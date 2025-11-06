# synaptica_clausewise_app.py
import os
import json
import base64
from datetime import datetime
import streamlit as st
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

from synaptica_clausewise_workflow import (
    run_pipeline,
    SUPPORTED_EXTS,
)
from legal_rag_bot import create_rag_bot, is_rag_available

APP_NAME = "Synaptica | clauseWise"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS for modern look ----------
st.markdown(
    """
    <style>
    .main { padding: 1.5rem; }
    .app-title { font-size: 2.1rem; font-weight: 800; letter-spacing: 0.5px; }
    .subtle { color: #667085; }
    .pill {
        display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; color:#4338ca;
        font-size:12px; font-weight:600; margin-left:8px;
    }
    .card {
        background: #ffffff;
        border: 1px solid #eaeaea;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        margin-bottom: 14px;
    }
    /* Fancy risk card inspired style */
    .fancy-card {
        position: relative;
        background: #fff;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(2, 6, 23, 0.08);
        border: 1px solid #eef2f7;
    }
    .fancy-header { display:flex; align-items:center; gap:10px; margin-bottom:8px; flex-wrap:wrap; }
    .fancy-title { font-weight:800; color:#0f172a; }
    .badge { display:inline-flex; align-items:center; padding:4px 10px; border-radius:999px; font-weight:700; font-size:12px; }
    .badge-soft { background:#eef2ff; color:#4338ca; }
    .badge-ctx { background:#f1f5f9; color:#0f172a; }
    .badge-risk { background:#fee2e2; color:#991b1b; }
    .badge-risk.medium { background:#ffedd5; color:#9a3412; }
    .badge-risk.low { background:#dcfce7; color:#065f46; }
    .quote { border-left:4px solid #e5e7eb; padding:10px 12px; margin:8px 0 2px; color:#334155; font-style:italic; background:#fafafa; border-radius:8px; }
    .actions { display:flex; gap:8px; align-items:center; margin-top:8px; }
    .btn-ghost { border:1px solid #e5e7eb; padding:6px 12px; border-radius:10px; background:#fff; font-weight:600; color:#0f172a; }
    .btn-ghost:hover { background:#f8fafc; }
    .btn-row { display:flex; gap:8px; align-items:center; margin: 6px 0 2px; }
    .muted { color:#64748b; font-size:12px; }
    .tag {
        display:inline-block; padding:2px 8px; border-radius:8px; background:#f1f5f9; margin-right:6px;
        font-size:12px; color:#0f172a; border:1px solid #e2e8f0;
    }
    .risk-low { background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }
    .risk-medium { background:#fffbeb; color:#92400e; border-color:#fde68a; }
    .risk-high { background:#fef2f2; color:#991b1b; border-color:#fecaca; }
    .clause-id { font-weight:700; color:#0f172a; }
    .clause-title { font-size:1.05rem; font-weight:700; margin-right:6px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    hr.sep { display:none; }
    
    /* Dashboard improvements */
    .dashboard-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 20px 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .safety-score {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 8px 0;
    }
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin-top: 8px;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 10px;
        transition: width 0.3s ease;
    }

    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .fancy-card { background:#0b1220; border-color:#1f2937; box-shadow: 0 8px 24px rgba(0,0,0,0.35); }
        .fancy-title, .clause-id { color:#e5e7eb; }
        .badge-ctx { background:#111827; color:#e5e7eb; border:1px solid #374151; }
        .badge-soft { background:#111827; color:#c7d2fe; border:1px solid #374151; }
        .quote { background:#0f172a; color:#cbd5e1; border-left-color:#334155; }
        .tag { background:#0f172a; color:#e5e7eb; border-color:#334155; }
        .risk-low { background:#052e1a; color:#86efac; border-color:#065f46; }
        .risk-medium { background:#3b2a12; color:#fdba74; border-color:#9a3412; }
        .risk-high { background:#3f0b0b; color:#fecaca; border-color:#b91c1c; }
        .muted { color:#94a3b8; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.caption("A modern contract analysis assistant powered by LangChain + Hugging Face.")
    st.divider()
    # Discover token from env -> secrets -> manual input (fallback)
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        try:
            token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")  # type: ignore[attr-defined]
            if token:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        except Exception:
            pass

    if token:
        masked = token[:6] + "..." + token[-4:] if len(token) > 10 else "******"
        st.success(f"Hugging Face token detected ({masked}).")
    else:
        st.warning("No token detected. Enter it once to set for this session.")
        typed = st.text_input("Hugging Face Token", type="password")
        if typed:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = typed
            st.session_state["_hf_token_set"] = True
            st.rerun()

    st.markdown("**IBM Granite (classification)** and **Mixtral (analysis & RAG bot)** are used by default.")
    st.caption("You can override models via environment variables: GRANITE_REPO, MAIN_LLM_REPO.")
    st.divider()
    st.markdown("### Processing Options")
    chunk_size = st.slider("Chunk size", 800, 3000, 1500, step=100)
    chunk_overlap = st.slider("Chunk overlap", 50, 600, 150, step=10)
    st.divider()
    st.markdown("### Export")
    download_name = st.text_input("Output JSON filename", value="synaptica_clausewise_output.json")

st.markdown(f"<div class='app-title'>🧠 {APP_NAME} <span class='pill'>LangChain • HuggingFace</span></div>", unsafe_allow_html=True)
st.write("Upload a contract (PDF / DOCX / TXT). The app classifies, extracts entities & clauses, explains them in simple language, and adds risks and real‑life situations.")

uploaded = st.file_uploader("Upload a legal document", type=[e.replace(".", "") for e in SUPPORTED_EXTS])

def _download_button(payload: dict, filename: str):
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    b64 = base64.b64encode(data).decode()
    st.download_button("📥 Download JSON", data=data, file_name=filename, mime="application/json")
    st.caption(f"Saved as **{filename}**")

def _risk_chip(score: str) -> str:
    s = (score or "").lower()
    if s == "high": return "risk-high"
    if s == "medium": return "risk-medium"
    return "risk-low"

def _first_nonempty(d: dict, keys: list[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # Fallback to the first key even if empty to preserve shape
    return d.get(keys[0], "") if keys else ""

def _calculate_risk_metrics(clauses: list) -> dict:
    """Calculate risk metrics from clauses."""
    total = len(clauses)
    high_count = 0
    medium_count = 0
    low_count = 0
    risk_tags = {}
    
    for clause in clauses:
        risk_score = (clause.get("Risk score", "low") or "low").lower()
        if risk_score == "high":
            high_count += 1
        elif risk_score == "medium":
            medium_count += 1
        else:
            low_count += 1
        
        # Track most common risk tag
        tag = clause.get("risk tag", "Legal")
        risk_tags[tag] = risk_tags.get(tag, 0) + 1
    
    # Calculate safety score: low risk = 100%, medium = 50%, high = 0%
    if total > 0:
        safety_score = int(((low_count * 100 + medium_count * 50) / total))
    else:
        safety_score = 100
    
    most_common_tag = max(risk_tags.items(), key=lambda x: x[1])[0] if risk_tags else "Legal"
    
    return {
        "total": total,
        "high": high_count,
        "medium": medium_count,
        "low": low_count,
        "safety_score": safety_score,
        "most_common_tag": most_common_tag,
    }

if uploaded:
    st.success(f"File selected: **{uploaded.name}**")
    
    # Text area to capture user background
    st.markdown("### About Yourself")
    st.caption("Tell us about your background (e.g., profession, role, location, experience) so we can personalize the risk scenarios.")
    user_background = st.text_area(
        "About yourself",
        placeholder="e.g., I'm a software developer working in Bangalore, India. I have 5 years of experience in tech startups...",
        height=100,
        key="user_background",
        help="This information will be used to generate personalized, realistic risk scenarios based on your actual circumstances."
    )
    
    run = st.button("▶️ Run analysis", type="primary", width='stretch', key="run_analysis")
    if run:
        with st.spinner("Analyzing document with IBM Granite + Mixtral..."):
            try:
                bg = user_background.strip() if user_background else ""
                result = run_pipeline(uploaded.name, uploaded.read(), chunk_size=chunk_size, chunk_overlap=chunk_overlap, user_background=bg)
                st.session_state["analysis_result"] = result
                st.session_state["uploaded_name"] = uploaded.name
            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.stop()

    # Use cached analysis if present
    result = st.session_state.get("analysis_result") if st.session_state.get("uploaded_name") == uploaded.name else None
    if result:
        # Check for errors in the result
        entities = result.get("entities", {})
        if isinstance(entities, dict) and entities.get("_error"):
            st.warning(f"⚠️ Entity extraction had issues: {entities.get('_error')}. Some entities may be missing.")
        
        clauses = result.get("clauses", [])
        if clauses:
            # Check if risk scores are using fallback (all medium)
            risk_scores = [c.get("Risk score", "medium") for c in clauses]
            if all(score == "medium" for score in risk_scores) and len(clauses) > 0:
                st.warning("⚠️ Risk assessment may have used fallback values. Some risk scores may not be accurate.")
        # ---- Header summary ----
        col1, col2, col3, col4 = st.columns([2,2,2,2])
        with col1:
            st.metric("Classification", result.get("classification", {}).get("classification", "—"))
        with col2:
            st.metric("# Chunks", result.get("chunks_count", 0))
        with col3:
            entities = result.get("entities", {}) or {}
            parties = entities.get("parties", []) if isinstance(entities, dict) else []
            st.metric("Parties found", len(parties) if isinstance(parties, list) else 0)
        with col4:
            st.metric("Clauses extracted", len(result.get("clauses", [])))

        # controls
        c1, c2, _ = st.columns([1,1,6])
        with c1:
            if st.button("🔁 Re-run analysis", key="rerun_analysis"):
                with st.spinner("Re-analyzing document..."):
                    try:
                        bg = user_background.strip() if user_background else ""
                        fresh = run_pipeline(uploaded.name, uploaded.read(), chunk_size=chunk_size, chunk_overlap=chunk_overlap, user_background=bg)
                        st.session_state["analysis_result"] = fresh
                        result = fresh
                    except Exception as e:
                        st.error(f"Re-run failed: {e}")
        with c2:
            if st.button("🗑️ Clear result", key="clear_result"):
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("uploaded_name", None)
                st.rerun()

        # ---- Risk Dashboard ----
        clauses = result.get("clauses", [])
        if clauses:
            metrics = _calculate_risk_metrics(clauses)
            safety_score = metrics["safety_score"]
            
            # Determine dominant risk level for center text
            dominant_risk = "Low"
            dominant_pct = 0
            if metrics["high"] > metrics["medium"] and metrics["high"] > metrics["low"]:
                dominant_risk = "High"
                dominant_pct = int((metrics["high"] / metrics["total"]) * 100) if metrics["total"] > 0 else 0
            elif metrics["medium"] > metrics["low"]:
                dominant_risk = "Medium"
                dominant_pct = int((metrics["medium"] / metrics["total"]) * 100) if metrics["total"] > 0 else 0
            else:
                dominant_pct = int((metrics["low"] / metrics["total"]) * 100) if metrics["total"] > 0 else 0
            
            # Dashboard container
            st.markdown("""
            <div class='dashboard-card'>
                <h2 style='margin-top:0; margin-bottom:24px; font-size:1.8rem; font-weight:700;'>📊 Risk Dashboard</h2>
            """, unsafe_allow_html=True)
            
            # Top metrics row
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='color:#94a3b8; font-size:0.9rem; margin-bottom:4px;'>Total Clauses</div>
                    <div style='font-size:2rem; font-weight:700; color:#fff;'>{metrics["total"]}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                risk_color = "#ef4444" if metrics["high"] > 0 else "#94a3b8"
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='color:#94a3b8; font-size:0.9rem; margin-bottom:4px;'>High Risk</div>
                    <div style='font-size:2rem; font-weight:700; color:{risk_color};'>{metrics["high"]}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                safety_color = "#22c55e" if safety_score >= 80 else "#fbbf24" if safety_score >= 50 else "#ef4444"
                st.markdown(f"""
                <div style='padding:16px;'>
                    <div style='color:#94a3b8; font-size:0.9rem; margin-bottom:8px;'>Document Safety</div>
                    <div class='safety-score' style='color:{safety_color};'>{safety_score}%</div>
                    <div class='progress-container'>
                        <div class='progress-fill' style='width:{safety_score}%; background:linear-gradient(90deg, {safety_color}, {safety_color}dd);'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Donut chart and legend
            col_chart, col_legend = st.columns([2, 1])
            
            with col_chart:
                if PLOTLY_AVAILABLE and (metrics["high"] > 0 or metrics["medium"] > 0 or metrics["low"] > 0):
                    # Create enhanced donut chart with center text
                    fig = go.Figure(data=[go.Pie(
                        labels=["High", "Medium", "Low"],
                        values=[metrics["high"], metrics["medium"], metrics["low"]],
                        hole=0.65,
                        marker_colors=["#3b82f6", "#fbbf24", "#f97316"],
                        textinfo="none",
                        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
                    )])
                    
                    # Add center text annotation
                    fig.add_annotation(
                        text=f"<b>{dominant_pct}%</b><br><span style='font-size:14px;'>{dominant_risk}</span>",
                        x=0.5, y=0.5,
                        font=dict(size=20, color="white"),
                        showarrow=False,
                        xref="paper", yref="paper"
                    )
                    
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=320,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ffffff"),
                    )
                    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            
            with col_legend:
                st.markdown("""
                <div style='padding-top:20px;'>
                    <div style='font-weight:600; font-size:1rem; margin-bottom:16px; color:#fff;'>Risk Levels</div>
                    <div style='display:flex; align-items:center; margin-bottom:12px; padding:8px; background:rgba(255,255,255,0.05); border-radius:8px;'>
                        <div style='width:14px; height:14px; background:#3b82f6; border-radius:50%; margin-right:10px; box-shadow:0 0 8px rgba(59,130,246,0.5);'></div>
                        <span style='color:#e5e7eb;'>High</span>
                    </div>
                    <div style='display:flex; align-items:center; margin-bottom:12px; padding:8px; background:rgba(255,255,255,0.05); border-radius:8px;'>
                        <div style='width:14px; height:14px; background:#fbbf24; border-radius:50%; margin-right:10px; box-shadow:0 0 8px rgba(251,191,36,0.5);'></div>
                        <span style='color:#e5e7eb;'>Medium</span>
                    </div>
                    <div style='display:flex; align-items:center; margin-bottom:12px; padding:8px; background:rgba(255,255,255,0.05); border-radius:8px;'>
                        <div style='width:14px; height:14px; background:#f97316; border-radius:50%; margin-right:10px; box-shadow:0 0 8px rgba(249,115,22,0.5);'></div>
                        <span style='color:#e5e7eb;'>Low</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary section
            safety_status = "highly safe ✅" if safety_score >= 80 else "moderately safe ⚠️" if safety_score >= 50 else "requires attention ⚠️"
            st.markdown(f"""
            <div style='margin-top:24px; padding-top:20px; border-top:1px solid rgba(255,255,255,0.1);'>
                <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); gap:16px;'>
                    <div style='padding:12px; background:rgba(255,255,255,0.05); border-radius:8px; border-left:3px solid #3b82f6;'>
                        <div style='color:#94a3b8; font-size:0.85rem;'>High-Risk Clauses</div>
                        <div style='color:#fff; font-size:1.1rem; font-weight:600; margin-top:4px;'>{metrics['high']} detected</div>
                    </div>
                    <div style='padding:12px; background:rgba(255,255,255,0.05); border-radius:8px; border-left:3px solid #fbbf24;'>
                        <div style='color:#94a3b8; font-size:0.85rem;'>Common Tag</div>
                        <div style='color:#fff; font-size:1.1rem; font-weight:600; margin-top:4px;'>{metrics['most_common_tag']}</div>
                    </div>
                    <div style='padding:12px; background:rgba(255,255,255,0.05); border-radius:8px; border-left:3px solid {safety_color};'>
                        <div style='color:#94a3b8; font-size:0.85rem;'>Safety Status</div>
                        <div style='color:#fff; font-size:1.1rem; font-weight:600; margin-top:4px;'>{safety_status}</div>
                    </div>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ---- Entities (card) ----
        st.markdown("<div class='fancy-card'>", unsafe_allow_html=True)
        with st.expander("🔎 Key Entities", expanded=True):
            st.json(result.get("entities", {}))
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Clause cards ----
        st.markdown("### 📑 Clauses")
        for i, clause in enumerate(result.get("clauses", []), start=1):
            # Handle None or empty ID values
            raw_id = clause.get("id")
            cid = str(raw_id) if raw_id is not None and str(raw_id).strip() else f"{i}"
            title = clause.get("title", "Untitled")
            ctype = clause.get("type", "Other")
            text = clause.get("text", "").strip()
            expl = _first_nonempty(clause, ["explanation", "Explanation", "plain_explanation"])
            rscore = clause.get("Risk score", "low")
            rtag = clause.get("risk tag", "Legal")
            rtext = _first_nonempty(clause, ["Risk", "risk", "risk_reason", "risk_text", "risk_explanation"])  # model may vary key
            scen = _first_nonempty(clause, ["Situation", "situation", "scenario"])  # model may vary key

            risk_class = _risk_chip(rscore)

            st.markdown("<div class='fancy-card'>", unsafe_allow_html=True)
            # Header row styled
            header_html = (
                f"<div class='fancy-header'>"
                f"<span class='clause-id'>§{cid}</span>"
                f"<span class='fancy-title'>{title}</span>"
                f"<span class='badge badge-ctx'>{ctype}</span>"
                f"<span class='badge badge-risk {rscore.lower()}'>"+("Critical Risk" if rscore.lower()=="high" else f"Risk: {rscore.capitalize()}")+"</span>"
                f"<span class='badge badge-soft'>{rtag}</span>"
                f"</div>"
            )
            st.markdown(header_html, unsafe_allow_html=True)

            # Display clause text directly as a quote (no container)
            st.markdown(f"<div class='quote'>"+ (text.replace("\n"," ")[:1200] or "—") +"</div>", unsafe_allow_html=True)

            st.markdown("**Explanation (plain language)**")
            st.write(expl or "—")

            # -- Toggle buttons to reveal Risk and Situation --
            safe_id = f"{i}_{cid}".replace(" ", "_").replace("None", str(i))
            risk_state_key = f"show_risk_{safe_id}"
            scen_state_key = f"show_scen_{safe_id}"
            if risk_state_key not in st.session_state:
                st.session_state[risk_state_key] = False
            if scen_state_key not in st.session_state:
                st.session_state[scen_state_key] = False

            col_btn1, col_btn2, col_btn3 = st.columns([1,1,6])
            with col_btn1:
                if st.button(
                    ("Hide risk" if st.session_state[risk_state_key] else "Show risk"),
                    key=f"btn_risk_{safe_id}",
                ):
                    st.session_state[risk_state_key] = not st.session_state[risk_state_key]
            with col_btn2:
                if st.button(
                    ("Hide situation" if st.session_state[scen_state_key] else "Show situation"),
                    key=f"btn_scen_{safe_id}",
                ):
                    st.session_state[scen_state_key] = not st.session_state[scen_state_key]
            with col_btn3:
                st.markdown("<span class='muted'>Use the buttons to reveal details</span>", unsafe_allow_html=True)

            if st.session_state[risk_state_key]:
                st.markdown("**Why this is a risk**")
                st.write(rtext or "—")

            if st.session_state[scen_state_key]:
                st.markdown("**Real‑life situation**")
                st.write(scen or "—")

            st.markdown("</div>", unsafe_allow_html=True)

        # subtle spacing only, no white divider
        _download_button(result, download_name)
        
        # ============================================
        # Legal RAG Bot Section (at the end)
        # ============================================
        st.markdown("---")
        st.markdown("## 🤖 Legal Query Assistant")
        st.markdown("Ask questions about the processed document using our AI assistant powered by Mixtral (stable conversational model).")
        
        # RAG bot is always available - lightweight implementation
        st.info("💡 The RAG bot uses lightweight text matching for fast responses on any device.")
        
        # Initialize chat history
        if "rag_chat_history" not in st.session_state:
            st.session_state.rag_chat_history = []
        
        # Initialize RAG bot if not already done
        if "rag_bot" not in st.session_state or st.session_state.get("rag_bot_result_key") != id(result):
            try:
                st.session_state.rag_bot = create_rag_bot(result)
                st.session_state.rag_bot_result_key = id(result)
                st.session_state.rag_chat_history = []  # Clear history when new document is processed
                
                # Check if knowledge base has content
                if st.session_state.rag_bot.kb_items and len(st.session_state.rag_bot.kb_items) > 0:
                    st.success("✅ RAG bot initialized with the current document analysis.")
                else:
                    st.warning("⚠️ RAG bot initialized but document has no extractable clauses or entities. The bot may not be able to answer questions.")
                
                # Check if LLM initialized properly
                if not st.session_state.rag_bot.llm:
                    error_msg = getattr(st.session_state.rag_bot, 'llm_error', 'Unknown error')
                    st.error(f"❌ RAG bot LLM initialization failed: {error_msg}. Please check your HUGGINGFACEHUB_API_TOKEN.")
            except Exception as e:
                st.error(f"Failed to initialize RAG bot: {str(e)}. Please check your configuration and try again.")
                st.session_state.rag_bot = None
        
        # Chat interface
        if st.session_state.rag_bot:
            # Display chat history
            st.markdown("### 💬 Chat History")
            
            # Chat container
            chat_container = st.container()
            with chat_container:
                for i, msg in enumerate(st.session_state.rag_chat_history):
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 12px 16px; border-radius: 12px; 
                                    margin-bottom: 12px; margin-left: 20%;'>
                            <div style='font-weight: 600; margin-bottom: 4px;'>You</div>
                            <div>{msg["content"]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: #f1f5f9; color: #0f172a; padding: 12px 16px; 
                                    border-radius: 12px; margin-bottom: 12px; margin-right: 20%;
                                    border-left: 4px solid #4338ca;'>
                            <div style='font-weight: 600; margin-bottom: 4px; color: #4338ca;'>🤖 Legal Assistant</div>
                            <div style='line-height: 1.6;'>{msg["content"]}</div>
                            {f'<div style="font-size: 0.85rem; color: #64748b; margin-top: 8px; padding-top: 8px; border-top: 1px solid #e2e8f0;">📚 Based on: {msg.get("sources", "Document context")}</div>' if msg.get("sources") else ''}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Query input
            st.markdown("### ✍️ Ask a Question")
            col_query, col_send = st.columns([5, 1])
            
            with col_query:
                user_query = st.text_input(
                    "Enter your question about the document",
                    placeholder="e.g., What are the high-risk clauses? What are the termination terms? Who are the parties?",
                    key="rag_query_input",
                    label_visibility="collapsed"
                )
            
            with col_send:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                send_button = st.button("💬 Ask", type="primary", width='stretch', key="rag_send_button")
            
            # Example questions
            st.markdown("**💡 Example Questions:**")
            example_cols = st.columns(4)
            examples = [
                "What are the high-risk clauses?",
                "What are the payment terms?",
                "Who are the parties involved?",
                "What is the termination clause?"
            ]
            example_selected = None
            for i, example in enumerate(examples):
                with example_cols[i]:
                    if st.button(example, key=f"example_{i}", width='stretch'):
                        example_selected = example
            
            # Process query (from input or example button)
            if (send_button and user_query) or example_selected:
                query_to_use = example_selected if example_selected else user_query
                with st.spinner("🤔 Analyzing your question..."):
                    try:
                        # Add user message to history
                        st.session_state.rag_chat_history.append({
                            "role": "user",
                            "content": query_to_use,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Get answer from RAG bot
                        response = st.session_state.rag_bot.answer_query(query_to_use)
                        
                        # Format sources
                        sources_text = ""
                        if response.get("context_sources"):
                            sources_list = []
                            for source in response["context_sources"][:3]:  # Show top 3 sources
                                if source.get("clause_id"):
                                    sources_list.append(f"Clause {source['clause_id']} ({source.get('title', 'Untitled')})")
                                elif source.get("type"):
                                    sources_list.append(source["type"].replace("_", " ").title())
                            if sources_list:
                                sources_text = ", ".join(sources_list)
                        
                        # Add assistant response to history
                        st.session_state.rag_chat_history.append({
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": sources_text,
                            "timestamp": response["timestamp"]
                        })
                        
                        # Clear input and rerun to show new message
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        st.session_state.rag_chat_history.append({
                            "role": "assistant",
                            "content": f"I encountered an error: {str(e)}. Please try again.",
                            "timestamp": datetime.now().isoformat()
                        })
                        st.rerun()
            
            # Clear chat button
            if st.session_state.rag_chat_history:
                if st.button("🗑️ Clear Chat History", key="clear_rag_chat"):
                    st.session_state.rag_chat_history = []
                    st.rerun()
        else:
            st.info("RAG bot is not available. Please ensure the document analysis completed successfully.")
else:
    st.info("Start by uploading a file. Supported: PDF, DOCX, TXT.")
