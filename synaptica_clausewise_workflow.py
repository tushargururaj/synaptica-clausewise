# synaptica_clausewise_workflow.py
"""
Synaptica | clauseWise - GenAI Workflow (LangChain + Hugging Face)
------------------------------------------------------------------
- Uses ONLY Hugging Face endpoints via LangChain's HuggingFaceEndpoint.
- IBM Granite is used for the *document classification* task.
- A strong instruction-tuned model (default Mixtral Instruct) is used for
  extraction / explanation / risk scoring / scenarios.
- Modular design for easy debugging and unit testing.
- No mock data; all steps invoke real LLMs.
- Keep HUGGINGFACEHUB_API_TOKEN set in your environment.

Install (minimum):
    pip install langchain langchain-core langchain-huggingface langchain-text-splitters pypdf python-docx

Environment:
    export HUGGINGFACEHUB_API_TOKEN="hf_xxx"

Optional overrides:
    export GRANITE_REPO="ibm-granite/granite-8b-instruct"
    export MAIN_LLM_REPO="mistralai/Mixtral-8x7B-Instruct-v0.1"
"""

from __future__ import annotations

import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# ---- Simple document parsing ----
from pypdf import PdfReader
from docx import Document as DocxDocument

# ---- LangChain Core ----
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ---- Hugging Face LLMs via LangChain ----
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# ---- Text splitters ----
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================
# Model registry / constructors
# =============================
GRANITE_REPO = os.getenv("GRANITE_REPO", "ibm-granite/granite-3.2-8b-instruct")
MAIN_LLM_REPO = os.getenv("MAIN_LLM_REPO", "mistralai/Mixtral-8x7B-Instruct-v0.1")

def _default_task_for_repo(repo_id: str) -> str:
    """Pick a reasonable HF task for the given repo.
    Some providers expose Mixtral/Mistral as 'conversational' instead of 'text-generation'."""
    # Allow explicit override
    task_override = os.getenv("HF_TASK")
    if task_override:
        return task_override
    rid = (repo_id or "").lower()
    if "mixtral" in rid or "mistral" in rid:
        return "conversational"
    return "text-generation"

def _require_hf_token():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is not set. Please export your Hugging Face token."
        )

def build_llm(repo_id: str, *, max_new_tokens: int = 512, temperature: float = 0.2):
    """
    Create a LangChain HuggingFaceEndpoint LLM with consistent defaults.
    """
    _require_hf_token()
    task = _default_task_for_repo(repo_id)
    base_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task=task,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        streaming=False,
        timeout=120,
    )
    # Use chat wrapper when provider exposes conversational endpoints (e.g., Mixtral)
    if task == "conversational":
        return ChatHuggingFace(llm=base_llm)
    # Otherwise return the base text-generation endpoint
    return base_llm


# ================
# File IO helpers
# ================
SUPPORTED_EXTS = (".pdf", ".docx", ".txt")

def _lower(s: str) -> str:
    return (s or "").lower()

def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """
    Extract text from PDF / DOCX / TXT. Keep it simple.
    """
    name = _lower(filename)
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    elif name.endswith(".docx"):
        doc = DocxDocument(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    elif name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Please upload .pdf, .docx, or .txt")


# ==================
# Prompt definitions
# ==================
classification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert legal document classifier. "
            "Classify the document into a single, precise type from the set: "
            "[NDA, Lease, Rental Agreement, Service Agreement, Employment Agreement, "
            "Consulting Agreement, Sales Agreement, License Agreement, MSA, SOW, Other]. "
            "Reply with STRICT JSON ONLY: {{\"classification\": \"<type>\"}}. No prose."
        ),
        (
            "human",
            "Document text:\n\n{doc_text}\n\n"
            "Return a single JSON object exactly like this: {{\"classification\": \"Lease\"}}."
        ),
    ]
)

entities_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful legal information extractor. Extract key entities and clauses from the document. "
            "Return STRICT JSON ONLY with these top-level keys:\n"
            "{{\n"
            "  \"entities\": {{\n"
            "    \"parties\": [\"...\"],\n"
            "    \"effective_date\": \"...\",\n"
            "    \"termination_date\": \"...\",\n"
            "    \"governing_law\": \"...\",\n"
            "    \"notice_periods\": \"...\",\n"
            "    \"liability_provisions\": \"...\",\n"
            "    \"confidentiality_terms\": \"...\",\n"
            "    \"payment_terms\": \"...\",\n"
            "    \"ip_rights\": \"...\"\n"
            "  }}\n"
            "}}\n"
            "Use null where unknown. No extra commentary."
        ),
        ("human", "Document text:\n\n{doc_text}"),
    ]
)

clauses_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a legal clause extractor. From the provided chunk, extract any self-contained legal clauses. "
            "Return STRICT JSON ARRAY ONLY. Each element is an object:\n"
            "{{\n"
            "  \"id\": \"<number or section>\",\n"
            "  \"title\": \"<short descriptive title>\",\n"
            "  \"type\": \"<one of: Termination, Confidentiality, Payment, IP, Liability, Notice, Governing Law, Definitions, Services, Warranty, Miscellaneous, Other>\",\n"
            "  \"text\": \"<verbatim clause text>\"\n"
            "}}\n"
            "If none found, return []."
        ),
        ("human", "Chunk:\n{chunk}"),
    ]
)




explain_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert legal translator who explains legal clauses in plain, accurate, and practical language for non-lawyers.\n\n"
            "YOUR ROLE:\n"
            "Transform complex contract clauses into clear explanations so that any reader—without legal training—can easily understand what the clause means and what it requires.\n"
            "You do NOT simplify to the point of losing meaning; your goal is accuracy + clarity.\n\n"
            "TASK:\n"
            "For EACH object in the input JSON array (each clause), ADD one field called 'explanation' that describes the clause in plain English.\n"
            "You must return the SAME JSON array structure, with all existing fields untouched and an added 'explanation' string for each clause.\n"
            "Return ONLY the JSON array — no markdown, no notes, no extra text.\n\n"
            "EXPLANATION REQUIREMENTS:\n"
            "- Write 1–3 short sentences (max ~45 words).\n"
            "- Use simple, everyday language.\n"
            "- Explain the PRACTICAL effect — what the signer must do, avoid, or understand.\n"
            "- Keep explanations fact-based and neutral.\n"
            "- Preserve the clause’s meaning; do not omit legal conditions.\n"
            "- Use active voice (e.g., 'You must pay...' instead of 'Payment shall be made...').\n"
            "- Define any uncommon legal term briefly when first mentioned (e.g., 'indemnify (pay for losses)').\n"
            "- Avoid restating the clause; explain it as if summarizing for clarity.\n"
            "- Avoid giving examples, opinions, or hypothetical situations.\n\n"
            "QUALITY EXPECTATIONS:\n"
            "- Each explanation should sound natural, human, and precise.\n"
            "- If the clause has complex conditions, break it into 2 short, clear sentences.\n"
            "- If the clause is vague, acknowledge it (e.g., 'This clause is broad and may be open to interpretation.').\n"
            "- Maintain consistent tone and readability throughout.\n"
            "- Every clause MUST have an explanation — no exceptions.\n\n"
            "DO NOT:\n"
            "- Include personal details or user roles (tenant, student, etc.).\n"
            "- Write hypotheticals or examples ('Imagine you...').\n"
            "- Give legal advice.\n"
            "- Add or remove any other JSON keys.\n\n"
            "STYLE GUIDELINES:\n"
            "- Be concise, yet complete.\n"
            "- Prefer direct language: 'You must notify' instead of 'It shall be incumbent upon you to notify.'\n"
            "- One main idea per sentence.\n"
            "- Use commas and conjunctions to connect closely related ideas when needed.\n"
            "- Keep tone professional, neutral, and explanatory.\n\n"
            "EXAMPLES OF IDEAL EXPLANATIONS:\n"
            "Clause: 'The tenant must obtain written consent from the landlord prior to any modifications.'\n"
            "Explanation: 'You must get written permission from the landlord before making any changes. You cannot modify anything without approval.'\n\n"
            "Clause: 'The supplier shall indemnify the buyer against all claims arising from defective goods.'\n"
            "Explanation: 'You must cover any losses or legal claims if the goods are defective. This means you are responsible for problems caused by faulty products.'\n\n"
            "Clause: 'This agreement remains in force until terminated by either party with thirty (30) days' written notice.'\n"
            "Explanation: 'This contract continues until one side ends it with 30 days’ written notice. You must inform the other side in writing at least 30 days before ending it.'\n\n"
            "Clause: 'All confidential information must be kept secret and not shared without prior consent.'\n"
            "Explanation: 'You must keep private information secret and cannot share it unless the other side allows it in writing.'\n\n"
            "Clause: 'The party in breach shall be liable for all consequential damages, including loss of profit.'\n"
            "Explanation: 'If you break this contract, you must pay for any resulting losses, including lost income or profits.'\n\n"
            "Clause: 'This agreement shall be governed by the laws of the State of Karnataka.'\n"
            "Explanation: 'This contract follows Karnataka law. Any dispute will be handled under that legal system.'\n\n"
            "Clause: 'Payments are due within fifteen (15) days of receipt of invoice.'\n"
            "Explanation: 'You must pay within 15 days after receiving the invoice.'\n\n"
            "Clause: 'The employee shall not disclose trade secrets during or after employment.'\n"
            "Explanation: 'You cannot share confidential company information while working there or after leaving the job.'\n\n"
            "QUALITY CONTROL (SILENT):\n"
            "- Before finishing, verify each clause: 1) explanation added, 2) ≤3 sentences, 3) clear and active, 4) accurate meaning, 5) plain language.\n"
            "- DO NOT output this checklist — only the final JSON array.\n\n"
            "FINAL REMINDER:\n"
            "→ Output STRICT JSON array only, containing all original fields + 'explanation'. Nothing else."
        ),
        ("human", "Clauses JSON array:\n{clauses_json}"),
    ]
)


risk_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior contract risk analyst with advanced expertise in legal, financial, compliance, and operational risk assessment. "
            "Your job is to evaluate each clause in the provided JSON and assign a precise, well-calibrated risk score and risk tag.\n\n"
            "TASK:\n"
            "For EACH clause in the JSON array, ADD two fields ONLY:\n"
            "- \"Risk score\": exactly one of [\"low\", \"medium\", \"high\"]\n"
            "- \"risk tag\": exactly one of [\"Financial\", \"Operational\", \"Strategic\", \"Compliance\", \"Legal\", \"Security\", \"Performance\", \"Reputation\"]\n"
            "Return STRICT JSON ARRAY with the SAME clause objects (copy all existing fields untouched) and append only these two keys.\n"
            "Do not add any commentary, rationale, or formatting beyond the JSON.\n\n"
            "RISK SCORING CRITERIA (DETAILED):\n\n"
            "HIGH RISK — Clauses that create significant exposure or uncontrolled consequences. Usually one-sided, broad, or unlimited in nature.\n"
            "Examples:\n"
            "- Financial: 'The supplier shall pay a penalty of ₹10,00,000 for any breach of this agreement.' (large, one-sided penalty)\n"
            "- Financial: 'The vendor will not be paid until the client receives payment from their end customer.' (cash flow dependency)\n"
            "- Legal: 'The company assumes unlimited liability for any direct or indirect losses.' (no cap on liability)\n"
            "- Legal: 'Disputes will be resolved only through binding arbitration in a foreign jurisdiction.' (no appeal, unfair venue)\n"
            "- Compliance: 'Data may be transferred internationally without explicit consent or security safeguards.' (regulatory violation)\n"
            "- Operational: 'The client may terminate this agreement at any time without notice.' (no operational stability)\n"
            "- Strategic: 'All inventions and intellectual property created shall belong solely to the client, including pre-existing materials.' (total IP loss)\n"
            "- Security: 'Vendor grants third-party access to client data for any business purpose.' (data security breach)\n"
            "- Performance: 'Any delay in service will result in immediate contract termination.' (disproportionate consequence)\n"
            "- Reputation: 'The client may publicly use the company’s name and performance data for promotion without approval.' (reputation risk)\n"
            "- High-risk clauses typically include uncapped obligations, unilateral rights, broad indemnities, or indefinite exposure.\n\n"
            "MEDIUM RISK — Clauses that present moderate exposure or ambiguity but are common in business contracts. Manageable through negotiation or monitoring.\n"
            "Examples:\n"
            "- Financial: 'Invoices are payable within 60 days; 1% interest per month applies to delays.' (standard but affects cash flow)\n"
            "- Financial: 'Client may withhold up to 10% of payment until final delivery.' (moderate financial impact)\n"
            "- Legal: 'Liability is limited to the value of the contract.' (reasonable cap but still exposure)\n"
            "- Legal: 'Disputes will be resolved by arbitration in the seller’s jurisdiction.' (potential inconvenience)\n"
            "- Compliance: 'Both parties agree to comply with applicable data protection laws.' (broad, standard compliance)\n"
            "- Operational: 'Either party may terminate with 30 days’ written notice.' (manageable operational flexibility)\n"
            "- Strategic: 'Client receives an exclusive right to distribute the product in one region for one year.' (short-term strategic limitation)\n"
            "- Security: 'Customer data will be stored with authorized vendors, retention period to be defined later.' (moderate uncertainty)\n"
            "- Performance: 'Service Level: 99% uptime, below 95% triggers 10% credit.' (reasonable but measurable risk)\n"
            "- Reputation: 'Client may reference company name in bid proposals.' (low reputational impact)\n"
            "- Medium-risk clauses often contain limitations, but with potential ambiguity or negotiation points.\n\n"
            "LOW RISK — Clauses that are standard, fair, and balanced with minimal exposure even in worst-case interpretation.\n"
            "Examples:\n"
            "- Financial: 'Payment to be made within 30 days of invoice receipt.' (industry standard)\n"
            "- Financial: 'Any applicable taxes shall be borne by the payer as required by law.' (neutral)\n"
            "- Legal: 'This agreement is governed by the laws of India.' (standard legal jurisdiction)\n"
            "- Compliance: 'Each party agrees to maintain confidentiality of personal data and follow applicable data laws.' (mutual protection)\n"
            "- Operational: 'The contract will automatically renew annually unless either party provides 60 days’ notice.' (predictable)\n"
            "- Strategic: 'This agreement does not restrict either party from entering into similar engagements.' (non-exclusive, safe)\n"
            "- Security: 'Vendor shall implement industry-standard security controls and notify of any breach within 24 hours.' (balanced and defined)\n"
            "- Performance: 'Service reports shall be shared monthly for review.' (transparent, low consequence)\n"
            "- Reputation: 'Neither party may use the other’s logo or name without prior written consent.' (standard mutual clause)\n"
            "- Low-risk clauses are balanced, clearly defined, mutual, and limited in scope.\n\n"
            "RISK TAG GUIDELINES:\n"
            "- Financial → Pricing, payments, penalties, late fees, indemnity, damages, or cost-related exposure.\n"
            "- Legal → Liability, jurisdiction, enforceability, warranties, governing law, indemnity, dispute resolution.\n"
            "- Compliance → Data protection, privacy, statutory adherence, government/regulatory requirements.\n"
            "- Operational → Delivery schedules, terminations, renewals, service interruptions, process disruption.\n"
            "- Strategic → IP rights, exclusivity, ownership, restrictions, long-term business impact.\n"
            "- Security → Cybersecurity, access control, data retention, breach, encryption, and confidentiality scope.\n"
            "- Performance → SLAs, uptime, deliverable quality, testing, measurement, or penalties for underperformance.\n"
            "- Reputation → Brand use, publicity rights, public statements, confidentiality breaches, moral obligations.\n\n"
            "ADJUSTMENT FACTORS:\n"
            "- Increase risk level if the clause is vague, one-sided, uncapped, or grants unlimited authority.\n"
            "- Decrease risk level if the clause is mutual, capped, or clearly defined with limited scope.\n"
            "- If no meaningful exposure exists, mark as LOW by default.\n"
            "- Choose the tag that represents the PRIMARY domain of risk.\n\n"
            "ADDITIONAL CALIBRATION EXAMPLES:\n"
            "1. 'The Service Provider shall indemnify the Client for all losses arising out of any claim.' → HIGH / Legal (broad indemnity = major exposure)\n"
            "2. 'The Client shall pay interest at 2% per month on delayed payments.' → MEDIUM / Financial (moderate cash flow risk)\n"
            "3. 'Either Party may terminate this Agreement upon 90 days’ notice.' → LOW / Operational (balanced and fair)\n"
            "4. 'The Company shall own all deliverables and underlying IP.' → HIGH / Strategic (complete IP loss)\n"
            "5. 'This Agreement shall be governed by the laws of Karnataka.' → LOW / Legal (standard jurisdiction)\n"
            "6. 'Supplier warrants that goods conform to specifications.' → MEDIUM / Performance (common, limited exposure)\n"
            "7. 'Vendor must comply with GDPR.' → MEDIUM / Compliance (standard regulatory term)\n"
            "8. 'The customer may publish performance results publicly without approval.' → HIGH / Reputation (reputational control loss)\n"
            "9. 'Service Level: 98% uptime per month; failure triggers 5% rebate.' → MEDIUM / Performance (reasonable but measurable risk)\n"
            "10. 'The customer can terminate without cause and without penalty.' → HIGH / Operational (one-sided termination)\n"
            "11. 'Either party must provide 30 days’ written notice before termination.' → LOW / Operational (balanced)\n"
            "12. 'Vendor must maintain ISO 27001 certification throughout the contract term.' → MEDIUM / Compliance (manageable regulatory requirement)\n"
            "13. 'The provider shall refund all fees if services fail to meet standards for two consecutive months.' → HIGH / Financial (major repayment exposure)\n"
            "14. 'All confidential information must be returned or destroyed upon termination.' → LOW / Compliance (standard confidentiality term)\n"
            "15. 'Supplier is responsible for all damages, direct or indirect, caused by its employees.' → HIGH / Legal (broad liability)\n"
            "16. 'Payments will be made after successful quality verification of deliverables.' → MEDIUM / Performance (moderate business impact)\n"
            "17. 'The client may request additional reports within reasonable limits.' → LOW / Operational (safe, bounded clause)\n"
            "18. 'Any breach of confidentiality shall lead to immediate termination.' → HIGH / Compliance (strict penalty for breach)\n"
            "19. 'The parties agree to act in good faith in all dealings under this contract.' → LOW / Legal (standard good-faith clause)\n"
            "20. 'If a dispute arises, parties will attempt mediation before arbitration.' → LOW / Legal (balanced dispute resolution)\n\n"
            "QUALITY CONTROL (SILENT)\n"
            "Before assigning a risk score:\n"
            "1. Assess the worst-case consequence if the clause is enforced unfavorably.\n"
            "2. Consider whether the clause is mutual, capped, or one-sided.\n"
            "3. Evaluate the clarity and fairness of obligations.\n"
            "4. Identify which domain the main risk affects.\n"
            "5. Adjust conservatively based on definitions, limits, or missing safeguards.\n"
            "Do NOT output reasoning. Only return the final JSON array.\n\n"
            "FORMAT RULES:\n"
            "- Keep all original keys.\n"
            "- Add exactly: \"Risk score\" and \"risk tag\".\n"
            "- Output STRICT JSON array only (no prose, no markdown, no commentary)."
        ),
        ("human", "Clauses with explanations:\n{explained_json}"),
    ]
)


risk_reason_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a contract risk analyst explaining risks to non-lawyers. For each clause with a risk score and tag, provide a clear, actionable risk explanation.\n\n"
            "Your task: Add a \"Risk\" field to each clause explaining:\n"
            "1. What specific risk exists (be concrete, not vague)\n"
            "2. Why this particular clause creates that risk\n"
            "3. What could go wrong in practical terms\n\n"
            "GUIDELINES:\n"
            "- Write in plain, simple language (avoid legal jargon)\n"
            "- Be specific: Instead of 'may cause problems', say 'could result in unlimited financial liability'\n"
            "- Explain the CONSEQUENCE: What happens if this clause is enforced?\n"
            "- Match the severity to the risk score:\n"
            "  * HIGH risk: Describe serious consequences (financial loss, legal liability, career impact)\n"
            "  * MEDIUM risk: Describe moderate concerns (inconvenience, extra costs, limited flexibility)\n"
            "  * LOW risk: Note minor concerns or standard industry practice\n"
            "- For LOW risk clauses, you can acknowledge it's standard or explain why it's generally acceptable\n"
            "- Keep it concise: 1-3 sentences, maximum 50 words\n"
            "- Focus on WHAT the risk is, not just restating the clause\n\n"
            "EXAMPLES:\n"
            "- Good: \"This clause requires unlimited liability for any damages, which means you could be held responsible for costs far exceeding the contract value if something goes wrong.\"\n"
            "- Good: \"The non-compete restriction prevents you from working in your industry for 2 years across the entire country, which could severely limit future employment opportunities.\"\n"
            "- Good: \"This is a standard confidentiality clause that protects both parties' information equally, which is typical and fair in most agreements.\"\n"
            "- Bad: \"This clause may have risks.\" (too vague)\n"
            "- Bad: \"The clause states that...\" (just restating, not explaining risk)\n\n"
            "Return STRICT JSON ARRAY with each clause having an added \"Risk\" field. No commentary."
        ),
        ("human", "Clauses with risk tags and scores:\n{risked_json}"),
    ]
)

def build_scenario_prompt(user_background: str = "") -> ChatPromptTemplate:
    """Build scenario prompt with optional user background for personalization."""
    system_msg = (
        "You generate a realistic, legally accurate scenario ('Situation') for each clause risk so a reader understands real‑life impact. "
        "This is SEPARATE from the explanation - the explanation explains what the clause means, while the Situation shows how it affects someone in real life.\n\n"
        "REQUIREMENTS:\n"
        "- Write 2-4 sentences describing a realistic scenario\n"
        "- Show how the clause could impact someone in a practical situation\n"
        "- Do not invent facts that conflict with the clause text\n"
        "- No names of real companies\n"
        "- Focus on practical consequences and real-world implications\n"
    )
    if user_background.strip():
        system_msg += (
            f"\n\nPERSONALIZATION:\n"
            f"The user's background context is: {user_background.strip()}\n"
            "Connect the situation to the user's actual life circumstances based on this background. "
            "DO NOT hallucinate or invent details not provided in the background. "
            "Use only the information provided: their profession, location, experience level, or other stated facts. "
            "If a specific detail isn't in the background, make the scenario generic but still relevant to their stated context. "
            "The situation must feel realistic and personally relevant to their circumstances.\n\n"
            "IMPORTANT: This Situation field is DIFFERENT from the explanation field. "
            "The explanation is generic and explains what the clause means. "
            "The Situation is personalized and shows how it affects THIS specific user based on their background."
        )
    else:
        system_msg += (
            "\n\nIMPORTANT: This Situation field is DIFFERENT from the explanation field. "
            "The explanation is generic and explains what the clause means. "
            "The Situation shows a realistic scenario of how this clause could affect someone in practice."
        )
    system_msg += "\n\nReturn STRICT JSON ARRAY with an added \"Situation\" field. Do NOT include this in the explanation field."
    
    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Clauses with risk explanations:\n{risk_text_json}"),
    ])


# ===================
# Helper: Merge LLM response with existing clauses
# ===================
def _merge_llm_response(existing_clauses: List[Dict[str, Any]], llm_response: List[Dict[str, Any]], 
                        field_to_match: str = "text") -> List[Dict[str, Any]]:
    """
    Intelligently merge LLM response back into existing clauses.
    Matches clauses by text/content (with fuzzy matching) and preserves all existing data.
    Only updates fields that LLM provided.
    Falls back to positional matching if text matching fails.
    """
    if not llm_response or len(llm_response) == 0:
        return existing_clauses
    
    # Normalize text for matching (remove extra whitespace, lowercase for comparison)
    def normalize_text(s: str) -> str:
        return " ".join(s.strip().lower().split()) if s else ""
    
    # Create lookup by normalized text
    llm_lookup = {}
    llm_by_position = {}
    for idx, item in enumerate(llm_response):
        key = normalize_text(item.get(field_to_match, ""))
        if key:
            llm_lookup[key] = item
        # Also store by position as fallback
        if idx < len(existing_clauses):
            llm_by_position[idx] = item
    
    # Merge: preserve existing, update with LLM data
    merged = []
    for idx, existing in enumerate(existing_clauses):
        existing_key = normalize_text(existing.get(field_to_match, ""))
        updated = existing.copy()  # Start with all existing data
        
        # Try to find matching LLM data
        llm_data = None
        if existing_key in llm_lookup:
            # Perfect match by text
            llm_data = llm_lookup[existing_key]
        elif idx in llm_by_position:
            # Fallback: positional match
            llm_data = llm_by_position[idx]
        
        # If LLM provided data for this clause, merge it
        if llm_data:
            # Special handling: Check for explanation with different case variations FIRST
            explanation_keys = ["explanation", "Explanation", "EXPLANATION", "plain_explanation", "plain_language"]
            found_explanation = False
            for key in explanation_keys:
                if key in llm_data and llm_data[key] and str(llm_data[key]).strip():
                    updated["explanation"] = str(llm_data[key]).strip()
                    found_explanation = True
                    break
            
            # Only update non-empty fields from LLM
            for k, v in llm_data.items():
                if v is not None:
                    # Skip explanation if we already handled it above
                    if k.lower() in [ek.lower() for ek in explanation_keys] and found_explanation:
                        continue
                    # For strings, only update if non-empty
                    if isinstance(v, str):
                        if v.strip():
                            updated[k] = v.strip()
                    else:
                        # For non-strings (numbers, booleans, etc.), always update
                        updated[k] = v
        
        merged.append(updated)
    
    return merged


# ===================
# Core step functions
# ===================
def classify_document(doc_text: str) -> Dict[str, Any]:
    candidates = [
        GRANITE_REPO,
        MAIN_LLM_REPO,
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    last_err: Optional[Exception] = None
    for repo in candidates:
        try:
            llm = build_llm(repo, max_new_tokens=128, temperature=0.0)
            chain = classification_prompt | llm | StrOutputParser()
            raw = chain.invoke({"doc_text": doc_text[:8000]})
            try:
                return json.loads(_extract_json(raw))
            except Exception:
                return {"classification": "Other"}
        except Exception as e:
            last_err = e
            continue
    # If all candidates failed, return a safe default
    return {"classification": "Other", "_error": str(last_err) if last_err else None}

def extract_entities(doc_text: str) -> Dict[str, Any]:
    llm = build_llm(MAIN_LLM_REPO, max_new_tokens=512, temperature=0.1)
    chain = entities_prompt | llm | StrOutputParser()
    raw = chain.invoke({"doc_text": doc_text[:10000]})
    try:
        return json.loads(_extract_json(raw))
    except Exception:
        return {"entities": {
            "parties": [],
            "effective_date": None,
            "termination_date": None,
            "governing_law": None,
            "notice_periods": None,
            "liability_provisions": None,
            "confidentiality_terms": None,
            "payment_terms": None,
            "ip_rights": None
        }}

def split_text(doc_text: str, *, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    return splitter.split_text(doc_text)

def extract_clauses_from_chunks(chunks: List[str]) -> List[Dict[str, Any]]:
    llm = build_llm(MAIN_LLM_REPO, max_new_tokens=700, temperature=0.1)
    chain = clauses_prompt | llm | StrOutputParser()
    results: List[Dict[str, Any]] = []
    for ch in chunks:
        raw = chain.invoke({"chunk": ch[:4000]})
        try:
            items = json.loads(_extract_json(raw))
            if isinstance(items, list):
                results.extend(items)
        except Exception:
            continue
    # De-dup by text hash
    uniq, seen = [], set()
    for c in results:
        t = (c.get("title",""), c.get("text",""))
        if t not in seen:
            uniq.append(c)
            seen.add(t)
    return uniq

def add_plain_explanations(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not clauses:
        return clauses
    
    llm = build_llm(MAIN_LLM_REPO, max_new_tokens=700, temperature=0.2)
    payload = json.dumps(clauses, ensure_ascii=False)
    chain = explain_prompt | llm | StrOutputParser()
    raw = chain.invoke({"clauses_json": payload})
    
    try:
        llm_result = json.loads(_extract_json(raw))
        if isinstance(llm_result, list) and len(llm_result) > 0:
            # Merge LLM response with existing clauses
            merged = _merge_llm_response(clauses, llm_result, field_to_match="text")
            
            # Verify all clauses have explanations, add fallback if missing
            # Also clean explanations to remove personalized content
            for i, clause in enumerate(merged):
                expl = clause.get("explanation", "").strip()
                
                # Remove personalized content from explanations
                if expl:
                    # Split on periods and clean each sentence
                    sentences = expl.replace('!', '.').replace('?', '.').split('.')
                    cleaned_sentences = []
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                            
                        lower_sent = sent.lower()
                        
                        # Check for personalized phrases anywhere in the sentence
                        personalized_phrases = [
                            "as a", "as an", "for someone", "for a", "imagine", 
                            "suppose", "consider", "if you're", "if you are",
                            "keep this in mind", "remember that", "note that for",
                            "this is important for", "especially if you", "particularly if"
                        ]
                        
                        # Check for personal identifiers
                        personal_indicators = [
                            "year-old", "student", "developer", "professional", 
                            "working in", "living in", "from", "academic year",
                            "study schedule", "move-in", "move-out", "busy",
                            "computer science", "engineering", "business",
                            "your schedule", "your studies", "your work"
                        ]
                        
                        # Skip if sentence contains personalized content
                        is_personalized = (
                            any(lower_sent.startswith(prefix) for prefix in personalized_phrases) or
                            any(phrase in lower_sent for phrase in personalized_phrases) or
                            any(indicator in lower_sent for indicator in personal_indicators)
                        )
                        
                        if is_personalized:
                            continue  # Skip this personalized sentence
                        
                        cleaned_sentences.append(sent)
                    
                    if cleaned_sentences:
                        clause["explanation"] = ". ".join(cleaned_sentences)
                        if not clause["explanation"].endswith("."):
                            clause["explanation"] += "."
                    else:
                        # If all sentences were removed, generate a generic one
                        clause_text = clause.get("text", "").strip()
                        if clause_text:
                            clause["explanation"] = f"This clause states: {clause_text[:100]}{'...' if len(clause_text) > 100 else ''}"
                
                if not clause.get("explanation") or not str(clause.get("explanation", "")).strip():
                    # Try to get from LLM result by position
                    if i < len(llm_result):
                        llm_clause = llm_result[i]
                        for key in ["explanation", "Explanation", "EXPLANATION", "plain_explanation"]:
                            if key in llm_clause and llm_clause[key] and str(llm_clause[key]).strip():
                                expl_text = str(llm_clause[key]).strip()
                                # Clean it before assigning
                                if any(expl_text.lower().startswith(prefix) for prefix in [
                                    "as a", "as an", "for someone", "imagine"
                                ]):
                                    continue  # Skip personalized explanations
                                clause["explanation"] = expl_text
                                break
                    # If still no explanation, generate a simple one from the text
                    if not clause.get("explanation") or not str(clause.get("explanation", "")).strip():
                        clause_text = clause.get("text", "").strip()
                        if clause_text:
                            clause["explanation"] = f"This clause states: {clause_text[:100]}{'...' if len(clause_text) > 100 else ''}"
            
            return merged
    except Exception as e:
        # If parsing fails, try to generate basic explanations
        pass
    
    # Final fallback: ensure explanation field exists for all clauses
    for c in clauses:
        if not c.get("explanation") or not str(c.get("explanation", "")).strip():
            clause_text = c.get("text", "").strip()
            if clause_text:
                c["explanation"] = f"This clause states: {clause_text[:100]}{'...' if len(clause_text) > 100 else ''}"
            else:
                c["explanation"] = "This clause requires review."
    
    return clauses

def add_risk_scores(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    llm = build_llm(MAIN_LLM_REPO, max_new_tokens=512, temperature=0.2)
    payload = json.dumps(clauses, ensure_ascii=False)
    chain = risk_prompt | llm | StrOutputParser()
    raw = chain.invoke({"explained_json": payload})
    try:
        llm_result = json.loads(_extract_json(raw))
        if isinstance(llm_result, list) and len(llm_result) > 0:
            # Merge LLM response with existing clauses
            return _merge_llm_response(clauses, llm_result, field_to_match="text")
    except Exception:
        pass
    # Fallback: ensure risk fields exist
    for c in clauses:
        c.setdefault("Risk score", "medium")
        c.setdefault("risk tag", "Legal")
    return clauses

def add_risk_text(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    llm = build_llm(MAIN_LLM_REPO, max_new_tokens=512, temperature=0.2)
    payload = json.dumps(clauses, ensure_ascii=False)
    chain = risk_reason_prompt | llm | StrOutputParser()
    raw = chain.invoke({"risked_json": payload})
    try:
        llm_result = json.loads(_extract_json(raw))
        if isinstance(llm_result, list) and len(llm_result) > 0:
            # Merge LLM response with existing clauses
            return _merge_llm_response(clauses, llm_result, field_to_match="text")
    except Exception:
        pass
    # Fallback: ensure Risk field exists
    for c in clauses:
        c.setdefault("Risk", "This clause may create exposure depending on how it is enforced.")
    return clauses

def add_scenarios(clauses: List[Dict[str, Any]], *, user_background: str = "") -> List[Dict[str, Any]]:
    llm = build_llm(MAIN_LLM_REPO, max_new_tokens=600, temperature=0.2)
    payload = json.dumps(clauses, ensure_ascii=False)
    prompt = build_scenario_prompt(user_background)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"risk_text_json": payload})
    try:
        llm_result = json.loads(_extract_json(raw))
        if isinstance(llm_result, list) and len(llm_result) > 0:
            # Merge LLM response with existing clauses
            merged = _merge_llm_response(clauses, llm_result, field_to_match="text")
            
            # Final validation: Ensure explanations and situations are completely separate
            for clause in merged:
                expl = clause.get("explanation", "").strip()
                situation = clause.get("Situation", "").strip()
                
                # If explanation contains scenario-like content, clean it
                if expl:
                    # Check if explanation contains scenario indicators
                    scenario_indicators = [
                        "keep this in mind", "remember that", "note that",
                        "as a", "as an", "for someone", "imagine",
                        "your schedule", "your studies", "your work",
                        "academic year", "study schedule", "move-in", "move-out"
                    ]
                    expl_lower = expl.lower()
                    if any(indicator in expl_lower for indicator in scenario_indicators):
                        # Remove sentences with scenario content
                        sentences = expl.replace('!', '.').replace('?', '.').split('.')
                        cleaned = [s.strip() for s in sentences 
                                 if s.strip() and not any(ind in s.lower() for ind in scenario_indicators)]
                        if cleaned:
                            clause["explanation"] = ". ".join(cleaned)
                            if not clause["explanation"].endswith("."):
                                clause["explanation"] += "."
                        else:
                            # Fallback to simple explanation
                            clause_text = clause.get("text", "").strip()
                            if clause_text:
                                clause["explanation"] = f"This clause states: {clause_text[:100]}{'...' if len(clause_text) > 100 else ''}"
            
            return merged
    except Exception:
        pass
    # Fallback: ensure Situation field exists
    for c in clauses:
        c.setdefault("Situation", "A typical dispute arises where timelines or definitions are unclear.")
    return clauses


# =====================
# Orchestration / Pipeline
# =====================
def run_pipeline(filename: str, file_bytes: bytes, *, chunk_size: int = 1500, chunk_overlap: int = 150, user_background: str = "") -> Dict[str, Any]:
    """
    Full end-to-end pipeline:
        1) Extract text
        2) Classify (IBM Granite)
        3) Split
        4) Entities
        5) Clause extraction
        6) Explanations
        7) Risk tags/scores
        8) Risk explanation
        9) Scenarios (personalized with user_background if provided)
    Returns a single dict ready for UI/JSON download.
    """
    text = extract_text_from_bytes(filename, file_bytes)
    if not text.strip():
        raise ValueError("No text extracted from the document.")

    classification = classify_document(text)
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    entities = extract_entities(text)
    clauses = extract_clauses_from_chunks(chunks)
    explained = add_plain_explanations(clauses)
    risked = add_risk_scores(explained)
    risk_text = add_risk_text(risked)
    final_clauses = add_scenarios(risk_text, user_background=user_background)

    return {
        "app": "Synaptica | clauseWise",
        "filename": filename,
        "classification": classification,
        "entities": entities.get("entities", entities),
        "chunks_count": len(chunks),
        "clauses": final_clauses,
    }


# ===============
# Utils / Parsing
# ===============
def _extract_json(s: str) -> str:
    """Best-effort to extract the first JSON object/array in a string."""
    s = s.strip()
    # Quick accept for clean JSON
    if s.startswith("{") and s.endswith("}"):
        return s
    if s.startswith("[") and s.endswith("]"):
        return s
    # Fallback: find first JSON-like region
    start = None
    stack = []
    for i, ch in enumerate(s):
        if ch in "[{" and start is None:
            start = i
            stack = [ch]
        elif start is not None:
            if ch in "[{": stack.append(ch)
            elif ch in "]}":
                if not stack:
                    break
                opener = stack.pop()
                if not stack:
                    # end index inclusive
                    end = i + 1
                    return s[start:end]
    return s  # last resort; let json.loads fail for caller


if __name__ == "__main__":
    # Simple manual test (reads a local file path for quick smoke-test)
    import sys
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        with open(path, "rb") as f:
            data = f.read()
        out = run_pipeline(os.path.basename(path), data)
        print(json.dumps(out, indent=2, ensure_ascii=False))
