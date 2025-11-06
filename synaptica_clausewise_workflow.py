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
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from app_logger import get_logger, init_logging

# Initialize logging early
init_logging()
logger = get_logger(__name__)

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
            "You are an expert legal document analyzer. Extract key entities and information from the legal document. "
            "You MUST return a VALID JSON object with this EXACT structure:\n"
            "{{\n"
            "  \"entities\": {{\n"
            "    \"parties\": [\"party name 1\", \"party name 2\"],\n"
            "    \"effective_date\": \"date or null\",\n"
            "    \"termination_date\": \"date or null\",\n"
            "    \"governing_law\": \"jurisdiction or null\",\n"
            "    \"notice_periods\": \"notice requirement or null\",\n"
            "    \"liability_provisions\": \"liability terms or null\",\n"
            "    \"confidentiality_terms\": \"confidentiality requirement or null\",\n"
            "    \"payment_terms\": \"payment details or null\",\n"
            "    \"ip_rights\": \"intellectual property terms or null\"\n"
            "  }}\n"
            "}}\n\n"
            "IMPORTANT:\n"
            "- Extract real values from the document text\n"
            "- Use null (not the string \"null\") only if information is truly not found\n"
            "- For parties: extract names of individuals, companies, or organizations\n"
            "- For dates: extract in format like \"January 1, 2024\" or \"2024-01-01\" or \"2024-05-01\"\n"
            "- For termination_date: Look for phrases like 'valid from X to Y', 'from X until Y', 'expires on Y', 'ends on Y', 'to Y', 'until Y'\n"
            "- If you see date ranges like 'from May 1, 2024, to March 31, 2025', extract:\n"
            "  * effective_date: the first date (May 1, 2024)\n"
            "  * termination_date: the second date (March 31, 2025)\n"
            "- Look carefully for ALL dates mentioned - don't miss termination dates in date ranges\n"
            "- Return ONLY valid JSON, no explanations, no markdown, no code blocks\n"
            "- If you find information, include it even if incomplete"
        ),
        ("human", "Extract entities from this legal document:\n\n{doc_text}\n\nReturn the JSON object now:"),
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
            "Add a \"Risk\" field to each clause explaining the risk in 1-2 sentences (max 50 words). "
            "Match severity to risk score: HIGH=serious consequences, MEDIUM=moderate concerns, LOW=standard/minor. "
            "Be specific about consequences. Return STRICT JSON ARRAY only, no commentary."
        ),
        ("human", "Clauses:\n{risked_json}"),
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

def _simple_entity_extraction(doc_text: str) -> Dict[str, Any]:
    """Simple regex-based fallback entity extraction."""
    entities = {
        "parties": [],
        "effective_date": None,
        "termination_date": None,
        "governing_law": None,
        "notice_periods": None,
        "liability_provisions": None,
        "confidentiality_terms": None,
        "payment_terms": None,
        "ip_rights": None
    }
    
    text_lower = doc_text.lower()
    
    # Try to find dates - improved patterns
    # Look for date ranges first (e.g., "from X to Y", "valid from X to Y")
    date_range_patterns = [
        r'(?:valid\s+)?from\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})\s+to\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        r'(?:from|starting)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})\s+(?:until|to|till)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})\s+to\s+(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})\s+to\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
    ]
    
    for pattern in date_range_patterns:
        match = re.search(pattern, doc_text, re.IGNORECASE)
        if match:
            entities["effective_date"] = match.group(1).strip()
            entities["termination_date"] = match.group(2).strip()
            break
    
    # If no range found, look for individual dates
    if not entities["termination_date"]:
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        ]
        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, doc_text, re.IGNORECASE)
            dates_found.extend(matches)
        
        if dates_found:
            entities["effective_date"] = dates_found[0] if len(dates_found) > 0 else None
            entities["termination_date"] = dates_found[1] if len(dates_found) > 1 else None
        
        # Also look for termination-specific phrases
        termination_patterns = [
            r'(?:expires?|ends?|terminates?|valid until)\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:expires?|ends?|terminates?|valid until)\s+(?:on\s+)?(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(?:expires?|ends?|terminates?|valid until)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        ]
        for pattern in termination_patterns:
            match = re.search(pattern, doc_text, re.IGNORECASE)
            if match:
                entities["termination_date"] = match.group(1).strip()
                break
    
    # Try to find payment terms (look for currency symbols and amounts)
    payment_patterns = [
        r'[₹$€£]\s*\d+[,\d]*',
        r'rupees?\s+\d+[,\d]*',
        r'payment.*?(\d+\s*(days?|months?|weeks?))',
    ]
    for pattern in payment_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            entities["payment_terms"] = match.group(0)
            break
    
    # Try to find governing law
    law_patterns = [
        r'governed?\s+by\s+the\s+laws?\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'jurisdiction.*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    for pattern in law_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            entities["governing_law"] = match.group(1) if match.lastindex else match.group(0)
            break
    
    return entities


def extract_entities(doc_text: str) -> Dict[str, Any]:
    """Extract entities from document text with fallback."""
    default_entities = {
        "parties": [],
        "effective_date": None,
        "termination_date": None,
        "governing_law": None,
        "notice_periods": None,
        "liability_provisions": None,
        "confidentiality_terms": None,
        "payment_terms": None,
        "ip_rights": None
    }
    
    try:
        # Use more tokens for better extraction - increased to handle full response
        llm = build_llm(MAIN_LLM_REPO, max_new_tokens=1200, temperature=0.1)
        chain = entities_prompt | llm | StrOutputParser()
        raw = chain.invoke({"doc_text": doc_text[:12000]})  # Increased text limit
        
        # Clean the response
        raw = raw.strip()
        # Remove markdown code blocks if present
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        
        try:
            json_str = _extract_json(raw)
            result = json.loads(json_str)
            
            # Validate and normalize structure
            if isinstance(result, dict):
                if "entities" in result:
                    entities = result["entities"]
                elif all(k in result for k in default_entities.keys()):
                    # If entities are at top level
                    entities = result
                else:
                    # Try to use the whole dict as entities
                    entities = result
                
                # Ensure all required keys exist
                final_entities = default_entities.copy()
                for key in default_entities.keys():
                    if key in entities:
                        value = entities[key]
                        # Convert string "null" to actual None
                        if value == "null" or value == "None":
                            value = None
                        final_entities[key] = value
                
                # Validate that we got at least some non-null values
                non_null_count = sum(1 for v in final_entities.values() if v is not None and v != [])
                
                if non_null_count > 0:
                    return {"entities": final_entities}
                else:
                    # All values are null, try fallback
                    logger.warning("Entities: All values null; using fallback extraction")
                    fallback = _simple_entity_extraction(doc_text)
                    return {"entities": fallback, "_warning": "Used fallback extraction - LLM returned all nulls"}
            
            return {"entities": default_entities, "_error": "Invalid response format from LLM"}
            
        except json.JSONDecodeError as e:
            # Try to repair truncated JSON
            logger.error("Entities: JSON parse error; attempting repair", exc_info=True)
            
            try:
                repaired = _repair_truncated_json(_extract_json(raw))
                result = json.loads(repaired)
                
                # Process the repaired JSON
                if isinstance(result, dict):
                    if "entities" in result:
                        entities = result["entities"]
                    else:
                        entities = result
                    
                    final_entities = default_entities.copy()
                    for key in default_entities.keys():
                        if key in entities:
                            value = entities[key]
                            if value == "null" or value == "None":
                                value = None
                            final_entities[key] = value
                    
                    non_null_count = sum(1 for v in final_entities.values() if v is not None and v != [])
                    if non_null_count > 0:
                        logger.info("Entities: Repaired and parsed truncated JSON successfully")
                        return {"entities": final_entities, "_warning": "JSON was truncated but repaired"}
            except Exception as repair_error:
                logger.error("Entities: JSON repair failed", exc_info=True)
            
            # Try to extract partial data from incomplete JSON
            try:
                partial_entities = _extract_partial_entities(raw)
                non_null_count = sum(1 for v in partial_entities.values() if v is not None and v != [])
                if non_null_count > 0:
                    logger.info("Entities: Extracted partial fields from incomplete JSON")
                    return {"entities": partial_entities, "_warning": "Extracted partial data from incomplete JSON"}
            except Exception as partial_error:
                logger.error("Entities: Partial extraction failed", exc_info=True)
            
            # Final fallback: regex extraction from document
            logger.warning("Entities: Using regex-based fallback extraction from document text")
            fallback = _simple_entity_extraction(doc_text)
            return {"entities": fallback, "_error": f"JSON parse error, used fallback: {str(e)}"}
            
    except StopIteration as e:
        logger.error("Entities: StopIteration from API (truncated response)", exc_info=True)
        # Try fallback
        fallback = _simple_entity_extraction(doc_text)
        return {"entities": fallback, "_error": "API returned incomplete response, used fallback extraction"}
        
    except Exception as e:
        logger.error("Entities: Extraction failed", exc_info=True)
        # Try fallback extraction
        fallback = _simple_entity_extraction(doc_text)
        return {"entities": fallback, "_error": str(e)}
    
    # Final fallback
    return {"entities": default_entities, "_error": "Unknown error"}

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
            # Also clean explanations to remove redundant prefixes and personalized content
            for i, clause in enumerate(merged):
                expl = clause.get("explanation", "").strip()
                
                # Remove redundant prefixes like "This clause states:"
                if expl:
                    redundant_prefixes = [
                        "this clause states:",
                        "this clause states",
                        "the clause states:",
                        "the clause states",
                        "clause states:",
                        "clause states",
                        "this states:",
                        "this states",
                    ]
                    
                    expl_lower = expl.lower()
                    for prefix in redundant_prefixes:
                        if expl_lower.startswith(prefix):
                            expl = expl[len(prefix):].strip()
                            # Remove leading colon if present
                            if expl.startswith(":"):
                                expl = expl[1:].strip()
                            break
                
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
                        
                        # Remove redundant prefixes from individual sentences
                        redundant_prefixes = [
                            "this clause states:",
                            "this clause states",
                            "the clause states:",
                            "the clause states",
                            "clause states:",
                            "clause states",
                            "this states:",
                            "this states",
                        ]
                        for prefix in redundant_prefixes:
                            if lower_sent.startswith(prefix):
                                sent = sent[len(prefix):].strip()
                                if sent.startswith(":"):
                                    sent = sent[1:].strip()
                                lower_sent = sent.lower()
                                break
                        
                        if not sent:  # Skip if sentence became empty after prefix removal
                            continue
                        
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
                        # If all sentences were removed, generate a generic one (without redundant prefix)
                        clause_text = clause.get("text", "").strip()
                        if clause_text:
                            clause["explanation"] = clause_text[:100] + ('...' if len(clause_text) > 100 else '')
                
                if not clause.get("explanation") or not str(clause.get("explanation", "")).strip():
                    # Try to get from LLM result by position
                    if i < len(llm_result):
                        llm_clause = llm_result[i]
                        for key in ["explanation", "Explanation", "EXPLANATION", "plain_explanation"]:
                            if key in llm_clause and llm_clause[key] and str(llm_clause[key]).strip():
                                expl_text = str(llm_clause[key]).strip()
                                
                                # Clean redundant prefixes before assigning
                                expl_lower = expl_text.lower()
                                redundant_prefixes_local = [
                                    "this clause states:",
                                    "this clause states",
                                    "the clause states:",
                                    "the clause states",
                                    "clause states:",
                                    "clause states",
                                    "this states:",
                                    "this states",
                                ]
                                for prefix in redundant_prefixes_local:
                                    if expl_lower.startswith(prefix):
                                        expl_text = expl_text[len(prefix):].strip()
                                        if expl_text.startswith(":"):
                                            expl_text = expl_text[1:].strip()
                                        break
                                
                                # Skip personalized explanations
                                if any(expl_text.lower().startswith(prefix) for prefix in [
                                    "as a", "as an", "for someone", "imagine"
                                ]):
                                    continue
                                clause["explanation"] = expl_text
                                break
                    # If still no explanation, generate a simple one from the text (without redundant prefix)
                    if not clause.get("explanation") or not str(clause.get("explanation", "")).strip():
                        clause_text = clause.get("text", "").strip()
                        if clause_text:
                            clause["explanation"] = clause_text[:100] + ('...' if len(clause_text) > 100 else '')
            
            return merged
    except Exception as e:
        # If parsing fails, try to generate basic explanations
        pass
    
    # Final fallback: ensure explanation field exists for all clauses (without redundant prefix)
    for c in clauses:
        if not c.get("explanation") or not str(c.get("explanation", "")).strip():
            clause_text = c.get("text", "").strip()
            if clause_text:
                c["explanation"] = clause_text[:100] + ('...' if len(clause_text) > 100 else '')
            else:
                c["explanation"] = "This clause requires review."
    
    return clauses

def add_risk_scores(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not clauses:
        return clauses
    
    try:
        # Increase token limit to prevent truncation
        llm = build_llm(MAIN_LLM_REPO, max_new_tokens=1500, temperature=0.2)
        # Limit payload size if too large
        payload = json.dumps(clauses, ensure_ascii=False)
        if len(payload) > 8000:
            # Process in smaller batches if payload is too large
            logger.warning("Risk scoring: payload too large (%d chars); using fallback for batch", len(payload))
            for c in clauses:
                c.setdefault("Risk score", "medium")
                c.setdefault("risk tag", "Legal")
            return clauses
        
        chain = risk_prompt | llm | StrOutputParser()
        raw = chain.invoke({"explained_json": payload})
        
        # Clean the response
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        
        try:
            json_str = _extract_json(raw)
            llm_result = json.loads(json_str)
            
            if isinstance(llm_result, list) and len(llm_result) > 0:
                # Merge LLM response with existing clauses
                merged = _merge_llm_response(clauses, llm_result, field_to_match="text")
                # Validate that risk scores were added
                if any(c.get("Risk score") for c in merged):
                    return merged
                else:
                    logger.warning("Risk scoring: LLM returned without risk fields; using fallback")
            else:
                logger.error("Risk scoring: Invalid response type: %s", type(llm_result))
        except json.JSONDecodeError as e:
            logger.error("Risk scoring: JSON parse error; attempting repair", exc_info=True)
            
            # Try to repair truncated JSON with improved method
            try:
                json_str = _extract_json(raw)
                repaired = _repair_truncated_json(json_str)
                llm_result = json.loads(repaired)
                
                if isinstance(llm_result, list) and len(llm_result) > 0:
                    merged = _merge_llm_response(clauses, llm_result, field_to_match="text")
                    # Check if we got at least some risk scores
                    risk_count = sum(1 for c in merged if c.get("Risk score"))
                    if risk_count > 0:
                        logger.info("Risk scoring: Repaired truncated JSON; got %d scores", risk_count)
                        # Fill in missing risk scores for remaining clauses
                        for c in merged:
                            if not c.get("Risk score"):
                                c.setdefault("Risk score", "medium")
                                c.setdefault("risk tag", "Legal")
                        return merged
            except Exception as repair_error:
                logger.error("Risk scoring: JSON repair failed", exc_info=True)
                # Try extracting partial risk scores from incomplete JSON
                try:
                    partial_risks = _extract_partial_risk_scores(raw, clauses)
                    if partial_risks:
                        logger.info("Risk scoring: Extracted %d partial scores from incomplete JSON", len(partial_risks))
                        return partial_risks
                except Exception as partial_error:
                    logger.error("Risk scoring: Partial extraction failed", exc_info=True)
            
            logger.debug("Risk scoring: Raw response preview: %s", raw[:500])
            
        except Exception as e:
            logger.error("Risk scoring: Merge error", exc_info=True)
    except StopIteration as e:
        logger.error("Risk scoring: StopIteration (truncated response)", exc_info=True)
    except Exception as e:
        logger.error("Risk scoring: LLM call failed", exc_info=True)
    
    # Fallback: ensure risk fields exist
    logger.warning("Risk scoring: Using fallback scores (medium/Legal)")
    for c in clauses:
        c.setdefault("Risk score", "medium")
        c.setdefault("risk tag", "Legal")
    return clauses

def add_risk_text(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not clauses:
        return clauses
    
    # Create a minimal payload with only essential fields to reduce size
    minimal_clauses = []
    for clause in clauses:
        minimal = {
            "id": clause.get("id"),
            "text": clause.get("text", "")[:300],  # Truncate long texts
            "title": clause.get("title", ""),
            "Risk score": clause.get("Risk score", "medium"),
            "risk tag": clause.get("risk tag", "Legal")
        }
        minimal_clauses.append(minimal)
    
    payload = json.dumps(minimal_clauses, ensure_ascii=False)
    
    # Process in batches if payload is too large
    if len(payload) > 6000:
        # Split into smaller batches
        batch_size = max(3, len(clauses) // 3)  # Process in 3 batches
        result = []
        
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i:i + batch_size]
            batch_minimal = []
            for clause in batch:
                minimal = {
                    "id": clause.get("id"),
                    "text": clause.get("text", "")[:300],
                    "title": clause.get("title", ""),
                    "Risk score": clause.get("Risk score", "medium"),
                    "risk tag": clause.get("risk tag", "Legal")
                }
                batch_minimal.append(minimal)
            
            batch_payload = json.dumps(batch_minimal, ensure_ascii=False)
            batch_result = _process_risk_text_batch(batch, batch_payload)
            result.extend(batch_result)
        
        return result
    
    # Process all at once if payload is small enough
    return _process_risk_text_batch(clauses, payload)


def _process_risk_text_batch(clauses: List[Dict[str, Any]], payload: str) -> List[Dict[str, Any]]:
    """Process a batch of clauses to add risk text."""
    try:
        # Increase token limit for better responses
        llm = build_llm(MAIN_LLM_REPO, max_new_tokens=800, temperature=0.2)
        chain = risk_reason_prompt | llm | StrOutputParser()
        raw = chain.invoke({"risked_json": payload})
        
        # Clean the response
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        
        try:
            json_str = _extract_json(raw)
            llm_result = json.loads(json_str)
            
            if isinstance(llm_result, list) and len(llm_result) > 0:
                # Merge LLM response with existing clauses
                merged = _merge_llm_response(clauses, llm_result, field_to_match="text")
                # Validate that risk text was added
                if any(c.get("Risk") for c in merged):
                    return merged
                else:
                    print("Risk text not found in LLM response")
        except json.JSONDecodeError as e:
            logger.error("Risk text: JSON parse error; attempting repair", exc_info=True)
            
            # Try to repair truncated JSON
            try:
                repaired = _repair_truncated_json(_extract_json(raw))
                llm_result = json.loads(repaired)
                
                if isinstance(llm_result, list) and len(llm_result) > 0:
                    merged = _merge_llm_response(clauses, llm_result, field_to_match="text")
                    risk_count = sum(1 for c in merged if c.get("Risk"))
                    if risk_count > 0:
                        logger.info("Risk text: Repaired truncated JSON; got %d texts", risk_count)
                        # Fill in missing risk text
                        for c in merged:
                            if not c.get("Risk"):
                                c.setdefault("Risk", _generate_default_risk_text(c.get("Risk score", "medium")))
                        return merged
            except Exception as repair_error:
                logger.error("Risk text: JSON repair failed", exc_info=True)
        
    except StopIteration as e:
        logger.error("Risk text: StopIteration", exc_info=True)
    except Exception as e:
        logger.error("Risk text: processing failed", exc_info=True)
    
    # Fallback: generate default risk text based on risk score
    for c in clauses:
        if not c.get("Risk"):
            c["Risk"] = _generate_default_risk_text(c.get("Risk score", "medium"))
    return clauses


def _generate_default_risk_text(risk_score: str) -> str:
    """Generate a default risk text based on risk score."""
    risk_lower = (risk_score or "medium").lower()
    if risk_lower == "high":
        return "This clause creates significant exposure that could result in substantial financial loss or legal liability."
    elif risk_lower == "medium":
        return "This clause may create moderate exposure or inconvenience depending on how it is enforced."
    else:
        return "This clause is generally standard and creates minimal exposure."

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
                            # Fallback to simple explanation (without redundant prefix)
                            clause_text = clause.get("text", "").strip()
                            if clause_text:
                                clause["explanation"] = clause_text[:100] + ('...' if len(clause_text) > 100 else '')
            
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
def _enhance_entities_from_clauses(entities: Dict[str, Any], clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Enhance entities by extracting missing information from clauses."""
    if not clauses:
        return entities
    
    # Combine all clause texts for better extraction
    all_clause_text = " ".join([c.get("text", "") for c in clauses])
    
    # If termination_date is missing, try to extract from clauses
    if not entities.get("termination_date") or entities.get("termination_date") == "null":
        # Look for date ranges in clause text
        date_range_patterns = [
            r'(?:valid\s+)?from\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})\s+to\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:from|starting)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})\s+(?:until|to|till)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})\s+to\s+(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})\s+to\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        ]
        
        for pattern in date_range_patterns:
            match = re.search(pattern, all_clause_text, re.IGNORECASE)
            if match:
                if not entities.get("effective_date"):
                    entities["effective_date"] = match.group(1).strip()
                entities["termination_date"] = match.group(2).strip()
                logger.info("Entities: Enhanced termination_date from clauses: %s", entities["termination_date"])
                break
        
        # Also check for termination-specific phrases in clauses
        if not entities.get("termination_date") or entities.get("termination_date") == "null":
            termination_patterns = [
                r'(?:expires?|ends?|terminates?|valid until)\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
                r'(?:expires?|ends?|terminates?|valid until)\s+(?:on\s+)?(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
                r'(?:expires?|ends?|terminates?|valid until)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            ]
            for pattern in termination_patterns:
                match = re.search(pattern, all_clause_text, re.IGNORECASE)
                if match:
                    entities["termination_date"] = match.group(1).strip()
                    logger.info("Entities: Enhanced termination_date from clause patterns: %s", entities["termination_date"])
                    break
    
    return entities


def run_pipeline(filename: str, file_bytes: bytes, *, chunk_size: int = 1500, chunk_overlap: int = 150, user_background: str = "") -> Dict[str, Any]:
    """
    Full end-to-end pipeline:
        1) Extract text
        2) Classify (IBM Granite)
        3) Split
        4) Entities
        5) Clause extraction
        6) Enhance entities from clauses (fill missing dates, etc.)
        7) Explanations
        8) Risk tags/scores
        9) Risk explanation
        10) Scenarios (personalized with user_background if provided)
    Returns a single dict ready for UI/JSON download.
    """
    text = extract_text_from_bytes(filename, file_bytes)
    if not text.strip():
        raise ValueError("No text extracted from the document.")

    classification = classify_document(text)
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    entities = extract_entities(text)
    clauses = extract_clauses_from_chunks(chunks)
    
    # Enhance entities with information from clauses (especially dates)
    entities_dict = entities.get("entities", entities)
    if isinstance(entities_dict, dict):
        entities_dict = _enhance_entities_from_clauses(entities_dict, clauses)
        entities = {"entities": entities_dict}
    
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


def _repair_truncated_json(json_str: str) -> str:
    """Try to repair truncated JSON by closing incomplete strings and objects."""
    if not json_str or not json_str.strip():
        return json_str
    
    json_str = json_str.strip()
    original = json_str
    
    # Track state while parsing
    in_string = False
    escape_next = False
    depth_braces = 0
    depth_brackets = 0
    last_valid_pos = -1
    
    # Find the last valid position before truncation
    for i, char in enumerate(json_str):
        if escape_next:
            escape_next = False
            last_valid_pos = i
            continue
        
        if char == '\\':
            escape_next = True
            last_valid_pos = i
            continue
        
        if char == '"':
            in_string = not in_string
            last_valid_pos = i
            continue
        
        if in_string:
            last_valid_pos = i
            continue
        
        # Outside string
        if char == '{':
            depth_braces += 1
            last_valid_pos = i
        elif char == '}':
            depth_braces -= 1
            last_valid_pos = i
        elif char == '[':
            depth_brackets += 1
            last_valid_pos = i
        elif char == ']':
            depth_brackets -= 1
            last_valid_pos = i
        else:
            last_valid_pos = i
    
    # If we're inside a string, we need to close it
    if in_string:
        # Find where the string started (go backwards to find the opening quote)
        # Look for the last complete key-value pair or array element
        # Try to find a reasonable cut point
        cut_pos = last_valid_pos + 1
        
        # Go backwards from truncation to find where we can safely cut
        # Look for the start of the current incomplete field
        # Try to find a pattern like: "field": "incomplete text
        # We want to close it at a reasonable point
        
        # Find the last complete key-value separator before truncation
        last_colon = json_str.rfind(':', 0, cut_pos)
        if last_colon > 0:
            # Check if there's an opening quote after the colon
            after_colon = json_str[last_colon+1:cut_pos].strip()
            if after_colon.startswith('"'):
                # We're in a string value, need to close it
                # Find the opening quote of this string
                string_start = last_colon + 1 + after_colon.index('"')
                # Close the string at the current position
                json_str = json_str[:cut_pos] + '"'
                last_valid_pos = cut_pos
            else:
                # Not in a string value, might be in a key name
                json_str = json_str[:cut_pos] + '"'
        else:
            # No colon found, just close the string
            json_str = json_str[:cut_pos] + '"'
    
    # Now close any incomplete objects/arrays
    # We need to close in reverse order: first the current object, then parent objects
    result = json_str[:last_valid_pos + 1 + (1 if in_string else 0)]
    
    # Close the current object if we're inside one
    if depth_braces > 0:
        result += '}' * depth_braces
    
    # Close the array if we're inside one
    if depth_brackets > 0:
        result += ']' * depth_brackets
    
    return result


def _extract_partial_entities(incomplete_json: str) -> Dict[str, Any]:
    """Try to extract partial entities from incomplete JSON."""
    entities = {
        "parties": [],
        "effective_date": None,
        "termination_date": None,
        "governing_law": None,
        "notice_periods": None,
        "liability_provisions": None,
        "confidentiality_terms": None,
        "payment_terms": None,
        "ip_rights": None
    }
    
    # Try regex extraction from the incomplete JSON string
    # Extract parties array - handle both complete and incomplete arrays
    # First try complete array
    parties_match = re.search(r'"parties"\s*:\s*\[(.*?)\]', incomplete_json, re.DOTALL)
    if not parties_match:
        # Try incomplete array (no closing bracket)
        parties_match = re.search(r'"parties"\s*:\s*\[(.*)', incomplete_json, re.DOTALL)
    
    if parties_match:
        parties_str = parties_match.group(1)
        # Extract quoted strings - handle incomplete strings at the end
        party_names = re.findall(r'"([^"]+)"', parties_str)
        # Also try to extract last incomplete string if any
        if '"' in parties_str and not parties_str.strip().endswith('"'):
            # Try to extract up to the last quote
            last_quote = parties_str.rfind('"')
            if last_quote > 0:
                before_quote = parties_str[:last_quote]
                # Extract complete strings before the incomplete one
                complete_parties = re.findall(r'"([^"]+)"', before_quote)
                party_names = complete_parties
        if party_names:
            entities["parties"] = party_names
    
    # Extract dates - handle incomplete strings
    # Pattern for complete dates
    date_pattern_complete = r'"(effective_date|termination_date)"\s*:\s*"([^"]+)"'
    date_matches = re.findall(date_pattern_complete, incomplete_json)
    for key, value in date_matches:
        if value and len(value) > 3:
            entities[key] = value
    
    # Pattern for incomplete dates (truncated strings)
    date_pattern_incomplete = r'"(effective_date|termination_date)"\s*:\s*"([^"]+)'
    date_matches_incomplete = re.findall(date_pattern_incomplete, incomplete_json)
    for key, value in date_matches_incomplete:
        # Only use if we haven't already found a complete value
        if not entities.get(key) and value and len(value) > 3:
            # Take what we have, even if incomplete
            entities[key] = value
    
    # Extract other string fields - handle both complete and incomplete
    string_fields = ["governing_law", "notice_periods", "liability_provisions", 
                     "confidentiality_terms", "payment_terms", "ip_rights"]
    for field in string_fields:
        # Try complete string first
        pattern_complete = f'"{field}"\\s*:\\s*"([^"]+)"'
        match = re.search(pattern_complete, incomplete_json)
        if match:
            value = match.group(1)
            if value and len(value) > 2:
                entities[field] = value
        else:
            # Try incomplete string
            pattern_incomplete = f'"{field}"\\s*:\\s*"([^"]+)'
            match = re.search(pattern_incomplete, incomplete_json)
            if match:
                value = match.group(1)
                # Only use if meaningful length (at least 3 chars)
                if value and len(value) > 2:
                    entities[field] = value
    
    return entities


def _extract_partial_risk_scores(incomplete_json: str, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract partial risk scores from incomplete JSON array."""
    result = []
    
    # Try to extract complete objects from the array
    # Pattern: { "id": ..., "Risk score": ..., ... }
    # We'll try to extract any complete risk score entries
    
    # Split by objects (look for complete { ... } patterns)
    # This is a simple approach - extract risk scores that are complete
    risk_score_pattern = r'"Risk score"\s*:\s*"(high|medium|low)"'
    risk_tag_pattern = r'"risk tag"\s*:\s*"([^"]+)"'
    
    # Find all risk scores and tags in the incomplete JSON
    risk_scores = re.findall(risk_score_pattern, incomplete_json, re.IGNORECASE)
    risk_tags = re.findall(risk_tag_pattern, incomplete_json, re.IGNORECASE)
    
    # Try to match by ID if possible
    id_pattern = r'"id"\s*:\s*"?(\d+)"?'
    ids = re.findall(id_pattern, incomplete_json)
    
    # Build result by matching clauses with extracted risk data
    for i, clause in enumerate(clauses):
        clause_copy = clause.copy()
        
        # Try to match by ID first
        clause_id = str(clause.get("id", i + 1))
        if clause_id in ids:
            idx = ids.index(clause_id)
            if idx < len(risk_scores):
                clause_copy["Risk score"] = risk_scores[idx].lower()
            if idx < len(risk_tags):
                clause_copy["risk tag"] = risk_tags[idx]
        else:
            # Match by position
            if i < len(risk_scores):
                clause_copy["Risk score"] = risk_scores[i].lower()
            if i < len(risk_tags):
                clause_copy["risk tag"] = risk_tags[i]
        
        # Ensure defaults if not found
        clause_copy.setdefault("Risk score", "medium")
        clause_copy.setdefault("risk tag", "Legal")
        
        result.append(clause_copy)
    
    return result if result else clauses


if __name__ == "__main__":
    # Simple manual test (reads a local file path for quick smoke-test)
    import sys
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        with open(path, "rb") as f:
            data = f.read()
        out = run_pipeline(os.path.basename(path), data)
        print(json.dumps(out, indent=2, ensure_ascii=False))
