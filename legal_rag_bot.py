# legal_rag_bot.py
"""
Legal RAG Bot - Simplified Retrieval-Augmented Generation for Legal Queries
----------------------------------------------------------------------------
Uses IBM Granite model to answer legal queries based on processed clauses and documents.
Lightweight implementation - no heavy vector stores, just simple text matching.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain imports (minimal)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Hugging Face LLMs
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from app_logger import get_logger, init_logging

# Initialize logging for RAG bot
init_logging()
logger = get_logger(__name__)


# =============================
# Configuration
# =============================
# Use the same stable model as the main workflow for better compatibility
RAG_LLM_REPO = os.getenv("RAG_LLM_REPO", "mistralai/Mixtral-8x7B-Instruct-v0.1")


def _require_hf_token():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is not set. Please export your Hugging Face token."
        )


def build_rag_llm(max_new_tokens: int = 512, temperature: float = 0.3):
    """Build conversational LLM for RAG bot - using stable Mixtral model."""
    _require_hf_token()
    
    # Use conversational task for better stability with chat models
    base_llm = HuggingFaceEndpoint(
        repo_id=RAG_LLM_REPO,
        task="conversational",  # Use conversational task for chat models
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        streaming=False,
        timeout=120,  # Increased timeout for stability
    )
    
    # Wrap with ChatHuggingFace for proper chat formatting
    try:
        return ChatHuggingFace(llm=base_llm)
    except Exception as e:
        logger.warning(f"ChatHuggingFace wrapper failed, trying text-generation task: {e}")
        # Fallback to text-generation task
        try:
            base_llm = HuggingFaceEndpoint(
                repo_id=RAG_LLM_REPO,
                task="text-generation",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                streaming=False,
                timeout=120,
            )
            return ChatHuggingFace(llm=base_llm)
        except Exception as e2:
            logger.error(f"Failed to build RAG LLM with both methods: {e2}")
            raise


# =============================
# Simple Knowledge Base (no vector store)
# =============================
def build_knowledge_base(analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build simple knowledge base from processed analysis result.
    Returns list of dicts with text content and metadata - no heavy processing.
    """
    kb_items = []
    
    # Add document metadata
    filename = analysis_result.get("filename", "Unknown")
    classification = analysis_result.get("classification", {}).get("classification", "Unknown")
    entities = analysis_result.get("entities", {})
    
    # Document summary
    doc_summary = f"""Document: {filename}
Classification: {classification}
Parties: {', '.join(entities.get('parties', [])) if entities.get('parties') else 'Not specified'}
Effective Date: {entities.get('effective_date', 'Not specified')}
Termination Date: {entities.get('termination_date', 'Not specified')}
Governing Law: {entities.get('governing_law', 'Not specified')}
"""
    kb_items.append({
        "content": doc_summary.strip(),
        "type": "document_metadata",
        "source": filename
    })
    
    # Add entity information
    entity_text = []
    for key, value in entities.items():
        if value and key != 'parties':
            entity_text.append(f"{key.replace('_', ' ').title()}: {value}")
    if entity_text:
        kb_items.append({
            "content": "\n".join(entity_text),
            "type": "entities",
            "source": filename
        })
    
    # Add clauses with key information
    clauses = analysis_result.get("clauses", [])
    for clause in clauses:
        clause_id = clause.get("id", "Unknown")
        title = clause.get("title", "Untitled")
        clause_type = clause.get("type", "Other")
        text = clause.get("text", "")
        explanation = clause.get("explanation", "")
        risk_score = clause.get("Risk score", "low")
        risk_tag = clause.get("risk tag", "Legal")
        risk_reason = clause.get("Risk", "")
        
        # Simplified clause content - only essential info
        clause_content = f"""Clause {clause_id}: {title}
Type: {clause_type}
Text: {text}
Explanation: {explanation}
Risk: {risk_score} ({risk_tag}) - {risk_reason}"""
        
        kb_items.append({
            "content": clause_content.strip(),
            "type": "clause",
            "clause_id": str(clause_id),
            "title": title,
            "clause_type": clause_type,
            "risk_score": risk_score,
            "source": filename
        })
    
    return kb_items


# =============================
# Simple Text Matching (lightweight)
# =============================
def simple_text_match(query: str, kb_items: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Simple keyword-based matching - very lightweight, no ML models needed.
    Returns top_k most relevant items based on keyword overlap.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_items = []
    for item in kb_items:
        content_lower = item["content"].lower()
        content_words = set(content_lower.split())
        
        # Count matching words
        matches = len(query_words.intersection(content_words))
        # Bonus for exact phrase matches
        if query_lower in content_lower:
            matches += 5
        
        if matches > 0:
            scored_items.append((matches, item))
    
    # Sort by score and return top_k
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored_items[:top_k]]


# =============================
# Simplified RAG Bot
# =============================
class LegalRAGBot:
    """Lightweight Legal RAG Bot - no heavy dependencies."""
    
    def __init__(self, analysis_result: Optional[Dict[str, Any]] = None):
        self.llm = None
        self.analysis_result = analysis_result
        self.kb_items = None
        self.llm_error = None
        
        if analysis_result:
            self.build_knowledge_base(analysis_result)
    
    def build_knowledge_base(self, analysis_result: Dict[str, Any]):
        """Build simple knowledge base - no vector stores."""
        self.analysis_result = analysis_result
        self.kb_items = build_knowledge_base(analysis_result)
        
        # Initialize LLM with stable conversational model
        try:
            self.llm = build_rag_llm(max_new_tokens=512, temperature=0.3)
            logger.info("RAG bot LLM initialized successfully with %s", RAG_LLM_REPO)
        except Exception as e:
            # Store error for later use
            self.llm_error = str(e)
            self.llm = None
            logger.error("RAG bot LLM initialization failed: %s", str(e), exc_info=True)
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context using simple text matching."""
        if not self.kb_items:
            return "No knowledge base available. Please process a document first."
        
        # Get top 3 most relevant items
        relevant_items = simple_text_match(query, self.kb_items, top_k=3)
        
        if not relevant_items:
            # If no matches, return all items (fallback)
            relevant_items = self.kb_items[:5]  # Limit to 5 items max
        
        # Format context simply
        context_parts = []
        for i, item in enumerate(relevant_items, 1):
            item_type = item.get("type", "unknown")
            source = item.get("source", "Unknown")
            context_parts.append(f"[{i}] {item_type} from {source}:\n{item['content']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Answer a legal query using simplified RAG."""
        if not self.analysis_result or not self.kb_items:
            return {
                "answer": "No document has been processed yet. Please upload and analyze a document first.",
                "context_sources": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if LLM is initialized
        if not self.llm:
            error_msg = getattr(self, 'llm_error', 'Unknown error during LLM initialization')
            return {
                "answer": f"RAG bot initialization failed: {error_msg}. Please check your HUGGINGFACEHUB_API_TOKEN and try re-running the analysis.",
                "context_sources": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if knowledge base is empty
        if not self.kb_items or len(self.kb_items) == 0:
            return {
                "answer": "The processed document has no extractable information. Please ensure the document was processed successfully and contains clauses or entities.",
                "context_sources": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Retrieve relevant context
        try:
            context = self.retrieve_context(query)
            
            if not context or context.strip() == "":
                return {
                    "answer": "I couldn't find any relevant information in the document to answer your question. Please try a different question or ensure the document contains the information you're looking for.",
                    "context_sources": [],
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "answer": f"Error retrieving context: {str(e)}. Please try again.",
                "context_sources": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Simplified prompt - shorter for faster processing
        # Format prompt as a simple string for better compatibility
        prompt_text = f"""You are a legal assistant. Answer questions based ONLY on the provided context.
Be concise and accurate. Cite specific clauses or information from the context.
If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            # Use ChatPromptTemplate for proper conversational format with Mixtral
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful legal assistant. Answer questions based ONLY on the provided context from a legal document. Be concise, accurate, and cite specific clauses when relevant. If the context doesn't contain the answer, say so clearly."),
                ("human", "Context from the document:\n{context}\n\nQuestion: {query}\n\nProvide a clear answer based on the context above:")
            ])
            
            # Build chain - use StrOutputParser for text extraction
            chain = rag_prompt | self.llm | StrOutputParser()
            
            # Invoke with proper error handling
            logger.info("RAG: Processing query: %s", query[:100] if len(query) > 100 else query)
            try:
                answer = chain.invoke({"query": query, "context": context})
                logger.debug("RAG: Received response from LLM")
            except (StopIteration, RuntimeError, ValueError) as e:
                logger.warning("RAG: Chain invoke failed, trying direct LLM call: %s", str(e))
                # Retry with direct LLM call
                try:
                    messages = rag_prompt.format_messages(query=query, context=context)
                    if isinstance(self.llm, ChatHuggingFace):
                        response = self.llm.invoke(messages)
                        if hasattr(response, 'content'):
                            answer = response.content
                        elif hasattr(response, 'text'):
                            answer = response.text
                        else:
                            answer = str(response)
                        logger.debug("RAG: Got response from direct LLM call")
                    else:
                        # Fallback: use direct prompt
                        answer = self.llm.invoke(prompt_text)
                        if hasattr(answer, 'content'):
                            answer = answer.content
                        elif not isinstance(answer, str):
                            answer = str(answer)
                except Exception as retry_error:
                    logger.error("RAG: Direct LLM call also failed", exc_info=True)
                    raise e from retry_error
            
            # Extract text from response if it's a message object
            if hasattr(answer, 'content'):
                answer = answer.content
            elif not isinstance(answer, str):
                answer = str(answer)
            
            # Clean up the answer
            if answer:
                # Remove any prompt remnants that might have been included
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
                answer = answer.strip()
            
            if not answer:
                raise ValueError("Empty response from LLM")
            
            # Simple source extraction
            sources = []
            try:
                relevant_items = simple_text_match(query, self.kb_items, top_k=2)
                for item in relevant_items:
                    if item.get("clause_id"):
                        sources.append({
                            "type": item.get("type", "clause"),
                            "clause_id": item.get("clause_id"),
                            "title": item.get("title", "Untitled"),
                            "source": item.get("source", "Unknown")
                        })
            except Exception:
                pass  # Sources are optional
            
            return {
                "answer": answer,
                "context_sources": sources,
                "timestamp": datetime.now().isoformat()
            }
        except StopIteration as e:
            # Handle StopIteration specifically - often caused by empty API responses
            logger.error("RAG: StopIteration after retries", exc_info=True)
            return {
                "answer": "The API returned an incomplete response. This might be due to rate limiting or API issues. Please try again in a moment, or check your HUGGINGFACEHUB_API_TOKEN.",
                "context_sources": [],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            error_str = str(e) if str(e) else type(e).__name__
            error_type = type(e).__name__
            
            # Provide more specific error messages
            if "timeout" in error_str.lower() or "Timeout" in error_type:
                error_msg = "The request timed out. The API might be slow or overloaded. Please try again."
            elif "authentication" in error_str.lower() or "token" in error_str.lower():
                error_msg = f"Authentication error: {error_str}. Please check your HUGGINGFACEHUB_API_TOKEN."
            elif "rate limit" in error_str.lower():
                error_msg = "Rate limit exceeded. Please wait a moment and try again."
            else:
                error_msg = f"Error: {error_str}. This might be due to API issues. Please check your HUGGINGFACEHUB_API_TOKEN and try again."
            
            logger.error("RAG: Query processing failed: %s", error_msg)
            return {
                "answer": error_msg,
                "context_sources": [],
                "timestamp": datetime.now().isoformat()
            }


# =============================
# Convenience Functions
# =============================
def create_rag_bot(analysis_result: Dict[str, Any]) -> LegalRAGBot:
    """Create a new RAG bot instance with the given analysis result."""
    return LegalRAGBot(analysis_result)


def is_rag_available() -> bool:
    """RAG is always available - no heavy dependencies required."""
    return True
