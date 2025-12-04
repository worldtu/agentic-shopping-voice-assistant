# graph/answerer/__init__.py
from graph.models.llm import get_llm
from graph.answerer.prompts import answerer_prompt
from graph.answerer.parser import parse_answer_with_citations
from langchain_core.runnables import RunnableLambda
import json

def format_answerer_input(state_dict: dict) -> dict:
    """Format state for answerer prompt."""
    
    # Format retrieved docs
    docs = state_dict.get("retrieved_docs", [])
    docs_text = ""
    
    for i, doc in enumerate(docs, 1):
        docs_text += f"\n[DOC {i}]"
        docs_text += f"\nTitle: {doc.get('title', 'N/A')}"
        
        # Handle price - can be None from web results
        price = doc.get('price')
        if price is not None:
            docs_text += f"\nPrice: ${price:.2f}"
        else:
            docs_text += f"\nPrice: N/A"
        
        docs_text += f"\nBrand: {doc.get('brand', 'N/A')}"
        docs_text += f"\nMaterial: {doc.get('material', 'N/A')}"
        docs_text += f"\nCategory: {doc.get('category', 'N/A')}"
        
        # Get content from either 'content' or 'snippet' field
        content = doc.get('content') or doc.get('snippet', '')
        if content:
            docs_text += f"\nContent: {content[:300]}..."
        
        # Include URL if available (from web results)
        if doc.get('url'):
            docs_text += f"\nURL: {doc.get('url')}"
        
        docs_text += f"\nDoc ID: {doc.get('doc_id', 'N/A')}"
        docs_text += "\n"
    
    return {
        "query": state_dict["query"],
        "task": state_dict["task"],
        "retrieved_docs": docs_text.strip(),
        "comparison_criteria": json.dumps(
            state_dict["plan"].get("comparison_criteria", [])
        )
    }

def create_answerer_chain():
    """Create the answerer LCEL chain."""
    llm = get_llm()
    
    chain = (
        RunnableLambda(format_answerer_input)
        | answerer_prompt
        | llm
        | parse_answer_with_citations
    )
    
    return chain

# Singleton pattern
_answerer_chain = None

def get_answerer_chain():
    """Get or create answerer chain (lazy loading)."""
    global _answerer_chain
    if _answerer_chain is None:
        _answerer_chain = create_answerer_chain()
    return _answerer_chain