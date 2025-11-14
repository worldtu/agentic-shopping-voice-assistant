# graph/answerer/parser.py
import re
from typing import Dict, List

def parse_answer_with_citations(text: str) -> Dict:
    """
    Parse LLM answer and extract citations.
    
    Returns:
        {
            "answer": str,  # Clean answer text
            "citations": List[str]  # List of doc IDs cited
        }
    """
    
    text = text.strip()
    
    # Extract citations from the end
    citations = []
    citation_match = re.search(r'Citations?:\s*(.+?)$', text, re.IGNORECASE)
    
    if citation_match:
        citation_text = citation_match.group(1)
        # Extract all [DOC X] patterns
        doc_refs = re.findall(r'\[DOC\s+(\d+)\]', citation_text)
        citations = [f"DOC {ref}" for ref in doc_refs]
        
        # Remove citation line from answer
        text = text[:citation_match.start()].strip()
    
    # Also find inline citations in answer
    inline_citations = re.findall(r'\[DOC\s+(\d+)\]', text)
    for ref in inline_citations:
        doc_id = f"DOC {ref}"
        if doc_id not in citations:
            citations.append(doc_id)
    
    return {
        "answer": text,
        "citations": citations
    }