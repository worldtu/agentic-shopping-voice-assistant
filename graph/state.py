# graph/state.py
from typing import TypedDict, List, Annotated
from langgraph.graph import add_messages
import operator

class GraphState(TypedDict):
    # Input
    query: str
    
    # Router outputs
    task: str  # One of: "product_search", "comparison", "recommendation", "availability_check"
    constraints: dict  # Contains: product, min_price, max_price, material, brand
    safety_flags: List[str]  # Filtered to valid flags only
    
    # Planner outputs
    plan: dict # Contains: sources, retrieval_fields, comparison_criteria, filters
    
    # Retriever outputs
    retrieved_docs: List[dict]
    
    # Answerer outputs
    answer: str
    citations: List[str]
    
    # Logging
    # step_log: Annotated[List[dict], operator.add]  # Accumulate logs
    step_log: List[dict] 
