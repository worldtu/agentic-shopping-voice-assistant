# graph/nodes.py
from graph.state import GraphState
from graph.router import get_router_chain
from graph.planner import get_planner_chain
from graph.retriever import retrieve_products
from graph.answerer import get_answerer_chain
import logging

logger = logging.getLogger(__name__)

def router_node(state: GraphState) -> GraphState:
    """Extract task, constraints, and safety flags using LangChain + HuggingFace."""
    
    query = state["query"]
    try:
        # Use the chain
        router_chain = get_router_chain()
        result = router_chain.invoke(query)
        
        # Update state
        state["task"] = result.task
        state["constraints"] = result.constraints.model_dump(exclude_none=True)
        state["safety_flags"] = result.safety_flags
        
        # Log
        state["step_log"].append({
            "node": "router",
            "input": query,
            "output": {
                "task": result.task,
                "constraints": state["constraints"],
                "safety_flags": result.safety_flags
            },
            "success": True
        })
        
    except Exception as e:
        # Fallback on error
        logger.error(f"Router error: {e}", exc_info=True)
        
        state["task"] = "product_search"
        state["constraints"] = {}
        state["safety_flags"] = []
        state["step_log"].append({
            "node": "router",
            "error": str(e),
            "success": False
        })
    
    return state

def planner_node(state: GraphState) -> GraphState:
    """Create retrieval plan using LLM."""
    
    try:
        # Use the chain
        planner_chain = get_planner_chain()
        
        # Prepare input for chain
        chain_input = {
            "query": state["query"],
            "task": state["task"],
            "constraints": state["constraints"]
        }
        
        # Invoke chain
        plan = planner_chain.invoke(chain_input)
        
        # Update state
        state["plan"] = plan
        
        # Log
        state["step_log"].append({
            "node": "planner",
            "input": chain_input,
            "output": plan,
            "success": True
        })
        
    except Exception as e:
        # Fallback
        state["plan"] = {
            "sources": ["private_rag"],
            "retrieval_fields": ["title", "brand", "price", "rating"],
            "comparison_criteria": ["price", "rating"],
            "filters": {}
        }
        state["step_log"].append({
            "node": "planner",
            "error": str(e),
            "success": False
        })
    
    return state

def retriever_node(state: GraphState) -> GraphState:
    """Retrieve products from private RAG using plan filters"""
    
    try:
        plan = state["plan"]
        query = state["query"]
        
        # Extract retrieval parameters from plan
        filters = plan.get("filters", {})
        retrieval_fields = plan.get("retrieval_fields", [])
        
        logger.info(f"Retrieving with filters: {filters}")
        
        # Retrieve documents
        docs = retrieve_products(
            query=query,
            filters=filters,
            retrieval_fields=retrieval_fields,
            k=5
        )
        
        # Update state
        state["retrieved_docs"] = docs
        
        # Log
        state["step_log"].append({
            "node": "retriever",
            "input": {"query": query, "filters": filters},
            "output": {
                "num_docs": len(docs),
                "top_results": [
                    {"title": d["title"], "price": d["price"]} 
                    for d in docs[:3]
                ]
            },
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Retriever error: {e}", exc_info=True)
        state["retrieved_docs"] = []
        state["step_log"].append({
            "node": "retriever",
            "error": str(e),
            "success": False
        })
    
    return state

def answerer_node(state: GraphState) -> GraphState:
    """Synthesize final answer from retrieved documents"""
    
    try:
        # Check if we have docs
        if not state.get("retrieved_docs"):
            state["answer"] = "I couldn't find any products matching your criteria. Try adjusting your search."
            state["citations"] = []
            state["step_log"].append({
                "node": "answerer",
                "output": {"answer": state["answer"]},
                "success": True
            })
            return state
        
        # Use the chain
        answerer_chain = get_answerer_chain()
        
        # Invoke chain
        result = answerer_chain.invoke(state)
        
        # Update state
        state["answer"] = result["answer"]
        state["citations"] = result["citations"]
        
        # Log
        state["step_log"].append({
            "node": "answerer",
            "input": {
                "query": state["query"],
                "num_docs": len(state["retrieved_docs"])
            },
            "output": {
                "answer": result["answer"][:100] + "...",
                "citations": result["citations"]
            },
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Answerer error: {e}", exc_info=True)
        
        # Fallback: simple summary
        docs = state.get("retrieved_docs", [])
        if docs:
            state["answer"] = f"Found {len(docs)} products. Top result: {docs[0]['title']} at ${docs[0]['price']:.2f}"
            state["citations"] = ["DOC 1"]
        else:
            state["answer"] = "No products found."
            state["citations"] = []
        
        state["step_log"].append({
            "node": "answerer",
            "error": str(e),
            "success": False
        })
    
    return state