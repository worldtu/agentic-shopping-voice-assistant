# graph/graph.py
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import router_node, planner_node, retriever_node, answerer_node

def create_graph():
    """Create the LangGraph workflow."""
    
    # Initialize graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("answerer", answerer_node)
    
    # Define edges
    workflow.set_entry_point("router")
    workflow.add_edge("router", "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "answerer")
    workflow.add_edge("answerer", END)
    
    # Compile
    app = workflow.compile()
    
    return app