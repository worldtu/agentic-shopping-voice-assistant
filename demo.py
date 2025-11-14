#!/usr/bin/env python3
"""
Demo script for complete agentic pipeline
Shows router -> planner -> retriever -> answerer flow
"""

from graph.graph import create_graph
import json

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def demo_query(graph, query):
    """Run and display results for a single query"""
    
    print_section(f"Query: {query}")
    
    result = graph.invoke({
        "query": query,
        "step_log": []
    })
    
    # Show router output
    print("🔀 ROUTER OUTPUT:")
    print(f"  Task: {result['task']}")
    print(f"  Constraints: {json.dumps(result['constraints'], indent=6)}")
    if result.get('safety_flags'):
        print(f"  ⚠️  Safety Flags: {result['safety_flags']}")
    
    # Show planner output
    print("\n📋 PLANNER OUTPUT:")
    plan = result['plan']
    print(f"  Sources: {plan['sources']}")
    print(f"  Fields: {plan['retrieval_fields']}")
    print(f"  Criteria: {plan['comparison_criteria']}")
    print(f"  Filters: {json.dumps(plan['filters'], indent=6)}")
    
    # Show retriever output
    print("\n🔍 RETRIEVER OUTPUT:")
    docs = result.get('retrieved_docs', [])
    print(f"  Found {len(docs)} products")
    for i, doc in enumerate(docs[:3], 1):
        print(f"  [{i}] {doc['title']}")
        print(f"      Price: ${doc['price']:.2f} | Brand: {doc.get('brand', 'N/A')}")
    
    # Show answerer output
    print("\n💬 FINAL ANSWER:")
    print(f"  {result.get('answer', 'NO ANSWER')}")
    print(f"\n📚 Citations: {', '.join(result.get('citations', []))}")
    
    # Show step log
    print("\n📊 EXECUTION LOG:")
    for log in result['step_log']:
        status = "✓" if log.get('success') else "✗"
        print(f"  {status} {log['node']}")
        if not log.get('success'):
            print(f"      Error: {log.get('error')}")
    
    print("\n" + "="*70)

def main():
    """Run demo queries"""
    
    print("\n" + "="*70)
    print("  AGENTIC PRODUCT DISCOVERY - DEMO")
    print("="*70)
    
    graph = create_graph()
    
    demo_queries = [
        "recommend longboard under $1000",
        "puzzles around $18",
    ]
    
    for query in demo_queries:
        demo_query(graph, query)
        input("\nPress Enter to continue to next query...")
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()