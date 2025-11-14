# tests/test_answerer.py
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.graph import create_graph

def test_answerer_basic():
    """Test basic answer generation with citations"""
    
    graph = create_graph()
    
    result = graph.invoke({
        "query": "organic shampoo under $20",
        "step_log": []
    })
    
    # Should have answer
    assert result.get("answer"), "Should have an answer"
    assert len(result["answer"]) > 0
    
    # Should have citations
    assert len(result.get("citations", [])) > 0, "Should have citations"
    
    print("✓ Answer generated:")
    print(f"  {result['answer'][:150]}...")
    print(f"  Citations: {result['citations']}")


def test_answerer_comparison():
    """Test comparison task answer"""
    
    graph = create_graph()
    
    result = graph.invoke({
        "query": "compare Nike vs Adidas shoes",
        "step_log": []
    })
    
    answer = result.get("answer", "")
    
    # Answer should mention both brands (if found)
    # Note: depends on what's in your dataset
    assert len(answer) > 0
    
    print("✓ Comparison answer:")
    print(f"  {answer}")


def test_answerer_recommendation():
    """Test recommendation task answer"""
    
    graph = create_graph()
    
    result = graph.invoke({
        "query": "recommend the best cleaner under $15",
        "step_log": []
    })
    
    answer = result.get("answer", "")
    
    # Should have a clear recommendation
    assert len(answer) > 0
    
    print("✓ Recommendation answer:")
    print(f"  {answer}")


def test_complete_pipeline():
    """Test the ENTIRE pipeline end-to-end"""
    
    test_queries = [
        "organic shampoo under $20",
        "Nike shoes around $80",
        "stainless steel kettle between $20 and $40",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        # Create fresh graph for each query to avoid log accumulation
        graph = create_graph()
        
        result = graph.invoke({
            "query": query,
            "step_log": []
        })
        
        # Check all nodes succeeded
        for log in result["step_log"]:
            node = log["node"]
            success = log.get("success", False)
            print(f"  [{node}] {'✓' if success else '✗'}")
            
            if not success:
                print(f"    Error: {log.get('error')}")
        
        # Check final output
        print(f"\nAnswer: {result.get('answer', 'NO ANSWER')}")
        print(f"Citations: {result.get('citations', [])}")
        
        # Assertions
        assert result.get("answer"), f"Query '{query}' produced no answer"
        
        # At least router, planner, retriever, answerer should succeed
        nodes_executed = [log["node"] for log in result["step_log"]]
        assert "router" in nodes_executed
        assert "planner" in nodes_executed
        assert "retriever" in nodes_executed
        assert "answerer" in nodes_executed
    
    print("\n✓ Complete pipeline test passed for all queries")


def test_no_results_handling():
    """Test graceful handling when no products found"""
    
    graph = create_graph()
    
    result = graph.invoke({
        "query": "purple unicorn shampoo for $0.01",
        "step_log": []
    })
    
    answer = result.get("answer", "")
    
    # Should acknowledge no results
    assert "couldn't find" in answer.lower() or "no products" in answer.lower()
    
    print("✓ No results handled gracefully:")
    print(f"  {answer}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])