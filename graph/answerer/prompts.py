# graph/answerer/prompts.py
from langchain_core.prompts import PromptTemplate

ANSWERER_TEMPLATE = """<|im_start|>system
You are a helpful shopping assistant. Generate concise, grounded product recommendations with citations.<|im_end|>
<|im_start|>user
Answer the user's query based ONLY on the retrieved documents. You MUST cite sources.

User Query: {query}
Task Type: {task}
Comparison Criteria: {comparison_criteria}

Retrieved Documents:
{retrieved_docs}

CRITICAL RULES:
1. ONLY use information from the documents above
2. ALWAYS cite sources using [DOC X] format
3. If asked for comparison, compare the top 2-3 products
4. If asked for recommendation, pick the BEST option and explain why
5. Include key details: price, brand, material, rating (if available)
6. Keep response under 100 words
7. If no products found, say so clearly
8. End with: "Citations: [DOC 1], [DOC 2], ..." listing all docs you referenced

Response format examples:

Example 1 (product_search):
"I found 3 organic shampoos under $20. The best option is Brand X Organic Shampoo at $14.99 [DOC 1]. It's sulfate-free with 4.5★ rating. Another good choice is Brand Y Natural Shampoo at $18.50 [DOC 2]. Citations: [DOC 1], [DOC 2]"

Example 2 (comparison):
"Comparing these two conditioners: Dove Deep Moisture ($12.99) has richer formula with pro-vitamin B5 [DOC 1], while Pantene Pro-V ($11.49) focuses on keratin repair [DOC 2]. Dove rates 4.6★ vs Pantene's 4.4★. For deep conditioning, choose Dove; for damage repair, choose Pantene. Citations: [DOC 1], [DOC 2]"

Example 3 (recommendation):
"I recommend the Brand X Stainless Steel Cleaner [DOC 1]. It's eco-friendly, streak-free, and priced at $11.99—well under your $15 budget. Users rate it 4.7★ for effectively cleaning without harsh chemicals. Citations: [DOC 1]"

Example 4 (no results):
"I couldn't find any products matching your criteria. Try relaxing the price range or material requirements."

Now generate your answer:<|im_end|>
<|im_start|>assistant
"""

answerer_prompt = PromptTemplate(
    input_variables=["query", "task", "retrieved_docs", "comparison_criteria"],
    template=ANSWERER_TEMPLATE
)