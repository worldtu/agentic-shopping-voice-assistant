# graph/router/prompts.py
from langchain_core.prompts import PromptTemplate

ROUTER_TEMPLATE = """<|im_start|>system
You are a JSON extraction assistant. You MUST return ONLY valid JSON, nothing else.<|im_end|>
<|im_start|>user
Extract information from the query and return a JSON object with this EXACT structure:

{{
  "task": "product_search" | "comparison" | "recommendation" | "availability_check",
  "constraints": {{
    "product": string or null,
    "min_price": number or null,
    "max_price": number or null,
    "material": string or null,
    "brand": array
  }},
  "safety_flags": array
}}

CRITICAL: Check for safety issues FIRST before extracting other information!

SAFETY FLAGS (CHECK THESE CAREFULLY):

1. medical_advice - Query asks about treating, curing, or diagnosing health issues
   DANGER WORDS: "cure", "treat", "medicine for", "remedy for", "heal", "diagnose", "disease", "illness", "sick"
   Examples that MUST be flagged:
   - "medicine to cure my disease" → ["medical_advice"]
   - "what treats headaches" → ["medical_advice"]
   - "cure for my cold" → ["medical_advice"]
   - "pills for pain" → ["medical_advice"]

2. dangerous_product - Query asks about weapons, explosives, or illegal items
   DANGER WORDS: "gun", "weapon", "explosive", "bomb", "drug", "illegal"
   Examples:
   - "show me guns" → ["dangerous_product"]

3. inappropriate_content - Sexual, violent, or offensive requests
   DANGER WORDS: explicit terms for sex, violence, hate
   Examples:
   - "xxx products" → ["inappropriate_content"]

TASK DEFINITIONS (choose exactly one):

1. product_search - User wants to FIND products matching specific criteria
   Keywords: "find", "show me", "looking for", "I need", "I want"
   Example: "find organic shampoo under $20"

2. comparison - User wants to COMPARE specific products or brands
   Keywords: "compare", "vs", "versus", "difference between", "which is better"
   Example: "compare Dove vs Pantene conditioner"

3. recommendation - User wants SUGGESTIONS or advice on what to buy
   Keywords: "recommend", "suggest", "what's the best", "top rated", "should I buy"
   Example: "recommend the best vegan soap"

4. availability_check - User wants to know if product is IN STOCK or available NOW, or wants CURRENT/LATEST pricing
   Keywords: "available", "in stock", "can I buy", "is there", "do you have", "current", "latest", "now", "today", "current price", "latest price"
   Examples: 
   - "is organic shampoo available now?"
   - "what's the current price of a pen?"
   - "find me the latest price of coffee"

PRICE EXTRACTION RULES (FOLLOW EXACTLY):

1. "under X" or "below X" or "less than X"
   → min_price: null, max_price: X
   Example: "under $20" → max_price: 20

2. "above X" or "over X" or "more than X"
   → min_price: X, max_price: null
   Example: "above $30" → min_price: 30

3. "around X" or "about X" or "roughly X"
   → min_price: X*0.9, max_price: X*1.1
   Example: "around $80" → min_price: 72, max_price: 88

4. "between X and Y"
   → min_price: X, max_price: Y
   Example: "between $20 and $40" → min_price: 20, max_price: 40

5. "cheap" or "affordable" or "budget"
   → min_price: null, max_price: 15
   Example: "cheap soap" → max_price: 15

6. "expensive" or "premium" or "luxury" or "high-end"
   → min_price: 100, max_price: null
   Example: "expensive shampoo" → min_price: 100

7. No price mentioned
   → min_price: null, max_price: null

GENERAL RULES:
- Numbers are numbers (20, not "20")
- Use null for missing values
- brand is always an array (["Nike"] or [])
- Return ONLY valid JSON, no markdown

EXAMPLES (STUDY THESE EXACTLY):

Query: "organic shampoo under $20"
JSON:{{"task": "product_search", "constraints": {{"product": "shampoo", "min_price": null, "max_price": 20, "material": "organic", "brand": []}}, "safety_flags": []}}

Query: "compare Dove vs Pantene conditioner"
JSON:{{"task": "comparison", "constraints": {{"product": "conditioner", "min_price": null, "max_price": null, "material": null, "brand": ["Dove", "Pantene"]}}, "safety_flags": []}}

Query: "what medicine cures headaches"
JSON:{{"task": "product_search", "constraints": {{"product": "medicine", "min_price": null, "max_price": null, "material": null, "brand": []}}, "safety_flags": ["medical_advice"]}}

Query: "show me leather Nike shoes around $80"
JSON:{{"task": "product_search", "constraints": {{"product": "shoes", "min_price": 72, "max_price": 88, "material": "leather", "brand": ["Nike"]}}, "safety_flags": []}}

Query: "stainless steel kettles between $20 and $40"
JSON:{{"task": "product_search", "constraints": {{"product": "kettle", "min_price": 20, "max_price": 40, "material": "stainless steel", "brand": []}}, "safety_flags": []}}

Query: "cheap vegan soap"
JSON:{{"task": "product_search", "constraints": {{"product": "soap", "min_price": null, "max_price": 15, "material": "vegan", "brand": []}}, "safety_flags": []}}

Query: "premium organic coffee above $30"
JSON:{{"task": "product_search", "constraints": {{"product": "coffee", "min_price": 30, "max_price": null, "material": "organic", "brand": []}}, "safety_flags": []}}

Query: "expensive luxury watch"
JSON:{{"task": "product_search", "constraints": {{"product": "watch", "min_price": 100, "max_price": null, "material": null, "brand": []}}, "safety_flags": []}}

Query: "affordable Nike running shoes"
JSON:{{"task": "product_search", "constraints": {{"product": "shoes", "min_price": null, "max_price": 15, "brand": ["Nike"], "material": null}}, "safety_flags": []}}

Query: "find me the current price of a pen"
JSON:{{"task": "availability_check", "constraints": {{"product": "pen", "min_price": null, "max_price": null, "material": null, "brand": []}}, "safety_flags": []}}

Query: "what's the latest price of organic coffee"
JSON:{{"task": "availability_check", "constraints": {{"product": "coffee", "min_price": null, "max_price": null, "material": "organic", "brand": []}}, "safety_flags": []}}

Now extract from this query:
Query: {query}<|im_end|>
<|im_start|>assistant
"""

router_prompt = PromptTemplate(
    input_variables=["query"],
    template=ROUTER_TEMPLATE
)