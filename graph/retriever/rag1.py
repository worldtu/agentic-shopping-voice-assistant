"""
Private RAG retrieval logic (FAISS + SentenceTransformer + Groq Llama3)
Supports auto-download embeddings from Google Drive & user-provided API key.
"""

import os
import json
import faiss
import torch
import logging
import pandas as pd
import numpy as np
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ===============================
# 🔹 Environment Setup
# ===============================
load_dotenv(dotenv_path=".env.local", override=True)
logger = logging.getLogger(__name__)

_vector_store = None
_stella_model = None
_df = None
_index = None
GROQ_API_KEY = None  # will be set dynamically by setup_env()

DATA_PATH = os.getenv("DATA_PATH", "./data_cleaned.csv")
EMB_PATH = "./text_emb.pt"
EMB_DRIVE_ID = os.getenv("EMB_DRIVE_ID")
DATA_DRIVE_ID = os.getenv("DATA_DRIVE_ID")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ===============================
# 1️⃣ Setup Function
# ===============================
def setup_env(GROQ_API_KEY: str = None):
    """
    Initialize environment:
    - User provides Groq API key
    - Downloads data_cleaned.csv and text_emb.pt if missing
    """
    if GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        logger.info("[Setup] GROQ API key loaded successfully.")
    else:
        raise ValueError("❌ Please provide your GROQ_API_KEY when calling setup_env().")

    if not os.path.exists(DATA_PATH):
        _download_data_from_drive()
    else:
        logger.info("[Setup] Dataset already cached locally.")

    if not os.path.exists(EMB_PATH):
        _download_embedding_from_drive()
    else:
        logger.info("[Setup] Embedding already cached locally.")


# ===============================
# 2️⃣ Download from Google Drive
# ===============================
def _download_embedding_from_drive():
    """Download text_emb.pt from Google Drive if missing."""
    import gdown
    url = f"https://drive.google.com/uc?id={EMB_DRIVE_ID}"
    os.makedirs(os.path.dirname(EMB_PATH) or ".", exist_ok=True)
    logger.info(f"[Download] Downloading embedding from {url} ...")
    gdown.download(url, EMB_PATH, quiet=False)
    logger.info(f"[Download] Saved to {EMB_PATH}")


def _download_data_from_drive():
    import gdown
    if not DATA_DRIVE_ID:
        raise ValueError("❌ DATA_DRIVE_ID missing in .env file")
    url = f"https://drive.google.com/uc?id={DATA_DRIVE_ID}"
    os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
    logger.info(f"[Download] Downloading dataset from {url} ...")
    gdown.download(url, DATA_PATH, quiet=False)
    logger.info(f"[Download] Saved to {DATA_PATH}")


# ===============================
# 3️⃣ Load Vector Store
# ===============================
def get_vector_store():
    """Load FAISS index and SentenceTransformer encoder."""
    global _vector_store, _stella_model, _df, _index

    if _vector_store is None:
        logger.info("[Init] Loading dataset and embeddings...")

        _df = pd.read_csv(DATA_PATH)
        text_emb = torch.load(EMB_PATH, map_location="cpu")
        text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
        text_emb_np = text_emb.numpy().astype("float32")

        _index = faiss.IndexFlatIP(text_emb_np.shape[1])
        _index.add(text_emb_np)

        _stella_model = SentenceTransformer("infgrad/stella-base-en-v2", trust_remote_code=True)
        _vector_store = {"index": _index, "df": _df, "model": _stella_model}

        logger.info(f"[Init] FAISS index with {_index.ntotal} vectors loaded.")
    return _vector_store


# ===============================
# 4️⃣ Filter Extraction
# ===============================
def extract_filters_from_text(query: str) -> Dict:
    """Extract structured filters using user-provided Groq API key."""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    print(f"[DEBUG] Loaded GROQ_API_KEY: {GROQ_API_KEY[:8]}******")
    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY not set. Please call setup_env(GROQ_API_KEY=...) first.")

    system_prompt = """You are a data extraction assistant.
Extract structured filters from a shopping query in JSON.
Keys: category, min_price, max_price, material, brand.
Omit keys not mentioned.
Example:
"Find eco-friendly stainless cleaner under $15" ->
{"category":"cleaner","material":"stainless","max_price":15}
Return only valid JSON."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "max_tokens": 200,
        "temperature": 0.2,
    }
    try:
        res = requests.post(GROQ_ENDPOINT, headers=headers, json=payload).json()
        print("[DEBUG] Groq API raw response:", res)
        text = res["choices"][0]["message"]["content"].strip()
        print("[DEBUG] Parsed filters:", _safe_json_parse(text))
        return _safe_json_parse(text)

    except Exception as e:
        logger.warning(f"[Groq] Filter extraction failed: {e}")
        return {}


def _safe_json_parse(text):
    import re, json
    try:
        for m in re.findall(r"\{[\s\S]*?\}", text):
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    return {}


# ===============================
# 5️⃣ Retrieval
# ===============================
def retrieve_from_rag(query: str, filters: Dict, k: int = 20) -> List[Dict]:
    """Retrieve top-k documents from FAISS index with filter constraints."""
    vs = get_vector_store()
    df, index, model = vs["df"], vs["index"], vs["model"]

    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, k * 5)
    indices, scores = indices[0], scores[0]

    filtered = []
    for idx, score in zip(indices, scores):
        row = df.iloc[idx]

        # Category filter
        if "category" in filters and pd.notna(row.get("category", None)):
            query_cat = str(filters["category"]).lower()
            if query_cat not in str(row["category"]).lower():
                continue

        # Price filter (inside loop)
        try:
            price = float(row.get("selling_price", 0))
            if "min_price" in filters and price < float(filters["min_price"]):
                continue
            if "max_price" in filters and price > float(filters["max_price"]):
                continue
        except Exception:
            continue

        filtered.append(_format_result(row, score))
        if len(filtered) >= k:
            break

    # Fallback: if no strict filter matches, return top semantic matches
    if not filtered:
        print("[DEBUG] No strict matches found — returning top semantic results (may include out-of-range prices)")
        fallback = [_format_result(df.iloc[idx], score) for idx, score in zip(indices[:k], scores[:k])]
        if "max_price" in filters:
            fallback = [f for f in fallback if f.get("price", 0) <= float(filters["max_price"])]
        filtered = fallback

    # Debug logs
    print(f"[DEBUG] Dataset found: {os.path.exists(DATA_PATH)} ({DATA_PATH})")
    print(f"[DEBUG] Retrieved {len(filtered)} items for query: '{query}' with filters: {filters}")

    if len(filtered) > 0:
        print(f"✅ Successfully retrieved {len(filtered)} results.")
    else:
        print("⚠️ No matching results found. Check filters or data content.")

    # Final strict price filter to guarantee correctness
    if "max_price" in filters:
        try:
            max_p = float(filters["max_price"])
            before = len(filtered)
            filtered = [f for f in filtered if f.get("price", 0) <= max_p]
            after = len(filtered)
            print(f"[DEBUG] Applied max_price filter {max_p}: reduced {before} → {after} results")
        except Exception as e:
            print(f"[DEBUG] Skipped price filter due to error: {e}")

    return filtered


def _format_result(row, score):
    """
    Normalize one row into the standard product dict schema.

    This matches what MCP rag.search + LangGraph answerer expect:
    - doc_id: unique id for citation
    - title: product name
    - price: numeric price
    - category / brand / material: metadata
    - ingredients / rating: optional but helpful for recommendations
    - content: rich text description
    - source: always "rag"
    """
    # Safe rating extraction
    try:
        rating_val = float(row["rating"]) if "rating" in row and pd.notna(row["rating"]) else None
    except Exception:
        rating_val = None

    return {
        "doc_id": row.get("uniq_id"),
        "title": row.get("product_name"),
        "price": float(row.get("selling_price", 0)),
        "category": row.get("category", ""),
        "brand": row.get("brand", ""),
        "material": row.get("material", ""),
        "ingredients": row.get("ingredients", ""),  # if column missing, returns ""
        "rating": rating_val,
        "content": row.get("rich_description", ""),
        "score": float(score),
        "source": "rag",
    }


# ===============================
# 6️⃣ Unified Pipeline
# ===============================
def rag_with_auto_filter(user_query: str, k: int = 20) -> List[Dict]:
    """
    Convenience pipeline:
    - Use Groq to extract filters from natural language query
    - Then run filtered retrieval.
    """
    print(f"[DEBUG] 🚀 rag_with_auto_filter triggered with query: {user_query}")
    filters = extract_filters_from_text(user_query)
    results = retrieve_from_rag(user_query, filters, k)
    return results


def rag_search(query: str, filters: Dict = None, k: int = 20) -> List[Dict]:
    """
    Unified interface for LangGraph and MCP:

    - If filters is provided:
        Use explicit filters from planner / caller.
    - If filters is None:
        Auto-extract filters from the query using Groq.
    """
    if filters:
        return retrieve_from_rag(query, filters, k)
    return rag_with_auto_filter(query, k)
