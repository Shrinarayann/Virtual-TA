import json
import os
import numpy as np
import faiss
import httpx
import logging
from typing import List, Dict
from pathlib import Path

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "data" / "faiss_index.bin"
MAPPING_FILE = BASE_DIR / "data" / "index_to_chunk_map.json"
API_URL = "https://aipipe.org/openai/v1/embeddings"
MODEL_NAME = "text-embedding-3-small"

# --- Load FAISS Index and Mapping ---
try:
    if not INDEX_FILE.exists() or not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Missing FAISS index or mapping file. Expected:\n"
            f" - {INDEX_FILE}\n"
            f" - {MAPPING_FILE}\n"
            f"Please run `python index_with_faiss.py create` to generate them."
        )
    index = faiss.read_index(str(INDEX_FILE))
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        index_to_chunk = json.load(f)
    logger.info("FAISS index and mapping file loaded successfully.")
except Exception as e:
    logger.error(f"Error loading FAISS index: {e}")
    raise

# --- Embedding API ---
async def embed_query(query_text: str) -> List[float]:
    """Embeds a single text query using the AI Pipe proxy."""
    api_key = os.getenv("AIPIPE_TOKEN")
    if not api_key:
        raise ValueError("AIPIPE_TOKEN environment variable not set.")

    payload = {"model": MODEL_NAME, "input": query_text}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return data['data'][0]['embedding']

# --- FAISS Search ---
async def search_index(query: str, k: int = 5) -> List[Dict]:
    """Searches the FAISS index for the k most similar chunks."""
    query_embedding = await embed_query(query)
    query_vector_np = np.array([query_embedding], dtype='float32')
    distances, indices = index.search(query_vector_np, k)

    results = []
    for i, idx in enumerate(indices[0]):
        retrieved_chunk = index_to_chunk.get(str(idx))
        if retrieved_chunk:
            results.append({
                "content": retrieved_chunk['page_content'],
                "metadata": retrieved_chunk['metadata'],
                "distance": float(distances[0][i])
            })

    return results
