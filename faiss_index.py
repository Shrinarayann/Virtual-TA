import json
import os
import sys
import numpy as np
import faiss
import httpx
import asyncio

# --- Configuration ---
EMBEDDINGS_FILE = './data/embeddings.json'
INDEX_FILE = './data/faiss_index.bin'
MAPPING_FILE = './data/index_to_chunk_map.json'

# AI Pipe Proxy Configuration (for embedding the search query)
API_URL = "https://aipipe.org/openai/v1/embeddings"
MODEL_NAME = "text-embedding-3-small"

def create_faiss_index():
    """Loads embeddings and builds a FAISS index."""
    print("Starting FAISS index creation...")
    
    # 1. Load the embeddings data
    try:
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        if not chunks:
            print(f"Error: {EMBEDDINGS_FILE} is empty. No index created.")
            return
    except FileNotFoundError:
        print(f"Error: {EMBEDDINGS_FILE} not found. Please run the embedding script first.")
        return

    # 2. Prepare data for FAISS
    embeddings = [chunk['embedding'] for chunk in chunks]
    
    # Convert to a NumPy array of type float32, which FAISS requires
    embeddings_matrix = np.array(embeddings, dtype='float32')
    vector_dimension = embeddings_matrix.shape[1]
    
    print(f"Loaded {len(embeddings)} vectors of dimension {vector_dimension}.")

    # 3. Create the FAISS index
    # IndexFlatL2 is a simple index that performs an exact search.
    # It's a great starting point for accuracy.
    index = faiss.IndexFlatL2(vector_dimension)
    
    # Check if the index is trained (IndexFlatL2 doesn't require training, but it's good practice)
    if not index.is_trained:
        print("Training index (not required for IndexFlatL2, but good practice)...")
        index.train(embeddings_matrix)
        
    # Add the vectors to the index
    index.add(embeddings_matrix)
    print(f"Index now contains {index.ntotal} vectors.")

    # 4. Save the index and the mapping file
    print(f"Saving FAISS index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)

    # Create a mapping from index ID to original chunk data (without the large embedding)
    index_to_chunk = {
        i: {
            "page_content": chunk["page_content"],
            "metadata": chunk["metadata"]
        }
        for i, chunk in enumerate(chunks)
    }
    
    print(f"Saving ID-to-chunk map to {MAPPING_FILE}...")
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_to_chunk, f, indent=2)

    print("\nFAISS index creation complete!")


async def embed_query(query_text: str):
    """Embeds a single text query using the AI Pipe proxy."""
    api_key = os.getenv("AIPIPE_TOKEN")
    if not api_key:
        raise ValueError("AIPIPE_TOKEN environment variable not set.")

    payload = {"model": MODEL_NAME, "input": query_text}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_URL, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data['data'][0]['embedding']
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"Error embedding query: {e}")
            return None

async def search_faiss_index(query: str, k: int = 5):
    """Searches the FAISS index for the k most similar chunks to the query."""
    print(f"Searching for top {k} results for query: '{query}'")
    
    # 1. Load the index and mapping
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            index_to_chunk = json.load(f)
    except FileNotFoundError:
        print("Error: Index files not found. Please run 'python index_with_faiss.py create' first.")
        return

    # 2. Embed the search query
    print("Embedding search query...")
    query_embedding = await embed_query(query)
    if query_embedding is None:
        return

    # 3. Prepare the query vector for FAISS
    # FAISS expects a 2D array for queries
    query_vector_np = np.array([query_embedding], dtype='float32')

    # 4. Perform the search
    # The search method returns distances and the indices (IDs) of the results
    distances, indices = index.search(query_vector_np, k)

    # 5. Format and print the results
    print("\n--- Search Results ---")
    if not indices.size > 0:
        print("No results found.")
        return

    results = []
    for i, idx in enumerate(indices[0]):
        # The index in the mapping file is a string, so we convert idx
        retrieved_chunk = index_to_chunk.get(str(idx))
        if retrieved_chunk:
            print(f"\nResult {i+1} (Distance: {distances[0][i]:.4f}):")
            print(f"Source: {retrieved_chunk['metadata']['source_url']}")
            print("-" * 20)
            print(retrieved_chunk['page_content'])
            print("-" * 20)
            results.append({
                "content": retrieved_chunk['page_content'],
                "metadata": retrieved_chunk['metadata'],
                "distance": float(distances[0][i])
            })
    return results


async def main():
    """Main function to handle command-line arguments."""
    if len(sys.argv) < 2 or sys.argv[1] not in ['create', 'search']:
        print("Usage:")
        print("  To create the index: python index_with_faiss.py create")
        print("  To search the index: python index_with_faiss.py search \"<your query>\"")
        return

    mode = sys.argv[1]

    if mode == 'create':
        create_faiss_index()
    elif mode == 'search':
        if len(sys.argv) < 3:
            print("Error: Search mode requires a query.")
            print("Usage: python index_with_faiss.py search \"<your query>\"")
            return
        query = sys.argv[2]
        await search_faiss_index(query)

if __name__ == '__main__':
    # Before running, ensure you have the required libraries:
    # pip install faiss-cpu numpy httpx
    # Note: 'faiss-cpu' is for CPU-only. Use 'faiss-gpu' if you have a compatible NVIDIA GPU and CUDA.
    asyncio.run(main())