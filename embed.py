import json
import os
import time
import httpx
import asyncio
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
INPUT_FILE = './data/chunks.json'
OUTPUT_FILE = './data/embeddings.json'

# AI Pipe Proxy Configuration
API_URL = "https://aipipe.org/openai/v1/embeddings"
MODEL_NAME = "text-embedding-3-small" 
# As per the curl example for the OpenAI proxy, we use the standard model name.

# The OpenAI API has a limit on how many texts can be sent in one go.
# A batch size of 200 is a safe and efficient choice.
BATCH_SIZE = 200

async def embed_batch(client, text_batch: list[str], retries: int = 3, delay: int = 5):
    """
    Sends a batch of texts to the AI Pipe embedding API and returns the embeddings.
    Includes retry logic for handling transient API errors.
    """
    api_key = os.getenv("AIPIPE_TOKEN")
    if not api_key:
        raise ValueError("AIPIPE_TOKEN environment variable not set.")

    payload = {
        "model": MODEL_NAME,
        "input": text_batch
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(retries):
        try:
            response = await client.post(API_URL, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()  # Raises an exception for 4xx or 5xx status codes
            
            data = response.json()
            # The API returns embeddings in the same order as the input texts
            batch_embeddings = [item['embedding'] for item in data['data']]
            return batch_embeddings
            
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"API request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print("Max retries reached. Could not embed batch.")
                return None # Failed to embed this batch

async def main():
    """Main function to load chunks, embed them in batches, and save the results."""
    
    # 1. Load the chunked data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {INPUT_FILE}.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Please run the chunking script first.")
        return

    # 2. Check for the API token
    if not os.getenv("AIPIPE_TOKEN"):
        print("Error: The AIPIPE_TOKEN environment variable is not set.")
        print("Please set it before running the script.")
        return

    embedded_chunks = []
    
    # 3. Process chunks in batches asynchronously
    async with httpx.AsyncClient() as client:
        # Create a list of all batch processing tasks
        tasks = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            tasks.append(process_batch(client, batch, i))
            
        # Run all tasks concurrently with a progress bar
        print(f"Starting embedding process in {len(tasks)} batches...")
        results = await tqdm_asyncio.gather(*tasks, desc="Embedding Chunks")

    # 4. Consolidate results
    for batch_result in results:
        if batch_result: # Check if the batch was processed successfully
            embedded_chunks.extend(batch_result)

    # 5. Save the final results
    if embedded_chunks:
        print(f"\nSuccessfully embedded {len(embedded_chunks)} chunks.")
        print(f"Saving embeddings to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
        print("Embedding process complete!")
    else:
        print("\nNo chunks were embedded. Please check the error messages above.")

async def process_batch(client, batch, start_index):
    """Helper function to process one batch of chunks."""
    # Extract the text content to send to the API
    texts_to_embed = [chunk['page_content'] for chunk in batch]
    
    # Get embeddings for the current batch
    embeddings = await embed_batch(client, texts_to_embed)
    
    if embeddings is None:
        print(f"Skipping batch starting at index {start_index} due to embedding failure.")
        return []

    # Combine the original chunk data with its new embedding
    batch_with_embeddings = []
    for i, chunk in enumerate(batch):
        chunk['embedding'] = embeddings[i]
        batch_with_embeddings.append(chunk)
        
    return batch_with_embeddings

if __name__ == '__main__':
    # Before running, ensure you have the required libraries:
    # pip install httpx tqdm
    asyncio.run(main())