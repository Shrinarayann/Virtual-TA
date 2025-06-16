# main.py
import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Import the search function from our retriever module
from retriever import search_index

# Load environment variables from a .env file (for local development)
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="TDS Virtual TA",
    description="An API to answer questions about the Tools in Data Science course.",
    version="1.0.0"
)

# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None # Optional base64 encoded image string

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]


# --- LLM and Prompt Configuration ---
CHAT_API_URL = "https://aipipe.org/openrouter/v1/chat/completions"
CHAT_MODEL = "google/gemini-2.0-flash-lite-001" # Fast and cost-effective

async def generate_final_answer(query: str, image_provided: bool, context_chunks: List[Dict]) -> str:
    """Generates a final answer using the retrieved context and an LLM."""
    api_key = os.getenv("AIPIPE_TOKEN")
    if not api_key:
        raise HTTPException(status_code=500, detail="AIPIPE_TOKEN is not configured on the server.")

    # Construct the prompt for the LLM
    context_str = "\n\n---\n\n".join([chunk['content'] for chunk in context_chunks])
    
    system_prompt = """
    You are an expert Teaching Assistant for the 'Tools in Data Science' course at IIT Madras.
    Your task is to answer a student's question based ONLY on the provided context below.
    Do not use any external knowledge. If the context does not contain the answer, state that you cannot answer the question with the information provided.
    Be concise and directly address the student's question.
    """
    
    user_prompt_content = f"""
    CONTEXT:
    {context_str}
    
    ---
    
    QUESTION: {query}
    """
    
    if image_provided:
        user_prompt_content += "\n\n(Note: The user also provided an image. Refer to it only if the question explicitly mentions an image.)"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_content}
    ]

    payload = {"model": CHAT_MODEL, "messages": messages}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(CHAT_API_URL, json=payload, headers=headers, timeout=25.0)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            raise HTTPException(status_code=502, detail=f"Failed to communicate with the LLM provider: {e}")


# --- API Endpoint ---
@app.post("/api/", response_model=QueryResponse, summary="Answer a student's question")
async def answer_question(request: QueryRequest):
    """
    This endpoint answers a student's question by performing a RAG search
    on the course content and discourse forum posts.
    """
    try:
        # 1. Retrieve relevant chunks using the FAISS index
        retrieved_chunks = await search_index(request.question, k=5)
        
        if not retrieved_chunks:
            raise HTTPException(status_code=404, detail="Could not find any relevant information for your question.")

        # 2. Generate a final answer using the LLM with the retrieved context
        final_answer = await generate_final_answer(
            query=request.question,
            image_provided=(request.image is not None),
            context_chunks=retrieved_chunks
        )

        # 3. Format the source links for the response
        # De-duplicate links based on URL
        seen_urls = set()
        source_links = []
        for chunk in retrieved_chunks:
            url = chunk['metadata'].get('source_url')
            if url and url not in seen_urls:
                source_links.append(Link(
                    url=url,
                    text=chunk['content'][:250] + '...' # Provide a snippet of the source text
                ))
                seen_urls.add(url)

        return QueryResponse(answer=final_answer, links=source_links)

    except Exception as e:
        # Catch-all for any other unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "Virtual TA API is running"}