import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
COURSE_FILE = './data/course_content.json'
DISCOURSE_FILE = './data/discourse.json'
OUTPUT_FILE = './data/chunks.json'

# Chunking parameters: These are good starting points.
# CHUNK_SIZE is the maximum number of characters in a chunk.
# CHUNK_OVERLAP is the number of characters to overlap between chunks to maintain context.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def chunk_course_content(text_splitter, all_chunks):
    """Processes and chunks the course_content.json file."""
    print(f"Processing {COURSE_FILE}...")
    try:
        with open(COURSE_FILE, 'r', encoding='utf-8') as f:
            course_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {COURSE_FILE} not found. Skipping.")
        return

    for item in course_data:
        content = item.get('content')
        if not content:
            continue

        # Use the text splitter to create documents (chunks)
        docs = text_splitter.create_documents([content])
        
        for i, doc in enumerate(docs):
            chunk_metadata = {
                "source_url": item.get('url'),
                "source_title": item.get('title'),
                # Create a unique ID for each chunk for easy reference
                "chunk_id": f"{item.get('id', 'course')}_chunk_{i}"
            }
            
            all_chunks.append({
                "page_content": doc.page_content,
                "metadata": chunk_metadata
            })
    print(f"Finished processing {COURSE_FILE}. Total chunks so far: {len(all_chunks)}")


def chunk_discourse_content(text_splitter, all_chunks):
    """Processes and chunks the discourse.json file."""
    print(f"\nProcessing {DISCOURSE_FILE}...")
    try:
        with open(DISCOURSE_FILE, 'r', encoding='utf-8') as f:
            discourse_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {DISCOURSE_FILE} not found. Skipping.")
        return

    initial_chunk_count = len(all_chunks)

    for i, thread in enumerate(discourse_data):
        # 1. Chunk the main question
        question_text = thread.get('question')
        if question_text:
            docs = text_splitter.create_documents([question_text])
            for j, doc in enumerate(docs):
                chunk_metadata = {
                    "source_url": thread.get('url'),
                    "source_title": f"Discourse Question: {question_text[:80]}...",
                    "chunk_id": f"discourse_q_{i}_chunk_{j}"
                }
                all_chunks.append({
                    "page_content": doc.page_content,
                    "metadata": chunk_metadata
                })

        # 2. Chunk the answers
        answers = thread.get('answers', [])
        for k, answer in enumerate(answers):
            answer_text = answer.get('text')
            if answer_text:
                docs = text_splitter.create_documents([answer_text])
                for l, doc in enumerate(docs):
                    chunk_metadata = {
                        "source_url": answer.get('url'),
                        "source_title": f"Discourse Answer to: {question_text[:80]}...",
                        "chunk_id": f"discourse_q_{i}_a_{k}_chunk_{l}"
                    }
                    all_chunks.append({
                        "page_content": doc.page_content,
                        "metadata": chunk_metadata
                    })

    print(f"Finished processing {DISCOURSE_FILE}. Added {len(all_chunks) - initial_chunk_count} new chunks.")

def main():
    """Main function to run the chunking process."""
    
    # Initialize the text splitter
    # This splitter tries to split on paragraphs, then lines, then words.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    
    # Process both files
    chunk_course_content(text_splitter, all_chunks)
    chunk_discourse_content(text_splitter, all_chunks)
    
    # Save the final chunks to a JSON file
    print(f"\nTotal chunks created: {len(all_chunks)}")
    print(f"Saving all chunks to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    print("Chunking complete! Your data is ready for the next step.")

if __name__ == '__main__':
    # Before running, make sure you have langchain installed:
    # pip install langchain-text-splitters
    #
    # Ensure 'course_content.json' and 'discourse.json' are in the same directory.
    main()