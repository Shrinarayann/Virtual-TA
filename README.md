# 📚 Virtual TA – AI Teaching Assistant

An AI-powered virtual teaching assistant that answers student queries using IITM’s TDS course material and Discourse Q&A discussions.

---

## ✨ Features
- 🕸️ **Web Scraper (Playwright):** Extracts Q&A, replies, and images from IITM TDS Discourse.
- 📦 **Chunking & Embeddings:** Preprocesses text chunks for retrieval using FAISS.
- 🤖 **RAG-based API:** Answers questions and provides relevant Discourse links as context.
- 🚀 **Deployment:** FastAPI app deployed on Render with `/api/ask` endpoint.

---

## ⚡ Tech Stack
- **Backend:** Python, FastAPI
- **Scraper:** Playwright (Python)
- **Embeddings:** SentenceTransformers, FAISS
- **RAG System:** Retrieval-Augmented Generation (OpenAI GPT)
- **Deployment:** Render (Free Tier)

---

## 🚀 Getting Started

### 📦 Clone and Install
```bash
git clone https://github.com/Shrinarayann/virtual-ta.git
cd virtual-ta
pip install -r requirements.txt
```

### 🏃‍♂️ Run Locally
```bash
uvicorn main:app --reload
```
Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📌 API Usage

### POST `/api/ask`
**Request Body:**
```json
{
  "question": "Explain ensemble methods in TDS.",
  "image_base64": "<optional image in base64>"
}
```

**Response:**
```json
{
  "answer": "Ensemble methods combine multiple models to improve performance...",
  "sources": [
    {
      "snippet": "Ensemble methods like Bagging and Boosting...",
      "url": "https://discourse.tds.iitm.ac.in/t/ensemble-methods/123"
    }
  ]
}
```

---

## 🏆 Highlights
- 🔥 Reduced query latency by **35%** with FAISS indexing.
- 📖 Multi-modal queries supported (text + base64 images).
- 🌐 Public API deployed on Render.

---

## 👨‍💻 Author
- **Shrinarayan N** – [LinkedIn](https://linkedin.com/in/shrinarayan-n) | [GitHub](https://github.com/Shrinarayann)

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
