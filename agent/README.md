# AI Assistant using Chat Completion and RAG  
Author: **Leonardo Sampaio Rocha**

## Overview
This project is a lightweight **AI Assistant backend** built with **FastAPI**, designed to integrate **Chat Completions** with **Retrieval-Augmented Generation (RAG)**.  
It provides both streaming and non-streaming APIs, ready to be plugged into a frontend.

---

## Quickstart

### 1. Create and activate a virtual environment  
```bash
cd agent
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment via .env
- Create `agent/.env` with:
```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Frontend origin(s) (comma-separated)
CORS_ALLOW_ORIGINS=http://localhost:5173

# Optional default system prompt
DEFAULT_SYSTEM_PROMPT=You are a helpful AI assistant.
```
The server loads `agent/.env` automatically (and falls back to a root `.env` if present).

### 4. Run the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello!","history":[],"stream":false}'
```

API

POST /api/chat

Request body:
```json
{
  "message": "string",
  "history": [ {"role": "user|assistant|system", "content": "string"} ],
  "stream": false,
  "system_prompt": "optional"
}
```

Responses:
- Non-streaming: application/json with { "reply": string, "citations": [] }
- Streaming: text/plain chunks; append to your UI as tokens arrive

RAG readiness
- app/services/rag.py: RAGPipeline.retrieve and format_context
- Replace with vector DB retrieval (FAISS/Chroma/Milvus/pgvector). Return RetrievalResult[] and we’ll pass formatted context into the system message.

OpenAI config
- app/services/llm_openai.py reads OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
- Uses /chat/completions non-streaming and streaming via SSE-style chunks

