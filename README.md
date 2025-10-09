# AI Startup Mentor — Interactive Multi-Expert Chat System

**Author:** Leonardo Sampaio Rocha  
**Acknowledgment:** Prototype concept inspired by an idea from **Gui Santa Rosa**.

---

## Overview
This project implements an **AI Startup Mentor**, a conversational system that helps founders refine startup ideas, analyze product strategies, and explore funding paths.

The mentor provides general guidance on building and growing a company. When you need deeper insights, you can call on specialized experts:
- **PM (Product Manager):** Product strategy, validation, and roadmap design
- **CTO (Chief Technology Officer):** Technical architecture, scalability, and cost trade-offs
- **VC (Venture Capital Partner):** Market opportunity, traction, and fundraising guidance
- Or call the **Committee**, which brings all three together to give multi-perspective feedback.

The backend is powered by **FastAPI**, using **OpenAI Chat Completions** and **LangGraph** to coordinate agent reasoning and persona routing. A minimal **HTML/JS frontend** is included under `webchat/`.

---

## Project Structure

```
AI-AGENT-CHAT-RAG-PROD/
├── agent/
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint (with GCS persistence)
│   │   ├── state_graph.py     # Persona routing & LangGraph orchestration
│   │   └── services/          # LLM and utility modules
│   ├── prompts/               # Persona prompt definitions (CTO, PM, VC)
│   └── data/                  # (Optional) Local SQLite or cache storage
│
├── webchat/
│   ├── index.html             # Chat UI
│   ├── script.js              # Frontend logic
│   └── styles.css             # UI styling
│
├── Dockerfile                 # Container build instructions
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Persistence Layer

By default, the system stores chat histories in **Google Cloud Storage (GCS)** — one JSON file per thread (conversation). Each file is stored at:

```
gs://<your-bucket>/threads/<thread_id>.json
```

This is defined by two environment variables:

```bash
GCS_BUCKET_NAME=ai-mentor-checkpoints
GCS_THREADS_PREFIX=threads
```

You can easily replace this with local SQLite persistence by editing a few lines in `agent/app/main.py` and `state_graph.py`, for example:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
checkpointer = AsyncSqliteSaver.from_file("local_threads.db")
```

This makes the system fully local, requiring no cloud services.

---

## Google Cloud Setup

To enable GCS persistence:

1. **Create a bucket:**
   ```bash
   gsutil mb -l us-central1 gs://ai-mentor-checkpoints
   ```

2. **Authenticate:**
   ```bash
   gcloud auth application-default login
   ```

3. **Set environment variables:**
   ```bash
   GCS_BUCKET_NAME=ai-mentor-checkpoints
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   ```

The application will automatically create and manage JSON thread files inside the bucket.

---

## Environment Configuration

Example `.env` file:

```bash
# OpenAI configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# CORS configuration
CORS_ALLOW_ORIGINS=http://localhost:5500

# System prompt
DEFAULT_SYSTEM_PROMPT=You are an AI Mentor inspired by Steve Jobs, acting as a mentor for startup founders.\nSpeak with sharp insight, challenge assumptions, and push people to think bigger.\nAlways focus on product excellence, user experience, innovation, and building impactful companies.\nKeep answers short (2–4 sentences), practical, and inspiring.\nPrefer natural conversation: share one point, then ask a follow-up question to gather context.\nDo not drift into unrelated topics.\nAvoid long encyclopedic responses or lengthy bullet lists unless explicitly asked.

# GCS persistence
GCS_BUCKET_NAME=ai-mentor-checkpoints
# GCS_THREADS_PREFIX=threads
# GOOGLE_CLOUD_PROJECT=my-gcp-project-id
```

---

## Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repo>/AI-AGENT-CHAT-RAG-PROD.git
   cd AI-AGENT-CHAT-RAG-PROD
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   # .\.venv\Scripts\Activate.ps1   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the API server:
   ```bash
   uvicorn agent.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. Open the web interface:
   **http://localhost:5500/webchat/index.html**

---

## Running with Docker

1. Build the image:
   ```bash
   docker build -t ai-startup-mentor .
   ```

2. Run the container:
   ```bash
   docker run -d \
     -p 8000:8000 \
     --env-file .env \
     --name ai-mentor \
     ai-startup-mentor
   ```

The API will be available at: **http://localhost:8000/api/chat**

---

## Deploying to Google Cloud Run

1. Authenticate and set your project:
   ```bash
   gcloud auth login
   gcloud config set project your-project-id
   ```

2. Build and deploy:
   ```bash
   gcloud run deploy ai-startup-mentor \
     --source . \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars "OPENAI_API_KEY=YOUR_KEY,OPENAI_MODEL=gpt-4o-mini,GCS_BUCKET_NAME=ai-mentor-checkpoints"
   ```

Once deployed, GCS persistence will work automatically. To switch to local storage, remove `GCS_BUCKET_NAME` from `.env` and use SQLite instead.

---

## API Endpoint

### `POST /api/chat`

#### Request
```json
{
  "message": "How should I test my MVP before launch?",
  "history": []
}
```

#### Response
```json
{
  "reply": "Start by interviewing potential users with a clickable prototype. Focus on learning, not selling."
}
```

---

## Tech Stack

- FastAPI — REST backend
- LangGraph — agent orchestration
- OpenAI API — LLM reasoning
- Google Cloud Storage — persistence layer (pluggable)
- HTML/CSS/JS — frontend
- Docker & Cloud Run — deployment

---

## Customization

You can modify:
- `prompts/system_prompt.json` → default mentor behavior
- `prompts/roles/*.txt` → persona definitions (CTO, PM, VC)
- `.env` → API key, model, bucket, system prompt

The system is designed to be modular — swap the LLM, storage backend, or frontend without changing the core logic.

