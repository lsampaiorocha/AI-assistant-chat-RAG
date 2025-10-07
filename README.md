# 🚀 AI Startup Mentor — Interactive Multi-Expert Chat System

**Author:** Leonardo Sampaio Rocha

---

## Overview
This project implements an **AI Startup Mentor**, a conversational system that helps founders refine startup ideas, analyze product strategies, and explore funding paths.

The mentor provides general guidance on building and growing a company.  
When you need deeper insights, you can call on specialized experts:
- **PM (Product Manager):** Product strategy, validation, and roadmap design  
- **CTO (Chief Technology Officer):** Technical architecture, scalability, and cost trade-offs  
- **VC (Venture Capital Partner):** Market opportunity, traction, and fundraising guidance  
- Or call the **Committee**, which brings all three together to give multi-perspective feedback.

The backend is powered by **FastAPI**, using **OpenAI Chat Completions** and **LangGraph** to coordinate agent reasoning and persona routing.  
A minimal **HTML/JS frontend** is included under `webchat/`.

---

## Project Structure

```
AI-AGENT-CHAT-RAG-PROD/
├── agent/
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint
│   │   ├── state_graph.py     # Persona routing & LangGraph orchestration
│   │   └── services/          # LLM, RAG, and utility modules
│   ├── prompts/               # Persona prompt definitions (CTO, PM, VC)
│   └── data/                  # (Optional) Storage, cache, or local RAG data
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

## Running Locally (Development)

### 1. Clone the repository
```bash
git clone https://github.com/<your-repo>/AI-AGENT-CHAT-RAG-PROD.git
cd AI-AGENT-CHAT-RAG-PROD
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file in the project root
```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
CORS_ALLOW_ORIGINS=http://localhost:5500
```

### 5. Run the API server
```bash
uvicorn agent.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open the web interface:  
**http://localhost:5500/webchat/index.html**

---

## Running with Docker

### 1. Build the image
```bash
docker build -t ai-startup-mentor .
```

### 2. Run the container
```bash
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name ai-mentor \
  ai-startup-mentor
```

The API will be available at:  
**http://localhost:8000/api/chat**

---

## API Endpoint

### `POST /api/chat`

#### Request Body
```json
{
  "message": "Hi, what’s a good way to validate my startup idea?",
  "history": [],
  "stream": false
}
```

#### Response
```json
{
  "reply": "Start by interviewing 5–10 potential users to confirm the problem before building anything.",
  "phase": "mentor"
}
```

You can mention roles directly in your message:
- "What does the CTO think about scalability?"
- "Ask the VC if this is investable at seed stage."
- "Get committee advice on our go-to-market."

---

## How It Works

- **LangGraph** routes each message to the appropriate expert (Mentor, PM, CTO, VC, or Committee).  
- **OpenAI API** provides reasoning and persona-based responses.  
- **Frontend (webchat/)** offers a lightweight chat interface.  

---

## Tech Stack

- **FastAPI** — REST backend  
- **LangGraph** — stateful agent orchestration  
- **OpenAI Chat API** — LLM interaction  
- **HTML/CSS/JS** — lightweight frontend  
- **Docker** — containerized deployment  

---
