# agent/app/main.py
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from .services.llm_openai import OpenAIChat
from .services.rag import RAGPipeline, RetrievalResult
from .services.ingest import ingest_document, ingest_from_dir
from .state_graph import graph


_here = Path(__file__).resolve().parent
_agent_root = _here.parent
_repo_root = _agent_root.parent
load_dotenv(dotenv_path=_agent_root / ".env", override=True)
load_dotenv(dotenv_path=_repo_root / ".env", override=True)

# ✅ Carrega system prompt de JSON
_system_prompt_file = _repo_root / "prompts" / "system_prompt.json"
if _system_prompt_file.exists():
    data = json.loads(_system_prompt_file.read_text(encoding="utf-8"))
    DEFAULT_SYSTEM_PROMPT = data.get("prompt", "")
else:
    DEFAULT_SYSTEM_PROMPT = (
        "You are Steve Jobs acting as a startup mentor. "
        "Speak with vision, challenge assumptions, and give sharp, practical advice "
        "focused on entrepreneurship, innovation, and building great products."
    )


class Message(BaseModel):
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = Field(default_factory=list)
    stream: bool = False
    system_prompt: Optional[str] = None
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    citations: Optional[List[RetrievalResult]] = None
    phase: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = OpenAIChat()
    rag = RAGPipeline()
    app.state.sessions: Dict[str, Dict] = {}

    try:
        stats = rag.collection.count()
    except Exception:
        stats = 0

    if stats == 0:
        print("Chroma collection is empty. Running default ingestion...")
        default_doc = Path(_repo_root / "docs" / "GenAI interview.txt")
        if default_doc.exists():
            text = default_doc.read_text(encoding="utf-8")
            ingest_document(text)
            print("Ingestion completed.")
        else:
            print("No default document found to ingest.")
    else:
        print(f"Chroma already has {stats} documents.")

    app.state.llm = llm
    app.state.rag = rag

    yield
    print("Shutting down... cleanup if necessary")


def create_app() -> FastAPI:
    app = FastAPI(title="Agent API", version="0.1.0", lifespan=lifespan)

    allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        rag: RAGPipeline = app.state.rag
        sessions: Dict[str, Dict] = app.state.sessions

        thread_id = req.thread_id or "default-thread"

        prev_state = sessions.get(thread_id) or {
            "messages": [],
            "intro_done": False,
            "tests_done": 0,
            "general_done": 0,
        }

        # ✅ Se ainda não existe system prompt, injeta do JSON
        system_msg = req.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = prev_state["messages"].copy()
        if not any(m.get("role") == "system" for m in messages if isinstance(m, dict)):
            messages.insert(0, {"role": "system", "content": system_msg})

        messages.append({"role": "user", "content": req.message})

        input_state = {**prev_state, "messages": messages}

        try:
            result = await graph.ainvoke(
                input_state,
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 1},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Graph error: {e}")

        sessions[thread_id] = result

        msgs = result.get("messages", [])
        normalized_msgs = []
        for m in msgs:
            if isinstance(m, dict):
                normalized_msgs.append(m)
            else:
                normalized_msgs.append({
                    "role": getattr(m, "role", "assistant"),
                    "content": getattr(m, "content", str(m)),
                })

        reply = next(
            (m["content"] for m in reversed(normalized_msgs) if m["role"] == "assistant"),
            ""
        ).strip()
        if not reply:
            reply = "Thanks! Let’s continue."

        intro_done = bool(result.get("intro_done", False))
        tests_done = int(result.get("tests_done", 0))
        general_done = int(result.get("general_done", 0))
        if general_done >= 4:
            phase = "feedback"
        elif tests_done >= 3:
            phase = "exploration"
        elif intro_done:
            phase = "testing"
        else:
            phase = "introduction"

        try:
            citations = await rag.retrieve(req.message)
        except Exception:
            citations = []

        return ChatResponse(reply=reply, citations=citations, phase=phase)

    @app.get("/debug/state")
    async def debug_state(session_id: str = "default"):
        s = app.state.sessions.get(session_id, {})
        return {
            "session_id": session_id,
            "intro_done": s.get("intro_done"),
            "tests_done": s.get("tests_done"),
            "general_done": s.get("general_done"),
            "messages_tail": s.get("messages", [])[-6:],
        }

    return app


app = create_app()
