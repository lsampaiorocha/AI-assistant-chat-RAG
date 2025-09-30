from __future__ import annotations

import os
from typing import AsyncGenerator, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

from .services.llm_openai import OpenAIChat
from .services.rag import RAGPipeline, RetrievalResult
from .services.ingest import ingest_document, ingest_from_dir
from .state_graph import conversation_app


# Load .env from agent/ and optionally repo root as fallback
_here = Path(__file__).resolve().parent  # agent/app
_agent_root = _here.parent               # agent/
_repo_root = _agent_root.parent          # project root
load_dotenv(dotenv_path=_agent_root / ".env", override=False)
load_dotenv(dotenv_path=_repo_root / ".env", override=False)


class Message(BaseModel):
    """Single message in the chat history."""
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = Field(default_factory=list)
    stream: bool = False
    system_prompt: Optional[str] = None
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Schema for chat responses.

    Attributes:
        reply: Assistant’s reply text.
        citations: Optional list of retrieval results (from RAG).
        phase: Optional current phase label (debug/helpful).
    """
    reply: str
    citations: Optional[List[RetrievalResult]] = None
    phase: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for FastAPI.

    - Initializes services (LLM and RAG).
    - Ensures ingestion if the ChromaDB is empty.
    - Initializes in-memory session state for LangGraph.
    """
    # Initialize services
    llm = OpenAIChat()
    rag = RAGPipeline()

    # Very simple in-memory sessions: session_id -> LangGraph state
    app.state.sessions: Dict[str, Dict] = {}

    # Ensure ingestion on startup if DB is empty
    try:
        stats = rag.collection.count()
    except Exception:
        stats = 0

    # rag.collection.delete(ids=rag.collection.get()["ids"])  # <- manual wipe, if needed
    # ingest_from_dir(file_types=["txt", "mp3"])               # <- bulk ingest, if desired

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

    # Store services in app.state for use inside endpoints
    app.state.llm = llm
    app.state.rag = rag

    yield  # control returns to FastAPI runtime here

    # Shutdown logic (if needed)
    print("Shutting down... cleanup if necessary")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    - Sets up CORS middleware.
    - Uses `lifespan` to manage startup/shutdown.
    - Defines health and chat endpoints.
    """
    app = FastAPI(title="Agent API", version="0.1.0", lifespan=lifespan)

    # Configure CORS
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
        """Health check endpoint."""
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """
        One-step-per-call chat endpoint using LangGraph (stateful) + RAG citations.
        We persist the graph state per thread_id in app.state.sessions so flags/counters
        (intro_done/tests_done/general_done) survive between requests.
        """
        rag: RAGPipeline = app.state.rag
        sessions: Dict[str, Dict] = app.state.sessions

        thread_id = req.thread_id or "default-thread"

        # 1) Load last state for this thread, or start fresh
        prev_state = sessions.get(thread_id) or {
            "messages": [],
            "intro_done": False,
            "tests_done": 0,
            "general_done": 0,
        }

        # 2) Merge the new user message with the previous state
        #    (we keep the flags/counters so router can progress)
        input_state = {
            **prev_state,
            "messages": prev_state["messages"] + [{"role": "user", "content": req.message}],
        }

        # 3) Run exactly one node (graph compiled with interrupt_after)
        try:
            result = conversation_app.invoke(
                input_state,
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 1},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Graph error: {e}")

        # 4) Persist the updated state for this thread
        sessions[thread_id] = result

        # 5) Extract the last assistant reply produced this step
        msgs = result.get("messages", [])
        reply = next((m["content"] for m in reversed(msgs) if m.get("role") == "assistant"), "").strip()
        if not reply:
            # very safe fallback so UI never shows an empty bubble
            reply = "Thanks! Let’s continue."

        # 6) Derive a phase label for your UI (optional)
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

        # 7) Best-effort RAG citations
        try:
            citations = await rag.retrieve(req.message)
        except Exception:
            citations = []

        return ChatResponse(reply=reply, citations=citations, phase=phase)


    @app.get("/debug/state")
    async def debug_state(session_id: str = "default"):
        """Peek at the saved LangGraph state for a session (debug only)."""
        s = app.state.sessions.get(session_id, {})
        return {
            "session_id": session_id,
            "intro_done": s.get("intro_done"),
            "tests_done": s.get("tests_done"),
            "general_done": s.get("general_done"),
            "messages_tail": s.get("messages", [])[-6:],  # last few messages
        }

    return app


app = create_app()
