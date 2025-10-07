from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import json
import os
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google.cloud import storage

from .services.llm_openai import OpenAIChat
from .services.rag import RAGPipeline, RetrievalResult
from .services.ingest import ingest_document
from .state_graph import graph


# -----------------------------------------------------------------------------
# Paths & environment
# -----------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
_agent_root = _here.parent
_repo_root = _agent_root.parent
load_dotenv(dotenv_path=_agent_root / ".env", override=True)
load_dotenv(dotenv_path=_repo_root / ".env", override=True)


# -----------------------------------------------------------------------------
# System prompt
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# GCS persistence (direct, per-thread JSON)
# -----------------------------------------------------------------------------
BUCKET_NAME = "ai-mentor-checkpoints"
THREADS_PREFIX = "threads"  # gs://<bucket>/threads/<thread_id>.json

_gcs_client: Optional[storage.Client] = None


def _gcs() -> storage.Client:
    """Gets a singleton GCS client. Respects GOOGLE_CLOUD_PROJECT if set."""
    global _gcs_client
    if _gcs_client is None:
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
        _gcs_client = storage.Client(project=project) if project else storage.Client()
    return _gcs_client


def _thread_blob(thread_id: str) -> storage.Blob:
    bucket = _gcs().bucket(BUCKET_NAME)
    path = f"{THREADS_PREFIX}/{thread_id}.json"
    return bucket.blob(path)


def load_thread_from_gcs(thread_id: str) -> List[dict]:
    """
    Return full message list from gs://bucket/threads/<thread_id>.json.
    If not found, return [].
    """
    try:
        blob = _thread_blob(thread_id)
        if not blob.exists():
            print(f"ℹ No thread file found for '{thread_id}' in gs://{BUCKET_NAME}/{THREADS_PREFIX}/")
            return []
        text = blob.download_as_text(encoding="utf-8")
        if not text:
            return []
        data = json.loads(text)
        msgs = data.get("messages", [])
        if not isinstance(msgs, list):
            msgs = []
        print(f"☁️ Loaded thread '{thread_id}' with {len(msgs)} message(s) from GCS")
        return msgs
    except Exception as e:
        print(f"⚠ Could not load thread '{thread_id}' from GCS: {e}")
        return []


def save_thread_to_gcs(thread_id: str, messages: List[dict]) -> None:
    """
    Overwrite gs://bucket/threads/<thread_id>.json with the full merged history.
    """
    try:
        payload = {"messages": messages}
        blob = _thread_blob(thread_id)
        blob.upload_from_string(
            data=json.dumps(payload, ensure_ascii=False),
            content_type="application/json",
        )
        print(f"✅ Saved thread '{thread_id}' with {len(messages)} message(s) to GCS")
    except Exception as e:
        print(f"⚠ Could not save thread '{thread_id}' to GCS: {e}")


def _normalize_msg(m: dict) -> dict:
    """Ensure a message is a plain dict with role/content."""
    if isinstance(m, dict):
        role = m.get("role")
        content = m.get("content")
        return {"role": role, "content": content}
    # If pydantic Message sneaks in, coerce
    role = getattr(m, "role", None)
    content = getattr(m, "content", None)
    return {"role": role, "content": content}


def _diff_new_assistant_messages(before: List[dict], after: List[dict]) -> List[dict]:
    """
    Return assistant messages that are present in 'after' but not in 'before'.
    We compare by sequence index from the tail and role, preferring simple diff:
    """
    # Fast path: if 'after' is longer, take the tail slice; keep only assistant roles.
    if len(after) > len(before):
        tail = after[len(before):]
        return [_normalize_msg(m) for m in tail if isinstance(m, dict) and m.get("role") == "assistant"]

    # Fallback: set-based difference by (role, content) tuples
    seen = {(m.get("role"), json.dumps(m.get("content"), ensure_ascii=False) if isinstance(m.get("content"), (dict, list)) else str(m.get("content"))) for m in before}
    out = []
    for m in after:
        key = (m.get("role"), json.dumps(m.get("content"), ensure_ascii=False) if isinstance(m.get("content"), (dict, list)) else str(m.get("content")))
        if key not in seen and m.get("role") == "assistant":
            out.append(_normalize_msg(m))
    return out


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class Message(BaseModel):
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str | list | dict


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = Field(default_factory=list)  # ignored; we always load from GCS
    stream: bool = False
    system_prompt: Optional[str] = None
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    citations: Optional[List[RetrievalResult]] = None
    phase: Optional[str] = None


# -----------------------------------------------------------------------------
# Debug helper
# -----------------------------------------------------------------------------
def _print_messages(label: str, messages: List[dict]) -> None:
    print(f"\n🔍 {label}: total {len(messages)} message(s)")
    for i, m in enumerate(messages[-6:], start=max(1, len(messages) - 5)):
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        elif isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        content = str(content).replace("\n", " ")
        print(f"   {i}. ({role}) {content[:160]}{'...' if len(content) > 160 else ''}")


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = OpenAIChat()
    rag = RAGPipeline()

    app.state.llm = llm
    app.state.rag = rag

    # Optional RAG ingestion (unchanged)
    try:
        stats = rag.collection.count()
    except Exception:
        stats = 0
    if stats == 0:
        default_doc = _repo_root / "docs" / "GenAI interview.txt"
        if default_doc.exists():
            ingest_document(default_doc.read_text(encoding="utf-8"))
            print("✅ Ingestion completed.")
        else:
            print("ℹ No default document found to ingest.")

    yield
    # No flush needed; persistence is immediate to GCS


# -----------------------------------------------------------------------------
# App factory
# -----------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="Agent API", version="0.4.1-gcs", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        rag: RAGPipeline = app.state.rag

        thread_id = req.thread_id or "web-fixed"
        system_msg = req.system_prompt or DEFAULT_SYSTEM_PROMPT

        # 1) Load full history from GCS
        history: List[dict] = [ _normalize_msg(m) for m in load_thread_from_gcs(thread_id) ]

        # If empty, start with system prompt
        if not history:
            history = [{"role": "system", "content": system_msg}]

        _print_messages(f"Loaded (thread={thread_id})", history)

        # 2) Append the new user message (APPEND)
        history.append({"role": "user", "content": req.message})

        # 3) Run the graph (no checkpointer; we own persistence)
        try:
            result = await graph.ainvoke(
                {"messages": history},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Graph error: {e}")

        # 4) Graph may return only the delta or a full list.
        #    We compute "new assistant messages" and append them to our local history.
        result_msgs: List[dict] = [ _normalize_msg(m) for m in result.get("messages", []) ]
        new_assistant = _diff_new_assistant_messages(before=history, after=result_msgs or history)

        # If graph returned nothing, we keep history as-is (already has user msg)
        if new_assistant:
            history.extend(new_assistant)

        # 5) Extract reply from the last assistant message
        last_assistant = next(
            (m for m in reversed(history) if m.get("role") == "assistant"),
            None,
        )
        reply = last_assistant.get("content", "").strip() if last_assistant else "(no assistant response found)"

        # 6) Save full merged history back to GCS (APPEND semantics via overwrite of merged)
        save_thread_to_gcs(thread_id, history)
        _print_messages(f"Saved (thread={thread_id})", history)

        # 7) Optional citations
        try:
            citations = await rag.retrieve(req.message)
        except Exception:
            citations = []

        return ChatResponse(reply=reply, citations=citations)

    return app


app = create_app()
