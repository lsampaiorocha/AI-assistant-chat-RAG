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
        "You are an AI Mentor inspired at Steve Jobs, acting as a mentor for startup founders.",
        "Speak with sharp insight, challenge assumptions, and push people to think bigger.",
        "Always focus on product excellence, user experience, innovation, and building impactful companies.",
        "Keep answers short (2–4 sentences), practical, and inspiring.",
        "Prefer natural conversation: share one point, then ask a follow-up question to gather context.",
        "Do not drift into unrelated topics.",
        "Avoid long encyclopedic responses or lengthy bullet lists unless explicitly asked."
    )


# -----------------------------------------------------------------------------
# GCS persistence (configurable via .env)
# -----------------------------------------------------------------------------
# Expect a variable in .env like:
#   GCS_BUCKET_NAME=ai-mentor-checkpoints
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
THREADS_PREFIX = os.getenv("GCS_THREADS_PREFIX", "threads")

if not BUCKET_NAME:
    raise RuntimeError(
        "Missing environment variable 'GCS_BUCKET_NAME'. "
        "Please set it in your .env file, e.g., GCS_BUCKET_NAME=ai-mentor-checkpoints"
    )

_gcs_client: Optional[storage.Client] = None


def _gcs() -> storage.Client:
    """Return a singleton GCS client, respecting GOOGLE_CLOUD_PROJECT if set."""
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
    """Load a thread’s messages from GCS. Return [] if not found."""
    try:
        blob = _thread_blob(thread_id)
        if not blob.exists():
            print(f"No thread file found for '{thread_id}' in gs://{BUCKET_NAME}/{THREADS_PREFIX}/")
            return []
        text = blob.download_as_text(encoding="utf-8")
        if not text:
            return []
        data = json.loads(text)
        msgs = data.get("messages", [])
        if not isinstance(msgs, list):
            msgs = []
        print(f"Loaded thread '{thread_id}' with {len(msgs)} message(s) from GCS")
        return msgs
    except Exception as e:
        print(f"Could not load thread '{thread_id}' from GCS: {e}")
        return []


def save_thread_to_gcs(thread_id: str, messages: List[dict]) -> None:
    """Save a thread’s messages back to GCS (overwrite full history)."""
    try:
        payload = {"messages": messages}
        blob = _thread_blob(thread_id)
        blob.upload_from_string(
            data=json.dumps(payload, ensure_ascii=False),
            content_type="application/json",
        )
        print(f"Saved thread '{thread_id}' with {len(messages)} message(s) to GCS")
    except Exception as e:
        print(f"Could not save thread '{thread_id}' to GCS: {e}")


def _normalize_msg(m: dict) -> dict:
    """Ensure a message is a plain dict with role/content."""
    if isinstance(m, dict):
        return {"role": m.get("role"), "content": m.get("content")}
    role = getattr(m, "role", None)
    content = getattr(m, "content", None)
    return {"role": role, "content": content}


def _diff_new_assistant_messages(before: List[dict], after: List[dict]) -> List[dict]:
    """Return assistant messages that appear in 'after' but not in 'before'."""
    if len(after) > len(before):
        tail = after[len(before):]
        return [_normalize_msg(m) for m in tail if isinstance(m, dict) and m.get("role") == "assistant"]

    seen = {
        (m.get("role"), json.dumps(m.get("content"), ensure_ascii=False)
         if isinstance(m.get("content"), (dict, list)) else str(m.get("content")))
        for m in before
    }
    out = []
    for m in after:
        key = (m.get("role"), json.dumps(m.get("content"), ensure_ascii=False)
               if isinstance(m.get("content"), (dict, list)) else str(m.get("content")))
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
    history: List[Message] = Field(default_factory=list)
    stream: bool = False
    system_prompt: Optional[str] = None
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


# -----------------------------------------------------------------------------
# Debug helper
# -----------------------------------------------------------------------------
def _print_messages(label: str, messages: List[dict]) -> None:
    print(f"\n{label}: total {len(messages)} message(s)")
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
    app.state.llm = llm
    yield


# -----------------------------------------------------------------------------
# App factory
# -----------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="Agent API", version="0.4.2-gcs-env", lifespan=lifespan)
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
        thread_id = req.thread_id or "web-fixed"
        system_msg = req.system_prompt or DEFAULT_SYSTEM_PROMPT

        # 1) Load history
        history: List[dict] = [_normalize_msg(m) for m in load_thread_from_gcs(thread_id)]
        if not history:
            history = [{"role": "system", "content": system_msg}]

        _print_messages(f"Loaded (thread={thread_id})", history)

        # 2) Add user message
        history.append({"role": "user", "content": req.message})

        # 3) Run graph
        try:
            result = await graph.ainvoke(
                {"messages": history},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Graph error: {e}")

        # 4) Detect new assistant messages
        result_msgs: List[dict] = [_normalize_msg(m) for m in result.get("messages", [])]
        new_assistant = _diff_new_assistant_messages(before=history, after=result_msgs or history)
        if new_assistant:
            history.extend(new_assistant)

        # 5) Extract reply
        last_assistant = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
        reply = last_assistant.get("content", "").strip() if last_assistant else "(no assistant response found)"

        # 6) Persist thread
        save_thread_to_gcs(thread_id, history)
        _print_messages(f"Saved (thread={thread_id})", history)

        return ChatResponse(reply=reply)

    return app


app = create_app()
