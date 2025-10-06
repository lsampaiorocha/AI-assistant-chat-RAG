from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import json
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.base import CheckpointTuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import CheckpointTuple

from fastapi import HTTPException


from .services.llm_openai import OpenAIChat
from .services.rag import RAGPipeline, RetrievalResult
from .services.ingest import ingest_document
from .state_graph import graph

_here = Path(__file__).resolve().parent
_agent_root = _here.parent
_repo_root = _agent_root.parent
load_dotenv(dotenv_path=_agent_root / ".env", override=True)
load_dotenv(dotenv_path=_repo_root / ".env", override=True)

# --- System Prompt -----------------------------------------------------------
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

# --- Schemas ----------------------------------------------------------------
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

# --- Lifespan ---------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = OpenAIChat()
    rag = RAGPipeline()

    db_path = _repo_root / "data"
    db_path.mkdir(exist_ok=True)
    memory_db = db_path / "memory.sqlite"

    # open/close saver for the whole app lifetime
    async with AsyncSqliteSaver.from_conn_string(str(memory_db)) as checkpointer:
        app.state.llm = llm
        app.state.rag = rag
        app.state.checkpointer = checkpointer
        print(f"AsyncSqliteSaver: {memory_db}")

        # one-time ingestion if empty
        try:
            stats = rag.collection.count()
        except Exception:
            stats = 0
        if stats == 0:
            default_doc = _repo_root / "docs" / "GenAI interview.txt"
            if default_doc.exists():
                ingest_document(default_doc.read_text(encoding="utf-8"))
                print("Ingestion completed.")
            else:
                print("No default document found to ingest.")

        yield

# --- App --------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="Agent API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        rag: RAGPipeline = app.state.rag
        checkpointer: AsyncSqliteSaver = app.state.checkpointer

        thread_id = req.thread_id or "default-thread"
        system_msg = req.system_prompt or DEFAULT_SYSTEM_PROMPT

        # Recupera o checkpoint salvo anteriormente (se existir)
        ckpt: Optional[CheckpointTuple] = await checkpointer.aget_tuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": "default"}}
        )
        prev_state = ckpt.checkpoint if ckpt else {"messages": []}

        print(f"=>DEBUG SAVED CHKP: {prev_state}")

        # Restaura histórico em formato LangChain
        messages = prev_state.get("messages", [])
        if not messages:
            messages = [SystemMessage(content=system_msg)]

        messages.append(HumanMessage(content=req.message))
        input_state = {"messages": messages}

        print(f"=>DEBUG INPUT STATE: {input_state}")

        # Executa o grafo com o estado anterior e persiste automaticamente
        try:
            result = await graph.ainvoke(
                input_state,
                config={
                    "configurable": {"thread_id": thread_id, "checkpoint_ns": "default"},
                    "checkpointer": checkpointer,
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Graph error: {e}")
        
        print(f"=>DEBUG RESULT: {result}")

        # Extrai a resposta do último AIMessage
        msgs = result.get("messages", [])
        reply_msg = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
        reply = reply_msg.content.strip() if reply_msg else "(no assistant response found)"


        print(f"=>DEBUG REPLY: {reply}")

        # Persiste manualmente o novo estado, garantindo ID único
        await checkpointer.aput(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": "default"}},
            checkpoint={"id": f"{thread_id}-latest", "messages": msgs},
            metadata={},
            new_versions=[],
        )

        # Recupera citações (opcional)
        try:
            citations = await rag.retrieve(req.message)
        except Exception:
            citations = []

        return ChatResponse(reply=reply, citations=citations)
    
    return app

app = create_app()
