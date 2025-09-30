from __future__ import annotations

import os
import asyncio
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

from .services.llm_openai import OpenAIChat
from .services.rag import RAGPipeline, RetrievalResult


# Load .env from agent/ and optionally repo root as fallback
_here = Path(__file__).resolve().parent  # agent/app
_agent_root = _here.parent               # agent/
_repo_root = _agent_root.parent          # project root
load_dotenv(dotenv_path=_agent_root / ".env", override=False)
load_dotenv(dotenv_path=_repo_root / ".env", override=False)


class Message(BaseModel):
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = Field(default_factory=list)
    stream: bool = False
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    citations: Optional[List[RetrievalResult]] = None


def create_app() -> FastAPI:
    app = FastAPI(title="Agent API", version="0.1.0")

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

    # Services
    llm = OpenAIChat()
    rag = RAGPipeline() 

    #Just for checking if it is working
    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        retrievals = await rag.retrieve(req.message)
        augmented_system = req.system_prompt or os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful AI assistant.")
        context_block = rag.format_context(retrievals)

        # Construct messages for the LLM
        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": augmented_system + ("\n\n" + context_block if context_block else "")})
        for m in req.history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": req.message})

        if not req.stream:
            try:
                text = await llm.complete(messages)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            return ChatResponse(reply=text, citations=retrievals)

        async def token_stream() -> AsyncGenerator[bytes, None]:
            try:
                async for token in llm.stream(messages):
                    yield token.encode("utf-8")
            except Exception as e:
                yield f"\n[error] {str(e)}".encode("utf-8")

        return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")

    return app


app = create_app()


