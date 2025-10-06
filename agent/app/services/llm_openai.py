from __future__ import annotations

import os
from typing import AsyncGenerator, Dict, List, Sequence, Union, Any

from openai import AsyncOpenAI

# Optional: only needed if you want to accept LangChain message classes
try:
    from langchain_core.messages import (
        BaseMessage,
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        FunctionMessage,
    )
    _HAS_LC = True
except Exception:
    BaseMessage = object  # type: ignore
    _HAS_LC = False


OpenAIMsg = Dict[str, Any]
MsgInput = Sequence[Union[OpenAIMsg, "BaseMessage"]]


def _coerce_content(content: Any) -> str:
    """
    OpenAI expects a string (or a list of parts). We normalize common cases.
    - If content is list[str] -> join with newlines
    - If content is list[dict] (LC tool/parts) -> best-effort str()
    - Else -> str(content)
    """
    if isinstance(content, list):
        try:
            # if it's list[str]
            return "\n".join([c for c in content if isinstance(c, str)])
        except Exception:
            return str(content)
    return "" if content is None else str(content)


def _to_role_and_content(m: Any) -> OpenAIMsg:
    """
    Convert either a LangChain message or a raw {role, content} dict
    into OpenAI's {role, content} shape.
    """
    # Already a dict in the right shape
    if isinstance(m, dict) and "role" in m and "content" in m:
        return {"role": str(m["role"]), "content": _coerce_content(m["content"])}

    # LangChain messages (only if available)
    if _HAS_LC and isinstance(m, BaseMessage):
        if isinstance(m, SystemMessage):
            return {"role": "system", "content": _coerce_content(m.content)}
        if isinstance(m, HumanMessage):
            return {"role": "user", "content": _coerce_content(m.content)}
        if isinstance(m, AIMessage):
            return {"role": "assistant", "content": _coerce_content(m.content)}
        if isinstance(m, (ToolMessage, FunctionMessage)):
            # Map tool/function messages to "assistant" with inline content.
            # Adjust if you later use the Tools API.
            return {
                "role": "assistant",
                "content": _coerce_content(getattr(m, "content", "")),
            }
        # Fallback for unknown LC message types
        return {"role": "user", "content": _coerce_content(getattr(m, "content", ""))}

    # Fallback: try to read attributes like a LC BaseMessage
    role = getattr(m, "role", None)
    content = getattr(m, "content", None)
    if role is not None and content is not None:
        return {"role": str(role), "content": _coerce_content(content)}

    # Last resort: treat as user text
    return {"role": "user", "content": _coerce_content(m)}


def _normalize_messages(messages: MsgInput) -> List[OpenAIMsg]:
    norm: List[OpenAIMsg] = []
    for m in messages:
        msg = _to_role_and_content(m)
        # Drop empty content messages to avoid confusing completions
        if str(msg.get("content", "")).strip():
            norm.append(msg)
    return norm


class OpenAIChat:
    """
    Thin wrapper over OpenAI Chat Completions using the official async client.
    Accepts either LangChain message classes or raw {role, content} dicts.

    Env vars:
      - OPENAI_API_KEY (required)
      - OPENAI_BASE_URL (optional; default https://api.openai.com/v1)
      - OPENAI_MODEL (default gpt-4o-mini)
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def complete(self, messages: MsgInput) -> str:
        payload = _normalize_messages(messages)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=0.2,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()

    async def stream(self, messages: MsgInput) -> AsyncGenerator[str, None]:
        payload = _normalize_messages(messages)
        async with self.client.chat.completions.stream(
            model=self.model,
            messages=payload,
            temperature=0.2,
        ) as stream:
            async for event in stream:
                if event.type == "message.delta" and getattr(event.delta, "content", None):
                    yield event.delta.content
                elif event.type == "message.stop":
                    break
