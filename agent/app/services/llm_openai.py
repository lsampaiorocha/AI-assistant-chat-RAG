from __future__ import annotations

import os
from typing import AsyncGenerator, Dict, List, Sequence, Union, Any

from openai import AsyncOpenAI


OpenAIMsg = Dict[str, Any]
MsgInput = Sequence[Union[OpenAIMsg, str]]


def _coerce_content(content: Any) -> str:
    """Normalize content into a string for OpenAI API."""
    if isinstance(content, list):
        try:
            return "\n".join([str(c) for c in content if c])
        except Exception:
            return str(content)
    return "" if content is None else str(content)


def _to_role_and_content(m: Any) -> OpenAIMsg:
    """Convert anything like a dict or text into OpenAI's {role, content} shape."""
    # already a dict
    if isinstance(m, dict) and "role" in m and "content" in m:
        return {"role": str(m["role"]), "content": _coerce_content(m["content"])}

    # allow shorthand "user text"
    if isinstance(m, str):
        return {"role": "user", "content": m}

    # fallback for objects with attributes
    role = getattr(m, "role", None)
    content = getattr(m, "content", None)
    if role and content:
        return {"role": str(role), "content": _coerce_content(content)}

    # last resort: treat as user input
    return {"role": "user", "content": _coerce_content(m)}


def _normalize_messages(messages: MsgInput) -> List[OpenAIMsg]:
    norm: List[OpenAIMsg] = []
    for m in messages:
        msg = _to_role_and_content(m)
        if str(msg.get("content", "")).strip():
            norm.append(msg)
    return norm


class OpenAIChat:
    """
    Minimal async OpenAI Chat wrapper (no LangChain).

    Env vars:
      - OPENAI_API_KEY (required)
      - OPENAI_BASE_URL (optional, default https://api.openai.com/v1)
      - OPENAI_MODEL (default gpt-4o-mini)
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def complete(self, messages: MsgInput) -> str:
        """Return a single chat completion response."""
        payload = _normalize_messages(messages)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=0.2,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()

    async def stream(self, messages: MsgInput) -> AsyncGenerator[str, None]:
        """Stream tokens incrementally."""
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
