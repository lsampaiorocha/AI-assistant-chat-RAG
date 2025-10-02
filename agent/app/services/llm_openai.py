

from __future__ import annotations

import os
from typing import AsyncGenerator, Dict, List

from openai import AsyncOpenAI


class OpenAIChat:
    """Thin wrapper over OpenAI Chat Completions/Responses using the official OpenAI client.

    Reads configuration from environment variables:
    - OPENAI_API_KEY (required)
    - OPENAI_BASE_URL (optional; for Azure/OpenAI-compatible endpoints)
    - OPENAI_MODEL (default: gpt-4o-mini)
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        # Official async client
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        """One-shot chat completion (non-streaming)."""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=False,
        )
        return resp.choices[0].message.content.strip()

    async def stream(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Streaming chat completion, yielding text chunks incrementally."""
        async with self.client.chat.completions.stream(
            model=self.model,
            messages=messages,
            temperature=0.2,
        ) as stream:
            async for event in stream:
                if event.type == "message.delta" and event.delta.content:
                    yield event.delta.content
                elif event.type == "message.stop":
                    break
