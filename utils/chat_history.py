"""
ChatHistory — lightweight in-memory chat history manager.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str
    sources: List[str] = field(default_factory=list)


class ChatHistory:
    """Manages conversation turns for the Streamlit session."""

    def __init__(self):
        self._messages: List[Message] = []

    def add(self, role: str, content: str, sources: Optional[List[str]] = None) -> None:
        self._messages.append(Message(role=role, content=content, sources=sources or []))

    def get_all(self) -> List[dict]:
        return [
            {"role": m.role, "content": m.content, "sources": m.sources}
            for m in self._messages
        ]

    def get_last_n(self, n: int = 6) -> List[dict]:
        return self.get_all()[-n:]

    def count(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages = []

    def to_langchain_format(self) -> List[dict]:
        """Convert to LangChain message format for multi-turn memory."""
        result = []
        for m in self._messages:
            if m.role == "user":
                result.append({"type": "human", "content": m.content})
            else:
                result.append({"type": "ai", "content": m.content})
        return result
