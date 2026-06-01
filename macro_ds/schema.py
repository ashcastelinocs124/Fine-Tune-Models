"""Normalized, provider-agnostic trace format.

This is THE contract that flows through the whole pipeline. Both the OpenAI driver
(teacher generation) and the vLLM driver (student serving) emit `Message`/`Trace`
objects, and `to_dataset` consumes them. Keeping one internal representation is what
makes train/serve parity checkable.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A single tool/function invocation requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One turn in a conversation, normalized across providers."""

    role: Role
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None  # only on `tool` messages
    name: Optional[str] = None  # tool name on `tool` messages

    def to_openai(self) -> dict[str, Any]:
        """Render to the OpenAI chat-completions message shape."""
        if self.role == "tool":
            out: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content or "",
            }
            if self.name:
                out["name"] = self.name
            return out
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        return out

    @classmethod
    def from_openai(cls, d: dict[str, Any]) -> "Message":
        """Parse an OpenAI message dict (or SDK object cast to dict) into a Message."""
        role = d["role"]
        if role == "tool":
            return cls(
                role="tool",
                content=d.get("content") or "",
                tool_call_id=d.get("tool_call_id"),
                name=d.get("name"),
            )
        tool_calls = None
        raw_calls = d.get("tool_calls") or []
        if raw_calls:
            tool_calls = []
            for tc in raw_calls:
                fn = tc["function"]
                args = fn.get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        # keep the raw string under a sentinel key so downstream filters can reject it
                        args = {"__raw__": args}
                tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=args))
        return cls(role=role, content=d.get("content"), tool_calls=tool_calls)


class Trace(BaseModel):
    """A full agent trajectory for one research question."""

    question: str
    asof_date: str
    messages: list[Message] = Field(default_factory=list)
    final_report: Optional[str] = None
    steps: int = 0
    tool_errors: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def completed(self) -> bool:
        """True if the agent produced a non-empty final report (did not force-stop empty)."""
        return bool(self.final_report and self.final_report.strip())

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Trace":
        return cls.model_validate_json(s)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trace":
        return cls.model_validate(d)
