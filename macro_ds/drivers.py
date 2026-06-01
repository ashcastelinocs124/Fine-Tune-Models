"""Model drivers — one `.step()` interface, two backends.

OpenAIDriver talks to the OpenAI API (teacher generation). VLLMDriver points the same
client at a local vLLM OpenAI-compatible endpoint (student serving). Because both return
a normalized `Message`, `agent.run_agent` is identical for teacher and student.
"""

from __future__ import annotations

from typing import Any

from macro_ds import config
from macro_ds.schema import Message


def _sdk_message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an OpenAI SDK chat message object into the dict shape Message.from_openai expects."""
    out: dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", None)}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    return out


class OpenAIDriver:
    """Drives the loop with an OpenAI(-compatible) chat-completions endpoint."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key or config.get_key("OPENAI_API_KEY"), base_url=base_url)
        self.usage: dict[str, int] = {}
        self.meta: dict[str, Any] = {"model": model, "base_url": base_url}

    def step(self, messages: list[Message], tool_schemas: list[dict]) -> Message:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[m.to_openai() for m in messages],
            tools=tool_schemas,
            tool_choice="auto",
        )
        self._accumulate_usage(resp)
        return Message.from_openai(_sdk_message_to_dict(resp.choices[0].message))

    def _accumulate_usage(self, resp: Any) -> None:
        u = getattr(resp, "usage", None)
        if not u:
            return
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = getattr(u, k, None)
            if v is not None:
                self.usage[k] = self.usage.get(k, 0) + v


class VLLMDriver(OpenAIDriver):
    """Drives the loop with a local vLLM server (student). Same wire format as OpenAI."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        super().__init__(model=model, base_url=base_url, api_key=api_key)
