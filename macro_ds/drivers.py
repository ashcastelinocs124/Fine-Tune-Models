"""Model drivers — one `.step()` interface, two backends.

OpenAIDriver talks to the OpenAI API (teacher generation). VLLMDriver points the same
client at a local vLLM OpenAI-compatible endpoint (student serving). Because both return
a normalized `Message`, `agent.run_agent` is identical for teacher and student.
"""

from __future__ import annotations

import json
import re
from typing import Any

from macro_ds import config
from macro_ds.schema import Message, ToolCall

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[str | None, list[ToolCall] | None]:
    """Parse a Qwen3 assistant generation into (content, tool_calls).

    Qwen3 emits tool calls as `<tool_call>\n{"name":..,"arguments":..}\n</tool_call>` blocks,
    optionally preceded by reasoning text (the visible Thought). Returns the surrounding text
    as content and the parsed calls (or None if there are none / they don't parse).
    """
    calls: list[ToolCall] = []
    for i, blob in enumerate(_TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"__raw__": args}
        calls.append(ToolCall(id=f"call_{i}", name=name, arguments=args or {}))
    content = _TOOL_CALL_RE.sub("", text).strip()
    return (content or None), (calls or None)


def _messages_to_hf(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert the live transcript (list[Message]) to HF chat-template message dicts."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in m.tool_calls
                    ],
                }
            )
        elif m.role == "tool":
            d: dict[str, Any] = {"role": "tool", "content": m.content or ""}
            if m.name:
                d["name"] = m.name
            out.append(d)
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


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

    def step(self, messages: list[Message], tool_schemas: list[dict], allow_tools: bool = True) -> Message:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
        }
        if allow_tools:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "auto"
        # allow_tools=False -> omit tools so the model must answer with text (forced final report)
        resp = self.client.chat.completions.create(**kwargs)
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


class HFDriver:
    """Drives the loop with a local HF model (base + optional LoRA adapter), no server.

    A no-vLLM serving/eval path: loads Qwen3 in 4-bit, applies the SAME chat template used
    in training (parity), generates, and parses Qwen3 `<tool_call>` tags into ToolCall objects.
    """

    def __init__(self, model_name: str, adapter: str | None = None, max_new_tokens: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0}
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model
        self.max_new_tokens = max_new_tokens
        # stop generation at end-of-turn, not just <|endoftext|>
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self._eos = [t for t in {self.tok.eos_token_id, im_end} if t is not None]
        self.usage: dict[str, int] = {}
        self.meta = {"model": model_name, "adapter": adapter}

    def step(self, messages: list[Message], tool_schemas: list[dict], allow_tools: bool = True) -> Message:
        import torch

        hf = _messages_to_hf(messages)
        prompt = self.tok.apply_chat_template(
            hf,
            tools=tool_schemas if allow_tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self._eos,
                pad_token_id=self.tok.pad_token_id,
            )
        gen = out[0][enc["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        if allow_tools:
            content, tool_calls = parse_tool_calls(text)
        else:
            content, tool_calls = text.strip() or None, None
        return Message(role="assistant", content=content, tool_calls=tool_calls)
