"""Independent loss-mask verification against the real Qwen3 chat template.

LLaMA-Factory computes the training loss mask internally. This module recomputes it
ourselves — rendering the record through the actual Qwen3 tokenizer and masking every
token except assistant-generated ones (reasoning + tool calls + final report). Stage 0
asserts our mask and LLaMA-Factory's agree, so a template/masking bug can't silently
poison training.

Technique:
- `start` of each assistant turn = len(render(messages[:i], add_generation_prompt=True)),
  i.e. the token right after that turn's `<|im_start|>assistant\n` header. (Verified
  prefix-consistent with the full transcript.)
- `end` = the index just past that turn's closing `<|im_end|>` token, found by scanning
  the transcript forward from `start` for the first `<|im_end|>`. This is robust to what
  FOLLOWS the turn — a bare assistant header (final turn) vs. a longer
  `<|im_start|>user\n<tool_response>` block (after a tool-calling turn) — which a
  fixed-length subtraction got wrong, overshooting into the next turn's framing.

Qwen3-specific notes:
- The template has NO `{% generation %}` blocks, so transformers'
  `return_assistant_tokens_mask` does not work — we diff prefixes ourselves.
- `ids_all` is built as `render(messages, add_generation_prompt=True)[:-H]` so the final
  assistant turn is rendered historical (consistent with the others) rather than getting
  the special last-message treatment of an add_generation_prompt=False render.
- We never pass `enable_thinking` (leaving it undefined avoids the prompt's empty-think
  injection at the start boundary).
"""

from __future__ import annotations

import json
from typing import Any

IGNORE = -100


def _maybe_json(s: Any) -> Any:
    if isinstance(s, str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    return s


def _to_hf_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the HF chat-template message list from our dataset record."""
    messages: list[dict[str, Any]] = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    for m in record["messages"]:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                # HF templates expect arguments as an object, not a JSON string
                                "arguments": _maybe_json(tc["function"]["arguments"]),
                            },
                        }
                        for tc in m["tool_calls"]
                    ],
                }
            )
        elif role == "tool":
            tmsg: dict[str, Any] = {"role": "tool", "content": m.get("content") or ""}
            if m.get("name"):
                tmsg["name"] = m["name"]
            messages.append(tmsg)
        else:
            messages.append({"role": role, "content": m.get("content") or ""})
    return messages


def _render(tok, messages, tools, add_generation_prompt: bool) -> str:
    # Deliberately do NOT pass enable_thinking — see module docstring.
    return tok.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _encode(tok, messages, tools, add_generation_prompt: bool) -> list[int]:
    # Render to text first (robust across transformers versions), then tokenize without
    # adding extra special tokens — the template already contains them as literal tokens.
    text = _render(tok, messages, tools, add_generation_prompt)
    return tok(text, add_special_tokens=False).input_ids


def render_and_mask(record: dict[str, Any], tok) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) where labels are -100 except on assistant-generated tokens."""
    messages = _to_hf_messages(record)
    tools = json.loads(record["tools"]) if record.get("tools") else None
    offset = 1 if record.get("system") else 0

    asst_indices = [i + offset for i, m in enumerate(record["messages"]) if m["role"] == "assistant"]
    if not asst_indices:
        raise RuntimeError("trace has no assistant turns to train on")

    # H = length of the trailing assistant generation-prompt header. messages[:first_asst]
    # ends in a non-assistant message, so its add_generation_prompt=False render is
    # think-injection-free and the difference is exactly the header.
    j0 = asst_indices[0]
    H = len(_encode(tok, messages[:j0], tools, True)) - len(_encode(tok, messages[:j0], tools, False))
    if H <= 0:
        raise RuntimeError("could not determine generation-prompt header length")

    # Fully-historical transcript (no last-turn special treatment): render with a trailing
    # generation prompt and strip it. All assistant turns are then rendered identically.
    ids_all = _encode(tok, messages, tools, True)[:-H]
    labels = [IGNORE] * len(ids_all)

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tok.unk_token_id:
        raise RuntimeError("could not resolve <|im_end|> token id")

    for hf_i in asst_indices:
        before = _encode(tok, messages[:hf_i], tools, True)  # transcript(:hf_i) + header
        start = len(before)
        if ids_all[:start] != before:
            raise RuntimeError(f"template not prefix-consistent at assistant turn hf_i={hf_i}")
        # end = just past this turn's own <|im_end|> (robust to whatever follows the turn)
        try:
            end = ids_all.index(im_end_id, start) + 1
        except ValueError:
            end = len(ids_all)
        for k in range(start, end):
            labels[k] = ids_all[k]

    return ids_all, labels
