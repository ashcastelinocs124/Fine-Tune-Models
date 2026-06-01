"""The provider-agnostic ReAct loop.

`run_agent` is the single loop used by BOTH teacher generation and student serving —
it never knows which model is behind the `driver`. That is the train/serve parity
guarantee: identical control flow, identical tool execution, identical formatting.
"""

from __future__ import annotations

from typing import Protocol

from macro_ds.prompts import build_system_prompt
from macro_ds.schema import Message, Trace
from macro_ds.tools import execute_tool, openai_tool_schemas


class Driver(Protocol):
    """Anything that can take the running transcript + tool schemas and return the
    next assistant Message. Implementations live in drivers.py."""

    def step(self, messages: list[Message], tool_schemas: list[dict]) -> Message: ...


def run_agent(question: str, asof_date: str, driver: Driver, max_steps: int = 12) -> Trace:
    system = build_system_prompt(asof_date=asof_date, max_steps=max_steps)
    messages: list[Message] = [
        Message(role="system", content=system),
        Message(role="user", content=question),
    ]
    tool_schemas = openai_tool_schemas()

    steps = 0
    tool_errors = 0
    final_report: str | None = None

    while steps < max_steps:
        assistant = driver.step(messages, tool_schemas)
        messages.append(assistant)
        steps += 1

        if assistant.tool_calls:
            for tc in assistant.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                if result.startswith("ERROR:"):
                    tool_errors += 1
                messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id, name=tc.name)
                )
            continue

        # no tool calls -> the assistant produced the final deliverable
        final_report = assistant.content
        break

    return Trace(
        question=question,
        asof_date=asof_date,
        messages=messages,
        final_report=final_report,
        steps=steps,
        tool_errors=tool_errors,
        usage=getattr(driver, "usage", {}) or {},
        meta=getattr(driver, "meta", {}) or {},
    )
