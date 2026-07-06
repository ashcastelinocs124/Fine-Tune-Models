"""The provider-agnostic ReAct loop.

`run_agent` is the single loop used by BOTH teacher generation and student serving —
it never knows which model is behind the `driver`. That is the train/serve parity
guarantee: identical control flow, identical tool execution, identical formatting.
"""

from __future__ import annotations

from typing import Protocol

from macro_ds.domains import MACRO, DomainProfile
from macro_ds.schema import Message, Trace
from macro_ds.tools import execute_tool, openai_tool_schemas


class Driver(Protocol):
    """Anything that can take the running transcript + tool schemas and return the
    next assistant Message. Implementations live in drivers.py.

    `allow_tools=False` must make the driver answer with plain text (no tool calls) —
    used to force a final report when the step budget is exhausted."""

    def step(self, messages: list[Message], tool_schemas: list[dict], allow_tools: bool = True) -> Message: ...


def run_agent(
    question: str,
    asof_date: str,
    driver: Driver,
    max_steps: int = 12,
    profile: DomainProfile | None = None,
) -> Trace:
    p = profile or MACRO
    system = p.build_system_prompt(asof_date=asof_date, max_steps=max_steps)
    messages: list[Message] = [
        Message(role="system", content=system),
        Message(role="user", content=question),
    ]
    tool_schemas = openai_tool_schemas(p.tool_names)

    steps = 0
    tool_errors = 0
    final_report: str | None = None

    while steps < max_steps:
        # Reserve the final step to force a written report (tools disabled), so a
        # thorough teacher that keeps researching still produces a completable trace
        # instead of running out the budget with no answer.
        last = steps == max_steps - 1
        if last:
            # force a clean written report on the final step
            messages.append(Message(role="user", content=p.finalize_instruction))
        assistant = driver.step(messages, tool_schemas, allow_tools=not last)
        messages.append(assistant)
        steps += 1

        if assistant.tool_calls and not last:
            for tc in assistant.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                if result.startswith("ERROR:"):
                    tool_errors += 1
                messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id, name=tc.name)
                )
            continue

        # no tool calls (or final step with tools disabled) -> the final deliverable
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
