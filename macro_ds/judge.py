"""LLM-as-judge quality scoring for traces (the second rejection-sampling gate).

The judge sees the question, the final report, and the sources the agent actually
retrieved, then scores groundedness / coverage / citation-faithfulness / structure on
a 1-5 rubric. We keep a trace only if `overall >= KEEP_THRESHOLD`.
"""

from __future__ import annotations

import json
from typing import Any

from macro_ds import config
from macro_ds.schema import Trace

KEEP_THRESHOLD = 4.0

RUBRIC = """You are a strict reviewer of global-macro research traces.
Score the analyst's FINAL REPORT against the SOURCES THAT WERE ACTUALLY RETRIEVED.

Rate each 1-5 (5 best):
- groundedness: every claim is supported by retrieved sources; no invented numbers.
- coverage: addresses the question's key drivers and counter-arguments.
- citation_faithfulness: cited URLs/series actually support the cited claims.
- structure: clear view, drivers, evidence, risks, and what would change the view.

Respond ONLY with a JSON object:
{"groundedness": int, "coverage": int, "citation_faithfulness": int,
 "structure": int, "overall": number, "rationale": string}
"""


def _retrieved_sources(trace: Trace) -> str:
    chunks = []
    for m in trace.messages:
        if m.role == "tool":
            chunks.append(f"[{m.name}] {m.content}")
    return "\n\n".join(chunks) if chunks else "(no sources retrieved)"


def build_judge_prompt(trace: Trace) -> str:
    return (
        f"QUESTION:\n{trace.question}\n\n"
        f"SOURCES RETRIEVED:\n{_retrieved_sources(trace)}\n\n"
        f"FINAL REPORT:\n{trace.final_report or '(none)'}\n"
    )


def _call_judge(messages: list[dict[str, Any]], model: str) -> str:
    """Call the judge model and return raw content. Isolated so tests can mock it."""
    from openai import OpenAI

    client = OpenAI(api_key=config.get_key("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def judge_trace(trace: Trace, model: str, rubric: str | None = None) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": rubric if rubric is not None else RUBRIC},
        {"role": "user", "content": build_judge_prompt(trace)},
    ]
    raw = _call_judge(messages, model)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"keep": False, "overall": 0, "scores": {}, "rationale": "unparseable judge output"}

    score_keys = ["groundedness", "coverage", "citation_faithfulness", "structure"]
    scores = {k: data.get(k) for k in score_keys if k in data}
    overall = data.get("overall")
    if overall is None:
        numeric = [v for v in scores.values() if isinstance(v, (int, float))]
        overall = sum(numeric) / len(numeric) if numeric else 0
    overall = float(overall or 0)
    return {
        "keep": overall >= KEEP_THRESHOLD,
        "overall": overall,
        "scores": scores,
        "rationale": data.get("rationale", ""),
    }
