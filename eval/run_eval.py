"""Three-way evaluation: base Qwen3-8B vs fine-tuned student vs GPT teacher, all run
through the SAME shared harness on the held-out eval set. RUN ON THE VPS.

For each (system, question) it runs the agent loop, then computes mechanical metrics
(macro_ds.metrics) and an absolute LLM-judge quality score (macro_ds.judge). Writes raw
traces + a results JSON and prints a markdown comparison table.

Prereqs: serve/serve_vllm.sh running (for base + student); OPENAI_API_KEY set (teacher + judge).

Usage:
    python eval/run_eval.py --eval-set eval/eval_set.jsonl --teacher gpt-5 --judge gpt-5 \
        --limit 30 --out eval/results
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from datetime import datetime, timezone

from macro_ds.agent import run_agent
from macro_ds.drivers import OpenAIDriver, VLLMDriver
from macro_ds.judge import judge_trace
from macro_ds.metrics import hallucinated_number_rate, mechanical_metrics


def _driver_factory(system: str, teacher_model: str, base_url: str):
    if system == "base":
        return lambda: VLLMDriver(model="Qwen/Qwen3-8B", base_url=base_url)
    if system == "student":
        return lambda: VLLMDriver(model="macro", base_url=base_url)
    if system == "teacher":
        return lambda: OpenAIDriver(model=teacher_model)
    raise ValueError(f"unknown system {system!r}")


def _aggregate(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "n": len(rows),
        "completion_rate": round(sum(r["completion"] for r in rows) / len(rows), 3) if rows else 0,
        "valid_tool_call_rate": mean("valid_tool_call_rate"),
        "tool_error_rate": mean("tool_error_rate"),
        "avg_steps": mean("steps"),
        "hallucinated_number_rate": mean("hallucinated_number_rate"),
        "judge_overall": mean("judge_overall"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval/eval_set.jsonl")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--systems", nargs="+", default=["base", "student", "teacher"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()

    questions = [json.loads(l) for l in pathlib.Path(args.eval_set).read_text().splitlines() if l.strip()]
    if args.limit:
        questions = questions[: args.limit]

    out_dir = pathlib.Path(args.out)
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for system in args.systems:
        factory = _driver_factory(system, args.teacher, args.base_url)
        rows = []
        for q in questions:
            try:
                trace = run_agent(q["question"], q["asof_date"], driver=factory(), max_steps=12)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {system} failed on {q['id']}: {e}")
                continue
            (out_dir / "traces" / f"{system}-{q['id']}.json").write_text(trace.to_json())
            mech = mechanical_metrics(trace)
            try:
                jr = judge_trace(trace, model=args.judge)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] judge failed for {system}/{q['id']}: {e}")
                jr = {"overall": None}
            rows.append(
                {
                    "id": q["id"],
                    **mech,
                    "hallucinated_number_rate": hallucinated_number_rate(trace),
                    "judge_overall": jr.get("overall"),
                }
            )
        summary[system] = _aggregate(rows)
        print(f"[{system}] {summary[system]}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (out_dir / f"{stamp}.json").write_text(json.dumps(summary, indent=2))
    _print_table(summary)


def _print_table(summary: dict[str, dict]) -> None:
    metrics = [
        "completion_rate",
        "valid_tool_call_rate",
        "tool_error_rate",
        "avg_steps",
        "hallucinated_number_rate",
        "judge_overall",
    ]
    systems = list(summary)
    print("\n| metric | " + " | ".join(systems) + " |")
    print("|" + "---|" * (len(systems) + 1))
    for m in metrics:
        print(f"| {m} | " + " | ".join(str(summary[s].get(m)) for s in systems) + " |")


if __name__ == "__main__":
    main()
