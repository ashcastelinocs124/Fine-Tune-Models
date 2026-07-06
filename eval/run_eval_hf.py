"""Three-way eval WITHOUT vLLM: base Qwen3 vs fine-tuned student (both via HFDriver) vs the
GPT teacher (OpenAIDriver), all through the SAME run_agent harness on held-out questions.
Computes mechanical metrics + an LLM-judge score and prints a comparison table. RUN ON THE VPS.

Usage:
    python eval/run_eval_hf.py --eval-set eval/eval_set.jsonl --limit 2 \
        --hf-model Qwen/Qwen3-8B --adapter /workspace/outputs/qwen3-8b-macro-lora \
        --teacher gpt-5 --judge gpt-5-mini --max-steps 6
"""

from __future__ import annotations

import argparse
import json
import statistics

from macro_ds.agent import run_agent
from macro_ds.domains import get_profile
from macro_ds.drivers import HFDriver, OpenAIDriver
from macro_ds.judge import judge_trace
from macro_ds.metrics import hallucinated_number_rate, mechanical_metrics


def _agg(rows: list[dict]) -> dict:
    def mean(k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "n": len(rows),
        "completion": mean("completion"),
        "valid_tool_call_rate": mean("valid_tool_call_rate"),
        "avg_steps": mean("steps"),
        "tool_error_rate": mean("tool_error_rate"),
        "hallucinated_number_rate": mean("hallucinated_number_rate"),
        "judge_overall": mean("judge_overall"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval/eval_set.jsonl")
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--systems", nargs="+", default=["base", "student", "teacher"])
    ap.add_argument("--domain", default="macro", help="domain profile: macro | general_search")
    args = ap.parse_args()
    profile = get_profile(args.domain)

    questions = [json.loads(l) for l in open(args.eval_set) if l.strip()][: args.limit]
    print(f"Evaluating {len(questions)} held-out questions across {args.systems}\n")

    # build one driver per system (HF models load once and are reused across questions)
    drivers = {}
    if "base" in args.systems:
        print("loading base Qwen3-8B...")
        drivers["base"] = HFDriver(args.hf_model, adapter=None, max_new_tokens=768)
    if "student" in args.systems:
        print("loading student (base + adapter)...")
        drivers["student"] = HFDriver(args.hf_model, adapter=args.adapter, max_new_tokens=768)
    if "teacher" in args.systems:
        drivers["teacher"] = OpenAIDriver(model=args.teacher)

    results = {s: [] for s in drivers}
    for q in questions:
        for name, drv in drivers.items():
            try:
                tr = run_agent(q["question"], q["asof_date"], driver=drv, max_steps=args.max_steps,
                               profile=profile)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {name} failed on {q['id']}: {e}")
                continue
            mech = mechanical_metrics(tr)
            try:
                jr = judge_trace(tr, model=args.judge, rubric=profile.judge_rubric)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] judge failed {name}/{q['id']}: {e}")
                jr = {"overall": None}
            results[name].append(
                {**mech, "hallucinated_number_rate": hallucinated_number_rate(tr), "judge_overall": jr.get("overall")}
            )
            print(
                f"[{name:7}] {q['id']:14} steps={mech['steps']} "
                f"valid_tool={mech['valid_tool_call_rate']:.2f} completed={mech['completion']} "
                f"judge={jr.get('overall')}"
            )

    summary = {s: _agg(rows) for s, rows in results.items()}
    metrics = ["completion", "valid_tool_call_rate", "tool_error_rate", "avg_steps",
               "hallucinated_number_rate", "judge_overall"]
    syss = list(summary)
    print("\n| metric | " + " | ".join(syss) + " |")
    print("|" + "---|" * (len(syss) + 1))
    for m in metrics:
        print(f"| {m} | " + " | ".join(str(summary[s].get(m)) for s in syss) + " |")


if __name__ == "__main__":
    main()
