"""Run the student agent through the SHARED harness against a vLLM endpoint. RUN ON THE VPS.

This is the parity payoff: the exact same `run_agent` loop and tools that generated the
teacher traces now drive the fine-tuned student. Only the driver changes.

Usage (with serve/serve_vllm.sh running):
    python serve/run_agent.py --question "Path of the US 10y yield next quarter?" --asof 2026-06-01
    python serve/run_agent.py --question "..." --asof 2026-06-01 --model Qwen/Qwen3-8B  # base baseline
"""

from __future__ import annotations

import argparse

from macro_ds.agent import run_agent
from macro_ds.drivers import VLLMDriver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--model", default="macro", help="'macro' (student+adapter) or 'Qwen/Qwen3-8B' (base)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-steps", type=int, default=12)
    args = ap.parse_args()

    driver = VLLMDriver(model=args.model, base_url=args.base_url)
    trace = run_agent(args.question, asof_date=args.asof, driver=driver, max_steps=args.max_steps)

    print(f"\n=== Trajectory ({trace.steps} steps, {trace.tool_errors} tool errors) ===")
    for m in trace.messages:
        if m.role == "assistant" and m.tool_calls:
            print(f"\n[assistant] {m.content}")
            for tc in m.tool_calls:
                print(f"  -> {tc.name}({tc.arguments})")
        elif m.role == "tool":
            preview = (m.content or "")[:200].replace("\n", " ")
            print(f"  <- [{m.name}] {preview}")
    print("\n=== FINAL REPORT ===")
    print(trace.final_report or "(no final report — hit step budget)")


if __name__ == "__main__":
    main()
