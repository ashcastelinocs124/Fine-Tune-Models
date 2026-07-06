"""Filter raw teacher traces by rejection sampling: mechanical gates then LLM judge.

Survivors are copied to the clean dir and recorded in manifest.jsonl. Prints a
kept/total count and a histogram of rejection reasons.

Usage:
    python gen/filter_traces.py --raw data/traces/raw --clean data/traces/clean \
        --judge gpt-5 [--skip-judge]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

from macro_ds.domains import get_profile
from macro_ds.filtering import mechanical_gates
from macro_ds.judge import judge_trace
from macro_ds.schema import Trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/traces/raw")
    ap.add_argument("--clean", default="data/traces/clean")
    ap.add_argument("--judge", default=None, help="judge model id; omit/--skip-judge to skip")
    ap.add_argument("--skip-judge", action="store_true", help="mechanical gates only")
    ap.add_argument("--domain", default="macro", help="domain profile: macro | general_search")
    args = ap.parse_args()
    profile = get_profile(args.domain)

    raw_dir = pathlib.Path(args.raw)
    clean_dir = pathlib.Path(args.clean)
    clean_dir.mkdir(parents=True, exist_ok=True)
    manifest = clean_dir / "manifest.jsonl"

    raw_files = sorted(raw_dir.glob("*.json"))
    kept = 0
    reasons_hist: collections.Counter[str] = collections.Counter()

    with manifest.open("w") as mf:
        for path in raw_files:
            # Per-file isolation: one corrupt file or judge API error must not abort the batch.
            try:
                trace = Trace.from_json(path.read_text())
            except Exception as e:  # noqa: BLE001
                print(f"[warn] could not parse {path.name}: {e}")
                reasons_hist["parse_error"] += 1
                continue

            gate = mechanical_gates(trace)
            if not gate.passed:
                for r in gate.reasons:
                    reasons_hist[r] += 1
                continue

            judge_result = {"keep": True, "overall": None, "scores": {}}
            if args.judge and not args.skip_judge:
                try:
                    judge_result = judge_trace(trace, model=args.judge, rubric=profile.judge_rubric)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] judge failed for {path.name}: {e}")
                    reasons_hist["judge_error"] += 1
                    continue
                if not judge_result["keep"]:
                    reasons_hist[f"judge<4 (overall={judge_result['overall']})"] += 1
                    continue

            (clean_dir / path.name).write_text(trace.to_json())
            mf.write(
                json.dumps(
                    {
                        "id": path.stem,
                        "question": trace.question,
                        "steps": trace.steps,
                        "overall": judge_result.get("overall"),
                        "scores": judge_result.get("scores"),
                    }
                )
                + "\n"
            )
            kept += 1

    print(f"Kept {kept}/{len(raw_files)} traces -> {clean_dir}")
    if reasons_hist:
        print("Rejection reasons:")
        for reason, n in reasons_hist.most_common():
            print(f"  {n:4d}  {reason}")


if __name__ == "__main__":
    main()
