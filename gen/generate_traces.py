"""Generate teacher traces by running GPT as an agent in the shared harness.

Each question is run through `macro_ds.agent.run_agent` with an OpenAIDriver, and the
full trajectory is saved as JSON. The run is:
- resumable: questions whose output file already exists are skipped.
- robust: a failure on one question is logged and skipped, not fatal to the batch.
- concurrent: a thread pool runs several questions at once.

Usage:
    python gen/generate_traces.py --bank data/questions/bank.jsonl \
        --out data/traces/raw --limit 20 --model gpt-5 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from macro_ds.agent import run_agent
from macro_ds.drivers import OpenAIDriver

MAX_STEPS = 12


def _load_bank(path: str) -> list[dict[str, Any]]:
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _generate_one(q: dict[str, Any], out_dir: pathlib.Path, driver_factory: Callable[[], Any]) -> bool:
    """Generate and save one trace. Returns True if a new trace was written."""
    dest = out_dir / f"{q['id']}.json"
    if dest.exists():
        return False  # resumable: already done
    try:
        driver = driver_factory()
        trace = run_agent(q["question"], q["asof_date"], driver=driver, max_steps=MAX_STEPS)
        trace.meta.setdefault("question_id", q["id"])
        trace.meta.setdefault("theme", q.get("theme"))
        # Atomic write: a kill mid-write must not leave a truncated file that resume treats
        # as "done" and that then crashes the filter stage.
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(trace.to_json())
        os.replace(tmp, dest)
        return True
    except Exception as e:  # noqa: BLE001 — one bad question must not kill the batch
        print(f"[warn] generation failed for {q['id']}: {e}")
        return False


def run_generation(
    bank: str,
    out: str,
    limit: int | None = None,
    model: str | None = None,
    concurrency: int = 4,
    driver_factory: Callable[[], Any] | None = None,
) -> int:
    """Generate traces for the question bank. Returns the count of newly written traces."""
    questions = _load_bank(bank)
    if limit is not None:
        questions = questions[:limit]
    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if driver_factory is None:
        if not model:
            raise ValueError("model is required when no driver_factory is provided")
        driver_factory = lambda: OpenAIDriver(model=model)  # noqa: E731

    written = 0
    if concurrency <= 1:
        for q in questions:
            written += int(_generate_one(q, out_dir, driver_factory))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_generate_one, q, out_dir, driver_factory) for q in questions]
            for fut in as_completed(futures):
                written += int(fut.result())

    print(f"Generated {written} new traces -> {out_dir} ({len(questions)} questions considered)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/questions/bank.jsonl")
    ap.add_argument("--out", default="data/traces/raw")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL"))
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    run_generation(args.bank, args.out, limit=args.limit, model=args.model, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
