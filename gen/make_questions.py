"""Write the macro question bank to a JSONL file.

Usage:
    python gen/make_questions.py --n 150 --asof 2026-06-01 --out data/questions/bank.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib

from macro_ds.questions import build_question_bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--asof", required=True, help="as-of date, e.g. 2026-06-01")
    ap.add_argument("--out", default="data/questions/bank.jsonl")
    args = ap.parse_args()

    bank = build_question_bank(asof_date=args.asof, n=args.n)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for q in bank:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(bank)} questions to {out}")


if __name__ == "__main__":
    main()
