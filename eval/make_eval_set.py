"""Write the held-out evaluation question set (disjoint from the training bank).

Usage:
    python eval/make_eval_set.py --n 48 --asof 2026-06-01 --out eval/eval_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib

from macro_ds.domains import get_profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--out", default="eval/eval_set.jsonl")
    ap.add_argument("--domain", default="macro", help="domain profile: macro | general_search")
    args = ap.parse_args()

    ev = get_profile(args.domain).build_eval_set(asof_date=args.asof, n=args.n)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for q in ev:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(ev)} held-out eval questions to {out}")


if __name__ == "__main__":
    main()
