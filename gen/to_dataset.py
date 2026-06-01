"""Convert clean traces into a LLaMA-Factory train/val dataset.

Writes data/dataset/{train,val}.json (arrays of OpenAI-style message records) plus a
candidate dataset_info.json snippet for LLaMA-Factory. The exact LLaMA-Factory column
mapping is confirmed by the Stage-0 template-parity check (train/check_template_parity.py)
before any real training.

Usage:
    python gen/to_dataset.py --clean data/traces/clean --out data/dataset --val-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import pathlib

from macro_ds.dataset import build_split
from macro_ds.schema import Trace

DATASET_INFO = {
    "macro_traces": {
        "file_name": "train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "system": "system", "tools": "tools"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "assistant",
            "system_tag": "system",
        },
    }
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/traces/clean")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clean_dir = pathlib.Path(args.clean)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = [Trace.from_json(p.read_text()) for p in sorted(clean_dir.glob("*.json"))]
    train, val = build_split(traces, val_frac=args.val_frac, seed=args.seed)

    (out_dir / "train.json").write_text(json.dumps(train, indent=2))
    (out_dir / "val.json").write_text(json.dumps(val, indent=2))
    (out_dir / "dataset_info.json").write_text(json.dumps(DATASET_INFO, indent=2))

    print(f"Wrote {len(train)} train + {len(val)} val records to {out_dir}")
    print("NOTE: confirm the LLaMA-Factory column mapping with train/check_template_parity.py (Stage 0).")


if __name__ == "__main__":
    main()
