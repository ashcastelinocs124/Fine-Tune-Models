#!/usr/bin/env bash
# Train the Qwen3-8B QLoRA adapter on the macro traces. RUN ON THE VPS.
#
#   pip install -e .                       # macro_ds + deps
#   pip install "llamafactory[torch,bitsandbytes,metrics]"  # or clone + pip install -e .
#   pip install vllm                       # for serving later
#
# Then:
#   bash train/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> Verifying template/mask parity before training..."
python train/check_template_parity.py

echo ">> Starting QLoRA training..."
llamafactory-cli train train/qlora_qwen3_8b.yaml

echo ">> Done. Adapter in outputs/qwen3-8b-macro-lora"
