#!/usr/bin/env bash
# Serve the fine-tuned student (Qwen3-8B + LoRA adapter) via vLLM. RUN ON THE VPS.
#
# Exposes an OpenAI-compatible endpoint at http://localhost:8000/v1 with two model names:
#   - "macro"      -> base + our LoRA adapter (the student)
#   - "Qwen/Qwen3-8B" -> base model (the eval baseline; vLLM serves the base under its name)
#
# NOTE: confirm the correct Qwen3 tool-call parser name for your vLLM version
# (try `hermes`, then `qwen3_coder`/`qwen` if tool calls don't parse). If LoRA + the tool
# parser misbehave together, merge the adapter first (see serve/README note) and serve the
# merged model without --enable-lora.
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${ADAPTER:-outputs/qwen3-8b-macro-lora}"

vllm serve Qwen/Qwen3-8B \
  --enable-lora \
  --lora-modules "macro=${ADAPTER}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 16384 \
  --port 8000
