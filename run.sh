#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
MLX_LORA="$ROOT/.venv/bin/mlx_lm.lora"
MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
if [[ ! -x "$PYTHON" || ! -x "$MLX_LORA" ]]; then
  echo "Missing MLX environment. Run: python3 -m venv .venv && .venv/bin/pip install 'mlx-lm[train]'"
  exit 1
fi
cd "$ROOT"
case "${1:-}" in
  baseline) "$PYTHON" scripts/evaluate.py --model "$MODEL" --name baseline ;;
  train) "$MLX_LORA" --config config/lora.yaml ;;
  adapted) "$PYTHON" scripts/evaluate.py --model "$MODEL" --adapter-path artifacts/adapters --name adapted && "$PYTHON" scripts/make_report.py ;;
  *) echo "Usage: ./run.sh {baseline|train|adapted}"; exit 1 ;;
esac
