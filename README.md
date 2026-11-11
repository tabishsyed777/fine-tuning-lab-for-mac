# End-to-End Fine-Tuning Lab (Apple Silicon)

A reproducible portfolio lab for the complete fine-tuning loop on a Mac: curate domain data, establish a baseline, train a LoRA adapter with MLX, and compare held-out scores.

The included IT-support triage dataset demonstrates the required JSONL format only; it is **not** large enough for a credible benchmark. Use approved, non-sensitive data and aim for 30+ examples per label.

## Pipeline

```text
raw examples -> validate/stratify -> baseline evaluation -> MLX LoRA -> adapted evaluation -> results/report.md
```

The test split is held out. The metric is deterministic label accuracy for one of `ACCESS`, `BILLING`, `HARDWARE`, `NETWORK`, or `SOFTWARE`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U "mlx-lm[train]"

python scripts/prepare_data.py
./run.sh baseline
./run.sh train
./run.sh adapted
```

`run.sh` stores adapters in `artifacts/adapters/` and writes raw predictions plus a before/after report under `results/`.

## Customize

1. Add `{"ticket":"...", "label":"ACCESS"}` rows to `data/raw/support_examples.jsonl`.
2. Keep allowed labels in `config/labels.txt`.
3. Run `python scripts/prepare_data.py`; the manifest records each split's label distribution.
4. Only tune `config/lora.yaml` after recording a baseline.

MLX LM’s [LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) documents the expected data and adapter workflow.
