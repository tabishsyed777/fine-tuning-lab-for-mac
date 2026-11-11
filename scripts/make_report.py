#!/usr/bin/env python3
import json
from pathlib import Path
base = json.loads(Path("results/baseline.json").read_text())
tuned = json.loads(Path("results/adapted.json").read_text())
delta = tuned["accuracy"] - base["accuracy"]
Path("results/report.md").write_text(f"# Fine-tuning result\n\n| Model | Held-out accuracy | Examples | Predictions |\n|---|---:|---:|---|\n| Base | {base['accuracy']:.1%} | {base['n']} | [JSON](baseline.json) |\n| LoRA adapter | {tuned['accuracy']:.1%} | {tuned['n']} | [JSON](adapted.json) |\n\n**Change:** {delta:+.1%}.\n\nThis comparison uses the same stratified, held-out split and deterministic decoding. Inspect predictions before making quality claims.\n")
