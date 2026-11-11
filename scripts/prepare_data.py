#!/usr/bin/env python3
"""Validate raw routing data and make deterministic MLX-compatible splits."""
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

INSTRUCTION = "Classify the IT support ticket. Reply with exactly one label: ACCESS, BILLING, HARDWARE, NETWORK, or SOFTWARE."

def as_chat(row):
    return {"messages": [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": "Ticket: " + row["ticket"]},
        {"role": "assistant", "content": row["label"]},
    ]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/support_examples.jsonl")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    allowed = {x.strip() for x in Path("config/labels.txt").read_text().splitlines() if x.strip()}
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    for n, row in enumerate(rows, 1):
        if set(row) != {"ticket", "label"} or not isinstance(row["ticket"], str) or row["label"] not in allowed:
            raise ValueError(f"Invalid row {n}: expected ticket and an allowed label")
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    if min(len(label_rows) for label_rows in by_label.values()) < 3:
        raise ValueError("Use at least 3 examples per label; aim for 30+ per label for a credible score.")
    rng = random.Random(args.seed)
    splits = {"train": [], "valid": [], "test": []}
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        holdout_n = max(1, round(len(label_rows) * .2))
        splits["test"].extend(label_rows[:holdout_n])
        splits["valid"].extend(label_rows[holdout_n:2 * holdout_n])
        splits["train"].extend(label_rows[2 * holdout_n:])
    for examples in splits.values():
        rng.shuffle(examples)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(as_chat(x)) + "\n" for x in examples))
    manifest = {"seed": args.seed, "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), "counts": {k: len(v) for k, v in splits.items()}, "label_counts": dict(Counter(x["label"] for x in rows)), "split_label_counts": {k: dict(Counter(x["label"] for x in v)) for k, v in splits.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
if __name__ == "__main__": main()
