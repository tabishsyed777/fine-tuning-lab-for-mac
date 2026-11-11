#!/usr/bin/env python3
"""Evaluate a base model or LoRA adapter on the untouched JSONL test split."""
import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
from mlx_lm import generate, load

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--adapter-path")
    p.add_argument("--data", default="data/processed/test.jsonl"); p.add_argument("--name", required=True)
    args = p.parse_args()
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    records = []
    labels = {"ACCESS", "BILLING", "HARDWARE", "NETWORK", "SOFTWARE"}
    for line in Path(args.data).read_text().splitlines():
        messages = json.loads(line)["messages"]
        ticket, expected = messages[1]["content"].removeprefix("Ticket: "), messages[2]["content"]
        prompt = tokenizer.apply_chat_template(messages[:2], add_generation_prompt=True)
        answer = generate(model, tokenizer, prompt=prompt, max_tokens=8, verbose=False)
        first_word = re.match(r"\s*([A-Z]+)\b", answer.upper())
        predicted = first_word.group(1) if first_word and first_word.group(1) in labels else "INVALID"
        records.append({"ticket": ticket, "expected": expected, "predicted": predicted, "correct": predicted == expected})
    score = sum(r["correct"] for r in records) / len(records)
    payload = {"run": args.name, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "accuracy": score, "n": len(records), "predictions": records}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{args.name}: {score:.1%} ({sum(r['correct'] for r in records)}/{len(records)})")
if __name__ == "__main__": main()
