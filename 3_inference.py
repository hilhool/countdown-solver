"""Solve the test puzzles and write submission.csv. No brute-force search.

    python 3_inference.py [--limit N]

The model proposes solution traces; exact arithmetic verifies each one. A puzzle
is solved as soon as any sampled trace is provably correct (generate-and-verify).
The search budget escalates only for the puzzles still unsolved:

    1. greedy / beam pass over every puzzle
    2. escalating rounds of temperature sampling, accept the first verified trace

Because every accepted answer is checked exactly, the reported solve rate is the
true accuracy on solvable puzzles. Unsolved puzzles fall back to a single number
(scored as wrong) rather than an exhaustive solver.
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import ast
import csv
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from countdown import make_prompt_numbers as make_prompt, parse_trace, trace_to_equation, verify

BASE_MODEL = "google/gemma-3-1b-it"
LORA_DIR = "./gemma-countdown"
TEST_FILE = "test_public.csv"
OUTPUT_CSV = "submission.csv"
# Verified solves are flushed here after every batch so an interrupted run can
# resume instead of re-solving from scratch. Holds only verified equations, never
# the placeholders emitted for unsolved puzzles.
CHECKPOINT_CSV = "submission_checkpoint.csv"

MAX_NEW_TOK = 96
BEAM_WIDTH = 16
BEAM_BATCH = 8
SAMPLE_BATCH = 4

# Escalating sampling schedule for unsolved puzzles: (num_samples, temperature).
SAMPLE_SCHEDULE = [(8, 0.7), (16, 0.8), (32, 1.0), (64, 1.1)]


def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="cuda")
    model = PeftModel.from_pretrained(base, LORA_DIR).eval()
    return model, tokenizer


def generate(model, tokenizer, prompts, *, beam=False, num_samples=1, temperature=1.0):
    messages = [[{"role": "user", "content": p}] for p in prompts]
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        if beam:
            # Return every beam hypothesis, not just the top one: the verifier
            # accepts any that checks out, so all BEAM_WIDTH beams are candidates
            # for roughly the cost of the search we already pay.
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOK, num_beams=BEAM_WIDTH,
                num_return_sequences=BEAM_WIDTH, early_stopping=True,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        else:
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOK, do_sample=True,
                temperature=temperature, top_p=0.95, num_return_sequences=num_samples,
                pad_token_id=tokenizer.eos_token_id,
            )
    decoded = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Regroup into one list of candidates per prompt.
    per_prompt = BEAM_WIDTH if beam else num_samples
    return [decoded[i * per_prompt:(i + 1) * per_prompt] for i in range(len(prompts))]


def write_results(path, results):
    """Persist {id: equation} to a CSV (id, equation), sorted by id."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "equation"])
        for rid in sorted(results):
            writer.writerow([rid, results[rid]])


def load_checkpoint(path):
    """Load previously verified solves so a restart skips them. Returns {id: eq}."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {int(r.id): str(r.equation) for r in df.itertuples()}


def first_valid(candidates, target, numbers):
    """Return the shortest verified equation among candidate traces, or None."""
    valid = []
    for raw in candidates:
        eq = trace_to_equation(parse_trace(raw), numbers)
        if eq and verify(eq, target, numbers):
            valid.append(eq)
    return min(valid, key=len) if valid else None


def solve(model, tokenizer, rows, checkpoint=None):
    results = dict(checkpoint or {})
    stats = {"resumed": len(results), "beam": 0, "sampling": 0, "unsolved": 0}
    # Skip puzzles already verified in a previous run.
    pending = [i for i in range(len(rows)) if rows[i]["id"] not in results]
    if stats["resumed"]:
        print(f"  resumed {stats['resumed']} verified solves, {len(pending)} still pending")

    # Stage 1: beam pass over everything.
    for b in tqdm(range(0, len(pending), BEAM_BATCH), desc="beam"):
        idxs = pending[b:b + BEAM_BATCH]
        groups = generate(model, tokenizer, [make_prompt(rows[i]["nums"], rows[i]["target"]) for i in idxs], beam=True)
        new = False
        for i, cands in zip(idxs, groups):
            eq = first_valid(cands, rows[i]["target"], rows[i]["nums"])
            if eq:
                results[rows[i]["id"]] = eq
                stats["beam"] += 1
                new = True
        if new:
            write_results(CHECKPOINT_CSV, results)
    pending = [i for i in pending if rows[i]["id"] not in results]
    print(f"  beam solved {stats['beam']}/{len(rows)} ({100*stats['beam']/len(rows):.1f}%), {len(pending)} pending")

    # Stage 2: escalating verified sampling on the remainder.
    for num_samples, temp in SAMPLE_SCHEDULE:
        if not pending:
            break
        still = []
        for b in tqdm(range(0, len(pending), SAMPLE_BATCH), desc=f"sample n={num_samples} t={temp}"):
            idxs = pending[b:b + SAMPLE_BATCH]
            groups = generate(
                model, tokenizer, [make_prompt(rows[i]["nums"], rows[i]["target"]) for i in idxs],
                num_samples=num_samples, temperature=temp,
            )
            new = False
            for i, cands in zip(idxs, groups):
                eq = first_valid(cands, rows[i]["target"], rows[i]["nums"])
                if eq:
                    results[rows[i]["id"]] = eq
                    stats["sampling"] += 1
                    new = True
                else:
                    still.append(i)
            if new:
                write_results(CHECKPOINT_CSV, results)
        pending = still
        solved = len(results)
        print(f"  after n={num_samples}: {solved}/{len(rows)} ({100*solved/len(rows):.1f}%), {len(pending)} pending")

    return results, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="solve only the first N puzzles")
    args = parser.parse_args()

    df = pd.read_csv(TEST_FILE)
    if args.limit:
        df = df.head(args.limit)
    rows = [
        {"id": int(r.id), "target": int(r.target),
         "nums": ast.literal_eval(r.nums) if isinstance(r.nums, str) else r.nums}
        for r in df.itertuples()
    ]
    print(f"Loaded {len(rows)} puzzles from {TEST_FILE}")

    checkpoint = load_checkpoint(CHECKPOINT_CSV)
    test_ids = {r["id"] for r in rows}
    checkpoint = {rid: eq for rid, eq in checkpoint.items() if rid in test_ids}

    model, tokenizer = load_model()
    t0 = time.time()
    results, stats = solve(model, tokenizer, rows, checkpoint)

    # results still holds only verified equations here; add placeholders for the
    # truly unsolved so submission.csv has a row per puzzle (checkpoint stays clean).
    for i in range(len(rows)):
        if rows[i]["id"] not in results:
            results[rows[i]["id"]] = str(rows[i]["nums"][0])
            stats["unsolved"] += 1
    write_results(OUTPUT_CSV, results)

    solved = len(rows) - stats["unsolved"]
    print(f"\nSolved (verified): {solved}/{len(rows)} = {100*solved/len(rows):.1f}%")
    print(f"  resumed: {stats['resumed']}, beam: {stats['beam']}, "
          f"sampling: {stats['sampling']}, unsolved: {stats['unsolved']}")
    print(f"  elapsed {time.time()-t0:.0f}s -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
