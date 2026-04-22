import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

"""
Script 3: Fine-tuned Gemma solves test puzzles → submission.csv
Run:    python 3_inference.py
Output: submission.csv

Inference pipeline:
1. First pass: beam search (or greedy)
2. Retry pass: 15 attempts with sampling + majority voting
3. Brute-force fallback for unsolved
"""

import re
import ast
import csv
import torch
import pandas as pd
from fractions import Fraction
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_MODEL = "google/gemma-3-1b-it"
LORA_DIR = "./gemma-countdown"
TEST_FILE = "test_public.csv"
OUTPUT_CSV = "submission.csv"

BATCH_SIZE = 8
MAX_NEW_TOK = 64              # Shorter - equations are brief

# First pass: beam search
USE_BEAM_SEARCH = True
NUM_BEAMS = 5
TEMP_FIRST = 0.1              # For non-beam fallback

# Retry pass
NUM_RETRIES = 15
TEMP_RETRY = 0.7

# Validation
STRICT_VALIDATION = True      # Check each number used at most once
# ──────────────────────────────────────────────────────────────────────────────


def make_prompt(target: int, nums: list) -> str:
    """Minimal prompt format - must match training."""
    nums_str = " ".join(map(str, nums))
    return f"Numbers: {nums_str}\nTarget: {target}"


def extract_equation(text: str) -> str:
    """Extract equation from model output."""
    # Remove thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|[^>]*\|>', '', text)

    # Normalize operators
    text = text.replace('×', '*').replace('÷', '/').replace('−', '-')

    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return ""

    def extract_expr_from_line(line: str) -> str | None:
        """Check both sides of = sign (model may write '61 = 90 - 80 + 75 - 24')."""
        parts = line.split('=')
        # Check each part for valid expression
        for part in parts:
            part = part.strip()
            if (re.fullmatch(r'[\d\s\+\-\*\/\(\)]+', part)
                    and re.search(r'\d+\s*[\+\-\*\/]\s*\d+', part)):
                return part
        return None

    # Try each line (prefer last valid one)
    for line in reversed(lines):
        expr = extract_expr_from_line(line)
        if expr:
            return expr

    # Fallback: extract any math expression
    for line in reversed(lines):
        for part in line.split('='):
            part = part.strip()
            m = re.search(r'([\d\(\)][\d\s\+\-\*\/\(\)]*[\d\)])', part)
            if m and re.search(r'\d+\s*[\+\-\*\/]\s*\d+', m.group(1)):
                return m.group(1).strip()

    return lines[-1] if lines else ""


def verify(equation: str, target: int, nums: list) -> bool:
    """Verify equation: correct result, uses only given numbers, each at most once."""
    # Extract numbers from equation
    try:
        used = list(map(int, re.findall(r'\d+', equation)))
    except ValueError:
        return False

    # Check each number is from the pool (at most once)
    pool = list(nums)
    for n in used:
        if n in pool:
            pool.remove(n)
        else:
            return False

    # Check equation syntax
    if not re.fullmatch(r'[\d\s\+\-\*\/\(\)]+', equation):
        return False

    # Evaluate with exact arithmetic
    try:
        expr = re.sub(r'(\d+)', r'Fraction(\1)', equation)
        result = eval(expr, {"__builtins__": {}, "Fraction": Fraction})
        return result == Fraction(target)
    except Exception:
        return False


def brute_force_solve(numbers: list, target: int, timeout: float = 5.0) -> str | None:
    """Exhaustive recursive solver using exact Fraction arithmetic."""
    import time
    target_f = Fraction(target)
    deadline = time.perf_counter() + timeout

    def solve(items):
        if time.perf_counter() > deadline:
            return None

        for val, expr in items:
            if val == target_f:
                return expr

        if len(items) < 2:
            return None

        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                a_val, a_expr = items[i]
                b_val, b_expr = items[j]
                rest = [items[k] for k in range(n) if k != i and k != j]

                candidates = [
                    (a_val + b_val, f"({a_expr} + {b_expr})"),
                    (a_val * b_val, f"({a_expr} * {b_expr})"),
                    (a_val - b_val, f"({a_expr} - {b_expr})"),
                    (b_val - a_val, f"({b_expr} - {a_expr})"),
                ]
                if b_val != 0:
                    candidates.append((a_val / b_val, f"({a_expr} / {b_expr})"))
                if a_val != 0:
                    candidates.append((b_val / a_val, f"({b_expr} / {a_expr})"))

                for nv, ne in candidates:
                    r = solve(rest + [(nv, ne)])
                    if r is not None:
                        return r
        return None

    items = [(Fraction(n), str(n)) for n in numbers]
    found = solve(items)
    if found is None:
        return None

    # Strip outer parentheses
    while found.startswith('(') and found.endswith(')'):
        inner = found[1:-1]
        # Check if parentheses actually wrap everything
        depth = 0
        valid = True
        for c in inner:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth < 0:
                valid = False
                break
        if valid and depth == 0:
            found = inner
        else:
            break

    return found


def generate_batch(model, tokenizer, prompts, temperature=0.1, use_beam=False, num_beams=5):
    """Generate responses for a batch of prompts."""
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=256,
    ).to("cuda")

    with torch.no_grad():
        if use_beam:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOK,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOK,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )

    new_toks = outputs[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_toks, skip_special_tokens=True)


def eval_equation(eq: str) -> int | None:
    """Evaluate equation and return integer result, or None if invalid."""
    try:
        expr = re.sub(r'(\d+)', r'Fraction(\1)', eq)
        result = eval(expr, {"__builtins__": {}, "Fraction": Fraction})
        if result.denominator == 1:
            return int(result)
    except Exception:
        pass
    return None


def majority_vote(equations: list[str]) -> str | None:
    """Select most common equation by evaluated result, not string comparison."""
    if not equations:
        return None

    # Group equations by their evaluated result
    by_result: dict[int, list[str]] = {}
    for eq in equations:
        val = eval_equation(eq)
        if val is not None:
            by_result.setdefault(val, []).append(eq)

    if not by_result:
        return Counter(equations).most_common(1)[0][0]

    # Find result with most votes
    best_result = max(by_result.keys(), key=lambda r: len(by_result[r]))
    # Return shortest equation for that result (cleaner)
    return min(by_result[best_result], key=len)


def main():
    print("=" * 60)
    print("Step 3: Inference — solving test puzzles")
    print("=" * 60)

    # ── Load test data ────────────────────────────────────────────────────
    print(f"\nLoading {TEST_FILE}...")
    df = pd.read_csv(TEST_FILE)
    print(f"  Puzzles: {len(df)}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading fine-tuned Gemma + LoRA from {LORA_DIR}...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    tokenizer.padding_side = "left"  # Left-pad for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="cuda",
    )
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    print("  Model loaded!")

    # ── Prepare rows ──────────────────────────────────────────────────────
    rows = []
    for _, r in df.iterrows():
        nums = ast.literal_eval(r['nums']) if isinstance(r['nums'], str) else r['nums']
        rows.append({"id": int(r['id']), "target": int(r['target']), "nums": nums})

    # Statistics tracking
    stats = {
        "first_pass": 0,
        "retry_pass": 0,
        "brute_force": 0,
        "fallback": 0,
        "by_num_count": Counter(),
        "by_target_range": Counter(),
    }

    # ── First pass: beam search or greedy ─────────────────────────────────
    method = "beam search" if USE_BEAM_SEARCH else "greedy"
    print(f"\nFirst pass ({method}, batch={BATCH_SIZE})...")
    results: dict[int, str] = {}
    need_retry: list[dict] = []

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="First pass"):
        batch = rows[i:i + BATCH_SIZE]
        prompts = [make_prompt(r["target"], r["nums"]) for r in batch]
        decoded = generate_batch(
            model, tokenizer, prompts,
            temperature=TEMP_FIRST,
            use_beam=USE_BEAM_SEARCH,
            num_beams=NUM_BEAMS,
        )

        for meta, raw in zip(batch, decoded):
            eq = extract_equation(raw)
            if verify(eq, meta["target"], meta["nums"]):
                results[meta["id"]] = eq
                stats["first_pass"] += 1
                stats["by_num_count"][len(meta["nums"])] += 1
                if meta["target"] <= 100:
                    stats["by_target_range"]["1-100"] += 1
                elif meta["target"] <= 500:
                    stats["by_target_range"]["101-500"] += 1
                else:
                    stats["by_target_range"]["501-999"] += 1
            else:
                need_retry.append(meta)

    print(f"  First pass solved: {stats['first_pass']}/{len(rows)}")

    # ── Retry pass with majority voting ───────────────────────────────────
    if need_retry:
        print(f"\nRetrying {len(need_retry)} puzzles ({NUM_RETRIES} attempts each, temp={TEMP_RETRY})...")
        still_failed = []

        for meta in tqdm(need_retry, desc="Retrying"):
            valid_equations = []

            # Batch all retries in single generate() call with num_return_sequences
            prompt = make_prompt(meta["target"], meta["nums"])
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOK,
                    temperature=TEMP_RETRY,
                    do_sample=True,
                    num_return_sequences=NUM_RETRIES,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_toks = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(new_toks, skip_special_tokens=True)

            for raw in decoded:
                eq = extract_equation(raw)
                if verify(eq, meta["target"], meta["nums"]):
                    valid_equations.append(eq)

            if valid_equations:
                # Majority voting
                best_eq = majority_vote(valid_equations)
                results[meta["id"]] = best_eq
                stats["retry_pass"] += 1
                stats["by_num_count"][len(meta["nums"])] += 1
                if meta["target"] <= 100:
                    stats["by_target_range"]["1-100"] += 1
                elif meta["target"] <= 500:
                    stats["by_target_range"]["101-500"] += 1
                else:
                    stats["by_target_range"]["501-999"] += 1
            else:
                still_failed.append(meta)

        print(f"  Retries solved: {stats['retry_pass']}")

        # ── Brute-force fallback ──────────────────────────────────────────
        if still_failed:
            print(f"\nBrute-forcing {len(still_failed)} remaining puzzles...")

            for meta in tqdm(still_failed, desc="Brute-force"):
                expr = brute_force_solve(meta["nums"], meta["target"])
                if expr is not None and verify(expr, meta["target"], meta["nums"]):
                    results[meta["id"]] = expr
                    stats["brute_force"] += 1
                    stats["by_num_count"][len(meta["nums"])] += 1
                    if meta["target"] <= 100:
                        stats["by_target_range"]["1-100"] += 1
                    elif meta["target"] <= 500:
                        stats["by_target_range"]["101-500"] += 1
                    else:
                        stats["by_target_range"]["501-999"] += 1
                else:
                    # Ultimate fallback - submit first number
                    results[meta["id"]] = str(meta["nums"][0])
                    stats["fallback"] += 1

            print(f"  Brute-force solved: {stats['brute_force']}/{len(still_failed)}")

    # ── Save submission ───────────────────────────────────────────────────
    print(f"\nSaving {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'equation'])
        for rid in sorted(results.keys()):
            writer.writerow([rid, results[rid]])

    # ── Statistics ────────────────────────────────────────────────────────
    total_model_solved = stats["first_pass"] + stats["retry_pass"]
    total_solved = total_model_solved + stats["brute_force"]

    print(f"\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print(f"\n  Total puzzles:      {len(rows)}")
    print(f"\n  Solved by stage:")
    print(f"    First pass:       {stats['first_pass']} ({100*stats['first_pass']/len(rows):.1f}%)")
    print(f"    Retries:          {stats['retry_pass']} ({100*stats['retry_pass']/len(rows):.1f}%)")
    print(f"    Brute-force:      {stats['brute_force']} ({100*stats['brute_force']/len(rows):.1f}%)")
    print(f"    Fallback:         {stats['fallback']}")
    print(f"\n  Model accuracy:     {total_model_solved}/{len(rows)} ({100*total_model_solved/len(rows):.1f}%)")
    print(f"  Total correct:      {total_solved}/{len(rows)} ({100*total_solved/len(rows):.1f}%)")

    # By number count
    print(f"\n  By number count:")
    for nc in sorted(stats["by_num_count"].keys()):
        # Count total puzzles with this num_count
        total_nc = sum(1 for r in rows if len(r["nums"]) == nc)
        solved_nc = stats["by_num_count"][nc]
        print(f"    {nc} nums: {solved_nc}/{total_nc} ({100*solved_nc/total_nc:.1f}%)" if total_nc > 0 else f"    {nc} nums: 0")

    # By target range
    print(f"\n  By target range:")
    for range_name in ["1-100", "101-500", "501-999"]:
        if range_name == "1-100":
            total_range = sum(1 for r in rows if r["target"] <= 100)
        elif range_name == "101-500":
            total_range = sum(1 for r in rows if 100 < r["target"] <= 500)
        else:
            total_range = sum(1 for r in rows if r["target"] > 500)
        solved_range = stats["by_target_range"].get(range_name, 0)
        if total_range > 0:
            print(f"    {range_name}: {solved_range}/{total_range} ({100*solved_range/total_range:.1f}%)")

    print(f"\n  Output: {OUTPUT_CSV}")
    print("\n→ Upload submission.csv to the platform!")


if __name__ == "__main__":
    main()
