import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

"""
Script 1: Generate verified training data for Countdown puzzles
Run:    python 1_generate.py
Output: train_verified.jsonl

Generates 300k SYNTHETIC puzzles:
- Operator distribution: 2 ops (40%), 3 ops (40%), 4 ops (20%)
- Number count: 3-6, uniform distribution
- Numbers: realistic Countdown pool [1-10, 25, 50, 75, 100] + random 1-100
- Targets: 1-999, maximally uniform coverage
- Up to 3 different valid equations per puzzle
- No duplicates (same puzzle + same equation)
"""

import json
import re

import random
import time
from fractions import Fraction
from collections import Counter, defaultdict
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OUTPUT_FILE = "train_verified.jsonl"
SEED = 42

# Distribution by operator count
# 2 ops: 120k (40%), 3 ops: 120k (40%), 4 ops: 60k (20%)
OPERATOR_DIST = {2: 120000, 3: 120000, 4: 60000}  # = 300k total

# Number count distribution (uniform across 3-6)
NUM_COUNTS = [3, 4, 5, 6]

# Realistic Countdown number pool
COUNTDOWN_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100]

# Max equations to collect per puzzle
MAX_EQUATIONS_PER_PUZZLE = 3

# Timeout for brute-force per puzzle
TIMEOUT_PER_PUZZLE = 0.1
# ──────────────────────────────────────────────────────────────────────────────


def generate_numbers(num_count: int) -> list[int]:
    """Generate numbers using realistic Countdown distribution, no duplicates.

    Mix of Countdown pool [1-10, 25, 50, 75, 100] and random 1-100.
    """
    # Determine how many from each pool
    from_countdown = sum(1 for _ in range(num_count) if random.random() < 0.7)
    from_random = num_count - from_countdown

    # Sample without replacement from each pool
    countdown_nums = random.sample(COUNTDOWN_POOL, min(from_countdown, len(COUNTDOWN_POOL)))
    random_pool = list(range(1, 101))
    random_nums = random.sample(random_pool, min(from_random, len(random_pool)))

    nums = countdown_nums + random_nums
    # Pad with random if needed (when pools exhausted)
    while len(nums) < num_count:
        n = random.randint(1, 100)
        if n not in nums:
            nums.append(n)

    random.shuffle(nums)
    return nums[:num_count]


def count_operators(expr: str) -> int:
    """Count arithmetic operators in an expression."""
    return sum(1 for c in expr if c in '+-*/')


def strip_outer_parens(expr: str) -> str:
    """Remove outermost parentheses if they wrap the entire expression."""
    while expr.startswith('(') and expr.endswith(')'):
        depth = 0
        can_strip = True
        for i, c in enumerate(expr):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and i < len(expr) - 1:
                can_strip = False
                break
        if can_strip:
            expr = expr[1:-1]
        else:
            break
    return expr


def brute_force_find_equations(numbers: list, target_ops: int,
                                max_solutions: int = 10,
                                timeout: float = 0.1) -> list[tuple[int, str]]:
    """Find equations with exactly target_ops operators.

    Returns list of (target, equation) tuples.
    Uses exact Fraction arithmetic.
    """
    solutions = []
    seen_equations = set()
    deadline = time.perf_counter() + timeout

    def solve(items, depth=0):
        if time.perf_counter() > deadline or len(solutions) >= max_solutions:
            return

        # Check all current values
        for val, expr in items:
            if val.denominator == 1 and 1 <= val <= 999:
                clean = strip_outer_parens(expr)
                ops = count_operators(clean)
                if ops == target_ops and clean not in seen_equations:
                    seen_equations.add(clean)
                    solutions.append((int(val), clean))
                    if len(solutions) >= max_solutions:
                        return

        if len(items) < 2:
            return

        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                if time.perf_counter() > deadline or len(solutions) >= max_solutions:
                    return

                a_val, a_expr = items[i]
                b_val, b_expr = items[j]
                rest = [items[k] for k in range(n) if k != i and k != j]

                candidates = [
                    (a_val + b_val, f"({a_expr} + {b_expr})"),
                    (a_val - b_val, f"({a_expr} - {b_expr})"),
                    (b_val - a_val, f"({b_expr} - {a_expr})"),
                    (a_val * b_val, f"({a_expr} * {b_expr})"),
                ]
                if b_val != 0:
                    candidates.append((a_val / b_val, f"({a_expr} / {b_expr})"))
                if a_val != 0:
                    candidates.append((b_val / a_val, f"({b_expr} / {a_expr})"))

                random.shuffle(candidates)  # Randomize for diversity
                for nv, ne in candidates:
                    solve(rest + [(nv, ne)], depth + 1)

    items = [(Fraction(x), str(x)) for x in numbers]
    solve(items)
    return solutions


def verify(equation: str, target: int, nums: list) -> bool:
    """Verify equation solves to target using only given numbers."""
    try:
        used = list(map(int, re.findall(r'\d+', equation)))
    except ValueError:
        return False

    pool = list(nums)
    for n in used:
        if n in pool:
            pool.remove(n)
        else:
            return False

    if not re.fullmatch(r'[\d\s\+\-\*\/\(\)]+', equation):
        return False

    try:
        expr = re.sub(r'(\d+)', r'Fraction(\1)', equation)
        result = eval(expr, {"__builtins__": {}, "Fraction": Fraction})
        return result == Fraction(target)
    except Exception:
        return False


def generate_puzzle_with_ops(target_ops: int, timeout: float = 0.1) -> list[dict] | None:
    """Generate a puzzle with specific operator count.

    Returns up to MAX_EQUATIONS_PER_PUZZLE different solutions.
    """
    num_count = random.choice(NUM_COUNTS)
    nums = generate_numbers(num_count)

    solutions = brute_force_find_equations(
        nums, target_ops,
        max_solutions=MAX_EQUATIONS_PER_PUZZLE * 2,  # Find extra to have choices
        timeout=timeout
    )

    if not solutions:
        return None

    # Select up to MAX_EQUATIONS_PER_PUZZLE unique solutions
    results = []
    seen = set()

    for target, equation in solutions:
        if len(results) >= MAX_EQUATIONS_PER_PUZZLE:
            break

        # Create unique key for deduplication
        key = (tuple(sorted(nums)), target, equation)
        if key in seen:
            continue

        if verify(equation, target, nums):
            seen.add(key)
            results.append({
                "numbers": nums,
                "target": target,
                "equation": equation,
            })

    return results if results else None


def main():
    print("=" * 60)
    print("Step 1: Generate SYNTHETIC training data (300k examples)")
    print("=" * 60)

    random.seed(SEED)

    print(f"\nConfiguration:")
    print(f"  2 operators: {OPERATOR_DIST[2]:,} (40%)")
    print(f"  3 operators: {OPERATOR_DIST[3]:,} (40%)")
    print(f"  4 operators: {OPERATOR_DIST[4]:,} (20%)")
    print(f"  Total: {sum(OPERATOR_DIST.values()):,} examples")
    print(f"  Number counts: {NUM_COUNTS} (uniform)")
    print(f"  Max equations per puzzle: {MAX_EQUATIONS_PER_PUZZLE}")

    t0 = time.time()
    all_records = []
    global_seen = set()  # For global deduplication

    for target_ops, target_count in OPERATOR_DIST.items():
        print(f"\nGenerating {target_ops}-operator examples ({target_count:,} target)...")

        op_records = []
        attempts = 0
        pbar = tqdm(total=target_count, desc=f"{target_ops} ops")

        while len(op_records) < target_count:
            attempts += 1
            results = generate_puzzle_with_ops(target_ops, timeout=TIMEOUT_PER_PUZZLE)

            if results is None:
                continue

            for rec in results:
                if len(op_records) >= target_count:
                    break

                # Global deduplication
                key = (tuple(sorted(rec["numbers"])), rec["target"], rec["equation"])
                if key in global_seen:
                    continue

                global_seen.add(key)
                op_records.append(rec)
                pbar.update(1)

        pbar.close()

        print(f"  Generated {len(op_records):,} examples in {attempts:,} attempts")
        all_records.extend(op_records)

    elapsed = time.time() - t0
    print(f"\nTotal: {len(all_records):,} examples in {elapsed:.1f}s")

    # ── Statistics ────────────────────────────────────────────────────────────
    print(f"\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)

    # Operator distribution
    op_counts = Counter(count_operators(r["equation"]) for r in all_records)
    print(f"\n  Operator distribution:")
    for ops in sorted(op_counts.keys()):
        pct = 100 * op_counts[ops] / len(all_records)
        print(f"    {ops} ops: {op_counts[ops]:,} ({pct:.1f}%)")

    # Number count distribution
    num_counts = Counter(len(r["numbers"]) for r in all_records)
    print(f"\n  Number count distribution:")
    for nc in sorted(num_counts.keys()):
        pct = 100 * num_counts[nc] / len(all_records)
        print(f"    {nc} nums: {num_counts[nc]:,} ({pct:.1f}%)")

    # Target coverage
    targets = [r["target"] for r in all_records]
    target_counter = Counter(targets)
    unique_targets = len(target_counter)
    print(f"\n  Target coverage:")
    print(f"    Range: {min(targets)} - {max(targets)}")
    print(f"    Unique targets: {unique_targets}/999 ({100*unique_targets/999:.1f}%)")
    print(f"    Mean: {sum(targets)/len(targets):.1f}")
    print(f"    Targets > 100: {sum(1 for t in targets if t > 100):,} ({100*sum(1 for t in targets if t > 100)/len(all_records):.1f}%)")

    # Average solutions per unique puzzle
    puzzle_solutions = defaultdict(int)
    for r in all_records:
        key = (tuple(sorted(r["numbers"])), r["target"])
        puzzle_solutions[key] += 1
    avg_solutions = sum(puzzle_solutions.values()) / len(puzzle_solutions) if puzzle_solutions else 0
    print(f"\n  Unique puzzles: {len(puzzle_solutions):,}")
    print(f"  Avg equations per puzzle: {avg_solutions:.2f}")

    # ── Examples ──────────────────────────────────────────────────────────────
    print(f"\n  10 example records:")
    for i, rec in enumerate(all_records[:10]):
        ops = count_operators(rec["equation"])
        print(f"    [{i+1}] nums={rec['numbers']} target={rec['target']} → {rec['equation']} ({ops} ops)")

    # ── Token length analysis (for MAX_SEQ_LEN verification) ───────────────────
    print(f"\n  Token length analysis (sample):")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

        def estimate_tokens(rec):
            prompt = f"Numbers: {' '.join(map(str, rec['numbers']))}\nTarget: {rec['target']}"
            # Approximate full chat template length
            return len(tokenizer.encode(prompt)) + len(tokenizer.encode(rec['equation'])) + 20

        sample = all_records[:10000] if len(all_records) > 10000 else all_records
        lengths = sorted([estimate_tokens(r) for r in sample])
        p50 = lengths[len(lengths) // 2]
        p90 = lengths[int(len(lengths) * 0.90)]
        p99 = lengths[int(len(lengths) * 0.99)]
        max_len = lengths[-1]
        print(f"    p50: {p50}, p90: {p90}, p99: {p99}, max: {max_len}")
        if p99 > 128:
            print(f"    WARNING: p99 ({p99}) exceeds MAX_SEQ_LEN=128, consider increasing")
        else:
            print(f"    OK: p99 ({p99}) fits within MAX_SEQ_LEN=128")
    except Exception as e:
        print(f"    Skipped (tokenizer not available): {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    random.shuffle(all_records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_records):,} examples to {OUTPUT_FILE}")
    print(f"\n→ Next: python 2_train.py")


if __name__ == "__main__":
    main()
