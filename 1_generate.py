"""Generate verified Countdown training data as step-by-step traces.

    python 1_generate.py            # ~200k examples -> train_traces.jsonl

For each puzzle we sample a set of numbers (matching the public test set:
3-6 values drawn from 1-99), enumerate every target reachable with exact
integer arithmetic, and emit a step trace for a sample of those targets. Every
record is verified before it is written, so the dataset is correct by
construction. Each record is:

    {"numbers": [...], "target": int, "trace": "a op b = c\\n..."}
"""

import json
import random
import time
from collections import Counter

from tqdm import tqdm

from countdown import trace_to_lines, parse_trace, trace_to_equation, verify

OUTPUT_FILE = "train_traces.jsonl"
SEED = 42
TARGET_RECORDS = 200_000

# Number-count weights mirror the public test set (3:14%, 4:42%, 5:29%, 6:15%).
NUM_COUNT_WEIGHTS = {3: 0.14, 4: 0.42, 5: 0.29, 6: 0.15}

# Targets sampled per number-set, so no single set dominates the data.
TARGETS_PER_SET = 4
NODE_BUDGET = 60_000  # search nodes per number-set before giving up


def sample_numbers(rng) -> list[int]:
    counts = list(NUM_COUNT_WEIGHTS)
    weights = list(NUM_COUNT_WEIGHTS.values())
    k = rng.choices(counts, weights=weights)[0]
    return [rng.randint(1, 99) for _ in range(k)]


def collect_reachable(numbers, budget):
    """Return {target_value: trace} for every integer in 1..999 reachable from
    `numbers` with positive-integer intermediates, exploring up to `budget`
    nodes. Each value keeps the first trace found."""
    found = {}
    nodes = 0

    def rec(items):
        nonlocal nodes
        for val, steps in items:
            if steps and 1 <= val <= 999 and val not in found:
                found[val] = list(steps)
        if len(items) < 2:
            return
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                nodes += 1
                if nodes > budget:
                    return
                a_val, a_steps = items[i]
                b_val, b_steps = items[j]
                rest = [items[k] for k in range(n) if k != i and k != j]
                moves = [
                    (a_val + b_val, (a_val, "+", b_val)),
                    (a_val * b_val, (a_val, "*", b_val)),
                ]
                hi, lo = (a_val, b_val) if a_val >= b_val else (b_val, a_val)
                if hi > lo:
                    moves.append((hi - lo, (hi, "-", lo)))
                if lo != 0 and hi % lo == 0:
                    moves.append((hi // lo, (hi, "/", lo)))
                for new_val, (x, op, y) in moves:
                    step = (x, op, y, new_val)
                    rec(rest + [(new_val, a_steps + b_steps + [step])])
                    if nodes > budget:
                        return

    rec([(int(x), []) for x in numbers])
    return found


def main():
    rng = random.Random(SEED)
    print(f"Generating up to {TARGET_RECORDS:,} verified trace examples...")

    records = []
    seen = set()
    t0 = time.time()
    pbar = tqdm(total=TARGET_RECORDS)

    while len(records) < TARGET_RECORDS:
        numbers = sample_numbers(rng)
        reachable = collect_reachable(numbers, NODE_BUDGET)
        if not reachable:
            continue

        targets = list(reachable)
        rng.shuffle(targets)
        for target in targets[:TARGETS_PER_SET]:
            key = (tuple(sorted(numbers)), target)
            if key in seen:
                continue
            trace = reachable[target]
            lines = trace_to_lines(trace)
            # Defensive re-verification through the same path inference uses.
            eq = trace_to_equation(parse_trace(lines), numbers)
            if eq is None or not verify(eq, target, numbers):
                continue
            seen.add(key)
            records.append({"numbers": numbers, "target": target, "trace": lines})
            pbar.update(1)
            if len(records) >= TARGET_RECORDS:
                break

    pbar.close()
    rng.shuffle(records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    num_dist = Counter(len(r["numbers"]) for r in records)
    step_dist = Counter(r["trace"].count("\n") + 1 for r in records)
    targets = [r["target"] for r in records]
    print(f"\nWrote {len(records):,} records to {OUTPUT_FILE} in {elapsed:.0f}s")
    print("  numbers per puzzle:", dict(sorted(num_dist.items())))
    print("  steps per solution:", dict(sorted(step_dist.items())))
    print(f"  unique targets: {len(set(targets))}/999, range {min(targets)}-{max(targets)}")
    print("\n  examples:")
    for rec in records[:4]:
        print(f"    nums={rec['numbers']} target={rec['target']}")
        for line in rec["trace"].splitlines():
            print(f"        {line}")


if __name__ == "__main__":
    main()
