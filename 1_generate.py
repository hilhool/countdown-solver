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

# The public test mix is 3:14%, 4:42%, 5:29%, 6:15%, but the model only fails on
# the hard tail (5-6 numbers). We oversample that tail relative to the test mix so
# the policy sees more long reductions, where first-pass errors actually happen.
NUM_COUNT_WEIGHTS = {3: 0.10, 4: 0.30, 5: 0.32, 6: 0.28}

# Targets sampled per number-set, so no single set dominates the data.
TARGETS_PER_SET = 4
NODE_BUDGET = 60_000  # search nodes per number-set before giving up


def sample_numbers(rng) -> list[int]:
    counts = list(NUM_COUNT_WEIGHTS)
    weights = list(NUM_COUNT_WEIGHTS.values())
    k = rng.choices(counts, weights=weights)[0]
    # Test values span 1..100 (100 does occur); sampling 1..99 would leave the
    # model never having seen 100 as an operand.
    return [rng.randint(1, 100) for _ in range(k)]


# Rank operators so equal-length traces have a single deterministic ordering.
_OP_RANK = {"+": 0, "-": 1, "*": 2, "/": 3}


def _trace_key(steps):
    """Sort key for choosing among traces: fewest steps first, then a fixed
    lexicographic order. Picking the same canonical trace every time keeps the
    training target low-entropy, which a greedy/beam decoder reproduces far more
    reliably than an arbitrary search-order trace."""
    return (len(steps), tuple((a, _OP_RANK[op], b, r) for (a, op, b, r) in steps))


def collect_reachable(numbers, budget):
    """Return {target_value: trace} for every integer in 1..999 reachable from
    `numbers` with positive-integer intermediates, exploring up to `budget`
    nodes.

    For each value we keep the SHORTEST trace (fewest steps, i.e. fewest numbers
    used), breaking ties deterministically via `_trace_key`. Shorter traces mean
    fewer chances for the decoder to deviate, and a single canonical form per
    puzzle is much easier to learn than the first trace a search happens to hit.
    Commutative steps are emitted larger-operand-first, matching '-' and '/', so
    every line has one canonical shape (e.g. always '9 + 3 = 12', never '3 + 9')."""
    found = {}  # value -> (key, steps)
    nodes = 0

    def consider(val, steps):
        if not (steps and 1 <= val <= 999):
            return
        key = _trace_key(steps)
        cur = found.get(val)
        if cur is None or key < cur[0]:
            found[val] = (key, list(steps))

    def rec(items):
        nonlocal nodes
        for val, steps in items:
            consider(val, steps)
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
                hi, lo = (a_val, b_val) if a_val >= b_val else (b_val, a_val)
                # Larger operand first for every op, so commutative steps share
                # the canonical shape used by '-' and '/'.
                moves = [
                    (hi + lo, (hi, "+", lo)),
                    (hi * lo, (hi, "*", lo)),
                ]
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
    return {val: steps for val, (key, steps) in found.items()}


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

        # Bias target selection toward longer reductions. Keeping the shortest
        # trace per target is the right supervision, but most targets are
        # reachable in 1-2 steps, so a uniform pick would flood the data with
        # trivial examples and starve the long reductions the model fails on.
        # Weight each target by trace length^2 and sample without replacement
        # (Efraimidis-Spirakis: key = u^(1/w), take the largest keys).
        scored = [
            (rng.random() ** (1.0 / (len(reachable[t]) ** 2)), t)
            for t in reachable
        ]
        scored.sort(reverse=True)
        for _, target in scored[:TARGETS_PER_SET]:
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
