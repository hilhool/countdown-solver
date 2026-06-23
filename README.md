# Countdown Solver

A 1-billion-parameter language model fine-tuned to solve the Countdown numbers
game, paired with a generate-and-verify decoder that guarantees every answer it
submits is arithmetically correct.

Given a set of numbers and a target, the solver must combine the numbers with
`+`, `-`, `*`, `/` (each number used at most once) to reach the target:

```
Numbers: 75 80 90 24
Target:  61
Answer:  (75 + 90) - (80 + 24)
```

## Result

On the 2,000-puzzle public test set, using only the model and an exact-arithmetic
verifier (no exhaustive search):

| Stage | Solve rate |
|------:|:-----------|
| Beam pass (single attempt)        | 85.2% (1704/2000) |
| + verified sampling (≤120 / puzzle) | **97.5% (1951/2000)** |

Every reported solve is checked with exact rational arithmetic, so the solve rate
is the true accuracy on solvable puzzles, not an estimate.

## Approach

The pipeline is three scripts plus a shared core module.

### 1. Data: distilling an exact solver into traces (`1_generate.py`)

A small language model is poor at producing a whole nested equation in one shot,
but good at producing one arithmetic step at a time. So the training target is a
**reduction trace** rather than a single expression:

```
Numbers: 11 83 62 23 68          90 - 80 = 10
Target:  870            →         11 * 68 = 748
                                  83 + 748 = 831
                                  62 - 23 = 39
                                  39 + 831 = 870
```

For each sampled set of numbers, an exact integer search enumerates every target
reachable under the canonical Countdown ruleset (positive-integer intermediates,
so subtraction stays positive and division is exact) and records one trace per
target. The number-count mix (3 to 6 numbers) matches the public test set. The
script writes about 200k puzzles, and re-verifies every record through the same
parse-and-check path the model's outputs take at inference time, so the dataset is
correct by construction.

This is the distillation step: it compresses the competence of a slow symbolic
solver into a fast neural policy.

### 2. Fine-tuning (`2_train.py`)

Supervised fine-tuning of `google/gemma-3-1b-it` with LoRA (r = 128), 4-bit base
weights, on roughly 200k traces for 3 epochs. Loss is computed on the trace only;
the prompt is masked. Training fits on a single 16 GB GPU.

### 3. Inference: generate and verify (`3_inference.py`)

No brute-force search. The model proposes traces; an exact verifier accepts the
first one that is provably correct. The search budget escalates only for puzzles
that are still unsolved:

1. a beam pass over every puzzle;
2. escalating rounds of temperature sampling (8 → 16 → 32 → 64 samples), stopping
   each puzzle the moment a verified trace appears.

Because the verifier is exact, a wrong sample can never be accepted. Accuracy
comes from coverage: does *any* sample land on a correct trace. Sampling is spent
only where it is needed, and the solver folds each accepted trace back into a
single nested equation for submission.

The distinction from brute force is deliberate. Brute force enumerates the entire
solution space independently of the model. Here the model's learned policy decides
what to try and the verifier only filters, so the full search space is never
enumerated.

## Reproduce

Requirements: an NVIDIA GPU with ~16 GB VRAM (CUDA 12+) and Python 3.10+.

```bash
pip install -r requirements.txt

python 1_generate.py     # ~25 min  -> train_traces.jsonl
python 2_train.py        # several hours on an RTX 4060 Ti -> ./gemma-countdown
python 3_inference.py    # solves test_public.csv -> submission.csv
```

Check the model on a slice without a full run:

```bash
python 3_inference.py --limit 200
```

## Layout

| Path | Purpose |
|------|---------|
| `countdown.py`    | Core logic: search, trace ↔ equation conversion, exact verifier (shared by all stages) |
| `1_generate.py`   | Build the verified trace dataset |
| `2_train.py`      | LoRA fine-tuning |
| `3_inference.py`  | Generate-and-verify solver, writes `submission.csv` |
| `test_public.csv` | Public test puzzles (`id, target, nums`) |

The trained adapter and the generated dataset are not committed (see
`.gitignore`); both are reproduced by the scripts above.

## Notes and limitations

- A small fraction of puzzles have no solution under the ruleset; no method can
  solve those, and they cap the achievable solve rate.
- Verified sampling trades compute for accuracy. The escalating schedule keeps the
  average sample count low because most puzzles are solved in the first one or two
  rounds; only the hard tail consumes the full budget.
- The verifier uses exact rational arithmetic (`fractions.Fraction`), so there are
  no floating-point false positives.
