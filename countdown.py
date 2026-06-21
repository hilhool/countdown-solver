"""Core Countdown logic shared by data generation and inference.

A puzzle gives a set of numbers and a target. A solution combines the numbers
with +, -, *, / (each number used at most once) to reach the target. We use the
canonical Countdown ruleset: every intermediate result is a positive integer,
so subtraction must stay positive and division must be exact.

Two representations are used:

* equation  - a single nested expression, e.g. "(90 - 80) + (75 - 24)".
              This is the format the competition expects in submission.csv.
* trace     - one binary operation per line, e.g.
                  90 - 80 = 10
                  75 - 24 = 51
                  10 + 51 = 61
              This is what the model is trained to produce: each line is a
              single step a small model can emit reliably.

`trace_to_equation` folds a trace back into a nested equation, and `verify`
checks a candidate equation with exact arithmetic.
"""

import re
import random
from fractions import Fraction

OPS = ("+", "-", "*", "/")


def make_prompt_numbers(numbers, target) -> str:
    """The user-turn text. Shared by training and inference so they never drift."""
    return f"Numbers: {' '.join(map(str, numbers))}\nTarget: {target}"


def find_trace(numbers, target, rng=random):
    """Search for a solution trace reaching `target` using integer intermediates.

    Returns a list of (a, op, b, result) steps, or None if no solution is found
    within the explored space. Operands are explored in randomized order so the
    same puzzle can yield different solutions across calls.
    """
    target = int(target)

    # Each item is (value, [steps that produced it]).
    def search(items):
        for val, steps in items:
            if val == target and steps:
                return steps
        if len(items) < 2:
            return None
        n = len(items)
        order = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(order)
        for i, j in order:
            a_val, a_steps = items[i]
            b_val, b_steps = items[j]
            rest = [items[k] for k in range(n) if k != i and k != j]
            moves = []
            moves.append((a_val + b_val, (a_val, "+", b_val)))
            moves.append((a_val * b_val, (a_val, "*", b_val)))
            hi, lo = (a_val, b_val) if a_val >= b_val else (b_val, a_val)
            if hi > lo:
                moves.append((hi - lo, (hi, "-", lo)))
            if lo != 0 and hi % lo == 0:
                moves.append((hi // lo, (hi, "/", lo)))
            rng.shuffle(moves)
            for new_val, (x, op, y) in moves:
                step = (x, op, y, new_val)
                res = search(rest + [(new_val, a_steps + b_steps + [step])])
                if res is not None:
                    return res
        return None

    items = [(int(x), []) for x in numbers]
    return search(items)


def trace_to_lines(trace) -> str:
    """Render a trace as text lines: 'a op b = result' per step."""
    return "\n".join(f"{a} {op} {b} = {r}" for (a, op, b, r) in trace)


def parse_trace(text: str):
    """Parse model output of the form 'a op b = c' lines into steps.

    Returns a list of (a, op, b, c) integer tuples. Tolerant of extra prose,
    operator glyphs, and a leading 'Target =' echo.
    """
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
    steps = []
    for line in text.splitlines():
        m = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)", line)
        if m:
            a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
            steps.append((a, op, b, c))
    return steps


def trace_to_equation(steps, numbers):
    """Fold parsed steps into a single nested equation string.

    Mirrors the bookkeeping the verifier uses: a pool of (value, expr) tokens.
    Each step consumes two matching tokens and pushes the combined expression.
    Returns the equation string, or None if a step references an unavailable
    value or the arithmetic does not check out.
    """
    pool = [(int(n), str(n)) for n in numbers]

    def take(value):
        for idx, (v, _) in enumerate(pool):
            if v == value:
                return pool.pop(idx)
        return None

    last_expr = None
    for a, op, b, c in steps:
        ta = take(a)
        if ta is None:
            return None
        tb = take(b)
        if tb is None:
            pool.append(ta)  # restore before failing
            return None
        if op == "-" and ta[0] - tb[0] != c:
            return None
        if op == "+" and ta[0] + tb[0] != c:
            return None
        if op == "*" and ta[0] * tb[0] != c:
            return None
        if op == "/" and (tb[0] == 0 or ta[0] % tb[0] != 0 or ta[0] // tb[0] != c):
            return None
        expr = f"({ta[1]} {op} {tb[1]})"
        pool.append((c, expr))
        last_expr = expr

    if last_expr is None:
        return None
    return strip_outer_parens(last_expr)


def strip_outer_parens(expr: str) -> str:
    """Remove redundant outermost parentheses wrapping the whole expression."""
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        wraps_all = True
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i < len(expr) - 1:
                wraps_all = False
                break
        if wraps_all:
            expr = expr[1:-1]
        else:
            break
    return expr


def verify(equation: str, target: int, numbers) -> bool:
    """Check an equation: well-formed, uses each given number at most once, and
    evaluates exactly to target using exact (Fraction) arithmetic."""
    if not equation or not re.fullmatch(r"[\d\s\+\-\*\/\(\)]+", equation):
        return False
    used = list(map(int, re.findall(r"\d+", equation)))
    pool = list(numbers)
    for n in used:
        if n in pool:
            pool.remove(n)
        else:
            return False
    try:
        expr = re.sub(r"(\d+)", r"Fraction(\1)", equation)
        return eval(expr, {"__builtins__": {}, "Fraction": Fraction}) == Fraction(int(target))
    except Exception:
        return False
