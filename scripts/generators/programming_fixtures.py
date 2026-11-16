#!/usr/bin/env python3
"""Generate original programming fixtures using analytic or exhaustive optima.

Copyright (c) 2026 kakune.
SPDX-License-Identifier: MIT OR Apache-2.0

This manual generator uses only Python's standard library and no external data.
"""

import argparse
import itertools
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypedDict


class Problem(TypedDict):
    name: str
    n: int
    m: int
    ptr: list[int]
    index: list[int]
    value: list[float]
    objective: list[float]
    offset: float
    lower: list[float | None]
    upper: list[float | None]
    row_lower: list[float | None]
    row_upper: list[float | None]
    integer: list[int]
    quadratic_diagonal: float


class Reference(TypedDict):
    status: str
    objective: float
    primal: list[float]


class ReferencedProblem(Problem):
    minimum: float
    reference: Reference


class AssignmentReference(TypedDict):
    objective: int
    permutation: list[int]


class Assignment(TypedDict):
    name: str
    n: int
    a: list[int]
    b: list[int]
    minimum: int
    minimizer: list[int]
    references: list[AssignmentReference]


class Fixtures(TypedDict):
    copyright: str
    license: str
    provenance: str
    lp: list[ReferencedProblem]
    qp: list[ReferencedProblem]
    mip: list[ReferencedProblem]
    qap: list[Assignment]


def case(
    name: str, matrix: Sequence[Sequence[float]], objective: Sequence[float],
    lower: Sequence[float | None], upper: Sequence[float | None],
    row_lower: Sequence[float | None], row_upper: Sequence[float | None],
    integer: Iterable[int] = (), h: float = 0,
) -> Problem:
    ptr, index, value = [0], list[int](), list[float]()
    for row in matrix:
        for j, entry in enumerate(row):
            if entry:
                index.append(j)
                value.append(entry)
        ptr.append(len(index))
    return dict(
        name=name, n=len(objective), m=len(matrix), ptr=ptr, index=index, value=value,
        objective=list(objective), offset=0, lower=list(lower), upper=list(upper),
        row_lower=list(row_lower), row_upper=list(row_upper), integer=list(integer),
        quadratic_diagonal=h,
    )


def objective(case: Problem, x: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(case["objective"], x)) + case["quadratic_diagonal"] * sum(v * v for v in x) / 2


def reference(case: Problem, x: Sequence[float], method: str) -> ReferencedProblem:
    """Return a case with its reference result, leaving the input untouched."""
    minimum = objective(case, x)
    return {
        **case,
        "minimum": minimum,
        "reference": dict(status=method + ".kOptimal", objective=minimum, primal=list(x)),
    }


def linear(n: int, variant: int) -> ReferencedProblem:
    target = [1 + (j + variant) % 3 for j in range(n)]
    matrix = [[(2 if i == j else 0.5 if j + 1 == i else 0) for j in range(n)] for i in range(n)]
    rhs = [sum(a * b for a, b in zip(row, target)) for row in matrix]
    senses = [(i + variant) % 3 for i in range(n)]  # equality, lower, upper
    dual = [1 if sense != 2 else -1 for sense in senses]
    costs = [sum(dual[i] * matrix[i][j] for i in range(n)) for j in range(n)]
    result = case(
        f"manufactured-linear-{variant}", matrix, costs, [0] * n, [None] * n,
        [None if s == 2 else b for s, b in zip(senses, rhs)],
        [None if s == 1 else b for s, b in zip(senses, rhs)],
    )
    # c = A^T dual; every one-sided slack increases c^T x. A is invertible.
    return reference(result, target, "AnalyticKKT")


def quadratic(linear_case: ReferencedProblem) -> ReferencedProblem:
    x = linear_case["reference"]["primal"]
    result: ReferencedProblem = {
        **linear_case,
        "name": linear_case["name"].replace("linear", "quadratic"),
        "quadratic_diagonal": 0.5,
        "objective": [c - 0.5 * v for c, v in zip(linear_case["objective"], x)],
    }
    return reference(result, x, "AnalyticKKT")


def integer(variant: int) -> ReferencedProblem:
    matrix, costs, rhs = (
        ([[3, 2], [1, 4]], [1, 1], [7, 6]) if variant == 0 else
        ([[2, 3, 1], [1, 1, 3]], [1, 2, 1], [11, 8])
    )
    n = len(costs)
    result = case(f"manufactured-integer-{variant}", matrix, costs, [0] * n, [5] * n, rhs, [None] * len(rhs), range(n))
    feasible = [x for x in itertools.product(range(6), repeat=n)
                if all(sum(a * b for a, b in zip(row, x)) >= bound for row, bound in zip(matrix, rhs))]
    return reference(result, min(feasible, key=lambda x: objective(result, x)), "ExhaustiveInteger")


def assignment(n: int) -> Assignment:
    a = [0 if i == j else 1 + (3 * i + 5 * j + i * j) % 9 for i in range(n) for j in range(n)]
    b = [0 if i == j else abs(i - j) + (2 * i + j) % 4 for i in range(n) for j in range(n)]

    def score(p: Sequence[int]) -> int:
        return sum(a[i * n + j] * b[p[i] * n + p[j]] for i in range(n) for j in range(n))

    optimum = min(itertools.permutations(range(n)), key=score)
    return dict(name=f"manufactured-assignment-{n}", n=n, a=a, b=b,
                minimum=score(optimum), minimizer=list(optimum),
                references=[dict(objective=score(p), permutation=list(p))
                            for p in [tuple(range(n)), tuple(reversed(range(n)))]])


def generate_data() -> Fixtures:
    lp = [linear(n, i) for i, n in enumerate([3, 4, 5, 7, 9, 12])]
    return dict(
        copyright="Copyright (c) 2026 kakune", license="MIT OR Apache-2.0",
        provenance="Original dyadic LP/QP constructions and exhaustive bounded MIP/QAP enumeration; no downloaded problem data.",
        lp=lp, qp=[quadratic(c) for c in lp[:2]],
        mip=[integer(i) for i in range(2)], qap=[assignment(n) for n in [4, 5, 6]],
    )


class Options(argparse.Namespace):
    check: bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the fixture without writing it")
    args = parser.parse_args(namespace=Options())
    dest = Path(__file__).resolve().parents[2] / "crates/scioxide-optimize/tests/data/programming.json"
    text = json.dumps(generate_data(), separators=(",", ":")) + "\n"
    if args.check:
        if dest.read_text() != text:
            parser.exit(1, "Programming fixtures differ from their original constructions\n")
        print("Programming fixtures reproduce exactly")
    else:
        dest.write_text(text)


if __name__ == "__main__":
    main()
