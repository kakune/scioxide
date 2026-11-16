#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["sympy==1.14.0", "mpmath==1.3.0"]
# ///
"""Rebuild the executable MGH residual definitions from mathematical formulas.

Copyright (c) 2026 kakune and SciOxide contributors.
SPDX-License-Identifier: MIT OR Apache-2.0

Run: uv run scripts/generators/mgh_models.py --check
     uv run scripts/generators/mgh_models.py --write
     uv run scripts/generators/mgh_models.py --output /tmp/mgh-models.json

Only the executable `models[*].residuals` field is regenerated. Historical
reference-tool results, starts, bounds, and solver work counts are preserved.
This is a new, auditable generator, not a recovered historical generator.
No fixture, Fortran source, or network resource is read to construct formulas.

Mathematical reference: More, Garbow & Hillstrom (1981), section 2,
doi:10.1145/355934.355936. Problem numbers below are the OBJFCN/SSQFCN
family numbering used by the existing test harness, not section-2 numbering.
Tabulated observations for Bard, Kowalik-Osborne, Meyer and Osborne problems
retain the MINPACK input notice in crates/scioxide-optimize/THIRD_PARTY_NOTICES.md.
Gaussian observations are recomputed from the normal density.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Sequence
from typing import Literal, TypedDict, cast

import sympy as s
import mpmath

Family = Literal["unconstrained", "least-squares"]
Token = float | str


class Model(TypedDict):
    family: Family
    problem: int
    n: int
    m: int | None
    residuals: list[list[Token]]


class GenerationRecord(TypedDict):
    script: str
    dependencies: dict[str, str]
    scope: str
    definitions_sha256: str
    record: str


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "crates/scioxide-optimize/tests/data/mgh.json"

# Explicit coverage, independent of the generated fixture.
SPECS: tuple[tuple[Family, int, int, int | None], ...] = tuple(
    [("unconstrained", p, n, None) for p, n in [
        (1, 3), (2, 6), (3, 3), (4, 2), (5, 3), (6, 10), (7, 9),
        (7, 12), (8, 10), (9, 1), (9, 4), (9, 10), (10, 2), (11, 4),
        (12, 3), (13, 10), (14, 2), (15, 4), (16, 2), (17, 4),
        (18, 7), (18, 8), (18, 9), (18, 10)]]
    + [("least-squares", p, n, m) for p, n, m in [
        (1, 5, 10), (1, 5, 50), (2, 5, 10), (2, 5, 50),
        (3, 5, 10), (3, 5, 50), (4, 2, 2), (5, 3, 3), (6, 4, 4),
        (7, 2, 2), (8, 3, 15), (9, 4, 11), (10, 3, 16),
        (11, 6, 31), (11, 9, 31), (11, 12, 31), (12, 3, 10),
        (13, 2, 10), (14, 4, 20), (15, 1, 8), (15, 8, 8),
        (15, 9, 9), (15, 10, 10), (16, 10, 10), (16, 30, 30),
        (16, 40, 40), (17, 5, 33), (18, 11, 65)]]
)

# Gaussian observations are the normal density at half-integers, rounded to
# four decimal places; compute them rather than transcribing a CALGO array.
GAUSSIAN = [round(math.exp(-((7-i)/2)**2/2)/math.sqrt(2*math.pi), 4) for i in range(15)]
# MINPACK problem input observations; not CALGO implementation code.
BARD = [.14, .18, .22, .25, .29, .32, .35, .39, .37, .58, .73, .96, 1.34, 2.1, 4.39]
KOWALIK_Y = [.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246]
KOWALIK_U = [4., 2., 1., .5, .25, .167, .125, .1, .0833, .0714, .0625]
MEYER = [34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744,
         8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872]
OSBORNE1 = [.844, .908, .932, .936, .925, .908, .881, .850, .818, .784,
            .751, .718, .685, .658, .628, .603, .580, .558, .538, .522,
            .506, .490, .478, .467, .457, .448, .438, .431, .424, .420,
            .414, .411, .406]
OSBORNE2 = [1.366, 1.191, 1.112, 1.013, .991, .885, .831, .847, .786, .725,
            .746, .679, .608, .655, .616, .606, .602, .626, .651, .724,
            .649, .649, .694, .644, .624, .661, .612, .558, .533, .495,
            .500, .423, .395, .375, .372, .391, .396, .405, .428, .429,
            .523, .562, .607, .653, .672, .708, .633, .668, .645, .632,
            .591, .559, .597, .625, .739, .710, .729, .720, .636, .581,
            .428, .292, .162, .098, .054]


def helical(x: Sequence[s.Expr]) -> list[s.Expr]:
    angle = s.atan2(x[1], x[0])
    # Angle in turns, continued through the negative-x half plane.
    turn = angle / (2 * s.pi) + s.Piecewise(
        (1, s.And(x[0] < 0, angle < 0)), (0, True))
    return [10 * (x[2] - 10 * turn), 10 * (s.sqrt(x[0]**2 + x[1]**2) - 1), x[2]]


def rosenbrock(x: Sequence[s.Expr]) -> list[s.Expr]:
    return [1 - x[0], 10 * (x[1] - x[0]**2)]


def powell_singular(x: Sequence[s.Expr]) -> list[s.Expr]:
    return [x[0] + 10*x[1], s.sqrt(5)*(x[2]-x[3]),
            (x[1]-2*x[2])**2, s.sqrt(10)*(x[0]-x[3])**2]


def powell_scaled(x: Sequence[s.Expr]) -> list[s.Expr]:
    return [10000*x[0]*x[1]-1, s.exp(-x[0])+s.exp(-x[1])-1.0001]


def box(x: Sequence[s.Expr], m: int = 10) -> list[s.Expr]:
    return [s.exp(-i*x[0]/10)-s.exp(-i*x[1]/10)
            + s.Mul(x[2], s.exp(-i)-s.exp(-s.Rational(i, 10)), evaluate=False) for i in range(1, m+1)]


def watson(x: Sequence[s.Expr]) -> list[s.Expr]:
    result = []
    for i in range(1, 30):
        t = s.Rational(i, 29)
        polynomial = sum(v*t**j for j, v in enumerate(x))
        derivative = sum(j*x[j]*t**(j-1) for j in range(1, len(x)))
        result.append(derivative-polynomial**2-1)
    return result + [x[0], x[1]-x[0]**2-1]


def brown_dennis(x: Sequence[s.Expr], m: int = 20) -> list[s.Expr]:
    return [(x[0]+i*x[1]/5-s.exp(s.Rational(i, 5)))**2
            + (x[2]+s.sin(s.Rational(i, 5))*x[3]-s.cos(s.Rational(i, 5)))**2 for i in range(1, m+1)]


def chebyquad(x: Sequence[s.Expr], m: int) -> list[s.Expr]:
    # Chebyshev recurrence; no coefficient table or expanded source program.
    previous = [s.Integer(1)] * len(x)
    current = [2*v-1 for v in x]
    residuals = []
    for degree in range(1, m+1):
        correction = s.Rational(1, degree**2-1) if degree % 2 == 0 else 0
        residuals.append(sum(current)/len(x) + correction)
        previous, current = current, [2*(2*v-1)*b-a for v, a, b in zip(x, previous, current)]
    return residuals


def unconstrained(problem: int, x: Sequence[s.Expr]) -> list[s.Expr]:
    n = len(x)
    if problem == 1:
        return helical(x)
    if problem == 2:  # Biggs EXP6
        return [x[2]*s.exp(-i*x[0]/10)-x[3]*s.exp(-i*x[1]/10)
                + x[5]*s.exp(-i*x[4]/10)
                - (s.exp(-s.Rational(i, 10))-5*s.exp(-i)+3*s.exp(-s.Rational(4*i, 10)))
                for i in range(1, 14)]
    if problem == 3:
        return [x[0]*s.exp(-x[1]*(s.Rational(7-i, 2)-x[2])**2/2)-y
                for i, y in enumerate(GAUSSIAN)]
    if problem == 4:
        return powell_scaled(x)
    if problem == 5:
        return box(x)
    if problem == 6:  # Variably dimensioned
        weighted = sum((j+1)*(v-1) for j, v in enumerate(x))
        return [v-1 for v in x] + [weighted, weighted**2]
    if problem == 7:
        return watson(x)
    if problem == 8:  # Penalty I, alpha = 10^-5
        return [s.sqrt(s.Rational(1, 100000))*(v-1) for v in x] + [sum(v*v for v in x)-s.Rational(1, 4)]
    if problem == 9:  # Penalty II
        small = s.sqrt(s.Rational(1, 100000))
        result = [sum((n-j)*v*v for j, v in enumerate(x))-1, x[0]-s.Rational(1, 5)]
        for j in range(1, n):
            target = float(s.exp(s.Rational(j+1, 10))+s.exp(s.Rational(j, 10)))
            result += [small*(s.exp(x[j]/10)+s.exp(x[j-1]/10)-target),
                       small*(s.exp(x[j]/10)-s.exp(-s.Rational(1, 10)))]
        return result
    if problem == 10:
        return [x[0]-1000000, x[1]-s.Rational(2, 1000000), x[0]*x[1]-2]
    if problem == 11:
        return brown_dennis(x)
    if problem == 12:  # Gulf research and development
        return [s.exp(-s.Abs(25+(-50*s.log(s.Rational(i, 100)))**(s.Rational(2, 3))-x[1])**x[2]/x[0])-s.Rational(i, 100)
                for i in range(1, 100)]
    if problem == 13:
        common = n-sum(s.cos(v) for v in x)
        return [common+(j+1)*(1-s.cos(v))-s.sin(v) for j, v in enumerate(x)]
    if problem == 14:
        return rosenbrock(x)
    if problem == 15:
        return powell_singular(x)
    if problem == 16:  # Beale
        return [y-x[0]*(1-x[1]**i) for i, y in enumerate([s.Rational(3, 2), s.Rational(9, 4), s.Rational(21, 8)], 1)]
    if problem == 17:  # Wood, factored cross term
        return [10*(x[1]-x[0]**2), 1-x[0], s.sqrt(90)*(x[3]-x[2]**2),
                1-x[2], s.sqrt(10)*(x[1]+x[3]-2), (x[1]-x[3])/s.sqrt(10)]
    if problem == 18:
        return chebyquad(x, n)
    raise ValueError(f"Unknown unconstrained problem {problem}")


def least_squares(problem: int, x: Sequence[s.Expr], m: int) -> list[s.Expr]:
    n = len(x)
    if problem == 1:  # Linear, full rank
        common = -2*sum(x)/m-1
        return [s.sympify(common+(x[i] if i < n else 0)) for i in range(m)]
    if problem == 2:  # Linear, rank one
        weighted = sum((j+1)*v for j, v in enumerate(x))
        return [s.Add(s.Mul(i+1, weighted, evaluate=False), -1, evaluate=False) for i in range(m)]
    if problem == 3:  # Linear, rank one with zero columns and rows
        weighted = sum((j+1)*x[j] for j in range(1, n-1))
        return [s.Add(s.Mul(i, weighted, evaluate=False), -1, evaluate=False) for i in range(m-1)] + [s.Integer(-1)]
    if problem == 4:
        # SSQFCN lists the scaled quadratic residual first.
        return list(reversed(rosenbrock(x)))
    if problem == 5:
        return helical(x)
    if problem == 6:
        return powell_singular(x)
    if problem == 7:  # Freudenstein and Roth
        return [x[0]+x[1]*((5-x[1])*x[1]-2)-13,
                x[0]+x[1]*((x[1]+1)*x[1]-14)-29]
    if problem == 8:
        return [y-x[0]-i/((16-i)*x[1]+min(i, 16-i)*x[2]) for i, y in enumerate(BARD, 1)]
    if problem == 9:
        inputs = [s.Rational(str(u)) for u in KOWALIK_U]
        return [y-u*x[0]*(u+x[1])/(u*u+u*x[2]+x[3]) for u, y in zip(inputs, KOWALIK_Y)]
    if problem == 10:
        return [x[0]*s.exp(x[1]/(45+5*i+x[2]))-y for i, y in enumerate(MEYER, 1)]
    if problem == 11:
        return watson(x)
    if problem == 12:
        return box(x, m)
    if problem == 13:  # Jennrich and Sampson
        return [2+2*i-s.exp(i*x[0])-s.exp(i*x[1]) for i in range(1, m+1)]
    if problem == 14:
        return brown_dennis(x, m)
    if problem == 15:
        return chebyquad(x, m)
    if problem == 16:  # Brown almost-linear
        return [v+sum(x)-(n+1) for v in x[:-1]] + [s.prod(x)-1]
    if problem == 17:
        return [y-x[0]-x[1]*s.exp(-10*i*x[3])-x[2]*s.exp(-10*i*x[4]) for i, y in enumerate(OSBORNE1)]
    if problem == 18:
        return [y-x[0]*s.exp(-s.Rational(i, 10)*x[4])-sum(
            x[j]*s.exp(-x[j+4]*(s.Rational(i, 10)-x[j+7])**2) for j in range(1, 4))
            for i, y in enumerate(OSBORNE2)]
    raise ValueError(f"Unknown least-squares problem {problem}")


def rpn(expr: s.Basic, variables: Sequence[s.Expr]) -> list[Token]:
    """Serialize our symbolic equations to the test interpreter's small language."""
    if expr in variables:
        return [f"x:{variables.index(expr)}"]
    if isinstance(expr, s.Expr) and expr.is_number:
        return [float(expr)]
    if isinstance(expr, (s.Add, s.Mul, s.And)):
        operator = "+" if isinstance(expr, s.Add) else "*" if isinstance(expr, s.Mul) else "and"
        args = expr.args
        result = rpn(args[0], variables)
        for arg in args[1:]:
            result += rpn(arg, variables) + [operator]
        return result
    if isinstance(expr, s.Pow):
        base, exponent = expr.args
        if exponent.is_Integer:
            return rpn(base, variables) + [f"powi:{exponent}"]
        if exponent.is_number:
            return rpn(base, variables) + [f"powf:{float(exponent)}"]
        return rpn(base, variables) + rpn(exponent, variables) + ["pow"]
    if isinstance(expr, s.Piecewise):
        yes_branch, no_branch = expr.args
        yes, condition = yes_branch.args
        no, otherwise = no_branch.args
        if otherwise != s.true:
            raise ValueError("Piecewise requires an unconditional fallback")
        return rpn(condition, variables) + rpn(yes, variables) + rpn(no, variables) + ["select"]
    if isinstance(expr, s.StrictLessThan):
        return rpn(expr.lhs, variables) + rpn(expr.rhs, variables) + ["<"]
    operators = {s.exp: "exp", s.log: "ln", s.sin: "sin", s.cos: "cos", s.Abs: "abs", s.atan2: "atan2"}
    if expr.func in operators:
        return [token for arg in expr.args for token in rpn(arg, variables)] + [operators[expr.func]]
    raise ValueError(f"Unsupported expression: {expr}")


def generate_models() -> list[Model]:
    result: list[Model] = []
    for family, problem, n, m in SPECS:
        x = list(s.symbols(f"x0:{n}", real=True))
        if family == "unconstrained":
            equations = unconstrained(problem, x)
        else:
            if m is None:
                raise ValueError("Least-squares models require a residual count")
            equations = least_squares(problem, x, m)
        result.append(dict(family=family, problem=problem, n=n, m=m,
                           residuals=[rpn(s.sympify(eq), x) for eq in equations]))
    return result


def update_deck(
    deck: dict[str, object], generated: Sequence[Model],
    provenance: GenerationRecord, *, check: bool = False,
) -> dict[str, object]:
    """Replace definitions without modifying the deck or its historical records."""
    old_models = deck.get("models")
    if not isinstance(old_models, list) or len(old_models) != len(generated):
        raise ValueError("Unexpected model count")
    models: list[dict[str, object]] = []
    for i, (old, new) in enumerate(zip(old_models, generated)):
        if not isinstance(old, dict):
            raise ValueError(f"Model {i} must be an object")
        for key in ("family", "problem", "n", "m"):
            if old.get(key) != new[key]:
                raise ValueError(f"Model {i} has unexpected {key}")
        if check and old.get("residuals") != new["residuals"]:
            raise ValueError(f"Model {i} differs from the formula generator")
        models.append({**old, "residuals": new["residuals"]})
    if check and deck.get("model_generation") != provenance:
        raise ValueError("MGH generation record differs; run --write to record it")
    return {**deck, "models": models, "model_generation": provenance}


class Options(argparse.Namespace):
    check: bool
    write: bool
    output: Path | None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--output", type=Path, help="Write standalone definitions, without reading the fixture")
    args = parser.parse_args(namespace=Options())
    if (s.__version__, mpmath.__version__) != ("1.14.0", "1.3.0"):
        parser.error("Use SymPy 1.14.0 and mpmath 1.3.0 (or run this script with uv run)")
    generated = generate_models()
    digest = hashlib.sha256(json.dumps(generated, separators=(",", ":")).encode()).hexdigest()
    if args.output:
        args.output.write_text(json.dumps(generated, separators=(",", ":")) + "\n")
        return
    deck = cast(dict[str, object], json.loads(FIXTURE.read_text()))
    provenance: GenerationRecord = dict(
        script="scripts/generators/mgh_models.py",
        dependencies={"sympy": "1.14.0", "mpmath": "1.3.0"},
        scope="models[*].residuals only; historical oracle results and solver work counts are retained",
        definitions_sha256=digest,
        record="DATA_PROVENANCE.md#reproducing-the-mgh-problem-definitions",
    )
    try:
        updated = update_deck(deck, generated, provenance, check=args.check)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    if args.write:
        FIXTURE.write_text(json.dumps(updated, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"52 MGH model definitions reproduced; SHA256 {digest}")


if __name__ == "__main__":
    main()
