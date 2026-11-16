"""Compare regenerated formulas with the separately recorded numerical oracle.

Run with the generator's pinned environment:
uv run --with sympy==1.14.0 --with mpmath==1.3.0 python -m unittest discover -s scripts/tests -p test_mgh_models.py
"""

import json
import copy
import math
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, TypedDict, cast

from scripts.generators import mgh_models as generator

ROOT = Path(__file__).resolve().parents[2]


class OraclePoint(TypedDict):
    x: list[float]
    f: float | None


class OracleResult(TypedDict):
    success: bool
    outcome: OraclePoint


class OracleCase(TypedDict):
    model: int
    factor: int
    initial: OraclePoint
    references: list[OracleResult]
    bounded: OracleResult


class OracleDeck(TypedDict):
    models: list[generator.Model]
    cases: list[OracleCase]


def evaluate(tokens: Sequence[generator.Token], x: Sequence[float]) -> float:
    stack: list[float] = []
    unary: dict[str, Callable[[float], float]] = {
        "exp": math.exp, "ln": math.log, "sin": math.sin, "cos": math.cos, "abs": abs,
    }
    for token in tokens:
        if isinstance(token, (int, float)):
            stack.append(token)
        elif token.startswith("x:"):
            stack.append(x[int(token[2:])])
        elif token.startswith(("powi:", "powf:")):
            power = int(token[5:]) if token.startswith("powi:") else float(token[5:])
            stack.append(stack.pop()**power)
        elif token in unary:
            stack.append(unary[token](stack.pop()))
        elif token == "select":
            no, yes, condition = stack.pop(), stack.pop(), stack.pop()
            stack.append(yes if condition else no)
        else:
            b, a = stack.pop(), stack.pop()
            operations: dict[str, Callable[[], float]] = {"+": lambda: a+b, "*": lambda: a*b, "pow": lambda: a**b,
                          "atan2": lambda: math.atan2(a, b), "<": lambda: a < b,
                          "and": lambda: bool(a and b)}
            stack.append(operations[token]())
    if len(stack) != 1:
        raise ValueError("Invalid RPN stack")
    return stack[0]


class MghGeneration(unittest.TestCase):
    generated: ClassVar[list[generator.Model]]
    deck: ClassVar[OracleDeck]

    @classmethod
    def setUpClass(cls) -> None:
        cls.generated = generator.generate_models()
        cls.deck = cast(OracleDeck, json.loads(generator.FIXTURE.read_text()))

    def test_every_executable_definition_is_reproducible(self) -> None:
        self.assertEqual(len(self.generated), 52)
        for generated, recorded in zip(self.generated, self.deck["models"]):
            self.assertEqual(generated, {
                "family": recorded["family"], "problem": recorded["problem"],
                "n": recorded["n"], "m": recorded["m"], "residuals": recorded["residuals"],
            })

    def test_generator_does_not_read_the_fixture(self) -> None:
        previous = generator.FIXTURE
        try:
            generator.FIXTURE = ROOT / "nonexistent-mgh-fixture.json"
            self.assertEqual(generator.generate_models(), self.generated)
        finally:
            generator.FIXTURE = previous

    def test_updating_definitions_preserves_input_and_historical_records(self) -> None:
        provenance: generator.GenerationRecord = {
            "script": "test", "dependencies": {}, "scope": "residuals",
            "definitions_sha256": "test", "record": "test",
        }
        models = self.generated[:2]
        deck: dict[str, object] = {
            "models": [{**model, "published_minima": [42.0]} for model in models],
            "cases": [{"historical": "retained"}],
        }
        original = copy.deepcopy(deck)
        updated = generator.update_deck(deck, models, provenance)
        self.assertEqual(deck, original)
        self.assertEqual(updated["models"], deck["models"])
        self.assertEqual(updated["cases"], deck["cases"])
        self.assertEqual(updated["model_generation"], provenance)
        # A late metadata mismatch must not leave earlier models partly updated.
        invalid: dict[str, object] = {
            **deck, "models": [{**models[0], "residuals": []}, {**models[1], "n": -1}],
        }
        original_invalid = copy.deepcopy(invalid)
        with self.assertRaisesRegex(ValueError, "unexpected n"):
            generator.update_deck(invalid, models, provenance)
        self.assertEqual(invalid, original_invalid)

    def test_against_historical_reference_objectives(self) -> None:
        self.assertEqual(len(self.deck["cases"]), 156)
        checked = 0
        for case in self.deck["cases"]:
            residuals = self.generated[case["model"]]["residuals"]
            points = [case["initial"]] + [r["outcome"] for r in case["references"]+[case["bounded"]] if r["success"]]
            for point in points:
                if point["f"] is None:
                    continue  # Nonfinite oracle records are not finite-value assertions.
                with self.subTest(model=case["model"], factor=case["factor"], x=point["x"]):
                    actual = math.fsum(evaluate(r, point["x"])**2 for r in residuals)
                    self.assertTrue(math.isfinite(actual))
                    self.assertLessEqual(abs(actual-point["f"]), 2e-8*(1+abs(point["f"])))
                    checked += 1
        self.assertGreater(checked, 500)


if __name__ == "__main__":
    unittest.main()
