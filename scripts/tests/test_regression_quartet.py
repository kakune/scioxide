"""Check the actual distributed coordinates with exact binary-rational sums."""

import ast
from fractions import Fraction
import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_array(text: str, name: str) -> object:
    match = re.search(rf"const {name}: .*?= (\[.*?\]);", text, re.S)
    if match is None:
        raise ValueError(f"Missing array: {name}")
    return ast.literal_eval(match.group(1))


def float_array(value: object) -> list[float]:
    if not isinstance(value, list) or not all(isinstance(v, (int, float)) for v in value):
        raise ValueError("Expected a numeric array")
    return [float(v) for v in value]


def read_arrays(locale: str) -> tuple[list[float], list[float], list[list[float]]]:
    path = ROOT / f"docs/handbook/content/docs/{locale}/scioxide-stats/examples/anscombe-quartet.mdx"
    text = path.read_text()
    ys = read_array(text, "Y")
    if not isinstance(ys, list):
        raise ValueError("Expected a response matrix")
    return (float_array(read_array(text, "X_SHARED")),
            float_array(read_array(text, "X_FOURTH")), [float_array(y) for y in ys])


class RegressionQuartet(unittest.TestCase):
    def test_locales_ship_the_same_original_coordinates(self) -> None:
        en, ja = read_arrays("en"), read_arrays("ja")
        self.assertEqual(en, ja)
        self.assertEqual(len(en[0]), 13)
        self.assertEqual(len(en[1]), 13)
        self.assertTrue(all(len(row) == 13 for row in en[2]))

    def test_summary_identities_of_distributed_binary64_values(self) -> None:
        shared, fourth, ys = read_arrays("en")
        for xs, row in zip([shared]*3+[fourth], ys):
            x, y = list(map(Fraction, xs)), list(map(Fraction, row))
            n = len(x)
            mx, my = sum(x)/n, sum(y)/n
            sxx = sum((v-mx)**2 for v in x)
            sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
            syy = sum((v-my)**2 for v in y)
            slope = sxy/sxx
            intercept = my-slope*mx
            rss = sum((b-intercept-slope*a)**2 for a, b in zip(x, y))
            for actual, expected in [(mx, 0), (my, 3), (sxx, 182), (syy, 91),
                                     (slope, .5), (intercept, 3), (rss, 45.5)]:
                self.assertAlmostEqual(float(actual), expected, delta=1e-12)

    def test_response_outlier_and_leverage_one_construction(self) -> None:
        shared, fourth, ys = read_arrays("en")
        # The other 12 points are collinear; the selected point is separated.
        slope = (ys[2][1]-ys[2][0])/(shared[1]-shared[0])
        intercept = ys[2][0]-slope*shared[0]
        for i, (x, y) in enumerate(zip(shared, ys[2])):
            distance = y-intercept-slope*x
            if i == 10:
                self.assertGreater(distance, 7)
            else:
                self.assertAlmostEqual(distance, 0, delta=2e-15)
        self.assertEqual(len(set(fourth[:-1])), 1)
        self.assertTrue(math.isclose(1/13+fourth[-1]**2/182, 1, abs_tol=1e-15))


if __name__ == "__main__":
    unittest.main()
