"""Guard the non-mutating transformations used by fixture generation."""

import copy
import unittest

from scripts.generators import programming_fixtures as generator


class ProgrammingFixtures(unittest.TestCase):
    def test_adding_a_reference_leaves_the_problem_untouched(self) -> None:
        problem = generator.case("one", [[1.0]], [2.0], [0.0], [None], [1.0], [None])
        original = copy.deepcopy(problem)
        result = generator.reference(problem, [1.0], "AnalyticKKT")
        self.assertEqual(problem, original)
        self.assertEqual(result["minimum"], 2.0)
        self.assertEqual(result["reference"]["primal"], [1.0])

    def test_quadratic_conversion_preserves_the_linear_reference(self) -> None:
        linear = generator.linear(3, 0)
        original = copy.deepcopy(linear)
        quadratic = generator.quadratic(linear)
        self.assertEqual(linear, original)
        self.assertNotEqual(quadratic["objective"], linear["objective"])
        self.assertEqual(quadratic["reference"]["primal"], linear["reference"]["primal"])
        self.assertEqual(generator.quadratic(linear), quadratic)


if __name__ == "__main__":
    unittest.main()
