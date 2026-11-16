"""Adding or removing crates must not silently lose CI coverage."""

import unittest
from unittest.mock import patch

from scripts.checks import ci_matrix


class TestMatrix(unittest.TestCase):
    def test_members_appear_once(self) -> None:
        with patch.dict(ci_matrix.SHARDS, {"first": ["a"], "second": ["b"]}, clear=True):
            self.assertEqual(ci_matrix.matrix({"scioxide-a", "scioxide-b"}), [
                {"name": "first", "crates": "scioxide-a"},
                {"name": "second", "crates": "scioxide-b"},
            ])

    def test_membership_changes_and_duplicates_fail(self) -> None:
        with patch.dict(ci_matrix.SHARDS, {"first": ["a"]}, clear=True):
            cases: tuple[set[str], ...] = (set(), {"scioxide-a", "scioxide-b"})
            for members in cases:
                with self.assertRaises(ValueError):
                    ci_matrix.matrix(members)
        with patch.dict(ci_matrix.SHARDS, {"first": ["a", "a"]}, clear=True):
            with self.assertRaises(ValueError):
                ci_matrix.matrix({"scioxide-a"})


if __name__ == "__main__":
    unittest.main()
