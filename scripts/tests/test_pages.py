"""Validate the file selection used for Pages artifacts."""

import tempfile
import unittest
from pathlib import Path

from scripts.checks import pages


class PagesTests(unittest.TestCase):
    def test_pages_size_excludes_staging_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_bytes(b"page")
            staging = root / ".staging"
            staging.mkdir()
            (staging / "unused.wasm").write_bytes(b"not served")
            self.assertEqual(pages.site_size(root), 4)
            (root / "linked.html").symlink_to(root / "index.html")
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                pages.site_size(root)


if __name__ == "__main__":
    unittest.main()
