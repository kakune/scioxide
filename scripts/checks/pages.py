#!/usr/bin/env python3
"""Validate the files included by actions/upload-pages-artifact."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 1_000_000_000  # GitHub Pages' published-site limit, not compressed upload size.


def site_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        # The pinned upload-pages-artifact action excludes hidden path components.
        # Oxiquill may leave hidden staging directories after an interrupted build.
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Pages artifacts must not contain symbolic links: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def main() -> None:
    directory = ROOT / "docs/handbook/dist"
    if not (directory / "index.html").is_file():
        raise SystemExit("Missing handbook dist/index.html; run the production build first")
    total = site_size(directory)
    if total > MAX_BYTES:
        raise SystemExit(f"Site is {total:,} bytes; GitHub Pages allows {MAX_BYTES:,} bytes")
    print(f"GitHub Pages artifact: {total:,} / {MAX_BYTES:,} bytes")


if __name__ == "__main__":
    main()
