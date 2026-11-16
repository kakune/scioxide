#!/usr/bin/env python3
"""Check notice retention and known data-license boundaries; not a legal determination."""

import argparse
import re
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast


class Workspace(TypedDict):
    members: list[str]


class Package(TypedDict):
    name: str


class Manifest(TypedDict):
    package: Package

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "scioxide-ode": ["Copyright (c) 2004, Ernst Hairer", "Redistribution and use in source and binary forms", "documentation and/or other materials provided with the distribution.", "SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."],
    "scioxide-core": ["Copyright (c) 2018 Jorge Aparicio", "Copyright (c) 2005 David Schultz", "Developed at SunSoft", "Developed at SunPro"],
    "scioxide-fft": ["Max-Planck-Society", "Matteo Frigo"],
    "scioxide-integrate": ["Zydrunas Gimbutas", "Robert Parrish"],
    "scioxide-linalg": ["University of Tennessee", "University of California Berkeley"],
    "scioxide-optimize": ["Johan Larsson", "Zaikun ZHANG", "Nick Gould", "MINPACK", "Minpack Copyright Notice"],
    "scioxide-random": ["Frances Y. Kuo and Stephen Joe", "D. E. Shaw Research"],
    "scioxide-rootfind": ["Minpack Copyright Notice", "SciPy"],
    "scioxide-signal": ["Filip Wasilewski", "PyWavelets Developers"],
    "scioxide-special": ["Cephes Math Library Release 2.8", "Copyright 1984, 1995, 2000", "Xiaogang Zhang", "Boost Software License", "Massachusetts Institute of Technology", "mpmath contributors"],
}


def validate_notice(text: str, required: Sequence[str], label: str) -> None:
    for phrase in required:
        if phrase not in text:
            raise ValueError(f"{label}: required upstream notice missing: {phrase}")


def validate_notices(root: Path) -> None:
    for name, required in REQUIRED.items():
        notice = root / "crates" / name / "THIRD_PARTY_NOTICES.md"
        validate_notice(notice.read_text(), required, str(notice))


def validate_data_header(header: str, label: str) -> None:
    if re.search(r"\b(?:A?GPL|LGPL)-[23]", header):
        raise ValueError(f"{label}: unresolved copyleft label on distributed reference data")


def validate_data_label(path: Path) -> None:
    try:
        header = path.read_text()[:2048]
    except UnicodeDecodeError:
        return
    validate_data_header(header, str(path))


def validate_site(root: Path, site: Path) -> None:
    """Inspect the built artifact, without regenerating or repairing its notices."""
    notice = (site / "scioxide-licenses/NOTICE.txt").read_text()
    sources = [root / "DATA_PROVENANCE.md", *sorted((root / "crates").glob("*/THIRD_PARTY_NOTICES.md"))]
    for name in ("LICENSE-MIT", "LICENSE-APACHE"):
        source = root / name
        if (site / "scioxide-licenses" / (name + ".txt")).read_bytes() != source.read_bytes():
            raise ValueError(f"{site}: stale or incomplete {name}")
        sources.append(source)
    for source in sources:
        if source.read_text() not in notice:
            raise ValueError(f"{site}: missing or stale notice: {source.relative_to(root)}")
    runtime = site / "oxiquill/licenses/THIRD_PARTY_LICENSES.txt"
    if not runtime.read_text().strip():
        raise ValueError(f"{runtime}: empty runtime notice")


def validate(root: Path, packages: bool = False) -> int:
    validate_notices(root)
    workspace = cast(Workspace, tomllib.loads((root / "Cargo.toml").read_text())["workspace"])
    for member in workspace["members"]:
        crate = root / member
        manifest = cast(Manifest, tomllib.loads((crate / "Cargo.toml").read_text()))
        name = manifest["package"]["name"]
        for filename in ["LICENSE-MIT", "LICENSE-APACHE"]:
            if (crate / filename).read_bytes() != (root / filename).read_bytes():
                raise ValueError(f"{member}: inconsistent {filename}")
        for directory in [crate / "tests/data", crate / "tests/fixtures"]:
            for file in directory.rglob("*"):
                if file.is_file():
                    validate_data_label(file)
        if packages:
            result = subprocess.run(
                ["cargo", "package", "-p", name, "--list", "--offline", "--allow-dirty"],
                cwd=root, check=True, text=True, capture_output=True,
            )
            included = set(result.stdout.splitlines())
            for notice in crate.rglob("*"):
                if notice.is_file() and (notice.name.startswith("LICENSE") or notice.name == "THIRD_PARTY_NOTICES.md"):
                    if notice.relative_to(crate).as_posix() not in included:
                        raise ValueError(f"{member}: {notice.name} excluded from package")
    subprocess.run([sys.executable, "scripts/generators/programming_fixtures.py", "--check"], cwd=root, check=True)
    subprocess.run([sys.executable, "scripts/generators/regression_quartet.py", "--check"], cwd=root, check=True)
    return len(workspace["members"])


class Options(argparse.Namespace):
    packages: bool
    site_dir: Path | None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", action="store_true", help="Also inspect Cargo's packaged file lists offline")
    parser.add_argument("--site-dir", type=Path, help="Also verify notices in an already built site")
    args = parser.parse_args(namespace=Options())
    try:
        count = validate(ROOT, args.packages)
        if args.site_dir is not None:
            validate_site(ROOT, args.site_dir)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"License notice check failed: {error}\n")
    print(f"License notices and data boundaries checked for {count} crates")


if __name__ == "__main__":
    main()
