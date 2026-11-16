"""Balance isolated crate tests without silently omitting workspace members."""

import json
from pathlib import Path
import tomllib
from typing import Final, TypedDict, cast

ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class TestShard(TypedDict):
    name: str
    crates: str


class Workspace(TypedDict):
    members: list[str]


class WorkspaceManifest(TypedDict):
    workspace: Workspace


class Package(TypedDict):
    name: str


class PackageManifest(TypedDict):
    package: Package

# Keep the expensive numerical suites on separate runners. Each crate still
# gets its own cargo invocation, preserving feature isolation.
SHARDS: Final[dict[str, list[str]]] = {
    "linalg": ["linalg"],
    "optimize": ["optimize"],
    "integration": ["integrate", "differentiate", "validated"],
    "transforms": ["fft", "signal", "poly", "special"],
    "equations": ["ode", "pde", "interp", "rootfind"],
    "foundations": ["core", "ad", "sparse", "spatial", "random", "stats"],
}


def matrix(members: set[str]) -> list[TestShard]:
    assigned = [f"scioxide-{name}" for names in SHARDS.values() for name in names]
    if len(assigned) != len(set(assigned)) or set(assigned) != members:
        raise ValueError("CI shards must cover every workspace package exactly once")
    return [
        {"name": name, "crates": " ".join(f"scioxide-{crate}" for crate in crates)}
        for name, crates in SHARDS.items()
    ]


def main() -> None:
    workspace = cast(
        WorkspaceManifest, tomllib.loads((ROOT / "Cargo.toml").read_text())
    )["workspace"]
    members = {
        cast(
            PackageManifest,
            tomllib.loads((ROOT / member / "Cargo.toml").read_text()),
        )["package"]["name"]
        for member in workspace["members"]
    }
    print(json.dumps(matrix(members), separators=(",", ":")))


if __name__ == "__main__":
    main()
