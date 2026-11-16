#!/usr/bin/env python3
"""Check the workspace's coordinated release contract without publishing."""

import argparse
import graphlib
import re
import tomllib
from pathlib import Path
from typing import NotRequired, TypedDict, cast

Inherited = dict[str, bool]
Dependency = TypedDict("Dependency", {
    "package": str, "version": str, "path": str, "default-features": bool,
}, total=False)
DependencyTable = TypedDict("DependencyTable", {
    "dependencies": dict[str, str | Dependency],
    "build-dependencies": dict[str, str | Dependency],
}, total=False)
Package = TypedDict("Package", {
    "name": str, "version": str | Inherited, "readme": str,
    "license": NotRequired[str | Inherited],
    "repository": NotRequired[str | Inherited],
    "rust-version": NotRequired[str | Inherited],
    "publish": NotRequired[bool | list[str]],
    "description": NotRequired[str], "documentation": NotRequired[str],
})


class Manifest(DependencyTable):
    package: Package
    target: NotRequired[dict[str, DependencyTable]]


class WorkspacePackage(TypedDict):
    version: str


class Workspace(TypedDict):
    package: WorkspacePackage
    members: list[str]

ROOT = Path(__file__).resolve().parents[2]
VERSION = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"


def release_notes(changelog: str, version: str) -> str:
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## |\Z)",
        changelog,
        re.MULTILINE | re.DOTALL,
    )
    if not match or not match[1].strip():
        raise ValueError(f"CHANGELOG.md needs nonempty notes for {version}")
    return match[1].strip() + "\n"


def validate_version(version: str, tag: str | None) -> None:
    if not re.fullmatch(VERSION, version):
        raise ValueError("Only stable x.y.z workspace versions are supported")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match workspace version v{version}")


def validate_package(package: Package, member: str) -> None:
    if package["version"] != {"workspace": True}:
        raise ValueError(f"{member}: version must inherit workspace.package.version")
    for field in ("license", "repository", "rust-version"):
        if package.get(field) != {"workspace": True}:
            raise ValueError(f"{member}: {field} must inherit workspace metadata")
    if package.get("publish") is False:
        raise ValueError(f"{member}: every release member must be publishable")
    if not package.get("description") or not package.get("documentation"):
        raise ValueError(f"{member}: missing package description or documentation")


def validate(root: Path, tag: str | None = None) -> tuple[str, str]:
    workspace = cast(Workspace, tomllib.loads((root / "Cargo.toml").read_text())["workspace"])
    version = workspace["package"]["version"]
    validate_version(version, tag)
    manifests: dict[str, tuple[Path, Manifest]] = {}
    for member in workspace["members"]:
        path = root / member
        manifest = cast(Manifest, tomllib.loads((path / "Cargo.toml").read_text()))
        package = manifest["package"]
        validate_package(package, member)
        for filename in (package["readme"], "LICENSE-MIT", "LICENSE-APACHE"):
            if not (path / filename).is_file():
                raise ValueError(f"{member}: missing {filename}")
        manifests[package["name"]] = (path, manifest)
    graph: dict[str, set[str]] = {name: set() for name in manifests}
    for name, (path, manifest) in manifests.items():
        tables: list[DependencyTable] = [manifest, *manifest.get("target", {}).values()]
        for table in tables:
            for kind, dependencies in (
                ("dependencies", table.get("dependencies", {})),
                ("build-dependencies", table.get("build-dependencies", {})),
            ):
                for alias, dep in dependencies.items():
                    actual = dep.get("package", alias) if isinstance(dep, dict) else alias
                    if not isinstance(dep, dict) or actual not in manifests:
                        raise ValueError(f"{name}: external {kind} dependency {alias}")
                    if dep.get("version") != version:
                        raise ValueError(f"{name}: {alias} must require {version}")
                    if (path / dep.get("path", "")).resolve() != manifests[actual][0].resolve():
                        raise ValueError(f"{name}: {alias} has an incorrect workspace path")
                    if dep.get("default-features") is not False:
                        raise ValueError(f"{name}: {alias} must disable default features")
                    graph[name].add(actual)
    tuple(graphlib.TopologicalSorter(graph).static_order())
    notes = release_notes((root / "CHANGELOG.md").read_text(), version)
    return version, notes


class Options(argparse.Namespace):
    tag: str | None
    notes_file: Path | None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--notes-file", type=Path)
    args = parser.parse_args(namespace=Options())
    try:
        version, notes = validate(ROOT, args.tag)
    except (ValueError, graphlib.CycleError) as error:
        parser.exit(1, f"Release validation failed: {error}\n")
    if args.notes_file:
        args.notes_file.write_text(notes)
    print(f"Release metadata valid: {version}")


if __name__ == "__main__":
    main()
