#!/usr/bin/env python3
"""Resume a verified workspace publication, refusing mismatched existing crates.

Run cargo publish --locked --workspace --registry crates-io --dry-run first.
Without --execute this only reads the registry and prints the remaining crates.
"""

import argparse
import email.utils
import hashlib
import json
import math
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import TypedDict


class Package(TypedDict):
    id: str
    name: str
    version: str


class Metadata(TypedDict):
    packages: list[Package]
    workspace_members: list[str]
    target_directory: str


class RegistryEntry(TypedDict):
    vers: str
    yanked: bool
    cksum: str

ROOT = Path(__file__).resolve().parents[2]


def parse_metadata(value: object) -> Metadata:
    """Narrow Cargo's JSON to the fields used for selecting release packages."""
    if not isinstance(value, dict):
        raise ValueError("Cargo metadata must be an object")
    packages, members, target = value.get("packages"), value.get("workspace_members"), value.get("target_directory")
    if not isinstance(packages, list) or not isinstance(members, list) or not isinstance(target, str):
        raise ValueError("Cargo metadata needs packages, workspace_members and target_directory")
    member_ids: list[str] = []
    for member in members:
        if not isinstance(member, str):
            raise ValueError("Workspace member IDs must be strings")
        member_ids.append(member)
    parsed: list[Package] = []
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("Cargo packages must be objects")
        package_id, name, version = package.get("id"), package.get("name"), package.get("version")
        if not isinstance(package_id, str) or not isinstance(name, str) or not isinstance(version, str):
            raise ValueError("Cargo packages need string id, name and version fields")
        parsed.append(Package(id=package_id, name=name, version=version))
    return Metadata(packages=parsed, workspace_members=member_ids, target_directory=target)


def parse_registry_entry(value: object) -> RegistryEntry:
    if not isinstance(value, dict):
        raise ValueError("Registry entry must be an object")
    version, yanked, checksum = value.get("vers"), value.get("yanked"), value.get("cksum")
    if not isinstance(version, str) or not isinstance(yanked, bool) or not isinstance(checksum, str):
        raise ValueError("Registry entry needs vers, yanked and cksum fields of the expected types")
    return RegistryEntry(vers=version, yanked=yanked, cksum=checksum)


def registry_version(name: str, version: str) -> RegistryEntry | None:
    # All workspace names are longer than three characters.
    url = f"https://index.crates.io/{name[:2]}/{name[2:4]}/{name}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "scioxide-release (https://github.com/kakune/scioxide)",
            "Cache-Control": "no-cache",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            entries = [parse_registry_entry(json.loads(line)) for line in response if line.strip()]
    except urllib.error.HTTPError as error:
        error.close()
        if error.code == 404:
            return None
        raise
    return next((entry for entry in entries if entry["vers"] == version), None)


def archive_checksum(target_directory: str | Path, name: str, version: str) -> str:
    # Cargo 1.95's workspace packaging writes archives to tmp-crate; single
    # package layouts can place the archive directly under package instead.
    package_dir = Path(target_directory) / "package"
    filename = f"{name}-{version}.crate"
    for directory in (package_dir / "tmp-crate", package_dir):
        artifact = directory / filename
        if artifact.is_file():
            return hashlib.sha256(artifact.read_bytes()).hexdigest()
    raise ValueError(f"Missing archive for {name}; run the workspace publish dry run first")


def already_published(package: str, checksum: str, entry: RegistryEntry | None) -> bool:
    if entry is None:
        return False
    if entry["yanked"]:
        raise ValueError(f"{package}: this version is yanked; investigate before resuming")
    if checksum != entry["cksum"]:
        raise ValueError(f"{package}: published archive differs from this tag; refusing to resume")
    return True


def retry_after_delay(output: str, now: float) -> int | None:
    """Recognize crates.io's HTTP 429 deadline; never retry unrelated failures."""
    if not re.search(r"\b429\b|Too Many Requests", output, re.IGNORECASE):
        return None
    match = re.search(r"Please try again after (.+?(?:GMT|UTC|[+-]\d{4}))", output, re.IGNORECASE)
    if not match:
        return None
    try:
        retry_at = email.utils.parsedate_to_datetime(match[1]).timestamp()
    except (ValueError, TypeError, OverflowError):
        return None
    return math.ceil(max(0, retry_at - now) + 5)


def remaining_packages(
    packages: Sequence[Package], checksums: Mapping[str, str],
    entries: Mapping[str, RegistryEntry | None],
) -> list[str]:
    """Select unpublished crates from a registry snapshot without performing I/O."""
    return [package["name"] for package in packages if not already_published(
        package["name"], checksums[package["name"]],
        entries[package["name"]],
    )]


def read_registry_entries(packages: Sequence[Package]) -> dict[str, RegistryEntry | None]:
    return {package["name"]: registry_version(package["name"], package["version"])
            for package in packages}


class Options(argparse.Namespace):
    execute: bool
    bootstrap: bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Wait for explicit crates.io 429 deadlines during initial token-based publication")
    args = parser.parse_args(namespace=Options())
    metadata = parse_metadata(json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
    )))
    packages = [p for p in metadata["packages"] if p["id"] in metadata["workspace_members"]]
    checksums = {
        package["name"]: archive_checksum(metadata["target_directory"], package["name"], package["version"])
        for package in packages
    }
    remaining = remaining_packages(packages, checksums, read_registry_entries(packages))
    print("Remaining crates: " + (", ".join(remaining) or "none"), flush=True)
    if not args.execute:
        return
    deadline = time.monotonic() + 3 * 60 * 60
    while remaining:
        command = ["cargo", "publish", "--locked", "--registry", "crates-io"]
        for name in remaining:
            command.extend(["--package", name])
        # Preserve Cargo's diagnostics for an actionable failure and, only in
        # bootstrap mode, the registry's explicit next-publish time.
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        print(result.stdout, end="", flush=True)
        if result.returncode == 0:
            break
        delay = retry_after_delay(result.stdout, time.time()) if args.bootstrap else None
        if delay is None or time.monotonic() + delay > deadline:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"crates.io rate limit: waiting {delay} seconds before resuming the same release.", flush=True)
        wake_at = time.monotonic() + delay
        while (left := wake_at - time.monotonic()) > 0:
            time.sleep(min(left, 30))
        remaining = remaining_packages(packages, checksums, read_registry_entries(packages))
        print("Remaining crates: " + (", ".join(remaining) or "none"), flush=True)
    # Do not announce a GitHub Release until every version is visible and matches.
    for package in packages:
        name, version = package["name"], package["version"]
        for attempt in range(6):
            if already_published(name, checksums[name], registry_version(name, version)):
                break
            if attempt == 5:
                raise RuntimeError(f"{name}@{version} is not yet visible; resume the same tag later")
            time.sleep(10)
    print("Every crate is published with the expected checksum.")


if __name__ == "__main__":
    main()
