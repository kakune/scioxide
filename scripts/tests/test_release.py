"""Regression checks for publication identity and partial-release recovery."""

import hashlib
import datetime
import io
import json
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from email.message import Message
from unittest.mock import patch
from pathlib import Path

from scripts.checks import release as check
from scripts.release import publish

SCRIPTS = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_workspace_and_tag_agree(self) -> None:
        version, notes = check.validate(SCRIPTS.parent)
        self.assertEqual(check.validate(SCRIPTS.parent, f"v{version}"), (version, notes))
        with self.assertRaisesRegex(ValueError, "does not match"):
            check.validate(SCRIPTS.parent, "v9.9.9")

    def test_notes_do_not_include_other_releases(self) -> None:
        changelog = "# Changes\n## [Unreleased]\nFuture\n## [0.1.1]\nFix\n## [0.1.0]\nInitial\n"
        self.assertEqual(check.release_notes(changelog, "0.1.1"), "Fix\n")
        self.assertEqual(check.release_notes(changelog, "0.1.0"), "Initial\n")
        with self.assertRaises(ValueError):
            check.release_notes(changelog, "0.2.0")

    def test_resume_only_accepts_identical_unyanked_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "test.crate"
            artifact.write_bytes(b"release archive")
            checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
            entry: publish.RegistryEntry = {"vers": "0.1.0", "yanked": False, "cksum": checksum}
            self.assertFalse(publish.already_published("test", checksum, None))
            self.assertTrue(publish.already_published("test", checksum, entry))
            with self.assertRaisesRegex(ValueError, "differs"):
                publish.already_published("test", checksum, {**entry, "cksum": "wrong"})
            with self.assertRaisesRegex(ValueError, "yanked"):
                publish.already_published("test", checksum, {**entry, "yanked": True})

    def test_workspace_packaging_archive_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_dir = Path(directory) / "package" / "tmp-crate"
            archive_dir.mkdir(parents=True)
            (archive_dir / "test-0.1.0.crate").write_bytes(b"archive")
            self.assertEqual(publish.archive_checksum(directory, "test", "0.1.0"),
                             hashlib.sha256(b"archive").hexdigest())
            with self.assertRaisesRegex(ValueError, "Missing archive"):
                publish.archive_checksum(directory, "test", "0.2.0")

    def test_bootstrap_respects_the_registry_retry_deadline(self) -> None:
        now = datetime.datetime(2026, 11, 16, 12, 0, tzinfo=datetime.timezone.utc).timestamp()
        message = 'status 429 Too Many Requests: Please try again after Mon, 16 Nov 2026 12:10:00 GMT'
        self.assertEqual(publish.retry_after_delay(message, now), 605)
        self.assertIsNone(publish.retry_after_delay('status 403 Forbidden', now))
        self.assertIsNone(publish.retry_after_delay('status 429 with an unknown deadline', now))

    def test_partial_bootstrap_retries_only_the_unpublished_package(self) -> None:
        names = ['scioxide-core', 'scioxide-leaf']
        metadata = {
            'packages': [{'name': name, 'id': name, 'version': '0.1.0'} for name in names],
            'workspace_members': names,
            'target_directory': '/unused',
        }
        published: set[str] = set()
        selections: list[tuple[str, ...]] = []

        def registry(name: str, version: str) -> publish.RegistryEntry | None:
            return {'vers': version, 'cksum': 'digest', 'yanked': False} if name in published else None

        def cargo(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            selections.append(tuple(command[i + 1] for i, arg in enumerate(command) if arg == '--package'))
            if not published:
                published.add('scioxide-core')
                return subprocess.CompletedProcess(command, 101, stdout='status 429')
            published.add('scioxide-leaf')
            return subprocess.CompletedProcess(command, 0, stdout='published\n')

        with (
            patch('sys.argv', ['publish.py', '--execute', '--bootstrap']),
            patch.object(subprocess, 'check_output', return_value=json.dumps(metadata).encode()),
            patch.object(subprocess, 'run', side_effect=cargo),
            patch.object(publish, 'archive_checksum', return_value='digest'),
            patch.object(publish, 'registry_version', side_effect=registry),
            patch.object(publish, 'retry_after_delay', return_value=0),
            redirect_stdout(io.StringIO()),
        ):
            publish.main()
        self.assertEqual(selections, [('scioxide-core', 'scioxide-leaf'), ('scioxide-leaf',)])
        self.assertEqual(published, set(names))

    def test_registry_outage_is_not_treated_as_an_unpublished_crate(self) -> None:
        with patch.object(urllib.request, "urlopen") as request:
            request.side_effect = urllib.error.HTTPError("url", 404, "missing", Message(), None)
            self.assertIsNone(publish.registry_version("test", "0.1.0"))
            request.side_effect = urllib.error.HTTPError("url", 503, "outage", Message(), None)
            with self.assertRaises(urllib.error.HTTPError):
                publish.registry_version("test", "0.1.0")

    def test_malformed_external_records_are_rejected(self) -> None:
        for invalid in (None, {}, {"vers": "0.1.0", "yanked": "false", "cksum": "digest"}):
            with self.subTest(entry=invalid), self.assertRaises(ValueError):
                publish.parse_registry_entry(invalid)
        with self.assertRaises(ValueError):
            publish.parse_metadata({"packages": [], "workspace_members": [123], "target_directory": "/tmp"})

    def test_remaining_packages_uses_only_the_supplied_snapshot(self) -> None:
        packages: list[publish.Package] = [
            {"id": name, "name": name, "version": "0.1.0"} for name in ("first", "second")
        ]
        entries: dict[str, publish.RegistryEntry | None] = {
            "first": {"vers": "0.1.0", "cksum": "digest", "yanked": False}, "second": None,
        }
        with patch.object(publish, "registry_version", side_effect=AssertionError("unexpected I/O")):
            self.assertEqual(publish.remaining_packages(
                packages, {"first": "digest", "second": "digest"}, entries,
            ), ["second"])



if __name__ == "__main__":
    unittest.main()
