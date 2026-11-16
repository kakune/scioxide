import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.checks import licenses as checks

ROOT = Path(__file__).resolve().parents[2]


class LicenseChecks(unittest.TestCase):
    def test_dop853_binary_redistribution_clause_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in checks.REQUIRED:
                dest = root / "crates" / name / "THIRD_PARTY_NOTICES.md"
                dest.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / "crates" / name / dest.name, dest)
            notice = root / "crates/scioxide-ode/THIRD_PARTY_NOTICES.md"
            notice.write_text(notice.read_text().replace(
                "documentation and/or other materials provided with the distribution.", ""))
            with self.assertRaisesRegex(ValueError, "scioxide-ode"):
                checks.validate_notices(root)

    def test_built_site_rejects_stale_notice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            dest = site / "scioxide-licenses"
            dest.mkdir()
            sources = [ROOT / "DATA_PROVENANCE.md", *sorted((ROOT / "crates").glob("*/THIRD_PARTY_NOTICES.md"))]
            for name in ("LICENSE-MIT", "LICENSE-APACHE"):
                shutil.copyfile(ROOT / name, dest / (name + ".txt"))
                sources.append(ROOT / name)
            complete = "\n".join(source.read_text() for source in sources)
            (dest / "NOTICE.txt").write_text(complete)
            runtime = site / "oxiquill/licenses/THIRD_PARTY_LICENSES.txt"
            runtime.parent.mkdir(parents=True)
            runtime.write_text("Runtime notice fixture\n")
            checks.validate_site(ROOT, site)
            ode = (ROOT / "crates/scioxide-ode/THIRD_PARTY_NOTICES.md").read_text()
            (dest / "NOTICE.txt").write_text(complete.replace(ode, ""))
            with self.assertRaisesRegex(ValueError, "scioxide-ode"):
                checks.validate_site(ROOT, site)

    def test_legacy_fixture_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "reference.tsv"
            fixture.write_text("# independent reference; GPL-3.0-only\n1\t2\n")
            with self.assertRaisesRegex(ValueError, "unresolved copyleft"):
                checks.validate_data_label(fixture)
            fixture.write_text("# SPDX-License-Identifier: MIT OR Apache-2.0\n1\t2\n")
            checks.validate_data_label(fixture)

    def test_removing_separate_cephes_notice_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in checks.REQUIRED:
                dest = root / "crates" / name / "THIRD_PARTY_NOTICES.md"
                dest.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / "crates" / name / dest.name, dest)
            checks.validate_notices(root)
            notice = root / "crates/scioxide-special/THIRD_PARTY_NOTICES.md"
            text = notice.read_text().split("## Cephes original redistribution notice")[0]
            notice.write_text(text)
            with self.assertRaisesRegex(ValueError, "Cephes Math Library"):
                checks.validate_notices(root)


if __name__ == "__main__":
    unittest.main()
