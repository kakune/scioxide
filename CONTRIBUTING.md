# Contributing

Use `main` as the integration branch and open a pull request from a short-lived branch such as `feat/sparse-solve`, `fix/fft-plan` or `docs/handbook`.
Keep `main` releasable and squash merge after `CI success` passes against the latest `main` and review conversations are resolved.
Delete the work branch after merging.
Use descriptive commit and PR titles; no particular commit-message parser is required by the release workflow.

All 19 crates use the same version and ship together.
There is no permanent `develop` branch or release branch.
A release preparation PR updates the workspace version, internal normal/build dependency versions, changelog and versioned documentation links, and is merged into `main` before tagging.
During `0.x`, breaking changes increment the minor version; compatible additions and fixes increment the patch version.
An MSRV increase requires a new minor version before 1.0 and must be documented in the release notes.
For now only the latest release line is maintained; fixes go through `main` and a new release.
If multiple supported lines become necessary, explicitly add maintenance branches and adapt release ancestry validation before using them.

Follow the [development contracts](README.md#development) and [handbook authoring guide](docs/handbook/README.md).
Run tests for changed crates, their relevant consumers and their doctests, including `--no-default-features` where applicable.
CI retains the existing numerical, architecture, feature-isolation, MSRV and Miri checks and adds packaging, rustdoc, workflow linting, a dependency audit and handbook validation.
The handbook build uses the production Pages base path so a root-only build cannot hide broken deployment links.
The final `CI success` job fails when any required job fails, is cancelled or is skipped.
Do not add path filters to the CI trigger that could leave `CI success` pending.

Keep changes to workflows and publishing credentials subject to careful review.

## Python maintenance scripts

Use Python 3.11 or later.
Scripts are grouped into `scripts/checks/`, `scripts/generators/` and `scripts/release/`, with regression tests in `scripts/tests/`.
Run commands from the repository root; for example, `python3 scripts/checks/licenses.py` or `python3 scripts/generators/programming_fixtures.py --check`.

Keep transformations independent of file access, subprocesses and network requests.
Return new values instead of modifying arguments; local accumulation inside a function is fine.
Annotate every function and describe structured records with `TypedDict`.
Validate external inputs at the I/O boundary.

Run the complete checks with the pinned development dependencies:

```sh
uv run --no-project --with-requirements scripts/requirements-dev.txt python -m unittest discover -s scripts/tests -v
uv run scripts/generators/mgh_models.py --check
```
