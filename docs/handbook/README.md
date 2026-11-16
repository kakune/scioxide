# SciOxide Handbook

The handbook teaches numerical methods and SciOxide APIs in English and Japanese, assuming basic Rust, calculus and linear algebra.
It uses [Oxiquill](https://github.com/kakune/oxiquill), pinned in [package.json](package.json), to publish MDX pages with executable Rust examples.
Keep both languages consistent; English is the default, and missing translations are omitted from navigation.

## Development

Run these commands from the repository root.
The Nix shell supplies the workspace Rust toolchain, Wasm target, `wasm-opt`, Node and pnpm.

```sh
nix develop .#handbook
pnpm --dir docs/handbook install --frozen-lockfile
pnpm --dir docs/handbook dev
```

Validate changes before publication:

```sh
pnpm --dir docs/handbook check
pnpm --dir docs/handbook exec oxiquill test-wasm --page ja/scioxide-poly/index.mdx --all-cases
SITE=https://kakune.github.io BASE_PATH=/scioxide pnpm --dir docs/handbook build
python3 scripts/checks/pages.py
python3 scripts/checks/licenses.py --site-dir docs/handbook/dist
```

`check` generates the runtime and checks Astro diagnostics.
For Wasm tests, select the changed content with repeatable `--page` or `--cell` options; `--all-cases` also exercises named cases without declared tests.
Finish with `build`, which checks production routes, anchors and redirects.
Use `preview` to inspect the result.

Keep caches for incremental builds; use `clean`, `--force` or cache deletion only to diagnose a specific cache problem.
Rust cells compile into one Wasm module per locale chapter, while cell dependencies remain local to each page.
The first generation after an Oxiquill update may need network access for matching tools.
To update Oxiquill, change its pinned version, run `pnpm --dir docs/handbook install`, and commit both the manifest and lockfile.

## Writing pages

Crate chapters follow [dependency order](scripts/crate-catalog.mjs) and may assume only earlier chapters.
Put shared numerical contracts in `concepts`, workflows spanning crates in `applications`, and the complete symbol index in `reference`.

| Path | `sidebar.order` | Content |
|---|---|---|
| `index.mdx` | 10 | Purpose, first computation, interpretation, chapter map |
| `choosing.mdx` | 20 | Selection flow and comparison tables; at most one Mermaid diagram per chapter |
| `algorithms/index.mdx` | 300 | Shared contracts, notation and page map, without repeating child summaries |
| `algorithms/*.mdx` | 305–395 | Mathematical explanations; use consistent increments of 5 or 10 |
| `examples/index.mdx` | 400 | Table of subjects, APIs and independent checks; no cells |
| `examples/*.mdx` | 405–495 | Worked examples |
| `api.mdx` or `api/index.mdx` | 500 | API guide; child pages start at 510 in increments of 10 |

Each directory's index must have its lowest order, which determines the group's position.
Aim for 120–280 lines per page and one to three cells of roughly 80 lines each, without compressing readable source to meet a target.
Split longer material by topic or experiment when useful.
Write one sentence per line, separate paragraphs and steps with blank lines, and end each page with a three-to-six-sentence summary and suggested next pages.
Format Rust cell bodies with the repository's rustfmt rules and long `json!` arguments with one key per line.

Explain assumptions, derivations, conditioning, error, computational cost and termination.
Distinguish precision, tolerances, estimates, certified bounds and statistical uncertainty, and state the implementation's supported types, domains and environments.
Keep repository procedures out of published explanations.

## API coverage and links

Treat the current Rust source and Cargo configuration as authoritative.
Introduce each crate's purpose, entry points, type relationships and a minimal example before explaining results, errors and diagnostics.
Group API tables by purpose, covering every public item and re-export, including the core backend namespace.
Each row needs the item's kind, role, error and allocation contract, an explanatory section, and an example that actually uses it.
The master index links to these groups without repeating their contracts.

Compare coverage against fresh rustdoc `all.html` from the documented release with all features enabled, including BLAS.
Link definitions to docs.rs version `0.1.0`, verify module paths and method anchors using the release's docs.rs feature settings, and use identical targets in both languages.
API changes also require updates to selection guides, algorithms, examples and the master index.

Use relative page links so they work under `/scioxide`.
Give cells a `ref` matching their `id`, name headings in `references` frontmatter, and link with `oxref:`.
For page moves, add redirects to `handbookRedirects` in [astro.config.mjs](astro.config.mjs) in the same commit.
Both routes must be base-relative with trailing slashes, and the old route must not belong to an existing page in either locale.
Heading anchors cannot be redirected; update affected links when changing them.

## Executable examples

Edit Rust in MDX, never in generated runtime files.
Every Rust fence needs a cell ID, title, run mode and explicit Cargo package list.
Use public APIs and the solver described in the text, with concise comments in the page's language and errors converted to strings at the cell boundary.
Explain relevant plan, workspace and provider lifetimes, partial results and unsupported environments.
Never report non-convergence as success or substitute zero for an unknown value.

Share models through `kind: definitions` and `uses`.
Share computed results through `exports` and `dependsOn`, exporting only finite values or null.
Use `run: button` and `cache: inputs` for expensive producers.

Use structured outputs:

- Keep table values numeric, set formatting with column descriptors, and put units in column labels.
- Label chart axes and units, show enough samples, and provide numerical outputs for independent checks.
  Use logarithmic axes, reference annotations, dashed theoretical curves and `band` or `errorbar` for uncertainty as appropriate.
- Link charts sharing an x quantity with `link.group`; use `outputLayout: columns` for direct comparisons and `emit_scene3d!` with equal axis units for spatial geometry.
- When termination determines the conclusion, emit the outcome last, including its status, residuals and iteration counts.
- Give tested tables and charts unique metadata IDs and select them with `artifact:`.
  Charts use `emit_chart!` to attach metadata; positional `/outputs/N` selectors are reserved for outcomes and text.

Validate defaults, vary every control independently, and check numerical relations against independent references.
Verify native-only arithmetic natively and explain browser limitations.
A successful build alone does not establish numerical correctness.

## Layout and publication

[handbook.css](src/styles/handbook.css) controls typography, output sizes and scrolling.
Keep prose at the normal reading width; use `contentWidth: full` for API guides, selection guides, the master index and other pages dominated by tables.
Set Mermaid's `config.flowchart.useMaxWidth: false` to preserve natural diagram sizes on screen.
Check desktop, mobile, zoomed and print layouts, including math, diagrams, charts, language navigation and runtime errors.

The [route middleware](src/route-middleware.ts) expands only the current chapter's sidebar and links to other chapter overviews to keep the site within GitHub Pages' size limit.
The [deployment workflow](../../.github/workflows/pages.yml) publishes `main` with the production base path used above.
The artifact checks enforce the 1 GB limit, reject symbolic links and verify bundled licenses.

[sync-licenses.mjs](scripts/sync-licenses.mjs) copies root licenses, crate notices and data attribution into `public/scioxide-licenses/` during generation, alongside Oxiquill's separate framework notices.
Edit the source notices and verify retention with `node --test docs/handbook/scripts/sync-licenses.test.mjs` from the repository root.

For substantive revisions, review API coverage, links, non-default controls and numerical correctness separately.
Use temporary review tooling without duplicating Rust cells or restoring retired test infrastructure.
Record the source revision, commands, failures and skipped checks with reasons in the PR description; do not commit review reports or evidence files.
See the [repository development guide](../../README.md#development) for Rust workspace checks.
