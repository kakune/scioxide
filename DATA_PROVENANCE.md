# Data provenance and redistribution

SciOxide's original source code, documentation, manufactured test inputs and independently computed reference values are Copyright (c) 2026 kakune and SciOxide contributors, licensed under MIT OR Apache-2.0.
This includes the original reference data under each crate's `tests/data/`, `tests/fixtures/` and test-support modules, subject to the third-party input notices identified below.
This statement does not change a third party's license, withdraw any previously granted license, or claim exclusive rights over mathematical formulas, numerical facts or scientific observations.

Using a reference program to compute numbers does not itself copy that program into the output.
The tool names and versions in fixture headers identify how the references were computed; they do not declare the tool's license to be the license of those numbers.
No MPFR, MPC, FLINT, GSL, SciPy, NumPy, mpmath or ArviZ implementation or compiled generator is bundled with these fixtures.
Copied inputs, coefficient tables and source adaptations retain their own conditions, even when their outputs or surrounding implementation are original.

## Original reference computations

| Crate | Reference data and computation |
|---|---|
| `scioxide-ad` | Analytic derivatives and independently evaluated high-precision derivatives, powers and Taylor coefficients |
| `scioxide-core` | Exact IEEE-input arithmetic and constants; independently computed MPFR/MPC elementary, logarithmic, normalization and boundary references |
| `scioxide-differentiate` | Analytic derivative values and independent finite-difference convention comparisons |
| `scioxide-fft` | Independently computed Fourier, cosine/sine, chirp-z, Hankel, FFTLog and nonuniform transform references |
| `scioxide-integrate` | Analytic integrals, exact polynomial identities and independent high-precision quadrature |
| `scioxide-interp` | Explicit interpolation nodes and values, analytic derivatives/integrals, and independently computed high-precision or SciPy comparisons |
| `scioxide-linalg` | Manufactured matrices; exact rational and high-precision spectra, factorizations, matrix functions and control references |
| `scioxide-ode` | Independently computed trajectories and work counts for mathematical ODE models; literature endpoint comparisons |
| `scioxide-optimize` | Independently encoded mathematical objective/constraint expressions and separately computed reference outcomes; original manufactured programming instances |
| `scioxide-pde` | Manufactured fields, meshes, equations and discrete verification data |
| `scioxide-poly` | Exact algebraic identities, manufactured roots and independently computed polynomial reference values |
| `scioxide-random` | Analytic laws, exact integer/rational engine and QMC checks, and independently computed high-precision distribution values |
| `scioxide-rootfind` | Independently computed scalar-root and square-system references; SciPy APS input cases and mathematical MINPACK problems as noted below |
| `scioxide-signal` | Independently computed filter, spectrum and wavelet comparisons, including numerical outputs of GSL, SciPy and PyWavelets |
| `scioxide-sparse` | Manufactured sparse patterns, graph/ordering cases and independently computed arithmetic references |
| `scioxide-spatial` | Manufactured geometric inputs, exact predicate checks and mathematical geometry references |
| `scioxide-special` | Independently computed mpmath/MPFR/FLINT references, exact identities, and separately evaluated comparison values |
| `scioxide-stats` | Exact Fraction/Decimal statistics, analytic integrals, enumerated resampling and independent distribution/diagnostic comparisons |
| `scioxide-validated` | Independently computed MPFR/FLINT values and enclosures for arithmetic, quadrature, roots and linear certificates |

Source versions, precision, domains and comparison conventions remain in the fixture headers and consuming tests.
An independent reference is a correctness check; it is not a certified error bound unless the specific fixture and test state one.

## Reproducing the MGH problem definitions

`scripts/generators/mgh_models.py` constructs all 52 executable residual models in `crates/scioxide-optimize/tests/data/mgh.json` from explicit mathematical formulas.
It uses the OBJFCN/SSQFCN family numbering already used by the tests; the mathematical reference is Moré, Garbow and Hillstrom (1981), [Testing Unconstrained Optimization Software](https://doi.org/10.1145/355934.355936), section 2.
The script states each problem's equations, dimensions and residual order, then serializes them to the test interpreter's RPN format with SymPy 1.14.0 and mpmath 1.3.0.
It does not download, parse, translate or execute CALGO or MINPACK programs.
The Gaussian observations are recomputed from the normal density; the Bard, Kowalik–Osborne, Meyer and Osborne input observations retain the MINPACK notice.

```bash
uv run scripts/generators/mgh_models.py --check
uv run scripts/generators/mgh_models.py --output /tmp/mgh-models.json
uv run scripts/generators/mgh_models.py --write
```

`--output` constructs standalone definitions without reading the fixture.
`--check` compares every generated residual token and checks the generation record.
`--write` updates only those definitions and their generation record, preserving the published-minimum metadata and the 156 historical reference cases.
The checked definition serialization has SHA256 `75496df77d5679a65bcf0f498df407456874cdae0c2419271eee9e24e6aec905`.
The generator reproduces all 52 residual definitions.
It does not regenerate their solver trajectories, evaluation counts or Fortran checkpoints.
The original reference-tool URLs and hashes remain in `sources`.
The fixture's `model_generation` field points to this script and record.

## Third-party input material

These are exceptions to the original-material license above.
The full notices are shipped inside the owning crate and are included in the handbook's downloadable SciOxide notices.

| Material | Scope and retained terms |
|---|---|
| fdlibm/FreeBSD/musl kernels via libm | `scioxide-core`: MIT, BSD and Sun permission notices in source and `THIRD_PARTY_NOTICES.md` |
| pocketfft transform reductions | `scioxide-fft`: BSD-3-Clause notice in `THIRD_PARTY_NOTICES.md`, including the permitted FFTW-derived odd DCT-IV portion |
| Xiao–Gimbutas simplex and Lebedev–Laikov sphere tables | `scioxide-integrate`: BSD notices in source and `THIRD_PARTY_NOTICES.md` |
| LAPACK SVD/secular adaptations | `scioxide-linalg`: upstream redistribution conditions in `THIRD_PARTY_NOTICES.md` |
| Hairer/Wanner DOP853 coefficient tables | `scioxide-ode`: original Hairer BSD-2-Clause notice in `THIRD_PARTY_NOTICES.md`, covering source and binary redistribution |
| PRIMA through Basin | `scioxide-optimize`: Basin MIT and PRIMA BSD-3-Clause notices in `THIRD_PARTY_NOTICES.md` |
| CUTEst/SIF input definitions | `scioxide-optimize/tests/data/hs.json`: the input material retains the pinned SIF BSD-3-Clause notice; original mathematical encoding and computed outputs retain SciOxide's license |
| MINPACK problem definitions and starts | `scioxide-optimize` and `scioxide-rootfind`: the complete MINPACK notice and acknowledgment are retained in their `THIRD_PARTY_NOTICES.md` |
| SciPy APS input corpus | `scioxide-rootfind/tests/data/aps.tsv`: SciPy BSD-3-Clause notice in `THIRD_PARTY_NOTICES.md` |
| Random123 known-answer vectors | `scioxide-random/tests/data/counter_reference.tsv`: upstream BSD notice in `THIRD_PARTY_NOTICES.md` |
| Joe–Kuo Sobol direction numbers | `scioxide-random/src/quasi/data/`: original BSD notice in `LICENSE` and `THIRD_PARTY_NOTICES.md` |
| PyWavelets filter banks | `scioxide-signal`: MIT notice in `THIRD_PARTY_NOTICES.md` |
| Cephes, Boost.Math, Faddeeva and mpmath source adaptations | `scioxide-special`: the applicable BSD, BSL-1.0 and MIT notices in `THIRD_PARTY_NOTICES.md` |

The optimizer's programming fixtures contain no Netlib LP, MIPLIB, COIN-OR Sample or QAPLIB instance arrays or solution files.
These fixtures use explicit dyadic constructions, analytic KKT optima and complete enumeration of small finite integer/permutation spaces.
The handbook's 12-site assignment example also uses original rank-one matrices with an optimum proved by the rearrangement inequality, rather than a copied QAPLIB matrix or solution file.

## Handbook examples

Original worked-example code and synthetic data use MIT OR Apache-2.0.
Literature examples identify the original scientific observations or mathematical constructions in both languages; numerical facts are not presented as newly owned SciOxide data.
Comparing independently transcribed observations against a software dataset does not grant a license to redistribute that software or its dataset files.
Source adaptations or copied input material must instead retain the corresponding third-party permission and notice.

### Original regression quartet

The English and Japanese `scioxide-stats/examples/anscombe-quartet.mdx` pages use SciOxide's own 13-point constructions, generated by `scripts/generators/regression_quartet.py` with Python's standard library.
The pages do not distribute Anscombe's Table 1 coordinates, a permutation or rescaling of that table, or a dataset downloaded from R or another package.
The arithmetic seeds, projection, normalization and resulting identities are given in the pages and generator.
They produce matching summaries while varying the residual shapes; the generator also produces both locales' numerical assertions.

The rights basis is original implementation and original manufactured inputs under MIT OR Apache-2.0, with Anscombe (1973) credited for the statistical idea.
The distinction between an idea/method and its particular expression is described in [U.S. Copyright Office Circular 33](https://www.copyright.gov/circs/circ33.pdf).
We do not infer permission to reproduce the original table from this general rule, its widespread use, a citation, or the license of software that includes it.

```bash
python3 scripts/generators/regression_quartet.py --check
python3 scripts/generators/regression_quartet.py --write
```

The check verifies zero residual mean, orthogonality to the design, the common residual sum of squares, and exact agreement of the generated source and assertions in both languages.
The generators can be run without building Rust or the handbook.

### Clinical observations

The leukemia example uses 42 remission times with treatment and censoring indicators from Freireich et al. (1963) and Gehan (1965).
Gehan's observations appear on p. 213, section 11, available from the [New Mexico Environment Department archive](https://hwbdocs.env.nm.gov/Los%20Alamos%20National%20Labs/TA%2054/11508.pdf), PDF SHA256 `7275152d12326ae07668a488a1a0ff0d29d7c78cdbf94299c83879d53d3fa8e3`.
The handbook arranges the full two-group sample by group and numerical time, interleaving censored and uncensored observations.
It does not include scanned tables, publisher typography, paper prose, or an R/MASS dataset file.

These observations are used as numerical facts; the original papers are scientific sources, not open-source licenses.
The public availability of a paper or a citation does not itself grant redistribution permission.
Japan's Agency for Cultural Affairs [explains the distinction for facts and data](https://www.bunka.go.jp/seisaku/chosakuken/seidokaisetsu/chosakukensha_fumei/), and the U.S. Copyright Office [distinguishes individual facts and mechanical ordering from original compilations](https://www.copyright.gov/register/tx-databases.html).
Creative selection or arrangement, database rights and separate acquisition terms require their own analysis where applicable.

## Maintaining the boundary

### What the automated checks establish

`python3 scripts/checks/licenses.py --packages` verifies known required notices, local license consistency, selected fixture regeneration, and Cargo package file lists.
`--site-dir docs/handbook/dist` additionally checks the built site's full SciOxide notices against the current source and requires a nonempty framework notice.
CI runs this artifact check before uploading the handbook.
These checks detect missing or stale known notices, not unknown copying or the legal validity of a claimed permission.

### Adding and distributing material

For new third-party material, record its precise source revision, affected files and applicable redistribution terms before adding it.
A citation, a download URL, a package-wide license on unrelated build scripts, or a generator's license is not a substitute for permission covering the material being distributed.
If that permission cannot be established, use an original manufactured fixture rather than copying the material and adding an unsupported license claim.
Do not remove required notices when reducing package contents.
Compiled examples and Wasm distributions also carry the notices required for binary redistribution.
