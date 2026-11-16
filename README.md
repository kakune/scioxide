# SciOxide

SciOxide is a collection of Rust libraries for numerical computing.
It covers linear algebra, special functions, statistics, optimization, differential equations, and related methods.
Use the crates you need; they share scalar types, errors, and numerical tolerances through `scioxide-core`.

[Handbook](https://kakune.github.io/scioxide/) · [API reference](https://docs.rs/scioxide-core) · [Changelog](CHANGELOG.md)

The libraries support `std` and `no_std + alloc`.
Their runtime dependencies are limited to other SciOxide crates.
Checked operations return structured errors, and reusable plans and workspaces support repeated calculations in caller-owned storage.

## Installation

SciOxide requires **Rust 1.93.0 or later**.
Add the crates you need to `Cargo.toml`:

```toml
[dependencies]
scioxide-core = "0.1"
scioxide-poly = "0.1"
```

For `no_std + alloc`, disable default features on each SciOxide dependency:

```toml
scioxide-poly = { version = "0.1", default-features = false }
```

SciOxide is under active development.
Breaking API changes use a new minor version while the crates are at `0.x`.

## Example

Interpolate three samples and evaluate the resulting polynomial:

```rust
use scioxide_core::SciOxideError;
use scioxide_poly::ApproxPolynomial;

fn main() -> Result<(), SciOxideError> {
    let p = ApproxPolynomial::interpolate_barycentric(
        vec![0.0_f64, 1.0, 2.0],
        vec![0.0, 1.0, 4.0],
    )?;
    assert!((p.evaluate(3.0)? - 9.0).abs() < 1e-12);
    Ok(())
}
```

## Crates

Each crate's README introduces its main APIs.
Full API documentation is available on [docs.rs](https://docs.rs/scioxide-core).

| Crate | Purpose |
|---|---|
| [`scioxide-core`](crates/scioxide-core) | Scalar types, errors, tolerances, and execution |
| [`scioxide-linalg`](crates/scioxide-linalg) | Dense matrices, factorizations, and iterative solvers |
| [`scioxide-sparse`](crates/scioxide-sparse) | Sparse matrices, preconditioners, and graph algorithms |
| [`scioxide-spatial`](crates/scioxide-spatial) | Geometry, spatial search, and meshes |
| [`scioxide-special`](crates/scioxide-special) | Gamma, Bessel, elliptic, and other special functions |
| [`scioxide-poly`](crates/scioxide-poly) | Exact and floating-point polynomials and rational functions |
| [`scioxide-fft`](crates/scioxide-fft) | Fourier, cosine, sine, and related transforms |
| [`scioxide-signal`](crates/scioxide-signal) | Filters, resampling, spectra, and wavelets |
| [`scioxide-random`](crates/scioxide-random) | Random numbers, probability distributions, and sampling |
| [`scioxide-stats`](crates/scioxide-stats) | Statistics, hypothesis tests, regression, and time series |
| [`scioxide-ad`](crates/scioxide-ad) | Forward and reverse automatic differentiation |
| [`scioxide-differentiate`](crates/scioxide-differentiate) | Finite differences and complex-step differentiation |
| [`scioxide-interp`](crates/scioxide-interp) | Interpolation and splines |
| [`scioxide-rootfind`](crates/scioxide-rootfind) | Scalar, system, and polynomial root finding |
| [`scioxide-integrate`](crates/scioxide-integrate) | Quadrature, cubature, and Monte Carlo integration |
| [`scioxide-optimize`](crates/scioxide-optimize) | Local, constrained, and global optimization |
| [`scioxide-ode`](crates/scioxide-ode) | Ordinary, differential-algebraic, delay, and stochastic equations |
| [`scioxide-pde`](crates/scioxide-pde) | Finite difference, element, and volume methods |
| [`scioxide-validated`](crates/scioxide-validated) | Interval arithmetic and verified numerical methods |

## Development

The repository pins development tools in `rust-toolchain.toml`.
Nix users can run `nix develop` for development or `nix develop .#msrv` to check Rust 1.93.0.
CI tests both the minimum supported version and stable Rust.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and review policy.

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p scioxide-core
cargo test -p scioxide-core --no-default-features
```

Run tests for the crates you change, including their doctests.

Published normal and build dependencies must stay within SciOxide, including optional and target-specific dependencies; external crates are allowed as dev-dependencies.
Keep the dependency graph acyclic and put shared functionality in its owning crate.
The default feature is `std`; internal dependencies disable default features and forward `std` explicitly.
Platform libraries are supplied by callers, as with the optional `scioxide-linalg/blas` feature, rather than linked by the crates.
Keep independent reference data with its tests and record its source, generation precision, domain and comparison tolerances; tests must not require the generators.

Shared numerical behavior is documented in [scioxide-core](https://docs.rs/scioxide-core/0.1.0/scioxide_core/); individual APIs document their domains, accuracy, failure behavior and allocation requirements in rustdoc.
The [handbook](docs/handbook/README.md) explains the mathematics and usage in English and Japanese.
Keep both languages consistent when changing an explanation, and link to the owning API instead of duplicating its specification.
Write Markdown and MDX prose with one sentence per line, without wrapping within a sentence.
Preserve paragraph and list structure, and keep code, math, tables, frontmatter and other structured markup in their required syntax.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party notices and data licenses are included in the relevant crates.
This license covers original source, documentation and independently generated reference data; third-party portions retain their own terms.
See [data provenance and redistribution](DATA_PROVENANCE.md) for the distinction between original outputs, reference tools and copied input material.
