# scioxide-special

Special functions for Rust.
The crate evaluates gamma and beta functions and their incomplete forms, error and normal-distribution functions, Bessel, Airy and related functions, elliptic integrals and Jacobi elliptic functions, hypergeometric functions, zeta functions and polylogarithms, exponential and related integrals, Coulomb wave functions, Legendre functions and angular-momentum coefficients, and Mathieu, spheroidal and ellipsoidal wave functions, for `f32` and `f64` and, where defined, complex arguments.
Each function can also return an error estimate and forms that survive overflow.

## Main entry points

- Gamma, beta and combinatorics: `gamma`, `ln_gamma`, `digamma`, `beta`, `gamma_p`, `gamma_p_inv`, `beta_inc`, `beta_inc_inv`, `factorial`, `binomial`, `pochhammer`, `bernoulli`.
- Error and probability functions: `erf`, `erfc`, `erfcx`, `erfinv`, `dawson`, `faddeeva`, `voigt_profile`, `normal_cdf`, `normal_quantile`, `owens_t`.
- Bessel family: `bessel_j`, `bessel_y`, `bessel_i`, `bessel_k`, `hankel_1`, `spherical_jn`, `airy_ai`, `airy_bi`, `kelvin`, `struve`, `parabolic_d`.
- Elliptic: `elliptic_k`, `elliptic_e`, `carlson_rf`, `jacobi_sn`.
- Hypergeometric: `hyp1f1`, `hyp2f1`, `hyperu`, `hyp_pfq`, `HypergeometricFunction`.
- Zeta and related: `riemann_zeta`, `hurwitz_zeta`, `polylog`, `lerch_phi`, `lambert_w0`, `wright_omega`.
- Integrals: `expint_e1`, `expint_ei`, `si`, `ci`, `fresnel_s`, `debye_1`, `fermi_dirac_half`.
- Physics: `coulomb_wave`, `hydrogenic_radial`, `legendre_p`, `spherical_harmonic`, `wigner_3j`, `mathieu_a`, `mathieu_ce`, `spheroidal_angular`, `ellipsoidal_harmonic`, and the `MathieuPlan`, `SpheroidalPlan` and `EllipsoidalHarmonicPlan` plans for many points.
- Detailed results: most functions `f` have `f_with_options`, returning a `SpecialFunctionResult`, and many have `f_scoped` for cancellable evaluation.

## Feature flags

- `std` (default): uses the standard library.
  Without it the crate is `no_std` and needs `alloc`.

See the [API documentation](https://docs.rs/scioxide-special) for a quick start, the function table, conventions, and the error and accuracy rules.
Third-party attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Third-party portions retain the conditions in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including notices required for binary redistribution.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
