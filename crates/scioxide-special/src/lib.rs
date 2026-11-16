//! Special functions of real and complex arguments for `f32` and `f64`.
//!
//! The crate evaluates the classical special functions of mathematical
//! physics and statistics: gamma and beta functions and their incomplete
//! forms, error functions and normal-distribution functions, Bessel, Airy and
//! related functions, elliptic integrals and Jacobi elliptic functions,
//! hypergeometric functions, zeta functions and polylogarithms, exponential
//! and related integrals, Coulomb wave functions, Legendre functions and
//! angular-momentum coefficients, and Mathieu, spheroidal and ellipsoidal
//! wave functions. Definitions and notation follow the NIST Digital Library
//! of Mathematical Functions (DLMF); each function cites the section it
//! implements.
//!
//! Functions with a [`RealScalar`](scioxide_core::RealScalar) parameter accept
//! `f32` and `f64`. Functions with a
//! [`NumericScalar`](scioxide_core::NumericScalar) parameter also accept
//! [`Complex<f32>`](scioxide_core::Complex) and `Complex<f64>`, and evaluate the
//! analytic continuation, with the branch cuts stated on each function.
//!
//! # Quick start
//!
//! ```
//! use core::f64::consts::{FRAC_PI_2, PI};
//! use scioxide_core::Complex;
//! use scioxide_special::{bessel_j, erf, gamma};
//!
//! // Γ(5) = 4! = 24
//! assert!((gamma(5.0_f64)? - 24.0).abs() < 1e-12);
//! // Γ(1/2) = √π (DLMF 5.4.6), here with a complex argument.
//! let half = gamma(Complex::new(0.5_f64, 0.0))?;
//! assert!((half - Complex::new(PI.sqrt(), 0.0)).norm() < 1e-12);
//! // erf(1) and J₀(1), rounded to 16 significant digits.
//! assert!((erf(1.0_f64)? - 0.842_700_792_949_714_9).abs() < 1e-12);
//! assert!((bessel_j(0.0_f64, 1.0)? - 0.765_197_686_557_966_6).abs() < 1e-12);
//! // J_½(z) = sin z / √(πz/2) (DLMF 10.16.1): a real order with a complex argument.
//! let z = Complex::new(1.0_f64, 1.0);
//! assert!((bessel_j(0.5, z)? - z.sin() / (z * FRAC_PI_2).sqrt()).norm() < 1e-12);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a function
//!
//! All functions are exported from the crate root.
//!
//! | Family | Main entry points | DLMF |
//! |---|---|---|
//! | Gamma, beta, digamma, polygamma | [`gamma`], [`ln_gamma`], [`digamma`], [`polygamma`], [`beta`], [`ln_beta`] | [5](https://dlmf.nist.gov/5) |
//! | Incomplete gamma and beta, with inverses | [`gamma_p`], [`gamma_q`], [`gamma_p_inv`], [`beta_inc`], [`beta_inc_q`], [`beta_inc_inv`], and `*_log` forms | [8](https://dlmf.nist.gov/8) |
//! | Factorials, binomials, Stirling, Bernoulli and Euler numbers | [`factorial`], [`binomial`], [`pochhammer`], [`stirling_first_kind`], [`stirling_second_kind`], [`bernoulli`], [`euler`] | [5](https://dlmf.nist.gov/5), [24](https://dlmf.nist.gov/24), [26](https://dlmf.nist.gov/26) |
//! | Error functions, Dawson, Faddeeva, Voigt | [`erf`], [`erfc`], [`erfcx`], [`erfi`], [`erfinv`], [`erfcinv`], [`dawson`], [`faddeeva`], [`voigt_profile`] | [7](https://dlmf.nist.gov/7) |
//! | Normal distribution, Owen's T | [`normal_cdf`], [`normal_sf`], [`normal_quantile`], [`normal_isf`], [`normal_log_cdf`], [`owens_t`] | [7](https://dlmf.nist.gov/7) |
//! | Bessel and Hankel functions | [`bessel_j`], [`bessel_y`], [`bessel_i`], [`bessel_k`], [`hankel_1`], [`hankel_2`], and `*_scaled`, `*_derivative` forms | [10](https://dlmf.nist.gov/10) |
//! | Spherical Bessel and Riccati–Bessel functions | [`spherical_jn`], [`spherical_yn`], [`spherical_in`], [`spherical_kn`], [`riccati_jn`], [`riccati_yn`] | [10.47](https://dlmf.nist.gov/10.47) |
//! | Airy functions | [`airy_ai`], [`airy_bi`], [`airy_ai_scaled`], [`airy_bi_scaled`] | [9](https://dlmf.nist.gov/9) |
//! | Kelvin, Struve, parabolic cylinder functions | [`ber`], [`bei`], [`ker`], [`kei`], [`kelvin`], [`struve`], [`modstruve`], [`parabolic_d`], [`parabolic_v`], [`parabolic_w`] | [10.61](https://dlmf.nist.gov/10.61), [11](https://dlmf.nist.gov/11), [12](https://dlmf.nist.gov/12) |
//! | Integrals of Bessel and Struve functions | [`bessel_j0_integral`], [`bessel_y0_integral`], [`bessel_i0_integral`], [`bessel_k0_integral`], [`struve_h0_integral`], [`struve_l0_integral`] | [10.22](https://dlmf.nist.gov/10.22), [11.7](https://dlmf.nist.gov/11.7) |
//! | Elliptic integrals and functions | [`elliptic_k`], [`elliptic_e`], [`elliptic_f`], [`elliptic_pi`], [`carlson_rf`], [`carlson_rd`], [`carlson_rj`], [`jacobi_sn`], [`jacobi_cn`], [`jacobi_dn`] | [19](https://dlmf.nist.gov/19), [22](https://dlmf.nist.gov/22) |
//! | Hypergeometric functions | [`hyp0f1`], [`hyp1f1`], [`hyperu`], [`hyp2f1`], [`hyp2f0`], [`hyp_pfq`], [`HypergeometricFunction`] | [13](https://dlmf.nist.gov/13), [15](https://dlmf.nist.gov/15), [16](https://dlmf.nist.gov/16) |
//! | Zeta functions, polylogarithms, Lerch transcendent | [`riemann_zeta`], [`hurwitz_zeta`], [`dirichlet_eta`], [`polylog`], [`dilog`], [`lerch_phi`], [`clausen`] | [25](https://dlmf.nist.gov/25) |
//! | Lambert W and Wright omega | [`lambert_w0`], [`lambert_wm1`], [`lambert_w_complex`], [`wright_omega`] | [4.13](https://dlmf.nist.gov/4.13) |
//! | Exponential, trigonometric and logarithmic integrals, Fresnel integrals | [`expint_e1`], [`expint_en`], [`expint_ei`], [`si`], [`ci`], [`shi`], [`chi`], [`logarithmic_integral`], [`fresnel_s`], [`fresnel_c`] | [6](https://dlmf.nist.gov/6), [7](https://dlmf.nist.gov/7) |
//! | Debye, Fermi–Dirac, synchrotron and transport integrals | [`debye_1`], [`fermi_dirac_half`], [`fermi_dirac_int`], [`synchrotron_1`], [`transport_2`] | [25.12(iii)](https://dlmf.nist.gov/25.12#iii) (Fermi–Dirac) |
//! | Coulomb wave and hydrogenic functions | [`coulomb_f`], [`coulomb_g`], [`coulomb_wave`], [`coulomb_wave_sequence`], [`hydrogenic_radial`] | [33](https://dlmf.nist.gov/33) |
//! | Legendre functions and spherical harmonics | [`legendre_p`], [`legendre_q`], [`legendre_p_assoc`], [`legendre_h3d`], [`spherical_harmonic`], [`spherical_harmonic_real`] | [14](https://dlmf.nist.gov/14) |
//! | Wigner and Clebsch–Gordan coefficients | [`wigner_3j`], [`wigner_6j`], [`wigner_9j`], [`clebsch_gordan`], [`racah_w`], [`gaunt`], [`AngularPlan`] | [34](https://dlmf.nist.gov/34) |
//! | Mathieu, spheroidal and ellipsoidal wave functions | [`mathieu_a`], [`mathieu_b`], [`mathieu_ce`], [`mathieu_se`], [`mathieu_mc`], [`mathieu_ms`], [`spheroidal_characteristic_value`], [`spheroidal_angular`], [`spheroidal_radial`], [`ellipsoidal_harmonic`]; [`MathieuPlan`], [`SpheroidalPlan`], [`EllipsoidalHarmonicPlan`] for many points | [28](https://dlmf.nist.gov/28), [30](https://dlmf.nist.gov/30), [29](https://dlmf.nist.gov/29) |
//!
//! Most functions come in several forms with the same arguments:
//!
//! | Form | Returns | Use it when |
//! |---|---|---|
//! | `f(x)` | the value | you need the value only |
//! | `f_with_options(x, options)` | a [`SpecialFunctionResult`] | you need an error estimate, a derivative, a scaled or logarithmic form, or a work budget other than the default; see [detailed results](#detailed-results) |
//! | `f_scoped(x, options, scope)` | a [`SpecialFunctionResult`] | the evaluation must observe cancellation or a deadline, or count toward a parent solver's budget; see [scoped evaluation](#scoped-evaluation) |
//!
//! The argument sets the scalar type of the result. An order, shape or other
//! parameter may be real when the argument is complex: `bessel_j(2.5, z)` with a
//! complex `z` evaluates `J_2.5(z)`. Each such function says which parameters
//! may be real; [`Parameter`] states the rule. A complex parameter needs a
//! complex argument.
//!
//! A scalar function has no batch form; evaluate it at many points by mapping
//! and collecting. A sequence form exists only for consecutive Coulomb orders,
//! where [`coulomb_wave_sequence`] reuses the order recurrence.
//!
//! ```
//! let xs = [0.25_f64, 0.5, 1.0];
//! let values = xs.iter().map(|&x| scioxide_special::erf(x)).collect::<Result<Vec<_>, _>>()?;
//! assert!((values[2] - 0.842_700_792_949_714_9).abs() < 1e-12);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! Zeros of these functions are computed by `scioxide-rootfind`. Exact integer
//! factorials and binomial coefficients are
//! [`checked_factorial`](scioxide_core::checked_factorial) and
//! [`checked_binomial`](scioxide_core::checked_binomial) in `scioxide-core`.
//!
//! Functions whose values routinely overflow or underflow have exponentially
//! scaled companions named `*_scaled`, such as [`bessel_i_scaled`] for
//! `e^(−|Re z|) I_ν(z)`, and logarithmic companions. Those of gamma, beta,
//! factorials and binomial coefficients are named `ln_*`, such as [`ln_gamma`];
//! elsewhere the name contains `log`, as in [`log_erfc`], [`gamma_log_p`],
//! [`beta_inc_log`], [`normal_log_cdf`], [`bessel_k_log`] and
//! [`hurwitz_zeta_log`]; inverses of logarithmic probabilities are named
//! `*_inv_log*`, such as [`gamma_p_inv_log`]. Each states its scale factor.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors) and
//! [non-finite rules](scioxide_core#non-finite-values). In this crate:
//!
//! - `NonFinite`: an argument is NaN or infinite. No function accepts infinite
//!   arguments; limits at infinity are not evaluated.
//! - `DomainError` with [`DomainReason::Pole`](scioxide_core::DomainReason::Pole):
//!   the argument is at a pole, such as `Γ(0)`. Some functions instead return
//!   the limit `±∞` at a pole or singular endpoint; those say so.
//! - `DomainError` with `OutsideDomain`: the argument is outside the real domain
//!   of a function, such as `gamma_p(a, x)` with `x < 0`. `Undefined` marks a
//!   point where the limit depends on the direction of approach.
//! - `InvalidInput` with `OutOfRange`: an integer parameter, such as an order or
//!   a degree, is outside its supported range.
//! - `NumericalInstability` with `Overflow`: the value overflows the scalar
//!   type. The detailed form still returns the scaled and logarithmic forms where
//!   they are representable. A value that underflows is returned as a signed
//!   zero or subnormal, never as an error.
//! - `NumericalInstability` with `LossOfSignificance`: cancellation that the
//!   method cannot resolve would leave no correct digits.
//! - `ConvergenceFailure` with `IterationLimit`: the work budget ran out. The
//!   error carries the attempted counts.
//! - `ExecutionStopped`: a scoped evaluation was cancelled or passed its
//!   deadline.
//!
//! Each function lists its own domain and poles under `# Errors`.
//!
//! # Detailed results
//!
//! `*_with_options` functions return a [`SpecialFunctionResult`]. Its
//! [`value`](SpecialFunctionResult::value) carries an absolute and, for nonzero
//! values, a relative error estimate. Where available, it also holds the
//! derivative with respect to the argument, a [scaled](SpecialFunctionResult::scaled)
//! form with a separate binary exponent, and a
//! [logarithmic](SpecialFunctionResult::logarithmic) form. The scaled and
//! logarithmic forms remain available when the value overflows. The ordinary
//! value is the same, bit for bit, as the value of the plain function.
//!
//! [`termination`](SpecialFunctionResult::termination) reports how the
//! evaluation finished. If the work budget runs out after a useful estimate,
//! the detailed form returns it with
//! [`SpecialFunctionTermination::IterationLimit`], and the plain form returns
//! `ConvergenceFailure`. A documented infinite limit has termination
//! [`Limit`](SpecialFunctionTermination::Limit) and representation
//! [`InfiniteLimit`](ScalarRepresentation::InfiniteLimit).
//!
//! ```
//! use scioxide_special::{CombinatorialOptions, ScalarRepresentation};
//! use scioxide_special::{factorial, factorial_with_options, ln_factorial};
//!
//! // 1000! ≈ 4.02 × 10²⁵⁶⁷ overflows f64, so the plain function fails.
//! assert!(factorial::<f64>(1000).is_err());
//! let result = factorial_with_options::<f64>(1000, CombinatorialOptions::default())?;
//! assert_eq!(result.scalar_representation(), ScalarRepresentation::Overflow);
//! let log = result.logarithmic().expect("log form").value().log_magnitude();
//! assert!((log.expect("nonzero") - ln_factorial::<f64>(1000)?).abs() < 1e-8);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! [`try_value`](SpecialFunctionResult::try_value) and
//! [`try_derivative`](SpecialFunctionResult::try_derivative) return the ordinary
//! value and the derivative without their estimates. A result has no derivative
//! when the function does not compute one, or when the derivative is outside the
//! range of the scalar type or singular at the argument; each function says
//! whether its detailed form returns one.
//!
//! ```
//! // Ai(1) and Ai′(1)
//! let ai = scioxide_special::airy_ai_with_options(1.0_f64, Default::default())?;
//! let (value, slope) = (ai.try_value()?, ai.try_derivative()?);
//! assert!((value - 0.135_292_416_312_881_4).abs() < 1e-14);
//! assert!((slope + 0.159_147_441_296_793_2).abs() < 1e-14);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! Every options type holds a work budget, counted in work units such as
//! series terms or recurrence steps; the defaults are listed on each type.
//! Only [`SpectralOptions`] and [`EllipsoidalOptions`] also take a tolerance,
//! which controls truncation or quadrature. Elsewhere the accuracy is fixed,
//! and a larger budget only lets slowly converging cases finish.
//!
//! # Scoped evaluation
//!
//! Functions whose work depends on their arguments have `*_scoped` forms.
//! They take any `impl Into<SolveScope>`: an
//! [`ExecutionContext`](scioxide_core::ExecutionContext) starts a new evaluation,
//! and a `&mut SolveContext` runs it inside a parent solver. The options budget
//! bounds the evaluation, the parent receives its work as nested counts, and
//! the parent's cancellation and deadline apply. See
//! [execution contexts](scioxide_core#execution-contexts) and
//! [solve accounting](scioxide_core#solve-accounting). A scoped form returns the
//! same result as the `*_with_options` form.
//!
//! A function has no scoped form when every evaluation does a bounded amount of
//! work that depends neither on its arguments nor on convergence: a fixed
//! table, a rational or Chebyshev approximation, or a fixed set of quadrature
//! nodes. These are [`gamma`], [`ln_gamma`], [`digamma`], [`beta`] and
//! [`ln_beta`]; [`erfinv`] and [`erfcinv`]; the normal distribution functions
//! ([`normal_cdf`], [`normal_sf`], their logarithms, [`normal_quantile`] and
//! [`normal_isf`]) and [`owens_t`]; the factorials, multifactorials and
//! binomial coefficients with their logarithms, the Pochhammer symbols, and
//! the Bernoulli and Euler numbers; the exponential integrals `E₁` and `Ei`
//! with their scaled forms, `E₃`, and the arctangent and logarithmic
//! integrals; the Fermi–Dirac integrals of order −1 and 0 and the incomplete
//! integral of order 0; the synchrotron functions; and [`clausen`], [`dilog`],
//! [`spence`] and [`coulomb_normalization`]. Functions evaluated by a series, a
//! continued fraction, a recurrence in order or degree, or a root iteration
//! have scoped forms.
//!
//! The scoped forms of the error functions and Dawson's integral accept complex
//! arguments only, because their real evaluation has a fixed cost.
//!
//! ```
//! use scioxide_core::ExecutionContext;
//! use scioxide_special::{EllipticOptions, carlson_rf_scoped, carlson_rf_with_options};
//!
//! let options = EllipticOptions::default();
//! let scoped = carlson_rf_scoped(1.0_f64, 2.0, 3.0, options, ExecutionContext::sequential())?;
//! assert_eq!(scoped, carlson_rf_with_options(1.0_f64, 2.0, 3.0, options)?);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Accuracy
//!
//! An ulp is the spacing of floating-point numbers at the exact result, and
//! `ε` is the machine epsilon of the scalar type. The bounds below hold on the
//! tested range of each function: the reference points checked by the crate's
//! tests against high-precision values. Near a zero of a function the bound is
//! absolute, and relative accuracy can be poor there even when the absolute
//! error is small.
//!
//! | Functions (real arguments) | `f64` | `f32` |
//! |---|---|---|
//! | [`gamma`], [`ln_gamma`], [`digamma`], [`polygamma`], [`beta`], [`ln_beta`] | 4 ulps | 1 ulp |
//! | [`factorial`], [`ln_factorial`], [`binomial`] | 4 ulps | 1 ulp |
//! | [`gamma_p`], [`gamma_q`], [`beta_inc`] and the other incomplete and inverse gamma and beta functions | 32 ulps | 1 ulp |
//! | [`erfinv`], [`erfcinv`], [`normal_quantile`], [`normal_isf`], [`erfcx`], [`log_erfc`], [`dawson`] | 4 ulps | 1 ulp |
//! | [`normal_cdf`], [`normal_sf`], [`normal_log_cdf`], [`normal_log_sf`] on `[−38, 38]` | 8 ulps | 1 ulp |
//! | [`expint_e1`], [`expint_ei`], [`expint_en`], [`expint_3`], [`si`], [`ci`], away from zeros | 8 ulps | — |
//! | [`elliptic_k`], [`elliptic_e`] on `0 ≤ m < 1` | 2 ulps | 1 ulp |
//! | Carlson integrals, incomplete elliptic integrals, Jacobi elliptic functions | 64 ulps | 64 ulps |
//! | [`bessel_j`], [`bessel_y`], [`bessel_i`], [`bessel_k`], Hankel and Airy functions | 64 ulps | 64 ulps |
//! | [`struve`], [`modstruve`] on `−5 ≤ ν ≤ 10`, `10⁻³ ≤ x ≤ 100` | 64 ulps | — |
//! | [`legendre_p`], [`legendre_q`], [`legendre_p_assoc`] | 64 ulps | 64 ulps |
//! | [`hyp0f1`], [`hyp1f1`], [`hyp2f1`], [`hyperu`] | 64 ulps | 64 ulps |
//! | [`lambert_w0`], [`lambert_wm1`], [`wright_omega`], [`dilog`], [`spence`] | 64 ulps | 64 ulps |
//! | [`coulomb_f`], [`coulomb_g`] and their derivatives | 64 ulps | 64 ulps |
//!
//! A separate sample of 4,160 `f64` points measures maxima of 3.7 ulps for
//! [`gamma`], 1.9 for [`ln_gamma`], 2.1 for [`digamma`] and 1.0 for [`beta`].
//! On that wider sample [`gamma_p`] and [`gamma_q`] reach 90 ulps and
//! [`beta_inc`] 12 ulps, so the 32-ulp bound applies to the tested range only. Near the zeros of
//! [`struve`] the absolute error is at most `4ε·max(1, |H_ν(x)|)`. The real
//! [`riemann_zeta`], [`dirichlet_eta`] and [`hurwitz_zeta`] with `s > 1` report
//! error estimates of at most `64ε|value|`.
//!
//! Complex arguments, and all functions not listed above, have no uniform
//! bound. For them, use the error estimate of the detailed result.
//!
//! # Error estimates
//!
//! The error estimate of a [`ValueWithError`] has one of three
//! [reliabilities](scioxide_core::EstimateReliability). `ModelBased` estimates
//! are derived from the truncation and rounding errors of the evaluation.
//! `StatisticallyCalibrated` estimates are envelopes measured against
//! high-precision reference values, scaled as `k·max(ε|v|, η)`, where `v` is
//! the value, `η` the smallest positive subnormal number and `k` a constant of
//! the function. `Heuristic` estimates have no coverage statement. None is a
//! guaranteed enclosure; use `scioxide-validated` for verified enclosures.
//!
//! A large estimate is reported, not turned into an error. Logarithmic errors
//! are in natural-log units, and the logarithmic form is omitted when the
//! relative error reaches ½.
//!
//! # Reusing plans and workspaces
//!
//! Four plans precompute data that many evaluations reuse: [`MathieuPlan`],
//! [`SpheroidalPlan`] and [`EllipsoidalHarmonicPlan`] solve a truncated
//! eigenproblem and store its coefficients, and [`AngularPlan`] sieves the primes
//! and least prime factors that factor the factorials of Racah's formulas. A plan
//! is immutable and keeps the options it was built with. It has `new` for default
//! execution and `build(.., scope)` for scoped preparation, and its evaluation
//! methods have `*_scoped` variants. For a single evaluation, the one-shot
//! functions such as [`mathieu_a`], [`spheroidal_angular`],
//! [`ellipsoidal_harmonic`] and [`wigner_3j`] build the plan and evaluate once.
//!
//! [`HypergeometricFunction`] stores a validated parameter set of `ₚF_q` or `U`
//! for evaluation at many arguments. It adds the regularized `ₚF_q` and first
//! derivatives with respect to the argument or one parameter to what
//! [`hyp_pfq`] and [`hyperu`] offer.
//!
//! The owned evaluation methods of [`AngularPlan`] and [`HypergeometricFunction`]
//! allocate a workspace per call. Their `*_scoped` methods reuse one you provide
//! ([`AngularWorkspace`], [`HypergeometricWorkspace`]) and allocate nothing. See
//! [plans and workspaces](scioxide_core#plans-and-workspaces).
//!
//! # Feature flags
//!
//! - `std` (default): uses the standard library. Without it, the crate builds
//!   for `no_std` targets with `alloc`, with the same API and results.
//!
//! # References
//!
//! F. W. J. Olver et al., eds., *NIST Digital Library of Mathematical
//! Functions*. <https://dlmf.nist.gov>

#![forbid(unsafe_code)]
#![deny(unreachable_pub)]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

extern crate alloc;

#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod bessel_airy;
mod combinatorial;
mod complex_support;
mod coulomb;
mod dispatch;
mod elliptic_family;
mod error_probability;
mod expansions;
mod gamma_beta;
mod hypergeometric_family;
mod integrals;
mod options;
mod orthogonal_angular;
mod result;
mod spectral;
mod support;
mod thermal_transport;
mod util;
mod value;
mod zeta_polylog;

// Results.
pub use dispatch::Parameter;
pub use result::{ScalarRepresentation, SpecialFunctionResult, SpecialFunctionTermination};
pub use value::ValueWithError;

// Gamma and beta functions.
pub use gamma_beta::api::{
    beta, beta_inc, beta_inc_scoped, beta_inc_with_options, beta_with_options, digamma,
    digamma_with_options, gamma, gamma_lower, gamma_lower_scoped, gamma_lower_with_options,
    gamma_p, gamma_p_scoped, gamma_p_with_options, gamma_q, gamma_q_scoped, gamma_q_with_options,
    gamma_upper, gamma_upper_scoped, gamma_upper_with_options, gamma_with_options, ln_beta,
    ln_beta_with_options, ln_gamma, ln_gamma_with_options, polygamma, polygamma_scoped,
    polygamma_with_options,
};
pub use gamma_beta::probability_tails::{
    beta_inc_inv, beta_inc_inv_log, beta_inc_inv_log_with_options, beta_inc_inv_with_options,
    beta_inc_log, beta_inc_log_q, beta_inc_log_q_with_options, beta_inc_log_with_options,
    beta_inc_q, beta_inc_q_inv, beta_inc_q_inv_log, beta_inc_q_inv_log_with_options,
    beta_inc_q_inv_with_options, beta_inc_q_with_options, gamma_log_p, gamma_log_p_with_options,
    gamma_log_q, gamma_log_q_with_options, gamma_p_inv, gamma_p_inv_log,
    gamma_p_inv_log_with_options, gamma_p_inv_with_options, gamma_q_inv, gamma_q_inv_log,
    gamma_q_inv_log_with_options, gamma_q_inv_with_options,
};
pub use gamma_beta::probability_tails::{
    beta_inc_inv_log_scoped, beta_inc_inv_scoped, beta_inc_log_q_scoped, beta_inc_log_scoped,
    beta_inc_q_inv_log_scoped, beta_inc_q_inv_scoped, beta_inc_q_scoped, gamma_log_p_scoped,
    gamma_log_q_scoped, gamma_p_inv_log_scoped, gamma_p_inv_scoped, gamma_q_inv_log_scoped,
    gamma_q_inv_scoped,
};
pub use options::{BetaOptions, GammaOptions};

// Error and probability functions.
pub use error_probability::api::{
    dawson, dawson_scoped, dawson_with_options, erf, erf_scoped, erf_with_options, erfc,
    erfc_scoped, erfc_with_options, erfcx, erfcx_scoped, erfcx_with_options, erfi, erfi_scoped,
    erfi_with_options, log_erfc, log_erfc_scoped, log_erfc_with_options,
};

pub use error_probability::faddeeva::{
    faddeeva, faddeeva_scoped, faddeeva_with_options, voigt_complex, voigt_complex_scoped,
    voigt_complex_with_options, voigt_profile, voigt_profile_scoped, voigt_profile_with_options,
};
pub use error_probability::owens_t::{owens_t, owens_t_with_options};
pub use error_probability::probability::{
    normal_cdf, normal_cdf_with_options, normal_inv_log_cdf, normal_inv_log_cdf_with_options,
    normal_inv_log_sf, normal_inv_log_sf_with_options, normal_isf, normal_isf_with_options,
    normal_log_cdf, normal_log_cdf_with_options, normal_log_sf, normal_log_sf_with_options,
    normal_quantile, normal_quantile_with_options, normal_sf, normal_sf_with_options,
};
pub use error_probability::probability::{normal_inv_log_cdf_scoped, normal_inv_log_sf_scoped};
pub use error_probability::real_api::{erfcinv, erfcinv_with_options, erfinv, erfinv_with_options};
pub use options::{ErrorFunctionOptions, ProbabilityOptions};

// Bessel, Airy and related functions.
pub use bessel_airy::api::{
    airy_ai, airy_ai_scaled, airy_ai_scaled_scoped, airy_ai_scaled_with_options, airy_ai_scoped,
    airy_ai_with_options, airy_bi, airy_bi_scaled, airy_bi_scaled_scoped,
    airy_bi_scaled_with_options, airy_bi_scoped, airy_bi_with_options,
};
pub use bessel_airy::api::{
    bessel_i, bessel_i_derivative, bessel_i_derivative_with_options, bessel_i_scaled,
    bessel_i_scaled_derivative, bessel_i_scaled_derivative_with_options, bessel_i_scaled_scoped,
    bessel_i_scaled_with_options, bessel_i_scoped, bessel_i_with_options, bessel_j,
    bessel_j_derivative, bessel_j_derivative_with_options, bessel_j_scoped, bessel_j_with_options,
    bessel_k, bessel_k_derivative, bessel_k_derivative_with_options, bessel_k_scaled,
    bessel_k_scaled_derivative, bessel_k_scaled_derivative_with_options, bessel_k_scaled_scoped,
    bessel_k_scaled_with_options, bessel_k_scoped, bessel_k_with_options, bessel_y,
    bessel_y_derivative, bessel_y_derivative_with_options, bessel_y_scoped, bessel_y_with_options,
    hankel_1, hankel_1_derivative, hankel_1_derivative_with_options, hankel_1_scoped,
    hankel_1_with_options, hankel_2, hankel_2_derivative, hankel_2_derivative_with_options,
    hankel_2_scoped, hankel_2_with_options, spherical_hankel_1, spherical_hankel_1_scoped,
    spherical_hankel_1_with_options, spherical_hankel_2, spherical_hankel_2_scoped,
    spherical_hankel_2_with_options, spherical_in, spherical_in_scaled, spherical_in_scaled_scoped,
    spherical_in_scaled_with_options, spherical_in_scoped, spherical_in_with_options, spherical_jn,
    spherical_jn_scoped, spherical_jn_with_options, spherical_kn, spherical_kn_scaled,
    spherical_kn_scaled_scoped, spherical_kn_scaled_with_options, spherical_kn_scoped,
    spherical_kn_with_options, spherical_yn, spherical_yn_scoped, spherical_yn_with_options,
};
pub use bessel_airy::api::{
    bessel_i_derivative_scoped, bessel_i_scaled_derivative_scoped, bessel_j_derivative_scoped,
    bessel_k_derivative_scoped, bessel_k_scaled_derivative_scoped, bessel_y_derivative_scoped,
    hankel_1_derivative_scoped, hankel_2_derivative_scoped,
};
pub use bessel_airy::bessel_integrals::{
    bessel_i0_integral, bessel_i0_integral_with_options, bessel_i0_remainder_integral,
    bessel_i0_remainder_integral_with_options, bessel_j0_complement_integral,
    bessel_j0_complement_integral_with_options, bessel_j0_integral,
    bessel_j0_integral_with_options, bessel_k0_integral, bessel_k0_integral_with_options,
    bessel_k0_tail_integral, bessel_k0_tail_integral_with_options, bessel_y0_integral,
    bessel_y0_integral_with_options, bessel_y0_tail_integral, bessel_y0_tail_integral_with_options,
    struve_h0_integral, struve_h0_integral_with_options, struve_h0_tail_integral,
    struve_h0_tail_integral_with_options, struve_l0_integral, struve_l0_integral_with_options,
};
pub use bessel_airy::bessel_integrals::{
    bessel_i0_integral_scoped, bessel_i0_remainder_integral_scoped,
    bessel_j0_complement_integral_scoped, bessel_j0_integral_scoped, bessel_k0_integral_scoped,
    bessel_k0_tail_integral_scoped, bessel_y0_integral_scoped, bessel_y0_tail_integral_scoped,
    struve_h0_integral_scoped, struve_h0_tail_integral_scoped, struve_l0_integral_scoped,
};
pub use bessel_airy::complex_bessel::{
    bessel_k_log, bessel_k_log_with_options, bessel_lambda, bessel_lambda_with_options,
    bessel_weighted_integral, bessel_weighted_integral_with_options,
};
pub use bessel_airy::complex_bessel::{
    bessel_k_log_scoped, bessel_lambda_scoped, bessel_weighted_integral_scoped,
};
pub use bessel_airy::kelvin::{
    bei, bei_with_options, beip, beip_with_options, ber, ber_with_options, berp, berp_with_options,
    kei, kei_with_options, keip, keip_with_options, kelvin, kelvin_with_options, ker,
    ker_with_options, kerp, kerp_with_options,
};
pub use bessel_airy::kelvin::{
    bei_scoped, beip_scoped, ber_scoped, berp_scoped, kei_scoped, keip_scoped, kelvin_scoped,
    ker_scoped, kerp_scoped,
};
pub use bessel_airy::parabolic::{
    parabolic_d, parabolic_d_complex, parabolic_d_complex_with_options, parabolic_d_with_options,
    parabolic_v, parabolic_v_with_options, parabolic_w, parabolic_w_with_options,
};
pub use bessel_airy::parabolic::{
    parabolic_d_complex_scoped, parabolic_d_scoped, parabolic_v_scoped, parabolic_w_scoped,
};
pub use bessel_airy::spherical::{
    legendre_h3d, legendre_h3d_with_options, riccati_jn, riccati_jn_with_options, riccati_yn,
    riccati_yn_with_options,
};
pub use bessel_airy::spherical::{legendre_h3d_scoped, riccati_jn_scoped, riccati_yn_scoped};
pub use bessel_airy::struve::{
    modstruve, modstruve_scoped, modstruve_with_options, struve, struve_scoped, struve_with_options,
};
pub use options::{
    AiryOptions, BesselIntegralOptions, BesselOptions, KelvinOptions, ParabolicCylinderOptions,
    SphericalBesselOptions, StruveOptions,
};

// Coulomb and hydrogenic functions.
pub use coulomb::{
    CoulombComponent, CoulombWaveResult, coulomb_f, coulomb_f_prime, coulomb_f_prime_scoped,
    coulomb_f_prime_with_options, coulomb_f_scoped, coulomb_f_with_options, coulomb_g,
    coulomb_g_prime, coulomb_g_prime_scoped, coulomb_g_prime_with_options, coulomb_g_scoped,
    coulomb_g_with_options, coulomb_normalization, coulomb_normalization_sequence,
    coulomb_normalization_sequence_into, coulomb_normalization_with_options, coulomb_sequence,
    coulomb_sequence_into, coulomb_spherical_f, coulomb_spherical_f_scoped,
    coulomb_spherical_f_with_options, coulomb_wave, coulomb_wave_scoped, coulomb_wave_sequence,
    coulomb_wave_sequence_into, coulomb_wave_with_options, hydrogenic_radial,
    hydrogenic_radial_scoped, hydrogenic_radial_with_options,
};
pub use options::CoulombOptions;

// Combinatorial functions.
pub use combinatorial::{
    bernoulli, bernoulli_with_options, binomial, binomial_with_options, checked_bernoulli,
    checked_euler, double_factorial, double_factorial_with_options, euler, euler_with_options,
    factorial, factorial_with_options, generalized_binomial, generalized_binomial_with_options,
    ln_binomial, ln_binomial_with_options, ln_double_factorial, ln_double_factorial_with_options,
    ln_factorial, ln_factorial_with_options, ln_multifactorial, ln_multifactorial_with_options,
    multifactorial, multifactorial_with_options, pochhammer, pochhammer_relative,
    pochhammer_relative_with_options, pochhammer_with_options, stirling_first_kind,
    stirling_first_kind_with_options, stirling_second_kind, stirling_second_kind_with_options,
};
pub use combinatorial::{
    generalized_binomial_scoped, stirling_first_kind_scoped, stirling_second_kind_scoped,
};
pub use options::CombinatorialOptions;

// Elliptic functions.
pub use elliptic_family::api::{
    elliptic_e, elliptic_e_inc, elliptic_e_inc_scoped, elliptic_e_inc_with_options,
    elliptic_e_scoped, elliptic_e_with_options, elliptic_f, elliptic_f_scoped,
    elliptic_f_with_options, elliptic_k, elliptic_k_scoped, elliptic_k_with_options, elliptic_pi,
    elliptic_pi_scoped, elliptic_pi_with_options,
};
pub use elliptic_family::carlson::{
    carlson_rc, carlson_rc_scoped, carlson_rc_with_options, carlson_rd, carlson_rd_scoped,
    carlson_rd_with_options, carlson_rf, carlson_rf_scoped, carlson_rf_with_options, carlson_rg,
    carlson_rg_scoped, carlson_rg_with_options, carlson_rj, carlson_rj_scoped,
    carlson_rj_with_options,
};
pub use elliptic_family::jacobi::{
    jacobi_amplitude, jacobi_amplitude_scoped, jacobi_amplitude_with_options, jacobi_cn,
    jacobi_cn_scoped, jacobi_cn_with_options, jacobi_dn, jacobi_dn_scoped, jacobi_dn_with_options,
    jacobi_sn, jacobi_sn_scoped, jacobi_sn_with_options,
};
pub use options::EllipticOptions;

// Hypergeometric functions.
pub use hypergeometric_family::hypergeometric::{
    HypergeometricDerivative, HypergeometricFamily, HypergeometricFunction,
    HypergeometricWorkspace, hyp_pfq, hyp_pfq_with_options, hyp0f1, hyp0f1_with_options, hyp1f1,
    hyp1f1_with_options, hyp2f0, hyp2f0_with_options, hyp2f1, hyp2f1_regularized,
    hyp2f1_regularized_with_options, hyp2f1_with_options, hyperu, hyperu_with_options,
};
pub use hypergeometric_family::hypergeometric::{
    hyp_pfq_scoped, hyp0f1_scoped, hyp1f1_scoped, hyp2f0_scoped, hyp2f1_regularized_scoped,
    hyp2f1_scoped, hyperu_scoped,
};
pub use options::HypergeometricOptions;

// Integrals.
pub use integrals::api::{
    fresnel_c, fresnel_c_scoped, fresnel_c_with_options, fresnel_s, fresnel_s_scoped,
    fresnel_s_with_options,
};
pub use integrals::{
    atanint, atanint_with_options, chi, chi_scaled, chi_scaled_with_options, chi_with_options, ci,
    ci_with_options, expint_3, expint_3_with_options, expint_e1, expint_e1_scaled,
    expint_e1_scaled_with_options, expint_e1_with_options, expint_e2, expint_e2_scaled,
    expint_e2_scaled_with_options, expint_e2_with_options, expint_ei, expint_ei_scaled,
    expint_ei_scaled_with_options, expint_ei_with_options, expint_en, expint_en_scaled,
    expint_en_scaled_with_options, expint_en_with_options, logarithmic_integral,
    logarithmic_integral_offset, logarithmic_integral_offset_with_options,
    logarithmic_integral_with_options, shi, shi_scaled, shi_scaled_with_options, shi_with_options,
    si, si_with_options,
};
pub use integrals::{
    chi_scaled_scoped, chi_scoped, ci_scoped, expint_e2_scaled_scoped, expint_e2_scoped,
    expint_en_scaled_scoped, expint_en_scoped, shi_scaled_scoped, shi_scoped, si_scoped,
};
pub use options::IntegralOptions;

// Orthogonal and angular functions.
pub use options::LegendreOptions;
pub use orthogonal_angular::angular::{
    AngularMomentum, AngularOptions, AngularPlan, AngularState, AngularWorkspace, clebsch_gordan,
    gaunt, racah_w, wigner_3j, wigner_6j, wigner_9j,
};
pub use orthogonal_angular::api::{
    legendre_p, legendre_p_assoc, legendre_p_assoc_scoped, legendre_p_assoc_with_options,
    legendre_p_scoped, legendre_p_with_options, legendre_q, legendre_q_scoped,
    legendre_q_with_options,
};
pub use orthogonal_angular::harmonics::{
    spherical_harmonic, spherical_harmonic_real, spherical_harmonic_real_scoped,
    spherical_harmonic_real_with_options, spherical_harmonic_scoped,
    spherical_harmonic_with_options,
};

// Spectral and ellipsoidal functions.
pub use options::EllipsoidalOptions;
pub use spectral::ellipsoidal::{
    EllipsoidalHarmonicPlan, EllipsoidalSigns, LameClass, ellipsoidal_harmonic,
    ellipsoidal_harmonic_second_kind, ellipsoidal_normalization,
};
pub use spectral::mathieu::{
    MathieuParity, MathieuPlan, mathieu_a, mathieu_b, mathieu_ce, mathieu_mc, mathieu_ms,
    mathieu_se,
};
pub use spectral::spheroidal::{
    SpheroidalKind, SpheroidalPlan, spheroidal_angular, spheroidal_characteristic_value,
    spheroidal_radial,
};
pub use spectral::{
    RadialKind, SpectralDiagnostics, SpectralNormalization, SpectralOptions, SpectralTerms,
};

// Thermal and transport functions.
pub use options::{DebyeOptions, FermiDiracOptions, SynchrotronOptions, TransportOptions};
pub use thermal_transport::synchrotron::{
    synchrotron_1, synchrotron_1_scaled, synchrotron_1_scaled_with_options,
    synchrotron_1_with_options, synchrotron_2, synchrotron_2_scaled,
    synchrotron_2_scaled_with_options, synchrotron_2_with_options,
};
pub use thermal_transport::thermal::{
    debye_1, debye_1_with_options, debye_2, debye_2_with_options, debye_3, debye_3_with_options,
    debye_4, debye_4_with_options, debye_5, debye_5_with_options, debye_6, debye_6_with_options,
    fermi_dirac_0, fermi_dirac_0_with_options, fermi_dirac_1, fermi_dirac_1_with_options,
    fermi_dirac_2, fermi_dirac_2_with_options, fermi_dirac_3half, fermi_dirac_3half_with_options,
    fermi_dirac_half, fermi_dirac_half_with_options, fermi_dirac_inc_0,
    fermi_dirac_inc_0_with_options, fermi_dirac_int, fermi_dirac_int_with_options, fermi_dirac_m1,
    fermi_dirac_m1_with_options, fermi_dirac_mhalf, fermi_dirac_mhalf_with_options,
};
pub use thermal_transport::thermal::{
    debye_1_scoped, debye_2_scoped, debye_3_scoped, debye_4_scoped, debye_5_scoped, debye_6_scoped,
    fermi_dirac_1_scoped, fermi_dirac_2_scoped, fermi_dirac_3half_scoped, fermi_dirac_half_scoped,
    fermi_dirac_int_scoped, fermi_dirac_mhalf_scoped,
};
pub use thermal_transport::transport::{
    transport_2, transport_2_scoped, transport_2_with_options, transport_3, transport_3_scoped,
    transport_3_with_options, transport_4, transport_4_scoped, transport_4_with_options,
    transport_5, transport_5_scoped, transport_5_with_options,
};

// Zeta, polylogarithms and related functions.
pub use options::{ClausenOptions, LambertOptions, LerchOptions, PolylogOptions, ZetaOptions};
pub use zeta_polylog::api::{
    dilog, dilog_with_options, lerch_phi, lerch_phi_scoped, lerch_phi_with_options, polylog,
    polylog_scoped, polylog_with_options, spence, spence_with_options, wright_omega,
    wright_omega_scoped, wright_omega_with_options,
};
pub use zeta_polylog::api::{
    dirichlet_eta, dirichlet_eta_scoped, dirichlet_eta_with_options, hurwitz_zeta,
    hurwitz_zeta_scoped, hurwitz_zeta_with_options, riemann_zeta, riemann_zeta_scoped,
    riemann_zeta_with_options,
};
pub use zeta_polylog::clausen::{clausen, clausen_with_options};
pub use zeta_polylog::lambert::{
    lambert_w_complex, lambert_w_complex_scoped, lambert_w_complex_with_options, lambert_w0,
    lambert_w0_scoped, lambert_w0_with_options, lambert_wm1, lambert_wm1_scoped,
    lambert_wm1_with_options,
};
pub use zeta_polylog::real_api::{
    hurwitz_zeta_log, hurwitz_zeta_log_scoped, hurwitz_zeta_log_with_options,
};

#[cfg(test)]
#[path = "../tests/support/publication.rs"]
mod test_support;

#[cfg(test)]
extern crate self as scioxide_special;

pub use bessel_airy::kelvin::{
    bei_scaled, bei_scaled_scoped, bei_scaled_with_options, beip_scaled, beip_scaled_scoped,
    beip_scaled_with_options, ber_scaled, ber_scaled_scoped, ber_scaled_with_options, berp_scaled,
    berp_scaled_scoped, berp_scaled_with_options, kei_scaled, kei_scaled_scoped,
    kei_scaled_with_options, keip_scaled, keip_scaled_scoped, keip_scaled_with_options, ker_scaled,
    ker_scaled_scoped, ker_scaled_with_options, kerp_scaled, kerp_scaled_scoped,
    kerp_scaled_with_options,
};

pub use bessel_airy::api::{
    bessel_j_scaled, bessel_j_scaled_scoped, bessel_j_scaled_with_options, bessel_y_scaled,
    bessel_y_scaled_scoped, bessel_y_scaled_with_options, hankel_1_scaled, hankel_1_scaled_scoped,
    hankel_1_scaled_with_options, hankel_2_scaled, hankel_2_scaled_scoped,
    hankel_2_scaled_with_options, spherical_hankel_1_scaled, spherical_hankel_1_scaled_scoped,
    spherical_hankel_1_scaled_with_options, spherical_hankel_2_scaled,
    spherical_hankel_2_scaled_scoped, spherical_hankel_2_scaled_with_options,
};
