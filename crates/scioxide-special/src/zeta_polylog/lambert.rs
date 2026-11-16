//! Lambert W branches and Wright omega.
//!
//! Real arguments use scalar Halley kernels; complex arguments keep their
//! sheet-aware continuation. Adjacent sheets use a Puiseux initializer and a
//! cancellation-resistant residual in `u = W + 1` near the branch point.
//! Principal seeds cover the moderate plane with local approximations and the
//! exterior with a logarithmic expansion. Halley stops at a residual supported
//! by floating-point precision, including before a rounded cycle repeats.
//! Exhausted budgets produce diagnostic reports; value-only
//! projections return typed convergence failures.
use crate::support::TRANSCENDENTAL_ERRORS as ERRORS;
use crate::support::TranscendentalEstimate as Estimate;
use crate::support::Work;
use crate::support::widen_numeric as widen;
use crate::support::{C, ONE, ZERO, real};
use crate::zeta_polylog::transcendental::{
    Detailed, RealKernelEvaluation, TranscendentalEvaluation, real_report, report,
};
use crate::{LambertOptions, SpecialFunctionResult};
use core::f64::consts::{E, PI};
use scioxide_core::{AlgorithmComponent, algorithm_component};
use scioxide_core::{
    Complex, ConvergenceDiagnostics, ConvergenceReason, DomainReason, EvaluationCounts,
    NormSelection, NumericScalar, RealScalar, ResidualReport, SciOxideError, SolveContext,
};

#[path = "lambert_real.rs"]
mod real_kernel;

const PUISEUX_COEFFICIENTS: [f64; 9] = [
    1.0,
    -1.0 / 3.0,
    11.0 / 72.0,
    -43.0 / 540.0,
    769.0 / 17280.0,
    -221.0 / 8505.0,
    680863.0 / 43545600.0,
    -1963.0 / 204120.0,
    226287557.0 / 37623398400.0,
];

fn branch_coordinate(z: C) -> C {
    // The low part of e prevents the rounded product e*z from erasing the
    // distance of an exact binary argument from -1/e (Corless et al., §4).
    C::new(
        z.re.mul_add(E, 1.0) + z.re * 1.4456468917292502e-16,
        z.im * E,
    )
}
/// Puiseux initializer for u = W + 1, without rounding away the small u.
fn puiseux(q: C, negative: bool, work: &mut Work<'_, '_, '_>) -> Result<C, SciOxideError> {
    let mut p = (q * 2.0).sqrt();
    if negative {
        p = -p;
    }
    let mut value = ZERO;
    for &coefficient in PUISEUX_COEFFICIENTS.iter().rev() {
        work.try_step()?;
        value = value * p + real(coefficient);
    }
    Ok(value * p)
}

// (u - 1)exp(u) + 1 = u² Σ (n - 1)u^(n - 2)/n!, n >= 2.
// On the branch-point region |q| <= 0.2, the selected root has |u| < 1.
// Terms through n=24 leave a tail bounded by 1.7e-24 |u|^25.
const BRANCH_COEFFICIENTS: [f64; 23] = {
    let mut coefficients = [0.0; 23];
    let mut factorial = 1.0;
    let mut n = 1;
    while n <= 24 {
        factorial *= n as f64;
        if n >= 2 {
            coefficients[n - 2] = (n - 1) as f64 / factorial;
        }
        n += 1;
    }
    coefficients
};

/// Residual, first and second derivatives, and residual rounding/truncation error.
fn branch_residual<V: NumericScalar<Real = f64>>(
    u: V,
    q: V,
    exp: impl Fn(V) -> V,
) -> (V, V, V, f64) {
    let polynomial = BRANCH_COEFFICIENTS
        .iter()
        .rev()
        .fold(V::ZERO, |sum, &coefficient| {
            if V::IS_REAL {
                V::from_real(
                    sum.real_component()
                        .mul_add(u.real_component(), coefficient),
                )
            } else {
                sum * u + V::from_real(coefficient)
            }
        });
    let h = u * u * polynomial;
    let exponential = exp(u);
    (
        h - q,
        u * exponential,
        (V::ONE + u) * exponential,
        8.0 * f64::EPSILON * (h.abs() + q.abs()) + 1.7e-24 * u.abs().powi(25),
    )
}

/// One stopping rule for shifted, exponential and logarithmic Halley iteration. Cycles are accepted
/// only when their best residual supports a finite estimate.
fn refine<V: NumericScalar<Real = f64>>(
    mut w: V,
    offset: f64,
    lip: Option<f64>,
    method: AlgorithmComponent,
    residual: impl Fn(V) -> (V, V, V, f64),
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate<V>, SciOxideError> {
    if let Some(lip) = lip {
        w = V::from_components(w.real_component(), lip);
    }
    let mut previous = [None; 2];
    let mut estimate = None;
    loop {
        if let Err(error) = work.try_step() {
            return if !work.complete {
                estimate.ok_or(error)
            } else {
                Err(error)
            };
        }
        let (f, d, second, _) = residual(w);
        let curvature = if V::IS_REAL {
            V::from_real(0.5) * f * (second / d)
        } else {
            f * (second / d) * V::from_real(0.5)
        };
        let step = f / (d - curvature);
        if !step.has_finite_components() {
            return Err(ERRORS.unstable());
        }
        let mut next = w - step;
        if let Some(lip) = lip {
            next = V::from_components(next.real_component(), lip);
        }
        if !next.has_finite_components() {
            return Err(ERRORS.unstable());
        }
        let cycle = next == w || previous.contains(&Some(next));
        let mut best = next;
        let mut best_residual = residual(next);
        if cycle {
            for candidate in [Some(w), previous[0], previous[1]].into_iter().flatten() {
                let candidate_residual = residual(candidate);
                if candidate_residual.0.abs() < best_residual.0.abs() {
                    best = candidate;
                    best_residual = candidate_residual;
                }
            }
        }
        let value = V::from_components(best.real_component() + offset, best.imaginary_component());
        let correction = (best_residual.0 / best_residual.1).abs();
        let uncertainty = best_residual.3 / best_residual.1.abs();
        let error = correction + uncertainty + 16.0 * f64::EPSILON * value.abs();
        if !correction.has_finite_components() || !error.has_finite_components() {
            return Err(ERRORS.unstable());
        }
        let tolerance = f64::EPSILON * value.abs().max(f64::MIN_POSITIVE);
        // Rounded complex iterates need not repeat within the short history.
        // Apply the same residual floor to every iterate; waiting for an exact
        // cycle can exhaust any budget after the value has already converged.
        if correction <= (if V::IS_REAL && !cycle { 0.5 } else { 8.0 }) * tolerance {
            return Ok(Estimate::new(value, error, method));
        }
        if cycle {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::Stagnation,
                ConvergenceDiagnostics::new()
                    .with_algorithm(scioxide_core::AlgorithmId::from_components(
                        algorithm_component!("lambert_polylog"),
                        method,
                        1,
                    ))
                    .with_counts(EvaluationCounts::from_iterations(work.used).into())
                    .with_residual(ResidualReport::unscaled(
                        best_residual.0.abs(),
                        NormSelection::L2,
                    )?),
                "Halley iterates cycled without an accurate residual",
            ));
        }
        estimate = Some(Estimate::new(value, error, method));
        previous = [Some(w), previous[0]];
        w = next;
    }
}

fn branch_refine(
    q: C,
    negative: bool,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let u = puiseux(q, negative, work)?;
    let lip = (q.im == 0.0 && q.re > 0.0).then_some(if negative { -q.im } else { q.im });
    refine(
        u,
        -1.0,
        lip,
        algorithm_component!("branch_point_halley"),
        |u| branch_residual(u, q, |u| u.exp()),
        work,
    )
}

// Region layout adapted from mpmath 1.3.0's hybrid principal-sheet initializer.
/// Principal-sheet initializer. Evaluate the local table on the upper lip and conjugate for
/// the lower lip; testing the sign bit also preserves the branch selected by negative zero.
/// The exterior expansion is Corless et al. (1996), equation (4.19).
#[allow(clippy::approx_constant)] // -0.318 approximates Re(W₀(-1)), not -1/π.
fn principal_seed(z: C, l: C) -> C {
    let upper = C::new(z.re, z.im.abs());
    if z.re > -1.0 && z.re < 2.5 && upper.im < 4.0 {
        let seed = if upper.im > 1.0 {
            C::new(0.876, 0.645) + C::new(0.118, -0.174) * (upper - C::new(0.75, 2.5))
        } else if upper.im > 0.25 {
            C::new(0.505, 0.204) + C::new(0.375, -0.132) * (upper - C::new(0.75, 0.5))
        } else if z.re < -0.5 {
            C::new(-0.318, 1.34) + C::new(-0.697, -0.593) * (upper + ONE)
        } else if z.re < -0.2 {
            let q = branch_coordinate(upper);
            -ONE + (q * 2.0).sqrt() - q * (2.0 / 3.0)
        } else if z.re < 0.5 {
            upper
        } else {
            real(0.2) + upper * 0.3
        };
        if z.im.is_sign_negative() {
            seed.conj()
        } else {
            seed
        }
    } else {
        let log_l = l.ln();
        let inverse = ONE / l;
        l - log_l + log_l * inverse + log_l * (log_l - real(2.0)) * inverse * inverse * 0.5
    }
}

pub(crate) fn kernel(
    z: C,
    branch: i32,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    kernel_with_log(z, branch, None, work)
}

fn kernel_with_log(
    z: C,
    branch: i32,
    log_z: Option<C>,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    crate::support::require_finite_param(z, "Lambert parameter z must be finite")?;
    if z == ZERO {
        if branch != 0 {
            return Err(ERRORS.domain(scioxide_core::argument!("z"), DomainReason::Pole));
        }
        let mut e = Estimate::new(z, 0.0, algorithm_component!("zero"));
        e.derivative = Some(ONE);
        return Ok(e);
    }
    let lower = z.im.is_sign_negative();
    let adjacent = branch == 0 || branch == -1 && !lower || branch == 1 && lower;
    let q = branch_coordinate(z);
    if adjacent && q.norm() <= 0.2 {
        return branch_refine(q, branch != 0, work).map(|e| with_derivative(e, z));
    }
    let l = log_z.unwrap_or_else(|| z.ln() + C::new(0.0, 2.0 * PI * f64::from(branch)));
    let mut w = if branch == 0 {
        principal_seed(z, l)
    } else if adjacent && (z - real(-1.0 / E)).norm() < 0.2 {
        puiseux(q, branch != 0, work)? - ONE
    } else if (branch == -1 && !lower || branch == 1 && lower)
        && z.im == 0.0
        && z.re < 0.0
        && z.re > -1.0 / E
    {
        C::new(
            (-z.re).ln() - (-(-z.re).ln()).ln(),
            if branch == -1 { -0.0 } else { 0.0 },
        )
    } else {
        l - l.ln()
    };
    let real_lower = adjacent && branch != 0 && z.im == 0.0 && z.re > -1.0 / E && z.re < 0.0;
    let real_lip = if branch == -1 { -0.0 } else { 0.0 };
    if real_lower {
        w.im = real_lip;
    }
    let e = refine(
        w,
        0.0,
        real_lower.then_some(real_lip),
        algorithm_component!("branch_aware_halley"),
        |w| {
            // Exponential residual for the principal region, logarithmic residual
            // for remote sheets and tiny nonprincipal inputs. Neither forms exp(w).
            if w.re > -500.0 && w.re < 500.0 && branch == 0 {
                let t = z * (-w).exp();
                let f = w - t;
                let d = ONE + t;
                (f, d, -t, 8.0 * f64::EPSILON * (w.norm() + t.norm()))
            } else {
                let logarithm = w.ln();
                let f = w + logarithm - l;
                let d = ONE + ONE / w;
                (
                    f,
                    d,
                    -ONE / w / w,
                    8.0 * f64::EPSILON * w.norm()
                        + 8.0 * f64::EPSILON * logarithm.norm()
                        + 8.0 * f64::EPSILON * l.norm(),
                )
            }
        },
        work,
    )?;
    // A defining-equation residual alone cannot identify a W sheet.
    let phase = e.value + e.value.ln() - l;
    if work.complete && phase.norm() > 1e-10 * (1.0 + l.norm()) {
        return Err(ERRORS.unstable());
    }
    Ok(with_derivative(e, z))
}
fn with_derivative(mut e: Estimate, z: C) -> Estimate {
    e.derivative = Some(e.value / z / (ONE + e.value));
    e.derivative_error = e.error / z.norm() / (ONE + e.value).norm().powi(2);
    e
}
fn omega_kernel(z: C, work: &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError> {
    crate::support::require_finite_param(z, "Lambert parameter z must be finite")?;
    if z == ONE {
        let mut e = Estimate::new(ONE, 0.0, algorithm_component!("identity"));
        e.derivative = Some(real(0.5));
        return Ok(e);
    }
    let delta = C::new(
        z.re + 1.0,
        (z.im.abs() - PI) - crate::support::constants::PI.lo,
    );
    if delta.norm() <= 0.05 {
        let mut e = branch_refine(-crate::support::complex_expm1(delta), delta.im > 0.0, work)?;
        if z.im < 0.0 {
            e.value = e.value.conj();
        }
        e.method = algorithm_component!("omega_branch_point_halley");
        e.derivative = Some(e.value / (ONE + e.value));
        e.derivative_error = e.error / (ONE + e.value).norm().powi(2);
        return Ok(e);
    }
    if z.re > -700.0 && z.re < 600.0 && z.im.abs() < 1e8 {
        work.try_step()?;
        let exp_z = z.exp();
        let k = ((z.im - exp_z.im.atan2(exp_z.re)) / (2.0 * PI)).round() as i32;
        let mut e = kernel_with_log(exp_z, k, Some(z), work)?;
        e.method = algorithm_component!("omega_unwinding");
        e.derivative = Some(e.value / (ONE + e.value));
        e.derivative_error = e.error / (ONE + e.value).norm().powi(2);
        return Ok(e);
    }
    if z.re <= -700.0 && z.im.abs() <= PI {
        work.try_step()?;
        let value = z.exp();
        let mut e = Estimate::new(
            value,
            value.norm() * 8.0 * f64::EPSILON,
            algorithm_component!("omega_small"),
        );
        e.derivative = Some(value);
        e.logarithm = Some(z);
        e.error = e.error.max(f64::from_bits(1));
        return Ok(e);
    }
    let mut e = refine(
        z - z.ln(),
        0.0,
        None,
        algorithm_component!("omega_logarithmic_halley"),
        |w| {
            let logarithm = w.ln();
            let f = w + logarithm - z;
            let d = ONE + ONE / w;
            (
                f,
                d,
                -ONE / w / w,
                // Scale each term before adding, including at z = f64::MAX.
                8.0 * f64::EPSILON * w.norm()
                    + 8.0 * f64::EPSILON * logarithm.norm()
                    + 8.0 * f64::EPSILON * z.norm(),
            )
        },
        work,
    )?;
    e.derivative = Some(e.value / (ONE + e.value));
    e.derivative_error = e.error / (ONE + e.value).norm().powi(2);
    Ok(e)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes branch `k` of the Lambert W function, `W_k(z)`, for complex `z`.

`W_k(z)` is the solution `w` of `w eʷ = z` on branch `k` (the `branch` argument), numbered as in
Corless et al. (1996) and DLMF §4.13. `W₀` is the principal branch with its cut on `(−∞, −1/e]`;
every other branch has its cut on `(−∞, 0]`. On a cut the sign of a zero imaginary part selects the
side. `W₀(0) = 0`, and `z = 0` is a singularity of every other branch. The argument nearest `−1/e`
is evaluated as the exact binary number it is, unlike the endpoint convention of
[`crate::lambert_w0`] and [`crate::lambert_wm1`]; at a singular point the detailed derivative may be
absent.

Halley iteration refines a starting value from a Puiseux series next to the branch point, from local
approximations for moderate `|z|` on the principal branch, and from the asymptotic expansion in
`ln z + 2πik` elsewhere. Iteration stops when the residual reaches its rounding level.

# Examples

```
use scioxide_core::Complex;

// W_k(z) solves w eʷ = z on every branch k
let z = Complex::new(-1.0_f64, 2.0);
for k in [-1, 0, 1] {
    let w = scioxide_special::lambert_w_complex(z, k)?;
    assert!((w * w.exp() - z).norm() < 1e-13);
}
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"There is no uniform ulp bound; the error estimate is derived from the Halley correction and the
residual. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `z = 0` for `k ≠ 0`.

# References

- NIST DLMF, §4.13. <https://dlmf.nist.gov/4.13>
- R. M. Corless, G. H. Gonnet, D. E. G. Hare, D. J. Jeffrey and D. E. Knuth, "On the Lambert W
  function", *Adv. Comput. Math.* 5 (1996), 329–359. <https://doi.org/10.1007/BF02124750>"###)]

    pub lambert_w_complex [S: RealScalar] (z: Complex<S>,
        branch: i32) -> Complex<S>
    where [Complex<S>: NumericScalar<Real = S>, ];
    options options: LambertOptions;
    detailed lambert_w_complex_with_options -> Detailed<S>;
    value {
        lambert_w_complex_evaluate(z, branch, LambertOptions::default())?.value()
    }
    report {
        lambert_w_complex_evaluate(z, branch, options)?.publish()
    }
    scoped pub lambert_w_complex_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            let z = widen(z, "Lambert parameter z must be finite")?;
            let mut work = Work::with_context(options.iteration_budget().get(), Some(context));
            let e = kernel(z, branch, &mut work);
            report(e, &work)?.publish()
        })
    }
}

/// Evaluate before projecting the value or forming the detailed result.
fn lambert_w_complex_evaluate<S: RealScalar>(
    z: Complex<S>,
    branch: i32,
    options: LambertOptions,
) -> Result<TranscendentalEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let z = widen(z, "Lambert parameter z must be finite")?;
    let mut work = Work::new(options.iteration_budget().get());
    let e = kernel(z, branch, &mut work);
    report(e, &work)
}

fn real_evaluate<S: RealScalar>(
    x: S,
    branch: i32,
    options: LambertOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<RealKernelEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "x must be finite",
        ));
    }
    let input = x.try_to_f64_exact()?;
    let endpoint = S::try_from_f64_lossy(-1.0 / E)?;
    if x < endpoint || branch == -1 && x >= S::ZERO {
        return Err(ERRORS.domain(
            scioxide_core::argument!("x"),
            if x == S::ZERO {
                DomainReason::Pole
            } else {
                DomainReason::OutsideDomain
            },
        ));
    }
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let e = if x == endpoint {
        Ok(Estimate::new(
            -1.0,
            0.0,
            algorithm_component!("rounded_branch_endpoint"),
        ))
    } else {
        real_kernel::lambert(input, branch == -1, &mut work)
    };
    real_report(e, &work)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the principal branch `W₀(x)` of the Lambert W function for real `x`.

`W₀(x) ≥ −1` is the solution `w` of `w eʷ = x` for `x ≥ −1/e` (DLMF §4.13). At the value of `−1/e`
rounded to the scalar type the result is exactly `−1`, and the detailed result has no finite
derivative; the next smaller representable `x` is outside the domain. The function has no poles.

Halley iteration on exponential or logarithmic residuals gives the value; next to `−1/e` the
iteration works in `u = W + 1` from a series in the branch coordinate.

# Examples

```
// W₀(e) = 1, and W₀(x) solves w eʷ = x
let w = scioxide_special::lambert_w0(core::f64::consts::E)?;
assert!((w - 1.0).abs() < 1e-14);
let w = scioxide_special::lambert_w0(-0.3_f64)?;
assert!((w * w.exp() + 0.3).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 64 ulps for `f32` and `f64` on the tested range, with absolute error bounds next
to the branch point `−1/e`. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`OutsideDomain`) if `x` is less than `−1/e` rounded to the scalar type.

# References

- NIST DLMF, §4.13. <https://dlmf.nist.gov/4.13>
- R. M. Corless, G. H. Gonnet, D. E. G. Hare, D. J. Jeffrey and D. E. Knuth, "On the Lambert W
  function", *Adv. Comput. Math.* 5 (1996), 329–359. <https://doi.org/10.1007/BF02124750>"###)]

    pub lambert_w0 [S: RealScalar] (x: S) -> S
    where [Complex<S>: NumericScalar<Real = S>, ];
    options options: LambertOptions;
    detailed lambert_w0_with_options -> SpecialFunctionResult<S>;
    value {
        let options: LambertOptions = LambertOptions::default();
        real_evaluate(x, 0, options, None)?.value()
    }
    report {
        real_evaluate(x, 0, options, None)?.publish()
    }
    scoped pub lambert_w0_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            real_evaluate(x, 0, options, Some(context))?.publish()
        })
    }
}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the lower real branch `W₋₁(x)` of the Lambert W function.

`W₋₁(x) ≤ −1` is the solution `w` of `w eʷ = x` for `−1/e ≤ x < 0` (DLMF §4.13). It tends to `−∞` as
`x → 0⁻`, so `x = 0` is a pole. At the value of `−1/e` rounded to the scalar type the result is
exactly `−1`, and the detailed result has no finite derivative; the next smaller representable `x`
is outside the domain.

Halley iteration on exponential or logarithmic residuals gives the value; next to `−1/e` the
iteration works in `u = W + 1` from a series in the branch coordinate.

# Examples

```
// W₋₁(−2e⁻²) = −2
let x = -2.0 * (-2.0_f64).exp();
let w = scioxide_special::lambert_wm1(x)?;
assert!((w + 2.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"The error is within 64 ulps for `f32` and `f64` on the tested range, with absolute error bounds next
to the branch point `−1/e`. See the [crate accuracy table](crate#accuracy)."###, r###"- `DomainError` (`Pole`) at `x = 0`.
- `DomainError` (`OutsideDomain`) if `x > 0` or `x` is less than `−1/e` rounded to the scalar type.

# References

- NIST DLMF, §4.13. <https://dlmf.nist.gov/4.13>
- R. M. Corless, G. H. Gonnet, D. E. G. Hare, D. J. Jeffrey and D. E. Knuth, "On the Lambert W
  function", *Adv. Comput. Math.* 5 (1996), 329–359. <https://doi.org/10.1007/BF02124750>"###)]

    pub lambert_wm1 [S: RealScalar] (x: S) -> S
    where [Complex<S>: NumericScalar<Real = S>, ];
    options options: LambertOptions;
    detailed lambert_wm1_with_options -> SpecialFunctionResult<S>;
    value {
        let options: LambertOptions = LambertOptions::default();
        real_evaluate(x, -1, options, None)?.value()
    }
    report {
        real_evaluate(x, -1, options, None)?.publish()
    }
    scoped pub lambert_wm1_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            real_evaluate(x, -1, options, Some(context))?.publish()
        })
    }
}

pub(crate) fn wright_omega_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    wright_omega_complex_evaluate(z, LambertOptions::default(), None)?.value()
}

/// Evaluate before projecting the value or forming the detailed result.
pub(crate) fn wright_omega_complex_evaluate<S: RealScalar>(
    z: Complex<S>,
    options: LambertOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<TranscendentalEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let z = widen(z, "Lambert parameter z must be finite")?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let e = omega_kernel(z, &mut work).and_then(|mut e| {
        let maximum = S::MAX.into_f64();
        if e.value.re.abs().max(e.value.im.abs()) <= maximum {
            let projected = C::new(
                S::try_from_f64_lossy(e.value.re)?.into_f64(),
                S::try_from_f64_lossy(e.value.im)?.into_f64(),
            );
            let rounding = (projected - e.value).norm();
            if rounding > 0.0 && projected != ZERO {
                // The ordinary bound covers both the value and its defining
                // residual. Destination rounding is amplified by 1 + 1/w in
                // w + Log(w), especially for small binary32 omega values.
                e.error += rounding * (ONE + ONE / projected).norm();
            }
        }
        Ok(e)
    });
    report(e, &work)
}
/// Real omega never exponentiates a large argument or enters complex continuation.
pub(crate) fn wright_omega_real_evaluate<S: RealScalar>(
    x: S,
    options: LambertOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<RealKernelEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    crate::support::require_finite_param(x, "Lambert parameter x must be finite")?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let estimate = real_kernel::omega(x.try_to_f64_exact()?, &mut work);
    real_report(estimate, &work)
}
pub(crate) fn wright_omega<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: LambertOptions = LambertOptions::default();

    wright_omega_real_evaluate(x, options, None)?.value()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;

    #[test]
    fn short_cycles_without_an_accurate_residual_are_typed_stagnation() {
        for length in [2, 3] {
            let mut work = Work::new(32);
            // Model a residual evaluation that drives a rounded two/three-cycle.
            // These deliberately inconsistent derivatives must never be accepted.
            let error = refine(
                ONE,
                0.0,
                None,
                scioxide_core::algorithm_component!("cycle_test"),
                |w| {
                    let next = if length == 2 {
                        -w
                    } else if w.re == 3.0 {
                        ONE
                    } else {
                        w + ONE
                    };
                    (w - next, ONE, ZERO, 0.0)
                },
                &mut work,
            )
            .err()
            .unwrap();
            match error.view() {
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason,
                    diagnostics,
                    ..
                } => {
                    assert_eq!(reason, ConvergenceReason::Stagnation);
                    assert_eq!(diagnostics.counts().own().iterations(), length);
                    assert_eq!(diagnostics.algorithm().unwrap().method(), "cycle_test");
                    assert!(diagnostics.residual().unwrap().raw_norm() > 0.0);
                }
                error => panic!("wrong cycle classification: {error:?}"),
            }
        }
    }

    #[test]
    fn accurate_residual_does_not_wait_for_a_rounded_cycle() {
        let mut work = Work::new(32);
        let result = refine(
            ONE,
            0.0,
            None,
            scioxide_core::algorithm_component!("rounding_test"),
            |w| {
                let f = if w.re == 1.0 {
                    -f64::EPSILON
                } else {
                    f64::EPSILON
                };
                (real(f), ONE, ZERO, f64::EPSILON)
            },
            &mut work,
        )
        .unwrap();
        assert_eq!(work.used, 1);
        assert!((result.value - ONE).norm() <= f64::EPSILON);
        assert!(result.error.is_finite());
    }
}
