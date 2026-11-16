//! Faddeeva/Voigt kernels: Algorithm 916 sums (Zaghloul and Ali, 2011) and the
//! Laplace continued fraction. On the axes, `w(iy) = erfcx(y)` and
//! `Im w(x) = 2F(x)/√π` come from the crate's `f64` erfcx and Dawson kernels
//! ([`crate::error_probability::real_faddeeva`]). The sums share one Gaussian
//! and decaying exponential factors per term. Finite values keep their checks
//! and work status until the detailed result forms its derivative and error.
// The sums and the continued fraction follow Steven G. Johnson's MIT
// implementation; see THIRD_PARTY_NOTICES.md.

use crate::ValueWithError;
use crate::error_probability::probability::{error, finish};
use crate::expansions::real;
use crate::result::{Bound, Either, Evaluation, Forms, Publish, Standard};
use crate::support::PROBABILITY_ERRORS as ERRORS;
use crate::support::Work;
use crate::support::require_finite_param;
use crate::{
    ProbabilityOptions, ScalarRepresentation, SpecialFunctionResult, SpecialFunctionTermination,
};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, Complex, EstimateReliability, EvaluationCounts, LogValue,
    NumericScalar, RealScalar, ScaledValue, SciOxideError, algorithm_component,
};

fn scale<S: RealScalar>(z: Complex<S>, s: S) -> Complex<S> {
    Complex::new(z.re * s, z.im * s)
}

type Detailed<S> = SpecialFunctionResult<Complex<S>>;

/// An `f64` axis kernel at `x`, charged as one unit of `w`.
fn axis<S: RealScalar>(kernel: fn(f64) -> f64, x: S, w: &mut Work) -> Result<S, SciOxideError> {
    if w.step() {
        S::try_from_f64_lossy(kernel(x.into_f64()))
    } else {
        Ok(S::ZERO)
    }
}
fn sinc<S: RealScalar>(x: S, sine: S) -> S {
    if x == S::ZERO { S::ONE } else { sine / x }
}

pub(crate) fn upper<S: RealScalar>(
    z: Complex<S>,
    w: &mut Work,
) -> Result<Complex<S>, SciOxideError> {
    let zero = S::ZERO;
    let one = S::ONE;
    let two = S::TWO;
    let x = z.re.abs();
    let y = z.im;
    let ispi = one / S::PI.sqrt();
    if x == zero {
        return Ok(Complex::new(
            axis(crate::error_probability::real_faddeeva::erfcx, y, w)?,
            z.re,
        ));
    }
    if y == zero {
        let dawson = axis(crate::error_probability::real_faddeeva::dawson, z.re, w)?;
        return Ok(Complex::new((-x * x).exp(), two * ispi * dawson));
    }
    if y > real::<S>(7.)?
        || (x > real::<S>(6.)?
            && (y > real::<S>(0.1)?
                || (x > real::<S>(8.)? && y > real::<S>(1e-10)?)
                || x > real::<S>(28.)?))
    {
        if x.max(y) > real::<S>(1e7)? {
            if !w.step() {
                return Ok(Complex::new(zero, zero));
            }
            return Ok(Complex::new(zero, ispi) / z);
        }
        let depth = (real::<S>(3.9)?
            + real::<S>(11.398)?
                / (real::<S>(0.08254)? * x + real::<S>(0.1421)? * y + real::<S>(0.2023)?))
        .floor()
        .try_to_usize_exact()?;
        let mut d = z;
        for n in (1..depth).rev() {
            if !w.step() {
                break;
            }
            d = z - Complex::new(S::try_from_usize_exact(n)? * S::HALF, zero) / d;
        }
        return Ok(Complex::new(zero, ispi) / d);
    }
    // t3 is the largest Gaussian. Recover t1 and t2 with exp(-2*a*x)^n,
    // whose factors never grow, avoiding overflow and two exponentials per term.
    // Evaluate t3 directly even when earlier terms underflow near the real axis.
    // Small-x sinh avoids cancellation in sum5-sum4.
    let a = real::<S>(0.518_321_480_430_085_9)?;
    let c = two * a / S::PI;
    let decay = (-two * a * x).exp();
    let mut decay_n = one;
    let mut s1 = zero;
    let mut s2 = zero;
    let mut s3 = zero;
    let mut sd = zero;
    let mut n = 0usize;
    while w.step() {
        n += 1;
        let an = a * S::try_from_usize_exact(n)?;
        let denom = an * an + y * y;
        let t3 = (-(an - x) * (an - x)).exp() / denom;
        decay_n *= decay;
        let t1 = t3 * decay_n;
        let t2 = t1 * decay_n;
        let diff = if x < real::<S>(5e-4)? {
            two * an * t1 * (two * an * x).sinh()
        } else {
            an * (t3 - t2)
        };
        s1 += t1;
        s2 += t2;
        s3 += t3;
        sd += diff;
        if an > x + one && an * t3 <= S::EPSILON * sd.abs().max(S::MIN_POSITIVE) {
            break;
        }
    }
    let e = (-x * x).exp();
    let base =
        e * S::try_from_f64_lossy(crate::error_probability::real_faddeeva::erfcx(y.into_f64()))?
            - c * y * s1;
    let xy = z.re * y;
    let coef = c * z.re * e;
    let sin_xy = xy.sin();
    let (sin_2xy, cos_2xy) = (two * xy).sin_cos();
    Ok(Complex::new(
        base * cos_2xy + coef * sin_xy * sinc(xy, sin_xy) + S::HALF * c * y * (s2 + s3),
        coef * sinc(two * xy, sin_2xy) - base * sin_2xy + S::HALF * c * sd.copysign(z.re),
    ))
}

/// Reflect in `f64`, keeping the Gaussian exponent as a separate scale.
fn reflection<S: RealScalar>(
    z: Complex<S>,
    w: &mut Work,
) -> Result<(Complex<f64>, f64), SciOxideError> {
    let point = Complex::new(z.re.into_f64(), z.im.into_f64());
    let (exponent, phase) = super::complex_error::gaussian(point)?;
    let carrier = if exponent.hi > S::MAX.into_f64().ln() * 0.5 {
        exponent.hi
    } else {
        0.0
    };
    let upper = upper(-point, w)?;
    let gaussian = (exponent - crate::support::wide::Wide::from(carrier)).exp();
    Ok((phase * (2.0 * gaussian) - upper * (-carrier).exp(), carrier))
}

/// Faddeeva derivatives need only the original argument until the detailed result is formed;
/// Voigt supplies its already scaled derivative and propagation magnitude.
#[derive(Clone, Copy)]
enum Derivative<S> {
    Faddeeva(Complex<S>),
    Voigt(Complex<S>, S),
}

#[derive(Clone, Copy)]
struct Finite<S> {
    derivative: Option<Derivative<S>>,
    complete: bool,
}

type DerivativeEstimate<S> = (Complex<S>, Bound<S>);

/// Recover a finite derivative even if an intermediate product or norm overflows.
fn derivative<S: RealScalar>(
    z: Complex<S>,
    v: Complex<S>,
) -> Result<Option<DerivativeEstimate<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let constant = S::TWO / S::PI.sqrt();
    let d = Complex::new(S::ZERO, constant) - scale(z * v, S::TWO);
    let magnitude = constant + S::TWO * z.norm() * v.norm();
    if d.is_finite() && magnitude.is_finite() {
        return Ok(Some((d, Bound::envelope(magnitude, 128.0)?)));
    }
    let product = ScaledValue::from_value(z)?
        .mul(ScaledValue::from_value(v)?)?
        .mul_real(scioxide_core::Scaled::from_value(S::TWO)?)?;
    let d = ScaledValue::from_value(Complex::new(S::ZERO, constant))?.sub(product)?;
    let projected = match (d.real().round(), d.imaginary().round()) {
        (Ok(re), Ok(im)) => Complex::new(re, im),
        _ => return Ok(None),
    };
    let magnitude = ScaledValue::from_value(z)?
        .magnitude()?
        .mul(ScaledValue::from_value(v)?.magnitude()?)?
        .mul(scioxide_core::Scaled::from_value(S::TWO)?)?
        .add(scioxide_core::Scaled::from_value(constant)?)?;
    let absolute = magnitude
        .mul(scioxide_core::Scaled::from_value(
            real::<S>(128.0)? * S::EPSILON,
        )?)?
        .round()?;
    Ok(Some((projected, Bound::calibrated_absolute(absolute))))
}

impl<S: RealScalar> Publish<Complex<S>> for Finite<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn publish(self, value: Option<Complex<S>>) -> Result<Forms<Complex<S>>, SciOxideError> {
        let Some(v) = value else {
            return Standard::Finite {
                bound: Bound::exact(),
                derivative: None,
            }
            .publish(None);
        };
        let norm = v.norm();
        // Keep the qualified 64-unit envelope and final subnormal rounding.
        let bound = if norm.is_finite() {
            error(norm)?
        } else {
            Bound::calibrated_absolute(
                ScaledValue::from_value(v)?
                    .magnitude()?
                    .mul(scioxide_core::Scaled::from_value(
                        real::<S>(64.)? * S::EPSILON,
                    )?)?
                    .round()?,
            )
        }
        .with_subnormal_rounding(norm, real::<S>(4.)?);
        let derivative = match self.derivative {
            Some(Derivative::Faddeeva(z)) => derivative(z, v)?,
            Some(Derivative::Voigt(d, magnitude)) if d.is_finite() => {
                Some((d, Bound::envelope(magnitude, 128.0)?))
            }
            _ => None,
        };
        Standard::Finite {
            bound: if self.complete {
                bound
            } else {
                Bound::heuristic(bound.absolute)
            },
            derivative,
        }
        .publish(Some(v))
    }
}

#[inline]
fn report<S: RealScalar>(
    v: Complex<S>,
    derivative: Option<Derivative<S>>,
    w: &mut Work,
    method: AlgorithmComponent,
) -> Result<Evaluation<Complex<S>, Finite<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    w.check()?;

    crate::support::check_finite(v.re)?;
    crate::support::check_finite(v.im)?;
    Ok(Evaluation::from_parts(
        Some(v),
        ScalarRepresentation::Finite,
        if w.complete {
            SpecialFunctionTermination::Converged
        } else {
            SpecialFunctionTermination::IterationLimit
        },
        EvaluationCounts::from_iterations(w.used),
        AlgorithmId::from_components(algorithm_component!("probability"), method, 1),
        Some(Finite {
            derivative,
            complete: w.complete,
        }),
    ))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Faddeeva function w(z) = exp(−z²) erfc(−iz).

```text
w(z) = exp(−z²) erfc(−iz) = exp(−z²) (1 + (2i/√π) ∫₀ᶻ exp(t²) dt)
```

(DLMF 7.2.3). w is entire. On the imaginary axis `w(iy) = erfcx(y)`; on the real axis
`Re w(x) = exp(−x²)` and `Im w(x) = (2/√π) D(x)`, where `D` is Dawson's integral.

In the closed upper half-plane the function is evaluated by the sums of Zaghloul and Ali
(Algorithm 916) and, for large `|z|`, by a continued fraction; the axes use the
[`erfcx`](crate::erfcx) and [`dawson`](crate::dawson) kernels. In the lower half-plane it uses
the reflection `w(z) = 2 exp(−z²) − w(−z)` (DLMF §7.4). There `exp(−z²)` grows without bound
where `|Im z| > |Re z|`, and [`faddeeva_with_options`](crate::faddeeva_with_options) returns
values beyond the scalar range in logarithmic form.

# Examples

```
// w(iy) = erfcx(y) on the imaginary axis
let w = scioxide_special::faddeeva(scioxide_core::Complex::new(0.0_f64, 1.0))?;
let expected = scioxide_special::erfcx(1.0_f64)?;
assert!((w.re - expected).abs() < 1e-15 && w.im.abs() < 1e-15);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target. The detailed form reports error bounds of 64 units for the value
and 128 units for the derivative ([error estimates](crate#error-estimates)), including
the reflection up to the overflow threshold of each precision. See the [crate accuracy
table](crate#accuracy)."###, r###"
- `NumericalInstability` (`Overflow`) if `Im z < 0` and `|w(z)|` exceeds the range of the scalar
  type.
- `NumericalInstability` (`LossOfSignificance`) if `Im z < 0` and `Re z · Im z` overflows, so the
  phase of `exp(−z²)` cannot be formed.

# References

- NIST Digital Library of Mathematical Functions, §7.2 and §7.4. <https://dlmf.nist.gov/7>
- M. R. Zaghloul and A. N. Ali, "Algorithm 916: Computing the Faddeyeva and Voigt functions", *ACM
  Trans. Math. Softw.* 38(2), 2011."###)]

    pub faddeeva [S: RealScalar] (z: Complex<S>) -> Complex<S>
    where [Complex<S>: NumericScalar<Real = S>, ];
    options options: ProbabilityOptions;
    detailed faddeeva_with_options -> Detailed<S>;
    value {
        faddeeva_value(z)
    }
    report {
        faddeeva_controlled(z, options, None)?.publish()
    }
    scoped pub faddeeva_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>) {
        crate::util::scoped(scope, options.iteration_budget(), |context| {
            faddeeva_controlled(z, options, Some(context))?.publish()
        })
    }
}

/// Checked ordinary projection of the shared upper-half-plane kernel.
#[inline]
fn faddeeva_value<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = ProbabilityOptions::default();
    if z.im < S::ZERO {
        return faddeeva_controlled(z, options, None)?.value();
    }
    require_finite_param(z.re, "probability parameter z must be finite")?;
    require_finite_param(z.im, "probability parameter z must be finite")?;
    let mut work = Work::new(options.iteration_budget().get());
    let value = upper(z, &mut work)?;
    work.check()?;
    crate::support::check_finite(value.re)?;
    crate::support::check_finite(value.im)?;
    if !work.complete {
        return Err(crate::result::exhausted_work(work.used));
    }
    Ok(value)
}

fn faddeeva_controlled<S: RealScalar>(
    z: Complex<S>,
    options: ProbabilityOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<FaddeevaEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    require_finite_param(z.re, "probability parameter z must be finite")?;
    require_finite_param(z.im, "probability parameter z must be finite")?;
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    if z.im < S::ZERO {
        let (v, exponent) = reflection(z, &mut w)?;
        return exponential_report(v, exponent, z, &mut w);
    }
    let v = upper(z, &mut w)?;
    Ok(report(
        v,
        Some(Derivative::Faddeeva(z)),
        &mut w,
        algorithm_component!("faddeeva_916_fraction"),
    )?
    .map_form(Either::Left))
}

/// Reject nonfinite Voigt arguments and invalid widths.
fn widths<S: RealScalar>(x: S, sigma: S, gamma: S) -> Result<(), SciOxideError> {
    require_finite_param(x, "probability parameter x must be finite")?;
    require_finite_param(sigma, "probability parameter sigma must be finite")?;
    require_finite_param(gamma, "probability parameter gamma must be finite")?;
    if sigma < S::ZERO || (sigma == S::ZERO && gamma == S::ZERO) {
        return Err(ERRORS.outside(scioxide_core::argument!("sigma")));
    }
    if gamma < S::ZERO {
        return Err(ERRORS.outside(scioxide_core::argument!("gamma")));
    }
    Ok(())
}

/// Whether the Gaussian width is negligible beside the Lorentzian scale.
fn lorentzian<S: RealScalar>(x: S, sigma: S, gamma: S) -> bool {
    sigma == S::ZERO || sigma / x.abs().max(gamma) < S::SQRT_EPSILON * S::HALF
}

/// The Faddeeva argument `(x + iγ)/(σ√2)` and the scale `1/(σ√2)`.
fn voigt_argument<S: RealScalar>(
    x: S,
    sigma: S,
    gamma: S,
) -> Result<(Complex<S>, S), SciOxideError> {
    let root = real::<S>(2.)?.sqrt();
    let z = Complex::new((x / sigma) / root, (gamma / sigma) / root);
    require_finite_param(z.re, "probability parameter x must be finite")?;
    require_finite_param(z.im, "probability parameter gamma must be finite")?;
    Ok((z, (S::ONE / sigma) / root))
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the complex Voigt function, the Faddeeva function scaled to a unit-area line shape.

```text
V(x; σ, γ) = w(z) / (σ√(2π)),   z = (x + iγ) / (σ√2),
```

where `w` is the [Faddeeva function](crate::faddeeva), `σ ≥ 0` is the standard deviation of
the Gaussian and `γ ≥ 0` the half-width at half-maximum of the Lorentzian. `Re V` is the Voigt
profile, the convolution of the two densities, and integrates to one over `x`; `Im V` is the
dispersion profile. At `σ = 0` the result is the Lorentzian limit `(γ + ix)/(π(x² + γ²))`; at
`γ = 0` the real part is the normal density with standard deviation `σ`.

The Lorentzian limit is also used when `σ < ½√ε · max(|x|, γ)`, with `ε` the machine epsilon,
because the Gaussian width is then below the resolution of the result.

# Examples

```
// σ = 1, γ = 0: the peak of the standard normal density, 1/√(2π)
let v = scioxide_special::voigt_complex(0.0_f64, 1.0, 0.0)?;
let peak = 1.0 / (2.0 * core::f64::consts::PI).sqrt();
assert!((v.re - peak).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target; the detailed form reports a propagated error bound for the value
and the derivative. See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `σ < 0`, if `γ < 0`, or if `σ = γ = 0`.

# References

- NIST Digital Library of Mathematical Functions, §7.19. <https://dlmf.nist.gov/7.19>"###)]

    pub voigt_complex [S: RealScalar] (x: S, sigma: S, gamma: S) -> Complex<S>
    where [Complex<S>: NumericScalar<Real = S>, ];
    options options: ProbabilityOptions;
    detailed voigt_complex_with_options -> Detailed<S>;
    evaluate |options, context| {
        voigt_complex_evaluate(x, sigma, gamma, options, context)
    }
    scoped pub voigt_complex_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Evaluates the detailed result before it is projected and returned.
fn voigt_complex_evaluate<S: RealScalar>(
    x: S,
    sigma: S,
    gamma: S,
    options: ProbabilityOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<Complex<S>, Finite<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    widths(x, sigma, gamma)?;
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    let (v, d) = if lorentzian(x, sigma, gamma) {
        w.step();
        let z = Complex::new(gamma, -x);
        let v = Complex::new(S::ONE / S::PI, S::ZERO) / z;
        let d = Complex::new(S::ZERO, S::ONE) * v / z;
        (v, Some((d, d.norm())))
    } else {
        let (z, scale) = voigt_argument(x, sigma, gamma)?;
        let raw = upper(z, &mut w)?;
        let factor = scale / S::PI.sqrt();
        (
            self::scale(raw, factor),
            Some((
                self::scale(
                    Complex::new(S::ZERO, real::<S>(2.)? / S::PI.sqrt())
                        - self::scale(z * raw, real::<S>(2.)?),
                    scale * factor,
                ),
                scale
                    * factor
                    * (real::<S>(2.)? / S::PI.sqrt() + real::<S>(2.)? * z.norm() * raw.norm()),
            )),
        )
    };
    report(
        v,
        d.map(|(d, magnitude)| Derivative::Voigt(d, magnitude)),
        &mut w,
        algorithm_component!("voigt"),
    )
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Voigt profile, the real part of [`voigt_complex`](crate::voigt_complex).

```text
V(x; σ, γ) = Re w((x + iγ)/(σ√2)) / (σ√(2π))
```

is the convolution of a normal density with standard deviation `σ ≥ 0` and a Lorentzian
(Cauchy) density with half-width at half-maximum `γ ≥ 0`; it integrates to one over `x`. At
`σ = 0` it is the Lorentzian `γ/(π(x² + γ²))`, and at `γ = 0` the normal density. The
evaluation and the limits are those of [`voigt_complex`](crate::voigt_complex), without the
imaginary part.

# Examples

```
// σ = 0, γ = 1: the peak of the standard Cauchy density, 1/π
let v = scioxide_special::voigt_profile(0.0_f64, 0.0, 1.0)?;
assert!((v - 1.0 / core::f64::consts::PI).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"
No uniform ulp target. The detailed form reports a bound of 64 units ([error
estimates](crate#error-estimates)). See the [crate accuracy table](crate#accuracy)."###, r###"
- `DomainError` (`OutsideDomain`) if `σ < 0`, if `γ < 0`, or if `σ = γ = 0`.

# References

- NIST Digital Library of Mathematical Functions, §7.19. <https://dlmf.nist.gov/7.19>"###)]

    pub voigt_profile [S: RealScalar] (x: S, sigma: S, gamma: S) -> S
    where [Complex<S>: NumericScalar<Real = S>, ];
    options options: ProbabilityOptions;
    detailed voigt_profile_with_options -> SpecialFunctionResult<S>;
    evaluate |options, context| {
        voigt_profile_evaluate(x, sigma, gamma, options, context)
    }
    scoped pub voigt_profile_scoped ['scope, 'cap] (scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>);
}

/// Evaluate the real part alone.
fn voigt_profile_evaluate<S: RealScalar>(
    x: S,
    sigma: S,
    gamma: S,
    options: ProbabilityOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    widths(x, sigma, gamma)?;
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    let v = if lorentzian(x, sigma, gamma) {
        w.step();
        // γ/(π(x² + γ²)), scaled through the hypotenuse.
        let h = x.hypot(gamma);
        (gamma / h) / (S::PI * h)
    } else {
        let (z, scale) = voigt_argument(x, sigma, gamma)?;
        upper(z, &mut w)?.re * (scale / S::PI.sqrt())
    };
    w.check()?;
    finish(v, error(v)?, &w, algorithm_component!("voigt_profile"))
}

/// Evaluation of the Faddeeva function: a standard finite form, or a reflection beyond the ordinary
/// range.
type FaddeevaEvaluation<S> = Evaluation<Complex<S>, Either<Finite<S>, Reflection<S>>>;

/// The reflection `v·exp(exponent)` beyond the ordinary scalar range.
#[derive(Clone, Copy)]
struct Reflection<S> {
    v: Complex<S>,
    exponent: S,
}

fn exponential_report<S: RealScalar>(
    v: Complex<f64>,
    exponent: f64,
    z: Complex<S>,
    w: &mut Work,
) -> Result<FaddeevaEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    w.check()?;
    // No logarithm of the significand: retain its precision through the final
    // destination rounding, including values near the overflow boundary.
    let projection = || -> Result<Complex<S>, SciOxideError> {
        if exponent == 0.0 {
            return Ok(Complex::new(real(v.re)?, real(v.im)?));
        }
        let factor = scioxide_core::Scaled::from_log(1.0, exponent)?;
        Ok(Complex::new(
            factor
                .mul(scioxide_core::Scaled::from_value(v.re)?)?
                .round_to()?,
            factor
                .mul(scioxide_core::Scaled::from_value(v.im)?)?
                .round_to()?,
        ))
    };
    let projected = match projection() {
        Ok(value) => Some(value),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow
                        | scioxide_core::NumericalIssue::ExponentOutOfRange,
                    ..
                }
            ) =>
        {
            None
        }
        Err(error) => return Err(error),
    };
    if let Some(projected) = projected {
        return Ok(report(
            projected,
            Some(Derivative::Faddeeva(z)),
            w,
            algorithm_component!("faddeeva_reflection"),
        )?
        .map_form(Either::Left));
    }
    let v = Complex::new(real::<S>(v.re)?, real::<S>(v.im)?);
    let exponent = real::<S>(exponent)?;
    let termination = if w.complete {
        SpecialFunctionTermination::Converged
    } else {
        SpecialFunctionTermination::IterationLimit
    };
    Ok(Evaluation::from_parts(
        None,
        ScalarRepresentation::Overflow,
        termination,
        EvaluationCounts::from_iterations(w.used),
        scioxide_core::algorithm_id!("probability", "faddeeva_scaled_reflection", 1),
        Some(Either::Right(Reflection { v, exponent })),
    ))
}

impl<S: RealScalar> Publish<Complex<S>> for Reflection<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn publish(self, _value: Option<Complex<S>>) -> Result<Forms<Complex<S>>, SciOxideError> {
        let Self { v, exponent } = self;
        let base_log = LogValue::from_value(v)?;
        let log = crate::support::check_finite(
            base_log.log_magnitude().ok_or_else(|| {
                SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "nonzero reflection magnitude",
                )
            })? + exponent,
        )?;
        let rel = (real::<S>(64.)? + real::<S>(4.)? * exponent.abs()) * S::EPSILON;
        let reliability = EstimateReliability::ModelBased;
        let scaled = match scioxide_core::Scaled::from_log(S::ONE, exponent) {
            Ok(magnitude) => {
                let scaled = ScaledValue::from_value(v)?.mul_real(magnitude)?;
                Some(ValueWithError::new(
                    scaled,
                    rel * scaled.scaled_magnitude(),
                    Some(rel),
                    reliability,
                )?)
            }
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::ExponentOutOfRange,
                        ..
                    }
                ) =>
            {
                None
            }
            Err(error) => return Err(error),
        };
        Ok(Forms {
            value: None,
            scaled,
            logarithmic: Some(ValueWithError::new(
                LogValue::from_scaled_log(base_log.scaled_phase(), log)?,
                rel,
                None,
                reliability,
            )?),
            derivative: None,
        })
    }
}
