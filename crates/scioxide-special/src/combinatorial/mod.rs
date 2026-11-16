//! Factorials, binomial coefficients, Pochhammer symbols, Stirling, Bernoulli and Euler numbers.
use crate::ValueWithError;
use crate::combinatorial::combinatorial_tables::{BERNOULLI, EULER};
use crate::result::{Evaluation, Forms, Publish};
use crate::support::COMBINATORIAL_ERRORS as ERRORS;
use crate::support::Work;
use crate::support::is_real;
use crate::support::narrow;
use crate::support::pole;
use crate::support::wide::Wide;
use crate::support::widen_numeric as widen;
use crate::support::{C, ONE, ZERO, real};
use crate::{
    CombinatorialOptions, ScalarRepresentation, SpecialFunctionResult, SpecialFunctionTermination,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::Scaled;
use scioxide_core::backend::scaled::factor_common_scale;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, DomainReason, EstimateReliability, EvaluationCounts, LogValue,
    NumericScalar, RealScalar, ScaledValue, SciOxideError, algorithm_component,
};

const EPS: f64 = f64::EPSILON;
type Report<T> = SpecialFunctionResult<T>;

use crate::support::invalid;

#[inline]
fn integer(n: usize) -> Wide {
    integer_u64(n as u64)
}
#[inline]
fn integer_u64(n: u64) -> Wide {
    Wide::from((n >> 32) as f64).mul_f64(4294967296.0) + Wide::from((n & 0xffff_ffff) as f64)
}
use crate::support::constants::LN_2;
fn pi() -> Wide {
    crate::support::constants::PI
}

/// Returns the Bernoulli number `B_n` as an exact reduced fraction `(numerator, denominator)`.
///
/// The convention is `t / (eᵗ − 1) = Σ B_n tⁿ / n!` (DLMF 24.2.1), so `B₁ = −1/2`. Odd orders
/// above one give `(0, 1)`. Even orders `n ≤ 34` and `n = 38` are available from a table; the
/// numerator of `B₃₆` does not fit in 64 bits. The denominator is positive. For a floating-point
/// value of any order use [`crate::bernoulli`].
///
/// # Errors
///
/// - `SizeOverflow` for even `n = 36` or `n > 38`, or if the numerator or denominator does not
///   fit in `isize` or `usize`.
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_special::checked_bernoulli(1)?, (-1, 2));
/// // B₁₂ = −691/2730 (DLMF Table 24.2.1)
/// assert_eq!(scioxide_special::checked_bernoulli(12)?, (-691, 2730));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - NIST DLMF, §24.2. <https://dlmf.nist.gov/24.2>
pub fn checked_bernoulli(n: usize) -> Result<(isize, usize), SciOxideError> {
    let (p, q) = if n == 38 {
        // Reduced numerators are not monotone: B36 overflows i64, but B38 fits.
        (2_929_993_913_841_559, 6)
    } else if n == 1 {
        (-1, 2)
    } else if !n.is_multiple_of(2) {
        (0, 1)
    } else {
        *BERNOULLI.get(n / 2).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("exact_bernoulli_number"))
        })?
    };
    Ok((
        isize::try_from(p).map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("bernoulli_numerator"))
        })?,
        usize::try_from(q).map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("bernoulli_denominator"))
        })?,
    ))
}
/// Returns the Euler number `E_n` as an exact integer.
///
/// The convention is `sech t = Σ E_n tⁿ / n!` (DLMF §24.2), so `E₂ = −1`. Odd orders give 0.
/// Even orders `n ≤ 22` are available from a table. For a floating-point value of any order
/// use [`crate::euler`].
///
/// # Errors
///
/// - `SizeOverflow` for even `n > 22`, or if the value does not fit in `isize`.
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_special::checked_euler(2)?, -1);
/// assert_eq!(scioxide_special::checked_euler(4)?, 5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - NIST DLMF, §24.2. <https://dlmf.nist.gov/24.2>
pub fn checked_euler(n: usize) -> Result<isize, SciOxideError> {
    if !n.is_multiple_of(2) {
        return Ok(0);
    }
    EULER
        .get(n / 2)
        .copied()
        .and_then(|v| isize::try_from(v).ok())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("exact_euler_number")))
}

#[derive(Clone, Copy)]
struct Candidate {
    ordinary: Option<C>,
    scaled: Option<ScaledValue<C>>,
    logarithmic: LogValue<C, f64>,
    relative: f64,
    absolute: f64,
    log_error: f64,
    method: AlgorithmComponent,
}
fn project(value: ScaledValue<C>) -> Result<Option<C>, SciOxideError> {
    match value.round() {
        Ok(value) => Ok(Some(value)),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}
impl Candidate {
    fn value(v: C, relative: f64, method: AlgorithmComponent) -> Result<Self, SciOxideError> {
        let logarithmic = LogValue::from_value(v)?;
        Ok(Self {
            ordinary: Some(v),
            scaled: Some(ScaledValue::from_value(v)?),
            logarithmic,
            relative,
            absolute: 0.0,
            log_error: relative + 4.0 * EPS * logarithmic.log_magnitude().unwrap_or(0.0).abs(),
            method,
        })
    }
    fn scaled(
        value: ScaledValue<C>,
        relative: f64,
        method: AlgorithmComponent,
    ) -> Result<Self, SciOxideError> {
        let ordinary = project(value)?;
        let logarithmic = if value.is_zero() {
            LogValue::from_value(value.round()?)?
        } else {
            // Factor only into scaled components. Materializing a shared IEEE
            // mantissa would erase mixed-scale directions before normalization.
            let ([re, im], exponent) = factor_common_scale([value.real(), value.imaginary()])?;
            let norm = ScaledValue::<C>::from_components(re, im).magnitude()?;
            let phase = ScaledValue::from_components(re.div(norm)?, im.div(norm)?);
            let exponent = i128::from(exponent) + i128::from(norm.exponent());
            let log = (Wide::from(norm.significand()).ln() + LN_2.mul_f64(exponent as f64)).value();
            LogValue::from_scaled_log(phase, log)?
        };
        Ok(Self {
            ordinary,
            scaled: Some(value),
            logarithmic,
            relative,
            absolute: 0.0,
            log_error: relative + 4.0 * EPS * logarithmic.log_magnitude().unwrap_or(0.0).abs(),
            method,
        })
    }
    fn logarithmic(
        phase: ScaledValue<C>,
        log: Wide,
        error: f64,
        method: AlgorithmComponent,
    ) -> Result<Self, SciOxideError> {
        if !log.value().is_finite() {
            return Err(SciOxideError::numerical_instability(
                if log.value().is_infinite() {
                    scioxide_core::NumericalIssue::Overflow
                } else {
                    scioxide_core::NumericalIssue::LossOfSignificance
                },
                "the logarithmic result exceeds the supported numerical range",
            ));
        }
        let logarithmic = LogValue::from_scaled_log(phase, log.value())?;
        fn logarithmic_scale(log: Wide) -> Result<Scaled<f64>, SciOxideError> {
            let ordinary = log.exp();
            if ordinary.is_normal() {
                Scaled::from_value(ordinary)
            } else {
                Scaled::from_log(1.0, log.hi)?.mul(Scaled::from_log(1.0, log.lo)?)
            }
        }
        let scale = logarithmic_scale(log);
        let scaled = match scale.and_then(|scale| phase.mul_real(scale)) {
            Ok(value) => Some(value),
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
        let ordinary = scaled.map(project).transpose()?.flatten();
        Ok(Self {
            ordinary,
            scaled,
            logarithmic,
            relative: error,
            absolute: 0.0,
            log_error: error + EPS * log.hi.abs(),
            method,
        })
    }
}
/// Evaluation of a combinatorial function.
pub(crate) type CombinatorialEvaluation<T> = Evaluation<T, CandidateForm>;

/// A candidate with the relative error of its narrowed ordinary value.
#[derive(Clone, Copy)]
pub(crate) struct CandidateForm {
    c: Candidate,
    relative: f64,
}

fn report<T: NumericScalar>(
    candidate: Result<Candidate, SciOxideError>,
    work: &Work<'_, '_, '_>,
) -> Result<CombinatorialEvaluation<T>, SciOxideError> {
    if work.exhausted() {
        return Err(crate::result::exhausted_work(work.used));
    }
    let c = candidate?;
    let maximum = T::Real::MAX.into_f64();
    let is_real = is_real::<T>();
    if is_real
        && !c
            .logarithmic
            .scaled_phase()
            .imaginary()
            .abs_le(Scaled::from_value(128.0 * EPS)?)
    {
        return Err(crate::support::instability(
            "computed combinatorial real projection has a significant imaginary component",
        ));
    }
    let mut relative = c.relative;
    // A nonzero magnitude below the extended exponent range rounds to a zero
    // signed like its phase.
    let ordinary = match c.ordinary {
        None if c.logarithmic.log_magnitude().is_some_and(|log| log < 0.0) => {
            let phase = c.logarithmic.scaled_phase();
            Some(C::new(
                0.0_f64.copysign(phase.real().significand()),
                0.0_f64.copysign(phase.imaginary().significand()),
            ))
        }
        ordinary => ordinary,
    };
    let projected = ordinary
        .filter(|v| v.re.abs().max(v.im.abs()) <= maximum)
        .map(narrow::<T>)
        .transpose()?;
    // A rounded zero carries its rounding as an absolute bound at publication;
    // the scaled and logarithmic forms keep the relative error.
    let rounded_zero = projected.is_some_and(|v| v.abs() == T::Real::ZERO)
        && c.logarithmic.log_magnitude().is_some();
    match (projected, ordinary) {
        (Some(v), Some(ordinary)) if !rounded_zero => {
            let rounded = widen(v, "combinatorial parameter v must be finite")?;
            if ordinary != ZERO && rounded != ordinary {
                relative += (rounded - ordinary).norm() / ordinary.norm();
            }
        }
        _ if T::Real::EPSILON.into_f64() > EPS => relative += T::Real::EPSILON.into_f64(),
        _ => {}
    }
    let termination = if relative == 0.0 && c.absolute == 0.0 && !rounded_zero {
        SpecialFunctionTermination::Exact
    } else {
        SpecialFunctionTermination::Converged
    };
    let representation = if projected.is_some() {
        ScalarRepresentation::Finite
    } else {
        ScalarRepresentation::Overflow
    };
    Ok(Evaluation::from_parts(
        projected,
        representation,
        termination,
        EvaluationCounts::from_iterations(work.used),
        AlgorithmId::from_components(algorithm_component!("combinatorial"), c.method, 1),
        Some(CandidateForm { c, relative }),
    ))
}

impl<T: NumericScalar> Publish<T> for CandidateForm {
    fn publish(self, value: Option<T>) -> Result<Forms<T>, SciOxideError> {
        let Self { c, relative } = self;
        let maximum = T::Real::MAX.into_f64();
        let bounded = |x: f64| T::Real::try_from_f64_lossy(x.min(maximum));
        // A nonzero result that rounds to zero is covered by one quantum.
        let rounded_zero =
            |v: T| v.abs() == T::Real::ZERO && c.logarithmic.log_magnitude().is_some();
        let value = value
            .map(|v| {
                if rounded_zero(v) {
                    return ValueWithError::new(
                        v,
                        bounded(c.absolute)? + T::Real::MIN_SUBNORMAL,
                        None,
                        EstimateReliability::ModelBased,
                    );
                }
                ValueWithError::new(
                    v,
                    bounded(c.absolute + v.abs().into_f64() * relative)?,
                    Some(bounded(relative)?),
                    EstimateReliability::ModelBased,
                )
            })
            .transpose()?;
        let scaled = c
            .scaled
            .map(|s| {
                let v = ScaledValue::from_component_parts(
                    narrow::<T>(s.component_significands())?,
                    s.component_exponents(),
                )?;
                ValueWithError::new(
                    v,
                    bounded(v.scaled_magnitude().into_f64() * relative)?,
                    Some(bounded(relative)?),
                    EstimateReliability::ModelBased,
                )
            })
            .transpose()?;
        let log = c
            .logarithmic
            .log_magnitude()
            .map(T::Real::try_from_f64_lossy)
            .transpose()?;
        let log_rounding = log
            .zip(c.logarithmic.log_magnitude())
            .map_or(0.0, |(a, b)| (a.into_f64() - b).abs());
        let phase = c.logarithmic.scaled_phase();
        let phase = ScaledValue::from_component_parts(
            narrow::<T>(phase.component_significands())?,
            phase.component_exponents(),
        )?;
        Ok(Forms {
            value,
            scaled,
            logarithmic: Some(ValueWithError::new(
                match log {
                    Some(log) => LogValue::from_scaled_log(phase, log)?,
                    None => LogValue::from_value(phase.round()?)?,
                },
                bounded(c.log_error + log_rounding)?,
                None,
                EstimateReliability::ModelBased,
            )?),
            derivative: None,
        })
    }
}

fn product_report(
    n: usize,
    factor: impl FnMut(usize) -> C,
    work: &mut Work<'_, '_, '_>,
    method: AlgorithmComponent,
) -> Result<Candidate, SciOxideError> {
    let (value, relative) = scaled_product(n, factor, work)?;
    Candidate::scaled(value, relative, method)
}
fn scaled_product(
    n: usize,
    mut factor: impl FnMut(usize) -> C,
    work: &mut Work<'_, '_, '_>,
) -> Result<(ScaledValue<C>, f64), SciOxideError> {
    enum Product {
        Real { mantissa: Wide, exponent: i64 },
        Complex(ScaledValue<scioxide_core::Complex<f64>>),
    }
    let mut product = Product::Real {
        mantissa: Wide::from(1.0),
        exponent: 0,
    };
    for i in 0..n {
        work.try_step()?;
        let value = factor(i);
        if value == ZERO {
            return Ok((ScaledValue::from_value(ZERO)?, 0.0));
        }
        product = match product {
            Product::Real { mantissa, exponent } if value.im == 0.0 => {
                let factor = Scaled::from_value(value.re)?;
                let mantissa = mantissa * Wide::from(factor.significand());
                let normalized = Scaled::from_value(mantissa.value())?;
                let combined = i128::from(exponent)
                    + i128::from(factor.exponent())
                    + i128::from(normalized.exponent());
                let exponent = i64::try_from(combined).map_err(|_| {
                    SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::ExponentOutOfRange,
                        "normalized product exponent is outside i64",
                    )
                })?;
                let shift = -i32::try_from(normalized.exponent()).map_err(|_| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("product_exponent"))
                })?;
                Product::Real {
                    mantissa: Wide {
                        hi: mantissa.hi.scalbn(shift),
                        lo: mantissa.lo.scalbn(shift),
                    },
                    exponent,
                }
            }
            Product::Real { mantissa, exponent } => Product::Complex(
                ScaledValue::from_components(
                    Scaled::from_parts(mantissa.value(), exponent)?,
                    Scaled::ZERO,
                )
                .mul(ScaledValue::from_value(value)?)?,
            ),
            Product::Complex(product) => {
                Product::Complex(product.mul(ScaledValue::from_value(value)?)?)
            }
        };
    }
    let (value, relative) = match product {
        Product::Real { mantissa, exponent } => (
            ScaledValue::from_parts(real(mantissa.value()), exponent)?,
            if n == 0 {
                0.0
            } else {
                (4.0 + 2.0 * n as f64) * EPS
            },
        ),
        Product::Complex(value) => (value, 8.0 * EPS * n as f64),
    };
    Ok((value, relative))
}

fn factorial_log(n: usize, work: &mut Work<'_, '_, '_>) -> Result<Wide, SciOxideError> {
    work.steps(80)?;
    Ok(crate::gamma_beta::gamma::real::ln_positive_wide(
        integer(n) + Wide::from(1.0),
    ))
}
fn multifactorial_kernel(
    n: usize,
    step: usize,
    w: &mut Work<'_, '_, '_>,
) -> Result<Candidate, SciOxideError> {
    if step == 0 {
        return Err(invalid(
            scioxide_core::argument!("step"),
            "multifactorial step must be positive",
        ));
    }
    if step == 1 && n < crate::gamma_beta::gamma::SMALL_FACTORIALS_F64.len() {
        return Candidate::value(
            real(crate::gamma_beta::gamma::SMALL_FACTORIALS_F64[n]),
            0.0,
            algorithm_component!("factorial_table"),
        );
    }
    if n == 0 {
        return Candidate::value(ONE, 0.0, algorithm_component!("empty_product"));
    }
    let count = (n - 1) / step + 1;
    if count <= 128 {
        return product_report(
            count,
            |i| real((n - i * step) as f64),
            w,
            algorithm_component!("multifactorial_product"),
        );
    }
    w.steps(160)?;
    let a = integer((n - 1) % step + 1) / integer(step);
    let m = integer(count);
    let log = m * integer(step).ln() + crate::gamma_beta::gamma::real::ln_positive_wide(a + m)
        - crate::gamma_beta::gamma::real::ln_positive_wide(a);
    Candidate::logarithmic(
        ScaledValue::from_value(ONE)?,
        log,
        8.0 * EPS * (1.0 + log.hi.abs()),
        algorithm_component!("multifactorial_gamma"),
    )
}

fn gamma_ratio_positive(a: Wide, x: Wide, w: &mut Work<'_, '_, '_>) -> Result<Wide, SciOxideError> {
    w.steps(160)?;
    let b = a + x;
    Ok(if a.hi >= 32.0 && b.hi >= 32.0 {
        x * a.ln() + (b - Wide::from(0.5)) * (x / a).ln_1p() - x
            + crate::support::gamma_series::stirling_wide(b)
            - crate::support::gamma_series::stirling_wide(a)
    } else {
        crate::gamma_beta::gamma::real::ln_positive_wide(b)
            - crate::gamma_beta::gamma::real::ln_positive_wide(a)
    })
}
fn psi(z: C, w: &mut Work<'_, '_, '_>) -> Result<C, SciOxideError> {
    if pole(z) {
        return Err(ERRORS.domain(scioxide_core::argument!("a"), DomainReason::Pole));
    }
    if z.im == 0.0 {
        w.steps(80)?;
        return Ok(real(crate::gamma_beta::gamma::digamma_kernel(z.re)?));
    }
    crate::gamma_beta::gamma::complex_log_gamma(z, &mut || w.try_step()).map(|(_, d)| d)
}
fn small_log_ratio(a: C, x: C, w: &mut Work<'_, '_, '_>) -> Result<C, SciOxideError> {
    // Eight-point Gauss--Legendre integral of psi along the short segment.
    // This preserves x even when a+x rounds back to a.
    const NODES: [(f64, f64); 4] = [
        (0.1834346424956498, 0.362683783378362),
        (0.525532409916329, 0.3137066458778873),
        (0.7966664774136267, 0.2223810344533745),
        (0.9602898564975363, 0.1012285362903763),
    ];
    let mut sum = ZERO;
    for (node, weight) in NODES {
        for sign in [-1.0, 1.0] {
            sum += psi(a + x * real((1.0 + sign * node) * 0.5), w)? * real(weight * 0.5);
        }
    }
    Ok(x * sum)
}
use crate::support::complex_expm1;

/// Reflection ratio with the common integer part removed before adding x.
/// Both sine arguments keep the displacement from the nearest gamma pole.
fn reflected_short_ratio(
    a: C,
    x: C,
    relative: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Candidate, SciOxideError> {
    let reduced = C::new(a.re - a.re.round(), a.im);
    let denominator = scioxide_core::sin_pi(reduced + x)?;
    let sine_ratio = scioxide_core::sin_pi(reduced)? / denominator;
    let l = -small_log_ratio(ONE - a, -x, w)?;
    let method = algorithm_component!("pochhammer_complex_reflection");
    if relative && l.re < 700.0 {
        w.steps(32)?;
        // sin(A)-sin(A+X) = -2 cos(A+X/2) sin(X/2).
        let sine_relative = -real(2.0)
            * scioxide_core::cos_pi(reduced + x * 0.5)?
            * (scioxide_core::sin_pi(x * 0.5)? / x)
            / denominator;
        let t1 = sine_relative * l.exp();
        let t2 = complex_expm1(l) / x;
        let value = t1 + t2;
        let error = 128.0
            * EPS
            * (1.0 + l.norm())
            * (1.0 + (t1.norm() + t2.norm()) / value.norm().max(f64::MIN_POSITIVE));
        return Candidate::value(value, error, method);
    }
    let sine = LogValue::from_value(sine_ratio)?;
    let log_sine = sine.log_magnitude().ok_or_else(|| ERRORS.unstable())?;
    let ratio = Candidate::logarithmic(
        sine.scaled_phase()
            .mul(ScaledValue::from_value(C::new(l.im.cos(), l.im.sin()))?)?,
        Wide::from(l.re) + Wide::from(log_sine),
        128.0 * EPS * (1.0 + l.norm() + log_sine.abs()),
        method,
    )?;
    if relative {
        relative_from_candidate(ratio, x)
    } else {
        Ok(ratio)
    }
}

fn pochhammer_kernel(
    a: C,
    x: C,
    relative: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Candidate, SciOxideError> {
    if x == ZERO {
        return if relative {
            Candidate::value(
                psi(a, w)?,
                32.0 * EPS,
                algorithm_component!("pochhammer_relative_limit"),
            )
        } else {
            Candidate::value(ONE, 0.0, algorithm_component!("empty_product"))
        };
    }
    let integer_shift = x.im == 0.0 && x.re == x.re.floor();
    let small_integer = integer_shift && x.re.abs() <= 128.0;
    if small_integer && !relative {
        if x.re >= 0.0 {
            return product_report(
                x.re as usize,
                |i| a + real(i as f64),
                w,
                algorithm_component!("pochhammer_product"),
            );
        }
        let result = product_report(
            (-x.re) as usize,
            |i| a - real((i + 1) as f64),
            w,
            algorithm_component!("pochhammer_reciprocal_product"),
        )?;
        if result.logarithmic.log_magnitude().is_none() {
            return Err(ERRORS.domain(scioxide_core::argument!("a"), DomainReason::Pole));
        }
        let s = result
            .scaled
            .ok_or_else(|| crate::support::COMBINATORIAL_ERRORS.unstable())?;
        return Candidate::scaled(
            ScaledValue::from_value(scioxide_core::Complex::new(
                1.0,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
            ))?
            .div(s)?,
            result.relative + 4.0 * EPS,
            algorithm_component!("pochhammer_reciprocal_product"),
        );
    }
    if integer_shift && x.re > 0.0 && pole(a) {
        if a.re + x.re > 0.0 {
            return Candidate::value(
                if relative { -ONE / x } else { ZERO },
                if relative { 4.0 * EPS } else { 0.0 },
                algorithm_component!("pochhammer_polynomial_zero"),
            );
        }
        let log = gamma_ratio_positive(Wide::from(1.0 - a.re - x.re), Wide::from(x.re), w)?;
        let sign = if (x.re % 2.0) == 0.0 { ONE } else { -ONE };
        let c = Candidate::logarithmic(
            ScaledValue::from_value(sign)?,
            log,
            8.0 * EPS * (1.0 + log.hi.abs()),
            algorithm_component!("pochhammer_negative_integer"),
        )?;
        return if relative {
            relative_from_candidate(c, x)
        } else {
            Ok(c)
        };
    }
    let b = a + x;
    if pole(b) {
        return Err(ERRORS.domain(scioxide_core::argument!("x"), DomainReason::Pole));
    }
    if pole(a) {
        return Candidate::value(
            if relative { -ONE / x } else { ZERO },
            if relative { 4.0 * EPS } else { 0.0 },
            algorithm_component!("reciprocal_gamma_zero"),
        );
    }
    let distance = if a.re <= 0.0 {
        (a.re - a.re.round()).hypot(a.im).min(1.0)
    } else {
        a.norm().min(1.0)
    };
    let shift = (4.5 - a.re.min(b.re)).ceil().max(0.0);
    let shift = if (a.im != 0.0 || x.im != 0.0) && shift <= 64.0 {
        shift as usize
    } else {
        0
    };
    if shift == 0 && x.norm() <= 0.01 * distance {
        let l = small_log_ratio(a, x, w)?;
        if relative {
            w.steps(32)?;
            return Candidate::value(
                complex_expm1(l) / x,
                64.0 * EPS,
                algorithm_component!("pochhammer_relative_short_shift"),
            );
        }
        let phase = C::new(l.im.cos(), l.im.sin());
        return Candidate::logarithmic(
            ScaledValue::from_value(phase)?,
            Wide::from(l.re),
            64.0 * EPS * (1.0 + l.norm()),
            algorithm_component!("pochhammer_short_shift"),
        );
    }
    let candidate = if a.im == 0.0 && x.im == 0.0 && a.re > 0.0 && b.re > 0.0 {
        let l = gamma_ratio_positive(Wide::from(a.re), Wide::from(x.re), w)?;
        if relative && l.hi.abs() < 700.0 {
            return Candidate::value(
                real((l.exp_m1() / Wide::from(x.re)).value()),
                16.0 * EPS,
                algorithm_component!("pochhammer_relative_expm1"),
            );
        }
        Candidate::logarithmic(
            ScaledValue::from_value(ONE)?,
            l,
            8.0 * EPS * (1.0 + l.hi.abs()),
            algorithm_component!("pochhammer_combined_gamma"),
        )?
    } else if a.im == 0.0 && x.im == 0.0 {
        w.steps(160)?;
        let log_gamma = |z: Wide| {
            if z.hi > 0.0 {
                crate::gamma_beta::gamma::real::ln_positive_wide(z)
            } else {
                crate::gamma_beta::gamma::real::ln_negative_wide(z)
            }
        };
        let sign_gamma = |z: f64| -> Result<f64, SciOxideError> {
            Ok(if z > 0.0 || scioxide_core::sin_pi(z)? > 0.0 {
                1.0
            } else {
                -1.0
            })
        };
        let la = log_gamma(Wide::from(a.re));
        let lb = log_gamma(Wide::from(a.re) + Wide::from(x.re));
        let sa = sign_gamma(a.re)?;
        let sb = sign_gamma(b.re)?;
        let l = lb - la;
        Candidate::logarithmic(
            ScaledValue::from_value(real(sa * sb))?,
            l,
            16.0 * EPS * (1.0 + la.hi.abs() + lb.hi.abs()),
            algorithm_component!("pochhammer_signed_gamma"),
        )?
    } else {
        // Shift nearby negative-axis parameters together before taking logs.
        // Multiplying the rational recurrence directly avoids subtracting
        // separately unwrapped reflection phases near a component zero.
        if shift == 0
            && a.re < 0.0
            && x.norm() <= 0.25 * (ONE - a).norm()
            && (a.im.abs() + x.im.abs()) * core::f64::consts::PI < f64::MAX.ln() - 4.0
        {
            return reflected_short_ratio(a, x, relative, w);
        }
        let ashift = a + real(shift as f64);
        let bshift = ashift + x;
        let short_shifted = ashift.re >= 4.5 && x.norm() <= 0.25 * ashift.norm();
        let (l, l_rel) = if short_shifted {
            let l = small_log_ratio(ashift, x, w)?;
            (l, 64.0 * EPS * (1.0 + l.norm()))
        } else {
            let (la, _) =
                crate::gamma_beta::gamma::complex_log_gamma(ashift, &mut || w.try_step())?;
            let (lb, _) =
                crate::gamma_beta::gamma::complex_log_gamma(bshift, &mut || w.try_step())?;
            (lb - la, 64.0 * EPS * (1.0 + la.norm() + lb.norm()))
        };
        if shift > 0 && short_shifted {
            w.steps(32)?;
            let r_shift = (0..shift).try_fold(ZERO, |r, i| -> Result<C, SciOxideError> {
                w.try_step()?;
                let ai = a + real(i as f64);
                Ok((ai * r - ONE) / (ai + x))
            })?;
            let t1 = r_shift * l.exp();
            let t2 = complex_expm1(l) / x;
            let w_rel = t1 + t2;
            if w_rel.re.is_finite() && w_rel.im.is_finite() {
                if relative {
                    let rel = (64.0 + 8.0 * shift as f64)
                        * EPS
                        * (1.0 + (t1.norm() + t2.norm()) / w_rel.norm().max(f64::MIN_POSITIVE));
                    return Candidate::value(
                        w_rel,
                        rel,
                        algorithm_component!("pochhammer_complex_recurrence"),
                    );
                }
                let d = x * w_rel;
                if d.norm() < 0.5 {
                    let val = ONE + d;
                    let rel = (64.0 + 8.0 * shift as f64)
                        * EPS
                        * (1.0 + x.norm() * (t1.norm() + t2.norm()));
                    return Candidate::value(
                        val,
                        rel,
                        algorithm_component!("pochhammer_complex_recurrence"),
                    );
                }
            }
        }
        let gamma = Candidate::logarithmic(
            ScaledValue::from_value(C::new(l.im.cos(), l.im.sin()))?,
            Wide::from(l.re),
            l_rel,
            algorithm_component!("pochhammer_complex_gamma"),
        )?;
        if shift == 0 {
            gamma
        } else {
            let recurrence = product_report(
                shift,
                |i| {
                    let ai = a + real(i as f64);
                    ai / (ai + x)
                },
                w,
                algorithm_component!("pochhammer_complex_recurrence"),
            )?;
            let s = recurrence
                .scaled
                .ok_or_else(|| crate::support::COMBINATORIAL_ERRORS.unstable())?;
            if let Some(value) = gamma.ordinary.filter(|v| *v != ZERO) {
                Candidate::scaled(
                    s.mul(ScaledValue::from_value(value)?)?,
                    gamma.relative + recurrence.relative,
                    algorithm_component!("pochhammer_complex_recurrence"),
                )?
            } else {
                let phase = gamma
                    .logarithmic
                    .scaled_phase()
                    .mul(recurrence.logarithmic.scaled_phase())?;
                Candidate::logarithmic(
                    phase,
                    Wide::from(l.re)
                        + Wide::from(
                            recurrence
                                .logarithmic
                                .log_magnitude()
                                .ok_or_else(|| crate::support::COMBINATORIAL_ERRORS.unstable())?,
                        ),
                    gamma.relative + recurrence.relative,
                    algorithm_component!("pochhammer_complex_recurrence"),
                )?
            }
        }
    };
    if relative {
        relative_from_candidate(candidate, x)
    } else {
        Ok(candidate)
    }
}
fn relative_from_candidate(c: Candidate, x: C) -> Result<Candidate, SciOxideError> {
    let divisor = ScaledValue::from_value(x)?;
    if let Some(ordinary) = c.ordinary {
        let difference = ordinary - ONE;
        let value = ScaledValue::from_value(difference)?.div(divisor)?;
        let mut result = Candidate::scaled(
            value,
            4.0 * EPS,
            algorithm_component!("pochhammer_relative_ratio"),
        )?;
        let uncertainty = ScaledValue::from_value(ordinary)?
            .magnitude()?
            .mul(Scaled::from_value(c.relative)?)?;
        if difference != ZERO {
            let relative = uncertainty
                .div(ScaledValue::from_value(difference)?.magnitude()?)?
                .round()?;
            result.relative += relative;
            result.log_error += relative;
        } else {
            result.absolute = uncertainty.div(divisor.magnitude()?)?.round()?;
        }
        return Ok(result);
    }
    let log_x = LogValue::from_value(x)?;
    let phase = c.logarithmic.scaled_phase().div(log_x.scaled_phase())?;
    Candidate::logarithmic(
        phase,
        Wide::from(
            c.logarithmic
                .log_magnitude()
                .ok_or_else(|| crate::support::COMBINATORIAL_ERRORS.unstable())?,
        ) - Wide::from(
            log_x
                .log_magnitude()
                .ok_or_else(|| crate::support::COMBINATORIAL_ERRORS.unstable())?,
        ),
        c.relative + 8.0 * EPS,
        algorithm_component!("pochhammer_relative_scaled"),
    )
}

fn sequence(n: usize, euler: bool, w: &mut Work<'_, '_, '_>) -> Result<Candidate, SciOxideError> {
    if n == 0 {
        return Candidate::value(ONE, 0.0, algorithm_component!("sequence_origin"));
    }
    if !n.is_multiple_of(2) {
        return Candidate::value(
            if !euler && n == 1 { real(-0.5) } else { ZERO },
            0.0,
            algorithm_component!("sequence_parity"),
        );
    }
    if euler && n / 2 < EULER.len() {
        return Candidate::value(
            real(EULER[n / 2] as f64),
            4.0 * EPS,
            algorithm_component!("euler_exact_table"),
        );
    }
    if !euler && n / 2 < BERNOULLI.len() {
        let (p, q) = BERNOULLI[n / 2];
        let numerator = integer_u64(p.unsigned_abs()).mul_f64(if p < 0 { -1.0 } else { 1.0 });
        return Candidate::value(
            real((numerator / integer(q as usize)).value()),
            4.0 * EPS,
            algorithm_component!("bernoulli_rational_table"),
        );
    }
    let mut sum = Wide::from(1.0);
    let order = n as f64 + if euler { 1.0 } else { 0.0 };
    for index in 1_usize.. {
        w.try_step()?;
        let base = if euler {
            2.0 * index as f64 + 1.0
        } else {
            index as f64 + 1.0
        };
        let term = (-order * base.ln()).exp();
        sum = sum
            + Wide::from(if euler && !index.is_multiple_of(2) {
                -term
            } else {
                term
            });
        let tail = if euler {
            ((base + 2.0) / base).powf(-order) * term
        } else {
            term * base / (order - 1.0)
        };
        if tail <= 0.25 * EPS {
            break;
        }
    }
    let log = factorial_log(n, w)?
        + sum.ln()
        + if euler {
            (Wide::from(2.0) / pi()).ln().mul_f64(n as f64 + 1.0) + LN_2
        } else {
            LN_2 - pi().mul_f64(2.0).ln().mul_f64(n as f64)
        };
    let sign = if (n / 2).is_multiple_of(2) {
        if euler { 1.0 } else { -1.0 }
    } else if euler {
        -1.0
    } else {
        1.0
    };
    Candidate::logarithmic(
        ScaledValue::from_value(real(sign))?,
        log,
        16.0 * EPS * (1.0 + log.hi.abs()),
        if euler {
            algorithm_component!("euler_beta_identity")
        } else {
            algorithm_component!("bernoulli_zeta_identity")
        },
    )
}

fn stirling(
    n: usize,
    k: usize,
    first: bool,
    cells: &mut [Scaled<f64>],
    w: &mut Work<'_, '_, '_>,
) -> Result<Candidate, SciOxideError> {
    if n == k {
        return Candidate::value(ONE, 0.0, algorithm_component!("stirling_diagonal"));
    }
    if k == 0 || k > n {
        return Candidate::value(ZERO, 0.0, algorithm_component!("stirling_zero"));
    }
    if !first && k == 1 {
        return Candidate::value(ONE, 0.0, algorithm_component!("stirling_single_block"));
    }
    if cells.len() <= k {
        return Err(invalid(
            scioxide_core::argument!("scratch"),
            "Stirling column exceeds reserved workspace",
        ));
    }
    w.steps(k + 1)?;
    cells[..=k].fill(Scaled::ZERO);
    cells[0] = Scaled::from_value(1.0)?;
    for row in 1..=n {
        let lower = k.saturating_sub(n - row).max(1);
        for column in (lower..=k.min(row)).rev() {
            w.try_step()?;
            let factor = if first { row - 1 } else { column };
            cells[column] = cells[column]
                .mul(Scaled::from_value(factor as f64)?)?
                .add(cells[column - 1])?;
        }
        cells[0] = Scaled::ZERO;
    }
    let ([v], exponent) = factor_common_scale([cells[k]])?;
    let sign = if first && !(n - k).is_multiple_of(2) {
        -1.0
    } else {
        1.0
    };
    Candidate::scaled(
        ScaledValue::from_parts(real(sign * v.round()?), exponent)?,
        8.0 * EPS * n as f64,
        if first {
            algorithm_component!("stirling_first_recurrence")
        } else {
            algorithm_component!("stirling_second_recurrence")
        },
    )
}

/// Scalar dispatch key shared by the combinatorial entry points. Integer-index queries promote
/// their real outputs to T without continuation in order.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CombinatorialQuery<T: NumericScalar> {
    /// Discrete factorial.
    Factorial(usize),
    /// Double factorial.
    DoubleFactorial(usize),
    /// Multifactorial (n, positive step).
    Multifactorial(usize, usize),
    /// Discrete binomial coefficient (n,k), k<=n.
    Binomial(usize, usize),
    /// Pochhammer (base, shift).
    Pochhammer(T, T),
    /// Cancellation-safe (Pochhammer(base,shift)-1)/shift, with digamma limit.
    PochhammerRelative(T, T),
    /// Generalized binomial (upper argument, nonnegative lower index).
    GeneralizedBinomial(T, usize),
    /// Bernoulli number.
    Bernoulli(usize),
    /// Euler number in the sech convention.
    Euler(usize),
    /// Signed first-kind Stirling number.
    StirlingFirst(usize, usize),
    /// Second-kind Stirling number.
    StirlingSecond(usize, usize),
}
impl<T: NumericScalar> CombinatorialQuery<T> {
    /// Validate.
    pub(crate) fn validate(self) -> Result<(), SciOxideError> {
        match self {
            Self::Multifactorial(_, 0) => Err(invalid(
                scioxide_core::argument!("step"),
                "multifactorial step must be positive",
            )),
            Self::Binomial(n, k) if k > n => Err(invalid(
                scioxide_core::argument!("k"),
                "binomial lower index exceeds upper index",
            )),
            Self::GeneralizedBinomial(z, _) => {
                widen(z, "combinatorial parameter z must be finite")?;
                Ok(())
            }
            Self::Pochhammer(a, x) | Self::PochhammerRelative(a, x) => {
                widen(a, "combinatorial parameter a must be finite")?;
                widen(x, "combinatorial parameter x must be finite")?;
                Ok(())
            }
            _ => Ok(()),
        }
    }
    fn evaluate(
        self,
        cells: &mut [Scaled<f64>],
        w: &mut Work<'_, '_, '_>,
    ) -> Result<Candidate, SciOxideError> {
        match self {
            Self::Factorial(n) => multifactorial_kernel(n, 1, w),
            Self::DoubleFactorial(n) => multifactorial_kernel(n, 2, w),
            Self::Multifactorial(n, step) => multifactorial_kernel(n, step, w),
            Self::Binomial(n, k) => {
                if k > n {
                    return Err(invalid(
                        scioxide_core::argument!("k"),
                        "binomial lower index exceeds upper index",
                    ));
                }
                let k = k.min(n - k);
                if k == 0 {
                    return Candidate::value(ONE, 0.0, algorithm_component!("binomial_endpoint"));
                }
                if k < 128 {
                    w.steps(k)?;
                    let mut value = Wide::from(1.0);
                    for i in 1..=k {
                        value = value * integer(n - k + i).div_f64(i as f64);
                    }
                    if value.hi.is_finite() {
                        return Candidate::value(
                            real(value.value()),
                            4.0 * EPS,
                            algorithm_component!("binomial_twofold"),
                        );
                    }
                }
                w.steps(39)?;
                let l = -crate::gamma_beta::beta::ln_beta_wide((n - k + 1) as f64, (k + 1) as f64)?
                    - (integer(n) + Wide::from(1.0)).ln();
                Candidate::logarithmic(
                    ScaledValue::from_value(ONE)?,
                    l,
                    8.0 * EPS * (1.0 + l.hi.abs()),
                    algorithm_component!("binomial_combined_gamma"),
                )
            }

            Self::Pochhammer(a, x) => pochhammer_kernel(
                widen(a, "combinatorial parameter a must be finite")?,
                widen(x, "combinatorial parameter x must be finite")?,
                false,
                w,
            ),
            Self::PochhammerRelative(a, x) => pochhammer_kernel(
                widen(a, "combinatorial parameter a must be finite")?,
                widen(x, "combinatorial parameter x must be finite")?,
                true,
                w,
            ),
            Self::GeneralizedBinomial(a, k) => {
                let a = widen(a, "combinatorial parameter a must be finite")?;
                if k == 0 {
                    return Candidate::value(ONE, 0.0, algorithm_component!("empty_product"));
                }
                let (value, relative) = scaled_product(k, |i| a - real(i as f64), w)?;
                if value.is_zero() {
                    return Candidate::scaled(
                        value,
                        relative,
                        algorithm_component!("generalized_binomial_product"),
                    );
                }
                // Retain numerator cancellation before the single division.
                // The existing factorial table contains exact binary64 integers;
                // larger denominators use the same compensated product kernel.
                let (divisor, denominator_error) = if let Some(&factorial) =
                    crate::gamma_beta::gamma::SMALL_FACTORIALS_F64.get(k)
                {
                    (ScaledValue::from_value(real(factorial))?, 0.0)
                } else {
                    scaled_product(k, |i| real((i + 1) as f64), w)?
                };
                Candidate::scaled(
                    value.div(divisor)?,
                    relative + denominator_error + 4.0 * EPS,
                    algorithm_component!("generalized_binomial_product"),
                )
            }
            Self::Bernoulli(n) => sequence(n, false, w),
            Self::Euler(n) => sequence(n, true, w),
            Self::StirlingFirst(n, k) => stirling(n, k, true, cells, w),
            Self::StirlingSecond(n, k) => stirling(n, k, false, cells, w),
        }
    }
}
/// Evaluate a query.
pub(crate) fn run<T: NumericScalar>(
    query: CombinatorialQuery<T>,
    options: CombinatorialOptions,
    cells: &mut [Scaled<f64>],
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<CombinatorialEvaluation<T>, SciOxideError> {
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    work.check()?;
    let candidate = query.evaluate(cells, &mut work);
    work.check()?;
    report(candidate, &work)
}
crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the double factorial `n!! = n (n − 2) (n − 4) ⋯`, the product of the positive integers
up to `n` with the parity of `n`.

`0!! = 1!! = 1`. Short products are formed in double-word arithmetic; products of more than 128
factors use the gamma-function form of [`crate::multifactorial`]. [`crate::ln_double_factorial`]
gives `ln(n!!)` without overflow.

# Examples

```
// 9!! = 9 · 7 · 5 · 3 · 1 = 945
let d = scioxide_special::double_factorial::<f64>(9)?;
assert!((d - 945.0).abs() < 1e-12);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `n!!` exceeds the range of the scalar type.

# References

- NIST DLMF, §5.2 and §26.1–26.8. <https://dlmf.nist.gov/26>"###)]

    pub double_factorial [S: RealScalar] (n: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed double_factorial_with_options -> Report<S>;
    evaluate |options, context| {
        run(
            CombinatorialQuery::DoubleFactorial(n),
            options,
            &mut [],
            None,
        )
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Bernoulli number `B_n`, defined by `t / (eᵗ − 1) = Σ B_n tⁿ / n!` (DLMF 24.2.1).

`B₀ = 1`, `B₁ = −1/2`, and `B_n = 0` for odd `n > 1`. Even orders `n ≤ 34` are rounded from
exact rationals; [`crate::checked_bernoulli`] returns those rationals. Larger even orders use

```text
B₂ₘ = (−1)ᵐ⁺¹ 2 (2m)! ζ(2m) / (2π)²ᵐ
```

(DLMF 25.6.2), where `ζ` is the Riemann zeta function, summed directly.

# Examples

```
// B₁ = −1/2 in the convention t / (eᵗ − 1)
let b = scioxide_special::bernoulli::<f64>(1)?;
assert!((b + 0.5).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|B_n|` exceeds the range of the scalar type.

# References

- NIST DLMF, §24.2 and §25.6. <https://dlmf.nist.gov/24>"###)]

    pub bernoulli [S: RealScalar] (n: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed bernoulli_with_options -> Report<S>;
    evaluate |options, context| {
        run(CombinatorialQuery::Bernoulli(n), options, &mut [], None)
    }

}

crate::dispatch::entry! {
    #[doc = crate::dispatch::entry!(@operation, r###"Computes the Euler number `E_n`, defined by `sech t = Σ E_n tⁿ / n!` (DLMF §24.2).

`E₀ = 1`, `E_n = 0` for odd `n`, and the signs of the even orders alternate: `E₂ = −1`,
`E₄ = 5`. Even orders `n ≤ 22` are rounded from exact integers; [`crate::checked_euler`] returns
those integers. Larger even orders use

```text
E₂ₘ = (−1)ᵐ 2²ᵐ⁺² (2m)! β(2m + 1) / π²ᵐ⁺¹,
```

where `β(s) = Σ_{j≥0} (−1)ʲ (2j + 1)⁻ˢ` is the Dirichlet beta function, summed directly.

# Examples

```
// sech t = 1 − t²/2 + …, so E₂ = −1
let e = scioxide_special::euler::<f64>(2)?;
assert!((e + 1.0).abs() < 1e-14);
# Ok::<(), scioxide_core::SciOxideError>(())
```"###, r###"No uniform ulp bound is stated; the detailed form returns an error estimate, and the tests check
the result against independent reference values and exact identities. See the [crate accuracy
table](crate#accuracy)."###, r###"- `NumericalInstability` (`Overflow`) if `|E_n|` exceeds the range of the scalar type.

# References

- NIST DLMF, §24.2 and §25.6. <https://dlmf.nist.gov/24>"###)]

    pub euler [S: RealScalar] (n: usize) -> S
    where [];
    options options: CombinatorialOptions;
    detailed euler_with_options -> Report<S>;
    evaluate |options, context| {
        run(CombinatorialQuery::Euler(n), options, &mut [], None)
    }

}

mod api;
pub use api::*;
mod combinatorial_tables;
