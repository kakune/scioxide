//! `f64` value kernels for real hypergeometric functions.
//!
//! Real 0F1, 1F1, 2F1, 2F0 and Tricomi U values carry no derivative
//! component. An argument derivative is a second value from the contiguous
//! relations d/dz pFq(a;b;z) = (Πa/Πb) pFq(a+1;b+1;z) and
//! U'(a,b,z) = -a U(a+1,b+1,z) (DLMF 16.3.1, 13.3.22).
//!
//! - Series keep their leading terms in two-term arithmetic until `f64`
//!   rounding of the remaining terms cannot reach the result, then
//!   continue in `f64` to a 2^-57 relative tail. Cancellation the two-term
//!   terms cannot resolve sends 1F1 to bounded 4/8/16-word series arithmetic.
//! - Negative-ray 0F1 outside the resolved small series uses its Bessel-J
//!   connection with reflection and a corrected square-root phase (10.16.9).
//! - 1F1 below z = -1 uses Kummer's e^z 1F1(b-a;b;-z), and 2F1 on [-9,-1/2)
//!   Pfaff's (1-z)^-a 2F1(a,c-b;c;z/(z-1)) (DLMF 13.2.39, 15.8.1). The
//!   transformed parameters, arguments and prefactor logarithms are two-term.
//! - U(a,b,z), z > 0, sums its asymptotic expansion at large z (DLMF 13.7.3).
//!   Otherwise Kummer's z^(1-b) U(1+a-b,2-b,z) makes b >= 1 (13.2.40),
//!   backward recurrence (13.3.7) gives the minimal ratio r = U(a+1)/U(a),
//!   and the Wronskian of M and U (13.2.34) normalizes it:
//!   U = Γ(b) z^(1-b) e^z / (Γ(a) Σ t_n (n + a(1 - (a-b+1) r))), where t_n
//!   are the terms of M(a,b,z), all positive for a > 0. Below z = 2, Taylor
//!   steps of the differential equation continue U and U' toward the origin.
//! - The compensated two-term kernels (`hypergeometric_real`) take the other
//!   regions: 1F1 asymptotic candidates and 2F1 at z = 1, on (0.9,1) and
//!   below z = -9. Integral numerator gaps use the reciprocal logarithmic
//!   limit (15.8.8). U leaves arguments below 2^-40, and
//!   parameters beyond the recurrence and continuation limits, to the complex
//!   kernel.
use crate::support::HYPERGEOMETRIC_ERRORS as ERRORS;
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::real;
use crate::support::wide::ln;
use crate::support::wide::project;
use crate::support::wide::{self as wide, Wide as D};
use scioxide_core::{
    AlgorithmComponent, DomainReason, EstimateReliability, RealScalar, SciOxideError,
    algorithm_component,
};

/// Unit roundoff of `f64`.
const UNIT: f64 = f64::EPSILON / 2.0;
/// Series stop once a majorant of their remainder is below this fraction of the sum.
const TAIL: f64 = 1.0 / 144_115_188_075_855_872.0;
/// An `f64` remainder starts once its rounding stays below this fraction
/// of a unit roundoff of the sum.
const HANDOFF: f64 = 0.25;
/// Largest two-term rounding, relative to the sum, a series may return.
const RESOLUTION: f64 = UNIT / 16.0;
/// Magnitudes beyond which series rescale by `SHRINK`.
const LARGE: f64 = 1.0e150;
/// 2^-512, an exact rescaling factor.
const SHRINK: f64 = 7.458_340_731_200_207e-155;
/// Arguments from which the 0F1 series resolves its cancellation.
const F01_SERIES: f64 = -128.0;
/// 1F1 arguments below this use Kummer's transformation.
const KUMMER: f64 = -1.0;
/// 2F1 arguments in [PFAFF_LIMIT, PFAFF) use Pfaff's transformation and those
/// in [PFAFF, DIRECT] the direct series.
const PFAFF: f64 = -0.5;
const PFAFF_LIMIT: f64 = -9.0;
const DIRECT: f64 = 0.9;
/// U arguments below this continue from it by Taylor steps.
const U_CONTINUATION: f64 = 2.0;
/// Smallest U argument reached by continuation.
const U_SMALLEST: f64 = 1.0 / 1_099_511_627_776.0;
/// Largest |a| of the U recurrence and normalization.
const U_PARAMETER: f64 = 64.0;
/// Largest b >= 1 of the U recurrence and normalization. Backward recurrence
/// resolves the minimal ratio only for b <= 2z + 2, so larger b move the
/// normalization point to z = (b-2)/2; beyond b = 12 its rounding grows.
const U_DENOMINATOR: f64 = 12.0;
/// Most downward U recurrence steps from a positive parameter.
const U_SHIFT: f64 = 64.0;
/// Largest |t1| + |t2| over |U| the small-argument U connection accepts.
const CONNECTION: f64 = 4.0;
/// 1/((k+2)(k+1)) for the leading Taylor coefficient recurrences of U.
const TAYLOR: [f64; 96] = {
    let mut table = [0.0; 96];
    let mut k = 0;
    while k < table.len() {
        table[k] = 1.0 / ((k + 2) * (k + 1)) as f64;
        k += 1;
    }
    table
};
use crate::support::constants::LN_2;

/// Real hypergeometric families with `f64` value kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    F01,
    F11,
    F21,
    F20,
    U,
}

impl Kind {
    /// The named generalized family with `p` numerator and `q` denominator
    /// parameters.
    pub(crate) fn pfq(p: usize, q: usize) -> Option<Self> {
        match (p, q) {
            (0, 1) => Some(Self::F01),
            (1, 1) => Some(Self::F11),
            (2, 1) => Some(Self::F21),
            (2, 0) => Some(Self::F20),
            _ => None,
        }
    }

    const fn numerators(self) -> usize {
        match self {
            Self::F01 => 0,
            Self::F11 | Self::U => 1,
            Self::F21 | Self::F20 => 2,
        }
    }

    const fn denominators(self) -> usize {
        match self {
            Self::F20 => 0,
            _ => 1,
        }
    }
}

/// `y·exp(log)` with an absolute error estimate of `y`.
#[derive(Clone, Copy)]
struct Value {
    y: D,
    log: D,
    error: f64,
    method: AlgorithmComponent,
    reliability: EstimateReliability,
}

impl Value {
    const fn new(y: D, error: f64, method: AlgorithmComponent) -> Self {
        Self {
            y,
            log: D::ZERO,
            error,
            method,
            reliability: EstimateReliability::ModelBased,
        }
    }

    /// The value times `exp(log)`.
    fn scaled(self, log: D) -> Self {
        Self {
            log: self.log + log,
            ..self
        }
    }

    /// The value times `factor`, which carries a relative error `relative`.
    fn times(self, factor: D, relative: f64) -> Self {
        let y = self.y * factor;
        Self {
            y,
            error: self.error * factor.hi.abs() + relative * y.hi.abs(),
            ..self
        }
    }

    fn with(self, method: AlgorithmComponent) -> Self {
        Self { method, ..self }
    }
}

/// Real value of a named family and, when `derivative` is set, its argument
/// derivative, on behalf of `w`'s public entry point. `a` and `b` are the
/// numerator and denominator parameters (`a` and `b` of U). `None` leaves
/// the evaluation to the complex kernels: U and 2F0 outside the real U
/// kernel's domain, and 2F1 or 2F0 across their branch cuts.
pub(crate) fn estimate(
    kind: Kind,
    a: &[f64],
    b: &[f64],
    z: f64,
    regularized: bool,
    derivative: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Estimate>, SciOxideError> {
    let wide = |x: &[f64]| {
        let mut out = [D::ZERO; 2];
        for (out, &x) in out.iter_mut().zip(x) {
            *out = D::from(x);
        }
        out
    };
    let (a, b) = (wide(a), wide(&b[..b.len().min(1)]));
    let (a, b) = (&a[..kind.numerators()], &b[..kind.denominators()]);
    let Some(result) = value(kind, a, b, z, regularized, w)? else {
        return Ok(None);
    };
    let slope = if derivative {
        if kind == Kind::U && z == 0.0 {
            return Err(ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain));
        }
        let Some(slope) = slope(kind, a, b, z, regularized, w)? else {
            return Ok(None);
        };
        Some(slope)
    } else {
        None
    };
    finish(result, slope).map(Some)
}

/// Outcome of a slope-only evaluation, independent of the base value's range.
pub(super) enum DerivativeEstimate {
    Unsupported,
    Unavailable,
    Estimated(Estimate),
}

/// Only the argument slope. Named detailed calls already own their value.
pub(crate) fn derivative(
    kind: Kind,
    a: &[f64],
    b: &[f64],
    z: f64,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<DerivativeEstimate, SciOxideError> {
    let aa = core::array::from_fn::<_, 2, _>(|i| D::from(a.get(i).copied().unwrap_or(0.0)));
    let bb = [D::from(b.first().copied().unwrap_or(0.0))];
    let Some(result) = slope(kind, &aa[..a.len()], &bb[..b.len()], z, regularized, w)? else {
        return Ok(DerivativeEstimate::Unsupported);
    };
    if ![
        result.y.hi,
        result.y.lo,
        result.log.hi,
        result.log.lo,
        result.error,
    ]
    .iter()
    .all(|x| x.is_finite())
    {
        return Ok(DerivativeEstimate::Unavailable);
    }
    finish(result, None).map(DerivativeEstimate::Estimated)
}

/// d/dz by the contiguous relations, evaluated by the value kernels.
fn slope(
    kind: Kind,
    a: &[D],
    b: &[D],
    z: f64,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Value>, SciOxideError> {
    let one = D::from(1.0);
    let factor = if kind == Kind::U {
        -a[0]
    } else {
        let numerator = a.iter().fold(one, |product, &x| product * x);
        if regularized {
            numerator
        } else {
            b.iter().fold(numerator, |quotient, &x| quotient / x)
        }
    };
    if factor.hi == 0.0 {
        return Ok(Some(Value::new(
            D::ZERO,
            0.0,
            algorithm_component!("contiguous_derivative"),
        )));
    }
    let mut shifted_a = [D::ZERO; 2];
    let mut shifted_b = [D::ZERO; 1];
    for (out, &x) in shifted_a.iter_mut().zip(a) {
        *out = x + one;
    }
    for (out, &x) in shifted_b.iter_mut().zip(b) {
        *out = x + one;
    }
    let shifted = value(
        kind,
        &shifted_a[..a.len()],
        &shifted_b[..b.len()],
        z,
        regularized,
        w,
    )?;
    Ok(shifted.map(|value| value.times(factor, 4.0 * UNIT * UNIT)))
}

fn value(
    kind: Kind,
    a: &[D],
    b: &[D],
    z: f64,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Value>, SciOxideError> {
    if regularized && let [c] = b {
        return regularize(kind, a, *c, z, w);
    }
    if z == 0.0 && kind != Kind::U {
        return Ok(Some(Value::new(
            D::from(1.0),
            0.0,
            algorithm_component!("zero_identity"),
        )));
    }
    match kind {
        Kind::F01 => f01(b[0], z, w).map(Some),
        Kind::F11 if same(a[0], b[0]) => Ok(Some(
            Value::new(
                D::from(1.0),
                0.0,
                algorithm_component!("exponential_identity"),
            )
            .scaled(D::from(z)),
        )),
        Kind::F11 => f11(a[0], b[0], z, w).map(Some),
        Kind::F21 => f21(a[0], a[1], b[0], z, w),
        Kind::F20 => f20(a[0], a[1], z, w),
        Kind::U => u(a[0].value(), b[0].value(), z, w),
    }
}

fn same(x: D, y: D) -> bool {
    x.hi == y.hi && x.lo == y.lo
}

/// The count m of a non-positive integer parameter -m.
fn terminating(x: D) -> Option<f64> {
    (x.lo == 0.0 && x.hi <= 0.0 && x.hi == x.hi.floor()).then_some(-x.hi)
}

/// pFq(a; c; z) / Γ(c). At a pole c = -m of Γ(c),
/// Π(a_i)_(m+1) z^(m+1)/(m+1)! pFq(a+m+1; m+2; z) (DLMF 15.2.3_5, 13.2.5).
fn regularize(
    kind: Kind,
    a: &[D],
    c: D,
    z: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Value>, SciOxideError> {
    let Some(m) = terminating(c) else {
        let Some(result) = value(kind, a, &[c], z, false, w)? else {
            return Ok(None);
        };
        w.try_step()?;
        let gamma = crate::support::gamma_core::gamma_value(c.value())?;
        let relative = (2.0 * crate::support::gamma_core::GAMMA_ULPS + 1.0) * UNIT;
        return Ok(Some(if gamma.is_finite() && gamma != 0.0 {
            result.times(D::from(1.0 / gamma), relative)
        } else {
            let (logarithm, sign) = crate::support::gamma_core::gamma_log_parts::<f64>(c.value())?;
            result
                .times(D::from(sign), relative * logarithm.abs())
                .scaled(D::from(-logarithm))
        }));
    };
    let count = m + 1.0;
    let mut factor = D::from(1.0);
    let mut exponent = 0.0;
    let mut j = 0.0;
    while j < count {
        w.try_step()?;
        factor = a
            .iter()
            .fold(factor, |product, &x| product * (x + D::from(j)))
            * D::from(z)
            / D::from(j + 1.0);
        if factor.hi.abs() > LARGE {
            factor = factor.mul_f64(SHRINK);
            exponent += 512.0;
        }
        j += 1.0;
    }
    if factor.hi == 0.0 {
        return Ok(Some(Value::new(
            D::ZERO,
            0.0,
            algorithm_component!("regularized_pole_zero"),
        )));
    }
    let mut shifted = [D::ZERO; 2];
    for (out, &x) in shifted.iter_mut().zip(a) {
        *out = x + D::from(count);
    }
    let Some(result) = value(
        kind,
        &shifted[..a.len()],
        &[D::from(count + 1.0)],
        z,
        false,
        w,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(
        result
            .times(factor, 4.0 * count * UNIT)
            .scaled(LN_2.mul_f64(exponent)),
    ))
}

fn f01(b: D, z: f64, w: &mut Work<'_, '_, '_>) -> Result<Value, SciOxideError> {
    if z < 0.0 && b.hi.abs() < f64::MIN_POSITIVE {
        return bessel_connection(b, z, w);
    }
    if z >= F01_SERIES
        && let Some(result) = series([], Some(b), D::from(z), w)?
    {
        return Ok(result);
    }
    if z < 0.0 {
        return bessel_connection(b, z, w);
    }
    compensated(&[], b, z, w)
}

/// Gamma(b) (-z)^((1-b)/2) J_(b-1)(2 sqrt(-z)), DLMF 10.16.9.
/// The square-root residual corrects the Bessel phase before scalar rounding.
fn bessel_connection(b: D, z: f64, w: &mut Work<'_, '_, '_>) -> Result<Value, SciOxideError> {
    use crate::bessel_airy::bessel::{bessel_j, bessel_jy_sqrt, bessel_y};
    let one = D::from(1.0);
    let root = (-z).sqrt();
    let residual = (D::from(-z) - wide::two_product(root, root)).div_f64(root);
    let x = 2.0 * root;
    let order = if b.hi < 0.5 { one - b } else { b - one };
    let (j, y, j_error, y_error) =
        if !residual.hi.is_finite() || residual.hi.abs() > f64::EPSILON.sqrt() {
            let Some([(j, je), (y, ye)]) = bessel_jy_sqrt(order, -z, w)? else {
                return Err(ERRORS.unstable());
            };
            (D::from(j), D::from(y), je, ye)
        } else {
            let corrected = |value: f64, next: f64| {
                let slope = order.hi / x * value - next;
                let error = 16.0
                    * f64::EPSILON
                    * residual.hi.abs()
                    * ((order.hi / x * value).abs() + next.abs())
                    + residual.hi
                        * residual.hi
                        * ((1.0 + (order.hi / x).powi(2)) * value.abs() + slope.abs() / x)
                    + 4.0 * f64::EPSILON * f64::EPSILON * x * slope.abs();
                (D::from(value) + residual.mul_f64(slope), error)
            };
            let (j, je) = corrected(bessel_j(order.hi, x, w)?, bessel_j(order.hi + 1.0, x, w)?);
            let (y, ye) = if b.hi < 0.5 {
                corrected(bessel_y(order.hi, x, w)?, bessel_y(order.hi + 1.0, x, w)?)
            } else {
                (D::ZERO, 0.0)
            };
            (j, y, je, ye)
        };
    let (value, magnitude, phase_error) = if b.hi < 0.5 {
        // Reflect using b itself, so subtracting one never loses its distance
        // from an integral denominator pole.
        let pi = crate::support::constants::PI;
        let sine =
            D::from(crate::util::sin_pi(b.hi)) + pi.mul_f64(b.lo * crate::util::cos_pi(b.hi));
        let cosine =
            D::from(crate::util::cos_pi(b.hi)) - pi.mul_f64(b.lo * crate::util::sin_pi(b.hi));
        (
            -(cosine * j + sine * y),
            (cosine.hi * j.hi).abs() + (sine.hi * y.hi).abs(),
            cosine.hi.abs() * j_error + sine.hi.abs() * y_error,
        )
    } else {
        (j, j.hi.abs(), j_error)
    };
    w.steps(256)?;
    let logarithm = if b.hi > 0.0 {
        crate::support::gamma_real::ln_positive_wide(b)
    } else {
        crate::support::gamma_real::ln_negative_wide(b)
    } + (one - b).mul_f64(0.5) * ln(D::from(-z));
    let sign = if b.hi > 0.0 {
        1.0
    } else {
        crate::util::sin_pi(b.hi).signum()
    };
    // The Bessel kernels' qualified 64-unit envelope, plus transformation
    // rounding. Near a reflected zero retain the individual contributions.
    let error = (64.0 * f64::EPSILON + order.lo.abs() * (4.0 + x.ln().abs())) * magnitude
        + 8.0 * f64::EPSILON * value.hi.abs()
        + phase_error;
    Ok(Value {
        reliability: EstimateReliability::StatisticallyCalibrated,
        ..Value::new(
            value.mul_f64(sign),
            error,
            algorithm_component!("real_bessel_connection"),
        )
        .scaled(logarithm)
    })
}

fn resolved_f11(a: D, b: D, z: f64, w: &mut Work<'_, '_, '_>) -> Result<Value, SciOxideError> {
    let (parameter, argument, scale) = if z < KUMMER && terminating(a).is_none() {
        (b - a, -z, D::from(z))
    } else {
        (a, z, D::ZERO)
    };
    let (y, log, error) = super::hypergeometric_resolved::kummer(parameter, b, argument, w)?;
    Ok(Value {
        y,
        log: log + scale,
        error,
        method: algorithm_component!("real_resolved_kummer_series"),
        reliability: EstimateReliability::ModelBased,
    })
}

fn f11(a: D, b: D, z: f64, w: &mut Work<'_, '_, '_>) -> Result<Value, SciOxideError> {
    if terminating(a).is_none() {
        // The compensated kernels own the exponential and algebraic
        // asymptotic expansions and their remainder checks.
        let cut = 40.0 + 2.0 * (a.hi.abs() + b.hi.abs());
        if z > cut && a.hi > 1e-6 || z < -cut {
            let result = compensated(&[a], b, z, w)?;
            return if result.error <= 8.0 * f64::EPSILON * result.y.hi.abs() {
                Ok(result)
            } else {
                resolved_f11(a, b, z, w)
            };
        }
        if z < KUMMER {
            return match series([b - a], Some(b), D::from(-z), w)? {
                Some(result) => Ok(result
                    .scaled(D::from(z))
                    .with(algorithm_component!("real_kummer_series"))),
                None => resolved_f11(a, b, z, w),
            };
        }
    }
    match series([a], Some(b), D::from(z), w)? {
        Some(result) => Ok(result),
        None => resolved_f11(a, b, z, w),
    }
}

fn f21(a: D, b: D, c: D, z: f64, w: &mut Work<'_, '_, '_>) -> Result<Option<Value>, SciOxideError> {
    if terminating(a).is_some() || terminating(b).is_some() {
        return match series([a, b], Some(c), D::from(z), w)? {
            Some(result) => Ok(Some(result)),
            None if z <= 1.0 => compensated(&[a, b], c, z, w).map(Some),
            None => Ok(None),
        };
    }
    if z > 1.0 {
        return Ok(None);
    }
    if z == 1.0 {
        return compensated(&[a, b], c, z, w).map(Some);
    }
    let one = D::from(1.0);
    if same(a, c) || same(b, c) {
        let exponent = if same(a, c) { b } else { a };
        w.try_step()?;
        return Ok(Some(
            Value::new(one, 0.0, algorithm_component!("parameter_identity"))
                .scaled(-(exponent * ln(one - D::from(z)))),
        ));
    }
    if [a, b, c].iter().all(|x| x.lo == 0.0) && a.hi == 1.0 && b.hi == 1.0 && c.hi == 2.0 {
        let value = -<f64 as RealScalar>::ln_1p(-z) / z;
        return Ok(Some(Value::new(
            D::from(value),
            8.0 * f64::EPSILON * value.abs(),
            algorithm_component!("logarithmic_identity"),
        )));
    }
    let difference = b - a;
    if z < PFAFF_LIMIT && difference.lo == 0.0 && difference.hi.fract() == 0.0 {
        let e = super::hypergeometric_real::integer_reciprocal(a, b, c, z, w)?;
        return Ok(Some(Value {
            y: D::from(e.y.v.re),
            log: D::from(e.scale.v.re),
            error: e.error,
            method: e.method,
            reliability: e.reliability(),
        }));
    }
    let result = if (PFAFF..=DIRECT).contains(&z) {
        series([a, b], Some(c), D::from(z), w)?
    } else if (PFAFF_LIMIT..PFAFF).contains(&z) {
        pfaff(a, b, c, z, w)?
    } else {
        None
    };
    match result {
        Some(result) => Ok(Some(result)),
        None => compensated(&[a, b], c, z, w).map(Some),
    }
}

/// (1-z)^-a 2F1(a,c-b;c;z/(z-1)), or the form with a and b exchanged whose
/// first term ratio is smaller; a terminating form when one exists.
fn pfaff(
    a: D,
    b: D,
    c: D,
    z: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Value>, SciOxideError> {
    w.try_step()?;
    let complement = D::from(1.0) - D::from(z);
    let argument = D::from(-z) / complement;
    let (first, second) = (c - b, c - a);
    let leading = terminating(first).is_some()
        || terminating(second).is_none() && (a.hi * first.hi).abs() <= (b.hi * second.hi).abs();
    let (p, q, exponent) = if leading {
        (a, first, a)
    } else {
        (second, b, b)
    };
    Ok(series([p, q], Some(c), argument, w)?.map(|result| {
        result
            .scaled(-(exponent * ln(complement)))
            .with(algorithm_component!("real_pfaff_series"))
    }))
}

/// 2F0(a,b;;z) = (-1/z)^a U(a,1+a-b,-1/z) for z < 0 (DLMF 13.6.21).
fn f20(a: D, b: D, z: f64, w: &mut Work<'_, '_, '_>) -> Result<Option<Value>, SciOxideError> {
    if terminating(a).is_some() || terminating(b).is_some() {
        return series([a, b], None, D::from(z), w);
    }
    if z >= 0.0 {
        return Ok(None);
    }
    w.try_step()?;
    let x = -1.0 / z;
    if !x.is_finite() {
        // Below |z| = 1/f64::MAX the U argument overflows, but the asymptotic
        // series 1 + abz + (a)_2(b)_2 z²/2 + ... has terms shrinking by at
        // least 2^-400 per step once |(a+n)(b+n)z| < 2^-400; its first
        // correction then bounds the remainder twice over.
        let ratio = |n: f64| ((a.value() + n) * (b.value() + n) * z).abs();
        if ratio(0.0).max(ratio(1.0)) >= f64::powi(2.0, -400) {
            return Err(ERRORS.unstable());
        }
        let first = wide::product(wide::product(a, b), D::from(z));
        return Ok(Some(Value::new(
            D::from(1.0) + first,
            2.0 * first.hi.abs(),
            algorithm_component!("2F0_subnormal_series"),
        )));
    }
    let parameter = (D::from(1.0) + a - b).value();
    let Some(result) = u(a.value(), parameter, x, w)? else {
        return Ok(None);
    };
    // x rounds -1/z, so x^a U(a,c,x) is 2F0 at z' = -1/x. Scaling by x
    // itself keeps that identity exact; z - z' is then bounded through
    // d/dz 2F0 = ab 2F0(a+1,b+1;z) (DLMF 16.3.1).
    let log_x = ln(D::from(x));
    let log = result.log + a * log_x;
    let shift = (D::from(z) + D::from(1.0) / D::from(x)).hi.abs();
    let sensitivity = (a * b).hi.abs() * shift;
    let slope = if sensitivity == 0.0 {
        0.0
    } else if (a.hi > 0.0 && b.hi > -1.0) || (b.hi > 0.0 && a.hi > -1.0) {
        // For a>0, 2F0(a+1,b+1;ξ) = E[(1-ξt)^(-b-1)] over t ~ Gamma(a+1)
        // (DLMF 16.5.3): it lies in (0,1] for ξ<0 and b>-1.
        project(D::from(2.0 * sensitivity), -log)
    } else {
        let Some(next) = u(a.value() + 1.0, parameter, x, w)? else {
            return Ok(None);
        };
        // The derivative magnitude at z, doubled for the excursion to z'.
        2.0 * sensitivity
            * (next.y.hi.abs() + next.error)
            * project(D::from(1.0), next.log - result.log + log_x)
    };
    // Rounding c = 1+a-b evaluates b shifted by the same amount; to first
    // order in z that moves the sum by about |a z| of itself.
    let rounded_c = (D::from(1.0) + a - b - D::from(parameter)).hi.abs();
    let shifted_b = 4.0 * rounded_c * (a.hi * z).abs() * (result.y.hi.abs() + result.error);
    // Two-word logarithms round by a few 2^-104 of their magnitude.
    let rounding = 8.0 * f64::EPSILON * f64::EPSILON * (result.log.hi.abs() + (a * log_x).hi.abs());
    Ok(Some(
        Value {
            error: result.error + slope + shifted_b + rounding * result.y.hi.abs(),
            ..result
        }
        .scaled(a * log_x)
        .with(algorithm_component!("2F0_U_connection")),
    ))
}

/// A value of the compensated two-term kernels.
fn compensated(a: &[D], b: D, z: f64, w: &mut Work<'_, '_, '_>) -> Result<Value, SciOxideError> {
    let mut numerator = [Jet::default(); 2];
    for (out, x) in numerator.iter_mut().zip(a) {
        *out = Jet::constant(real(x.value()));
    }
    let e = crate::hypergeometric_family::hypergeometric_real::evaluate(
        &numerator[..a.len()],
        Jet::constant(real(b.value())),
        Jet::constant(real(z)),
        w,
    )?;
    Ok(Value {
        y: D::from(e.y.v.re),
        log: D::from(e.scale.v.re),
        error: e.error,
        method: e.method,
        reliability: e.reliability(),
    })
}

/// A bound on every term ratio |r_m|, m >= k, of Σ Π(a_i)_n/((b)_n n!) x^n,
/// valid once k + b > 0.
fn majorant(a: &[D], b: Option<D>, x: f64, k: f64) -> f64 {
    let x = x.abs();
    let up = |p: D| k + p.hi.abs();
    match (a, b) {
        ([], Some(b)) => x / ((k + 1.0) * (k + b.hi)),
        ([p], Some(b)) => x * (up(*p) / (k + b.hi)).max(1.0) / (k + 1.0),
        ([p, q], Some(b)) => {
            let first = (up(*p) / (k + b.hi)).max(1.0) * (up(*q) / (k + 1.0)).max(1.0);
            let second = (up(*q) / (k + b.hi)).max(1.0) * (up(*p) / (k + 1.0)).max(1.0);
            x * first.min(second)
        }
        // 2F0 terms eventually grow; only terminating sums reach their end.
        _ => f64::INFINITY,
    }
}

/// Σ_n Π(a_i)_n/((b)_n n!) x^n for at most two numerator parameters and an
/// optional denominator parameter (DLMF 16.2.1); `None` when the two-term
/// rounding the cancellation amplifies could reach an `f64` rounding.
///
/// Terms are two-term while the `f64` remainder's accumulated rounding
/// could exceed `HANDOFF` units of the sum, given the majorant ρ of the
/// remaining term ratios, after which `f64` terms carry relative errors
/// of at most (1 + 9m) units m terms past the handoff.
fn series<const P: usize>(
    a: [D; P],
    b: Option<D>,
    x: D,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Value>, SciOxideError> {
    let end = a
        .iter()
        .filter_map(|&p| terminating(p))
        .fold(f64::INFINITY, f64::min);
    let mut term = D::from(1.0);
    let mut sum = term;
    let mut magnitude = 1.0_f64;
    let mut weighted = 0.0_f64;
    let mut exponent = 0.0_f64;
    let mut n = 0.0_f64;
    let mut done = n >= end;
    while !done {
        w.try_step()?;
        let numerator = a
            .iter()
            .fold(x, |product, &p| wide::product(product, p + D::from(n)));
        let denominator = match b {
            Some(q) => wide::scale(q + D::from(n), n + 1.0),
            None => D::from(n + 1.0),
        };
        term = wide::product(term, wide::quotient(numerator, denominator));
        sum = sum + term;
        n += 1.0;
        let size = term.hi.abs();
        magnitude += size;
        weighted += n * size;
        if !(sum.hi.is_finite() && sum.lo.is_finite() && size.is_finite()) {
            return Err(ERRORS.unstable());
        }
        if magnitude > LARGE {
            term = term.mul_f64(SHRINK);
            sum = sum.mul_f64(SHRINK);
            magnitude *= SHRINK;
            weighted *= SHRINK;
            exponent += 512.0;
        }
        done = n >= end;
        // Neither stopping test can pass while the term exceeds the sum.
        if done || size > sum.hi.abs() || b.is_some_and(|q| n + q.hi <= 0.0) {
            continue;
        }
        let rho = majorant(&a, b, x.hi, n);
        if rho < 1.0 {
            let total = sum.hi.abs() * (1.0 - rho);
            done = size * rho <= TAIL * total;
            if done || size * rho * (0.5 + 9.0 / (1.0 - rho)) <= HANDOFF * total {
                break;
            }
        }
    }
    // Two-term rounding: each step's few products and sums, relative 2^-104
    // apiece, accumulate along the term recurrence and into the sum.
    let mut error = 16.0 * UNIT * UNIT * (weighted + n * magnitude);
    let mut tail = 0.0;
    if !done {
        let parameters = a.map(|p| p.hi);
        let (denominator, x) = (b.map(|q| q.hi), x.hi);
        let mut t = term.hi;
        let mut rounding = 0.0;
        let mut m = 0.0;
        while n < end {
            w.try_step()?;
            let ratio = parameters.iter().fold(x, |product, &p| product * (p + n))
                / match denominator {
                    Some(q) => (q + n) * (n + 1.0),
                    None => n + 1.0,
                };
            t *= ratio;
            tail += t;
            n += 1.0;
            m += 1.0;
            rounding += t.abs() * (1.0 + 9.0 * m);
            if t.abs() <= TAIL * sum.hi.abs() {
                let rho = majorant(&a, b, x, n);
                if rho < 1.0 && t.abs() * rho <= TAIL * (1.0 - rho) * sum.hi.abs() {
                    error += t.abs() * rho / (1.0 - rho);
                    break;
                }
            }
            if t == 0.0 {
                break;
            }
        }
        error += UNIT * rounding;
    }
    let total = sum + D::from(tail);
    if 16.0 * UNIT * UNIT * (weighted + n * magnitude) > RESOLUTION * total.hi.abs() {
        return Ok(None);
    }
    Ok(Some(Value {
        y: total,
        log: LN_2.mul_f64(exponent),
        error,
        method: algorithm_component!("real_series"),
        reliability: EstimateReliability::ModelBased,
    }))
}

/// The kernel estimate: ordinary values are projected here so that two-term
/// scales keep their low parts; exact identities keep their scale for the
/// report's projection. The derivative shares the value's scale.
fn finish(value: Value, slope: Option<Value>) -> Result<Estimate, SciOxideError> {
    let Value {
        y,
        log,
        error,
        method,
        reliability,
    } = value;
    let reliability =
        if slope.is_some_and(|slope| slope.reliability != EstimateReliability::ModelBased) {
            EstimateReliability::StatisticallyCalibrated
        } else {
            reliability
        };
    if ![y.hi, y.lo, log.hi, log.lo, error]
        .iter()
        .all(|x| x.is_finite())
    {
        return Err(ERRORS.unstable());
    }
    let exact = error == 0.0 && log.lo == 0.0 && y.lo == 0.0;
    let ordinary = project(y, log);
    let materialized = !exact && ordinary.is_finite() && (ordinary != 0.0 || y.hi == 0.0);
    let (v, scale, error, base) = if materialized {
        (
            ordinary,
            0.0,
            project(D::from(error), log).abs() + f64::EPSILON * ordinary.abs(),
            D::ZERO,
        )
    } else if exact {
        (y.hi, log.hi, 0.0, log)
    } else {
        (
            y.value(),
            log.value(),
            error + f64::EPSILON * y.hi.abs(),
            log,
        )
    };
    let (d, derivative_error) = match slope {
        None => (0.0, None),
        Some(slope) => {
            let shift = slope.log - base;
            let d = project(slope.y, shift);
            let e = project(D::from(slope.error), shift).abs() + f64::EPSILON * d.abs();
            if d.is_finite() && e.is_finite() {
                (d, Some(e))
            } else {
                (0.0, Some(f64::INFINITY))
            }
        }
    };
    Ok(Estimate::from(crate::support::JetData {
        y: Jet {
            v: real(v),
            d: real(d),
        },
        scale: Jet::constant(real(scale)),
        error,
        method,
        derivative_error,
    })
    .with_reliability(reliability))
}

/// Tricomi U(a,b,z) for real parameters and z >= 0; `None` outside the
/// recurrence and continuation limits.
fn u(a: f64, b: f64, z: f64, w: &mut Work<'_, '_, '_>) -> Result<Option<Value>, SciOxideError> {
    if a <= 0.0 && a == a.floor() {
        return u_polynomial(a, b, z, w).map(Some);
    }
    if z == 0.0 {
        return u_origin(a, b, w).map(Some);
    }
    if z < 0.0 {
        return Ok(None);
    }
    let one = D::from(1.0);
    let gap = D::from(a) - D::from(b) + one;
    if gap.hi == 0.0 && gap.lo == 0.0 {
        // U(a,a+1,z) = z^-a (DLMF 13.6.4).
        w.try_step()?;
        return Ok(Some(
            Value::new(one, 0.0, algorithm_component!("U_power_identity"))
                .scaled(-(D::from(a) * ln(D::from(z)))),
        ));
    }
    if let Some(result) = u_asymptotic(a, gap.value(), z, w)? {
        return Ok(Some(result));
    }
    if z < U_CONTINUATION
        && b != b.floor()
        && let Some(result) = u_connection(a, b, z, w)?
    {
        return Ok(Some(result));
    }
    // The transformed parameters round once; their sensitivities stay below
    // |ln z| plus a unit each.
    let (a, b, prefactor, transformed) = if b >= 1.0 {
        (a, b, D::ZERO, 0.0)
    } else {
        w.try_step()?;
        let prefactor = (one - D::from(b)) * ln(D::from(z));
        let (a, b) = (gap.value(), (D::from(2.0) - D::from(b)).value());
        (
            a,
            b,
            prefactor,
            UNIT * (a.abs() + b.abs()) * (2.0 + 2.0 * z.ln().abs()),
        )
    };
    if a <= 0.0 && a == a.floor() {
        let polynomial = u_polynomial(a, b, z, w)?;
        let relative = transformed * polynomial.y.hi.abs();
        return Ok(Some(
            Value {
                error: polynomial.error + relative,
                ..polynomial
            }
            .scaled(prefactor),
        ));
    }
    let shift = if a > 0.0 { 0.0 } else { (1.0 - a).floor() };
    if a.abs() > U_PARAMETER || b > U_DENOMINATOR || shift > U_SHIFT || z < U_SMALLEST {
        return Ok(None);
    }
    let point = z.max(U_CONTINUATION).max(0.5 * b - 1.0);
    let Some((seed, ratio)) = u_casoratian(a + shift, b, point, w)? else {
        return Ok(None);
    };
    // Recur to a: U(c-1) = (2c-b+z) U(c) - c(c-b+1) U(c+1), relative to the
    // seed, propagating first-order absolute errors of both members.
    let (mut lower, mut upper) = (1.0, ratio);
    let (mut lower_error, mut upper_error) = (0.0, 8.0 * UNIT * ratio.abs());
    let mut exponent = 0.0;
    let mut c = a + shift;
    for _ in 0..shift as usize {
        w.try_step()?;
        let (p, q) = (2.0 * c - b + point, c * (c - b + 1.0));
        let previous = p * lower - q * upper;
        let error = p.abs() * lower_error
            + q.abs() * upper_error
            + 4.0 * UNIT * ((p * lower).abs() + (q * upper).abs());
        (lower, upper) = (previous, lower);
        (lower_error, upper_error) = (error, lower_error);
        c -= 1.0;
        if lower.abs() > LARGE {
            (lower, upper) = (lower * SHRINK, upper * SHRINK);
            (lower_error, upper_error) = (lower_error * SHRINK, upper_error * SHRINK);
            exponent += 512.0;
        }
    }
    let (factor, relative, method) = if z < point {
        let slope = a * ((a - b + 1.0) * upper - lower) / point;
        let slope_error = a.abs() * ((a - b + 1.0).abs() * upper_error + lower_error) / point;
        let (value, error) = u_continuation(
            a,
            b,
            point,
            z,
            (lower, lower_error),
            (slope, slope_error),
            w,
        )?;
        (
            value,
            error / value.abs() + transformed,
            algorithm_component!("U_taylor_continuation"),
        )
    } else {
        (lower, lower_error / lower.abs() + transformed, seed.method)
    };
    if !(factor.is_finite() && factor != 0.0 && relative.is_finite()) {
        return Ok(None);
    }
    Ok(Some(
        seed.times(D::from(factor), relative)
            .scaled(prefactor + LN_2.mul_f64(exponent))
            .with(method),
    ))
}

/// U(-m,b,z) by its three-term recurrence in m (DLMF 13.3.7), with
/// first-order rounding propagation.
fn u_polynomial(a: f64, b: f64, z: f64, w: &mut Work<'_, '_, '_>) -> Result<Value, SciOxideError> {
    let method = algorithm_component!("U_terminating_polynomial");
    let n = -a;
    if n == 0.0 {
        return Ok(Value::new(D::from(1.0), 0.0, method));
    }
    let mut previous = 1.0_f64;
    let mut y = z - b;
    let mut error = 2.0 * UNIT * y.abs();
    let mut previous_error = 0.0;
    let mut exponent = 0.0;
    let mut k = 1.0;
    while k < n {
        w.try_step()?;
        if y.abs().max(previous.abs()) > LARGE {
            (y, previous) = (y * SHRINK, previous * SHRINK);
            (error, previous_error) = (error * SHRINK, previous_error * SHRINK);
            exponent += 512.0;
        }
        let p = z - b - 2.0 * k;
        let q = k * (k + b - 1.0);
        let next = p * y - q * previous;
        let next_error = error * p.abs()
            + previous_error * q.abs()
            + 8.0 * UNIT * (p.abs() * y.abs() + q.abs() * previous.abs());
        (previous, y) = (y, next);
        (previous_error, error) = (error, next_error);
        k += 1.0;
    }
    Ok(Value {
        log: LN_2.mul_f64(exponent),
        ..Value::new(D::from(y), error, method)
    })
}

/// U(a,b,0) = Γ(1-b)/Γ(1+a-b) for b < 1 (DLMF 13.2.16_2).
fn u_origin(a: f64, b: f64, w: &mut Work<'_, '_, '_>) -> Result<Value, SciOxideError> {
    let method = algorithm_component!("U_origin_limit");
    if b >= 1.0 {
        return Err(ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain));
    }
    w.try_step()?;
    let one = D::from(1.0);
    let lower = (one + D::from(a) - D::from(b)).value();
    if lower <= 0.0 && lower == lower.floor() {
        return Ok(Value::new(D::ZERO, 0.0, method));
    }
    let (upper, _) =
        crate::support::gamma_core::gamma_log_parts::<f64>((one - D::from(b)).value())?;
    let (denominator, sign) = crate::support::gamma_core::gamma_log_parts::<f64>(lower)?;
    Ok(Value::new(D::from(sign), 16.0 * f64::EPSILON, method)
        .scaled(D::from(upper) - D::from(denominator)))
}

/// The asymptotic expansion z^-a Σ (a)_n (a-b+1)_n (-1/z)^n / n!, accepted
/// when its terms fall below 2^-57 of the sum before they grow, or below
/// 2^-54 where they turn.
fn u_asymptotic(
    a: f64,
    gap: f64,
    z: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Value>, SciOxideError> {
    if z < 40.0 + 2.0 * (a.abs() * gap.abs()).sqrt().max(a.abs() + gap.abs()) {
        return Ok(None);
    }
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    let mut magnitude = 1.0_f64;
    let mut previous = f64::INFINITY;
    let mut k = 0.0;
    let remainder = loop {
        w.try_step()?;
        let next = term * -(a + k) * (gap + k) / ((k + 1.0) * z);
        let size = next.abs();
        if size > previous {
            if previous <= 8.0 * TAIL * sum.abs() {
                break 2.0 * previous;
            }
            return Ok(None);
        }
        term = next;
        sum += term;
        magnitude += size;
        if size <= TAIL * sum.abs() {
            break 2.0 * size;
        }
        previous = size;
        k += 1.0;
    };
    w.try_step()?;
    Ok(Some(
        Value::new(
            D::from(sum),
            remainder + 2.0 * UNIT * magnitude,
            algorithm_component!("U_asymptotic_expansion"),
        )
        .scaled(-(D::from(a) * ln(D::from(z)))),
    ))
}

/// U(a,b,z) and U(a+1,b,z)/U(a,b,z) for a > 0, b >= 1 and z >= 2.
///
/// The ratio of the minimal solution comes from backward recurrence and the
/// M-U Wronskian normalizes it: with t_n the terms of M(a,b,z),
/// U = Γ(b) z^(1-b) e^z / (Γ(a) Σ t_n (n + a(1 - (a-b+1) r))).
fn u_casoratian(
    a: f64,
    b: f64,
    z: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<(Value, f64)>, SciOxideError> {
    // Calibrated on a <= 128, b <= 2z + 2 against 40000-step references.
    let depth = (1.25 * (a.sqrt() + 3.0) * (2.0 + 24.0 / z)).ceil() + 4.0;
    let (mut next, mut current) = (0.0_f64, 1.0_f64);
    let mut k = depth;
    while k >= 1.0 {
        w.try_step()?;
        let c = a + k;
        let previous = (2.0 * c - b + z) * current - c * (c - b + 1.0) * next;
        (next, current) = (current, previous);
        if current.abs() > LARGE {
            (next, current) = (next * SHRINK, current * SHRINK);
        }
        k -= 1.0;
    }
    let ratio = next / current;
    let weight = a * (a - b + 1.0) * ratio;
    let offset = a - weight;
    if !(ratio.is_finite() && offset > 0.0) {
        return Ok(None);
    }
    // Σ t_n (n + offset): positive terms.
    let (mut t, mut sum, mut plain, mut rounding) = (1.0_f64, offset, 1.0_f64, 0.0_f64);
    let mut n = 0.0;
    loop {
        w.try_step()?;
        t *= (a + n) * z / ((b + n) * (n + 1.0));
        n += 1.0;
        let term = t * (n + offset);
        sum += term;
        plain += t;
        rounding += term * (6.0 * n + 2.0);
        if !sum.is_finite() {
            return Ok(None);
        }
        if term <= TAIL * sum {
            let rho = z * ((n + a) / (n + b)).max(1.0) / (n + 1.0) * (1.0 + 1.0 / (n + offset));
            if rho < 1.0 && term * rho <= TAIL * (1.0 - rho) * sum {
                break;
            }
        }
    }
    w.try_step()?;
    let numerator = crate::support::gamma_core::gamma_value(b)?;
    let denominator = crate::support::gamma_core::gamma_value(a)?;
    let y = numerator / denominator * z.powf(1.0 - b) / sum;
    if !(y.is_finite() && y != 0.0) {
        return Ok(None);
    }
    // Two gamma values, the power, four roundings, the sum's rounding and
    // the ratio's few-unit error through the offset.
    let relative = UNIT
        * (4.0 * crate::support::gamma_core::GAMMA_ULPS
            + 6.0
            + rounding / sum
            + 8.0 * weight.abs() * plain / sum);
    Ok(Some((
        Value::new(
            D::from(y),
            relative * y.abs(),
            algorithm_component!("U_casoratian"),
        )
        .scaled(D::from(z)),
        ratio,
    )))
}

/// U(a,b,z) for non-integer b and z < 2 by the M connection
/// U = Γ(1-b)/Γ(a-b+1) M(a,b,z) + Γ(b-1)/Γ(a) z^(1-b) M(a-b+1,2-b,z)
/// (DLMF 13.2.42), when its terms cancel by at most `CONNECTION`. The
/// shifted parameters are two-term, so neither the gamma arguments nor the
/// power's exponent carry their rounding.
fn u_connection(
    a: f64,
    b: f64,
    z: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Value>, SciOxideError> {
    w.steps(4)?;
    let one = D::from(1.0);
    let complement = one - D::from(b);
    let alpha = D::from(a) + complement;

    let gamma = |x: D| -> Result<f64, SciOxideError> {
        if x.hi <= 0.0 && x.hi == x.hi.floor() && x.lo == 0.0 {
            return Ok(f64::INFINITY);
        }
        let value = crate::support::gamma_core::gamma_value(x.hi)?;
        Ok(if x.lo == 0.0 {
            value
        } else {
            // Γ(x+δ) = Γ(x)(1 + δψ(x)) to first order.
            value * (1.0 + x.lo * crate::support::gamma_core::digamma_kernel::<f64>(x.hi)?)
        })
    };
    let first = gamma(complement)? / gamma(alpha)?;
    let second = gamma(-complement)? / gamma(D::from(a))?;
    let (upper, upper_rounding) = kummer(a, b, z, w)?;
    let (lower, lower_rounding) = kummer(alpha.value(), (one + complement).value(), z, w)?;
    let first = first * upper;
    let second = project(D::from(second * lower), complement * ln(D::from(z)));
    let value = first + second;
    let scale = first.abs() + second.abs();
    if !(value.is_finite() && scale.is_finite()) || scale > CONNECTION * value.abs() {
        return Ok(None);
    }
    // Two gamma values (and their digamma corrections), the power and four
    // roundings per term.
    let base = 4.0 * crate::support::gamma_core::GAMMA_ULPS + 8.0;
    Ok(Some(Value::new(
        D::from(value),
        UNIT * ((base + upper_rounding) * first.abs() + (base + lower_rounding) * second.abs()),
        algorithm_component!("U_connection"),
    )))
}

/// M(a,b,z) for |z| < 2 in `f64`, with its rounding relative to the sum
/// in units of the unit roundoff.
fn kummer(a: f64, b: f64, z: f64, w: &mut Work<'_, '_, '_>) -> Result<(f64, f64), SciOxideError> {
    let (mut t, mut sum, mut magnitude, mut rounding) = (1.0_f64, 1.0_f64, 1.0_f64, 0.0_f64);
    let mut n = 0.0;
    loop {
        w.try_step()?;
        t *= (a + n) * z / ((b + n) * (n + 1.0));
        n += 1.0;
        sum += t;
        magnitude += t.abs();
        rounding += t.abs() * (6.0 * n + 2.0);
        if t == 0.0 {
            break;
        }
        if b + n > 0.0 && t.abs() <= TAIL * sum.abs() {
            let rho = z.abs() * ((n + a.abs()) / (n + b)).max(1.0) / (n + 1.0);
            if rho < 1.0 && t.abs() * rho <= TAIL * (1.0 - rho) * sum.abs() {
                break;
            }
        }
    }
    Ok((sum, (rounding + magnitude) / sum.abs()))
}

/// U(target) and its absolute error from (U, U') at `start` > `target`,
/// each with an absolute error, by `f64` Taylor steps of
/// zU'' + (b-z)U' - aU = 0, each at most half way to the origin.
///
/// Every step advances the unit solutions (1, 0) and (0, 1); the value is
/// their combination and the errors travel through the step's transition
/// matrix, so growth of the other solution is never under-counted.
fn u_continuation(
    a: f64,
    b: f64,
    start: f64,
    target: f64,
    (mut y, mut y_error): (f64, f64),
    (mut slope, mut slope_error): (f64, f64),
    w: &mut Work<'_, '_, '_>,
) -> Result<(f64, f64), SciOxideError> {
    let mut center = start;
    loop {
        w.try_step()?;
        let last = center - target <= 0.5 * center;
        let h = if last { target - center } else { -0.5 * center };
        let scale = 1.0 / center;
        let (square, step) = (h * h * scale, h * scale);
        // v_k = u_k h^k: c(k+2)(k+1)u_(k+2) = (k+a)u_k - (k+1)(k+b-c)u_(k+1).
        // Per unit solution: [v_k, v_(k+1), Σ v, Σ k v, Σ|v|, Σ k|v|]. The
        // coefficients depend only on k, off the recurrences' critical paths.
        let mut first = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut second = [0.0, h, h, h, h.abs(), h.abs()];
        let mut quiet = 0;
        let mut k = 0.0;
        while quiet < 3 {
            w.try_step()?;
            let reciprocal = match TAYLOR.get(k as usize) {
                Some(&reciprocal) => reciprocal,
                None => 1.0 / ((k + 2.0) * (k + 1.0)),
            };
            let p = (k + a) * square * reciprocal;
            let q = (k + 1.0) * (k + b - center) * step * reciprocal;
            let advance = |[previous, current, value, moment, magnitude, weighted]: [f64; 6]| {
                let next = p * previous - q * current;
                [
                    current,
                    next,
                    value + next,
                    moment + (k + 2.0) * next,
                    magnitude + next.abs(),
                    weighted + (k + 2.0) * next.abs(),
                ]
            };
            (first, second) = (advance(first), advance(second));
            let contribution = (first[1] * y).abs() + (second[1] * slope).abs();
            quiet = if contribution <= TAIL * (first[2] * y + second[2] * slope).abs() {
                quiet + 1
            } else {
                0
            };
            k += 1.0;
        }
        let value = first[2] * y + second[2] * slope;
        let derivative = (first[3] * y + second[3] * slope) / h;
        if !(value.is_finite() && derivative.is_finite()) {
            return Err(ERRORS.unstable());
        }
        let rounding = 4.0 * UNIT * (first[4] * y.abs() + second[4] * slope.abs());
        let slope_rounding = 4.0 * UNIT * (first[5] * y.abs() + second[5] * slope.abs()) / h.abs();
        (y_error, slope_error) = (
            first[2].abs() * y_error + second[2].abs() * slope_error + rounding,
            (first[3].abs() * y_error + second[3].abs() * slope_error) / h.abs() + slope_rounding,
        );
        (y, slope) = (value, derivative);
        if last {
            return Ok((y, y_error));
        }
        center += h;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    fn work() -> Work<'static, 'static, 'static> {
        Work::new(1 << 22)
    }

    fn ulps(x: f64, y: f64) -> f64 {
        let ordered = |v: f64| {
            let bits = v.to_bits() as i64;
            i128::from(if bits < 0 { i64::MIN - bits } else { bits })
        };
        (ordered(x) - ordered(y)).abs() as f64
    }

    /// The value and its reported absolute error.
    fn real(kind: Kind, a: &[f64], b: &[f64], z: f64) -> (f64, f64) {
        let e = estimate(kind, a, b, z, false, false, &mut work())
            .unwrap()
            .unwrap();
        assert_eq!(e.scale.v.re, 0.0, "{kind:?} {a:?} {b:?} {z}: unprojected");
        (e.y.v.re, e.error)
    }

    /// The compensated two-term kernels' value and error at the same point.
    fn reference(a: &[f64], b: f64, z: f64) -> (f64, f64) {
        let wide: [D; 2] = core::array::from_fn(|i| D::from(a.get(i).copied().unwrap_or(0.0)));
        let v = compensated(&wide[..a.len()], D::from(b), z, &mut work()).unwrap();
        (project(v.y, v.log), project(D::from(v.error), v.log))
    }

    fn kind(a: &[f64]) -> Kind {
        Kind::pfq(a.len(), 1).unwrap()
    }

    /// Points in every `f64` region: direct, Kummer and Pfaff series on
    /// both sides of each cut.
    fn grid() -> Vec<(Vec<f64>, f64, f64)> {
        let mut points = Vec::new();
        for b in [0.5, 2.25, 7.5] {
            for z in [-120.0, -30.0, -8.5, -0.3, 0.7, 6.0, 45.0] {
                points.push((vec![], b, z));
            }
        }
        for a in [-3.5, 0.75, 4.25] {
            for b in [0.5, 3.25] {
                for z in [-30.0, -1.5, -1.0, -0.75, 0.6, 12.0, 35.0] {
                    points.push((vec![a], b, z));
                }
            }
        }
        for a in [-2.5, 0.75, 2.25] {
            for b in [-1.25, 1.5] {
                for c in [0.5, 4.5] {
                    for z in [-8.5, -2.0, -0.5, -0.4, 0.3, 0.85] {
                        points.push((vec![a, b], c, z));
                    }
                }
            }
        }
        points
    }

    #[test]
    fn series_agree_with_the_compensated_kernels() {
        for (a, b, z) in grid() {
            let (value, error) = real(kind(&a), &a, &[b], z);
            let (expected, expected_error) = reference(&a, b, z);
            // Two units apart, unless the compensated value reports more (its
            // reciprocal connection below z = -4 rounds its gamma ratios); each
            // estimate covers the other kernel's value within both errors.
            let gap = (value - expected).abs();
            assert!(
                ulps(value, expected) <= 2.0 || gap <= expected_error,
                "{a:?} {b} {z}: {value:e} vs {expected:e}"
            );
            assert!(
                gap <= error + expected_error + f64::EPSILON * value.abs(),
                "{a:?} {b} {z}: gap {gap:e}, reported {error:e}"
            );
        }
    }

    #[test]
    fn kummer_and_pfaff_join_the_direct_series_at_their_cuts() {
        for cut in [KUMMER, PFAFF, PFAFF_LIMIT, DIRECT] {
            for z in [cut.next_down(), cut, cut.next_up()] {
                let cases: [(&[f64], f64); 2] = [(&[0.75], 2.5), (&[0.5, 1.25], 2.25)];
                for (a, b) in cases {
                    if a.len() != 2 && [PFAFF, PFAFF_LIMIT, DIRECT].contains(&cut)
                        || a.len() == 2 && cut == KUMMER
                    {
                        continue;
                    }
                    let (value, _) = real(kind(a), a, &[b], z);
                    let (expected, _) = reference(a, b, z);
                    assert!(ulps(value, expected) <= 2.0, "{a:?} {b} {z}");
                }
            }
        }
    }

    #[test]
    fn u_matches_closed_forms_in_every_region() {
        let u = |a: f64, b: f64, z: f64| real(Kind::U, &[a], &[b], z);
        // U(a,a+1,z) = z^-a (DLMF 13.6.4).
        for z in [0.25, 3.0, 70.0] {
            let (value, _) = u(1.75, 2.75, z);
            assert!(ulps(value, z.powf(-1.75)) <= 2.0);
        }
        // U(1/2,1/2,x^2) = sqrt(pi) exp(x^2) erfc(x) (DLMF 13.6.7) through the
        // connection, the normalized recurrence and the asymptotic expansion.
        for x in [0.3_f64, 1.2, 2.5, 5.0, 9.0] {
            let z = x * x;
            let expected = core::f64::consts::PI.sqrt() * z.exp() * crate::erfc(x).unwrap();
            let (value, error) = u(0.5, 0.5, z);
            assert!(
                ((value - expected) / expected).abs() < 32.0 * f64::EPSILON,
                "{z}"
            );
            assert!(error < 512.0 * f64::EPSILON * value.abs());
        }
        // U(nu+1/2,2nu+1,2x) = exp(x) (2x)^-nu K_nu(x)/sqrt(pi) (DLMF 13.6.10),
        // b >= 1 and Kummer's b < 1 orientation, on both sides of z = 2.
        for nu in [1.25_f64, -0.3] {
            for x in [0.05_f64, 0.6, 0.99, 1.01, 3.0, 12.0, 30.0] {
                let expected = x.exp() * (2.0 * x).powf(-nu) * crate::bessel_k(nu, x).unwrap()
                    / core::f64::consts::PI.sqrt();
                let (value, error) = u(nu + 0.5, 2.0 * nu + 1.0, 2.0 * x);
                let observed = ((value - expected) / expected).abs();
                assert!(observed < 64.0 * f64::EPSILON, "{nu} {x}: {observed:e}");
                assert!(error > 0.0 && error < 256.0 * f64::EPSILON * value.abs());
            }
        }
    }

    #[test]
    fn u_satisfies_its_recurrence_across_negative_parameters() {
        // U(a-1) + (b-2a-z) U(a) + a(a-b+1) U(a+1) = 0
        // (DLMF 13.3.7), with a crossing zero and z inside and outside the
        // continuation range.
        for (a, b) in [(-1.3, 1.5), (-2.7, 0.4), (0.6, 3.5)] {
            for z in [0.4, 1.7, 2.6, 9.0] {
                let u = |a: f64| real(Kind::U, &[a], &[b], z);
                let ((lower, e0), (middle, e1), (upper, e2)) = (u(a - 1.0), u(a), u(a + 1.0));
                let terms = [lower, (b - 2.0 * a - z) * middle, a * (a - b + 1.0) * upper];
                let residual = terms.iter().sum::<f64>().abs();
                let scale = terms.iter().map(|t| t.abs()).sum::<f64>();
                let bound = e0 + (b - 2.0 * a - z).abs() * e1 + (a * (a - b + 1.0)).abs() * e2;
                assert!(
                    residual <= bound + 8.0 * f64::EPSILON * scale,
                    "{a} {b} {z}: {residual:e} > {bound:e}"
                );
            }
        }
    }

    #[test]
    fn argument_derivatives_follow_the_contiguous_relations() {
        for (a, b, z) in grid() {
            let k = kind(&a);
            let e = estimate(k, &a, &[b], z, false, true, &mut work())
                .unwrap()
                .unwrap();
            let z = Jet::variable(real_part(z));
            let jets: Vec<_> = a.iter().map(|&x| Jet::constant(real_part(x))).collect();
            let r = crate::hypergeometric_family::hypergeometric_real::evaluate(
                &jets,
                Jet::constant(real_part(b)),
                z,
                &mut work(),
            )
            .unwrap();
            let expected = (r.y.d.re + r.y.v.re * r.scale.d.re) * r.scale.v.re.exp();
            assert!(
                (e.y.d.re - expected).abs() <= 1e-13 * expected.abs().max(1e-300),
                "{a:?} {b}: {:e} vs {expected:e}",
                e.y.d.re
            );
        }
        // z U'(a,b,z) = -a U(a,b,z) + a(a-b+1) U(a+1,b,z), independent of the
        // contiguous form -a U(a+1,b+1,z) the kernel uses.
        for (a, b, z) in [(0.75, 1.5, 0.8), (2.5, 0.25, 6.0), (1.25, 3.0, 60.0)] {
            let e = estimate(Kind::U, &[a], &[b], z, false, true, &mut work())
                .unwrap()
                .unwrap();
            let (upper, _) = real(Kind::U, &[a + 1.0], &[b], z);
            let expected = (-a * e.y.v.re + a * (a - b + 1.0) * upper) / z;
            assert!(((e.y.d.re - expected) / expected).abs() < 1e-13);
        }
    }

    fn real_part(x: f64) -> crate::support::C {
        crate::support::real(x)
    }

    #[test]
    fn regularized_poles_use_the_shifted_series() {
        // 2F1(a,b;-1;z)/Γ(-1) = ab(a+1)(b+1) z^2/2 2F1(a+2,b+2;3;z).
        let (a, b, z) = (0.5, 0.75, 0.3);
        let e = estimate(Kind::F21, &[a, b], &[-1.0], z, true, false, &mut work())
            .unwrap()
            .unwrap();
        let (shifted, _) = real(Kind::F21, &[a + 2.0, b + 2.0], &[3.0], z);
        let expected = a * b * (a + 1.0) * (b + 1.0) * z * z / 2.0 * shifted;
        assert!(ulps(e.y.v.re, expected) <= 4.0);
        // A numerator that terminates before the pole leaves an exact zero.
        let e = estimate(Kind::F11, &[-1.0], &[-2.0], 0.4, true, false, &mut work())
            .unwrap()
            .unwrap();
        assert_eq!(e.y.v.re, 0.0);
    }

    #[test]
    fn every_term_is_charged_to_the_work_allowance() {
        let mut w = Work::new(5);
        assert!(estimate(Kind::F11, &[0.5], &[1.5], 5.0, false, false, &mut w).is_err());
        assert!(w.exhausted());
        assert_eq!(w.used, 5);
        let mut w = Work::new(3);
        assert!(estimate(Kind::U, &[0.75], &[1.5], 4.0, false, false, &mut w).is_err());
        assert_eq!(w.used, 3);
    }

    #[test]
    fn two_term_logarithms_and_projections_keep_their_low_parts() {
        // ln(1 + 2^-40) = 2^-40 - 2^-81 + ...: the head alone would lose it.
        let x = D::from(1.0) + D::from(2.0_f64.powi(-40));
        let l = ln(x);
        assert_eq!(l.value(), 2.0_f64.powi(-40) - 2.0_f64.powi(-81));
        for value in [1e-300, 0.75, 3.0, 1e300] {
            assert!(ulps(ln(D::from(value)).value(), value.ln()) <= 1.0);
        }
        // An exact power of two, and a scale beyond the binary64 range.
        assert_eq!(
            project(D::from(3.0), LN_2.mul_f64(-1074.0)),
            f64::from_bits(3)
        );
        assert_eq!(project(D::from(1.5), LN_2.mul_f64(5.0)), 48.0);
        assert_eq!(project(D::from(1.0), D::from(800.0)), f64::INFINITY);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod detailed_work_tests {
    use super::*;
    use crate::{
        HypergeometricOptions, hyp0f1_with_options, hyp1f1_with_options, hyp2f1_with_options,
        hyperu_with_options,
    };
    #[test]
    fn detailed_calls_charge_exactly_one_value_and_one_slope() {
        for (kind, a, b, z) in [
            (Kind::F01, &[][..], 1.25, -32.0),
            (Kind::F11, &[0.5][..], 1.5, 5.0),
            (Kind::F21, &[0.5, 0.75][..], 2.0, 0.8),
            (Kind::U, &[0.75][..], 1.5, 4.0),
        ] {
            let mut work = Work::new(1 << 20);
            let expected = estimate(kind, a, &[b], z, false, true, &mut work)
                .unwrap()
                .unwrap();
            let options = HypergeometricOptions::default()
                .with_iteration_budget(scioxide_core::Budget::try_from(work.used).unwrap());
            let result = match kind {
                Kind::F01 => hyp0f1_with_options(b, z, options),
                Kind::F11 => hyp1f1_with_options(a[0], b, z, options),
                Kind::F21 => hyp2f1_with_options(a[0], a[1], b, z, options),
                _ => hyperu_with_options(a[0], b, z, options),
            }
            .unwrap();
            assert_eq!(result.counts().iterations(), work.used);
            assert!(result.termination().is_success());
            let slope = (expected.y.d.re + expected.y.v.re * expected.scale.d.re)
                * expected.scale.v.re.exp();
            assert!(
                (result.derivative().unwrap().value() - slope).abs()
                    <= 32.0 * f64::EPSILON * slope.abs()
            );
        }
    }
}
