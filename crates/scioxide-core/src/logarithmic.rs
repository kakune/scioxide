//! Log-sum-exp reductions with masked terms and exact zero sums.
//!
//! A `−∞` logit is a masked term, an exact zero exponential. Complex logits may
//! be `−∞` only in their real part; every imaginary part and weight component
//! must be finite. All components are checked before masked or zero-weight
//! terms are skipped, and NaN or `+∞` logits are always errors.
//!
//! | Case | Sum as `LogValue` | Real logarithm | Principal complex logarithm | Normalization |
//! |---|---|---|---|---|
//! | Empty or all masked | zero | `−∞` | `(−∞, +0)` | `DomainError` |
//! | Finite and masked terms | sum of the finite terms | same | same | masked softmax `+0`, masked log-softmax `−∞` |
//! | Zero weights | zero if no other term contributes | use the value form for signed weights | use the value form for complex weights | not weighted |
//! | Exact cancellation | zero, with zero phase | use the value form | `DomainError` | not applicable |
//! | NaN or `+∞` logit, non-finite weight or angle | `NonFinite`, also on skipped terms | same | same | same |
//!
//! An empty batch of lanes is valid. A single-term normalization is exactly
//! `(1, +0)` in probability and log-probability form. A negative log
//! probability that underflows is `−0`; probability zeros, including masked
//! entries, are `+0`. Overflow from finite arithmetic is a `NumericalInstability`.

// # Cancellation-aware reductions and certified tiering
//
// Positive sums first try working-precision exponentials and compensated
// summation, with an outward certificate on the destination grid. The owned
// Gal table uses a split ln(2) shift and degree-five polynomial; extreme tails
// retain the owned fdlibm exponential. Error-free shift residuals give a
// 4*u relative term bound plus four minimum subnormals. Addition residuals
// are retained in four independent sums and merged without partial rounding:
// g=(n+8)*u/(1-(n+8)*u) includes the merge and gives a 4*g² mass bound, plus
// final working rounding. This tier admits n<=2^24; longer inputs fall back.
// Owned log1p contributes one binary64 ULP. Reconstruction includes addition
// rounding; certification reserves two destination ULP for final projection.
// Near-zero/cancelling and subnormal logarithms retain the existing tiers.
// Normalization certifies its shifted log within eight destination ULP and
// denominator within 7*u, reserving room for subsequent output arithmetic.
//
// Uncertified inputs try double-double exponentials and one logarithmic
// Newton correction. Shifts use error-free TwoSum; FMA products retain their
// residuals. A three-part ln(2) reduction, degree-16 reciprocal-factorial
// Horner polynomial and four expm1 doublings give eta_exp=2^-100 relative
// error, plus at most two binary64 minimum subnormals per term. The exact
// accumulator sums both components with no n*epsilon term; extraction drops
// <2^-105 of its positive binary frame. For seed error delta<=8*ULP(ln1p(r_hi)),
// eta_log=2^-98*(1+|shifted_log|)+delta² bounds logarithmic evaluation error.
// See the double-double helper for the arithmetic and remainder derivation.
//
// Rounded shifts below -740, including overflowed negative differences, are
// dropped with mass bound n*2^-1067 (>n*exp(-740+2^-43)). Term, underflow,
// extraction and tail errors are propagated through the logarithm, then added
// to eta_log and the final TwoSum reconstruction bound. Bound arithmetic rounds
// outward. Accept iff B<=4*ulp_S(|L_hat|-B), conservatively using the low interval
// endpoint on S's own normal/subnormal grid. Destination rounding adds at most
// half a ULP, within the unchanged 16-destination-ULP envelope. Normalization
// certifies shifted_log instead, independently of cancellation in m+shifted_log.
// Centered positive inputs also try this tier: for n>=2 their full logarithm
// is at least ln(2)-2^-27. The centered series remains available in the fallback.
//
// Weighted and general-phase complex sums first try compensated binary64
// components and an ordinary absolute-component mass sum. Bounded shifted
// exponentials and products contribute 8*u of that mass for Cartesian weights
// or 34*u for normalized unit phases; compensation adds 4*g², with
// g=n*u/(1-n*u). Outward mass intervals bound the true vector conditioning
// using L1/sqrt(2) <= vector mass <= L1. Hypot adds at most 10*u of the norm.
// For norm error E and computed norm h, rho=E/(h-E) bounds logarithmic error;
// 2*rho bounds phase error. Elementary log, reconstruction and destination
// rounding are included before accepting the unchanged positive or conditioned
// envelopes. This tier admits n<=2^24, including short and centered inputs.
// Weighted tails below -700, subnormal products, range hazards, unresolved
// component zeros, conditioning above 2^20 and near-zero logarithms fall back.
// No weighted tail is discarded. See the weighted helper for the derivation.
//
// Uncertified positive and weighted sums retain the four-component
// kernel. Shifts are error-free two-term differences. Its exponential
// uses a four-component ln(2) reduction, degree-24 Taylor expansion on
// |r| <= ln(2)/512, and eight expm1 doublings. The series remainder is below
// 2^-315 before doubling; the split-constant error contributes less than
// 2^-206 for |binary exponent| <= 1732. Fewer than 2^16 component insertions
// per exponential, each discarding only beyond four components, leave ample
// margin in a conservative 2^-180 relative exponential error budget.
//
// For logits of magnitude at most 2^-27, the same owner accumulates 2*exp(x)
// as 2 + 2*x + x²*C(x), where C(x)=2*(exp(x)-1-x)/x². The constant and
// linear terms are exact, and the quadratic term has an independent binary
// scale. The coefficient series through x^8 has remainder <2^-267; below
// 2^-600, C=1 drops only O(x³) in the exponential. This retains curvature
// after linear cancellation, even at the subnormal output grid. Small unit
// residuals use a factored ten-term log1p series (relative remainder <2^-270).
//
// Products with supplied Cartesian weights and accumulation of all expansion
// components are exact. Thus term errors add as sum(|term_i| * eta_i), not as
// an n*epsilon error from floating-point accumulation. Four-component extraction
// discards less than 2^-211 of its binary frame. Three logarithmic Newton
// corrections square the initial elementary error on each step; reconstruction
// adds the anchor, mantissa logarithm and split binary exponent before rounding.
// The accumulator's binary frame is [-3222,3177], with an optional exact
// division by two for the centered series. Multiplying split ln(2) therefore
// contributes <2^-205 over the complete reconstruction exponent range.
//
// If Z is the exact sum, A=sum(|term_i|), and accumulated term error is E<|Z|,
// its contribution to log-magnitude error is at most -ln(1-E/|Z|).
// This explains the conditioning factor kappa=A/|Z| for signed/complex sums;
// positive sums have kappa=1. Reconstruction error is additional and must be
// compared with ULP(log(|Z|)), not ULP(Z). The committed two-precision campaign
// enforces 16 destination ULP on positive sums and real nonnegative weights,
// including translated zero neighborhoods, for lengths through 100000.
// This is sampled qualification, not an exhaustive IEEE-domain certificate.
//
// Complex-logit phases have at most the elementary trigonometric error; their
// radial error is removed before accumulation. A common input phase factors
// out exactly and delegates its magnitude to the positive reduction. For k
// occupied windows, a conservative signed/complex envelope is
// 64*k*epsilon*kappa*(1+|log(|Z|)|) + 8*minimum_subnormal, in the resolved
// regime 32*k*epsilon*kappa < 1/2; outside it forward conditioning is unbounded.
// Exact equal-logit cancellations and zero identities remain separate rules.
//
// Fallback positive discarded tails contribute at most n*exp(-1200); n<2^64
// makes this smaller than 2^-1667. Weighted tails are deferred, never dropped across
// cancelling windows. Certified sums take O(n) work; the weighted/complex
// fallback takes O(n*k) (k<=n), at most O(n²), with
// fixed storage and no length cap beyond representable slices/accumulator
// capacity. Tiny real/imaginary-axis logarithms subtract the unit in the exact
// accumulator and reconstruct on a grid raised by 2^512, preserving zero signs.
// Projection sums retained components directly on the destination IEEE grid;
// binary32 midpoints are not rounded through an intermediate binary64 log.

use crate::stable::{domain, input, output};

/// A logit that may be masked: `−∞` stands for an exact zero exponential term.
#[derive(Clone, Copy)]
pub(crate) enum Logit {
    Masked,
    Finite(f64),
}
pub(crate) fn logit<S: RealScalar>(x: S) -> Result<Logit, SciOxideError> {
    if x == S::NEG_INFINITY {
        Ok(Logit::Masked)
    } else {
        input(x).map(Logit::Finite)
    }
}
#[derive(Clone, Copy)]
pub(crate) enum ZeroSum {
    Empty,
    Cancelled,
}
pub(crate) enum WeightedSum<S> {
    Zero(ZeroSum),
    Nonzero { phase: Complex<f64>, log: S },
}
impl<S: RealScalar> WeightedSum<S> {
    pub(crate) fn principal(self, argument: crate::Argument) -> Result<Complex<S>, SciOxideError> {
        match self {
            Self::Zero(ZeroSum::Empty) => Ok(Complex::new(S::NEG_INFINITY, S::ZERO)),
            Self::Zero(ZeroSum::Cancelled) => Err(domain(argument)),
            Self::Nonzero { phase, log } => Ok(Complex::new(
                log,
                output(RealScalar::atan2(phase.im, phase.re))?,
            )),
        }
    }
}

use crate::size::validate_same_len;
use crate::{Complex, LogValue, RangeAccumulator, RealScalar, SciOxideError};

mod expansion;
use expansion::Expansion;
mod double_double;
use double_double::DoubleDouble;
mod weighted_precision;
mod working_precision;
pub(crate) use working_precision::{PositiveInputs, positive_inputs, working_sum};

// Constant-storage two-term path. Native exp/log1p use SciOxide's documented
// four-ULP elementary envelope (not an assumed correctly rounded host libm).
#[inline]
pub(crate) fn two_term_logarithm<S: RealScalar>(x: S, y: S) -> Result<S, SciOxideError> {
    let x = logit(x)?;
    let y = logit(y)?;
    let (x, y) = match (x, y) {
        (Logit::Masked, Logit::Masked) => return Ok(S::NEG_INFINITY),
        (Logit::Finite(x), Logit::Masked) | (Logit::Masked, Logit::Finite(x)) => {
            return output(x + 0.0);
        }
        (Logit::Finite(x), Logit::Finite(y)) => (x, y),
    };
    let max = x.max(y);
    let difference = (x - y).abs();
    let term = RealScalar::exp(-difference);
    let correction = RealScalar::ln_1p(term);
    let value = max + correction;
    // Rounded subtraction has |error|<=2*u*|difference|. exp's four
    // ULP are <=8*u*exp; log1p is 1-Lipschitz on nonnegative arguments.
    // The factor 16 covers reference/computed conversion and the tiny
    // quadratic shift error. Absolute underflow is covered independently.
    let local = if term == 0.0 {
        0.0
    } else {
        ((2.0 * difference + 16.0) * term + 16.0 * correction) * f64::EPSILON * 0.5
    };
    let bound = add_up(
        add_up(local, 8.0 * f64::from_bits(1)),
        destination_ulp::<f64>(value),
    );
    let low = RealScalar::next_after(value.abs() - bound, 0.0);
    if low >= S::MIN_POSITIVE.into_f64()
        && bound.is_finite()
        && bound <= 14.0 * destination_ulp::<S>(low)
        && value.abs() > (1.0 + max.abs() + correction) / (1_u64 << 20) as f64
    {
        return output(value);
    }
    let values = [S::try_from_f64_lossy(x)?, S::try_from_f64_lossy(y)?];
    positive_sum(values.into_iter(), CertificationTarget::Logarithm)?.logarithm()
}

/// The quantity whose accuracy in the output type must be certified.
#[derive(Clone, Copy)]
pub(crate) enum CertificationTarget {
    Logarithm,
    Normalization,
}

// All operands are nonnegative. Advancing one IEEE value includes rounding
// of the bound itself, even when a product underflows to zero.
#[inline]
fn add_up(a: f64, b: f64) -> f64 {
    let value = a + b;
    if value.is_finite() {
        f64::from_bits(value.to_bits() + 1)
    } else {
        value
    }
}

#[inline]
fn mul_up(a: f64, b: f64) -> f64 {
    let value = a * b;
    if value.is_finite() {
        f64::from_bits(value.to_bits() + 1)
    } else {
        value
    }
}

// Evaluate the destination ULP at a binary64 interval endpoint without first
// rounding that endpoint to S (which could cross a destination binade).
#[inline]
fn destination_ulp<S: RealScalar>(value: f64) -> f64 {
    let bits = value.abs().to_bits();
    let field = ((bits >> 52) & 0x7ff) as i32;
    let exponent = if field == 0 {
        63 - bits.leading_zeros() as i32 - 1074
    } else {
        field - 1023
    };
    let minimum = if S::MANTISSA_DIGITS == 24 {
        -149
    } else {
        -1074
    };
    let grid = (exponent - S::MANTISSA_DIGITS as i32 + 1).max(minimum);
    if grid >= -1022 {
        f64::from_bits(((grid + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (grid + 1074))
    }
}

struct DoubleDoubleSum {
    shifted_log: DoubleDouble,
    logarithm: DoubleDouble,
    shifted_bound: f64,
    bound: f64,
}

impl DoubleDoubleSum {
    fn certifies<S: RealScalar>(&self, target: CertificationTarget) -> bool {
        let (value, bound) = match target {
            CertificationTarget::Logarithm => (self.logarithm, self.bound),
            CertificationTarget::Normalization => (self.shifted_log, self.shifted_bound),
        };
        let magnitude = RealScalar::next_after(value.hi.abs() - value.lo.abs(), 0.0);
        let low = RealScalar::next_after(magnitude - bound, 0.0);
        low > 0.0 && bound.is_finite() && bound <= 4.0 * destination_ulp::<S>(low)
    }

    fn finish(self, anchor: f64) -> PositiveSum {
        PositiveSum {
            anchor,
            shifted_log: self.shifted_log.expansion(),
            logarithm: PositiveLogarithm::Expansion(self.logarithm.expansion()),
        }
    }
}

fn double_double_positive_sum<S: RealScalar>(
    values: impl Iterator<Item = S>,
    max: f64,
) -> Result<DoubleDoubleSum, SciOxideError> {
    let mut sum = RangeAccumulator::<f64>::new();
    let mut omitted_maximum = false;
    let mut terms = 0_usize;
    let mut dropped = 0_usize;
    for x in values {
        let x = x.into_f64();
        if x == f64::NEG_INFINITY {
            continue;
        }
        if !omitted_maximum && x == max {
            omitted_maximum = true;
            continue;
        }
        terms += 1;
        if x - max < -740.0 {
            dropped += 1;
            continue;
        }
        let (hi, lo) = crate::arithmetic::two_sum(x, -max);
        let term = DoubleDouble { hi, lo }.exp();
        sum.add(term.hi)?;
        sum.add(term.lo)?;
    }
    let r = DoubleDouble::from_sum(&sum);
    let (shifted_log, logarithm_error) = r.ln_1p();
    let logarithm = DoubleDouble::from(max).add(shifted_log);
    let r_upper = add_up(r.hi.abs(), r.lo.abs());
    // A term's relative evaluation error, referred to its computed value, is
    // <=2*eta_exp. Extraction discards <2^-105 of the positive exact sum.
    let relative = add_up(
        RealScalar::scalbn(1.0_f64, -99),
        RealScalar::scalbn(1.0_f64, -104),
    );
    let underflow = mul_up(
        RealScalar::next_after(terms as f64, f64::INFINITY),
        RealScalar::scalbn(1.0_f64, -1073),
    );
    // Rounded shifts below -740 have exact shifts below -740+2^-43.
    // exp(-740+2^-43)<2^-1067, also covering overflowed negative differences.
    let tail = mul_up(
        RealScalar::next_after(dropped as f64, f64::INFINITY),
        RealScalar::scalbn(1.0_f64, -1067),
    );
    let mass_error = add_up(mul_up(relative, r_upper), add_up(underflow, tail));
    // The true total is >=max(1,r_hi)/2. Since mass_error/total<1/2,
    // -ln(1-mass_error/total)<=2*mass_error/total.
    let mass_log_error =
        RealScalar::next_after(mul_up(4.0, mass_error) / r.hi.max(1.0), f64::INFINITY);
    let shifted_bound = add_up(logarithm_error, mass_log_error);
    // TwoSum itself is exact. Low-component additions incur at most 4*u²
    // of the operand magnitudes; 2^-102 leaves a factor-four safety margin.
    // Scale operands separately to keep the bound finite at f64::MAX.
    let reconstruction_error = add_up(
        mul_up(RealScalar::scalbn(1.0_f64, -102), max.abs()),
        mul_up(RealScalar::scalbn(1.0_f64, -102), shifted_log.hi.abs()),
    );
    Ok(DoubleDoubleSum {
        shifted_log,
        logarithm,
        shifted_bound,
        bound: add_up(shifted_bound, reconstruction_error),
    })
}

// Unit phases are an operation-specific capability: Cartesian input weights
// must retain their supplied magnitude, while exp(i*angle) has exact norm one.
#[derive(Clone, Copy)]
pub(crate) enum LogWeight {
    Cartesian(Complex<f64>),
    UnitPhase(f64),
}

impl LogWeight {
    fn nonzero(self) -> bool {
        match self {
            Self::Cartesian(w) => w.re != 0.0 || w.im != 0.0,
            Self::UnitPhase(_) => true,
        }
    }
    fn validate(self) -> Result<bool, SciOxideError> {
        match self {
            Self::Cartesian(w) => {
                input(w.re)?;
                input(w.im)?;
                Ok(w.re != 0.0 || w.im != 0.0)
            }
            Self::UnitPhase(angle) => {
                input(angle)?;
                Ok(true)
            }
        }
    }

    fn components(self) -> [Expansion; 2] {
        match self {
            Self::Cartesian(w) => [Expansion::from(w.re), Expansion::from(w.im)],
            Self::UnitPhase(angle) => {
                let (s, c) = RealScalar::sin_cos(angle);
                let parts = [Expansion::from(c), Expansion::from(s)];
                let norm = parts[0].mul(parts[0]).add(parts[1].mul(parts[1])).sqrt();
                parts.map(|part| part.div(norm))
            }
        }
    }
}

pub(crate) enum PositiveReduction {
    Zero,
    Nonzero(PositiveSum),
}
impl PositiveReduction {
    pub(crate) fn logarithm<S: RealScalar>(&self) -> Result<S, SciOxideError> {
        match self {
            Self::Zero => Ok(S::NEG_INFINITY),
            Self::Nonzero(sum) => sum.logarithm(),
        }
    }
    pub(crate) fn nonzero(self, argument: crate::Argument) -> Result<PositiveSum, SciOxideError> {
        match self {
            Self::Zero => Err(domain(argument)),
            Self::Nonzero(sum) => Ok(sum),
        }
    }
}
pub(crate) struct PositiveSum {
    pub(crate) anchor: f64,
    shifted_log: Expansion,
    logarithm: PositiveLogarithm,
}

enum PositiveLogarithm {
    Working(f64),
    Expansion(Expansion),
}

#[cfg(test)]
impl PositiveLogarithm {
    fn expansion(&self) -> Expansion {
        match *self {
            Self::Working(value) => Expansion::from(value),
            Self::Expansion(value) => value,
        }
    }
}

impl PositiveSum {
    pub(crate) fn shifted_log(&self) -> f64 {
        self.shifted_log.rounded()
    }

    pub(crate) fn logarithm<S: RealScalar>(&self) -> Result<S, SciOxideError> {
        match self.logarithm {
            PositiveLogarithm::Working(value) => output(value),
            PositiveLogarithm::Expansion(value) => value.round_to(),
        }
    }
}

// `argument` names its logit parameter.
pub(crate) fn positive_sum<S: RealScalar>(
    values: impl Iterator<Item = S> + Clone,
    target: CertificationTarget,
) -> Result<PositiveReduction, SciOxideError> {
    let inputs = positive_inputs(values.clone())?;
    let PositiveInputs {
        max,
        centered,
        active,
    } = inputs;
    if active == 0 {
        return Ok(PositiveReduction::Zero);
    }
    if active == 1 {
        // The singleton's shifted sum is exactly one. Reconstructing then
        // subtracting its logit would inject a residual into log_softmax.
        return Ok(PositiveReduction::Nonzero(PositiveSum {
            anchor: max,
            shifted_log: Expansion::ZERO,
            logarithm: PositiveLogarithm::Working(max + 0.0),
        }));
    }
    let working = working_sum(values.clone(), inputs, |_| Ok(()))?;
    if working.certifies::<S>(target) {
        return Ok(PositiveReduction::Nonzero(working.finish(max)));
    }
    let fast = double_double_positive_sum(values.clone(), max)?;
    if fast.certifies::<S>(target) {
        return Ok(PositiveReduction::Nonzero(fast.finish(max)));
    }
    exact_positive_sum(values, max, centered).map(PositiveReduction::Nonzero)
}

// The qualification kernel is retained unchanged for uncertified inputs.
fn exact_positive_sum<S: RealScalar>(
    values: impl Iterator<Item = S>,
    max: f64,
    centered: bool,
) -> Result<PositiveSum, SciOxideError> {
    let mut sum = RangeAccumulator::<f64>::new();
    for x in values {
        let x = x.into_f64();
        if x == f64::NEG_INFINITY {
            continue;
        }
        if centered {
            Expansion::accumulate_centered(&mut sum, Expansion::ONE, x)?;
            continue;
        }
        // Far tails cannot affect even a subnormal logarithm: exp(-1200)
        // times the maximum supported usize length is below 2^-1667.
        if x - max < -1200.0 {
            continue;
        }
        let delta = Expansion::from(x).sub(Expansion::from(max));
        let (mantissa, exponent) = delta.exp_scaled();
        mantissa.accumulate(&mut sum, 1.0, exponent)?;
    }
    let (mantissa, exponent) = Expansion::from_sum(&sum);
    let anchor = if centered { 0.0 } else { max };
    let scale = if centered { -1 } else { 0 };
    let exponent = exponent + scale;
    let logarithm = match Expansion::small_real_log(anchor, &sum, scale)? {
        Some(log) => log,
        None => Expansion::reconstruct(Expansion::from(anchor), mantissa, exponent),
    };
    Ok(PositiveSum {
        anchor: max,
        shifted_log: if centered {
            logarithm.sub(Expansion::from(max))
        } else {
            mantissa.ln().add(Expansion::log_two(exponent))
        },
        logarithm: PositiveLogarithm::Expansion(logarithm),
    })
}

/// Computes `logsumexp(x) = ln Σᵢ e^(xᵢ)` without overflow.
///
/// A `−∞` logit is a masked term that contributes zero, and an empty or fully
/// masked slice returns `−∞`; use [`logsumexp_value`] to get an exact zero
/// instead. Complex logits give the principal logarithm. Their imaginary parts
/// must be finite, also on masked terms, and an empty or fully masked complex sum
/// returns `(−∞, +0)`.
///
/// The terms are summed without a rounding error that grows with the input
/// length. Real sums take O(n) time. Complex sums whose phases differ can take
/// O(n k) time, where `k ≤ n` is the number of occupied windows of width 1200
/// among the real parts of the logits.
///
/// # Accuracy
///
/// Real results are within 16 ulps, including results near zero and subnormal
/// results. For complex logits the error depends on the cancellation
/// `κ = Σᵢ |tᵢ| / |Σᵢ tᵢ|` among the terms `tᵢ = e^(xᵢ)`: the logarithm of the
/// magnitude is within `64 k ε κ (1 + |ln|Σᵢ tᵢ||) + 8 η` when
/// `32 k ε κ < 1/2`, where `ε` is the machine epsilon and `η` the smallest
/// positive subnormal.
///
/// # Errors
///
/// - `NonFinite` if a logit is NaN or `+∞`, or an imaginary part is not finite.
/// - `DomainError` if complex terms cancel exactly.
/// - `NumericalInstability` if the result overflows.
///
/// # Examples
///
/// ```
/// // The f64 nearest to −ln 2 is slightly above it, so the exact sum
/// // 2 e^x is slightly above 1 and its logarithm is tiny but nonzero.
/// let x = -core::f64::consts::LN_2;
/// let log = scioxide_core::logsumexp(&[x, x])?;
/// assert_eq!(log.to_bits(), 0x3c7a_bc9e_3b39_803f);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn logsumexp<T: crate::NumericScalar>(values: &[T]) -> Result<T, SciOxideError> {
    if T::IS_REAL {
        let sum = positive_sum(
            values.iter().map(|value| value.real_component()),
            CertificationTarget::Logarithm,
        )?;
        return Ok(T::from_real(sum.logarithm()?));
    }
    let value = weighted_components(
        values.len(),
        |i| values[i].real_component(),
        |i| LogWeight::UnitPhase(values[i].imaginary_component().into_f64()),
    )?
    .principal(crate::argument!("values"))?;
    Ok(T::from_components(value.re, value.im))
}

/// Computes `Σᵢ e^(xᵢ)` as a [`LogValue`].
///
/// The result holds the sign or complex phase and the logarithm of the
/// magnitude. Unlike [`logsumexp`], an empty or fully masked input, and complex
/// terms that cancel exactly, return the zero `LogValue` instead of `−∞` or an
/// error. A nonzero sum has the accuracy documented on [`logsumexp`].
///
/// # Errors
///
/// - `NonFinite` if a logit is NaN or `+∞`, or an imaginary part is not finite.
/// - `NumericalInstability` if the logarithm of the magnitude overflows.
pub fn logsumexp_value<T: crate::NumericScalar>(
    values: &[T],
) -> Result<LogValue<T, T::Real>, SciOxideError> {
    if T::IS_REAL {
        return match positive_sum(
            values.iter().map(|value| value.real_component()),
            CertificationTarget::Logarithm,
        )? {
            PositiveReduction::Zero => Ok(LogValue::ZERO),
            PositiveReduction::Nonzero(sum) => LogValue::from_log(T::ONE, sum.logarithm()?),
        };
    }
    match weighted_components(
        values.len(),
        |i| values[i].real_component(),
        |i| LogWeight::UnitPhase(values[i].imaginary_component().into_f64()),
    )? {
        WeightedSum::Zero(_) => Ok(LogValue::ZERO),
        WeightedSum::Nonzero { phase, log } => LogValue::from_log(
            T::from_components(output(phase.re)?, output(phase.im)?),
            log,
        ),
    }
}

// Each window fits the exact triple-product accumulator, including the entire input weight
// range. An exactly cancelled window is discarded before the next window is scaled. This
// retains small survivors behind arbitrarily large, exactly cancelling equal-logit terms
// without allocating sorted storage.
pub(crate) fn weighted_components<S: RealScalar>(
    len: usize,
    value: impl Fn(usize) -> S,
    weight: impl Fn(usize) -> LogWeight,
) -> Result<WeightedSum<S>, SciOxideError> {
    if len != 0
        && let LogWeight::UnitPhase(angle) = weight(0)
        && (0..len).all(|i| matches!(weight(i), LogWeight::UnitPhase(a) if a == angle))
    {
        input(angle)?;
        let sum = positive_sum((0..len).map(&value), CertificationTarget::Logarithm)?;
        return match sum {
            PositiveReduction::Zero => Ok(WeightedSum::Zero(ZeroSum::Empty)),
            PositiveReduction::Nonzero(sum) => {
                let (s, c) = RealScalar::sin_cos(angle);
                Ok(WeightedSum::Nonzero {
                    phase: Complex::new(c, s),
                    log: sum.logarithm()?,
                })
            }
        };
    }
    // Validate every component before ignoring masked or zero-weight terms.
    let mut anchor = f64::NEG_INFINITY;
    for i in 0..len {
        let x = logit(value(i))?;
        let w = weight(i).validate()?;
        if let Logit::Finite(x) = x
            && w
        {
            anchor = anchor.max(x);
        }
    }
    if anchor == f64::NEG_INFINITY {
        return Ok(WeightedSum::Zero(ZeroSum::Empty));
    }
    if let Some(sum) = weighted_precision::weighted_sum(len, &value, &weight, anchor)? {
        #[cfg(test)]
        WEIGHTED_TIERS.with(|counts| {
            let [cheap, exact] = counts.get();
            counts.set([cheap + 1, exact]);
        });
        return Ok(sum);
    }
    windowed_components(len, value, weight)
}

#[cfg(test)]
std::thread_local! {
    static WEIGHTED_TIERS: core::cell::Cell<[usize; 2]> = const { core::cell::Cell::new([0; 2]) };
}

// All components were validated by the public reduction owner. The strict
// window path preserves centered reconstruction and cancellation survivors.
fn windowed_components<S: RealScalar>(
    len: usize,
    value: impl Fn(usize) -> S,
    weight: impl Fn(usize) -> LogWeight,
) -> Result<WeightedSum<S>, SciOxideError> {
    #[cfg(test)]
    WEIGHTED_TIERS.with(|counts| {
        let [cheap, exact] = counts.get();
        counts.set([cheap, exact + 1]);
    });
    let mut ceiling = f64::INFINITY;
    let mut result = [Expansion::ZERO; 2];
    let mut result_log = Expansion::ZERO;
    loop {
        // Window invariant: every logit at or above `ceiling` was accumulated by
        // an earlier window. The anchor is the largest remaining logit with a
        // nonzero weight.
        let mut anchor = f64::NEG_INFINITY;
        let mut centered = true;
        for i in 0..len {
            let x = value(i).into_f64();
            let w = weight(i);
            if x != f64::NEG_INFINITY && w.nonzero() && x < ceiling {
                anchor = anchor.max(x);
                centered &= x.abs() <= RealScalar::scalbn(1.0_f64, -27);
            }
        }
        if anchor == f64::NEG_INFINITY {
            break;
        }
        let mut re = RangeAccumulator::<f64>::new();
        let mut im = RangeAccumulator::<f64>::new();
        let mut next = f64::NEG_INFINITY;
        for i in 0..len {
            let x = value(i);
            let x = x.into_f64();
            if x == f64::NEG_INFINITY || x >= ceiling {
                continue;
            }
            let w = weight(i).components();
            if w.iter().all(|part| part.leading() == 0.0) {
                continue;
            }
            if centered {
                for (sum, part) in [(&mut re, w[0]), (&mut im, w[1])] {
                    Expansion::accumulate_centered(sum, part, x)?;
                }
                continue;
            }
            // A 1200-wide window leaves room for all four exponential
            // components and the complete weight range in the exact bins.
            if x - anchor < -1200.0 {
                next = next.max(x);
                continue;
            }
            let delta = Expansion::from(x).sub(Expansion::from(anchor));
            let (mantissa, exponent) = delta.exp_scaled();
            for (sum, part) in [(&mut re, w[0]), (&mut im, w[1])] {
                mantissa.accumulate_weight(sum, part, exponent)?;
            }
        }
        let (real, er) = Expansion::from_sum(&re);
        let (imaginary, ei) = Expansion::from_sum(&im);
        let (r, i) = (real.leading(), imaginary.leading());
        if r != 0.0 || i != 0.0 {
            // Turn the window sum into (unit phase, log magnitude) with a shared
            // binary exponent, then merge it into the running result by log-domain
            // addition relative to the larger magnitude.
            let exponent = if r == 0.0 {
                ei
            } else if i == 0.0 {
                er
            } else {
                er.max(ei)
            };
            let z = [real.scale(er - exponent), imaginary.scale(ei - exponent)];
            let scale = if centered { -1 } else { 0 };
            let exponent = exponent + scale;
            let anchor = if centered { 0.0 } else { anchor };
            let magnitude = z[0].mul(z[0]).add(z[1].mul(z[1])).sqrt();
            let small = if i == 0.0 {
                Expansion::small_real_log(anchor, &re, scale)?
            } else if r == 0.0 {
                Expansion::small_real_log(anchor, &im, scale)?
            } else {
                None
            };
            let log = small.unwrap_or_else(|| {
                Expansion::reconstruct(Expansion::from(anchor), magnitude, exponent)
            });
            let phase = z.map(|part| part.div(magnitude));
            if result.iter().all(|part| part.leading() == 0.0) {
                result = phase;
                result_log = log;
            } else {
                let scale = if result_log.leading() >= log.leading() {
                    result_log
                } else {
                    log
                };
                // Check large differences before error-free subtraction: the
                // anchors themselves can span the whole finite IEEE range.
                let factor = |value: Expansion| {
                    if value.leading() - scale.leading() < -1200.0 {
                        Expansion::ZERO
                    } else {
                        value.sub(scale).exp()
                    }
                };
                let left = factor(result_log);
                let right = factor(log);
                let sum = core::array::from_fn::<_, 2, _>(|j| {
                    result[j].mul(left).add(phase[j].mul(right))
                });
                if sum.iter().all(|part| part.leading() == 0.0) {
                    result = [Expansion::ZERO; 2];
                    result_log = Expansion::ZERO;
                } else {
                    // Normalize before squaring: cancellation can leave a tiny
                    // surviving component whose square would underflow.
                    let mut real = RangeAccumulator::<f64>::new();
                    let mut imaginary = RangeAccumulator::<f64>::new();
                    sum[0].accumulate(&mut real, 1.0, 0)?;
                    sum[1].accumulate(&mut imaginary, 1.0, 0)?;
                    let (r, er) = Expansion::from_sum(&real);
                    let (i, ei) = Expansion::from_sum(&imaginary);
                    let exponent = if r.leading() == 0.0 {
                        ei
                    } else if i.leading() == 0.0 {
                        er
                    } else {
                        er.max(ei)
                    };
                    let z = [r.scale(er - exponent), i.scale(ei - exponent)];
                    let norm = z[0].mul(z[0]).add(z[1].mul(z[1])).sqrt();
                    result = z.map(|part| part.div(norm));
                    result_log = Expansion::reconstruct(scale, norm, exponent);
                }
            }
        }
        if next == f64::NEG_INFINITY {
            break;
        }
        ceiling = RealScalar::next_after(next, f64::INFINITY);
    }
    if result.iter().all(|part| part.leading() == 0.0) {
        Ok(WeightedSum::Zero(ZeroSum::Cancelled))
    } else {
        Ok(WeightedSum::Nonzero {
            phase: Complex::new(result[0].rounded(), result[1].rounded()),
            log: result_log.round_to()?,
        })
    }
}

/// Computes `Σᵢ wᵢ e^(xᵢ)` for real logits `x` and real or complex weights `w`, as a [`LogValue`].
///
/// The result holds the sign or complex phase and the logarithm of the
/// magnitude. Masked logits and zero weights contribute nothing; an empty input
/// or exact cancellation returns the zero `LogValue`. Every weight component must
/// be finite, also for masked and zero-weight terms. Takes O(n) time in most
/// cases and at most O(n k), where `k ≤ n` is the number of occupied windows of
/// width 1200 among the logits.
///
/// # Accuracy
///
/// With nonnegative weights, the logarithm of the magnitude is within 16 ulps.
/// Signed and complex weights can cancel: with `κ = Σᵢ |tᵢ| / |Σᵢ tᵢ|` for the
/// terms `tᵢ = wᵢ e^(xᵢ)`, the error is at most
/// `64 k ε κ (1 + |ln|Σᵢ tᵢ||) + 8 η` when `32 k ε κ < 1/2`, where `ε` is the
/// machine epsilon and `η` the smallest positive subnormal.
///
/// # Errors
///
/// - `DimensionMismatch` if `values` and `weights` differ in length.
/// - `NonFinite` if a logit is NaN or `+∞`, or a weight component is not finite.
/// - `NumericalInstability` if the logarithm of the magnitude overflows.
///
/// # Examples
///
/// ```
/// let x = -core::f64::consts::LN_2;
/// // Same sum as the `logsumexp` example.
/// let sum = scioxide_core::weighted_logsumexp(&[x, x], &[1.0_f64, 1.0])?;
/// assert_eq!(
///     sum.log_magnitude().map(f64::to_bits),
///     Some(0x3c7a_bc9e_3b39_803f)
/// );
/// // Equal terms with opposite signs cancel exactly.
/// let zero = scioxide_core::weighted_logsumexp(&[x, x], &[1.0_f64, -1.0])?;
/// assert!(zero.log_magnitude().is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn weighted_logsumexp<T: crate::NumericScalar>(
    values: &[T::Real],
    weights: &[T],
) -> Result<LogValue<T, T::Real>, SciOxideError> {
    validate_same_len(values.len(), weights.len())?;
    weighted_value(values.len(), |i| values[i], |i| weights[i])
}

pub(crate) fn weighted_value<T: crate::NumericScalar>(
    len: usize,
    value: impl Fn(usize) -> T::Real,
    weight: impl Fn(usize) -> T,
) -> Result<LogValue<T, T::Real>, SciOxideError> {
    let sum = weighted_components(len, value, |i| {
        let w = weight(i);
        LogWeight::Cartesian(Complex::new(
            w.real_component().into_f64(),
            w.imaginary_component().into_f64(),
        ))
    })?;
    match sum {
        WeightedSum::Zero(_) => Ok(LogValue::ZERO),
        WeightedSum::Nonzero { phase, log } => LogValue::from_log(
            T::from_components(output(phase.re)?, output(phase.im)?),
            log,
        ),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;
    use alloc::vec::Vec;

    fn assert_bound(actual: Expansion, expected: Expansion, bound: f64) {
        let error = actual.sub(expected);
        let error = if error.leading() < 0.0 {
            error.neg()
        } else {
            error
        };
        assert!(
            error.sub(Expansion::from(bound)).leading() <= 0.0,
            "error {:?} exceeds bound {bound:e}; actual={actual:?}, expected={expected:?}",
            error,
        );
    }

    #[test]
    fn double_double_constants_match_expansion() {
        let ln_two = double_double::LN_2
            .into_iter()
            .fold(Expansion::ZERO, |s, x| s.add(Expansion::from(x)));
        assert_bound(
            ln_two,
            Expansion::log_two(1),
            RealScalar::scalbn(1.0_f64, -140),
        );
        for k in -1100..=1100 {
            let k = f64::from(k);
            let high = double_double::LN_2[0];
            assert_eq!(RealScalar::mul_add(k, high, -(k * high)), 0.0);
        }
        let mut factorial = 1.0;
        for (index, coefficient) in double_double::RECIPROCAL_FACTORIALS.into_iter().enumerate() {
            factorial *= (index + 2) as f64;
            let reference = Expansion::ONE.div(Expansion::from(factorial));
            assert_bound(
                coefficient.expansion(),
                reference,
                RealScalar::scalbn(reference.leading(), -106),
            );
        }
    }

    #[test]
    fn double_double_exponential_bound() {
        for hi in [
            -740.0, -739.5, -709.0, -500.0, -100.0, -50.0, -1.0, -0.35, -0.01, 0.0, 0.35, 1.0,
            10.0, 44.0,
        ] {
            for lo in [-1.0, 0.0, 1.0] {
                let value = DoubleDouble::pair(hi, lo * RealScalar::scalbn(hi.abs().max(1.0), -55));
                let reference = value.expansion().exp();
                let bound = add_up(
                    mul_up(RealScalar::scalbn(1.0_f64, -100), reference.leading()),
                    RealScalar::scalbn(1.0_f64, -1073),
                );
                assert_bound(value.exp().expansion(), reference, bound);
            }
        }
        // Exercise both sides of every Cody-Waite boundary in the supported
        // domain, where the rounded choice of binary exponent can change.
        for k in -1067..=64 {
            let boundary = (f64::from(k) + 0.5) * core::f64::consts::LN_2;
            for hi in [
                RealScalar::next_after(boundary, f64::NEG_INFINITY),
                boundary,
                RealScalar::next_after(boundary, f64::INFINITY),
            ] {
                let value = DoubleDouble::from(hi);
                let reference = value.expansion().exp();
                let bound = add_up(
                    mul_up(RealScalar::scalbn(1.0_f64, -100), reference.leading()),
                    RealScalar::scalbn(1.0_f64, -1073),
                );
                assert_bound(value.exp().expansion(), reference, bound);
            }
        }
    }

    fn certification_boundaries<S: RealScalar>() {
        let ulp = destination_ulp::<S>(1.0);
        let candidate = |value, bound| DoubleDoubleSum {
            shifted_log: DoubleDouble::from(value),
            logarithm: DoubleDouble::from(value),
            shifted_bound: bound,
            bound,
        };
        assert!(
            candidate(1.0 + 8.0 * ulp, 4.0 * ulp).certifies::<S>(CertificationTarget::Logarithm)
        );
        assert!(
            !candidate(1.0 + 8.0 * ulp, mul_up(4.0, ulp))
                .certifies::<S>(CertificationTarget::Logarithm)
        );
        // Below a binade boundary the grid is twice as fine. Converting the
        // low endpoint to f32 first would incorrectly accept some intervals.
        assert!(!candidate(1.0, 4.0 * ulp).certifies::<S>(CertificationTarget::Logarithm));
        assert!(candidate(1.0, 2.0 * ulp).certifies::<S>(CertificationTarget::Logarithm));
        assert_eq!(
            destination_ulp::<S>(RealScalar::next_after(1.0_f64, 0.0)),
            ulp * 0.5
        );
        let subnormal = S::ZERO.next_after(S::ONE).into_f64();
        assert_eq!(destination_ulp::<S>(0.0), subnormal);
        assert_eq!(destination_ulp::<S>(subnormal), subnormal);
        assert!(!candidate(subnormal, subnormal).certifies::<S>(CertificationTarget::Logarithm));
        let minimum = S::try_from_f64_lossy(subnormal).unwrap();
        let fast = double_double_positive_sum([minimum, -S::MAX].into_iter(), subnormal).unwrap();
        assert!(!fast.certifies::<S>(CertificationTarget::Logarithm));
    }

    #[test]
    fn destination_grid_certification_boundaries() {
        certification_boundaries::<f32>();
        certification_boundaries::<f64>();
    }

    #[test]
    fn far_tail_and_extreme_shifts() {
        for max in [0.0, 3.0] {
            for shift in [
                RealScalar::next_after(-740.0_f64, f64::NEG_INFINITY),
                -740.0,
                RealScalar::next_after(-740.0_f64, 0.0),
                -f64::MAX,
            ] {
                let values = [max, max + shift];
                let fast = double_double_positive_sum(values.into_iter(), max).unwrap();
                let exact = exact_positive_sum(values.into_iter(), max, false).unwrap();
                assert_bound(
                    fast.logarithm.expansion(),
                    exact.logarithm.expansion(),
                    fast.bound,
                );
                assert_eq!(
                    fast.certifies::<f64>(CertificationTarget::Logarithm),
                    max != 0.0
                );
            }
        }
    }

    fn compare_seeded<S: RealScalar>() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut certified = 0;
        for exponent in [-1074, -1022, -64, -27, -10, 0, 5, 10, 100, 1023] {
            if S::MANTISSA_DIGITS == 24 && !(-149..=127).contains(&exponent) {
                continue;
            }
            for size in [2, 3, 8, 17, 64, 256, 1024, 4096] {
                let values: Vec<S> = (0..size)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let unit = (seed >> 11) as f64 * RealScalar::scalbn(1.0_f64, -53);
                        let lane_exponent = exponent - (seed & 7) as i32;
                        S::try_from_f64_lossy(RealScalar::scalbn(2.0 * unit - 1.0, lane_exponent))
                            .unwrap()
                    })
                    .collect();
                let max = values
                    .iter()
                    .map(|x| x.into_f64())
                    .fold(f64::NEG_INFINITY, f64::max);
                let fast = double_double_positive_sum(values.iter().copied(), max).unwrap();
                let centered = values
                    .iter()
                    .all(|x| x.into_f64().abs() <= RealScalar::scalbn(1.0_f64, -27));
                let exact = exact_positive_sum(values.iter().copied(), max, centered).unwrap();
                if fast.certifies::<S>(CertificationTarget::Logarithm) {
                    certified += 1;
                    assert_bound(
                        fast.logarithm.expansion(),
                        exact.logarithm.expansion(),
                        fast.bound,
                    );
                    // Destination projection must use the same owner, including
                    // binary32 midpoints and the overflow policy.
                    let projected = fast.logarithm.expansion().round_to::<S>().unwrap();
                    let reference = exact.logarithm::<S>().unwrap();
                    let ulp = destination_ulp::<S>(reference.into_f64());
                    assert!((projected.into_f64() - reference.into_f64()).abs() <= 16.0 * ulp);
                }
                if fast.certifies::<S>(CertificationTarget::Normalization) {
                    assert_bound(
                        fast.shifted_log.expansion(),
                        exact.shifted_log,
                        fast.shifted_bound,
                    );
                }
            }
        }
        assert!(certified >= 40);
    }

    #[test]
    fn certified_fast_matches_exact_f64() {
        compare_seeded::<f64>();
    }

    #[test]
    fn certified_fast_matches_exact_f32() {
        compare_seeded::<f32>();
    }

    #[test]
    fn tier_selection_and_normalization_target() {
        for values in [
            &[3.0, -1.5][..],
            &[0.0, 0.0],
            &[-50.0, -25.0, 0.0, 25.0, 50.0],
        ] {
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let fast = double_double_positive_sum(values.iter().copied(), max).unwrap();
            assert!(fast.certifies::<f64>(CertificationTarget::Logarithm));
            assert!(fast.certifies::<f32>(CertificationTarget::Logarithm));
        }
        let values = [-core::f64::consts::LN_2; 2];
        let fast = double_double_positive_sum(values.into_iter(), values[0]).unwrap();
        assert!(!fast.certifies::<f64>(CertificationTarget::Logarithm));
        assert!(fast.certifies::<f64>(CertificationTarget::Normalization));
        let result = positive_sum(values.into_iter(), CertificationTarget::Logarithm).unwrap();
        assert_eq!(
            result.logarithm::<f64>().unwrap().to_bits(),
            0x3c7a_bc9e_3b39_803f
        );
        let subnormal = f64::from_bits(1);
        for max in [-subnormal, subnormal] {
            let fast = double_double_positive_sum([max, -f64::MAX].into_iter(), max).unwrap();
            assert!(!fast.certifies::<f64>(CertificationTarget::Logarithm));
        }
        let fast = double_double_positive_sum([f64::MAX, -f64::MAX].into_iter(), f64::MAX).unwrap();
        assert!(fast.certifies::<f64>(CertificationTarget::Logarithm));
        assert_eq!(fast.finish(f64::MAX).logarithm::<f64>().unwrap(), f64::MAX);
    }

    fn fixture_tiers<S: RealScalar>() {
        let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
        let mut fallback = 0;
        let mut accepted = 0;
        for line in include_str!("../tests/data/logarithmic.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields[0] != width
                || fields[1] != "positive"
                || !(fields[2] == "ln2-neighborhood"
                    || fields[2].starts_with("unit-sum-")
                    || fields[2] == "dominant-tiny")
            {
                continue;
            }
            let mut values = Vec::new();
            for group in fields[8..].chunks_exact(5) {
                let bits = u64::from_str_radix(group[1], 16).unwrap();
                let x = if width == "32" {
                    f32::from_bits(bits as u32) as f64
                } else {
                    f64::from_bits(bits)
                };
                values.extend(core::iter::repeat_n(
                    S::try_from_f64_lossy(x).unwrap(),
                    group[0].parse::<usize>().unwrap(),
                ));
            }
            let max = values
                .iter()
                .map(|x| x.into_f64())
                .fold(f64::NEG_INFINITY, f64::max);
            let fast = double_double_positive_sum(values.iter().copied(), max).unwrap();
            if fast.certifies::<S>(CertificationTarget::Logarithm) {
                accepted += 1;
                let exact = exact_positive_sum(values.iter().copied(), max, false).unwrap();
                assert_bound(
                    fast.logarithm.expansion(),
                    exact.logarithm.expansion(),
                    fast.bound,
                );
            } else {
                fallback += 1;
            }
            if width == "64" && fields[2] != "dominant-tiny" {
                assert!(
                    !fast.certifies::<S>(CertificationTarget::Logarithm),
                    "{line}"
                );
            }
        }
        assert!(fallback > 0 && accepted > 0);
    }

    #[test]
    fn fixture_tier_selection_both_widths() {
        fixture_tiers::<f64>();
        fixture_tiers::<f32>();
    }

    fn weighted_tiers<S: RealScalar>() {
        let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
        let mut accepted = 0;
        let mut fallback = 0;
        for line in include_str!("../tests/data/logarithmic.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields[0] != width || fields[1] != "weighted" {
                continue;
            }
            // Lengths through 100000 are qualified against MPFR/MPC in
            // tests/logarithmic_reconstruction.rs. The exact accumulator has no
            // length-dependent path, so longer rows here would only repeat
            // identical strict-kernel exponentials.
            let lanes = fields[8..]
                .chunks_exact(5)
                .map(|group| group[0].parse::<usize>().unwrap())
                .sum::<usize>();
            if lanes > 1000 {
                continue;
            }
            let decode = |bits: &str| {
                let bits = u64::from_str_radix(bits, 16).unwrap();
                if width == "32" {
                    f32::from_bits(bits as u32) as f64
                } else {
                    f64::from_bits(bits)
                }
            };
            let mut values = Vec::new();
            let mut weights = Vec::new();
            for group in fields[8..].chunks_exact(5) {
                values.extend(core::iter::repeat_n(
                    decode(group[1]),
                    group[0].parse::<usize>().unwrap(),
                ));
                weights.extend(core::iter::repeat_n(
                    LogWeight::Cartesian(Complex::new(decode(group[3]), decode(group[4]))),
                    group[0].parse::<usize>().unwrap(),
                ));
            }
            let anchor = values
                .iter()
                .copied()
                .zip(&weights)
                .filter(|(x, w)| *x != f64::NEG_INFINITY && w.nonzero())
                .map(|(x, _)| x)
                .fold(f64::NEG_INFINITY, f64::max);
            if !anchor.is_finite() {
                continue;
            }
            let value = |i| {
                if values[i] == f64::NEG_INFINITY {
                    S::NEG_INFINITY
                } else {
                    S::try_from_f64_lossy(values[i]).unwrap()
                }
            };
            if let Some(candidate) =
                weighted_precision::weighted_sum(values.len(), value, |i| weights[i], anchor)
                    .unwrap()
            {
                accepted += 1;
                let strict = windowed_components(values.len(), value, |i| weights[i]).unwrap();
                match (candidate, strict) {
                    (
                        WeightedSum::Nonzero { phase, log },
                        WeightedSum::Nonzero { phase: p, log: l },
                    ) => {
                        let condition = f64::from_bits(u64::from_str_radix(fields[6], 16).unwrap());
                        let epsilon = S::EPSILON.into_f64();
                        let positive = weights.iter().all(
                            |w| matches!(w, LogWeight::Cartesian(w) if w.im == 0.0 && w.re >= 0.0),
                        );
                        let bound = if positive {
                            16.0 * destination_ulp::<S>(l.into_f64())
                        } else {
                            64.0 * epsilon * condition * (1.0 + l.into_f64().abs())
                        };
                        assert!((log.into_f64() - l.into_f64()).abs() <= bound, "{line}");
                        assert!(
                            (phase.re - p.re).abs().max((phase.im - p.im).abs())
                                <= 64.0 * epsilon * condition,
                            "{line}"
                        );
                    }
                    _ => panic!("certified tier changed zero identity: {line}"),
                }
            } else {
                fallback += 1;
            }
        }
        assert!(
            accepted >= 10 && fallback > 0,
            "accepted={accepted}, fallback={fallback}"
        );
    }

    #[test]
    fn weighted_fixture_tiers_both_widths() {
        weighted_tiers::<f32>();
        weighted_tiers::<f64>();
    }
}
