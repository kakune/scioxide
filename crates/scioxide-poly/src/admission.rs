//! Range checks that let polynomial kernels run in native floating point.
//!
//! Scaled arithmetic itself is in `scioxide-core`. This module decides when native
//! arithmetic is safe for any basis: arithmetic tiers, checkpoint tiers and
//! their step counts, range checks for points and coefficients, and the bounds
//! for Newton divided differences. Recurrence classification for orthogonal
//! families is in the orthogonal module.
use alloc::vec::Vec;
use core::ops::Deref;

use scioxide_core::{
    AlgebraicScalar, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError,
};

#[derive(Debug)]
pub(crate) enum ArithmeticError {
    Retry,
    Numerical(SciOxideError),
}
impl From<SciOxideError> for ArithmeticError {
    fn from(error: SciOxideError) -> Self {
        Self::Numerical(error)
    }
}
pub(crate) type ArithmeticResult<T> = Result<T, ArithmeticError>;

/// A retry discards staging storage; a numerical error never becomes a retry.
#[inline]
pub(crate) fn candidate<T>(result: ArithmeticResult<T>) -> Result<Option<T>, SciOxideError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(ArithmeticError::Retry) => Ok(None),
        Err(ArithmeticError::Numerical(error)) => Err(error),
    }
}
#[inline]
pub(crate) fn extended<T>(result: ArithmeticResult<T>) -> Result<T, SciOxideError> {
    match result {
        Ok(value) => Ok(value),
        Err(ArithmeticError::Numerical(error)) => Err(error),
        Err(ArithmeticError::Retry) => Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "extended arithmetic rejected its range",
        )),
    }
}
#[inline]
pub(crate) fn require(admitted: bool) -> ArithmeticResult<()> {
    if admitted {
        Ok(())
    } else {
        Err(ArithmeticError::Retry)
    }
}

/// Generic products require sqrt-range operands. Real-factor kernels can use
/// the normal range because they check both components of each product and
/// quotient, including vanished nonzero components, before the next operation.
pub(crate) enum ProductOperands {}
pub(crate) enum RealFactors {}
pub(crate) enum WideHorner {}
pub(crate) trait NativePolicy: sealed::Sealed {
    const REAL_FACTORS: bool;
    const WIDE_HORNER: bool = false;
}
impl sealed::Sealed for ProductOperands {}
impl sealed::Sealed for RealFactors {}
impl sealed::Sealed for WideHorner {}
impl NativePolicy for WideHorner {
    const REAL_FACTORS: bool = false;
    const WIDE_HORNER: bool = true;
}
impl NativePolicy for ProductOperands {
    const REAL_FACTORS: bool = false;
}
impl NativePolicy for RealFactors {
    const REAL_FACTORS: bool = true;
}
pub(crate) struct Native<P = ProductOperands>(core::marker::PhantomData<P>);
pub(crate) struct Extended;
pub(crate) struct Lanes<const COUNT: usize>;

/// Operations are unchecked inside a proved block. Checked coefficient kernels
/// use the `checked_*` methods; every failed range check has the same signal.
pub(crate) trait Arithmetic<C: NumericScalar> {
    const NATIVE: bool;
    /// Checkpoint tier of blocked monomial Horner; paired updates consume two
    /// coefficients per step.
    const HORNER_TIER: CheckpointTier = CheckpointTier::Narrow(Narrow(()));
    type Frame: Copy + PartialEq;
    fn frame() -> Self::Frame;
    fn frame_value(value: Self::Value, frame: Self::Frame) -> ArithmeticResult<Self::Value>;
    #[inline]
    fn forcing(value: C, frame: Self::Frame) -> ArithmeticResult<Self::Value> {
        Self::frame_value(Self::scalar(value)?, frame)
    }
    fn reframe<const FORCING: bool>(
        values: &mut [Self::Value],
        bound: &mut Self::Bound,
        frame: &mut Self::Frame,
        narrow: bool,
    ) -> ArithmeticResult<()>;
    fn restore(value: Self::Value, frame: Self::Frame) -> ArithmeticResult<Self::Value>;
    fn restore_bound(value: Self::Bound, frame: Self::Frame) -> ArithmeticResult<Self::Bound>;
    type Value: Copy + PartialEq;
    type Bound: Copy;
    type Projected;
    fn scalar(value: C) -> ArithmeticResult<Self::Value>;
    fn add(a: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value>;
    fn sub(a: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value>;
    fn mul(a: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value>;
    fn div(a: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value>;
    fn mul_add(a: Self::Value, x: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value>;
    fn magnitude(a: Self::Value) -> ArithmeticResult<Self::Bound>;
    fn bound_mul_add(
        a: Self::Bound,
        x: Self::Bound,
        b: Self::Bound,
    ) -> ArithmeticResult<Self::Bound>;
    fn bound_add(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound>;
    fn bound_mul(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound>;
    fn bound_max(a: Self::Bound, b: Self::Bound) -> Self::Bound;
    fn bound_scalar(value: C::Real) -> ArithmeticResult<Self::Bound>;
    fn bound_sub(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound>;
    fn bound_div(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound>;
    fn mul_bound(a: Self::Value, b: Self::Bound) -> ArithmeticResult<Self::Value>;
    #[inline]
    fn checked_bound(a: Self::Bound) -> ArithmeticResult<Self::Bound> {
        require(Self::safe_bound(a))?;
        Ok(a)
    }
    #[inline]
    fn checked_bound_scalar(value: C::Real) -> ArithmeticResult<Self::Bound> {
        Self::checked_bound(Self::bound_scalar(value)?)
    }
    #[inline]
    fn checked_bound_mul(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound> {
        Self::checked_bound(Self::bound_mul(
            Self::checked_bound(a)?,
            Self::checked_bound(b)?,
        )?)
    }
    #[inline]
    fn checked_bound_div(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound> {
        Self::checked_bound(Self::bound_div(
            Self::checked_bound(a)?,
            Self::checked_bound(b)?,
        )?)
    }
    fn safe(a: Self::Value) -> bool;
    fn safe_bound(a: Self::Bound) -> bool;
    fn checkpoint(a: Self::Value, narrow: bool) -> bool;
    fn bound_checkpoint(a: Self::Bound, narrow: bool) -> bool;
    fn project(a: Self::Value) -> ArithmeticResult<Self::Projected>;
    fn finish(a: Self::Value, factor: Scaled<C::Real>) -> ArithmeticResult<Self::Projected>;
    fn real(value: C::Real) -> ArithmeticResult<Self::Value> {
        Self::scalar(C::from_real(value))
    }
    fn double(value: Self::Value) -> ArithmeticResult<Self::Value> {
        Self::add(value, value)
    }
    #[inline]
    fn checked(a: Self::Value) -> ArithmeticResult<Self::Value> {
        require(Self::safe(a))?;
        Ok(a)
    }
    #[inline]
    fn checked_scalar(value: C) -> ArithmeticResult<Self::Value> {
        Self::checked(Self::scalar(value)?)
    }
    #[inline]
    fn checked_real(value: C::Real) -> ArithmeticResult<Self::Value> {
        Self::checked(Self::real(value)?)
    }
}
/// Every nonzero component used as a product operand is in sqrt-range. Products
/// are consequently normal and finite; /4 leaves headroom for complex component
/// sums and derivative doubling. Cancellation between normal nearby products is
/// exact (Sterbenz); check the completed sum before it becomes another operand.
#[inline]
pub(crate) fn native_range<R: RealScalar>(value: R) -> bool {
    // Exact widened IEEE encodings of sqrt(MIN_POSITIVE) and sqrt(MAX)/4.
    // sqrt(MAX) rounds to the predecessor of 2^512 (f64) or 2^64 (f32).
    // Keeping the constants here avoids invoking scalar math dispatch merely
    // to classify each live operand. This is the same interval, bit for bit.
    let (lower, upper) = if R::MANTISSA_DIGITS == 53 {
        ((1023_u64 - 511) << 52, ((1023_u64 + 510) << 52) - 1)
    } else {
        ((1023_u64 - 63) << 52, ((1023_u64 + 62) << 52) - (1 << 29))
    };
    component_range(value, lower, upper)
}

#[inline]
fn policy_range<R: RealScalar, P: NativePolicy>(value: R) -> bool {
    if P::REAL_FACTORS {
        component_range(
            value,
            R::MIN_POSITIVE.into_f64().to_bits(),
            (R::MAX / (R::TWO + R::TWO)).into_f64().to_bits(),
        )
    } else {
        native_range(value)
    }
}
/// Widening f32 is exact; the integer interval also rejects NaN and infinity.
#[inline]
fn component_range<R: RealScalar>(value: R, lower: u64, upper: u64) -> bool {
    let bits = value.into_f64().to_bits() & 0x7fff_ffff_ffff_ffff;
    (bits == 0) | (bits.wrapping_sub(lower) <= upper - lower)
}

#[inline]
/// Native norm, used only after the sqrt-range check has made the squares safe.
pub(crate) fn native_magnitude<C: NumericScalar>(value: C) -> C::Real {
    let re = value.real_component();
    if C::IS_REAL {
        re.abs()
    } else {
        let im = value.imaginary_component();
        (re * re + im * im).sqrt()
    }
}

#[inline]
pub(crate) fn native_value<C: NumericScalar>(value: C) -> bool {
    native_range(value.real_component()) & native_range(value.imaginary_component())
}
/// Classify arbitrary Horner iterators without collecting or allocating.
pub(crate) fn native_coefficients<C: NumericScalar>(values: impl Iterator<Item = C>) -> bool {
    values.fold(true, |admitted, value| admitted & native_value(value))
}
// A narrower checkpoint permits six moderate real steps: its minimum dyadic
// quantum is 2^-180, giving 2^-996 after six factors of quantum 2^-136.
// Six general steps or twelve Chebyshev steps grow magnitudes by less
// than 2^220. An asymmetric upper checkpoint of 2^768 therefore leaves
// headroom below 2^1008, including coefficient-weighted envelopes. Failure retries the wider native admission before
// scaled arithmetic; neither native candidate can silently underflow/overflow.
#[inline]
fn narrow_range<R: RealScalar>(value: R) -> bool {
    component_range(value, (1023_u64 - 128) << 52, (1023_u64 + 768) << 52)
}

/// `f64` real blocks accept points and coefficients in [2^-16,2^16] or zero.
/// A factor a*x+b has magnitude at most `2^33` and quantum at least `2^-136`.
/// Starting from sqrt-range states (quantum at least `2^-563`), three steps keep
/// quantum at least `2^-971` and magnitudes below `2^640`, including derivatives and
/// envelopes. Thus every intermediate remains finite and normal or exact zero.
/// Both live states and the envelope are checked before starting another block.
#[inline]
pub(crate) fn moderate<R: RealScalar>(value: R) -> bool {
    component_range(value, (1023_u64 - 16) << 52, (1023_u64 + 16) << 52)
}
#[inline]
pub(crate) fn block_point<C: NumericScalar>(point: C) -> bool {
    C::IS_REAL && C::Real::MANTISSA_DIGITS == 53 && moderate(point.real_component())
}

#[inline]
pub(crate) fn paired_point<C: NumericScalar>(point: C) -> bool {
    // binary64 sqrt-range inputs have a dyadic quantum >= 2^-563.
    // Two steps with point components in [2^-16,2^16] can lower it only
    // to 2^(-563-2*68) = 2^-699, well above MIN_POSITIVE. Short sums,
    // complex products and derivative doubling stay below 2^550. Thus
    // every intermediate between passed checkpoints is finite and normal (or exact
    // zero), even if a state temporarily leaves the admission interval.
    // Wider points and binary32 retain a checkpoint after every step.
    C::Real::MANTISSA_DIGITS == 53
        && moderate(point.real_component()) & moderate(point.imaginary_component())
}

/// Full normal output range is valid after the last proved block. A subnormal
/// projection retries core arithmetic so gradual underflow happens only once.
#[inline]
fn normal_output<R: RealScalar>(value: R) -> bool {
    component_range(
        value,
        R::MIN_POSITIVE.into_f64().to_bits(),
        R::MAX.into_f64().to_bits(),
    )
}

/// Distinct real nodes bounded by `MAX/2` have finite, nonzero differences, so
/// [`difference_quotient`] need not test its denominator. Complex nodes never
/// qualify: their quotients use extended component arithmetic.
#[inline]
pub(crate) fn finite_gaps<C: NumericScalar>(nodes: &[C]) -> bool {
    C::IS_REAL
        && nodes
            .iter()
            .all(|node| node.real_component().abs() <= C::Real::MAX / C::Real::TWO)
}

/// One native Newton divided difference `(a - b) / (x - y)`, or `None` when it
/// needs the extended path. Only real quotients are native.
///
/// The native path requires a finite quotient that is normal or has an exactly zero
/// numerator: a subnormal quotient is representable but can lose a coefficient
/// needed by later orders. With `FINITE_GAPS = false` the operand differences
/// are also checked, so success proves finite operands and a nonzero
/// denominator; with `FINITE_GAPS = true` the caller has proved
/// [`finite_gaps`] for the table's nodes, and a finite quotient then also
/// proves a finite numerator.
#[inline]
pub(crate) fn difference_quotient<C: NumericScalar, const FINITE_GAPS: bool>(
    a: C,
    b: C,
    x: C,
    y: C,
) -> Option<C> {
    if !C::IS_REAL {
        return None;
    }
    let numerator = a.real_component() - b.real_component();
    let denominator = x.real_component() - y.real_component();
    let value = numerator / denominator;
    ((FINITE_GAPS | (numerator.is_finite() & denominator.is_finite()))
        & value.is_finite()
        & ((value.abs() >= C::Real::MIN_POSITIVE) | (numerator == C::Real::ZERO)))
        .then_some(C::from_real(value))
}

/// Whole-table certificate for an unchecked in-place Newton table over finite
/// `values` (any number of interleaved components) at distinct real `nodes`.
///
/// Strictly monotone nodes with [`finite_gaps`] make every denominator at least
/// the smallest adjacent gap `g`, and each order at most doubles the preceding
/// magnitude bound `B` before dividing, so `B_k <= 2 B_(k-1) / g`. The chain
/// starts from the sample bound `B_0` and is carried with a factor `4 / g`,
/// whose slack absorbs the rounding of the bound itself; requiring every
/// `B_k <= MAX/4`, `B_0` included, keeps each numerator `<= MAX/2`.
/// Quotients are therefore finite, and may round to subnormals exactly as the
/// checked per-entry path would project them. The certificate is `O(n)`,
/// independent of the `O(n²)` table work.
pub(crate) fn difference_table<R: RealScalar>(nodes: &[R], values: &[R]) -> bool {
    let increasing = nodes.windows(2).all(|pair| pair[0] < pair[1]);
    if !(increasing || nodes.windows(2).all(|pair| pair[0] > pair[1])) || !finite_gaps(nodes) {
        return false;
    }
    let limit = R::MAX / (R::TWO + R::TWO);
    let gap = nodes
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(R::MAX, |smallest, gap| smallest.min(gap));
    let growth = (R::TWO + R::TWO) / gap;
    let bound = values
        .iter()
        .fold(R::ZERO, |bound, value| bound.max(value.abs()));
    // The first-order numerators difference raw samples, so B_0 obeys the same
    // limit; a wide gap alone cannot certify samples near the overflow threshold.
    bound <= limit
        && (1..nodes.len())
            .try_fold(bound, |bound, _| {
                let next = bound * growth;
                (next <= limit).then_some(next)
            })
            .is_some()
}

/// Exact common-frame change at a proved checkpoint. Narrow states span
/// [2^-128,2^768]; three-step wide states span [2^-511,2^896]. The minimum
/// quantum is respectively 2^-180 or 2^-563. Six or three factors of quantum
/// `>=2^-136` leave `>=2^-996` or `>=2^-971`; coefficient-weighted envelope products
/// remain normal. At most 35 growth bits per general step (18 for classical
/// Chebyshev) leave headroom for derivatives, envelope terms and their sums.
///
/// Series forcing lies in [2^-64,2^16] or is zero before framing. Restrict the cumulative exponent so
/// scaled forcing also lies in the checkpoint window. An envelope term uses
/// the basis before the last forward update of a six-step block: its magnitude
/// is at least 2^(-180-5*136-64) = 2^-924 or zero, hence still normal.
/// Twelve integer-factor steps similarly keep envelope terms >=2^-992.
/// The envelope and all live states must fit simultaneously; otherwise retry, never discard a component.
#[inline]
fn reframe<C: NumericScalar, const FORCING: bool>(
    values: &mut [C],
    bound: &mut C::Real,
    exponent: &mut i32,
    narrow: bool,
) -> ArithmeticResult<()> {
    require(C::IS_REAL)?;
    let (lower, upper) = if narrow { (-128, 768) } else { (-511, 896) };
    let low_bits = ((1023 + lower) as u64) << 52;
    let high_bits = ((1023 + upper) as u64) << 52;
    let admitted = |v: C::Real| component_range(v, low_bits, high_bits);
    if values.iter().all(|v| admitted(v.real_component())) && admitted(*bound) {
        return Ok(());
    }
    require(C::IS_REAL && C::Real::MANTISSA_DIGITS == 53)?;
    let mut minimum = i32::MAX;
    let mut maximum = i32::MIN;
    for v in values
        .iter()
        .map(|v| v.real_component())
        .chain(core::iter::once(*bound))
    {
        require(normal_output(v))?;
        if v != C::Real::ZERO {
            let bits = v.into_f64().to_bits() & 0x7fff_ffff_ffff_ffff;
            let floor = ((bits >> 52) as i32) - 1023;
            let ceil = floor + i32::from(bits & ((1_u64 << 52) - 1) != 0);
            minimum = minimum.min(floor);
            maximum = maximum.max(ceil);
        }
    }
    let mut shift_low = maximum - upper;
    let mut shift_high = minimum - lower;
    if FORCING {
        shift_low = shift_low.max(16 - upper - *exponent);
        shift_high = shift_high.min(-64 - lower - *exponent);
    }
    require(shift_low <= shift_high)?;
    let shift = 0.clamp(shift_low, shift_high);
    let next_exponent = exponent.checked_add(shift).ok_or(ArithmeticError::Retry)?;
    for value in values {
        *value = C::from_real(value.real_component().scalbn(-shift));
    }
    *bound = bound.scalbn(-shift);
    *exponent = next_exponent;
    Ok(())
}

impl<C: NumericScalar, P: NativePolicy> Arithmetic<C> for Native<P> {
    const NATIVE: bool = true;
    // Six wide coefficients are three paired updates: from quantum 2^-563,
    // rounded x^2 (quantum >=2^-84) leaves >=2^-815. Magnitudes remain
    // below 2^998 from the wide checkpoint's 2^896 upper bound.
    const HORNER_TIER: CheckpointTier = if P::WIDE_HORNER {
        CheckpointTier::Wide(Wide(()))
    } else {
        CheckpointTier::Narrow(Narrow(()))
    };
    type Frame = i32;
    #[inline]
    fn frame() -> i32 {
        0
    }
    #[inline]
    fn frame_value(value: C, frame: i32) -> ArithmeticResult<C> {
        if frame == 0 {
            return Ok(value);
        }
        Ok(C::from_real(value.real_component().scalbn(
            frame.checked_neg().ok_or(ArithmeticError::Retry)?,
        )))
    }
    #[inline(always)]
    fn reframe<const FORCING: bool>(
        values: &mut [C],
        bound: &mut C::Real,
        frame: &mut i32,
        narrow: bool,
    ) -> ArithmeticResult<()> {
        if values
            .iter()
            .all(|&v| <Self as Arithmetic<C>>::checkpoint(v, narrow))
            && <Self as Arithmetic<C>>::bound_checkpoint(*bound, narrow)
        {
            return Ok(());
        }
        reframe::<C, FORCING>(values, bound, frame, narrow)
    }
    #[inline(always)]
    fn restore(value: C, frame: i32) -> ArithmeticResult<C> {
        let value = if frame == 0 {
            value
        } else {
            C::from_real(value.real_component().scalbn(frame))
        };
        require(
            normal_output(value.real_component()) & normal_output(value.imaginary_component()),
        )?;
        Ok(value)
    }
    #[inline(always)]
    fn restore_bound(value: C::Real, frame: i32) -> ArithmeticResult<C::Real> {
        let value = if frame == 0 {
            value
        } else {
            value.scalbn(frame)
        };
        require(normal_output(value))?;
        Ok(value)
    }
    type Value = C;
    type Bound = C::Real;
    type Projected = C;
    #[inline]
    fn scalar(value: C) -> ArithmeticResult<C> {
        Ok(value)
    }
    #[inline]
    fn add(a: C, b: C) -> ArithmeticResult<C> {
        Ok(a + b)
    }
    #[inline]
    fn sub(a: C, b: C) -> ArithmeticResult<C> {
        Ok(a - b)
    }
    #[inline]
    fn mul(a: C, b: C) -> ArithmeticResult<C> {
        if P::REAL_FACTORS {
            require(b.imaginary_component() == C::Real::ZERO)?;
            let value = C::from_components(
                a.real_component() * b.real_component(),
                a.imaginary_component() * b.real_component(),
            );
            require(
                b.real_component() == C::Real::ZERO
                    || !((a.real_component() != C::Real::ZERO
                        && value.real_component() == C::Real::ZERO)
                        || (a.imaginary_component() != C::Real::ZERO
                            && value.imaginary_component() == C::Real::ZERO)),
            )?;
            Ok(value)
        } else {
            Ok(a * b)
        }
    }
    #[inline]
    fn div(a: C, b: C) -> ArithmeticResult<C> {
        require(!P::REAL_FACTORS || b.imaginary_component() == C::Real::ZERO)?;
        let value = if C::IS_REAL {
            a / b
        } else if b.imaginary_component() == C::Real::ZERO {
            C::from_components(
                a.real_component() / b.real_component(),
                a.imaginary_component() / b.real_component(),
            )
        } else {
            a / b
        };
        // Coefficient kernels only divide by real factors. Do not accept a
        // vanished nonzero component, including an exact zero quotient.
        if P::REAL_FACTORS {
            require(
                !((a.real_component() != C::Real::ZERO && value.real_component() == C::Real::ZERO)
                    || (a.imaginary_component() != C::Real::ZERO
                        && value.imaginary_component() == C::Real::ZERO)),
            )?;
        }
        Ok(value)
    }
    #[inline]
    fn mul_add(a: C, x: C, b: C) -> ArithmeticResult<C> {
        Ok(if C::IS_REAL {
            C::from_real(scioxide_core::fast_mul_add(
                a.real_component(),
                x.real_component(),
                b.real_component(),
            ))
        } else {
            a * x + b
        })
    }
    #[inline]
    fn magnitude(value: C) -> ArithmeticResult<C::Real> {
        Ok(if P::REAL_FACTORS {
            value.abs()
        } else {
            native_magnitude(value)
        })
    }
    #[inline]
    fn bound_add(a: C::Real, b: C::Real) -> ArithmeticResult<C::Real> {
        Ok(a + b)
    }
    #[inline]
    fn bound_mul(a: C::Real, b: C::Real) -> ArithmeticResult<C::Real> {
        Ok(a * b)
    }
    #[inline]
    fn bound_mul_add(a: C::Real, x: C::Real, b: C::Real) -> ArithmeticResult<C::Real> {
        Ok(scioxide_core::fast_mul_add(a, x, b))
    }
    #[inline]
    fn bound_max(a: C::Real, b: C::Real) -> C::Real {
        if a > b { a } else { b }
    }
    #[inline]
    fn bound_scalar(value: C::Real) -> ArithmeticResult<C::Real> {
        Ok(value)
    }
    #[inline]
    fn bound_sub(a: C::Real, b: C::Real) -> ArithmeticResult<C::Real> {
        Ok(a - b)
    }
    #[inline]
    fn bound_div(a: C::Real, b: C::Real) -> ArithmeticResult<C::Real> {
        let value = a / b;
        if P::REAL_FACTORS {
            require(a == C::Real::ZERO || value != C::Real::ZERO)?;
        }
        Ok(value)
    }
    #[inline]
    fn mul_bound(a: C, b: C::Real) -> ArithmeticResult<C> {
        Ok(C::from_components(
            a.real_component() * b,
            a.imaginary_component() * b,
        ))
    }
    #[inline]
    fn safe(a: C) -> bool {
        policy_range::<C::Real, P>(a.real_component())
            & policy_range::<C::Real, P>(a.imaginary_component())
    }
    #[inline]
    fn safe_bound(a: C::Real) -> bool {
        policy_range::<C::Real, P>(a)
    }
    #[inline]
    fn checkpoint(a: C, narrow: bool) -> bool {
        if narrow {
            narrow_range(a.real_component()) & narrow_range(a.imaginary_component())
        } else {
            native_value(a)
        }
    }
    #[inline]
    fn bound_checkpoint(a: C::Real, narrow: bool) -> bool {
        if narrow {
            narrow_range(a)
        } else {
            native_range(a)
        }
    }
    #[inline]
    fn project(a: C) -> ArithmeticResult<C> {
        Ok(a)
    }
    fn finish(a: C, factor: Scaled<C::Real>) -> ArithmeticResult<C> {
        Ok(ScaledValue::from_value(a)?.mul_real(factor)?.round()?)
    }
}
impl<C: NumericScalar> Arithmetic<C> for Extended {
    const NATIVE: bool = false;
    type Frame = ();
    fn frame() {}
    fn frame_value(value: Self::Value, _: ()) -> ArithmeticResult<Self::Value> {
        Ok(value)
    }
    fn reframe<const FORCING: bool>(
        _: &mut [Self::Value],
        _: &mut Self::Bound,
        _: &mut (),
        _: bool,
    ) -> ArithmeticResult<()> {
        Ok(())
    }
    fn restore(value: Self::Value, _: ()) -> ArithmeticResult<Self::Value> {
        Ok(value)
    }
    fn restore_bound(value: Self::Bound, _: ()) -> ArithmeticResult<Self::Bound> {
        Ok(value)
    }
    type Value = ScaledValue<C>;
    type Bound = Scaled<C::Real>;
    type Projected = C;
    fn scalar(value: C) -> ArithmeticResult<Self::Value> {
        Ok(ScaledValue::from_value(value)?)
    }
    fn add(a: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value> {
        Ok(a.add(b)?)
    }
    fn sub(a: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value> {
        Ok(a.sub(b)?)
    }
    fn mul(a: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value> {
        Ok(a.mul(b)?)
    }
    fn div(a: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value> {
        Ok(a.div(b)?)
    }
    fn mul_add(a: Self::Value, x: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value> {
        Ok(a.mul(x)?.add(b)?)
    }
    fn magnitude(a: Self::Value) -> ArithmeticResult<Self::Bound> {
        Ok(a.magnitude()?)
    }
    fn bound_add(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound> {
        Ok(a.add(b)?)
    }
    fn bound_mul(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound> {
        Ok(a.mul(b)?)
    }
    fn bound_mul_add(
        a: Self::Bound,
        x: Self::Bound,
        b: Self::Bound,
    ) -> ArithmeticResult<Self::Bound> {
        Ok(a.mul(x)?.add(b)?)
    }
    fn bound_max(a: Self::Bound, b: Self::Bound) -> Self::Bound {
        if a.abs_cmp(b).is_gt() { a } else { b }
    }
    fn bound_scalar(value: C::Real) -> ArithmeticResult<Self::Bound> {
        Ok(Scaled::from_value(value)?)
    }
    fn bound_sub(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound> {
        Ok(a.sub(b)?)
    }
    fn bound_div(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound> {
        Ok(a.div(b)?)
    }
    fn mul_bound(a: Self::Value, b: Self::Bound) -> ArithmeticResult<Self::Value> {
        Ok(a.mul_real(b)?)
    }
    fn safe(_: Self::Value) -> bool {
        true
    }
    fn safe_bound(_: Self::Bound) -> bool {
        true
    }
    fn checkpoint(_: Self::Value, _: bool) -> bool {
        true
    }
    fn bound_checkpoint(_: Self::Bound, _: bool) -> bool {
        true
    }
    fn project(a: Self::Value) -> ArithmeticResult<C> {
        Ok(a.round()?)
    }
    fn finish(a: Self::Value, factor: Scaled<C::Real>) -> ArithmeticResult<C> {
        Ok(a.mul_real(factor)?.round()?)
    }
}
impl<C: NumericScalar, const COUNT: usize> Arithmetic<C> for Lanes<COUNT> {
    const NATIVE: bool = true;
    type Frame = [i32; COUNT];
    #[inline]
    fn frame() -> Self::Frame {
        [0; COUNT]
    }
    #[inline]
    fn frame_value(mut value: Self::Value, frame: Self::Frame) -> ArithmeticResult<Self::Value> {
        if frame == [0; COUNT] {
            return Ok(value);
        }
        for (out, exponent) in value.iter_mut().zip(frame) {
            *out = <Native as Arithmetic<C>>::frame_value(*out, exponent)?;
        }
        Ok(value)
    }
    #[inline(always)]
    fn reframe<const FORCING: bool>(
        values: &mut [Self::Value],
        bound: &mut Self::Bound,
        frame: &mut Self::Frame,
        narrow: bool,
    ) -> ArithmeticResult<()> {
        require(C::IS_REAL && C::Real::MANTISSA_DIGITS == 53)?;
        if values.iter().all(|&v| Self::checkpoint(v, narrow))
            && <Self as Arithmetic<C>>::bound_checkpoint(*bound, narrow)
        {
            return Ok(());
        }
        require(values.len() <= 6)?;
        // Each lane owns its exponent. Never let one point discard another's
        // small components by sharing a floating-point frame.
        for i in 0..COUNT {
            let mut lane = [C::ZERO; 6];
            for (out, value) in lane.iter_mut().zip(values.iter()) {
                *out = value[i];
            }
            reframe::<C, FORCING>(
                &mut lane[..values.len()],
                &mut bound[i],
                &mut frame[i],
                narrow,
            )?;
            for (value, component) in values.iter_mut().zip(lane) {
                value[i] = component;
            }
        }
        Ok(())
    }
    #[inline(always)]
    fn restore(mut value: Self::Value, frame: Self::Frame) -> ArithmeticResult<Self::Value> {
        for (out, exponent) in value.iter_mut().zip(frame) {
            *out = <Native as Arithmetic<C>>::restore(*out, exponent)?;
        }
        Ok(value)
    }
    #[inline(always)]
    fn restore_bound(mut value: Self::Bound, frame: Self::Frame) -> ArithmeticResult<Self::Bound> {
        for (out, exponent) in value.iter_mut().zip(frame) {
            *out = <Native as Arithmetic<C>>::restore_bound(*out, exponent)?;
        }
        Ok(value)
    }
    type Value = [C; COUNT];
    type Projected = [C; COUNT];
    type Bound = [C::Real; COUNT];
    #[inline]
    fn scalar(value: C) -> Result<Self::Value, ArithmeticError> {
        Ok([value; COUNT])
    }
    #[inline]
    fn add(a: Self::Value, b: Self::Value) -> Result<Self::Value, ArithmeticError> {
        Ok(core::array::from_fn(|i| a[i] + b[i]))
    }
    #[inline]
    fn sub(a: Self::Value, b: Self::Value) -> Result<Self::Value, ArithmeticError> {
        Ok(core::array::from_fn(|i| a[i] - b[i]))
    }
    #[inline]
    fn mul(a: Self::Value, b: Self::Value) -> Result<Self::Value, ArithmeticError> {
        Ok(core::array::from_fn(|i| a[i] * b[i]))
    }
    #[inline]
    fn mul_add(
        a: Self::Value,
        x: Self::Value,
        b: Self::Value,
    ) -> Result<Self::Value, ArithmeticError> {
        Ok(core::array::from_fn(|i| {
            if C::IS_REAL {
                C::from_real(scioxide_core::fast_mul_add(
                    a[i].real_component(),
                    x[i].real_component(),
                    b[i].real_component(),
                ))
            } else {
                a[i] * x[i] + b[i]
            }
        }))
    }
    #[inline]
    fn magnitude(a: Self::Value) -> Result<Self::Bound, ArithmeticError> {
        Ok(core::array::from_fn(|i| native_magnitude(a[i])))
    }
    #[inline]
    fn bound_add(a: Self::Bound, b: Self::Bound) -> Result<Self::Bound, ArithmeticError> {
        Ok(core::array::from_fn(|i| a[i] + b[i]))
    }
    #[inline]
    fn bound_mul(a: Self::Bound, b: Self::Bound) -> Result<Self::Bound, ArithmeticError> {
        Ok(core::array::from_fn(|i| a[i] * b[i]))
    }
    #[inline]
    fn bound_max(a: Self::Bound, b: Self::Bound) -> Self::Bound {
        core::array::from_fn(|i| if a[i] > b[i] { a[i] } else { b[i] })
    }
    fn bound_scalar(value: C::Real) -> ArithmeticResult<Self::Bound> {
        Ok([value; COUNT])
    }
    fn bound_sub(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound> {
        Ok(core::array::from_fn(|i| a[i] - b[i]))
    }
    fn bound_div(a: Self::Bound, b: Self::Bound) -> ArithmeticResult<Self::Bound> {
        Ok(core::array::from_fn(|i| a[i] / b[i]))
    }
    fn mul_bound(a: Self::Value, b: Self::Bound) -> ArithmeticResult<Self::Value> {
        Ok(core::array::from_fn(|i| {
            C::from_components(
                a[i].real_component() * b[i],
                a[i].imaginary_component() * b[i],
            )
        }))
    }
    #[inline]
    fn safe(a: Self::Value) -> bool {
        a.into_iter().fold(true, |safe, value| {
            safe & <Native as Arithmetic<C>>::safe(value)
        })
    }
    #[inline]
    fn safe_bound(a: Self::Bound) -> bool {
        a.into_iter()
            .fold(true, |safe, value| safe & native_range(value))
    }
    #[inline]
    fn checkpoint(a: Self::Value, narrow: bool) -> bool {
        a.into_iter().fold(true, |safe, value| {
            safe & <Native as Arithmetic<C>>::checkpoint(value, narrow)
        })
    }
    #[inline]
    fn bound_checkpoint(a: Self::Bound, narrow: bool) -> bool {
        a.into_iter().fold(true, |safe, value| {
            safe & <Native as Arithmetic<C>>::bound_checkpoint(value, narrow)
        })
    }
    #[inline]
    fn div(a: Self::Value, b: Self::Value) -> ArithmeticResult<Self::Value> {
        Ok(core::array::from_fn(|i| a[i] / b[i]))
    }
    #[inline]
    fn bound_mul_add(
        a: Self::Bound,
        x: Self::Bound,
        b: Self::Bound,
    ) -> ArithmeticResult<Self::Bound> {
        Ok(core::array::from_fn(|i| {
            scioxide_core::fast_mul_add(a[i], x[i], b[i])
        }))
    }
    fn project(a: Self::Value) -> ArithmeticResult<Self::Projected> {
        Ok(a)
    }
    fn finish(a: Self::Value, factor: Scaled<C::Real>) -> ArithmeticResult<Self::Projected> {
        let mut output = [C::ZERO; COUNT];
        for (out, value) in output.iter_mut().zip(a) {
            *out = <Native as Arithmetic<C>>::finish(value, factor)?;
        }
        Ok(output)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Coefficients<C: NumericScalar> {
    values: Vec<C>,
    bounded: bool,
    native: bool,
}
impl<C: NumericScalar> Coefficients<C> {
    pub(crate) fn new(values: Vec<C>) -> Self {
        let view = CoefficientView::new(&values);
        let (bounded, native) = (view.bounded, view.native);
        Self {
            values,
            bounded,
            native,
        }
    }
    pub(crate) fn view(&self) -> CoefficientView<'_, C> {
        CoefficientView {
            values: &self.values,
            bounded: self.bounded,
            native: self.native,
        }
    }
    pub(crate) fn into_vec(self) -> Vec<C> {
        self.values
    }
}
impl<C: NumericScalar> Deref for Coefficients<C> {
    type Target = [C];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}
#[derive(Clone, Copy)]
pub(crate) struct CoefficientView<'a, C: NumericScalar> {
    values: &'a [C],
    bounded: bool,
    native: bool,
}
impl<'a, C: NumericScalar> CoefficientView<'a, C> {
    pub(crate) fn values(self) -> &'a [C] {
        self.values
    }
    /// Equal even/odd coefficient lanes reuse the scalar Horner certificate
    /// at x^2. Moderate x^2 bounds |x| to [2^-8,2^8] or zero. Combining the
    /// value, derivative and envelope after narrow checkpoints stays normal
    /// or exact zero, with magnitude below 2^788.
    #[inline(always)]
    pub(crate) fn horner_parity(self, point: C) -> bool {
        self.values.len() > 2 && self.values.len().is_multiple_of(2) && self.bounded && {
            let squared = point * point;
            // A nonzero moderate square already bounds the original point.
            // Reject a nonzero point whose square underflowed to zero.
            block_point(squared) && (squared != C::ZERO || point == C::ZERO)
        }
    }
    /// The grouped Horner update processes twelve coefficients between
    /// narrow checkpoints, including derivatives and envelope. Rounded x^2
    /// has quantum >=2^-84 and magnitude <=2^32; six paired updates leave
    /// quantum >=2^-688 and magnitude <2^972 from the checkpoint interval.
    /// The peeled first pair and forcing have quantum >=2^-184 and remain normal.
    #[inline(always)]
    pub(crate) fn horner_tier(self, point: C) -> CheckpointTier {
        if self.bounded && block_point(point) {
            CheckpointTier::Narrow(Narrow(()))
        } else {
            CheckpointTier::CHECKED
        }
    }
    pub(crate) fn native(self) -> bool {
        self.native
    }
    /// Binary64 real coefficients in `[2^-64, 2^16]` or zero.
    pub(crate) fn bounded(self) -> bool {
        self.bounded
    }

    pub(crate) fn new(values: &'a [C]) -> Self {
        let bounded = C::IS_REAL
            && C::Real::MANTISSA_DIGITS == 53
            && values.iter().all(|c| {
                component_range(
                    c.real_component(),
                    (1023_u64 - 64) << 52,
                    (1023_u64 + 16) << 52,
                )
            });
        // Binary64 bounded coefficients remain inside sqrt-range.
        let native = bounded || native_coefficients(values.iter().copied());
        Self {
            values,
            bounded,
            native,
        }
    }
}
/// Sealed checkpoint policies keep compile-time loop specialization.
mod sealed {
    pub(crate) trait Sealed {}
}
pub(crate) trait Checkpoint: sealed::Sealed {
    const STEPS: usize;
    const NARROW: bool;
}
macro_rules! checkpoint {
    ($name:ident,$steps:expr,$narrow:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) struct $name(());
        impl sealed::Sealed for $name {}
        impl Checkpoint for $name {
            const STEPS: usize = $steps;
            const NARROW: bool = $narrow;
        }
    };
}
checkpoint!(PerStep, 0, false);
checkpoint!(Wide, 3, false);
checkpoint!(Narrow, 6, true);
checkpoint!(Integer, 12, true);
/// Only checked constructors can produce a blocked policy: [`CheckpointTier::blocked`]
/// for recurrences and the coefficient-view Horner tiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CheckpointTier {
    PerStep(PerStep),
    Wide(Wide),
    Narrow(Narrow),
    Integer(Integer),
}
impl CheckpointTier {
    pub(crate) const CHECKED: Self = Self::PerStep(PerStep(()));
    /// Narrow (or, for integer-factor recurrences, twelve-step integer) blocks
    /// for a recurrence whose caller-proved terms are `bounded` when every point
    /// is a moderate binary64 block point; otherwise every step is checked.
    /// The caller proves the recurrence bound; this function checks the points.
    #[inline(always)]
    pub(crate) fn blocked<C: NumericScalar>(
        bounded: bool,
        integer_factors: bool,
        points: impl IntoIterator<Item = C>,
    ) -> Self {
        if !bounded || !points.into_iter().all(block_point) {
            Self::CHECKED
        } else if integer_factors {
            Self::Integer(Integer(()))
        } else {
            Self::Narrow(Narrow(()))
        }
    }
    /// Steps between checkpoints, from the tier's [`Checkpoint`] marker.
    pub(crate) const fn steps(self) -> usize {
        match self {
            Self::PerStep(_) => PerStep::STEPS,
            Self::Wide(_) => Wide::STEPS,
            Self::Narrow(_) => Narrow::STEPS,
            Self::Integer(_) => Integer::STEPS,
        }
    }
    /// Whether checkpoints use the narrow window, from the tier's marker.
    pub(crate) const fn narrow(self) -> bool {
        match self {
            Self::PerStep(_) => PerStep::NARROW,
            Self::Wide(_) => Wide::NARROW,
            Self::Narrow(_) => Narrow::NARROW,
            Self::Integer(_) => Integer::NARROW,
        }
    }
    pub(crate) fn wide(self) -> Self {
        match self {
            Self::PerStep(_) => self,
            _ => Self::Wide(Wide(())),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::{Arithmetic, ArithmeticError, Native, RealFactors, native_range, native_value};
    use scioxide_core::{Complex, RealScalar};

    #[test]
    fn real_only_reframing_rejects_complex_states_before_the_early_return() {
        for imaginary in [0.5_f64, 2.0_f64.powi(800)] {
            let original = [Complex::new(1.0, imaginary)];
            let mut live = original;
            let mut bound = 1.0;
            let mut exponent = 0;
            assert!(matches!(
                super::reframe::<_, false>(&mut live, &mut bound, &mut exponent, true),
                Err(ArithmeticError::Retry)
            ));
            assert_eq!(live, original);
            assert_eq!(bound, 1.0);
            assert_eq!(exponent, 0);
        }
    }

    #[test]
    fn exact_frames_preserve_live_states_forcing_and_zero_signs() {
        type A = Native;
        let original = [2.0_f64.powi(800), -2.0_f64.powi(100), -0.0];
        let mut live = original;
        let mut bound = 2.0_f64.powi(801);
        let mut frame = 0;
        <A as Arithmetic<f64>>::reframe::<false>(&mut live, &mut bound, &mut frame, true).unwrap();
        assert_eq!(frame, 33);
        for (actual, expected) in live.into_iter().zip(original) {
            assert_eq!(
                <A as Arithmetic<f64>>::restore(actual, frame)
                    .unwrap()
                    .to_bits(),
                expected.to_bits()
            );
        }
        assert_eq!(
            <A as Arithmetic<f64>>::restore_bound(bound, frame).unwrap(),
            2.0_f64.powi(801)
        );
        assert_eq!(
            <A as Arithmetic<f64>>::forcing(0.5, frame).unwrap(),
            0.5_f64.scalbn(-frame)
        );
        // Wide checkpoints can retain states whose exponent span does not fit
        // a narrow frame; an unsuccessful frame never discards a small state.
        let original = [2.0_f64.powi(900), 2.0_f64.powi(-200)];
        let mut live = original;
        assert!(
            <A as Arithmetic<f64>>::reframe::<true>(&mut live, &mut 0.0, &mut 0, true).is_err()
        );
        assert_eq!(live, original);
        <A as Arithmetic<f64>>::reframe::<true>(&mut live, &mut 0.0, &mut 0, false).unwrap();
    }

    #[test]
    fn independent_lanes_keep_separate_exact_frames() {
        type A = super::Lanes<2>;
        let original = [
            [2.0_f64.powi(800), 2.0_f64.powi(200)],
            [2.0_f64.powi(100), 2.0_f64.powi(-100)],
        ];
        let mut live = original;
        let mut bound = [2.0_f64.powi(801), 2.0_f64.powi(201)];
        let mut frame = [0; 2];
        <A as Arithmetic<f64>>::reframe::<false>(&mut live, &mut bound, &mut frame, true).unwrap();
        assert_eq!(frame, [33, 0]);
        for (actual, expected) in live.into_iter().zip(original) {
            assert_eq!(
                <A as Arithmetic<f64>>::restore(actual, frame).unwrap(),
                expected
            );
        }
    }

    fn range_edges<R: RealScalar>() {
        let lower = R::MIN_POSITIVE.sqrt();
        let upper = R::MAX.sqrt() / (R::TWO + R::TWO);
        for value in [R::ZERO, -R::ZERO, lower, -lower, upper, -upper] {
            assert!(native_range(value));
        }
        for value in [
            lower.next_after(R::ZERO),
            upper.next_after(R::MAX),
            R::MIN_POSITIVE,
            R::MAX,
            R::INFINITY,
            R::NAN,
        ] {
            assert!(!native_range(value));
            assert!(!native_value(Complex::new(R::ONE, value)));
            assert!(!native_value(Complex::new(value, R::ONE)));
        }
        let numerator =
            <Native as Arithmetic<Complex<R>>>::checked_scalar(Complex::new(lower, lower)).unwrap();
        let divisor =
            <Native as Arithmetic<Complex<R>>>::checked_scalar(Complex::new(upper, R::ZERO))
                .unwrap();
        let result = <Native as Arithmetic<Complex<R>>>::div(numerator, divisor)
            .and_then(<Native as Arithmetic<Complex<R>>>::checked);
        assert!(matches!(result, Err(ArithmeticError::Retry)));
    }

    #[test]
    fn component_admission_and_division_cover_both_widths() {
        range_edges::<f32>();
        range_edges::<f64>();
    }

    #[test]
    fn real_factor_policy_retains_normal_range_and_detects_lost_components() {
        fn check<R: RealScalar>() {
            let large = R::MAX / (R::TWO + R::TWO + R::TWO + R::TWO);
            let value = Complex::new(large, R::MIN_POSITIVE);
            let one = Complex::new(R::ONE, R::ZERO);
            assert!(<Native as Arithmetic<Complex<R>>>::checked_scalar(value).is_err());
            assert_eq!(
                <Native<RealFactors> as Arithmetic<Complex<R>>>::mul(value, one)
                    .and_then(<Native<RealFactors> as Arithmetic<Complex<R>>>::checked)
                    .unwrap(),
                value
            );
            let small = Complex::new(R::MIN_POSITIVE, R::MIN_POSITIVE);
            for factor in [
                Complex::new(R::MIN_POSITIVE, R::ZERO),
                Complex::new(R::HALF, R::ZERO),
                Complex::new(R::ONE, R::ONE),
            ] {
                assert!(matches!(
                    <Native<RealFactors> as Arithmetic<Complex<R>>>::mul(small, factor)
                        .and_then(<Native<RealFactors> as Arithmetic<Complex<R>>>::checked),
                    Err(ArithmeticError::Retry)
                ));
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
