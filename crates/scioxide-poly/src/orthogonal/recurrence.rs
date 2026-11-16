//! Differentiated three-term recurrence, native range checks and scaled restart.

use crate::horner::{HornerEvaluation, HornerState};
use crate::util::validate_slice;
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, ScaledValue, SciOxideError};

use super::admission::{ClassicalRecurrence, SeriesAdmission};
use crate::admission::{
    Arithmetic, ArithmeticError, ArithmeticResult, Checkpoint, CheckpointTier, Extended, Integer,
    Lanes, Native, candidate, extended, require,
};
pub(super) type LaneEvaluations<C, const COUNT: usize> =
    [(C, C, <C as NumericScalar>::Real); COUNT];
type Jet<C, A> = [<A as Arithmetic<C>>::Value; 3];
type State<C, A> = HornerState<<A as Arithmetic<C>>::Value, <A as Arithmetic<C>>::Bound>;
/// Independent points share coefficient loads and native arithmetic.
/// A lane that fails the range check retries through the ordinary pointwise
/// entry points. Twelve-step blocks are allowed only for classical Chebyshev T:
/// its integer recurrence coefficients leave factor quantum at least `2^-68`, so
/// narrow states keep quantum `2^-996` and magnitudes below `2^1008` for twelve
/// steps.
#[inline]
pub(super) fn series_lanes<C: NumericScalar, const COUNT: usize>(
    admission: SeriesAdmission<'_, C>,
    points: [C; COUNT],
    p0: C::Real,
) -> Result<Option<LaneEvaluations<C, COUNT>>, SciOxideError> {
    let tier = admission.tier(points, true);
    let coefficients = admission.coefficients();
    let recurrence = admission.recurrence();
    let result = match tier {
        CheckpointTier::Integer(t) => {
            series_kernel_impl::<C, Lanes<COUNT>, 1, false, true, false, _>(
                coefficients.iter().copied(),
                points,
                p0,
                |n| crate::OrthogonalFamily::ChebyshevT.recurrence(n),
                t,
            )
        }
        CheckpointTier::Narrow(t) => {
            series_kernel_impl::<C, Lanes<COUNT>, 1, false, true, false, _>(
                coefficients.iter().copied(),
                points,
                p0,
                |n| Ok(recurrence[n]),
                t,
            )
        }
        _ => return Ok(None),
    };
    let mut state = candidate(result)?;
    for tier in [tier, tier.wide()] {
        if state.is_some() {
            break;
        }
        state = candidate(series_kernel::<C, Lanes<COUNT>, 1, false, true, true>(
            coefficients.iter().copied(),
            points,
            p0,
            |n| Ok(recurrence[n]),
            tier,
        ))?;
    }
    Ok(state.map(|state| {
        core::array::from_fn(|i| (state.value[i], state.derivative[i], state.bound[i]))
    }))
}

/// Emit each basis member with the same forward update used by series bounds.
/// Output is staging storage, so an unsuccessful native pass can be discarded.
#[inline]
pub(crate) fn members_into<C: NumericScalar>(
    point: C,
    p0: C::Real,
    recurrence: impl Fn(usize) -> (C::Real, C::Real, C::Real),
    output: &mut [C],
) -> Result<(), SciOxideError> {
    if candidate(members_kernel::<C, Native>(point, p0, &recurrence, output))?.is_some() {
        return Ok(());
    }
    extended_members(point, p0, recurrence, output)
}

/// Forward recurrence of an integer-coefficient family (Chebyshev T/U, Hermite
/// H/He), `P_(k+1) = factor P_k - c_k P_(k-1)` from `P_0 = 1` and `P_1 = first`.
/// `GROWING` carries `c_k = c_1 + (k - 1) increment` exactly (Hermite);
/// otherwise `c_k = 1` and the step has no multiplication by it. The kernel
/// runs under the existing twelve-step integer-factor certificate: fixed-size
/// blocks remove per-step family and checkpoint dispatch, and every step rounds
/// as the generic recurrence does, except that an exact zero keeps the sign of
/// its product. Both live states must pass the same checkpoint as the generic
/// recurrence; a failure discards this candidate and keeps its range fallback.
#[inline]
fn integer_member<R: RealScalar, const GROWING: bool>(
    degree: usize,
    factor: R,
    first: R,
    (mut c, increment): (R, R),
    _tier: Integer,
) -> Option<R> {
    if degree == 0 {
        return Some(R::ONE);
    }
    let mut previous = R::ONE;
    let mut current = first;
    let mut remaining = degree - 1;
    let checkpoint = |previous, current| {
        <Native as Arithmetic<R>>::checkpoint(previous, Integer::NARROW)
            & <Native as Arithmetic<R>>::checkpoint(current, Integer::NARROW)
    };
    let mut step = |previous: &mut R, current: &mut R| {
        let next = if GROWING {
            let next = factor * *current - c * *previous;
            c += increment;
            next
        } else {
            factor * *current - *previous
        };
        (*previous, *current) = (*current, next);
    };
    while remaining >= Integer::STEPS {
        for _ in 0..Integer::STEPS {
            step(&mut previous, &mut current);
        }
        if !checkpoint(previous, current) {
            return None;
        }
        remaining -= Integer::STEPS;
    }
    for _ in 0..remaining {
        step(&mut previous, &mut current);
    }
    checkpoint(previous, current).then_some(current)
}

// Isolate each family's monomorphized loop from the public dispatch, so other
// families' cold arithmetic and errors do not spill its small live state.
#[inline(never)]
fn evaluate_member<R: RealScalar, const ORDER: usize>(
    degree: usize,
    point: R,
    recurrence: impl Fn(usize) -> Result<((R, R, R), R), SciOxideError>,
    tier: CheckpointTier,
) -> Result<R, SciOxideError> {
    if let Some(value) = candidate(member_kernel::<R, Native, ORDER, false>(
        degree,
        point,
        &recurrence,
        tier,
    ))? {
        return Ok(value[ORDER]);
    }
    if tier.steps() != 0 {
        for tier in [tier, tier.wide()] {
            if let Some(value) = candidate(member_kernel::<R, Native, ORDER, true>(
                degree,
                point,
                &recurrence,
                tier,
            ))? {
                return Ok(value[ORDER]);
            }
        }
    }
    extended_member::<R, ORDER>(degree, point, recurrence)
}

#[cold]
#[inline(never)]
fn extended_members<C: NumericScalar>(
    point: C,
    p0: C::Real,
    recurrence: impl Fn(usize) -> (C::Real, C::Real, C::Real),
    output: &mut [C],
) -> Result<(), SciOxideError> {
    extended(members_kernel::<C, Extended>(point, p0, recurrence, output))
}

#[inline(always)]
fn members_kernel<C: NumericScalar, A: Arithmetic<C, Projected = C>>(
    point: C,
    p0: C::Real,
    recurrence: impl Fn(usize) -> (C::Real, C::Real, C::Real),
    output: &mut [C],
) -> ArithmeticResult<()> {
    let x = A::scalar(point)?;
    let zero = A::scalar(C::ZERO)?;
    let mut previous = [zero; 3];
    let mut current = [A::real(p0)?, zero, zero];
    if !A::safe(x) || !A::safe(current[0]) {
        return Err(ArithmeticError::Retry);
    }
    let count = output.len();
    for (index, out) in output.iter_mut().enumerate() {
        *out = A::project(current[0])?;
        if index + 1 < count {
            let next = step::<C, A, 0>(current, previous, x, recurrence(index), zero, true)?;
            (previous, current) = (current, next);
        }
    }
    Ok(())
}

/// Both forward recurrence and backward Clenshaw differentiate the same update:
/// `next = forcing + (a*x+b)*current - c*previous`. Only the source of c differs.
/// ORDER selects requested derivatives, so unused derivatives cannot fail a call.
#[inline(always)]
fn step<C: NumericScalar, A: Arithmetic<C>, const ORDER: usize>(
    current: Jet<C, A>,
    previous: Jet<C, A>,
    x: A::Value,
    coefficients: (C::Real, C::Real, C::Real),
    forcing: A::Value,
    checked: bool,
) -> ArithmeticResult<Jet<C, A>> {
    let (a, b, c) = coefficients;
    let (a, b, c) = (A::real(a)?, A::real(b)?, A::real(c)?);
    if checked && !(A::safe(a) & A::safe(b) & A::safe(c) & A::safe(forcing)) {
        return Err(ArithmeticError::Retry);
    }
    let factor = A::mul_add(a, x, b)?;
    if checked && !A::safe(factor) {
        return Err(ArithmeticError::Retry);
    }
    let zero = A::scalar(C::ZERO)?;
    let mut sources = [forcing, zero, zero];
    if ORDER >= 1 {
        sources[1] = A::mul(a, current[0])?;
    }
    if ORDER >= 2 {
        let first = A::mul(a, current[1])?;
        sources[2] = A::add(first, first)?;
    }
    let mut next = [zero; 3];
    // Identical lane operations let native value/derivative arithmetic share
    // SIMD instructions; the differentiated forcing is the only difference.
    for (index, out) in next[..=ORDER].iter_mut().enumerate() {
        *out = A::sub(
            A::mul_add(factor, current[index], sources[index])?,
            A::mul(c, previous[index])?,
        )?;
    }
    require(!checked || next[..=ORDER].iter().copied().all(A::safe))?;
    Ok(next)
}

#[inline(always)]
pub(super) fn member<R: RealScalar, const ORDER: usize>(
    degree: usize,
    point: R,
    family: crate::OrthogonalFamily<R>,
) -> Result<R, SciOxideError> {
    if ORDER == 0 && matches!(family, crate::OrthogonalFamily::Legendre) {
        return scioxide_special::legendre_p(degree, point);
    }
    let tier = super::admission::family_tier(family, degree, point);
    if ORDER == 0
        && let CheckpointTier::Integer(integer) = tier
    {
        // Signed zero follows the generic `a*x + 0` update: `-0` becomes `+0`.
        let (single, double) = (point + R::ZERO, point + point + R::ZERO);
        let value = match family {
            crate::OrthogonalFamily::ChebyshevT => {
                integer_member::<R, false>(degree, double, single, (R::ONE, R::ZERO), integer)
            }
            crate::OrthogonalFamily::ChebyshevU => {
                integer_member::<R, false>(degree, double, double, (R::ONE, R::ZERO), integer)
            }
            crate::OrthogonalFamily::HermiteH => {
                integer_member::<R, true>(degree, double, double, (R::TWO, R::TWO), integer)
            }
            crate::OrthogonalFamily::HermiteHe => {
                integer_member::<R, true>(degree, single, single, (R::ONE, R::ONE), integer)
            }
            _ => None,
        };
        if let Some(value) = value {
            return Ok(value);
        }
    }
    // Match outside the hot recurrence so constant-family terms specialize.
    macro_rules! evaluate {
        ($family:expr) => {{
            let recurrence = |n| {
                // A family-wide certificate limits every index to 2^14,
                // within u32 and exactly representable in both widths. Retain checked casts
                // when that certificate is unavailable.
                let chebyshev = {
                    let family: F<R> = $family;
                    matches!(family, F::ChebyshevT | F::ChebyshevU)
                };
                let k = if chebyshev {
                    R::ZERO
                } else if tier.steps() != 0 {
                    R::try_from_f64_lossy(f64::from(n as u32))?
                } else {
                    R::try_from_usize_exact(n)?
                };
                $family.recurrence_at(n, k).map(|terms| (terms, R::ONE))
            };
            evaluate_member::<R, ORDER>(degree, point, recurrence, tier)
        }};
    }
    use crate::OrthogonalFamily as F;
    match family {
        F::ChebyshevT => evaluate!(F::ChebyshevT),
        F::ChebyshevU => evaluate!(F::ChebyshevU),
        F::Legendre => evaluate!(F::Legendre),
        F::HermiteH => evaluate!(F::HermiteH),
        F::HermiteHe => evaluate!(F::HermiteHe),
        F::Laguerre => evaluate!(F::Laguerre),
        F::Gegenbauer { lambda } if tier.steps() != 0 => {
            // The certificate admits all coefficients through this degree;
            // share the pure formula without rechecking it at every step.
            let recurrence = |n| {
                extended(super::family::gegenbauer_coefficients::<R, Native>(
                    lambda,
                    R::try_from_f64_lossy(f64::from(n as u32))?,
                    n == 0,
                ))
                .map(|terms| (terms, R::ONE))
            };
            evaluate_member::<R, ORDER>(degree, point, recurrence, tier)
        }
        F::Gegenbauer { lambda } => evaluate!(F::Gegenbauer { lambda }),
        F::GeneralizedLaguerre { alpha } if tier.steps() != 0 => {
            let recurrence = |n| {
                let k = R::try_from_f64_lossy(f64::from(n as u32))?;
                F::GeneralizedLaguerre { alpha }
                    .laguerre_numerators(n, k)
                    .map(|terms| (terms, k + R::ONE))
            };
            evaluate_member::<R, ORDER>(degree, point, recurrence, tier)
        }
        F::Jacobi { alpha, beta } if tier.steps() != 0 => {
            let recurrence = |n| {
                let k = R::try_from_f64_lossy(f64::from(n as u32))?;
                F::Jacobi { alpha, beta }
                    .certified_coefficients(n, k)
                    .map(|terms| (terms, R::ONE))
            };
            evaluate_member::<R, ORDER>(degree, point, recurrence, tier)
        }
        F::Jacobi { alpha, beta } => evaluate!(F::Jacobi { alpha, beta }),
        F::GeneralizedLaguerre { alpha } => evaluate!(F::GeneralizedLaguerre { alpha }),
    }
}

#[cold]
#[inline(never)]
fn extended_member<C: NumericScalar, const ORDER: usize>(
    degree: usize,
    point: C,
    recurrence: impl Fn(usize) -> Result<((C::Real, C::Real, C::Real), C::Real), SciOxideError>,
) -> Result<C, SciOxideError> {
    let result = extended(member_kernel::<C, Extended, ORDER, false>(
        degree,
        point,
        recurrence,
        CheckpointTier::CHECKED,
    ))?;
    result[ORDER].round()
}

#[inline(always)]
fn member_kernel<C: NumericScalar, A: Arithmetic<C>, const ORDER: usize, const FRAMED: bool>(
    degree: usize,
    point: C,
    recurrence: impl Fn(usize) -> Result<((C::Real, C::Real, C::Real), C::Real), SciOxideError>,
    tier: CheckpointTier,
) -> ArithmeticResult<Jet<C, A>> {
    match tier {
        CheckpointTier::PerStep(t) => {
            member_kernel_impl::<C, A, ORDER, FRAMED, _>(degree, point, recurrence, t)
        }
        CheckpointTier::Wide(t) => {
            member_kernel_impl::<C, A, ORDER, FRAMED, _>(degree, point, recurrence, t)
        }
        CheckpointTier::Narrow(t) => {
            member_kernel_impl::<C, A, ORDER, FRAMED, _>(degree, point, recurrence, t)
        }
        CheckpointTier::Integer(t) => {
            member_kernel_impl::<C, A, ORDER, FRAMED, _>(degree, point, recurrence, t)
        }
    }
}
#[inline(always)]
fn member_kernel_impl<
    C: NumericScalar,
    A: Arithmetic<C>,
    const ORDER: usize,
    const FRAMED: bool,
    T: Checkpoint,
>(
    degree: usize,
    point: C,
    recurrence: impl Fn(usize) -> Result<((C::Real, C::Real, C::Real), C::Real), SciOxideError>,
    _tier: T,
) -> ArithmeticResult<Jet<C, A>> {
    let x = A::scalar(point)?;
    if T::STEPS == 0 && !A::safe(x) {
        return Err(ArithmeticError::Retry);
    }
    let zero = A::scalar(C::ZERO)?;
    let mut previous = [zero; 3];
    let mut current = [A::scalar(C::ONE)?, zero, zero];
    let mut frame = A::frame();
    let mut frame_bound = A::magnitude(zero)?;
    let mut remaining = T::STEPS.max(1);
    for index in 0..degree {
        let (coefficients, divisor) = recurrence(index)?;
        // The reciprocal depends only on degree, so division can execute
        // independently of the live recurrence instead of extending its chain.
        let reciprocal = if divisor != C::Real::ONE {
            Some(A::div(A::real(C::Real::ONE)?, A::real(divisor)?)?)
        } else {
            None
        };
        let mut next =
            step::<C, A, ORDER>(current, previous, x, coefficients, zero, T::STEPS == 0)?;
        if let Some(reciprocal) = reciprocal {
            for value in &mut next[..=ORDER] {
                *value = A::mul(*value, reciprocal)?;
            }
        }
        (previous, current) = (current, next);
        remaining -= 1;
        if T::STEPS != 0 && (remaining == 0 || index + 1 == degree) {
            if FRAMED {
                let mut live = [
                    current[0],
                    previous[0],
                    current[1],
                    previous[1],
                    current[2],
                    previous[2],
                ];
                A::reframe::<false>(
                    &mut live[..2 * (ORDER + 1)],
                    &mut frame_bound,
                    &mut frame,
                    T::NARROW,
                )?;
                for i in 0..=ORDER {
                    current[i] = live[2 * i];
                    previous[i] = live[2 * i + 1];
                }
            } else {
                require(
                    current[..=ORDER]
                        .iter()
                        .copied()
                        .all(|v| A::checkpoint(v, T::NARROW))
                        & previous[..=ORDER]
                            .iter()
                            .copied()
                            .all(|v| A::checkpoint(v, T::NARROW)),
                )?;
            }
        }
        if remaining == 0 {
            remaining = T::STEPS.max(1);
        }
    }
    if FRAMED && T::STEPS != 0 && frame != A::frame() {
        current[ORDER] = A::restore(current[ORDER], frame)?;
    }
    Ok(current)
}

/// Evaluate an unprojected member from numerator coefficients and divisors.
/// The symmetric Jacobi recurrence shares the differentiated update above.
pub(super) fn scaled_member<R: RealScalar>(
    degree: usize,
    point: R,
    recurrence: impl Fn(usize) -> ((R, R, R), R),
) -> Result<scioxide_core::Scaled<R>, SciOxideError> {
    let x = ScaledValue::from_value(point)?;
    let mut previous = [ScaledValue::ZERO; 3];
    let mut current = [ScaledValue::ONE, ScaledValue::ZERO, ScaledValue::ZERO];
    for n in 0..degree {
        let (coefficients, divisor) = recurrence(n);
        let mut next = extended(step::<R, Extended, 0>(
            current,
            previous,
            x,
            coefficients,
            ScaledValue::ZERO,
            true,
        ))?;
        next[0] = next[0].div(ScaledValue::from_value(divisor)?)?;
        (previous, current) = (current, next);
    }
    Ok(current[0].real())
}

/// Clenshaw value/derivatives and a forward pass for the actual basis envelope.
/// NORMWISE allows every coefficient to vary by max |c|; otherwise the envelope
/// is componentwise. Recurrence data is read once per step in each direction.
#[inline]
pub(crate) fn series<
    C: NumericScalar,
    const ORDER: usize,
    const NORMWISE: bool,
    const BOUND: bool,
>(
    coefficients: impl DoubleEndedIterator<Item = C> + ExactSizeIterator + Clone,
    point: C,
    p0: C::Real,
    recurrence: impl Fn(usize) -> Result<(C::Real, C::Real, C::Real), SciOxideError>,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    evaluate_series::<C, ORDER, NORMWISE, BOUND>(
        coefficients,
        point,
        p0,
        CheckpointTier::CHECKED,
        recurrence,
    )
}

/// The checked view binds coefficients to the recurrence whose bounds were checked.
#[inline]
pub(super) fn admitted_series<
    C: NumericScalar,
    const ORDER: usize,
    const NORMWISE: bool,
    const BOUND: bool,
>(
    admission: SeriesAdmission<'_, C>,
    point: C,
    p0: C::Real,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    let tier = admission.tier([point], false);
    let recurrence = admission.recurrence();
    let coefficients = admission.coefficients();
    match recurrence.classical() {
        ClassicalRecurrence::ChebyshevT => evaluate_series::<C, ORDER, NORMWISE, BOUND>(
            coefficients.iter().copied(),
            point,
            p0,
            tier,
            |n| crate::OrthogonalFamily::ChebyshevT.recurrence(n),
        ),
        ClassicalRecurrence::ChebyshevU => evaluate_series::<C, ORDER, NORMWISE, BOUND>(
            coefficients.iter().copied(),
            point,
            p0,
            tier,
            |n| crate::OrthogonalFamily::ChebyshevU.recurrence(n),
        ),
        ClassicalRecurrence::Other => evaluate_series::<C, ORDER, NORMWISE, BOUND>(
            coefficients.iter().copied(),
            point,
            p0,
            tier,
            |n| Ok(recurrence[n]),
        ),
    }
}
#[inline(always)]
fn evaluate_series<
    C: NumericScalar,
    const ORDER: usize,
    const NORMWISE: bool,
    const BOUND: bool,
>(
    coefficients: impl DoubleEndedIterator<Item = C> + ExactSizeIterator + Clone,
    point: C,
    p0: C::Real,
    tier: CheckpointTier,
    recurrence: impl Fn(usize) -> Result<(C::Real, C::Real, C::Real), SciOxideError>,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    validate_slice(core::slice::from_ref(&point))?;
    let narrow = candidate(series_kernel::<C, Native, ORDER, NORMWISE, BOUND, false>(
        coefficients.clone(),
        point,
        p0,
        &recurrence,
        tier,
    ))?;
    let native = if narrow.is_some() || tier.steps() == 0 {
        narrow
    } else {
        let framed = candidate(series_kernel::<C, Native, ORDER, NORMWISE, BOUND, true>(
            coefficients.clone(),
            point,
            p0,
            &recurrence,
            tier,
        ))?;
        if framed.is_some() {
            framed
        } else {
            candidate(series_kernel::<C, Native, ORDER, NORMWISE, BOUND, true>(
                coefficients.clone(),
                point,
                p0,
                &recurrence,
                tier.wide(),
            ))?
        }
    };
    if let Some(state) = native {
        return Ok(HornerEvaluation::Native(state));
    }
    extended_series::<C, ORDER, NORMWISE, BOUND>(coefficients, point, p0, recurrence)
}

#[cold]
#[inline(never)]
fn extended_series<
    C: NumericScalar,
    const ORDER: usize,
    const NORMWISE: bool,
    const BOUND: bool,
>(
    coefficients: impl DoubleEndedIterator<Item = C> + ExactSizeIterator + Clone,
    point: C,
    p0: C::Real,
    recurrence: impl Fn(usize) -> Result<(C::Real, C::Real, C::Real), SciOxideError>,
) -> Result<HornerEvaluation<C>, SciOxideError> {
    let state = extended(series_kernel::<C, Extended, ORDER, NORMWISE, BOUND, false>(
        coefficients,
        ScaledValue::from_value(point)?,
        p0,
        recurrence,
        CheckpointTier::CHECKED,
    ))?;
    Ok(HornerEvaluation::Extended(state))
}

#[inline(always)]
fn series_kernel<
    C: NumericScalar,
    A: Arithmetic<C>,
    const ORDER: usize,
    const NORMWISE: bool,
    const BOUND: bool,
    const FRAMED: bool,
>(
    coefficients: impl DoubleEndedIterator<Item = C> + ExactSizeIterator + Clone,
    point: A::Value,
    p0: C::Real,
    recurrence: impl Fn(usize) -> Result<(C::Real, C::Real, C::Real), SciOxideError>,
    tier: CheckpointTier,
) -> ArithmeticResult<State<C, A>> {
    match tier {
        CheckpointTier::PerStep(t) => {
            series_kernel_impl::<C, A, ORDER, NORMWISE, BOUND, FRAMED, _>(
                coefficients,
                point,
                p0,
                recurrence,
                t,
            )
        }
        CheckpointTier::Wide(t) => series_kernel_impl::<C, A, ORDER, NORMWISE, BOUND, FRAMED, _>(
            coefficients,
            point,
            p0,
            recurrence,
            t,
        ),
        CheckpointTier::Narrow(t) => series_kernel_impl::<C, A, ORDER, NORMWISE, BOUND, FRAMED, _>(
            coefficients,
            point,
            p0,
            recurrence,
            t,
        ),
        CheckpointTier::Integer(t) => {
            series_kernel_impl::<C, A, ORDER, NORMWISE, BOUND, FRAMED, _>(
                coefficients,
                point,
                p0,
                recurrence,
                t,
            )
        }
    }
}

#[inline(never)]
fn series_kernel_impl<
    C: NumericScalar,
    A: Arithmetic<C>,
    const ORDER: usize,
    const NORMWISE: bool,
    const BOUND: bool,
    const FRAMED: bool,
    T: Checkpoint,
>(
    coefficients: impl DoubleEndedIterator<Item = C> + ExactSizeIterator + Clone,
    point: A::Value,
    p0: C::Real,
    recurrence: impl Fn(usize) -> Result<(C::Real, C::Real, C::Real), SciOxideError>,
    _tier: T,
) -> ArithmeticResult<State<C, A>> {
    let x = point;
    let p0 = A::real(p0)?;
    if !(A::safe(x) & A::safe(p0)) {
        return Err(ArithmeticError::Retry);
    }
    if T::NARROW && !A::checkpoint(p0, true) {
        return Err(ArithmeticError::Retry);
    }
    let zero = A::scalar(C::ZERO)?;
    let mut next = [zero; 3];
    let mut next2 = next;
    let mut cnext = C::Real::ZERO;
    let mut previous = [zero; 3];
    let mut current = [p0, zero, zero];
    let mut bound = A::magnitude(zero)?;
    let mut coefficient_max = bound;
    let mut backward_frame = A::frame();
    let mut forward_frame = A::frame();
    let mut backward_bound = bound;
    let len = coefficients.len();
    let mut remaining = T::STEPS.max(1);
    // These traversals are independent. Interleaving them exposes their
    // arithmetic to the processor without allocating basis values or scratch.
    for (index, (coefficient, bound_coefficient)) in
        coefficients.clone().rev().zip(coefficients).enumerate()
    {
        let (a, b, c) = recurrence(len - 1 - index)?;
        let value = step::<C, A, ORDER>(
            next,
            next2,
            x,
            (a, b, cnext),
            if FRAMED {
                A::forcing(coefficient, backward_frame)?
            } else {
                A::scalar(coefficient)?
            },
            T::STEPS == 0,
        )?;
        (next2, next) = (next, value);
        cnext = c;
        if BOUND {
            let coefficient = A::scalar(bound_coefficient)?;
            if T::STEPS == 0 && !A::safe(coefficient) {
                return Err(ArithmeticError::Retry);
            }
            let magnitude = A::magnitude(coefficient)?;
            let basis = A::magnitude(current[0])?;
            if NORMWISE {
                coefficient_max = A::bound_max(coefficient_max, magnitude);
                bound = A::bound_add(bound, basis)?;
            } else {
                bound = A::bound_add(bound, A::bound_mul(magnitude, basis)?)?;
            }
            if T::STEPS == 0 && !A::safe_bound(bound) {
                return Err(ArithmeticError::Retry);
            }
            if index + 1 < len {
                let value = step::<C, A, 0>(
                    current,
                    previous,
                    x,
                    recurrence(index)?,
                    zero,
                    T::STEPS == 0,
                )?;
                (previous, current) = (current, value);
            }
        }
        remaining -= 1;
        if T::STEPS != 0 && (remaining == 0 || index + 1 == len) {
            if FRAMED {
                let mut live = [next[0], next2[0], next[1], next2[1], next[2], next2[2]];
                A::reframe::<true>(
                    &mut live[..2 * (ORDER + 1)],
                    &mut backward_bound,
                    &mut backward_frame,
                    T::NARROW,
                )?;
                for i in 0..=ORDER {
                    next[i] = live[2 * i];
                    next2[i] = live[2 * i + 1];
                }
                if BOUND {
                    let mut live = [current[0], previous[0]];
                    A::reframe::<false>(&mut live, &mut bound, &mut forward_frame, T::NARROW)?;
                    current[0] = live[0];
                    previous[0] = live[1];
                }
            } else {
                require(
                    next[..=ORDER]
                        .iter()
                        .copied()
                        .all(|v| A::checkpoint(v, T::NARROW))
                        & next2[..=ORDER]
                            .iter()
                            .copied()
                            .all(|v| A::checkpoint(v, T::NARROW))
                        & (!BOUND
                            || (A::checkpoint(current[0], T::NARROW)
                                & A::checkpoint(previous[0], T::NARROW)
                                & A::bound_checkpoint(bound, T::NARROW))),
                )?;
            }
        }
        if remaining == 0 {
            remaining = T::STEPS.max(1);
        }
    }
    for value in &mut next[..=ORDER] {
        if FRAMED && T::STEPS != 0 {
            *value = A::restore(*value, backward_frame)?;
        }
        *value = A::mul(*value, p0)?;
        if FRAMED && T::STEPS != 0 {
            *value = A::restore(*value, A::frame())?;
        }
    }
    if (!FRAMED || T::STEPS == 0) && !next[..=ORDER].iter().copied().all(A::safe) {
        return Err(ArithmeticError::Retry);
    }
    if FRAMED && BOUND && T::STEPS != 0 {
        bound = A::restore_bound(bound, forward_frame)?;
    }
    if BOUND && NORMWISE {
        bound = A::bound_mul(coefficient_max, bound)?;
    }
    if (!FRAMED || T::STEPS == 0) && !A::safe_bound(bound) {
        return Err(ArithmeticError::Retry);
    }
    if FRAMED && BOUND && T::STEPS != 0 {
        bound = A::restore_bound(bound, A::frame())?;
    }
    Ok(HornerState {
        value: next[0],
        derivative: next[1],
        second_derivative: next[2],
        bound,
    })
}

#[cfg(test)]
pub(super) fn checked_member<R: RealScalar, const ORDER: usize>(
    degree: usize,
    point: R,
    family: crate::OrthogonalFamily<R>,
) -> Result<R, SciOxideError> {
    if let Some(value) = candidate(member_kernel::<R, Native, ORDER, false>(
        degree,
        point,
        |n| family.recurrence(n).map(|terms| (terms, R::ONE)),
        CheckpointTier::CHECKED,
    ))? {
        return Ok(value[ORDER]);
    }
    extended_member::<R, ORDER>(degree, point, |n| {
        family.recurrence(n).map(|terms| (terms, R::ONE))
    })
}

#[cfg(test)]
mod tests {
    use super::super::admission::RecurrenceData;
    use super::*;
    use crate::OrthogonalFamily as F;
    use crate::admission::{CoefficientView, block_point};
    use scioxide_core::{Complex, RealScalar};

    fn near<C: NumericScalar>(actual: C, expected: C, factor: usize) {
        let tolerance = C::Real::EPSILON
            * C::Real::try_from_usize_exact(factor).unwrap()
            * (C::Real::ONE + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:?} != {expected:?}"
        );
    }

    /// The blocked integer-family kernel rounds every step as the checked
    /// generic recurrence does; only the sign of an exact zero may differ.
    #[test]
    fn integer_family_members_equal_the_checked_forward_recurrence() {
        for family in [F::ChebyshevT, F::ChebyshevU, F::HermiteH, F::HermiteHe] {
            for degree in 0..=128 {
                for point in [
                    -16.0_f64, -1.0, -0.875, -0.125, -0.0, 0.0, 0.25, 0.875, 1.0, 4.0, 10.5,
                ] {
                    // `eval` returns the identity members T_1 and He_1 bit for bit.
                    let identity = degree == 1 && matches!(family, F::ChebyshevT | F::HermiteHe);
                    let expected = if identity {
                        point
                    } else {
                        checked_member::<_, 0>(degree, point, family).unwrap()
                    };
                    let actual = family.evaluate(degree, point).unwrap();
                    assert_eq!(
                        actual, expected,
                        "{family:?}, n={degree}, x={point}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn chebyshev_checkpoints_match_checked_forward_recurrence_at_boundaries() {
        for family in [F::ChebyshevT, F::ChebyshevU] {
            for degree in [
                0, 1, 2, 11, 12, 13, 14, 24, 25, 26, 16, 32, 48, 64, 256, 1024, 16384,
            ] {
                for point in [
                    -1.0_f64.next_down(),
                    -1.0,
                    -0.5,
                    2.0_f64.powi(-17),
                    2.0_f64.powi(-16),
                    0.0,
                    0.5,
                    1.0,
                    1.0_f64.next_down(),
                ] {
                    let expected = checked_member::<_, 0>(degree, point, family).unwrap();
                    let actual = family.evaluate(degree, point).unwrap();
                    let envelope = if family == F::ChebyshevU {
                        (degree + 1) as f64
                    } else {
                        1.0
                    };
                    assert!(
                        (actual - expected).abs() <= 2.0 * degree as f64 * f64::EPSILON * envelope
                    );
                }
            }
        }
    }

    #[test]
    fn chebyshev_forward_blocks_retain_the_generic_range_fallback() {
        for family in [F::ChebyshevT, F::ChebyshevU] {
            for degree in [400, 512, 1024] {
                for point in [-2.0_f64, 2.0, 16.0] {
                    match checked_member::<_, 0>(degree, point, family) {
                        Ok(expected) => {
                            let actual = family.evaluate(degree, point).unwrap();
                            assert!(
                                (actual / expected - 1.0).abs()
                                    <= 2.0 * degree as f64 * f64::EPSILON
                            );
                        }
                        Err(expected) => assert_eq!(
                            family.evaluate(degree, point).unwrap_err().kind(),
                            expected.kind()
                        ),
                    }
                }
            }
        }
    }

    #[test]
    fn small_coefficients_keep_complete_reports_in_native_lanes() {
        let mut coefficients = [0.25_f64; 64];
        for small in [2.0_f64.powi(-17), 2.0_f64.powi(-64)] {
            coefficients[31] = small;
            for family in [F::ChebyshevT, F::Legendre, F::Laguerre] {
                let recurrence =
                    RecurrenceData::new((0..64).map(|n| family.recurrence(n).unwrap()).collect());
                let points = [-0.5, 0.0, 0.25, 0.875];
                let result = series_lanes(
                    recurrence.series(CoefficientView::new(&coefficients)),
                    points,
                    1.0,
                )
                .unwrap()
                .unwrap();
                for ((value, derivative, bound), point) in result.into_iter().zip(points) {
                    let expected = series_kernel::<_, Extended, 1, false, true, false>(
                        coefficients.into_iter(),
                        ScaledValue::from_value(point).unwrap(),
                        1.0,
                        |n| family.recurrence(n),
                        CheckpointTier::CHECKED,
                    )
                    .unwrap();
                    near(value, expected.value.round().unwrap(), 2048);
                    near(derivative, expected.derivative.round().unwrap(), 2048);
                    near(bound, expected.bound.round().unwrap(), 2048);
                }
            }
        }
    }

    #[test]
    fn formerly_restarting_series_remain_native_and_match_scaled() {
        for family in [F::HermiteH, F::Legendre, F::Laguerre] {
            for length in [40, 48, 56, 64, 80, 128, 192, 256] {
                let coefficients: alloc::vec::Vec<f64> = (0..length)
                    .map(|i| if i % 3 == 0 { -0.25 } else { 0.5 })
                    .collect();
                let recurrence = RecurrenceData::new(
                    (0..length).map(|n| family.recurrence(n).unwrap()).collect(),
                );
                for point in [0.0, 0.25, -0.5, 0.875] {
                    let actual = admitted_series::<_, 1, false, true>(
                        recurrence.series(CoefficientView::new(&coefficients)),
                        point,
                        1.0,
                    )
                    .unwrap();
                    assert!(
                        matches!(actual, HornerEvaluation::Native(_)),
                        "{family:?}/{length} at {point}"
                    );
                    let expected = series_kernel::<_, Extended, 1, false, true, false>(
                        coefficients.iter().copied(),
                        ScaledValue::from_value(point).unwrap(),
                        1.0,
                        |n| family.recurrence(n),
                        CheckpointTier::CHECKED,
                    )
                    .unwrap();
                    let (value, derivative, bound) = actual.rounded().unwrap();
                    near(value, expected.value.round().unwrap(), 8192);
                    near(derivative, expected.derivative.round().unwrap(), 8192);
                    near(bound, expected.bound.round().unwrap(), 8192);
                }
            }
        }
    }

    #[test]
    fn native_lanes_and_value_only_series_use_only_their_outputs() {
        let coefficients = [0.25_f64; 128];
        let recurrence = RecurrenceData::new(
            (0..128)
                .map(|n| F::HermiteH.recurrence(n).unwrap())
                .collect(),
        );
        let points = [0.0, -0.25, 0.5, 0.875];
        let admission = recurrence.series(CoefficientView::new(&coefficients));
        let lanes = series_lanes(admission, points, 1.0).unwrap().unwrap();
        for ((value, derivative, bound), point) in lanes.into_iter().zip(points) {
            let expected = admitted_series::<_, 1, false, true>(admission, point, 1.0)
                .unwrap()
                .rounded()
                .unwrap();
            near(value, expected.0, 8192);
            near(derivative, expected.1, 8192);
            near(bound, expected.2, 8192);
        }
        let large = f64::MAX * 0.75;
        let actual = series::<_, 0, false, false>([-large, large].into_iter(), 1.0, 1.0, |n| {
            F::ChebyshevT.recurrence(n)
        })
        .unwrap()
        .into_scaled()
        .unwrap();
        assert_eq!(actual.value.round().unwrap(), 0.0);
        assert!(
            actual.derivative.is_zero()
                && actual.second_derivative.is_zero()
                && actual.bound.is_zero()
        );
    }

    fn paths<C: NumericScalar>(imaginary: f64) {
        let real = |value| C::Real::try_from_f64_exact(value).unwrap();
        let point = C::from_components(real(0.25), real(imaginary));
        let coefficients = [
            C::ONE,
            C::ZERO,
            C::from_real(real(-0.5)),
            C::from_components(real(0.25), real(imaginary)),
        ];
        for family in [
            F::ChebyshevT,
            F::ChebyshevU,
            F::Legendre,
            F::HermiteH,
            F::HermiteHe,
            F::Laguerre,
            F::GeneralizedLaguerre { alpha: real(0.5) },
            F::Jacobi {
                alpha: real(0.25),
                beta: real(0.75),
            },
            F::Gegenbauer {
                lambda: real(-0.25),
            },
        ] {
            let native = series_kernel::<C, Native, 2, false, true, false>(
                coefficients.iter().copied(),
                point,
                C::Real::ONE,
                |k| family.recurrence(k),
                CheckpointTier::CHECKED,
            )
            .unwrap();
            let scaled = series_kernel::<C, Extended, 2, false, true, false>(
                coefficients.iter().copied(),
                ScaledValue::from_value(point).unwrap(),
                C::Real::ONE,
                |k| family.recurrence(k),
                CheckpointTier::CHECKED,
            )
            .unwrap();
            near(native.value, scaled.value.round().unwrap(), 256);
            near(native.derivative, scaled.derivative.round().unwrap(), 256);
            near(
                native.second_derivative,
                scaled.second_derivative.round().unwrap(),
                256,
            );
            near(
                C::from_real(native.bound),
                C::from_real(scaled.bound.round().unwrap()),
                256,
            );
        }
    }

    #[test]
    fn native_and_scaled_series_agree_in_every_scalar_width() {
        paths::<f32>(0.0);
        paths::<f64>(0.0);
        paths::<Complex<f32>>(0.125);
        paths::<Complex<f64>>(0.125);
    }

    fn range<R: RealScalar>(hermite_degree: usize, laguerre_degree: usize, point: R) {
        for (family, degree, point) in [
            (F::HermiteH, hermite_degree, R::ZERO),
            (F::Laguerre, laguerre_degree, point),
        ] {
            assert!(
                member_kernel::<R, Native, 0, false>(
                    degree,
                    point,
                    |k| family.recurrence(k).map(|terms| (terms, R::ONE)),
                    CheckpointTier::CHECKED
                )
                .is_err()
            );
            let expected = member_kernel::<R, Extended, 0, false>(
                degree,
                point,
                |k| family.recurrence(k).map(|terms| (terms, R::ONE)),
                CheckpointTier::CHECKED,
            )
            .unwrap()[0]
                .round()
                .unwrap();
            near(family.evaluate(degree, point).unwrap(), expected, 2048);
            let derivative = member_kernel::<R, Extended, 1, false>(
                degree,
                point,
                |k| family.recurrence(k).map(|terms| (terms, R::ONE)),
                CheckpointTier::CHECKED,
            )
            .unwrap()[1]
                .round()
                .unwrap();
            near(
                family.evaluate_derivative(degree, point).unwrap(),
                derivative,
                2048,
            );
        }
    }

    #[test]
    fn high_degree_members_restart_in_extended_range() {
        range::<f32>(40, 32, -40.0);
        range::<f64>(200, 128, -1000.0);
        // Only requested outputs are projected: these zero results survive
        // intermediate values and unused derivatives beyond binary64 range.
        assert_eq!(F::HermiteH.evaluate(301, 0.0_f64).unwrap(), 0.0);
        assert_eq!(F::HermiteH.evaluate_derivative(300, 0.0_f64).unwrap(), 0.0);
        assert!(F::HermiteH.evaluate(300, 0.0_f64).is_err());
    }

    #[test]
    fn sparse_coefficients_keep_the_distinct_envelope_models() {
        let coefficients = [0.0_f64, 0.0, 1.0];
        let componentwise =
            series::<_, 1, false, true>(coefficients.iter().copied(), 0.0, 1.0, |k| {
                F::ChebyshevT.recurrence(k)
            })
            .unwrap()
            .rounded()
            .unwrap();
        let normwise = series::<_, 2, true, true>(coefficients.iter().copied(), 0.0, 1.0, |k| {
            F::ChebyshevT.recurrence(k)
        })
        .unwrap()
        .rounded()
        .unwrap();
        assert_eq!(componentwise.0, -1.0);
        assert_eq!(componentwise.2, 1.0);
        assert_eq!(normwise.0, -1.0);
        assert_eq!(normwise.2, 2.0);
    }

    #[test]
    fn blocked_checkpoints_agree_at_admission_edges_and_restart() {
        let coefficients = [0.25_f64, -0.5, 0.125, 0.25, -0.125];
        for point in [1.0 / 65536.0, 0.25, 65536.0] {
            for p0 in [f64::MIN_POSITIVE.sqrt(), 1.0, f64::MAX.sqrt() / 4.0] {
                let recurrence = RecurrenceData::new(
                    (0..coefficients.len())
                        .map(|k| F::ChebyshevT.recurrence(k).unwrap())
                        .collect(),
                );
                let actual = admitted_series::<_, 2, false, true>(
                    recurrence.series(CoefficientView::new(&coefficients)),
                    point,
                    p0,
                )
                .unwrap()
                .into_scaled()
                .unwrap();
                let expected = series_kernel::<_, Extended, 2, false, true, false>(
                    coefficients.iter().copied(),
                    ScaledValue::from_value(point).unwrap(),
                    p0,
                    |k| F::ChebyshevT.recurrence(k),
                    CheckpointTier::CHECKED,
                )
                .unwrap();
                near(
                    actual.value.round().unwrap(),
                    expected.value.round().unwrap(),
                    256,
                );
                near(
                    actual.derivative.round().unwrap(),
                    expected.derivative.round().unwrap(),
                    256,
                );
                near(
                    actual.second_derivative.round().unwrap(),
                    expected.second_derivative.round().unwrap(),
                    256,
                );
                near(
                    actual.bound.round().unwrap(),
                    expected.bound.round().unwrap(),
                    256,
                );
            }
        }
        for point in [
            (1.0_f64 / 65536.0).next_after(0.0),
            65536.0_f64.next_after(f64::MAX),
        ] {
            assert!(!block_point(point));
        }
        assert!(!block_point(0.25_f32));
        assert!(!block_point(Complex::new(0.25_f64, 0.0)));
    }
    #[test]
    fn inadmissible_coefficients_retry_without_unchecked_lanes() {
        let terms = RecurrenceData::new(
            (0..16)
                .map(|n| F::ChebyshevT.recurrence(n).unwrap())
                .collect(),
        );
        for small in [2.0_f64.powi(-65), f64::from_bits(1)] {
            let mut coefficients = [0.25; 16];
            coefficients[7] = small;
            let view = terms.series(CoefficientView::new(&coefficients));
            assert!(series_lanes(view, [0.25, 0.5], 1.0).unwrap().is_none());
            let actual = admitted_series::<_, 1, false, true>(view, 0.5, 1.0).unwrap();
            if small == f64::from_bits(1) {
                assert!(matches!(actual, HornerEvaluation::Extended(_)));
            }
            let actual = actual.into_scaled().unwrap();
            let expected = series_kernel::<f64, Extended, 1, false, true, false>(
                coefficients.into_iter(),
                ScaledValue::from_value(0.5).unwrap(),
                1.0,
                |n| F::ChebyshevT.recurrence(n),
                CheckpointTier::CHECKED,
            )
            .unwrap();
            near(
                actual.value.round().unwrap(),
                expected.value.round().unwrap(),
                256,
            );
            near(
                actual.derivative.round().unwrap(),
                expected.derivative.round().unwrap(),
                256,
            );
        }
    }
}
