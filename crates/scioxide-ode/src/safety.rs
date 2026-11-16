//! Checked numerical and lifecycle kernels shared by ODE orchestration.

mod implicit;
pub(crate) use implicit::{
    implicit_stage_rows, normalized_stage_residual, scalar_stage_residual, stage_derivative,
    stage_residual,
};

use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::checked_final_rounding;
use scioxide_core::{NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{checked_difference, checked_float_product};
use scioxide_linalg::Vector;

/// Unsupported configuration of `argument`.
pub(crate) fn invalid(argument: scioxide_core::Argument, message: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        message,
    )
}

/// Reject non-finite provider output.
#[inline(always)]
pub(crate) fn finite<T: NumericScalar>(
    values: &[T],
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if values.iter().all(|v| v.has_finite_components()) {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            origin,
            "all components must be finite",
        ))
    }
}

#[inline]
pub(crate) fn validate_finite_time<S: RealScalar>(
    time: S,
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if time.is_finite() {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(origin, "time must be finite"))
    }
}

pub(crate) fn validate_finite_state<S>(
    state: &Vector<S>,
    dimension: usize,
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError>
where
    S: NumericScalar,
{
    validate_vector(state.as_slice(), dimension, origin)
}

/// Validate the shape and components of a borrowed state or provider vector.
#[inline]
pub(crate) fn validate_vector<T: NumericScalar>(
    values: &[T],
    dimension: usize,
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if values.len() != dimension {
        return Err(SciOxideError::dimension_mismatch(dimension, values.len()));
    }
    finite(values, origin)
}

/// Validate a square callback matrix without assuming contiguous storage.
pub(crate) fn validate_matrix_shape<T: NumericScalar>(
    dimension: usize,
    matrix: scioxide_linalg::MatrixView<'_, T>,
) -> Result<(), SciOxideError> {
    if matrix.rows() != dimension || matrix.cols() != dimension {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(dimension, dimension),
            scioxide_core::Dimensions::two(matrix.rows(), matrix.cols()),
        ));
    }
    Ok(())
}

/// Validate callback matrix components, including strided views.
pub(crate) fn validate_matrix<T: NumericScalar>(
    dimension: usize,
    matrix: scioxide_linalg::MatrixView<'_, T>,
) -> Result<(), SciOxideError> {
    validate_matrix_shape(dimension, matrix)?;
    for row in 0..dimension {
        for col in 0..dimension {
            finite(
                core::slice::from_ref(&matrix.get(row, col)?),
                NonFiniteOrigin::Callback,
            )?;
        }
    }
    Ok(())
}

/// Poll execution capabilities before counting or committing work.
#[inline]
pub(crate) fn checkpoint(cx: &scioxide_core::SolveContext<'_, '_>) -> Result<(), SciOxideError> {
    cx.execution().check()
}

/// Infinity norm of finite numerical components.
pub(crate) fn max_abs<T: NumericScalar>(values: impl IntoIterator<Item = T>) -> T::Real {
    values
        .into_iter()
        .fold(T::Real::ZERO, |maximum, value| maximum.max(value.abs()))
}

pub(crate) fn checked_step_endpoint<S: RealScalar>(time: S, step: S) -> Result<S, SciOxideError> {
    validate_finite_time(time, NonFiniteOrigin::Input)?;
    if !step.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "step size must be finite",
        ));
    }
    if step == S::ZERO {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("dt"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "step size must be nonzero",
        ));
    }
    let endpoint =
        checked_final_rounding(time + step, "step endpoint is outside the finite range")?;
    if endpoint == time {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "step size does not advance representable time",
        ));
    }
    Ok(endpoint)
}

pub(crate) fn checked_stage_time<S: RealScalar>(
    time: S,
    step: S,
    node: S,
) -> Result<S, SciOxideError> {
    if !node.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "stage node must be finite",
        ));
    }
    let offset = checked_float_product(step, node)?;
    checked_final_rounding(time + offset, "stage time is outside the finite range")
}

/// The caller has checked finite time/step and endpoint, and the prepared
/// node lies in [0,1]. Monotonicity keeps every stage inside those endpoints.
/// An exact zero node keeps the positive zero of the checked product.
#[inline(always)]
pub(crate) fn prepared_stage_time<S: RealScalar>(time: S, step: S, node: S) -> S {
    time + if node == S::ZERO {
        S::ZERO
    } else {
        step * node
    }
}

/// Signed distance to the adjacent finite time toward a distinct finite endpoint.
///
/// This direction-dependent increment includes signed zero and subnormals; it
/// is not an accuracy tolerance. O(1), no storage. Callers validate a finite,
/// nonempty span before using this helper.
pub(crate) fn adjacent_time_step<S: RealScalar>(time: S, toward: S) -> Result<S, SciOxideError> {
    checked_difference(time.next_after(toward), time)
}

/// Base-2 logarithm of the relative bound, in machine epsilons, on the rounding
/// that accumulated step additions leave in a time near an endpoint.
///
/// Each addition `t + h` rounds by at most `ε·|t + h| / 2`, so `k` steps to `end`
/// accumulate at most about `k·ε·max(|t|, |end|) / 2`. The bound `2¹⁰ = 1024`
/// covers about two thousand fixed or halved steps; larger remainders are real time.
const ENDPOINT_ROUNDING_DOUBLINGS: u32 = 10;

/// Whether `time` lies within the rounding of accumulated times of `end`: at most
/// `1024·ε·max(|time|, |end|)` away, where `ε` is the machine epsilon.
pub(crate) fn within_endpoint_rounding<S: RealScalar>(time: S, end: S) -> bool {
    let bound = (0..ENDPOINT_ROUNDING_DOUBLINGS).fold(S::EPSILON, |bound, _| bound * S::TWO);
    (end - time).abs() <= bound * time.abs().max(end.abs())
}

/// Select a bounded, directed step and its already-checked endpoint; `argument` names the
/// step configuration. `None` means no representable progress fits the proposal and maximum
/// bound. A final clipped step may be shorter than the caller's minimum retry step. A step
/// that advances but would stop short of `end` by no more than
/// [`within_endpoint_rounding`] ends at `end` instead, so accumulated time rounding never leaves a rounding-sized last step;
/// such a step may exceed the maximum by that remainder.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_step<S: RealScalar>(
    time: S,
    end: S,
    direction: S,
    proposed_magnitude: S,
    min_step: Option<S>,
    max_step: Option<S>,
    argument: scioxide_core::Argument,
) -> Result<Option<(S, S)>, SciOxideError> {
    if !proposed_magnitude.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "proposed step magnitude must be finite",
        ));
    }
    if proposed_magnitude <= S::ZERO {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "proposed step magnitude must be positive",
        ));
    }
    if direction != S::ONE && direction != -S::ONE {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "integration direction must be -1 or 1",
        ));
    }

    let mut magnitude = proposed_magnitude;
    if let Some(maximum) = max_step {
        magnitude = magnitude.min(maximum);
    }
    if let Some(minimum) = min_step {
        magnitude = magnitude.max(minimum);
    }
    let signed = direction * magnitude;
    let candidate =
        match checked_final_rounding(time + signed, "proposed time is outside the finite range") {
            Ok(candidate) => candidate,
            Err(ref error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    }
                ) =>
            {
                end
            }
            Err(error) => return Err(error),
        };
    let reaches_end = if direction > S::ZERO {
        candidate >= end
    } else {
        candidate <= end
    };
    let snapped = !reaches_end && candidate != time && within_endpoint_rounding(candidate, end);
    let mut next = if reaches_end || snapped {
        end
    } else {
        candidate
    };
    let mut step = checked_difference(next, time)?;
    // Endpoint rounding can enlarge the increment beyond an explicit maximum.
    if !snapped && max_step.is_some_and(|maximum| step.abs() > maximum) {
        next = next.next_after(time);
        step = checked_difference(next, time)?;
    }
    // Subtraction need not round-trip across zero or widely separated scales.
    // Publish only the endpoint the method actually evaluates. If subtraction
    // rounded past the bound, shorten the increment by one float first.
    next = time + step;
    if (direction > S::ZERO && next > end) || (direction < S::ZERO && next < end) {
        step = step.next_after(S::ZERO);
        next = time + step;
    }
    // Distinct finite floating-point endpoints are at least one representable
    // increment apart. Equality therefore detects stagnation without another
    // next-after operation or distance subtraction.
    if next == time {
        return Ok(None);
    }
    if (step > S::ZERO) != (direction > S::ZERO) {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "step proposal does not follow the integration direction",
        ));
    }
    Ok(Some((step, next)))
}

pub(crate) fn scaled_rms_norm<T: NumericScalar>(values: &[T]) -> Result<T::Real, SciOxideError> {
    if values.is_empty() {
        return Ok(<T::Real as scioxide_core::AlgebraicScalar>::ZERO);
    }
    let mut norm = scioxide_core::ScaledNorm::new();
    for &value in values {
        norm.add_magnitude(value.abs())?;
    }
    let dimension = <T::Real>::try_from_usize_exact(values.len())?;
    Ok(norm.finish()? / dimension.sqrt())
}

/// Apply the real time increment to both state components, rescaling to avoid
/// overflow and underflow. No real/imaginary coordinate repacking is involved.
pub(crate) fn state_increment<T: NumericScalar>(
    value: T,
    step: T::Real,
    slope: T,
) -> Result<T, SciOxideError> {
    let real =
        scioxide_core::checked_mul_add(step, slope.real_component(), value.real_component())?;
    let imaginary = if matches!(
        T::kind(),
        scioxide_core::backend::ScalarKind::Complex32
            | scioxide_core::backend::ScalarKind::Complex64
    ) {
        scioxide_core::checked_mul_add(
            step,
            slope.imaginary_component(),
            value.imaginary_component(),
        )?
    } else {
        T::Real::ZERO
    };
    Ok(T::from_components(real, imaginary))
}

pub(crate) fn state_difference<T: NumericScalar>(a: T, b: T) -> Result<T, SciOxideError> {
    let real = checked_difference(a.real_component(), b.real_component())?;
    let imaginary = if matches!(
        T::kind(),
        scioxide_core::backend::ScalarKind::Complex32
            | scioxide_core::backend::ScalarKind::Complex64
    ) {
        checked_difference(a.imaginary_component(), b.imaginary_component())?
    } else {
        T::Real::ZERO
    };
    Ok(T::from_components(real, imaginary))
}

/// Native real-weighted state combination, with replay from the original
/// operands when a sum overflows or a large step amplifies a zero or subnormal
/// weighted sum. A zero sum whose products all have an exactly zero operand is
/// exact and stays native. The step factor is applied only after reducing the
/// weights. Ordinary sums have the method's usual floating-point rounding error.
#[inline]
pub(crate) fn stage_affine<T: NumericScalar, I>(
    state: T,
    step: T::Real,
    terms: I,
) -> Result<T, SciOxideError>
where
    I: IntoIterator<Item = (T::Real, T)>,
    I::IntoIter: Clone,
{
    let terms = terms.into_iter();
    let component = |initial: T::Real, imaginary: bool| {
        let mut weighted = T::Real::ZERO;
        for (weight, value) in terms.clone() {
            let slope = if imaginary {
                value.imaginary_component()
            } else {
                value.real_component()
            };
            weighted = scioxide_core::fast_mul_add(weight, slope, weighted);
        }
        let sum = scioxide_core::fast_mul_add(step, weighted, initial);
        if native_affine_component(step, weighted, sum)
            || (weighted == T::Real::ZERO
                && sum.is_finite()
                && exact_zero_products(terms.clone().map(|(weight, value)| {
                    (
                        weight,
                        if imaginary {
                            value.imaginary_component()
                        } else {
                            value.real_component()
                        },
                    )
                })))
        {
            Ok(sum)
        } else {
            affine_component_fallback(
                initial,
                step,
                terms.clone().map(|(weight, value)| {
                    (
                        weight,
                        if imaginary {
                            value.imaginary_component()
                        } else {
                            value.real_component()
                        },
                    )
                }),
            )
        }
    };
    let real = component(state.real_component(), false)?;
    if T::IS_REAL {
        Ok(T::from_real(real))
    } else {
        Ok(T::from_components(
            real,
            component(state.imaginary_component(), true)?,
        ))
    }
}

/// A large scale can restore terms lost below the normal range. Classify the
/// native reduction, including zero, before accepting the scaled result.
#[inline(always)]
fn native_affine_component<S: RealScalar>(scale: S, weighted: S, result: S) -> bool {
    weighted.is_finite()
        & result.is_finite()
        & ((scale.abs() <= S::ONE) | (weighted.abs() >= S::MIN_POSITIVE))
}

/// Whether every product has an exactly zero operand. Their sum is then an
/// exact zero that no scale can amplify, whereas a product of two nonzero
/// operands that rounds to zero or a subnormal has lost range.
#[inline]
fn exact_zero_products<S: RealScalar>(mut terms: impl Iterator<Item = (S, S)>) -> bool {
    terms.all(|(weight, value)| weight == S::ZERO || value == S::ZERO)
}

#[inline(always)]
fn native_affine<T: NumericScalar>(scale: T::Real, weighted: T, result: T) -> bool {
    native_affine_component(scale, weighted.real_component(), result.real_component())
        & (T::IS_REAL
            || native_affine_component(
                scale,
                weighted.imaginary_component(),
                result.imaginary_component(),
            ))
}

#[cold]
#[inline(never)]
fn affine_component_fallback<S: RealScalar>(
    initial: S,
    step: S,
    terms: impl Iterator<Item = (S, S)>,
) -> Result<S, SciOxideError> {
    let mut sum = scioxide_core::RangeAccumulator::new();
    sum.add(initial)?;
    sum.add_triple_products(step, terms)?;
    sum.finish()
}

/// Apply one tableau row across contiguous state storage. Native rows stream
/// through each stage vector; exceptional lanes replay their original terms.
#[inline(always)]
pub(crate) fn combine_stages<T: NumericScalar>(
    initial: Option<&[T]>,
    step: T::Real,
    weights: &[T::Real],
    stages: &[Vector<T>],
    output: &mut [T],
) -> Result<(), SciOxideError> {
    let count = weights.len().min(stages.len());
    combine_rows(
        initial,
        step,
        None,
        count,
        |j| (weights[j], stages[j].as_slice()),
        output,
        |i, old| {
            stage_affine(
                old,
                step,
                (0..count).map(|j| (weights[j], stages[j].as_slice()[i])),
            )
        },
    )
}

/// Reduce two real-weighted rows over the same stage vectors, sharing loads.
/// Exceptional lanes and short tails replay the original operands separately.
#[inline(never)]
pub(crate) fn combine_stage_pair<T: NumericScalar>(
    weights: [&[T::Real]; 2],
    stages: &[Vector<T>],
    output: &mut [T],
    mut combine: impl FnMut(T, T) -> Result<T, SciOxideError>,
) -> Result<(), SciOxideError> {
    let count = weights[0].len().min(weights[1].len()).min(stages.len());
    let length = output.len();
    let replay = |row: usize, i| {
        stage_affine(
            T::ZERO,
            T::Real::ONE,
            (0..count).map(|j| (weights[row][j], stages[j].as_slice()[i])),
        )
    };
    let end = length / 4 * 4;
    for offset in (0..end).step_by(4) {
        let mut a = [T::ZERO; 4];
        let mut b = [T::ZERO; 4];
        for j in 0..count {
            let x = weights[0][j];
            let y = weights[1][j];
            if x == T::Real::ZERO && y == T::Real::ZERO {
                continue;
            }
            let values = block4(stages[j].as_slice(), offset);
            if x != T::Real::ZERO {
                a[0] = native_add(x, values[0], a[0]);
                a[1] = native_add(x, values[1], a[1]);
                a[2] = native_add(x, values[2], a[2]);
                a[3] = native_add(x, values[3], a[3]);
            }
            if y != T::Real::ZERO {
                b[0] = native_add(y, values[0], b[0]);
                b[1] = native_add(y, values[1], b[1]);
                b[2] = native_add(y, values[2], b[2]);
                b[3] = native_add(y, values[3], b[3]);
            }
        }
        let finite = a[0].has_finite_components()
            & a[1].has_finite_components()
            & a[2].has_finite_components()
            & a[3].has_finite_components()
            & b[0].has_finite_components()
            & b[1].has_finite_components()
            & b[2].has_finite_components()
            & b[3].has_finite_components();
        if finite {
            output[offset] = combine(a[0], b[0])?;
            output[offset + 1] = combine(a[1], b[1])?;
            output[offset + 2] = combine(a[2], b[2])?;
            output[offset + 3] = combine(a[3], b[3])?;
        } else {
            for (j, slot) in output[offset..offset + 4].iter_mut().enumerate() {
                let i = offset + j;
                *slot = combine(replay(0, i)?, replay(1, i)?)?;
            }
        }
    }
    for (j, slot) in output[end..].iter_mut().enumerate() {
        let i = end + j;
        *slot = combine(replay(0, i)?, replay(1, i)?)?;
    }
    Ok(())
}

#[inline(always)]
fn native_add<T: NumericScalar>(weight: T::Real, value: T, sum: T) -> T {
    T::from_components(
        scioxide_core::fast_mul_add(weight, value.real_component(), sum.real_component()),
        if T::IS_REAL {
            T::Real::ZERO
        } else {
            scioxide_core::fast_mul_add(
                weight,
                value.imaginary_component(),
                sum.imaginary_component(),
            )
        },
    )
}

#[inline(always)]
fn native_scale<T: NumericScalar>(scale: T::Real, value: T) -> T {
    T::from_components(
        scale * value.real_component(),
        if T::IS_REAL {
            T::Real::ZERO
        } else {
            scale * value.imaginary_component()
        },
    )
}

#[inline(always)]
fn block4<T: Copy>(values: &[T], offset: usize) -> [T; 4] {
    let values = &values[offset..offset + 4];
    [values[0], values[1], values[2], values[3]]
}

/// Centering constant and origin samples.
type RowReference<'a, T> = (<T as NumericScalar>::Real, &'a [T]);

/// Four-coordinate register blocking avoids repeatedly storing each partial
/// sum. Rows, optional origin and initial state have the prepared output shape.
/// A centered row starts with `constant*origin` and subtracts that origin from
/// each sample. Exceptional coordinates and the final short block use `scalar`.
/// Every scale uses this kernel; large scales also replay zero/subnormal sums
/// unless every product of the lane has an exactly zero operand.
#[inline(always)]
pub(crate) fn combine_rows<'a, T: NumericScalar>(
    initial: Option<&[T]>,
    scale: T::Real,
    reference: Option<RowReference<'_, T>>,
    count: usize,
    term: impl Fn(usize) -> (T::Real, &'a [T]),
    output: &mut [T],
    mut scalar: impl FnMut(usize, T) -> Result<T, SciOxideError>,
) -> Result<(), SciOxideError> {
    let length = output.len();
    let mut chunks = output.chunks_exact_mut(4);
    for (block, out) in chunks.by_ref().enumerate() {
        let offset = block * 4;
        // Keep these four-lane values in registers. Generic Option/array
        // combinators can become calls returning arrays through stack storage.
        let (origin, constant) = match reference {
            Some((constant, row)) => (block4(row, offset), constant),
            None => ([T::ZERO; 4], T::Real::ZERO),
        };
        let mut sums = [
            native_scale(constant, origin[0]),
            native_scale(constant, origin[1]),
            native_scale(constant, origin[2]),
            native_scale(constant, origin[3]),
        ];
        for j in 0..count {
            let (weight, row) = term(j);
            if weight == T::Real::ZERO {
                continue;
            }
            let row = &row[offset..offset + 4];
            sums[0] = native_add(weight, row[0] - origin[0], sums[0]);
            sums[1] = native_add(weight, row[1] - origin[1], sums[1]);
            sums[2] = native_add(weight, row[2] - origin[2], sums[2]);
            sums[3] = native_add(weight, row[3] - origin[3], sums[3]);
        }
        let old = match initial {
            Some(row) => block4(row, offset),
            None => [T::ZERO; 4],
        };
        let values = [
            native_add(scale, sums[0], old[0]),
            native_add(scale, sums[1], old[1]),
            native_add(scale, sums[2], old[2]),
            native_add(scale, sums[3], old[3]),
        ];
        let ordinary = [
            native_affine(scale, sums[0], values[0]),
            native_affine(scale, sums[1], values[1]),
            native_affine(scale, sums[2], values[2]),
            native_affine(scale, sums[3], values[3]),
        ];
        if ordinary[0] & ordinary[1] & ordinary[2] & ordinary[3] {
            out.copy_from_slice(&values);
        } else {
            for (j, out) in out.iter_mut().enumerate() {
                *out = if ordinary[j]
                    || exact_zero_lane(
                        scale,
                        sums[j],
                        values[j],
                        (constant, origin[j]),
                        count,
                        &term,
                        offset + j,
                    ) {
                    values[j]
                } else {
                    scalar(offset + j, old[j])?
                };
            }
        }
    }
    let offset = length / 4 * 4;
    for (j, out) in chunks.into_remainder().iter_mut().enumerate() {
        let i = offset + j;
        *out = scalar(i, initial.map_or(T::ZERO, |v| v[i]))?;
    }
    Ok(())
}

/// Classify an exceptional row lane component by component: a component is
/// exact when it is ordinary or when its zero sum has only products with an
/// exactly zero operand, including the centered start `constant*origin`.
#[cold]
#[inline(never)]
fn exact_zero_lane<'a, T: NumericScalar>(
    scale: T::Real,
    sum: T,
    value: T,
    (constant, origin): (T::Real, T),
    count: usize,
    term: &impl Fn(usize) -> (T::Real, &'a [T]),
    index: usize,
) -> bool {
    let exact = |imaginary: bool| {
        let part = |value: T| {
            if imaginary {
                value.imaginary_component()
            } else {
                value.real_component()
            }
        };
        native_affine_component(scale, part(sum), part(value))
            || (part(sum) == T::Real::ZERO
                && part(value).is_finite()
                && exact_zero_products(core::iter::once((constant, part(origin))).chain(
                    (0..count).map(|j| {
                        let (weight, row) = term(j);
                        (weight, part(row[index] - origin))
                    }),
                )))
    };
    exact(false) && (T::IS_REAL || exact(true))
}

/// Complex stage coordinates, including for a real final state. Intermediate
/// imaginary parts may cancel across stages; round projects to the declared
/// state domain only at the final boundary. Core owns all component arithmetic.
#[derive(Clone, Copy)]
pub(crate) struct StageValue<T: NumericScalar> {
    pub(crate) value: scioxide_core::ScaledValue<scioxide_core::Complex<T::Real>>,
    marker: core::marker::PhantomData<T>,
}
impl<T: NumericScalar> StageValue<T> {
    pub(crate) fn zero() -> Self {
        Self {
            value: scioxide_core::ScaledValue::ZERO,
            marker: core::marker::PhantomData,
        }
    }
    pub(crate) fn from_value<V: NumericScalar<Real = T::Real>>(
        value: V,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            value: scioxide_core::ScaledValue::from_value(scioxide_core::Complex::new(
                value.real_component(),
                value.imaginary_component(),
            ))?,
            marker: core::marker::PhantomData,
        })
    }
    pub(crate) fn add(self, other: Self) -> Result<Self, SciOxideError> {
        Ok(Self {
            value: self.value.add(other.value)?,
            ..self
        })
    }
    pub(crate) fn sub(self, other: Self) -> Result<Self, SciOxideError> {
        Ok(Self {
            value: self.value.sub(other.value)?,
            ..self
        })
    }
    pub(crate) fn mul(self, other: Self) -> Result<Self, SciOxideError> {
        Ok(Self {
            value: self.value.mul(other.value)?,
            ..self
        })
    }
    pub(crate) fn div(self, other: Self) -> Result<Self, SciOxideError> {
        Ok(Self {
            value: self.value.div(other.value)?,
            ..self
        })
    }
    pub(crate) fn round(self) -> Result<T, SciOxideError> {
        let real = self.value.real().round()?;
        let imaginary = if matches!(
            T::kind(),
            scioxide_core::backend::ScalarKind::Complex32
                | scioxide_core::backend::ScalarKind::Complex64
        ) {
            self.value.imaginary().round()?
        } else {
            T::Real::ZERO
        };
        Ok(T::from_components(real, imaginary))
    }
}

#[inline(always)]
fn native_product_loses_range<S: RealScalar>(left: S, right: S) -> bool {
    left != S::ZERO && right != S::ZERO && !(left * right).is_normal()
}

/// Native weighted stage sum under the same ordinary-rounding contract as
/// `stage_sum`; tiny totals replay if individual products lose range.
#[inline]
pub(crate) fn native_stage_sum<S: RealScalar, I>(terms: I) -> Option<S>
where
    I: IntoIterator<Item = (S, S)>,
    I::IntoIter: Clone,
{
    let mut terms = terms.into_iter();
    let native = terms.clone().fold(S::ZERO, |sum, (weight, value)| {
        scioxide_core::fast_mul_add(weight, value, sum)
    });
    (native.is_finite()
        && (native.is_normal()
            || !terms.any(|(weight, value)| native_product_loses_range(weight, value))))
    .then_some(native)
}

/// Accumulate weighted stage values before rounding a scaled sum.
pub(crate) fn stage_sum<S: RealScalar, I>(
    terms: I,
) -> Result<scioxide_core::Scaled<S>, SciOxideError>
where
    I: IntoIterator<Item = (S, S)>,
    I::IntoIter: Clone,
{
    use scioxide_core::{Scaled, ScaledAccumulator};
    let terms = terms.into_iter();
    if let Some(native) = native_stage_sum(terms.clone()) {
        return Scaled::from_value(native);
    }
    let mut sum = ScaledAccumulator::ZERO;
    for (weight, value) in terms {
        if weight == S::ZERO || value == S::ZERO {
            continue;
        }
        if weight == S::ONE {
            sum.add_value(value)?;
        } else {
            sum.add_product(Scaled::from_value(weight)?, Scaled::from_value(value)?)?;
        }
    }
    sum.total()
}

pub(crate) fn validate_positive_finite<S: RealScalar>(
    value: S,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if !RealScalar::is_finite(value) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "step size must be finite",
        ));
    }
    if value <= <S as scioxide_core::AlgebraicScalar>::ZERO {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "step size must be positive",
        ));
    }
    Ok(())
}

pub(crate) fn validate_step_bounds<S: RealScalar>(
    minimum: Option<S>,
    maximum: Option<S>,
) -> Result<(), SciOxideError> {
    if let (Some(minimum), Some(maximum)) = (minimum, maximum)
        && minimum > maximum
    {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("step"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "min_step must not exceed max_step",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn steps_ending_within_endpoint_rounding_snap_to_the_end() {
        // 0.09 + 0.01 rounds to 0.09999999999999999, one ulp below 0.1.
        let (step, next) = select_step(
            0.09_f64,
            0.1,
            1.,
            0.01,
            None,
            Some(0.01),
            scioxide_core::argument!("s"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(next, 0.1);
        assert!((step - 0.01).abs() <= 1024. * f64::EPSILON * 0.1);
        // A remainder larger than the rounding of the endpoint is kept.
        let (_, next) = select_step(
            0.0_f64,
            0.1,
            1.,
            0.1 - 1e-12,
            None,
            None,
            scioxide_core::argument!("s"),
        )
        .unwrap()
        .unwrap();
        assert!(next < 0.1);
        // Backward steps snap symmetrically.
        let (_, next) = select_step(
            -0.09_f64,
            -0.1,
            -1.,
            0.01,
            None,
            None,
            scioxide_core::argument!("s"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(next, -0.1);
        assert!(within_endpoint_rounding(0.09999999999999999_f64, 0.1));
        assert!(!within_endpoint_rounding(0.0999999999999_f64, 0.1));
    }

    #[test]
    fn many_halved_steps_end_without_a_rounding_sized_step() {
        // Ninety-six additions of 0.1/64 reach 0.14999999999999972, 2.8e-16 short.
        let (end, h) = (0.15_f64, 0.1 / 64.0);
        let (mut time, mut steps) = (0.0_f64, 0);
        while time < end {
            let (_, next) = select_step(
                time,
                end,
                1.,
                h,
                None,
                Some(h),
                scioxide_core::argument!("s"),
            )
            .unwrap()
            .unwrap();
            time = next;
            steps += 1;
        }
        assert_eq!((steps, time), (96, end));
    }

    #[test]
    fn prepared_stage_times_preserve_checked_rounding_and_signed_zero() {
        for (time, step) in [
            (0.0_f64, 0.1),
            (-0.0, -f64::from_bits(1)),
            (-f64::MAX, f64::MAX),
            (f64::MAX, -f64::MAX),
            (1e300, 1e290),
        ] {
            checked_step_endpoint(time, step).unwrap();
            for node in [0.0, 0.1, 0.5, 0.9, 1.0] {
                assert_eq!(
                    prepared_stage_time(time, step, node).to_bits(),
                    checked_stage_time(time, step, node).unwrap().to_bits()
                );
            }
        }
        for (time, step) in [
            (0.0_f32, 0.1),
            (-0.0, -f32::from_bits(1)),
            (-f32::MAX, f32::MAX),
            (f32::MAX, -f32::MAX),
        ] {
            checked_step_endpoint(time, step).unwrap();
            for node in [0.0, 0.1, 0.5, 0.9, 1.0] {
                assert_eq!(
                    prepared_stage_time(time, step, node).to_bits(),
                    checked_stage_time(time, step, node).unwrap().to_bits()
                );
            }
        }
    }

    #[test]
    fn native_stage_overflow_replays_original_operands() {
        // Both products and the native partial sums overflow, but the complete
        // affine expression is exactly representable.
        assert_eq!(
            stage_affine(
                0.0_f64,
                1.0,
                [(2.0, f64::MAX), (-2.0, f64::MAX), (1.0, 1.0)]
            )
            .unwrap(),
            1.0
        );
        assert_eq!(
            stage_affine(
                0.0_f32,
                1.0,
                [(2.0, f32::MAX), (-2.0, f32::MAX), (1.0, 1.0)]
            )
            .unwrap(),
            1.0
        );
        let value = scioxide_core::Complex::new(f64::MAX, -f64::MAX);
        assert_eq!(
            stage_affine(
                scioxide_core::Complex::new(1.0, 2.0),
                1.0,
                [(2.0, value), (-2.0, value)]
            )
            .unwrap(),
            scioxide_core::Complex::new(1.0, 2.0)
        );
        assert!(stage_affine(0.0_f64, 1.0, [(2.0, f64::MAX)]).is_err());
    }

    #[test]
    fn paired_rows_replay_overflow_in_blocks_and_tails() {
        for dimension in [1, 4, 7] {
            let stages = [
                Vector::from_vec(alloc::vec![f64::MAX; dimension]),
                Vector::from_vec(alloc::vec![f64::MAX; dimension]),
                Vector::from_vec(alloc::vec![1.0; dimension]),
            ];
            let mut first = alloc::vec![0.0; dimension];
            let mut second = alloc::vec![0.0; dimension];
            let weights = [&[2.0, -2.0, 1.0][..], &[-2.0, 2.0, 2.0][..]];
            combine_stage_pair(weights, &stages, &mut first, |a, _| Ok(a)).unwrap();
            combine_stage_pair(weights, &stages, &mut second, |_, b| Ok(b)).unwrap();
            assert!(first.iter().all(|x| (x - 1.0).abs() <= 8.0 * f64::EPSILON));
            assert!(second.iter().all(|x| (x - 2.0).abs() <= 8.0 * f64::EPSILON));
        }
    }

    #[test]
    fn native_stage_preserves_tiny_time_weight_factors() {
        let tiny = f64::from_bits(1);
        let expected = (tiny * f64::MAX) * 0.5;
        let actual = stage_affine(0.0, tiny, [(0.5, f64::MAX)]).unwrap();
        assert!((actual - expected).abs() <= f64::EPSILON * expected);
        let tiny = f32::from_bits(1);
        let expected = (tiny * f32::MAX) * 0.5;
        let actual = stage_affine(0.0, tiny, [(0.5, f32::MAX)]).unwrap();
        assert!((actual - expected).abs() <= f32::EPSILON * expected);

        for dimension in [1, 4, 7] {
            let stages = [Vector::from_vec(alloc::vec![f64::MAX; dimension])];
            let mut output = alloc::vec![0.0; dimension];
            combine_stages(None, f64::from_bits(1), &[0.5], &stages, &mut output).unwrap();
            let expected = (f64::from_bits(1) * f64::MAX) * 0.5;
            assert!(
                output
                    .iter()
                    .all(|&x| (x - expected).abs() <= f64::EPSILON * expected)
            );
        }
    }

    #[test]
    fn large_steps_restore_zero_and_subnormal_weighted_lanes() {
        fn check<T: NumericScalar>() {
            let half = T::Real::ONE / T::Real::TWO;
            for dimension in [1, 4, 7, 8] {
                // The tiny slopes round to zero or to an inaccurate subnormal
                // when weighted. Normal lanes share the same block and tail.
                let slopes: alloc::vec::Vec<_> = (0..dimension)
                    .map(|i| match i % 3 {
                        0 => T::Real::MIN_SUBNORMAL,
                        1 => T::Real::MIN_SUBNORMAL * (T::Real::TWO + T::Real::ONE),
                        _ => T::Real::ONE,
                    })
                    .collect();
                let stages = [Vector::from_vec(
                    slopes.iter().map(|&x| T::from_components(x, -x)).collect(),
                )];
                for step in [T::Real::MAX, -T::Real::MAX] {
                    let mut output = alloc::vec![T::ZERO; dimension];
                    combine_stages(None, step, &[half], &stages, &mut output).unwrap();
                    for (i, (&slope, value)) in slopes.iter().zip(&output).enumerate() {
                        let expected = (slope * step) * half;
                        let tolerance = T::Real::EPSILON * expected.abs() * T::Real::TWO;
                        let scalar =
                            stage_affine(T::ZERO, step, [(half, stages[0].as_slice()[i])]).unwrap();
                        for actual in [*value, scalar] {
                            assert!((actual.real_component() - expected).abs() <= tolerance);
                            if !T::IS_REAL {
                                assert!(
                                    (actual.imaginary_component() + expected).abs() <= tolerance
                                );
                            }
                        }
                    }
                }
            }
        }
        check::<f32>();
        check::<f64>();
        check::<scioxide_core::Complex<f32>>();
        check::<scioxide_core::Complex<f64>>();
    }

    #[test]
    fn ordinary_blocks_keep_native_combinations_at_every_step_scale() {
        let row = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let initial = [2.0; 8];
        for step in [-1e6, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 1e6] {
            let mut output = [0.0; 8];
            let mut replays = 0;
            combine_rows(
                Some(&initial),
                step,
                None,
                1,
                |_| (0.5, &row),
                &mut output,
                |i, old| {
                    replays += 1;
                    stage_affine(old, step, [(0.5, row[i])])
                },
            )
            .unwrap();
            assert_eq!(replays, 0);
            for (&actual, slope) in output.iter().zip(row) {
                let expected = 2.0 + step * (0.5 * slope);
                assert!((actual - expected).abs() <= f64::EPSILON * expected.abs());
            }
        }
    }

    #[test]
    fn mixed_blocks_replay_only_exceptional_lanes_and_the_tail() {
        use scioxide_core::Complex;
        let tiny = f64::from_bits(1);
        let row = [
            Complex::new(1.0, 2.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, tiny),
            Complex::new(tiny, 1.0),
            Complex::new(4.0, 8.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 4.0),
        ];
        for step in [2.0, -2.0] {
            let mut output = [Complex::ZERO; 7];
            let mut replays = alloc::vec::Vec::new();
            combine_rows(
                None,
                step,
                None,
                1,
                |_| (0.5, &row),
                &mut output,
                |i, old| {
                    replays.push(i);
                    stage_affine(old, step, [(0.5, row[i])])
                },
            )
            .unwrap();
            // Lane 1 sums exact zeros; lanes 2 and 3 lose a tiny product to underflow.
            assert_eq!(replays, [2, 3, 4, 5, 6]);
            for (&actual, slope) in output.iter().zip(row) {
                let expected = slope * (0.5 * step);
                assert!(
                    (actual.real_component() - expected.real_component()).abs()
                        <= f64::EPSILON * expected.real_component().abs()
                );
                assert!(
                    (actual.imaginary_component() - expected.imaginary_component()).abs()
                        <= f64::EPSILON * expected.imaginary_component().abs()
                );
            }
        }
    }

    #[test]
    fn exact_zero_products_stay_native_at_large_steps() {
        use scioxide_core::{Complex, fast_mul_add};
        let naive = |initial: f64, step: f64, terms: &[(f64, f64)]| {
            let weighted = terms
                .iter()
                .fold(0.0, |sum, &(w, v)| fast_mul_add(w, v, sum));
            fast_mul_add(step, weighted, initial)
        };
        let tiny = f64::from_bits(1);
        for step in [10.0, -1e3, 1e300] {
            let terms = [(0.5_f64, 0.0_f64), (0.0, 3.0), (-0.25, -0.0)];
            for initial in [1.5, 0.0, -0.0, -2e-310] {
                let actual = stage_affine(initial, step, terms).unwrap();
                assert_eq!(actual.to_bits(), naive(initial, step, &terms).to_bits());
            }
            // Lanes 0 and 2 have identically zero slopes, lane 4 is the tail.
            let rows = [[0.0_f64, 1.0, 0.0, -2.0, 0.0], [0.0, 3.0, -0.0, 0.5, 4.0]];
            let (weights, initial) = ([0.25, -0.75], [1.0, 2.0, -3.0, 4.0, 5.0]);
            let mut output = [0.0_f64; 5];
            let mut replays = alloc::vec::Vec::new();
            let lane = |i: usize| [(weights[0], rows[0][i]), (weights[1], rows[1][i])];
            combine_rows(
                Some(&initial),
                step,
                None,
                2,
                |j| (weights[j], &rows[j][..]),
                &mut output,
                |i, old| {
                    replays.push(i);
                    stage_affine(old, step, lane(i))
                },
            )
            .unwrap();
            assert_eq!(replays, [4]);
            for (i, actual) in output.iter().enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    naive(initial[i], step, &lane(i)).to_bits()
                );
            }
            // Component-wise: exact zero parts beside ordinary parts, and a
            // centered start whose origin is zero.
            let rows = [
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(-0.0, 2.0),
                Complex::new(0.0, 0.0),
            ];
            let origin = [Complex::new(0.0, 0.0); 4];
            let mut output = [Complex::ZERO; 4];
            let mut replays = 0;
            combine_rows(
                Some(&rows),
                step,
                Some((0.5, &origin)),
                1,
                |_| (0.25, &rows[..]),
                &mut output,
                |_, _| {
                    replays += 1;
                    Ok(Complex::ZERO)
                },
            )
            .unwrap();
            assert_eq!(replays, 0);
            for (actual, &row) in output.iter().zip(&rows) {
                for imaginary in [false, true] {
                    let part = |v: Complex<f64>| {
                        if imaginary {
                            v.imaginary_component()
                        } else {
                            v.real_component()
                        }
                    };
                    let expected = naive(part(row), step, &[(0.25, part(row))]);
                    assert_eq!(part(*actual).to_bits(), expected.to_bits());
                }
            }
            // A product of nonzero operands that underflows still replays.
            let rows = [tiny, 1.0, 1.0, 1.0];
            let mut output = [0.0; 4];
            let mut replays = alloc::vec::Vec::new();
            combine_rows(
                None,
                step,
                None,
                1,
                |_| (0.5, &rows[..]),
                &mut output,
                |i, old| {
                    replays.push(i);
                    stage_affine(old, step, [(0.5, rows[i])])
                },
            )
            .unwrap();
            assert_eq!(replays, [0]);
            let expected = (tiny * step) * 0.5;
            assert!((output[0] - expected).abs() <= f64::EPSILON * expected.abs());
            let actual = stage_affine(0.0, step, [(0.5, tiny), (0.0, 1.0)]).unwrap();
            assert!((actual - expected).abs() <= f64::EPSILON * expected.abs());
        }
    }

    #[test]
    fn large_step_blocks_recover_overflow_and_reject_unrepresentable_results() {
        for dimension in [1, 4, 7] {
            let stages = [
                Vector::from_vec(alloc::vec![f64::MAX; dimension]),
                Vector::from_vec(alloc::vec![f64::MAX; dimension]),
                Vector::from_vec(alloc::vec![1.0; dimension]),
            ];
            for step in [2.0, -2.0] {
                let mut output = alloc::vec![0.0; dimension];
                combine_stages(None, step, &[2.0, -2.0, 1.0], &stages, &mut output).unwrap();
                assert!(
                    output
                        .iter()
                        .all(|&x| (x - step).abs() <= f64::EPSILON * step.abs())
                );
                let error = combine_stages(None, step, &[2.0], &stages, &mut output).unwrap_err();
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    }
                ));
            }
        }
    }

    #[test]
    fn scaled_forcing_replays_products_before_step_amplifies_underflow() {
        fn check<S: RealScalar + NumericScalar<Real = S>>(tiny: S, large: S) {
            let half = S::ONE / S::TWO;
            let expected = (tiny * large) * half;
            let factor = scioxide_core::Scaled::from_value(large).unwrap();
            let tolerance = (S::TWO + S::TWO) * S::EPSILON * expected;
            let scalar = stage_sum([(half, tiny)])
                .unwrap()
                .mul(factor)
                .unwrap()
                .round()
                .unwrap();
            assert!((scalar - expected).abs() <= tolerance);
            let real = stage_affine(S::ZERO, large, [(half, tiny)]).unwrap();
            assert!((real - expected).abs() <= tolerance);
            let complex = stage_affine(
                scioxide_core::Complex::new(S::ZERO, S::ZERO),
                large,
                [(half, scioxide_core::Complex::new(tiny, tiny))],
            )
            .unwrap();
            assert!((complex.real_component() - expected).abs() <= tolerance);
            assert!((complex.imaginary_component() - expected).abs() <= tolerance);
        }
        check(f64::from_bits(1), f64::MAX);
        check(f32::from_bits(1), f32::MAX);
    }

    #[test]
    fn mass_residual_retains_real_and_complex_cancellation() {
        assert_eq!(
            stage_residual(
                &[1_f64; 3],
                1.,
                &[1e300, 1., -1e300],
                &[0.; 3],
                1.0,
                [],
                0.0,
            )
            .unwrap(),
            1.
        );
        use scioxide_core::Complex;
        let large = Complex::new(1e300, 1e300);
        let mass = [
            Complex::new(1., 1.),
            Complex::new(1., -1.),
            Complex::new(-1., -1.),
        ];
        let state = [large, Complex::new(1., 2.), large];
        let zero = [Complex::new(0., 0.); 3];
        assert_eq!(
            stage_residual(&mass, 1., &state, &zero, 1.0, [], Complex::ZERO).unwrap(),
            Complex::new(3., 1.)
        );
        let large = Complex::new(f64::MAX, f64::MAX);
        assert_eq!(
            stage_residual(
                &[Complex::new(0.5, 0.5)],
                2.,
                &[large],
                &[large],
                1.0,
                [],
                Complex::ZERO,
            )
            .unwrap(),
            Complex::new(0., f64::MAX)
        );
    }

    #[test]
    fn scaled_norms_remain_finite_when_raw_squares_overflow() {
        let magnitude = f64::MAX / 4.0;
        let norm = scaled_rms_norm(&[magnitude, magnitude]).unwrap();
        assert!(norm.is_finite());
        assert_eq!(norm, magnitude);

        let tolerance = scioxide_core::ErrorTolerance::scalar(
            1.0e-12,
            1.0e-8,
            scioxide_core::NormSelection::RootMeanSquare,
        )
        .unwrap();
        let previous = [1.0e300_f64];
        let accuracy = crate::StageAccuracy::new(&previous, &tolerance, 20);
        let difference = accuracy.correction_norm(&previous, &[1.0e290]).unwrap();
        assert!(difference.is_finite() && difference < 1.0);
    }
}
