//! Batch transforms, softmax normalization and log-sum-exp reductions.
//!
//! A `−∞` logit is a masked entry: it contributes `e^(−∞) = 0`. NaN and `+∞`
//! logits, and non-finite weights and angles, are rejected even when another
//! component would make the term zero.
mod owned;
use crate::backend::PartitionSlot;
use crate::logarithmic::{
    CertificationTarget, Logit, logit, positive_inputs, positive_sum, working_sum,
};
use crate::size::validate_same_len;
use crate::stable::output;
use crate::{
    BatchLayout, Dimensions, ExecutionContext, ExecutionPolicy, LogValue, NdView, NdViewMut,
    NumericScalar, RealScalar, SciOxideError,
};
use core::num::NonZeroUsize;
pub use owned::{
    log_softmax, log_softmax_axis, logsumexp_axis, softmax, softmax_axis, weighted_logsumexp_axis,
};

// Output of a softmax normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Normalization {
    // Probabilities `pᵢ`; masked entries are `+0`.
    Probability,
    // Log probabilities `ln pᵢ`; masked entries are `−∞`.
    Log,
}

fn storage(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if actual >= expected {
        Ok(())
    } else {
        validate_same_len(expected, actual)
    }
}

// Shape and staging length were checked before dispatch. Publication cannot
// fail and performs only the one copy required for transactional output.
fn publish<S: Copy>(values: &[S], target: &mut NdViewMut<'_, '_, S>) {
    if let Some(slots) = target.contiguous_mut() {
        slots.copy_from_slice(values);
    } else {
        let mut index = 0;
        target.for_each_mut(|slot| {
            *slot = values[index];
            index += 1;
        });
    }
}

// Fixed partition admission storage is bounded and never allocated. Each slot exclusively
// owns its disjoint staging rows, through the shared core protocol.

fn stage_partitions<S: Send>(
    scratch: &mut [S],
    width: usize,
    count: usize,
    policy: ExecutionPolicy,
    context: ExecutionContext<'_>,
    work_per_item: NonZeroUsize,
    apply: impl Fn(core::ops::Range<usize>, &mut [S], ExecutionContext<'_>) -> Result<(), SciOxideError>
    + Sync,
) -> Result<(), SciOxideError> {
    context.check()?;
    // At most 64 partitions, so the slot array below is fixed-size. Each slot
    // receives the contiguous block of scratch rows matching its partition
    // range. Results go to scratch only, so callers publish to the output
    // after the whole batch has succeeded.
    let plan = context.plan(

        policy,
        count,
        crate::PlanLimits {
            work_per_item,
            auto_threshold: const { NonZeroUsize::new(32).expect("resource limit must be positive") },
            worker_limit: const { NonZeroUsize::new(64).expect("resource limit must be positive") },
        },
    );
    let mut remaining = scratch;
    let mut partitions = plan.partitions();
    let slots: [_; 64] = core::array::from_fn(|_| {
        partitions
            .next()
            .map_or_else(PartitionSlot::empty, |partition| {
                let size = partition.range().len() * width;
                let (current, rest) = core::mem::take(&mut remaining).split_at_mut(size);
                remaining = rest;
                PartitionSlot::new(current)
            })
    });
    context.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, data, child| apply(partition.range(), data, child),
    )?;
    context.check()
}

fn rows<S: Send>(
    scratch: &mut [S],
    width: usize,
    count: usize,
    policy: ExecutionPolicy,
    context: ExecutionContext<'_>,
    work_per_item: NonZeroUsize,
    apply: impl Fn(usize, &mut [S]) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    stage_partitions(
        scratch,
        width,
        count,
        policy,
        context,
        work_per_item,
        |range, data, child| {
            if width == 0 {
                for index in range {
                    child.check()?;
                    apply(index, &mut [])?;
                }
            } else {
                for (index, row) in range.zip(data.chunks_exact_mut(width)) {
                    child.check()?;
                    apply(index, row)?;
                }
            }
            Ok(())
        },
    )
}

/// Applies a scalar function to every element of a strided view.
///
/// `output` must have the same shape as `input`, and `scratch` must hold at least
/// one value per element. The input may use a broadcast layout; the output
/// layout must not have repeated addresses. The function runs in parallel according to `policy`
/// (see [execution contexts](crate#execution-contexts)); with `Auto`, it estimates
/// one work unit per element and goes parallel from 32 elements. Results do not
/// depend on scheduling. Cancellation and deadlines are checked at least every 256
/// calls per partition, and a call that is already running is not interrupted.
/// On any error, `output` is unchanged.
///
/// # Errors
///
/// - `DimensionMismatch` if the shapes differ or `scratch` is too short.
/// - `NonFinite` (intermediate) if `transform` returns NaN or infinity.
/// - Any error returned by `transform`.
/// - `ExecutionStopped` on cancellation or deadline, and `ExecutionFailure` if
///   the executor fails.
pub fn transform_into<S: NumericScalar>(
    input: NdView<'_, '_, S>,
    transform: impl Fn(S) -> Result<S, SciOxideError> + Sync,
    policy: ExecutionPolicy,
    mut output: NdViewMut<'_, '_, S>,
    scratch: &mut [S],
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    const POLL_ITEMS: usize = 256;
    if input.layout().shape().extents() != output.as_view().layout().shape().extents() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::from(input.layout().shape().extents()),
            Dimensions::from(output.as_view().layout().shape().extents()),
        ));
    }
    let n = input.layout().shape().len();
    storage(n, scratch.len())?;
    let data = input.storage();
    let contiguous = input.layout().is_contiguous();
    let checked = |value| {
        let value = transform(value)?;
        if !value.has_finite_components() {
            return Err(SciOxideError::non_finite(
                crate::NonFiniteOrigin::Intermediate,
                "non-finite transform output",
            ));
        }
        Ok(value)
    };
    stage_partitions(
        &mut scratch[..n],
        1,
        n,
        policy,
        cx,
        NonZeroUsize::MIN,
        |range, staged, child| {
            if contiguous {
                for (values, slots) in data[range]
                    .chunks(POLL_ITEMS)
                    .zip(staged.chunks_mut(POLL_ITEMS))
                {
                    child.check()?;
                    for (&value, slot) in values.iter().zip(slots) {
                        *slot = checked(value)?;
                    }
                }
            } else {
                for (chunk, slots) in staged.chunks_mut(POLL_ITEMS).enumerate() {
                    child.check()?;
                    let start = range.start + chunk * POLL_ITEMS;
                    for (index, slot) in (start..).zip(slots) {
                        *slot = checked(data[input.layout().linear_offset(index, None)])?;
                    }
                }
            }
            Ok(())
        },
    )?;
    publish(&scratch[..n], &mut output);
    Ok(())
}

// Its logits are `input`.
fn normalize<S: RealScalar>(
    values: impl Iterator<Item = S> + Clone,
    result: &mut [S],
    normalization: Normalization,
) -> Result<(), SciOxideError> {
    let inputs = positive_inputs(values.clone())?;
    if inputs.active > 1 {
        let working = match normalization {
            Normalization::Probability => {
                let mut slots = result.iter_mut();
                working_sum(values.clone(), inputs, |term| {
                    if let Some(slot) = slots.next() {
                        *slot = S::try_from_f64_lossy(term)?;
                    }
                    Ok(())
                })?
            }
            Normalization::Log => working_sum(values.clone(), inputs, |_| Ok(()))?,
        };
        if working.certifies::<S>(CertificationTarget::Normalization) {
            match normalization {
                Normalization::Probability => {
                    let inverse = 1.0 / working.sum;
                    for slot in result {
                        *slot = S::try_from_f64_lossy(slot.into_f64() * inverse)?;
                    }
                }
                Normalization::Log => {
                    for (x, slot) in values.zip(result) {
                        *slot = if x == S::NEG_INFINITY {
                            S::NEG_INFINITY
                        } else {
                            let value = (x.into_f64() - inputs.max) - working.shifted_log;
                            output(if value == 0.0 { -0.0 } else { value })?
                        };
                    }
                }
            }
            return Ok(());
        }
    }
    // softmax_i = e^(x_i - max) / sum_j e^(x_j - max). Shifting by the maximum
    // keeps every exponential in (0, 1], and `log_sum` is the logarithm of the
    // shifted sum.
    let reduction = positive_sum(values.clone(), CertificationTarget::Normalization)?
        .nonzero(crate::argument!("input"))?;
    let max = reduction.anchor;
    let log_sum = reduction.shifted_log();
    let sum = RealScalar::exp(log_sum);
    // Every active log probability is strictly negative when two terms exist,
    // even when a dominant lane's correction underflows. A singleton is exact +0.
    let multiple = values
        .clone()
        .filter(|&x| x != S::NEG_INFINITY)
        .take(2)
        .count()
        == 2;
    for (x, out) in values.zip(result) {
        *out = match logit(x)? {
            Logit::Masked => match normalization {
                Normalization::Probability => S::ZERO,
                Normalization::Log => S::NEG_INFINITY,
            },
            Logit::Finite(x) => {
                let shifted = x - max;
                output(match normalization {
                    Normalization::Log => {
                        let value = shifted - log_sum;
                        if value == 0.0 && multiple {
                            -0.0
                        } else {
                            value
                        }
                    }
                    Normalization::Probability => RealScalar::exp(shifted) / sum,
                })?
            }
        };
    }
    Ok(())
}

fn axis_normalize<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axis: usize,
    mut target: NdViewMut<'_, '_, S>,
    scratch: &mut [S],
    normalization: Normalization,
    policy: ExecutionPolicy,
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if input.layout().shape().extents() != target.as_view().layout().shape().extents() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::from(input.layout().shape().extents()),
            Dimensions::from(target.as_view().layout().shape().extents()),
        ));
    }
    let batches = input.layout().axis(axis)?;
    let shape = input.layout().shape();
    let width = shape.extents()[axis];
    let n = shape.len();
    storage(n, scratch.len())?;
    let data = input.storage();
    rows(
        &mut scratch[..n],
        width,
        batches.len(),
        policy,
        cx,
        NonZeroUsize::new(width).unwrap_or(NonZeroUsize::MIN),
        |index, row| {
            let lane = batches.lane(index)?;
            if lane.stride() == 1 {
                normalize(
                    data[lane.start()..lane.start() + lane.len()]
                        .iter()
                        .copied(),
                    row,
                    normalization,
                )
            } else {
                normalize(lane.offsets().map(|i| data[i]), row, normalization)
            }
        },
    )?;
    if n == 0 {
        return Ok(());
    }
    // Scratch holds one contiguous row of `width` values per lane, while the
    // output is visited in row-major order. With extents viewed as
    // (outer, width, inner), the row-major index maps to lane
    // outer * inner + inner_index and coordinate (index / inner) % width, which
    // matches the lane order of `layout.axis`.
    let inner = shape.extents()[axis + 1..].iter().product::<usize>();
    if inner == 1 {
        publish(&scratch[..n], &mut target);
        return Ok(());
    }
    let mut index = 0;
    target.for_each_mut(|x| {
        let outer = index / (width * inner);
        let lane = outer * inner + index % inner;
        let coordinate = (index / inner) % width;
        *x = scratch[lane * width + coordinate];
        index += 1;
    });
    Ok(())
}

/// Computes the softmax `pᵢ = e^(xᵢ − m) / Σⱼ e^(xⱼ − m)`, `m = maxⱼ xⱼ`, into `output`.
///
/// `scratch` must hold at least `input.len()` values. Masked entries (`−∞`)
/// receive `+0`. Takes O(n) time.
///
/// # Errors
///
/// - `DimensionMismatch` if `output` and `input` differ in length, or `scratch`
///   is too short.
/// - `NonFinite` if an input is NaN or `+∞`.
/// - `DomainError` if the input is empty or every entry is masked.
/// - `NumericalInstability` if a result overflows.
pub fn softmax_into<S: RealScalar>(
    input: &[S],
    output: &mut [S],
    scratch: &mut [S],
) -> Result<(), SciOxideError> {
    validate_same_len(input.len(), output.len())?;
    storage(input.len(), scratch.len())?;
    normalize(
        input.iter().copied(),
        &mut scratch[..input.len()],
        Normalization::Probability,
    )?;
    output.copy_from_slice(&scratch[..input.len()]);
    Ok(())
}

/// Computes the log-softmax `ln pᵢ = xᵢ − ln Σⱼ e^(xⱼ)` into `output`.
///
/// `scratch` must hold at least `input.len()` values. Masked entries (`−∞`)
/// receive `−∞`. Takes O(n) time.
///
/// # Errors
///
/// - `DimensionMismatch` if `output` and `input` differ in length, or `scratch`
///   is too short.
/// - `NonFinite` if an input is NaN or `+∞`.
/// - `DomainError` if the input is empty or every entry is masked.
/// - `NumericalInstability` if a result overflows.
pub fn log_softmax_into<S: RealScalar>(
    input: &[S],
    output: &mut [S],
    scratch: &mut [S],
) -> Result<(), SciOxideError> {
    validate_same_len(input.len(), output.len())?;
    storage(input.len(), scratch.len())?;
    normalize(
        input.iter().copied(),
        &mut scratch[..input.len()],
        Normalization::Log,
    )?;
    output.copy_from_slice(&scratch[..input.len()]);
    Ok(())
}

/// Computes the softmax of every lane of `input` along `axis` into `output`.
///
/// Each lane `x` along `axis` is normalized independently, as in
/// [`softmax`](crate::softmax):
///
/// ```text
/// pᵢ = e^(xᵢ − m) / Σⱼ e^(xⱼ − m),   m = maxⱼ xⱼ
/// ```
///
/// A `−∞` logit is a masked entry and receives `+0`. The lanes are those of
/// [`StridedLayout::axis`](crate::StridedLayout::axis); an array with no lanes
/// writes nothing. `input` may use a broadcast layout. `output` has the extents
/// of `input` and a layout without repeated addresses, and `scratch` holds at
/// least `input.layout().shape().len()` values. Each lane is computed the same
/// way under every `policy`, so the result does not depend on it.
///
/// With [`ExecutionPolicy::Auto`], lanes run in parallel from 32 lanes on, in
/// at most 64 partitions (see [execution contexts](crate#execution-contexts)).
/// Cancellation and deadlines are checked before every lane, so each partition
/// finishes at most one more lane after a stop request. Takes O(n) time for `n`
/// elements. On error, `output` is unchanged.
///
/// # Errors
///
/// - `DimensionMismatch` if the extents of `output` differ from those of
///   `input`, or `scratch` is too short.
/// - `OutOfBounds` if `axis` is not an axis of `input`.
/// - `SizeOverflow` if the number of lanes overflows `usize`.
/// - `DomainError` if a lane is empty or fully masked.
/// - `NonFinite` if an input is NaN or `+∞`.
/// - `NumericalInstability` if a result overflows.
/// - `ExecutionStopped` on cancellation or deadline, `ExecutionFailure` if the
///   executor fails, and `CapabilityFailure` if the clock fails.
pub fn softmax_axis_into<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axis: usize,
    policy: ExecutionPolicy,
    output: NdViewMut<'_, '_, S>,
    scratch: &mut [S],
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    axis_normalize(
        input,
        axis,
        output,
        scratch,
        Normalization::Probability,
        policy,
        cx,
    )
}

/// Computes the log-softmax of every lane of `input` along `axis` into `output`.
///
/// Each lane `x` along `axis` receives `ln pᵢ = xᵢ − m − ln Σⱼ e^(xⱼ − m)`,
/// `m = maxⱼ xⱼ`, as in [`log_softmax`](crate::log_softmax). A `−∞` logit is a
/// masked entry and receives `−∞`. Lanes, storage, execution and errors are
/// those of [`softmax_axis_into`].
///
/// # Errors
///
/// The errors of [`softmax_axis_into`].
pub fn log_softmax_axis_into<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axis: usize,
    policy: ExecutionPolicy,
    output: NdViewMut<'_, '_, S>,
    scratch: &mut [S],
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    axis_normalize(input, axis, output, scratch, Normalization::Log, policy, cx)
}

// Work estimate of one lane along `axis`, for the `Auto` threshold.
fn lane_work(batches: BatchLayout<'_>, extents: &[usize]) -> NonZeroUsize {
    NonZeroUsize::new(extents[batches.axis()]).unwrap_or(NonZeroUsize::MIN)
}

// Writes the log-sum-exp of lane `i` to `staged[i]`; `staged` holds one value
// per lane of `batches`, which are lanes of `input`.
fn stage_logsumexp<T: NumericScalar>(
    input: NdView<'_, '_, T>,
    batches: BatchLayout<'_>,
    staged: &mut [T],
    policy: ExecutionPolicy,
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let data = input.storage();
    rows(
        staged,
        1,
        batches.len(),
        policy,
        cx,
        lane_work(batches, input.layout().shape().extents()),
        |index, row| {
            let lane = batches.lane(index)?;
            let value = |j| data[lane.start() + j * lane.stride()];
            row[0] = if T::IS_REAL {
                let sum = positive_sum(
                    lane.offsets().map(|i| data[i].real_component()),
                    CertificationTarget::Logarithm,
                )?;
                T::from_real(sum.logarithm()?)
            } else {
                let sum = crate::logarithmic::weighted_components(
                    lane.len(),
                    |j| value(j).real_component(),
                    |j| {
                        crate::logarithmic::LogWeight::UnitPhase(
                            value(j).imaginary_component().into_f64(),
                        )
                    },
                )?;
                let value = sum.principal(crate::argument!("input"))?;
                T::from_components(value.re, value.im)
            };
            Ok(())
        },
    )
}

/// Computes the log-sum-exp of every lane of `input` along `axis` into `output`.
///
/// Each lane `x` along `axis` is reduced as in [`logsumexp`](crate::logsumexp),
/// with its accuracy:
///
/// ```text
/// s = ln Σᵢ e^(xᵢ)
/// ```
///
/// A `−∞` logit is a masked entry and contributes `0`; an empty or fully masked
/// lane gives `−∞` (`(−∞, +0)` for complex values). Complex logits give the
/// principal logarithm, and their imaginary parts must be finite, also on
/// masked terms. The output values follow the lanes of
/// [`StridedLayout::axis`](crate::StridedLayout::axis), in row-major order of
/// the remaining axes. `input` may use a broadcast layout. `output` has the
/// extents of `input` without `axis`, and `scratch` holds at least
/// `input.layout().axis(axis)?.len()` values. Each lane is reduced in the same
/// order under every `policy`, so the result does not depend on it.
///
/// With [`ExecutionPolicy::Auto`], lanes run in parallel from 32 lanes on, in
/// at most 64 partitions (see [execution contexts](crate#execution-contexts)).
/// Cancellation and deadlines are checked before every lane, so each partition
/// finishes at most one more lane after a stop request. On error, `output` is
/// unchanged.
///
/// # Errors
///
/// - `DimensionMismatch` if the extents of `output` differ from those of
///   `input` without `axis`, or `scratch` is too short.
/// - `OutOfBounds` if `axis` is not an axis of `input`.
/// - `SizeOverflow` if the number of lanes overflows `usize`.
/// - `NonFinite` if a logit is NaN or `+∞`, or an imaginary part is not finite.
/// - `DomainError` if the terms of a complex lane cancel exactly.
/// - `NumericalInstability` if a result overflows.
/// - `ExecutionStopped` on cancellation or deadline, `ExecutionFailure` if the
///   executor fails, and `CapabilityFailure` if the clock fails.
pub fn logsumexp_axis_into<T: NumericScalar>(
    input: NdView<'_, '_, T>,
    axis: usize,
    policy: ExecutionPolicy,
    mut output: NdViewMut<'_, '_, T>,
    scratch: &mut [T],
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let batches = input.layout().axis(axis)?;
    reduced_shape(
        input.layout().shape().extents(),
        axis,
        output.as_view().layout().shape().extents(),
    )?;
    storage(batches.len(), scratch.len())?;
    let staged = &mut scratch[..batches.len()];
    stage_logsumexp(input, batches, staged, policy, cx)?;
    publish(staged, &mut output);
    Ok(())
}

// Lanes of `input` and `weights` along `axis`, after checking that both have
// the same extents.
fn weighted_lanes<'a, 'b, T: NumericScalar>(
    input: NdView<'_, 'a, T::Real>,
    weights: NdView<'_, 'b, T>,
    axis: usize,
) -> Result<(BatchLayout<'a>, BatchLayout<'b>), SciOxideError> {
    let batches = input.layout().axis(axis)?;
    if input.layout().shape().extents() != weights.layout().shape().extents() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::from(input.layout().shape().extents()),
            Dimensions::from(weights.layout().shape().extents()),
        ));
    }
    Ok((batches, weights.layout().axis(axis)?))
}

// Writes the weighted log-sum-exp of lane `i` to `staged[i]`; `lanes` comes
// from `weighted_lanes`.
fn stage_weighted_logsumexp<T: NumericScalar>(
    input: NdView<'_, '_, T::Real>,
    weights: NdView<'_, '_, T>,
    (batches, weight_batches): (BatchLayout<'_>, BatchLayout<'_>),
    staged: &mut [LogValue<T, T::Real>],
    policy: ExecutionPolicy,
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    rows(
        staged,
        1,
        batches.len(),
        policy,
        cx,
        lane_work(batches, input.layout().shape().extents()),
        |i, row| {
            let x = batches.lane(i)?;
            let w = weight_batches.lane(i)?;
            row[0] = crate::logarithmic::weighted_value(
                x.len(),
                |j| input.storage()[x.start() + j * x.stride()],
                |j| weights.storage()[w.start() + j * w.stride()],
            )?;
            Ok(())
        },
    )
}

/// Computes the weighted log-sum-exp of every lane along `axis` into `output`.
///
/// For real logits `x` and real or complex weights `w` of the same extents,
/// each lane along `axis` is reduced as in
/// [`weighted_logsumexp`](crate::weighted_logsumexp), with its accuracy:
///
/// ```text
/// s = ln Σᵢ wᵢ e^(xᵢ)
/// ```
///
/// The result is a [`LogValue`], which represents an exact zero sum, such as
/// an empty or fully masked lane or exact cancellation, without an infinity.
/// Weights must be finite, also on masked terms. `weights` may use other
/// strides than `input`, including a broadcast layout. Lanes, storage and
/// execution are those of [`logsumexp_axis_into`], with `scratch` counting log
/// values.
///
/// # Errors
///
/// - `DimensionMismatch` if the extents of `weights` differ from those of
///   `input`, the extents of `output` differ from those of `input` without
///   `axis`, or `scratch` is too short.
/// - `OutOfBounds` if `axis` is not an axis of `input`.
/// - `SizeOverflow` if the number of lanes overflows `usize`.
/// - `NonFinite` if a logit is NaN or `+∞`, or a weight is not finite.
/// - `NumericalInstability` if a result overflows.
/// - `ExecutionStopped` on cancellation or deadline, `ExecutionFailure` if the
///   executor fails, and `CapabilityFailure` if the clock fails.
pub fn weighted_logsumexp_axis_into<T: NumericScalar>(
    input: NdView<'_, '_, T::Real>,
    weights: NdView<'_, '_, T>,
    axis: usize,
    policy: ExecutionPolicy,
    mut output: NdViewMut<'_, '_, LogValue<T, T::Real>>,
    scratch: &mut [LogValue<T, T::Real>],
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let lanes = weighted_lanes(input, weights, axis)?;
    reduced_shape(
        input.layout().shape().extents(),
        axis,
        output.as_view().layout().shape().extents(),
    )?;
    storage(lanes.0.len(), scratch.len())?;
    let staged = &mut scratch[..lanes.0.len()];
    stage_weighted_logsumexp(input, weights, lanes, staged, policy, cx)?;
    publish(staged, &mut output);
    Ok(())
}

fn reduced_shape(input: &[usize], axis: usize, actual: &[usize]) -> Result<(), SciOxideError> {
    let expected = input
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| (i != axis).then_some(n));
    if expected.clone().eq(actual.iter().copied()) {
        Ok(())
    } else {
        let dimensions = match input.len() - 1 {
            0 => Dimensions::scalar(),
            1 => Dimensions::one(input[usize::from(axis == 0)]),
            2 => {
                let mut axes = [0; 2];
                for (slot, extent) in axes.iter_mut().zip(expected) {
                    *slot = extent;
                }
                Dimensions::from(axes)
            }
            _ => Dimensions::from(expected.collect::<alloc::vec::Vec<_>>()),
        };
        Err(SciOxideError::dimension_mismatch(
            dimensions,
            Dimensions::from(actual),
        ))
    }
}
