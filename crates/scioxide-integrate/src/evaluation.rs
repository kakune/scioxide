//! Shared integrand evaluation: value rows and ordered sequential/executor sampling.
//!
//! Every family evaluates independent callback points into flat rows of real value
//! components. Rows are indexed by point, so executor partitioning never changes a
//! value or a reduction order; only the dispatch differs from the sequential loop.

use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::backend::{PartitionSlot, checked_mul};
use scioxide_core::{
    AffineValue, EvaluationRole, ExecutionContext, ExecutionPolicy, PlanLimits, RealScalar,
    SciOxideError, SolveContext,
};

use crate::kernel::checked_callback_value;

/// Largest worker count any integration plan requests from an executor.
pub(crate) const MAX_WORKERS: usize = WORKERS.get();
/// Callback points sampled per dispatch by refinement families whose levels can
/// exceed any fixed region size; rows are reduced in point order after each block.
/// Unrelated to the public stochastic `SAMPLE_BLOCK` of random observations.
pub(crate) const POINT_BLOCK: usize = 64;
const WORKERS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(31);

/// Copy one callback value into its row, checking shape and finiteness.
#[inline(always)]
pub(crate) fn write_row<S: RealScalar, V: AffineValue<Scale = S> + ?Sized>(
    value: &V,
    row: &mut [S],
) -> Result<(), SciOxideError> {
    crate::kernel::shape(row.len(), value.dimension()?)?;
    for (index, slot) in row.iter_mut().enumerate() {
        *slot = checked_callback_value(value.component(index)?)?;
    }
    Ok(())
}

/// Extract the one checked callback component as a value, without an output
/// pointer that could hide the floating-point data flow from the sampler.
#[inline(always)]
pub(crate) fn scalar_value<S: RealScalar, V: AffineValue<Scale = S> + ?Sized>(
    value: &V,
) -> Result<S, SciOxideError> {
    checked_callback_value(value.try_to_scalar()?)
}

/// Sample a complete short scalar block with one immutable weight per point.
/// Four lanes reduce the weighted values; same-sign products need no compensation
/// within the 50-epsilon floor. Cancellation and range failures recompute from the
/// original weights and stored rows without calling the integrand again. Unit weights
/// give Romberg the same reduction and provider accounting as fixed rules.
/// `N = 0` stores rows only, `N = 1` also reduces weighted values, and
/// `N = 2` additionally reduces absolute mass.
/// Actual attempts, including the first failing callback, are reconciled once.
/// `POSITIVE_NONUNIT` requires an owner proof that every weight is positive
/// and differs from one; other callers keep the full product check.
#[inline(always)]
pub(crate) fn sample_scalar<S: RealScalar, P, const N: usize, const POSITIVE_NONUNIT: bool>(
    points: &[P],
    rows: &mut [S],
    context: &mut SolveContext<'_, '_>,
    evaluate: impl FnMut(&P) -> Result<S, SciOxideError>,
    weight: impl Fn(&P) -> S,
) -> Result<[scioxide_core::PromotingCompensatedSum<S>; N], SciOxideError> {
    sample_scalar_with::<S, P, N, POSITIVE_NONUNIT, _>(
        points,
        rows,
        context,
        evaluate,
        weight,
        {
            #[inline(always)]
            |values| {
                let mut sums = [scioxide_core::PromotingCompensatedSum::ZERO; N];
                for (sum, value) in sums.iter_mut().zip(values) {
                    *sum = scioxide_core::PromotingCompensatedSum::from_compensated(
                        scioxide_core::CompensatedSum::from_parts(value, S::ZERO)?,
                    );
                }
                Ok(sums)
            }
        },
        Ok,
    )
}

/// The shared sampling traversal, with pure finalizers for its native and
/// compensated representations. Scalar consumers can return a rounded value
/// directly without materializing an extended accumulator on the ordinary path.
#[inline(always)]
pub(crate) fn sample_scalar_with<
    S: RealScalar,
    P,
    const N: usize,
    const POSITIVE_NONUNIT: bool,
    T,
>(
    points: &[P],
    rows: &mut [S],
    context: &mut SolveContext<'_, '_>,
    mut evaluate: impl FnMut(&P) -> Result<S, SciOxideError>,
    weight: impl Fn(&P) -> S,
    native: impl FnOnce([S; N]) -> Result<T, SciOxideError>,
    compensated: impl FnOnce([scioxide_core::PromotingCompensatedSum<S>; N]) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let count = points.len();
    crate::kernel::shape(count, rows.len())?;
    if rows.len() > 64 {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("rows"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "short scalar blocks contain at most 64 rows",
        ));
    }
    let mut attempts = 0;
    let sampled = {
        #[inline(always)]
        || {
            let mut sample = {
                #[inline(always)]
                |point: &P| {
                    attempts += 1;
                    evaluate(point)
                }
            };
            let (point_chunks, point_tail) = points.as_chunks::<4>();
            let (row_chunks, row_tail) = rows.as_chunks_mut::<4>();
            let mut quartet = {
                #[inline(always)]
                |points: &[P; 4], row: &mut [S; 4]| -> Result<(), SciOxideError> {
                    let first = sample(&points[0])?;
                    row[0] = first;
                    let second = sample(&points[1])?;
                    row[1] = second;
                    let third = sample(&points[2])?;
                    row[2] = third;
                    let fourth = sample(&points[3])?;
                    row[3] = fourth;
                    Ok(())
                }
            };
            for (points, row) in point_chunks.iter().zip(row_chunks.iter_mut()) {
                quartet(points, row)?;
            }
            for (point, slot) in point_tail.iter().zip(row_tail) {
                let value = sample(point)?;
                *slot = value;
            }
            Ok::<(), SciOxideError>(())
        }
    }();
    context
        .record_providers(EvaluationRole::Function, attempts)?
        .require()?;
    sampled?;
    contract_scalar_rows::<S, P, N, POSITIVE_NONUNIT, T>(points, rows, weight, native, compensated)
}

/// The same ordered contraction for rows sampled through an executor. Stored
/// scalar rows keep the native check and compensated recomputation independent of dispatch.
#[inline(always)]
pub(crate) fn contract_scalar_rows<
    S: RealScalar,
    P,
    const N: usize,
    const POSITIVE_NONUNIT: bool,
    T,
>(
    points: &[P],
    rows: &[S],
    weight: impl Fn(&P) -> S,
    native: impl FnOnce([S; N]) -> Result<T, SciOxideError>,
    compensated: impl FnOnce([scioxide_core::PromotingCompensatedSum<S>; N]) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    use scioxide_core::PromotingCompensatedSum;
    let count = points.len();
    crate::kernel::shape(count, rows.len())?;
    if N == 0 {
        return compensated([PromotingCompensatedSum::ZERO; N]);
    }
    let mut sign_sums = [S::ZERO; 4];
    let mut normal = true;
    let mut products_normal = true;
    let mut values = [S::ZERO; 4];
    let mut accumulate = {
        #[inline(always)]
        |point: &P, slot: S, lane: usize| {
            let w = weight(point);
            let product = w * slot;
            let product_normal = (product.abs() >= S::MIN_POSITIVE)
                | (!POSITIVE_NONUNIT && w == S::ZERO)
                | (slot == S::ZERO);
            if N == 1 {
                products_normal &= product_normal;
            }
            normal &= (!POSITIVE_NONUNIT && w == S::ONE) | product_normal;
            let sign = S::ONE.copysign(if POSITIVE_NONUNIT { slot } else { product });
            sign_sums[lane] += sign;
            values[lane] += product;
        }
    };
    let (point_chunks, point_tail) = points.as_chunks::<4>();
    let (row_chunks, row_tail) = rows.as_chunks::<4>();
    for (points, row) in point_chunks.iter().zip(row_chunks) {
        accumulate(&points[0], row[0], 0);
        accumulate(&points[1], row[1], 1);
        accumulate(&points[2], row[2], 2);
        accumulate(&points[3], row[3], 3);
    }
    for (lane, (point, &slot)) in point_tail.iter().zip(row_tail).enumerate() {
        accumulate(point, slot, lane);
    }
    // At most 64 signs of magnitude one sum exactly in either scalar width.
    // copysign includes signed zeros, preserving the original replay decision.
    let sign_total = (sign_sums[0] + sign_sums[1]) + (sign_sums[2] + sign_sums[3]);
    let length = S::from_f64_ieee(count as f64);
    let signs = if count == 0 {
        0
    } else if sign_total == length {
        1
    } else if sign_total == -length {
        2
    } else {
        3
    };
    let value = (values[0] + values[1]) + (values[2] + values[3]);
    if normal && signs != 3 && value.is_finite() {
        // With one sign, taking the absolute value commutes with every
        // addition in this tree. Its mass is the same reduction without
        // maintaining a second set of lanes while calling the integrand.
        return native(core::array::from_fn(|i| {
            if i == 0 { value } else { value.abs() }
        }));
    }
    compensated(contract_rows::<S, P, N>(
        points,
        rows,
        weight,
        products_normal,
        signs,
    )?)
}

/// Contraction of stored rows, independent of callback and context types.
/// Keeps the compensated recomputation outside the short sampling loop.
#[inline(never)]
fn contract_rows<S: RealScalar, P, const N: usize>(
    points: &[P],
    rows: &[S],
    weight: impl Fn(&P) -> S,
    products_normal: bool,
    signs: u8,
) -> Result<[scioxide_core::PromotingCompensatedSum<S>; N], SciOxideError> {
    use scioxide_core::PromotingCompensatedSum;
    let count = points.len();
    if N == 1
        && signs == 3
        && products_normal
        && let Some([sum]) = scioxide_core::backend::compensated_indexed(count, |index| {
            [weight(&points[index]) * rows[index]]
        })
    {
        // Product admission is already complete. Indexed compensation has the
        // same lane parts as the retained-products reduction, without another
        // staging block or a second admission pass.
        return Ok(core::array::from_fn(|_| {
            PromotingCompensatedSum::from_compensated(sum)
        }));
    }
    let terms = || (0..count).map(|index| (weight(&points[index]), rows[index]));
    let sum = PromotingCompensatedSum::from_products(terms())?;
    let absolute = if N > 1 {
        PromotingCompensatedSum::from_products(terms().map(|(w, v)| (w.abs(), v.abs())))?
    } else {
        PromotingCompensatedSum::ZERO
    };
    Ok(core::array::from_fn(
        |i| if i == 0 { sum } else { absolute },
    ))
}

/// Evaluate `count` independent rows of `components` values in index order.
///
/// The caller checks the budgets first. Each invocation is counted once as a
/// function evaluation, including invocations that fail. Counts are reconciled
/// after the batch, including the last failing row, or after executor dispatch.
#[inline(always)]
pub(crate) fn sample_rows<S, E>(
    count: usize,
    components: usize,
    rows: &mut [S],
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
    evaluate: &E,
) -> Result<(), SciOxideError>
where
    S: Send,
    E: Fn(usize, &mut [S]) -> Result<(), SciOxideError> + Sync + ?Sized,
{
    sample_rows_counted(
        count,
        components,
        rows,
        policy,
        context,
        &|index, row, counts| {
            *counts = [1, 1];
            evaluate(index, row)
        },
    )
    .map(|_| ())
}

/// Sample ordered rows while reconciling separate provider and function attempts.
/// The callback updates both counters before each provider call, including failures.
#[inline(always)]
pub(crate) fn sample_rows_counted<S, E>(
    count: usize,
    components: usize,
    rows: &mut [S],
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
    evaluate: &E,
) -> Result<usize, SciOxideError>
where
    S: Send,
    E: Fn(usize, &mut [S], &mut [usize; 2]) -> Result<(), SciOxideError> + Sync + ?Sized,
{
    crate::kernel::shape(
        checked_mul(count, components, scioxide_core::quantity!("sample_rows"))?,
        rows.len(),
    )?;
    if policy == ExecutionPolicy::Sequential || count < 2 {
        return if components == 1 {
            sample_sequential::<S, E, 1>(count, components, rows, context, evaluate)
        } else {
            sample_sequential::<S, E, 0>(count, components, rows, context, evaluate)
        };
    }
    sample_rows_parallel(count, components, rows, policy, context, evaluate)
}

/// A fixed scalar stride exposes singleton row shape without changing the
/// callback-first failure semantics of other requested component counts.
#[inline(always)]
fn sample_sequential<S: Send, E, const WIDTH: usize>(
    count: usize,
    requested: usize,
    rows: &mut [S],
    context: &mut SolveContext<'_, '_>,
    evaluate: &E,
) -> Result<usize, SciOxideError>
where
    E: Fn(usize, &mut [S], &mut [usize; 2]) -> Result<(), SciOxideError> + Sync + ?Sized,
{
    let components = if WIDTH == 0 { requested } else { WIDTH };

    let mut functions = 0;
    let mut attempts = 0usize;
    for index in 0..count {
        let mut counts = [0; 2];
        let result = evaluate(
            index,
            &mut rows[index * components..(index + 1) * components],
            &mut counts,
        );
        attempts += counts[0];
        functions += counts[1];
        if let Err(error) = result {
            context
                .record_providers(EvaluationRole::Function, attempts)?
                .require()?;
            return Err(error);
        }
    }
    context
        .record_providers(EvaluationRole::Function, attempts)?
        .require()?;
    Ok(functions)
}

/// Keep executor state out of the sequential sampler's stack frame.
fn sample_rows_parallel<S, E>(
    count: usize,
    components: usize,
    rows: &mut [S],
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
    evaluate: &E,
) -> Result<usize, SciOxideError>
where
    S: Send,
    E: Fn(usize, &mut [S], &mut [usize; 2]) -> Result<(), SciOxideError> + Sync + ?Sized,
{
    let execution = context.execution();
    let plan = execution.plan(policy, count, LIMITS);
    let mut slots: [PartitionSlot<&mut [S]>; MAX_WORKERS] =
        core::array::from_fn(|_| PartitionSlot::empty());
    let attempts: [AtomicUsize; MAX_WORKERS] = core::array::from_fn(|_| AtomicUsize::new(0));
    let functions: [AtomicUsize; MAX_WORKERS] = core::array::from_fn(|_| AtomicUsize::new(0));
    let mut remaining = rows;
    for partition in plan.partitions() {
        let (head, tail) = remaining.split_at_mut(checked_mul(
            partition.range().len(),
            components,
            scioxide_core::quantity!("partition_rows"),
        )?);
        remaining = tail;
        slots[partition.worker_index()] = PartitionSlot::new(head);
    }
    let outcome = execution.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, rows, _| {
            let first = partition.range().start;
            for index in partition.range() {
                let offset = (index - first) * components;
                let mut counts = [0; 2];
                let result = evaluate(index, &mut rows[offset..offset + components], &mut counts);
                attempts[partition.worker_index()].fetch_add(counts[0], Ordering::Relaxed);
                functions[partition.worker_index()].fetch_add(counts[1], Ordering::Relaxed);
                result?;
            }
            Ok(())
        },
    );
    context
        .record_providers(
            EvaluationRole::Function,
            attempts.iter().map(|n| n.load(Ordering::Relaxed)).sum(),
        )?
        .require()?;
    outcome?;
    Ok(functions.iter().map(|n| n.load(Ordering::Relaxed)).sum())
}

/// Deterministic partition limits shared by every integration dispatch.
pub(crate) const LIMITS: PlanLimits = PlanLimits {
    work_per_item: NonZeroUsize::MIN,
    auto_threshold: NonZeroUsize::MIN,
    worker_limit: WORKERS,
};

/// Run `run(index, item)` once for every item: in index order when sequential, or
/// over contiguous executor partitions otherwise.
///
/// Items are disjoint units of independent work (for example sample blocks with
/// their own scratch and summaries), so each result depends only on its item and
/// never on the partition; callers reduce items in index order afterwards. The
/// executor's identity, duplicate and completion checks apply.
pub(crate) fn dispatch<T: Send>(
    items: &mut [T],
    policy: ExecutionPolicy,
    execution: ExecutionContext<'_>,
    run: &(dyn Fn(usize, &mut T) -> Result<(), SciOxideError> + Sync),
) -> Result<(), SciOxideError> {
    if policy == ExecutionPolicy::Sequential || items.len() < 2 {
        return items
            .iter_mut()
            .enumerate()
            .try_for_each(|(index, item)| run(index, item));
    }
    let plan = execution.plan(policy, items.len(), LIMITS);
    let mut slots: [PartitionSlot<&mut [T]>; MAX_WORKERS] =
        core::array::from_fn(|_| PartitionSlot::empty());
    let mut remaining = items;
    for partition in plan.partitions() {
        let (head, tail) = core::mem::take(&mut remaining).split_at_mut(partition.range().len());
        remaining = tail;
        slots[partition.worker_index()] = PartitionSlot::new(head);
    }
    execution.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, items, _| {
            partition
                .range()
                .zip(items.iter_mut())
                .try_for_each(|(index, item)| run(index, item))
        },
    )
}

/// Ordered short scalar sampling with optional unit-weight reductions.
#[inline(always)]
pub(crate) fn sample_scalar_1d<S, V, F, const N: usize>(
    function: &F,
    points: &[S],
    rows: &mut [S],
    context: &mut SolveContext<'_, '_>,
) -> Result<[scioxide_core::PromotingCompensatedSum<S>; N], SciOxideError>
where
    S: RealScalar,
    V: AffineValue<Scale = S>,
    F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
{
    sample_scalar::<S, S, N, false>(
        points,
        rows,
        context,
        {
            #[inline(always)]
            |point: &S| scalar_value(&function(*point)?)
        },
        |_| S::ONE,
    )
}

/// Evaluate a one-dimensional integrand at `points` into rows.
#[inline(always)]
pub(crate) fn sample_1d<S, V, F>(
    function: &F,
    points: &[S],
    components: usize,
    rows: &mut [S],
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    V: AffineValue<Scale = S>,
    F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
{
    if components == 1 && points.len() <= 64 && policy == ExecutionPolicy::Sequential {
        return sample_scalar_1d::<S, V, F, 0>(function, points, rows, context).map(|_| ());
    }
    sample_rows(
        points.len(),
        components,
        rows,
        policy,
        context,
        &|index, row| write_row(&function(points[index])?, row),
    )
}

/// Evaluate a multidimensional integrand at `count` points stored row-major.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn sample_nd<S, V, F>(
    function: &F,
    points: &[S],
    dimension: usize,
    count: usize,
    components: usize,
    rows: &mut [S],
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    V: AffineValue<Scale = S>,
    F: Fn(&[S]) -> Result<V, SciOxideError> + Sync + ?Sized,
{
    crate::kernel::shape(
        checked_mul(count, dimension, scioxide_core::quantity!("sample_points"))?,
        points.len(),
    )?;
    sample_rows(count, components, rows, policy, context, &|index, row| {
        write_row(
            &function(&points[index * dimension..(index + 1) * dimension])?,
            row,
        )
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn scalar_attempts<const N: usize>() {
        for count in [0, 1, 3, 4, 5, 20, 63, 64] {
            let points: alloc::vec::Vec<usize> = (0..count).collect();
            let mut rows = alloc::vec![42.0; count];
            let mut state = scioxide_core::SolveState::new();
            for failure in core::iter::once(None).chain((0..count).map(Some)) {
                rows.fill(42.0);
                let mut calls = 0;
                scioxide_core::SolveScope::from(ExecutionContext::sequential_native())
                    .run(
                        &mut state,
                        crate::workspace::identity(
                            scioxide_core::algorithm_id!("integrate", "sample_scalar", 1),
                            count,
                        ),
                        scioxide_core::SolveBudgets::unbounded(),
                        |cx| {
                            let result = sample_scalar::<f64, _, N, false>(
                                &points,
                                &mut rows,
                                cx,
                                |&point| {
                                    assert_eq!(point, calls);
                                    calls += 1;
                                    if failure == Some(point) {
                                        Err(SciOxideError::non_finite(
                                            scioxide_core::NonFiniteOrigin::Callback,
                                            "deliberate sampling failure",
                                        ))
                                    } else {
                                        Ok(1.0)
                                    }
                                },
                                |_| 1.0,
                            );
                            assert_eq!(calls, failure.map_or(count, |index| index + 1));
                            assert_eq!(cx.provider_count(EvaluationRole::Function).unwrap(), calls);
                            if failure.is_some() {
                                assert_eq!(
                                    result.unwrap_err().kind(),
                                    scioxide_core::SciOxideErrorKind::NonFinite
                                );
                            } else {
                                assert!(result.is_ok());
                            }
                            let completed = failure.unwrap_or(count);
                            assert!(rows[..completed].iter().all(|&value| value == 1.0));
                            assert!(rows[completed..].iter().all(|&value| value == 42.0));
                            Ok(())
                        },
                    )
                    .unwrap();
            }
        }
    }

    #[test]
    fn scalar_blocks_reconcile_success_and_every_failure_position() {
        scalar_attempts::<0>();
        scalar_attempts::<1>();
        scalar_attempts::<2>();
    }

    #[test]
    fn opposite_zero_signs_preserve_compensated_replay() {
        fn check<S: RealScalar, const POSITIVE_NONUNIT: bool>(weight: S) {
            let large = S::ONE / S::EPSILON;
            let values = [large, S::HALF, S::HALF, S::HALF, S::HALF, S::NEG_ZERO];
            for negative in [false, true] {
                let points = values.map(|value| if negative { -value } else { value });
                let mut rows = [S::ZERO; 6];
                let mut state = scioxide_core::SolveState::new();
                scioxide_core::SolveScope::from(ExecutionContext::sequential_native())
                    .run(
                        &mut state,
                        crate::workspace::identity(
                            scioxide_core::algorithm_id!("integrate", "sample_scalar", 1),
                            6,
                        ),
                        scioxide_core::SolveBudgets::unbounded(),
                        |cx| {
                            let [sum] = sample_scalar::<S, _, 1, POSITIVE_NONUNIT>(
                                &points,
                                &mut rows,
                                cx,
                                |&value| Ok(value),
                                |_| weight,
                            )?;
                            let expected = (large + S::TWO) * weight;
                            let expected = if negative { -expected } else { expected };
                            assert_eq!(
                                sum.round()?.into_f64().to_bits(),
                                expected.into_f64().to_bits()
                            );
                            Ok(())
                        },
                    )
                    .unwrap();
            }
        }
        check::<f32, false>(1.0);
        check::<f64, false>(1.0);
        check::<f32, true>(0.5);
        check::<f64, true>(0.5);
    }
}
