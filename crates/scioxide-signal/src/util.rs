//! Validation, range scaling, allocation and execution helpers.
use crate::{SignalBoundary, SignalLayout, SignalRequirements, SignalWorkspace};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_mul, try_filled};
use scioxide_core::{
    Argument, Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, NumericScalar,
    RealScalar, SciOxideError, SolveContext,
};
/// Four-lane working-precision contraction; exceptional range replays exactly.
/// Callers validate operands before entering this hot kernel. The componentwise
/// error is bounded by gamma(2*m+8) times absolute product mass, plus gradual
/// underflow. Cloning the iterator must replay the same immutable operands.
#[inline]
pub(crate) fn working_sum_products<T: NumericScalar>(
    pairs: impl Iterator<Item = (T, T)> + Clone,
) -> Result<T, SciOxideError> {
    let candidate = scioxide_core::backend::standard::dot(pairs.clone());
    if candidate.has_finite_components() {
        Ok(candidate)
    } else {
        sum_products(pairs)
    }
}

/// Contiguous short filters use explicit four-element blocks so the compiler can
/// vectorize independent lanes even when samples and coefficients run oppositely.
#[inline]
pub(crate) fn contiguous_products<T: NumericScalar, C: Copy, const REVERSE: bool>(
    samples: &[T],
    coefficients: &[C],
    convert: impl Fn(C) -> T + Copy,
) -> Result<T, SciOxideError> {
    exact_len(samples.len(), coefficients.len())?;
    let n = samples.len();
    let (mut a, mut b, mut c, mut d) = (T::ZERO, T::ZERO, T::ZERO, T::ZERO);
    for (i, h) in coefficients.chunks_exact(4).enumerate() {
        let base = if REVERSE { n - 4 * (i + 1) } else { 4 * i };
        let x = &samples[base..base + 4];
        a = a + x[if REVERSE { 3 } else { 0 }] * convert(h[0]);
        b = b + x[if REVERSE { 2 } else { 1 }] * convert(h[1]);
        c = c + x[if REVERSE { 1 } else { 2 }] * convert(h[2]);
        d = d + x[if REVERSE { 0 } else { 3 }] * convert(h[3]);
    }
    let mut lanes = [a, b, c, d];
    for (lane, &h) in coefficients[n / 4 * 4..].iter().enumerate() {
        let index = n / 4 * 4 + lane;
        let x = samples[if REVERSE { n - 1 - index } else { index }];
        lanes[lane] = lanes[lane] + x * convert(h);
    }
    let candidate = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    if candidate.has_finite_components() {
        return Ok(candidate);
    }
    sum_products(
        coefficients
            .iter()
            .enumerate()
            .map(|(i, &h)| (samples[if REVERSE { n - 1 - i } else { i }], convert(h))),
    )
}

// Use the core range accumulator so cancellation does not overflow products or discard small
// terms while normalizing a large dynamic range.
pub(crate) fn sum_products<T: scioxide_core::NumericScalar>(
    pairs: impl Iterator<Item = (T, T)>,
) -> Result<T, SciOxideError> {
    let mut real = scioxide_core::RangeAccumulator::new();
    let mut imaginary = scioxide_core::RangeAccumulator::new();
    let is_complex =
        T::from_components(T::Real::ZERO, T::Real::ONE).imaginary_component() != T::Real::ZERO;
    for (a, b) in pairs {
        real.add_product(a.real_component(), b.real_component())?;
        if is_complex {
            real.subtract_product(a.imaginary_component(), b.imaginary_component())?;
            imaginary.add_product(a.real_component(), b.imaginary_component())?;
            imaginary.add_product(a.imaginary_component(), b.real_component())?;
        }
    }
    Ok(T::from_components(
        real.finish()?,
        if is_complex {
            imaginary.finish()?
        } else {
            T::Real::ZERO
        },
    ))
}
/// Out-of-range signal parameter `argument`.
pub(crate) fn invalid(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::OutOfRange,
        detail,
    )
}
/// Empty required collection or transform length `argument`.
pub(crate) fn empty(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, scioxide_core::InvalidInputReason::Empty, detail)
}
/// Option combination of `argument` that this configuration does not support.
pub(crate) fn unsupported(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        detail,
    )
}
/// Shape disagreement of `actual` with `reference` whose `axis` extent is `len`.
///
/// Shapes of rank at most two do not allocate.
pub(crate) fn axis_mismatch(
    reference: &[usize],
    axis: usize,
    len: usize,
    actual: &[usize],
) -> SciOxideError {
    if reference.len() <= 2 {
        let mut expected = [0; 2];
        expected[..reference.len()].copy_from_slice(reference);
        if let Some(slot) = expected.get_mut(axis) {
            *slot = len;
        }
        SciOxideError::dimension_mismatch(&expected[..reference.len()], actual)
    } else {
        let mut expected = reference.to_vec();
        if let Some(slot) = expected.get_mut(axis) {
            *slot = len;
        }
        SciOxideError::dimension_mismatch(expected, actual)
    }
}
/// Require `actual` to equal `reference` with its `axis` extent replaced by `len`.
pub(crate) fn check_axis_shape(
    reference: &[usize],
    axis: usize,
    len: usize,
    actual: &[usize],
) -> Result<(), SciOxideError> {
    if actual.len() == reference.len()
        && reference
            .iter()
            .zip(actual)
            .enumerate()
            .all(|(i, (&a, &b))| b == if i == axis { len } else { a })
    {
        Ok(())
    } else {
        Err(axis_mismatch(reference, axis, len, actual))
    }
}
/// Require a view shape equal to the prepared `expected` shape.
pub(crate) fn exact_shape(expected: &[usize], actual: &[usize]) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(expected, actual))
    }
}
pub(crate) fn capacity(required: usize, actual: usize) -> Result<(), SciOxideError> {
    if actual < required {
        Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(required),
            scioxide_core::Dimensions::one(actual),
        ))
    } else {
        Ok(())
    }
}
/// Return `value` when finite, else an overflow.
pub(crate) fn finite<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    require_finite(value, output_overflow)
}
/// Unrepresentable result.
pub(crate) fn output_overflow() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Overflow,
        "result is not representable",
    )
}
/// Checked product of finite factors.
/// callers map `None` to their own typed error.
pub(crate) fn checked_factor_product<S: RealScalar>(values: &[S]) -> Option<S> {
    scioxide_core::checked_float_product_factors(values).ok()
}
pub(crate) fn scalar_from_usize_rounded<S: RealScalar>(value: usize) -> Result<S, SciOxideError> {
    scioxide_core::backend::scalar_from_u128_rounded(value as u128)
}
pub(crate) fn check_real_slice_finite<S: scioxide_core::RealScalar>(
    input: &[S],
) -> Result<(), SciOxideError> {
    check_finite_values(input.iter().map(|x| x.is_finite()), || {
        SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "input contains a non-finite value",
        )
    })
}

/// Row-major strides of `shape`, allocated.
pub(crate) fn strides(shape: &[usize]) -> Result<Vec<usize>, SciOxideError> {
    let mut result = try_filled(shape.len(), 0, scioxide_core::quantity!("signal_strides"))?;
    if shape.contains(&0) {
        return Ok(result);
    }
    let mut stride = 1;
    for (&extent, slot) in shape.iter().zip(result.iter_mut()).rev() {
        *slot = stride;
        stride = checked_mul(stride, extent, scioxide_core::quantity!("signal_shape"))?;
    }
    Ok(result)
}
/// Owned copy of plan metadata, allocated.
pub(crate) fn copy<T: Copy>(values: &[T]) -> Result<Vec<T>, SciOxideError> {
    scioxide_core::backend::try_copy_slice(values, scioxide_core::quantity!("signal_metadata"))
}
pub(crate) fn complex<T: NumericScalar>(v: T) -> Complex<T::Real> {
    Complex::new(v.real_component(), v.imaginary_component())
}
pub(crate) fn scale<T: NumericScalar>(values: &[T]) -> T::Real {
    values.iter().fold(T::Real::ZERO, |s, v| {
        s.max(v.real_component().abs())
            .max(v.imaginary_component().abs())
    })
}
/// Reject non-finite samples.
pub(crate) fn inputs<T: NumericScalar>(values: &[T]) -> Result<(), SciOxideError> {
    check_finite_values(values.iter().map(|v| v.has_finite_components()), || {
        SciOxideError::non_finite(scioxide_core::NonFiniteOrigin::Input, "non-finite sample")
    })
}
pub(crate) fn boundary_index(i: i128, n: usize, boundary: SignalBoundary) -> Option<usize> {
    if i >= 0 && i < n as i128 {
        return Some(i as usize);
    }
    match boundary {
        SignalBoundary::Zero => None,
        SignalBoundary::Periodic => Some(i.rem_euclid(n as i128) as usize),
        SignalBoundary::Reflect => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * (n as i128 - 1);
            let j = i.rem_euclid(period);
            Some(j.min(period - j) as usize)
        }
    }
}

/// Zero-filled storage of `n` values, allocated.
pub(crate) fn zeros<T: NumericScalar>(n: usize) -> Result<Vec<T>, SciOxideError> {
    try_filled(n, T::ZERO, scioxide_core::quantity!("signal_workspace"))
}
/// Checked workspace size `n * k`.
pub(crate) fn size(n: usize, k: usize) -> Result<usize, SciOxideError> {
    checked_mul(n, k, scioxide_core::quantity!("signal_workspace"))
}
/// Return `x` when both components are finite, else an overflow.
#[inline(always)]
pub(crate) fn check<T: NumericScalar>(x: T) -> Result<T, SciOxideError> {
    require_finite(x, output_overflow)
}
/// Check execution and charge one shared iteration.
pub(crate) fn admit(context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
    context.execution().check()?;
    if !context.record_iteration()?.is_granted() {
        return Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new(),
            "signal line budget exhausted",
        ));
    }
    Ok(())
}

/// Lane geometry `(axis length, lane count)` of a request layout.
///
/// A contiguous record is one lane. An axis layout validates the axis and the
/// checked element count; its lanes are the product of the other extents.
pub(crate) fn layout_lanes(layout: SignalLayout<'_>) -> Result<(usize, usize), SciOxideError> {
    match layout {
        SignalLayout::Contiguous(n) => Ok((n, 1)),
        SignalLayout::Axis { shape, axis } => {
            if axis >= shape.len() {
                return Err(invalid(
                    scioxide_core::argument!("axis"),
                    "axis must be below the shape rank",
                ));
            }
            scioxide_core::Shape::new(shape)?;
            let lanes = shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != axis)
                .try_fold(1, |lanes, (_, &extent)| {
                    checked_mul(lanes, extent, scioxide_core::quantity!("signal_lanes"))
                })?;
            Ok((shape[axis], lanes))
        }
    }
}

/// Sample requirements of [`axis_filter`] with `stride` staged values per element.
pub(crate) fn lane_requirements(
    layout: SignalLayout<'_>,
    stride: usize,
) -> Result<SignalRequirements, SciOxideError> {
    let (n, lanes) = layout_lanes(layout)?;
    Ok(SignalRequirements::values(size(size(n, lanes)?, stride)?))
}

/// Stage selected-axis lanes in blocks of `stride * n` values and run `filter`
/// per lane on `(input, output, work)`; `work` holds `(stride - 2) * n` values.
/// The output is written only after every lane succeeds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn axis_filter<T: NumericScalar>(
    input: NdView<'_, '_, T>,
    mut output: NdViewMut<'_, '_, T>,
    axis: usize,
    workspace: &mut SignalWorkspace<T>,
    stride: usize,
    policy: ExecutionPolicy,
    execution: ExecutionContext<'_>,
    filter: impl Fn(&[T], &mut [T], &mut [T]) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    let (n, total) = crate::axis::shape(&input, &output, axis)?;
    let blocks = workspace
        .buffers(&SignalRequirements::values(size(total, stride)?))?
        .values;
    execution.check()?;
    if total == 0 {
        return Ok(());
    }
    crate::axis::gather_blocks(&input, axis, blocks, stride * n, |_| Ok(()))?;
    execute_blocks(blocks, stride * n, policy, execution, |chunk| {
        for block in chunk.chunks_exact_mut(stride * n) {
            execution.check()?;
            let (x, rest) = block.split_at_mut(n);
            let (y, w) = rest.split_at_mut(n);
            filter(x, y, w)?;
        }
        Ok(())
    })?;
    execution.check()?;
    crate::axis::publish_blocks(&mut output, axis, blocks, stride * n, n)?;
    Ok(())
}
/// Run `work` over disjoint `stride`-sized blocks on at most 32 workers.
pub(crate) fn execute_blocks<T: NumericScalar>(
    data: &mut [T],
    stride: usize,
    policy: ExecutionPolicy,
    execution: ExecutionContext<'_>,
    work: impl Fn(&mut [T]) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    use core::num::NonZeroUsize;
    use scioxide_core::backend::PartitionSlot;
    const WORKER_LIMIT: NonZeroUsize = NonZeroUsize::MIN.saturating_add(31);
    if data.is_empty() {
        return Ok(());
    }
    let plan = execution.plan(
        policy,
        data.len() / stride,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: WORKER_LIMIT,
        },
    );
    if plan.worker_count() == 1 {
        return work(data);
    }
    let mut slots: [PartitionSlot<&mut [T]>; 32] = core::array::from_fn(|_| PartitionSlot::empty());
    let mut remaining = data;
    for part in plan.partitions() {
        let (chunk, rest) = remaining.split_at_mut(part.range().len() * stride);
        slots[part.worker_index()] = PartitionSlot::new(chunk);
        remaining = rest;
    }
    execution.execute_partitioned(plan, &slots[..plan.worker_count()], &|_, chunk, _| {
        work(chunk)
    })?;

    Ok(())
}

/// Shared finite predicate with the caller's established error classification.
pub(crate) fn check_finite_values(
    values: impl IntoIterator<Item = bool>,
    error: impl FnOnce() -> SciOxideError,
) -> Result<(), SciOxideError> {
    if values.into_iter().all(core::convert::identity) {
        Ok(())
    } else {
        Err(error())
    }
}
pub(crate) fn clamped_index(i: usize, offset: isize, len: usize) -> usize {
    if offset < 0 {
        i.saturating_sub(offset.unsigned_abs())
    } else {
        i.saturating_add(offset as usize).min(len - 1)
    }
}

pub(crate) fn exact_len(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(expected),
            scioxide_core::Dimensions::one(actual),
        ))
    }
}

/// Classify and scale finite samples in one traversal.
pub(crate) fn input_scale<T: NumericScalar>(
    values: impl IntoIterator<Item = T>,
) -> Result<T::Real, SciOxideError> {
    values.into_iter().try_fold(T::Real::ZERO, |scale, value| {
        check_finite_values([value.has_finite_components()], || {
            SciOxideError::non_finite(scioxide_core::NonFiniteOrigin::Input, "non-finite sample")
        })?;
        Ok(scale
            .max(value.real_component().abs())
            .max(value.imaginary_component().abs()))
    })
}

/// Select an exceptional binary frame once for a complete FFT operand. The
/// ordinary frame reserves room for two forward transforms, their product and
/// an unnormalized inverse. Tiny operands also use a frame so their spectrum
/// products do not underflow before a large partner can restore them.
pub(crate) fn fft_exponent<S: RealScalar>(maximum: S, volume: usize) -> Result<i32, SciOxideError> {
    let exponent = scioxide_core::backend::standard::exponent(maximum)?;
    let limit = (scioxide_core::backend::standard::exponent(S::MAX)? - 4) / 4
        - volume.max(1).ilog2() as i32;
    Ok(if (-limit..=limit).contains(&exponent) {
        0
    } else {
        exponent
    })
}

/// Restore a whole staged buffer with one binary frame. The ordinary frame
/// needs only a finite-result check; exceptional frames permit gradual underflow.
pub(crate) fn restore_fft<T: NumericScalar>(
    values: &mut [T],
    exponent: i32,
) -> Result<(), SciOxideError> {
    if exponent == 0 {
        values
            .iter()
            .try_for_each(|&value| check(value).map(|_| ()))
    } else {
        for value in values {
            *value = scioxide_core::backend::standard::restore(*value, exponent)?;
        }
        Ok(())
    }
}
pub(crate) fn require_finite<T: NumericScalar>(
    value: T,
    error: impl FnOnce() -> SciOxideError,
) -> Result<T, SciOxideError> {
    check_finite_values([value.has_finite_components()], error)?;
    Ok(value)
}
pub(crate) fn inconsistent(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::Inconsistent,
        detail,
    )
}
/// Finite LTI arithmetic: a non-finite value from finite operands is an overflow.
pub(crate) fn lti_finite<T: NumericScalar>(value: T) -> Result<T, SciOxideError> {
    require_finite(value, || {
        SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "LTI arithmetic produced a non-finite value",
        )
    })
}
pub(crate) fn window_finite(value: f64) -> Result<f64, SciOxideError> {
    require_finite(value, || {
        SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "window calculation must remain finite",
        )
    })
}
pub(crate) fn positive<S: RealScalar>(
    x: S,
    argument: Argument,
    error: impl FnOnce() -> SciOxideError,
) -> Result<S, SciOxideError> {
    require_finite(x, error)?;
    if x <= S::ZERO {
        Err(invalid(argument, "parameter must be positive"))
    } else {
        Ok(x)
    }
}

pub(crate) fn zeroed<T: NumericScalar>(
    len: usize,
    quantity: scioxide_core::Quantity,
) -> Result<Vec<T>, SciOxideError> {
    try_filled(len, T::ZERO, quantity)
}

/// Contiguous vector storage, excluding padding and trailing samples.
pub(crate) fn vector_storage<'d, T>(input: &NdView<'d, '_, T>) -> Option<&'d [T]> {
    let layout = input.layout();
    (layout.shape().rank() == 1 && layout.strides()[0] == 1)
        .then(|| &input.storage()[..layout.shape().len()])
}
pub(crate) fn copy_view<T: Copy>(input: &NdView<'_, '_, T>, output: &mut [T]) {
    if let Some(values) = vector_storage(input) {
        output.copy_from_slice(values);
    } else {
        for (slot, &value) in output.iter_mut().zip(input.iter()) {
            *slot = value;
        }
    }
}

/// Kind of an expected failure in unit tests; panics at the caller on `Ok`.
#[cfg(test)]
#[track_caller]
pub(crate) fn error_kind<T>(result: Result<T, SciOxideError>) -> scioxide_core::SciOxideErrorKind {
    #[allow(clippy::panic)]
    match result {
        Ok(_) => panic!("expected an error, got Ok"),
        Err(error) => error.kind(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod contraction_tests {
    use super::*;
    use scioxide_core::SciOxideErrorKind;

    fn compare<T: NumericScalar>() {
        let value = |x| T::Real::try_from_f64_lossy(x).unwrap();
        let mut pairs = Vec::new();
        for i in 0..257 {
            let sign = if i % 2 == 0 { 1. } else { -1. };
            let scale = T::Real::ONE.scalbn((i % 11) * 10 - 50);
            pairs.push((
                T::from_components(
                    value(sign * (1.1 + (i % 7) as f64 / 37.)) * scale,
                    value(sign * 0.1) * scale,
                ),
                T::from_components(value(0.1 + (i % 11) as f64 / 19.), value(-0.3)),
            ));
        }
        for length in [0, 1, 2, 3, 4, 5, 31, 63, 257] {
            let terms = pairs[..length].iter().copied();
            let actual = working_sum_products(terms.clone()).unwrap();
            let left: Vec<_> = terms.clone().map(|(a, _)| a).collect();
            let right: Vec<_> = terms.clone().map(|(_, b)| b).collect();
            let forward =
                contiguous_products::<T, T, false>(&left, &right, core::convert::identity).unwrap();
            let reverse =
                contiguous_products::<T, T, true>(&left, &right, core::convert::identity).unwrap();
            let reverse_exact =
                sum_products(left.iter().rev().zip(&right).map(|(&a, &b)| (a, b))).unwrap();
            let exact = sum_products(terms.clone()).unwrap();
            let mass = terms.fold(T::Real::ZERO, |s, (a, b)| {
                s + (a.real_component().abs() + a.imaginary_component().abs())
                    * (b.real_component().abs() + b.imaginary_component().abs())
            });
            let ku = value((2 * length + 8) as f64) * T::Real::EPSILON * T::Real::HALF;
            let bound = ku / (T::Real::ONE - ku) * mass;
            assert!((actual.real_component() - exact.real_component()).abs() <= bound);
            assert!((actual.imaginary_component() - exact.imaginary_component()).abs() <= bound);
            for (a, b) in [(forward, exact), (reverse, reverse_exact)] {
                assert!((a.real_component() - b.real_component()).abs() <= bound);
                assert!((a.imaginary_component() - b.imaginary_component()).abs() <= bound);
            }
        }
        // Products overflow individually; the completed contraction is finite.
        let large = T::from_real(T::Real::MAX);
        let two = T::from_real(value(2.));
        let cancellation = [(large, two), (-large, two)];
        assert_eq!(
            working_sum_products(cancellation.into_iter()).unwrap(),
            T::ZERO
        );
        assert_eq!(
            error_kind(working_sum_products([(large, two)].into_iter())),
            SciOxideErrorKind::NumericalInstability
        );
        let tiny = T::Real::MIN_POSITIVE * T::Real::EPSILON;
        let left = [large, T::from_real(tiny), -large];
        let right = [two, T::ONE, two];
        let retained =
            contiguous_products::<T, T, false>(&left, &right, core::convert::identity).unwrap();
        assert!(retained.real_component() > T::Real::ZERO);
        assert!(
            (retained.real_component() / tiny - T::Real::ONE).abs() <= value(8.) * T::Real::EPSILON
        );
        assert_eq!(
            error_kind(contiguous_products::<T, T, true>(
                &[large],
                &[two],
                core::convert::identity
            )),
            SciOxideErrorKind::NumericalInstability
        );
        let subnormal = [(T::from_real(tiny), T::from_real(value(0.5))); 8];
        let actual = working_sum_products(subnormal.into_iter()).unwrap();
        let exact = sum_products(subnormal.into_iter()).unwrap();
        assert!((actual.real_component() - exact.real_component()).abs() <= value(8.) * tiny);
    }

    #[test]
    fn working_contractions_follow_absolute_bounds() {
        compare::<f32>();
        compare::<f64>();
        compare::<Complex<f32>>();
        compare::<Complex<f64>>();
    }
}
