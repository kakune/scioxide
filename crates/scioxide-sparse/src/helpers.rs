//! Private shape checks, iteration accounting, indexing and scalar helpers.
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, try_copy_slice, try_filled};
use scioxide_core::{
    Argument, ConvergenceDiagnostics, ConvergenceReason, Dimensions, InvalidInputReason,
    NonFiniteOrigin, NumericScalar, SciOxideError, SolveContext,
};

pub(crate) fn invalid(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input_static(argument, InvalidInputReason::Inconsistent, detail)
}

/// Exact mathematical extent (vectors, shapes, and stored structural arrays).
pub(crate) fn exact(actual: usize, expected: usize) -> Result<(), SciOxideError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            Dimensions::one(expected),
            Dimensions::one(actual),
        ))
    }
}

/// Borrowed scratch is a minimum; only the required prefix may be touched.
pub(crate) fn minimum(actual: usize, required: usize) -> Result<(), SciOxideError> {
    if actual >= required {
        Ok(())
    } else {
        exact(actual, required)
    }
}

/// Charge one iteration, checking cancellation and the deadline first.
pub(crate) fn admit(context: &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> {
    context.execution().check()?;
    if context.record_iteration()?.is_granted() {
        Ok(())
    } else {
        Err(exhausted(context, ConvergenceReason::IterationLimit))
    }
}
pub(crate) fn exhausted(
    context: &SolveContext<'_, '_>,
    reason: ConvergenceReason,
) -> SciOxideError {
    SciOxideError::convergence_failure(
        reason,
        ConvergenceDiagnostics::new().with_counts(context.counts()),
        "sparse work budget exhausted",
    )
}

pub(crate) fn finite_with_origin<S: NumericScalar>(
    values: &[S],
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if values.iter().all(|v| v.has_finite_components()) {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            origin,
            "sparse scalar components must be finite",
        ))
    }
}
pub(crate) fn finite<S: NumericScalar>(values: &[S]) -> Result<(), SciOxideError> {
    finite_with_origin(values, NonFiniteOrigin::Input)
}

/// Stable counting transpose of already validated bucket indices. Payloads may
/// be source positions, row indices, or scalar/block data. The iterator must
/// replay the same entries on cloning. Counts and prefix sums are checked;
/// storage allocation is fallible. O(buckets + entries), with no numerical work.
pub(crate) fn counting_transpose<T>(
    buckets: usize,
    entries: impl Iterator<Item = (usize, T)> + Clone,
    mut scatter: impl FnMut(usize, T),
) -> Result<Vec<usize>, SciOxideError> {
    let quantity = scioxide_core::quantity!("sparse_transpose");
    let mut ptr = try_filled(checked_add(buckets, 1, quantity)?, 0usize, quantity)?;
    for (bucket, _) in entries.clone() {
        ptr[bucket + 1] = checked_add(ptr[bucket + 1], 1, quantity)?;
    }
    for i in 1..ptr.len() {
        ptr[i] = checked_add(ptr[i], ptr[i - 1], quantity)?;
    }
    let mut cursor = try_copy_slice(&ptr[..buckets], quantity)?;
    for (bucket, value) in entries {
        scatter(cursor[bucket], value);
        cursor[bucket] += 1;
    }
    Ok(ptr)
}

/// Final IEEE projection. Strict projection rejects a disappearing component;
/// gradual projection permits signed underflow, including rounding to zero.
#[derive(Clone, Copy)]
pub(crate) enum Projection {
    Gradual,
    Strict,
}

/// Convert scalar components without introducing arithmetic rounding.
pub(crate) fn scaled<S: NumericScalar>(
    value: S,
) -> Result<scioxide_core::ScaledValue<scioxide_core::Complex<S::Real>>, SciOxideError> {
    scioxide_core::ScaledValue::from_value(scioxide_core::Complex::new(
        value.real_component(),
        value.imaginary_component(),
    ))
}

/// Project the represented scaled value; this does not imply a single rounding
/// of the mathematical expression that produced it. Each scaled operation has
/// its own documented rounding boundary.
pub(crate) fn project<S: NumericScalar>(
    value: scioxide_core::ScaledValue<scioxide_core::Complex<S::Real>>,
    policy: Projection,
) -> Result<S, SciOxideError> {
    let z = match policy {
        Projection::Gradual => value.round()?,
        Projection::Strict => value.try_value()?,
    };
    Ok(S::from_components(z.re, z.im))
}

/// Scaled division, which avoids overflow and underflow, followed by explicit
/// IEEE projection. Scaled division rounds its significands before projection;
/// this is not a claim of correctly rounded exact division. A zero denominator
/// is `Singular`.
#[cold]
#[inline(never)]
pub(crate) fn divide<S: NumericScalar>(a: S, b: S, policy: Projection) -> Result<S, SciOxideError> {
    finite(&[a, b])?;
    if b == S::ZERO {
        return Err(SciOxideError::singular(None));
    }
    project(scaled(a)?.div(scaled(b)?)?, policy)
}

/// Exact `a - b*c`, rounded once to nearest, ties to even, by core ProductSum.
/// Intermediate products may exceed scalar range. Final overflow is an error;
/// final gradual underflow follows the core accumulator contract.
#[cold]
#[inline(never)]
pub(crate) fn subtract_product<S: NumericScalar>(a: S, b: S, c: S) -> Result<S, SciOxideError> {
    let mut sum = scioxide_core::ProductSum::new();
    sum.add_product(a, S::ONE)?;
    sum.subtract_product(b, c)?;
    sum.finish()
}

pub(crate) fn shape(actual: (usize, usize), expected: (usize, usize)) -> Result<(), SciOxideError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            Dimensions::two(expected.0, expected.1),
            Dimensions::two(actual.0, actual.1),
        ))
    }
}

/// Resizes a preparation buffer; capacity is never released when active shapes shrink.
pub(crate) fn resize<T: Clone>(
    values: &mut Vec<T>,
    len: usize,
    value: T,
) -> Result<(), SciOxideError> {
    let additional = len.saturating_sub(values.len());
    scioxide_core::backend::try_reserve(
        values,
        additional,
        scioxide_core::quantity!("sparse_preparation"),
    )?;
    values.resize(len, value);
    Ok(())
}

/// Counting transpose into caller-reserved buffers, using pointers as cursors.
pub(crate) fn transpose_into(
    pattern: scioxide_core::StructuralPatternView<'_>,
    ptr: &mut [usize],
    indices: &mut [usize],
    positions: &mut [usize],
) {
    ptr.fill(0);
    for &column in pattern.col_indices() {
        ptr[column + 1] += 1;
    }
    for i in 1..ptr.len() {
        ptr[i] += ptr[i - 1];
    }
    for (row, bounds) in pattern.row_ptr().windows(2).enumerate() {
        for pos in bounds[0]..bounds[1] {
            let col = pattern.col_indices()[pos];
            let slot = ptr[col];
            ptr[col] += 1;
            indices[slot] = row;
            positions[slot] = pos;
        }
    }
    for i in (1..ptr.len()).rev() {
        ptr[i] = ptr[i - 1];
    }
    ptr[0] = 0;
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{Complex64, SciOxideErrorKind};
    #[test]
    fn subtraction_rounds_only_after_cancellation() {
        let b = 1. + 2_f64.powi(-27);
        let c = 1. - 2_f64.powi(-27);
        assert_eq!(1. - b * c, 0.);
        assert_eq!(subtract_product(1., b, c).unwrap(), 2_f64.powi(-54));
        assert_eq!(
            subtract_product(
                Complex64::new(1., 0.),
                Complex64::new(b, 0.),
                Complex64::new(c, 0.)
            )
            .unwrap(),
            Complex64::new(2_f64.powi(-54), 0.)
        );
        assert_eq!(subtract_product(f64::MAX, f64::MAX, 2.).unwrap(), -f64::MAX);
    }
    #[test]
    fn division_projection_is_explicit() {
        let tiny = f64::from_bits(1);
        assert_eq!(divide(tiny, 2., Projection::Gradual).unwrap(), 0.);
        assert_eq!(
            divide(tiny, 2., Projection::Strict).unwrap_err().kind(),
            SciOxideErrorKind::NumericalInstability
        );
        assert_eq!(divide(f64::MAX, f64::MAX, Projection::Strict).unwrap(), 1.);
        assert_eq!(
            divide(1_f64, 0., Projection::Gradual).unwrap_err().kind(),
            SciOxideErrorKind::Singular
        );
    }
    #[test]
    fn counting_transpose_is_stable_and_accepts_empty_buckets() {
        let source = [(2, 9), (0, 7), (2, 8), (0, 6)];
        let mut output = [0; 4];
        let ptr = counting_transpose(4, source.into_iter(), |slot, v| output[slot] = v).unwrap();
        assert_eq!(ptr, [0, 2, 2, 4, 4]);
        assert_eq!(output, [7, 6, 9, 8]);
        assert_eq!(
            counting_transpose(0, core::iter::empty::<(usize, ())>(), |_, ()| {}).unwrap(),
            [0]
        );
    }
}
