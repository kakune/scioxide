//! Private checked arithmetic shared by linear-algebra kernels.

use alloc::borrow::Cow;

use scioxide_core::{NonFiniteOrigin, NumericScalar, NumericalIssue, SciOxideError};

/// Round a count to `f64` and then to the scalar type.
/// Counts used as weights or error scales need not be exactly representable.
pub(crate) fn rounded_count<R: scioxide_core::RealScalar>(
    count: usize,
) -> Result<R, SciOxideError> {
    R::try_from_f64_lossy(count as f64)
}

/// Dimension-scaled roundoff guard capped at one half, not a forward-error
/// certificate. Positive collinear vectors remain distinguishable from breakdown.
pub(crate) fn dimension_epsilon<R: scioxide_core::RealScalar>(
    dimension: usize,
) -> Result<R, SciOxideError> {
    Ok((rounded_count::<R>(dimension)? * R::EPSILON).min(R::ONE / (R::ONE + R::ONE)))
}

pub(crate) fn validate_inputs<S: NumericScalar>(values: &[S]) -> Result<(), SciOxideError> {
    values.iter().enumerate().try_for_each(|(index, &value)| {
        if value.has_finite_components() {
            Ok(())
        } else {
            Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                alloc::format!("entry {index} has a non-finite component"),
            ))
        }
    })
}

pub(crate) fn validate_input<S: NumericScalar>(
    value: S,
    name: &'static str,
) -> Result<(), SciOxideError> {
    if value.has_finite_components() {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            alloc::format!("{name} has a non-finite component"),
        ))
    }
}

/// Largest component magnitude of contiguous input values, rejecting a
/// non-finite value with the error of [`validate_input`].
pub(crate) fn validated_maximum<S: NumericScalar>(
    values: &[S],
    name: &'static str,
    context: crate::KernelContext<'_>,
) -> Result<S::Real, SciOxideError> {
    crate::kernel::arch::maximum(values, context.dispatch())
        .and_then(|maximum| match maximum {
            Some(maximum) => Ok(maximum),
            None => crate::kernel::standard::finite_slice_maximum(values),
        })
        .or_else(|error| {
            values
                .iter()
                .try_for_each(|&value| validate_input(value, name))?;
            Err(error)
        })
}

#[inline]
pub(crate) fn finite_candidate<S: NumericScalar>(candidate: S) -> Result<S, SciOxideError> {
    finite_candidate_with_detail(candidate, || {
        "finite operands produced a non-finite result".into()
    })
}

pub(crate) fn add<S: NumericScalar>(left: S, right: S) -> Result<S, SciOxideError> {
    finite_candidate(left + right)
}

pub(crate) fn sub<S: NumericScalar>(left: S, right: S) -> Result<S, SciOxideError> {
    finite_candidate(left - right)
}

pub(crate) fn mul<S: NumericScalar>(left: S, right: S) -> Result<S, SciOxideError> {
    finite_candidate(left * right)
}

pub(crate) fn div<S: NumericScalar>(numerator: S, denominator: S) -> Result<S, SciOxideError> {
    finite_candidate(numerator / denominator)
}

/// Validate a scalar boundary.
pub(crate) fn finite<S: NumericScalar>(
    value: S,
    origin: NonFiniteOrigin,
    detail: &'static str,
) -> Result<S, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(SciOxideError::non_finite(origin, detail))
    }
}

/// Reject an unrepresentable candidate with a lazily constructed diagnostic.
#[inline]
pub(crate) fn finite_candidate_with_detail<S: NumericScalar>(
    value: S,
    detail: impl FnOnce() -> Cow<'static, str>,
) -> Result<S, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            detail(),
        ))
    }
}

/// Decode finite binary parts at the boundary; no finite classification is repeated by callers.
pub(crate) fn finite_parts<R: scioxide_core::RealScalar>(
    value: R,
) -> Result<(u64, i16, bool), SciOxideError> {
    match value.decompose() {
        scioxide_core::FloatDecomposition::Finite {
            significand,
            exponent,
            negative,
        } => Ok((significand, exponent, negative)),
        _ => Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "finite scalar required",
        )),
    }
}

/// Check a matrix's shape and that every entry is finite.
pub(crate) fn validate_matrix<S: NumericScalar>(
    v: crate::MatrixView<'_, S>,
    rows: usize,
    cols: usize,
) -> Result<(), SciOxideError> {
    if (v.rows(), v.cols()) != (rows, cols) {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(rows, cols),
            scioxide_core::Dimensions::two(v.rows(), v.cols()),
        ));
    }
    for i in 0..rows {
        for j in 0..cols {
            validate_input(v.get_element(i, j), "matrix entry")?;
        }
    }
    Ok(())
}

/// Check that the output is exactly `order × order` before writing it.
pub(crate) fn validate_square_output<S: NumericScalar>(
    output: &crate::MatrixViewMut<'_, S>,
    order: usize,
) -> Result<(), SciOxideError> {
    if (output.rows(), output.cols()) != (order, order) {
        return Err(SciOxideError::dimension_mismatch(
            [order, order],
            [output.rows(), output.cols()],
        ));
    }
    Ok(())
}

/// Require an `order × k` right-hand side and a matching output block.
pub(crate) fn check_block_shape<S: NumericScalar>(
    order: usize,
    rhs: crate::MatrixView<'_, S>,
    output: &crate::MatrixViewMut<'_, S>,
) -> Result<(), SciOxideError> {
    let columns = rhs.cols();
    if rhs.rows() != order {
        return Err(SciOxideError::dimension_mismatch(
            [order, columns],
            [rhs.rows(), columns],
        ));
    }
    if output.rows() != order || output.cols() != columns {
        return Err(SciOxideError::dimension_mismatch(
            [order, columns],
            [output.rows(), output.cols()],
        ));
    }
    Ok(())
}

/// Check that every value is finite, reporting `origin` and `detail` on failure.
pub(crate) fn validate_values<S: NumericScalar>(
    values: &[S],
    origin: NonFiniteOrigin,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    values
        .iter()
        .try_for_each(|&value| finite(value, origin, detail).map(|_| ()))
}

/// An argument value outside its accepted range or set.
pub(crate) fn out_of_range(
    argument: scioxide_core::Argument,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::OutOfRange,
        detail,
    )
}

/// Individually valid arguments, or an argument and its preparation, that
/// contradict each other.
pub(crate) fn inconsistent(
    argument: scioxide_core::Argument,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::Inconsistent,
        detail,
    )
}

/// A requested combination, or a capability an argument lacks, that the
/// operation does not support.
pub(crate) fn unsupported(
    argument: scioxide_core::Argument,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        detail,
    )
}

/// Establish finiteness of operator, inverse or preconditioner output immediately
/// after the caller's provider returns; failures have callback origin.
pub(crate) fn provider_output<S: NumericScalar>(output: &[S]) -> Result<(), SciOxideError> {
    validate_values(
        output,
        NonFiniteOrigin::Callback,
        "provider returned a non-finite component",
    )
}

/// Require an argument's shape to equal `expected`, reporting both on mismatch.
pub(crate) fn expect_shape<const N: usize>(
    expected: [usize; N],
    actual: [usize; N],
) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(&expected, &actual))
    }
}

/// Require a square `rows × cols` operator or matrix.
pub(crate) fn square(rows: usize, cols: usize) -> Result<(), SciOxideError> {
    expect_shape([cols, cols], [rows, cols])
}

/// Check exact conjugate symmetry, including real diagonals, on a square finite view.
/// The caller establishes shape and finiteness before this structural check.
pub(crate) fn validate_exact_hermitian<S: NumericScalar>(
    matrix: crate::MatrixView<'_, S>,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    for row in 0..matrix.rows() {
        for col in 0..=row {
            if matrix.get_element(row, col) != matrix.get_element(col, row).conj() {
                return Err(SciOxideError::invalid_input(
                    argument,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    alloc::format!("matrix is not exactly Hermitian at ({row}, {col})"),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use scioxide_core::RealScalar;

    #[test]
    fn input_maximum_preserves_errors_across_dispatch() -> Result<(), SciOxideError> {
        use scioxide_core::{Complex, ExecutionContext, SciOxideErrorView};
        for execution in [
            ExecutionContext::sequential(),
            ExecutionContext::sequential_native(),
        ] {
            let context = crate::KernelContext::new(execution);
            assert_eq!(validated_maximum::<f64>(&[], "coefficient", context)?, 0.0);
            // Several vector-width chunks and a scalar tail.
            let mut values = [1.0; 37];
            values[35] = -f64::MAX;
            assert_eq!(
                validated_maximum(&values, "coefficient", context)?,
                f64::MAX
            );
            assert_eq!(
                validated_maximum(&[-f32::MAX; 37], "coefficient", context)?,
                f32::MAX
            );
            assert!(matches!(
                validated_maximum(&[f32::NAN; 37], "coefficient", context)
                    .unwrap_err()
                    .view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
            for index in [0, 17, 36] {
                for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    values[index] = bad;
                    let error = validated_maximum(&values, "coefficient", context).unwrap_err();
                    assert!(matches!(
                        error.view(),
                        SciOxideErrorView::NonFinite {
                            origin: NonFiniteOrigin::Input,
                            ..
                        }
                    ));
                    assert_eq!(
                        error.to_string(),
                        validate_input(bad, "coefficient").unwrap_err().to_string()
                    );
                }
                values[index] = 1.0;
            }
            assert_eq!(
                validated_maximum(&[Complex::new(3.0, -4.0)], "coefficient", context)?,
                4.0
            );
            assert!(matches!(
                validated_maximum(&[Complex::new(1.0, f64::NAN)], "coefficient", context)
                    .unwrap_err()
                    .view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
        Ok(())
    }

    #[test]
    fn dimension_guards_and_rank_cutoffs_accept_large_binary32_orders() {
        for n in [
            0,
            1,
            (1 << 23) - 1,
            1 << 23,
            1 << 24,
            (1 << 24) + 1,
            usize::MAX,
        ] {
            let factor = dimension_epsilon::<f32>(n).unwrap();
            assert!(factor.is_finite() && (0.0..1.0).contains(&factor));
            assert_eq!(factor == 0., n == 0);
            let cutoff = crate::RankPolicy::<f32>::DimensionScaledEpsilon
                .cutoff(2., n, n)
                .unwrap();
            assert!(cutoff.is_finite() && cutoff < 2.);
            let wide = dimension_epsilon::<f64>(n).unwrap();
            assert!(wide.is_finite() && (0.0..1.0).contains(&wide));
        }
        assert_eq!(
            rounded_count::<f32>((1 << 24) + 1).unwrap(),
            (1 << 24) as f32
        );
    }

    #[test]
    fn finite_parts_preserve_zero_sign_and_binary_boundaries() {
        for value in [
            0.0_f64,
            -0.0,
            f64::MIN_SUBNORMAL,
            f64::MIN_POSITIVE,
            1.0,
            -1.5,
            f64::MAX,
        ] {
            let (significand, exponent, negative) = finite_parts(value).unwrap();
            let restored = (significand as f64).scalbn(i32::from(exponent));
            let restored = if negative { -restored } else { restored };
            assert_eq!(restored.to_bits(), value.to_bits());
        }
        for value in [
            0.0_f32,
            -0.0,
            f32::MIN_SUBNORMAL,
            f32::MIN_POSITIVE,
            1.0,
            -1.5,
            f32::MAX,
        ] {
            let (significand, exponent, negative) = finite_parts(value).unwrap();
            let restored = (significand as f32).scalbn(i32::from(exponent));
            let restored = if negative { -restored } else { restored };
            assert_eq!(restored.to_bits(), value.to_bits());
        }
        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(
                finite_parts(value).unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::NonFinite
            );
        }
    }
}
