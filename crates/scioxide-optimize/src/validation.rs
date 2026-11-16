//! Shared shape, finite-value and parameter validation.
use scioxide_core::{Argument, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError};

pub(crate) fn invalid(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::Inconsistent,
        detail,
    )
}
pub(crate) fn number<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(value)
}
pub(crate) fn shape(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch([expected], [actual]))
    }
}
pub(crate) fn slice_shape<S>(expected: usize, values: &[S]) -> Result<(), SciOxideError> {
    shape(expected, values.len())
}
pub(crate) fn matrix_shape<S: NumericScalar>(
    rows: usize,
    cols: usize,
    matrix: &scioxide_linalg::Matrix<S>,
) -> Result<(), SciOxideError> {
    if rows == matrix.rows() && cols == matrix.cols() {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            [rows, cols],
            [matrix.rows(), matrix.cols()],
        ))
    }
}
fn finite_components(
    mut values: impl Iterator<Item = bool>,
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if values.all(|finite| finite) {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(origin, "non-finite component"))
    }
}
pub(crate) fn finite_real<S: RealScalar>(
    values: &[S],
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    finite_components(values.iter().map(|v| v.is_finite()), origin)
}
pub(crate) fn finite_numeric<S: NumericScalar>(
    values: &[S],
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    finite_components(values.iter().map(|v| v.has_finite_components()), origin)
}
pub(crate) fn finite_input<S: RealScalar>(values: &[S]) -> Result<(), SciOxideError> {
    finite_real(values, NonFiniteOrigin::Input)
}
pub(crate) fn finite_callback<S: RealScalar>(values: &[S]) -> Result<(), SciOxideError> {
    finite_real(values, NonFiniteOrigin::Callback)
}
pub(crate) fn input<S: RealScalar>(expected: usize, values: &[S]) -> Result<(), SciOxideError> {
    slice_shape(expected, values)?;
    finite_input(values)
}
pub(crate) fn scalar<S: RealScalar>(value: S, origin: NonFiniteOrigin) -> Result<S, SciOxideError> {
    finite_real(&[value], origin)?;
    Ok(value)
}
pub(crate) fn callback_scalar<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    scalar(value, NonFiniteOrigin::Callback)
}
/// Validate the start of a minimizer that does not accept problem bounds.
pub(crate) fn unbounded<S, F>(f: &F, x0: &[S]) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: crate::ObjectiveFunction<S> + ?Sized,
{
    if f.bounds().is_some() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("f"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "this method has no bound handling; use a bound-aware method",
        ));
    }
    crate::utils::validate_x0(f, x0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn zero_shapes_and_failure_origins() {
        shape(0, 0).unwrap();
        assert!(slice_shape(1, &[] as &[f64]).is_err());
        for origin in [
            NonFiniteOrigin::Input,
            NonFiniteOrigin::Callback,
            NonFiniteOrigin::Intermediate,
        ] {
            let error = finite_real(&[f64::NAN], origin).unwrap_err();
            assert!(
                matches!(error.view(),scioxide_core::SciOxideErrorView::NonFinite { origin: actual,.. } if actual==origin)
            );
        }
        let error = input(1, &[f64::INFINITY]).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }
}
