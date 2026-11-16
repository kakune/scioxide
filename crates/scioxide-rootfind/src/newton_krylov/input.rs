//! Input for Newton–Krylov solves.
use super::*;

/// Holds the initial point of a [`NewtonKrylovSolver`] solve.
///
/// Slices, arrays, `Vec`s and rank-one [`NdView`]s convert into this type, so
/// `solve(&problem, &[1.0])` and `solve(&problem, view)` both work. A view may
/// have any valid stride.
#[derive(Debug)]
pub enum NewtonKrylovInput<'a, 'layout, T: NumericScalar> {
    /// Contiguous coordinates.
    Slice(&'a [T]),
    /// A rank-one strided view. A view of another rank is an `InvalidInput` error
    /// when the solve starts.
    View(NdView<'a, 'layout, T>),
}
impl<T: NumericScalar> NewtonKrylovInput<'_, '_, T> {
    pub(crate) fn len(&self) -> Result<usize, SciOxideError> {
        match self {
            Self::Slice(values) => Ok(values.len()),
            Self::View(values) => {
                if values.layout().shape().rank() != 1 {
                    return Err(invalid(
                        scioxide_core::argument!("initial"),
                        InvalidInputReason::Inconsistent,
                        "initial view must have rank one",
                    ));
                }
                Ok(values.iter().len())
            }
        }
    }
}
impl<'a, T: NumericScalar> From<&'a [T]> for NewtonKrylovInput<'a, '_, T> {
    fn from(values: &'a [T]) -> Self {
        Self::Slice(values)
    }
}
impl<'a, T: NumericScalar, const N: usize> From<&'a [T; N]> for NewtonKrylovInput<'a, '_, T> {
    fn from(values: &'a [T; N]) -> Self {
        Self::Slice(values)
    }
}
impl<'a, T: NumericScalar> From<&'a Vec<T>> for NewtonKrylovInput<'a, '_, T> {
    fn from(values: &'a Vec<T>) -> Self {
        Self::Slice(values.as_slice())
    }
}
impl<'a, 'layout, T: NumericScalar> From<NdView<'a, 'layout, T>>
    for NewtonKrylovInput<'a, 'layout, T>
{
    fn from(values: NdView<'a, 'layout, T>) -> Self {
        Self::View(values)
    }
}
