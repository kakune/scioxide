//! Jacobi matrices of orthogonal families and zeros by the Golub–Welsch method.
use crate::util::{finite_arithmetic, finite_input, finite_output};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{Argument, RealScalar, Scaled, SciOxideError};
use scioxide_linalg::HermitianEigenvaluesResult;

/// Holds the symmetric tridiagonal Jacobi matrix `Jₙ` of an orthogonal family.
///
/// Let `p₀, p₁, …` be the polynomials of the family that are orthonormal for
/// the weight normalized to total mass one, with positive leading
/// coefficients. They satisfy
///
/// ```text
/// x pₖ(x) = βₖ₊₁ pₖ₊₁(x) + αₖ pₖ(x) + βₖ pₖ₋₁(x),   p₋₁ = 0,  p₀ = 1,
/// ```
///
/// with `βₖ > 0`. `Jₙ` has diagonal `α₀, …, αₙ₋₁` and off-diagonal
/// `β₁, …, βₙ₋₁`. Its eigenvalues are the zeros of `pₙ`, and it does not
/// depend on the normalization chosen for the family. Construction from the
/// three-term recurrence takes O(n) time and memory.
///
/// # References
///
/// G. H. Golub and J. H. Welsch, "Calculation of Gauss quadrature rules",
/// *Math. Comp.* 23 (1969), 221–230.
///
/// ```
/// use scioxide_poly::*;
/// let family = OrthogonalFamily::<f64>::Legendre;
/// let recurrence = family.jacobi_recurrence(3)?;
/// assert_eq!(recurrence.diagonal().len(), 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct JacobiRecurrence<S: RealScalar> {
    diagonal: Vec<S>,
    off_diagonal: Vec<S>,
}
impl<S: RealScalar> JacobiRecurrence<S> {
    /// Returns the diagonal `α₀, …, αₙ₋₁`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let family = OrthogonalFamily::<f64>::Legendre;
    /// let recurrence = family.jacobi_recurrence(3)?;
    /// assert_eq!(recurrence.diagonal(), &[0.0, 0.0, 0.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn diagonal(&self) -> &[S] {
        &self.diagonal
    }
    /// Returns the off-diagonal `β₁, …, βₙ₋₁`; every entry is positive.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let family = OrthogonalFamily::<f64>::Legendre;
    /// let recurrence = family.jacobi_recurrence(3)?;
    /// assert!(recurrence.off_diagonal().iter().all(|value| *value > 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn off_diagonal(&self) -> &[S] {
        &self.off_diagonal
    }
    /// Evaluates the orthonormal polynomial `pₙ₋₁(x)` in extended exponent range.
    ///
    /// `n` is the order of the matrix, and `pₙ₋₁` is defined on
    /// [`JacobiRecurrence`]. The last diagonal entry `αₙ₋₁` is not used. The
    /// cost is O(n) time with no allocation. The value is returned as a
    /// [`Scaled`] number, so it does not overflow or underflow even for large
    /// `n` or large `|x|`. Convert it with [`Scaled::round`], which fails if the
    /// value is outside the range of `S`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` is NaN or infinite.
    /// - `NumericalInstability` if the value exceeds even the extended exponent
    ///   range.
    ///
    /// ```
    /// use scioxide_poly::OrthogonalFamily;
    /// // p₁(x) = √3 x is orthonormal for dx/2 on [−1, 1].
    /// let recurrence = OrthogonalFamily::Legendre.jacobi_recurrence(2)?;
    /// let value = recurrence.last_member_scaled(0.5_f64)?.round()?;
    /// assert!((value - 3.0_f64.sqrt() * 0.5).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn last_member_scaled(&self, x: S) -> Result<Scaled<S>, SciOxideError> {
        finite_input(x, "evaluation point must be finite")?;
        super::recurrence::scaled_member(self.off_diagonal.len(), x, |k| {
            let previous = if k == 0 {
                S::ZERO
            } else {
                self.off_diagonal[k - 1]
            };
            ((S::ONE, -self.diagonal[k], previous), self.off_diagonal[k])
        })
    }

    /// Returns the diagonal and the off-diagonal without copying.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let family = OrthogonalFamily::<f64>::Legendre;
    /// let recurrence = family.jacobi_recurrence(3)?;
    /// let (diagonal, off_diagonal) = recurrence.into_parts();
    /// assert_eq!(diagonal, vec![0.0, 0.0, 0.0]);
    /// assert_eq!(off_diagonal.len(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn into_parts(self) -> (Vec<S>, Vec<S>) {
        (self.diagonal, self.off_diagonal)
    }
}
/// Assembles the symmetric Jacobi matrix of order `n` from a recurrence.
///
/// A zero order is reported against `order`, the argument that supplied `n`.
pub(crate) fn jacobi_recurrence<S: RealScalar>(
    n: usize,
    recurrence_fn: impl Fn(usize) -> Result<(S, S, S), SciOxideError>,
    order: Argument,
) -> Result<JacobiRecurrence<S>, SciOxideError> {
    if n == 0 {
        return Err(SciOxideError::invalid_input(
            order,
            scioxide_core::InvalidInputReason::OutOfRange,
            "cannot compute zeros for degree 0",
        ));
    }

    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let mut diagonal = try_filled(n, zero, scioxide_core::quantity!("golub_welsch_diagonal"))?;
    let mut off_diagonal = try_filled(
        n.saturating_sub(1),
        zero,
        scioxide_core::quantity!("golub_welsch_off_diagonal"),
    )?;

    let mut previous_a = None;
    for (k, entry) in diagonal.iter_mut().enumerate() {
        let raw = recurrence_fn(k)?;
        let (value, beta) = super::coefficients::symmetric_step(raw, previous_a)?;
        *entry = value;
        if k > 0 {
            off_diagonal[k - 1] = beta;
        }
        previous_a = Some(raw.0);
    }

    Ok(JacobiRecurrence {
        diagonal,
        off_diagonal,
    })
}

pub(super) fn golub_welsch_zeros<R: RealScalar>(
    n: usize,
    recurrence: impl Fn(usize) -> Result<(R, R, R), SciOxideError>,
) -> Result<Vec<R>, SciOxideError> {
    let (diagonal, off_diagonal) =
        jacobi_recurrence(n, recurrence, scioxide_core::argument!("n"))?.into_parts();
    // The default options deflate relative to the machine epsilon and sort
    // the eigenvalues in ascending order.
    let values =
        HermitianEigenvaluesResult::tridiagonal(&diagonal, &off_diagonal)?.into_eigenvalues();
    values
        .iter()
        .try_for_each(|&x| finite_output(x, "eigensolver returned a nonfinite zero").map(|_| ()))?;
    Ok(values)
}

pub(super) fn chebyshev<R: RealScalar>(n: usize, second: bool) -> Result<Vec<R>, SciOxideError> {
    if n == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("n"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "cannot compute zeros for degree zero",
        ));
    }
    let order = if second {
        checked_add(n, 1, scioxide_core::quantity!("zero_angle_denominator"))?
    } else {
        n
    };
    let mut values = try_filled(n, R::ZERO, scioxide_core::quantity!("chebyshev_zeros"))?;
    for index in 0..n / 2 {
        let value = finite_arithmetic(
            chebyshev_angle::<R>(n, order, index)?.sin(),
            "zero formula is not finite",
        )?;
        values[index] = -value;
        values[n - 1 - index] = value;
    }
    Ok(values)
}

/// Returns the angle `(n − 1 − 2 index) π / (2 order)` of the Chebyshev point
/// `−sin(angle)`, for `index < n / 2`.
///
/// Mirroring the lower half keeps the points exactly symmetric, with an exact
/// zero at the center.
pub(crate) fn chebyshev_angle<R: RealScalar>(
    n: usize,
    order: usize,
    index: usize,
) -> Result<R, SciOxideError> {
    let denominator = R::try_from_usize_exact(checked_mul(
        2,
        order,
        scioxide_core::quantity!("zero_angle_denominator"),
    )?)?;
    Ok(R::try_from_usize_exact(n - 1 - 2 * index)? / denominator * R::PI)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_zeros_are_sorted_exactly_symmetric_and_centered() {
        fn check<R: RealScalar>() {
            for n in [1, 2, 3, 4, 16, 64, 255, 256] {
                for family in [
                    crate::OrthogonalFamily::<R>::ChebyshevT,
                    crate::OrthogonalFamily::ChebyshevU,
                ] {
                    let values = family.zeros(n).unwrap();
                    assert_eq!(values.len(), n);
                    assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
                    for (&left, &right) in values.iter().zip(values.iter().rev()) {
                        assert_eq!(left, -right);
                        assert!(left > -R::ONE && left < R::ONE);
                    }
                    if n % 2 == 1 {
                        assert_eq!(values[n / 2], R::ZERO);
                    }
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn values_only_solver_preserves_one_node_and_known_legendre_roots() {
        assert_eq!(
            golub_welsch_zeros(1, |_| Ok((2.0_f64, -6.0, 0.0))).unwrap(),
            [3.0]
        );
        let values = crate::OrthogonalFamily::Legendre.zeros(3).unwrap();
        for (actual, expected) in values
            .iter()
            .zip([-(0.6_f64).sqrt(), 0.0, (0.6_f64).sqrt()])
        {
            assert!((actual - expected).abs() <= 32.0 * f64::EPSILON);
        }
    }
    #[test]
    fn test_golub_welsch_rejects_any_non_positive_beta_sq() {
        assert!(matches!(
            (golub_welsch_zeros(2, |k| match k {
                0 => Ok((1.0, 0.0, 0.0)),
                1 => Ok((1.0, 0.0, -1e-12)),
                _ => Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("recurrence_index"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "invalid recurrence index",
                )),
            },))
            .unwrap_err()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Breakdown,
                ..
            }
        ));
    }

    #[test]
    fn test_golub_welsch_rejects_large_negative_beta_sq() {
        assert!(matches!(
            (golub_welsch_zeros(2, |k| match k {
                0 => Ok((1.0, 0.0, 0.0)),
                1 => Ok((1.0, 0.0, -1e-6)),
                _ => Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("recurrence_index"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "invalid recurrence index"
                )),
            },))
            .unwrap_err()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }

    #[test]
    fn golub_welsch_reports_non_finite_recurrence_callback_values() {
        assert!(matches!(
            (golub_welsch_zeros(1, |_| Ok((f64::NAN, 0.0, 0.0))))
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Callback,
                ..
            }
        ));
    }
}
