//! Normalization and multiplication data derived from classical coefficients.
use super::{BasisNormalization, OrthogonalBasisSpec};
use crate::admission::{Arithmetic, ArithmeticResult};
use crate::util::finite_callback;
use scioxide_core::backend::checked_add;
use scioxide_core::{NumericScalar, RealScalar, Scaled, SciOxideError};

/// Contributions to `(Q_{k-1}, Q_k, Q_{k+1})` from `value * x * Q_k`.
/// Keep the division before multiplication so extended execution does not
/// project an unrepresentable reciprocal into the scalar range.
pub(crate) fn multiplication<C: NumericScalar, A: Arithmetic<C>>(
    value: A::Value,
    (a, b, c): (C::Real, C::Real, C::Real),
    has_previous: bool,
) -> ArithmeticResult<[A::Value; 3]> {
    let zero = A::scalar(C::ZERO)?;
    let upper = A::checked(A::div(value, A::checked_real(a)?)?)?;
    let middle = A::checked(A::sub(
        zero,
        A::checked(A::mul(upper, A::checked_real(b)?)?)?,
    )?)?;
    let lower = if has_previous {
        A::checked(A::mul(upper, A::checked_real(c)?)?)?
    } else {
        zero
    };
    Ok([lower, middle, upper])
}

/// Symmetric multiplication data `(diagonal_k, beta_k)`, where beta_0=0.
/// Positive beta removes classical sign conventions by a diagonal similarity.
pub(super) fn symmetric_step<R: RealScalar>(
    (a, b, c): (R, R, R),
    previous_a: Option<R>,
) -> Result<(R, R), SciOxideError> {
    for value in [a, b, c] {
        finite_callback(value, "recurrence provider returned non-finite coefficient")?;
    }
    if a == R::ZERO {
        return Err(breakdown("recurrence leading coefficient is zero"));
    }
    let a = Scaled::from_value(a)?;
    let diagonal = (-Scaled::from_value(b)?).div(a)?.round()?;
    let beta = match previous_a {
        None => R::ZERO,
        Some(previous) => {
            finite_callback(
                previous,
                "recurrence provider returned non-finite coefficient",
            )?;
            if previous == R::ZERO {
                return Err(breakdown("recurrence leading coefficient is zero"));
            }
            let square = Scaled::from_value(c)?.div(Scaled::from_value(previous)?.mul(a)?)?;
            if square.significand() <= R::ZERO {
                return Err(breakdown("non-positive off-diagonal square"));
            }
            // Take the square root before projection, including when the
            // square itself lies below the scalar's subnormal range.
            let beta = square.sqrt()?.round()?;
            if beta <= R::ZERO {
                return Err(underflow("off-diagonal coefficient underflow"));
            }
            beta
        }
    };
    Ok((diagonal, beta))
}

impl<R: RealScalar> OrthogonalBasisSpec<R> {
    pub(super) fn normalized_step(self, n: usize) -> Result<(R, R, R), SciOxideError> {
        let s = Scaled::from_value;
        let raw @ (a, b, c) = self.family.recurrence(n)?;
        let scale = s(self.scale)?;
        let shift = s(self.shift)?;
        let (an, bn, cn) = match self.normalization {
            BasisNormalization::Classical => (
                s(a)?.div(scale)?,
                s(b)?.sub(s(a)?.mul(shift)?.div(scale)?)?,
                s(c)?,
            ),
            normalization => {
                let previous_a = if n == 0 {
                    None
                } else {
                    Some(self.family.recurrence(n - 1)?.0)
                };
                let (diagonal, beta) = symmetric_step(raw, previous_a)?;
                match normalization {
                    BasisNormalization::Monic => (
                        Scaled::ONE,
                        -scale.mul(s(diagonal)?)?.add(shift)?,
                        scale.square()?.mul(s(beta)?.square()?)?,
                    ),
                    BasisNormalization::Orthonormal => {
                        let next = self.family.recurrence(checked_add(
                            n,
                            1,
                            scioxide_core::quantity!("recurrence_index"),
                        )?)?;
                        let (_, next_beta) = symmetric_step(next, Some(a))?;
                        // Unit-norm scaling preserves each classical member's
                        // sign, unlike the positive Jacobi-matrix convention.
                        let upper = s(if a < R::ZERO { -next_beta } else { next_beta })?;
                        let lower = s(if previous_a.is_some_and(|a| a < R::ZERO) {
                            -beta
                        } else {
                            beta
                        })?;
                        (
                            Scaled::ONE.div(upper)?.div(scale)?,
                            -s(diagonal)?.add(shift.div(scale)?)?.div(upper)?,
                            lower.div(upper)?,
                        )
                    }
                    BasisNormalization::Classical => unreachable!(),
                }
            }
        };
        let a = an.round()?;
        if a == R::ZERO {
            return Err(underflow("recurrence leading coefficient underflow"));
        }
        Ok((a, bn.round()?, if n == 0 { R::ZERO } else { cn.round()? }))
    }
}

fn breakdown(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::Breakdown, detail)
}

fn underflow(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::Underflow, detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_coupling_does_not_project_the_square_before_sqrt() {
        let (_, beta) = symmetric_step((1e30_f32, 0.0, 1.0), Some(1e30)).unwrap();
        assert!((beta * 1e30 - 1.0).abs() <= 8.0 * f32::EPSILON);
        let (_, beta) = symmetric_step((1e300_f64, 0.0, 1.0), Some(1e300)).unwrap();
        assert!((beta * 1e300 - 1.0).abs() <= 8.0 * f64::EPSILON);
    }
}
