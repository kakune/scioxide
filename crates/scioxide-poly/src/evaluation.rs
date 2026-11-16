//! Checked point evaluation at a complex point, used by polynomial root finding.
use crate::horner::{HornerEvaluation, HornerState, admitted_complex_horner};
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledValue};

/// Coefficient perturbations allowed by the backward error of a
/// [`CheckedBasisEvaluation`].
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::*;
/// let basis = PolynomialRootBasisRef::Monomial;
/// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
/// assert_eq!(evaluation.backward_error_model(), PolynomialBackwardErrorModel::Componentwise);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolynomialBackwardErrorModel {
    /// Each coefficient `aᵢ` may change by a multiple of `|aᵢ|`; zero
    /// coefficients stay zero. Used for the monomial basis.
    Componentwise,
    /// Each coefficient may change by a multiple of `maxᵢ |aᵢ|`, so zero
    /// coefficients may change too. Used for orthogonal bases.
    Normwise,
}

/// Value, first two derivatives and backward error of a polynomial at one
/// complex point, from [`checked_basis_evaluation`].
///
/// When native arithmetic would overflow or underflow, the evaluation is done
/// in extended exponent range. The plain accessors round to the scalar type
/// and return an error if the result does not fit; the `scaled_*` accessors
/// return the [`ScaledValue`] without rounding. The residual and rounding
/// allowance are estimates, not rigorous bounds.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::*;
/// let basis = PolynomialRootBasisRef::Monomial;
/// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
/// assert!((evaluation.value()?.re - 2.25).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CheckedBasisEvaluation<S: RealScalar> {
    state: HornerEvaluation<Complex<S>>,
    residual: S,
    rounding: S,
    point: Complex<S>,
    model: PolynomialBackwardErrorModel,
}

impl<S: RealScalar> CheckedBasisEvaluation<S> {
    /// Returns the perturbation model of the backward error.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
    /// assert_eq!(evaluation.backward_error_model(), PolynomialBackwardErrorModel::Componentwise);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn backward_error_model(self) -> PolynomialBackwardErrorModel {
        self.model
    }

    /// Returns `p(z)`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the value overflows the scalar type.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
    /// assert_eq!(evaluation.value()?, Complex::new(2.25, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn value(self) -> Result<Complex<S>, SciOxideError> {
        self.project(|state| state.value, |state| state.value)
    }

    /// Returns `p(z)` in extended exponent range.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the native value cannot be converted.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
    /// assert_eq!(evaluation.scaled_value()?.round()?, Complex::new(2.25, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn scaled_value(self) -> Result<ScaledValue<Complex<S>>, SciOxideError> {
        self.lift(|state| state.value, |state| state.value)
    }

    /// Returns `p′(z)`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the derivative overflows the scalar type.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
    /// assert_eq!(evaluation.derivative()?, Complex::new(3.0, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative(self) -> Result<Complex<S>, SciOxideError> {
        self.project(|state| state.derivative, |state| state.derivative)
    }

    /// Returns `p′(z)` in extended exponent range.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the native value cannot be converted.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
    /// assert_eq!(evaluation.scaled_derivative()?.round()?, Complex::new(3.0, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn scaled_derivative(self) -> Result<ScaledValue<Complex<S>>, SciOxideError> {
        self.lift(|state| state.derivative, |state| state.derivative)
    }

    /// Returns `p″(z)`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the second derivative overflows the scalar
    ///   type.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
    /// assert_eq!(evaluation.second_derivative()?, Complex::new(2.0, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn second_derivative(self) -> Result<Complex<S>, SciOxideError> {
        self.project(
            |state| state.second_derivative,
            |state| state.second_derivative,
        )
    }

    /// Returns `p″(z)` in extended exponent range.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the native value cannot be converted.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
    /// assert_eq!(evaluation.scaled_second_derivative()?.round()?, Complex::new(2.0, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn scaled_second_derivative(self) -> Result<ScaledValue<Complex<S>>, SciOxideError> {
        self.lift(
            |state| state.second_derivative,
            |state| state.second_derivative,
        )
    }

    /// Returns the Newton correction `p(z) / p′(z)`, or `None` if `p′(z) = 0`.
    ///
    /// The quotient is formed in extended range when needed, so it fits even
    /// when `p(z)` or `p′(z)` alone does not.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the correction overflows the scalar type.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[-1.0_f64, 0.0, 1.0], Complex::new(2.0, 0.0), basis)?;
    /// assert_eq!(evaluation.newton_correction()?, Some(Complex::new(0.75, 0.0)));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn newton_correction(self) -> Result<Option<Complex<S>>, SciOxideError> {
        let zero = Complex::new(S::ZERO, S::ZERO);
        if let HornerEvaluation::Native(state) = self.state {
            if state.derivative == zero {
                return Ok(None);
            }
            let correction = state.value / state.derivative;
            if correction.re.is_finite()
                && correction.im.is_finite()
                && (correction != zero || state.value == zero)
            {
                return Ok(Some(correction));
            }
        }
        let derivative = self.scaled_derivative()?;
        if derivative.is_zero() {
            return Ok(None);
        }
        Ok(Some(self.scaled_value()?.div(derivative)?.round()?))
    }

    /// Returns the backward error `η = |p(z)| / e(z)` of `z` as a root.
    ///
    /// `e(z) = Σ |aᵢ| |z|ⁱ` for the monomial basis and
    /// `e(z) = maxᵢ |aᵢ| Σ |φᵢ(z)|` for an orthogonal basis `φᵢ`. `η` is the
    /// smallest relative perturbation of the coefficients, in the sense of
    /// [`backward_error_model`](Self::backward_error_model), that makes `z` an
    /// exact root. For the zero polynomial, `e(z) = 0` and `|p(z)|` is returned.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
    /// assert_eq!(evaluation.normalized_residual(), 1.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn normalized_residual(self) -> S {
        self.residual
    }

    /// Returns the backward error plus the rounding allowance.
    ///
    /// The allowance estimates the rounding error of the evaluation itself; see
    /// [`checked_basis_evaluation`]. Use the sum to compare candidate roots
    /// conservatively. It is an estimate, not an enclosure.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[-1.0_f64, 0.0, 1.0], Complex::new(1.0, 0.0), basis)?;
    /// assert!(evaluation.residual_with_roundoff() >= evaluation.normalized_residual());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn residual_with_roundoff(self) -> S {
        self.residual + self.rounding
    }

    /// Returns `true` if the backward error is at most the rounding allowance.
    ///
    /// Then `p(z)` cannot be told apart from zero in floating point, and
    /// further iteration cannot improve `z`.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[-1.0_f64, 0.0, 1.0], Complex::new(1.0, 0.0), basis)?;
    /// assert!(evaluation.roundoff_limited());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn roundoff_limited(self) -> bool {
        self.residual <= self.rounding
    }

    /// Returns the condition of `z` as a root, computed from this evaluation.
    ///
    /// The absolute condition number is `e(z) / |p′(z)|` and the relative one
    /// `e(z) / (|z| |p′(z)|)`, with the envelope `e(z)` of
    /// [`normalized_residual`](Self::normalized_residual); see
    /// [`RootConditionReport`](crate::RootConditionReport).
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a rescaled intermediate overflows.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let basis = PolynomialRootBasisRef::Monomial;
    /// let evaluation = checked_basis_evaluation(&[1.0_f64, 2.0, 1.0], Complex::new(0.5, 0.0), basis)?;
    /// let absolute = evaluation.root_condition()?.absolute();
    /// assert!(absolute.value().is_some_and(|value| (value - 0.75).abs() < 1e-13));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn root_condition(self) -> Result<crate::RootConditionReport<S>, SciOxideError> {
        let state = self.state.into_scaled()?;
        crate::root_diagnostics::condition_from_scaled(
            state.bound,
            state.value.magnitude()?,
            state.derivative.magnitude()?,
            ScaledValue::from_value(self.point)?.magnitude()?,
        )
    }

    #[inline]
    fn project(
        self,
        native: impl FnOnce(HornerState<Complex<S>, S>) -> Complex<S>,
        extended: impl FnOnce(
            HornerState<ScaledValue<Complex<S>>, Scaled<S>>,
        ) -> ScaledValue<Complex<S>>,
    ) -> Result<Complex<S>, SciOxideError> {
        match self.state {
            HornerEvaluation::Native(state) => Ok(native(state)),
            HornerEvaluation::Extended(state) => extended(state).round(),
        }
    }

    #[inline]
    fn lift(
        self,
        native: impl FnOnce(HornerState<Complex<S>, S>) -> Complex<S>,
        extended: impl FnOnce(
            HornerState<ScaledValue<Complex<S>>, Scaled<S>>,
        ) -> ScaledValue<Complex<S>>,
    ) -> Result<ScaledValue<Complex<S>>, SciOxideError> {
        match self.state {
            HornerEvaluation::Native(state) => ScaledValue::from_value(native(state)),
            HornerEvaluation::Extended(state) => Ok(extended(state)),
        }
    }
}

/// Evaluates a polynomial, its first two derivatives and its backward error at
/// a complex point `z`.
///
/// `coefficients` are monomial coefficients `a₀, …, aₙ`, or the coefficients
/// of an expansion `Σ aᵢ φᵢ(x)` in a prepared orthogonal basis; `basis` says
/// which. Monomial coefficients are evaluated by Horner's rule and orthogonal
/// expansions by Clenshaw's recurrence, in O(n) operations. Derivatives are
/// taken with respect to the physical variable `x` of the basis. When native
/// arithmetic would overflow or underflow, the whole evaluation restarts in
/// extended exponent range. Root finding uses this function for its residual
/// tests.
///
/// The result also holds the backward error of `z` as a root (see
/// [`CheckedBasisEvaluation::normalized_residual`]) and a rounding allowance
/// `γ₈ₘ` for `m` monomial coefficients or `γ₃₂ₘ` for `m` orthogonal ones,
/// with `γₖ = k ε / (1 − k ε)` and machine epsilon `ε`. When the evaluation
/// rescales to avoid underflow, the allowance also covers gradual underflow.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `coefficients` is empty.
/// - `NonFinite` if a coefficient or `z` has a NaN or infinite component.
/// - `InvalidInput` (`Inconsistent`) if there are more orthogonal coefficients
///   than the prepared degree allows.
/// - `NumericalInstability` (`IllConditioned`) if `k ε ≥ 1`, so that no
///   rounding allowance exists.
/// - `NumericalInstability` if a rescaled intermediate overflows.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::{BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};
/// use scioxide_poly::{PolynomialRootBasisRef, checked_basis_evaluation};
///
/// // Legendre P₂ with shift 2 and scale 3, monic in x, is (x − 2)² − 3; P₂(2) = −3.
/// let (family, normalization) = (OrthogonalFamily::Legendre, BasisNormalization::Monic);
/// let spec = OrthogonalBasisSpec::new(family, normalization, 2.0_f64, 3.0)?;
/// let plan = spec.prepare(2, Default::default())?;
/// let basis = PolynomialRootBasisRef::Orthogonal(&plan);
/// let result = checked_basis_evaluation(&[0.0_f64, 0.0, 1.0], Complex::new(2.0, 0.0), basis)?;
/// assert!((result.value()?.re + 3.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
/// SIAM, 2002, chapter 5.
pub fn checked_basis_evaluation<C: NumericScalar>(
    coefficients: &[C],
    point: Complex<C::Real>,
    basis: crate::PolynomialRootBasisRef<'_, C::Real>,
) -> Result<CheckedBasisEvaluation<C::Real>, SciOxideError> {
    if coefficients.is_empty() {
        return Err(crate::util::empty_error(scioxide_core::argument!(
            "coefficients"
        )));
    }
    // Natively admitted coefficients are finite, so one admission scan serves
    // validation and the monomial recurrence; the finiteness scan runs only
    // for coefficients outside the native range.
    let admitted = crate::admission::native_coefficients(coefficients.iter().copied());
    if !admitted && !coefficients.iter().all(|c| c.has_finite_components()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "coefficients must have finite components",
        ));
    }
    crate::util::finite_input(point.re, "the evaluation point must have finite components")?;
    crate::util::finite_input(point.im, "the evaluation point must have finite components")?;
    // Eight real rounding operations per complex Horner step. Orthogonal
    // recurrences and their envelope need additional forward/backward steps.
    let (model, operations) = if basis.is_monomial() {
        (PolynomialBackwardErrorModel::Componentwise, 8)
    } else {
        (PolynomialBackwardErrorModel::Normwise, 32)
    };
    let operations = coefficients.len().saturating_mul(operations);
    let rounding = crate::util::rounding_error::<C::Real>(operations)?;
    let state = match basis {
        crate::PolynomialRootBasisRef::Monomial => {
            admitted_complex_horner::<_, 2, true>(coefficients, point, admitted)?
        }
        crate::PolynomialRootBasisRef::Orthogonal(plan) => {
            plan.root_iteration_evaluation::<_, 2>(coefficients, point)?
        }
    };
    assess(state, point, model, rounding, operations)
}

/// Residual and rounding allowance from either recurrence state.
fn assess<S: RealScalar>(
    state: HornerEvaluation<Complex<S>>,
    point: Complex<S>,
    model: PolynomialBackwardErrorModel,
    gamma: S,
    operations: usize,
) -> Result<CheckedBasisEvaluation<S>, SciOxideError> {
    // `|p(z)| / bound` is the backward error of the evaluation: the smallest
    // relative coefficient perturbation (componentwise for monomials, normwise
    // for orthogonal bases) that makes `z` an exact root. A zero bound only
    // occurs for the zero polynomial, where the raw magnitude is reported.
    if let HornerEvaluation::Native(native) = state
        && let Some(magnitude) = native_magnitude(native.value)
    {
        // Native admission keeps products and sums normal (or exactly zero);
        // gradual-underflow allowance is needed only after extended projection.
        return Ok(CheckedBasisEvaluation {
            state,
            residual: if native.bound.is_zero() {
                magnitude
            } else {
                magnitude / native.bound
            },
            rounding: gamma,
            point,
            model,
        });
    }
    let scaled = state.into_scaled()?;
    let magnitude = scaled.value.magnitude()?;
    let (residual, underflow) = if scaled.bound.is_zero() {
        (magnitude.round()?, S::ZERO)
    } else {
        let allowance = Scaled::from_value(S::MIN_SUBNORMAL)?
            .mul(Scaled::from_value(S::try_from_usize_exact(operations)?)?)?
            .div(scaled.bound)?;
        (
            magnitude.div(scaled.bound)?.round()?,
            if allowance.abs_le(Scaled::from_value(S::MAX)?) {
                allowance.round()?
            } else {
                S::INFINITY
            },
        )
    };
    Ok(CheckedBasisEvaluation {
        state,
        residual,
        rounding: gamma + underflow,
        point,
        model,
    })
}

/// Native `|z|` when the squared norm is finite and normal (or exactly zero).
#[inline]
fn native_magnitude<S: RealScalar>(value: Complex<S>) -> Option<S> {
    let square = value.re * value.re + value.im * value.im;
    (square.is_finite()
        && (square >= S::MIN_POSITIVE || (value.re.is_zero() && value.im.is_zero())))
    .then(|| square.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_horner_reports_derivatives_and_backward_residual() {
        let evaluation = checked_basis_evaluation(
            &[-1.0_f64, 0.0, 1.0],
            Complex::new(1.0, 0.0),
            crate::PolynomialRootBasisRef::Monomial,
        )
        .unwrap();
        assert_eq!(evaluation.value().unwrap(), Complex::new(0.0, 0.0));
        assert_eq!(evaluation.derivative().unwrap(), Complex::new(2.0, 0.0));
        assert_eq!(
            evaluation.second_derivative().unwrap(),
            Complex::new(2.0, 0.0)
        );
        assert_eq!(evaluation.normalized_residual(), 0.0);
    }

    #[test]
    fn native_magnitude_rejects_unsafe_squares() {
        assert_eq!(native_magnitude(Complex::new(3.0_f64, 4.0)), Some(5.0));
        assert_eq!(native_magnitude(Complex::new(0.0_f64, -0.0)), Some(0.0));
        assert_eq!(native_magnitude(Complex::new(1e200_f64, 0.0)), None);
        assert_eq!(native_magnitude(Complex::new(1e-200_f64, 0.0)), None);
    }
}

#[cfg(test)]
mod basis_root_tests {
    use super::*;
    use crate::{
        BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily, PolynomialExecutionOptions,
        PolynomialRootBasisRef,
    };
    #[test]
    fn complex_coefficients_and_native_bases_have_independent_derivatives() {
        // T3=4z^3-3z; P3=(5z^3-3z)/2, with an imaginary constant.
        let z = Complex::new(0.25_f64, 0.5);
        let c = [
            Complex::new(0.0, 2.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        for (kind, a, b) in [
            (OrthogonalFamily::ChebyshevT, 4.0, 3.0),
            (OrthogonalFamily::Legendre, 2.5, 1.5),
        ] {
            let plan = OrthogonalBasisSpec::new(kind, BasisNormalization::Classical, 0.0, 1.0)
                .unwrap()
                .prepare(3, PolynomialExecutionOptions::new())
                .unwrap();
            let e =
                checked_basis_evaluation(&c, z, PolynomialRootBasisRef::Orthogonal(&plan)).unwrap();
            let expected = z * z * z * Complex::new(a, 0.0) - z * Complex::new(b, 0.0) + c[0];
            assert!((e.value().unwrap() - expected).norm() < 1e-14);
            assert!(
                (e.derivative().unwrap()
                    - (z * z * Complex::new(3.0 * a, 0.0) - Complex::new(b, 0.0)))
                .norm()
                    < 1e-14
            );
            assert!(
                (e.second_derivative().unwrap() - z * Complex::new(6.0 * a, 0.0)).norm() < 1e-14
            );
        }
        let e = checked_basis_evaluation(
            &[Complex::new(1.0, 2.0), Complex::new(2.0, -1.0)],
            z,
            crate::PolynomialRootBasisRef::Monomial,
        )
        .unwrap();
        assert_eq!(
            e.value().unwrap(),
            Complex::new(1.0, 2.0) + Complex::new(2.0, -1.0) * z
        );
    }
}
