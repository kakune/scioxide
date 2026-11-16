use super::*;

impl<T> StateSpace<T>
where
    T: NumericScalar,
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Converts a continuous model to discrete time with sample period `h = sample_period`.
    ///
    /// [`DiscretizationMethod`] defines the discrete matrices. Zero-order hold
    /// evaluates one matrix exponential of order `n + m`, and the generalized
    /// bilinear transform one LU factorization of order `n`. Both cost O(n³)
    /// operations when `m` and `p` are at most `n`. The result has time domain
    /// `LtiTime::Discrete { sample_period }`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the model is already discrete, if
    ///   `sample_period` is not positive, or if `alpha` is outside `[0, 1]`.
    /// - `NonFinite` if `sample_period` or `alpha` is NaN or infinite.
    /// - `Singular` if `I − α h A` is singular.
    /// - `NumericalInstability` if the matrix exponential or a discrete matrix
    ///   overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    /// use scioxide_signal::{DiscretizationMethod, LtiTime, StateSpace};
    /// let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
    /// let b = Matrix::from_vec(1, 1, vec![1.0])?;
    /// let d = Matrix::zeros(1, 1)?;
    /// let model = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), b.view(), d.view())?;
    /// let discrete = model.discretize(0.5, DiscretizationMethod::ZeroOrderHold)?;
    /// // x' = −x + u held over h = 0.5 gives A_d = exp(−0.5) and B_d = 1 − exp(−0.5).
    /// let a_d = discrete.a().as_slice()[0];
    /// assert!((a_d - (-0.5_f64).exp()).abs() < 1e-12);
    /// assert!((discrete.b().as_slice()[0] - (1.0 - a_d)).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// G. F. Franklin, J. D. Powell and M. L. Workman, *Digital Control of Dynamic
    /// Systems*, 3rd ed., Addison-Wesley, 1998.
    pub fn discretize(
        &self,
        sample_period: T::Real,
        method: DiscretizationMethod<T::Real>,
    ) -> Result<Self, SciOxideError> {
        if !matches!(self.time, LtiTime::Continuous) {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "only continuous systems can be discretized",
            ));
        }
        crate::util::check_real_slice_finite(&[sample_period])?;
        if sample_period <= T::Real::ZERO {
            return Err(invalid(
                scioxide_core::argument!("sample_period"),
                "sample period must be positive",
            ));
        }
        match method {
            DiscretizationMethod::ZeroOrderHold => {
                self.discretize_zoh(sample_period, ExecutionContext::sequential_native())
            }
            DiscretizationMethod::GeneralizedBilinear { alpha } => {
                self.discretize_gbt(sample_period, alpha)
            }
        }
    }

    pub(super) fn discretize_zoh(
        &self,
        h: T::Real,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let n = self.states();
        let m = self.inputs();
        let order = n.checked_add(m).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("lti_augmented_matrix"))
        })?;
        let mut augmented = Matrix::zeros(order, order)?;
        let scale = T::from_real(h);
        for row in 0..n {
            for col in 0..n {
                augmented.as_mut_slice()[row * order + col] =
                    self.a.as_slice()[row * n + col] * scale;
            }
            for col in 0..m {
                augmented.as_mut_slice()[row * order + n + col] =
                    self.b.as_slice()[row * m + col] * scale;
            }
        }
        let plan = MatrixFunctionPlan::build(
            &augmented,
            MatrixFunction::Exp,
            &MatrixFunctionOptions::default(),
            context,
        )?;
        let mut exponential = Matrix::zeros(order, order)?;
        plan.evaluate_into(exponential.view_mut(), &mut LinalgWorkspace::new(), context)?;
        let mut a = Matrix::zeros(n, n)?;
        let mut b = Matrix::zeros(n, m)?;
        for row in 0..n {
            a.as_mut_slice()[row * n..(row + 1) * n]
                .copy_from_slice(&exponential.as_slice()[row * order..row * order + n]);
            b.as_mut_slice()[row * m..(row + 1) * m]
                .copy_from_slice(&exponential.as_slice()[row * order + n..row * order + n + m]);
        }
        Self::new(
            LtiTime::Discrete { sample_period: h },
            a.view(),
            b.view(),
            self.c.view(),
            self.d.view(),
        )
    }

    fn discretize_gbt(&self, h: T::Real, alpha: T::Real) -> Result<Self, SciOxideError> {
        crate::util::check_real_slice_finite(&[alpha])?;
        if alpha < T::Real::ZERO || alpha > T::Real::ONE {
            return Err(invalid(
                scioxide_core::argument!("method"),
                "generalized bilinear alpha must be in [0, 1]",
            ));
        }
        let n = self.states();
        let ah = T::from_real(alpha * h);
        let bh = T::from_real((T::Real::ONE - alpha) * h);
        let h_scalar = T::from_real(h);
        let mut left = Matrix::identity(n)?;
        let mut right = Matrix::identity(n)?;
        for index in 0..n * n {
            left.as_mut_slice()[index] = left.as_slice()[index] - self.a.as_slice()[index] * ah;
            right.as_mut_slice()[index] = right.as_slice()[index] + self.a.as_slice()[index] * bh;
        }
        let factor = LuDecomposition::factor(&left)?;
        let a = factor.solve_matrix(&right)?;
        let scaled_b = self.b.view().try_scale(h_scalar)?;
        let b = factor.solve_matrix(&scaled_b)?;
        let inverse = factor.inverse()?;
        let c = self.c.view().mul_mat(&inverse)?;
        let cb = self.c.view().mul_mat(&b)?;
        let mut d = self.d.clone();
        let alpha_scalar = T::from_real(alpha);
        for index in 0..d.as_slice().len() {
            d.as_mut_slice()[index] = d.as_slice()[index] + cb.as_slice()[index] * alpha_scalar;
        }
        Self::new(
            LtiTime::Discrete { sample_period: h },
            a.view(),
            b.view(),
            c.view(),
            d.view(),
        )
    }
}
