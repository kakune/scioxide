use super::*;

impl<T> StateSpace<T>
where
    T: NumericScalar,
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Computes the frequency response `H(λ) = C (λI − A)⁻¹ B + D` at each frequency.
    ///
    /// Continuous models use `λ = iω`, with `ω` in radians per time unit.
    /// Discrete models use `λ = exp(iπf)`, with `f` in multiples of the Nyquist
    /// frequency, so the sample period does not enter. These are the
    /// conventions of [`TransferFunction::response`] in the matching
    /// [`IirDomain`]. The result holds a `p × m` matrix per frequency.
    ///
    /// This owned form prepares a [`FrequencyResponsePlan`], which reduces `A`
    /// to Hessenberg form once, and runs it with sequential execution. Reuse a
    /// plan for repeated sweeps of one model.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a frequency is NaN or infinite.
    /// - `Singular` if `λI − A` is exactly singular at a frequency, that is,
    ///   the frequency hits a pole.
    /// - `NumericalInstability` if a response value overflows.
    /// - `AllocationFailure` if the result cannot be stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    /// use scioxide_signal::{LtiTime, StateSpace};
    /// let a = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
    /// let b = Matrix::from_vec(1, 1, vec![1.0])?;
    /// let d = Matrix::zeros(1, 1)?;
    /// let model = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), b.view(), d.view())?;
    /// // H(s) = 1 / (s + 1), so H(i) = (1 − i) / 2.
    /// let response = model.frequency_response(&[1.0])?;
    /// let h = response.values()[0];
    /// assert!((h.re - 0.5).abs() < 1e-14 && (h.im + 0.5).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn frequency_response(
        &self,
        frequencies: &[T::Real],
    ) -> Result<FrequencyResponse<T::Real>, SciOxideError> {
        let plan = FrequencyResponsePlan::new(self)?;
        let mut workspace = plan.workspace(frequencies.len())?;
        let mut values = try_zeros(
            plan.workspace_requirements(frequencies.len())?.output_len(),
            scioxide_core::quantity!("frequency_response"),
        )?;
        plan.frequency_response_into(
            frequencies,
            &mut values,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(FrequencyResponse {
            frequencies: copy_slice(frequencies, scioxide_core::quantity!("frequency_grid"))?,
            values,
            outputs: self.outputs(),
            inputs: self.inputs(),
        })
    }

    /// Computes the poles of the model, the `n` eigenvalues of `A`.
    ///
    /// The eigenvalues come from
    /// [`GeneralEigenvaluesResult::compute`](scioxide_linalg::GeneralEigenvaluesResult::compute):
    /// balancing, Hessenberg reduction and shifted QR iteration, in O(n³) time.
    /// For a real model, complex poles come in exact conjugate pairs.
    ///
    /// # Errors
    ///
    /// - `ConvergenceFailure` if the QR iteration does not converge.
    /// - `NumericalInstability` if scaling an eigenvalue overflows.
    /// - `AllocationFailure` if the result cannot be stored.
    pub fn poles(&self) -> Result<Vec<Complex<T::Real>>, SciOxideError> {
        let result = scioxide_linalg::GeneralEigenvaluesResult::compute(&self.a)?;
        copy_slice(result.eigenvalues(), scioxide_core::quantity!("lti_poles"))
    }

    /// Returns the controllability matrix `[B, AB, …, Aⁿ⁻¹B]`, of shape `n × nm`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `n m` overflows `usize`.
    /// - `NumericalInstability` if a matrix product overflows.
    /// - `AllocationFailure` if the matrix cannot be stored.
    pub fn controllability_matrix(&self) -> Result<Matrix<T>, SciOxideError> {
        controllability_matrix(&self.a, &self.b)
    }

    /// Returns the observability matrix `[C; CA; …; CAⁿ⁻¹]`, of shape `np × n`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `n p` overflows `usize`.
    /// - `NumericalInstability` if a matrix product overflows.
    /// - `AllocationFailure` if the matrix cannot be stored.
    pub fn observability_matrix(&self) -> Result<Matrix<T>, SciOxideError> {
        observability_matrix(&self.a, &self.c)
    }

    /// Computes the numerical rank of the controllability matrix.
    ///
    /// The pair `(A, B)` is controllable when the rank is `n`. The rank counts
    /// the singular values of [`Self::controllability_matrix`] above the
    /// threshold of `policy`. The thin SVD costs O(n³ m) operations.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::controllability_matrix`].
    /// - `ConvergenceFailure` if the SVD does not converge.
    /// - The errors of `policy` when its threshold is evaluated.
    pub fn controllability(
        &self,
        policy: RankPolicy<T::Real>,
    ) -> Result<RankReport<T::Real>, SciOxideError> {
        scioxide_linalg::ThinSvdDecomposition::compute(&controllability_matrix(&self.a, &self.b)?)?
            .analyze_rank(policy)
    }

    /// Computes the numerical rank of the observability matrix.
    ///
    /// The pair `(A, C)` is observable when the rank is `n`. The rank counts
    /// the singular values of [`Self::observability_matrix`] above the
    /// threshold of `policy`. The thin SVD costs O(n³ p) operations.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::observability_matrix`].
    /// - `ConvergenceFailure` if the SVD does not converge.
    /// - The errors of `policy` when its threshold is evaluated.
    pub fn observability(
        &self,
        policy: RankPolicy<T::Real>,
    ) -> Result<RankReport<T::Real>, SciOxideError> {
        scioxide_linalg::ThinSvdDecomposition::compute(&observability_matrix(&self.a, &self.c)?)?
            .analyze_rank(policy)
    }
}

fn observability_matrix<T: NumericScalar>(
    a: &Matrix<T>,
    c: &Matrix<T>,
) -> Result<Matrix<T>, SciOxideError> {
    let n = a.rows();
    let p = c.rows();
    let mut output = Matrix::zeros(
        n.checked_mul(p).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("observability_matrix"))
        })?,
        n,
    )?;
    let mut block = c.clone();
    for power in 0..n {
        output.as_mut_slice()[power * p * n..(power + 1) * p * n].copy_from_slice(block.as_slice());
        if power + 1 < n {
            block = block.view().mul_mat(a)?;
        }
    }
    Ok(output)
}
