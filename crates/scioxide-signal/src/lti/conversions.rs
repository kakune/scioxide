use super::*;

impl<T> StateSpace<T>
where
    T: NumericScalar,
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Converts the channel from input `input` to output `output` to a transfer function.
    ///
    /// With `b` column `input` of `B`, `c` row `output` of `C` and `d` the
    /// matching entry of `D`, the result is
    ///
    /// ```text
    /// H(λ) = c (λI − A)⁻¹ b + d = N(λ) / det(λI − A)
    /// ```
    ///
    /// in the [`IirDomain`] that matches the model's time domain. The
    /// denominator has degree `n` and is expanded from the eigenvalues of `A`.
    /// The numerator follows from the matrix determinant lemma
    /// `det(λI − A + b c) = det(λI − A) + c adj(λI − A) b`, using the
    /// eigenvalues of `A − b c`. Nothing is cancelled, so uncontrollable or
    /// unobservable modes remain as common factors. A model without states gives
    /// the constant `d`.
    ///
    /// The cost is O(n³) operations and O(n²) memory. The coefficients of a
    /// high-order polynomial are ill-conditioned functions of its roots, so
    /// evaluate responses of high-order models with
    /// [`Self::frequency_response`] instead.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `input ≥ m` or `output ≥ p`.
    /// - `ConvergenceFailure` if an eigenvalue iteration does not converge.
    /// - `NumericalInstability` if a coefficient overflows.
    pub fn to_transfer(
        &self,
        input: usize,
        output: usize,
    ) -> Result<TransferFunction<T>, SciOxideError> {
        if input >= self.inputs() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("lti_input_channel"),
                input,
                self.inputs(),
            ));
        }
        if output >= self.outputs() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("lti_output_channel"),
                output,
                self.outputs(),
            ));
        }
        let n = self.states();
        if n == 0 {
            return TransferFunction::new(
                domain(self.time),
                &[self.d.as_slice()[output * self.inputs() + input]],
                &[T::ONE],
            );
        }
        let denominator = characteristic_coefficients(&self.a)?;
        let direct = self.d.as_slice()[output * self.inputs() + input];
        let mut shifted = Matrix::from_vec(
            n,
            n,
            copy_slice(
                self.a.as_slice(),
                scioxide_core::quantity!("lti_shifted_matrix"),
            )?,
        )?;
        for (row, values) in shifted.as_mut_slice().chunks_exact_mut(n).enumerate() {
            for (col, value) in values.iter_mut().enumerate() {
                *value = finite(
                    *value
                        - self.b.as_slice()[row * self.inputs() + input]
                            * self.c.as_slice()[output * n + col],
                )?;
            }
        }
        // Matrix determinant lemma: det(sI-A+BC) = den + C adj(sI-A) B.
        // This also handles zero feedthrough without division by d.
        let shifted_coefficients = characteristic_coefficients(&shifted)?;
        let mut numerator = scioxide_core::backend::try_filled(
            n + 1,
            T::ZERO,
            scioxide_core::quantity!("lti_transfer_numerator"),
        )?;
        for ((value, shifted), &den) in numerator
            .iter_mut()
            .zip(shifted_coefficients)
            .zip(&denominator)
        {
            *value = scioxide_core::ScaledValue::from_value(shifted)?
                .add(
                    scioxide_core::ScaledValue::from_value(direct - T::ONE)?
                        .mul(scioxide_core::ScaledValue::from_value(den)?)?,
                )?
                .round()?;
        }
        numerator[0] = direct;
        TransferFunction::new(domain(self.time), &numerator, &denominator)
    }

    /// Converts the channel from input `input` to output `output` to zeros, poles and gain.
    ///
    /// The zeros and poles are the roots of the numerator and denominator of
    /// [`Self::to_transfer`], computed and checked by
    /// [`TransferFunction::to_zpk`] in a fresh root scope;
    /// [`to_zpk_scoped`](Self::to_zpk_scoped) runs the root solves inside the
    /// caller's solve.
    ///
    /// ```
    /// use scioxide_linalg::Matrix;
    /// use scioxide_signal::{LtiTime, StateSpace};
    /// let a = Matrix::from_vec(2, 2, vec![0.0_f64, 1.0, -2.0, -3.0])?;
    /// let b = Matrix::from_vec(2, 1, vec![0.0, 1.0])?;
    /// let c = Matrix::from_vec(1, 2, vec![1.0, 0.0])?;
    /// let d = Matrix::zeros(1, 1)?;
    /// let system = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view())?;
    /// // 1 / ((s + 1)(s + 2)) has poles -1 and -2.
    /// let zpk = system.to_zpk(0, 0)?;
    /// let mut poles: Vec<f64> = zpk.poles().iter().map(|p| p.re).collect();
    /// poles.sort_by(f64::total_cmp);
    /// assert!((poles[0] + 2.0).abs() < 1e-12 && (poles[1] + 1.0).abs() < 1e-12);
    /// let transfer = system.to_transfer(0, 0)?;
    /// assert!((zpk.response(0.5)? - transfer.response(0.5)?).norm() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::to_transfer`] and [`TransferFunction::to_zpk`].
    pub fn to_zpk(
        &self,
        input: usize,
        output: usize,
    ) -> Result<PoleZeroGain<T::Real>, SciOxideError> {
        self.to_zpk_scoped(input, output, ExecutionContext::sequential_native())
    }

    /// Converts a channel to zeros, poles and gain as [`to_zpk`](Self::to_zpk)
    /// does, as one epoch of `scope`.
    ///
    /// The root solves count toward the limits of `scope` and observe its
    /// cancellation and deadline, as described in
    /// [scoped designs](crate#scoped-designs).
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::to_transfer`] and
    ///   [`TransferFunction::to_zpk_scoped`].
    pub fn to_zpk_scoped<'scope, 'capability>(
        &self,
        input: usize,
        output: usize,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<PoleZeroGain<T::Real>, SciOxideError> {
        self.to_transfer(input, output)?.to_zpk_scoped(scope)
    }
}

impl<T> TransferFunction<T>
where
    T: NumericScalar,
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Realizes the transfer function as a single-input, single-output state-space model.
    ///
    /// The realization is the controllable canonical form. For the normalized
    /// denominator `1, a₁, …, a_n` and the numerator `b₀, …, b_n`, padded with
    /// zeros to the same length, `A` is the companion matrix with first row
    /// `−a₁, …, −a_n` and ones on the subdiagonal, `B = e₁`, `C` has entries
    /// `b_k − b₀ a_k` and `D = b₀`. The order `n` is the analog denominator
    /// degree, or the longer digital coefficient list minus one.
    ///
    /// An analog transfer function needs `LtiTime::Continuous`, and a digital
    /// one needs `LtiTime::Discrete`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `time` does not match the domain.
    /// - `InvalidInput` (`OutOfRange`) if a discrete sample period is not positive.
    /// - `NonFinite` if the sample period is NaN or infinite.
    /// - `AllocationFailure` if the model cannot be stored.
    ///
    /// # References
    ///
    /// G. F. Franklin, J. D. Powell and M. L. Workman, *Digital Control of Dynamic
    /// Systems*, 3rd ed., Addison-Wesley, 1998.
    pub fn to_state_space(&self, time: LtiTime<T::Real>) -> Result<StateSpace<T>, SciOxideError> {
        let time_argument = scioxide_core::argument!("time");
        time.validate(time_argument)?;
        match (self.domain(), time) {
            (IirDomain::Analog, LtiTime::Continuous)
            | (IirDomain::Digital, LtiTime::Discrete { .. }) => {}
            _ => {
                return Err(inconsistent(
                    time_argument,
                    "transfer and state-space time domains disagree",
                ));
            }
        }
        let n = match self.domain() {
            IirDomain::Analog => self.denominator().len() - 1,
            IirDomain::Digital => self.denominator().len().max(self.numerator().len()) - 1,
        };
        let mut denominator = scioxide_core::backend::try_filled(
            n + 1,
            T::ZERO,
            scioxide_core::quantity!("state_space_denominator"),
        )?;
        denominator[..self.denominator().len()].copy_from_slice(self.denominator());
        if n == 0 {
            let a = Matrix::zeros(0, 0)?;
            let b = Matrix::zeros(0, 1)?;
            let c = Matrix::zeros(1, 0)?;
            let d = Matrix::from_vec(1, 1, vec![self.numerator()[0]])?;
            return StateSpace::new(time, a.view(), b.view(), c.view(), d.view());
        }
        let mut coefficients = scioxide_core::backend::try_filled(
            n + 1,
            T::ZERO,
            scioxide_core::quantity!("state_space_numerator"),
        )?;
        match self.domain() {
            IirDomain::Analog => {
                let start = coefficients.len() - self.numerator().len();
                coefficients[start..].copy_from_slice(self.numerator());
            }
            IirDomain::Digital => {
                coefficients[..self.numerator().len()].copy_from_slice(self.numerator())
            }
        }
        let direct = coefficients[0];
        let mut a = Matrix::zeros(n, n)?;
        for col in 0..n {
            a.as_mut_slice()[col] = -denominator[col + 1];
        }
        for row in 1..n {
            a.as_mut_slice()[row * n + row - 1] = T::ONE;
        }
        let mut b = Matrix::zeros(n, 1)?;
        b.as_mut_slice()[0] = T::ONE;
        let mut c = Matrix::zeros(1, n)?;
        for col in 0..n {
            c.as_mut_slice()[col] = coefficients[col + 1] - direct * denominator[col + 1];
        }
        let d = Matrix::from_vec(1, 1, vec![direct])?;
        StateSpace::new(time, a.view(), b.view(), c.view(), d.view())
    }
}

impl<S> PoleZeroGain<S>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    /// Realizes the filter as a cascade of first- and second-order state-space sections.
    ///
    /// The poles and zeros are grouped into sections of degree at most two.
    /// Each section is realized in controllable canonical form by
    /// [`TransferFunction::to_state_space`], and the sections are connected in
    /// series. The model has complex matrices. Grouping avoids expanding a
    /// high-order polynomial, whose coefficients are ill-conditioned. Unstable
    /// poles are allowed.
    ///
    /// An analog filter needs `LtiTime::Continuous`, and a digital one needs
    /// `LtiTime::Discrete`.
    ///
    /// # Errors
    ///
    /// - The errors of [`TransferFunction::to_state_space`].
    /// - `NumericalInstability` if the gain per section underflows or a model
    ///   entry overflows.
    pub fn to_state_space(
        &self,
        time: LtiTime<S>,
    ) -> Result<StateSpace<Complex<S>>, SciOxideError> {
        let sections = self.section_coefficients()?;
        let mut result = TransferFunction::new(self.domain(), &[Complex::ONE], &[Complex::ONE])?
            .to_state_space(time)?;
        for row in sections {
            // Analog rows encode first-order sections with a leading zero denominator.
            let first = match self.domain() {
                IirDomain::Analog => row[3..5]
                    .iter()
                    .take_while(|value| **value == Complex::ZERO)
                    .count(),
                IirDomain::Digital => 0,
            };
            let next = TransferFunction::new(self.domain(), &row[..3], &row[3 + first..])?
                .to_state_space(time)?;
            let n = result.states();
            let m = next.states();
            let total = n.checked_add(m).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("lti_states"))
            })?;
            let mut a = Matrix::zeros(total, total)?;
            let mut b = Matrix::zeros(total, 1)?;
            let mut c = Matrix::zeros(1, total)?;
            let d1 = result.d.as_slice()[0];
            let d2 = next.d.as_slice()[0];
            for i in 0..n {
                for j in 0..n {
                    a.as_mut_slice()[i * total + j] = result.a.as_slice()[i * n + j];
                }
                b.as_mut_slice()[i] = result.b.as_slice()[i];
                c.as_mut_slice()[i] = finite(d2 * result.c.as_slice()[i])?;
            }
            for i in 0..m {
                for j in 0..n {
                    a.as_mut_slice()[(n + i) * total + j] =
                        finite(next.b.as_slice()[i] * result.c.as_slice()[j])?;
                }
                for j in 0..m {
                    a.as_mut_slice()[(n + i) * total + n + j] = next.a.as_slice()[i * m + j];
                }
                b.as_mut_slice()[n + i] = finite(next.b.as_slice()[i] * d1)?;
                c.as_mut_slice()[n + i] = next.c.as_slice()[i];
            }
            let d = Matrix::from_vec(1, 1, vec![finite(d2 * d1)?])?;
            result = StateSpace::new(time, a.view(), b.view(), c.view(), d.view())?;
        }
        Ok(result)
    }
}

fn characteristic_coefficients<T>(a: &Matrix<T>) -> Result<Vec<T>, SciOxideError>
where
    T: NumericScalar,
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    use scioxide_poly::{ApproxPolynomial, PolynomialBasis};
    let values = scioxide_linalg::GeneralEigenvaluesResult::compute(a)?;
    let real =
        T::from_components(T::Real::ZERO, T::Real::ONE).imaginary_component() == T::Real::ZERO;
    let mut coefficients = if real {
        // Real Schur returns conjugate pairs; construct real quadratic factors
        // directly rather than discarding rounded imaginary coefficients.
        let mut roots = Vec::new();
        let mut pairs = Vec::new();
        roots.try_reserve_exact(a.rows()).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("lti_real_roots"), a.rows())
        })?;
        pairs.try_reserve_exact(a.rows() / 2).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("lti_conjugate_roots"),
                a.rows() / 2,
            )
        })?;
        for &value in values.eigenvalues() {
            if value.im == T::Real::ZERO {
                roots.push(value.re);
            } else if value.im > T::Real::ZERO {
                pairs.push(value);
            }
        }
        let polynomial = ApproxPolynomial::from_conjugate_pairs(&roots, &pairs)?;
        let mut coefficients = try_zeros(
            a.rows() + 1,
            scioxide_core::quantity!("characteristic_polynomial"),
        )?;
        for (out, &value) in coefficients
            .iter_mut()
            .zip(polynomial.basis().coefficients())
        {
            *out = T::from_real(value);
        }
        coefficients
    } else {
        let polynomial = ApproxPolynomial::from_roots(values.eigenvalues())?;
        let mut coefficients = try_zeros(
            a.rows() + 1,
            scioxide_core::quantity!("characteristic_polynomial"),
        )?;
        for (out, &value) in coefficients
            .iter_mut()
            .zip(polynomial.basis().coefficients())
        {
            *out = T::from_components(value.re, value.im);
        }
        coefficients
    };
    coefficients.reverse();
    Ok(coefficients)
}

fn domain<S: RealScalar>(time: LtiTime<S>) -> IirDomain {
    match time {
        LtiTime::Continuous => IirDomain::Analog,
        LtiTime::Discrete { .. } => IirDomain::Digital,
    }
}
