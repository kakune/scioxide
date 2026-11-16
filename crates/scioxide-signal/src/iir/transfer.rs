use super::*;

impl<T: NumericScalar> TransferFunction<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Creates a transfer function from numerator `b` and denominator `a`.
    ///
    /// The coefficient order follows `domain`, as described on
    /// [`TransferFunction`]. Every coefficient is divided by `a[0]`. Leading
    /// zeros of an analog numerator are dropped before the degree check.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `InvalidInput` if `b` or `a` is empty, if `a[0]` is zero, or if an
    ///   analog numerator has a higher degree than the denominator.
    /// - `NumericalInstability` if dividing by `a[0]` overflows or turns a
    ///   nonzero coefficient into zero.
    pub fn new(domain: IirDomain, b: &[T], a: &[T]) -> Result<Self, SciOxideError> {
        inputs(b)?;
        inputs(a)?;
        if b.is_empty() {
            return Err(empty(
                scioxide_core::argument!("b"),
                "numerator must be nonempty",
            ));
        }
        if a.is_empty() {
            return Err(empty(
                scioxide_core::argument!("a"),
                "denominator must be nonempty",
            ));
        }
        if a[0] == T::ZERO {
            return Err(invalid(
                scioxide_core::argument!("a"),
                "denominator must have a nonzero leading coefficient",
            ));
        }
        let mut b = copy(b)?;
        let mut a = copy(a)?;
        let scale = a[0];
        for v in b.iter_mut().chain(a.iter_mut()) {
            let normalized = check(*v / scale)?;
            if *v != T::ZERO && normalized == T::ZERO {
                return Err(failure("transfer normalization lost a coefficient"));
            }
            *v = normalized;
        }
        if domain == IirDomain::Analog {
            let first = b.iter().position(|x| *x != T::ZERO).unwrap_or(b.len() - 1);
            b.drain(..first);
            if b.len() > a.len() {
                return Err(inconsistent(
                    scioxide_core::argument!("b"),
                    "analog numerator degree exceeds the denominator degree",
                ));
            }
        } else {
            while b.len() > 1 && b.last() == Some(&T::ZERO) {
                b.pop();
            }
            while a.len() > 1 && a.last() == Some(&T::ZERO) {
                a.pop();
            }
        }
        Ok(Self { domain, b, a })
    }
    /// Returns the numerator coefficients in the order of the domain.
    pub fn numerator(&self) -> &[T] {
        &self.b
    }
    /// Returns the denominator coefficients, normalized so that the first is one.
    pub fn denominator(&self) -> &[T] {
        &self.a
    }
    /// Returns the domain of the coefficients.
    pub fn domain(&self) -> IirDomain {
        self.domain
    }
    /// Evaluates the frequency response at `frequency`.
    ///
    /// The response is `H(iω)` for an analog filter, with `ω` in radians per
    /// time unit, and `H(exp(iπf))` for a digital one, with `f` in multiples of
    /// Nyquist. Digital frequencies are reduced modulo two before forming the
    /// angle. Horner evaluation retains extended exponents through the final
    /// quotient, including when the polynomials individually overflow or underflow.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `frequency` is NaN or infinite.
    /// - `NumericalInstability` (`Breakdown`) if the denominator vanishes at
    ///   `frequency`.
    /// - `NumericalInstability` (`Overflow`) if the response is not
    ///   representable.
    pub fn response(&self, frequency: T::Real) -> Result<Complex<T::Real>, SciOxideError> {
        let x = frequency_variable(self.domain, frequency)?;
        let (b, _) = polynomial(&self.b, x, self.domain == IirDomain::Digital)?;
        let (a, _) = polynomial(&self.a, x, self.domain == IirDomain::Digital)?;
        if a.is_zero() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "response has a pole at this frequency",
            ));
        }
        b.div(a)?.round()
    }
    /// Computes the group delay `τ = −dφ/dω` at `frequency`.
    ///
    /// `φ` is the phase of the response and `ω` the angular frequency, so `τ`
    /// is in time units for an analog filter and in samples for a digital one.
    /// `frequency` is interpreted as in [`response`](Self::response). The delay
    /// comes from the logarithmic derivatives `B′/B` and `A′/A` of numerator
    /// and denominator, without differencing the phase.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `frequency` is NaN or infinite.
    /// - `NumericalInstability` (`Breakdown`) if `frequency` is a zero or a
    ///   pole of the response, where the group delay is undefined.
    /// - `NumericalInstability` (`Overflow`) if the delay is not representable.
    pub fn group_delay(&self, frequency: T::Real) -> Result<T::Real, SciOxideError> {
        let x = frequency_variable(self.domain, frequency)?;
        check(rational_group_delay(&self.b, &self.a, x, self.domain)?)
    }
    /// Computes the zeros, poles and gain of the transfer function.
    ///
    /// A digital numerator delay becomes poles at the origin, and a digital
    /// denominator of higher degree than the numerator gives zeros at the
    /// origin. The roots of each polynomial come from the polynomial root
    /// solver, with the eigenvalues of the balanced companion matrix as a
    /// fallback. A root set `r₁, …, rₙ` of a polynomial with coefficients
    /// `a₀, …, aₙ` in ascending powers is accepted only when
    ///
    /// ```text
    /// maxⱼ |aₙ eⱼ − aⱼ| ≤ 16 n ε max(maxⱼ |aⱼ|, |aₙ| maxⱼ qⱼ)
    /// ```
    ///
    /// where `eⱼ` and `qⱼ` are the coefficients of `∏ₖ (x − rₖ)` and
    /// `∏ₖ (x + |rₖ|)` and `ε` is the machine epsilon. The roots are then the
    /// exact roots of a polynomial within this normwise backward error. The
    /// forward error depends on root conditioning: an `m`-fold root is accurate
    /// to about `ε^(1/m)`. Clustered roots, such as the `N`-fold zero at
    /// `z = −1` of a digital Butterworth filter, therefore convert but lose
    /// accuracy. The root solves run in a fresh root scope;
    /// [`to_zpk_scoped`](Self::to_zpk_scoped) runs them inside the caller's
    /// solve.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`LossOfSignificance`) if no computed root set
    ///   passes the backward-error test.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{IirDomain, TransferFunction};
    /// // (1 + z⁻¹)² / (1 − 0.25 z⁻²): a double zero at z = −1 and poles at ±0.5.
    /// let (b, a) = ([1.0_f64, 2.0, 1.0], [1.0, 0.0, -0.25]);
    /// let transfer = TransferFunction::new(IirDomain::Digital, &b, &a)?;
    /// let zpk = transfer.to_zpk()?;
    /// // A double root is accurate to about sqrt(ε).
    /// assert!(zpk.zeros().iter().all(|z| (z.re + 1.0).abs() < 1e-7));
    /// assert!((transfer.response(0.3)? - zpk.response(0.3)?).norm() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_zpk(&self) -> Result<PoleZeroGain<T::Real>, SciOxideError>
    where
        T::Real: NumericScalar<Real = T::Real>,
        Complex<T::Real>: NumericScalar<Real = T::Real>,
    {
        self.to_zpk_scoped(scioxide_core::ExecutionContext::sequential_native())
    }
    /// Computes the zeros, poles and gain as [`to_zpk`](Self::to_zpk) does,
    /// as one epoch of `scope`.
    ///
    /// The root solves count toward the limits of `scope` and observe its
    /// cancellation and deadline, as described in
    /// [scoped designs](crate#scoped-designs).
    ///
    /// # Errors
    ///
    /// - The errors of [`to_zpk`](Self::to_zpk).
    /// - `ConvergenceFailure` if the root solves exhaust the limits of `scope`.
    /// - `ExecutionStopped` on cancellation or a passed deadline.
    pub fn to_zpk_scoped<'scope, 'capability>(
        &self,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<PoleZeroGain<T::Real>, SciOxideError>
    where
        T::Real: NumericScalar<Real = T::Real>,
        Complex<T::Real>: NumericScalar<Real = T::Real>,
    {
        run_design(
            scope,
            scioxide_core::algorithm_id!("signal", "transfer_zpk", 1),
            |context| self.zpk_in(context),
        )
    }
    /// Pole-zero conversion inside an existing solve.
    pub(crate) fn zpk_in(
        &self,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PoleZeroGain<T::Real>, SciOxideError>
    where
        T::Real: NumericScalar<Real = T::Real>,
        Complex<T::Real>: NumericScalar<Real = T::Real>,
    {
        let first = self.b.iter().position(|x| *x != T::ZERO);
        let degree = self.a.len().max(self.b.len()) - 1;
        let mut a = copy(&self.a)?;
        let mut b = copy(&self.b[first.unwrap_or(self.b.len() - 1)..])?;
        if self.domain == IirDomain::Digital {
            a.try_reserve_exact(degree + 1 - a.len()).map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("iir_denominator"),
                    degree + 1,
                )
            })?;
            a.resize(degree + 1, T::ZERO);
            b.try_reserve_exact(degree + 1 - b.len()).map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("iir_numerator"),
                    degree + 1,
                )
            })?;
            b.resize(degree + 1 - first.unwrap_or(degree), T::ZERO);
        }
        let gain = Complex::new(b[0].real_component(), b[0].imaginary_component());
        a.reverse();
        b.reverse();
        let poles = roots(&a, Some(&mut *context))?;
        let z = if first.is_none() {
            Vec::new()
        } else {
            roots(&b, Some(context))?
        };
        PoleZeroGain::new(self.domain, &z, &poles, gain)
    }
}
