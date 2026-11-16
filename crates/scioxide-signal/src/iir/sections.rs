use super::*;

impl<T: NumericScalar> SecondOrderSections<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Creates second-order sections from rows `[b₀, b₁, b₂, a₀, a₁, a₂]`.
    ///
    /// Digital rows are in ascending powers of `z⁻¹`. Analog rows are in
    /// descending powers of `s`, and a leading zero of the denominator lowers
    /// the section order. Each row is divided by its leading nonzero
    /// denominator coefficient. The poles of each normalized row come from the
    /// quadratic formula in its cancellation-free form and must lie strictly
    /// inside the stability region.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `InvalidInput` if a denominator is zero, if `a₀` of a digital row is
    ///   zero, if an analog numerator has a higher degree than its
    ///   denominator, or if a section pole is not strictly stable.
    /// - `NumericalInstability` (`Overflow`) if normalizing a row overflows.
    pub fn new(domain: IirDomain, sections: &[[T; 6]]) -> Result<Self, SciOxideError> {
        let sections_argument = scioxide_core::argument!("sections");
        let mut result = Vec::new();
        result.try_reserve_exact(sections.len()).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("sos"), sections.len())
        })?;
        for row in sections {
            inputs(row)?;
            let mut row = *row;
            let first = if domain == IirDomain::Digital {
                0
            } else {
                row[3..].iter().position(|x| *x != T::ZERO).ok_or_else(|| {
                    invalid(sections_argument, "section denominator must be nonzero")
                })?
            };
            if domain == IirDomain::Analog && row[..first].iter().any(|x| *x != T::ZERO) {
                return Err(invalid(
                    sections_argument,
                    "analog section numerator degree exceeds the denominator degree",
                ));
            }
            if row[3 + first] == T::ZERO {
                return Err(invalid(
                    sections_argument,
                    "section denominator must have a nonzero leading coefficient",
                ));
            }
            let scale = row[3 + first];
            for x in &mut row {
                *x = check(*x / scale)?;
            }
            let z = |i: usize| Complex::new(row[i].real_component(), row[i].imaginary_component());
            let mut p = [c(T::Real::ZERO); 2];
            let degree = 2 - first;
            if degree == 2 {
                let half = c(T::Real::HALF);
                let q = z(4) * half;
                let d = check(q * q - z(5))?.sqrt();
                let first = if (-q + d).norm() > (-q - d).norm() {
                    -q + d
                } else {
                    -q - d
                };
                p = if first.norm() == T::Real::ZERO {
                    [first, first]
                } else {
                    [first, z(5) / first]
                };
            } else if degree == 1 {
                p[0] = -z(5);
            }
            for root in &p[..degree] {
                check(*root)?;
            }
            if p[..degree].iter().any(|p| match domain {
                IirDomain::Analog => p.re >= T::Real::ZERO,
                IirDomain::Digital => p.norm() >= T::Real::ONE,
            }) {
                return Err(invalid(
                    sections_argument,
                    "rounded section poles must be strictly stable",
                ));
            }
            result.push(row);
        }
        Ok(Self {
            domain,
            sections: result,
        })
    }
    /// Returns the domain of the sections.
    pub fn domain(&self) -> IirDomain {
        self.domain
    }
    /// Returns the normalized coefficient rows.
    pub fn sections(&self) -> &[[T; 6]] {
        &self.sections
    }
    /// Computes the zeros, poles and gain of the cascade, section by section.
    ///
    /// Each section is factored and checked as in [`TransferFunction::to_zpk`],
    /// so no high-order polynomial is expanded. The section gains are
    /// multiplied with an extended exponent range and rounded once. The cost
    /// is `O(L)` operations for `L` sections. The root solves run in a fresh
    /// root scope; [`to_zpk_scoped`](Self::to_zpk_scoped) runs them inside the
    /// caller's solve.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`LossOfSignificance`) if no computed root set
    ///   of a section passes the backward-error test.
    /// - `NumericalInstability` (`Overflow` or `Underflow`) if the product of
    ///   the section gains is above the finite range or nonzero and below the
    ///   normal range.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{IirDomain, SecondOrderSections};
    /// let sos = SecondOrderSections::new(
    ///     IirDomain::Digital,
    ///     &[
    ///         [0.5_f64, 0.5, 0.0, 1.0, -0.5, 0.0],
    ///         [1.0, 0.0, 0.0, 1.0, 0.0, -0.25],
    ///     ],
    /// )?;
    /// let zpk = sos.to_zpk()?;
    /// // z^2 / (z^2 - 0.25) contributes two zeros at the origin.
    /// assert_eq!((zpk.zeros().len(), zpk.poles().len()), (3, 3));
    /// assert!((zpk.response(0.2)? - sos.response(0.2)?).norm() < 1e-14);
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
            scioxide_core::algorithm_id!("signal", "sections_zpk", 1),
            |context| self.zpk_in(context),
        )
    }
    fn zpk_in(
        &self,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<PoleZeroGain<T::Real>, SciOxideError>
    where
        T::Real: NumericScalar<Real = T::Real>,
        Complex<T::Real>: NumericScalar<Real = T::Real>,
    {
        let mut z = Vec::new();
        let mut p = Vec::new();

        let mut gain = ZpkGain::new(c(T::Real::ONE))?;
        let count = crate::util::size(self.sections.len(), 2)?;
        z.try_reserve_exact(count).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("sos_zeros"), count)
        })?;
        p.try_reserve_exact(count).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("sos_poles"), count)
        })?;
        for row in &self.sections {
            let first = if self.domain == IirDomain::Analog {
                row[3..].iter().position(|x| *x != T::ZERO).ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("self"),
                        "section denominator must be nonzero",
                    )
                })?
            } else {
                0
            };
            let tf = TransferFunction::new(self.domain, &row[..3], &row[3 + first..])?;
            let section = tf.zpk_in(context)?;
            gain = gain.times(section.gain)?;
            z.extend_from_slice(&section.zeros);
            p.extend_from_slice(&section.poles);
        }
        PoleZeroGain::new(self.domain, &z, &p, gain.finish()?)
    }
    /// Computes the group delay of the cascade as the sum of the section delays.
    ///
    /// The frequency and the units of the result follow
    /// [`TransferFunction::group_delay`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `frequency` is NaN or infinite.
    /// - `NumericalInstability` (`Breakdown`) if `frequency` is a zero or a
    ///   pole of a section.
    /// - `NumericalInstability` (`Overflow`) if the delay is not representable.
    pub fn group_delay(&self, frequency: T::Real) -> Result<T::Real, SciOxideError> {
        let x = frequency_variable(self.domain, frequency)?;
        self.sections.iter().try_fold(T::Real::ZERO, |delay, row| {
            check(delay + rational_group_delay(&row[..3], &row[3..], x, self.domain)?)
        })
    }
    /// Converts to real sections by dropping imaginary parts within `tolerance`.
    ///
    /// Every coefficient `x` must satisfy `|Im x| ≤ atol + rtol·|Re x|`, where
    /// `atol` and `rtol` are the absolute and relative parts of `tolerance`.
    /// The real parts then form new sections, which are checked as in
    /// [`SecondOrderSections::new`].
    ///
    /// # Errors
    ///
    /// - `DomainError` (`OutsideDomain`) if an imaginary part exceeds the
    ///   tolerance.
    /// - The errors of [`SecondOrderSections::new`] for the real rows.
    pub fn to_real(
        &self,
        tolerance: scioxide_core::ResidualTolerance<T::Real>,
    ) -> Result<SecondOrderSections<T::Real>, SciOxideError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.sections.len()).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("real_sos"),
                self.sections.len(),
            )
        })?;
        for row in &self.sections {
            let mut r = [T::Real::ZERO; 6];
            for (i, x) in row.iter().enumerate() {
                if !tolerance
                    .is_satisfied(x.imaginary_component().abs(), x.real_component().abs())?
                {
                    return Err(SciOxideError::domain_error(
                        scioxide_core::argument!("result"),
                        scioxide_core::DomainReason::OutsideDomain,
                        "sections have imaginary parts beyond the tolerance",
                    ));
                }
                r[i] = x.real_component();
            }
            rows.push(r);
        }
        SecondOrderSections::new(self.domain, &rows)
    }
    /// Expands the cascade into one transfer function.
    ///
    /// The section polynomials are multiplied out, so the coefficients of a
    /// high-order cascade are ill-conditioned.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if an expanded coefficient
    ///   overflows.
    /// - `NumericalInstability` (`Underflow`) if an expanded analog denominator
    ///   underflows to zero.
    /// - The errors of [`TransferFunction::new`] for the expanded coefficients.
    pub fn to_transfer(&self) -> Result<TransferFunction<T>, SciOxideError> {
        let mut b = zeros(1)?;
        b[0] = T::ONE;
        let mut a = copy(&b)?;
        for row in &self.sections {
            b = product(&b, &row[..3])?;
            a = product(&a, &row[3..])?;
        }
        if self.domain == IirDomain::Analog {
            let first = a.iter().position(|x| *x != T::ZERO).ok_or_else(|| {
                SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "cascade denominator underflowed to zero",
                )
            })?;
            a.drain(..first);
        }
        TransferFunction::new(self.domain, &b, &a)
    }
    /// Evaluates the frequency response as the product of the section responses.
    ///
    /// `frequency` is interpreted as in [`TransferFunction::response`].
    /// The cascade retains extended exponents until its final response is rounded.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `frequency` is NaN or infinite.
    /// - `NumericalInstability` (`Breakdown`) if `frequency` is a pole.
    /// - `NumericalInstability` (`Overflow`) if the response is not
    ///   representable.
    pub fn response(&self, frequency: T::Real) -> Result<Complex<T::Real>, SciOxideError> {
        let x = frequency_variable(self.domain, frequency)?;
        let mut y = ScaledValue::ONE;
        for row in &self.sections {
            let (b, _) = polynomial(&row[..3], x, self.domain == IirDomain::Digital)?;
            let (a, _) = polynomial(&row[3..], x, self.domain == IirDomain::Digital)?;
            if a.is_zero() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "response has a pole at this frequency",
                ));
            }
            y = y.mul(b)?.div(a)?;
        }
        y.round()
    }
}
fn product<T: NumericScalar>(a: &[T], b: &[T]) -> Result<Vec<T>, SciOxideError> {
    // Array order is immaterial for convolution; retain exact output padding.
    let left = ApproxPolynomial::monomial(copy(a)?)?;
    let right = ApproxPolynomial::monomial(copy(b)?)?;
    let result = left.try_mul(&right)?;
    let mut out = zeros(add(a.len(), b.len() - 1)?)?;
    out[..result.basis().coefficients().len()].copy_from_slice(result.basis().coefficients());
    Ok(out)
}
