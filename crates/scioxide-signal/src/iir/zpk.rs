use super::*;

impl<S: RealScalar> PoleZeroGain<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Creates a pole-zero-gain model from zeros `z`, poles `p` and `gain`.
    ///
    /// The roots are stored as given, without cancelling common roots or
    /// reflecting unstable poles.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a zero, pole or gain component is NaN or infinite.
    /// - `InvalidInput` if there are more zeros than poles.
    pub fn new(
        domain: IirDomain,
        z: &[Complex<S>],
        p: &[Complex<S>],
        gain: Complex<S>,
    ) -> Result<Self, SciOxideError> {
        inputs(z)?;
        inputs(p)?;
        inputs(&[gain])?;
        if z.len() > p.len() {
            return Err(inconsistent(
                scioxide_core::argument!("z"),
                "more zeros than poles",
            ));
        }
        Ok(Self {
            domain,
            zeros: copy(z)?,
            poles: copy(p)?,
            gain,
        })
    }
    /// Returns the domain of the roots.
    pub fn domain(&self) -> IirDomain {
        self.domain
    }
    /// Returns the zeros.
    pub fn zeros(&self) -> &[Complex<S>] {
        &self.zeros
    }
    /// Returns the poles, including the origin poles of digital delays.
    pub fn poles(&self) -> &[Complex<S>] {
        &self.poles
    }
    /// Returns the gain `k`.
    pub fn gain(&self) -> Complex<S> {
        self.gain
    }
    /// Returns the smallest distance of a pole to the stability boundary.
    ///
    /// The distance is `−Re p` for an analog pole `p` and `1 − |p|` for a
    /// digital one. A positive margin means the filter is strictly stable, and
    /// a model without poles has an infinite margin.
    pub fn stability_margin(&self) -> S {
        self.poles
            .iter()
            .map(|p| match self.domain {
                IirDomain::Analog => -p.re,
                IirDomain::Digital => S::ONE - p.norm(),
            })
            .fold(S::INFINITY, S::min)
    }
    /// Returns whether every pole lies strictly inside the stability region.
    pub fn is_stable(&self) -> bool {
        self.stability_margin() > S::ZERO
    }
    /// Evaluates the frequency response at `frequency` from the factored form.
    ///
    /// `frequency` is interpreted as in [`TransferFunction::response`]. The
    /// product of the root factors retains extended exponents through the
    /// final quotient, without expanding polynomials.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `frequency` is NaN or infinite.
    /// - `NumericalInstability` (`Breakdown`) if `frequency` is a pole.
    /// - `NumericalInstability` (`Overflow`) if the response is not
    ///   representable.
    pub fn response(&self, frequency: S) -> Result<Complex<S>, SciOxideError> {
        let mut x = frequency_variable(self.domain, frequency)?;
        if self.domain == IirDomain::Digital {
            x = x.conj();
        }
        let x = ScaledValue::from_value(x)?;
        let mut y = ScaledValue::from_value(self.gain)?;
        for (i, p) in self.poles.iter().enumerate() {
            let denominator = x.sub(ScaledValue::from_value(*p)?)?;
            if denominator.is_zero() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "response has a pole at this frequency",
                ));
            }
            if let Some(zero) = self.zeros.get(i) {
                y = y.mul(x.sub(ScaledValue::from_value(*zero)?)?)?;
            }
            y = y.div(denominator)?;
        }
        y.round()
    }
    /// Expands the roots into a transfer function with complex coefficients.
    ///
    /// The gain multiplies the numerator. A digital numerator is delayed by the
    /// number of poles minus the number of zeros, so that both polynomials are
    /// in powers of `z⁻¹`. The coefficients of a high-order filter are
    /// ill-conditioned: filter with [`to_sos`](Self::to_sos) instead.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if an expanded coefficient
    ///   overflows.
    /// - The errors of [`TransferFunction::new`] for the expanded coefficients.
    pub fn to_transfer(&self) -> Result<TransferFunction<Complex<S>>, SciOxideError> {
        let mut b = expand(&self.zeros)?;
        for v in &mut b {
            *v = check(*v * self.gain)?;
        }
        if self.domain == IirDomain::Digital {
            let mut delayed = zeros(self.poles.len() + 1)?;
            let delay = self.poles.len() - self.zeros.len();
            delayed[delay..].copy_from_slice(&b);
            b = delayed;
        }
        TransferFunction::new(self.domain, &b, &expand(&self.poles)?)
    }
    /// Groups the roots into second-order sections with complex coefficients.
    ///
    /// Poles are taken in order of distance to the stability boundary, nearest
    /// first. Each pole is paired with the remaining pole nearest to its
    /// conjugate, real poles with real poles, and each pair takes the nearest
    /// remaining zeros. The section order is then reversed, so the poles
    /// nearest the boundary are filtered last. Each of the `K` sections gets
    /// the gain magnitude `|k|^(1/K)`, and the first section also gets the
    /// phase of `k`. Use [`SecondOrderSections::to_real`] to obtain real
    /// sections of a real filter.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a pole is not strictly stable, before or after the
    ///   section coefficients are rounded.
    /// - `NumericalInstability` if the gain per section underflows, if a
    ///   section coefficient overflows, or if zeros remain unpaired.
    pub fn to_sos(&self) -> Result<SecondOrderSections<Complex<S>>, SciOxideError> {
        if !self.is_stable() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "poles must be strictly stable",
            ));
        }
        SecondOrderSections::new(self.domain, &self.section_coefficients()?)
    }

    /// Low-order factors used by filtering and by state-space realization.
    pub(crate) fn section_coefficients(&self) -> Result<Vec<[Complex<S>; 6]>, SciOxideError> {
        let mut poles = copy(&self.poles)?;
        let mut z = copy(&self.zeros)?;
        let real_poles = self.poles.iter().filter(|p| p.im == S::ZERO).count();
        let count = (real_poles.div_ceil(2) + (self.poles.len() - real_poles).div_ceil(2)).max(1);
        let mut sections = Vec::new();
        sections.try_reserve_exact(count).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("sos"), count)
        })?;
        // Build most sensitive sections first, then run them last.
        let magnitude = ScaledValue::from_value(self.gain)?.magnitude()?;
        let per_gain = if count == 1 {
            // Its magnitude need not fit when both finite components are large.
            self.gain
        } else {
            let native_magnitude = self.gain.norm();
            let root = if native_magnitude.is_finite() {
                native_magnitude.powf(S::ONE / scalar::<S>(count)?)
            } else {
                scioxide_core::Scaled::from_log(S::ONE, magnitude.ln_abs()? / scalar::<S>(count)?)?
                    .round()?
            };
            if !magnitude.is_zero() && root == S::ZERO {
                return Err(failure("SOS gain underflow"));
            }
            c(root)
        };
        while !poles.is_empty() {
            let worst = nearest_index(&poles, |p| match self.domain {
                IirDomain::Digital => (S::ONE - p.norm()).abs(),
                IirDomain::Analog => p.re.abs(),
            });
            let p = poles.remove(worst);
            let mut pair = [p, c(S::ZERO)];
            let degree = if !poles.iter().any(|q| (p.im == S::ZERO) == (q.im == S::ZERO)) {
                1
            } else {
                2
            };
            if degree == 2 {
                let i = nearest_index(&poles, |q| {
                    if (p.im == S::ZERO) != (q.im == S::ZERO) {
                        S::INFINITY
                    } else {
                        (*q - p.conj()).norm()
                    }
                });
                pair[1] = poles.remove(i);
            }
            let zd = if degree == 1
                && p.im == S::ZERO
                && !z.iter().any(|q| q.im == S::ZERO)
                && z.len() < poles.len() + degree
            {
                0
            } else {
                z.len().min(degree)
            };
            let mut zp = [c(S::ZERO); 2];
            for j in 0..zd {
                let target = if j == 1 && zp[0].im != S::ZERO {
                    zp[0].conj()
                } else {
                    pair[j]
                };
                let real_count = z.iter().filter(|q| q.im == S::ZERO).count();
                let i = nearest_index(&z, |q| {
                    if (degree == 1 && q.im != S::ZERO)
                        || (j == 0 && zd == 2 && real_count == 1 && q.im == S::ZERO)
                        || (j == 1 && ((zp[0].im == S::ZERO) != (q.im == S::ZERO)))
                    {
                        S::INFINITY
                    } else {
                        (*q - target).norm()
                    }
                });
                zp[j] = z.remove(i);
            }
            let a = expand(&pair[..degree])?;
            let b = expand(&zp[..zd])?;
            let mut row = [c(S::ZERO); 6];
            let ao = if self.domain == IirDomain::Analog {
                2 - degree
            } else {
                0
            };
            let bo = if self.domain == IirDomain::Analog {
                2 - zd
            } else {
                degree - zd
            };
            row[3 + ao..3 + ao + a.len()].copy_from_slice(&a);
            for (i, v) in b.iter().enumerate() {
                row[bo + i] = check(*v * per_gain)?;
            }
            sections.push(row);
        }
        if !z.is_empty() {
            return Err(failure("SOS pairing left unmatched zeros"));
        }
        if sections.is_empty() {
            sections.push(if self.domain == IirDomain::Analog {
                [
                    c(S::ZERO),
                    c(S::ZERO),
                    self.gain,
                    c(S::ZERO),
                    c(S::ZERO),
                    c(S::ONE),
                ]
            } else {
                [
                    self.gain,
                    c(S::ZERO),
                    c(S::ZERO),
                    c(S::ONE),
                    c(S::ZERO),
                    c(S::ZERO),
                ]
            });
        } else {
            sections.reverse();
            let phase = if count == 1 || magnitude.is_zero() {
                c(S::ONE)
            } else {
                ScaledValue::from_value(self.gain)?
                    .mul_real(scioxide_core::Scaled::ONE.div(magnitude)?)?
                    .round()?
            };
            for v in &mut sections[0][..3] {
                *v = check(*v * phase)?;
            }
        }
        Ok(sections)
    }
}
fn nearest_index<S: RealScalar>(
    values: &[Complex<S>],
    distance: impl Fn(&Complex<S>) -> S,
) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if distance(&values[i]) < distance(&values[best]) {
            best = i;
        }
    }
    best
}
