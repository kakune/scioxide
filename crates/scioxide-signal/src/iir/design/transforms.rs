use super::*;

impl<S: RealScalar> PoleZeroGain<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Maps an analog lowpass prototype with cutoff one to `band`.
    ///
    /// The frequency transformations substitute for `s`:
    ///
    /// ```text
    /// lowpass:   s → s/ω_c
    /// highpass:  s → ω_c/s
    /// bandpass:  s → (s² + ω₀²)/(B s)
    /// bandstop:  s → B s/(s² + ω₀²)
    /// ```
    ///
    /// where `ω_c` is the cutoff, and `ω₀ = √(ω₁ω₂)` and `B = ω₂ − ω₁` are the
    /// center and width of a band with edges `ω₁ < ω₂`. Bandpass and bandstop
    /// transformations double the number of poles. The gain is rescaled with
    /// an extended exponent range and rounded once, so intermediate overflow
    /// or underflow does not affect a representable result.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the model is digital, if an edge is not positive,
    ///   if band edges are not increasing, or if `band` is highpass or
    ///   bandstop and the model has more poles than zeros at the origin.
    /// - `NonFinite` if an edge is NaN or infinite.
    /// - `NumericalInstability` (`Overflow` or `Underflow`) if a transformed
    ///   root overflows, or if the gain is above the finite range or nonzero
    ///   and below the normal range.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{IirBand, IirFamily, iir_prototype};
    /// let prototype = iir_prototype(4, IirFamily::<f64>::Butterworth)?;
    /// let highpass = prototype.transform(IirBand::Highpass(2.0))?;
    /// // Four zeros at the origin; unit gain at high frequency, half power at 2.
    /// assert_eq!(highpass.zeros().len(), 4);
    /// assert!((highpass.response(2.0)?.norm_sqr() - 0.5).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
    /// 3rd ed., Prentice Hall, 2010, ch. 7.
    pub fn transform(&self, band: IirBand<S>) -> Result<Self, SciOxideError> {
        let band_argument = scioxide_core::argument!("band");
        if self.domain != IirDomain::Analog {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "frequency transformation requires an analog prototype",
            ));
        }
        let band = map_band(
            band,
            |x| positive_parameter(x, band_argument),
            band_argument,
        )?;
        let relative = self.poles.len() - self.zeros.len();
        let mut gain = ZpkGain::new(self.gain)?;
        let invert = matches!(band, IirBand::Highpass(_) | IirBand::Bandstop(..));
        if invert {
            let z0 = self.zeros.iter().filter(|r| r.norm() == S::ZERO).count();
            let p0 = self.poles.iter().filter(|r| r.norm() == S::ZERO).count();
            if p0 > z0 {
                return Err(invalid(
                    scioxide_core::argument!("self"),
                    "inversion produces an improper analog function",
                ));
            }
            for z in self.zeros.iter().filter(|r| r.norm() != S::ZERO) {
                gain = gain.times(-*z)?;
            }
            for p in self.poles.iter().filter(|r| r.norm() != S::ZERO) {
                gain = gain.over(-*p)?;
            }
            let scale = match band {
                IirBand::Highpass(w) => w,
                IirBand::Bandstop(a, b) => b - a,
                _ => S::ONE,
            };
            for _ in 0..z0 - p0 {
                gain = gain.scale(scale)?;
            }
        }
        let convert = |roots: &[Complex<S>]| -> Result<Vec<Complex<S>>, SciOxideError> {
            let factor = if matches!(band, IirBand::Bandpass(..) | IirBand::Bandstop(..)) {
                2
            } else {
                1
            };
            let capacity = crate::util::size(roots.len(), factor)?;
            let mut out = Vec::new();
            out.try_reserve_exact(capacity).map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("transformed_roots"),
                    capacity,
                )
            })?;
            for r in roots {
                match band {
                    IirBand::Lowpass(w) => out.push(check(r.scale(w))?),
                    IirBand::Highpass(w) => {
                        if r.norm() != S::ZERO {
                            out.push(check(c(w) / *r)?);
                        }
                    }
                    IirBand::Bandpass(a, b) | IirBand::Bandstop(a, b) => {
                        let half = (b - a) * S::HALF;
                        let center = a * b;
                        if invert && r.norm() == S::ZERO {
                            out.push(c(S::ZERO));
                            continue;
                        }
                        let q = if invert { c(half) / *r } else { r.scale(half) };
                        let d = (q * q - c(center)).sqrt();
                        let first = if (q + d).norm() > (q - d).norm() {
                            q + d
                        } else {
                            q - d
                        };
                        out.push(check(first)?);
                        out.push(check(c(center) / first)?);
                    }
                }
            }
            Ok(out)
        };
        let p = convert(&self.poles)?;
        let mut z = convert(&self.zeros)?;
        let extra = if matches!(band, IirBand::Bandstop(..)) {
            crate::util::size(relative, 2)?
        } else {
            relative
        };
        z.try_reserve_exact(extra).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("transformed_zeros"), extra)
        })?;
        match band {
            IirBand::Lowpass(w) => {
                for _ in 0..relative {
                    gain = gain.scale(w)?;
                }
            }
            IirBand::Highpass(_) => z.resize(z.len() + relative, c(S::ZERO)),
            IirBand::Bandpass(a, b) => {
                for _ in 0..relative {
                    gain = gain.scale(b - a)?;
                    z.push(c(S::ZERO));
                }
            }
            IirBand::Bandstop(a, b) => {
                let w = (a * b).sqrt();
                for _ in 0..relative {
                    z.push(Complex::new(S::ZERO, w));
                    z.push(Complex::new(S::ZERO, -w));
                }
            }
        }
        Self::new(IirDomain::Analog, &z, &p, gain.finish()?)
    }
    /// Maps an analog model to a digital one by the bilinear transform.
    ///
    /// The substitution is `s = 2fₛ (z − 1)/(z + 1)` with sample rate `fₛ`
    /// (`sample_rate`). The analog frequency `ω` and the digital frequency `f`,
    /// in multiples of Nyquist, are then related by `ω = 2fₛ tan(πf/2)`, and
    /// no prewarping is applied. Each analog pole or zero `r` maps to
    /// `(2fₛ + r)/(2fₛ − r)`, and each zero at infinity maps to `z = −1`. The
    /// gain is rescaled as in [`transform`](Self::transform).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the model is digital or `sample_rate` is not
    ///   positive.
    /// - `NonFinite` if `sample_rate` is NaN or infinite.
    /// - `NumericalInstability` if a pole or zero equals `2fₛ`, if a mapped
    ///   root overflows, or if the gain is above the finite range or nonzero
    ///   and below the normal range.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_signal::{IirFamily, iir_prototype};
    /// let analog = iir_prototype(2, IirFamily::<f64>::Butterworth)?;
    /// let digital = analog.bilinear(1.0)?;
    /// assert!(
    ///     digital
    ///         .zeros()
    ///         .iter()
    ///         .all(|z| *z == Complex::new(-1.0, 0.0))
    /// );
    /// // DC maps to z = 1, where the response stays one.
    /// assert!((digital.response(0.0)?.norm() - 1.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*,
    /// 3rd ed., Prentice Hall, 2010, ch. 7.
    pub fn bilinear(&self, sample_rate: S) -> Result<Self, SciOxideError> {
        if self.domain != IirDomain::Analog {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "bilinear transformation requires an analog model",
            ));
        }
        let t = check(
            positive_parameter(sample_rate, scioxide_core::argument!("sample_rate"))? * (S::TWO),
        )?;
        let mut gain = ZpkGain::new(self.gain)?;
        for (i, p) in self.poles.iter().enumerate() {
            if let Some(z) = self.zeros.get(i) {
                gain = gain.times(c(t) - *z)?;
            }
            gain = gain.over(c(t) - *p)?;
        }
        let mut z = zeros(self.poles.len())?;
        let mut p = zeros(self.poles.len())?;
        for (i, r) in self.poles.iter().enumerate() {
            p[i] = check((c(t) + *r) / (c(t) - *r))?;
            z[i] = if let Some(r) = self.zeros.get(i) {
                check((c(t) + *r) / (c(t) - *r))?
            } else {
                c(-S::ONE)
            };
        }
        Self::new(IirDomain::Digital, &z, &p, gain.finish()?)
    }
}
