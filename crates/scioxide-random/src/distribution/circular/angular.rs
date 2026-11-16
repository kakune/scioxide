use super::*;
use crate::distribution::LawOptions;
use crate::{Cdf, InverseSurvival, LogCdf, LogSurvival, Quantile, Support, Survival};
use scioxide_core::ExtendedInterval;

/// Von Mises distribution on the circle with location `μ` and concentration `κ ≥ 0`.
///
/// ```text
/// f(x) = exp(κ cos(x − μ)) / (2π I₀(κ))
/// ```
///
/// where `I₀` is the modified Bessel function of the first kind. The density is
/// `2π`-periodic. The CDF, quantiles, moments and samples refer to the
/// principal interval `[μ − π, μ + π]`; [`Self::winding_cdf`] extends the CDF
/// to the real line. `κ = 0` is the uniform distribution on the circle. Draws use
/// the rejection method of Best and Fisher (1979).
///
/// ```
/// use scioxide_random::{
///     Cdf, Density, Distribution, Quantile, SeedableRng, SplitMix64, Support, VonMises,
/// };
/// use core::f64::consts::TAU;
/// let law = VonMises::<f64>::new(0.0, 2.0)?;
/// assert!((law.pdf(1.0)? - law.pdf(1.0 + TAU)?).abs() < 1e-15);
/// let x = law.quantile(0.8)?;
/// assert!((law.cdf(x)? - 0.8).abs() < 1e-12);
/// assert!(law.mean_resultant_length()? > 0.0);
/// let mut rng = SplitMix64::from_seed(71);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. J. Best and N. I. Fisher, "Efficient simulation of the von Mises
/// distribution", *Appl. Statist.* 28(2), 1979, 152–157.
#[derive(Debug, Clone)]
pub struct VonMises<S: RealScalar> {
    location: S,
    kappa: f64,
    log_peak: f64,
    resultant: f64,
    deficit: f64,
    angle: AngleKernel,
    options: LawOptions<S>,
}
impl<S: RealScalar> VonMises<S> {
    /// Creates the law with location `μ = location` (radians) and `κ = concentration`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `location` or `concentration` is NaN or infinite.
    /// - `DomainError` if `concentration < 0`, or if `location ± π` rounds to `location`.
    /// - `ConvergenceFailure` if the Bessel function evaluation exhausts the work limit.
    pub fn new(location: S, concentration: S) -> Result<Self, SciOxideError> {
        Self::with_options(location, concentration, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is out of range.
    pub fn with_options(
        location: S,
        concentration: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(location, "location")?;
        let kappa =
            nonnegative(concentration, scioxide_core::argument!("concentration"))?.into_f64();
        let mu = location.into_f64();
        let low: S = number(mu - PI)?;
        let high: S = number(mu + PI)?;
        if low >= location || high <= location {
            return Err(crate::law::domain(
                scioxide_core::argument!("location"),
                "location cannot resolve a principal angular interval",
            ));
        }
        let mut left = options.work.get();
        let i0 = scaled_bessel(0.0, kappa, &mut left)?;
        let i1 = scaled_bessel(1.0, kappa, &mut left)?;
        crate::law::positive_result(i0)?;
        let (resultant, deficit) = if kappa > 1e8 {
            let deficit = 0.5 / kappa * (1.0 + 0.25 / kappa);
            (1.0 - deficit, deficit)
        } else {
            let resultant = i1 / i0;
            (resultant, 1.0 - resultant)
        };
        Ok(Self {
            location,
            kappa,
            log_peak: -crate::distribution::special::log_two_pi::<f64>() - i0.ln(),
            resultant,
            deficit,
            angle: AngleKernel::new(kappa),
            options,
        })
    }
    /// Returns the location `μ` in radians.
    pub fn location(&self) -> S {
        self.location
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
    /// Returns the mean resultant length `I₁(κ)/I₀(κ)`, the length of `E[(cos X, sin X)]`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the value is outside the range of `S`.
    pub fn mean_resultant_length(&self) -> Result<S, SciOxideError> {
        number(self.resultant)
    }
    /// Returns the circular variance `1 − I₁(κ)/I₀(κ)`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the value is outside the range of `S`.
    pub fn circular_variance(&self) -> Result<S, SciOxideError> {
        number(self.deficit)
    }
    /// Returns the mean direction `μ`.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `κ = 0`: the uniform circle has no mean direction.
    pub fn mean_direction(&self) -> Result<S, SciOxideError> {
        if self.kappa == 0.0 {
            return Err(crate::law::undefined_moment(
                "uniform circle has no unique mean direction",
            ));
        }
        Ok(self.location)
    }
    /// Computes the CDF lifted to the real line: `F(x) + k` when `x` lies `k` full
    /// turns from the principal interval.
    ///
    /// The value increases by 1 per revolution, so it is a probability only
    /// on the principal interval.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` is NaN or infinite.
    /// - `ConvergenceFailure` if the tail quadrature exhausts the work limit.
    pub fn winding_cdf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let offset = x.into_f64() - self.location.into_f64();
        let (sin, cos) = offset.sin_cos();
        let theta = sin.atan2(cos);
        let winding = ((offset - theta) / TAU).round();
        number(winding + self.tail(theta, false, &mut self.options.work.get())?.exp())
    }
    fn log_density(&self, theta: f64) -> f64 {
        self.log_peak - self.kappa * (2.0 * (theta * 0.5).sin().powi(2))
    }
    // Direct smaller tail; rational rescaling resolves the mass near the lower
    // integration endpoint even when kappa is large or the tail is remote.
    fn small_tail(&self, theta: f64, left: &mut usize) -> Result<f64, SciOxideError> {
        if theta == 0.0 {
            return Ok(-core::f64::consts::LN_2);
        }
        if self.kappa == 0.0 {
            return Ok(((PI - theta) / TAU).ln());
        }
        let width = 1.0 / self.kappa.sqrt().max(self.kappa * theta.sin()).max(1.0);
        let end = (PI - theta) / (width + PI - theta);
        let integral = crate::distribution::quadrature::positive_integral(
            0.0,
            end,
            self.options.relative(),
            left,
            &|t: f64| {
                let step = width * t / (1.0 - t);
                let difference = 2.0 * (theta + step * 0.5).sin() * (step * 0.5).sin();
                Ok(-self.kappa * difference - 2.0 * (-t).ln_1p())
            },
            None,
        )?;
        Ok(self.log_density(theta) + width.ln() + integral)
    }
    fn tail(&self, theta: f64, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        if theta <= -PI {
            return Ok(if upper { 0.0 } else { f64::NEG_INFINITY });
        }
        if theta >= PI {
            return Ok(if upper { f64::NEG_INFINITY } else { 0.0 });
        }
        let small = crate::law::log_probability(self.small_tail(theta.abs(), left)?)?;
        Ok(if upper == (theta >= 0.0) {
            small
        } else {
            log1mexp(small)?
        })
    }
    fn log_tail(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        // Compare against rounded public endpoints so endpoint quantiles are exact.
        let (low, high) = self.bounds()?;
        if x <= low {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        if x >= high {
            return Ok(if upper { S::NEG_INFINITY } else { S::ZERO });
        }
        number(finite(self.tail(
            x.into_f64() - self.location.into_f64(),
            upper,
            left,
        )?)?)
    }
    fn bounds(&self) -> Result<(S, S), SciOxideError> {
        Ok((
            number(self.location.into_f64() - PI)?,
            number(self.location.into_f64() + PI)?,
        ))
    }
    fn inverse(&self, p: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        probability(p, scioxide_core::argument!("p"))?;
        let (low, high) = self.bounds()?;
        if p == S::ZERO {
            return Ok(if upper { high } else { low });
        }
        if p == S::ONE {
            return Ok(if upper { low } else { high });
        }
        if p == S::HALF {
            return Ok(self.location);
        }
        // A private adapter shares the complete work ledger across root callbacks.
        let adapter = TailLedger {
            law: self,
            left: core::cell::Cell::new(*left),
            scale: self.kappa.sqrt().max(1.0),
        };
        let p = p.into_f64();
        let (logp, upper) = if p > 0.5 {
            ((-p).ln_1p(), !upper)
        } else {
            (p.ln(), upper)
        };
        let options = crate::inversion::law_options(self.options.work)?;
        let report = crate::invert_log_probability_with_options(&adapter, logp, upper, &options);
        *left = adapter.left.get();
        let report = report?;
        if !report.root().is_success() {
            return Err(crate::law::iteration_limit(report.evaluations()));
        }
        let coordinate = report.value();
        let theta = PI * (coordinate / adapter.scale).tanh();
        let value = number(self.location.into_f64() + theta)?;
        if value <= low.next_after(S::INFINITY) || value >= high.next_after(S::NEG_INFINITY) {
            return Err(scioxide_core::SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "interior circular quantile rounded to a support endpoint",
            ));
        }
        Ok(value)
    }
    fn linear_variance(&self, left: &mut usize) -> Result<S, SciOxideError> {
        // Integrate only the positive half; symmetry supplies the factor two.
        let width = 1.0 / self.kappa.sqrt().max(1.0);
        let end = PI / (width + PI);
        let value = crate::distribution::quadrature::positive_integral(
            0.0,
            end,
            self.options.relative(),
            left,
            &|t: f64| {
                let theta = width * t / (1.0 - t);
                let scaled = t / (1.0 - t);
                Ok(
                    -self.kappa * (2.0 * (theta * 0.5).sin().powi(2)) + 2.0 * scaled.ln()
                        - 2.0 * (-t).ln_1p(),
                )
            },
            None,
        )?;
        number((value + self.log_peak + 3.0 * width.ln() + core::f64::consts::LN_2).exp())
    }
}
struct TailLedger<'a, S: RealScalar> {
    law: &'a VonMises<S>,
    left: core::cell::Cell<usize>,
    // y=sqrt(max(kappa,1))*atanh(theta/pi) has true unbounded support and
    // resolves concentrated mass in O(1) coordinates. Generic inversion can
    // discover its bracket by checked expansion instead of halving pi hundreds
    // of times. This is a monotone coordinate transform, not a tail approximation.
    scale: f64,
}
impl<S: RealScalar> Support<f64> for TailLedger<'_, S> {
    fn support(&self) -> Result<ExtendedInterval<f64>, SciOxideError> {
        Ok(ExtendedInterval::entire())
    }
}
impl<S: RealScalar> LogCdf<f64> for TailLedger<'_, S> {
    fn log_cdf(&self, x: f64) -> Result<f64, SciOxideError> {
        let mut left = self.left.get();
        crate::law::charge(&mut left)?;
        let result = self
            .law
            .tail(PI * (x / self.scale).tanh(), false, &mut left);
        self.left.set(left);
        result
    }
}
impl<S: RealScalar> LogSurvival<f64> for TailLedger<'_, S> {
    fn log_sf(&self, x: f64) -> Result<f64, SciOxideError> {
        let mut left = self.left.get();
        crate::law::charge(&mut left)?;
        let result = self.law.tail(PI * (x / self.scale).tanh(), true, &mut left);
        self.left.set(left);
        result
    }
}

/// Von Mises distribution as a law on the real interval `[μ − π, μ + π]`.
///
/// The density is the [`VonMises`] density on that interval and zero outside it.
///
/// ```
/// use scioxide_random::{
///     Cdf, Density, Distribution, SeedableRng, SplitMix64, VonMisesLine,
/// };
/// use core::f64::consts::PI;
/// let law = VonMisesLine::<f64>::new(0.0, 2.0)?;
/// assert_eq!(law.pdf(4.0)?, 0.0);
/// assert!((law.cdf(0.0)? - 0.5).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(72);
/// assert!(law.sample(&mut rng)?.abs() <= PI);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct VonMisesLine<S: RealScalar>(VonMises<S>);
impl<S: RealScalar> VonMisesLine<S> {
    /// Creates the law with location `μ = location` and `κ = concentration`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `location` or `concentration` is NaN or infinite.
    /// - `DomainError` if `concentration < 0`, or if `location ± π` rounds to `location`.
    /// - `ConvergenceFailure` if the Bessel function evaluation exhausts the work limit.
    pub fn new(location: S, concentration: S) -> Result<Self, SciOxideError> {
        Self::with_options(location, concentration, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is out of range.
    pub fn with_options(
        location: S,
        concentration: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self(VonMises::with_options(
            location,
            concentration,
            options,
        )?))
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.0.options
    }
}
impl<S: RealScalar> LogDensity<S> for VonMises<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        // Core's full-range trigonometric reduction preserves finite large
        // arguments; reducing by the rounded constant TAU first would not.
        number(finite(
            self.log_density(x.into_f64() - self.location.into_f64()),
        )?)
    }
}
impl<S: RealScalar> LogDensity<S> for VonMisesLine<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        if !self.contains(x)? {
            return Ok(S::NEG_INFINITY);
        }
        number(finite(
            self.0
                .log_density(x.into_f64() - self.0.location.into_f64()),
        )?)
    }
}
macro_rules! von_capabilities {
    ($name:ident, $kernel:ident) => {
        impl<S: RealScalar> $name<S> {
            fn kernel(&self) -> &VonMises<S> {
                $kernel(self)
            }
        }
        impl<S: RealScalar> Density<S> for $name<S> {
            fn pdf(&self, x: S) -> Result<S, SciOxideError> {
                finite(self.log_pdf(x)?.exp())
            }
        }
        impl<S: RealScalar> Support<S> for $name<S> {
            fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
                let (a, b) = self.kernel().bounds()?;
                ExtendedInterval::new(Some(a), Some(b))
            }
        }
        impl<S: RealScalar> LogCdf<S> for $name<S> {
            fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
                self.kernel()
                    .log_tail(x, false, &mut self.kernel().options.work_budget().get())
            }

            fn log_cdf_scoped(
                &self,
                x: S,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                let k = self.kernel();
                crate::law::with_work(context, k.options.work_budget(), |left| {
                    k.log_tail(x, false, left)
                })
            }
        }
        impl<S: RealScalar> LogSurvival<S> for $name<S> {
            fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
                self.kernel()
                    .log_tail(x, true, &mut self.kernel().options.work_budget().get())
            }

            fn log_sf_scoped(
                &self,
                x: S,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                let k = self.kernel();
                crate::law::with_work(context, k.options.work_budget(), |left| {
                    k.log_tail(x, true, left)
                })
            }
        }
        impl<S: RealScalar> Cdf<S> for $name<S> {
            fn cdf(&self, x: S) -> Result<S, SciOxideError> {
                Ok(self.log_cdf(x)?.exp())
            }
        }
        impl<S: RealScalar> Survival<S> for $name<S> {
            fn sf(&self, x: S) -> Result<S, SciOxideError> {
                Ok(self.log_sf(x)?.exp())
            }
        }
        impl<S: RealScalar> Quantile<S> for $name<S> {
            fn quantile(&self, p: S) -> Result<S, SciOxideError> {
                self.kernel()
                    .inverse(p, false, &mut self.kernel().options.work_budget().get())
            }

            fn quantile_scoped(
                &self,
                p: S,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                let k = self.kernel();
                crate::law::with_work(context, k.options.work_budget(), |left| {
                    k.inverse(p, false, left)
                })
            }
        }
        impl<S: RealScalar> InverseSurvival<S> for $name<S> {
            fn isf(&self, p: S) -> Result<S, SciOxideError> {
                self.kernel()
                    .inverse(p, true, &mut self.kernel().options.work_budget().get())
            }

            fn isf_scoped(
                &self,
                p: S,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                let k = self.kernel();
                crate::law::with_work(context, k.options.work_budget(), |left| {
                    k.inverse(p, true, left)
                })
            }
        }
        impl<S: RealScalar> Mean<S> for $name<S> {
            fn mean(&self) -> Result<S, SciOxideError> {
                Ok(self.kernel().location)
            }
        }
        impl<S: RealScalar> Variance<S> for $name<S> {
            fn variance(&self) -> Result<S, SciOxideError> {
                self.kernel()
                    .linear_variance(&mut self.kernel().options.work_budget().get())
            }

            fn variance_scoped(
                &self,
                context: &mut scioxide_core::SolveContext<'_, '_>,
            ) -> Result<S, SciOxideError> {
                let k = self.kernel();
                crate::law::with_work(context, k.options.work_budget(), |left| {
                    k.linear_variance(left)
                })
            }
        }
        impl<S: RealScalar> Entropy<S> for $name<S> {
            fn entropy(&self) -> Result<S, SciOxideError> {
                let k = self.kernel();
                number(-k.log_peak + k.kappa * k.deficit)
            }
        }
        impl<S: RealScalar> Distribution<S> for $name<S> {
            scalar_batches!();
            fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
                let k = self.kernel();
                let theta = k
                    .angle
                    .sample(&mut crate::distribution::words::Words::widening(
                        rng,
                        k.options.sample_words.get(),
                    ))?;
                number(k.location.into_f64() + theta)
            }
        }
    };
}
fn circle<S: RealScalar>(x: &VonMises<S>) -> &VonMises<S> {
    x
}
fn line<S: RealScalar>(x: &VonMisesLine<S>) -> &VonMises<S> {
    &x.0
}
von_capabilities!(VonMises, circle);
von_capabilities!(VonMisesLine, line);

/// Wrapped Cauchy distribution with location `μ` and mean resultant length `0 ≤ ρ < 1`.
///
/// ```text
/// f(x) = (1 − ρ²) / (2π (1 + ρ² − 2ρ cos(x − μ))),   μ ≤ x ≤ μ + 2π
/// ```
///
/// The support is the interval `[μ, μ + 2π]`, so the mode is at both ends and
/// the linear mean is `μ + π`. `ρ = 0` is the uniform distribution on the circle.
/// Draws invert the CDF.
///
/// ```
/// use scioxide_random::{
///     Cdf, Distribution, Mean, Quantile, SeedableRng, SplitMix64, Support,
///     WrappedCauchy,
/// };
/// use core::f64::consts::PI;
/// let law = WrappedCauchy::<f64>::new(0.0, 0.5)?;
/// assert!((law.cdf(PI)? - 0.5).abs() < 1e-15);
/// let x = law.quantile(0.1)?;
/// assert!((law.cdf(x)? - 0.1).abs() < 1e-14);
/// assert!((law.mean()? - PI).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(73);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct WrappedCauchy<S: RealScalar> {
    location: S,
    rho: f64,
    options: LawOptions<S>,
}
impl<S: RealScalar> WrappedCauchy<S> {
    /// Creates the law with location `μ = location` (radians) and `ρ = rho`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `location` or `rho` is NaN or infinite.
    /// - `DomainError` if `rho` is outside `[0, 1)`, or if `location + 2π` rounds to `location`.
    pub fn new(location: S, rho: S) -> Result<Self, SciOxideError> {
        Self::with_options(location, rho, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is out of range.
    pub fn with_options(
        location: S,
        rho: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        argument(location, "location")?;
        probability(rho, scioxide_core::argument!("rho"))?;
        if rho == S::ONE {
            return Err(crate::law::domain(
                scioxide_core::argument!("rho"),
                "wrapped Cauchy rho must be less than one",
            ));
        }
        let high: S = number(location.into_f64() + TAU)?;
        if high <= location {
            return Err(crate::law::domain(
                scioxide_core::argument!("location"),
                "location cannot resolve wrapped interval",
            ));
        }
        Ok(Self {
            location,
            rho: rho.into_f64(),
            options,
        })
    }
    /// Returns the [`LawOptions`] given at construction.
    pub fn options(&self) -> LawOptions<S> {
        self.options
    }
    /// Returns the mean resultant length `ρ`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if `ρ` is outside the range of `S`.
    pub fn mean_resultant_length(&self) -> Result<S, SciOxideError> {
        number(self.rho)
    }
    /// Returns the circular variance `1 − ρ`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the value is outside the range of `S`.
    pub fn circular_variance(&self) -> Result<S, SciOxideError> {
        number(1.0 - self.rho)
    }
    /// Returns the mean direction `μ`.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `ρ = 0`: the uniform circle has no mean direction.
    pub fn mean_direction(&self) -> Result<S, SciOxideError> {
        if self.rho == 0.0 {
            return Err(crate::law::undefined_moment(
                "uniform circle has no unique mean direction",
            ));
        }
        Ok(self.location)
    }
    fn tail(&self, x: S, upper: bool) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        let high: S = number(self.location.into_f64() + TAU)?;
        if x <= self.location {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        if x >= high {
            return Ok(if upper { S::NEG_INFINITY } else { S::ZERO });
        }
        let theta = x.into_f64() - self.location.into_f64();
        let upper_half = theta > PI;
        let small = if upper_half {
            high.into_f64() - x.into_f64()
        } else {
            theta
        };
        let h = small * 0.5;
        let numerator = (1.0 + self.rho) * h.sin();
        let denominator = (1.0 - self.rho) * h.cos();
        let log_small = (numerator.atan2(denominator) / PI).ln();
        number(if upper == upper_half {
            log_small
        } else {
            log1mexp(log_small)?
        })
    }
    fn inverse(&self, p: S, upper: bool) -> Result<S, SciOxideError> {
        probability(p, scioxide_core::argument!("p"))?;
        let p = p.into_f64();
        let small = p.min(1.0 - p);
        // Around p=1/2, cos(pi*p) must vanish exactly at the median. Form
        // its small argument from the exact distance to 1/2 instead.
        let (sin, cos) = if small <= 0.25 {
            (PI * small).sin_cos()
        } else {
            let (sin, cos) = (PI * (0.5 - small)).sin_cos();
            (cos, sin)
        };
        let theta = 2.0 * ((1.0 - self.rho) * sin).atan2((1.0 + self.rho) * cos);
        let theta = if (p > 0.5) ^ upper {
            TAU - theta
        } else {
            theta
        };
        number(self.location.into_f64() + theta)
    }
}
impl<S: RealScalar> Support<S> for WrappedCauchy<S> {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        ExtendedInterval::new(
            Some(self.location),
            Some(number(self.location.into_f64() + TAU)?),
        )
    }
}
impl<S: RealScalar> LogDensity<S> for WrappedCauchy<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        argument(x, "x")?;
        if !self.contains(x)? {
            return Ok(S::NEG_INFINITY);
        }
        let high: S = number(self.location.into_f64() + TAU)?;
        let distance =
            (x.into_f64() - self.location.into_f64()).min(high.into_f64() - x.into_f64());
        let half = distance * 0.5;
        let denominator = (1.0 - self.rho).hypot(2.0 * self.rho.sqrt() * half.sin());
        number(
            (-self.rho).ln_1p() + self.rho.ln_1p()
                - crate::distribution::special::log_two_pi::<f64>()
                - 2.0 * denominator.ln(),
        )
    }
}
impl<S: RealScalar> Density<S> for WrappedCauchy<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        finite(self.log_pdf(x)?.exp())
    }
}
impl<S: RealScalar> LogCdf<S> for WrappedCauchy<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.tail(x, false)
    }
}
impl<S: RealScalar> LogSurvival<S> for WrappedCauchy<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        self.tail(x, true)
    }
}
impl<S: RealScalar> Cdf<S> for WrappedCauchy<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_cdf(x)?.exp())
    }
}
impl<S: RealScalar> Survival<S> for WrappedCauchy<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(self.log_sf(x)?.exp())
    }
}
impl<S: RealScalar> Quantile<S> for WrappedCauchy<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        self.inverse(p, false)
    }
}
impl<S: RealScalar> InverseSurvival<S> for WrappedCauchy<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        self.inverse(p, true)
    }
}
impl<S: RealScalar> Entropy<S> for WrappedCauchy<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        number(
            crate::distribution::special::log_two_pi::<f64>()
                + (-self.rho).ln_1p()
                + self.rho.ln_1p(),
        )
    }
}
impl<S: RealScalar> Mean<S> for WrappedCauchy<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        number(self.location.into_f64() + PI)
    }
}
impl<S: RealScalar> Variance<S> for WrappedCauchy<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        self.variance_work(&mut self.options.work_budget().get())
    }

    fn variance_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.variance_work(left)
        })
    }
}
impl<S: RealScalar> WrappedCauchy<S> {
    fn variance_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        let width = (1.0 - self.rho) / (1.0 + self.rho);
        let end = PI / (width + PI);
        let value = crate::distribution::quadrature::positive_integral(
            0.0,
            end,
            self.options.relative(),
            left,
            &|u: f64| {
                let t = width * u / (1.0 - u);
                let denominator = (1.0 - self.rho).hypot(2.0 * self.rho.sqrt() * (t * 0.5).sin());
                Ok(2.0 * (PI - t).ln() + (-self.rho).ln_1p() + self.rho.ln_1p()
                    - crate::distribution::special::log_two_pi::<f64>()
                    - 2.0 * denominator.ln()
                    + width.ln()
                    - 2.0 * (-u).ln_1p())
            },
            None,
        )?;
        number((value + core::f64::consts::LN_2).exp())
    }
}

impl<S: RealScalar> Distribution<S> for WrappedCauchy<S> {
    scalar_batches!();
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let p = crate::util::validated_unit_float(
            &mut crate::distribution::words::Words::widening(rng, self.options.sample_words.get()),
            scioxide_core::argument!("self"),
        )?;
        self.inverse(number(p)?, false)
    }
}

macro_rules! indexed_sampling {
    ($name:ident) => {
        impl<S: RealScalar> $name<S> {
            /// Fills `output` with samples, drawing sample `i` from the stream `root.split(i)`.
            ///
            /// The execution policy of the [`LawOptions`] schedules the samples on
            /// `cx`. The result does not depend on the schedule. Samples are
            /// staged in `workspace`, which grows as needed, and `output` is
            /// written only if every draw succeeds.
            ///
            /// # Errors
            ///
            /// - Any error of `sample`.
            /// - `ResourceLimit` if the staging storage exceeds [`LawOptions::max_bytes`].
            /// - Stream splitting, cancellation and executor errors.
            pub fn sample_indexed_into<R: Rng<S> + crate::SplittableRng + Send + Sync>(
                &self,
                root: &R,
                output: &mut [S],
                workspace: &mut crate::SamplingWorkspace<S>,
                cx: scioxide_core::ExecutionContext<'_>,
            ) -> Result<(), SciOxideError> {
                let options = self.options();
                crate::workspace::sample_indexed(
                    root,
                    output,
                    1,
                    S::ZERO,
                    workspace,
                    cx,
                    options.execution_policy(),
                    crate::workspace::IndexedBytes {
                        limit: options.max_bytes(),
                        retained: 0,
                        per_worker: 0,
                    },
                    || Ok(()),
                    |rng, (), row| {
                        row[0] = self.sample(rng)?;
                        Ok(())
                    },
                )
            }
        }
    };
}
indexed_sampling!(VonMises);
indexed_sampling!(VonMisesLine);
indexed_sampling!(WrappedCauchy);

/// Best and Fisher (1979) angle sampler, computing `r − 1` and `1 − cos θ`
/// without cancellation. Both rejection loops stop after 1,024 proposals.
#[derive(Debug, Clone, Copy)]
pub(super) struct AngleKernel {
    kappa: f64,
    /// Best-Fisher `r - 1 = (1 - b)^2 / (2 b)`; unused below `kappa = 0.5`.
    delta: f64,
}
impl AngleKernel {
    pub(super) fn new(kappa: f64) -> Self {
        if kappa < 0.5 {
            return Self { kappa, delta: 0.0 };
        }
        // b = (a - sqrt(2a))/(2k), r=(1+b^2)/(2b).
        // Compute 1-b by rationalization, then r-1=(1-b)^2/(2b).
        let inverse = 1.0 / kappa;
        let a_over_k = inverse + inverse.hypot(2.0);
        let root_over_k = (2.0 * a_over_k * inverse).sqrt();
        let one_minus_b =
            (root_over_k - inverse - inverse * inverse / (inverse.hypot(2.0) + 2.0)) * 0.5;
        let b = 1.0 - one_minus_b;
        Self {
            kappa,
            delta: one_minus_b * one_minus_b / (2.0 * b),
        }
    }
    pub(super) fn sample<R: Rng<f64>>(&self, rng: &mut R) -> Result<f64, SciOxideError> {
        let Self { kappa, delta } = *self;
        if kappa == 0.0 {
            return Ok(TAU * rng.next_float()? - PI);
        }
        if kappa < 0.5 {
            // Uniform-envelope rejection avoids the small-kappa cancellation of
            // the Best-Fisher proposal. Its acceptance is at least exp(-1).
            return crate::util::rejection(|| {
                let theta = TAU * rng.next_float()? - PI;
                let accept = crate::util::validated_open_float(rng)?.ln()
                    <= -kappa * (2.0 * (theta * 0.5).sin().powi(2));
                Ok(accept.then_some(theta))
            });
        }
        crate::util::rejection(|| {
            let u = rng.next_float()?;
            let z = (PI * u).cos();
            let one_minus_f = delta * (1.0 - z) / (1.0 + z + delta);
            let c = kappa * (delta + one_minus_f);
            let v = crate::util::validated_open_float(rng)?;
            if v < c * (2.0 - c) || v.ln() <= c.ln() + 1.0 - c {
                let theta = 2.0 * (0.5 * one_minus_f).sqrt().min(1.0).asin();
                return Ok(Some(if rng.next_float()? < 0.5 {
                    -theta
                } else {
                    theta
                }));
            }
            Ok(None)
        })
    }
}
