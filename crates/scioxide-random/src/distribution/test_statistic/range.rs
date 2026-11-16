//! Studentized range distribution; both tails are positive integrals.
use super::*;
use crate::distribution::LawOptions;
use chi_f_t::scale_integral;
use math::{boundary, log_interval, log_phi, normal_log, real_integral};
/// Studentized range distribution with `k > 1` groups and `ν > 0` degrees of freedom.
///
/// For integer `k`, `Q = (max Zᵢ − min Zᵢ) / √(V/ν)`, where `Z₁, …, Z_k` are
/// independent standard normal and `V` is an independent chi-squared variate
/// with `ν` degrees of freedom. `ν = +∞` gives the range of `k` normals. The
/// distribution function is a double integral and is also defined for real
/// `k > 1`. Integer `k` draws follow the definition, and other `k` draws
/// invert the distribution function. Moments and quantiles repeat the tail
/// integration and may need a work limit above the [`LawOptions`] default.
///
/// ```
/// use scioxide_random::{Cdf, Distribution, LawOptions, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, StudentizedRange};
/// let work = scioxide_core::Budget::ONE.saturating_add(1 << 24);
/// let options = LawOptions::default().with_work_budget(work);
/// // k = 2 with infinite degrees is |Z1 - Z2| = sqrt(2) |Z|.
/// let law = StudentizedRange::with_options(2.0_f64, f64::INFINITY, options)?;
/// assert!((law.cdf(core::f64::consts::SQRT_2)? - 0.682_689_492_137_085_9).abs() < 1e-9);
/// assert!((law.mean()? - 2.0 / core::f64::consts::PI.sqrt()).abs() < 1e-9);
/// assert!((law.cdf(law.quantile(0.95)?)? - 0.95).abs() < 1e-9);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!(law.sample(&mut rng)? >= 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct StudentizedRange<S: RealScalar> {
    k: f64,
    nu: f64,
    /// Prepared chi scale; absent for infinite degrees.
    chi: Option<crate::distribution::chi_squared::ChiSquareKernel<f64>>,
    /// The folded Student t of two groups with finite degrees.
    pair: Option<crate::StudentT<f64>>,
    options: LawOptions<S>,
}
impl<S: RealScalar> StudentizedRange<S> {
    /// Creates a studentized range law with `k` groups and `degrees` degrees of freedom.
    ///
    /// `degrees` may be `+∞`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `k` is NaN or infinite, or if `degrees` is NaN or `−∞`.
    /// - `DomainError` if `k ≤ 1` or `degrees ≤ 0`.
    pub fn new(k: S, degrees: S) -> Result<Self, SciOxideError> {
        Self::with_options(k, degrees, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(k: S, degrees: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        let k = positive(k, scioxide_core::argument!("k"))?.into_f64();
        if k <= 1.0 {
            return Err(crate::law::domain(
                scioxide_core::argument!("k"),
                "studentized range requires k > 1",
            ));
        }
        let nu = if degrees == S::INFINITY {
            f64::INFINITY
        } else {
            positive(degrees, scioxide_core::argument!("degrees"))?.into_f64()
        };
        let chi = if nu.is_finite() {
            Some(crate::distribution::chi_squared::ChiSquareKernel::new(nu)?)
        } else {
            None
        };
        let pair = (k == 2.0 && nu.is_finite())
            .then(|| crate::StudentT::new(nu))
            .transpose()?;
        Ok(Self {
            k,
            nu,
            chi,
            pair,
            options,
        })
    }
    fn raw_moment(&self, order: f64, w: &mut Work) -> Result<f64, SciOxideError> {
        if self.nu <= order {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        // Integrate normal-range tails first, then use independence of the chi scale.
        let range = real_integral(
            |r, w| {
                let x = r.exp();
                if !x.is_finite() || x > f64::MAX.sqrt() {
                    return Ok(f64::NEG_INFINITY);
                }
                Ok(order.ln() + order * r + normal_range(self.k, x, Query::Upper, w)?)
            },
            0.0,
            1.0,
            w,
        )?;
        let scale = chi_f_t::scale_moment_log(self.nu, order)?;
        finite((range + scale).exp())
    }
}
impl<S: RealScalar> Statistic<S> for StudentizedRange<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn hull(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    // Normal-range median 2 Φ⁻¹(2^(-1/k)) over the chi-scale median; it only
    // orders tail evaluations.
    fn median(&self) -> f64 {
        let range =
            scioxide_special::normal_quantile(0.5_f64.powf(1. / self.k)).map_or(1., |z| 2. * z);
        if self.nu.is_finite() {
            range / (math::chi_square_center(self.nu, self.nu) / self.nu).sqrt()
        } else {
            range
        }
    }
    fn kernel(&self, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
        if x == 0.0 && q == Query::Density {
            if self.k < 2.0 {
                return Ok(f64::INFINITY);
            }
            if self.k == 2.0 {
                return Ok(2.0_f64.sqrt().ln()
                    + match &self.pair {
                        Some(t) => t.log_pdf(0.0)?,
                        None => log_phi(0.0),
                    });
            }
        }
        if let Some(v) = boundary(x, 0.0, f64::INFINITY, q) {
            return Ok(v);
        }
        // With two groups the range is |Z1 - Z2| / s, i.e. sqrt(2) |T|, so the
        // law reduces exactly to a folded Student t (or normal for nu = inf).
        if self.k == 2.0 {
            w.charge(1)?;
            let z = x / core::f64::consts::SQRT_2;
            if q == Query::Density {
                let ld = match &self.pair {
                    Some(t) => t.log_pdf(z)?,
                    None => log_phi(z),
                };
                return Ok(ld + 0.5 * core::f64::consts::LN_2);
            }
            let ls = match &self.pair {
                Some(t) => t.log_sf(z)?,
                None => normal_log(z, true)?,
            };
            return Ok(if q == Query::Upper {
                ls + core::f64::consts::LN_2
            } else {
                log1mexp(ls + core::f64::consts::LN_2)?
            });
        }
        // Studentization: Q = R / s with s^2 ~ chi^2_nu / nu independent of the
        // normal range R, so P(Q <= x) = E[P(R <= x s)]. The outer integral over
        // the log scale r is done by `scale_integral`; the inner normal-range
        // evaluation gets a tighter tolerance so its error does not dominate
        // the outer quadrature.
        scale_integral(
            self.nu,
            |r, w| {
                let threshold = x * r.exp();
                if !threshold.is_finite() || threshold > f64::MAX.sqrt() {
                    return Ok(if q == Query::Lower {
                        0.0
                    } else {
                        f64::NEG_INFINITY
                    });
                }
                let tolerance = w.tolerance;
                w.tolerance = (tolerance * 0.03125).max(64.0 * f64::EPSILON);
                let result = normal_range(self.k, threshold, q, w);
                w.tolerance = tolerance;
                Ok(result? + if q == Query::Density { r } else { 0.0 })
            },
            w,
        )
    }
    fn variate<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        // Integer k samples by definition (range of k normals over an
        // independent chi scale); fractional k has no generative model and is
        // sampled by inversion instead.
        if self.k.fract() != 0.0 {
            return self.quantile(crate::float::open_word_probability(
                rng,
                self.options.sample_words,
            )?);
        }
        if self.k > self.options.sample_words.get() as f64 {
            return Err(exhausted());
        }
        let mut r = crate::distribution::words::Words::new(rng, self.options.sample_words.get());
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for _ in 0..self.k as usize {
            let z = crate::distribution::normal::standard_normal::<f64, _>(&mut r)?;
            low = low.min(z);
            high = high.max(z);
        }
        let scale = match &self.chi {
            Some(chi) => (chi.sample(&mut r)? / self.nu).sqrt(),
            None => 1.0,
        };
        number(finite((high - low) / scale)?)
    }
}
statistic_law!(StudentizedRange, report);
fn normal_range(k: f64, x: f64, q: Query, w: &mut Work) -> Result<f64, SciOxideError> {
    if x == 0.0 {
        return Ok(if q == Query::Upper {
            0.0
        } else {
            f64::NEG_INFINITY
        });
    }
    // Small-range limit: all k normals must fall within width x, giving
    // F(x) ~ sqrt(k) (x / sqrt(2 pi))^(k-1) with relative error O(k x^2).
    if x * x * k < w.tolerance * 0.01 {
        let lower = 0.5 * k.ln()
            + (k - 1.0) * (x.ln() - 0.5 * crate::distribution::special::log_two_pi::<f64>());
        return Ok(match q {
            Query::Lower => lower,
            Query::Upper => log1mexp(lower)?,
            Query::Density => lower + (k - 1.0).ln() - x.ln(),
        });
    }
    // Large-range expansion: one ordered pair supplies the leading tail.
    // Intersections of overlapping pair exceedances have the next exponential
    // scale exp(-x^2/3), compared with exp(-x^2/4) in the leading term.
    if 2.0 * k.ln() - x * x / 12.0 < (w.tolerance * 0.001).ln() {
        let coefficient = k.ln() + (k - 1.0).ln();
        let tail = coefficient + normal_log(x / core::f64::consts::SQRT_2, true)?;
        return Ok(match q {
            Query::Upper => tail,
            Query::Lower => log1mexp(tail)?,
            Query::Density => {
                coefficient + log_phi(x / core::f64::consts::SQRT_2) - 0.5 * core::f64::consts::LN_2
            }
        });
    }
    // For three groups the range law has a closed form in terms of Owen's T
    // (S(x) = 12 T(x / sqrt 2, 1 / sqrt 3)), which is far cheaper than the
    // general double integral.
    if k == 3.0 {
        let density = |y: f64| -> Result<f64, SciOxideError> {
            Ok((3.0 / core::f64::consts::PI.sqrt()).ln() - y * y / 4.0
                + log_interval(-y / 6.0_f64.sqrt(), y / 6.0_f64.sqrt())?)
        };
        if q == Query::Density {
            w.charge(1)?;
            return density(x);
        }
        if q == Query::Lower && x < 1.0 {
            return math::integral(|y, _| density(y), 0.0, x, w);
        }
        let remaining = w.limit - w.used;
        if remaining == 0 {
            return Err(exhausted());
        }
        let options = scioxide_special::ProbabilityOptions::default()
            .with_iteration_budget(Budget::try_from(remaining)?);
        let report = scioxide_special::owens_t_with_options(
            x / core::f64::consts::SQRT_2,
            1.0 / 3.0_f64.sqrt(),
            options,
        )?;
        w.charge(report.counts().iterations())?;
        let tail = 12.0 * report.try_value()?;
        return Ok(if q == Query::Upper {
            tail.ln()
        } else {
            (-tail).ln_1p()
        });
    }
    // Minimum density k*phi(z)*SF(z)^(k-1); the difference below integrates
    // the upper tail positively without subtracting a completed CDF from one.
    real_integral(
        |z, _| {
            let interval = log_interval(z, z + x)?;
            let conditional = match q {
                Query::Lower => (k - 1.0) * interval,
                Query::Upper => {
                    let base = normal_log(z, true)?;
                    let ratio = normal_log(z + x, true)? - base;
                    let difference = if ratio < -36.0 {
                        (k - 1.0).ln() + ratio
                    } else {
                        log1mexp((k - 1.0) * log1mexp(crate::law::log_probability(ratio)?)?)?
                    };
                    (k - 1.0) * base + difference
                }
                Query::Density => (k - 1.0).ln() + log_phi(z + x) + (k - 2.0) * interval,
            };
            Ok(k.ln() + log_phi(z) + conditional)
        },
        if q == Query::Lower {
            -(2.0 * k.ln()).sqrt()
        } else {
            -x * 0.5
        },
        1.0,
        w,
    )
}
impl<S: RealScalar> Mean<S> for StudentizedRange<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        number(self.raw_moment(1.0, &mut Work::new(self.options))?)
    }

    fn mean_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            let mut work = Work::new(self.options);
            work.limit = *left;
            let w = &mut work;
            let result = (|| number(self.raw_moment(1.0, w)?))();
            *left = work.limit - work.used;
            result
        })
    }
}
impl<S: RealScalar> Variance<S> for StudentizedRange<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let mut w = Work::new(self.options);
        // Reserve 1/128 of the requested accuracy per raw moment; the guard
        // below accounts for its amplification by the final subtraction.
        w.tolerance = (w.tolerance / 128.0).max(64.0 * f64::EPSILON);
        let mean = self.raw_moment(1.0, &mut w)?;
        let second = self.raw_moment(2.0, &mut w)?;
        let value = crate::distribution::moment_math::variance_from_moments_with_accuracy(
            second,
            mean,
            w.error,
            self.options.tolerance.relative().into_f64(),
        )?;
        super::super::moment_math::variance(value.ln())
    }

    fn variance_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            let mut work = Work::new(self.options);
            work.limit = *left;
            let w = &mut work;
            let result = (|| {
                // Reserve 1/128 of the requested accuracy per raw moment; the guard
                // below accounts for its amplification by the final subtraction.
                w.tolerance = (w.tolerance / 128.0).max(64.0 * f64::EPSILON);
                let mean = self.raw_moment(1.0, w)?;
                let second = self.raw_moment(2.0, w)?;
                let value = crate::distribution::moment_math::variance_from_moments_with_accuracy(
                    second,
                    mean,
                    w.error,
                    self.options.tolerance.relative().into_f64(),
                )?;
                super::super::moment_math::variance(value.ln())
            })();
            *left = work.limit - work.used;
            result
        })
    }
}
