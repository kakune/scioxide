//! Signed differences of independent Poisson counts.
use super::count_math::{self as math};
use crate::distribution::LawOptions;
use crate::law::logadd;
use crate::{BitGenerator, SignedCountSupport};
use scioxide_core::{RealScalar, SciOxideError};

/// Skellam distribution: the difference `X − Y` of independent Poisson counts with rates `μ₁` and
/// `μ₂`.
///
/// For `μ₁, μ₂ > 0` the mass function is
///
/// ```text
/// P(k) = e^(−(μ₁ + μ₂)) (μ₁/μ₂)^(k/2) I_|k|(2 √(μ₁ μ₂)),   k ∈ ℤ
/// ```
///
/// where `I_ν` is the modified Bessel function of the first kind. The mean is
/// `μ₁ − μ₂` and the variance `μ₁ + μ₂`. If one rate is zero the law is a
/// Poisson law on the nonnegative or nonpositive integers, and if both are
/// zero it is a point mass at zero. Counts are `i64`. Each call of `sample`,
/// `sample_n` or `sample_n_into` ([`CountDistribution`](crate::CountDistribution))
/// prepares a [`SkellamSampler`] without allocating and draws through it; for a
/// loop of single draws, prepare it once with [`Skellam::prepare_sampler`].
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, Mass, Mean, Quantile, SeedableRng, Skellam};
/// use scioxide_random::{SplitMix64, Survival};
/// let law = Skellam::new(3.0_f64, 1.0)?;
/// assert!((law.pmf(0)? - 0.131_121_595_373_807_7).abs() < 1e-13);
/// assert!((law.pmf(1)? / law.pmf(-1)? - 3.0).abs() < 1e-12); // (mu1/mu2)^k
/// assert!((law.cdf(-1)? + law.sf(-1)? - 1.0).abs() < 1e-8);
/// assert!((law.mean()? - 2.0).abs() < 1e-15 && law.quantile(0.5)? == 2);
/// let k: i64 = law.sample(&mut SplitMix64::from_seed(42))?;
/// assert!(law.pmf(k)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Skellam<S: RealScalar> {
    a: f64,
    b: f64,
    options: LawOptions<S>,
}
impl<S: RealScalar> Skellam<S> {
    /// Creates a Skellam law with rates `mu1` and `mu2`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `mu1` or `mu2` is NaN or infinite.
    /// - `DomainError` if `mu1 < 0` or `mu2 < 0`.
    pub fn new(mu1: S, mu2: S) -> Result<Self, SciOxideError> {
        Self::with_options(mu1, mu2, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(mu1: S, mu2: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self {
            a: math::parameter(mu1, false, scioxide_core::argument!("mu1"))?,
            b: math::parameter(mu2, false, scioxide_core::argument!("mu2"))?,
            options,
        })
    }
    /// Returns a sampler that draws the two Poisson counts and subtracts them.
    ///
    /// The setup reads no random words.
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if a rate exceeds the range of
    ///   [`Poisson::prepare_sampler`](crate::Poisson::prepare_sampler).
    pub fn prepare_sampler(&self) -> Result<SkellamSampler, SciOxideError> {
        Ok(SkellamSampler {
            a: super::Poisson::with_options(self.a, self.options.widened()?)?.prepare_sampler()?,
            b: super::Poisson::with_options(self.b, self.options.widened()?)?.prepare_sampler()?,
            limits: math::DrawLimits::of(self.options),
            work: self.options.work_budget().get(),
        })
    }
    fn mass(&self, k: i64) -> Result<f64, SciOxideError> {
        let left = &mut self.options.work_budget().get();
        math::signed_count_value(k)?;
        if self.a == 0.0 || self.b == 0.0 {
            return point_or_poisson(self.a, self.b, k, None, left);
        }
        // P(K = k) = e^{-(a+b)} (a/b)^{k/2} I_|k|(2 sqrt(ab)). With the
        // exponentially scaled Bessel function e^{-x} I(x) the exponent
        // combines to -(sqrt a - sqrt b)^2, which is small even when a and b
        // are huge. If the Bessel evaluation fails or underflows, fall back to
        // the explicit Poisson convolution.
        let root_a = self.a.sqrt();
        let root_b = self.b.sqrt();
        let x = 2.0 * root_a * root_b;
        if x.is_finite() {
            let result = scioxide_special::bessel_i_scaled_with_options(
                k.unsigned_abs() as f64,
                x,
                scioxide_special::BesselOptions::default()
                    .with_iteration_budget(math::remaining_budget(left)?),
            );
            let report = result?;
            *left = left
                .checked_sub(report.counts().iterations())
                .ok_or_else(math::exhausted)?;
            if !report.termination().is_success() {
                return report.try_value();
            }
            if let Ok(value) = report.try_value()
                && value > 0.0
            {
                return math::log_probability(
                    -(root_a - root_b).powi(2)
                        + (k as f64) * 0.5 * (self.a.ln() - self.b.ln())
                        + value.ln(),
                );
            }
        }
        convolution(self.a, self.b, k, None, self.options, left)
    }
    fn tail(&self, k: i64, upper: bool) -> Result<f64, SciOxideError> {
        self.tail_with_work(k, upper, &mut self.options.work_budget().get())
    }
    fn tail_with_work(&self, k: i64, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        math::signed_count_value(k)?;
        if self.a == 0.0 || self.b == 0.0 {
            return point_or_poisson(self.a, self.b, k, Some(upper), left);
        }
        // The convolution sums over the second Poisson variable, so it is
        // arranged to run over the smaller rate (fewer significant terms):
        // A - B <= k  <=>  B - A >= -k  <=>  B - A > -k - 1.
        if self.a < self.b {
            convolution(self.b, self.a, -k - 1, Some(!upper), self.options, left)
        } else {
            convolution(self.a, self.b, k, Some(upper), self.options, left)
        }
    }
    fn inverse(&self, p: S, upper: bool, left: &mut usize) -> Result<i64, SciOxideError> {
        let (lo, hi) = self.signed_count_support();
        math::inverse_signed(p, upper, lo, hi, left, |k, u, left| {
            self.tail_with_work(k, u, left)
        })
    }
}
fn point_or_poisson(
    a: f64,
    b: f64,
    k: i64,
    tail: Option<bool>,
    left: &mut usize,
) -> Result<f64, SciOxideError> {
    if a == 0.0 && b == 0.0 {
        return Ok(match tail {
            None => {
                if k == 0 {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
            Some(u) => {
                if (k >= 0) != u {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            }
        });
    }
    if a == 0.0 {
        return match tail {
            None => {
                if k > 0 {
                    Ok(f64::NEG_INFINITY)
                } else {
                    crate::LogMass::log_pmf(&super::Poisson::new(b)?, (-k) as usize)
                }
            }
            Some(u) => {
                if k >= 0 {
                    Ok(if u { f64::NEG_INFINITY } else { 0.0 })
                } else {
                    poisson_tail(b, (-k - 1) as usize, !u, left)
                }
            }
        };
    }
    match tail {
        None => {
            if k < 0 {
                Ok(f64::NEG_INFINITY)
            } else {
                crate::LogMass::log_pmf(&super::Poisson::new(a)?, k as usize)
            }
        }
        Some(u) => {
            if k < 0 {
                Ok(if u { 0.0 } else { f64::NEG_INFINITY })
            } else {
                poisson_tail(a, k as usize, u, left)
            }
        }
    }
}
fn poisson_tail(rate: f64, k: usize, upper: bool, left: &mut usize) -> Result<f64, SciOxideError> {
    if rate == 0.0 {
        return Ok(if upper { f64::NEG_INFINITY } else { 0.0 });
    }
    let shape = super::count_math::analytic_count(k)?;
    if shape >= super::count_math::MAX_EXACT_COUNT as f64 {
        return Err(SciOxideError::not_implemented(
            "Poisson tail shape exceeds exact counts",
        ));
    }
    crate::law::special(left, |budget| {
        if upper {
            scioxide_special::gamma_log_p_with_options(
                shape + 1.0,
                rate,
                scioxide_special::GammaOptions::new(budget),
            )
        } else {
            scioxide_special::gamma_log_q_with_options(
                shape + 1.0,
                rate,
                scioxide_special::GammaOptions::new(budget),
            )
        }
    })
}
fn convolution<S: RealScalar>(
    a: f64,
    b: f64,
    k: i64,
    tail: Option<bool>,
    o: LawOptions<S>,
    left: &mut usize,
) -> Result<f64, SciOxideError> {
    let center = math::count(b.floor())?;
    let term = |j: usize, left: &mut usize| -> Result<f64, SciOxideError> {
        let shifted = (j as i128) + (k as i128);
        let value = if shifted < 0 {
            if tail == Some(true) {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            let shifted = usize::try_from(shifted)
                .map_err(|_| SciOxideError::not_implemented("Skellam convolution count range"))?;
            match tail {
                Some(u) => poisson_tail(a, shifted, u, left)?,
                None => super::discrete_math::poisson_log_mass(
                    super::count_math::analytic_count(shifted)?,
                    a,
                ),
            }
        };
        Ok(
            super::discrete_math::poisson_log_mass(super::count_math::analytic_count(j)?, b)
                + value,
        )
    };
    // Sum over j (the value of Poisson(b)) outward from its mode. Each term is
    // at most P(B = j), so the untouched mass of B on both sides bounds the
    // remaining contribution; stop when that bound is below the relative
    // tolerance of the running total.
    let mut lo = center;
    let mut hi = center;
    crate::law::charge(left)?;
    let mut total = term(center, left)?;
    loop {
        let remainder = logadd(
            if lo == 0 {
                f64::NEG_INFINITY
            } else {
                poisson_tail(b, lo - 1, false, left)?
            },
            poisson_tail(b, hi, true, left)?,
        )?;
        if remainder <= total + o.relative().ln() {
            return math::log_probability(total);
        }
        if lo > 0 {
            lo -= 1;
            crate::law::charge(left)?;
            total = logadd(total, term(lo, left)?)?;
        }
        hi = hi
            .checked_add(1)
            .ok_or_else(|| SciOxideError::not_implemented("Skellam tail count range"))?;
        crate::law::charge(left)?;
        total = logadd(total, term(hi, left)?)?;
    }
}
impl<S: RealScalar> SignedCountSupport for Skellam<S> {
    fn signed_count_support(&self) -> (Option<i64>, Option<i64>) {
        (
            if self.b == 0.0 { Some(0) } else { None },
            if self.a == 0.0 { Some(0) } else { None },
        )
    }
}
/// Skellam sampler: the difference of two Poisson draws.
///
/// The two Poisson draws of one sample share the
/// [`LawOptions::sample_word_budget`] of the law. A difference outside the
/// range of `i64` is a `NotImplemented` error.
///
/// ```
/// use scioxide_random::{CountDistribution, SeedableRng, Skellam, SplitMix64};
/// let sampler = Skellam::new(5.0_f64, 0.0)?.prepare_sampler()?;
/// let mut rng = SplitMix64::from_seed(7);
/// for _ in 0..4 {
///     assert!(sampler.sample(&mut rng)? >= 0); // mu2 = 0 leaves a Poisson count
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SkellamSampler {
    a: super::PoissonSampler,
    b: super::PoissonSampler,
    limits: math::DrawLimits,
    work: usize,
}
impl SkellamSampler {
    fn draw<R: BitGenerator + ?Sized>(&self, rng: &mut R) -> Result<i64, SciOxideError> {
        let mut left = self.work;
        let a = self.a.draw_with_work(rng, &mut left)? as i128;
        let b = self.b.draw_with_work(rng, &mut left)? as i128;
        i64::try_from(a - b).map_err(|_| {
            SciOxideError::not_implemented("Skellam sample exceeds signed output range")
        })
    }
    fn limits(&self) -> math::DrawLimits {
        self.limits
    }
}
math::count_distribution!([] SkellamSampler, i64);
math::prepared_count_distribution!(Skellam, i64);
impl<S: RealScalar> Skellam<S> {
    fn average(&self) -> Result<f64, SciOxideError> {
        crate::law::finite(self.a - self.b)
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        crate::law::finite(self.a + self.b)
    }
}
math::signed_capabilities!(Skellam);
