//! Count laws on the nonnegative integers. Computation is in `f64`.
use super::count_math::analytic_count;
use super::count_math::{self as math, Kernel, count_capabilities};
use crate::distribution::LawOptions;
use crate::distribution::beta::BetaKernel;
use crate::distribution::gamma::GammaKernel;
use crate::law::{log1mexp, logadd};
use crate::{BitGenerator, CountSupport};
use scioxide_core::{RealScalar, SciOxideError};

/// Geometric distribution: the number of trials up to and including the first success.
///
/// With success probability `p ∈ (0, 1]` and `q = 1 − p`, the mass function is
/// `P(k) = p q^(k−1)` for `k = 1, 2, …`, with mean `1/p` and variance `q/p²`.
/// Draws invert the distribution function.
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, Geometric, Mass, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// let law = Geometric::new(0.25_f64)?;
/// assert!((law.pmf(1)? - 0.25).abs() < 1e-15);
/// assert!((law.cdf(2)? - 0.4375).abs() < 1e-15 && (law.sf(2)? - 0.5625).abs() < 1e-15);
/// assert_eq!(law.quantile(0.5)?, 3);
/// assert!(law.sample(&mut SplitMix64::from_seed(42))? >= 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Geometric<S: RealScalar> {
    p: f64,
    log_q: f64,
    log_p: f64,
    options: LawOptions<S>,
}
impl<S: RealScalar> Geometric<S> {
    /// Creates a geometric law with success probability `p`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `p` is NaN or infinite.
    /// - `DomainError` if `p` is outside `(0, 1]`.
    pub fn new(p: S) -> Result<Self, SciOxideError> {
        Self::with_options(p, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(p: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        let p = math::probability(p, false, true, scioxide_core::argument!("p"))?;
        Ok(Self {
            p,
            log_q: (-p).ln_1p(),
            log_p: p.ln(),
            options,
        })
    }
    /// Creates a geometric law with failure probability `q = exp(−λ)`.
    ///
    /// The rate `λ > 0` gives `p = 1 − exp(−λ)`. The law keeps `ln q = −λ`
    /// exactly, so tails stay accurate when `p` rounds to one. The support
    /// starts at one; subtract one from draws for the Planck (discrete
    /// exponential) law on `k = 0, 1, …`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::from_rate_with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `rate` is NaN or infinite.
    /// - `DomainError` if `rate ≤ 0`.
    pub fn from_rate(rate: S) -> Result<Self, SciOxideError> {
        Self::from_rate_with_options(rate, LawOptions::default())
    }

    /// Creates the law of [`Self::from_rate`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::from_rate`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn from_rate_with_options(rate: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        let rate = math::parameter(rate, true, scioxide_core::argument!("rate"))?;
        Ok(Self {
            p: -(-rate).exp_m1(),
            log_q: -rate,
            log_p: log1mexp(-rate)?,
            options,
        })
    }
    /// Returns the mean number of failures before the first success, `q/p`.
    ///
    /// This is the mean of the Planck law. It is computed directly, so it keeps
    /// its relative accuracy when the mean trial count `1/p` rounds to one.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if `q/p` overflows the scalar type.
    pub fn mean_failures(&self) -> Result<S, SciOxideError> {
        crate::law::finite(S::try_from_f64_lossy(self.log_q.exp() / self.p)?)
    }
}
impl<S: RealScalar> CountSupport for Geometric<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (
            1,
            if self.log_q == f64::NEG_INFINITY {
                Some(1)
            } else {
                None
            },
        )
    }
}
impl<S: RealScalar> Kernel<S> for Geometric<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        if k == 0 {
            return Ok(f64::NEG_INFINITY);
        }
        if self.log_q == f64::NEG_INFINITY {
            return Ok(if k == 1 { 0.0 } else { f64::NEG_INFINITY });
        }
        Ok(self.log_p + crate::law::finite(analytic_count(k - 1)? * self.log_q)?)
    }
    fn tail_with_work(&self, k: usize, u: bool, _left: &mut usize) -> Result<f64, SciOxideError> {
        let sf = if k == 0 {
            0.0
        } else if self.log_q == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            analytic_count(k)? * self.log_q
        };
        Ok(if u {
            if self.log_q.is_finite() {
                crate::law::finite(sf)?
            } else {
                sf
            }
        } else if k == 1 {
            self.log_p
        } else {
            log1mexp(sf)?
        })
    }
    // S(k) = q^k by one exponential and the CDF as -expm1 of the same exponent,
    // without the logarithmic tail round trip.
    fn probability(&self, k: usize, upper: bool) -> Result<f64, SciOxideError> {
        if k == 0 {
            return Ok(if upper { 1.0 } else { 0.0 });
        }
        if self.log_q == f64::NEG_INFINITY {
            return Ok(if upper { 0.0 } else { 1.0 });
        }
        if k == 1 && !upper {
            return Ok(self.p);
        }
        let exponent = analytic_count(k)? * self.log_q;
        Ok(if upper {
            exponent.exp()
        } else {
            -exponent.exp_m1()
        })
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        Ok(1.0 / self.p)
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        Ok(self.log_q.exp() / self.p / self.p)
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        if self.log_q == f64::NEG_INFINITY {
            return Ok(1);
        }
        // Inversion of S(k) = q^k: K = floor(ln U / ln q) + 1. `log_q` is
        // cached from `ln_1p(-p)` (or -rate), so tiny p keeps its precision.
        math::count((math::open(rng)?.ln() / self.log_q).floor() + 1.0)
    }
}
count_capabilities!(Geometric);

/// Negative binomial distribution: the number of failures before the `r`-th success.
///
/// With real shape `r > 0`, success probability `p ∈ (0, 1]` and `q = 1 − p`,
/// the mass function is
///
/// ```text
/// P(k) = Γ(k + r) / (Γ(r) k!) · p^r q^k,   k = 0, 1, …
/// ```
///
/// with mean `r q / p` and variance `r q / p²`. Tails use the regularized
/// incomplete beta function `I_p(r, k + 1)`. Draws are Poisson counts whose
/// rate is a gamma variate with shape `r` and scale `q/p` (gamma–Poisson
/// mixture). A rate that underflows gives the count zero.
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, Mass, Mean, NegativeBinomial, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Survival};
/// // r = 2, p = 1/2: P(k) = (k + 1) / 2^(k + 2).
/// let law = NegativeBinomial::new(2.0_f64, 0.5)?;
/// assert!((law.pmf(2)? - 0.1875).abs() < 1e-14);
/// assert!((law.cdf(2)? - 0.6875).abs() < 1e-14 && (law.sf(2)? - 0.3125).abs() < 1e-14);
/// assert!((law.mean()? - 2.0).abs() < 1e-14 && law.quantile(0.6)? == 2);
/// assert!(law.pmf(law.sample(&mut SplitMix64::from_seed(42))?)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct NegativeBinomial<S: RealScalar> {
    r: f64,
    p: f64,
    log_p_mass: f64,
    log_q: f64,
    shape_correction: f64,
    gamma: GammaKernel<f64>,
    options: LawOptions<S>,
}
impl<S: RealScalar> NegativeBinomial<S> {
    /// Creates a negative binomial law with shape `r` and success probability `p`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `r` or `p` is NaN or infinite.
    /// - `DomainError` if `r ≤ 0` or `p` is outside `(0, 1]`.
    pub fn new(r: S, p: S) -> Result<Self, SciOxideError> {
        Self::with_options(r, p, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(r: S, p: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        let r = math::parameter(r, true, scioxide_core::argument!("r"))?;
        let p = math::probability(p, false, true, scioxide_core::argument!("p"))?;
        Ok(Self {
            r,
            p,
            log_p_mass: r * p.ln(),
            log_q: (-p).ln_1p(),
            shape_correction: super::discrete_math::real_stirling_correction(r)?,
            gamma: GammaKernel::new(r)?,
            options,
        })
    }
}
impl<S: RealScalar> CountSupport for NegativeBinomial<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (0, if self.p == 1.0 { Some(0) } else { None })
    }
}
fn nb_tail(r: f64, p: f64, k: usize, u: bool, left: &mut usize) -> Result<f64, SciOxideError> {
    if p == 1.0 {
        return Ok(if u { f64::NEG_INFINITY } else { 0.0 });
    }
    // P(X <= k) = I_p(r, k + 1); both tails come straight from the regularized
    // incomplete beta and its complement, never from 1 - CDF.
    let k = next_count(k)?;
    let options = scioxide_special::BetaOptions::new(math::remaining_budget(left)?);
    let report = if u {
        scioxide_special::beta_inc_log_q_with_options(r, k, p, options)?
    } else {
        scioxide_special::beta_inc_log_with_options(r, k, p, options)?
    };
    math::special_value(report, left)
}
// Add in the integer domain before checking exact floating representation.
fn next_count(k: usize) -> Result<f64, SciOxideError> {
    analytic_count(
        k.checked_add(1)
            .ok_or_else(|| SciOxideError::not_implemented("count plus one overflows"))?,
    )
}
/// Gamma-Poisson draw with a prepared standard gamma kernel of shape `r`.
fn nb_draw<R: BitGenerator + ?Sized>(
    gamma: &GammaKernel<f64>,
    log_odds: f64,
    rng: &mut crate::distribution::words::Words<'_, R>,
    left: &mut usize,
) -> Result<usize, SciOxideError> {
    if log_odds == f64::NEG_INFINITY {
        return Ok(0);
    }
    // Gamma-Poisson mixture: Poisson(Lambda) with Lambda ~ Gamma(r, q/p) is
    // NegativeBinomial(r, p), valid for non-integer r as well.
    let rate = crate::law::finite((gamma.sample_log_power(rng, 1.0)? + log_odds).exp())?;
    if rate == 0.0 {
        return Ok(0);
    }
    super::poisson::conditional_poisson(rate, rng, left)
}
impl<S: RealScalar> Kernel<S> for NegativeBinomial<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        if self.p == 1.0 {
            return Ok(if k == 0 { 0.0 } else { f64::NEG_INFINITY });
        }
        if k == 0 {
            return Ok(self.log_p_mass);
        }
        let kp1 = next_count(k)?;
        let k = analytic_count(k)?;
        let n = self.r + k;
        let successes = n * self.p;
        let failures = n * (1.0 - self.p);
        // NB(k;r,p) = r/(r+k) Binomial(r+k,r;p), extended to real r.
        // Loader's two deviances avoid subtracting large log-gamma terms.
        // Keep the beta form when addition loses a shape or a mean underflows,
        // or when forming 1-p would discard a small success probability.
        if n > self.r
            && n > k
            && successes >= f64::MIN_POSITIVE
            && failures >= f64::MIN_POSITIVE
            && self.p >= f64::EPSILON
        {
            use super::discrete_math::{
                deviance, factorial_stirling_correction, real_stirling_correction,
            };
            let correction = real_stirling_correction(n)?
                - self.shape_correction
                - factorial_stirling_correction(k);
            return Ok(correction
                - deviance(self.r, successes)
                - deviance(k, failures)
                - 0.5
                    * (crate::distribution::special::log_two_pi::<f64>()
                        + k.ln()
                        + (k / self.r).ln_1p()));
        }
        Ok(-scioxide_special::ln_beta(self.r, kp1)? - n.ln() + self.log_p_mass + k * self.log_q)
    }
    fn probability(&self, k: usize, upper: bool) -> Result<f64, SciOxideError> {
        if self.p == 1.0 {
            return Ok(if upper { 0.0 } else { 1.0 });
        }
        let count = next_count(k)?;
        let mut left = self.options.work_budget().get();
        let options = scioxide_special::BetaOptions::default()
            .with_iteration_budget(math::remaining_budget(&left)?);
        // Reflect only when 1-p is exact. The ordinary lower-beta kernel is
        // cheaper than the log-pair kernel used by beta_inc_q.
        let reflected = self.p > 0.5;
        let report = if reflected {
            scioxide_special::beta_inc_with_options(count, self.r, 1.0 - self.p, options)?
        } else {
            scioxide_special::beta_inc_with_options(self.r, count, self.p, options)?
        };
        let value = math::special_value(report, &mut left)?;
        if upper == reflected {
            Ok(value)
        } else if value <= 0.5 {
            Ok(1.0 - value)
        } else {
            // Do not subtract a near-one tail. Charge both attempts against
            // the same budget, including the logarithmic fallback.
            Ok(math::log_probability(self.tail_with_work(k, upper, &mut left)?)?.exp())
        }
    }
    fn tail_with_work(&self, k: usize, u: bool, _left: &mut usize) -> Result<f64, SciOxideError> {
        nb_tail(self.r, self.p, k, u, _left)
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        Ok(self.r * ((1.0 - self.p) / self.p))
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        Ok(self.average()? / self.p)
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        nb_draw(
            &self.gamma,
            self.log_q - self.p.ln(),
            rng,
            &mut self.options.work_budget().get(),
        )
    }
}
count_capabilities!(NegativeBinomial);

/// Beta-binomial distribution: binomial counts whose success probability is beta distributed.
///
/// With `n` trials and shapes `α, β > 0`, the mass function is
///
/// ```text
/// P(k) = C(n, k) B(k + α, n − k + β) / B(α, β),   k = 0, …, n
/// ```
///
/// where `B` is the beta function. The mean is `n α / (α + β)`. Draws take
/// `p` from Beta(α, β) and then a binomial count.
///
/// ```
/// use scioxide_random::{BetaBinomial, Cdf, CountDistribution, Mass, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Survival};
/// // alpha = beta = 1 mixes uniformly over p: every count in 0..=4 has mass 1/5.
/// let law = BetaBinomial::new(4, 1.0_f64, 1.0)?;
/// assert!((law.pmf(3)? - 0.2).abs() < 1e-14);
/// assert!((law.cdf(1)? - 0.4).abs() < 1e-14 && (law.sf(1)? - 0.6).abs() < 1e-14);
/// assert!((law.mean()? - 2.0).abs() < 1e-14 && law.quantile(0.5)? == 2);
/// assert!(law.sample(&mut SplitMix64::from_seed(42))? <= 4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct BetaBinomial<S: RealScalar> {
    n: usize,
    a: f64,
    b: f64,
    beta: BetaKernel<f64>,
    log_normalizer: Result<f64, SciOxideError>,
    options: LawOptions<S>,
}
impl<S: RealScalar> BetaBinomial<S> {
    /// Creates a beta-binomial law with `n` trials and shapes `alpha` and `beta`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite.
    /// - `DomainError` if `alpha ≤ 0` or `beta ≤ 0`.
    pub fn new(n: usize, alpha: S, beta: S) -> Result<Self, SciOxideError> {
        Self::with_options(n, alpha, beta, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(
        n: usize,
        alpha: S,
        beta: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        let a = math::parameter(alpha, true, scioxide_core::argument!("alpha"))?;
        let b = math::parameter(beta, true, scioxide_core::argument!("beta"))?;
        Ok(Self {
            n,
            a,
            b,
            beta: BetaKernel::new(a, b)?,
            log_normalizer: scioxide_special::ln_beta(a, b),
            options,
        })
    }
}
impl<S: RealScalar> CountSupport for BetaBinomial<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (0, Some(self.n))
    }
}
impl<S: RealScalar> Kernel<S> for BetaBinomial<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        if k > self.n {
            return Ok(f64::NEG_INFINITY);
        }
        if self.n == 0 {
            return Ok(0.0);
        }
        // For huge shapes, ln B(k + a, n - k + b) - ln B(a, b) cancels almost
        // completely. With small n the ratio is instead a product of n simple
        // fractions (a + i) / (a + b + ...), which is accurate term by term.
        if self.a.max(self.b) > 1024.0 && self.n <= 128 {
            let count = analytic_count(k)?;
            let coefficient = super::discrete_math::log_choose(self.n, k)?;
            let successes = (0..k)
                .map(|i| log_fraction(self.a + i as f64, self.b))
                .sum::<f64>();
            let failures = (0..self.n - k)
                .map(|i| log_fraction(self.b + i as f64, self.a + count))
                .sum::<f64>();
            return Ok(coefficient + successes + failures);
        }
        if self.a + 1.0 == self.a || self.b + 1.0 == self.b {
            return Err(SciOxideError::not_implemented(
                "beta-binomial shape increments beyond working precision for n>128",
            ));
        }
        let coefficient = super::discrete_math::log_choose(self.n, k)?;
        let n = analytic_count(self.n)?;
        let k = analytic_count(k)?;
        Ok(
            coefficient + scioxide_special::ln_beta(k + self.a, n - k + self.b)?
                - self.log_normalizer.clone()?,
        )
    }
    fn tail_with_work(&self, k: usize, u: bool, _left: &mut usize) -> Result<f64, SciOxideError> {
        if k >= self.n {
            return Ok(if u { f64::NEG_INFINITY } else { 0.0 });
        }
        if u {
            math::sum(k + 1, self.n, _left, |j| self.mass(j))
        } else {
            math::sum(0, k, _left, |j| self.mass(j))
        }
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        let n = analytic_count(self.n)?;
        let fraction = shape_fraction(self.a, self.b);
        Ok(if n == 0.0 {
            0.0
        } else if fraction < f64::MIN_POSITIVE {
            (n.ln() + log_fraction(self.a, self.b)).exp()
        } else {
            n * fraction
        })
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        if self.n == 0 {
            return Ok(0.0);
        }
        let n = analytic_count(self.n)?;
        let a = shape_fraction(self.a, self.b);
        let b = shape_fraction(self.b, self.a);
        let factor = 1.0 + (n - 1.0) / (self.a + self.b + 1.0);
        Ok(if a.min(b) < f64::MIN_POSITIVE {
            (n.ln() + log_fraction(self.a, self.b) + log_fraction(self.b, self.a) + factor.ln())
                .exp()
        } else {
            n * a * b * factor
        })
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        if self.n == 0 {
            return Ok(0);
        }
        // Refuse counts beyond the binomial sampler before consuming any word.
        super::binomial::sampled_trials(self.n)?;
        if self.n <= 128 && (self.a.min(self.b) < 1.0 || self.a.max(self.b) > 1024.0) {
            // Beta conjugacy gives the exact Pólya conditional law. This avoids
            // rounded Gamma-mixture endpoints for small, U-shaped beta laws.
            let mut successes = 0usize;
            for trial in 0..self.n {
                let a = self.a + successes as f64;
                let b = self.b + (trial - successes) as f64;
                let reflected = a > b;
                let probability = if reflected {
                    shape_fraction(b, a)
                } else {
                    shape_fraction(a, b)
                };
                if probability == 0.0 {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Underflow,
                        "positive conditional probability is not representable",
                    ));
                }
                let event = super::discrete_math::bernoulli(probability, rng)?;
                successes += usize::from(event != reflected);
            }
            return Ok(successes);
        }
        // Reflect before materializing the probability, retaining a tiny
        // complement even when the ordinary Beta variate rounds to one.
        // An underflowed smaller probability uses Binomial's exact endpoint.
        let odds = self.beta.sample_log_odds(rng)?;
        let reflected = odds > 0.0;
        let small = (-odds.abs()).exp();
        let p = small / (1.0 + small);
        let draw = super::binomial::conditional_binomial(
            self.n,
            p,
            rng,
            &mut self.options.work_budget().get(),
        )?;
        Ok(if reflected { self.n - draw } else { draw })
    }
}
count_capabilities!(BetaBinomial);

// A/(A+B), retaining the small fraction without overflowing the shape sum.
fn shape_fraction(a: f64, b: f64) -> f64 {
    if a >= b {
        1.0 / (1.0 + b / a)
    } else {
        let ratio = a / b;
        ratio / (1.0 + ratio)
    }
}

// log(A/(A+B)), retaining subnormal ratios until the final probability rounding.
fn log_fraction(a: f64, b: f64) -> f64 {
    if a > b {
        -(b / a).ln_1p()
    } else {
        let ratio = a / b;
        let log_ratio = if ratio < f64::MIN_POSITIVE {
            a.ln() - b.ln()
        } else {
            ratio.ln()
        };
        log_ratio - ratio.ln_1p()
    }
}

/// Beta negative binomial distribution: negative binomial counts whose success probability is beta
/// distributed.
///
/// With shape `r ≥ 0` and beta shapes `α, β > 0`, the mass function is
///
/// ```text
/// P(k) = Γ(r + k) / (k! Γ(r)) · B(α + r, β + k) / B(α, β),   k = 0, 1, …
/// ```
///
/// where `B` is the beta function. `r = 0` is a point mass at zero. Draws take
/// `p` from Beta(α, β) and then a negative binomial count through a
/// gamma–Poisson mixture, working with log odds and log rates. A rate that
/// underflows gives the count zero.
///
/// ```
/// use scioxide_random::{BetaNegativeBinomial, Cdf, CountDistribution, Mass, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Survival};
/// // r = 1, alpha = 2, beta = 1: P(k) = 4 / ((k + 1)(k + 2)(k + 3)).
/// let law = BetaNegativeBinomial::new(1.0_f64, 2.0, 1.0)?;
/// assert!((law.pmf(1)? - 1.0 / 6.0).abs() < 1e-13);
/// assert!((law.cdf(1)? - 5.0 / 6.0).abs() < 1e-13 && (law.sf(1)? - 1.0 / 6.0).abs() < 1e-13);
/// assert!((law.mean()? - 1.0).abs() < 1e-13 && law.quantile(0.5)? == 0);
/// assert!(law.pmf(law.sample(&mut SplitMix64::from_seed(42))?)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct BetaNegativeBinomial<S: RealScalar> {
    r: f64,
    a: f64,
    b: f64,
    beta: BetaKernel<f64>,
    log_normalizer: Result<f64, SciOxideError>,
    /// Prepared gamma kernel of shape `r`; absent for the point mass `r = 0`.
    gamma: Option<GammaKernel<f64>>,
    options: LawOptions<S>,
}
impl<S: RealScalar> BetaNegativeBinomial<S> {
    /// Creates a beta negative binomial law with shape `r` and beta shapes `alpha` and `beta`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `DomainError` if `r < 0`, `alpha ≤ 0` or `beta ≤ 0`.
    pub fn new(r: S, alpha: S, beta: S) -> Result<Self, SciOxideError> {
        Self::with_options(r, alpha, beta, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(
        r: S,
        alpha: S,
        beta: S,
        options: LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        let r = math::parameter(r, false, scioxide_core::argument!("r"))?;
        let a = math::parameter(alpha, true, scioxide_core::argument!("alpha"))?;
        let b = math::parameter(beta, true, scioxide_core::argument!("beta"))?;
        Ok(Self {
            r,
            a,
            b,
            beta: BetaKernel::new(a, b)?,
            log_normalizer: scioxide_special::ln_beta(a, b),
            gamma: if r == 0.0 {
                None
            } else {
                Some(GammaKernel::new(r)?)
            },
            options,
        })
    }
}
impl<S: RealScalar> CountSupport for BetaNegativeBinomial<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (0, if self.r == 0.0 { Some(0) } else { None })
    }
}
impl<S: RealScalar> Kernel<S> for BetaNegativeBinomial<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        if self.r == 0.0 {
            return Ok(if k == 0 { 0.0 } else { f64::NEG_INFINITY });
        }
        let kp1 = next_count(k)?;
        let k = analytic_count(k)?;
        Ok(-scioxide_special::ln_beta(self.r, kp1)? - (self.r + k).ln()
            + scioxide_special::ln_beta(self.a + self.r, self.b + k)?
            - self.log_normalizer.clone()?)
    }
    fn tail_with_work(&self, k: usize, u: bool, _left: &mut usize) -> Result<f64, SciOxideError> {
        if self.r == 0.0 {
            return Ok(if u { f64::NEG_INFINITY } else { 0.0 });
        }
        // With r = 1 the survival function has the closed form
        // B(a, b + k + 1) / B(a, b). It is used directly when it is the small
        // side (< 1/2); otherwise the CDF is summed so neither tail is formed
        // by complementing a value near one.
        if self.r == 1.0 {
            let sf = scioxide_special::ln_beta(self.a, self.b + next_count(k)?)?
                - self.log_normalizer.clone()?;
            if sf > -core::f64::consts::LN_2 {
                let cdf = math::sum(0, k, _left, |j| self.mass(j))?;
                return Ok(if u { log1mexp(cdf)? } else { cdf });
            }
            return Ok(if u { sf } else { log1mexp(sf)? });
        }
        if !u && k < math::DIRECT_TERMS {
            return math::sum(0, k, _left, |j| self.mass(j));
        }
        self.mixture_tail(k, u, _left)
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        if self.r == 0.0 {
            Ok(0.0)
        } else if self.a <= 1.0 {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            Ok(self.r * self.b / (self.a - 1.0))
        }
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        if self.r == 0.0 {
            Ok(0.0)
        } else if self.a <= 2.0 {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            Ok(self.average()?
                * ((self.a + self.b - 1.0) / (self.a - 1.0))
                * ((self.a + self.r - 1.0) / (self.a - 2.0)))
        }
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        let Some(gamma) = &self.gamma else {
            return Ok(0);
        };
        let log_odds = -self.beta.sample_log_odds(rng)?;
        nb_draw(gamma, log_odds, rng, &mut self.options.work_budget().get())
    }
}
count_capabilities!(BetaNegativeBinomial);

impl<S: RealScalar> BetaNegativeBinomial<S> {
    fn mixture_tail(&self, k: usize, u: bool, left: &mut usize) -> Result<f64, SciOxideError> {
        let (r, a, b, o) = (self.r, self.a, self.b, self.options);
        // Tail of the beta mixture as \int NB_tail(k; r, p) Beta(p; a, b) dp,
        // integrated in logit coordinate t = ln(p / q). Both ln p and ln q are
        // obtained stably from t, the Beta density becomes a p^a q^b weight
        // (the logit Jacobian p q absorbs the "-1" exponents), and the
        // unbounded t-line suits the shared tail quadrature around a mode guess.
        let norm = self.log_normalizer.clone()?;
        let f = |t: f64, left: &mut usize| -> Result<f64, SciOxideError> {
            let (lp, lq) = (
                scioxide_core::log_sigmoid(t)?,
                scioxide_core::log_sigmoid(-t)?,
            );
            let p = lp.exp();
            let tail = if p == 0.0 {
                if u { 0.0 } else { f64::NEG_INFINITY }
            } else if lq.exp() == 0.0 {
                if u { f64::NEG_INFINITY } else { 0.0 }
            } else if t > 0.0 {
                // Reflect to keep probabilities close to one out of the beta argument.
                let kk = next_count(k)?;
                if u {
                    math::special_value(
                        scioxide_special::beta_inc_log_with_options(
                            kk,
                            r,
                            lq.exp(),
                            scioxide_special::BetaOptions::new(math::remaining_budget(left)?),
                        )?,
                        left,
                    )?
                } else {
                    math::special_value(
                        scioxide_special::beta_inc_log_q_with_options(
                            kk,
                            r,
                            lq.exp(),
                            scioxide_special::BetaOptions::new(math::remaining_budget(left)?),
                        )?,
                        left,
                    )?
                }
            } else {
                nb_tail(r, p, k, u, left)?
            };
            Ok(a * lp + b * lq - norm + tail)
        };
        let mode = if u {
            (a / (b + analytic_count(k)? / (r + 1.0))).ln()
        } else {
            ((a + r) / b).ln()
        };
        let width = (1.0 / a + 1.0 / b).sqrt().max(1.0 / a).max(1.0 / b);
        let tolerance = scioxide_core::Tolerance::new(0.0, o.relative())?;
        let options = super::LawOptions::default()
            .with_tolerance(tolerance)
            .with_work_budget(o.work_budget());
        logadd(
            crate::distribution::quadrature::tail(f, mode, false, mode, width, options, left)?,
            crate::distribution::quadrature::tail(f, mode, true, mode, width, options, left)?,
        )
    }
}

/// Logarithmic series distribution with parameter `p ∈ (0, 1)`.
///
/// The mass function is `P(k) = −p^k / (k ln(1 − p))` for `k = 1, 2, …`.
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, LogSeries, Mass, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// // P(k) = p^k / (k ln(1 / (1 - p))).
/// let law = LogSeries::new(0.5_f64)?;
/// let ln2 = core::f64::consts::LN_2;
/// assert!((law.pmf(1)? - 0.5 / ln2).abs() < 1e-14 && (law.cdf(1)? - 0.5 / ln2).abs() < 1e-14);
/// assert!((law.sf(1)? - (1.0 - 0.5 / ln2)).abs() < 1e-14);
/// assert!((law.mean()? - 1.0 / ln2).abs() < 1e-14 && law.quantile(0.5)? == 1);
/// assert!(law.sample(&mut SplitMix64::from_seed(42))? >= 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LogSeries<S: RealScalar> {
    p: f64,
    /// Prepared `ln(1 - p)`, negative and finite.
    log_q: f64,
    options: LawOptions<S>,
}
impl<S: RealScalar> LogSeries<S> {
    /// Creates a logarithmic series law with parameter `p`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `p` is NaN or infinite.
    /// - `DomainError` if `p` is outside `(0, 1)`.
    pub fn new(p: S) -> Result<Self, SciOxideError> {
        Self::with_options(p, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(p: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        let p = math::probability(p, false, false, scioxide_core::argument!("p"))?;
        Ok(Self {
            p,
            log_q: (-p).ln_1p(),
            options,
        })
    }
}
impl<S: RealScalar> CountSupport for LogSeries<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}
impl<S: RealScalar> Kernel<S> for LogSeries<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        if k == 0 {
            Ok(f64::NEG_INFINITY)
        } else {
            let k = analytic_count(k)?;
            Ok(k * self.p.ln() - k.ln() - (-self.log_q).ln())
        }
    }
    fn tail_with_work(&self, k: usize, u: bool, _left: &mut usize) -> Result<f64, SciOxideError> {
        if k == 0 {
            return Ok(if u { 0.0 } else { f64::NEG_INFINITY });
        }
        if !u && k < math::DIRECT_TERMS {
            return math::sum(1, k, _left, |j| self.mass(j));
        }
        // sum_{j >= n} p^j / j = \int_0^p x^(n-1) / (1 - x) dx. With
        // x = p e^(-t/n) this becomes p^n / n \int_0^inf e^-t / (1 - p e^(-t/n)) dt,
        // a smooth, exponentially decaying log-integrand for the shared
        // half-line quadrature. `scale` is the natural decay length near t = 0.
        let n = next_count(k)?;
        let scale = n * (1.0 - self.p) / self.p;
        let f = |t: f64, _: &mut usize| -> Result<f64, SciOxideError> {
            Ok(n * self.p.ln() - n.ln() - (-self.log_q).ln() - t - log1mexp(self.p.ln() - t / n)?)
        };
        let tol = scioxide_core::Tolerance::new(0.0, self.options.relative())?;
        let o = super::LawOptions::default()
            .with_tolerance(tol)
            .with_work_budget(self.options.work_budget());
        let sf =
            crate::distribution::quadrature::tail(f, 0.0, true, 0.0, scale.min(1.0), o, _left)?;
        Ok(if u { sf } else { log1mexp(sf)? })
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        Ok(-self.p / ((1.0 - self.p) * self.log_q))
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        // Avoid subtracting two almost equal raw moments near the point-mass limit.
        if self.p < 1e-4 {
            let p = self.p;
            return Ok(p * (0.5 + p * (5.0 / 6.0 + p * (9.0 / 8.0 + p * (251.0 / 180.0)))));
        }
        let m = self.average()?;
        Ok(m / (1.0 - self.p) - m * m)
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        // Kemp's two-uniform logarithmic-series generator (LK).
        let v = math::open(rng)?;
        if v >= self.p {
            return Ok(1);
        }
        let q = -(math::open(rng)? * self.log_q).exp_m1();
        if v <= q * q {
            math::count((1.0 + v.ln() / q.ln()).floor())
        } else if v > q {
            Ok(1)
        } else {
            Ok(2)
        }
    }
}
count_capabilities!(LogSeries);

/// Yule–Simon distribution with shape `ρ > 0`.
///
/// The mass function is `P(k) = ρ B(k, ρ + 1)` for `k = 1, 2, …`, where `B`
/// is the beta function. The tail decays like `k^(−ρ)`.
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, Mass, Mean, Quantile, SeedableRng, SplitMix64};
/// use scioxide_random::{Survival, YuleSimon};
/// // rho = 2: P(k) = 4 / (k (k + 1)(k + 2)) and P(X > k) = 2 / ((k + 1)(k + 2)).
/// let law = YuleSimon::new(2.0_f64)?;
/// assert!((law.pmf(2)? - 1.0 / 6.0).abs() < 1e-13);
/// assert!((law.cdf(1)? - 2.0 / 3.0).abs() < 1e-13 && (law.sf(2)? - 1.0 / 6.0).abs() < 1e-13);
/// assert!((law.mean()? - 2.0).abs() < 1e-13 && law.quantile(0.5)? == 1);
/// assert!(law.sample(&mut SplitMix64::from_seed(42))? >= 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct YuleSimon<S: RealScalar> {
    rho: f64,
    options: LawOptions<S>,
}
impl<S: RealScalar> YuleSimon<S> {
    /// Creates a Yule–Simon law with shape `rho`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `rho` is NaN or infinite.
    /// - `DomainError` if `rho ≤ 0`.
    pub fn new(rho: S) -> Result<Self, SciOxideError> {
        Self::with_options(rho, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(rho: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self {
            rho: math::parameter(rho, true, scioxide_core::argument!("rho"))?,
            options,
        })
    }
}
impl<S: RealScalar> CountSupport for YuleSimon<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}
impl<S: RealScalar> Kernel<S> for YuleSimon<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        if k == 0 {
            Ok(f64::NEG_INFINITY)
        } else {
            Ok(self.rho.ln() + scioxide_special::ln_beta(analytic_count(k)?, self.rho + 1.0)?)
        }
    }
    fn tail_with_work(&self, k: usize, u: bool, _left: &mut usize) -> Result<f64, SciOxideError> {
        // Closed form S(k) = rho B(k + 1, rho). When S(k) > 1/2 the CDF is the
        // small side and is summed directly instead of complemented.
        let sf = if k == 0 {
            0.0
        } else {
            self.rho.ln() + scioxide_special::ln_beta(next_count(k)?, self.rho)?
        };
        if k > 0 && sf > -core::f64::consts::LN_2 {
            let cdf = math::sum(1, k, _left, |j| self.mass(j))?;
            return Ok(if u { log1mexp(cdf)? } else { cdf });
        }
        Ok(if u { sf } else { log1mexp(sf)? })
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        if self.rho <= 1.0 {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            Ok(self.rho / (self.rho - 1.0))
        }
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        if self.rho <= 2.0 {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            Ok((self.rho / (self.rho - 1.0)).powi(2) / (self.rho - 2.0))
        }
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        // Yule-Simon is a geometric law whose success probability is
        // e^{-E / rho} with E ~ Exp(1): draw that probability in log form,
        // then invert the geometric tail with it.
        let logp = math::open(rng)?.ln() / self.rho;
        let logq = log1mexp(logp)?;
        math::count((math::open(rng)?.ln() / logq).floor() + 1.0)
    }
}
count_capabilities!(YuleSimon);

/// Zipf (zeta) distribution with exponent `a > 1`.
///
/// The mass function is `P(k) = k^(−a) / ζ(a)` for `k = 1, 2, …`, where `ζ`
/// is the Riemann zeta function. The constructor computes `ln ζ(a)` once,
/// within the [work limit](crate#work-limits).
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, Mass, Quantile, SeedableRng, SplitMix64};
/// use scioxide_random::{Survival, Zipf};
/// // a = 2: P(k) = 6 / (pi k)^2.
/// let law = Zipf::new(2.0_f64)?;
/// let first = 6.0 / core::f64::consts::PI.powi(2);
/// assert!((law.pmf(2)? - first / 4.0).abs() < 1e-14);
/// assert!((law.cdf(1)? - first).abs() < 1e-14 && (law.sf(1)? - (1.0 - first)).abs() < 1e-14);
/// assert_eq!(law.quantile(0.5)?, 1);
/// assert!(law.sample(&mut SplitMix64::from_seed(42))? >= 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Zipf<S: RealScalar> {
    a: f64,
    log_normalizer: f64,
    options: LawOptions<S>,
}
impl<S: RealScalar> Zipf<S> {
    /// Creates a Zipf law with exponent `a`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` is NaN or infinite.
    /// - `DomainError` if `a ≤ 1`.
    /// - `ConvergenceFailure` if computing `ζ(a)` exhausts the work limit.
    pub fn new(a: S) -> Result<Self, SciOxideError> {
        Self::with_options(a, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(a: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        let a = math::parameter(a, true, scioxide_core::argument!("a"))?;
        if a <= 1.0 {
            return Err(crate::law::domain(
                scioxide_core::argument!("a"),
                "Zipf requires a>1",
            ));
        }
        Ok(Self {
            a,
            log_normalizer: zeta_log(a, 1.0, options)?,
            options,
        })
    }
}
impl<S: RealScalar> CountSupport for Zipf<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}
fn zeta_log<S: RealScalar>(a: f64, q: f64, o: LawOptions<S>) -> Result<f64, SciOxideError> {
    zeta_log_with_work(a, q, &mut o.work_budget().get())
}
fn zeta_log_with_work(a: f64, q: f64, left: &mut usize) -> Result<f64, SciOxideError> {
    let report = scioxide_special::hurwitz_zeta_log_with_options(
        a,
        q,
        scioxide_special::ZetaOptions::default()
            .with_iteration_budget(math::remaining_budget(left)?),
    )?;
    math::special_value(report, left)
}
impl<S: RealScalar> Kernel<S> for Zipf<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        if k == 0 {
            Ok(f64::NEG_INFINITY)
        } else {
            Ok(-self.a * analytic_count(k)?.ln() - self.log_normalizer)
        }
    }
    fn tail_with_work(&self, k: usize, u: bool, _left: &mut usize) -> Result<f64, SciOxideError> {
        if k == 0 {
            return Ok(if u { 0.0 } else { f64::NEG_INFINITY });
        }
        let norm = self.log_normalizer;
        let sf = zeta_log_with_work(self.a, next_count(k)?, _left)? - norm;
        if sf > -core::f64::consts::LN_2 {
            // Near the zeta pole, subtracting the two log normalizers loses
            // the small lower probability. Sum that probability positively.
            let cdf = math::sum(
                1,
                k,
                _left,
                |j| Ok(-self.a * analytic_count(j)?.ln() - norm),
            )?;
            return Ok(if u { log1mexp(cdf)? } else { cdf });
        }
        Ok(if u { sf } else { log1mexp(sf)? })
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        self.average_with_work(&mut self.options.work_budget().get())
    }
    fn average_with_work(&self, left: &mut usize) -> Result<f64, SciOxideError> {
        if self.a <= 2.0 {
            Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ))
        } else {
            Ok((zeta_log_with_work(self.a - 1.0, 1.0, left)? - self.log_normalizer).exp())
        }
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        self.spread_with_work(&mut self.options.work_budget().get())
    }
    fn spread_with_work(&self, left: &mut usize) -> Result<f64, SciOxideError> {
        if self.a <= 3.0 {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        let norm = self.log_normalizer;
        let z0 = zeta_log_with_work(self.a, 2.0, left)?;
        let z1 = zeta_log_with_work(self.a - 1.0, 2.0, left)?;
        let z2 = zeta_log_with_work(self.a - 2.0, 2.0, left)?;
        // Moments are rebuilt from Hurwitz zeta values at q = 2 (terms j >= 2),
        // adding the j = 1 term analytically, so every quantity is a
        // difference of ordered log values that `logdiff` can resolve.
        let first = (crate::law::logdiff(z1, z0, scioxide_core::argument!("self"))? - norm).exp();
        let second = (crate::law::logdiff(
            logadd(z2, z0)?,
            core::f64::consts::LN_2 + z1,
            scioxide_core::argument!("self"),
        )? - norm)
            .exp();
        Ok(second - first * first)
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        // Devroye's rejection algorithm, with acceptance evaluated in logs.
        // Proposal X = floor(U^{-1/(a-1)}) and acceptance
        // V X (T - 1) / (b - 1) <= T / b with T = (1 + 1/X)^(a-1), b = 2^(a-1).
        // Evaluated in logs via `ln_1p`/`log1mexp`, it remains valid for a
        // close to 1, where the naive powers round to 1.
        let am1 = self.a - 1.0;
        let logb = am1 * core::f64::consts::LN_2;
        crate::util::rejection(|| {
            let x = math::count((-math::open(rng)?.ln() / am1).exp().floor())?;
            let xf = analytic_count(x)?;
            let logt = am1 * (1.0 / xf).ln_1p();
            let accept = log1mexp(-logb)? - xf.ln() - log1mexp(-logt)?;
            Ok((math::open(rng)?.ln() <= accept).then_some(x))
        })
    }
}
count_capabilities!(Zipf);

/// Zipfian distribution: the Zipf law truncated to `1, …, n`.
///
/// With exponent `a ≥ 0`, the mass function is `P(k) = k^(−a) / H(n, a)` for
/// `k = 1, …, n`, where `H(n, a) = Σ_{j=1}^n j^(−a)` is the generalized
/// harmonic number. `a = 0` is the discrete uniform law. Normalizer and tails
/// use direct sums and the Euler–Maclaurin formula, so storage does not grow
/// with `n`. Draws use rejection-inversion with O(1) storage.
///
/// # References
///
/// W. Hörmann and G. Derflinger, "Rejection-inversion to generate variates
/// from monotone discrete distributions", *ACM Trans. Model. Comput. Simul.*
/// 6(3), 1996, 169–184.
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, Mass, Quantile, SeedableRng, SplitMix64};
/// use scioxide_random::Zipfian;
/// let law = Zipfian::new(1.1_f64, 1000)?;
/// assert!(law.pmf(1)? > law.pmf(2)? && (law.cdf(1000)? - 1.0).abs() < 1e-15);
/// assert!(law.cdf(law.quantile(0.5)?)? >= 0.5);
/// let mut rng = SplitMix64::from_seed(42);
/// assert!((1..=1000).contains(&law.sample(&mut rng)?));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Zipfian<S: RealScalar> {
    a: f64,
    n: usize,
    log_normalizer: f64,
    options: LawOptions<S>,
    integral_low: f64,
    integral_high: f64,
}
impl<S: RealScalar> Zipfian<S> {
    /// Creates a Zipfian law with exponent `a` on `1, …, n`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` is NaN or infinite.
    /// - `DomainError` if `a < 0` or `n = 0`.
    /// - `NotImplemented` if `a > 0` and `n > 2⁵³`.
    /// - `ConvergenceFailure` if computing the normalizer exhausts the work limit.
    pub fn new(a: S, n: usize) -> Result<Self, SciOxideError> {
        Self::with_options(a, n, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(a: S, n: usize, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        if n == 0 {
            return Err(crate::law::domain(
                scioxide_core::argument!("n"),
                "Zipfian requires n>=1",
            ));
        }
        let a = math::parameter(a, false, scioxide_core::argument!("a"))?;
        let (integral_low, integral_high) = if a == 0.0 || n == 1 {
            (0.0, 0.0)
        } else {
            (
                zipf_integral(a, 1.5) - 1.0,
                zipf_integral(a, analytic_count(n)? + 0.5),
            )
        };
        Ok(Self {
            a,
            n,
            log_normalizer: if a == 0.0 {
                (n as f64).ln()
            } else {
                super::count_power::power_sum(a, 1, n, &mut options.work_budget().get())?
            },
            options,
            integral_low,
            integral_high,
        })
    }
    fn power_sum_with_work(
        &self,
        a: f64,
        start: usize,
        end: usize,
        left: &mut usize,
    ) -> Result<f64, SciOxideError> {
        super::count_power::power_sum(a, start, end, left)
    }
}
// Hörmann and Derflinger (1996), rejection-inversion for monotone masses.
// H(x) = integral_1^x t^-a dt. expm1/log1p retain the a -> 1 limit;
// the exact a=1 branch avoids dividing by zero without a tolerance switch.
fn zipf_integral(a: f64, x: f64) -> f64 {
    let exponent = 1.0 - a;
    if exponent == 0.0 {
        x.ln()
    } else {
        (exponent * x.ln()).exp_m1() / exponent
    }
}
fn zipf_integral_inverse(a: f64, u: f64) -> f64 {
    let exponent = 1.0 - a;
    if exponent == 0.0 {
        u.exp()
    } else {
        ((exponent * u).max(-1.0).ln_1p() / exponent).exp()
    }
}
impl<S: RealScalar> CountSupport for Zipfian<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (1, Some(self.n))
    }
}
impl<S: RealScalar> Kernel<S> for Zipfian<S> {
    fn options(&self) -> LawOptions<S> {
        self.options
    }
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut crate::distribution::words::Words<'_, R>,
    ) -> Result<usize, SciOxideError> {
        if self.n == 1 {
            return Ok(1);
        }
        if self.a == 0.0 {
            return Ok(crate::sampling::uniform_below(self.n as u64, rng)? as usize + 1);
        }
        crate::util::rejection(|| {
            let u =
                self.integral_high + math::open(rng)? * (self.integral_low - self.integral_high);
            let x = zipf_integral_inverse(self.a, u);
            if x.is_nan() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "Zipfian rejection-inversion produced an indeterminate candidate",
                ));
            }
            // Only endpoint rounding can put the proposal outside its support.
            let k = (x + 0.5).max(1.0).min(self.n as f64) as usize;
            let threshold =
                zipf_integral(self.a, k as f64 + 0.5) - (-self.a * (k as f64).ln()).exp();
            Ok((u >= threshold).then_some(k))
        })
    }
    fn mass(&self, k: usize) -> Result<f64, SciOxideError> {
        if k == 0 || k > self.n {
            Ok(f64::NEG_INFINITY)
        } else if self.a == 0.0 {
            Ok(-analytic_count(self.n)?.ln())
        } else {
            Ok(-self.a * analytic_count(k)?.ln() - self.log_normalizer)
        }
    }
    fn tail_with_work(&self, k: usize, u: bool, _left: &mut usize) -> Result<f64, SciOxideError> {
        if k == 0 {
            return Ok(if u { 0.0 } else { f64::NEG_INFINITY });
        }
        if k >= self.n {
            return Ok(if u { f64::NEG_INFINITY } else { 0.0 });
        }
        if self.a == 0.0 {
            return Ok(
                (analytic_count(if u { self.n - k } else { k })? / analytic_count(self.n)?).ln(),
            );
        }
        let tail = self.power_sum_with_work(
            self.a,
            if u { k + 1 } else { 1 },
            if u { self.n } else { k },
            _left,
        )? - self.log_normalizer;
        if tail > -core::f64::consts::LN_2 {
            let other = self.power_sum_with_work(
                self.a,
                if u { 1 } else { k + 1 },
                if u { k } else { self.n },
                _left,
            )? - self.log_normalizer;
            log1mexp(math::log_probability(other)?)
        } else {
            Ok(tail)
        }
    }
    fn average(&self) -> Result<f64, SciOxideError> {
        self.average_with_work(&mut self.options.work_budget().get())
    }
    fn average_with_work(&self, left: &mut usize) -> Result<f64, SciOxideError> {
        if self.a == 0.0 {
            Ok((analytic_count(self.n)? + 1.0) / 2.0)
        } else {
            Ok(
                (self.power_sum_with_work(self.a - 1.0, 1, self.n, left)? - self.log_normalizer)
                    .exp(),
            )
        }
    }
    fn spread(&self) -> Result<f64, SciOxideError> {
        self.spread_with_work(&mut self.options.work_budget().get())
    }
    fn spread_with_work(&self, left: &mut usize) -> Result<f64, SciOxideError> {
        if self.a == 0.0 {
            let n = analytic_count(self.n)?;
            return Ok((n - 1.0) * (n + 1.0) / 12.0);
        }
        let mean = self.average_with_work(left)?;
        let norm = self.log_normalizer;
        Ok(math::sum(1, self.n, left, |k| {
            let x = analytic_count(k)?;
            Ok(2.0 * (x - mean).abs().ln() - self.a * x.ln() - norm)
        })?
        .exp())
    }
}
count_capabilities!(Zipfian);
