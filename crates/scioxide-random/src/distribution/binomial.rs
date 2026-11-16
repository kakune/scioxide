//! Binomial distribution.
//!
//! Sampling uses direct Bernoulli trials for very small `n`, binomial inversion
//! (BINV) when `n * min(p, 1 - p) < 10`, and transformed rejection (BTPE)
//! otherwise. Inversion and rejection stop after a fixed number of steps and
//! return an error instead of a truncated sample.
use crate::distribution::LawOptions;

#[cfg(test)]
use alloc::vec::Vec;
use scioxide_core::backend::checked_add;
use scioxide_core::{NumericalIssue, RealScalar, ScalarConversionReason, SciOxideError};

use super::count_math::analytic_count;
use super::discrete_math::binomial_log_mass;
use crate::traits::{BitGenerator, Cdf, Mass};
use crate::util::MAX_SAMPLING_ATTEMPTS;

/// Direct Bernoulli trials have low overhead at this size.
const DIRECT_SIMULATION_MAX_TRIALS: usize = 16;

/// BINV has expected work proportional to the smaller mean.
const BINV_MEAN_THRESHOLD: f64 = 10.0;

/// Exact probability ratios are cheap and stable this close to the mode.
const BTPE_NEAR_MODE_DISTANCE: usize = 20;

#[derive(Debug, Clone)]
enum SamplingMethod {
    Constant,
    Direct { probability: f64 },
    Binv(Binv),
    Btpe(Btpe),
}

/// Immutable constants for binomial inversion.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Binv {
    initial_probability: f64,
    positive_probability: f64,
    recurrence_scale: f64,
    odds: f64,
    mode: usize,
}

impl Binv {
    #[allow(clippy::cast_precision_loss)]
    fn new(n: usize, probability: f64) -> Result<Self, SciOxideError> {
        let n_f64 = n as f64;
        let mode = mode(n, probability)?;
        let odds = probability / (1.0 - probability);

        // Inversion is split into "X = 0?" and a search conditional on X > 0.
        // P(X = 0) = q^n and P(X > 0) = 1 - q^n are formed via `ln_1p` and
        // `expm1` so tiny n p neither rounds q^n to 1 nor loses P(X > 0).
        Ok(Self {
            initial_probability: (n_f64 * (-probability).ln_1p()).exp(),
            positive_probability: -(n_f64 * (-probability).ln_1p()).exp_m1(),
            recurrence_scale: (n_f64 + 1.0) * odds,
            odds,
            mode,
        })
    }
}

/// Immutable proposal and acceptance constants for BTPE.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Btpe {
    n: usize,
    n_f64: f64,
    variance: f64,
    mode: usize,
    proposal_center: f64,
    left: f64,
    right: f64,
    parallelogram_slope: f64,
    triangle_area: f64,
    parallelogram_area: f64,
    left_tail_area: f64,
    total_area: f64,
    left_tail_rate: f64,
    right_tail_rate: f64,
    recurrence_scale: f64,
    odds: f64,
    log_mode_mass: f64,
    probability: f64,
}

impl Btpe {
    #[allow(clippy::cast_precision_loss)]
    #[inline]
    fn new(n: usize, probability: f64) -> Result<Self, SciOxideError> {
        let n_f64 = n as f64;
        let failure_probability = 1.0 - probability;
        let mean = n_f64 * probability;
        let variance = mean * failure_probability;
        let mode = mode(n, probability)?;
        let mode_f64 = mode as f64;
        // Kachitvichyanukul & Schmeiser (1988) BTPE setup. The hat function
        // over the pmf consists of a central triangle of half-width
        // `triangle_area` around mode + 1/2, two parallelogram flanks with
        // slope `parallelogram_slope`, and exponential left/right tails with
        // rates fitted at the triangle edges. The numeric constants (2.195,
        // 4.6, 0.134, 20.5, 15.3) are the published fitted values; the areas
        // below are cumulative so one uniform selects the region.
        let proposal_center = mode_f64 + 0.5;
        let triangle_area = (2.195 * variance.sqrt() - 4.6 * failure_probability).floor() + 0.5;
        let left = proposal_center - triangle_area;
        let right = proposal_center + triangle_area;
        let parallelogram_slope = 0.134 + 20.5 / (15.3 + mode_f64);
        let left_tail_argument =
            (mean + probability - left) / (mean + probability - left * probability);
        let right_tail_argument = (right - mean - probability) / (right * failure_probability);
        let left_tail_rate = left_tail_argument * (1.0 + 0.5 * left_tail_argument);
        let right_tail_rate = right_tail_argument * (1.0 + 0.5 * right_tail_argument);
        let parallelogram_area = triangle_area * (1.0 + 2.0 * parallelogram_slope);
        let left_tail_area = parallelogram_area + parallelogram_slope / left_tail_rate;
        let total_area = left_tail_area + parallelogram_slope / right_tail_rate;
        let odds = probability / failure_probability;
        let log_mode_mass = binomial_log_mass(n_f64, mode_f64, probability);

        let parameters = Self {
            n,
            n_f64,
            variance,
            mode,
            proposal_center,
            left,
            right,
            parallelogram_slope,
            triangle_area,
            parallelogram_area,
            left_tail_area,
            total_area,
            left_tail_rate,
            right_tail_rate,
            recurrence_scale: (n_f64 + 1.0) * odds,
            odds,
            log_mode_mass,
            probability,
        };

        if parameters.constants_are_valid() {
            Ok(parameters)
        } else {
            Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "BTPE constants were non-finite or outside their required ranges",
            ))
        }
    }

    fn constants_are_valid(self) -> bool {
        [
            self.variance,
            self.proposal_center,
            self.left,
            self.right,
            self.parallelogram_slope,
            self.triangle_area,
            self.parallelogram_area,
            self.left_tail_area,
            self.total_area,
            self.left_tail_rate,
            self.right_tail_rate,
            self.recurrence_scale,
            self.odds,
            self.log_mode_mass,
            self.probability,
        ]
        .into_iter()
        .all(f64::is_finite)
            && self.variance > 0.0
            && self.triangle_area > 0.0
            && self.left_tail_rate > 0.0
            && self.right_tail_rate > 0.0
            && self.total_area > 0.0
            && self.log_mode_mass < 0.0
    }
}

/// The mode `floor((n + 1) p)` shared by BINV and BTPE.
#[allow(clippy::cast_precision_loss)]
fn mode(n: usize, probability: f64) -> Result<usize, SciOxideError> {
    let mode = super::count_math::count(((n as f64 + 1.0) * probability).floor())?;
    if mode > n {
        return Err(SciOxideError::scalar_conversion(
            scioxide_core::ConversionType::F64,
            scioxide_core::ConversionType::Usize,
            ScalarConversionReason::OutOfRange,
        ));
    }
    Ok(mode)
}

/// Binomial distribution: the number of successes in `n` independent trials with success
/// probability `p`.
///
/// The mass function is `P(k) = C(n, k) p^k (1 − p)^(n−k)` for `k = 0, …, n`,
/// with mean `n p` and variance `n p (1 − p)`. Any `n` and any `p ∈ [0, 1]`
/// are accepted. `n = 0`, `p = 0` and `p = 1` give point masses. The mass
/// function uses Loader's saddle-point form, which avoids cancellation for
/// large `n`.
///
/// Quantiles start from a Cornish–Fisher approximation and finish with a
/// discrete search. The search returns the smallest count that reaches the
/// requested probability, so the approximation only sets the starting point.
/// The search and the entropy sum count toward the
/// [work limit](crate#work-limits). Each call of `sample`, `sample_n` or
/// `sample_n_into` ([`CountDistribution`](crate::CountDistribution)) prepares
/// a [`BinomialSampler`] without allocating and draws through it; for a loop of
/// single draws, prepare it once with [`Binomial::prepare_sampler`].
///
/// ```
/// use scioxide_random::{Binomial, Cdf, CountDistribution, Mass, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// let law = Binomial::new(10, 0.3_f64)?;
/// assert!((law.pmf(3)? - 0.266_827_932).abs() < 1e-12);
/// assert!((law.cdf(3)? - 0.649_610_718_4).abs() < 1e-12);
/// assert!((law.sf(3)? - 0.350_389_281_6).abs() < 1e-12);
/// assert_eq!(law.quantile(0.5)?, 3);
/// let draws = law.sample_n(&mut SplitMix64::from_seed(42), 100)?;
/// assert!(draws.iter().all(|&k| k <= 10));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// C. Loader, "Fast and accurate computation of binomial probabilities", 2000.
#[derive(Debug, Clone)]
#[must_use]
pub struct Binomial<S: RealScalar> {
    options: LawOptions<S>,
    n: usize,
    prob: S,
    /// Loader's mass constants; `None` beyond the exact-count range.
    mass: Option<super::discrete_math::BinomialMass>,
}

/// Binomial sampler with its setup constants computed once.
///
/// The sampler works with `p' = min(p, 1 − p)` and reflects the count when
/// `p > 1/2`. It counts direct Bernoulli trials for `n ≤ 16`, uses inversion
/// (BINV) when `n p' < 10`, and uses BTPE transformed rejection otherwise.
///
/// A draw through [`CountDistribution`](crate::CountDistribution) reads at most
/// 272 uniform words for direct trials, 18 for inversion and 2,048 for BTPE.
/// These counts are also limited by [`LawOptions::sample_word_budget`].
/// Inversion and rejection stop after 1,024 steps or candidates.
///
/// ```
/// use scioxide_random::{Binomial, CountDistribution, SeedableRng, SplitMix64};
/// let sampler = Binomial::new(1000, 0.4_f64)?.prepare_sampler()?;
/// let mut rng = SplitMix64::from_seed(7);
/// for _ in 0..4 {
///     assert!(sampler.sample(&mut rng)? <= 1000);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// V. Kachitvichyanukul and B. W. Schmeiser, "Binomial random variate
/// generation", *Commun. ACM* 31(2), 1988, 216–222.
#[derive(Debug, Clone)]
#[must_use]
pub struct BinomialSampler {
    n: usize,
    reflected: bool,
    sampling_method: SamplingMethod,
    limits: super::count_math::DrawLimits,
    work: usize,
}

/// Every method computes in f64, and counts beyond 2^52 are not all exactly
/// representable, so nondegenerate laws with more trials are refused
/// rather than sampled with a silently discretized support.
pub(super) fn sampled_trials(n: usize) -> Result<(), SciOxideError> {
    if (n as u128) >= (1_u128 << 52) {
        return Err(SciOxideError::not_implemented(
            "binomial sampling above 2^52 - 1 trials",
        ));
    }
    Ok(())
}

impl<S: RealScalar> Binomial<S> {
    /// Creates a binomial law with `n` trials and success probability `p`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `p` is NaN or infinite.
    /// - `DomainError` if `p` is outside `[0, 1]`.
    pub fn new(n: usize, p: S) -> Result<Self, SciOxideError> {
        Self::with_options(n, p, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`].
    pub fn with_options(n: usize, p: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        let mass = analytic_count(n)
            .ok()
            .map(|trials| super::discrete_math::BinomialMass::new(trials, p.into_f64()));
        Ok(Self {
            options,
            n,
            prob: p,
            mass,
        })
    }

    /// Returns a sampler with the setup constants for this law.
    ///
    /// The setup reads no random words. Point-mass laws accept every `n`.
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if `0 < p < 1` and `n ≥ 2⁵²`.
    /// - `NumericalInstability` if the BTPE setup constants are not finite.
    pub fn prepare_sampler(&self) -> Result<BinomialSampler, SciOxideError> {
        let n = self.n;
        let p = self.prob.into_f64();
        // Sample with min(p, 1 - p) and reflect k -> n - k afterwards. Every
        // method below assumes p <= 1/2 (BINV's recurrence stability and the
        // BTPE constants were fitted for that range).
        let reflected = p > 0.5;
        let working_probability = if reflected { 1.0 - p } else { p };
        let sampling_method = SamplingMethod::prepare(n, working_probability)?;
        Ok(BinomialSampler {
            n,
            reflected,
            sampling_method,
            limits: super::count_math::DrawLimits::of(self.options),
            work: self.options.work_budget().get(),
        })
    }

    /// Loader's log mass at `k` from the prepared constants.
    fn log_mass(&self, k: usize) -> Result<f64, SciOxideError> {
        let mass = match self.mass {
            Some(mass) => mass,
            None => super::discrete_math::BinomialMass::new(
                analytic_count(self.n)?,
                self.prob.into_f64(),
            ),
        };
        Ok(mass.log_mass(analytic_count(k)?))
    }

    /// `ln P(X <= k)` (`upper`: `ln P(X > k)`) for `0 <= k < n` and `0 < p < 1`
    /// from the tail away from the mode, summed outward from its largest mass by
    /// the positive pmf ratio recurrence; the requested side is its `ln1p`
    /// complement. Terms are log-concave, so the sum stops once the geometric
    /// bound on the rest is below an eighth of an epsilon. `None` when the
    /// starting mass is not normal or the tail is longer than `2 DIRECT_TERMS`.
    fn direct_log_tail(
        &self,
        k: usize,
        upper: bool,
        left: &mut usize,
    ) -> Result<Option<f64>, SciOxideError> {
        let (n, p) = (self.n, self.prob.into_f64());
        let n_scalar = analytic_count(n)?;
        let lower_side = analytic_count(k)? <= n_scalar * p;
        let start = if lower_side { k } else { k + 1 };
        let mass = self.log_mass(start)?.exp();
        if !mass.is_normal() {
            return Ok(None);
        }
        // pmf(i + 1) / pmf(i) = (n - i) p / ((i + 1) q).
        let odds = p / (1.0 - p);
        let (mut sum, mut term, mut index) = (mass, mass, start);
        let mut complete = false;
        for _ in 0..2 * super::count_math::DIRECT_TERMS {
            // The support end completes the sum exactly.
            if (lower_side && index == 0) || (!lower_side && index == n) {
                complete = true;
                break;
            }
            crate::law::charge(left)?;
            let ratio = if lower_side {
                index as f64 / ((n - index + 1) as f64 * odds)
            } else {
                (n - index) as f64 * odds / (index + 1) as f64
            };
            term *= ratio;
            sum += term;
            index = if lower_side { index - 1 } else { index + 1 };
            if ratio < 1.0 && term * ratio / (1.0 - ratio) <= 0.125 * f64::EPSILON * sum {
                complete = true;
                break;
            }
        }
        if !complete {
            return Ok(None);
        }
        let tail = sum.min(1.0);
        Ok(Some(if upper != lower_side {
            tail.ln()
        } else {
            (-tail).ln_1p()
        }))
    }

    fn log_cdf_work(&self, k: usize, left: &mut usize) -> Result<S, SciOxideError> {
        if k >= self.n || self.prob == S::ZERO {
            return Ok(S::ZERO);
        }
        if self.prob == S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        if let Some(value) = self.direct_log_tail(k, false, left)? {
            return S::try_from_f64_lossy(value);
        }
        let a = analytic_count(k + 1)?;
        let b = analytic_count(self.n - k)?;
        S::try_from_f64_lossy(crate::law::special(left, |budget| {
            scioxide_special::beta_inc_log_q_with_options(
                a,
                b,
                self.prob.into_f64(),
                scioxide_special::BetaOptions::new(budget),
            )
        })?)
    }

    fn log_sf_work(&self, k: usize, left: &mut usize) -> Result<S, SciOxideError> {
        if k >= self.n || self.prob == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if self.prob == S::ONE {
            return Ok(S::ZERO);
        }
        if let Some(value) = self.direct_log_tail(k, true, left)? {
            return S::try_from_f64_lossy(value);
        }
        // P(X > k) = I_p(k + 1, n - k), evaluated directly in log form so
        // upper tails far below the smallest normal float stay representable.
        let a = analytic_count(k + 1)?;
        let b = analytic_count(self.n - k)?;
        S::try_from_f64_lossy(crate::law::special(left, |budget| {
            scioxide_special::beta_inc_log_with_options(
                a,
                b,
                self.prob.into_f64(),
                scioxide_special::BetaOptions::new(budget),
            )
        })?)
    }
}

impl SamplingMethod {
    // The law or conditional-count caller has already validated 0 <= p <= 1/2.
    fn prepare(n: usize, probability: f64) -> Result<Self, SciOxideError> {
        debug_assert!((0.0..=0.5).contains(&probability));
        Ok(if n == 0 || probability == 0.0 {
            SamplingMethod::Constant
        } else {
            sampled_trials(n)?;
            if n <= DIRECT_SIMULATION_MAX_TRIALS {
                SamplingMethod::Direct { probability }
            } else if (n as f64) * probability < BINV_MEAN_THRESHOLD {
                SamplingMethod::Binv(Binv::new(n, probability)?)
            } else {
                SamplingMethod::Btpe(Btpe::new(n, probability)?)
            }
        })
    }
    #[inline]
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        left: &mut usize,
    ) -> Result<usize, SciOxideError> {
        match self {
            Self::Constant => Ok(0),
            Self::Direct { probability } => direct_sim(n, *probability, rng, left),
            Self::Binv(parameters) => binv(n, parameters, rng, left),
            Self::Btpe(parameters) => btpe(n, parameters, rng, left),
        }
    }
}

/// Conditional count kernel with a validated smaller-tail probability. The
/// enclosing vector owns word accounting; only trial-dependent work remains here.
pub(super) fn conditional_binomial<R: BitGenerator + ?Sized>(
    n: usize,
    probability: f64,
    rng: &mut R,
    left: &mut usize,
) -> Result<usize, SciOxideError> {
    let sample = SamplingMethod::prepare(n, probability)?.draw(n, rng, left)?;
    validate_and_reflect_sample(n, sample, false)
}

/// Direct simulation: count how many of `n` uniform draws fall below `p`.
#[inline(always)]
fn direct_sim<R: BitGenerator + ?Sized>(
    n: usize,
    p: f64,
    rng: &mut R,
    left: &mut usize,
) -> Result<usize, SciOxideError> {
    // Decode a dyadic probability once per small batch when it fits in one
    // raw word. Finer probabilities retain the exact multiword comparison.
    let scaled = p * 18_446_744_073_709_551_616.0;
    let threshold = scaled as u64;
    if scaled == threshold as f64 {
        (0..n).try_fold(0, |count, _| {
            crate::law::charge(left)?;
            Ok(count + usize::from(rng.next_uniform_u64()? < threshold))
        })
    } else {
        (0..n).try_fold(0, |count, _| {
            crate::law::charge(left)?;
            Ok(count + usize::from(super::discrete_math::bernoulli(p, rng)?))
        })
    }
}

/// Resolve the zero atom, then invert conditional positive masses by ratios.
#[inline(always)]
fn binv<R: BitGenerator + ?Sized>(
    n: usize,
    parameters: &Binv,
    rng: &mut R,
    left: &mut usize,
) -> Result<usize, SciOxideError> {
    if !super::discrete_math::bernoulli(parameters.positive_probability, rng)? {
        return Ok(0);
    }
    // Sequential search of the conditional CDF starting from
    // P(X = 1 | X > 0) = n p q^(n-1) / (1 - q^n). Each step applies the pmf
    // ratio p_{k+1}/p_k = ((n + 1)/(k + 1) - 1) * p/q, so no factorials or
    // powers are evaluated inside the loop.
    let mut residual = super::words::uniform(rng)?;
    let mut probability = parameters.initial_probability
        * ((parameters.recurrence_scale - parameters.odds) / parameters.positive_probability);
    for candidate in 1..=MAX_SAMPLING_ATTEMPTS {
        crate::law::charge(left)?;
        if residual < probability && candidate <= n {
            return Ok(candidate);
        }
        if candidate >= n {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "CDF inversion passed the upper support boundary",
            ));
        }
        residual -= probability;
        let ratio = parameters.recurrence_scale / (candidate + 1) as f64 - parameters.odds;
        probability *= ratio;
        debug_assert!(candidate <= parameters.mode || ratio <= 1.0);
    }

    Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BtpeProposal {
    Immediate(usize),
    RejectionTest { candidate: usize, acceptance: f64 },
    Reject,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
#[inline]
fn proposal_candidate(value: f64, n: usize) -> Option<usize> {
    let candidate = value.floor();
    let usize_upper_bound = usize::MAX as f64 + 1.0;
    if candidate >= 0.0 && candidate < usize_upper_bound && candidate <= n as f64 {
        Some(candidate as usize)
    } else {
        None
    }
}

#[inline]
fn btpe_proposal(parameters: &Btpe, area_uniform: f64, shape_uniform: f64) -> BtpeProposal {
    let area = area_uniform * parameters.total_area;

    // Region 1 (triangle): lies entirely under the pmf, so the candidate is
    // accepted without evaluating any mass. This is the common case.
    if area <= parameters.triangle_area {
        let value = parameters.proposal_center - parameters.triangle_area * shape_uniform + area;
        return proposal_candidate(value, parameters.n)
            .map_or(BtpeProposal::Reject, BtpeProposal::Immediate);
    }

    // Region 2 (parallelograms): candidate is uniform in x; the acceptance
    // height is rescaled so that values above 1 fall outside the hat.
    if area <= parameters.parallelogram_area {
        let value =
            parameters.left + (area - parameters.triangle_area) / parameters.parallelogram_slope;
        let acceptance = shape_uniform * parameters.parallelogram_slope + 1.0
            - (value - parameters.proposal_center).abs() / parameters.triangle_area;
        if acceptance > 1.0 {
            return BtpeProposal::Reject;
        }
        return proposal_candidate(value, parameters.n).map_or(BtpeProposal::Reject, |candidate| {
            BtpeProposal::RejectionTest {
                candidate,
                acceptance,
            }
        });
    }

    if shape_uniform <= 0.0 {
        return BtpeProposal::Reject;
    }

    // Regions 3 and 4: exponential tails beyond the triangle edges, sampled
    // by inversion of the exponential hat.
    if area <= parameters.left_tail_area {
        let value = parameters.left + shape_uniform.ln() / parameters.left_tail_rate;
        let acceptance =
            shape_uniform * (area - parameters.parallelogram_area) * parameters.left_tail_rate;
        return proposal_candidate(value, parameters.n).map_or(BtpeProposal::Reject, |candidate| {
            BtpeProposal::RejectionTest {
                candidate,
                acceptance,
            }
        });
    }

    let value = parameters.right - shape_uniform.ln() / parameters.right_tail_rate;
    let acceptance =
        shape_uniform * (area - parameters.left_tail_area) * parameters.right_tail_rate;
    proposal_candidate(value, parameters.n).map_or(BtpeProposal::Reject, |candidate| {
        BtpeProposal::RejectionTest {
            candidate,
            acceptance,
        }
    })
}

#[allow(clippy::cast_precision_loss)]
#[inline]
fn near_mode_mass_ratio(parameters: &Btpe, candidate: usize) -> f64 {
    if candidate > parameters.mode {
        ((parameters.mode + 1)..=candidate).fold(1.0, |ratio, value| {
            ratio * (parameters.recurrence_scale / value as f64 - parameters.odds)
        })
    } else {
        ((candidate + 1)..=parameters.mode).fold(1.0, |ratio, value| {
            ratio / (parameters.recurrence_scale / value as f64 - parameters.odds)
        })
    }
}

#[allow(clippy::cast_precision_loss)]
#[inline]
fn btpe_accepts(parameters: &Btpe, candidate: usize, acceptance: f64) -> bool {
    if acceptance <= 0.0 {
        return true;
    }

    // Near the mode, the exact ratio f(k)/f(mode) is cheapest as a short
    // product of pmf recurrence ratios.
    let distance = candidate.abs_diff(parameters.mode);
    if distance <= BTPE_NEAR_MODE_DISTANCE {
        return acceptance <= near_mode_mass_ratio(parameters, candidate);
    }

    let distance_f64 = distance as f64;
    let log_acceptance = acceptance.ln();
    // Kachitvichyanukul & Schmeiser's BTPE squeeze is valid only for
    // 20 < d < npq/2 - 1. Outside that interval compare exact log masses.
    if distance_f64 < parameters.variance / 2.0 - 1.0 {
        let scaled_distance = distance_f64 / parameters.variance;
        let rho = scaled_distance * (distance_f64 * (distance_f64 / 3.0 + 0.625) + 1.0 / 6.0)
            / parameters.variance
            + 0.5 * scaled_distance;
        let center = -distance_f64 * distance_f64 / (2.0 * parameters.variance);
        if log_acceptance < center - rho {
            return true;
        }
        if log_acceptance > center + rho {
            return false;
        }
    }

    let log_mass_ratio =
        binomial_log_mass(parameters.n_f64, candidate as f64, parameters.probability)
            - parameters.log_mode_mass;

    log_acceptance <= log_mass_ratio
}

/// BTPE transformed rejection for medium and large means.
#[inline(always)]
fn btpe<R: BitGenerator + ?Sized>(
    n: usize,
    parameters: &Btpe,
    rng: &mut R,
    left: &mut usize,
) -> Result<usize, SciOxideError> {
    for _ in 0..MAX_SAMPLING_ATTEMPTS {
        crate::law::charge(left)?;
        let area_uniform = super::words::uniform(rng)?;
        let shape_uniform = super::words::uniform(rng)?;
        match btpe_proposal(parameters, area_uniform, shape_uniform) {
            BtpeProposal::Immediate(candidate) => return Ok(candidate),
            BtpeProposal::RejectionTest {
                candidate,
                acceptance,
            } if btpe_accepts(parameters, candidate, acceptance) => return Ok(candidate),
            BtpeProposal::RejectionTest { .. } | BtpeProposal::Reject => {}
        }
    }

    debug_assert!(n > 0);
    Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
}

#[inline(always)]
fn validate_and_reflect_sample(
    n: usize,
    sample: usize,
    reflected: bool,
) -> Result<usize, SciOxideError> {
    if sample > n {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "sampling method produced a value outside the distribution support",
        ));
    }

    Ok(if reflected { n - sample } else { sample })
}

impl BinomialSampler {
    /// Uncharged draw; nested callers share their own word allowance.
    #[inline]
    pub(super) fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<usize, SciOxideError> {
        let sample = self.sampling_method.draw(self.n, rng, &mut { self.work })?;

        validate_and_reflect_sample(self.n, sample, self.reflected)
    }
    fn limits(&self) -> super::count_math::DrawLimits {
        self.limits
    }
}
super::count_math::count_distribution!([] BinomialSampler, usize);
super::count_math::prepared_count_distribution!(Binomial, usize);

impl<S: RealScalar> Mass<S> for Binomial<S> {
    fn pmf(&self, k: usize) -> Result<S, SciOxideError> {
        if k > self.n {
            return Ok(S::ZERO);
        }

        // Handle edge cases
        if self.prob == S::ZERO {
            return if k == 0 { Ok(S::ONE) } else { Ok(S::ZERO) };
        }
        if self.prob == S::ONE {
            return if k == self.n { Ok(S::ONE) } else { Ok(S::ZERO) };
        }

        if self.n == 0 {
            return Ok(S::ONE);
        }
        S::try_from_f64_lossy(self.log_mass(k)?.exp())
    }
}

impl<S: RealScalar> Cdf<S, usize> for Binomial<S> {
    fn cdf(&self, k: usize) -> Result<S, SciOxideError> {
        S::try_from_f64_lossy(binomial_cdf(
            self.n,
            self.prob.into_f64(),
            k,
            &mut self.options.work_budget().get(),
        )?)
    }
}

fn binomial_cdf(n: usize, p: f64, k: usize, left: &mut usize) -> Result<f64, SciOxideError> {
    if k >= n {
        return Ok(1.0);
    }
    if p == 0.0 {
        return Ok(1.0);
    }
    if p == 1.0 {
        return Ok(0.0);
    }

    let lower_terms = checked_add(k, 1, scioxide_core::quantity!("binomial_lower_cdf_terms"))?;
    let upper_terms = n - k;
    let n_scalar = analytic_count(n)?;
    let k_scalar = analytic_count(k)?;
    let mean = n_scalar * p;
    // Short explicit sums are used only for tails on the far side of the mean,
    // where terms increase monotonically toward k. Such sums of positive
    // terms have no cancellation and are cheaper than the incomplete beta.
    if lower_terms <= super::count_math::DIRECT_TERMS && k_scalar <= mean {
        let zero_mass = (n_scalar * (-p).ln_1p()).exp();
        let cdf = if zero_mass >= f64::MIN_POSITIVE {
            // A normal zero atom retains relative precision throughout this
            // short increasing recurrence and avoids log-mass setup.
            (0..k)
                .try_fold((zero_mass, zero_mass), |(sum, mass), index| {
                    crate::law::charge(left)?;
                    let next = mass * (n - index) as f64 * p / ((index + 1) as f64 * (1.0 - p));
                    Ok::<_, SciOxideError>((sum + next, next))
                })?
                .0
        } else {
            // Start at the largest mass when the zero atom underflows.
            let initial = binomial_log_mass(n_scalar, k_scalar, p).exp();
            (1..=k)
                .rev()
                .try_fold((initial, initial), |(sum, mass), index| {
                    crate::law::charge(left)?;
                    let next = (mass / p) * (index as f64 / (n - index + 1) as f64) * (1.0 - p);
                    Ok::<_, SciOxideError>((sum + next, next))
                })?
                .0
        };
        if (0.0..=1.0).contains(&cdf) {
            return Ok(cdf);
        }
    } else if upper_terms <= super::count_math::DIRECT_TERMS && k_scalar >= mean {
        let end_mass = (n_scalar * p.ln()).exp();
        let survival = if end_mass >= f64::MIN_POSITIVE {
            (0..(upper_terms - 1))
                .try_fold((end_mass, end_mass), |(sum, mass), step| {
                    crate::law::charge(left)?;
                    let index = n - step;
                    let next = mass * index as f64 * (1.0 - p) / ((n - index + 1) as f64 * p);
                    Ok::<_, SciOxideError>((sum + next, next))
                })?
                .0
        } else {
            let initial = binomial_log_mass(n_scalar, (k + 1) as f64, p).exp();
            ((k + 1)..n)
                .try_fold((initial, initial), |(sum, mass), index| {
                    crate::law::charge(left)?;
                    let next = (mass / (1.0 - p)) * ((n - index) as f64 / (index + 1) as f64) * p;
                    Ok::<_, SciOxideError>((sum + next, next))
                })?
                .0
        };
        let cdf = 1.0 - survival;
        if (0.0..=1.0).contains(&cdf) {
            return Ok(cdf);
        }
    }

    // Reflect only for p > 1/2, where Sterbenz's lemma makes 1-p exact.
    // Tiny p must enter the complementary beta directly to retain its digits.
    let k_plus1 = checked_add(k, 1, scioxide_core::quantity!("binomial_cdf_successes"))?;
    let k_plus1 = analytic_count(k_plus1)?;
    let n_minus_k = analytic_count(n - k)?;
    if p > 0.5 {
        crate::law::special(left, |budget| {
            scioxide_special::beta_inc_with_options(
                n_minus_k,
                k_plus1,
                1.0 - p,
                scioxide_special::BetaOptions::new(budget),
            )
        })
    } else {
        crate::law::special(left, |budget| {
            scioxide_special::beta_inc_q_with_options(
                k_plus1,
                n_minus_k,
                p,
                scioxide_special::BetaOptions::new(budget),
            )
        })
    }
}

impl<S: RealScalar> crate::LogSurvival<S, usize> for Binomial<S> {
    fn log_sf(&self, k: usize) -> Result<S, SciOxideError> {
        self.log_sf_work(k, &mut self.options.work_budget().get())
    }

    fn log_sf_scoped(
        &self,
        k: usize,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_sf_work(k, left)
        })
    }
}

impl<S: RealScalar> crate::LogCdf<S, usize> for Binomial<S> {
    fn log_cdf(&self, k: usize) -> Result<S, SciOxideError> {
        self.log_cdf_work(k, &mut self.options.work_budget().get())
    }

    fn log_cdf_scoped(
        &self,
        k: usize,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_cdf_work(k, left)
        })
    }
}

impl<S: RealScalar> crate::Survival<S, usize> for Binomial<S> {
    fn sf(&self, k: usize) -> Result<S, SciOxideError> {
        S::try_from_f64_lossy(
            crate::LogSurvival::log_sf(
                &Binomial::with_options(self.n, self.prob.into_f64(), self.options.widened()?)?,
                k,
            )?
            .exp(),
        )
    }
}

impl<S: RealScalar> crate::LogMass<S> for Binomial<S> {
    fn log_pmf(&self, k: usize) -> Result<S, SciOxideError> {
        if k > self.n {
            return Ok(S::NEG_INFINITY);
        }
        if self.prob == S::ZERO || self.prob == S::ONE {
            return Ok(self.pmf(k)?.ln());
        }
        S::try_from_f64_lossy(self.log_mass(k)?)
    }
}

impl<S: RealScalar> Binomial<S> {
    fn inverse_probability(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<usize, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        let probability = p.into_f64();
        let success = self.prob.into_f64();
        let guess = if probability == 0.0
            || probability == 1.0
            || success == 0.0
            || success == 1.0
            || self.n == 0
        {
            None
        } else {
            let smaller = if probability > 0.5 {
                1.0 - probability
            } else {
                probability
            };
            let z = scioxide_special::normal_quantile(smaller)?;
            let z = if upper != (probability > 0.5) { -z } else { z };
            let mean = analytic_count(self.n)? * success;
            let sigma = (mean * (1.0 - success)).sqrt();
            // First Cornish-Fisher skewness correction, with continuity rounding.
            let start = mean + sigma * z + (1.0 - 2.0 * success) * (z * z - 1.0) / 6.0;
            Some(start.round().clamp(0.0, self.n as f64) as usize)
        };
        // The law's own tails decide every step, so the threshold is exact.
        let (lower, maximum) = crate::CountSupport::count_support(self);
        super::count_math::inverse(p, upper, lower, maximum, guess, left, |k, upper, left| {
            Ok(if upper {
                self.log_sf_work(k, left)?
            } else {
                self.log_cdf_work(k, left)?
            }
            .into_f64())
        })
    }
}

impl<S: RealScalar> crate::Quantile<S, usize> for Binomial<S> {
    fn quantile(&self, p: S) -> Result<usize, SciOxideError> {
        self.inverse_probability(p, false, &mut self.options.work_budget().get())
    }
    fn quantile_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<usize, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.inverse_probability(p, false, left)
        })
    }
}
impl<S: RealScalar> crate::InverseSurvival<S, usize> for Binomial<S> {
    fn isf(&self, p: S) -> Result<usize, SciOxideError> {
        self.inverse_probability(p, true, &mut self.options.work_budget().get())
    }
    fn isf_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<usize, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.inverse_probability(p, true, left)
        })
    }
}

impl<S: RealScalar> crate::Mean<S> for Binomial<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        crate::law::finite(S::try_from_f64_lossy(analytic_count(self.n)?)? * self.prob)
    }
}
impl<S: RealScalar> crate::Variance<S> for Binomial<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        crate::law::finite(crate::Mean::mean(self)? * (S::ONE - self.prob))
    }
}

impl<S: RealScalar> crate::CountSupport for Binomial<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        if self.prob == S::ZERO {
            (0, Some(0))
        } else if self.prob == S::ONE {
            (self.n, Some(self.n))
        } else {
            (0, Some(self.n))
        }
    }
}
impl<S: RealScalar> crate::Entropy<S> for Binomial<S> {
    /// Returns the entropy `−Σ P(k) ln P(k)` as a finite sum over `k = 0, …, n`.
    ///
    /// The sum takes O(n) time and O(1) storage. A point-mass law returns zero.
    /// Otherwise the work limit in [`LawOptions`] must cover `n + 1` terms, or
    /// the call returns `ConvergenceFailure` before summing.
    fn entropy(&self) -> Result<S, SciOxideError> {
        self.entropy_with_work(&mut self.options.work_budget().get())
    }
    fn entropy_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.entropy_with_work(left)
        })
    }
}
impl<S: RealScalar> Binomial<S> {
    fn entropy_with_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        if self.prob == S::ZERO || self.prob == S::ONE || self.n == 0 {
            return Ok(S::ZERO);
        }
        if self.n >= *left {
            return Err(crate::law::iteration_limit(0));
        }
        (0..=self.n).try_fold(S::ZERO, |sum, k| {
            crate::law::charge(left)?;
            let l = crate::LogMass::log_pmf(self, k)?;
            Ok(if l == S::NEG_INFINITY {
                sum
            } else {
                sum - l.exp() * l
            })
        })
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::super::test_rng::ScriptedRng;
    use super::*;
    use crate::CountDistribution;
    use crate::prng::SplitMix64;
    use crate::traits::SeedableRng;

    #[test]
    fn test_new_valid() {
        assert!(Binomial::<f64>::new(10, 0.5).is_ok());
    }

    #[test]
    fn test_f32_and_f64_constructors_choose_equivalent_precomputation() {
        for (n, probability_f32) in [
            (16_usize, 0.25_f32),
            (1_000, 1.0_f32 / 1_024.0),
            (1_000, 0.25_f32),
        ] {
            let f32_distribution = Binomial::new(n, probability_f32).unwrap();
            let f64_distribution = Binomial::new(n, f64::from(probability_f32)).unwrap();
            match (
                f32_distribution.prepare_sampler().unwrap().sampling_method,
                f64_distribution.prepare_sampler().unwrap().sampling_method,
            ) {
                (SamplingMethod::Direct { .. }, SamplingMethod::Direct { .. }) => {}
                (SamplingMethod::Binv(f32_parameters), SamplingMethod::Binv(f64_parameters)) => {
                    assert_eq!(f32_parameters, f64_parameters);
                }
                (SamplingMethod::Btpe(f32_parameters), SamplingMethod::Btpe(f64_parameters)) => {
                    assert_eq!(f32_parameters, f64_parameters);
                }
                (f32_method, f64_method) => {
                    panic!("constructor mismatch: f32={f32_method:?}, f64={f64_method:?}");
                }
            }
        }
    }

    #[test]
    fn test_sampling_methods_switch_at_documented_boundaries() {
        let constant = Binomial::new(usize::MAX, 0.0_f64).unwrap();
        let direct = Binomial::new(DIRECT_SIMULATION_MAX_TRIALS, 0.5_f64).unwrap();
        let binv_after_direct = Binomial::new(DIRECT_SIMULATION_MAX_TRIALS + 1, 0.5_f64).unwrap();
        let binv_below_mean = Binomial::new(1_000, 0.009_999_f64).unwrap();
        let btpe_at_mean = Binomial::new(1_000, 0.01_f64).unwrap();

        assert!(matches!(
            constant.prepare_sampler().unwrap().sampling_method,
            SamplingMethod::Constant
        ));
        assert!(matches!(
            direct.prepare_sampler().unwrap().sampling_method,
            SamplingMethod::Direct { .. }
        ));
        assert!(matches!(
            binv_after_direct.prepare_sampler().unwrap().sampling_method,
            SamplingMethod::Binv(_)
        ));
        assert!(matches!(
            binv_below_mean.prepare_sampler().unwrap().sampling_method,
            SamplingMethod::Binv(_)
        ));
        assert!(matches!(
            btpe_at_mean.prepare_sampler().unwrap().sampling_method,
            SamplingMethod::Btpe(_)
        ));
    }

    #[test]
    fn test_new_zero_n() {
        let law = Binomial::<f64>::new(0, 0.5).unwrap();
        assert_eq!(law.pmf(0).unwrap(), 1.0);
    }

    #[test]
    fn test_new_p_out_of_range() {
        let error = Binomial::<f64>::new(10, 1.5).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_p_non_finite() {
        let error = Binomial::<f64>::new(10, f64::NAN).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_range() {
        let dist = Binomial::new(10_usize, 0.5_f64).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..1000 {
            let k = dist.prepare_sampler().unwrap().sample(&mut rng).unwrap();
            assert!(k <= 10);
        }
    }

    #[test]
    fn test_direct_path_matches_bernoulli_simulation() {
        let dist = Binomial::new(16_usize, 0.25_f64).unwrap();
        let mut actual_rng = SplitMix64::from_seed(73);
        let mut expected_rng = SplitMix64::from_seed(73);

        for _ in 0..100 {
            let expected = direct_sim(16, 0.25, &mut expected_rng, &mut 65_536).unwrap();
            assert_eq!(
                dist.prepare_sampler()
                    .unwrap()
                    .sample(&mut actual_rng)
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn test_n_one_handles_valid_uniform_endpoints() {
        let distribution = Binomial::new(1_usize, 0.5_f64).unwrap();
        let mut zero_rng = ScriptedRng::constant(0.0);
        let mut max_rng = ScriptedRng::constant(1.0 - f64::EPSILON);

        assert_eq!(
            distribution
                .prepare_sampler()
                .unwrap()
                .sample(&mut zero_rng)
                .unwrap(),
            1
        );
        assert_eq!(
            distribution
                .prepare_sampler()
                .unwrap()
                .sample(&mut max_rng)
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_endpoints_are_domain_values_without_random_draws() {
        let zero = Binomial::new(usize::MAX, 0.0_f64).unwrap();
        let one = Binomial::new(usize::MAX, 1.0_f64).unwrap();
        let mut rng = ScriptedRng::constant(0.5);

        assert_eq!(zero.prepare_sampler().unwrap().sample(&mut rng).unwrap(), 0);
        assert_eq!(
            one.prepare_sampler().unwrap().sample(&mut rng).unwrap(),
            usize::MAX
        );
        assert_eq!(rng.draws, 0);
    }

    #[test]
    fn test_binv_zero_uniform_returns_no_successes() {
        let dist = Binomial::new(1_000_000_usize, 0.000_001_f64).unwrap();
        let mut rng = ScriptedRng::constant(1.0_f64.next_down());

        assert_eq!(dist.prepare_sampler().unwrap().sample(&mut rng).unwrap(), 0);
        assert_eq!(rng.draws, 1);
    }

    #[test]
    fn test_binv_precomputes_initial_and_mode_probabilities() {
        let n = 1_000_usize;
        let probability = 0.001_f64;
        let parameters = Binv::new(n, probability).unwrap();
        let distribution = Binomial::new(n, probability).unwrap();
        let mut mode_probability = parameters.initial_probability;
        for candidate in 1..=parameters.mode {
            mode_probability *= parameters.recurrence_scale / candidate as f64 - parameters.odds;
        }

        assert_eq!(parameters.mode, 1);
        assert!((parameters.initial_probability - distribution.pmf(0).unwrap()).abs() < 1e-14);
        assert!((mode_probability - distribution.pmf(parameters.mode).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_btpe_triangle_and_parallelogram_proposals() {
        let dist = Binomial::new(20_usize, 0.5_f64).unwrap();
        let SamplingMethod::Btpe(parameters) = dist.prepare_sampler().unwrap().sampling_method
        else {
            panic!("expected BTPE");
        };
        let triangle_uniform = 0.5 * parameters.triangle_area / parameters.total_area;
        let triangle = btpe_proposal(&parameters, triangle_uniform, 0.5);
        assert_eq!(triangle, BtpeProposal::Immediate(parameters.mode));

        let parallelogram_area = parameters.triangle_area
            + 0.5 * parameters.triangle_area * parameters.parallelogram_slope;
        let parallelogram =
            btpe_proposal(&parameters, parallelogram_area / parameters.total_area, 0.0);
        let BtpeProposal::RejectionTest {
            candidate,
            acceptance,
        } = parallelogram
        else {
            panic!("expected a parallelogram proposal");
        };
        assert_eq!(candidate, 9);
        assert!(btpe_accepts(&parameters, candidate, acceptance));

        let mut triangle_rng = ScriptedRng::new(vec![triangle_uniform, 0.5], 0.0);
        let mut parallelogram_rng =
            ScriptedRng::new(vec![parallelogram_area / parameters.total_area, 0.0], 0.0);

        assert_eq!(
            dist.prepare_sampler()
                .unwrap()
                .sample(&mut triangle_rng)
                .unwrap(),
            parameters.mode
        );
        assert_eq!(triangle_rng.draws, 2);
        assert_eq!(
            dist.prepare_sampler()
                .unwrap()
                .sample(&mut parallelogram_rng)
                .unwrap(),
            candidate
        );
        assert_eq!(parallelogram_rng.draws, 2);
    }

    #[test]
    fn test_btpe_tail_proposals_are_bounded() {
        let dist = Binomial::new(20_usize, 0.5_f64).unwrap();
        let SamplingMethod::Btpe(parameters) = dist.prepare_sampler().unwrap().sampling_method
        else {
            panic!("expected BTPE");
        };
        let left_area = 0.5 * (parameters.parallelogram_area + parameters.left_tail_area);
        let right_area = 0.5 * (parameters.left_tail_area + parameters.total_area);

        for proposal in [
            btpe_proposal(&parameters, left_area / parameters.total_area, 0.5),
            btpe_proposal(&parameters, right_area / parameters.total_area, 0.5),
        ] {
            let BtpeProposal::RejectionTest { candidate, .. } = proposal else {
                panic!("expected a tail proposal");
            };
            assert!(candidate <= 20);
        }
    }

    #[test]
    fn test_btpe_far_acceptance_uses_squeeze_and_stirling_fallback() {
        let dist = Binomial::new(1_000_usize, 0.5_f64).unwrap();
        let SamplingMethod::Btpe(parameters) = dist.prepare_sampler().unwrap().sampling_method
        else {
            panic!("expected BTPE");
        };
        let candidate = parameters.mode - 50;
        let distance = candidate.abs_diff(parameters.mode) as f64;
        let scaled_distance = distance / parameters.variance;
        let rho = scaled_distance
            * ((distance * (distance / 3.0 + 0.625) + 1.0 / 6.0) / parameters.variance + 0.5);
        let center = -distance * distance / (2.0 * parameters.variance);

        assert!(btpe_accepts(
            &parameters,
            candidate,
            (center - rho - 1.0).exp()
        ));
        assert!(!btpe_accepts(
            &parameters,
            candidate,
            (center + rho + 1.0).exp()
        ));

        let fallback_acceptance = center.exp();
        let mode = parameters.mode as f64;
        let candidate_f64 = candidate as f64;
        let exact_log_ratio = scioxide_special::ln_gamma(mode + 1.0).unwrap()
            + scioxide_special::ln_gamma(parameters.n_f64 - mode + 1.0).unwrap()
            - scioxide_special::ln_gamma(candidate_f64 + 1.0).unwrap()
            - scioxide_special::ln_gamma(parameters.n_f64 - candidate_f64 + 1.0).unwrap()
            + (candidate as f64 - parameters.mode as f64)
                * (parameters.probability.ln() - (-parameters.probability).ln_1p());
        assert_eq!(
            btpe_accepts(&parameters, candidate, fallback_acceptance),
            fallback_acceptance.ln() <= exact_log_ratio
        );
    }

    #[test]
    fn btpe_outside_squeeze_domain_uses_the_true_mass_ratio() {
        for (n, p) in [(5_000, 0.01), (1_000, 0.5)] {
            let parameters = Btpe::new(n, p).unwrap();
            for candidate in [
                0,
                parameters.mode.saturating_sub(21),
                parameters.mode + 21,
                n,
            ] {
                let log_ratio =
                    binomial_log_mass(n as f64, candidate as f64, p) - parameters.log_mode_mass;
                for offset in [-0.1_f64, 0.1] {
                    let height = (log_ratio + offset).exp();
                    if height > 0.0 && height <= 1.0 {
                        assert_eq!(
                            btpe_accepts(&parameters, candidate, height),
                            offset < 0.0,
                            "n={n}, p={p}, candidate={candidate}, log height={}",
                            height.ln()
                        );
                    }
                }
            }
        }
        // Reported counterexample: the old unrestricted lower squeeze
        // accepted exp(-45), although the true ratio is below exp(-47).
        let parameters = Btpe::new(5_000, 0.01).unwrap();
        assert!(!btpe_accepts(&parameters, 0, (-45.0_f64).exp()));
    }

    #[test]
    fn test_btpe_retries_before_triangle_acceptance() {
        let dist = Binomial::new(20_usize, 0.5_f64).unwrap();
        let SamplingMethod::Btpe(parameters) = dist.prepare_sampler().unwrap().sampling_method
        else {
            panic!("expected BTPE");
        };
        let triangle_uniform = 0.5 * parameters.triangle_area / parameters.total_area;
        let mut rng = ScriptedRng::new(vec![1.0_f64.next_down(), 0.0, triangle_uniform, 0.5], 0.0);

        assert_eq!(
            dist.prepare_sampler().unwrap().sample(&mut rng).unwrap(),
            parameters.mode
        );
        assert_eq!(rng.draws, 4);
    }

    #[test]
    fn test_btpe_exhaustion_is_bounded() {
        let dist = Binomial::new(20_usize, 0.5_f64).unwrap();
        let mut rng = ScriptedRng::new(Vec::new(), 1.0_f64.next_down());

        let error = dist
            .prepare_sampler()
            .unwrap()
            .sample(&mut rng)
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
        assert_eq!(rng.draws, 2 * MAX_SAMPLING_ATTEMPTS);
    }

    #[test]
    fn test_btpe_valid_uniform_extremes_preserve_the_sampling_contract() {
        let distribution = Binomial::new(1_000_usize, 0.5_f64).unwrap();
        let mut zero_rng = ScriptedRng::constant(0.0);
        let mut max_rng = ScriptedRng::constant(1.0 - f64::EPSILON);

        assert!(
            distribution
                .prepare_sampler()
                .unwrap()
                .sample(&mut zero_rng)
                .unwrap()
                <= 1_000
        );
        match distribution.prepare_sampler().unwrap().sample(&mut max_rng) {
            Ok(value) => assert!(value <= 1_000),
            Err(error) => assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: scioxide_core::ConvergenceReason::IterationLimit,
                    ..
                }
            )),
        }
    }

    #[test]
    fn test_each_nontrivial_method_is_reproducible() {
        for (n, probability) in [(1_000_usize, 0.001_f64), (1_000, 0.25)] {
            let distribution = Binomial::new(n, probability).unwrap();
            let mut first_rng = SplitMix64::from_seed(0x4249_4e4f);
            let mut second_rng = SplitMix64::from_seed(0x4249_4e4f);
            let first: Vec<usize> = (0..128)
                .map(|_| {
                    distribution
                        .prepare_sampler()
                        .unwrap()
                        .sample(&mut first_rng)
                        .unwrap()
                })
                .collect();
            let second: Vec<usize> = (0..128)
                .map(|_| {
                    distribution
                        .prepare_sampler()
                        .unwrap()
                        .sample(&mut second_rng)
                        .unwrap()
                })
                .collect();
            assert_eq!(first, second);
        }
    }

    #[test]
    fn test_sampling_symmetry_uses_the_same_canonical_stream() {
        for (n, probability, seed) in [
            (16_usize, 0.25_f64, 0x16),
            (1_000, 1.0 / 1_024.0, 0x1_024),
            (100, 0.25, 0x100),
        ] {
            let lower = Binomial::new(n, probability).unwrap();
            let upper = Binomial::new(n, 1.0 - probability).unwrap();
            let mut lower_rng = SplitMix64::from_seed(seed);
            let mut upper_rng = SplitMix64::from_seed(seed);

            for _ in 0..100 {
                let lower_sample = lower
                    .prepare_sampler()
                    .unwrap()
                    .sample(&mut lower_rng)
                    .unwrap();
                let upper_sample = upper
                    .prepare_sampler()
                    .unwrap()
                    .sample(&mut upper_rng)
                    .unwrap();
                assert_eq!(lower_sample + upper_sample, n);
            }
        }
    }

    #[test]
    fn test_out_of_support_internal_sample_is_a_structural_error() {
        let error = validate_and_reflect_sample(10, 11, false).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Breakdown,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_mean_approx() {
        let n = 20_usize;
        let p = 0.3_f64;
        let dist = Binomial::new(n, p).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let trials = 10_000;
        let sum: f64 = (0..trials)
            .map(|_| dist.prepare_sampler().unwrap().sample(&mut rng).unwrap() as f64)
            .sum();
        let sample_mean = sum / f64::from(trials);
        let expected = n as f64 * p;
        assert!(
            (sample_mean - expected).abs() < 0.5,
            "sample_mean={sample_mean}, expected={expected}"
        );
    }

    #[test]
    fn test_pmf_fair_coin() {
        // Binomial(10, 0.5), P(k=5) = C(10,5) * 0.5^10
        let dist = Binomial::new(10_usize, 0.5_f64).unwrap();
        let expected = 252.0 / 1024.0; // C(10,5) / 2^10
        assert!((dist.pmf(5).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_pmf_boundary() {
        let dist = Binomial::new(5_usize, 0.3_f64).unwrap();
        // P(0) = 0.7^5
        let expected = 0.7_f64.powi(5);
        assert!((dist.pmf(0).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_pmf_preserves_extreme_failure_probability() {
        let n = 1_000_000_000_000_usize;
        let probability = 1e-20_f64;
        let dist = Binomial::new(n, probability).unwrap();
        let expected = (n as f64 * (-probability).ln_1p()).exp();

        assert!((dist.pmf(0).unwrap() - expected).abs() < 1e-15);
        assert!(dist.pmf(0).unwrap() < 1.0);
    }

    #[test]
    fn test_pmf_beyond_n() {
        let dist = Binomial::new(5_usize, 0.5_f64).unwrap();
        assert!((dist.pmf(6).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_pmf_sum_near_one() {
        let dist = Binomial::new(10_usize, 0.4_f64).unwrap();
        let sum: f64 = (0..=10).map(|k| dist.pmf(k).unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum={sum}");
    }

    #[test]
    fn test_cdf_monotone() {
        let dist = Binomial::new(10_usize, 0.5_f64).unwrap();
        let mut prev = 0.0_f64;
        for k in 0..=10 {
            let c = dist.cdf(k).unwrap();
            assert!(c >= prev, "CDF not monotone at k={k}");
            prev = c;
        }
    }

    #[test]
    fn test_cdf_at_n() {
        let dist = Binomial::new(10_usize, 0.5_f64).unwrap();
        assert!((dist.cdf(10).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cdf_extreme_probability_uses_the_stable_beta_tail() {
        let n = 1_000_usize;
        let probability = 1.0 - 1e-12;
        let dist = Binomial::new(n, probability).unwrap();
        let cdf = dist.cdf(n - 2).unwrap();

        assert!(cdf > 0.0);
        assert!(cdf < 1e-12);
    }

    #[test]
    fn test_p_zero() {
        let dist = Binomial::new(5_usize, 0.0_f64).unwrap();
        assert!((dist.pmf(0).unwrap() - 1.0).abs() < 1e-12);
        assert!((dist.pmf(1).unwrap()).abs() < 1e-12);
        assert_eq!(dist.cdf(0).unwrap(), 1.0);
        assert_eq!(dist.cdf(4).unwrap(), 1.0);
    }

    #[test]
    fn test_p_one() {
        let dist = Binomial::new(5_usize, 1.0_f64).unwrap();
        assert!((dist.pmf(5).unwrap() - 1.0).abs() < 1e-12);
        assert!((dist.pmf(4).unwrap()).abs() < 1e-12);
        assert_eq!(dist.cdf(0).unwrap(), 0.0);
        assert_eq!(dist.cdf(4).unwrap(), 0.0);
        assert_eq!(dist.cdf(5).unwrap(), 1.0);
    }
    #[test]
    fn inversion_exhaustion_reports_attempted_work() {
        let parameters = Binv {
            initial_probability: 0.0,
            positive_probability: 1.0,
            recurrence_scale: 1.0,
            odds: 0.0,
            mode: MAX_SAMPLING_ATTEMPTS,
        };
        let mut rng = ScriptedRng::constant(0.5);
        let error = binv(usize::MAX, &parameters, &mut rng, &mut 65_536).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
        assert_eq!(rng.draws, 1);
    }
}
