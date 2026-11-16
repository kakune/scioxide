//! Poisson distribution.
//!
//! Sampling uses inversion below rate 12 and Hörmann's PTRS above it. Both
//! stop after a fixed number of steps and return an error instead of a
//! clipped sample.

use super::count_math::analytic_count;
use super::discrete_math::poisson_log_mass;
use crate::traits::{BitGenerator, Cdf, Mass};
use crate::util::MAX_SAMPLING_ATTEMPTS;
#[cfg(test)]
use alloc::vec::Vec;
use scioxide_core::{RealScalar, ScalarConversionReason, SciOxideError};

/// PTRS is designed for medium and large Poisson parameters.
const PTRS_THRESHOLD: f64 = 12.0;

#[derive(Debug, Clone)]
enum SamplingMethod {
    Constant,
    Inversion {
        lambda: f64,
        positive: f64,
        first_mass: f64,
    },
    Ptrs(Ptrs),
}

/// Immutable constants for Hörmann's PTRS algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ptrs {
    lambda: f64,
    a: f64,
    b: f64,
    log_inverse_alpha: f64,
    squeeze_threshold: f64,
}

impl Ptrs {
    fn new(lambda: f64) -> Self {
        // Hörmann's (1993) PTRS constants: the transformed-rejection hat
        // (2a / (1/2 - |u|) + b) u + lambda + 0.43 and its squeeze threshold
        // are the published fits in b = 0.931 + 2.53 sqrt(lambda); they are
        // valid for lambda >= 10, which the 12 threshold respects.
        let b = 0.931 + 2.53 * lambda.sqrt();
        let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
        Self {
            lambda,
            a: -0.059 + 0.024_83 * b,
            b,
            log_inverse_alpha: inverse_alpha.ln(),
            squeeze_threshold: 0.9277 - 3.6224 / (b - 2.0),
        }
    }
}

/// Poisson distribution with rate `λ ≥ 0`.
///
/// The mass function is `P(k) = λ^k e^(−λ) / k!` for `k = 0, 1, …`, with mean
/// and variance `λ`. `λ = 0` is a point mass at zero. The mass function uses
/// Loader's saddle-point form. Each call of `sample`, `sample_n` or
/// `sample_n_into` ([`CountDistribution`](crate::CountDistribution)) prepares
/// a [`PoissonSampler`] without allocating and draws through it; for a loop of
/// single draws, prepare it once with [`Poisson::prepare_sampler`].
///
/// ```
/// use scioxide_random::{Cdf, CountDistribution, Mass, Poisson, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival};
/// let law = Poisson::new(4.0_f64)?;
/// assert!((law.pmf(2)? - 0.146_525_111_109_873_4).abs() < 1e-14);
/// assert!((law.cdf(2)? - 0.238_103_305_553_544_3).abs() < 1e-14);
/// assert!((law.sf(2)? - 0.761_896_694_446_455_7).abs() < 1e-14);
/// assert_eq!(law.quantile(0.5)?, 4);
/// assert!(law.pmf(law.sample(&mut SplitMix64::from_seed(42))?)? > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct Poisson<S: RealScalar> {
    options: crate::LawOptions<S>,
    lambda: S,
    /// Loader's mass constants of the rate.
    mass: super::discrete_math::PoissonMass,
}

/// Poisson sampler with its setup constants computed once.
///
/// The sampler uses sequential inversion for `λ < 12` and Hörmann's PTRS
/// transformed rejection for `λ ≥ 12`. A draw through
/// [`CountDistribution`](crate::CountDistribution) reads at most 18 uniform
/// words for inversion and 2,048 for PTRS, and `λ = 0` reads none. Inversion
/// and rejection stop after 1,024 steps or candidates. The
/// [work limits](crate#work-limits) of the law also apply. An accepted count
/// that does not fit in `usize` is a `ScalarConversion` error.
///
/// ```
/// use scioxide_random::{CountDistribution, Mass, Poisson, SeedableRng, SplitMix64};
/// let law = Poisson::new(30.0_f64)?;
/// let sampler = law.prepare_sampler()?;
/// let mut rng = SplitMix64::from_seed(7);
/// for _ in 0..4 {
///     assert!(law.pmf(sampler.sample(&mut rng)?)? > 0.0);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// W. Hörmann, "The transformed rejection method for generating Poisson random
/// variables", *Insurance: Mathematics and Economics* 12(1), 1993, 39–45.
#[derive(Debug, Clone)]
#[must_use]
pub struct PoissonSampler {
    limits: super::count_math::DrawLimits,
    work: usize,
    sampling_method: SamplingMethod,
}

impl<S: RealScalar> Poisson<S> {
    /// Creates a Poisson law with rate `lambda`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lambda` is NaN or infinite.
    /// - `DomainError` if `lambda < 0`.
    pub fn new(lambda: S) -> Result<Self, SciOxideError> {
        Self::with_options(lambda, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside the range
    ///   accepted by [`LawOptions`](crate::LawOptions).
    pub fn with_options(lambda: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        crate::law::nonnegative(lambda, scioxide_core::argument!("lambda"))?;
        Ok(Self {
            options,
            lambda,
            mass: super::discrete_math::PoissonMass::new(lambda.into_f64()),
        })
    }

    /// Returns a sampler with the setup constants for this law.
    ///
    /// The setup reads no random words.
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if `λ > min(usize::MAX / 2, 2⁵¹)`.
    pub fn prepare_sampler(&self) -> Result<PoissonSampler, SciOxideError> {
        let lambda = self.lambda.into_f64();
        let maximum = ((usize::MAX / 2) as f64).min(2.0_f64.powi(51));
        if lambda > maximum {
            return Err(SciOxideError::not_implemented(
                "Poisson sampling above min(usize::MAX/2, 2^51)",
            ));
        }
        let sampling_method = if lambda == 0.0 {
            SamplingMethod::Constant
        } else if lambda < PTRS_THRESHOLD {
            // As for binomial BINV, "X = 0?" is decided first with an exact
            // Bernoulli of P(X > 0) = -expm1(-lambda), and the search runs on
            // the conditional law starting at P(X = 1 | X > 0).
            SamplingMethod::Inversion {
                lambda,
                positive: -(-lambda).exp_m1(),
                first_mass: (lambda / -(-lambda).exp_m1()) * (-lambda).exp(),
            }
        } else {
            SamplingMethod::Ptrs(Ptrs::new(lambda))
        };
        Ok(PoissonSampler {
            sampling_method,
            limits: super::count_math::DrawLimits::of(self.options),
            work: self.options.work_budget().get(),
        })
    }

    fn log_cdf_work(&self, k: usize, left: &mut usize) -> Result<S, SciOxideError> {
        if self.lambda == S::ZERO {
            return Ok(S::ZERO);
        }
        let shape = analytic_count(k)?;
        if shape >= super::count_math::MAX_EXACT_COUNT as f64 {
            return Err(SciOxideError::not_implemented(
                "Poisson tail shape exceeds exact counts",
            ));
        }
        S::try_from_f64_lossy(crate::law::special(left, |budget| {
            scioxide_special::gamma_log_q_with_options(
                shape + 1.0,
                self.lambda.into_f64(),
                scioxide_special::GammaOptions::new(budget),
            )
        })?)
    }

    fn log_sf_work(&self, k: usize, left: &mut usize) -> Result<S, SciOxideError> {
        if self.lambda == S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        let shape = analytic_count(k)?;
        if shape >= super::count_math::MAX_EXACT_COUNT as f64 {
            return Err(SciOxideError::not_implemented(
                "Poisson tail shape exceeds exact counts",
            ));
        }
        // P(X > k) = P(k + 1, lambda), the regularized lower incomplete gamma.
        S::try_from_f64_lossy(crate::law::special(left, |budget| {
            scioxide_special::gamma_log_p_with_options(
                shape + 1.0,
                self.lambda.into_f64(),
                scioxide_special::GammaOptions::new(budget),
            )
        })?)
    }
}

/// Invert conditional positive masses after an accurately resolved zero atom.
fn inversion<R: BitGenerator + ?Sized>(
    lambda: f64,
    positive: f64,
    first_mass: f64,
    rng: &mut R,
    left: &mut usize,
) -> Result<usize, SciOxideError> {
    if !super::discrete_math::bernoulli(positive, rng)? {
        return Ok(0);
    }
    let mut residual = super::words::uniform(rng)?;
    let mut mass = first_mass;
    for k in 1..=MAX_SAMPLING_ATTEMPTS {
        crate::law::charge(left)?;
        if residual < mass {
            return Ok(k);
        }
        residual -= mass;
        mass *= lambda / (k + 1) as f64;
    }
    Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
}

#[cold]
fn sample_out_of_range() -> SciOxideError {
    SciOxideError::scalar_conversion(
        scioxide_core::ConversionType::F64,
        scioxide_core::ConversionType::Usize,
        ScalarConversionReason::OutOfRange,
    )
}

/// Hörmann's transformed-rejection method for medium and large `lambda`.
#[inline(always)]
fn ptrs<R: BitGenerator + ?Sized>(
    parameters: &Ptrs,
    rng: &mut R,
    left: &mut usize,
) -> Result<usize, SciOxideError> {
    #[allow(clippy::cast_precision_loss)]
    let usize_upper_bound = usize::MAX as f64 + 1.0;

    for _ in 0..MAX_SAMPLING_ATTEMPTS {
        crate::law::charge(left)?;
        let centered_uniform = super::words::uniform(rng)? - 0.5;
        let acceptance_uniform = super::words::uniform(rng)?;
        let distance_from_edge = 0.5 - centered_uniform.abs();

        // A zero endpoint makes the PTRS transformation singular. It is a
        // valid RNG value, so reject this candidate within the common budget.
        if distance_from_edge <= 0.0 {
            continue;
        }

        let proposal = (2.0 * parameters.a / distance_from_edge + parameters.b) * centered_uniform
            + parameters.lambda
            + 0.43;

        if proposal < 0.0 {
            continue;
        }
        if !proposal.is_finite() {
            continue;
        }

        let candidate = proposal.floor();
        // Squeeze: away from the edges of u the hat lies within a known factor
        // of the pmf, so small acceptance values accept without evaluating
        // the log-mass. Very close to the edges the hat is poor and those
        // candidates are rejected cheaply when v exceeds the edge distance.
        if distance_from_edge >= 0.07 && acceptance_uniform <= parameters.squeeze_threshold {
            return accepted_count(candidate, usize_upper_bound);
        }

        if distance_from_edge < 0.013 && acceptance_uniform > distance_from_edge {
            continue;
        }

        // log(0) is the limiting value -infinity and therefore accepts every
        // otherwise admissible candidate without creating an invalid output.
        if acceptance_uniform == 0.0 {
            return accepted_count(candidate, usize_upper_bound);
        }

        let log_acceptance = acceptance_uniform.ln() + parameters.log_inverse_alpha
            - (parameters.a / (distance_from_edge * distance_from_edge) + parameters.b).ln();
        let log_mass = poisson_log_mass(candidate, parameters.lambda);

        if log_acceptance <= log_mass {
            return accepted_count(candidate, usize_upper_bound);
        }
    }

    Err(crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS))
}

fn accepted_count(candidate: f64, usize_upper_bound: f64) -> Result<usize, SciOxideError> {
    if candidate >= usize_upper_bound {
        return Err(sample_out_of_range());
    }
    // PTRS floors a continuous proposal, which has no fractional resolution
    // from 2^52 on; the count conversion itself is the shared exact range.
    if candidate >= 2.0_f64.powi(52) {
        return Err(SciOxideError::not_implemented(
            "Poisson accepted count beyond exact proposal range",
        ));
    }
    super::count_math::count(candidate)
}

impl PoissonSampler {
    fn limits(&self) -> super::count_math::DrawLimits {
        self.limits
    }
    fn draw<R: BitGenerator + ?Sized>(&self, rng: &mut R) -> Result<usize, SciOxideError> {
        self.draw_with_work(rng, &mut { self.work })
    }
    pub(super) fn draw_with_work<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        left: &mut usize,
    ) -> Result<usize, SciOxideError> {
        match &self.sampling_method {
            SamplingMethod::Constant => Ok(0),
            SamplingMethod::Inversion {
                lambda,
                positive,
                first_mass,
            } => inversion(*lambda, *positive, *first_mass, rng, left),
            SamplingMethod::Ptrs(parameters) => ptrs(parameters, rng, left),
        }
    }
}

/// A changing conditional rate needs only the inversion/PTRS constants, with
/// all attempts charged to the enclosing compound draw's remaining allowance.
pub(super) fn conditional_poisson<R: BitGenerator + ?Sized>(
    lambda: f64,
    rng: &mut R,
    left: &mut usize,
) -> Result<usize, SciOxideError> {
    crate::law::nonnegative(lambda, scioxide_core::argument!("lambda"))?;
    if lambda > ((usize::MAX / 2) as f64).min(2.0_f64.powi(51)) {
        return Err(SciOxideError::not_implemented(
            "Poisson sampling above min(usize::MAX/2, 2^51)",
        ));
    }
    if lambda == 0.0 {
        return Ok(0);
    }
    if lambda < PTRS_THRESHOLD {
        let positive = -(-lambda).exp_m1();
        inversion(
            lambda,
            positive,
            (lambda / positive) * (-lambda).exp(),
            rng,
            left,
        )
    } else {
        ptrs(&Ptrs::new(lambda), rng, left)
    }
}
super::count_math::count_distribution!([] PoissonSampler, usize);
super::count_math::prepared_count_distribution!(Poisson, usize);

impl<S: RealScalar> Mass<S> for Poisson<S> {
    fn pmf(&self, k: usize) -> Result<S, SciOxideError> {
        let lambda = self.lambda.into_f64();
        if lambda == 0.0 {
            return Ok(if k == 0 { S::ONE } else { S::ZERO });
        }
        S::try_from_f64_lossy(self.mass.log_mass(analytic_count(k)?).exp())
    }
}

impl<S: RealScalar> Cdf<S, usize> for Poisson<S> {
    fn cdf(&self, k: usize) -> Result<S, SciOxideError> {
        let left = &mut self.options.work_budget().get();
        let lambda = self.lambda.into_f64();
        if lambda == 0.0 {
            return Ok(S::ONE);
        }
        let k_f64 = analytic_count(k)?;
        if k < super::count_math::DIRECT_TERMS {
            // Sum from the largest term toward the smaller ones so the
            // accumulated rounding stays relative to the dominant mass: below
            // the mean start at k and walk down, above it start at 0 and walk up.
            let cdf = if k_f64 <= lambda {
                let initial = self.mass.log_mass(k_f64).exp();
                (1..=k)
                    .rev()
                    .try_fold((initial, initial), |(sum, mass), index| {
                        crate::law::charge(left)?;
                        let next = (mass / lambda) * index as f64;
                        Ok::<_, SciOxideError>((sum + next, next))
                    })?
                    .0
            } else {
                // Here lambda < k < 32, so the zero atom cannot underflow.
                let initial = (-lambda).exp();
                (0..k)
                    .try_fold((initial, initial), |(sum, mass), index| {
                        crate::law::charge(left)?;
                        let next = mass * (lambda / (index + 1) as f64);
                        Ok::<_, SciOxideError>((sum + next, next))
                    })?
                    .0
            };
            if (0.0..=1.0).contains(&cdf) {
                return S::try_from_f64_lossy(cdf);
            }
        }
        if k_f64 >= super::count_math::MAX_EXACT_COUNT as f64 {
            return Err(SciOxideError::not_implemented(
                "Poisson CDF shape beyond exact count precision",
            ));
        }
        S::try_from_f64_lossy(crate::law::special(left, |budget| {
            scioxide_special::gamma_q_with_options(
                k_f64 + 1.0,
                lambda,
                scioxide_special::GammaOptions::new(budget),
            )
        })?)
    }
}

impl<S: RealScalar> crate::LogSurvival<S, usize> for Poisson<S> {
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

impl<S: RealScalar> crate::LogCdf<S, usize> for Poisson<S> {
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

impl<S: RealScalar> crate::Survival<S, usize> for Poisson<S> {
    fn sf(&self, k: usize) -> Result<S, SciOxideError> {
        S::try_from_f64_lossy(
            crate::LogSurvival::log_sf(
                &Poisson::with_options(self.lambda.into_f64(), self.options.widened()?)?,
                k,
            )?
            .exp(),
        )
    }
}

impl<S: RealScalar> crate::LogMass<S> for Poisson<S> {
    fn log_pmf(&self, k: usize) -> Result<S, SciOxideError> {
        if self.lambda == S::ZERO {
            return Ok(self.pmf(k)?.ln());
        }
        S::try_from_f64_lossy(self.mass.log_mass(analytic_count(k)?))
    }
}
impl<S: RealScalar> crate::CountSupport for Poisson<S> {
    fn count_support(&self) -> (usize, Option<usize>) {
        (
            0,
            if self.lambda == S::ZERO {
                Some(0)
            } else {
                None
            },
        )
    }
}
impl<S: RealScalar> crate::Mean<S> for Poisson<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(self.lambda)
    }
}
impl<S: RealScalar> crate::Variance<S> for Poisson<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        Ok(self.lambda)
    }
}

impl<S: RealScalar> Poisson<S> {
    fn inverse_probability(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<usize, SciOxideError> {
        let (lower, maximum) = crate::CountSupport::count_support(self);
        super::count_math::inverse(p, upper, lower, maximum, None, left, |k, upper, left| {
            Ok(if upper {
                self.log_sf_work(k, left)?
            } else {
                self.log_cdf_work(k, left)?
            }
            .into_f64())
        })
    }
}
impl<S: RealScalar> crate::Quantile<S, usize> for Poisson<S> {
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
impl<S: RealScalar> crate::InverseSurvival<S, usize> for Poisson<S> {
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
        assert!(Poisson::<f64>::new(5.0).is_ok());
    }

    #[test]
    fn test_f32_and_f64_constructors_choose_equivalent_precomputation() {
        for lambda_f32 in [1.0_f32, 10.0, 12.0, 100.0] {
            let f32_distribution = Poisson::new(lambda_f32).unwrap();
            let f64_distribution = Poisson::new(f64::from(lambda_f32)).unwrap();
            match (
                f32_distribution.prepare_sampler().unwrap().sampling_method,
                f64_distribution.prepare_sampler().unwrap().sampling_method,
            ) {
                (SamplingMethod::Inversion { .. }, SamplingMethod::Inversion { .. }) => {}
                (SamplingMethod::Ptrs(f32_parameters), SamplingMethod::Ptrs(f64_parameters)) => {
                    assert_eq!(f32_parameters, f64_parameters);
                }
                (f32_method, f64_method) => {
                    panic!("constructor mismatch: f32={f32_method:?}, f64={f64_method:?}");
                }
            }
        }
    }

    #[test]
    fn test_sampling_method_switches_at_twelve() {
        let listed_ten = Poisson::new(10.0_f64).unwrap();
        let below = Poisson::new(11.999_f64).unwrap();
        let at = Poisson::new(12.0_f64).unwrap();

        assert!(matches!(
            listed_ten.prepare_sampler().unwrap().sampling_method,
            SamplingMethod::Inversion { .. }
        ));
        assert!(matches!(
            below.prepare_sampler().unwrap().sampling_method,
            SamplingMethod::Inversion { .. }
        ));
        assert!(matches!(
            at.prepare_sampler().unwrap().sampling_method,
            SamplingMethod::Ptrs(_)
        ));
    }

    #[test]
    fn test_new_invalid() {
        let error = Poisson::<f64>::new(-1.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite() {
        let error = Poisson::<f64>::new(f64::NAN).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_small_lambda() {
        let dist = Poisson::new(3.0_f64).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let n = 10_000;
        let sum: f64 = (0..n)
            .map(|_| dist.prepare_sampler().unwrap().sample(&mut rng).unwrap() as f64)
            .sum();
        let sample_mean = sum / f64::from(n);
        assert!((sample_mean - 3.0).abs() < 0.2, "sample_mean={sample_mean}");
    }

    #[test]
    fn test_small_lambda_zero_is_a_sample_not_an_error() {
        let dist = Poisson::new(3.0_f64).unwrap();
        let mut rng = ScriptedRng::constant(1.0_f64.next_down());

        assert_eq!(dist.prepare_sampler().unwrap().sample(&mut rng).unwrap(), 0);
        assert_eq!(rng.draws, 1);
    }

    #[test]
    fn test_ptrs_fast_and_full_acceptance() {
        let dist = Poisson::new(12.0_f64).unwrap();
        let mut fast_rng = ScriptedRng::new(vec![0.5, 0.1], 0.0);
        let mut full_rng = ScriptedRng::new(vec![0.5, 0.8], 0.0);

        assert_eq!(
            dist.prepare_sampler()
                .unwrap()
                .sample(&mut fast_rng)
                .unwrap(),
            12
        );
        assert_eq!(fast_rng.draws, 2);
        assert_eq!(
            dist.prepare_sampler()
                .unwrap()
                .sample(&mut full_rng)
                .unwrap(),
            12
        );
        assert_eq!(full_rng.draws, 2);
    }

    #[test]
    fn test_ptrs_retries_before_accepting() {
        let dist = Poisson::new(12.0_f64).unwrap();
        let mut rng = ScriptedRng::new(vec![0.5, 0.99, 0.5, 0.1], 0.0);

        assert_eq!(
            dist.prepare_sampler().unwrap().sample(&mut rng).unwrap(),
            12
        );
        assert_eq!(rng.draws, 4);
    }

    #[test]
    fn test_ptrs_exhaustion_is_bounded() {
        let dist = Poisson::new(12.0_f64).unwrap();
        let mut rng = ScriptedRng::constant(0.0);

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
    fn test_ptrs_maximum_uniform_is_bounded() {
        let distribution = Poisson::new(12.0_f64).unwrap();
        let mut rng = ScriptedRng::constant(1.0 - f64::EPSILON);

        let error = distribution
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
    fn test_ptrs_is_fixed_seed_reproducible() {
        let dist = Poisson::new(1000.0_f64).unwrap();
        let mut first_rng = SplitMix64::from_seed(89);
        let mut second_rng = SplitMix64::from_seed(89);

        let first: Vec<usize> = (0..100)
            .map(|_| {
                dist.prepare_sampler()
                    .unwrap()
                    .sample(&mut first_rng)
                    .unwrap()
            })
            .collect();
        let second: Vec<usize> = (0..100)
            .map(|_| {
                dist.prepare_sampler()
                    .unwrap()
                    .sample(&mut second_rng)
                    .unwrap()
            })
            .collect();

        assert_eq!(first, second);
    }

    #[test]
    fn test_ptrs_log_mass_uses_exact_table_and_stirling_regions() {
        let parameters = Ptrs::new(100.0);
        for candidate in [0.0_f64, 1.0, 15.0, 16.0, 100.0, 1_000_000.0] {
            let actual = poisson_log_mass(candidate, parameters.lambda);
            let expected = -parameters.lambda + candidate * parameters.lambda.ln()
                - scioxide_special::ln_gamma(candidate + 1.0).unwrap();
            let tolerance = 32.0
                * f64::EPSILON
                * (parameters.lambda
                    + (candidate * parameters.lambda.ln()).abs()
                    + scioxide_special::ln_gamma(candidate + 1.0).unwrap().abs())
                .max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "candidate={candidate}, actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
        assert_eq!(
            parameters.log_inverse_alpha,
            (1.1239 + 1.1328 / (parameters.b - 3.4)).ln()
        );
    }

    #[test]
    fn test_sample_large_lambda() {
        let dist = Poisson::new(50.0_f64).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let n = 10_000;
        let sum: f64 = (0..n)
            .map(|_| dist.prepare_sampler().unwrap().sample(&mut rng).unwrap() as f64)
            .sum();
        let sample_mean = sum / f64::from(n);
        assert!(
            (sample_mean - 50.0).abs() < 2.0,
            "sample_mean={sample_mean}"
        );
    }

    #[test]
    fn test_pmf_mode() {
        // For λ=5, mode is at k=4 or k=5
        let dist = Poisson::new(5.0_f64).unwrap();
        let p4 = dist.pmf(4).unwrap();
        let p5 = dist.pmf(5).unwrap();
        // Both should be near the mode; pmf(4) ≈ pmf(5) for λ=5
        assert!((p4 - p5).abs() < 0.01);
    }

    #[test]
    fn test_pmf_zero() {
        let dist = Poisson::new(3.0_f64).unwrap();
        // P(0) = exp(-3) ≈ 0.04979
        let expected = (-3.0_f64).exp();
        assert!((dist.pmf(0).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_cdf_monotone() {
        let dist = Poisson::new(5.0_f64).unwrap();
        let mut prev = 0.0_f64;
        for k in 0..20 {
            let c = dist.cdf(k).unwrap();
            assert!(c >= prev, "CDF not monotone at k={k}");
            prev = c;
        }
    }

    #[test]
    fn test_cdf_large_k() {
        let dist = Poisson::new(2.0_f64).unwrap();
        // CDF at large k should be near 1
        let c = dist.cdf(20).unwrap();
        assert!((c - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_cdf_reports_count_precision_limit() {
        let dist = Poisson::new(2.0_f64).unwrap();

        if usize::BITS > 53 {
            let error = dist.cdf(usize::MAX).unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::NotImplemented
            );
        } else {
            // On 32-bit targets, k+1 is still exact in the f64 analytic core.
            assert_eq!(dist.cdf(usize::MAX).unwrap(), 1.0);
        }
    }

    #[test]
    fn test_pmf_sum_near_one() {
        let dist = Poisson::new(5.0_f64).unwrap();
        let sum: f64 = (0..30).map(|k| dist.pmf(k).unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-8, "sum={sum}");
    }
    #[test]
    fn inversion_exhaustion_reports_attempted_work() {
        let mut rng = ScriptedRng::constant(0.5);
        let error = inversion(1.0, 1.0, 0.0, &mut rng, &mut 65_536).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ));
        assert_eq!(rng.draws, 1);
    }

    #[test]
    fn accepted_counts_are_not_saturated_or_redrawn() {
        let platform_end = 2.0_f64.powi(usize::BITS as i32);
        let error = accepted_count(platform_end, platform_end).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ScalarConversion
        );
        if usize::BITS > 52 {
            let error = accepted_count(2.0_f64.powi(52), platform_end).unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::NotImplemented
            );
        }
        assert_eq!(accepted_count(0.0, platform_end).unwrap(), 0);
    }
}
