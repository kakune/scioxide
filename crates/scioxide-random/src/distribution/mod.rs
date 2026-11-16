//! Probability laws, organized by family.
//!
//! Scalar laws implement one trait per operation (density, tails, inverses,
//! moments), and [`Distribution`](crate::Distribution) draws scalar samples.
//! Count laws draw exact integers through
//! [`CountDistribution`](crate::CountDistribution), directly or through the
//! prepared samplers [`BinomialSampler`], [`PoissonSampler`] and
//! [`SkellamSampler`]. Vector-valued laws provide `sample` and, where batches
//! are supported, `sample_n_into`.
//!
//! Each file holds one family: for example `normal` (and `normal/related`),
//! `gamma`, `beta`, `weibull`, `extreme_value` and `pareto` for continuous
//! laws; `binomial`, `poisson`, `count`, `discrete_elementary` and `finite_urn`
//! for counts; `circular` for angles and directions; `test_statistic` for
//! sampling distributions of tests; and `multivariate` for vector, matrix and
//! group laws. The crate documentation lists every law by family.

use scioxide_core::{Budget, ExecutionPolicy, RealScalar, SciOxideError, Tolerance};

/// Tolerance and resource limits for laws that compute numerically.
///
/// Laws whose operations need iteration, quadrature or a numerical search take
/// `LawOptions` as the last argument of their `with_options` constructor; their
/// `new` constructor uses the defaults. Closed-form laws take no options.
/// The setters do not check their arguments. The law constructor checks the
/// complete options before any numerical work or allocation, and every query,
/// moment and draw of the law then uses them.
///
/// - **Tolerance.** The relative component must lie in `[64ε, 0.1)`, where `ε`
///   is the machine epsilon of `S`. The default is `√ε` in both components.
///   Sums of positive terms and quadrature stop at the relative tolerance.
///   Probabilities may also stop at the absolute tolerance. The result is an
///   estimate, not a guaranteed bound. The test-statistic laws
///   ([`StudentizedRange`](crate::StudentizedRange), [`Rice`](crate::Rice),
///   the Kolmogorov–Smirnov laws, [`FisherF`](crate::FisherF) and the
///   noncentral χ², F and t laws) cap the relative target of their internal
///   sums and quadrature at `√ε` of `f64` (about 1.5·10⁻⁸), so a looser
///   tolerance does not shorten them.
/// - **Work limit** ([`Self::work_budget`]). One unit of work is one recurrence
///   update, integrand evaluation, special-function iteration or preparation
///   step. All steps of one operation count toward the same limit, including
///   bracketing, root finding, nested integrals and queries of component laws.
///   Special functions with a fixed iteration bound keep that bound.
/// - **Word limit** ([`Self::sample_word_budget`]). It caps the uniform words
///   one draw of a law that combines several sampling steps may request.
/// - **Byte limit** ([`Self::max_bytes`]). It caps each buffer the law
///   allocates.
/// - **Execution policy** ([`Self::execution_policy`]). It schedules
///   `sample_indexed_into`. Each row uses its own stream, so the samples do not
///   depend on the policy.
///
/// Every rejection loop or stepwise search within one draw also stops after
/// 1,024 candidates, for every law. Exhausting the work limit, the word limit or
/// the candidate limit is a `ConvergenceFailure`. Words consumed before the error
/// are handled as described in [batch sampling](crate#batch-sampling). See also
/// [work limits](crate#work-limits).
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_random::{LawOptions, LevyStable, Quantile};
///
/// let options = LawOptions::default().with_work_budget(Budget::try_from(100_000)?);
/// let law = LevyStable::with_options(1.5_f64, 0.0, options)?;
/// // A symmetric stable law (β = 0) has median 0.
/// assert_eq!(law.quantile(0.5)?, 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct LawOptions<S: RealScalar> {
    tolerance: Tolerance<S>,
    work: Budget,
    sample_words: Budget,
    max_bytes: usize,
    execution_policy: ExecutionPolicy,
}

impl<S: RealScalar> Default for LawOptions<S> {
    fn default() -> Self {
        Self {
            tolerance: Tolerance::from_sqrt_epsilon(),
            work: Budget::ONE.saturating_add(65_535),
            sample_words: Budget::ONE.saturating_add(65_535),
            max_bytes: 64 * 1024 * 1024,
            execution_policy: ExecutionPolicy::Sequential,
        }
    }
}

impl<S: RealScalar> LawOptions<S> {
    /// Returns the tolerance (default `√ε` in both components).
    pub const fn tolerance(self) -> Tolerance<S> {
        self.tolerance
    }
    /// Sets the tolerance. The law constructor checks that the relative
    /// component lies in `[64ε, 0.1)`.
    pub const fn with_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Returns the work limit of one operation (default 65,536).
    ///
    /// An operation is one preparation, query or moment computation. A numerical
    /// search inside one draw, such as the tail inversion of a count law, has the
    /// same limit.
    pub const fn work_budget(self) -> Budget {
        self.work
    }
    /// Sets the work limit of one operation.
    pub const fn with_work_budget(mut self, work: Budget) -> Self {
        self.work = work;
        self
    }
    /// Returns the limit on uniform words requested by one draw (default 65,536).
    ///
    /// The limit applies to count, circular, multivariate, matrix and
    /// test-statistic laws, and to [`Rice`], [`IrwinHall`] and
    /// [`Mixture`](crate::Mixture). Requests of nested draws count, and so do
    /// requests that fail. Angular laws count uniform floats instead of words.
    /// One word may take several steps of the underlying generator.
    /// [`BinomialSampler`], [`PoissonSampler`] and [`SkellamSampler`] use the
    /// limit of the law they are prepared from.
    pub const fn sample_word_budget(self) -> Budget {
        self.sample_words
    }
    /// Sets the limit on uniform words requested by one draw.
    pub const fn with_sample_word_budget(mut self, sample_words: Budget) -> Self {
        self.sample_words = sample_words;
        self
    }
    /// Returns the byte limit of each buffer the law allocates (default 64 MiB).
    ///
    /// The limit applies separately to the stored parameters, to each returned
    /// `Vec`, to internal buffers and to the capacity of a sampling workspace.
    /// Output slices you provide do not count. Exceeding the limit is a
    /// `ResourceLimit` error. Matrix factorizations computed by a constructor
    /// follow their own storage limits.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
    /// Sets the byte limit of each buffer. A limit of zero allows only empty
    /// buffers.
    pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }
    /// Returns the execution policy of `sample_indexed_into` (default
    /// sequential).
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution_policy
    }
    /// Sets the execution policy of `sample_indexed_into`.
    pub const fn with_execution_policy(mut self, execution_policy: ExecutionPolicy) -> Self {
        self.execution_policy = execution_policy;
        self
    }
    /// Checks storage for `elements` values of `T` against the byte limit.
    pub(crate) fn admit<T>(self, elements: usize) -> Result<(), SciOxideError> {
        crate::workspace::admit_bytes(
            crate::workspace::storage_bytes::<T>(elements)?,
            self.max_bytes,
        )
    }
    pub(crate) fn validate(self) -> Result<(), SciOxideError> {
        let relative = self.tolerance.relative();
        let floor = crate::law::constant::<S>(64.0)? * S::EPSILON;
        if relative < floor || relative >= crate::law::constant::<S>(0.1)? {
            return Err(crate::law::invalid(
                scioxide_core::argument!("options.tolerance"),
                "relative tolerance must lie in [64*S::EPSILON, 0.1)",
            ));
        }
        Ok(())
    }
    pub(crate) fn widened(self) -> Result<LawOptions<f64>, SciOxideError> {
        Ok(LawOptions::default()
            .with_tolerance(Tolerance::new(
                self.tolerance.absolute().into_f64(),
                self.tolerance.relative().into_f64(),
            )?)
            .with_work_budget(self.work)
            .with_sample_word_budget(self.sample_words)
            .with_max_bytes(self.max_bytes)
            .with_execution_policy(self.execution_policy))
    }
    fn relative(self) -> f64 {
        self.tolerance.relative().into_f64()
    }
}

mod beta;
mod binomial;
mod breit_wigner;
mod cauchy;
mod chi_squared;
mod circular;
pub use circular::{UniformDirection, VonMises, VonMisesFisher, VonMisesLine, WrappedCauchy};
mod formula;
mod moment_math;
mod quadrature;
mod rows;
mod special;
mod symmetric;
#[cfg(test)]
mod test_rng;
pub(crate) mod words;
pub(crate) use quadrature::log_integral;
mod central_urn;
mod count_math;
mod count_power;
mod discrete_math;
mod exponential;
mod gamma;
mod gausshyper;
mod irwin_hall;
mod log_normal;
mod normal;
mod poisson;
mod student_t;
mod uniform;

pub use beta::{Beta, BetaPrime, Burr12, Mielke};
pub use binomial::{Binomial, BinomialSampler};
pub use breit_wigner::RelativisticBreitWigner;
pub use cauchy::Cauchy;
pub use chi_squared::ChiSquared;
pub use exponential::{Exponential, ExponentialPower, GeneralizedExponential};
pub use gamma::{DoubleGamma, Gamma, GeneralizedGamma, Pearson3};
pub use gausshyper::GaussHypergeometric;
pub use irwin_hall::IrwinHall;
pub use log_normal::LogNormal;
pub use multivariate::Multinomial;
pub use multivariate::{MultivariateNormal, MultivariateNormalOptions};
pub(crate) use normal::standard_normal_quantile;
pub use normal::{
    Alpha, DoubleParetoLogNormal, ExponentiallyModifiedNormal, FatigueLife, GeneralizedNormal,
    Normal, PowerNormal,
};
pub use poisson::{Poisson, PoissonSampler};
pub use student_t::StudentT;
pub use uniform::Uniform;
mod logistic;
pub use logistic::{GeneralizedHalfLogistic, GeneralizedLogistic, HyperbolicSecant, Logistic};
mod asymmetric;
mod hyperbolic;
pub use asymmetric::{CrystalBall, JonesFaddySkewT, SkewCauchy, SkewNormal};
pub use hyperbolic::{GeneralizedHyperbolic, GeneralizedInverseGaussian, NormalInverseGaussian};

mod count;
pub use count::{
    BetaBinomial, BetaNegativeBinomial, Geometric, LogSeries, NegativeBinomial, YuleSimon, Zipf,
    Zipfian,
};

mod poisson_binomial;
pub use poisson_binomial::PoissonBinomial;

mod skellam;
mod stable;
pub use skellam::{Skellam, SkellamSampler};
pub use stable::LevyStable;

mod discrete_elementary;
mod finite_table;
mod finite_urn;
pub use discrete_elementary::{Boltzmann, DiscreteLaplace, DiscreteUniform};
pub use finite_urn::{
    FisherNoncentralHypergeometric, Hypergeometric, WalleniusNoncentralHypergeometric,
};

mod test_statistic;
pub use test_statistic::{
    FisherF, Kolmogorov, KolmogorovRequirements, KolmogorovSmirnovOneSided,
    KolmogorovSmirnovTwoSided, KolmogorovWorkspace, NoncentralChiSquared, NoncentralF,
    NoncentralStudentT, Rice, StudentizedRange, TestStatisticResult,
};

mod multivariate;
pub use multivariate::{Dirichlet, DirichletMultinomial, MultivariateHypergeometric, RandomTable};
pub use multivariate::{InverseWishart, MatrixNormal, MatrixT, Wishart};
pub use multivariate::{
    MultivariateProbabilityOptions, MultivariateProbabilityPlan, MultivariateProbabilityReport,
    MultivariateProbabilityTermination, MultivariateProbabilityWorkspace,
};
pub use multivariate::{MultivariateStudentT, NormalInverseGamma};
pub use multivariate::{OrthogonalGroup, RandomCorrelation, SpecialOrthogonalGroup, UnitaryGroup};

mod weibull;
pub use weibull::{DoubleWeibull, ExponentiatedWeibull, InverseWeibull, WeibullMin};

mod extreme_value;
pub use extreme_value::{GeneralizedExtremeValue, Gompertz, GumbelRight, Kappa4};

mod johnson;
pub use johnson::{JohnsonSB, JohnsonSU};

mod inverse_gaussian;
pub use inverse_gaussian::{InverseGaussian, ReciprocalInverseGaussian};

mod pareto;
pub use pareto::{GeneralizedPareto, Pareto, TruncatedPareto};

mod bounded_elementary;
pub use bounded_elementary::{Anglit, Argus, Bradford, Cosine, Trapezoid};

mod laplace;
pub use laplace::AsymmetricLaplace;

mod tukey_lambda;
pub use tukey_lambda::TukeyLambda;

mod moyal;
pub use moyal::Moyal;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod symmetric_laws_tests {
    use super::*;
    use crate::{Cdf, Density, LogSurvival, Mean, Variance};

    #[test]
    fn symmetric_law_identities() {
        let cosine = Cosine::<f64>::new().unwrap();
        assert!((cosine.pdf(0.).unwrap() - 1. / core::f64::consts::PI).abs() < 1e-14);
        assert!((cosine.cdf(0.).unwrap() - 0.5).abs() < 1e-14);
        let dg = DoubleGamma::<f64>::new(2.).unwrap();
        assert!((dg.variance().unwrap() - 6.).abs() < 1e-14);
        let dw = DoubleWeibull::<f64>::new(1.).unwrap();
        assert!((dw.pdf(0.).unwrap() - 0.5).abs() < 1e-14);
        assert!((dw.log_sf(40.).unwrap() + 40. + core::f64::consts::LN_2).abs() < 1e-12);
        let gl = GeneralizedLogistic::<f64>::new(1.).unwrap();
        assert!((gl.cdf(0.).unwrap() - 0.5).abs() < 1e-14);
        let gn = GeneralizedNormal::<f64>::new(2.).unwrap();
        assert!((gn.variance().unwrap() - 0.5).abs() < 1e-12);
        let gu = GumbelRight::<f64>::new();
        assert!((gu.cdf(0.).unwrap() - (-1_f64).exp()).abs() < 1e-14);
        assert!((gu.log_sf(100.).unwrap() + 100.).abs() < 1e-12);
        let hs = HyperbolicSecant::<f64>::new();
        assert!((hs.pdf(0.).unwrap() - 0.5).abs() < 1e-14);
        let al = AsymmetricLaplace::<f64>::new(2.).unwrap();
        assert!((al.cdf(0.).unwrap() - 0.8).abs() < 1e-14);
        assert!((al.mean().unwrap() + 1.5).abs() < 1e-14);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod bounded_laws_tests {
    use super::*;
    use crate::{Cdf, Density, Mean};
    #[test]
    fn bounded_family_identities() {
        let a = Anglit::<f64>::new();
        assert!((a.cdf(0.).unwrap() - 0.5).abs() < 1e-14);
        let br = Bradford::<f64>::new(2.).unwrap();
        assert!((br.cdf(1.).unwrap() - 1.).abs() < 1e-14);
        let j = JohnsonSB::<f64>::new(0., 1.).unwrap();
        assert!((j.cdf(0.5).unwrap() - 0.5).abs() < 1e-14);
        let t = Trapezoid::<f64>::new(0.25, 0.75).unwrap();
        assert!((t.cdf(0.5).unwrap() - 0.5).abs() < 1e-14);
        assert!((t.mean().unwrap() - 0.5).abs() < 1e-14);
        let ar = Argus::<f64>::new(2.).unwrap();
        assert!((ar.cdf(0.).unwrap()).abs() < 1e-14);
        assert!(ar.pdf(0.5).unwrap() > 0.);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod bounded_advanced_tests {
    use super::*;
    use crate::{Cdf, LogSurvival, Quantile};
    #[test]
    fn kappa_and_tukey_reductions() {
        let k = Kappa4::<f64>::new(0., 0.).unwrap();
        assert!((k.cdf(0.).unwrap() - (-1_f64).exp()).abs() < 1e-14);
        let t = TukeyLambda::<f64>::new(0.).unwrap();
        assert!((t.cdf(1.).unwrap() - 1. / (1. + (-1_f64).exp())).abs() < 1e-12);
        let u = TukeyLambda::<f64>::new(1.).unwrap();
        assert!((u.quantile(0.25).unwrap() + 0.5).abs() < 1e-14);
        assert!((u.log_sf(0.).unwrap() + core::f64::consts::LN_2).abs() < 1e-12);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod positive_laws_tests {
    use super::*;
    use crate::{Cdf, Density, Mean, Survival};

    #[test]
    fn positive_family_identities() {
        let b = BetaPrime::<f64>::new(2., 3.).unwrap();
        assert!((b.cdf(1.).unwrap() - 0.6875).abs() < 1e-13);
        let m = Mielke::<f64>::new(1., 2.).unwrap();
        assert!((m.cdf(1.).unwrap() - 1. / 2_f64.sqrt()).abs() < 1e-14);
        let al = Alpha::<f64>::new(2.).unwrap();
        assert!(al.sf(1.).unwrap() > 0.);
        let w = WeibullMin::<f64>::new(2.).unwrap();
        assert!((w.sf(2.).unwrap() - (-4_f64).exp()).abs() < 1e-14);
        let g = GeneralizedGamma::<f64>::new(2., 1.).unwrap();
        assert!(
            (g.cdf(2.).unwrap() - super::super::Gamma::new(2., 1.).unwrap().cdf(2.).unwrap()).abs()
                < 1e-13
        );
        let e = ExponentialPower::<f64>::new(1.).unwrap();
        assert!((e.cdf(1.).unwrap() - (1. - (1. - core::f64::consts::E).exp())).abs() < 1e-13);
        let f = FatigueLife::<f64>::new(0.5).unwrap();
        assert!((f.cdf(1.).unwrap() - 0.5).abs() < 1e-14);
        let o = Gompertz::<f64>::new(2.).unwrap();
        assert!((o.sf(1.).unwrap() - (-2_f64 * (core::f64::consts::E - 1.)).exp()).abs() < 1e-14);
        let i = InverseGaussian::<f64>::new(1.).unwrap();
        assert!((i.mean().unwrap() - 1.).abs() < 1e-14);
        let r = ReciprocalInverseGaussian::<f64>::new(1.).unwrap();
        assert!((r.mean().unwrap() - 2.).abs() < 1e-14);
        assert!((r.pdf(1.).unwrap() - i.pdf(1.).unwrap()).abs() < 1e-14);
        let p = Pearson3::<f64>::new(0.).unwrap();
        assert!((p.cdf(0.).unwrap() - 0.5).abs() < 1e-14);
    }
}
