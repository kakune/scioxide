//! Goodness-of-fit, normality, two-sample and k-sample distribution tests, and tests of
//! equal scale.
use crate::error_helpers::{finite, invalid};
mod normality;

pub use normality::{
    jarque_bera_test, jarque_bera_test_with_options, kurtosis_test, kurtosis_test_with_options,
    normality_test, normality_test_with_options, skew_test, skew_test_with_options,
};
mod one_sample;
pub use one_sample::{
    anderson_darling_statistic, anderson_darling_statistic_with_options,
    cramer_von_mises_statistic, cramer_von_mises_statistic_with_options, cramer_von_mises_test,
    cramer_von_mises_test_with_options, ks_one_sample_statistic,
    ks_one_sample_statistic_with_options, ks_one_sample_test, ks_one_sample_test_with_options,
};
mod variance;
pub use variance::{
    VarianceCenter, bartlett_test, bartlett_test_with_options, fligner_test,
    fligner_test_with_options, levene_test, levene_test_with_options,
};
mod two_sample;
pub use two_sample::{
    KsPermutationResult, cramer_von_mises_two_sample_test,
    cramer_von_mises_two_sample_test_with_options, ks_two_sample_permutation_test,
    ks_two_sample_permutation_test_into, ks_two_sample_test, ks_two_sample_test_with_options,
};
mod scale_rank;
pub use scale_rank::{
    ansari_bradley_test, ansari_bradley_test_with_options, mood_scale_test,
    mood_scale_test_with_options,
};
mod median;
pub use median::{MedianTestOutcome, MedianTies, median_test, median_test_with_options};
mod epps;
pub use epps::{epps_singleton_two_sample_test, epps_singleton_two_sample_test_with_options};
mod shapiro;
pub use shapiro::{
    shapiro_wilk_statistic, shapiro_wilk_statistic_with_options, shapiro_wilk_test,
    shapiro_wilk_test_with_options,
};
mod anderson_k;
pub use anderson_k::{
    AndersonKVariant, anderson_k_sample_test, anderson_k_sample_test_with_options,
};
mod batch;
pub use batch::{DistributionBatchTest, DistributionBatchWorkspace, distribution_tests_into};
mod goodness;
pub use goodness::{
    GoodnessOfFitModel, GoodnessOfFitResult, GoodnessOfFitStatistic, goodness_of_fit_test_into,
};

use crate::{ClassicalOptions, TestAlternative, TestCalibration};
use scioxide_core::{NonFiniteOrigin, RealScalar, SciOxideError};
use scioxide_random::{ChiSquared, FisherF, Normal, Survival};

/// Null distribution a distribution test actually used for its p-value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionMethod {
    /// Exact finite-sample null distribution, or enumeration of every label
    /// assignment.
    Exact,
    /// Normal, chi-square, F or another large-sample approximation documented
    /// on the test.
    Asymptotic,
    /// Permutation distribution of the group labels.
    Permutation {
        /// Whether every distinct label assignment was evaluated.
        exact: bool,
        /// Number of assignments evaluated.
        replicates: usize,
    },
    /// Parametric bootstrap: samples drawn from the fitted null model, refitting
    /// the parameters on each sample when they were estimated.
    MonteCarlo {
        /// Number of simulated samples.
        replicates: usize,
    },
}

/// Records whether the null distribution had fixed or estimated parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullParameters {
    /// Every parameter was specified in advance.
    Known,
    /// At least one parameter was estimated from the sample.
    Estimated,
    /// The null hypothesis is that the groups share an unspecified common
    /// distribution or scale.
    Exchangeable,
}

/// Result of a distribution test: statistic, p-value and the null
/// distribution used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistributionTestResult<S: RealScalar> {
    pub(crate) statistic: S,
    pub(crate) p_value: f64,
    pub(crate) method: DistributionMethod,
    pub(crate) null_parameters: NullParameters,
    pub(crate) observations: usize,
    pub(crate) groups: usize,
    pub(crate) monte_carlo_interval: Option<scioxide_core::Interval<f64>>,
}
impl<S: RealScalar> DistributionTestResult<S> {
    /// Returns the test statistic, as documented on the test.
    pub const fn statistic(&self) -> S {
        self.statistic
    }
    /// Returns the p-value: the upper tail probability at the observed statistic,
    /// or the tail selected by the alternative for directional tests.
    pub const fn p_value(&self) -> f64 {
        self.p_value
    }
    /// Returns the null distribution used for the p-value.
    pub const fn method(&self) -> DistributionMethod {
        self.method
    }
    /// Returns whether the null distribution had fixed or estimated parameters.
    pub const fn null_parameters(&self) -> NullParameters {
        self.null_parameters
    }
    /// Returns the total number of observations used, over all groups.
    pub const fn observations(&self) -> usize {
        self.observations
    }
    /// Returns the number of independent samples.
    pub const fn groups(&self) -> usize {
        self.groups
    }
    /// Returns the 95% Wilson score interval for the simulated upper tail
    /// probability. It is present only for Monte Carlo and randomized
    /// permutation p-values.
    pub const fn monte_carlo_interval(&self) -> Option<scioxide_core::Interval<f64>> {
        self.monte_carlo_interval
    }
}

fn sample<S: RealScalar>(
    x: &[S],
    minimum: usize,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    crate::error_helpers::count::<S>(x.len())?;
    if x.len() < minimum {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient observations",
        ));
    }
    x.iter().try_for_each(|v| {
        if v.is_finite() {
            Ok(())
        } else {
            Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "finite observations required",
            ))
        }
    })
}

fn result<S: RealScalar>(
    statistic: f64,
    p_value: f64,
    method: DistributionMethod,
    null_parameters: NullParameters,
    observations: usize,
    groups: usize,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    let statistic = S::try_from_f64_lossy(finite(statistic)?)?;
    let p_value = finite(p_value)?;
    if !(0.0..=1.0).contains(&p_value) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "probability outside [0,1]",
        ));
    }
    Ok(DistributionTestResult {
        statistic,
        p_value,
        method,
        null_parameters,
        observations,
        groups,
        monte_carlo_interval: None,
    })
}

fn require_asymptotic(options: ClassicalOptions) -> Result<(), SciOxideError> {
    if options.calibration() == TestCalibration::Exact {
        Err(invalid(
            scioxide_core::argument!("options.calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "exact calibration is unavailable for this statistic",
        ))
    } else {
        Ok(())
    }
}

fn require_omnibus(options: ClassicalOptions) -> Result<(), SciOxideError> {
    if options.alternative() != TestAlternative::TwoSided {
        Err(invalid(
            scioxide_core::argument!("options.alternative"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "omnibus test requires a two-sided alternative",
        ))
    } else {
        Ok(())
    }
}

fn normal_tail(z: f64, alternative: TestAlternative) -> Result<f64, SciOxideError> {
    let normal = Normal::new(0.0, 1.0)?;
    Ok(match alternative {
        TestAlternative::Less => scioxide_random::Cdf::cdf(&normal, z)?,
        TestAlternative::Greater => normal.sf(z)?,
        TestAlternative::TwoSided => (2.0 * normal.sf(z.abs())?).min(1.0),
    })
}

fn chi_tail(x: f64, df: f64) -> Result<f64, SciOxideError> {
    ChiSquared::new(df)?.sf(x)
}

fn f_tail(x: f64, numerator: f64, denominator: f64) -> Result<f64, SciOxideError> {
    FisherF::with_options(
        numerator,
        denominator,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )?
    .sf(x)
}

pub(crate) use one_sample::sorted;
