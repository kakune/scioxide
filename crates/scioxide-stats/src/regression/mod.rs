//! Linear and generalized linear models, robust regression, and tests that compare groups.
use crate::TestAlternative;
use crate::error_helpers::{finite, invalid, number, zeros};
mod glm;
mod groups;
mod linear;
mod operator;
mod posthoc;
mod robust;
mod simple;
pub use glm::{
    GlmAccepted, GlmBatchInputs, GlmBatchOutput, GlmBatchWorkspace, GlmCoefficientInference,
    GlmCovariance, GlmFamily, GlmFit, GlmFitView, GlmInputs, GlmLink, GlmObjective,
    GlmOperatorWorkspace, GlmOptions, GlmOutput, GlmReport, GlmResult, GlmSolver,
    GlmStorageOptions, GlmWorkspace, PreparedGlm, PreparedOperatorGlm, glm_batch_into,
};
pub use groups::{
    GroupTest, GroupTestMethod, PreparedGroups, friedman_test, friedman_test_with_options,
    group_tests_into, kruskal_wallis_test, kruskal_wallis_test_with_options,
};
pub use linear::{
    LinearRegressionFit, LinearRegressionOptions, PreparedRegression, RegressionContrast,
    RegressionCovariance, RegressionFactorization, RegressionWorkspace, regression_batch_into,
};
pub use operator::{
    OperatorRegressionMethod, OperatorRegressionOptions, OperatorRegressionReport,
    OperatorRegressionResult, OperatorRegressionWorkspace, PreparedOperatorRegression,
};
pub use posthoc::{PostHocComparison, PostHocMethod, PostHocOptions, PostHocResult};
pub use robust::{
    HuberAccepted, HuberFit, HuberFitView, HuberOptions, HuberOutput, HuberReport, HuberWorkspace,
    PreparedHuber, RobustIntercept, RobustSlope, RobustSlopeMethod, RobustSlopeWorkspace,
    huber_location, huber_location_with_options, robust_slope, robust_slope_with_options,
};
pub use simple::{LinearRegression, linear_regression};

use scioxide_core::SciOxideError;

fn budget_bytes(elements: usize, width: usize, limit: usize) -> Result<(), SciOxideError> {
    let bytes = scioxide_core::backend::checked_mul(
        elements,
        width,
        scioxide_core::quantity!("regression_storage_bytes"),
    )?;
    crate::error_helpers::limit_bytes(bytes, limit)
}

fn t_probability(t: f64, degrees: f64, alternative: TestAlternative) -> Result<f64, SciOxideError> {
    if degrees <= 0.0 || !degrees.is_finite() {
        return Err(invalid(
            scioxide_core::argument!("response"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "positive residual degrees required",
        ));
    }
    let x = if t.abs() > 1.0 {
        let r = scioxide_core::RealScalar::sqrt(degrees) / t.abs();
        r * r / (1.0 + r * r)
    } else {
        degrees / (degrees + t * t)
    };
    let two = scioxide_special::beta_inc(degrees / 2.0, 0.5, x)?;
    Ok(match alternative {
        TestAlternative::TwoSided => two,
        TestAlternative::Less => {
            if t < 0.0 {
                two / 2.0
            } else {
                1.0 - two / 2.0
            }
        }
        TestAlternative::Greater => {
            if t > 0.0 {
                two / 2.0
            } else {
                1.0 - two / 2.0
            }
        }
    })
}
fn f_probability(f: f64, numerator: f64, denominator: f64) -> Result<f64, SciOxideError> {
    if f < 0.0 || numerator <= 0.0 || denominator <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("response"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid F statistic or degrees",
        ));
    }
    scioxide_random::Survival::sf(
        &scioxide_random::FisherF::with_options(
            numerator,
            denominator,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )?,
        f,
    )
}

pub use linear::ordinary_least_squares;

pub use glm::glm_fit;
