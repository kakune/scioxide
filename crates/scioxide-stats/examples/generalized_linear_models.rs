//! Poisson rate regression with explicit exposure and model/robust uncertainty.
use scioxide_core::SciOxideError;
use scioxide_linalg::MatrixView;
use scioxide_stats::{
    ConfidenceProbability, GlmCovariance, GlmFamily, GlmOptions, PreparedGlm, Unweighted,
};

fn main() -> Result<(), SciOxideError> {
    let design = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let exposure = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let counts = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let model = PreparedGlm::new(
        MatrixView::from_slice(6, 1, &design)?,
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        Some(&exposure),
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(1_000_000),
    )?;
    let result = model.fit(scioxide_stats::GlmInputs::new(
        &counts,
        GlmOptions::default(),
    ))?;
    let Some(fit) = result.fit() else {
        println!(
            "termination={:?}; no accepted iterate",
            result.report().termination
        );
        return Ok(());
    };
    let model_interval = model.coefficient_inference(
        &counts,
        &fit,
        GlmCovariance::Model,
        ConfidenceProbability::new(0.95)?,
    )?;
    let robust_interval = model.coefficient_inference(
        &counts,
        &fit,
        GlmCovariance::Hc0,
        ConfidenceProbability::new(0.95)?,
    )?;
    println!(
        "termination={:?}, log rate={}, deviance={}, model CI={:?}, HC0 CI={:?}",
        result.report().termination,
        fit.coefficients().view().get(0)?,
        fit.deviance(),
        model_interval[0].interval(),
        robust_interval[0].interval(),
    );
    Ok(())
}
