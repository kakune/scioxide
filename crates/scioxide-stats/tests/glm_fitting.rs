use scioxide_core::{RealScalar, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_stats::{
    Analytic, FitDerivatives, FitObjective, Frequency, GlmFamily, GlmObjective, PreparedGlm,
    Unweighted, Weights,
};

fn score_checks<S: RealScalar>(step: f64, tolerance: f64) -> Result<(), SciOxideError> {
    let s = |v| S::try_from_f64_lossy(v);
    let design = Matrix::from_vec(3, 1, vec![s(1.)?; 3])?;
    let weights = [s(1.)?, s(2.)?, s(3.)?];
    let offsets = [s(0.1)?; 3];
    for (family, values, parameter, expected) in [
        (GlmFamily::Gaussian, [1., 2., 3.], 0.5, -10.4),
        (
            GlmFamily::Binomial,
            [0., 0.5, 1.],
            0.5,
            6. * (1. / (1. + (-0.6_f64).exp())) - 4.,
        ),
        (
            GlmFamily::Poisson,
            [1., 2., 3.],
            0.5,
            6. * 0.6_f64.exp() - 14.,
        ),
        (GlmFamily::Gamma, [1., 2., 3.], 0.5, 14. - 6. / 0.6),
        (
            GlmFamily::InverseGaussian,
            [1., 2., 3.],
            0.5,
            (14. - 6. / 0.6_f64.sqrt()) / 2.,
        ),
    ] {
        let model = PreparedGlm::new(
            design.view(),
            family,
            Weights::<_, Frequency>::new(&weights),
            None,
            Some(&offsets),
            None,
            scioxide_stats::GlmStorageOptions::new().with_max_bytes(4096),
        )?;
        let response = [s(values[0])?, s(values[1])?, s(values[2])?];
        let objective = GlmObjective::new(&model, &response)?;
        let mut gradient = [s(0.)?];
        objective.gradient(&[s(parameter)?], &mut gradient)?;
        assert!((gradient[0].try_to_f64_exact()? - expected).abs() < tolerance);
        let upper = objective
            .value(&[s(parameter + step)?], &mut ())?
            .try_to_f64_exact()?;
        let lower = objective
            .value(&[s(parameter - step)?], &mut ())?
            .try_to_f64_exact()?;
        assert!(((upper - lower) / (2. * step) - expected).abs() < tolerance * 10.);
        assert_eq!(objective.workspace_bytes()?, 0);
    }
    Ok(())
}
#[test]
fn analytic_family_scores_f64() -> Result<(), SciOxideError> {
    score_checks::<f64>(1e-5, 1e-8)
}
#[test]
fn analytic_family_scores_f32() -> Result<(), SciOxideError> {
    score_checks::<f32>(0.005, 0.003)
}
#[test]
fn binomial_trials_exposure_and_invalid_shapes() -> Result<(), SciOxideError> {
    let design = Matrix::from_vec(2, 1, vec![1.; 2])?;
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Binomial,
        Unweighted,
        Some(&[2, 4]),
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(4096),
    )?;
    let objective = GlmObjective::new(&model, &[0., 1.])?;
    let mut gradient = [0.];
    objective.gradient(&[0.], &mut gradient)?;
    assert_eq!(gradient, [-1.]);
    let poisson = PreparedGlm::new(
        design.view(),
        GlmFamily::Poisson,
        Unweighted,
        None,
        None,
        Some(&[2., 3.]),
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(4096),
    )?;
    let objective = GlmObjective::new(&poisson, &[1., 2.])?;
    objective.gradient(&[0.], &mut gradient)?;
    assert!((gradient[0] - 2_f64).abs() < 1e-12);
    assert!(GlmObjective::new(&model, &[2., 0.]).is_err());
    assert!(GlmObjective::new(&model, &[0.]).is_err());
    assert!(objective.value(&[], &mut ()).is_err());
    assert!(objective.gradient(&[0.], &mut []).is_err());
    let gamma = PreparedGlm::new(
        design.view(),
        GlmFamily::Gamma,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(4096),
    )?;
    assert!(GlmObjective::new(&gamma, &[0., 1.]).is_err());
    let objective = GlmObjective::new(&gamma, &[1., 2.])?;
    assert!(objective.validate(&[-1.], &mut ()).is_err());
    Ok(())
}

#[test]
fn overflowing_gamma_objective_is_a_structured_failure() -> Result<(), SciOxideError> {
    let design = Matrix::from_vec(3, 1, vec![1.0_f64; 3])?;
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Gamma,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(4096),
    )?;
    let objective = GlmObjective::new(&model, &[1e308; 3])?;
    assert!(matches!(objective.value(&[1e10], &mut ()), Err(ref error)
    if matches!(error.view(), scioxide_core::SciOxideErrorView::NumericalInstability {
        issue: scioxide_core::NumericalIssue::Overflow, ..
    })));
    // Value and derivative have independent representability limits.
    let objective = GlmObjective::new(&model, &[1e307; 3])?;
    assert!(objective.value(&[1e10], &mut ()).is_err());
    let mut gradient = [0.0];
    objective.gradient(&[1e10], &mut gradient)?;
    assert!((gradient[0] / 3e307 - 1.0).abs() < 1e-14);
    Ok(())
}

fn weighted_gamma_range<S: RealScalar>(
    response: f64,
    expected_value: f64,
    expected_gradient: f64,
) -> Result<(), SciOxideError> {
    let s = |v| S::try_from_f64_lossy(v);
    let design = Matrix::from_vec(3, 1, vec![s(1.0)?; 3])?;
    let weights = [s(1e-20)?; 3];
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Gamma,
        Weights::<_, Analytic>::new(&weights),
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(4096),
    )?;
    let response = [s(response)?; 3];
    let objective = GlmObjective::new(&model, &response)?;
    let value = objective.value(&[s(1e10)?], &mut ())?;
    let mut gradient = [S::ZERO];
    objective.gradient(&[s(1e10)?], &mut gradient)?;
    let tolerance = 16.0 * S::EPSILON.try_to_f64_exact()?;
    assert!((value.try_to_f64_exact()? / expected_value - 1.0).abs() < tolerance);
    assert!((gradient[0].try_to_f64_exact()? / expected_gradient - 1.0).abs() < tolerance);
    Ok(())
}

#[test]
fn weighting_precedes_deviance_projection() -> Result<(), SciOxideError> {
    weighted_gamma_range::<f64>(1e308, 3e298, 3e288)?;
    weighted_gamma_range::<f32>(1e30, 3e20, 3e10)
}

#[test]
fn half_deviance_and_subnormal_weights_preserve_representable_values() -> Result<(), SciOxideError>
{
    let design = Matrix::from_vec(3, 1, vec![1.0_f64; 3])?;
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Gaussian,
        Unweighted,
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(4096),
    )?;
    let objective = GlmObjective::new(&model, &[1.5e154, 0.0, 0.0])?;
    let value = objective.value(&[0.0], &mut ())?;
    assert!((value / 1.125e308 - 1.0).abs() < 1e-14);
    let tiny = f64::from_bits(1);
    let model = PreparedGlm::new(
        design.view(),
        GlmFamily::Gaussian,
        Weights::<_, Analytic>::new(&[tiny; 3]),
        None,
        None,
        None,
        scioxide_stats::GlmStorageOptions::new().with_max_bytes(4096),
    )?;
    let objective = GlmObjective::new(&model, &[1e154; 3])?;
    let value = objective.value(&[0.0], &mut ())?;
    assert!((value / 7.410984687618698e-16 - 1.0).abs() < 1e-14);
    Ok(())
}

#[test]
fn all_families_reject_nonfinite_responses_and_parameters() -> Result<(), SciOxideError> {
    let design = Matrix::from_vec(3, 1, vec![1.0_f64; 3])?;
    for family in [
        GlmFamily::Gaussian,
        GlmFamily::Binomial,
        GlmFamily::Poisson,
        GlmFamily::Gamma,
        GlmFamily::InverseGaussian,
    ] {
        let model = PreparedGlm::new(
            design.view(),
            family,
            Unweighted,
            None,
            None,
            None,
            scioxide_stats::GlmStorageOptions::new().with_max_bytes(4096),
        )?;
        let objective = GlmObjective::new(&model, &[0.5; 3])?;
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(GlmObjective::new(&model, &[0.5, bad, 0.5]).is_err());
            assert!(objective.value(&[bad], &mut ()).is_err());
            assert!(objective.gradient(&[bad], &mut [0.0]).is_err());
        }
    }
    Ok(())
}
