//! Public regressions for numerical failures versus rejected parameters.
#![allow(clippy::unwrap_used)]
#[path = "../support/errors.rs"]
mod errors;
use scioxide_core::{NumericScalar, NumericalIssue, RealScalar, SciOxideErrorKind, Tolerance};
use scioxide_linalg::Matrix;
use scioxide_random::*;

#[test]
fn alpha_inverse_range_failure_is_numerical_overflow() {
    for error in [
        Alpha::new(2.0_f64)
            .unwrap()
            .isf(f64::from_bits(1))
            .unwrap_err(),
        Alpha::new(2.0_f32)
            .unwrap()
            .isf(f32::from_bits(1))
            .unwrap_err(),
    ] {
        errors::numerical(error, NumericalIssue::Overflow);
    }
}

fn derived_constants<S: RealScalar>() {
    let tiny = S::ZERO.next_after(S::ONE);
    let large = S::MAX;
    for error in [
        ExponentiallyModifiedNormal::new(tiny).unwrap_err(),
        DoubleParetoLogNormal::new(S::ZERO, S::ONE, large, large).unwrap_err(),
        TruncatedPareto::new(large, S::try_from_f64_lossy(4.0).unwrap()).unwrap_err(),
        GeneralizedHalfLogistic::new(tiny).unwrap_err(),
        GeneralizedPareto::new(-tiny).unwrap_err(),
        GeneralizedExtremeValue::new(tiny).unwrap_err(),
        Argus::new(large).unwrap_err(),
        GaussHypergeometric::new(S::ONE, S::ONE, large, large).unwrap_err(),
        RelativisticBreitWigner::new(large).unwrap_err(),
    ] {
        errors::numerical(error, NumericalIssue::Overflow);
    }
    errors::numerical(
        RelativisticBreitWigner::new(tiny).unwrap_err(),
        NumericalIssue::Underflow,
    );
}

#[test]
fn derived_constants_retain_numerical_range_errors() {
    derived_constants::<f32>();
    derived_constants::<f64>();
}

#[test]
fn mathematical_parameter_violations_are_domain_errors() {
    errors::domain(
        RandomCorrelation::<f64>::new(&[1.0, 2.0]).unwrap_err(),
        scioxide_core::argument!("eigenvalues"),
    );
    errors::domain(
        VonMisesFisher::<f64>::new(&[1.0, 1.0], 1.0).unwrap_err(),
        scioxide_core::argument!("mean"),
    );
    assert_eq!(
        VonMisesFisher::<f64>::new(&[1.0], 1.0).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn underflowed_draws_do_not_blame_valid_parameters() {
    let law = DoubleParetoLogNormal::new(-10_000.0_f64, 1.0, 2.0, 2.0).unwrap();
    errors::numerical(
        law.sample(&mut SplitMix64::from_seed(7)).unwrap_err(),
        NumericalIssue::Underflow,
    );
    let law = NormalInverseGamma::new(0.0_f64, 1.0, 1e6, f64::from_bits(1)).unwrap();
    errors::numerical(
        law.sample(&mut SplitMix64::from_seed(7)).unwrap_err(),
        NumericalIssue::Underflow,
    );
    let law32 = NormalInverseGamma::new(0.0_f32, 1.0, 1e6, f32::from_bits(1)).unwrap();
    errors::numerical(
        law32.sample(&mut SplitMix64::from_seed(7)).unwrap_err(),
        NumericalIssue::Underflow,
    );
    let law = Wishart::new(1e-3, &Matrix::<f64>::identity(1).unwrap()).unwrap();
    let mut rng = SplitMix64::from_seed(7);
    let mut underflows = 0;
    for _ in 0..32 {
        match law.sample(&mut rng) {
            Ok(value) => assert!(value[0] > 0.0 && value[0].is_finite()),
            Err(error) => {
                errors::numerical(error, NumericalIssue::Underflow);
                underflows += 1;
            }
        }
    }
    assert!(underflows > 0);
}

fn gaussian_limit<S: RealScalar + NumericScalar<Real = S>>() {
    let shape = Matrix::<S>::identity(2).unwrap();
    let location = [S::ZERO, S::ONE];
    let t = MultivariateStudentT::gaussian_limit(&location, &shape).unwrap();
    let borrowed = MultivariateStudentT::gaussian_limit(&location, shape.view()).unwrap();
    let normal = MultivariateNormal::new(&location, &shape).unwrap();
    for df in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
        assert_eq!(
            MultivariateStudentT::new(&location, &shape, df)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(
            MultivariateStudentT::new(&location, shape.view(), df)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    for df in [S::ZERO, -S::ONE] {
        errors::domain(
            MultivariateStudentT::new(&location, &shape, df).unwrap_err(),
            scioxide_core::argument!("df"),
        );
    }
    assert_eq!(t.mean().unwrap(), location);
    assert_eq!(t.variance().unwrap(), vec![S::ONE; 2]);
    assert_eq!(t.covariance().unwrap().as_slice(), shape.as_slice());
    assert_eq!(
        t.log_pdf(&location).unwrap(),
        borrowed.log_pdf(&location).unwrap()
    );
    let error = (t.log_pdf(&location).unwrap() - normal.log_pdf(&location).unwrap()).abs();
    assert!(error <= S::EPSILON * S::try_from_f64_lossy(8.0).unwrap());
    let root = Philox4x64::from_seed(1);
    let tp = t.prepare_probability(Default::default(), &root).unwrap();
    let np = normal
        .prepare_probability(Default::default(), &root)
        .unwrap();
    let a = tp
        .rectangle_scoped(
            &[f64::NEG_INFINITY; 2],
            &[0.0, 1.0],
            &mut tp.workspace().unwrap(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    let b = np.rectangle(&[f64::NEG_INFINITY; 2], &[0.0, 1.0]).unwrap();
    assert_eq!(a.probability, b.probability);
    let singular = Matrix::from_vec(2, 2, vec![S::ONE, S::ZERO, S::ZERO, S::ZERO]).unwrap();
    let t = MultivariateStudentT::gaussian_limit(&location, &singular).unwrap();
    assert_eq!(t.rank(), 1);
    assert!(
        t.log_pseudo_density(&location, Tolerance::from_machine_epsilon())
            .unwrap()
            .is_finite()
    );
    let zero = Matrix::zeros(2, 2).unwrap();
    let t = MultivariateStudentT::gaussian_limit(&location, &zero).unwrap();
    assert_eq!(t.rank(), 0);
    assert_eq!(t.sample(&mut SplitMix64::from_seed(1)).unwrap(), location);
    assert_eq!(t.variance().unwrap(), vec![S::ZERO; 2]);
}

#[test]
fn gaussian_limit_is_explicit_in_both_precisions() {
    gaussian_limit::<f32>();
    gaussian_limit::<f64>();
}

#[derive(Clone)]
struct ZeroBits;
impl BitGenerator for ZeroBits {
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn fill_bytes(&mut self, bytes: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
        bytes.fill(0);
        Ok(())
    }
}
#[test]
fn singular_generated_group_factors_are_breakdowns() {
    errors::numerical(
        OrthogonalGroup::<f64>::new(2)
            .unwrap()
            .sample(&mut ZeroBits)
            .unwrap_err(),
        NumericalIssue::Breakdown,
    );
    errors::numerical(
        UnitaryGroup::<f64>::new(2)
            .unwrap()
            .sample(&mut ZeroBits)
            .unwrap_err(),
        NumericalIssue::Breakdown,
    );
}

#[test]
fn extreme_kolmogorov_logs_are_range_failures() {
    let law = Kolmogorov::<f64>::new().unwrap();
    errors::numerical(
        law.log_cdf(f64::from_bits(1)).unwrap_err(),
        NumericalIssue::Overflow,
    );
    errors::numerical(law.log_sf(f64::MAX).unwrap_err(), NumericalIssue::Overflow);
    assert_eq!(law.cdf(f64::from_bits(1)).unwrap(), 0.0);
    assert_eq!(law.sf(f64::MAX).unwrap(), 0.0);
}

#[test]
fn non_finite_parameters_name_the_parameter() {
    for (error, name) in [
        (TukeyLambda::new(f64::NAN).unwrap_err(), "lambda"),
        (Pearson3::new(f64::INFINITY).unwrap_err(), "skew"),
        (
            GeneralizedHyperbolic::new(1.0, 2.0, f64::NAN).unwrap_err(),
            "b",
        ),
        (
            Normal::new(0.0_f64, 1.0)
                .unwrap()
                .cdf(f64::NAN)
                .unwrap_err(),
            "x",
        ),
    ] {
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite, "{error}");
        assert!(
            error.to_string().contains(&format!("{name} is non-finite")),
            "expected {name}: {error}"
        );
    }
}

#[test]
fn generalized_hyperbolic_boundary_rejects_the_index() {
    errors::domain(
        GeneralizedHyperbolic::new(0.5_f64, 1.0, 1.0).unwrap_err(),
        scioxide_core::argument!("p"),
    );
    errors::domain(
        GeneralizedHyperbolic::new(-0.5_f64, 1.0, 1.5).unwrap_err(),
        scioxide_core::argument!("b"),
    );
    errors::domain(
        GeneralizedHyperbolic::new(-0.5_f64, 0.0, 0.0).unwrap_err(),
        scioxide_core::argument!("a"),
    );
}
