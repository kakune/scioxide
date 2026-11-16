//! Independent extreme-shape moment and entropy regression tests.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{
    Budget, NumericalIssue, RealScalar, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
    Tolerance,
};
use scioxide_random::*;

#[test]
fn extreme_value_variance_is_continuous_at_zero_shape() {
    let limit = core::f64::consts::PI.powi(2) / 6.0;
    for shape in [-1e-8_f64, 1e-8, -1e-7, 1e-7] {
        let variance = GeneralizedExtremeValue::new(shape)
            .unwrap()
            .variance()
            .unwrap();
        assert!((variance / limit - 1.0).abs() < 5.0 * shape.abs());
    }
}

#[test]
fn beta_moments_do_not_depend_on_rounded_means_or_reciprocal_shapes() {
    assert!((Beta::new(1e-310_f64, 1e-310).unwrap().variance().unwrap() - 0.25).abs() < 1e-12);
    assert!((Beta::new(1e20_f64, 1.0).unwrap().variance().unwrap() / 1e-40 - 1.0).abs() < 1e-12);
    assert!(
        (BetaPrime::new(1e200_f64, 1e200)
            .unwrap()
            .variance()
            .unwrap()
            / 2e-200
            - 1.0)
            .abs()
            < 1e-12
    );
}

#[test]
fn inverse_weibull_concentration_uses_gamma_moment_differences() {
    // Var(X) = pi^2/(6c^2) * (1 + O(1/c)); these bounds include the next term.
    let expected = core::f64::consts::PI.powi(2) / 6.0;
    assert!(
        (InverseWeibull::new(1e8_f64).unwrap().variance().unwrap() * 1e16 / expected - 1.0).abs()
            < 1e-7
    );
    assert!(
        (f64::from(InverseWeibull::new(1e4_f32).unwrap().variance().unwrap()) * 1e8 / expected
            - 1.0)
            .abs()
            < 1e-3
    );
}

#[test]
fn pareto_variances_survive_intermediate_overflow() {
    for (actual, expected, tolerance) in [
        (
            Pareto::new(1e110_f64).unwrap().variance().unwrap(),
            1e-220,
            1e-12,
        ),
        (
            f64::from(Pareto::new(1e15_f32).unwrap().variance().unwrap()),
            1e-30,
            1e-6,
        ),
        (
            GeneralizedPareto::new(-1e103_f64)
                .unwrap()
                .variance()
                .unwrap(),
            5e-310,
            1e-12,
        ),
        (
            f64::from(
                GeneralizedPareto::new(-1e13_f32)
                    .unwrap()
                    .variance()
                    .unwrap(),
            ),
            5e-40,
            1e-5,
        ),
    ] {
        assert!(
            (actual / expected - 1.0).abs() < tolerance,
            "{actual:e} != {expected:e}"
        );
    }
    for result in [
        Pareto::new(f64::MAX).unwrap().variance(),
        GeneralizedPareto::new(-f64::MAX).unwrap().variance(),
    ] {
        assert!(issue(&result.unwrap_err(), NumericalIssue::Underflow));
    }
}

#[test]
fn histogram_and_mixture_variances_scale_before_squaring() {
    let histogram = Histogram::from_masses(&[-1e200_f64, 0.0, 1.0], &[0.0, 1.0]).unwrap();
    assert!((histogram.variance().unwrap() - 1.0 / 12.0).abs() < 1e-15);
    let histogram = Histogram::from_masses(&[0.0_f64, 1.0, 1e200], &[1.0, 1e-300]).unwrap();
    assert!((histogram.variance().unwrap() / (1e100 / 3.0) - 1.0).abs() < 1e-12);
    let histogram = Histogram::from_masses(&[0.0_f64, 1e-200, 1e200], &[1e300, 1e-300]).unwrap();
    assert!((histogram.variance().unwrap() / (1e-200 / 3.0) - 1.0).abs() < 1e-12);
    let mixture = Mixture::new(
        vec![
            Normal::new(0.0_f64, 1.0).unwrap(),
            Normal::new(1e200, 1.0).unwrap(),
        ],
        &[1.0, 1e-300],
    )
    .unwrap();
    assert!((mixture.variance().unwrap() / 1e100 - 1.0).abs() < 1e-12);
    let tiny = f64::from_bits(1);
    let histogram = Histogram::from_masses(&[tiny, 2.0 * tiny], &[1.0]).unwrap();
    assert_eq!(histogram.mean().unwrap(), 2.0 * tiny);
}

fn query<S: RealScalar>(
    name: &str,
    p: [S; 4],
    operation: &str,
    work: usize,
) -> Result<S, SciOxideError> {
    let options = LawOptions::default().with_work_budget(Budget::ONE.saturating_add(work - 1));
    let [a, b, c, d] = p;
    match name {
        "weibull" => WeibullMin::new(a)?.variance(),
        "johnson" => JohnsonSU::new(a, b)?.variance(),
        "bradford" if operation == "mean" => Bradford::new(a)?.mean(),
        "bradford" => Bradford::new(a)?.variance(),
        "gamma" => Gamma::new(a, b)?.entropy(),
        "beta" => Beta::new(a, b)?.entropy(),
        "student" => StudentT::new(a)?.entropy(),
        "generalized" if operation == "mean" => GeneralizedGamma::new(a, b)?.mean(),
        "generalized" => GeneralizedGamma::new(a, b)?.variance(),
        "mielke" => Mielke::new(a, b)?.variance(),
        "trapezoid" => Trapezoid::new(a, b)?.variance(),
        "gausshyper" => GaussHypergeometric::with_options(a, b, c, d, options)?.variance(),
        "range" => StudentizedRange::with_options(a, b, options)?.variance(),
        "ksone" => {
            KolmogorovSmirnovOneSided::with_options(a.into_f64() as usize, options)?.variance()
        }
        "kstwo" => {
            KolmogorovSmirnovTwoSided::with_options(a.into_f64() as usize, options)?.variance()
        }
        _ => panic!("unknown fixture law {name}"),
    }
}

fn issue(error: &SciOxideError, expected: NumericalIssue) -> bool {
    matches!(error.view(), SciOxideErrorView::NumericalInstability { issue, .. } if issue == expected)
}

fn fixtures<S: RealScalar>(bits: &str) {
    for row in include_str!("../data/moment_cancellation.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = row.split('\t').collect();
        if fields[1] != bits {
            continue;
        }
        let p = core::array::from_fn(|i| {
            S::try_from_f64_lossy(fields[i + 2].parse().unwrap()).unwrap()
        });
        let entropy = fields[6] == "entropy";
        let result = query(
            fields[0],
            p,
            fields[6],
            if fields[8] == "work" { 256 } else { 1 << 24 },
        );
        match (fields[8], result) {
            ("ok" | "checked", Ok(value)) => {
                let actual = value.into_f64();
                let expected: f64 = fields[7].parse().unwrap();
                let scale = if entropy {
                    1.0_f64.max(expected.abs())
                } else {
                    expected.abs()
                };
                assert!(
                    actual.is_finite() && (entropy || actual > 0.0),
                    "{row}: {value:?}"
                );
                assert!(
                    (actual - expected).abs() <= S::SQRT_EPSILON.into_f64() * scale,
                    "{row}: {actual:e} != {expected:e}"
                );
            }
            ("overflow", Err(error)) => {
                assert!(issue(&error, NumericalIssue::Overflow), "{row}: {error}")
            }
            ("underflow", Err(error)) => {
                assert!(issue(&error, NumericalIssue::Underflow), "{row}: {error}")
            }
            ("checked", Err(error)) => assert!(
                issue(&error, NumericalIssue::Cancellation)
                    || issue(&error, NumericalIssue::Underflow)
                    || error.kind() == SciOxideErrorKind::ConvergenceFailure,
                "{row}: {error}"
            ),
            ("work", Err(error)) => assert!(
                matches!(
                    error.kind(),
                    SciOxideErrorKind::ConvergenceFailure | SciOxideErrorKind::ResourceLimit
                ),
                "{row}: {error}"
            ),
            (_, result) => panic!("{row}: {result:?}"),
        }
    }
}

#[test]
fn independent_extreme_moments_f64() {
    fixtures::<f64>("64");
}

#[test]
fn independent_extreme_moments_f32() {
    fixtures::<f32>("32");
}

fn positive_grid<S: RealScalar>() {
    for exponent in -12..=15 {
        let shape = S::try_from_f64_lossy(10.0_f64.powi(exponent)).unwrap();
        let three = S::try_from_f64_lossy(3.0).unwrap();
        let options = LawOptions::default().with_work_budget(Budget::ONE.saturating_add(255));
        let endpoint = shape / (S::ONE + shape);
        let count = shape.into_f64().max(1.0) as usize;
        let results = [
            Gamma::new(shape, S::ONE).unwrap().variance(),
            Beta::new(shape, shape).unwrap().variance(),
            StudentT::new(shape).unwrap().variance(),
            Mielke::new(shape, shape + three).unwrap().variance(),
            Trapezoid::new(endpoint, endpoint).unwrap().variance(),
            GaussHypergeometric::with_options(shape, S::ONE, S::ZERO, S::ZERO, options)
                .and_then(|law| law.variance()),
            StudentizedRange::with_options(three, shape, options).and_then(|law| law.variance()),
            KolmogorovSmirnovOneSided::with_options(count, options).and_then(|law| law.variance()),
            KolmogorovSmirnovTwoSided::with_options(count, options).and_then(|law| law.variance()),
        ];
        for result in results {
            match result {
                Ok(value) => assert!(
                    value.is_finite() && value > S::ZERO,
                    "shape={shape:?}: {value:?}"
                ),
                Err(error) => assert!(
                    matches!(
                        error.kind(),
                        SciOxideErrorKind::DomainError
                            | SciOxideErrorKind::ConvergenceFailure
                            | SciOxideErrorKind::ResourceLimit
                            | SciOxideErrorKind::NumericalInstability
                    ),
                    "shape={shape:?}: {error}"
                ),
            }
        }
    }
}

#[test]
fn shape_grids_never_succeed_with_nonpositive_variance() {
    positive_grid::<f32>();
    positive_grid::<f64>();
}

#[test]
fn variance_failures_are_typed_and_work_is_shared() {
    for error in [
        GeneralizedGamma::new(2.0_f64, -1.0)
            .unwrap()
            .variance()
            .unwrap_err(),
        Mielke::new(1.0_f64, 2.0).unwrap().variance().unwrap_err(),
    ] {
        assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
    }
    assert!(issue(
        &WeibullMin::new(f64::MAX).unwrap().variance().unwrap_err(),
        NumericalIssue::Underflow
    ));
    assert!(issue(
        &JohnsonSU::new(0.0, f64::MAX)
            .unwrap()
            .variance()
            .unwrap_err(),
        NumericalIssue::Underflow
    ));
    assert!(issue(
        &Mielke::new(1e15_f64, 1e15).unwrap().variance().unwrap_err(),
        NumericalIssue::Cancellation
    ));
    assert!(issue(
        &Mielke::new(1e-320_f64, 3.0)
            .unwrap()
            .variance()
            .unwrap_err(),
        NumericalIssue::Underflow
    ));
    let options = LawOptions::default().with_work_budget(Budget::ONE);
    assert_eq!(
        StudentizedRange::with_options(3.0_f64, 5.0, options)
            .unwrap()
            .variance()
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
}

#[test]
fn configurable_variance_accuracy_is_enforced() {
    for relative in [1e-4_f64, 1e-8, 1e-11] {
        let options = LawOptions::default().with_tolerance(Tolerance::new(0.0, relative).unwrap());
        let law = GaussHypergeometric::with_options(2.0, 3.0, 0.0, 0.0, options).unwrap();
        let actual = law.variance().unwrap();
        assert!((actual - 0.04).abs() <= relative * 0.04);
    }
    let options =
        LawOptions::default().with_tolerance(Tolerance::new(0.0, 64.0 * f64::EPSILON).unwrap());
    let error = GaussHypergeometric::with_options(2.0, 3.0, 0.0, 0.0, options)
        .unwrap()
        .variance()
        .unwrap_err();
    assert!(issue(&error, NumericalIssue::Cancellation));
}
