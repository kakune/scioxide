//! Discrete Hankel fixtures, inverse and adjoint identities and resource limits.
#![expect(clippy::unwrap_used)]
#[path = "support/hankel_zeros.rs"]
mod hankel_zeros;

use scioxide_core::{AdjointOperator, Complex, Tolerance};
use scioxide_fft::*;
#[test]
fn analytic_radial_transforms() {
    let p = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.0_f64)[..(32) + 1],
        scioxide_fft::DiscreteHankelPlanSpec::new(
            0.0_f64,
            8.0,
            scioxide_fft::FftDirection::Forward,
        )
        .with_tolerance(Tolerance::new(1e-13, 1e-13).unwrap())
        .with_limits(HankelLimits::default()),
    )
    .unwrap();
    let input: Vec<_> = p
        .nodes()
        .iter()
        .map(|&r| Complex::new((-0.5 * r * r).exp(), 0.0))
        .collect();
    let mut output = vec![Complex::new(0.0, 0.0); input.len()];
    p.execute_into(
        &input,
        &mut output,
        &mut p.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    for (&k, &v) in p.frequencies().iter().zip(&output) {
        assert!((v.re - (-0.5 * k * k).exp()).abs() < 5e-9, "{k} {v:?}");
    }
}
#[test]
fn half_order_involution_and_transactions() {
    let n = 16;
    let p = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.5_f64)[..(n) + 1],
        scioxide_fft::DiscreteHankelPlanSpec::new(
            0.5_f64,
            4.0,
            scioxide_fft::FftDirection::Forward,
        )
        .with_tolerance(Tolerance::new(1e-13, 1e-13).unwrap())
        .with_limits(HankelLimits::default()),
    )
    .unwrap();
    let p_inverse = DiscreteHankelPlan::new(
        &hankel_zeros::zeros(0.5_f64)[..(n) + 1],
        scioxide_fft::DiscreteHankelPlanSpec::new(
            0.5_f64,
            4.0,
            scioxide_fft::FftDirection::Inverse,
        )
        .with_tolerance(Tolerance::new(1e-13, 1e-13).unwrap())
        .with_limits(HankelLimits::default()),
    )
    .unwrap();
    let x: Vec<_> = (0..n).map(|j| Complex::new(j as f64, 1.0)).collect();
    let mut y = vec![Complex::new(0.0, 0.0); n];
    let mut z = y.clone();
    let mut w = p.workspace(()).unwrap();
    p.execute_into(
        &x,
        &mut y,
        &mut w,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    p_inverse
        .execute_into(
            &y,
            &mut z,
            &mut w,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
    assert!(x.iter().zip(&z).all(|(a, b)| (*a - *b).norm() < 1e-9));
    let before = y.clone();
    assert!(
        p.execute_into(
            &x,
            &mut y,
            &mut scioxide_fft::FftWorkspace::new(),
            scioxide_fft::FftPlanningContext::<f64>::default().execution()
        )
        .is_err()
    );
    assert_eq!(before, y);
}
#[path = "support/radial_fixtures.rs"]
mod radial_fixtures;
fn fixture_width<S: scioxide_core::RealScalar>(width: &str) {
    radial_fixtures::check_width::<S>(
        width,
        "dht",
        |fields, input, result| {
            let number = |j: usize| S::try_from_f64_lossy(fields[j].parse().unwrap()).unwrap();
            let n = input.len();
            let direction = match fields[11] {
                "forward" => FftDirection::Forward,
                "inverse" => FftDirection::Inverse,
                _ => unreachable!("unknown fixture direction"),
            };
            let plan = DiscreteHankelPlan::new(
                &hankel_zeros::zeros(number(3))[..n + 1],
                DiscreteHankelPlanSpec::new(number(3), number(4), direction).with_tolerance(
                    Tolerance::new(
                        S::EPSILON * S::try_from_f64_lossy(32.0).unwrap(),
                        S::EPSILON * S::try_from_f64_lossy(32.0).unwrap(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
            let mut workspace = plan.workspace(()).unwrap();
            if fields[15] == "1" {
                plan.apply_adjoint_into(input, result, &mut workspace)
                    .unwrap();
            } else {
                plan.execute_into(
                    input,
                    result,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            }
        },
        |actual, expected, mass, n| {
            let actual = actual.try_to_f64_exact().unwrap();
            let eta = if width == "32" {
                f32::from_bits(1) as f64
            } else {
                f64::from_bits(1)
            };
            let bound =
                8.0 * S::EPSILON.try_to_f64_exact().unwrap() * (n + 2) as f64 * mass + 2.0 * eta;
            assert!(
                (actual - expected).abs() <= bound,
                "n={n}: {actual:e} vs {expected:e}, bound {bound:e}"
            );
        },
    );
}
#[test]
fn independent_f64() {
    fixture_width::<f64>("64");
}
#[test]
fn independent_f32() {
    fixture_width::<f32>("32");
}

fn zero_sequences<S: scioxide_core::RealScalar>(width: u32) {
    #[derive(serde::Deserialize)]
    struct Case {
        width: u32,
        order: f64,
        zeros: Vec<f64>,
    }
    let cases: Vec<Case> = serde_json::from_str(include_str!("fixtures/dht_zeros.json")).unwrap();
    for case in cases.into_iter().filter(|case| case.width == width) {
        let scalar = |x| S::try_from_f64_lossy(x).unwrap();
        let zeros: Vec<_> = case.zeros.into_iter().map(scalar).collect();
        let tolerance = S::EPSILON * scalar(32.0);
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let spec = DiscreteHankelPlanSpec::new(scalar(case.order), scalar(4.0), direction)
                .with_tolerance(Tolerance::new(tolerance, tolerance).unwrap());
            DiscreteHankelPlan::new(&zeros[..6], spec).unwrap();
            // Omitting an even number of roots preserves derivative parity.
            for indices in [
                [0, 2, 3, 4, 5, 6],
                [0, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7],
            ] {
                let supplied = indices.map(|i| zeros[i]);
                let error = DiscreteHankelPlan::new(&supplied, spec).unwrap_err();
                assert!(
                    matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
                    if argument == scioxide_core::argument!("zeros")),
                    "order {}: {error:?}",
                    case.order
                );
            }
            let mut perturbed = zeros[..6].to_vec();
            perturbed[2] += scalar(0.01);
            assert!(
                matches!(DiscreteHankelPlan::new(&perturbed, spec).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("zeros"))
            );
        }
    }
}

#[test]
fn supplied_zeros_are_consecutive_in_both_precisions() {
    zero_sequences::<f32>(32);
    zero_sequences::<f64>(64);
}

fn extreme_output_failures<S: scioxide_core::RealScalar>(radius: f64) {
    let scalar = |x| S::try_from_f64_lossy(x).unwrap();
    let zeros = hankel_zeros::zeros(S::HALF);
    let tolerance = S::EPSILON * scalar(32.0);
    for (extent, direction) in [
        (radius, FftDirection::Forward),
        (1.0 / radius, FftDirection::Inverse),
    ] {
        let plan = DiscreteHankelPlan::new(
            &zeros[..6],
            DiscreteHankelPlanSpec::new(S::HALF, scalar(extent), direction)
                .with_tolerance(Tolerance::new(tolerance, tolerance).unwrap()),
        )
        .unwrap();
        let mut workspace = plan.workspace(()).unwrap();
        let input = [Complex::new(S::ONE, -S::HALF); 5];
        let initial = [Complex::new(scalar(123.0), scalar(-456.0)); 5];
        let mut output = initial;
        let error = plan
            .execute_into(
                &input,
                &mut output,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
        assert_eq!(output, initial);
        let error = plan
            .apply_adjoint_into(&input, &mut output, &mut workspace)
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
        assert_eq!(output, initial);
        let zero = [Complex::new(S::ZERO, S::ZERO); 5];
        plan.execute_into(
            &zero,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(output, zero);
        plan.apply_adjoint_into(&zero, &mut output, &mut workspace)
            .unwrap();
        assert_eq!(output, zero);
    }
}

#[test]
fn extreme_prepared_scales_defer_output_overflow_and_preserve_transactions() {
    extreme_output_failures::<f32>(1e20);
    extreme_output_failures::<f64>(1e160);
}
#[test]
fn large_supplied_zero_dhts_succeed_with_default_limits() {
    #[derive(serde::Deserialize)]
    struct Reference {
        n: usize,
        output: Vec<f64>,
    }
    let references: Vec<Reference> =
        serde_json::from_str(include_str!("fixtures/dht_large.json")).unwrap();
    for reference in references {
        let n = reference.n;
        assert_eq!(reference.output.len(), n);
        let zeros: Vec<_> = (1..=n + 1)
            .map(|j| j as f64 * std::f64::consts::PI)
            .collect();
        let p = DiscreteHankelPlan::new(
            &zeros,
            DiscreteHankelPlanSpec::new(0.5, 4.0, scioxide_fft::FftDirection::Forward)
                .with_tolerance(Tolerance::new(1e-12, 1e-12).unwrap()),
        )
        .unwrap();
        let mut input = vec![Complex::new(0.0, 0.0); n];
        input[0] = Complex::new(1.0, -0.5);
        let mut output = input.clone();
        p.execute_into(
            &input,
            &mut output,
            &mut p.workspace(()).unwrap(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        // For nu=1/2, J_(3/2)(pi)^2 = 2/pi^2 and the
        // first weighted matrix column is 16/(n+1)^2 J_(1/2)(x).
        for (i, value) in output.iter().enumerate() {
            let x = (i + 1) as f64 * std::f64::consts::PI / (n + 1) as f64;
            let expected = 16.0 / ((n + 1) as f64).powi(2)
                * (2.0 / (std::f64::consts::PI * x)).sqrt()
                * x.sin();
            assert!((value.re - expected).abs() <= 5e-11 * expected.abs() + 1e-18);
            let oracle = reference.output[i];
            assert!((value.re - oracle).abs() <= 5e-11 * oracle.abs() + 1e-18);
            assert!((value.im + 0.5 * expected).abs() <= 5e-11 * expected.abs() + 1e-18);
        }
    }
}

#[test]
fn default_limits_are_unbounded() {
    assert_eq!(
        scioxide_fft::HankelLimits::default().preparation_bytes,
        usize::MAX
    );
    assert_eq!(
        scioxide_fft::HankelLimits::default().workspace_bytes,
        usize::MAX
    );
}

fn default_tolerance<S: scioxide_core::RealScalar>() {
    let spec = DiscreteHankelPlanSpec::new(S::HALF, S::ONE, FftDirection::Forward);
    let (actual, expected) = (spec.tolerance(), Tolerance::<S>::from_sqrt_epsilon());
    assert_eq!(actual.absolute(), expected.absolute());
    assert_eq!(actual.relative(), expected.relative());
    let tight = Tolerance::new(S::EPSILON, S::EPSILON).unwrap();
    assert_eq!(
        spec.with_tolerance(tight).tolerance().absolute(),
        S::EPSILON
    );
}

#[test]
fn spec_defaults_to_sqrt_epsilon_zero_tolerance() {
    default_tolerance::<f32>();
    default_tolerance::<f64>();
    let zeros: [f64; 6] = core::array::from_fn(|k| (k + 1) as f64 * core::f64::consts::PI);
    let spec = DiscreteHankelPlanSpec::new(0.5, 4.0, FftDirection::Forward);
    let tight = Tolerance::new(1e-13, 1e-13).unwrap();
    let mut perturbed = zeros;
    perturbed[2] *= 1.0 + 1e-10;
    assert!(DiscreteHankelPlan::new(&perturbed, spec).is_ok());
    let mut moved = zeros;
    moved[2] += 0.01;
    for (supplied, spec) in [(perturbed, spec.with_tolerance(tight)), (moved, spec)] {
        let error = DiscreteHankelPlan::new(&supplied, spec).unwrap_err();
        assert!(
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    argument,
                    reason: scioxide_core::InvalidInputReason::OutOfRange,
                    ..
                } if argument == scioxide_core::argument!("zeros")
            ),
            "{error:?}"
        );
    }
    let with = spec.with_tolerance(tight).tolerance();
    assert_eq!((with.absolute(), with.relative()), (1e-13, 1e-13));
}
