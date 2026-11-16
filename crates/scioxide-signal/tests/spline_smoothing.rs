//! Independent polynomial, frequency-response, and dense-objective tests.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::{error, error_kind};
use scioxide_core::{
    AlgebraicScalar, Complex, ExecutionContext, NdView, NdViewMut, NumericScalar, NumericalIssue,
    RealScalar, SciOxideErrorKind, SciOxideErrorView, StridedLayout,
};
use scioxide_linalg::{LuDecomposition, Matrix, Vector};
use scioxide_signal::{
    SignalExecutionOptions, SignalLayout, SignalWorkspace, SplineDegree, SplineFilter,
    SplineRequest, SymmetricFilter, WhittakerHendersonPlan,
};
fn native() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}
const SEQUENTIAL: SignalExecutionOptions = SignalExecutionOptions::new();

#[test]
fn reconstruction_preserves_finite_extreme_coefficients() {
    fn run<T: NumericScalar>() {
        for degree in [SplineDegree::Quadratic, SplineDegree::Cubic] {
            let plan = SplineFilter::<T::Real>::new(degree, SEQUENTIAL).unwrap();
            let magnitude = T::Real::MAX * s(0.75);
            let coefficient = T::from_components(magnitude, -magnitude);
            let coefficients = [coefficient; 3];
            let mut output = [T::ZERO; 3];
            let mut scratch = plan
                .workspace(SplineRequest::Reconstruct(SignalLayout::Contiguous(3)))
                .unwrap();
            plan.reconstruct_into(&coefficients, &mut output, &mut scratch, native())
                .unwrap();
            for y in output {
                assert!(
                    (y.real_component() / magnitude - T::Real::ONE).abs()
                        <= s::<T::Real>(8.) * T::Real::EPSILON
                );
                if !T::IS_REAL {
                    assert!(
                        (y.imaginary_component() / magnitude + T::Real::ONE).abs()
                            <= s::<T::Real>(8.) * T::Real::EPSILON
                    );
                }
            }
        }
    }
    run::<f32>();
    run::<f64>();
    run::<Complex<f32>>();
    run::<Complex<f64>>();
}

#[test]
fn spline_queries_preserve_coordinate_offsets_and_range() {
    let plan = SplineFilter::<f32>::new(SplineDegree::Cubic, SEQUENTIAL).unwrap();
    let mut coefficients = vec![0.0_f32; 10000];
    coefficients[1] = 1.0;
    let mut workspace = plan.workspace(SplineRequest::Evaluate(1)).unwrap();
    let mut out = [0.0];
    let x = 0.123456_f32;
    plan.evaluate_into(
        &coefficients,
        &[x],
        0.0,
        1.0,
        &mut out,
        &mut workspace,
        native(),
    )
    .unwrap();
    let expected = scioxide_interp::cardinal_bspline(3, x - 1.0).unwrap();
    assert!((out[0] - expected).abs() < 8.0 * f32::EPSILON);
    plan.evaluate_into(
        &coefficients,
        &[f32::MAX],
        -f32::MAX,
        f32::MAX,
        &mut out,
        &mut workspace,
        native(),
    )
    .unwrap();
    assert!((out[0] - 1.0 / 6.0).abs() < 8.0 * f32::EPSILON);
}
fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn value<T: NumericScalar>(x: f64) -> T {
    T::from_real(s(x))
}
fn close<T: NumericScalar>(a: T, b: T, tol: f64) {
    let error = (a.real_component() - b.real_component()).abs().into_f64()
        + (a.imaginary_component() - b.imaginary_component())
            .abs()
            .into_f64();
    let scale = 1. + b.real_component().abs().into_f64() + b.imaginary_component().abs().into_f64();
    assert!(error <= tol * scale, "{a:?} != {b:?}; error {error}");
}
fn filtering<T: NumericScalar>(tol: f64) {
    for degree in [SplineDegree::Quadratic, SplineDegree::Cubic] {
        let plan = SplineFilter::<T::Real>::new(degree, SEQUENTIAL).unwrap();
        for n in [0, 1, 2, 3, 17, 129] {
            let input: Vec<T> = (0..n)
                .map(|i| T::from_components(s((i as f64 * 0.37).sin()), s((i as f64 * 0.13).cos())))
                .collect();
            let coefficients = plan.coefficients(&input).unwrap();
            let mut output = vec![T::ZERO; n];
            let mut scratch = SignalWorkspace::new();
            for request in [
                SplineRequest::Reconstruct(SignalLayout::Contiguous(n)),
                SplineRequest::Evaluate(n),
            ] {
                scratch
                    .try_reserve(&plan.workspace_requirements(request).unwrap())
                    .unwrap();
            }
            plan.reconstruct_into(&coefficients, &mut output, &mut scratch, native())
                .unwrap();
            for (&a, &b) in output.iter().zip(&input) {
                close(a, b, tol);
            }
            let coordinates: Vec<_> = (0..n).map(|i| s(i as f64)).collect();
            plan.evaluate_into(
                &coefficients,
                &coordinates,
                s(0.),
                s(1.),
                &mut output,
                &mut scratch,
                native(),
            )
            .unwrap();
            for (&a, &b) in output.iter().zip(&input) {
                close(a, b, tol);
            }
            if n > 0 {
                let constants = plan.coefficients(&vec![value::<T>(1.); n]).unwrap();
                for &x in &constants {
                    close(x, value(1.), tol);
                }
            }
        }
        // Polynomial reproduction from the analytic cardinal moments, away from endpoints.
        let degree_number = if degree == SplineDegree::Quadratic {
            2
        } else {
            3
        };
        for power in 0..=degree_number {
            let shift = if degree_number == 2 { 0.25 } else { 1. / 3. };
            let coeff: Vec<T> = (0..24)
                .map(|i| {
                    let x = i as f64;
                    value(match power {
                        0 => 1.,
                        1 => x,
                        2 => x * x - shift,
                        _ => x * x * x - x,
                    })
                })
                .collect();
            let coords: Vec<_> = (0..33).map(|i| s(4. + i as f64 / 4.)).collect();
            let mut y = vec![T::ZERO; 33];
            let mut scratch = plan.workspace(SplineRequest::Evaluate(33)).unwrap();
            plan.evaluate_into(
                &coeff,
                &coords,
                s(0.),
                s(1.),
                &mut y,
                &mut scratch,
                native(),
            )
            .unwrap();
            for (&x, &y) in coords.iter().zip(&y) {
                close(y, value(x.into_f64().powi(power)), tol);
            }
        }
    }
    // DCT-II modes are exact eigenvectors of half-sample reflected convolution.
    let n = 23;
    for mode in [0, 1, 6, 22] {
        let omega = std::f64::consts::PI * mode as f64 / n as f64;
        let x: Vec<T> = (0..n)
            .map(|i| value((omega * (i as f64 + 0.5)).cos()))
            .collect();
        for pole in [-0.7, 0., 0.8] {
            let plan =
                SymmetricFilter::first_order(s::<T::Real>(pole), s(0.7), SEQUENTIAL).unwrap();
            let gain = 0.7 / (1. + pole * pole - 2. * pole * omega.cos());
            for (&a, &b) in plan.filter(&x).unwrap().iter().zip(&x) {
                close(a, b * value(gain), tol * 8.);
            }
        }
        for (radius, angle) in [(0., 0.), (0.3, 0.), (0.8, 0.6), (0.7, std::f64::consts::PI)] {
            let a = 2. * radius * angle.cos();
            let b = -radius * radius;
            let re = 1. - a * omega.cos() - b * (2. * omega).cos();
            let im = a * omega.sin() + b * (2. * omega).sin();
            let plan =
                SymmetricFilter::second_order(s::<T::Real>(radius), s(angle), s(0.3), SEQUENTIAL)
                    .unwrap();
            let gain = 0.3 / (re * re + im * im);
            for (&actual, &input) in plan.filter(&x).unwrap().iter().zip(&x) {
                close(actual, input * value(gain), tol * 12.);
            }
        }
    }
}
fn smoothing<T: NumericScalar>(tol: f64) {
    let n = 9;
    for order in [1, 2, 3, 4] {
        for lambda in [0., 0.125, 10.] {
            let weights: Vec<T::Real> = (0..n)
                .map(|i| {
                    s(if i % 3 == 0 && lambda > 0. {
                        0.
                    } else {
                        1. + i as f64 / 8.
                    })
                })
                .collect();
            let plan = WhittakerHendersonPlan::<T>::new(
                n,
                order,
                s(lambda),
                Some(&weights),
                scioxide_signal::SignalExecutionOptions::new(),
            )
            .unwrap();
            let input: Vec<T> = (0..n)
                .map(|i| T::from_components(s((i as f64).sin()), s((i as f64 * 0.37).cos())))
                .collect();
            let (actual, diagnostics) = plan.smooth(&input).unwrap();
            // Independently construct D by repeated row differences of an identity matrix.
            let mut d: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1. } else { 0. }).collect())
                .collect();
            for _ in 0..order {
                d = d
                    .windows(2)
                    .map(|pair| pair[1].iter().zip(&pair[0]).map(|(a, b)| a - b).collect())
                    .collect();
            }
            let mut a = vec![T::ZERO; n * n];
            for (i, row) in a.chunks_exact_mut(n).enumerate() {
                for (j, element) in row.iter_mut().enumerate() {
                    *element = value(
                        lambda * d.iter().map(|r| r[i] * r[j]).sum::<f64>()
                            + if i == j { weights[i].into_f64() } else { 0. },
                    );
                }
            }
            let rhs = Vector::from_vec(
                input
                    .iter()
                    .zip(&weights)
                    .map(|(&x, &w)| x * T::from_real(w))
                    .collect(),
            );
            let expected = LuDecomposition::factor(&Matrix::from_vec(n, n, a).unwrap())
                .unwrap()
                .solve(&rhs)
                .unwrap();
            for (i, &a) in actual.iter().enumerate() {
                close(a, expected.view().get(i).unwrap(), tol * 20.);
            }
            assert!(diagnostics.scaled_optimality_residual().into_f64() < tol);
            assert!(diagnostics.data_fidelity() >= T::Real::ZERO);
            assert!(diagnostics.penalty() >= T::Real::ZERO);
            if lambda == 0. {
                for (&a, &b) in actual.iter().zip(&input) {
                    close(a, b, tol);
                }
            }
            for power in 0..order {
                let polynomial: Vec<T> = (0..n)
                    .map(|i| value((i as f64 / 8.).powi(power as i32)))
                    .collect();
                let (result, _) = plan.smooth(&polynomial).unwrap();
                for (&a, &b) in result.iter().zip(&polynomial) {
                    close(a, b, tol * 25.);
                }
            }
        }
    }
    let plan = WhittakerHendersonPlan::<T>::new(
        7,
        2,
        s(4.),
        Some(&[s(1.), s(0.), s(0.), s(0.), s(0.), s(0.), s(1.)]),
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let (line, _) = plan
        .smooth(&[
            value(0.),
            value(9.),
            value(-5.),
            value(3.),
            value(9.),
            value(-1.),
            value(6.),
        ])
        .unwrap();
    for (i, &x) in line.iter().enumerate() {
        close(x, value(i as f64), tol * 10.);
    }
}
#[test]
fn f32_filters() {
    filtering::<f32>(2e-6);
}
#[test]
fn f64_filters() {
    filtering::<f64>(2e-14);
}
#[test]
fn complex32_filters() {
    filtering::<Complex<f32>>(2e-6);
}
#[test]
fn complex64_filters() {
    filtering::<Complex<f64>>(2e-14);
}
#[test]
fn f32_smoothing() {
    smoothing::<f32>(3e-6);
}
#[test]
fn f64_smoothing() {
    smoothing::<f64>(2e-14);
}
#[test]
fn complex32_smoothing() {
    smoothing::<Complex<f32>>(3e-6);
}
#[test]
fn complex64_smoothing() {
    smoothing::<Complex<f64>>(2e-14);
}
#[test]
fn separable_views() {
    let shape = [4, 7];
    let layout = StridedLayout::new(&shape, &[7, 1]).unwrap();
    let input: Vec<f64> = (0..28).map(|i| (i as f64 * 0.31).sin()).collect();
    let mut rows = vec![0.; 28];
    let mut both = vec![0.; 28];
    let mut recovered = vec![0.; 28];
    let mut tmp = vec![0.; 28];
    let spline = SplineFilter::new(SplineDegree::Cubic, SEQUENTIAL).unwrap();
    let mut scratch = spline
        .workspace(SplineRequest::Coefficients(SignalLayout::Axis {
            shape: &shape,
            axis: 1,
        }))
        .unwrap();
    let seq = ExecutionContext::sequential();
    spline
        .coefficients_axis_into(
            NdView::new(&input, layout).unwrap(),
            1,
            NdViewMut::new(&mut rows, layout).unwrap(),
            &mut scratch,
            seq,
        )
        .unwrap();
    spline
        .coefficients_axis_into(
            NdView::new(&rows, layout).unwrap(),
            0,
            NdViewMut::new(&mut both, layout).unwrap(),
            &mut scratch,
            seq,
        )
        .unwrap();
    spline
        .reconstruct_axis_into(
            NdView::new(&both, layout).unwrap(),
            0,
            NdViewMut::new(&mut tmp, layout).unwrap(),
            &mut scratch,
            seq,
        )
        .unwrap();
    spline
        .reconstruct_axis_into(
            NdView::new(&tmp, layout).unwrap(),
            1,
            NdViewMut::new(&mut recovered, layout).unwrap(),
            &mut scratch,
            seq,
        )
        .unwrap();
    for (&a, &b) in recovered.iter().zip(&input) {
        close(a, b, 2e-14);
    }
    // A transposed output proves scatter respects the borrowed strides.
    let transposed = StridedLayout::new(&shape, &[1, 4]).unwrap();
    let wh = WhittakerHendersonPlan::new(
        7,
        2,
        0.5,
        None,
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let mut workspace = wh.workspace(4).unwrap();
    wh.smooth_axis_into(
        NdView::new(&input, layout).unwrap(),
        1,
        NdViewMut::new(&mut recovered, transposed).unwrap(),
        &mut workspace,
        seq,
    )
    .unwrap();
    for (i, row) in input.chunks(7).enumerate() {
        let (expected, _) = wh.smooth(row).unwrap();
        for (j, &x) in expected.iter().enumerate() {
            close(recovered[j * 4 + i], x, 2e-14);
        }
    }
}
#[test]
fn failures_are_transactional() {
    let spline = SplineFilter::<f64>::new(SplineDegree::Quadratic, SEQUENTIAL).unwrap();
    let mut output = [43.; 4];
    let mut scratch = spline
        .workspace(SplineRequest::Coefficients(SignalLayout::Contiguous(4)))
        .unwrap();
    assert_eq!(
        error_kind(spline.coefficients_into(
            &[1., f64::NAN, 2., 3.],
            &mut output,
            &mut scratch,
            native()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(output, [43.; 4]);
    let mut short = spline
        .workspace(SplineRequest::Coefficients(SignalLayout::Contiguous(3)))
        .unwrap();
    assert_eq!(
        spline
            .coefficients_into(&[1.; 4], &mut output, &mut short, native())
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [43.; 4]);
    assert_eq!(
        error_kind(spline.evaluate_into(
            &[1.; 4],
            &[f64::INFINITY],
            0.,
            1.,
            &mut output,
            &mut scratch,
            native()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(output, [43.; 4]);
    let mut empty = SignalWorkspace::new();
    assert_eq!(
        spline
            .evaluate_into(&[1.; 4], &[0.5], 0., 1., &mut output, &mut empty, native())
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [43.; 4]);
    assert_eq!(
        error_kind(SymmetricFilter::first_order(1., 1., SEQUENTIAL)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(SymmetricFilter::second_order(0.5, -1., 1., SEQUENTIAL)),
        SciOxideErrorKind::InvalidInput
    );
    for (n, d, lambda) in [(0, 2, 1.), (2, 2, 1.), (5, 0, 1.), (5, 2, -1.)] {
        assert_eq!(
            error_kind(WhittakerHendersonPlan::<f64>::new(
                n,
                d,
                lambda,
                None,
                scioxide_signal::SignalExecutionOptions::new()
            )),
            SciOxideErrorKind::InvalidInput
        );
    }
    assert_eq!(
        error_kind(WhittakerHendersonPlan::<f64>::new(
            4,
            2,
            1.,
            Some(&[0.; 4]),
            scioxide_signal::SignalExecutionOptions::new()
        )),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(WhittakerHendersonPlan::<f64>::new(
            4,
            2,
            0.,
            Some(&[1., 0., 1., 1.]),
            scioxide_signal::SignalExecutionOptions::new()
        )),
        SciOxideErrorKind::InvalidInput
    );
    let plan = WhittakerHendersonPlan::new(
        4,
        2,
        1.,
        None,
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let mut workspace = plan.workspace(1).unwrap();
    assert_eq!(
        error_kind(plan.smooth_into(
            &[1., 2., f64::NAN, 4.],
            &mut output,
            &mut workspace,
            native()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(output, [43.; 4]);
}

#[test]
fn zero_penalty_preserves_tiny_samples_with_extreme_positive_weights() {
    let input = [1e-100_f64, 2e-100, 3e-100];
    let plan = WhittakerHendersonPlan::new(
        3,
        2,
        0.,
        Some(&[1e-300, 1., 1.]),
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let (actual, diagnostics) = plan.smooth(&input).unwrap();
    assert_eq!(actual, input);
    assert_eq!(diagnostics.scaled_optimality_residual(), 0.);
}

fn zero_identity<T: NumericScalar>(weight: f64, sample: f64) {
    let input = [
        T::from_components(s(sample), s(-sample)),
        value(2.),
        value(3.),
    ];
    let plan = WhittakerHendersonPlan::new(
        3,
        2,
        s(0.),
        Some(&[s(weight), s(1.), s(1.)]),
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let (actual, diagnostics) = plan.smooth(&input).unwrap();
    assert_eq!(actual, input);
    assert_eq!(diagnostics.scaled_optimality_residual(), T::Real::ZERO);
    let shape = [2, 3];
    let layout = StridedLayout::new(&shape, &[3, 1]).unwrap();
    let input = input.repeat(2);
    let mut output = vec![T::ZERO; 6];
    let mut workspace = plan.workspace(2).unwrap();
    plan.smooth_axis_into(
        NdView::new(&input, layout).unwrap(),
        1,
        NdViewMut::new(&mut output, layout).unwrap(),
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, input);
}
#[test]
fn zero_penalty_identity_covers_both_precisions_complex_channels_and_axes() {
    zero_identity::<f32>(1e-30, 1e-20);
    zero_identity::<f64>(1e-300, 1e-100);
    zero_identity::<Complex<f32>>(1e-30, 1e-20);
    zero_identity::<Complex<f64>>(1e-300, 1e-100);
    let input = [f64::from_bits(1), -0., f64::MAX];
    let plan = WhittakerHendersonPlan::new(
        3,
        2,
        0.,
        Some(&[f64::from_bits(1), f64::MAX, 1.]),
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let (actual, _) = plan.smooth(&input).unwrap();
    assert_eq!(
        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        input.map(f64::to_bits)
    );
}
#[test]
fn unrepresentable_positive_penalty_rhs_is_an_error_without_output_commit() {
    let plan = WhittakerHendersonPlan::new(
        3,
        2,
        1e-300,
        Some(&[1e-300, 1., 1.]),
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let mut output = [43.; 3];
    let mut workspace = plan.workspace(1).unwrap();
    assert!(matches!(
        error(plan.smooth_into(&[1e-100, 0., 0.], &mut output, &mut workspace, native())).view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Underflow,
            ..
        }
    ));
    assert_eq!(output, [43.; 3]);
}

fn conditioning_threshold<S: RealScalar + NumericScalar<Real = S>>() {
    use scioxide_core::{NumericalIssue, SciOxideErrorView};
    use scioxide_signal::SmoothingConditioning;
    let bad = S::ONE / S::EPSILON;
    assert!(
        matches!(WhittakerHendersonPlan::<S>::new(8, 2, bad, None, scioxide_signal::SignalExecutionOptions::new()), Err(e)
        if matches!(e.view(), SciOxideErrorView::NumericalInstability { issue: NumericalIssue::IllConditioned, .. }))
    );
    let lambda = s::<S>(100.);
    let plan = WhittakerHendersonPlan::<S>::new(
        8,
        2,
        lambda,
        None,
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let bound = match plan.conditioning() {
        SmoothingConditioning::ReciprocalSpectralBound(value) => Some(value),
        _ => None,
    }
    .unwrap();
    assert!((bound.into_f64() - 1. / 1601.).abs() <= 4. * S::EPSILON.into_f64());
    let below = S::ONE / (s::<S>(48.) * S::EPSILON);
    assert!(
        WhittakerHendersonPlan::<S>::new(
            8,
            2,
            below,
            None,
            scioxide_signal::SignalExecutionOptions::new()
        )
        .is_ok()
    );
    let weights = [S::ONE, S::ZERO, S::ZERO, S::ONE];
    let missing = WhittakerHendersonPlan::<S>::new(
        4,
        2,
        S::ONE,
        Some(&weights),
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let estimate = match missing.conditioning() {
        SmoothingConditioning::OneNormEstimate(value) => Some(value),
        _ => None,
    }
    .unwrap();
    assert!(estimate.reciprocal().value() > S::ZERO);
    assert!(estimate.reciprocal().value() <= S::ONE);
}

#[test]
fn precision_loss_is_rejected_and_conditioning_is_reported() {
    conditioning_threshold::<f32>();
    conditioning_threshold::<f64>();
}

#[test]
fn extreme_range_diagnostics_recover_original_weight_units() {
    let values = [1., -2., 0.5, 3., -1.];
    let (reference, diagnostics) = WhittakerHendersonPlan::<f64>::new(
        5,
        2,
        1.,
        None,
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap()
    .smooth(&values)
    .unwrap();
    for (weight, amplitude, objective_scale) in [(1e-200, 1e200, 1e200), (1e200, 1e-200, 1e-200)] {
        let input = values.map(|x| x * amplitude);
        let plan = WhittakerHendersonPlan::<f64>::new(
            5,
            2,
            weight,
            Some(&[weight; 5]),
            scioxide_signal::SignalExecutionOptions::new(),
        )
        .unwrap();
        let (actual, observed) = plan.smooth(&input).unwrap();
        for (a, b) in actual.iter().zip(&reference) {
            assert!((a / amplitude - b).abs() < 1e-13);
        }
        assert!(
            (observed.data_fidelity() / objective_scale - diagnostics.data_fidelity()).abs()
                < 1e-12
        );
        assert!((observed.penalty() / objective_scale - diagnostics.penalty()).abs() < 1e-12);
        assert!(observed.scaled_optimality_residual() < 1e-13);
    }
    let plan = WhittakerHendersonPlan::<f64>::new(
        5,
        2,
        1.,
        None,
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let mut output = [43.; 5];
    let mut workspace = plan.workspace(1).unwrap();
    assert!(
        matches!(plan.smooth_into(&values.map(|x| x * 1e200), &mut output, &mut workspace, native()),
        Err(e) if matches!(e.view(), scioxide_core::SciOxideErrorView::NumericalInstability { issue: scioxide_core::NumericalIssue::Overflow, .. }))
    );
    assert_eq!(output, [43.; 5]);
}
