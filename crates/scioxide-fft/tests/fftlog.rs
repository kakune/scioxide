//! FFTLog fixtures, inverse identities, low-ringing offsets and axis execution.
#![expect(clippy::unwrap_used)]

use scioxide_core::{AdjointOperator, Complex, ExecutionPolicy};
use scioxide_fft::*;
fn spec(n: usize, bias: Complex<f64>, direction: FftDirection) -> FftLogPlanSpec<f64> {
    FftLogPlanSpec::new(n, 0.125, 0.0, 0.0, bias, direction)
        .with_execution_policy(ExecutionPolicy::Sequential)
        .with_limits(FftLogLimits::default())
}
#[test]
fn fftlog_roundtrip_and_adjoint() {
    for n in [1, 15, 16, 31, 32] {
        for bias in [Complex::new(0.0, 0.0), Complex::new(0.4, 0.2)] {
            let p = FftLogPlan::new(spec(n, bias, FftDirection::Forward)).unwrap();
            let inverse = FftLogPlan::new(spec(n, bias, FftDirection::Inverse)).unwrap();
            let x: Vec<_> = (0..n)
                .map(|i| Complex::new((i as f64 * 0.7).sin(), (i as f64 * 0.4).cos()))
                .collect();
            let y: Vec<_> = (0..n)
                .map(|i| Complex::new((i as f64 * 0.3).cos(), -(i as f64 * 0.5).sin()))
                .collect();
            let mut out = vec![Complex::new(0.0, 0.0); n];
            let mut back = out.clone();
            let mut scratch = p.workspace(()).unwrap();
            p.execute_into(
                &x,
                &mut out,
                &mut scratch,
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
            .unwrap();
            inverse
                .execute_into(
                    &out,
                    &mut back,
                    &mut scratch,
                    scioxide_fft::FftPlanningContext::<f64>::default().execution(),
                )
                .unwrap();
            assert!(
                x.iter().zip(&back).all(|(a, b)| (*a - *b).norm() < 5e-12),
                "{n} {bias:?} {x:?} {back:?}"
            );
            p.apply_adjoint_into(&y, &mut back, &mut scratch).unwrap();
            let lhs = out
                .iter()
                .zip(&y)
                .fold(Complex::new(0.0, 0.0), |a, (x, y)| a + x.conj() * *y);
            let rhs = x
                .iter()
                .zip(&back)
                .fold(Complex::new(0.0, 0.0), |a, (x, y)| a + x.conj() * *y);
            assert!((lhs - rhs).norm() < 2e-11, "{n} {bias:?}: {lhs:?} {rhs:?}");
            scioxide_core::LinearOperator::apply_into(&inverse, &x, &mut out, &mut scratch)
                .unwrap();
            inverse
                .apply_adjoint_into(&y, &mut back, &mut scratch)
                .unwrap();
            let lhs = out
                .iter()
                .zip(&y)
                .fold(Complex::new(0.0, 0.0), |sum, (a, b)| sum + a.conj() * *b);
            let rhs = x
                .iter()
                .zip(&back)
                .fold(Complex::new(0.0, 0.0), |sum, (a, b)| sum + a.conj() * *b);
            assert!(
                (lhs - rhs).norm() < 2e-11 * (1.0 + lhs.norm()),
                "inverse {n} {bias:?}: {lhs:?} {rhs:?}"
            );
        }
    }
}
#[test]
fn analytic_fftlog() {
    let spacing = 0.06;
    let offset = fftlog_low_ringing_offset(spacing, 0.0, 0.0, 0.0).unwrap();
    let config = FftLogPlanSpec::new(
        512,
        spacing,
        0.0_f64,
        offset,
        Complex::new(0.0, 0.0),
        scioxide_fft::FftDirection::Forward,
    );
    let p = FftLogPlan::new(config).unwrap();
    let input: Vec<_> = (0..512)
        .map(|j| {
            let r = p.log_radius(j).unwrap().exp();
            Complex::new(r * (-0.5 * r * r).exp(), 0.0)
        })
        .collect();
    let mut output = vec![Complex::new(0.0, 0.0); 512];
    p.execute_into(
        &input,
        &mut output,
        &mut p.workspace(()).unwrap(),
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    for (j, &v) in output.iter().enumerate() {
        let k = p.log_frequency(j).unwrap().exp();
        if (0.1..10.0).contains(&k) {
            assert!((v.re - k * (-0.5 * k * k).exp()).abs() < 5e-8, "{k} {v:?}");
        }
    }
}
#[test]
fn short_workspace_and_preparation_limit_are_rejected() {
    let n = 16;
    let x: Vec<_> = (0..n).map(|j| Complex::new(j as f64, 1.0)).collect();
    let mut y = vec![Complex::new(123.0, -456.0); n];
    let before = y.clone();
    let p = FftLogPlan::new(spec(n, Complex::new(0.0, 0.0), FftDirection::Forward)).unwrap();
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
    let mut config = spec(n, Complex::new(0.0, 0.0), FftDirection::Forward);
    config = config.with_limits(FftLogLimits {
        preparation_bytes: 1,
        ..config.limits()
    });
    assert!(FftLogPlan::new(config).is_err());
}

#[test]
fn borrowed_real_axis_and_reused_storage() {
    use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    let n = 5;
    let plan = FftLogPlan::new(spec(n, Complex::new(0.25, 0.125), FftDirection::Forward)).unwrap();
    let shape = [n, 2];
    let strides = [2, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let input: Vec<f64> = (0..n * 2).map(|j| j as f64 * 0.1).collect();
    let mut out = vec![Complex::new(0.0, 0.0); n * 2];
    let mut work = scioxide_fft::FftWorkspace::from_vec(vec![
        Complex::new(0.0, 0.0);
        plan.workspace_requirements(
            scioxide_fft::AxisWorkspaceRequest::Axis {
                shape: &shape,
                axis: 0
            }
        )
        .map(|r| r.complex_len())
        .unwrap()
    ]);
    {
        plan.execute_axis_into(
            NdView::new(&input, layout).unwrap(),
            0,
            NdViewMut::new(&mut out, layout).unwrap(),
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    };
    for k in 0..2 {
        let lane: Vec<_> = (0..n)
            .map(|j| Complex::new(input[j * 2 + k], 0.0))
            .collect();
        let mut expected = vec![Complex::new(0.0, 0.0); n];
        plan.execute_into(
            &lane,
            &mut expected,
            &mut plan.workspace(()).unwrap(),
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
        assert!((0..n).all(|j| (out[j * 2 + k] - expected[j]).norm() < 2e-12));
    }
    let before = out.clone();
    let mut invalid = input;
    invalid[n * 2 - 1] = f64::NAN;
    assert!(
        plan.execute_axis_into(
            NdView::new(&invalid, layout).unwrap(),
            0,
            NdViewMut::new(&mut out, layout).unwrap(),
            &mut work,
            ExecutionContext::sequential(),
        )
        .is_err()
    );
    assert_eq!(out, before);
}
#[path = "support/radial_fixtures.rs"]
mod radial_fixtures;
fn fixture_width<S: scioxide_core::RealScalar>(width: &str) {
    radial_fixtures::check_width::<S>(
        width,
        "fftlog",
        |fields, input, result| {
            let number = |j: usize| S::try_from_f64_lossy(fields[j].parse().unwrap()).unwrap();
            let plan = FftLogPlan::new(FftLogPlanSpec::new(
                input.len(),
                number(4),
                number(3),
                number(7),
                Complex::new(number(5), number(6)),
                FftDirection::Forward,
            ))
            .unwrap();
            plan.execute_into(
                input,
                result,
                &mut plan.workspace(()).unwrap(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        },
        |actual, expected, _, n| {
            let expected = S::try_from_f64_lossy(expected).unwrap();
            let error = (actual - expected).abs().try_to_f64_exact().unwrap();
            let limit = if width == "64" { 5e-11 } else { 2e-5 };
            assert!(
                error <= limit * (1.0 + expected.abs().try_to_f64_exact().unwrap()),
                "n={n}: {actual:?} vs {expected:?}"
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
