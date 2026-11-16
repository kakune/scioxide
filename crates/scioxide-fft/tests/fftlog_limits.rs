//! Large-grid and direction-specific regressions for FFTLog.
#![expect(clippy::unwrap_used)]
#[path = "support/numerics.rs"]
mod numerics;
use numerics::{dot, norm, widen};
use scioxide_core::{AdjointOperator, Complex, ExecutionContext, LinearOperator};
use scioxide_fft::{FftLogPlan, FftLogPlanSpec, fht, ifht};

#[test]
fn final_underflow_is_ordinary_rounding() {
    use scioxide_core::RealScalar;
    let plan = FftLogPlan::new(FftLogPlanSpec::new(
        16,
        3.0,
        0.0,
        0.0,
        Complex::new(-8.0, 0.0),
        scioxide_fft::FftDirection::Forward,
    ))
    .unwrap();
    let mut work = plan.workspace(()).unwrap();
    let input = [Complex::new(1e-321_f64, 0.0); 16];
    let mut output = [Complex::new(17.0, -23.0); 16];
    plan.execute_into(
        &input,
        &mut output,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(output.iter().all(|value| value.is_finite()));
    assert!(output.iter().any(|value| *value == Complex::new(0.0, 0.0)));
    // Binary rescaling is exact on input and gives an independent final
    // projection where the original computation rounded to subnormal or zero.
    let scaled = input.map(|value| Complex::new(value.re.scalbn(512), 0.0));
    let mut reference = output;
    plan.execute_into(
        &scaled,
        &mut reference,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    for (actual, reference) in output.into_iter().zip(reference) {
        for (actual, reference) in [(actual.re, reference.re), (actual.im, reference.im)] {
            let expected = reference.scalbn(-512);
            assert!((actual - expected).abs() <= 5e-11 * expected.abs() + 4.0 * f64::from_bits(1));
        }
    }
}

#[test]
fn large_default_grids_match_high_precision() {
    let mut reference = include_bytes!("fixtures/fftlog_large.bin").chunks_exact(8);
    for n in [65536, 65537] {
        let input: Vec<_> = (0..n)
            .map(|j| ((j * 17 % 257) as f64 - 128.0) / 128.0)
            .collect();
        for inverse in [false, true] {
            let actual = if inverse {
                ifht(&input, 0.01, 0.0, 0.0, 0.0)
            } else {
                fht(&input, 0.01, 0.0, 0.0, 0.0)
            }
            .unwrap();
            let (error, mass) = actual.iter().fold((0.0, 0.0), |(error, mass), value| {
                let expected = f64::from_le_bytes(reference.next().unwrap().try_into().unwrap());
                (error + (value - expected).powi(2), mass + expected.powi(2))
            });
            assert!(
                (error / mass).sqrt() <= 5e-11,
                "n={n} inverse={inverse}: {}",
                (error / mass).sqrt()
            );
        }
    }
    assert!(reference.next().is_none());
}

#[derive(serde::Deserialize)]
struct SingularCase {
    n: usize,
    order: f64,
    bias: f64,
    output: Vec<f64>,
}

fn singular_width<S: scioxide_core::RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let cases: Vec<SingularCase> =
        serde_json::from_str(include_str!("fixtures/fftlog_singular.json")).unwrap();
    let scalar = |x| S::try_from_f64_lossy(x).unwrap();
    for case in cases {
        let plan = FftLogPlan::new(FftLogPlanSpec::new(
            case.n,
            scalar(0.1),
            scalar(case.order),
            S::ZERO,
            Complex::new(scalar(case.bias), S::ZERO),
            scioxide_fft::FftDirection::Forward,
        ))
        .unwrap();
        let plan_inverse = FftLogPlan::new(FftLogPlanSpec::new(
            case.n,
            scalar(0.1),
            scalar(case.order),
            S::ZERO,
            Complex::new(scalar(case.bias), S::ZERO),
            scioxide_fft::FftDirection::Inverse,
        ))
        .unwrap();
        let input: Vec<_> = (1..=case.n)
            .map(|j| Complex::new(scalar(j as f64), S::ZERO))
            .collect();
        let mut output = input.clone();
        let mut work = plan.workspace(()).unwrap();
        plan.execute_into(
            &input,
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let tolerance = if S::EPSILON.try_to_f64_exact().unwrap() < 1e-10 {
            5e-11
        } else {
            2e-5
        };
        for (actual, expected) in output.iter().zip(&case.output) {
            assert!(
                (actual.re.try_to_f64_exact().unwrap() - expected).abs()
                    <= tolerance * (1.0 + expected.abs())
            );
            assert!(
                actual.im.try_to_f64_exact().unwrap().abs() <= tolerance * (1.0 + expected.abs())
            );
        }
        let expected = output.clone();
        plan.apply_into(&input, &mut output, &mut work).unwrap();
        assert_eq!(output, expected);
        if case.bias != 0.0 {
            // Each case has a zero coefficient: the inverse plan and its
            // adjoint are singular, while the forward adjoint conjugates the
            // same coefficients and remains admitted.
            for adjoint in [false, true] {
                singular(if adjoint {
                    plan_inverse.apply_adjoint_into(&input, &mut output, &mut work)
                } else {
                    plan_inverse.execute_into(
                        &input,
                        &mut output,
                        &mut work,
                        ExecutionContext::sequential(),
                    )
                });
                assert_eq!(output, expected);
            }
            let shape = [2, case.n];
            let strides = [case.n, 1];
            let layout = scioxide_core::StridedLayout::new(&shape, &strides).unwrap();
            let axis_input = input.repeat(2);
            let mut axis_output = vec![Complex::new(scalar(123.0), scalar(456.0)); 2 * case.n];
            let before = axis_output.clone();
            let mut axis_work = plan_inverse
                .workspace(scioxide_fft::AxisWorkspaceRequest::Axis {
                    shape: &shape,
                    axis: 1,
                })
                .unwrap();
            singular(plan_inverse.execute_axis_into(
                scioxide_core::NdView::new(&axis_input, layout).unwrap(),
                1,
                scioxide_core::NdViewMut::new(&mut axis_output, layout).unwrap(),
                &mut axis_work,
                ExecutionContext::sequential_native(),
            ));
            assert_eq!(axis_output, before);
            let real_input: Vec<_> = input.iter().map(|z| z.re).collect();
            let before = real_input.clone();
            singular(
                ifht(
                    &real_input,
                    scalar(0.1),
                    scalar(case.order),
                    S::ZERO,
                    scalar(case.bias),
                )
                .map(|_| ()),
            );
            assert_eq!(real_input, before);
            assert_adjoint(&plan, &probe(case.n, 0.3), &probe(case.n, 0.5));
        } else {
            plan_inverse
                .execute_into(
                    &expected,
                    &mut output,
                    &mut work,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            for (a, b) in input.iter().zip(&output) {
                assert!((*a - *b).norm().try_to_f64_exact().unwrap() < tolerance * 10.0);
            }
        }
    }
}
#[test]
fn singular_f64() {
    singular_width::<f64>();
}
#[test]
fn singular_f32() {
    singular_width::<f32>();
}

fn singular(result: Result<(), scioxide_core::SciOxideError>) {
    let error = result.unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ),
        "{error:?}"
    );
}

fn probe<S: scioxide_core::RealScalar>(n: usize, frequency: f64) -> Vec<Complex<S>> {
    (0..n)
        .map(|j| {
            let phase = j as f64 * frequency + 0.25;
            Complex::new(
                S::try_from_f64_lossy(phase.sin()).unwrap(),
                S::try_from_f64_lossy(phase.cos()).unwrap(),
            )
        })
        .collect()
}

/// Check `<A x, y> = <x, A* y>` relative to both products' scales; return
/// that scale, which is zero only for the zero map.
fn assert_adjoint<S: scioxide_core::RealScalar>(
    plan: &FftLogPlan<S>,
    x: &[Complex<S>],
    y: &[Complex<S>],
) -> f64
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let mut work = plan.workspace(()).unwrap();
    let mut image = x.to_vec();
    let mut adjoint = y.to_vec();
    plan.apply_into(x, &mut image, &mut work).unwrap();
    plan.apply_adjoint_into(y, &mut adjoint, &mut work).unwrap();
    let scale = norm(&image) * norm(y) + norm(x) * norm(&adjoint);
    let tolerance = 64.0 * S::EPSILON.try_to_f64_exact().unwrap();
    let (lhs, rhs) = (dot(&image, y), dot(x, &adjoint));
    assert!(
        (lhs - rhs).norm() <= tolerance * scale,
        "{lhs:?} {rhs:?} {scale}"
    );
    scale
}

/// Adjoints share their plan's coefficients, so singularity follows the
/// direction: zeros are admitted forward and poles inverse, in both the plan
/// map and its adjoint, while the other direction refuses both without
/// publishing output.
fn adjoints_follow_the_plan_direction<S: scioxide_core::RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    use scioxide_fft::FftDirection::{Forward, Inverse};
    let n = 5;
    // Bias 1 makes the zero-frequency order-zero Mellin coefficient zero;
    // bias -1 makes it a pole.
    let plan = |bias: f64, direction| {
        FftLogPlan::new(FftLogPlanSpec::new(
            n,
            S::try_from_f64_lossy(0.1).unwrap(),
            S::ZERO,
            S::ZERO,
            Complex::new(S::try_from_f64_lossy(bias).unwrap(), S::ZERO),
            direction,
        ))
        .unwrap()
    };
    for admitted in [plan(1.0, Forward), plan(-1.0, Inverse)] {
        assert!(
            admitted
                .diagnostics()
                .log_multiplier_condition
                .is_infinite()
        );
        assert!(assert_adjoint(&admitted, &probe(n, 0.7), &probe(n, 0.4)) > 0.0);
    }
    for refused in [plan(-1.0, Forward), plan(1.0, Inverse)] {
        let input = probe::<S>(n, 0.7);
        let initial = probe::<S>(n, 0.4);
        let mut output = initial.clone();
        let mut work = refused.workspace(()).unwrap();
        singular(refused.execute_into(
            &input,
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        ));
        singular(refused.apply_into(&input, &mut output, &mut work));
        singular(refused.apply_adjoint_into(&input, &mut output, &mut work));
        assert_eq!(output, initial);
    }
}
#[test]
fn adjoints_follow_the_plan_direction_f64() {
    adjoints_follow_the_plan_direction::<f64>();
}
#[test]
fn adjoints_follow_the_plan_direction_f32() {
    adjoints_follow_the_plan_direction::<f32>();
}

/// Bias weights whose materialized power law over- or underflows S. Moving the
/// offset by whole periods `m n d` keeps every multiplier phase `e^(-i w o)`
/// and scales the plan map by `e^(-q m n d)` forward (`e^(q m n d)` inverse,
/// conjugated bias for adjoints). The far plan's input carries `2^(-+e)` so
/// its result is representable, while `e^(-+q o)` itself, and the staged
/// input times its weight, are not.
fn periodic_offsets_scale_out_of_range_weights<S: scioxide_core::RealScalar>(
    n: usize,
    spacing: f64,
    bias: Complex<f64>,
    periods: f64,
    exponent: i32,
    tolerance: f64,
) where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    use scioxide_fft::FftDirection::{Forward, Inverse};
    let scalar = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let shift = periods * n as f64 * spacing;
    for (direction, sign) in [(Forward, -1.0), (Inverse, 1.0)] {
        let plan = |offset: f64| {
            FftLogPlan::new(FftLogPlanSpec::new(
                n,
                scalar(spacing),
                S::ZERO,
                scalar(offset),
                Complex::new(scalar(bias.re), scalar(bias.im)),
                direction,
            ))
            .unwrap()
        };
        let (near, far) = (plan(0.0), plan(shift));
        let input = probe::<S>(n, 0.7);
        let lifted: Vec<_> = input
            .iter()
            .map(|z| {
                Complex::new(
                    z.re.scalbn(-sign as i32 * exponent),
                    z.im.scalbn(-sign as i32 * exponent),
                )
            })
            .collect();
        for adjoint in [false, true] {
            let run = |plan: &FftLogPlan<S>, input: &[Complex<S>]| {
                let mut output = input.to_vec();
                let mut work = plan.workspace(()).unwrap();
                if adjoint {
                    plan.apply_adjoint_into(input, &mut output, &mut work)
                } else {
                    plan.apply_into(input, &mut output, &mut work)
                }
                .unwrap();
                output
            };
            let q = if adjoint { bias.conj() } else { bias };
            let factor = Complex::new(
                sign * (q.re * shift - f64::from(exponent) * core::f64::consts::LN_2),
                sign * q.im * shift,
            )
            .exp();
            let expected: Vec<_> = run(&near, &input)
                .into_iter()
                .map(|z| widen(z) * factor)
                .collect();
            let actual: Vec<_> = run(&far, &lifted).into_iter().map(widen).collect();
            let error = expected
                .iter()
                .zip(&actual)
                .map(|(e, a)| (*e - *a).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let scale = norm(&expected);
            assert!(
                scale > 0.0 && error <= tolerance * scale,
                "{direction:?} adjoint={adjoint}: {error} vs {scale}"
            );
        }
    }
}
#[test]
fn periodic_offsets_scale_out_of_range_weights_f32() {
    // e^(2.8 * 32) = 2^129 leaves f32; two weight blocks per grid.
    periodic_offsets_scale_out_of_range_weights::<f32>(
        32,
        0.5,
        Complex::new(2.8, 0.5),
        2.0,
        100,
        1e-4,
    );
}
#[test]
fn periodic_offsets_scale_out_of_range_weights_f64() {
    // e^(2.8 * 272) = 2^1099 leaves f64; two weight blocks per grid.
    periodic_offsets_scale_out_of_range_weights::<f64>(
        32,
        0.5,
        Complex::new(2.8, 0.5),
        17.0,
        1000,
        1e-10,
    );
    // Steep weights: e^(6 * 128) = 2^1108, with 16 four-sample blocks.
    periodic_offsets_scale_out_of_range_weights::<f64>(
        64,
        1.0,
        Complex::new(6.0, 0.5),
        2.0,
        1000,
        1e-10,
    );
}

#[test]
fn nonfinite_bias_weights_report_overflow_at_preparation() {
    // The outer grid coordinates (j - 7.5) * MAX/4 overflow binary64. A zero
    // bias contributes nothing there and prepares; any nonzero component
    // would need an infinite power law, so preparation reports `Overflow`.
    let spec = |bias| {
        FftLogPlanSpec::new(
            16,
            f64::MAX / 4.0,
            0.0_f64,
            0.0,
            bias,
            scioxide_fft::FftDirection::Forward,
        )
    };
    assert!(FftLogPlan::new(spec(Complex::new(0.0, 0.0))).is_ok());
    for bias in [Complex::new(0.5, 0.0), Complex::new(0.0, 0.5)] {
        let error = FftLogPlan::new(spec(bias)).unwrap_err();
        assert!(
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ),
            "{error:?}"
        );
    }
}
