#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;
use scioxide_core::{
    Cancellation, CancellationToken, Capacity, Complex, ExecutionContext, NumericScalar,
    RealScalar, SciOxideErrorKind,
};

use core::sync::atomic::{AtomicUsize, Ordering};

use scioxide_linalg::{LuDecomposition, Matrix};
use scioxide_signal::{
    FrequencyResponsePlan, FrequencyResponseWorkspace, IirDomain, LtiTime, PoleZeroGain,
    SecondOrderSections, StateSpace, TimeResponsePlan, TransferFunction,
};

fn model(time: LtiTime<f64>) -> StateSpace<f64> {
    let a = Matrix::from_vec(3, 3, vec![-2., 1., 0.5, 0.2, -3., 1., 0.3, -0.4, -4.]).unwrap();
    let b = Matrix::from_vec(3, 2, vec![1., 0., 0.5, 1., -1., 2.]).unwrap();
    let c = Matrix::from_vec(2, 3, vec![1., -0.5, 2., 0., 1., 0.5]).unwrap();
    let d = Matrix::from_vec(2, 2, vec![0.2, 0.1, -0.3, 0.5]).unwrap();
    StateSpace::new(time, a.view(), b.view(), c.view(), d.view()).unwrap()
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 3e-11 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn mimo_frequency_matches_independent_dense_solve_and_transfer_functions() {
    for time in [
        LtiTime::Continuous,
        LtiTime::Discrete { sample_period: 0.2 },
    ] {
        let model = model(time);
        let frequencies = [-3., 0., 0.2, 1., 4.];
        let response = model.frequency_response(&frequencies).unwrap();
        for (sample, frequency) in frequencies.into_iter().enumerate() {
            // Continuous: radians per time unit; discrete: multiples of Nyquist.
            let shift = match time {
                LtiTime::Continuous => Complex::new(0., frequency),
                LtiTime::Discrete { .. } => {
                    Complex::from_polar(1., frequency * core::f64::consts::PI)
                }
            };
            let a = Matrix::from_vec(
                3,
                3,
                model
                    .a()
                    .as_slice()
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        (if i / 3 == i % 3 {
                            shift
                        } else {
                            Complex::new(0., 0.)
                        }) - Complex::new(x, 0.)
                    })
                    .collect(),
            )
            .unwrap();
            let b = Matrix::from_vec(
                3,
                2,
                model
                    .b()
                    .as_slice()
                    .iter()
                    .map(|&x| Complex::new(x, 0.))
                    .collect(),
            )
            .unwrap();
            let solved = LuDecomposition::factor(&a)
                .unwrap()
                .solve_matrix(&b)
                .unwrap();
            for row in 0..2 {
                for col in 0..2 {
                    let expected = (0..3).fold(
                        Complex::new(model.d().as_slice()[row * 2 + col], 0.),
                        |sum, k| {
                            sum + Complex::new(model.c().as_slice()[row * 3 + k], 0.)
                                * solved.as_slice()[k * 2 + col]
                        },
                    );
                    let actual = response.values()[sample * 4 + row * 2 + col];
                    near(actual.re, expected.re);
                    near(actual.im, expected.im);
                    // Transfer functions share the state-space frequency convention.
                    let tf = model.to_transfer(col, row).unwrap();
                    let expected = tf.response(frequency).unwrap();
                    near(actual.re, expected.re);
                    near(actual.im, expected.im);
                }
            }
        }
    }
}

#[test]
fn prepared_responses_execute_on_first_and_repeated_execution() {
    for time in [
        LtiTime::Continuous,
        LtiTime::Discrete { sample_period: 0.1 },
    ] {
        let model = model(time);
        let plan = FrequencyResponsePlan::new(&model).unwrap();
        let mut ws = FrequencyResponseWorkspace::default();
        assert_eq!(
            error_kind(plan.workspace_requirements(usize::MAX)),
            SciOxideErrorKind::SizeOverflow
        );
        ws.try_reserve(&plan.workspace_requirements(1).unwrap())
            .unwrap();
        let req = plan.workspace_requirements(5).unwrap();
        assert_eq!(ws.check(&req).unwrap(), Capacity::NeedsGrowth);
        ws.try_reserve(&req).unwrap();
        assert_eq!(ws.check(&req).unwrap(), Capacity::Sufficient);
        ws.try_reserve(&plan.workspace_requirements(1).unwrap())
            .unwrap();
        assert_eq!(ws.check(&req).unwrap(), Capacity::Sufficient);
        let mut output = vec![Complex::new(0., 0.); req.output_len()];
        for _ in 0..2 {
            plan.frequency_response_into(
                &[0., 0.2, 0.5, 1., 2.],
                &mut output,
                &mut ws,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        }
        let plan = TimeResponsePlan::new(&model, &[0., 0.1, 0.2, 0.3, 0.4]).unwrap();
        let mut ws = plan.workspace().unwrap();
        let inputs: Vec<f64> = (0..plan.simulation_input_len())
            .map(|i| (i as f64 * 0.7).sin())
            .collect();
        let mut simulated = vec![0.; plan.simulation_output_len()];
        let mut history = vec![0.; plan.simulation_states_len()];
        for _ in 0..2 {
            plan.simulate_into(
                &inputs,
                &[0.5, -0.25, 1.],
                &mut simulated,
                &mut history,
                &mut ws,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        }
        let times = [0., 0.1, 0.2, 0.3, 0.4];
        let owned = model
            .simulate_zoh(&times, &inputs, &[0.5, -0.25, 1.])
            .unwrap();
        assert_eq!(owned.values(), simulated.as_slice());
        assert_eq!(owned.states(), history.as_slice());
        let mut output = vec![0.; plan.output_len()];
        let mut states = vec![0.; plan.states_len()];
        for _ in 0..2 {
            plan.impulse_response_into(
                &mut output,
                &mut states,
                &mut ws,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
            plan.step_response_into(
                &mut output,
                &mut states,
                &mut ws,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        }
    }
}

#[test]
fn time_responses_match_closed_forms_and_discrete_input_simulation() {
    let a = Matrix::from_vec(1, 1, vec![-2.0_f64]).unwrap();
    let b = Matrix::from_vec(1, 2, vec![1., 3.]).unwrap();
    let c = Matrix::from_vec(2, 1, vec![2., -1.]).unwrap();
    let d = Matrix::from_vec(2, 2, vec![0.5, 1., 2., -0.5]).unwrap();
    let model =
        StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap();
    let times = [0., 0.1, 0.2, 0.3];
    let impulse = model.impulse_response(&times).unwrap();
    let step = model.step_response(&times).unwrap();
    for (sample, t) in times.into_iter().enumerate() {
        for col in 0..2 {
            let decay = (-2. * t).exp();
            near(
                impulse.states()[sample * 2 + col],
                b.as_slice()[col] * decay,
            );
            near(
                step.states()[sample * 2 + col],
                b.as_slice()[col] * (1. - decay) / 2.,
            );
            for row in 0..2 {
                near(
                    impulse.values()[sample * 4 + row * 2 + col],
                    c.as_slice()[row] * b.as_slice()[col] * decay,
                );
                near(
                    step.values()[sample * 4 + row * 2 + col],
                    c.as_slice()[row] * b.as_slice()[col] * (1. - decay) / 2.
                        + d.as_slice()[row * 2 + col],
                );
            }
        }
    }
    let model = model
        .discretize(0.1, scioxide_signal::DiscretizationMethod::ZeroOrderHold)
        .unwrap();
    for step in [false, true] {
        let expected = if step {
            model.step_response(&times)
        } else {
            model.impulse_response(&times)
        }
        .unwrap();
        for col in 0..2 {
            let input: Vec<_> = (0..times.len() * 2)
                .map(|i| {
                    if i % 2 == col && (step || i / 2 == 0) {
                        1.
                    } else {
                        0.
                    }
                })
                .collect();
            let simulated = model.simulate_zoh(&times, &input, &[0.]).unwrap();
            for sample in 0..times.len() {
                near(
                    expected.states()[sample * 2 + col],
                    simulated.states()[sample],
                );
                for row in 0..2 {
                    near(
                        expected.values()[sample * 4 + row * 2 + col],
                        simulated.values()[sample * 2 + row],
                    );
                }
            }
        }
    }
}

struct StopAfter(AtomicUsize);
impl Cancellation for StopAfter {
    fn is_cancelled(&self) -> bool {
        self.0.fetch_sub(1, Ordering::Relaxed) == 0
    }
}

#[test]
fn errors_preserve_output_and_workspaces_can_be_reused() {
    let model = model(LtiTime::Continuous);
    let plan = FrequencyResponsePlan::new(&model).unwrap();
    let mut ws = plan.workspace(2).unwrap();
    let mut output = [Complex::new(7., 8.); 8];
    let sentinel = output;
    let stop = StopAfter(AtomicUsize::new(2));
    assert_eq!(
        error_kind(plan.frequency_response_into(
            &[0., 1.],
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native().with_cancellation(&stop)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(output, sentinel);
    assert_eq!(
        error_kind(plan.frequency_response_into(
            &[0., f64::NAN],
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(output, sentinel);
    assert_eq!(
        error_kind(plan.frequency_response_into(
            &[0., 1.],
            &mut output,
            &mut FrequencyResponseWorkspace::default(),
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, sentinel);
    assert_eq!(
        error_kind(plan.frequency_response_into(
            &[0., 1.],
            &mut output[..7],
            &mut ws,
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    plan.frequency_response_into(
        &[0., 1.],
        &mut output,
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    plan.frequency_response_into(&[], &mut [], &mut ws, ExecutionContext::sequential_native())
        .unwrap();

    let plan = TimeResponsePlan::new(&model, &[0., 0.1, 0.2]).unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut output = vec![7.; plan.output_len()];
    let mut states = vec![8.; plan.states_len()];
    assert_eq!(
        error_kind(plan.step_response_into(
            &mut output[1..],
            &mut states,
            &mut ws,
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        error_kind(plan.impulse_response_into(
            &mut output,
            &mut states[1..],
            &mut ws,
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert!(output.iter().all(|&v| v == 7.));
    assert!(states.iter().all(|&v| v == 8.));
    let stop = StopAfter(AtomicUsize::new(2));
    assert_eq!(
        error_kind(plan.step_response_into(
            &mut output,
            &mut states,
            &mut ws,
            ExecutionContext::sequential_native().with_cancellation(&stop)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert!(output.iter().all(|&v| v == 7.));
    assert!(states.iter().all(|&v| v == 8.));
    let short = TimeResponsePlan::new(&model, &[0.]).unwrap();
    assert_eq!(
        error_kind(plan.impulse_response_into(
            &mut output,
            &mut states,
            &mut short.workspace().unwrap(),
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert!(output.iter().all(|&v| v == 7.));
    assert!(states.iter().all(|&v| v == 8.));
    let mut simulated = vec![7.; plan.simulation_output_len()];
    let mut history = vec![8.; plan.simulation_states_len()];
    let stop = StopAfter(AtomicUsize::new(2));
    assert_eq!(
        error_kind(plan.simulate_into(
            &[1.; 6],
            &[0.; 3],
            &mut simulated,
            &mut history,
            &mut ws,
            ExecutionContext::sequential_native().with_cancellation(&stop)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(
        error_kind(plan.simulate_into(
            &[1.; 6],
            &[0.; 3],
            &mut simulated,
            &mut history,
            &mut short.workspace().unwrap(),
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert!(simulated.iter().all(|&v| v == 7.));
    assert!(history.iter().all(|&v| v == 8.));
    plan.step_response_into(
        &mut output,
        &mut states,
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(
        error_kind(TimeResponsePlan::new(&model, &[])),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(TimeResponsePlan::new(&model, &[0., 0.1, 0.3])),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(TimeResponsePlan::new(&model, &[0., f64::INFINITY])),
        SciOxideErrorKind::NonFinite
    );
    let token = CancellationToken::new();
    token.cancel();
    assert_eq!(
        error_kind(TimeResponsePlan::build(
            &model,
            &[0.],
            ExecutionContext::sequential_native().with_cancellation(&token)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
}

#[test]
fn late_singular_and_overflow_failures_are_atomic() {
    let a = Matrix::from_vec(1, 1, vec![0.0_f64]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1.]).unwrap();
    let d = Matrix::zeros(1, 1).unwrap();
    let model =
        StateSpace::new(LtiTime::Continuous, a.view(), b.view(), b.view(), d.view()).unwrap();
    let plan = FrequencyResponsePlan::new(&model).unwrap();
    let mut ws = plan.workspace(2).unwrap();
    let mut output = [Complex::new(7., 8.); 2];
    assert_eq!(
        error_kind(plan.frequency_response_into(
            &[1., 0.],
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::Singular
    );
    assert_eq!(output, [Complex::new(7., 8.); 2]);
    plan.frequency_response_into(
        &[1., 2.],
        &mut output,
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let mut near_pole = [Complex::new(0., 0.)];
    plan.frequency_response_into(
        &[1e-12],
        &mut near_pole,
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    near(near_pole[0].im, -1e12);

    let a = Matrix::from_vec(1, 1, vec![f64::MAX]).unwrap();
    let model = StateSpace::new(
        LtiTime::Discrete { sample_period: 1. },
        a.view(),
        b.view(),
        b.view(),
        d.view(),
    )
    .unwrap();
    let plan = TimeResponsePlan::new(&model, &[0., 1., 2., 3.]).unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut output = [7.; 4];
    let mut states = [8.; 4];
    assert_eq!(
        error_kind(plan.step_response_into(
            &mut output,
            &mut states,
            &mut ws,
            ExecutionContext::sequential_native()
        )),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(output, [7.; 4]);
    assert_eq!(states, [8.; 4]);
}

#[test]
fn zero_states_and_singleton_grids_keep_feedthrough_conventions() {
    for time in [
        LtiTime::Continuous,
        LtiTime::Discrete { sample_period: 0.5 },
    ] {
        let a = Matrix::<f64>::zeros(0, 0).unwrap();
        let b = Matrix::zeros(0, 1).unwrap();
        let c = Matrix::zeros(1, 0).unwrap();
        let d = Matrix::from_vec(1, 1, vec![2.]).unwrap();
        let model = StateSpace::new(time, a.view(), b.view(), c.view(), d.view()).unwrap();
        near(model.frequency_response(&[0.]).unwrap().values()[0].re, 2.);
        near(model.step_response(&[0.]).unwrap().values()[0], 2.);
        near(
            model.impulse_response(&[0.]).unwrap().values()[0],
            if matches!(time, LtiTime::Continuous) {
                0.
            } else {
                2.
            },
        );
    }
}

fn scalar_contract<T>()
where
    T: NumericScalar,
    T::Real: NumericScalar<Real = T::Real>,
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    let real = |x| T::Real::try_from_f64_lossy(x).unwrap();
    let scalar = |re, im| T::from_components(real(re), real(if T::IS_REAL { 0. } else { im }));
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            scalar(0.5, 0.1),
            scalar(0.1, -0.2),
            scalar(0.2, 0.3),
            scalar(0.4, -0.1),
        ],
    )
    .unwrap();
    let b = Matrix::from_vec(2, 1, vec![scalar(1., 0.2), scalar(0.5, -0.1)]).unwrap();
    let c = Matrix::from_vec(1, 2, vec![scalar(2., -0.1), scalar(1., 0.3)]).unwrap();
    let d = Matrix::from_vec(1, 1, vec![scalar(0.25, 0.1)]).unwrap();
    let model = StateSpace::new(
        LtiTime::Discrete {
            sample_period: real(0.1),
        },
        a.view(),
        b.view(),
        c.view(),
        d.view(),
    )
    .unwrap();
    let plan = FrequencyResponsePlan::new(&model).unwrap();
    let mut ws = plan.workspace(1).unwrap();
    let mut output = [Complex::new(real(0.), real(0.))];
    plan.frequency_response_into(
        &[real(0.3)],
        &mut output,
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let complex = |v: T| Complex::new(v.real_component(), v.imaginary_component());
    // Discrete frequencies are multiples of Nyquist; the sample period does not enter.
    let z = Complex::from_polar(real(1.), real(0.3) * T::Real::PI);
    let aa = z - complex(a.as_slice()[0]);
    let bb = -complex(a.as_slice()[1]);
    let cc = -complex(a.as_slice()[2]);
    let dd = z - complex(a.as_slice()[3]);
    let determinant = aa * dd - bb * cc;
    let x0 = (dd * complex(b.as_slice()[0]) - bb * complex(b.as_slice()[1])) / determinant;
    let x1 = (aa * complex(b.as_slice()[1]) - cc * complex(b.as_slice()[0])) / determinant;
    let expected =
        complex(c.as_slice()[0]) * x0 + complex(c.as_slice()[1]) * x1 + complex(d.as_slice()[0]);
    assert!(
        (output[0] - expected).norm()
            <= real(256.) * T::Real::EPSILON * (real(1.) + expected.norm())
    );
    let plan = TimeResponsePlan::new(&model, &[real(0.), real(0.1), real(0.2)]).unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut output = [T::ZERO; 3];
    let mut states = [T::ZERO; 6];
    plan.impulse_response_into(
        &mut output,
        &mut states,
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    for row in 0..2 {
        let ab =
            a.as_slice()[row * 2] * b.as_slice()[0] + a.as_slice()[row * 2 + 1] * b.as_slice()[1];
        assert!((states[4 + row] - ab).abs() <= real(64.) * T::Real::EPSILON);
    }
    plan.step_response_into(
        &mut output,
        &mut states,
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    for row in 0..2 {
        let expected = a.as_slice()[row * 2] * b.as_slice()[0]
            + a.as_slice()[row * 2 + 1] * b.as_slice()[1]
            + b.as_slice()[row];
        assert!((states[4 + row] - expected).abs() <= real(64.) * T::Real::EPSILON);
    }
}

#[test]
fn both_precisions_and_scalar_domains_match_analytic_responses_with_reused_storage() {
    scalar_contract::<f32>();
    scalar_contract::<f64>();
    scalar_contract::<Complex<f32>>();
    scalar_contract::<Complex<f64>>();
}

/// One system in every representation: transfer coefficients, roots, sections
/// and both state-space realizations must agree at the same frequencies.
fn representations_agree(domain: IirDomain, time: LtiTime<f64>, frequencies: &[f64]) {
    let close = |actual: Complex<f64>, expected: Complex<f64>| {
        assert!(
            (actual - expected).norm() <= 1e-12 * (1. + expected.norm()),
            "{actual:?} != {expected:?}"
        );
    };
    let (zeros, poles, gain, tf, sections) = match domain {
        // H(s) = 2 (s + 0.5) / ((s + 1)(s^2 + s + 2.5)).
        IirDomain::Analog => (
            [Complex::new(-0.5, 0.)],
            [
                Complex::new(-1., 0.),
                Complex::new(-0.5, 1.5),
                Complex::new(-0.5, -1.5),
            ],
            Complex::new(2., 0.),
            TransferFunction::new(domain, &[2., 1.], &[1., 2., 3.5, 2.5]).unwrap(),
            SecondOrderSections::new(
                domain,
                &[[0., 2., 1., 1., 1., 2.5], [0., 0., 1., 0., 1., 1.]],
            )
            .unwrap(),
        ),
        // H(z) = 0.7 (z + 0.5) / ((z - 0.5)(z^2 - 0.6 z + 0.25)), in powers of z^-1.
        IirDomain::Digital => (
            [Complex::new(-0.5, 0.)],
            [
                Complex::new(0.5, 0.),
                Complex::new(0.3, 0.4),
                Complex::new(0.3, -0.4),
            ],
            Complex::new(0.7, 0.),
            TransferFunction::new(domain, &[0., 0., 0.7, 0.35], &[1., -1.1, 0.55, -0.125]).unwrap(),
            SecondOrderSections::new(
                domain,
                &[[0., 0.7, 0.35, 1., -0.6, 0.25], [0., 1., 0., 1., -0.5, 0.]],
            )
            .unwrap(),
        ),
    };
    let zpk = PoleZeroGain::new(domain, &zeros, &poles, gain).unwrap();
    let roots = tf.to_zpk().unwrap();
    let realized = tf.to_state_space(time).unwrap();
    let cascade = zpk.to_state_space(time).unwrap();
    let sweep = realized.frequency_response(frequencies).unwrap();
    let cascade_sweep = cascade.frequency_response(frequencies).unwrap();
    let plan = FrequencyResponsePlan::new(&realized).unwrap();
    let mut ws = plan.workspace(frequencies.len()).unwrap();
    let mut planned = vec![Complex::new(0., 0.); frequencies.len()];
    plan.frequency_response_into(
        frequencies,
        &mut planned,
        &mut ws,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(sweep.frequencies(), frequencies);
    for (index, &frequency) in frequencies.iter().enumerate() {
        let expected = tf.response(frequency).unwrap();
        close(zpk.response(frequency).unwrap(), expected);
        close(roots.response(frequency).unwrap(), expected);
        close(sections.response(frequency).unwrap(), expected);
        close(sweep.values()[index], expected);
        close(cascade_sweep.values()[index], expected);
        close(planned[index], expected);
        let delay: f64 = tf.group_delay(frequency).unwrap();
        let section_delay: f64 = sections.group_delay(frequency).unwrap();
        assert!(
            (section_delay - delay).abs() <= 1e-11 * (1. + delay.abs()),
            "{section_delay} != {delay}"
        );
    }
}

#[test]
fn transfer_roots_sections_and_state_space_responses_agree_per_domain() {
    representations_agree(
        IirDomain::Analog,
        LtiTime::Continuous,
        &[0., 0.3, 1., 1.5, 2.5, 10.],
    );
    // Digital frequencies are multiples of Nyquist for every representation;
    // the state-space sample period does not enter.
    for sample_period in [0.25, 3.] {
        representations_agree(
            IirDomain::Digital,
            LtiTime::Discrete { sample_period },
            &[0., 0.1, 0.35, 0.8, 1.],
        );
    }
}

/// Uniform grid on `[0, stop]`: `k * (stop / (n - 1))` with the exact endpoint.
fn linspace(stop: f64, n: usize) -> Vec<f64> {
    let step = stop / (n - 1) as f64;
    let mut times: Vec<f64> = (0..n).map(|k| k as f64 * step).collect();
    times[n - 1] = stop;
    times
}

#[test]
fn generated_uniform_grids_are_accepted_at_every_magnitude() {
    let continuous = model(LtiTime::Continuous);
    let discrete = model(LtiTime::Discrete { sample_period: 0.1 });
    // Spacing deviations of these grids reach 2e-14 to 1.5e-12, beyond an
    // absolute tolerance, but stay within rounding of `k h` at `|t_k|`.
    let grids = [
        linspace(200.0, 20_001),
        linspace(1000.0, 100_001),
        (0..36_001).map(|k| k as f64 * 0.1).collect(),
        // User-built `k * (t1 - t0) / (n - 1)`.
        (0..50_001)
            .map(|k| k as f64 * 5000.0 / 50_000.0)
            .collect::<Vec<_>>(),
    ];
    for times in &grids {
        let deviation = times
            .windows(2)
            .map(|pair| ((pair[1] - pair[0]) - times[1]).abs())
            .fold(0.0, f64::max);
        assert!(deviation > 64.0 * f64::EPSILON * (1.0 + times[1]));
        TimeResponsePlan::new(&continuous, times).unwrap();
    }
    TimeResponsePlan::new(&discrete, &grids[2]).unwrap();
    // A displaced sample or a drifting step is still nonuniform.
    let mut displaced = linspace(1000.0, 100_001);
    displaced[50_000] *= 1.0 + 1e-9;
    let drifting: Vec<f64> = (0..10_001)
        .map(|k| k as f64 * 0.1 * (1.0 + 1e-12 * k as f64))
        .collect();
    for times in [&displaced, &drifting] {
        assert_eq!(
            error_kind(TimeResponsePlan::new(&continuous, times)),
            SciOxideErrorKind::InvalidInput
        );
    }
}
