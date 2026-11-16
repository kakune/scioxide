//! Owned answers agree with configured execution and preserve failure diagnostics.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Complex, ConvergenceReason, InvalidInputReason, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView, argument,
};
use scioxide_linalg::{Matrix, MatrixView};
use scioxide_ode::*;

fn decay(_: f64, y: &[f64], dy: &mut [f64]) -> Result<(), SciOxideError> {
    dy[0] = -y[0];
    Ok(())
}
#[test]
fn ivp_defaults_sampling_and_backward_direction() {
    let system = OdeFn::new(1, decay).unwrap();
    let configured = DormandPrince5::new()
        .solve(
            &OdeProblem::new(&system, 1.0),
            &OdeOptions::default(),
            OdeState::from_slice(0.0, &[1.0]).unwrap(),
        )
        .unwrap();
    let end = solve_ivp(decay, 0.0, 1.0, &[1.0]).unwrap();
    assert_eq!(end.as_slice(), configured.state().values());
    assert!((end.as_slice()[0] - (-1.0_f64).exp()).abs() < 1e-3);
    let times = [0., 0., 0.5, 0.5, 1., 1.];
    let sampled = solve_ivp_at(decay, &times, &[1.]).unwrap();
    let options =
        OdeOptions::default().with_output_schedule(OdeOutputSchedule::times(times.to_vec()));
    let configured = DormandPrince5::new()
        .solve(
            &OdeProblem::new(&system, 1.),
            &options,
            OdeState::from_slice(0., &[1.]).unwrap(),
        )
        .unwrap();
    for (i, values) in configured.trajectory().states().iter().enumerate() {
        assert_eq!(sampled.view().get(i, 0).unwrap(), values.as_slice()[0]);
    }
    assert_eq!(sampled.view().get(5, 0).unwrap(), end.as_slice()[0]);
    assert_eq!(
        sampled.view().get(2, 0).unwrap(),
        sampled.view().get(3, 0).unwrap()
    );
    let back = solve_ivp_at(decay, &[1., 1., 0.5, 0.], &[(-1.0_f64).exp()]).unwrap();
    assert!((back.view().get(3, 0).unwrap() - 1.).abs() < 1e-3);
    assert_eq!(
        back.view().get(3, 0).unwrap(),
        solve_ivp(decay, 1., 0., &[(-1.0_f64).exp()])
            .unwrap()
            .as_slice()[0]
    );
    let oscillator = solve_ivp_at(
        |_: f64, y: &[f64], dy: &mut [f64]| {
            dy[0] = y[1];
            dy[1] = -y[0];
            Ok(())
        },
        &(0..=10).map(f64::from).collect::<Vec<_>>(),
        &[1., 0.],
    )
    .unwrap();
    assert!((oscillator.view().get(10, 0).unwrap() - 10.0_f64.cos()).abs() < 3e-3);
}
#[test]
fn real_precisions_and_complex_states() {
    let y = solve_ivp(
        |_: f32, y: &[f32], dy: &mut [f32]| {
            dy[0] = -y[0];
            Ok(())
        },
        0.,
        1.,
        &[1.],
    )
    .unwrap();
    assert!((y.as_slice()[0] - (-1.0_f32).exp()).abs() < 1e-3);
    let z = Complex::new(1.0_f64, 2.0);
    let y = solve_ivp(
        |_: f64, y: &[Complex<f64>], dy: &mut [Complex<f64>]| {
            dy[0] = -y[0];
            Ok(())
        },
        0.,
        1.,
        &[z],
    )
    .unwrap();
    assert!((y.as_slice()[0].re - (-1.0_f64).exp()).abs() < 1e-3);
    assert!((y.as_slice()[0].im - 2.0 * (-1.0_f64).exp()).abs() < 1e-3);
}
fn invalid(error: &SciOxideError, name: scioxide_core::Argument, expected: InvalidInputReason) {
    assert!(
        matches!(error.view(),SciOxideErrorView::InvalidInput{argument,reason,..} if argument==name && reason==expected),
        "{error}"
    );
}
#[test]
fn ivp_input_errors_name_one_shot_arguments() {
    invalid(
        solve_ivp(decay, 0., 1., &[]).unwrap_err().error(),
        argument!("y0"),
        InvalidInputReason::Empty,
    );
    invalid(
        solve_ivp(decay, 0., 0., &[1.]).unwrap_err().error(),
        argument!("t1"),
        InvalidInputReason::Inconsistent,
    );
    for times in [&[][..], &[0.][..]] {
        invalid(
            solve_ivp_at(decay, times, &[1.]).unwrap_err().error(),
            argument!("times"),
            InvalidInputReason::Empty,
        );
    }
    invalid(
        solve_ivp_at(decay, &[0., 0.], &[1.]).unwrap_err().error(),
        argument!("times"),
        InvalidInputReason::Inconsistent,
    );
    for times in [[0., 2., 1.], [0., -1., 1.], [1., 2., 0.]] {
        invalid(
            solve_ivp_at(decay, &times, &[1.]).unwrap_err().error(),
            argument!("times"),
            InvalidInputReason::OutOfRange,
        );
    }
    for times in [[f64::NAN, 1.], [0., f64::INFINITY]] {
        assert_eq!(
            solve_ivp_at(decay, &times, &[1.])
                .unwrap_err()
                .error()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    assert_eq!(
        solve_ivp(decay, 0., 1., &[f64::NAN])
            .unwrap_err()
            .error()
            .kind(),
        SciOxideErrorKind::NonFinite
    );
    let error = solve_ivp(
        |_: f64, _: &[f64], dy: &mut [f64]| {
            dy.fill(f64::NAN);
            Ok(())
        },
        0.,
        1.,
        &[1.],
    )
    .unwrap_err();
    assert!(matches!(
        error.error().view(),
        SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Callback,
            ..
        }
    ));
}
#[test]
fn unsuccessful_stops_carry_the_configured_progress_and_callback_errors() {
    let rhs = |_: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[0] * y[0];
        Ok(())
    };
    let system = OdeFn::new(1, rhs).unwrap();
    let configured = DormandPrince5::new()
        .solve(
            &OdeProblem::new(&system, 2.),
            &OdeOptions::default(),
            OdeState::from_slice(0., &[1.]).unwrap(),
        )
        .unwrap();
    assert_eq!(
        configured.report().termination(),
        Some(OdeTermination::StepSizeLimit)
    );
    for failure in [
        solve_ivp(rhs, 0., 2., &[1.]).unwrap_err(),
        solve_ivp_at(rhs, &[0., 2.], &[1.]).unwrap_err(),
    ] {
        assert!(matches!(
            failure.error().view(),
            SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::StepSizeLimit,
                ..
            }
        ));
        assert_eq!(
            failure.progress().unwrap().time(),
            configured.report().final_time()
        );
        assert_eq!(
            failure.progress().unwrap().state(),
            configured.state().values()
        );
    }
    let expected = SciOxideError::invalid_input(
        argument!("callback"),
        InvalidInputReason::Inconsistent,
        "callback sentinel",
    );
    let failure = solve_ivp(
        |_: f64, _: &[f64], _: &mut [f64]| Err(expected.clone()),
        0.,
        1.,
        &[1.],
    )
    .unwrap_err();
    assert_eq!(failure.error(), &expected);
}
#[test]
fn dde_defaults_match_configured_execution() {
    let rhs = |_: f64, _: &[f64], d: MatrixView<'_, f64>, dy: &mut [f64]| {
        dy[0] = d.get(0, 0)?;
        Ok(())
    };
    let history = |_: f64, y: &mut [f64]| {
        y.fill(1.);
        Ok(())
    };
    let value = solve_dde(rhs, history, &[1.], 0., 3., 1).unwrap();
    let options = DdeOptions::default();
    assert_eq!(options.history_capacity().get(), usize::MAX);
    assert_eq!(options.schedule_capacity().get(), usize::MAX);
    let solution = DdePlan::new(1, (0., 3.), &[1.], &[], options)
        .unwrap()
        .solve(&DdeProblem::new(&rhs, &history))
        .unwrap();
    assert_eq!(value.as_slice(), solution.state().values());
    assert!((value.as_slice()[0] - 37. / 6.).abs() < 1e-12);
    invalid(
        solve_dde(rhs, history, &[1.], 1., 0., 1)
            .unwrap_err()
            .error(),
        argument!("t1"),
        InvalidInputReason::OutOfRange,
    );
    assert!(solve_dde(rhs, history, &[], 0., 1., 1).is_err());
}
#[test]
fn sde_defaults_match_every_configured_step() {
    let drift = |_: f64, x: &[f64], a: &mut [f64]| {
        a[0] = 0.4 * x[0];
        Ok(())
    };
    let diffusion = |_: f64, x: &[f64], dw: &[f64], b: &mut [f64]| {
        b[0] = 0.7 * x[0] * dw[0];
        Ok(())
    };
    let times = [0., 0.25, 0.5, 0.75, 1.];
    let increments = MatrixView::from_slice(4, 1, &[0.1, -0.2, 0.3, -0.1]).unwrap();
    let path = solve_sde(drift, diffusion, &times, increments, &[1.]).unwrap();
    let system = SdeFn::new(1, 1, drift, diffusion);
    let problem = SdeProblem::new(&system, &times, increments);
    let options = SdeOptions::default();
    let mut state = OdeState::from_slice(0., &[1.]).unwrap();
    let mut ws = SdeMethod::EulerMaruyama
        .workspace(&problem, &options)
        .unwrap();
    let mut rows = vec![1.];
    let _ = SdeMethod::EulerMaruyama
        .advance(
            &problem,
            &options,
            &mut state,
            &mut |_, y: &[f64]| {
                rows.push(y[0]);
                Ok(())
            },
            &mut ws,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(path.as_slice(), rows);
    let explicit = problem.with_step_boundaries(&[0, 1, 2, 3, 4]);
    assert_eq!(
        SdeMethod::EulerMaruyama
            .solve(&explicit, &options, &[1.])
            .unwrap()
            .state()
            .values(),
        state.values()
    );
    assert!(solve_sde(drift, diffusion, &[], increments, &[1.]).is_err());
    assert!(
        solve_sde(
            drift,
            diffusion,
            &times,
            MatrixView::from_slice(0, 1, &[]).unwrap(),
            &[1.]
        )
        .is_err()
    );
}
#[test]
fn mechanical_defaults_match_configured_execution() {
    let velocity = |p: &[f64], v: &mut [f64]| {
        v.copy_from_slice(p);
        Ok(())
    };
    let gradient = |q: &[f64], g: &mut [f64]| {
        g.copy_from_slice(q);
        Ok(())
    };
    let value = solve_hamiltonian(velocity, gradient, &[1.], &[0.], 0.01, 100).unwrap();
    let system = HamiltonianFn::new(1, velocity, gradient);
    let configured = StormerVerlet
        .solve(
            &system,
            0.01,
            100,
            &MechanicalOptions::default(),
            PhaseSpace::new(vec![1.], vec![0.]).unwrap(),
        )
        .unwrap();
    assert_eq!(value, configured.state().clone());
    assert!((value.position()[0] - 1.0_f64.cos()).abs() < 1e-4);
    invalid(
        solve_hamiltonian(velocity, gradient, &[1.], &[0.], 0., 1)
            .unwrap_err()
            .error(),
        argument!("step"),
        InvalidInputReason::OutOfRange,
    );
    assert!(solve_hamiltonian(velocity, gradient, &[1.], &[], 0.01, 1).is_err());
}
#[test]
fn bvp_defaults_and_both_evaluation_forms() {
    let rhs = |_: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = -y[0];
        Ok(())
    };
    let boundary = |a: &[f64], b: &[f64], r: &mut [f64]| {
        r[0] = a[0];
        r[1] = b[0] - 1.;
        Ok(())
    };
    let guess = Matrix::zeros(3, 2).unwrap();
    let mesh = [0., 0.5, 1.];
    let solution = solve_bvp(rhs, boundary, &mesh, guess.view()).unwrap();
    let system = BvpFn::new(2, rhs, boundary);
    let options = BvpOptions::default();
    assert_eq!(options.max_nodes(), usize::MAX);
    assert_eq!(options.max_workspace_bytes(), usize::MAX);
    let configured = BvpPlan::new(2, 0, &mesh, options)
        .unwrap()
        .solve(&BvpProblem::numerical(&system), guess.view(), &[])
        .unwrap();
    let mut value = [0.; 2];
    let mut expected = [0.; 2];
    let mut derivative = [0.; 2];
    solution.evaluate_into(0.5, &mut value).unwrap();
    configured
        .evaluate_with_derivative_into(0.5, &mut expected, &mut derivative)
        .unwrap();
    assert_eq!(value, expected);
    assert!((value[0] - 0.5_f64.sin() / 1.0_f64.sin()).abs() < 1e-3);
    let large = solve_bvp(
        |_: f64, _: &[f64], dy: &mut [f64]| {
            dy.fill(0.);
            Ok(())
        },
        |a: &[f64], _: &[f64], r: &mut [f64]| {
            for (r, a) in r.iter_mut().zip(a) {
                *r = *a - 1.;
            }
            Ok(())
        },
        &[0., 1.],
        Matrix::zeros(2, 14).unwrap().view(),
    )
    .unwrap();
    assert!(large.report().unwrap().is_success());
    invalid(
        &solve_bvp(rhs, boundary, &mesh, Matrix::zeros(3, 0).unwrap().view()).unwrap_err(),
        argument!("guess"),
        InvalidInputReason::Empty,
    );
}

#[test]
fn other_one_shots_support_f32() {
    let path = solve_sde(
        |_: f32, _: &[f32], a: &mut [f32]| {
            a[0] = 1.0;
            Ok(())
        },
        |_: f32, _: &[f32], dw: &[f32], b: &mut [f32]| {
            b[0] = 0.5 * dw[0];
            Ok(())
        },
        &[0.0, 0.5, 1.0],
        MatrixView::from_slice(2, 1, &[0.25, -0.125]).unwrap(),
        &[1.0],
    )
    .unwrap();
    assert!((path.view().get(2, 0).unwrap() - 2.0625).abs() < f32::EPSILON);
    let state = solve_hamiltonian(
        |p: &[f32], v: &mut [f32]| {
            v.copy_from_slice(p);
            Ok(())
        },
        |q: &[f32], g: &mut [f32]| {
            g.copy_from_slice(q);
            Ok(())
        },
        &[1.0],
        &[0.0],
        0.01,
        100,
    )
    .unwrap();
    assert!((state.position()[0] - 1.0_f32.cos()).abs() < 1e-4);
    let delayed = solve_dde(
        |_: f32, _: &[f32], d: MatrixView<'_, f32>, dy: &mut [f32]| {
            dy[0] = d.get(0, 0)?;
            Ok(())
        },
        |_: f32, y: &mut [f32]| {
            y.fill(1.0);
            Ok(())
        },
        &[1.0],
        0.0,
        3.0,
        1,
    )
    .unwrap();
    assert!((delayed.as_slice()[0] - 37.0 / 6.0).abs() < 1e-4);
    let solution = solve_bvp(
        |_: f32, _: &[f32], dy: &mut [f32]| {
            dy[0] = 1.0;
            Ok(())
        },
        |a: &[f32], _: &[f32], r: &mut [f32]| {
            r[0] = a[0];
            Ok(())
        },
        &[0.0, 1.0],
        Matrix::zeros(2, 1).unwrap().view(),
    )
    .unwrap();
    let mut y = [0.0];
    solution.evaluate_into(0.5, &mut y).unwrap();
    assert!((y[0] - 0.5).abs() < 1e-5);
}

#[test]
fn complex_sde_mechanics_and_bvp_preserve_both_components() {
    type C = Complex<f64>;
    let path = solve_sde(
        |_: f64, _: &[C], a: &mut [C]| {
            a[0] = C::new(1.0, 2.0);
            Ok(())
        },
        |_: f64, _: &[C], dw: &[f64], b: &mut [C]| {
            b[0] = C::new(dw[0], -dw[0]);
            Ok(())
        },
        &[0.0, 1.0],
        MatrixView::from_slice(1, 1, &[0.25]).unwrap(),
        &[C::new(0.0, 0.0)],
    )
    .unwrap();
    let y = path.view().get(1, 0).unwrap();
    assert!((y.re - 1.25).abs() < 1e-14 && (y.im - 1.75).abs() < 1e-14);
    let state = solve_hamiltonian(
        |p: &[C], v: &mut [C]| {
            v.copy_from_slice(p);
            Ok(())
        },
        |q: &[C], g: &mut [C]| {
            g.copy_from_slice(q);
            Ok(())
        },
        &[C::new(1.0, 2.0)],
        &[C::new(0.0, 0.0)],
        0.01,
        100,
    )
    .unwrap();
    let q = state.position()[0];
    assert!((q.re - 1.0_f64.cos()).abs() < 1e-4 && (q.im - 2.0 * 1.0_f64.cos()).abs() < 1e-4);
    let solution = solve_bvp(
        |_: f64, _: &[C], dy: &mut [C]| {
            dy[0] = C::new(1.0, 2.0);
            Ok(())
        },
        |a: &[C], _: &[C], r: &mut [C]| {
            r[0] = a[0];
            Ok(())
        },
        &[0.0, 1.0],
        Matrix::zeros(2, 1).unwrap().view(),
    )
    .unwrap();
    let mut y = [C::new(0.0, 0.0)];
    solution.evaluate_into(0.5, &mut y).unwrap();
    assert!((y[0].re - 0.5).abs() < 1e-10 && (y[0].im - 1.0).abs() < 1e-10);
}
