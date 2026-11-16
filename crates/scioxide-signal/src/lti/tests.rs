use super::*;
use crate::util::error_kind;
use scioxide_core::{AlgebraicScalar, SciOxideErrorKind};

fn first_order(time: LtiTime<f64>) -> StateSpace<f64> {
    let a = Matrix::from_vec(1, 1, vec![-1.0]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
    let c = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
    let d = Matrix::from_vec(1, 1, vec![0.0]).unwrap();
    StateSpace::new(time, a.view(), b.view(), c.view(), d.view()).unwrap()
}

#[test]
fn equivalent_realizations_and_analytic_responses() {
    let system = first_order(LtiTime::Continuous);
    let transfer = system.to_transfer(0, 0).unwrap();
    assert!((transfer.denominator()[1] - 1.0).abs() < 1e-12);
    let rebuilt = transfer.to_state_space(LtiTime::Continuous).unwrap();
    let response = rebuilt.frequency_response(&[0.0, 1.0]).unwrap();
    assert!((response.values()[0].re - 1.0).abs() < 1e-12);
    assert!((response.values()[1].norm() - 0.5_f64.sqrt()).abs() < 1e-12);
    let times = [0.0, 0.25, 0.5, 0.75, 1.0];
    let impulse = system.impulse_response(&times).unwrap();
    let step = system.step_response(&times).unwrap();
    for (index, &time) in times.iter().enumerate() {
        assert!((impulse.values()[index] - (-time).exp()).abs() < 2e-12);
        assert!((step.values()[index] - (1.0 - (-time).exp())).abs() < 2e-12);
    }
}

#[test]
fn discretization_and_sampled_simulation_match_closed_form() {
    let system = first_order(LtiTime::Continuous);
    let discrete = system
        .discretize(0.1, DiscretizationMethod::ZeroOrderHold)
        .unwrap();
    assert!((discrete.a().as_slice()[0] - (-0.1_f64).exp()).abs() < 2e-12);
    assert!((discrete.b().as_slice()[0] - (1.0 - (-0.1_f64).exp())).abs() < 2e-12);
    let times = [0.0, 0.1, 0.2, 0.3];
    let response = system.simulate_zoh(&times, &[1.0; 4], &[0.0]).unwrap();
    for (index, &time) in times.iter().enumerate() {
        assert!((response.values()[index] - (1.0 - (-time).exp())).abs() < 2e-12);
    }
    let tustin = system
        .discretize(
            0.1,
            DiscretizationMethod::GeneralizedBilinear { alpha: 0.5 },
        )
        .unwrap();
    assert!((tustin.a().as_slice()[0] - 0.95 / 1.05).abs() < 1e-12);
}

#[test]
fn mimo_shapes_rank_and_pole_placement_are_checked() {
    let a = Matrix::from_vec(2, 2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
    let b = Matrix::from_vec(2, 1, vec![0.0, 1.0]).unwrap();
    let c = Matrix::identity(2).unwrap();
    let d = Matrix::zeros(2, 1).unwrap();
    let system =
        StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap();
    assert!(
        system
            .controllability(RankPolicy::DimensionScaledEpsilon)
            .unwrap()
            .is_full_rank()
    );
    assert!(
        system
            .observability(RankPolicy::DimensionScaledEpsilon)
            .unwrap()
            .is_full_rank()
    );
    let frequency = system.frequency_response(&[0.5]).unwrap();
    assert_eq!(frequency.values().len(), 2);
    let placed = system
        .place_poles_with_options(
            &[Complex::new(-2.0, 0.0), Complex::new(-1.0, 0.0)],
            PolePlacementOptions::default(),
        )
        .unwrap();
    assert!(placed.maximum_residual() < 1e-10);
    assert!((placed.gain().as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((placed.gain().as_slice()[1] - 3.0).abs() < 1e-10);
    let conjugate = system
        .place_poles_with_options(
            &[Complex::new(-1.0, -1.0), Complex::new(-1.0, 1.0)],
            PolePlacementOptions::default(),
        )
        .unwrap();
    assert!(conjugate.maximum_residual() < 1e-10);
    assert!((conjugate.gain().as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((conjugate.gain().as_slice()[1] - 2.0).abs() < 1e-10);
}

#[test]
fn complex_and_full_input_repeated_poles_are_supported() {
    let a = Matrix::<f64>::zeros(2, 2).unwrap();
    let b = Matrix::identity(2).unwrap();
    let c = Matrix::identity(2).unwrap();
    let d = Matrix::zeros(2, 2).unwrap();
    let system =
        StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap();
    let placed = system
        .place_poles_with_options(
            &[Complex::new(-1.0, 0.0), Complex::new(-1.0, 0.0)],
            PolePlacementOptions::default(),
        )
        .unwrap();
    assert!(placed.maximum_residual() < 1e-12);

    let tf = TransferFunction::new(
        IirDomain::Analog,
        &[Complex::new(1.0, 1.0)],
        &[Complex::new(1.0, 0.0), Complex::new(1.0, -1.0)],
    )
    .unwrap();
    let complex = tf.to_state_space(LtiTime::Continuous).unwrap();
    let response = complex.frequency_response(&[0.25]).unwrap();
    assert!(response.values()[0].has_finite_components());
}

#[test]
fn invalid_shapes_domains_and_times_are_rejected() {
    let a = Matrix::<f64>::identity(2).unwrap();
    let bad = Matrix::<f64>::zeros(1, 1).unwrap();
    assert_eq!(
        error_kind(StateSpace::new(
            LtiTime::Continuous,
            a.view(),
            bad.view(),
            bad.view(),
            bad.view()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    let transfer = TransferFunction::new(IirDomain::Digital, &[1.0], &[1.0, -0.5]).unwrap();
    assert_eq!(
        error_kind(transfer.to_state_space(LtiTime::Continuous)),
        SciOxideErrorKind::InvalidInput
    );
    let system = first_order(LtiTime::Continuous);
    assert_eq!(
        error_kind(system.step_response(&[0.0, 0.2, 0.3])),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(system.discretize(-1.0, DiscretizationMethod::ZeroOrderHold)),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn digital_transfer_delay_round_trips_through_state_space() {
    let transfer = TransferFunction::new(IirDomain::Digital, &[0.0_f64, 1.0], &[1.0]).unwrap();
    let system = transfer
        .to_state_space(LtiTime::Discrete {
            sample_period: 0.25,
        })
        .unwrap();
    assert_eq!(system.states(), 1);
    let rebuilt = system.to_transfer(0, 0).unwrap();
    assert_eq!(rebuilt.numerator(), &[0.0, 1.0]);
    assert_eq!(rebuilt.denominator(), &[1.0]);
    // Discrete frequencies are multiples of Nyquist: z = exp(i*pi*f).
    let response = system
        .frequency_response(&[0.5 / core::f64::consts::PI])
        .unwrap();
    let expected = Complex::from_polar(1.0, -0.5);
    assert!((response.values()[0] - expected).norm() < 1e-12);
    // The sample period does not enter; f = 1 is Nyquist, z = -1.
    let nyquist = system.frequency_response(&[1.0]).unwrap();
    assert!((nyquist.values()[0] + Complex::new(1.0, 0.0)).norm() < 1e-12);
}

#[test]
fn both_precisions_and_complex32_follow_the_same_low_order_contract() {
    let a = Matrix::from_vec(1, 1, vec![-1.0_f32]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1.0_f32]).unwrap();
    let c = Matrix::from_vec(1, 1, vec![1.0_f32]).unwrap();
    let d = Matrix::zeros(1, 1).unwrap();
    let real =
        StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap();
    let zoh = real
        .discretize(0.125, DiscretizationMethod::ZeroOrderHold)
        .unwrap();
    assert!((zoh.a().as_slice()[0] - (-0.125_f32).exp()).abs() < 2e-6);

    let ac = Matrix::from_vec(1, 1, vec![Complex::new(-1.0_f32, 0.25)]).unwrap();
    let bc = Matrix::from_vec(1, 1, vec![Complex::ONE]).unwrap();
    let cc = bc.clone();
    let dc = Matrix::zeros(1, 1).unwrap();
    let complex = StateSpace::new(
        LtiTime::Continuous,
        ac.view(),
        bc.view(),
        cc.view(),
        dc.view(),
    )
    .unwrap();
    let response = complex.frequency_response(&[0.5]).unwrap();
    let expected = Complex::<f32>::ONE / Complex::new(1.0_f32, 0.25_f32);
    assert!((response.values()[0] - expected).norm() < 2e-5);
}

#[test]
fn borrowed_and_into_forms_preserve_outputs_on_failure() {
    let system = first_order(LtiTime::Continuous);
    let owned = system.view().to_owned().unwrap();
    assert_eq!(owned.a(), system.a());
    let mut frequency = [Complex::new(9.0, 9.0)];
    let plan = FrequencyResponsePlan::new(&owned).unwrap();
    let mut ws = plan.workspace(1).unwrap();
    plan.frequency_response_into(
        &[1.0],
        &mut frequency,
        &mut ws,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!((frequency[0].norm() - 0.5_f64.sqrt()).abs() < 1e-12);

    let mut output = [7.0; 3];
    let mut states = [8.0; 3];
    assert_eq!(
        error_kind(TimeResponsePlan::new(&owned, &[0.0, 0.2, 0.3])),
        SciOxideErrorKind::InvalidInput
    );
    let plan = TimeResponsePlan::new(&owned, &[0.0, 0.1, 0.2]).unwrap();
    let mut ws = plan.workspace().unwrap();
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    assert_eq!(
        error_kind(plan.step_response_into(
            &mut output,
            &mut states,
            &mut ws,
            ExecutionContext::sequential().with_cancellation(&token)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(output, [7.0; 3]);
    assert_eq!(states, [8.0; 3]);
}

#[test]
fn rank_deficient_multi_input_system_places_allowed_repeated_poles() {
    // This controllable pair has no controllable single-input direction:
    // the repeated zero eigenspace requires both input columns.
    let a = Matrix::from_vec(3, 3, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let b = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
    let c = Matrix::identity(3).unwrap();
    let d = Matrix::zeros(3, 2).unwrap();
    let system =
        StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap();
    let report = system
        .place_poles_with_options(
            &[
                Complex::new(-2.0, 0.0),
                Complex::new(-1.0, 0.0),
                Complex::new(-1.0, 0.0),
            ],
            PolePlacementOptions::new()
                .with_rank_policy(RankPolicy::DimensionScaledEpsilon)
                .with_residual_tolerance(1e-9),
        )
        .unwrap();
    assert!(report.maximum_residual() < 1e-9);
    assert_eq!(report.gain().rows(), 2);
    assert_eq!(report.gain().cols(), 3);
}

#[test]
fn pole_placement_validates_the_tolerance_when_placing() {
    let system = first_order(LtiTime::Continuous);
    for (tolerance, kind) in [
        (0.0, scioxide_core::SciOxideErrorKind::InvalidInput),
        (-1.0, scioxide_core::SciOxideErrorKind::InvalidInput),
        (f64::NAN, scioxide_core::SciOxideErrorKind::NonFinite),
        (f64::INFINITY, scioxide_core::SciOxideErrorKind::NonFinite),
    ] {
        let options = PolePlacementOptions::new().with_residual_tolerance(tolerance);
        assert_eq!(options.residual_tolerance().to_bits(), tolerance.to_bits());
        let error = system
            .place_poles_with_options(&[Complex::new(-2.0, 0.0)], options)
            .unwrap_err();
        assert_eq!(error.kind(), kind);
    }
    assert_eq!(
        PolePlacementOptions::<f64>::default(),
        PolePlacementOptions::new()
    );
}

#[test]
fn simulation_plan_writes_directly_and_preserves_outputs_on_failure() {
    let system = first_order(LtiTime::Continuous);
    let times = [0.0, 0.1, 0.2, 0.3];
    let plan = TimeResponsePlan::new(&system, &times).unwrap();
    assert_eq!(plan.simulation_input_len(), 4);
    assert_eq!(plan.simulation_output_len(), 4);
    assert_eq!(plan.simulation_states_len(), 4);
    let mut ws = plan.workspace().unwrap();
    let mut output = [7.0; 4];
    let mut states = [8.0; 4];
    plan.simulate_into(
        &[1.0; 4],
        &[0.5],
        &mut output,
        &mut states,
        &mut ws,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let owned = system.simulate_zoh(&times, &[1.0; 4], &[0.5]).unwrap();
    assert_eq!(owned.values(), &output);
    assert_eq!(owned.states(), &states);
    for (index, &time) in times.iter().enumerate() {
        let exact = 1.0 - 0.5 * (-time).exp();
        assert!((output[index] - exact).abs() < 2e-12);
    }
    let mut output = [7.0; 4];
    let mut states = [8.0; 4];
    for (inputs, initial, kind) in [
        (
            &[1.0, 1.0, f64::NAN, 1.0][..],
            &[0.0][..],
            SciOxideErrorKind::NonFinite,
        ),
        (
            &[1.0; 3][..],
            &[0.0][..],
            SciOxideErrorKind::DimensionMismatch,
        ),
        (&[1.0; 4][..], &[][..], SciOxideErrorKind::DimensionMismatch),
    ] {
        assert_eq!(
            error_kind(plan.simulate_into(
                inputs,
                initial,
                &mut output,
                &mut states,
                &mut ws,
                ExecutionContext::sequential()
            )),
            kind
        );
    }
    assert_eq!(
        error_kind(plan.simulate_into(
            &[1.0; 4],
            &[0.0],
            &mut output[..3],
            &mut states,
            &mut ws,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    assert_eq!(
        error_kind(plan.simulate_into(
            &[1.0; 4],
            &[0.0],
            &mut output,
            &mut states,
            &mut ws,
            ExecutionContext::sequential().with_cancellation(&token)
        )),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(output, [7.0; 4]);
    assert_eq!(states, [8.0; 4]);
}
