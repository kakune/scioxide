//! Errors identify rejected arguments and checked resources.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Argument, Complex, DomainReason, ExecutionContext, InvalidInputReason, NumericalIssue,
    SciOxideError, SciOxideErrorView, argument,
};
use scioxide_linalg::Matrix;
use scioxide_signal::{
    DetrendKind, FirBand, FirSymmetry, IirDomain, IirOrderOptions, LtiTime, PolePlacementOptions,
    PolyphasePlan, RemezOptions, ResampleAlignment, SignalLayout, StateSpace, TransferFunction,
    Window, WindowKind, WindowNormalization, WindowSymmetry, design_fir_least_squares,
    design_fir_remez_with_options, detrend_into, select_iir_order_scoped,
    select_iir_order_with_options,
};

/// Argument of an invalid-input error; `None` for any other outcome.
fn invalid<T: core::fmt::Debug>(result: Result<T, SciOxideError>) -> Option<Argument> {
    reason(result).map(|(argument, _)| argument)
}

/// Argument and reason of an invalid-input error; `None` for any other outcome.
fn reason<T: core::fmt::Debug>(
    result: Result<T, SciOxideError>,
) -> Option<(Argument, InvalidInputReason)> {
    match result.unwrap_err().view() {
        SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => Some((argument, reason)),
        _ => None,
    }
}

/// Expected and actual extents of a shape error; `None` for any other outcome.
fn shapes<T: core::fmt::Debug>(
    result: Result<T, SciOxideError>,
) -> Option<(Vec<usize>, Vec<usize>)> {
    match result.unwrap_err().view() {
        SciOxideErrorView::DimensionMismatch { expected, actual } => {
            Some((expected.to_vec(), actual.to_vec()))
        }
        _ => None,
    }
}

/// Argument and reason of a domain error; `None` for any other outcome.
fn domain<T: core::fmt::Debug>(
    result: Result<T, SciOxideError>,
) -> Option<(Argument, DomainReason)> {
    match result.unwrap_err().view() {
        SciOxideErrorView::DomainError {
            argument, reason, ..
        } => Some((argument, reason)),
        _ => None,
    }
}

/// Issue of a numerical-instability error; `None` for any other outcome.
fn instability<T: core::fmt::Debug>(result: Result<T, SciOxideError>) -> Option<NumericalIssue> {
    match result.unwrap_err().view() {
        SciOxideErrorView::NumericalInstability { issue, .. } => Some(issue),
        _ => None,
    }
}

fn band(lower: f64, upper: f64) -> FirBand<f64> {
    FirBand::new(lower, upper, 1.0, 1.0)
}

#[test]
fn shared_parameter_checks_name_the_public_parameter() {
    let mut output = [0.0_f64; 4];
    assert_eq!(
        invalid(detrend_into(
            &[1.0_f64, 2.0, 3.0, 4.0],
            DetrendKind::Linear,
            &[3, 2],
            &mut output,
            scioxide_core::ExecutionContext::sequential_native(),
        )),
        Some(argument!("breakpoints"))
    );
    // `Window::new` only forwards, so the validating callee is reported.
    assert_eq!(
        invalid(Window::<f64>::new(
            WindowKind::Kaiser { beta: -1.0 },
            8,
            WindowSymmetry::Symmetric,
            WindowNormalization::Native,
        )),
        Some(argument!("kind"))
    );
}

#[test]
fn fir_designs_name_rejected_arguments() {
    assert_eq!(
        reason(design_fir_least_squares::<f64>(9, FirSymmetry::Even, &[])),
        Some((argument!("bands"), InvalidInputReason::Empty))
    );
    let options = RemezOptions::default().with_grid_density(8);
    assert_eq!(
        invalid(design_fir_remez_with_options(
            9,
            FirSymmetry::Even,
            &[band(0.0, 0.4)],
            options,
        )),
        Some(argument!("options.grid_density"))
    );
}

#[test]
fn lti_and_iir_errors_distinguish_parameters_and_numerical_failures() {
    let a = Matrix::from_vec(1, 1, vec![-1.0_f64]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
    let system =
        StateSpace::new(LtiTime::Continuous, a.view(), b.view(), b.view(), a.view()).unwrap();
    // One requested pole per state: a count disagreement is a shape error.
    assert_eq!(
        shapes(system.place_poles_with_options(
            &[Complex::new(-1.0, 0.0), Complex::new(-2.0, 0.0)],
            PolePlacementOptions::default(),
        )),
        Some((vec![1], vec![2]))
    );
    // Plain LTI range errors are out of range, contradictions inconsistent.
    assert_eq!(
        reason(system.place_poles_with_options(
            &[Complex::new(-1.0, 0.0)],
            PolePlacementOptions::default().with_residual_tolerance(-1.0),
        )),
        Some((
            argument!("options.residual_tolerance"),
            InvalidInputReason::OutOfRange
        ))
    );
    let digital = TransferFunction::new(IirDomain::Digital, &[1.0_f64], &[1.0, -0.5]).unwrap();
    assert_eq!(
        reason(digital.to_state_space(LtiTime::Continuous)),
        Some((argument!("time"), InvalidInputReason::Inconsistent))
    );
    let pole = TransferFunction::new(IirDomain::Analog, &[1.0_f64], &[1.0, 0.0]).unwrap();
    assert!(
        matches!(pole.response(0.0), Err(e) if matches!(e.view(), SciOxideErrorView::NumericalInstability { issue: scioxide_core::NumericalIssue::Breakdown, .. }))
    );
    let roots = scioxide_signal::PoleZeroGain::new(
        IirDomain::Analog,
        &[],
        &[Complex::new(0., 0.)],
        Complex::new(1., 0.),
    )
    .unwrap();
    assert!(
        matches!(roots.response(0.), Err(e) if matches!(e.view(), SciOxideErrorView::NumericalInstability { issue: scioxide_core::NumericalIssue::Breakdown, .. }))
    );
    let zero = TransferFunction::new(IirDomain::Digital, &[1., -1.], &[1.]).unwrap();
    assert!(
        matches!(zero.group_delay(0.), Err(e) if matches!(e.view(), SciOxideErrorView::NumericalInstability { issue: scioxide_core::NumericalIssue::Breakdown, .. }))
    );
    let sections =
        scioxide_signal::SecondOrderSections::new(IirDomain::Digital, &[[1., -1., 0., 1., 0., 0.]])
            .unwrap();
    assert!(
        matches!(sections.group_delay(0.), Err(e) if matches!(e.view(), SciOxideErrorView::NumericalInstability { issue: scioxide_core::NumericalIssue::Breakdown, .. }))
    );
}

#[test]
fn polyphase_errors_classify_samples_and_factors() {
    let plan = PolyphasePlan::rational(&[0.5_f64, 0.5], 2, 1, ResampleAlignment::Causal).unwrap();
    let input = [1.0, f64::NAN];
    let mut output = vec![0.0; plan.output_len(input.len()).unwrap()];
    let mut scratch = plan
        .workspace(SignalLayout::Contiguous(input.len()))
        .unwrap();
    let error = plan
        .resample_into(
            &input,
            &mut output,
            &mut scratch,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(error.view(), SciOxideErrorView::NonFinite { .. }));
    assert_eq!(
        invalid(PolyphasePlan::rational(
            &[1.0_f64],
            0,
            1,
            ResampleAlignment::Causal
        )),
        Some(argument!("up"))
    );
}

#[test]
fn dpss_convergence_keeps_work_and_residual_diagnostics() {
    use scioxide_core::{Budget, ConvergenceReason};
    use scioxide_signal::{DpssOptions, DpssTapers, DpssTermination};
    let options = DpssOptions::default();
    let plan = DpssTapers::<f64>::new(32, 3., 4, options).unwrap();
    assert_eq!(plan.report().termination(), DpssTermination::Completed);
    assert!(plan.report().counts().total().unwrap().iterations() > 0);
    assert!(plan.report().residual().scaled_norm() < 64. * f64::EPSILON);
    let eigen = options.eigen();
    let options = options.with_eigen(
        eigen.with_execution_options(eigen.execution_options().with_iteration_budget(Budget::ONE)),
    );
    let error = DpssTapers::<f64>::new(32, 3., 4, options).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ConvergenceFailure {
        reason: ConvergenceReason::IterationLimit, diagnostics, ..
    } if diagnostics.counts().total().unwrap().iterations() == 1)
    );
}

#[test]
fn capped_order_keeps_best_candidate_instead_of_rejecting_input() {
    use scioxide_signal::{IirBand, IirOrderFamily, IirOrderTermination};
    for family in [
        IirOrderFamily::Butterworth,
        IirOrderFamily::ChebyshevI,
        IirOrderFamily::ChebyshevII,
        IirOrderFamily::Elliptic,
    ] {
        let capped = select_iir_order_with_options(
            family,
            IirBand::Lowpass(0.2),
            IirBand::Lowpass(0.3),
            1.,
            80.,
            IirDomain::Digital,
            IirOrderOptions::new().with_maximum_order(2),
        )
        .unwrap();
        assert_eq!(capped.termination(), IirOrderTermination::OrderLimit);
        assert!(!capped.is_success());
        assert_eq!(capped.order(), 2);
        assert!(capped.order_deficit().raw_norm() > 0.);
        // The order is closed-form: no local iterations, whatever the cap.
        assert_eq!(capped.counts().own().iterations(), 0);
        assert_eq!(
            capped.trust(),
            scioxide_core::PartialResultTrust::DiagnosticOnly
        );
        let complete = select_iir_order_with_options(
            family,
            IirBand::Lowpass(0.2),
            IirBand::Lowpass(0.3),
            1.,
            80.,
            IirDomain::Digital,
            IirOrderOptions::new().with_maximum_order(64),
        )
        .unwrap();
        assert!(complete.is_success());
        assert!(complete.order() > capped.order());
        assert_eq!(complete.order_deficit().raw_norm(), 0.);
        if matches!(family, IirOrderFamily::Elliptic) {
            assert!(complete.counts().nested().iterations() > 0);
        }
    }
    assert_eq!(
        invalid(select_iir_order_with_options(
            IirOrderFamily::Butterworth,
            IirBand::Lowpass(0.2),
            IirBand::Lowpass(0.3),
            1.,
            80.,
            IirDomain::Digital,
            IirOrderOptions::new().with_maximum_order(0),
        )),
        Some(argument!("options.maximum_order"))
    );
}

#[test]
fn order_search_runs_in_a_child_epoch_with_nested_counts() {
    use scioxide_core::{
        Budget, BudgetKind, PartialResultTrust, ProblemToken, SolveBudgets, SolveIdentity,
        SolveState,
    };
    use scioxide_signal::{
        IirBand, IirOrderFamily, IirOrderReport, IirOrderTermination, select_iir_order,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(2196),
        scioxide_core::algorithm_id!("signal", "order_test", 1),
        0,
        0,
    );
    let root = select_iir_order_with_options(
        IirOrderFamily::Elliptic,
        IirBand::Lowpass(0.2_f64),
        IirBand::Lowpass(0.3),
        1.,
        80.,
        IirDomain::Digital,
        IirOrderOptions::new().with_maximum_order(64),
    )
    .unwrap();
    assert!(root.is_success());
    // The owned selection is the scoped one in a fresh native root scope.
    let bits = |report: IirOrderReport<f64>| {
        let edges = match report.band() {
            IirBand::Lowpass(edge) | IirBand::Highpass(edge) => [edge, edge],
            IirBand::Bandpass(lower, upper) | IirBand::Bandstop(lower, upper) => [lower, upper],
        };
        let deficit = report.order_deficit().raw_norm();
        let outcome = (report.order(), report.termination(), report.counts());
        (outcome, edges.map(f64::to_bits), deficit.to_bits())
    };
    let (pass, stop) = (IirBand::Lowpass(0.2_f64), IirBand::Lowpass(0.3));
    let family = IirOrderFamily::Elliptic;
    let one_shot = select_iir_order(family, pass, stop, 1., 80., IirDomain::Digital).unwrap();
    let native = ExecutionContext::sequential_native();
    let options = IirOrderOptions::default();
    let scoped = select_iir_order_scoped(
        family,
        pass,
        stop,
        1.,
        80.,
        IirDomain::Digital,
        options,
        native,
    )
    .unwrap();
    assert_eq!(bits(one_shot), bits(scoped));
    // Iteration allowances are scope-local: a parent's iteration limit does not
    // stop the child's elliptic integrals, whose work reaches the parent as
    // nested counts.
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential_native(),
            identity,
            &mut budgets,
        )
        .unwrap();
    let report = select_iir_order_scoped(
        IirOrderFamily::Elliptic,
        IirBand::Lowpass(0.2_f64),
        IirBand::Lowpass(0.3),
        1.,
        80.,
        IirDomain::Digital,
        IirOrderOptions::new().with_maximum_order(64),
        &mut context,
    )
    .unwrap();
    assert_eq!(report.termination(), IirOrderTermination::Selected);
    assert_eq!(report.order(), root.order());
    assert_eq!(report.trust(), PartialResultTrust::ResidualChecked);
    assert_eq!(report.counts().own().iterations(), 0);
    let iterations = report.counts().nested().iterations();
    assert!(iterations > 1);
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), iterations);

    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential_native().with_cancellation(&token),
            identity,
            &mut budgets,
        )
        .unwrap();
    let options = IirOrderOptions::new().with_maximum_order(64);
    assert!(
        matches!(select_iir_order_scoped(IirOrderFamily::Butterworth, IirBand::Lowpass(0.2_f64),
        IirBand::Lowpass(0.3), 1., 80., IirDomain::Digital, options, &mut context),
        Err(e) if matches!(e.view(), SciOxideErrorView::ExecutionStopped { reason: scioxide_core::ExecutionStopReason::Cancelled }))
    );
    assert!(
        matches!(select_iir_order_scoped(IirOrderFamily::Butterworth, IirBand::Lowpass(0.2_f64),
        IirBand::Lowpass(0.3), 1., 80., IirDomain::Digital, options,
        ExecutionContext::sequential_native().with_cancellation(&token)),
        Err(e) if matches!(e.view(), SciOxideErrorView::ExecutionStopped { reason: scioxide_core::ExecutionStopReason::Cancelled }))
    );
}

#[test]
fn order_selection_is_closed_form_and_independent_of_the_cap() {
    use scioxide_signal::{IirBand, IirOrderFamily, IirOrderTermination};
    let select = |stop: f64, maximum_order: usize| {
        select_iir_order_with_options(
            IirOrderFamily::Butterworth,
            IirBand::Lowpass(0.2_f64),
            IirBand::Lowpass(stop),
            0.1,
            100.,
            IirDomain::Digital,
            IirOrderOptions::new().with_maximum_order(maximum_order),
        )
        .unwrap()
    };
    // SciPy `buttord(0.2, 0.200001, 0.1, 100)` gives 2_505_756 from the
    // continuous requirement 2_505_755.9029; an incremental search took that
    // many steps.
    let selected = select(0.200001, usize::MAX);
    assert_eq!(selected.termination(), IirOrderTermination::Selected);
    assert_eq!(selected.order(), 2_505_756);
    assert_eq!(selected.counts().own().iterations(), 0);
    let capped = select(0.200001, 1_000_000);
    assert_eq!(capped.termination(), IirOrderTermination::OrderLimit);
    assert_eq!(capped.order(), 1_000_000);
    assert!((capped.order_deficit().raw_norm() - 1_505_755.902_9).abs() < 1e-3);
    // An exactly sufficient cap selects; one below it reports the limit.
    let order = select(0.3, 64).order();
    assert!(select(0.3, order).is_success());
    assert_eq!(
        select(0.3, order - 1).termination(),
        IirOrderTermination::OrderLimit
    );
}

#[test]
fn missed_pole_tolerance_retains_gain_spectrum_and_conditioning() {
    use scioxide_signal::PolePlacementTermination;
    let a = Matrix::from_vec(3, 3, vec![0.1_f64, 1., 0., 0., 0.2, 1., 0., 0., 0.3]).unwrap();
    let b = Matrix::from_vec(3, 1, vec![0.1, 0.2, 1.]).unwrap();
    let c = Matrix::identity(3).unwrap();
    let d = Matrix::zeros(3, 1).unwrap();
    let system =
        StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view()).unwrap();
    let requested = [
        Complex::new(-0.7, 0.),
        Complex::new(-1.3, 0.),
        Complex::new(-2.9, 0.),
    ];
    let report = system
        .place_poles_with_options(
            &requested,
            PolePlacementOptions::new()
                .with_rank_policy(scioxide_linalg::RankPolicy::DimensionScaledEpsilon)
                .with_residual_tolerance(f64::MIN_POSITIVE),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        PolePlacementTermination::ResidualToleranceNotMet
    );
    assert!(!report.is_success());
    assert_eq!(report.achieved_poles().len(), 3);
    assert_eq!(report.gain().as_slice().len(), 3);
    assert!(report.maximum_residual() > f64::MIN_POSITIVE && report.maximum_residual() < 1e-10);
    assert!(report.transfer_condition().is_some());
    assert!(report.counts().own().iterations() > 0);
    assert!(report.counts().nested().iterations() > 0);
    assert_eq!(
        report.trust(),
        scioxide_core::PartialResultTrust::DiagnosticOnly
    );

    let b = Matrix::from_vec(3, 3, vec![1., 0., 0., 0., 0.1, 0., 0., 0., 0.01]).unwrap();
    let d = Matrix::zeros(3, 3).unwrap();
    let full = StateSpace::new(LtiTime::Continuous, a.view(), b.view(), c.view(), d.view())
        .unwrap()
        .place_poles_with_options(&requested, PolePlacementOptions::default())
        .unwrap();
    assert!(full.is_success());
    assert!(full.transfer_condition().is_none());
    assert!((full.input_condition().reciprocal().value() - 0.01).abs() < 1e-14);
}

/// Whether a result is the `NonFinite` error of a caller-supplied value.
fn non_finite_input<T>(result: Result<T, SciOxideError>) -> bool {
    result.err().is_some_and(|error| {
        matches!(
            error.view(),
            SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        )
    })
}

#[test]
fn non_finite_parameters_report_input_origin() {
    use scioxide_signal::{
        DiscretizationMethod, RidgeOptions, RidgePlan, SignalExecutionOptions, StftOptions,
        StftPlan, SymmetricFilter, WhittakerHendersonPlan,
    };
    let options = SignalExecutionOptions::new();
    for (pole, gain) in [(f64::NAN, 1.0), (0.5, f64::INFINITY)] {
        assert!(non_finite_input(SymmetricFilter::first_order(
            pole, gain, options
        )));
    }
    assert!(non_finite_input(SymmetricFilter::second_order(
        0.5,
        f64::NAN,
        1.0,
        options
    )));
    assert!(non_finite_input(WhittakerHendersonPlan::<f64>::new(
        8,
        2,
        f64::NAN,
        None,
        scioxide_signal::SignalExecutionOptions::new()
    )));
    assert!(non_finite_input(RidgePlan::new(
        &[1.0_f64, 2.0, 4.0],
        RidgeOptions::default().with_min_strength(f64::NAN)
    )));
    assert!(non_finite_input(StftPlan::new(
        &[1.0_f64; 4],
        2,
        StftOptions::new()
            .with_fft_len(Some(4))
            .with_sample_rate(f64::INFINITY)
    )));
    let one = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
    let system = StateSpace::new(
        LtiTime::Continuous,
        one.view(),
        one.view(),
        one.view(),
        one.view(),
    )
    .unwrap();
    assert!(non_finite_input(
        system.discretize(f64::NAN, DiscretizationMethod::ZeroOrderHold)
    ));
    assert!(non_finite_input(system.discretize(
        0.1,
        DiscretizationMethod::GeneralizedBilinear { alpha: f64::NAN }
    )));
    assert!(non_finite_input(system.place_poles_with_options(
        &[Complex::new(f64::NAN, 0.0)],
        PolePlacementOptions::new(),
    )));
    assert!(non_finite_input(StateSpace::new(
        LtiTime::Discrete {
            sample_period: f64::INFINITY
        },
        one.view(),
        one.view(),
        one.view(),
        one.view(),
    )));
    // Finite out-of-range values remain invalid input.
    assert_eq!(
        reason(system.discretize(-1.0, DiscretizationMethod::ZeroOrderHold)),
        Some((argument!("sample_period"), InvalidInputReason::OutOfRange))
    );
}

#[test]
fn positional_lengths_and_resolved_fft_lengths_name_their_parameters() {
    use InvalidInputReason::{Empty, Inconsistent, OutOfRange, UnsupportedConfiguration};
    use scioxide_signal::{
        SavitzkyGolayOptions, SavitzkyGolayPlan, SpectralEstimateOptions, SpectralEstimatePlan,
        StftOptions, StftPlan,
    };
    let window = [1.0_f64; 4];
    let short = StftOptions::new().with_fft_len(Some(3));
    assert_eq!(
        reason(StftPlan::new(&window, 0, StftOptions::new())),
        Some((argument!("hop"), OutOfRange))
    );
    assert_eq!(
        reason(StftPlan::from_dual(&window, 0, StftOptions::new())),
        Some((argument!("hop"), OutOfRange))
    );
    assert_eq!(
        reason(StftPlan::new(&window, 2, short)),
        Some((argument!("options.fft_len"), Inconsistent))
    );
    assert_eq!(
        reason(StftPlan::new(
            &window,
            2,
            short.with_fft_len(Some(usize::MAX))
        )),
        Some((argument!("options.fft_len"), OutOfRange))
    );
    assert_eq!(
        reason(StftPlan::new(
            &window,
            2,
            StftOptions::new().with_sample_rate(0.0)
        )),
        Some((argument!("options.sample_rate"), OutOfRange))
    );
    let spectral = SpectralEstimateOptions::new().with_fft_len(Some(3));
    assert_eq!(
        reason(SpectralEstimatePlan::<f64>::new(&window, spectral)),
        Some((argument!("options.fft_len"), Inconsistent))
    );
    let options = SavitzkyGolayOptions::<f64>::new();
    assert_eq!(
        reason(SavitzkyGolayPlan::new(0, 0, options)),
        Some((argument!("window_length"), Empty))
    );
    assert_eq!(
        reason(SavitzkyGolayPlan::new(5, 5, options)),
        Some((argument!("polynomial_degree"), OutOfRange))
    );
    assert_eq!(
        reason(SavitzkyGolayPlan::new(4, 2, options)),
        Some((argument!("window_length"), UnsupportedConfiguration))
    );
}

#[test]
fn non_finite_iir_parameters_report_input_origin() {
    use scioxide_signal::{
        IirBand, IirFamily, IirOrderFamily, ResonanceKind, design_comb, design_gammatone,
        design_iir, design_notch_peak, design_real_gammatone, iir_prototype,
    };
    let design =
        |family: IirFamily<f64>, band: IirBand<f64>, domain| design_iir(4, family, band, domain);
    for band in [
        IirBand::Lowpass(f64::NAN),
        IirBand::Highpass(f64::INFINITY),
        IirBand::Bandpass(0.1, f64::INFINITY),
        IirBand::Bandstop(f64::NAN, 0.4),
    ] {
        for domain in [IirDomain::Digital, IirDomain::Analog] {
            assert!(non_finite_input(design(
                IirFamily::Butterworth,
                band,
                domain
            )));
        }
    }
    for family in [
        IirFamily::ChebyshevI { ripple: f64::NAN },
        IirFamily::ChebyshevII {
            attenuation: f64::INFINITY,
        },
        IirFamily::Elliptic {
            ripple: 1.0,
            attenuation: f64::INFINITY,
        },
        IirFamily::Elliptic {
            ripple: f64::NAN,
            attenuation: 40.0,
        },
    ] {
        assert!(non_finite_input(design(
            family,
            IirBand::Lowpass(0.3),
            IirDomain::Digital
        )));
        assert!(non_finite_input(iir_prototype(4, family)));
    }
    let order = |pass: f64, ripple: f64, attenuation: f64| {
        select_iir_order_with_options(
            IirOrderFamily::Butterworth,
            IirBand::Lowpass(pass),
            IirBand::Lowpass(0.3),
            ripple,
            attenuation,
            IirDomain::Digital,
            IirOrderOptions::new().with_maximum_order(20),
        )
    };
    assert!(non_finite_input(order(f64::NAN, 1.0, 40.0)));
    assert!(non_finite_input(order(0.2, f64::NAN, 40.0)));
    assert!(non_finite_input(order(0.2, 1.0, f64::INFINITY)));
    let prototype = iir_prototype(3, IirFamily::<f64>::Butterworth).unwrap();
    assert!(non_finite_input(prototype.bilinear(f64::NAN)));
    assert!(non_finite_input(
        prototype.transform(IirBand::Highpass(f64::INFINITY))
    ));
    assert!(non_finite_input(prototype.response(f64::NAN)));
    assert!(non_finite_input(design_notch_peak(
        f64::NAN,
        30.0,
        ResonanceKind::Notch
    )));
    assert!(non_finite_input(design_notch_peak(
        0.3,
        f64::INFINITY,
        ResonanceKind::Peak
    )));
    assert!(non_finite_input(design_comb(
        4,
        f64::NAN,
        ResonanceKind::Notch
    )));
    assert!(non_finite_input(design_gammatone(4, f64::NAN, 0.05)));
    assert!(non_finite_input(design_gammatone(4, 0.25, f64::INFINITY)));
    assert!(non_finite_input(design_real_gammatone(2, 0.25, f64::NAN)));
    let transfer = TransferFunction::new(IirDomain::Digital, &[1.0_f64], &[1.0, -0.5]).unwrap();
    assert!(non_finite_input(transfer.response(f64::INFINITY)));
    assert!(non_finite_input(transfer.group_delay(f64::NAN)));
    // Finite out-of-range parameters remain invalid input naming the argument.
    assert_eq!(
        invalid(design(
            IirFamily::Butterworth,
            IirBand::Lowpass(-0.3),
            IirDomain::Digital
        )),
        Some(argument!("band"))
    );
    assert_eq!(
        invalid(prototype.bilinear(-1.0)),
        Some(argument!("sample_rate"))
    );
}

#[test]
fn prepared_plans_report_input_and_output_shape_disagreement() {
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_signal::{
        ConvolutionNdPlan, ConvolutionOptions, MinimumPhaseOptions, MinimumPhasePlan,
        SignalExecutionOptions, WaveletFamily, WaveletOptions, WaveletPlan, WhittakerHendersonPlan,
    };
    let cx = ExecutionContext::sequential_native;
    let wavelet =
        WaveletPlan::<f64>::new(WaveletFamily::Haar, 4, 1, WaveletOptions::default()).unwrap();
    let mut ws = wavelet.workspace(2).unwrap();
    let mut output = [7.0; 4];
    assert_eq!(
        shapes(wavelet.analyze_into(&[1.0; 3], &mut output, &mut ws, cx())),
        Some((vec![4], vec![3]))
    );
    assert_eq!(
        shapes(wavelet.analyze_into(&[1.0; 4], &mut output[..3], &mut ws, cx())),
        Some((vec![4], vec![3]))
    );
    assert_eq!(output, [7.0; 4]);
    // An axis form reports the whole shape with the record length on its axis.
    let mut samples = [1.0; 10];
    let mut coefficients = [7.0; 8];
    let out = StridedLayout::new(&[2, 4], &[4, 1]).unwrap();
    assert_eq!(
        shapes(wavelet.analyze_axis_into(
            NdView::new(&samples, StridedLayout::new(&[2, 5], &[5, 1]).unwrap()).unwrap(),
            1,
            NdViewMut::new(&mut coefficients, out).unwrap(),
            &mut ws,
            cx(),
        )),
        Some((vec![2, 4], vec![2, 5]))
    );
    assert_eq!(
        reason(wavelet.synthesize_axis_into(
            NdView::new(&coefficients, out).unwrap(),
            2,
            NdViewMut::new(&mut samples[..8], out).unwrap(),
            &mut ws,
            cx(),
        )),
        Some((argument!("axis"), InvalidInputReason::OutOfRange))
    );
    assert_eq!(coefficients, [7.0; 8]);

    let kernel = [1.0_f64, 2.0, 1.0];
    let plan = ConvolutionNdPlan::new(
        &[8],
        NdView::new(&kernel, StridedLayout::new(&[3], &[1]).unwrap()).unwrap(),
        &[0],
        ConvolutionOptions::new(),
    )
    .unwrap();
    assert_eq!(plan.output_shape(), [10]);
    let mut ws = plan
        .workspace(scioxide_signal::ConvolutionKernel::Prepared)
        .unwrap();
    let input = [1.0; 8];
    let mut output = [7.0; 10];
    assert_eq!(
        shapes(plan.convolve_into(
            NdView::new(&input[..7], StridedLayout::new(&[7], &[1]).unwrap()).unwrap(),
            NdViewMut::new(&mut output, StridedLayout::new(&[10], &[1]).unwrap()).unwrap(),
            &mut ws,
            cx(),
        )),
        Some((vec![8], vec![7]))
    );
    assert_eq!(
        shapes(plan.convolve_into(
            NdView::new(&input, StridedLayout::new(&[8], &[1]).unwrap()).unwrap(),
            NdViewMut::new(&mut output[..9], StridedLayout::new(&[9], &[1]).unwrap()).unwrap(),
            &mut ws,
            cx(),
        )),
        Some((vec![10], vec![9]))
    );
    assert_eq!(output, [7.0; 10]);

    let smoother =
        WhittakerHendersonPlan::<f64>::new(8, 2, 1.0, None, SignalExecutionOptions::new()).unwrap();
    let mut ws = smoother.workspace(1).unwrap();
    let mut output = [7.0; 8];
    assert_eq!(
        shapes(smoother.smooth_into(&[1.0; 7], &mut output, &mut ws, cx())),
        Some((vec![8], vec![7]))
    );
    assert_eq!(
        shapes(smoother.smooth_into(&[1.0; 8], &mut output[..7], &mut ws, cx())),
        Some((vec![8], vec![7]))
    );
    assert_eq!(output, [7.0; 8]);
    assert_eq!(
        shapes(WhittakerHendersonPlan::<f64>::new(
            8,
            2,
            1.0,
            Some(&[1.0; 7]),
            SignalExecutionOptions::new()
        )),
        Some((vec![8], vec![7]))
    );

    let minimum = MinimumPhasePlan::<f64>::new(3, MinimumPhaseOptions::new()).unwrap();
    let mut ws = minimum.workspace(()).unwrap();
    let mut output = vec![7.0; minimum.output_len()];
    assert_eq!(
        shapes(minimum.reconstruct_into(&[1.0, 2.0], &mut output, &mut ws, cx())),
        Some((vec![3], vec![2]))
    );
    assert!(output.iter().all(|&v| v == 7.0));
}

#[test]
fn empty_collections_and_contradictions_have_distinct_reasons() {
    use scioxide_signal::{
        LocalFilterPlan, MinimumPhaseOptions, MinimumPhasePlan, SignalExecutionOptions,
        window_metrics,
    };
    assert_eq!(
        reason(window_metrics::<f64>(&[])),
        Some((argument!("samples"), InvalidInputReason::Empty))
    );
    assert_eq!(
        reason(TransferFunction::new(IirDomain::Digital, &[], &[1.0_f64])),
        Some((argument!("b"), InvalidInputReason::Empty))
    );
    // Split checks: an empty collection is `Empty`, a disordered one out of range.
    assert_eq!(
        reason(LocalFilterPlan::<f64>::order(&[], 0)),
        Some((argument!("offsets"), InvalidInputReason::Empty))
    );
    assert_eq!(
        reason(LocalFilterPlan::<f64>::order(&[1, 0], 0)),
        Some((argument!("offsets"), InvalidInputReason::OutOfRange))
    );
    assert_eq!(
        reason(scioxide_signal::AnalyticSignalNdPlan::<f64>::new(
            &[3, 4],
            &[],
            SignalExecutionOptions::new()
        )),
        Some((argument!("axes"), InvalidInputReason::Empty))
    );
    assert_eq!(
        reason(MinimumPhasePlan::<f64>::new(
            3,
            MinimumPhaseOptions::new().with_fft_length(Some(2))
        )),
        Some((
            argument!("options.fft_length"),
            InvalidInputReason::Inconsistent
        ))
    );
    assert_eq!(
        reason(scioxide_signal::select_iir_order_with_options(
            scioxide_signal::IirOrderFamily::Butterworth,
            scioxide_signal::IirBand::Lowpass(0.2_f64),
            scioxide_signal::IirBand::Lowpass(0.3),
            3.0,
            3.0,
            IirDomain::Digital,
            IirOrderOptions::new().with_maximum_order(8),
        )),
        Some((argument!("attenuation"), InvalidInputReason::Inconsistent))
    );
}

#[test]
fn unsupported_option_combinations_are_unsupported_configurations() {
    use scioxide_core::{NdView, StridedLayout};
    use scioxide_signal::{
        ConvolutionExtent, ConvolutionNdPlan, ConvolutionOptions, SpectralAverage,
        SpectralEstimateOptions, SpectralEstimatePlan,
    };
    let median = SpectralEstimatePlan::<f64>::new(
        &[0.5, 1.0, 0.5, 0.25],
        SpectralEstimateOptions::new()
            .with_fft_len(Some(4))
            .with_sample_rate(1.0)
            .with_average(SpectralAverage::Median),
    )
    .unwrap();
    let mut ws = median
        .workspace(scioxide_signal::SignalLayout::Contiguous(8))
        .unwrap();
    let mut output = vec![7.0; median.bins()];
    assert_eq!(
        reason(median.coherence_into(
            &[1.0; 8],
            &[1.0; 8],
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native(),
        )),
        Some((
            argument!("self"),
            InvalidInputReason::UnsupportedConfiguration
        ))
    );
    assert!(output.iter().all(|&v| v == 7.0));
    let kernel = [1.0_f64, 1.0];
    assert_eq!(
        reason(ConvolutionNdPlan::new(
            &[4],
            NdView::new(&kernel, StridedLayout::new(&[2], &[1]).unwrap()).unwrap(),
            &[0],
            ConvolutionOptions::new().with_extent(ConvolutionExtent::Circular),
        )),
        Some((
            argument!("options.boundary"),
            InvalidInputReason::UnsupportedConfiguration
        ))
    );
}

#[test]
fn domain_and_numerical_failures_are_not_invalid_input() {
    use scioxide_signal::{ChirpLaw, Waveform, waveform_into};
    // f(t) = f0 / (1 - t/2) has its pole at t = 2.
    let chirp = Waveform::Chirp {
        initial: 1.0_f64,
        final_frequency: 2.0,
        duration: 1.0,
        phase: 0.0,
        law: ChirpLaw::Hyperbolic,
    };
    let mut output = [7.0; 3];
    assert_eq!(
        domain(waveform_into(
            chirp,
            &[0.0, 1.0, 2.5],
            &mut output,
            ExecutionContext::sequential_native(),
        )),
        Some((argument!("waveform"), DomainReason::Pole))
    );
    assert_eq!(output, [7.0; 3]);
    assert_eq!(
        domain(chirp.sample(3.0)),
        Some((argument!("self"), DomainReason::Pole))
    );
    assert!(chirp.sample(1.5).unwrap().is_finite());
    assert_eq!(
        domain(scioxide_signal::window_metrics(&[1.0_f64, -1.0])),
        Some((argument!("samples"), DomainReason::Undefined))
    );

    // A finite discrete state that doubles past the range overflows.
    let two = Matrix::from_vec(1, 1, vec![2.0_f64]).unwrap();
    let one = Matrix::from_vec(1, 1, vec![1.0]).unwrap();
    let zero = Matrix::from_vec(1, 1, vec![0.0]).unwrap();
    let system = StateSpace::new(
        LtiTime::Discrete { sample_period: 1.0 },
        two.view(),
        one.view(),
        one.view(),
        zero.view(),
    )
    .unwrap();
    assert_eq!(
        instability(system.simulate_zoh(&[0.0, 1.0, 2.0], &[0.0; 3], &[f64::MAX])),
        Some(NumericalIssue::Overflow)
    );
}
