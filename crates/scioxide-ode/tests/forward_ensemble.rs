//! Forward tangents and the independent-member ensemble runner.
#![allow(clippy::unwrap_used, clippy::panic)]
use core::cell::Cell;
use core::num::NonZeroUsize;
use scioxide_core::*;
use scioxide_linalg::{MatrixView, MatrixViewMut, Vector};
use scioxide_ode::*;

/// `y' = p y` with tangent `p dy + dp y`; `scratch` asks for that much provider scratch.
struct Growth {
    scratch: usize,
    calls: Cell<usize>,
}
impl Growth {
    const fn new(scratch: usize) -> Self {
        Self {
            scratch,
            calls: Cell::new(0),
        }
    }
}
impl<S: RealScalar, T: NumericScalar<Real = S>> OdeSensitivitySystem<S, T> for Growth {
    fn dimension(&self) -> usize {
        1
    }
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn rhs(
        &self,
        _: S,
        y: &[T],
        p: &[S],
        o: &mut [T],
        scratch: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), self.scratch);
        self.calls.set(self.calls.get() + 1);
        o[0] = T::from_real(p[0]) * y[0];
        Ok(())
    }
    fn tangent_rhs(
        &self,
        _: S,
        y: &[T],
        p: &[S],
        dy: &[T],
        dp: &[S],
        o: &mut [T],
        scratch: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), self.scratch);
        o[0] = T::from_real(p[0]) * dy[0] + T::from_real(dp[0]) * y[0];
        Ok(())
    }
}

fn tangent<T: NumericScalar>() {
    let f = |v| T::Real::try_from_f64_lossy(v).unwrap();

    let provider = Growth::new(0);
    let p = [f(0.5)];
    let dp = [f(1.), f(0.)];
    let mut workspace: ForwardSensitivityWorkspace<T::Real, T> = ForwardSensitivityWorkspace::new();
    let system = scioxide_ode::ForwardSensitivitySystem::bind(
        &provider,
        &p,
        MatrixView::from_slice(2, 1, &dp).unwrap(),
        &mut workspace,
    )
    .unwrap();
    let initial = T::from_components(f(2.), f(1.));
    let tangents = [T::ZERO, T::ONE];
    let mut augmented = vec![T::ZERO; 3];
    system
        .initialize_into(
            &[initial],
            MatrixView::from_slice(1, 2, &tangents).unwrap(),
            &mut augmented,
        )
        .unwrap();
    let tol = T::Real::SQRT_EPSILON * f(0.1);
    let options = OdeOptions::new().with_error_tolerance(
        system
            .error_tolerance(tol, tol, f(0.), NormSelection::Infinity)
            .unwrap(),
    );
    let ode = (DormandPrince5::<T::Real, T>::new(), options);
    let solution = ode
        .0
        .solve(
            &OdeProblem::new(&system, f(1.)),
            &ode.1,
            OdeState::new(f(0.), Vector::from_vec(augmented)),
        )
        .unwrap();
    let report = solution.report();
    assert!(report.is_success());
    let mut y = [T::ZERO];
    let mut sensitivities = [T::ZERO; 2];
    system
        .extract_into(
            solution.state().values(),
            &mut y,
            MatrixViewMut::from_slice(1, 2, &mut sensitivities).unwrap(),
        )
        .unwrap();
    let exponential = T::from_real(f(0.5).exp());
    for (actual, expected) in [
        (y[0], initial * exponential),
        (sensitivities[0], initial * exponential),
        (sensitivities[1], exponential),
    ] {
        assert!((actual - expected).abs() < f(100.) * tol);
    }
    assert!(report.counts().provider(EvaluationRole::LinearOperator) > 0);
}
#[test]
fn sensitivity_plans_report_their_dimensions() {
    let provider = Shape(3, 2);
    let mut ws = ForwardSensitivityWorkspace::new();
    let system = scioxide_ode::ForwardSensitivitySystem::bind(
        &provider,
        &[0.; 2],
        MatrixView::from_slice(4, 2, &[0.; 8]).unwrap(),
        &mut ws,
    )
    .unwrap();
    assert_eq!(system.augmented_dimension(), 15);
    assert!(ForwardSensitivityRequirements::new(&Shape(0, 2), 4).is_err());
    assert!(ForwardSensitivityRequirements::new(&provider, 0).is_err());
    assert!(ForwardSensitivityRequirements::new(&Shape(usize::MAX, 0), 1).is_err());
}
#[test]
fn f32_tangents() {
    tangent::<f32>();
}
#[test]
fn f64_tangents() {
    tangent::<f64>();
}
#[test]
fn complex32_tangents() {
    tangent::<Complex<f32>>();
}
#[test]
fn complex64_tangents() {
    tangent::<Complex<f64>>();
}

#[test]
fn binding_follows_the_requirements_protocol_and_reuses_the_workspace() {
    let provider = Growth::new(3);
    let requirements = ForwardSensitivityRequirements::new::<f64, f64>(&provider, 2).unwrap();
    assert_eq!(
        (
            requirements.packed_directions(),
            requirements.provider_scratch()
        ),
        (2, 3)
    );
    let mut workspace = ForwardSensitivityWorkspace::<f64>::new();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    // A padded `[k,p]` view packs like the contiguous one.
    let directions = [1., 0.];
    let padded = [1., 9., 0.];
    let strided = MatrixView::from_slice_with_strides(2, 1, &padded, 2, 1).unwrap();
    let evaluate = |system: &ForwardSensitivitySystem<'_, f64>| {
        let mut out = [0.; 3];
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = SolveIdentity::new(
            ProblemToken::new(1),
            scioxide_core::algorithm_id!("test", "tangent", 1),
            0,
            0,
        );
        let mut cx = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        system.rhs(0., &[2., 1., 0.5], &mut out, &mut cx).unwrap();
        assert_eq!(
            cx.counts().own().provider(EvaluationRole::LinearOperator),
            2
        );
        out
    };
    let grown = evaluate(
        &scioxide_ode::ForwardSensitivitySystem::bind(&provider, &[0.5], strided, &mut workspace)
            .unwrap(),
    );
    assert_eq!(grown, [1., 2.5, 0.25]);
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let rows = MatrixView::from_slice(2, 1, &directions).unwrap();
    let bound = (&provider, &mut workspace);
    let system = {
        let (provider, workspace) = bound;
        scioxide_ode::ForwardSensitivitySystem::bind(provider, &[0.5], rows, workspace).unwrap()
    };
    assert_eq!(evaluate(&system), grown);
    // Shape and finiteness are validated before anything is bound.
    assert!(
        scioxide_ode::ForwardSensitivitySystem::bind(&provider, &[], rows, &mut workspace).is_err()
    );
    assert!(
        scioxide_ode::ForwardSensitivitySystem::bind(&provider, &[f64::NAN], rows, &mut workspace)
            .is_err()
    );
    let wide = MatrixView::from_slice(1, 2, &directions[..2]).unwrap();
    assert!(
        scioxide_ode::ForwardSensitivitySystem::bind(&provider, &[0.5], wide, &mut workspace)
            .is_err()
    );
}

/// The non-finite origin of `result`'s error.
fn origin<T>(result: Result<T, SciOxideError>) -> NonFiniteOrigin {
    let Err(error) = result else {
        panic!("expected a non-finite error");
    };
    match error.view() {
        SciOxideErrorView::NonFinite { origin, .. } => origin,
        view => panic!("expected a non-finite error, found {view:?}"),
    }
}

#[test]
fn nonfinite_caller_data_is_input_and_provider_output_is_callback() {
    let layout_provider = Shape(1, 1);
    let mut layout_workspace = ForwardSensitivityWorkspace::new();
    let layout_parameters = vec![0.0; 1];
    let layout_directions = vec![0.0; 1];
    let plan = scioxide_ode::ForwardSensitivitySystem::bind(
        &layout_provider,
        &layout_parameters,
        MatrixView::from_slice(1, 1, &layout_directions).unwrap(),
        &mut layout_workspace,
    )
    .unwrap();
    let provider = Growth::new(0);
    let mut workspace = ForwardSensitivityWorkspace::new();
    let one = MatrixView::from_slice(1, 1, &[1.]).unwrap();
    let nan = MatrixView::from_slice(1, 1, &[f64::NAN]).unwrap();
    assert_eq!(
        origin(scioxide_ode::ForwardSensitivitySystem::bind(
            &provider,
            &[f64::NAN],
            one,
            &mut workspace
        )),
        NonFiniteOrigin::Input
    );
    assert_eq!(
        origin(scioxide_ode::ForwardSensitivitySystem::bind(
            &provider,
            &[0.5],
            nan,
            &mut workspace
        )),
        NonFiniteOrigin::Input
    );
    let mut augmented = [0.; 2];
    assert_eq!(
        origin(plan.initialize_into(&[f64::NAN], one, &mut augmented)),
        NonFiniteOrigin::Input
    );
    assert_eq!(
        origin(plan.initialize_into(&[1.], nan, &mut augmented)),
        NonFiniteOrigin::Input
    );
    let mut state = [0.];
    let mut tangent = [0.];
    assert_eq!(
        origin(plan.extract_into(
            &[1., f64::INFINITY],
            &mut state,
            MatrixViewMut::from_slice(1, 1, &mut tangent).unwrap()
        )),
        NonFiniteOrigin::Input
    );
    let event = |fixed: f64, slope: f64, numerator: f64, event_slope: f64| {
        let fixed = [fixed];
        let mut times = [0.];
        let mut states = [0.];
        plan.event_sensitivities_into(
            EventSensitivityInput {
                fixed_time: MatrixView::from_slice(1, 1, &fixed).unwrap(),
                state_slope: &[slope],
                numerators: &[numerator],
                event_slope,
                minimum_slope: 1e-12,
                isolation: EventIsolation::Isolated,
            },
            &mut times,
            MatrixViewMut::from_slice(1, 1, &mut states).unwrap(),
        )
    };
    assert!(event(0.5, 4., 0.5, 4.).is_ok());
    for (fixed, slope, numerator, event_slope) in [
        (f64::NAN, 4., 0.5, 4.),
        (0.5, f64::NAN, 0.5, 4.),
        (0.5, 4., f64::INFINITY, 4.),
        (0.5, 4., 0.5, f64::NAN),
    ] {
        assert_eq!(
            origin(event(fixed, slope, numerator, event_slope)),
            NonFiniteOrigin::Input
        );
    }
    // A provider that overflows returns a callback-origin error.
    let system =
        scioxide_ode::ForwardSensitivitySystem::bind(&provider, &[2.], one, &mut workspace)
            .unwrap();
    let mut solve = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "tangent-origin", 1),
        0,
        0,
    );
    let mut cx = solve
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut out = [0.; 2];
    assert_eq!(
        origin(system.rhs(0., &[f64::MAX, 1.], &mut out, &mut cx)),
        NonFiniteOrigin::Callback
    );
}

#[test]
fn event_time_and_terminal_state_derivatives() {
    let provider = Shape(1, 1);
    let mut workspace = ForwardSensitivityWorkspace::new();
    let system = scioxide_ode::ForwardSensitivitySystem::bind(
        &provider,
        &[0.],
        MatrixView::from_slice(1, 1, &[0.]).unwrap(),
        &mut workspace,
    )
    .unwrap();

    // y'=p, y(0)=0, terminal y=2: tau=2/p, d_tau/dp=-2/p²,
    // terminal y is constant even though fixed-time dy/dp=tau.
    let fixed = [0.5];
    let mut times = [7.];
    let mut states = [8.];
    let status = system
        .event_sensitivities_into(
            EventSensitivityInput {
                fixed_time: MatrixView::from_slice(1, 1, &fixed).unwrap(),
                state_slope: &[4.],
                numerators: &fixed,
                event_slope: 4.,
                minimum_slope: 1e-12,
                isolation: EventIsolation::Isolated,
            },
            &mut times,
            MatrixViewMut::from_slice(1, 1, &mut states).unwrap(),
        )
        .unwrap();
    assert_eq!(status, EventSensitivityStatus::Available);
    assert_eq!(times, [-0.125]);
    assert_eq!(states, [0.]);
    let tiny = 2.0_f64.powi(-800);
    let large = 2.0_f64.powi(800);
    system
        .event_sensitivities_into(
            EventSensitivityInput {
                fixed_time: MatrixView::from_slice(1, 1, &[0.0]).unwrap(),
                state_slope: &[large],
                numerators: &[tiny],
                event_slope: large,
                minimum_slope: 0.0,
                isolation: EventIsolation::Isolated,
            },
            &mut times,
            MatrixViewMut::from_slice(1, 1, &mut states).unwrap(),
        )
        .unwrap();
    assert_eq!(times, [-0.0]);
    assert_eq!(states, [-tiny]);
    let saved = (times, states);
    assert_eq!(
        system
            .event_sensitivities_into(
                EventSensitivityInput {
                    fixed_time: MatrixView::from_slice(1, 1, &fixed).unwrap(),
                    state_slope: &[0.],
                    numerators: &fixed,
                    event_slope: 0.,
                    minimum_slope: 1e-12,
                    isolation: EventIsolation::Isolated
                },
                &mut times,
                MatrixViewMut::from_slice(1, 1, &mut states).unwrap()
            )
            .unwrap(),
        EventSensitivityStatus::NonTransverse
    );
    assert_eq!((times, states), saved);
    for isolation in [EventIsolation::Isolated, EventIsolation::Ambiguous] {
        let result = system.event_sensitivities_into(
            EventSensitivityInput {
                fixed_time: MatrixView::from_slice(1, 1, &[f64::NAN]).unwrap(),
                state_slope: &[0.0],
                numerators: &[0.0],
                event_slope: 0.0,
                minimum_slope: 0.0,
                isolation,
            },
            &mut times,
            MatrixViewMut::from_slice(1, 1, &mut states).unwrap(),
        );
        assert!(matches!(
            result.unwrap_err().view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
        assert_eq!((times, states), saved);
    }
}

#[test]
fn event_state_corrections_retain_range_for_all_scalars() {
    fn check<T: NumericScalar>() {
        let provider = Growth::new(0);
        let mut workspace = ForwardSensitivityWorkspace::<T::Real, T>::new();
        let parameters = [T::Real::ZERO];
        let system = ForwardSensitivitySystem::bind(
            &provider,
            &parameters,
            MatrixView::from_slice(1, 1, &[T::Real::ONE]).unwrap(),
            &mut workspace,
        )
        .unwrap();
        let mut times = [T::Real::ONE];
        let mut states = [T::ONE];
        let tiny = T::Real::MIN_POSITIVE;
        let large = T::Real::ONE / tiny;
        let slope = T::from_components(large, if T::IS_REAL { T::Real::ZERO } else { -large });
        let result = system
            .event_sensitivities_into(
                EventSensitivityInput {
                    fixed_time: MatrixView::from_slice(1, 1, &[T::ZERO]).unwrap(),
                    state_slope: &[slope],
                    numerators: &[tiny],
                    event_slope: large,
                    minimum_slope: T::Real::ZERO,
                    isolation: EventIsolation::Isolated,
                },
                &mut times,
                MatrixViewMut::from_slice(1, 1, &mut states).unwrap(),
            )
            .unwrap();
        assert_eq!(result, EventSensitivityStatus::Available);
        assert_eq!(times[0], T::Real::ZERO);
        assert_eq!(states[0].real_component(), -tiny);
        assert_eq!(
            states[0].imaginary_component(),
            if T::IS_REAL { T::Real::ZERO } else { tiny }
        );
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}

#[test]
fn strided_tangent_outputs_preserve_padding_and_fail_transactionally() {
    let layout_provider = Shape(2, 0);
    let mut layout_workspace = ForwardSensitivityWorkspace::new();
    let layout_parameters = vec![0.0; 0];
    let layout_directions = vec![];
    let plan = scioxide_ode::ForwardSensitivitySystem::bind(
        &layout_provider,
        &layout_parameters,
        MatrixView::from_slice(2, 0, &layout_directions).unwrap(),
        &mut layout_workspace,
    )
    .unwrap();
    let augmented = [1., 2., 3., 4., 5., 6.];
    let mut state = [9.; 2];
    let mut backing = [7.; 9];
    plan.extract_into(
        &augmented,
        &mut state,
        MatrixViewMut::from_slice_with_strides(2, 2, &mut backing, 5, 2).unwrap(),
    )
    .unwrap();
    assert_eq!(state, [1., 2.]);
    assert_eq!(backing, [3., 7., 5., 7., 7., 4., 7., 6., 7.]);
    let before = (state, backing);
    let invalid = [1., 2., 3., f64::NAN, 5., 6.];
    assert!(
        plan.extract_into(
            &invalid,
            &mut state,
            MatrixViewMut::from_slice_with_strides(2, 2, &mut backing, 5, 2).unwrap()
        )
        .is_err()
    );
    assert_eq!((state, backing), before);
}

type Method = DormandPrince5<f64>;
type Outcome = Option<Result<OdeSolveReport<f64>, SciOxideError>>;

/// One member: caller-owned state and workspace, plus observations of its run.
struct Member {
    state: OdeState<f64>,
    workspace: OdeWorkspace<f64, Method>,
    /// Address of the parameter row the member received.
    parameters: Option<usize>,
    thread: Option<std::thread::ThreadId>,
}
impl Member {
    fn new() -> Self {
        Self {
            state: OdeState::new(0., Vector::from_vec(vec![1.])),
            workspace: OdeWorkspace::new(),
            parameters: None,
            thread: None,
        }
    }
}

fn ode() -> (Method, OdeOptions<f64>) {
    let tolerance = ErrorTolerance::scalar(1e-10, 1e-8, NormSelection::Infinity).unwrap();
    (
        DormandPrince5::new(),
        OdeOptions::new()
            .with_initial_step(0.01)
            .with_error_tolerance(tolerance),
    )
}

/// Integrate `y' = p[0] y` over `[0, 0.2]`; member `failing` fails in its first call.
fn member_run(
    ode: &(Method, OdeOptions<f64>),
    failing: Option<usize>,
) -> impl Fn(
    usize,
    &[f64],
    &mut Member,
    &mut SolveContext<'_, '_>,
) -> Result<OdeSolveReport<f64>, SciOxideError>
+ Sync
+ '_ {
    move |index, p, member, cx| {
        member.parameters = Some(p.as_ptr().addr());
        member.thread = Some(std::thread::current().id());
        let rate = p[0];
        let rhs = OdeFn::new(1, move |_: f64, y: &[f64], dy: &mut [f64]| {
            if failing == Some(index) {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("dy"),
                    InvalidInputReason::Inconsistent,
                    "deliberate member failure",
                ));
            }
            dy[0] = rate * y[0];
            Ok(())
        })?;
        ode.0.advance(
            &OdeProblem::new(&rhs, 0.2),
            &ode.1,
            &mut member.state,
            &mut (),
            &mut member.workspace,
            cx,
        )
    }
}

struct Run {
    states: Vec<u64>,
    outcomes: Vec<Option<Result<OdeSolveReport<f64>, SciOxideErrorKind>>>,
    report: EnsembleReport,
    parent: SolveCounts,
    threads: usize,
}

/// Run four members, member 1 failing, under `policy` with an optional shared
/// evaluation budget of the enclosing solve.
fn ensemble(policy: ExecutionPolicy, evaluations: Option<usize>) -> Run {
    let ode = ode();
    let rates = [-1., -2., -3., -4.];
    let parameters = EnsembleParameters::Independent(MatrixView::from_slice(4, 1, &rates).unwrap());
    let plan = policy;
    let mut workspace = {
        let mut workspace = scioxide_ode::EnsembleWorkspace::new();
        workspace
            .try_reserve(&scioxide_ode::EnsembleRequirements::new(4, parameters).unwrap())
            .unwrap();
        workspace
    };
    let mut members: Vec<Member> = (0..4).map(|_| Member::new()).collect();
    let mut outcomes: Vec<Outcome> = (0..4).map(|_| None).collect();
    #[cfg(feature = "std")]
    let executor = ThreadExecutor::new(NonZeroUsize::new(3).unwrap());
    #[cfg(feature = "std")]
    let execution = ExecutionContext::new(&executor);
    #[cfg(not(feature = "std"))]
    let execution = ExecutionContext::new(&SequentialExecutor);
    let mut state = SolveState::new();
    let mut budgets = match evaluations {
        Some(limit) => SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap()),
        None => SolveBudgets::unbounded(),
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(622),
        scioxide_core::algorithm_id!("test", "ensemble-parent", 1),
        0,
        0,
    );
    let mut parent = state
        .begin_solve(execution, identity, &mut budgets)
        .unwrap();
    let report = scioxide_ode::solve_ensemble_into(
        parameters,
        &member_run(&ode, Some(1)),
        plan,
        &mut members,
        &mut outcomes,
        &mut workspace,
        &mut parent,
    )
    .unwrap();
    let successful = if evaluations.is_some() { 0 } else { 3 };
    assert_eq!((report.completed(), report.successful()), (4, successful));
    assert_eq!(
        outcomes[1].as_ref().unwrap().as_ref().unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
    assert_eq!(attempt.progress(), Some(&report));
    let failed = members[1].workspace.last_attempt().unwrap();
    assert_eq!(failed.outcome(), AttemptOutcome::Failed);
    let mut threads = Vec::new();
    for thread in members.iter().map(|m| m.thread.unwrap()) {
        if !threads.contains(&thread) {
            threads.push(thread);
        }
    }
    Run {
        states: members
            .iter()
            .map(|m| m.state.values()[0].to_bits())
            .collect(),
        outcomes: outcomes
            .into_iter()
            .map(|outcome| outcome.map(|result| result.map_err(|error| error.kind())))
            .collect(),
        report,
        parent: parent.counts(),
        threads: threads.len(),
    }
}

#[test]
fn parallel_and_sequential_ensembles_are_bit_identical() {
    for evaluations in [None, Some(80)] {
        let serial = ensemble(ExecutionPolicy::Sequential, evaluations);
        let parallel = ensemble(
            ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap()),
            evaluations,
        );
        assert_eq!(serial.states, parallel.states);
        assert_eq!(serial.outcomes, parallel.outcomes);
        assert_eq!(serial.report, parallel.report);
        assert!(!serial.report.is_success());
        assert_eq!(serial.parent, parallel.parent);
        assert_eq!(
            serial.parent.total().unwrap(),
            serial.report.accounting().total().unwrap()
        );
        assert_eq!(serial.threads, 1);
        #[cfg(feature = "std")]
        assert!(parallel.threads > 1, "the parallel policy used one thread");
        for i in [0, 2, 3] {
            let report = serial.outcomes[i].unwrap().unwrap();
            let y = f64::from_bits(serial.states[i]);
            match evaluations {
                None => {
                    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
                    assert!((y - (-((i + 1) as f64) * 0.2).exp()).abs() < 1e-8);
                }
                // Each member receives a deterministic quarter of the shared budget.
                Some(limit) => {
                    assert_eq!(
                        report.termination(),
                        Some(OdeTermination::BudgetLimit(BudgetKind::Evaluation))
                    );
                    assert!(report.final_time() < 0.2);
                    assert!(
                        report
                            .accounting()
                            .total()
                            .unwrap()
                            .total_providers()
                            .unwrap()
                            <= limit / 4
                    );
                }
            }
        }
    }
}

#[test]
fn members_borrow_shared_and_contiguous_rows_and_stage_strided_rows() {
    let ode = ode();
    let plan = scioxide_core::ExecutionPolicy::Sequential;
    let shared = [-1., 7.];
    let rows = [-1., 1., -2., 2., -3., 3.];
    let columns = [-1., -2., -3., 1., 2., 3.];
    let cases = [
        (EnsembleParameters::Shared(&shared), 0),
        (
            EnsembleParameters::Independent(MatrixView::from_slice(3, 2, &rows).unwrap()),
            0,
        ),
        (
            EnsembleParameters::Independent(
                MatrixView::from_slice_with_strides(3, 2, &columns, 1, 3).unwrap(),
            ),
            6,
        ),
    ];
    let mut workspace = EnsembleWorkspace::new();
    for (case, (parameters, staged)) in cases.into_iter().enumerate() {
        let requirements = scioxide_ode::EnsembleRequirements::new(3, parameters).unwrap();
        assert_eq!(
            (requirements.members(), requirements.staged_parameters()),
            (3, staged)
        );
        let mut members: Vec<Member> = (0..3).map(|_| Member::new()).collect();
        let mut outcomes: Vec<Outcome> = (0..3).map(|_| None).collect();
        let report = scioxide_ode::solve_ensemble_into(
            parameters,
            &member_run(&ode, None),
            plan,
            &mut members,
            &mut outcomes,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(report.successful(), 3);
        assert!(report.is_success());
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::Sufficient
        );
        for (i, member) in members.iter().enumerate() {
            let address = member.parameters.unwrap();
            match case {
                0 => assert_eq!(address, shared.as_ptr().addr()),
                1 => assert_eq!(address, rows[2 * i..].as_ptr().addr()),
                _ => {
                    let source = columns.as_ptr_range();
                    assert!(!(source.start.addr()..source.end.addr()).contains(&address));
                }
            }
            let rate = if case == 0 { -1. } else { -((i + 1) as f64) };
            let exact = (rate * 0.2).exp();
            assert!((member.state.values()[0] - exact).abs() < 1e-8);
        }
    }
}

#[test]
fn preflight_rejections_leave_members_and_outcomes_unchanged() {
    assert_eq!(
        scioxide_ode::EnsembleRequirements::new(0, EnsembleParameters::Shared(&[0.0; 1]))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        scioxide_core::ExecutionPolicy::Sequential,
        ExecutionPolicy::Sequential
    );
    let ode = ode();
    let plan = scioxide_core::ExecutionPolicy::Sequential;
    let good = [-1., -2.];
    let nonfinite = [-1., f64::NAN];
    let short = [-1.];
    let mut workspace = EnsembleWorkspace::new();
    let cases = [
        (
            EnsembleParameters::Independent(MatrixView::from_slice(2, 1, &good).unwrap()),
            1,
            SciOxideErrorKind::DimensionMismatch,
        ),
        (
            EnsembleParameters::Independent(MatrixView::from_slice(1, 1, &short).unwrap()),
            2,
            SciOxideErrorKind::DimensionMismatch,
        ),
        (
            EnsembleParameters::Shared(&good),
            0,
            SciOxideErrorKind::InvalidInput,
        ),
        (
            EnsembleParameters::Independent(MatrixView::from_slice(2, 1, &nonfinite).unwrap()),
            2,
            SciOxideErrorKind::NonFinite,
        ),
    ];
    for (parameters, count, kind) in cases {
        let mut members: Vec<Member> = (0..count).map(|_| Member::new()).collect();
        let mut outcomes: Vec<Outcome> = (0..count)
            .map(|_| {
                Some(Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("outcome"),
                    InvalidInputReason::Inconsistent,
                    "sentinel",
                )))
            })
            .collect();
        let error = scioxide_ode::solve_ensemble_into(
            parameters,
            &member_run(&ode, None),
            plan,
            &mut members,
            &mut outcomes,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert!(members.iter().all(|m| m.thread.is_none()));
        assert!(outcomes.iter().all(|o| matches!(o, Some(Err(_)))));
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
        assert!(attempt.progress().is_none());
    }
}

#[test]
fn prepared_sequential_ensemble_allocates_nothing() {
    let ode = ode();
    let plan = scioxide_core::ExecutionPolicy::Sequential;
    let rates = [-1., -2., -3.];
    let parameters = EnsembleParameters::Independent(MatrixView::from_slice(3, 1, &rates).unwrap());
    let mut workspace = {
        let mut workspace = scioxide_ode::EnsembleWorkspace::new();
        workspace
            .try_reserve(&scioxide_ode::EnsembleRequirements::new(3, parameters).unwrap())
            .unwrap();
        workspace
    };
    let probe = OdeFn::new(1, |_: f64, _: &[f64], _: &mut [f64]| Ok(())).unwrap();
    let mut members: Vec<Member> = (0..3)
        .map(|_| {
            let mut member = Member::new();
            member.workspace = ode
                .0
                .workspace(&OdeProblem::new(&probe, 0.2), &ode.1)
                .unwrap();
            member
        })
        .collect();
    let mut outcomes: Vec<Outcome> = (0..3).map(|_| None).collect();
    let run = member_run(&ode, None);
    // The member closure records its thread; initialize the handle outside measurement.
    let _ = std::thread::current();
    for _ in 0..2 {
        for member in &mut members {
            member.state.reset(0., &[1.]).unwrap();
        }
        let report = scioxide_ode::solve_ensemble_into(
            parameters,
            &run,
            plan,
            &mut members,
            &mut outcomes,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(report.successful(), 3);
        assert!(report.is_success());
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Succeeded
        );
    }
}

#[test]
fn event_sensitivity_inputs_validate_plan_shapes_and_isolation() {
    let layout_provider = Shape(2, 1);
    let mut layout_workspace = ForwardSensitivityWorkspace::new();
    let layout_parameters = vec![0.0; 1];
    let layout_directions = vec![0.0; 1];
    let plan = scioxide_ode::ForwardSensitivitySystem::bind(
        &layout_provider,
        &layout_parameters,
        MatrixView::from_slice(1, 1, &layout_directions).unwrap(),
        &mut layout_workspace,
    )
    .unwrap();
    let mut times = [7.];
    let mut states = [8., 9.];
    let wrong = EventSensitivityInput {
        fixed_time: MatrixView::from_slice(1, 1, &[1.]).unwrap(),
        state_slope: &[1., 1.],
        numerators: &[1.],
        event_slope: 1.,
        minimum_slope: 0.,
        isolation: EventIsolation::Isolated,
    };
    assert!(
        plan.event_sensitivities_into(
            wrong,
            &mut times,
            MatrixViewMut::from_slice(2, 1, &mut states).unwrap()
        )
        .is_err()
    );
    assert_eq!((times, states), ([7.], [8., 9.]));
    let ambiguous = EventSensitivityInput {
        fixed_time: MatrixView::from_slice(2, 1, &[1., 2.]).unwrap(),
        state_slope: &[1., 1.],
        numerators: &[1.],
        event_slope: 1.,
        minimum_slope: 0.,
        isolation: EventIsolation::Ambiguous,
    };
    assert_eq!(
        plan.event_sensitivities_into(
            ambiguous,
            &mut times,
            MatrixViewMut::from_slice(2, 1, &mut states).unwrap()
        )
        .unwrap(),
        EventSensitivityStatus::Ambiguous
    );
    assert_eq!((times, states), ([7.], [8., 9.]));
}

#[test]
fn an_incomplete_ensemble_is_unsuccessful_even_if_completed_members_succeed() {
    #[derive(Clone, Copy)]
    struct Finished;
    impl EnsembleOutcome for Finished {
        fn is_success(&self) -> bool {
            true
        }
    }
    struct PartialExecutor;
    impl Executor for PartialExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(3).unwrap()
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            if let Some(partition) = plan.partitions().next() {
                operation(partition, (), context)?;
            }
            Err(SciOxideError::execution_stopped(
                ExecutionStopReason::Cancelled,
            ))
        }
    }
    let plan = ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap());
    let parameters = EnsembleParameters::Shared(&[] as &[f64]);
    let mut workspace = {
        let mut workspace = scioxide_ode::EnsembleWorkspace::new();
        workspace
            .try_reserve(&scioxide_ode::EnsembleRequirements::new(3, parameters).unwrap())
            .unwrap();
        workspace
    };
    let mut members = [(); 3];
    let mut outcomes = [const { None }; 3];
    let run = |_: usize, _: &[f64], _: &mut (), _: &mut SolveContext<'_, '_>| Ok(Finished);
    assert!(
        scioxide_ode::solve_ensemble_into(
            parameters,
            &run,
            plan,
            &mut members,
            &mut outcomes,
            &mut workspace,
            ExecutionContext::new(&PartialExecutor)
        )
        .is_err()
    );
    let report = workspace.last_attempt().unwrap().progress().unwrap();
    assert_eq!(report.completed(), 1);
    assert_eq!(report.successful(), 1);
    assert!(!report.is_success());
}

struct Shape(usize, usize);
impl OdeSensitivitySystem<f64> for Shape {
    fn dimension(&self) -> usize {
        self.0
    }
    fn parameter_dimension(&self) -> usize {
        self.1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        _: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn tangent_rhs(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        _: &[f64],
        _: &[f64],
        _: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
}
