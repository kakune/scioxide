#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
use super::*;
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    Budget, ErrorTolerance, ExecutionContext, NormSelection, NumericScalar, ProblemToken,
    RealScalar, SciOxideError, SolveBudgets, SolveIdentity, SolveState, StructuralPatternView,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{OdeMethod, OdeOptions, OdeProblem, OdeState, OdeSystem};
use scioxide_spatial::SegmentMesh;

use crate::support::{
    DISCRETIZATIONS, Discretization, End, Ends, Heat as HeatProblem, HyperbolicPde, ParabolicPde,
    Plan1d, TestScalar, Wave as WaveProblem, value,
};

const TEST_ARGUMENT: scioxide_core::Argument = scioxide_core::argument!("fixture");
/// Run direct callback evaluations in a private, unbounded root epoch.
fn context<R>(
    run: impl FnOnce(&mut scioxide_core::SolveContext<'_, '_>) -> Result<R, SciOxideError>,
) -> Result<R, SciOxideError> {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "mol", 1),
        0,
        0,
    );
    run(&mut state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?)
}
fn s<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}

// Independent three-site complete-graph Laplacian with reaction. The
// zero-sum mode [1,-1,0] has eigenvalue -4, including the off-band (0,2) edge.
struct Graph<T: NumericScalar> {
    mass: [T::Real; 3],
}
impl<T: NumericScalar> Graph<T> {
    fn entry(&self, i: usize, j: usize) -> T {
        T::from_components(
            self.mass[i] * s(if i == j { -3. } else { 1. }),
            T::Real::ZERO,
        )
    }
}
impl<T: NumericScalar> OdeSystem<T::Real, T> for Graph<T> {
    fn dimension(&self) -> usize {
        3
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
    fn time_derivative(
        &self,
        _: T::Real,
        _: &[T],
        output: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(T::ZERO);
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: T::Real,
        y: &[T],
        out: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for (i, v) in out.iter_mut().enumerate() {
            *v = (0..3).fold(T::ZERO, |a, j| a + self.entry(i, j) * y[j]);
        }
        Ok(())
    }
    fn jacobian(
        &self,
        _: T::Real,
        _: &[T],
        out: &mut MatrixViewMut<'_, T>,
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..3 {
            for j in 0..3 {
                out.set(i, j, self.entry(i, j))?;
            }
        }
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: T::Real,
        _: &[T],
        p: StructuralPatternView<'_>,
        out: &mut [T],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..3 {
            for (k, value) in out
                .iter_mut()
                .enumerate()
                .take(p.row_ptr()[i + 1])
                .skip(p.row_ptr()[i])
            {
                *value = self.entry(i, p.col_indices()[k]);
            }
        }
        Ok(Some(()))
    }
    fn jacobian_product(
        &self,
        t: T::Real,
        _: &[T],
        x: &[T],
        out: &mut [T],
        _: &mut [T],
        cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.rhs(t, x, out, cx).map(Some)
    }
}

#[derive(Clone)]
struct Heat;
impl<S: RealScalar> ParabolicPde<S> for Heat {
    fn diffusion(&self, _: &[S], _: S) -> Result<S, SciOxideError> {
        Ok(S::ONE)
    }
    fn source(&self, _: &[S], _: S) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
    fn initial_condition(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok((s::<S>(core::f64::consts::PI) * x[0]).sin())
    }
}
impl<S: RealScalar> HyperbolicPde<S> for Heat {
    fn wave_speed(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::ONE)
    }
    fn source(&self, _: &[S], _: S) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
    fn initial_condition(&self, x: &[S]) -> Result<S, SciOxideError> {
        <Self as ParabolicPde<S>>::initial_condition(self, x)
    }
    fn initial_velocity(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
}

fn mesh<S: TestScalar>(n: usize) -> SegmentMesh<S> {
    crate::mesh_fixture::uniform(S::ZERO, S::ONE, n)
}

fn zero_ends<S: TestScalar>() -> Ends<S> {
    Ends::new(
        End::dirichlet(value(|_| Ok(S::ZERO))),
        End::dirichlet(value(|_| Ok(S::ZERO))),
    )
}

/// Heat plan of one family with homogeneous Dirichlet ends.
fn heat_plan<S: TestScalar>(discretization: Discretization, n: usize) -> Plan1d<S> {
    discretization
        .plan(&mesh::<S>(n), &HeatProblem(Heat), &zero_ends())
        .unwrap()
}

fn heat_source<S: TestScalar>() -> FieldData<S, S, 1> {
    crate::support::source(&HeatProblem(Heat))
}

fn heat_initial<S: TestScalar>() -> FieldData<S, S, 1> {
    crate::support::initial(&Heat)
}

/// Result, caller state and workspace of one transient attempt.
type Advanced<S> = (
    Result<PdeSolveReport<S>, SciOxideError>,
    PdeState<S>,
    PdeWorkspace<S>,
);

/// Advance (or, given a checkpoint, resume) `discretization` from `initial` under
/// `source` to `end`, streaming to `sink`.
fn attempt_on<S: TestScalar, P: SpatialDiscretization<S, S, 1>>(
    discretization: &P,
    transient: &ParabolicOptions<S>,
    [source, initial]: [&FieldData<S, S, 1>; 2],
    end: S,
    checkpoint: Option<&PdeCheckpoint<S>>,
    sink: &mut dyn PdeOutputSink<S>,
    execution: ExecutionContext<'_>,
) -> Advanced<S> {
    let problem = ParabolicProblem::new(discretization, source, end).unwrap();
    let mut state = problem.initial_state(initial, transient).unwrap();
    let mut workspace = PdeWorkspace::new();
    let result = match checkpoint {
        Some(checkpoint) => problem.resume(
            transient,
            checkpoint,
            &mut state,
            sink,
            &mut workspace,
            execution,
        ),
        None => problem.advance(transient, &mut state, sink, &mut workspace, execution),
    };
    (result, state, workspace)
}

/// Advance `plan` from `initial` under `source` to `end`, streaming to `sink`.
fn advance<S: TestScalar>(
    plan: &Plan1d<S>,
    options: ParabolicOptions<S>,
    data: [&FieldData<S, S, 1>; 2],
    end: S,
    sink: &mut dyn PdeOutputSink<S>,
    execution: ExecutionContext<'_>,
) -> Advanced<S> {
    let transient = options;
    crate::with_plan!(plan, |p| attempt_on(
        p, &transient, data, end, None, sink, execution
    ))
}

/// Sequential heat advance from the initial data to `end`.
fn heat_advance<S: TestScalar>(
    plan: &Plan1d<S>,
    options: ParabolicOptions<S>,
    end: S,
    sink: &mut dyn PdeOutputSink<S>,
) -> Advanced<S> {
    advance(
        plan,
        options,
        [&heat_source(), &heat_initial()],
        end,
        sink,
        ExecutionContext::sequential_native(),
    )
}

/// Resume the heat problem to `end` from `checkpoint`, restoring into a fresh state.
fn heat_resume<S: TestScalar>(
    plan: &Plan1d<S>,
    options: ParabolicOptions<S>,
    end: S,
    checkpoint: &PdeCheckpoint<S>,
    sink: &mut dyn PdeOutputSink<S>,
) -> Advanced<S> {
    let (source, initial) = (heat_source(), heat_initial());
    let transient = options;
    crate::with_plan!(plan, |p| attempt_on(
        p,
        &transient,
        [&source, &initial],
        end,
        Some(checkpoint),
        sink,
        ExecutionContext::sequential_native(),
    ))
}

/// Owned heat solve from the initial data to `end`.
fn heat_solve<S: TestScalar>(
    plan: &Plan1d<S>,
    options: ParabolicOptions<S>,
    end: S,
) -> PdeSolution<S> {
    let (source, initial) = (heat_source(), heat_initial());
    let transient = options;
    crate::with_plan!(plan, |p| ParabolicProblem::new(p, &source, end)
        .unwrap()
        .solve_with_options(&initial, &transient))
    .unwrap()
}

fn assert_invalid_input(error: &SciOxideError) {
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ),
        "{error}"
    );
}

fn builtin<S: TestScalar>() {
    let tol = if S::EPSILON > s(1e-10) { 1e-5 } else { 1e-9 };
    for scheme in [
        ParabolicScheme::BackwardEuler,
        ParabolicScheme::CrankNicolson,
        ParabolicScheme::ExplicitDopri5,
    ] {
        let options = ParabolicOptions::new()
            .with_scheme(scheme)
            .with_step(s(0.002))
            .with_error_tolerance(
                ErrorTolerance::scalar(s(tol), s(tol), NormSelection::Infinity).unwrap(),
            )
            .with_output_schedule(PdeOutputSchedule::TimeSteps);
        for discretization in DISCRETIZATIONS {
            let plan = heat_plan::<S>(discretization, 17);
            let solution = heat_solve(&plan, options.clone(), s(0.01));
            let report = solution.report();
            let times = solution.trajectory().times();
            assert!(
                solution
                    .trajectory()
                    .states()
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite())
            );
            assert!(report.is_success(), "{:?}", report.termination());
            assert_eq!(report.final_time(), s(0.01));
            assert_eq!(times.len(), report.diagnostics().num_accepted_steps() + 1);
            assert!(times.windows(2).all(|p| p[0] < p[1]));
            let maximum = solution
                .values()
                .iter()
                .copied()
                .fold(S::ZERO, |a, b| a.max(b));
            assert!(
                (maximum - s::<S>((-core::f64::consts::PI.powi(2) * 0.01).exp())).abs() < s(0.01),
                "{discretization:?} {scheme:?}"
            );
        }
    }
}
#[test]
fn builtins_f32() {
    builtin::<f32>();
}
#[test]
fn builtins_f64() {
    builtin::<f64>();
}

#[test]
fn invalid_scaling_and_transactional_conversion() {
    for bad in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(MethodOfLinesPlan::new(1, Some(&[bad]), None).is_err());
        assert!(MethodOfLinesPlan::new(1, None, Some(&[bad])).is_err());
    }
    let plan = MethodOfLinesPlan::new(2, None, Some(&[1., f64::MAX])).unwrap();
    let mut out = [7., 8.];
    assert!(plan.decode_into(&[1., 2.], &mut out).is_err());
    assert_eq!(out, [7., 8.]);
    assert!(
        plan.error_tolerance(
            &ErrorTolerance::try_vector(&[1.], 0., NormSelection::Infinity).unwrap()
        )
        .is_err()
    );
}

#[test]
fn declared_time_invariant_loads_are_reused_only_after_success() {
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicUsize, Ordering};
    for family in DISCRETIZATIONS {
        let natural = Ends::new(
            End::natural(BoundaryData::zero()),
            End::natural(BoundaryData::zero()),
        );
        let plan = family
            .plan(&mesh::<f64>(5), &HeatProblem(Heat), &natural)
            .unwrap();
        for time_dependent in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let callback_calls = Arc::clone(&calls);
            let callback = move |_: FieldPoint<f64, 1>, time: f64| {
                if callback_calls.fetch_add(1, Ordering::Relaxed) == 0 {
                    return Err(SciOxideError::invalid_input(
                        TEST_ARGUMENT,
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "first load fails",
                    ));
                }
                Ok(1.0 + time)
            };
            let source = if time_dependent {
                FieldData::time_dependent(callback)
            } else {
                FieldData::time_invariant(move |p| callback(p, 0.0))
            };
            crate::with_plan!(&plan, |p| {
                let semi = SemidiscreteSystem::new(p, &source).unwrap();
                assert_eq!(calls.load(Ordering::Relaxed), 0);
                let input = vec![0.0; semi.dimension()];
                let mut out = vec![7.0; semi.dimension()];
                assert!(context(|cx| semi.rhs(0.0, &input, &mut out, cx)).is_err());
                assert!(out.iter().all(|&value| value == 7.0));
                context(|cx| semi.rhs(0.0, &input, &mut out, cx)).unwrap();
                let first = out.clone();
                let count = calls.load(Ordering::Relaxed);
                context(|cx| semi.rhs(1.0, &input, &mut out, cx)).unwrap();
                if time_dependent {
                    assert!(calls.load(Ordering::Relaxed) > count);
                    for (&value, &initial) in out.iter().zip(&first) {
                        assert!(
                            (value - 2.0 * initial).abs() <= 16.0 * f64::EPSILON * initial.abs()
                        );
                    }
                } else {
                    assert_eq!(calls.load(Ordering::Relaxed), count);
                    assert_eq!(out, first);
                }
                let previous = out.clone();
                assert!(context(|cx| semi.rhs(f64::NAN, &input, &mut out, cx)).is_err());
                assert_eq!(out, previous);
            });
        }
        let moving = Ends::new(
            End::natural(BoundaryData::time_dependent(|_, t| Ok(t))),
            End::natural(BoundaryData::zero()),
        );
        let plan = family
            .plan(&mesh::<f64>(5), &HeatProblem(Heat), &moving)
            .unwrap();
        crate::with_plan!(&plan, |p| {
            let source = FieldData::zero();
            let semi = SemidiscreteSystem::new(p, &source).unwrap();
            let input = vec![0.0; semi.dimension()];
            let mut out = vec![0.0; semi.dimension()];
            context(|cx| semi.rhs(0.0, &input, &mut out, cx)).unwrap();
            assert!(out.iter().all(|&value| value == 0.0));
            context(|cx| semi.rhs(1.0, &input, &mut out, cx)).unwrap();
            assert!(out.iter().any(|&value| value != 0.0));
        });
    }
}

#[test]
fn tolerance_reaches_all_wave_paths() {
    for discretization in DISCRETIZATIONS {
        let plan = discretization
            .plan(&mesh::<f64>(9), &WaveProblem(Heat), &zero_ends())
            .unwrap();
        let field = crate::with_plan!(&plan, |p| p.unknowns());
        let mut observations = Vec::new();
        for tolerance in [1e-3, 1e-9] {
            let mut absolute = vec![1.; 2 * field];
            absolute[field..].fill(tolerance);
            let options = HyperbolicOptions::new()
                .with_initial_step(0.5)
                .with_cfl_policy(CflPolicy::Disabled)
                .with_error_tolerance(
                    ErrorTolerance::try_vector(&absolute, tolerance, NormSelection::Infinity)
                        .unwrap(),
                );
            let transient = options;
            let source = heat_source::<f64>();
            let initial = heat_initial::<f64>();
            let velocity = FieldData::zero();
            let solution = crate::with_plan!(&plan, |p| HyperbolicProblem::new(p, &source, 1.)
                .unwrap()
                .solve_with_options(&initial, &velocity, &transient))
            .unwrap();
            observations.push(solution.report().diagnostics().num_accepted_steps());
        }
        assert!(
            observations[1] > observations[0],
            "{discretization:?} {observations:?}"
        );
    }
}

#[test]
fn rannacher_and_global_budgets() {
    let plan = heat_plan::<f64>(Discretization::Fdm, 9);
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::CrankNicolson)
        .with_step(0.02)
        .with_rannacher_startup(Budget::try_from(1).unwrap());
    let solution = heat_solve(
        &plan,
        options
            .clone()
            .with_output_schedule(PdeOutputSchedule::TimeSteps),
        0.05,
    );
    assert!(solution.report().is_success());
    assert_eq!(solution.trajectory().times(), [0., 0.01, 0.02, 0.04, 0.05]);
    let solution = heat_solve(
        &plan,
        options.with_step_budget(Budget::try_from(3).unwrap()),
        0.05,
    );
    let report = solution.report();
    assert_eq!(report.termination(), PdeTermination::TimeStepLimit);
    assert_eq!(report.diagnostics().num_accepted_steps(), 3);
    assert_eq!(report.final_time(), 0.04);
    // Final-only output still emits the field reached by the budget stop.
    assert_eq!(solution.trajectory().times(), [0.04]);
}

#[derive(Clone)]
struct Moving;
impl ParabolicPde<f64> for Moving {
    fn diffusion(&self, _: &[f64], _: f64) -> Result<f64, SciOxideError> {
        Ok(2.)
    }
    fn source(&self, x: &[f64], _: f64) -> Result<f64, SciOxideError> {
        Ok(x[0])
    }
    fn initial_condition(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0])
    }
}
#[test]
fn time_dependent_lifting_and_executor_replay() {
    let ends = Ends::new(
        End::dirichlet(crate::support::moving(|x, t| Ok(x * (1. + t)))),
        End::dirichlet(crate::support::moving(|x, t| Ok(x * (1. + t)))),
    );
    for scheme in [
        ParabolicScheme::BackwardEuler,
        ParabolicScheme::CrankNicolson,
        ParabolicScheme::ExplicitDopri5,
    ] {
        let options = ParabolicOptions::new()
            .with_scheme(scheme)
            .with_step(0.01)
            .with_output_schedule(PdeOutputSchedule::TimeSteps);
        for discretization in DISCRETIZATIONS {
            let plan = discretization
                .plan(&mesh::<f64>(9), &HeatProblem(Moving), &ends)
                .unwrap();
            let offset = if discretization == Discretization::Fvm {
                0.5
            } else {
                0.
            };
            let mut check = |observation: PdeObservation<'_, f64, f64>| {
                let t = observation.time();
                for (i, v) in observation.values().iter().enumerate() {
                    let x = (i as f64 + offset) / 8.;
                    assert!(
                        (v - x * (1. + t)).abs() < 1e-8,
                        "{discretization:?} {scheme:?}"
                    );
                }
                Ok(())
            };
            let source = crate::support::source(&HeatProblem(Moving));
            let initial = FieldData::time_invariant(|p: FieldPoint<f64, 1>| Ok(p.point[0]));
            let (first, first_state, _) = advance(
                &plan,
                options.clone(),
                [&source, &initial],
                0.03,
                &mut check,
                ExecutionContext::sequential_native(),
            );
            let executor = scioxide_core::SequentialExecutor;
            let (second, second_state, _) = advance(
                &plan,
                options.clone(),
                [&source, &initial],
                0.03,
                &mut check,
                ExecutionContext::new(&executor),
            );
            assert!(first.unwrap().is_success() && second.unwrap().is_success());
            assert_eq!(first_state.time(), 0.03);
            assert_eq!(first_state.coordinates(), second_state.coordinates());
        }
    }
}

#[test]
fn default_cfl_caps_zero_error_growth() {
    let plan = heat_plan::<f64>(Discretization::Fdm, 9);
    for policy in [
        CflPolicy::Automatic,
        CflPolicy::MaximumStep(0.002),
        CflPolicy::Disabled,
    ] {
        let options = ParabolicOptions::new()
            .with_scheme(ParabolicScheme::ExplicitDopri5)
            .with_step(0.5)
            .with_cfl_policy(policy)
            .with_output_schedule(PdeOutputSchedule::TimeSteps);
        let mut previous = 0.;
        let mut emitted = 0;
        let zero = FieldData::zero();
        let (report, _, _) = advance(
            &plan,
            options,
            [&zero, &zero],
            0.03,
            &mut |observation: PdeObservation<'_, f64, f64>| {
                let t = observation.time();
                // Gershgorin bound 4/h² of M⁻¹K and the DOPRI5 real-axis segment.
                let cap = match policy {
                    CflPolicy::Automatic => 0.9 * 3.3065 / 256.,
                    CflPolicy::MaximumStep(h) => h,
                    CflPolicy::Disabled => 1.,
                };
                assert!(t - previous <= cap + 1e-15);
                previous = t;
                emitted += 1;
                Ok(())
            },
            ExecutionContext::sequential_native(),
        );
        assert!(report.unwrap().is_success());
        if policy == CflPolicy::Disabled {
            assert_eq!(emitted, 2);
        } else {
            assert!(emitted > 2);
        }
    }
}

#[test]
fn fully_constrained_and_one_cell_domains() {
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::BackwardEuler)
        .with_step(0.01);
    let zero = FieldData::zero();
    for (discretization, expected) in [(Discretization::Fdm, 2), (Discretization::Fvm, 1)] {
        let plan = heat_plan::<f64>(discretization, 2);
        let transient = options.clone();
        let solution = crate::with_plan!(&plan, |p| ParabolicProblem::new(p, &zero, 0.1)
            .unwrap()
            .solve_with_options(&zero, &transient))
        .unwrap();
        let r = solution.report();
        assert!(r.is_success());
        assert_eq!(solution.values(), vec![0.; expected]);
        if discretization == Discretization::Fdm {
            assert_eq!(r.diagnostics().num_time_steps(), 0);
        }
    }
}

fn requested_times<S: TestScalar>() {
    let plan = heat_plan::<S>(Discretization::Fdm, 9);
    let requests = [s(0.), s(0.), s(0.003), s(0.003), s(0.02), s(0.02)];
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(s(0.005))
        .with_output_schedule(PdeOutputSchedule::Times(requests.to_vec()));
    let mut seen = Vec::new();
    let (report, state, _) = heat_advance(
        &plan,
        options,
        s(0.02),
        &mut |observation: PdeObservation<'_, S, S>| {
            assert_eq!(observation.values().len(), 9);
            assert!(observation.values().iter().all(|v| v.is_finite()));
            seen.push((observation.request_index().unwrap(), observation.time()));
            Ok(())
        },
    );
    assert!(report.unwrap().is_success());
    assert_eq!(seen, requests.into_iter().enumerate().collect::<Vec<_>>());
    let Plan1d::Fdm(fdm) = &plan else {
        unreachable!()
    };
    assert!(
        PdeCheckpoint::new(&state)
            .unwrap()
            .mesh_token()
            .is_compatible_with(fdm.mesh().mesh_token())
    );
}

#[test]
fn requested_times_f32() {
    requested_times::<f32>();
}
#[test]
fn requested_times_f64() {
    requested_times::<f64>();
}

#[test]
fn requested_times_reject_invalid_order_range_and_missing_dense_output() {
    let plan = heat_plan::<f64>(Discretization::Fdm, 9);
    let explicit = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(0.005);
    // Unordered and non-finite requests are rejected by the plan.
    for times in [vec![0.01, 0.0], vec![f64::NAN]] {
        let options = explicit
            .clone()
            .with_output_schedule(PdeOutputSchedule::Times(times));
        let source = heat_source();
        let initial = heat_initial();
        let error = crate::with_plan!(&plan, |p| ParabolicProblem::new(p, &source, 0.02)
            .unwrap()
            .initial_state(&initial, &options))
        .unwrap_err();
        assert_invalid_input(&error);
    }
    // Requests beyond the problem end are rejected at preflight.
    let (result, state, workspace) = heat_advance(
        &plan,
        explicit.with_output_schedule(PdeOutputSchedule::Times(vec![0.01, 0.03])),
        0.02,
        &mut PdeTrajectory::new(),
    );
    assert_invalid_input(&result.unwrap_err());
    assert_eq!(state.time(), 0.0);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
    // Methods without dense output cannot serve requested times of free dynamics: the
    // built-in implicit schemes, Rannacher startup and injected methods are rejected
    // at preflight, before any callback.
    let times = PdeOutputSchedule::Times(vec![0.01]);
    let fixed = |scheme| {
        ParabolicOptions::new()
            .with_scheme(scheme)
            .with_step(0.01)
            .with_output_schedule(times.clone())
    };
    for options in [
        fixed(ParabolicScheme::BackwardEuler),
        fixed(ParabolicScheme::CrankNicolson).with_rannacher_startup(Budget::ONE),
        fixed(ParabolicScheme::BackwardEuler).with_ode_method(scioxide_ode::ForwardEuler::new()),
    ] {
        let (result, state, workspace) =
            heat_advance(&plan, options, 0.02, &mut PdeTrajectory::new());
        assert_invalid_input(&result.unwrap_err());
        assert_eq!(state.time(), 0.0);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::PreflightRejected
        );
    }
}

#[test]
fn checkpoint_resume_preserves_committed_heat_and_mesh_identity() {
    let plan = heat_plan::<f64>(Discretization::Fdm, 9);
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::BackwardEuler)
        .with_step(0.01)
        .with_output_schedule(PdeOutputSchedule::TimeSteps);
    let full = heat_solve(&plan, options.clone(), 0.04);
    let mut calls = 0;
    let (result, state, workspace) =
        heat_advance(&plan, options.clone(), 0.04, &mut |_: PdeObservation<
            '_,
            f64,
            f64,
        >| {
            calls += 1;
            if calls == 4 {
                Err(crate::validation::invalid(TEST_ARGUMENT, "observer stop"))
            } else {
                Ok(())
            }
        });
    assert!(result.is_err());
    assert_eq!(state.time(), 0.02);
    let progress = workspace.last_attempt().unwrap().progress().unwrap();
    assert_eq!(progress.time(), 0.02);
    let checkpoint = PdeCheckpoint::new(progress.state()).unwrap();
    assert_eq!(checkpoint.time(), state.time());
    let mut trajectory = PdeTrajectory::new();
    let (resumed, _, _) = heat_resume(&plan, options.clone(), 0.04, &checkpoint, &mut trajectory);
    assert!(resumed.unwrap().is_success());
    let last = trajectory.states().last().unwrap();
    for (&a, &b) in last.iter().zip(full.values()) {
        assert!((a - b).abs() < 1e-8);
    }
    let other = heat_plan::<f64>(Discretization::Fdm, 9);
    assert!(
        heat_resume(
            &other,
            options,
            0.04,
            &checkpoint,
            &mut PdeTrajectory::new()
        )
        .0
        .is_err()
    );
}

#[test]
fn wave_checkpoint_retains_velocity_and_resumes_with_executor() {
    let plan = Discretization::Fdm
        .plan(&mesh::<f64>(9), &WaveProblem(Heat), &zero_ends())
        .unwrap();
    let Plan1d::Fdm(fdm) = &plan else {
        unreachable!()
    };
    let transient = HyperbolicOptions::new()
        .with_initial_step(0.01)
        .with_output_schedule(PdeOutputSchedule::TimeSteps);
    let (source, initial, velocity) = (
        heat_source::<f64>(),
        heat_initial::<f64>(),
        FieldData::zero(),
    );
    let problem = HyperbolicProblem::new(fdm, &source, 0.04).unwrap();
    let full = problem
        .solve_with_options(&initial, &velocity, &transient)
        .unwrap();
    let mut calls = 0;
    let mut state = problem.initial_state(&initial, &velocity).unwrap();
    let mut workspace = PdeWorkspace::<f64>::new();
    let failure = problem.advance(
        &transient,
        &mut state,
        &mut |_: PdeObservation<'_, f64, f64>| {
            calls += 1;
            if calls == 3 {
                Err(crate::validation::invalid(TEST_ARGUMENT, "observer stop"))
            } else {
                Ok(())
            }
        },
        &mut workspace,
        ExecutionContext::sequential_native(),
    );
    assert!(failure.is_err());
    // The checkpoint holds displacement and velocity of every free unknown.
    let checkpoint = PdeCheckpoint::new(&state).unwrap();
    assert!(checkpoint.time() > 0.0);
    assert_eq!(checkpoint.state().coordinates().len(), 2 * 7);
    let executor = scioxide_core::SequentialExecutor;
    let mut resumed_state = problem.initial_state(&initial, &velocity).unwrap();
    let mut trajectory = PdeTrajectory::new();
    let resumed = problem
        .resume(
            &transient,
            &checkpoint,
            &mut resumed_state,
            &mut trajectory,
            &mut PdeWorkspace::<f64>::new(),
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert!(resumed.is_success());
    let last = trajectory.states().last().unwrap();
    for (&a, &b) in last.iter().zip(full.values()) {
        assert!((a - b).abs() < 1e-6);
    }
    let omega = 16.0 * (core::f64::consts::PI / 16.0).sin();
    let error = last
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let x = i as f64 / 8.0;
            (value - (core::f64::consts::PI * x).sin() * (omega * 0.04).cos()).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        error < 5e-5,
        "independent discrete wave mode error {error:e}"
    );
}

#[test]
fn rannacher_restart_keeps_half_step_phase() {
    let plan = heat_plan::<f64>(Discretization::Fdm, 9);
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::CrankNicolson)
        .with_step(0.02)
        .with_rannacher_startup(Budget::try_from(1).unwrap())
        .with_output_schedule(PdeOutputSchedule::TimeSteps);
    let full = heat_solve(&plan, options.clone(), 0.05);
    let mut calls = 0;
    let (result, state, _) =
        heat_advance(&plan, options.clone(), 0.05, &mut |_: PdeObservation<
            '_,
            f64,
            f64,
        >| {
            calls += 1;
            if calls == 3 {
                Err(crate::validation::invalid(TEST_ARGUMENT, "stop midway"))
            } else {
                Ok(())
            }
        });
    assert!(result.is_err());
    let checkpoint = PdeCheckpoint::new(&state).unwrap();
    assert_eq!(checkpoint.time(), 0.01);
    assert_eq!(checkpoint.state().startup_end(), Some(0.02));
    let mut trajectory = PdeTrajectory::new();
    let (resumed, resumed_state, _) =
        heat_resume(&plan, options, 0.05, &checkpoint, &mut trajectory);
    assert!(resumed.unwrap().is_success());
    assert_eq!(trajectory.times(), [0.01, 0.02, 0.04, 0.05]);
    assert_eq!(resumed_state.startup_end(), None);
    let last = trajectory.states().last().unwrap();
    for (&a, &b) in last.iter().zip(full.values()) {
        assert!((a - b).abs() < 1e-8);
    }
}

#[test]
fn cancellation_returns_resumable_accepted_state() {
    let plan = heat_plan::<f64>(Discretization::Fdm, 9);
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::BackwardEuler)
        .with_step(0.01);
    let cancellation = scioxide_core::CancellationToken::new();
    let mut observed = 0;
    let (source, initial) = (heat_source::<f64>(), heat_initial::<f64>());
    let (result, state, _) = advance(
        &plan,
        options
            .clone()
            .with_output_schedule(PdeOutputSchedule::TimeSteps),
        [&source, &initial],
        0.04,
        &mut |_: PdeObservation<'_, f64, f64>| {
            observed += 1;
            if observed == 3 {
                cancellation.cancel();
            }
            Ok(())
        },
        ExecutionContext::sequential().with_cancellation(&cancellation),
    );
    if let Ok(report) = &result {
        assert!(!report.is_success());
    }
    let checkpoint = PdeCheckpoint::new(&state).unwrap();
    assert!(checkpoint.time() > 0.0 && checkpoint.time() < 0.04);
    let (resumed, _, _) = heat_resume(&plan, options, 0.04, &checkpoint, &mut ());
    assert!(resumed.unwrap().is_success());
}

#[test]
fn requested_times_keep_cell_values_and_duplicate_indices() {
    let plan = heat_plan::<f64>(Discretization::Fvm, 9);
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(0.005)
        .with_output_schedule(PdeOutputSchedule::Times(vec![0.0, 0.01, 0.01, 0.02]));
    let mut indices = Vec::new();
    let (report, _, _) = heat_advance(&plan, options, 0.02, &mut |observation: PdeObservation<
        '_,
        f64,
        f64,
    >| {
        assert_eq!(observation.values().len(), 8);
        indices.push(observation.request_index().unwrap());
        Ok(())
    });
    assert!(report.unwrap().is_success());
    assert_eq!(indices, [0, 1, 2, 3]);
}

#[test]
fn fully_constrained_requested_output_failure_keeps_prior_field() {
    let plan = heat_plan::<f64>(Discretization::Fdm, 2);
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::BackwardEuler)
        .with_step(0.01)
        .with_output_schedule(PdeOutputSchedule::Times(vec![0.0, 0.01, 0.01, 0.02]));
    let mut seen = Vec::new();
    let (result, state, workspace) =
        heat_advance(&plan, options, 0.02, &mut |sample: PdeObservation<
            '_,
            f64,
            f64,
        >| {
            assert_eq!(sample.values(), [0.0, 0.0]);
            seen.push((sample.request_index().unwrap(), sample.time()));
            if seen.len() == 3 {
                Err(crate::validation::invalid(TEST_ARGUMENT, "observer stop"))
            } else {
                Ok(())
            }
        });
    assert!(result.is_err());
    assert_eq!(seen, [(0, 0.0), (1, 0.01), (2, 0.01)]);
    assert_eq!(state.time(), 0.01);
    assert!(state.coordinates().is_empty());
    let progress = workspace.last_attempt().unwrap().progress().unwrap();
    assert_eq!(progress.time(), 0.01);
    assert_eq!(PdeCheckpoint::new(progress.state()).unwrap().time(), 0.01);
}

#[test]
fn malformed_tolerance_fails_before_source_callbacks() {
    let plan = heat_plan::<f64>(Discretization::Fdm, 9);
    let options = ParabolicOptions::new()
        .with_scheme(ParabolicScheme::ExplicitDopri5)
        .with_step(0.1)
        .with_error_tolerance(
            ErrorTolerance::try_vector(&[1e-5], 1e-5, NormSelection::Infinity).unwrap(),
        );
    let never =
        FieldData::time_dependent(|_: FieldPoint<f64, 1>, _| -> Result<f64, SciOxideError> {
            unreachable!()
        });
    let (result, state, workspace) = advance(
        &plan,
        options,
        [&never, &FieldData::zero()],
        0.1,
        &mut |_: PdeObservation<'_, f64, f64>| -> Result<(), SciOxideError> { unreachable!() },
        ExecutionContext::sequential_native(),
    );
    assert!(result.is_err());
    assert_eq!(state.time(), 0.0);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
}

#[test]
fn decoded_reports_preserve_negative_times_and_preproposal_budget_stops() {
    use scioxide_core::BudgetKind;
    let plan = MethodOfLinesPlan::new(3, None, None).unwrap();
    let provider = Graph::<f64> { mass: [1.; 3] };
    let mut work = plan.workspace().unwrap();
    let system = plan.bind(&provider, &mut work).unwrap();
    let options = OdeOptions::new().with_initial_step(0.01);
    let solve = (scioxide_ode::DormandPrince5::new(), options.clone());
    let problem = OdeProblem::new(&system, -0.2);
    let mut workspace = solve.0.workspace(&problem, &solve.1).unwrap();
    let mut state = OdeState::new(-0.1, Vector::from_vec(vec![0.; 3]));
    let raw = solve
        .0
        .advance(
            &problem,
            &solve.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let decoded = PdeSolveReport::try_from(&raw).unwrap();
    assert_eq!(decoded.final_time(), -0.2);
    assert_eq!(state.time(), -0.2);
    assert!(decoded.is_success());
    // Cancellation stops the driver before its first attempt.
    let cancelled = scioxide_core::CancellationToken::new();
    cancelled.cancel();
    let mut state = OdeState::new(-0.1, Vector::from_vec(vec![0.; 3]));
    let raw = solve
        .0
        .advance(
            &problem,
            &solve.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancelled),
        )
        .unwrap();
    assert_eq!(raw.attempted_steps(), 0);
    let decoded = PdeSolveReport::try_from(&raw).unwrap();
    assert_eq!(
        decoded.termination(),
        PdeTermination::ExecutionStopped(scioxide_core::ExecutionStopReason::Cancelled)
    );
    assert_eq!(decoded.final_time(), -0.1);
    // An exhausted shared evaluation budget of the parent denies the first attempt's
    // stage evaluation: one unclassified attempt and no accepted step.
    let mut root_state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut root = root_state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(630),
                scioxide_core::algorithm_id!("test", "parent", 1),
                1,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    root.record_provider(scioxide_core::EvaluationRole::Function)
        .unwrap()
        .require()
        .unwrap();
    let mut state = OdeState::new(-0.1, Vector::from_vec(vec![0.; 3]));
    let raw = solve
        .0
        .advance(
            &problem,
            &solve.1,
            &mut state,
            &mut (),
            &mut workspace,
            &mut root,
        )
        .unwrap();
    assert_eq!(raw.attempted_steps(), 1);
    assert_eq!(raw.accepted_steps(), 0);
    let decoded = PdeSolveReport::try_from(&raw).unwrap();
    assert_eq!(decoded.termination(), PdeTermination::EvaluationLimit);
    assert_eq!(decoded.diagnostics().num_accepted_steps(), 0);
    assert_eq!(decoded.final_time(), -0.1);
    // An attempted-step budget stop decodes as the time-step limit.
    let solve = (
        scioxide_ode::DormandPrince5::new(),
        options.with_max_steps(Budget::ONE),
    );
    let mut state = OdeState::new(-0.1, Vector::from_vec(vec![0.; 3]));
    let raw = solve
        .0
        .advance(
            &problem,
            &solve.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(raw.attempted_steps(), 1);
    let decoded = PdeSolveReport::try_from(&raw).unwrap();
    assert_eq!(decoded.termination(), PdeTermination::TimeStepLimit);
    assert!((decoded.final_time() + 0.11).abs() < 1e-15);
}

/// Deliberately fallible provider to exercise adapter publication boundaries.
struct KernelProvider {
    values: core::cell::Cell<[f64; 2]>,
    fail: core::cell::Cell<bool>,
    calls: core::cell::Cell<usize>,
}
impl KernelProvider {
    fn finish(&self) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        if self.fail.get() {
            Err(SciOxideError::singular(None))
        } else {
            Ok(())
        }
    }
}
impl OdeSystem<f64> for KernelProvider {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(&self.values.get());
        self.finish()
    }
    fn jacobian_product(
        &self,
        t: f64,
        y: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut [f64],
        c: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.rhs(t, y, out, c).map(Some)
    }
    fn time_derivative(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        c: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.rhs(t, y, out, c).map(Some)
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for (i, v) in self.values.get().into_iter().enumerate() {
            for j in 0..2 {
                out.set(i, j, v)?;
            }
        }
        self.finish().map(Some)
    }
    fn sparse_jacobian(
        &self,
        t: f64,
        y: &[f64],
        _: StructuralPatternView<'_>,
        out: &mut [f64],
        c: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.rhs(t, y, out, c).map(Some)
    }
}

#[test]
fn dense_derivatives_require_preparation_and_conversion_preserves_boundaries() {
    let provider = KernelProvider {
        values: core::cell::Cell::new([2., 3.]),
        fail: core::cell::Cell::new(false),
        calls: core::cell::Cell::new(0),
    };
    let plan = MethodOfLinesPlan::new(2, None, None).unwrap();
    let mut work = plan.workspace().unwrap();
    let system = plan.bind(&provider, &mut work).unwrap();
    let mut matrix = scioxide_linalg::Matrix::from_vec(2, 2, vec![7.; 4]).unwrap();
    assert!(
        context(|c| system.jacobian(0., &[0., 0.], &mut matrix.view_mut(), c))
            .unwrap()
            .is_none()
    );
    assert_eq!(provider.calls.get(), 0);
    for bad in [f64::NAN, f64::INFINITY, -f64::INFINITY] {
        let mut output = [7., 8.];
        assert!(plan.encode_into(&[0., bad], &mut output).is_err());
        assert_eq!(output, [7., 8.]);
    }
    let plan = MethodOfLinesPlan::new(2, None, Some(&[2., 0.5])).unwrap();
    let mut output = [0.; 2];
    plan.decode_into(&[f64::MAX * 0.5, f64::MIN_SUBNORMAL * 2.], &mut output)
        .unwrap();
    assert_eq!(output, [f64::MAX, f64::MIN_SUBNORMAL]);
    plan.encode_into(&[f64::MIN_SUBNORMAL * 2., f64::MAX * 0.5], &mut output)
        .unwrap();
    assert_eq!(output, [f64::MIN_SUBNORMAL, f64::MAX]);
    output = [7., 8.];
    assert!(plan.encode_into(&[1., f64::MAX], &mut output).is_err());
    assert_eq!(output, [7., 8.]);
}
