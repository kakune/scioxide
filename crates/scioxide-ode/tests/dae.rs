//! Residual index-1 DAE plans: accuracy, initialization, budgets, attempts and allocation.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{
    AlgebraicScalar, AttemptOutcome, Budget, BudgetKind, Complex, ErrorTolerance, EvaluationRole,
    ExecutionContext, NormSelection, NumericScalar, ProblemToken, RealScalar, ResidualTolerance,
    SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveState, StructuralPattern,
    StructuralPatternView,
};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::{
    DaeBdf, DaeInitialization, DaeNumericalOptions, DaeOptions, DaePartition, DaeProblem,
    DaeReport, DaeState, DaeSystem, DaeTermination, DaeWorkspace, OdeStageBackend,
};
use scioxide_sparse::SparseLuLimits;

fn real<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}
fn failure(message: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("output"),
        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        message,
    )
}

/// x' = -x, 0 = z - x² - t.
struct Decay;
impl<T: NumericScalar> DaeSystem<T> for Decay {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        t: T::Real,
        y: &[T],
        d: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] + y[0];
        out[1] = y[1] - y[0] * y[0] - T::from_real(t);
        Ok(())
    }
    fn derivatives(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        a: &mut MatrixViewMut<'_, T>,
        b: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, T::ZERO)?;
                b.set(i, j, T::ZERO)?;
            }
        }
        a.set(0, 0, T::ONE)?;
        a.set(1, 0, -(y[0] + y[0]))?;
        a.set(1, 1, T::ONE)?;
        b.set(0, 0, T::ONE)?;
        Ok(Some(()))
    }
    fn sparse_derivatives(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        p: StructuralPatternView<'_>,
        a: &mut [T],
        b: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for k in p.row_ptr()[i]..p.row_ptr()[i + 1] {
                let j = p.col_indices()[k];
                a[k] = match (i, j) {
                    (0, 0) | (1, 1) => T::ONE,
                    (1, 0) => -(y[0] + y[0]),
                    _ => T::ZERO,
                };
                b[k] = if i == 0 && j == 0 { T::ONE } else { T::ZERO };
            }
        }
        Ok(Some(()))
    }
    fn derivative_product(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        v: &[T],
        w: &[T],
        out: &mut [T],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out[0] = v[0] + w[0];
        out[1] = v[1] - (y[0] + y[0]) * v[0];
        Ok(Some(()))
    }
}

fn partition() -> DaePartition {
    DaePartition::new(&[true, false], &[true, false]).unwrap()
}
fn numerics<S: RealScalar>() -> DaeNumericalOptions<S> {
    let error = ErrorTolerance::scalar(real(1e-4), S::ZERO, NormSelection::Infinity).unwrap();
    let residual = real::<S>(1e-10).max(real::<S>(32.) * S::EPSILON);
    let residual = ResidualTolerance::new(residual, S::ZERO).unwrap();
    DaeNumericalOptions::new()
        .with_differential_error(error.clone())
        .with_algebraic_error(error)
        .with_differential_residual(residual)
        .with_algebraic_residual(residual)
}
fn options<S: RealScalar>() -> DaeOptions<S> {
    DaeOptions::new().with_numerics(numerics())
}
fn backends() -> Vec<OdeStageBackend> {
    vec![
        OdeStageBackend::Dense,
        OdeStageBackend::Sparse {
            pattern: StructuralPattern::new(2, 2, vec![0, 1, 3], vec![0, 0, 1]).unwrap(),
            limits: SparseLuLimits::new(32, 1024 * 1024),
        },
        OdeStageBackend::Krylov {
            iterations: Budget::try_from(20).unwrap(),
            restart: Budget::try_from(2).unwrap(),
            operator_scratch: 0,
            preconditioner_scratch: 0,
        },
    ]
}
fn plan<T: NumericScalar>(options: DaeOptions<T::Real>) -> (DaeBdf, DaeOptions<T::Real>)
where
    T::Real: NumericScalar<Real = T::Real>,
{
    (
        DaeBdf::new(partition(), OdeStageBackend::Dense).unwrap(),
        options,
    )
}
fn state(time: f64, y: [f64; 2], d: [f64; 2]) -> DaeState<f64> {
    DaeState::from_slices(time, &y, &d).unwrap()
}
fn run<Sys: DaeSystem<f64> + ?Sized>(
    plan: &(DaeBdf, DaeOptions<f64>),
    system: &Sys,
    state: &mut DaeState<f64>,
    end: f64,
) -> Result<DaeReport<f64>, SciOxideError> {
    let mut workspace = plan.0.workspace().unwrap();
    plan.0.advance(
        &DaeProblem::new(system, end),
        &plan.1,
        state,
        &mut (),
        &mut workspace,
        ExecutionContext::sequential(),
    )
}
/// The algebraic constraint of [`Decay`] at the state's time.
fn constraint(state: &DaeState<f64>) -> f64 {
    let y = state.values();
    (y[1] - y[0] * y[0] - state.time()).abs()
}
/// `Check` initialization rejects the caller's pair as inconsistent input.
fn assert_inconsistent_state(error: &SciOxideError) {
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                argument,
                reason: scioxide_core::InvalidInputReason::Inconsistent,
                ..
            } if argument == scioxide_core::argument!("state")
        ),
        "{error:?}"
    );
}
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(623),
        scioxide_core::algorithm_id!("test", "dae", 1),
        2,
        0,
    )
}

#[test]
fn partition_rejects_invalid_roles() {
    assert!(DaePartition::new(&[], &[]).is_err());
    assert!(DaePartition::new(&[true, false], &[true, true]).is_err());
}

fn decay_case<T: NumericScalar>()
where
    T::Real: NumericScalar<Real = T::Real>,
{
    let rone = T::Real::ONE;
    let initial = T::from_components(rone, real::<T::Real>(0.25));
    for backend in backends() {
        let plan = (
            DaeBdf::new(partition(), backend).unwrap(),
            options().with_initial_step(real(0.2)),
        );
        let mut state = DaeState::from_slices(
            T::Real::ZERO,
            &[initial, initial * initial],
            &[-initial, T::ZERO],
        )
        .unwrap();
        let mut workspace = plan.0.workspace().unwrap();
        let report = plan
            .0
            .advance(
                &DaeProblem::new(&Decay, rone),
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            Some(DaeTermination::ReachedEnd),
            "{report:?}"
        );
        assert!(report.accepted_steps() > 1);
        assert_eq!(state.time(), rone);
        let y = state.values();
        let exact = initial * T::from_real((-rone).exp());
        let tolerances = options::<T::Real>();
        assert!(
            (y[0] - exact).abs()
                < real::<T::Real>(10.)
                    * tolerances
                        .numerics()
                        .differential_error()
                        .threshold(0, 1, exact.abs())
                        .unwrap(),
            "{y:?}"
        );
        assert!(
            (y[1] - exact * exact - T::ONE).abs()
                < real::<T::Real>(10.)
                    * tolerances
                        .numerics()
                        .algebraic_error()
                        .threshold(0, 1, (exact * exact + T::ONE).abs())
                        .unwrap(),
            "{y:?}"
        );
        assert!(
            report.algebraic_residual()
                <= tolerances
                    .numerics()
                    .algebraic_residual()
                    .map_or(T::Real::SQRT_EPSILON, |t| t.absolute())
        );
        assert!(report.differential_error() <= rone && report.algebraic_error() <= rone);
    }
}
#[test]
fn dae_real_and_complex_dense_sparse_operator() {
    decay_case::<f64>();
    decay_case::<f32>();
    decay_case::<Complex<f64>>();
    decay_case::<Complex<f32>>();
}

#[test]
fn dae_initialization_is_explicit_and_transactional() {
    let check = plan::<f64>(options());
    let project = plan::<f64>(
        options().with_numerics(numerics().with_initialization(DaeInitialization::Project)),
    );
    let mut initial = state(0., [1., 7.], [3., 19.]);
    let mut workspace = check.0.workspace().unwrap();
    let error = check
        .0
        .advance(
            &DaeProblem::new(&Decay, 0.),
            &check.1,
            &mut initial,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_inconsistent_state(&error);
    assert_eq!(initial, state(0., [1., 7.], [3., 19.]));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert!(!progress.has_accepted_state() && progress.termination().is_none());
    let observed = progress.consistency().unwrap();
    assert_eq!(observed.initial_differential_residual(), 4.);
    assert_eq!(observed.initial_algebraic_residual(), 6.);
    let report = run(&project, &Decay, &mut initial, 0.).unwrap();
    assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
    assert_eq!(initial.values(), [1., 1.]);
    assert_eq!(initial.derivative(), [-1., 19.]);
    let c = report.consistency().unwrap();
    assert_eq!(c.state_correction(), 6.);
    assert_eq!(c.derivative_correction(), 4.);
    let saved = initial.clone();
    let report = run(&check, &Decay, &mut initial, 0.).unwrap();
    assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
    assert_eq!(initial, saved);
}

/// Rank-deficient mixed initialization Jacobian.
struct Deficient;
impl DaeSystem<f64> for Deficient {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o.fill(0.);
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, 0.)?;
                b.set(i, j, 0.)?;
            }
        }
        b.set(0, 0, 1.)?;
        Ok(Some(()))
    }
    fn sparse_derivatives(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        pattern: StructuralPatternView<'_>,
        a: &mut [f64],
        b: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        a.fill(0.);
        b.fill(0.);
        let row = pattern.row_ptr()[0]..pattern.row_ptr()[1];
        for (&column, value) in pattern.col_indices()[row.clone()].iter().zip(&mut b[row]) {
            if column == 0 {
                *value = 1.;
            }
        }
        Ok(Some(()))
    }
}

#[test]
fn dae_rank_failure_is_not_consistency_success() {
    for backend in backends().into_iter().take(2) {
        let plan = (DaeBdf::new(partition(), backend).unwrap(), options());
        let mut workspace = plan.0.workspace().unwrap();
        let mut initial = state(0., [1., 2.], [0., 0.]);
        assert!(
            plan.0
                .advance(
                    &DaeProblem::new(&Deficient, 0.),
                    &plan.1,
                    &mut initial,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .is_err()
        );
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
        assert!(!attempt.progress().unwrap().has_accepted_state());
        assert_eq!(initial, state(0., [1., 2.], [0., 0.]));
    }
}

#[test]
fn dae_budget_admission_precedes_callbacks() {
    let plan = plan::<f64>(
        options()
            .with_initial_step(0.1)
            .with_max_evaluations(Budget::try_from(1).unwrap()),
    );
    let mut initial = state(0., [1., 1.], [-1., 0.]);
    let mut workspace = plan.0.workspace().unwrap();
    // A denial before the initial pair is accepted leaves no state to stop with.
    let error = plan
        .0
        .advance(
            &DaeProblem::new(&Decay, 1.),
            &plan.1,
            &mut initial,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. }
                if diagnostics.budget().map(|report| report.kind()) == Some(BudgetKind::Evaluation)
        ),
        "{error:?}"
    );
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert!(!progress.has_accepted_state() && progress.termination().is_none());
    assert_eq!(progress.counts().total_providers().unwrap(), 1);
    assert_eq!(initial, state(0., [1., 1.], [-1., 0.]));
}

#[test]
fn dae_sink_receives_committed_consistent_endpoints() {
    let plan = plan::<f64>(options().with_initial_step(0.1));
    let mut workspace = plan.0.workspace().unwrap();
    let mut initial = state(0., [1., 1.], [-1., 0.]);
    let mut calls = 0;
    let report = plan
        .0
        .advance(
            &DaeProblem::new(&Decay, 0.2),
            &plan.1,
            &mut initial,
            &mut |t: f64, y: &[f64], d: &[f64]| {
                assert!((y[1] - y[0] * y[0] - t).abs() < 1e-10);
                assert!((d[0] + y[0]).abs() < 1e-10);
                calls += 1;
                Ok(())
            },
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
    assert_eq!(calls, report.accepted_steps());
    assert_eq!(initial.time(), 0.2);
    assert_eq!(report.final_time(), 0.2);
}

#[test]
fn dae_prepared_calls_allocate_no_storage() {
    for backend in backends() {
        let plan = (
            DaeBdf::new(partition(), backend).unwrap(),
            options().with_initial_step(0.1),
        );
        let mut workspace = DaeWorkspace::new();
        let requirements = plan.0.workspace_requirements().unwrap();
        assert_eq!(requirements.dimension(), 2);
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            scioxide_core::Capacity::NeedsGrowth
        );
        workspace.try_reserve(&requirements).unwrap();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            scioxide_core::Capacity::Sufficient
        );
        let problem = DaeProblem::new(&Decay, 0.1);
        let mut initial = state(0., [1., 1.], [-1., 0.]);
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut initial,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Succeeded
        );
    }
}

#[test]
fn dae_independent_state_error_controls() {
    let mut counts = [0; 3];
    let mut errors = [0_f64; 3];
    for case in 0..3 {
        let plan = plan::<f64>(
            options()
                .with_numerics(
                    numerics()
                        .with_differential_error(
                            ErrorTolerance::scalar(
                                if case == 1 { 1e-5 } else { 1e-2 },
                                0.,
                                NormSelection::Infinity,
                            )
                            .unwrap(),
                        )
                        .with_algebraic_error(
                            ErrorTolerance::scalar(
                                if case == 2 { 1e-5 } else { 1e-2 },
                                0.,
                                NormSelection::Infinity,
                            )
                            .unwrap(),
                        ),
                )
                .with_initial_step(0.5),
        );
        let mut initial = state(0., [1., 1.], [-1., 0.]);
        let mut workspace = plan.0.workspace().unwrap();
        // Compare trajectory maxima: a loose solve can cancel its global error
        // at the final endpoint, especially with portable scalar arithmetic.
        let r = plan
            .0
            .advance(
                &DaeProblem::new(&Decay, 1.),
                &plan.1,
                &mut initial,
                &mut |time: f64, state: &[f64], _: &[f64]| {
                    errors[case] = errors[case].max((state[0] - (-time).exp()).abs());
                    Ok(())
                },
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(r.termination(), Some(DaeTermination::ReachedEnd));
        counts[case] = r.accepted_steps();
    }
    assert!(counts[1] > counts[0] && counts[2] > counts[0]);
    assert!(
        errors[1] < errors[0] * 0.2 && errors[2] < errors[0] * 0.2,
        "counts={counts:?}, errors={errors:?}"
    );
}

#[test]
fn dae_fresh_epochs_use_backward_euler_startup() {
    let mut errors = [0.; 3];
    for (case, n) in [4, 8, 16].into_iter().enumerate() {
        let h = 0.5 / n as f64;
        let loose = ErrorTolerance::scalar(1., 0., NormSelection::Infinity).unwrap();
        let plan = plan::<f64>(
            options()
                .with_numerics(
                    numerics()
                        .with_differential_error(loose.clone())
                        .with_algebraic_error(loose),
                )
                .with_initial_step(h),
        );
        let mut workspace = plan.0.workspace().unwrap();
        let mut current = state(0., [1., 1.], [-1., 0.]);
        for i in 0..n {
            let r = plan
                .0
                .advance(
                    &DaeProblem::new(&Decay, (i + 1) as f64 * h),
                    &plan.1,
                    &mut current,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(r.accepted_steps(), 1);
            assert_eq!(r.termination(), Some(DaeTermination::ReachedEnd));
        }
        let discrete = (1. + h).powi(-n);
        assert!((current.values()[0] - discrete).abs() < 1e-10);
        errors[case] = (current.values()[0] - (-0.5f64).exp()).abs();
    }
    for pair in errors.windows(2) {
        assert!(pair[0] / pair[1] > 1.8 && pair[0] / pair[1] < 2.1);
    }
}

#[test]
fn dae_backward_and_adjacent_times() {
    let tight = ErrorTolerance::scalar(1e-5, 0., NormSelection::Infinity).unwrap();
    let backward = plan::<f64>(
        options()
            .with_numerics(
                numerics()
                    .with_differential_error(tight.clone())
                    .with_algebraic_error(tight),
            )
            .with_initial_step(0.1),
    );
    let x = (-1.0f64).exp();
    let mut current = state(1., [x, (-2.0f64).exp() + 1.], [-x, 0.]);
    let r = run(&backward, &Decay, &mut current, 0.).unwrap();
    assert_eq!(r.termination(), Some(DaeTermination::ReachedEnd));
    assert!((current.values()[0] - 1.).abs() < 0.004);
    let adjacent = f64::from_bits(1.0f64.to_bits() + 1);
    for step in [Some(1.), None] {
        let options = options();
        let plan = plan::<f64>(match step {
            Some(step) => options.with_initial_step(step),
            None => options,
        });
        let mut current = state(1., [1., 2.], [-1., 0.]);
        let r = run(&plan, &Decay, &mut current, adjacent).unwrap();
        assert_eq!(r.termination(), Some(DaeTermination::ReachedEnd));
        assert_eq!(r.final_time(), adjacent);
        assert_eq!(current.time(), adjacent);
        assert!((current.values()[0] - 1.).abs() <= 2. * f64::EPSILON);
    }
}

/// The algebraic equation z² + 1 = 0 has no real solution.
struct Impossible;
impl DaeSystem<f64> for Impossible {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: f64,
        y: &[f64],
        d: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = d[0] + y[0];
        o[1] = y[1] * y[1] + 1.;
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, 0.)?;
                b.set(i, j, 0.)?;
            }
        }
        a.set(0, 0, 1.)?;
        a.set(1, 1, 2. * y[1])?;
        b.set(0, 0, 1.)?;
        Ok(Some(()))
    }
}
#[test]
fn dae_incompatible_projection_preserves_inputs() {
    let plan = plan::<f64>(
        options().with_numerics(numerics().with_initialization(DaeInitialization::Project)),
    );
    let mut initial = state(0., [1., 0.5], [9., 7.]);
    let r = run(&plan, &Impossible, &mut initial, 0.);
    assert!(!matches!(r, Ok(report) if report.is_success()));
    assert_eq!(initial, state(0., [1., 0.5], [9., 7.]));
}

/// [`Decay`] whose residual fails after a time.
struct FailAfter(f64);
impl DaeSystem<f64> for FailAfter {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        t: f64,
        y: &[f64],
        d: &[f64],
        o: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if t > self.0 {
            return Err(failure("intentional provider failure"));
        }
        Decay.residual(t, y, d, o, cx)
    }
    fn derivatives(
        &self,
        t: f64,
        y: &[f64],
        d: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        Decay.derivatives(t, y, d, a, b, cx)
    }
}
#[test]
fn dae_callback_and_sink_failures_retain_accepted_pair() {
    let plan = plan::<f64>(options().with_initial_step(0.05));
    let mut workspace = plan.0.workspace().unwrap();
    let mut current = state(0., [1., 1.], [-1., 0.]);
    assert!(
        plan.0
            .advance(
                &DaeProblem::new(&FailAfter(0.15), 1.),
                &plan.1,
                &mut current,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert!(progress.has_accepted_state());
    assert!(progress.accepted_steps() > 0);
    assert!(progress.final_time() <= 0.15);
    assert_eq!(current.time(), progress.final_time());
    assert!(constraint(&current) < 1e-10);

    let mut current = state(0., [1., 1.], [-1., 0.]);
    assert!(
        plan.0
            .advance(
                &DaeProblem::new(&Decay, 1.),
                &plan.1,
                &mut current,
                &mut |_: f64, _: &[f64], _: &[f64]| Err(failure("sink failure")),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    let progress = workspace.last_attempt().unwrap().progress().unwrap();
    assert_eq!(progress.accepted_steps(), 1);
    assert!(progress.final_time() > 0.);
    assert_eq!(current.time(), progress.final_time());
    assert!(constraint(&current) < 1e-10);
}

#[test]
fn dae_owned_solve_returns_partial_progress() {
    let plan = plan::<f64>(options().with_initial_step(0.05));
    let solution = plan
        .0
        .solve(
            &DaeProblem::new(&Decay, 0.2),
            &plan.1,
            state(0., [1., 1.], [-1., 0.]),
        )
        .unwrap();
    assert!(solution.report().is_success());
    assert_eq!(solution.state().time(), 0.2);
    assert!(constraint(solution.state()) < 1e-10);
    let (error, progress) = plan
        .0
        .solve(
            &DaeProblem::new(&FailAfter(0.15), 1.),
            &plan.1,
            state(0., [1., 1.], [-1., 0.]),
        )
        .unwrap_err()
        .into_parts();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    let progress = progress.unwrap();
    assert!(progress.has_accepted_state());
    assert!(progress.accepted_steps() > 0 && progress.final_time() <= 0.15);
    let failure = plan
        .0
        .solve(
            &DaeProblem::new(&Decay, 1.),
            &plan.1,
            DaeState::from_slices(0., &[1.], &[-1., 0.]).unwrap(),
        )
        .unwrap_err();
    assert!(failure.progress().is_none());
}

/// A residual that leaves its second component unwritten.
struct Unwritten;
impl DaeSystem<f64> for Unwritten {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = 0.;
        Ok(())
    }
}
#[test]
fn dae_invalid_shapes_options_and_unwritten_outputs() {
    let plan = plan::<f64>(options());
    let mut workspace = plan.0.workspace().unwrap();
    let mut initial = state(0., [1., 1.], [-1., 0.]);
    assert!(
        plan.0
            .advance(
                &DaeProblem::new(&Unwritten, 0.),
                &plan.1,
                &mut initial,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
    assert_eq!(initial, state(0., [1., 1.], [-1., 0.]));
    let mut short = DaeState::from_slices(0., &[1.], &[-1., 0.]).unwrap();
    assert!(
        plan.0
            .advance(
                &DaeProblem::new(&Decay, 0.),
                &plan.1,
                &mut short,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert!(
        plan.0
            .advance(
                &DaeProblem::new(&Decay, f64::INFINITY),
                &plan.1,
                &mut initial,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(initial, state(0., [1., 1.], [-1., 0.]));
    for step in [0., -1., f64::NAN] {
        assert!(
            run(
                &self::plan::<f64>(options().with_initial_step(step)),
                &Decay,
                &mut initial,
                1.0
            )
            .is_err()
        );
    }
    let vector = ErrorTolerance::try_vector(&[1e-4, 1e-4], 0., NormSelection::Infinity).unwrap();
    assert!(
        run(
            &self::plan::<f64>(options().with_numerics(numerics().with_algebraic_error(vector))),
            &Decay,
            &mut initial,
            1.0
        )
        .is_err()
    );
    assert!(
        run(
            &self::plan::<f64>(options().with_numerics(numerics().with_algebraic_scale(-1.))),
            &Decay,
            &mut initial,
            1.0
        )
        .is_err()
    );
    assert!(
        DaeBdf::new(
            partition(),
            OdeStageBackend::Sparse {
                pattern: StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 0]).unwrap(),
                limits: SparseLuLimits::new(32, 1024 * 1024),
            }
        )
        .is_err()
    );
}

#[test]
fn dae_ancestor_evaluation_budget_stops_a_child_epoch() {
    let plan = plan::<f64>(options().with_initial_step(0.01));
    let mut workspace = plan.0.workspace().unwrap();
    let mut current = state(0., [1., 1.], [-1., 0.]);
    let mut epoch = SolveState::new();
    // The whole integration takes 190 evaluations.
    let mut budgets = SolveBudgets::unbounded()
        .with_limit(BudgetKind::Evaluation, Budget::try_from(120).unwrap());
    let mut parent = epoch
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let r = plan
        .0
        .advance(
            &DaeProblem::new(&Decay, 1.),
            &plan.1,
            &mut current,
            &mut (),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert_eq!(
        r.termination(),
        Some(DaeTermination::BudgetLimit(BudgetKind::Evaluation))
    );
    assert!(r.has_accepted_state() && r.accepted_steps() > 0, "{r:?}");
    assert_eq!(parent.budget_report(BudgetKind::Evaluation).consumed(), 120);
    assert_eq!(current.time(), r.final_time());
    assert!(constraint(&current) < 1e-10);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
}

#[test]
fn dae_step_budget_stops_after_accepted_steps() {
    for steps in [1, 20] {
        let plan = plan::<f64>(
            options()
                .with_initial_step(0.01)
                .with_max_steps(Budget::try_from(steps).unwrap()),
        );
        let mut current = state(0., [1., 1.], [-1., 0.]);
        let r = run(&plan, &Decay, &mut current, 1.).unwrap();
        assert_eq!(
            r.termination(),
            Some(DaeTermination::BudgetLimit(BudgetKind::Proposal))
        );
        assert!(r.has_accepted_state());
        assert_eq!(r.counts().proposals(), steps);
        assert!(r.accepted_steps() <= steps);
        assert_eq!(current.time(), r.final_time());
        assert!(constraint(&current) < 1e-10);
    }
}

#[test]
fn dae_cancellation_after_an_endpoint_preserves_it() {
    let cancellation = scioxide_core::CancellationToken::new();
    let plan = plan::<f64>(options().with_initial_step(0.01));
    let mut workspace = plan.0.workspace().unwrap();
    let mut current = state(0., [1., 1.], [-1., 0.]);
    let r = plan
        .0
        .advance(
            &DaeProblem::new(&Decay, 1.),
            &plan.1,
            &mut current,
            &mut |_: f64, _: &[f64], _: &[f64]| {
                cancellation.cancel();
                Ok(())
            },
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert!(matches!(
        r.termination(),
        Some(DaeTermination::ExecutionStopped(_))
    ));
    assert_eq!(r.accepted_steps(), 1);
    assert!(constraint(&current) < 1e-10);
}

#[cfg(feature = "std")]
#[test]
fn dae_injected_executor_matches_sequential() {
    use scioxide_core::ThreadExecutor;
    let executor = ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    for backend in backends() {
        let plan = (
            DaeBdf::new(partition(), backend).unwrap(),
            options().with_initial_step(0.1),
        );
        let mut workspace = plan.0.workspace().unwrap();
        let problem = DaeProblem::new(&Decay, 0.2);
        let mut parallel = state(0., [1., 1.], [-1., 0.]);
        let r = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut parallel,
                &mut (),
                &mut workspace,
                ExecutionContext::new(&executor),
            )
            .unwrap();
        let mut sequential = state(0., [1., 1.], [-1., 0.]);
        let s = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut sequential,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(r.termination(), Some(DaeTermination::ReachedEnd));
        assert_eq!(parallel, sequential);
        assert_eq!(r.accounting(), s.accounting());
    }
}

/// Differential and algebraic roles in reversed positions.
struct Reordered;
impl DaeSystem<f64> for Reordered {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        t: f64,
        y: &[f64],
        d: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = y[0] - y[1] * y[1] - t;
        o[1] = d[1] + y[1];
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, 0.)?;
                b.set(i, j, 0.)?;
            }
        }
        a.set(0, 0, 1.)?;
        a.set(0, 1, -2. * y[1])?;
        a.set(1, 1, 1.)?;
        b.set(1, 1, 1.)?;
        Ok(Some(()))
    }
}
#[test]
fn dae_noncontiguous_roles_and_component_scales() {
    let plan = (
        DaeBdf::new(
            DaePartition::new(&[false, true], &[false, true]).unwrap(),
            OdeStageBackend::Dense,
        )
        .unwrap(),
        options()
            .with_numerics(numerics().with_differential_error(
                ErrorTolerance::try_vector(&[1e-4], 0., NormSelection::Infinity).unwrap(),
            ))
            .with_initial_step(0.1),
    );
    let mut current = state(0., [1., 1.], [0., -1.]);
    let r = run(&plan, &Reordered, &mut current, 0.2).unwrap();
    assert_eq!(r.termination(), Some(DaeTermination::ReachedEnd));
    let y = current.values();
    assert!((y[0] - y[1] * y[1] - 0.2).abs() < 1e-10);
}

/// Independent pairs x' = -x, z = x² + t with singular diagonal mass diag(1, 0).
struct Pairs(usize);
impl<T: NumericScalar> DaeSystem<T> for Pairs {
    fn dimension(&self) -> usize {
        2 * self.0
    }
    fn residual(
        &self,
        time: T::Real,
        y: &[T],
        d: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for ((y, d), out) in y
            .chunks_exact(2)
            .zip(d.chunks_exact(2))
            .zip(out.chunks_exact_mut(2))
        {
            out[0] = d[0] + y[0];
            out[1] = y[1] - y[0] * y[0] - T::from_real(time);
        }
        Ok(())
    }
    fn derivatives(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        a: &mut MatrixViewMut<'_, T>,
        b: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 * self.0 {
            for j in 0..2 * self.0 {
                a.set(i, j, T::ZERO)?;
                b.set(i, j, T::ZERO)?;
            }
        }
        for i in (0..2 * self.0).step_by(2) {
            a.set(i, i, T::ONE)?;
            a.set(i + 1, i, -(y[i] + y[i]))?;
            a.set(i + 1, i + 1, T::ONE)?;
            b.set(i, i, T::ONE)?;
        }
        Ok(Some(()))
    }
    fn sparse_derivatives(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        p: StructuralPatternView<'_>,
        a: &mut [T],
        b: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..2 * self.0 {
            for k in p.row_ptr()[row]..p.row_ptr()[row + 1] {
                let col = p.col_indices()[k];
                a[k] = if col == row {
                    T::ONE
                } else if row % 2 == 1 && col == row - 1 {
                    -(y[col] + y[col])
                } else {
                    T::ZERO
                };
                b[k] = if row.is_multiple_of(2) && row == col {
                    T::ONE
                } else {
                    T::ZERO
                };
            }
        }
        Ok(Some(()))
    }
    fn derivative_product(
        &self,
        _: T::Real,
        y: &[T],
        _: &[T],
        v: &[T],
        w: &[T],
        out: &mut [T],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in (0..2 * self.0).step_by(2) {
            out[i] = v[i] + w[i];
            out[i + 1] = v[i + 1] - (y[i] + y[i]) * v[i];
        }
        Ok(Some(()))
    }
}
fn pair_initial<T: NumericScalar>(pair: usize, pairs: usize) -> T {
    T::from_components(real(1. + pair as f64 / pairs as f64), real(0.25))
}
fn pair_reset<T: NumericScalar>(state: &mut [T], derivative: &mut [T], project: bool) {
    let pairs = state.len() / 2;
    for (i, (y, d)) in state
        .chunks_exact_mut(2)
        .zip(derivative.chunks_exact_mut(2))
        .enumerate()
    {
        y[0] = pair_initial(i, pairs);
        y[1] = if project { T::ZERO } else { y[0] * y[0] };
        d[0] = if project { T::ZERO } else { -y[0] };
        d[1] = T::ZERO;
    }
}
fn pair_error<T: NumericScalar>(state: &[T], time: f64) -> f64 {
    let pairs = state.len() / 2;
    state
        .chunks_exact(2)
        .enumerate()
        .map(|(i, y)| {
            let x = pair_initial::<T>(i, pairs) * T::from_real(real((-time).exp()));
            let z = x * x + T::from_real(real(time));
            (y[0] - x).abs().into_f64().max((y[1] - z).abs().into_f64())
        })
        .fold(0., f64::max)
}
fn manufactured_pairs<T: NumericScalar>()
where
    T::Real: NumericScalar<Real = T::Real>,
{
    let n = 8_usize;
    let roles: Vec<_> = (0..n).map(|i| i.is_multiple_of(2)).collect();
    let mut row_ptr = vec![0];
    let mut columns = vec![];
    for i in 0..n {
        if i % 2 == 1 {
            columns.push(i - 1);
        }
        columns.push(i);
        row_ptr.push(columns.len());
    }
    let residual = real::<T::Real>(1e-10).max(real::<T::Real>(64.) * T::Real::EPSILON);
    let residual = ResidualTolerance::new(residual, T::Real::ZERO).unwrap();
    for backend in [
        OdeStageBackend::Dense,
        OdeStageBackend::Sparse {
            pattern: StructuralPattern::new(n, n, row_ptr.clone(), columns.clone()).unwrap(),
            limits: SparseLuLimits::new(8 * n, 8 * 1024 * 1024),
        },
        OdeStageBackend::Krylov {
            iterations: Budget::try_from(32).unwrap(),
            restart: Budget::try_from(8).unwrap(),
            operator_scratch: 0,
            preconditioner_scratch: 0,
        },
    ] {
        let mut y = vec![T::ZERO; n];
        let mut d = vec![T::ZERO; n];
        let mut workspace = DaeWorkspace::new();
        let mut current = DaeState::from_slices(T::Real::ZERO, &y, &d).unwrap();
        for project in [true, false] {
            let plan = (
                DaeBdf::new(DaePartition::new(&roles, &roles).unwrap(), backend.clone()).unwrap(),
                options::<T::Real>()
                    .with_numerics(
                        numerics()
                            .with_differential_residual(residual)
                            .with_algebraic_residual(residual)
                            .with_initialization(if project {
                                DaeInitialization::Project
                            } else {
                                DaeInitialization::Check
                            }),
                    )
                    .with_initial_step(real(0.05)),
            );
            workspace
                .try_reserve(&plan.0.workspace_requirements().unwrap())
                .unwrap();
            pair_reset(&mut y, &mut d, project);
            current.reset(T::Real::ZERO, &y, &d).unwrap();
            let end = if project { 0. } else { 0.25 };
            let system = Pairs(n / 2);
            let problem = DaeProblem::new(&system, real(end));
            let report = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut current,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(
                report.termination(),
                Some(DaeTermination::ReachedEnd),
                "{report:?}"
            );
            let limit = if project { 1e-5 } else { 0.008 };
            assert!(pair_error(current.values(), end) <= limit);
        }
    }
}
#[test]
fn dae_manufactured_pairs_accuracy_and_allocation() {
    manufactured_pairs::<f32>();
    manufactured_pairs::<f64>();
    manufactured_pairs::<Complex<f32>>();
    manufactured_pairs::<Complex<f64>>();
}

/// Cubic differential residual and cubic algebraic constraint.
struct Nonlinear;
impl DaeSystem<f64> for Nonlinear {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let v = d[0] + y[0];
        out[0] = v * v * v + v;
        out[1] = y[1] * y[1] * y[1] + y[1] - y[0] * y[0] * y[0] - y[0];
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        y: &[f64],
        d: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let v = d[0] + y[0];
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, 0.)?;
                b.set(i, j, 0.)?;
            }
        }
        a.set(0, 0, 3. * v * v + 1.)?;
        b.set(0, 0, 3. * v * v + 1.)?;
        a.set(1, 0, -3. * y[0] * y[0] - 1.)?;
        a.set(1, 1, 3. * y[1] * y[1] + 1.)?;
        Ok(Some(()))
    }
}
#[test]
fn dae_nonlinear_derivative_and_constraint_projection() {
    let project = plan::<f64>(
        options().with_numerics(numerics().with_initialization(DaeInitialization::Project)),
    );
    let mut current = state(0., [1., 3.], [2., 0.]);
    let report = run(&project, &Nonlinear, &mut current, 0.).unwrap();
    assert_eq!(
        report.termination(),
        Some(DaeTermination::ReachedEnd),
        "{report:?}"
    );
    assert!((current.values()[1] - 1.).abs() < 1e-10);
    assert!((current.derivative()[0] + 1.).abs() < 1e-10);
    assert_eq!(current.values()[0], 1.);
    let check = plan::<f64>(options().with_initial_step(0.1));
    let report = run(&check, &Nonlinear, &mut current, 0.25).unwrap();
    assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
    let y = current.values();
    assert!((y[0] - (-0.25f64).exp()).abs() < 0.003);
    assert!((y[1] - y[0]).abs() < 1e-10);
}
#[test]
fn dae_residual_roles_are_independent() {
    for differential_strict in [false, true] {
        let (differential, algebraic) = if differential_strict {
            (1e-12, 0.1)
        } else {
            (0.1, 1e-12)
        };
        let plan = plan::<f64>(
            options().with_numerics(
                numerics()
                    .with_differential_residual(ResidualTolerance::new(differential, 0.).unwrap())
                    .with_algebraic_residual(ResidualTolerance::new(algebraic, 0.).unwrap()),
            ),
        );
        let mut current = state(0., [1., 1.001], [-0.999, 0.]);
        assert_inconsistent_state(&run(&plan, &Nonlinear, &mut current, 0.).unwrap_err());
        assert_eq!(current, state(0., [1., 1.001], [-0.999, 0.]));
    }
}

#[test]
fn dae_nonfinite_input_is_a_preflight_rejection() {
    let plan = plan::<f64>(options());
    let mut workspace = plan.0.workspace().unwrap();
    let mut current = state(0., [f64::INFINITY, 1.], [-1., 0.]);
    let error = plan
        .0
        .advance(
            &DaeProblem::new(&Decay, 0.),
            &plan.1,
            &mut current,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.progress().is_none());
    assert_eq!(
        attempt.counts().total().unwrap().total_providers().unwrap(),
        0
    );
}

/// x' = -x, 0 = z - x.
struct LinearDecay;
impl<T: NumericScalar> DaeSystem<T> for LinearDecay {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: T::Real,
        y: &[T],
        d: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] + y[0];
        out[1] = y[1] - y[0];
        Ok(())
    }
    fn derivatives(
        &self,
        _: T::Real,
        _: &[T],
        _: &[T],
        a: &mut MatrixViewMut<'_, T>,
        b: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, T::ZERO)?;
                b.set(i, j, T::ZERO)?;
            }
        }
        a.set(0, 0, T::ONE)?;
        a.set(1, 0, -T::ONE)?;
        a.set(1, 1, T::ONE)?;
        b.set(0, 0, T::ONE)?;
        Ok(Some(()))
    }
}
fn default_decay<S: RealScalar + NumericScalar<Real = S>>() {
    for step in [Some(real::<S>(0.1)), None] {
        let options = DaeOptions::new();
        let options = match step {
            Some(step) => options.with_initial_step(step),
            None => options,
        };
        let plan = (
            DaeBdf::new(partition(), OdeStageBackend::Dense).unwrap(),
            options.clone(),
        );
        let mut current =
            DaeState::from_slices(S::ZERO, &[S::ONE, S::ONE], &[-S::ONE, S::ZERO]).unwrap();
        let mut workspace = plan.0.workspace().unwrap();
        let report = plan
            .0
            .advance(
                &DaeProblem::new(&LinearDecay, S::ONE),
                &plan.1,
                &mut current,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
        let exact = RealScalar::exp(-S::ONE);
        let threshold = options
            .numerics()
            .differential_error()
            .threshold(0, 1, exact)
            .unwrap();
        let y = current.values();
        assert!(
            NumericScalar::abs(y[0] - exact) <= S::TEN * threshold,
            "{report:?}, y={y:?}, exact={exact:?}"
        );
        assert!(NumericScalar::abs(y[1] - exact) <= S::TEN * threshold);
        assert!(report.accepted_steps() < 200, "{report:?}");
    }
}
#[test]
fn dae_default_tolerance_meets_decay_accuracy() {
    default_decay::<f64>();
    default_decay::<f32>();
}

#[test]
fn dae_termination_shares_the_time_evolution_vocabulary() {
    let shared = [
        DaeTermination::ReachedEnd,
        DaeTermination::TerminalEvent,
        DaeTermination::BudgetLimit(BudgetKind::Proposal),
        DaeTermination::StepSizeLimit,
        DaeTermination::ExecutionStopped(scioxide_core::ExecutionStopReason::Cancelled),
    ];
    assert!(shared[0].is_success() && shared[1].is_success());
    assert!(
        shared[2..]
            .iter()
            .all(|termination| !termination.is_success())
    );
    assert!(!DaeTermination::RejectionLimit.is_success());
}

/// Deliberately supplies only residuals, exercising the default fallback.
struct ResidualOnly;
impl<T: NumericScalar> DaeSystem<T> for ResidualOnly {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        t: T::Real,
        y: &[T],
        v: &[T],
        out: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Decay.residual(t, y, v, out, cx)
    }
}
#[test]
fn residual_only_strategies_and_prepared_allocations() {
    use scioxide_differentiate::DerivativeStrategy;
    for strategy in [
        DerivativeStrategy::AnalyticOrAutomaticOrForward,
        DerivativeStrategy::Forward,
        DerivativeStrategy::Central,
    ] {
        for backend in backends() {
            let options = options()
                .with_numerics(numerics().with_derivative_strategy(strategy))
                .with_initial_step(0.01);
            assert_eq!(options.numerics().derivative_strategy(), strategy);
            let plan = (DaeBdf::new(partition(), backend).unwrap(), options);
            let mut workspace = plan.0.workspace().unwrap();
            let mut initial = state(0., [1., 1.], [-1., 0.]);
            let report = plan
                .0
                .advance(
                    &DaeProblem::new(&ResidualOnly, 0.1),
                    &plan.1,
                    &mut initial,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(
                report.termination(),
                Some(DaeTermination::ReachedEnd),
                "{strategy:?}"
            );
            assert!((initial.values()[0] - (-0.1_f64).exp()).abs() < 1e-3);
            assert!(constraint(&initial) < 1e-9);
            assert!(
                report
                    .accounting()
                    .total()
                    .unwrap()
                    .provider(scioxide_core::EvaluationRole::Residual)
                    > report.accepted_steps()
            );
        }
    }
    for strategy in [DerivativeStrategy::Analytic, DerivativeStrategy::Automatic] {
        let plan =
            plan::<f64>(options().with_numerics(numerics().with_derivative_strategy(strategy)));
        assert!(
            run(
                &plan,
                &ResidualOnly,
                &mut state(0., [1., 1.], [-1., 0.]),
                0.1
            )
            .is_err()
        );
    }
}

struct CountedResidual(core::cell::Cell<usize>);
impl DaeSystem<f64> for CountedResidual {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        t: f64,
        y: &[f64],
        v: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.0.set(self.0.get() + 1);
        Decay.residual(t, y, v, out, cx)
    }
}
#[test]
fn difference_residuals_are_admitted_and_counted_before_callbacks() {
    use scioxide_core::EvaluationRole;
    use scioxide_differentiate::DerivativeStrategy;
    for strategy in [
        DerivativeStrategy::AnalyticOrAutomaticOrForward,
        DerivativeStrategy::Forward,
        DerivativeStrategy::Central,
    ] {
        let system = CountedResidual(core::cell::Cell::new(0));
        let plan = plan::<f64>(
            options()
                .with_numerics(numerics().with_derivative_strategy(strategy))
                .with_max_evaluations(Budget::try_from(3).unwrap()),
        );
        let mut initial = state(0., [1., 1.], [-1., 0.]);
        let mut workspace = plan.0.workspace().unwrap();
        // The allowance ends during initialization, before any pair is accepted.
        assert!(
            plan.0
                .advance(
                    &DaeProblem::new(&system, 0.1),
                    &plan.1,
                    &mut initial,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .is_err()
        );
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
        let counts = attempt.progress().unwrap().accounting().total().unwrap();
        assert_eq!(counts.total_providers().unwrap(), 3);
        assert_eq!(counts.provider(EvaluationRole::Residual), system.0.get());
        assert_eq!(
            counts.provider(EvaluationRole::Jacobian),
            usize::from(strategy == DerivativeStrategy::AnalyticOrAutomaticOrForward)
        );
        assert_eq!(initial.time(), 0.);
    }
}

struct InvalidDerivative(bool);
impl DaeSystem<f64> for InvalidDerivative {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        t: f64,
        y: &[f64],
        v: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Decay.residual(t, y, v, out, cx)
    }
    fn derivatives(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        _: &mut MatrixViewMut<'_, f64>,
        _: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        if self.0 {
            Err(failure("derivative provider failed"))
        } else {
            Ok(Some(()))
        }
    }
}
#[test]
fn derivative_errors_and_unwritten_blocks_do_not_request_fallback() {
    for fail in [false, true] {
        let plan = plan::<f64>(options());
        let result = run(
            &plan,
            &InvalidDerivative(fail),
            &mut state(0., [1., 1.], [-1., 0.]),
            0.1,
        );
        let error = result.unwrap_err();
        if fail {
            assert_eq!(error, failure("derivative provider failed"));
        } else {
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }
}

fn residual_scalar<T: NumericScalar>()
where
    T::Real: NumericScalar<Real = T::Real>,
{
    for backend in backends() {
        let plan = (
            DaeBdf::new(partition(), backend).unwrap(),
            options::<T::Real>().with_initial_step(real(0.01)),
        );
        let initial =
            DaeState::from_slices(real(0.), &[T::ONE, T::ONE], &[-T::ONE, T::ZERO]).unwrap();
        let solution = plan
            .0
            .solve(&DaeProblem::new(&ResidualOnly, real(0.1)), &plan.1, initial)
            .unwrap();
        assert!(solution.report().is_success());
        assert!(
            (solution.state().values()[0] - T::from_real(real::<T::Real>(-0.1).exp())).abs()
                < real(1e-3)
        );
    }
}
#[test]
fn residual_only_real_and_complex_precisions() {
    residual_scalar::<f32>();
    residual_scalar::<f64>();
    residual_scalar::<Complex<f32>>();
    residual_scalar::<Complex<f64>>();
}

#[test]
fn default_nonlinear_workflow_retains_reuse_without_excessive_step_holds() {
    let plan = (
        DaeBdf::new(partition(), OdeStageBackend::Dense).unwrap(),
        DaeOptions::new().with_initial_step(0.05),
    );
    let mut current = state(0., [1., 1.], [-1., 0.]);
    let report = run(&plan, &Decay, &mut current, 0.25).unwrap();
    assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
    assert!((current.values()[0] - (-0.25_f64).exp()).abs() < 1e-7);
    assert!((current.values()[1] - (-0.5_f64).exp() - 0.25).abs() < 1e-7);
    // Holding every new width for q+1 steps over-resolved this smooth problem
    // with 90 steps. Retain adaptation while exercising cross-step block reuse.
    assert!(report.accepted_steps() <= 50, "{report:?}");
    assert!(report.counts().provider(EvaluationRole::Jacobian) < report.accepted_steps() / 2);
}

/// Robertson's chemical kinetics with the mass-conservation constraint `y₁ + y₂ + y₃ = 1`.
struct Robertson;
impl<S: RealScalar + NumericScalar<Real = S>> DaeSystem<S> for Robertson {
    fn dimension(&self) -> usize {
        3
    }
    fn residual(
        &self,
        _: S,
        y: &[S],
        d: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let a = real::<S>(0.04) * y[0] - real::<S>(1e4) * y[1] * y[2];
        out[0] = d[0] + a;
        out[1] = d[1] - a + real::<S>(3e7) * y[1] * y[1];
        out[2] = y[0] + y[1] + y[2] - S::ONE;
        Ok(())
    }
    fn derivatives(
        &self,
        _: S,
        y: &[S],
        _: &[S],
        a: &mut MatrixViewMut<'_, S>,
        b: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let (k1, k2, k3) = (real::<S>(0.04), real::<S>(1e4), real::<S>(3e7));
        let rows = [
            [k1, -k2 * y[2], -k2 * y[1]],
            [-k1, k2 * y[2] + S::TWO * k3 * y[1], k2 * y[1]],
            [S::ONE, S::ONE, S::ONE],
        ];
        for (i, row) in rows.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                a.set(i, j, value)?;
                b.set(i, j, if i == j && i < 2 { S::ONE } else { S::ZERO })?;
            }
        }
        Ok(Some(()))
    }
}
/// Robertson from `(1, 0, 0)` with the stiff reference tolerances `atol = 10⁻⁴ rtol`.
fn robertson(rtol: f64, end: f64, initial_step: Option<f64>) -> (DaeReport<f64>, DaeState<f64>) {
    let tolerance =
        ErrorTolerance::scalar(1e-4 * rtol, rtol, NormSelection::RootMeanSquare).unwrap();
    let options = DaeOptions::new().with_numerics(
        DaeNumericalOptions::new()
            .with_differential_error(tolerance.clone())
            .with_algebraic_error(tolerance),
    );
    let options = match initial_step {
        Some(step) => options.with_initial_step(step),
        None => options,
    };
    let plan = (
        DaeBdf::new(
            DaePartition::new(&[true, true, false], &[true, true, false]).unwrap(),
            OdeStageBackend::Dense,
        )
        .unwrap(),
        options,
    );
    let mut current = DaeState::from_slices(0., &[1., 0., 0.], &[-0.04, 0.04, 0.]).unwrap();
    let report = run(&plan, &Robertson, &mut current, end).unwrap();
    (report, current)
}
/// Reference components of a `tests/data/stiff.tsv` problem.
fn stiff_reference(problem: &str) -> Vec<f64> {
    let row = include_str!("data/stiff.tsv")
        .lines()
        .find(|line| line.split_whitespace().next() == Some(problem))
        .unwrap();
    let column = row.split_whitespace().nth(5).unwrap();
    column
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect()
}
/// Largest error in units of `rtol (10⁻⁴ + |reference|)`.
fn error_units(values: &[f64], reference: &[f64], rtol: f64) -> f64 {
    values
        .iter()
        .zip(reference)
        .map(|(value, r)| (value - r).abs() / (rtol * (1e-4 + r.abs())))
        .fold(0., f64::max)
}

#[test]
fn dae_default_first_step_integrates_robertson_over_every_span() {
    // A tight run with an explicit first step supplies the reference at t = 4·10⁵; at
    // t = 40 it agrees with the checked-in reference to a tenth of rtol = 10⁻⁸.
    let (tight_report, tight) = robertson(1e-11, 4e5, Some(1e-12));
    assert!(tight_report.is_success());
    let (_, tight_40) = robertson(1e-11, 40., Some(1e-12));
    let tight_units = error_units(tight_40.values(), &stiff_reference("rober40"), 1e-8);
    assert!(tight_units < 0.1, "{tight_units}");
    for (end, reference) in [
        (40., stiff_reference("rober40")),
        (4e5, tight.values().to_vec()),
        (1e11, stiff_reference("rober")),
    ] {
        for rtol in [1e-6, 1e-8] {
            let (report, current) = robertson(rtol, end, None);
            assert_eq!(
                report.termination(),
                Some(DaeTermination::ReachedEnd),
                "end={end} rtol={rtol}: {report:?}"
            );
            assert_eq!(current.time(), end);
            assert!(report.rejected_steps() <= 10, "{report:?}");
            let units = error_units(current.values(), &reference, rtol);
            // To 4·10⁵ the decaying y₁ accumulates about ten units of global error at
            // rtol = 10⁻⁸ whatever the first step; an explicit 10⁻⁶ start measures it.
            let (_, explicit) = robertson(rtol, end, Some(1e-6));
            let baseline = error_units(explicit.values(), &reference, rtol);
            assert!(
                units <= 5_f64.max(1.05 * baseline) && units <= 12.,
                "end={end} rtol={rtol}: {units} (explicit start {baseline})"
            );
        }
    }
}

#[test]
fn dae_default_first_step_reaches_the_end_of_long_decays() {
    let options = DaeOptions::<f64>::new();
    let threshold = options
        .numerics()
        .differential_error()
        .threshold(0, 1, 0.)
        .unwrap();
    for end in [1e2, 1e3, 1e4] {
        let plan = (
            DaeBdf::new(partition(), OdeStageBackend::Dense).unwrap(),
            options.clone(),
        );
        let mut current = state(0., [1., 1.], [-1., 0.]);
        let report = run(&plan, &LinearDecay, &mut current, end).unwrap();
        assert_eq!(
            report.termination(),
            Some(DaeTermination::ReachedEnd),
            "{report:?}"
        );
        assert_eq!(current.time(), end);
        let y = current.values();
        assert!((y[0] - (-end).exp()).abs() <= threshold, "{y:?}");
        assert!((y[1] - y[0]).abs() <= threshold);
    }
}

/// [`LinearDecay`] or the ramp `x' = 1, 0 = z - x`, recording every proposal's endpoint.
struct Proposals {
    ramp: bool,
    pending: core::cell::Cell<bool>,
    endpoints: core::cell::RefCell<Vec<f64>>,
}
impl Proposals {
    fn new(ramp: bool) -> Self {
        Self {
            ramp,
            pending: core::cell::Cell::new(false),
            endpoints: core::cell::RefCell::new(Vec::new()),
        }
    }
}
impl DaeSystem<f64> for Proposals {
    fn dimension(&self) -> usize {
        2
    }
    fn step_limit(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<f64>, SciOxideError> {
        self.pending.set(true);
        Ok(None)
    }
    fn residual(
        &self,
        t: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if self.pending.replace(false) {
            self.endpoints.borrow_mut().push(t);
        }
        if self.ramp {
            out[0] = d[0] - 1.;
            out[1] = y[1] - y[0];
            Ok(())
        } else {
            LinearDecay.residual(t, y, d, out, cx)
        }
    }
    fn derivatives(
        &self,
        t: f64,
        y: &[f64],
        d: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        LinearDecay.derivatives(t, y, d, a, b, cx)?;
        if self.ramp {
            a.set(0, 0, 0.)?;
        }
        Ok(Some(()))
    }
}

#[test]
fn dae_default_first_step_is_bounded_by_the_initial_rate_norm() {
    let absolute = ErrorTolerance::scalar(1e-6, 0., NormSelection::Infinity).unwrap();
    let relative = ErrorTolerance::scalar(0., 1e-6, NormSelection::Infinity).unwrap();
    // h₀ = min(span/1000, 0.5/‖ẋ₀‖) with ‖ẋ₀‖ = 1/10⁻⁶; the ramp starts at x₀ = 0, where
    // the relative threshold is exactly zero: the moving component carries no scale and
    // bounds h₀ by the unscaled starting step 10⁻⁶.
    for (ramp, tolerance, end, expected) in [
        (false, absolute.clone(), 1., 5e-7),
        (false, absolute.clone(), 1e-4, 1e-7),
        (true, relative.clone(), 1., 1e-6),
        (true, relative, 1e-4, 1e-7),
    ] {
        let plan = plan::<f64>(
            DaeOptions::new().with_numerics(
                DaeNumericalOptions::new()
                    .with_differential_error(tolerance)
                    .with_algebraic_error(absolute.clone()),
            ),
        );
        let system = Proposals::new(ramp);
        let mut current = if ramp {
            state(0., [0., 0.], [1., 0.])
        } else {
            state(0., [1., 1.], [-1., 0.])
        };
        let report = run(&plan, &system, &mut current, end).unwrap();
        assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
        let first = system.endpoints.borrow()[0];
        assert!(
            (first - expected).abs() <= 4. * f64::EPSILON * expected,
            "{first} vs {expected}"
        );
    }
}

#[test]
fn dae_startup_estimate_rejects_an_oversized_first_step_few_times() {
    let threshold = DaeOptions::<f64>::new()
        .numerics()
        .differential_error()
        .threshold(0, 1, 1.)
        .unwrap();
    // The supplied algebraic rate is either consistent (-1) or the unused zero.
    for algebraic_rate in [-1., 0.] {
        let plan = plan::<f64>(DaeOptions::new().with_initial_step(0.01));
        let system = Proposals::new(false);
        let mut workspace = plan.0.workspace().unwrap();
        let mut current = state(0., [1., 1.], [-1., algebraic_rate]);
        let mut first = None;
        let report = plan
            .0
            .advance(
                &DaeProblem::new(&system, 1.),
                &plan.1,
                &mut current,
                &mut |t: f64, y: &[f64], _: &[f64]| {
                    first.get_or_insert((t, y[0], system.endpoints.borrow().len()));
                    Ok(())
                },
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
        let (time, value, proposals) = first.unwrap();
        // The BDF1 local error h²/2 ≤ tolerance/4 holds near h = 10⁻⁴, three or four
        // shrinkages by at most the minimum factor 0.2 from h = 10⁻².
        assert!(proposals - 1 <= 4, "{proposals} proposals");
        assert!(time < 4e-4, "{time}");
        assert!((value - (-time).exp()).abs() <= threshold);
    }
}

/// [`Robertson`] without derivative callbacks: blocks come from residual differences.
struct ResidualRobertson;
impl DaeSystem<f32> for ResidualRobertson {
    fn dimension(&self) -> usize {
        3
    }
    fn residual(
        &self,
        t: f32,
        y: &[f32],
        d: &[f32],
        out: &mut [f32],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Robertson.residual(t, y, d, out, cx)
    }
}

#[test]
fn dae_f32_robertson_factors_stiff_stages_without_a_defect_target() {
    // The mass balance resolves y₃ only to the absolute rounding of y₁ ≈ 1, a few
    // ulp(1) ≈ 6·10⁻⁸, so an algebraic absolute tolerance of 10⁻⁶ is attainable and the
    // stiff reference tolerance 10⁻⁴·rtol is not. Below it the steps shrink until a
    // backward-stable LU solve leaves a relative defect near κ·ε, above any useful
    // forcing; the integration then stops with a typed report instead of an error.
    let reference = stiff_reference("rober40");
    for rtol in [1e-3_f32, 1e-4, 1e-5] {
        let tolerance = |absolute| {
            ErrorTolerance::scalar(absolute, rtol, NormSelection::RootMeanSquare).unwrap()
        };
        for algebraic in [1e-6, 1e-4 * rtol] {
            let plan = (
                DaeBdf::new(
                    DaePartition::new(&[true, true, false], &[true, true, false]).unwrap(),
                    OdeStageBackend::Dense,
                )
                .unwrap(),
                DaeOptions::new().with_numerics(
                    DaeNumericalOptions::new()
                        .with_differential_error(tolerance(1e-4 * rtol))
                        .with_algebraic_error(tolerance(algebraic)),
                ),
            );
            let mut workspace = plan.0.workspace().unwrap();
            let problems: [&dyn DaeSystem<f32>; 2] = [&Robertson, &ResidualRobertson];
            for system in problems {
                let mut current =
                    DaeState::from_slices(0., &[1., 0., 0.], &[-0.04, 0.04, 0.]).unwrap();
                let report = plan
                    .0
                    .advance(
                        &DaeProblem::new(system, 40.),
                        &plan.1,
                        &mut current,
                        &mut (),
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                if algebraic < 1e-6 {
                    assert!(report.has_accepted_state(), "{report:?}");
                    continue;
                }
                assert_eq!(
                    report.termination(),
                    Some(DaeTermination::ReachedEnd),
                    "rtol={rtol}: {report:?}"
                );
                let values: Vec<f64> = current.values().iter().map(|&y| f64::from(y)).collect();
                let units = error_units(&values, &reference, f64::from(rtol));
                assert!(units <= 2., "rtol={rtol}: {units}");
            }
        }
    }
}

/// `x' = -x` with the unsolvable constraint `0 = z² + 1`.
struct Unsolvable;
impl DaeSystem<f64> for Unsolvable {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] + y[0];
        out[1] = y[1] * y[1] + 1.;
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, 0.)?;
                b.set(i, j, 0.)?;
            }
        }
        a.set(0, 0, 1.)?;
        a.set(1, 1, 2. * y[1])?;
        b.set(0, 0, 1.)?;
        Ok(Some(()))
    }
}

#[test]
fn dae_failed_projection_is_a_convergence_failure_without_accepted_state() {
    let limit = 6;
    let plan = plan::<f64>(
        options().with_numerics(
            numerics()
                .with_initialization(DaeInitialization::Project)
                .with_newton_limit(Budget::try_from(limit).unwrap()),
        ),
    );
    let mut workspace = plan.0.workspace().unwrap();
    let mut initial = state(0., [1., 0.7], [-1., 0.]);
    let error = plan
        .0
        .advance(
            &DaeProblem::new(&Unsolvable, 1.),
            &plan.1,
            &mut initial,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    let scioxide_core::SciOxideErrorView::ConvergenceFailure {
        reason,
        diagnostics,
        ..
    } = error.view()
    else {
        panic!("{error:?}");
    };
    assert!(
        matches!(
            reason,
            scioxide_core::ConvergenceReason::IterationLimit
                | scioxide_core::ConvergenceReason::Divergence
        ),
        "{error:?}"
    );
    let iterations = diagnostics.counts().own().iterations();
    assert!((1..=limit).contains(&iterations), "{error:?}");
    assert!(diagnostics.budget().is_none());
    assert_eq!(initial, state(0., [1., 0.7], [-1., 0.]));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert!(!progress.has_accepted_state() && progress.termination().is_none());
    assert!(progress.consistency().is_some());
}

#[test]
fn dae_cancellation_before_the_initial_pair_is_an_error() {
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let plan = plan::<f64>(options());
    let mut workspace = plan.0.workspace().unwrap();
    let mut initial = state(0., [1., 1.], [-1., 0.]);
    let error = plan
        .0
        .advance(
            &DaeProblem::new(&Decay, 1.),
            &plan.1,
            &mut initial,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped {
            reason: scioxide_core::ExecutionStopReason::Cancelled,
            ..
        }
    ));
    assert_eq!(initial, state(0., [1., 1.], [-1., 0.]));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert!(!progress.has_accepted_state() && progress.termination().is_none());
    assert_eq!(
        progress.trust(),
        scioxide_core::PartialResultTrust::DiagnosticOnly
    );
}

/// `x' = -k x`, `0 = z - x` with analytic blocks whose differential row is scaled by
/// `factor`, so that the chord iteration contracts at `θ = |1 - 1/factor|`.
struct ScaledJacobian {
    k: f64,
    factor: f64,
}
impl DaeSystem<f64> for ScaledJacobian {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] + self.k * y[0];
        out[1] = y[1] - y[0];
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for (i, j, state, rate) in [
            (0, 0, self.factor * self.k, self.factor),
            (0, 1, 0., 0.),
            (1, 0, -1., 0.),
            (1, 1, 1., 0.),
        ] {
            a.set(i, j, state)?;
            b.set(i, j, rate)?;
        }
        Ok(Some(()))
    }
}

#[test]
fn dae_inexact_jacobian_raises_the_order_past_newton_limited_steps() {
    // θ ≈ 0.67 with the 10⁻⁸ default residual blocks needs every Newton retry at order
    // one near h = 3·10⁻⁸; the order must rise there instead of cycling through failures.
    let rtol = 1e-6;
    let tolerance = ErrorTolerance::scalar(1e-10, rtol, NormSelection::RootMeanSquare).unwrap();
    for initial_step in [None, Some(1e-8), Some(1e-7), Some(1e-6)] {
        let options = DaeOptions::new().with_numerics(
            DaeNumericalOptions::new()
                .with_differential_error(tolerance.clone())
                .with_algebraic_error(tolerance.clone()),
        );
        let options = match initial_step {
            Some(step) => options.with_initial_step(step),
            None => options,
        };
        let plan = plan::<f64>(options);
        let system = ScaledJacobian {
            k: 1e3,
            factor: 0.6,
        };
        let mut current = state(0., [1., 1.], [-1e3, 0.]);
        let report = run(&plan, &system, &mut current, 5e-3).unwrap();
        assert_eq!(
            report.termination(),
            Some(DaeTermination::ReachedEnd),
            "{initial_step:?}: {report:?}"
        );
        assert!(
            report.accepted_steps() <= 200 && report.rejected_steps() <= 20,
            "{initial_step:?}: {report:?}"
        );
        let exact = (-5_f64).exp();
        let y = current.values();
        assert!((y[0] - exact).abs() <= 30. * rtol * exact, "{y:?}");
        assert!((y[1] - y[0]).abs() <= 30. * rtol * exact);
    }
}

/// `x' = 1 - x`, alone or with the algebraic copy `0 = z - x`.
struct Relaxation {
    algebraic: bool,
}
impl DaeSystem<f64> for Relaxation {
    fn dimension(&self) -> usize {
        if self.algebraic { 2 } else { 1 }
    }
    fn residual(
        &self,
        _: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] - (1. - y[0]);
        if self.algebraic {
            out[1] = y[1] - y[0];
        }
        Ok(())
    }
}

#[test]
fn dae_pure_relative_tolerance_starts_from_a_zero_state() {
    let rtol = 1e-6;
    let relative = ErrorTolerance::scalar(0., rtol, NormSelection::RootMeanSquare).unwrap();
    for algebraic in [true, false] {
        let roles: &[bool] = if algebraic { &[true, false] } else { &[true] };
        for end in [1e1, 1e5] {
            for initial_step in [None, Some(1e-6)] {
                let options = DaeOptions::new().with_numerics(
                    DaeNumericalOptions::new()
                        .with_differential_error(relative.clone())
                        .with_algebraic_error(relative.clone()),
                );
                let options = match initial_step {
                    Some(step) => options.with_initial_step(step),
                    None => options,
                };
                let plan = (
                    DaeBdf::new(
                        DaePartition::new(roles, roles).unwrap(),
                        OdeStageBackend::Dense,
                    )
                    .unwrap(),
                    options,
                );
                let mut current =
                    DaeState::from_slices(0., &[0., 0.][..roles.len()], &[1., 0.][..roles.len()])
                        .unwrap();
                let report = run(&plan, &Relaxation { algebraic }, &mut current, end).unwrap();
                assert_eq!(
                    report.termination(),
                    Some(DaeTermination::ReachedEnd),
                    "algebraic={algebraic} end={end} {initial_step:?}: {report:?}"
                );
                assert!(report.rejected_steps() <= 6, "{report:?}");
                let exact = 1. - (-end).exp();
                for &value in current.values() {
                    assert!(
                        (value - exact).abs() <= 10. * rtol * exact,
                        "{value} vs {exact}"
                    );
                }
            }
        }
    }
}

/// `x' = 10¹⁰`, `0 = z - x`.
struct Ramp;
impl DaeSystem<f64> for Ramp {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] - 1e10;
        out[1] = y[1] - y[0];
        Ok(())
    }
}

#[test]
fn dae_overflowing_initial_rate_norm_starts_at_the_smallest_normal_step() {
    // ‖ẋ₀‖ = 10¹⁰/10⁻³⁰⁰ exceeds the scalar range: the rate norm is infinite.
    let tolerance = ErrorTolerance::scalar(1e-300, 1e-6, NormSelection::RootMeanSquare).unwrap();
    let plan = plan::<f64>(
        DaeOptions::new().with_numerics(
            DaeNumericalOptions::new()
                .with_differential_error(tolerance.clone())
                .with_algebraic_error(tolerance),
        ),
    );
    let mut workspace = plan.0.workspace().unwrap();
    let mut current = state(0., [0., 0.], [1e10, 0.]);
    let mut first = None;
    let report = plan
        .0
        .advance(
            &DaeProblem::new(&Ramp, 1.),
            &plan.1,
            &mut current,
            &mut |t: f64, _: &[f64], _: &[f64]| {
                first.get_or_insert(t);
                Ok(())
            },
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
    assert_eq!(first, Some(f64::MIN_POSITIVE));
    for &value in current.values() {
        assert!((value - 1e10).abs() <= 1e-6 * 1e10, "{value}");
    }
}

/// [`LinearDecay`] whose matrix-free product adds one to its first entry away from `t = 0`:
/// an affine operator whose GMRES solutions miss their verified defect.
struct OffsetProduct;
impl DaeSystem<f64> for OffsetProduct {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        t: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        LinearDecay.residual(t, y, d, out, cx)
    }
    fn derivative_product(
        &self,
        t: f64,
        _: &[f64],
        _: &[f64],
        v: &[f64],
        w: &[f64],
        out: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out[0] = v[0] + w[0] + if t == 0. { 0. } else { 1. };
        out[1] = v[1] - v[0];
        Ok(Some(()))
    }
}

#[test]
fn dae_unverified_gmres_correction_rejects_the_proposal() {
    let plan = (
        DaeBdf::new(partition(), backends().remove(2)).unwrap(),
        options(),
    );
    let mut current = state(0., [1., 1.], [-1., 0.]);
    let report = run(&plan, &OffsetProduct, &mut current, 1.).unwrap();
    assert_eq!(report.termination(), Some(DaeTermination::RejectionLimit));
    assert_eq!(report.accepted_steps(), 0);
    assert_eq!(current, state(0., [1., 1.], [-1., 0.]));
}

#[test]
fn dae_gmres_iteration_exhaustion_rejects_the_proposal() {
    // One GMRES iteration cannot solve the coupled 2×2 stage system, so every
    // correction stops short of its forcing; that fails the stage and rejects the
    // proposal instead of ending the integration with an error.
    let backend = OdeStageBackend::Krylov {
        iterations: Budget::ONE,
        restart: Budget::ONE,
        operator_scratch: 0,
        preconditioner_scratch: 0,
    };
    let plan = (DaeBdf::new(partition(), backend).unwrap(), options());
    let mut current = state(0., [1., 1.], [-1., 0.]);
    let report = run(&plan, &LinearDecay, &mut current, 1.).unwrap();
    let termination = report.termination().unwrap();
    assert!(
        matches!(
            termination,
            DaeTermination::RejectionLimit | DaeTermination::ReachedEnd
        ),
        "{termination:?}"
    );
    assert!(report.rejected_steps() > 0);
}
