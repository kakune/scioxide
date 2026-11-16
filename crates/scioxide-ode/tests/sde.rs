//! SDE plans: accuracy, preflight and progressive failure contracts, budgets, refinement,
//! workspace requirements and ensembles.
#![allow(clippy::unwrap_used, clippy::panic)]

use core::cell::{Cell, RefCell};
use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, CancellationToken, Capacity, Complex, ErrorTolerance,
    EvaluationCounts, ExecutionContext, InvalidInputReason, NormSelection, NumericScalar,
    ProblemToken, RealScalar, SciOxideError, SciOxideErrorKind, SolveBudgets, SolveContext,
    SolveIdentity, SolveState,
};
use scioxide_linalg::MatrixView;
use scioxide_ode::{
    EnsembleParameters, OdeState, SdeMethod, SdeOptions, SdeProblem, SdeReport, SdeSystem,
    SdeTermination, SdeWorkspace,
};
use scioxide_random::{
    BrownianBridgePlan, Distribution, Normal, SamplingWorkspace, SeedableRng, SplitMix64,
};

fn r<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn failure(detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("output"),
        InvalidInputReason::UnsupportedConfiguration,
        detail,
    )
}

struct Linear<T: NumericScalar> {
    n: usize,
    m: usize,
    method: SdeMethod,
    a: T,
    b: T,
    multiplicative: bool,
    calls: Cell<usize>,
    fail: Option<usize>,
    incomplete: bool,
}
impl<T: NumericScalar> Linear<T> {
    fn new(method: SdeMethod, multiplicative: bool) -> Self {
        Self {
            n: 1,
            m: 1,
            method,
            a: T::from_real(r(0.4)),
            b: T::from_real(r(0.7)),
            multiplicative,
            calls: Cell::new(0),
            fail: None,
            incomplete: false,
        }
    }
    fn observe(&self) -> Result<(), SciOxideError> {
        let calls = self.calls.get() + 1;
        self.calls.set(calls);
        if self.fail == Some(calls) {
            Err(failure("injected provider failure"))
        } else {
            Ok(())
        }
    }
}
impl<T: NumericScalar> SdeSystem<T> for Linear<T> {
    fn dimension(&self) -> usize {
        self.n
    }
    fn noise_dimension(&self) -> usize {
        self.m
    }
    fn drift(
        &self,
        _: T::Real,
        y: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.observe()?;
        if !self.incomplete {
            for (o, &y) in out.iter_mut().zip(y) {
                *o = self.a * if self.multiplicative { y } else { T::ONE };
            }
        }
        Ok(())
    }
    fn diffusion_product(
        &self,
        _: T::Real,
        y: &[T],
        dw: &[T::Real],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.observe()?;
        for (o, &y) in out.iter_mut().zip(y) {
            *o = dw
                .iter()
                .fold(T::ZERO, |sum, &w| sum + self.b * T::from_real(w))
                * if self.multiplicative { y } else { T::ONE };
        }
        Ok(())
    }
}

fn options<S: RealScalar>(tolerance: Option<f64>) -> SdeOptions<S> {
    match tolerance {
        Some(tol) => SdeOptions::new().with_refinement(
            ErrorTolerance::scalar(r(tol), r(0.), NormSelection::Infinity).unwrap(),
        ),
        None => SdeOptions::new(),
    }
}

/// A plan with its reusable workspace; `solve` integrates `y` like caller-owned state.
struct Solver<T: NumericScalar> {
    method: SdeMethod,
    times: Vec<T::Real>,
    boundaries: Vec<usize>,
    options: SdeOptions<T::Real>,
    workspace: SdeWorkspace<T>,
}
impl<T: NumericScalar> Solver<T> {
    fn new(
        method: SdeMethod,
        times: &[T::Real],
        boundaries: &[usize],
        options: SdeOptions<T::Real>,
    ) -> Self {
        Self {
            method,
            times: times.to_vec(),
            boundaries: boundaries.to_vec(),
            options,
            workspace: SdeWorkspace::new(),
        }
    }
    fn solve(
        &mut self,
        system: &dyn SdeSystem<T>,
        y: &mut [T],
        increments: MatrixView<'_, T::Real>,
    ) -> Result<SdeReport<T::Real>, SciOxideError> {
        let mut state = OdeState::from_slice(self.times[0], y).unwrap();
        let result = self.method.advance(
            &SdeProblem::new(system, &self.times, increments)
                .with_step_boundaries(&self.boundaries),
            &self.options,
            &mut state,
            &mut (),
            &mut self.workspace,
            ExecutionContext::sequential(),
        );
        y.copy_from_slice(state.values());
        result
    }
    fn progress(&self) -> SdeReport<T::Real> {
        *self.workspace.last_attempt().unwrap().progress().unwrap()
    }
    fn outcome(&self) -> AttemptOutcome {
        self.workspace.last_attempt().unwrap().outcome()
    }
}

fn providers<S: RealScalar>(report: &SdeReport<S>) -> usize {
    report
        .accounting()
        .total()
        .unwrap()
        .total_providers()
        .unwrap()
}

fn additive<T: NumericScalar>() {
    for method in [SdeMethod::EulerMaruyama, SdeMethod::StochasticHeun] {
        let mut p = Linear::<T>::new(method, false);
        p.n = 2;
        p.m = 2;
        p.a = T::from_components(r(0.25), r(-0.125));
        p.b = T::from_components(r(0.5), r(0.25));
        let times = [r(0.), r(0.125), r(0.5), r(1.)];
        let noise = [r(0.25), r(0.5), r(-0.25), r(0.25), r(0.5), r(-0.25)];
        let view = MatrixView::from_slice(3, 2, &noise).unwrap();
        let initial = T::from_components(r(2.), r(1.));
        let mut s = Solver::new(method, &times, &[0, 1, 2, 3], SdeOptions::new());
        let mut y = [initial; 2];
        let report = s.solve(&p, &mut y, view).unwrap();
        assert!(report.is_success());
        assert_eq!(report.termination(), Some(SdeTermination::ReachedEnd));
        let expected = initial + p.a + p.b;
        assert!(
            y.iter()
                .all(|&v| (v - expected).abs() <= r::<T::Real>(32.) * T::Real::EPSILON)
        );
        assert_eq!(report.grid_index(), 3);
        assert_eq!(report.final_time(), r(1.));
        assert_eq!(report.counts().accepted(), 3);
        assert_eq!(providers(&report), p.calls.get());
        assert_eq!(s.outcome(), AttemptOutcome::Succeeded);
        assert_eq!(s.progress(), report);
        let first = y;
        y.fill(initial);
        assert!(s.solve(&p, &mut y, view).unwrap().is_success());
        assert_eq!(y, first);
    }
}
#[test]
fn additive_all_scalar_domains_and_replay() {
    additive::<f32>();
    additive::<f64>();
    additive::<Complex<f32>>();
    additive::<Complex<f64>>();
}

#[test]
fn shapes_and_nonfinite_inputs_fail_before_callbacks() {
    let p = Linear::<f64>::new(SdeMethod::EulerMaruyama, true);
    let mut s = Solver::new(
        SdeMethod::EulerMaruyama,
        &[0., 1.],
        &[0, 1],
        SdeOptions::new(),
    );
    let mut y = [2.];
    for noise in [f64::NAN, f64::INFINITY] {
        let error = s
            .solve(&p, &mut y, MatrixView::from_slice(1, 1, &[noise]).unwrap())
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
        assert_eq!(p.calls.get(), 0);
        assert_eq!(y, [2.]);
    }
    let error = s
        .solve(&p, &mut y, MatrixView::from_slice(1, 0, &[]).unwrap())
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    for times in [
        &[][..],
        &[0., 0.],
        &[1., 0.],
        &[0., f64::NAN],
        &[-f64::MAX, f64::MAX],
    ] {
        assert!(
            p.method
                .workspace_requirements(
                    &SdeProblem::new(
                        &scioxide_ode::SdeFn::new(
                            1,
                            1,
                            |_t, _x: &[f64], _out: &mut [f64]| Ok(()),
                            |_t, _x: &[f64], _dw: &[f64], _out: &mut [f64]| Ok(())
                        ),
                        times,
                        MatrixView::from_slice(0, 0, &[]).unwrap()
                    )
                    .with_step_boundaries(&[0, 1]),
                    &SdeOptions::new()
                )
                .is_err()
        );
    }
    for partition in [&[][..], &[1], &[0, 2], &[0, 0, 1]] {
        assert!(
            p.method
                .workspace_requirements(
                    &SdeProblem::new(
                        &scioxide_ode::SdeFn::new(
                            1,
                            1,
                            |_t, _x: &[f64], _out: &mut [f64]| Ok(()),
                            |_t, _x: &[f64], _dw: &[f64], _out: &mut [f64]| Ok(())
                        ),
                        &[0., 1.],
                        MatrixView::from_slice(0, 0, &[]).unwrap()
                    )
                    .with_step_boundaries(partition),
                    &SdeOptions::new()
                )
                .is_err()
        );
    }
    let mismatched = SdeOptions::new()
        .with_refinement(ErrorTolerance::try_vector(&[], 0., NormSelection::Infinity).unwrap());
    let error = SdeMethod::StochasticHeun
        .workspace_requirements(
            &SdeProblem::new(
                &scioxide_ode::SdeFn::new(
                    1,
                    1,
                    |_t, _x: &[f64], _out: &mut [f64]| Ok(()),
                    |_t, _x: &[f64], _dw: &[f64], _out: &mut [f64]| Ok(()),
                ),
                &[0., 1.],
                MatrixView::from_slice(0, 0, &[]).unwrap(),
            )
            .with_step_boundaries(&[0, 1]),
            &mismatched,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
}

#[test]
fn euler_maruyama_rejects_path_dependent_refinement() {
    let times = [0., 0.25, 0.5, 0.75, 1.];
    for tolerance in [1e-3, 1e3] {
        let error = SdeMethod::EulerMaruyama
            .workspace_requirements(
                &SdeProblem::new(
                    &scioxide_ode::SdeFn::new(
                        1,
                        1,
                        |_t, _x: &[f64], _out: &mut [f64]| Ok(()),
                        |_t, _x: &[f64], _dw: &[f64], _out: &mut [f64]| Ok(()),
                    ),
                    &times,
                    MatrixView::from_slice(0, 0, &[]).unwrap(),
                )
                .with_step_boundaries(&[0, 4]),
                &options(Some(tolerance)),
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::UnsupportedConfiguration,
                ..
            }
        ));
    }
    // The same grid is valid for fixed Euler–Maruyama steps and refined Heun.
    assert!(
        SdeMethod::EulerMaruyama
            .workspace_requirements(
                &SdeProblem::new(
                    &scioxide_ode::SdeFn::new(
                        1,
                        1,
                        |_t, _x: &[f64], _out: &mut [f64]| Ok(()),
                        |_t, _x: &[f64], _dw: &[f64], _out: &mut [f64]| Ok(())
                    ),
                    &times,
                    MatrixView::from_slice(0, 0, &[]).unwrap()
                )
                .with_step_boundaries(&[0, 4]),
                &options(None)
            )
            .is_ok()
    );
    assert!(
        SdeMethod::StochasticHeun
            .workspace_requirements(
                &SdeProblem::new(
                    &scioxide_ode::SdeFn::new(
                        1,
                        1,
                        |_t, _x: &[f64], _out: &mut [f64]| Ok(()),
                        |_t, _x: &[f64], _dw: &[f64], _out: &mut [f64]| Ok(())
                    ),
                    &times,
                    MatrixView::from_slice(0, 0, &[]).unwrap()
                )
                .with_step_boundaries(&[0, 4]),
                &options(Some(1e-3))
            )
            .is_ok()
    );
}

#[test]
fn heun_refinement_converges_to_the_stratonovich_solution() {
    // Stratonovich GBM dX = 0.4 X dt + 0.7 X ∘ dW has X(1) = exp(0.4 + 0.7 W(1)).
    let fine: u32 = 1 << 12;
    let rows = fine as usize;
    let times: Vec<_> = (0..=fine).map(|i| f64::from(i) / f64::from(fine)).collect();
    let boundaries: Vec<_> = (0..=8).map(|j| j * rows / 8).collect();
    let tolerances = [1e-2, 1e-3, 1e-4];
    let mut solvers = tolerances.map(|tolerance| {
        Solver::<f64>::new(
            SdeMethod::StochasticHeun,
            &times,
            &boundaries,
            options(Some(tolerance)),
        )
    });
    let p = Linear::<f64>::new(SdeMethod::StochasticHeun, true);
    let mut rng = SplitMix64::from_seed(2029);
    let normal = Normal::<f64>::new(0., (1. / f64::from(fine)).sqrt()).unwrap();
    let paths = 128;
    let (mut absolute, mut steps) = ([0.; 3], [0; 3]);
    for _ in 0..paths {
        let noise: Vec<_> = (0..fine)
            .map(|_| normal.sample(&mut rng).unwrap())
            .collect();
        let exact = (0.4 + 0.7 * noise.iter().sum::<f64>()).exp();
        let view = MatrixView::from_slice(rows, 1, &noise).unwrap();
        for ((solver, error), count) in solvers.iter_mut().zip(&mut absolute).zip(&mut steps) {
            let mut y = [1.];
            let report = solver.solve(&p, &mut y, view).unwrap();
            assert_eq!(report.termination(), Some(SdeTermination::ReachedEnd));
            *error += (y[0] - exact).abs() / f64::from(paths);
            *count += report.accepted_steps();
        }
    }
    eprintln!("tolerances {tolerances:?}: mean |error| {absolute:?}, accepted steps {steps:?}");
    assert!(absolute.windows(2).all(|pair| pair[1] < 0.75 * pair[0]));
    assert!(absolute[2] < 0.25 * absolute[0]);
    assert!(steps.windows(2).all(|pair| pair[1] > pair[0]));
}

#[test]
fn preflight_rejection_leaves_state_and_records_the_attempt() {
    let p = Linear::<f64>::new(SdeMethod::EulerMaruyama, false);
    let plan = (
        p.method,
        [0., 1.].to_vec(),
        [0, 1].to_vec(),
        SdeOptions::new(),
    );
    let mut workspace = SdeWorkspace::new();
    let problem = SdeProblem::new(&p, &plan.1, MatrixView::from_slice(1, 1, &[0.5]).unwrap())
        .with_step_boundaries(&plan.2);
    for (time, values, kind) in [
        (0.5, &[1.][..], SciOxideErrorKind::InvalidInput),
        (0., &[1., 2.], SciOxideErrorKind::DimensionMismatch),
        (0., &[f64::NAN], SciOxideErrorKind::NonFinite),
    ] {
        let mut state = OdeState::from_slice(time, values).unwrap();
        let error = plan
            .0
            .advance(
                &problem,
                &plan.3,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(state.time(), time);
        assert_eq!(state.values().len(), values.len());
        assert_eq!(p.calls.get(), 0);
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
        assert!(attempt.epoch().is_none() && attempt.progress().is_none());
    }
    // A rejection does not disturb the next call.
    let mut state = OdeState::from_slice(0., &[1.]).unwrap();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.3,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert!((state.values()[0] - 1.75).abs() < 1e-15);
    assert_eq!(state.time(), 1.);
}

#[test]
fn provider_failures_keep_exact_accepted_boundary_and_counts() {
    for method in [SdeMethod::EulerMaruyama, SdeMethod::StochasticHeun] {
        let per_step = if method == SdeMethod::EulerMaruyama {
            2
        } else {
            4
        };
        for fail in 1..=3 * per_step {
            let mut p = Linear::<f64>::new(method, false);
            p.fail = Some(fail);
            let mut s = Solver::new(
                method,
                &[0., 0.25, 0.5, 0.75],
                &[0, 1, 2, 3],
                SdeOptions::new(),
            );
            let mut y = [1.];
            let error = s
                .solve(&p, &mut y, MatrixView::from_slice(3, 1, &[0.; 3]).unwrap())
                .unwrap_err();
            let accepted = (fail - 1) / per_step;
            assert_eq!(s.outcome(), AttemptOutcome::Failed);
            let progress = s.progress();
            assert_eq!(progress.termination(), None);
            assert!(!progress.is_success());
            assert_eq!(progress.accepted_steps(), accepted);
            assert_eq!(progress.grid_index(), accepted);
            assert_eq!(progress.final_time(), 0.25 * accepted as f64);
            assert_eq!(providers(&progress), fail);
            assert_eq!(
                s.workspace.last_attempt().unwrap().counts(),
                progress.accounting()
            );
            assert_eq!(p.calls.get(), fail);
            assert!((y[0] - (1. + accepted as f64 * 0.1)).abs() < 1e-14);
            assert_eq!(
                error.to_string(),
                failure("injected provider failure").to_string()
            );
        }
    }
    let mut p = Linear::<f64>::new(SdeMethod::EulerMaruyama, true);
    p.incomplete = true;
    let mut s = Solver::new(p.method, &[0., 1.], &[0, 1], SdeOptions::new());
    let mut y = [1.];
    let error = s
        .solve(&p, &mut y, MatrixView::from_slice(1, 1, &[0.]).unwrap())
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(y, [1.]);
}

#[test]
fn owned_solve_returns_the_final_state_or_partial_progress() {
    let mut p = Linear::<f64>::new(SdeMethod::EulerMaruyama, false);
    let plan = (
        p.method,
        [0., 0.25, 0.5].to_vec(),
        [0, 1, 2].to_vec(),
        SdeOptions::new(),
    );
    let noise = [0.125, -0.25];
    let problem = SdeProblem::new(&p, &plan.1, MatrixView::from_slice(2, 1, &noise).unwrap())
        .with_step_boundaries(&plan.2);
    let solution = plan.0.solve(&problem, &plan.3, &[1.]).unwrap();
    assert!(solution.report().is_success());
    assert_eq!(solution.state().time(), 0.5);
    let expected = 1. + 0.4 * 0.5 + 0.7 * (0.125 - 0.25);
    assert!((solution.state().values()[0] - expected).abs() < 1e-15);
    let (state, report) = solution.into_parts();
    assert_eq!(report.accepted_steps(), 2);
    assert_eq!(state.values().len(), 1);

    p.fail = Some(3);
    p.calls.set(0);
    let problem = SdeProblem::new(&p, &plan.1, MatrixView::from_slice(2, 1, &noise).unwrap())
        .with_step_boundaries(&plan.2);
    let failure = plan.0.solve(&problem, &plan.3, &[1.]).unwrap_err();
    let progress = failure.progress().unwrap();
    assert_eq!(progress.termination(), None);
    assert_eq!(
        (progress.accepted_steps(), progress.final_time()),
        (1, 0.25)
    );
    assert_eq!(providers(progress), 3);
    let (error, _) = failure.into_parts();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);

    let failure = plan.0.solve(&problem, &plan.3, &[1., 2.]).unwrap_err();
    assert!(failure.progress().is_none());
    assert_eq!(failure.error().kind(), SciOxideErrorKind::DimensionMismatch);
}

#[test]
fn every_budget_boundary_counts_only_admitted_work() {
    for limit in 1..=6 {
        let p = Linear::<f64>::new(SdeMethod::EulerMaruyama, false);
        let budgeted = SdeOptions::new().with_max_evaluations(Budget::try_from(limit).unwrap());
        let mut s = Solver::new(p.method, &[0., 0.25, 0.5, 0.75], &[0, 1, 2, 3], budgeted);
        let mut y = [1.];
        let report = s
            .solve(&p, &mut y, MatrixView::from_slice(3, 1, &[0.; 3]).unwrap())
            .unwrap();
        assert_eq!(p.calls.get(), limit);
        assert_eq!(providers(&report), limit);
        assert_eq!(report.accepted_steps(), limit / 2);
        assert_eq!(report.is_success(), limit == 6);
        if limit < 6 {
            assert_eq!(
                report.termination(),
                Some(SdeTermination::BudgetLimit(BudgetKind::Evaluation))
            );
            assert_eq!(s.outcome(), AttemptOutcome::Stopped);
            assert_eq!(s.progress(), report);
        }
    }
    let p = Linear::<f64>::new(SdeMethod::EulerMaruyama, false);
    let budgeted = SdeOptions::new().with_max_steps(Budget::try_from(1).unwrap());
    let mut s = Solver::new(p.method, &[0., 0.5, 1.], &[0, 1, 2], budgeted);
    let report = s
        .solve(
            &p,
            &mut [1.],
            MatrixView::from_slice(2, 1, &[0.; 2]).unwrap(),
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        Some(SdeTermination::BudgetLimit(BudgetKind::Proposal))
    );
    assert_eq!(report.accepted_steps(), 1);
    assert_eq!(p.calls.get(), 2);
}

#[test]
fn child_epochs_share_the_ancestor_evaluation_budget() {
    let p = Linear::<f64>::new(SdeMethod::EulerMaruyama, false);
    let plan = (
        p.method,
        [0., 0.5, 1.].to_vec(),
        [0, 1, 2].to_vec(),
        SdeOptions::new(),
    );
    let mut workspace = SdeWorkspace::new();
    let problem = SdeProblem::new(&p, &plan.1, MatrixView::from_slice(2, 1, &[0.; 2]).unwrap())
        .with_step_boundaries(&plan.2);
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "parent", 1),
        0,
        0,
    );
    let mut epoch = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(3).unwrap());
    let mut parent = epoch
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut state = OdeState::from_slice(0., &[1.]).unwrap();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.3,
            &mut state,
            &mut (),
            &mut workspace,
            &mut parent,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        Some(SdeTermination::BudgetLimit(BudgetKind::Evaluation))
    );
    assert_eq!((report.accepted_steps(), p.calls.get()), (1, 3));
    assert_eq!(state.time(), 0.5);
    assert_eq!(parent.counts().nested().total_providers().unwrap(), 3);
}

#[test]
fn empty_and_deterministic_domains_and_exact_workspace_limit() {
    let mut p = Linear::<f64>::new(SdeMethod::EulerMaruyama, false);
    let mut s = Solver::new(p.method, &[3.], &[0], SdeOptions::new());
    let mut y = [2.];
    let report = s
        .solve(&p, &mut y, MatrixView::from_slice(0, 1, &[]).unwrap())
        .unwrap();
    assert!(report.is_success());
    assert_eq!((report.final_time(), report.accepted_steps()), (3., 0));
    assert_eq!(y, [2.]);
    assert_eq!(p.calls.get(), 0);
    p.n = 0;
    p.m = 0;
    let mut s = Solver::new(
        SdeMethod::StochasticHeun,
        &[0., 0.5, 1.],
        &[0, 2],
        options(Some(1e-6)),
    );
    let report = s
        .solve(&p, &mut [], MatrixView::from_slice(2, 0, &[]).unwrap())
        .unwrap();
    assert!(report.is_success());
    assert_eq!(p.calls.get(), 0);
    p.n = 1;
    let plan = (
        p.method,
        [0., 1.].to_vec(),
        [0, 1].to_vec(),
        SdeOptions::new(),
    );
    let problem = SdeProblem::new(&p, &plan.1, MatrixView::from_slice(1, 0, &[]).unwrap())
        .with_step_boundaries(&plan.2);
    let bytes = plan
        .0
        .workspace_requirements::<f64>(&problem, &plan.3)
        .unwrap()
        .required_bytes();
    let tight = SdeOptions::new().with_max_workspace_bytes(bytes - 1);
    let tight = (p.method, [0., 1.].to_vec(), [0, 1].to_vec(), tight);
    let error = tight.0.workspace::<f64>(&problem, &tight.3).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    let exact = SdeOptions::new().with_max_workspace_bytes(bytes);
    let mut s = Solver::new(p.method, &[0., 1.], &[0, 1], exact);
    let report = s
        .solve(&p, &mut y, MatrixView::from_slice(1, 0, &[]).unwrap())
        .unwrap();
    assert!(report.is_success());
    assert!((y[0] - 2.4).abs() < 1e-14);
    assert_eq!(p.calls.get(), 1);
}

#[test]
fn workspaces_follow_the_requirements_protocol_across_plans() {
    let large = (
        SdeMethod::StochasticHeun,
        [0., 0.25, 0.5, 0.75, 1.].to_vec(),
        [0, 4].to_vec(),
        options(Some(1e-3)),
    );
    let small = (
        SdeMethod::StochasticHeun,
        [0., 1.].to_vec(),
        [0, 1].to_vec(),
        SdeOptions::new(),
    );
    let mut large_system = Linear::<Complex<f64>>::new(SdeMethod::StochasticHeun, false);
    large_system.n = 3;
    large_system.m = 2;
    let small_system = Linear::<Complex<f64>>::new(SdeMethod::StochasticHeun, false);
    let large_problem = SdeProblem::new(
        &large_system,
        &large.1,
        MatrixView::from_slice(4, 2, &[0.; 8]).unwrap(),
    )
    .with_step_boundaries(&large.2);
    let small_problem = SdeProblem::new(
        &small_system,
        &small.1,
        MatrixView::from_slice(1, 1, &[0.]).unwrap(),
    )
    .with_step_boundaries(&small.2);
    let large_requirements = large
        .0
        .workspace_requirements::<Complex<f64>>(&large_problem, &large.3)
        .unwrap();
    let small_requirements = small
        .0
        .workspace_requirements::<Complex<f64>>(&small_problem, &small.3)
        .unwrap();
    assert_eq!(
        (
            large_requirements.dimension(),
            large_requirements.noise_dimension()
        ),
        (3, 2)
    );
    assert!(large_requirements.required_bytes() > small_requirements.required_bytes());
    assert_eq!(small_requirements.byte_limit(), None);
    let mut workspace = SdeWorkspace::<Complex<f64>>::new();
    assert_eq!(
        workspace.check(&small_requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&large_requirements).unwrap();
    for requirements in [&large_requirements, &small_requirements] {
        assert_eq!(workspace.check(requirements).unwrap(), Capacity::Sufficient);
    }
    // Retained capacity from the larger plan counts against a smaller plan's limit.
    let limited = SdeOptions::new().with_max_workspace_bytes(small_requirements.required_bytes());
    let limited = (
        SdeMethod::StochasticHeun,
        [0., 1.].to_vec(),
        [0, 1].to_vec(),
        limited,
    );
    let limited_requirements = limited
        .0
        .workspace_requirements::<Complex<f64>>(&small_problem, &limited.3)
        .unwrap();
    assert_eq!(
        workspace.check(&limited_requirements).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut p = Linear::<Complex<f64>>::new(SdeMethod::StochasticHeun, false);
    p.n = 1;
    let problem = SdeProblem::new(&p, &small.1, MatrixView::from_slice(1, 1, &[0.5]).unwrap())
        .with_step_boundaries(&small.2);
    let mut state = OdeState::from_slice(0., &[Complex::new(1., 0.)]).unwrap();
    let error = limited
        .0
        .advance(
            &problem,
            &limited.3,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(state.values(), [Complex::new(1., 0.)]);
    // An empty workspace is grown by preflight.
    let mut fresh = SdeWorkspace::new();
    let report = small
        .0
        .advance(
            &problem,
            &small.3,
            &mut state,
            &mut (),
            &mut fresh,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.is_success());
    assert_eq!(
        fresh.check(&small_requirements).unwrap(),
        Capacity::Sufficient
    );
}

#[test]
fn prepared_advance_allocates_nothing() {
    for tolerance in [None, Some(0.005)] {
        let p = Linear::<f64>::new(SdeMethod::StochasticHeun, true);
        let times: Vec<_> = (0..=64).map(|i| f64::from(i) / 64.).collect();
        let noise: Vec<_> = (0..64)
            .map(|i| if i % 2 == 0 { 0.025 } else { -0.0125 })
            .collect();
        let boundaries = [0, 16, 32, 48, 64];
        let plan = (
            p.method,
            times.to_vec(),
            boundaries.to_vec(),
            options(tolerance),
        );
        let mut workspace = SdeWorkspace::new();
        let problem = SdeProblem::new(&p, &plan.1, MatrixView::from_slice(64, 1, &noise).unwrap())
            .with_step_boundaries(&plan.2);
        workspace
            .try_reserve(&plan.0.workspace_requirements(&problem, &plan.3).unwrap())
            .unwrap();
        let mut state = OdeState::from_slice(0., &[1.]).unwrap();
        let mut samples = 0;
        let mut sink = |_: f64, _: &[f64]| {
            samples += 1;
            Ok(())
        };
        let report = plan
            .0
            .advance(
                &problem,
                &plan.3,
                &mut state,
                &mut sink,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(samples, report.accepted_steps());
        assert_eq!(state.time(), 1.);
    }
}

#[test]
fn prepared_budget_stops_execute() {
    let p = Linear::<f64>::new(SdeMethod::StochasticHeun, true);
    let times: Vec<_> = (0..=64).map(|i| f64::from(i) / 64.).collect();
    let noise: Vec<_> = (0..64)
        .map(|i| if i % 2 == 0 { 0.025 } else { -0.0125 })
        .collect();
    let boundaries: Vec<_> = (0..=64).collect();
    let problem = SdeProblem::new(&p, &times, MatrixView::from_slice(64, 1, &noise).unwrap())
        .with_step_boundaries(&boundaries);
    let budget = |limit: usize| Budget::try_from(limit).unwrap();
    let cases = core::iter::once((
        SdeOptions::new().with_max_steps(budget(10)),
        BudgetKind::Proposal,
    ))
    .chain((5..=24).map(|limit| {
        (
            SdeOptions::new().with_max_evaluations(budget(limit)),
            BudgetKind::Evaluation,
        )
    }));
    for (options, kind) in cases {
        let plan = (p.method, times.to_vec(), boundaries.to_vec(), options);
        let mut workspace = SdeWorkspace::new();
        let mut state = OdeState::from_slice(0., &[1.]).unwrap();
        let warm = plan
            .0
            .advance(
                &problem,
                &plan.3,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(!warm.is_success());
        state.reset(0., &[1.]).unwrap();
        let report = plan
            .0
            .advance(
                &problem,
                &plan.3,
                &mut state,
                &mut (),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            Some(SdeTermination::BudgetLimit(kind))
        );
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Stopped
        );
    }
}

#[test]
fn refinement_replays_noise_and_reports_grid_exhaustion() {
    let times: Vec<_> = (0..=64).map(|i| i as f64 / 64.).collect();
    let noise: Vec<_> = (0..64)
        .map(|i| if i % 2 == 0 { 0.025 } else { -0.0125 })
        .collect();
    let view = MatrixView::from_slice(64, 1, &noise).unwrap();
    let p = Linear::<f64>::new(SdeMethod::StochasticHeun, true);
    let mut s = Solver::new(p.method, &times, &[0, 64], options(Some(0.005)));
    let mut y = [1.];
    let first = s.solve(&p, &mut y, view).unwrap();
    assert!(first.is_success(), "{first:?}");
    assert!(first.counts().rejected() >= 2);
    assert!(
        first
            .last_step_error_estimate()
            .is_some_and(|error| error <= 1.)
    );
    let expected = y;
    y = [1.];
    let repeat = s.solve(&p, &mut y, view).unwrap();
    assert_eq!(y, expected);
    assert_eq!(repeat.accounting(), first.accounting());
    let mut tiny = Solver::new(p.method, &[0., 0.5, 1.], &[0, 2], options(Some(1e-12)));
    y = [1.];
    let report = tiny
        .solve(
            &p,
            &mut y,
            MatrixView::from_slice(2, 1, &[0.25, -0.1]).unwrap(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(SdeTermination::StepSizeLimit));
    assert_eq!(report.counts().rejected(), 1);
    assert_eq!(y, [1.]);
    assert_eq!(report.grid_index(), 0);
}

#[test]
fn brownian_bridge_path_matches_additive_solution() {
    let times = [0., 0.125, 0.25, 0.5, 1.];
    let bridge = BrownianBridgePlan::new(&times).unwrap();
    let mut path = [0.; 4];
    bridge
        .apply_into(
            &[0.4, -0.7, 0.2, 0.8],
            &mut path,
            &mut SamplingWorkspace::new(),
        )
        .unwrap();
    let noise: Vec<_> = core::iter::once(path[0])
        .chain(path.windows(2).map(|w| w[1] - w[0]))
        .collect();
    let p = Linear::<f64>::new(SdeMethod::StochasticHeun, false);
    for boundaries in [&[0, 4][..], &[0, 2, 4], &[0, 1, 2, 3, 4]] {
        let mut s = Solver::new(p.method, &times, boundaries, SdeOptions::new());
        let mut y = [1.];
        let report = s
            .solve(&p, &mut y, MatrixView::from_slice(4, 1, &noise).unwrap())
            .unwrap();
        assert_eq!(report.accepted_steps(), boundaries.len() - 1);
        assert!((y[0] - (1. + 0.4 + 0.7 * path[3])).abs() < 2e-15);
    }
}

#[test]
fn fixed_grid_strong_orders_use_coupled_exact_gbm_paths() {
    // Strong order is a property of the method; per-domain step arithmetic is
    // checked exactly by `real_and_complex_multiplicative_steps_keep_interpretations_distinct`.
    let times: Vec<_> = (0..=128).map(|i| f64::from(i) / 128.).collect();
    let mut rng = SplitMix64::from_seed(663);
    let normal = Normal::<f64>::new(0., (1.0_f64 / 128.).sqrt()).unwrap();
    for method in [SdeMethod::EulerMaruyama, SdeMethod::StochasticHeun] {
        let mut solvers: Vec<_> = [8, 16, 32, 64]
            .map(|steps| {
                Solver::<f64>::new(
                    method,
                    &times,
                    &(0..=steps).map(|j| j * 128 / steps).collect::<Vec<_>>(),
                    SdeOptions::new(),
                )
            })
            .into_iter()
            .collect();
        let mut squared = [0.; 4];
        let p = Linear::<f64>::new(method, true);
        for _ in 0..1024 {
            let noise: Vec<_> = (0..128).map(|_| normal.sample(&mut rng).unwrap()).collect();
            let view = MatrixView::from_slice(128, 1, &noise).unwrap();
            let w: f64 = noise.iter().sum();
            let exact =
                (0.4 - if method == SdeMethod::EulerMaruyama {
                    0.5 * 0.7 * 0.7
                } else {
                    0.
                } + 0.7 * w)
                    .exp();
            for (s, error) in solvers.iter_mut().zip(&mut squared) {
                let mut y = [1.];
                assert!(s.solve(&p, &mut y, view).unwrap().is_success());
                let delta = y[0] - exact;
                *error += delta * delta;
            }
        }
        let rms = squared.map(|v| (v / 1024.).sqrt());
        let order = (rms[0] / rms[3]).ln() / 8.0_f64.ln();
        eprintln!("{method:?}: RMS={rms:?}, strong order={order}");
        let (low, high) = if method == SdeMethod::EulerMaruyama {
            (0.3, 0.7)
        } else {
            (0.75, 1.25)
        };
        assert!(order > low && order < high, "{order}, {rms:?}");
        assert!(rms.windows(2).all(|w| w[1] < w[0]));
    }
}

#[test]
fn weak_order_against_exact_moments_without_monte_carlo_noise() {
    // Three Gaussian quadrature nodes integrate the step multiplier and its
    // square exactly (degree <=4). Independent increments multiply moments.
    for method in [SdeMethod::EulerMaruyama, SdeMethod::StochasticHeun] {
        let mut errors = [0.; 4];
        for (slot, steps) in errors.iter_mut().zip([16, 32, 64, 128]) {
            let h = 1. / steps as f64;
            let mut s = Solver::new(method, &[0., h], &[0, 1], SdeOptions::new());
            let mut mean = 0.;
            let mut second = 0.;
            for (z, weight) in [
                (-(3.0_f64).sqrt(), 1. / 6.),
                (0., 2. / 3.),
                ((3.0_f64).sqrt(), 1. / 6.),
            ] {
                let mut y = [1.];
                let p = Linear::<f64>::new(method, true);
                let report = s
                    .solve(
                        &p,
                        &mut y,
                        MatrixView::from_slice(1, 1, &[z * h.sqrt()]).unwrap(),
                    )
                    .unwrap();
                assert!(report.is_success());
                mean += weight * y[0];
                second += weight * y[0] * y[0];
            }
            let ito_a = if method == SdeMethod::EulerMaruyama {
                0.4_f64
            } else {
                0.4 + 0.5 * 0.49
            };
            let first_error = (mean.powi(steps) - ito_a.exp()).abs();
            let second_error = (second.powi(steps) - (2. * ito_a + 0.49).exp()).abs();
            *slot = first_error.max(second_error);
        }
        let order = (errors[0] / errors[3]).ln() / 8.0_f64.ln();
        eprintln!("{method:?}: weak errors={errors:?}, order={order}");
        assert!((0.85..1.15).contains(&order));
    }
}

#[test]
fn strided_noise_sink_failure_and_cancellation_preserve_accepted_state() {
    let mut p = Linear::<f64>::new(SdeMethod::StochasticHeun, false);
    p.n = 2;
    let plan = (
        p.method,
        [0., 0.5, 1.].to_vec(),
        [0, 1, 2].to_vec(),
        SdeOptions::new(),
    );
    let mut workspace = SdeWorkspace::new();
    let noise = [0.25, f64::NAN, -0.125];
    let noise = MatrixView::from_slice_with_strides(2, 1, &noise, 2, 1).unwrap();
    let problem = SdeProblem::new(&p, &plan.1, noise).with_step_boundaries(&plan.2);
    let mut state = OdeState::from_slice(0., &[1., 2.]).unwrap();
    let mut seen = 0;
    let mut sink = |t: f64, y: &[f64]| {
        seen += 1;
        assert_eq!(t, 0.5);
        assert!((y[0] - 1.375).abs() < 1e-14);
        Err(failure("sink failed"))
    };
    let error = plan
        .0
        .advance(
            &problem,
            &plan.3,
            &mut state,
            &mut sink,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.to_string(), failure("sink failed").to_string());
    assert_eq!(seen, 1);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    let progress = attempt.progress().unwrap();
    assert_eq!(progress.grid_index(), 1);
    assert_eq!(state.time(), 0.5);
    assert_eq!(state.values(), [1.375, 2.375]);
    // Sink calls are output, not provider work.
    assert_eq!(providers(progress), 4);

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    state.reset(0., &[1.375, 2.375]).unwrap();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.3,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert!(matches!(
        report.termination(),
        Some(SdeTermination::ExecutionStopped(_))
    ));
    assert_eq!(providers(&report), 0);
    assert_eq!(state.values(), [1.375, 2.375]);
    assert_eq!(state.time(), 0.);
    // Cancellation arriving during a stage must prevent the candidate commit.
    struct Cancel<'a>(&'a CancellationToken);
    impl SdeSystem<f64> for Cancel<'_> {
        fn dimension(&self) -> usize {
            2
        }
        fn noise_dimension(&self) -> usize {
            1
        }
        fn drift(
            &self,
            _: f64,
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.fill(1.);
            self.0.cancel();
            Ok(())
        }
        fn diffusion_product(
            &self,
            _: f64,
            _: &[f64],
            _: &[f64],
            _: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            unreachable!("cancelled before diffusion")
        }
    }
    cancellation.reset();
    let cancel = Cancel(&cancellation);
    let report = plan
        .0
        .advance(
            &SdeProblem::new(&cancel, &plan.1, noise).with_step_boundaries(&plan.2),
            &plan.3,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert!(matches!(
        report.termination(),
        Some(SdeTermination::ExecutionStopped(_))
    ));
    assert_eq!(providers(&report), 1);
    assert_eq!(state.values(), [1.375, 2.375]);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
}

#[test]
fn real_and_complex_multiplicative_steps_keep_interpretations_distinct() {
    fn case<T: NumericScalar>() {
        let initial = T::from_components(r(1.), r(0.5));
        let mut results = [T::ZERO; 2];
        for (i, method) in [SdeMethod::EulerMaruyama, SdeMethod::StochasticHeun]
            .into_iter()
            .enumerate()
        {
            let mut p = Linear::<T>::new(method, true);
            p.a = T::from_components(r(0.2), r(0.1));
            p.b = T::from_components(r(0.4), r(-0.25));
            let mut s = Solver::new(method, &[r(0.), r(0.125)], &[0, 1], SdeOptions::new());
            let mut y = [initial];
            let report = s
                .solve(
                    &p,
                    &mut y,
                    MatrixView::from_slice(1, 1, &[r(0.25)]).unwrap(),
                )
                .unwrap();
            assert!(report.is_success());
            let q = p.a * T::from_real(r(0.125)) + p.b * T::from_real(r(0.25));
            let multiplier = T::ONE
                + q
                + if i == 1 {
                    q * q * T::from_real(r(0.5))
                } else {
                    T::ZERO
                };
            assert!((y[0] - initial * multiplier).abs() < r::<T::Real>(16.) * T::Real::EPSILON);
            results[i] = y[0];
        }
        assert!((results[0] - results[1]).abs() > r(0.001));
    }
    case::<f32>();
    case::<f64>();
    case::<Complex<f32>>();
    case::<Complex<f64>>();
}

#[test]
fn dense_sparse_and_operator_diffusion_actions_agree() {
    use scioxide_core::LinearOperator;
    use scioxide_linalg::Matrix;
    use scioxide_sparse::CsrMatrix;
    struct Diffusion<O> {
        operator: O,
        scratch: RefCell<Vec<f64>>,
    }
    impl<O: LinearOperator<f64>> SdeSystem<f64> for Diffusion<O> {
        fn dimension(&self) -> usize {
            2
        }
        fn noise_dimension(&self) -> usize {
            2
        }
        fn drift(
            &self,
            _: f64,
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            out.fill(0.);
            Ok(())
        }
        fn diffusion_product(
            &self,
            _: f64,
            _: &[f64],
            dw: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            self.operator
                .apply_into(dw, out, &mut self.scratch.borrow_mut())
        }
    }
    struct Action;
    impl LinearOperator<f64> for Action {
        fn rows(&self) -> usize {
            2
        }
        fn cols(&self) -> usize {
            2
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = 2. * x[0] - x[1];
            out[1] = 3. * x[1];
            Ok(())
        }
    }
    fn run<O: LinearOperator<f64>>(operator: O) -> [f64; 2] {
        let scratch = RefCell::new(vec![0.; operator.apply_workspace_len().unwrap()]);
        let p = Diffusion { operator, scratch };
        let mut s = Solver::new(
            SdeMethod::StochasticHeun,
            &[0., 0.5, 1.],
            &[0, 1, 2],
            SdeOptions::new(),
        );
        let mut y = [1., 2.];
        let report = s
            .solve(
                &p,
                &mut y,
                MatrixView::from_slice(2, 2, &[0.25, -0.5, -0.125, 0.25]).unwrap(),
            )
            .unwrap();
        assert!(report.is_success());
        y
    }
    let dense = Matrix::from_vec(2, 2, vec![2., -1., 0., 3.]).unwrap();
    let sparse =
        CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![2., -1., 3.]).unwrap();
    assert_eq!(run(dense.view().try_finite().unwrap()), [1.5, 1.25]);
    assert_eq!(run(sparse), run(Action));
}

#[test]
fn sde_ensembles_preserve_member_noise_and_heterogeneous_outcomes() {
    use core::num::NonZeroUsize;
    use scioxide_core::ExecutionPolicy;
    type Member = (
        (SdeMethod, Vec<f64>, Vec<usize>, SdeOptions<f64>),
        SdeWorkspace<f64>,
        Linear<f64>,
        OdeState<f64>,
        [f64; 2],
    );
    type EnsembleRun = (
        Vec<Vec<f64>>,
        Vec<Option<Result<SdeReport<f64>, SciOxideError>>>,
        EvaluationCounts,
    );
    fn run(policy: ExecutionPolicy) -> EnsembleRun {
        let plan = policy;
        let parameters = [0.1, 0.2, 0.3, 0.4];
        let parameters =
            EnsembleParameters::Independent(MatrixView::from_slice(4, 1, &parameters).unwrap());
        let mut workspace = {
            let mut workspace = scioxide_ode::EnsembleWorkspace::new();
            workspace
                .try_reserve(&scioxide_ode::EnsembleRequirements::new(4, parameters).unwrap())
                .unwrap();
            workspace
        };
        let mut members: Vec<Member> = (0..4)
            .map(|i| {
                let mut p = Linear::<f64>::new(SdeMethod::EulerMaruyama, false);
                p.fail = if i == 1 { Some(1) } else { None };
                let sde = (
                    p.method,
                    [0., 0.5, 1.].to_vec(),
                    [0, 1, 2].to_vec(),
                    SdeOptions::new(),
                );
                let workspace = SdeWorkspace::new();
                (
                    sde,
                    workspace,
                    p,
                    OdeState::from_slice(0., &[1.]).unwrap(),
                    [i as f64 * 0.125, -0.0625],
                )
            })
            .collect();
        let mut output: Vec<_> = (0..4).map(|_| None).collect();
        #[cfg(feature = "std")]
        let executor = scioxide_core::ThreadExecutor::new(NonZeroUsize::new(3).unwrap());
        #[cfg(feature = "std")]
        let execution = ExecutionContext::new(&executor);
        #[cfg(not(feature = "std"))]
        let execution = ExecutionContext::sequential();
        let operation =
            |_: usize, parameters: &[f64], m: &mut Member, context: &mut SolveContext<'_, '_>| {
                m.2.a = parameters[0];
                let problem =
                    SdeProblem::new(&m.2, &m.0.1, MatrixView::from_slice(2, 1, &m.4).unwrap())
                        .with_step_boundaries(&m.0.2);
                m.0.0
                    .advance(&problem, &m.0.3, &mut m.3, &mut (), &mut m.1, context)
            };
        let report = scioxide_ode::solve_ensemble_into(
            parameters,
            &operation,
            plan,
            &mut members,
            &mut output,
            &mut workspace,
            execution,
        )
        .unwrap();
        assert_eq!((report.completed(), report.successful()), (4, 3));
        assert!(output[1].as_ref().unwrap().is_err());
        let failed = members[1].1.last_attempt().unwrap();
        assert_eq!(failed.outcome(), AttemptOutcome::Failed);
        assert_eq!(failed.progress().unwrap().accepted_steps(), 0);
        (
            members.iter().map(|m| m.3.values().to_vec()).collect(),
            output,
            report.accounting().total().unwrap(),
        )
    }
    let (serial, a, counts) = run(ExecutionPolicy::Sequential);
    let (parallel, b, other) = run(ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap()));
    assert_eq!(serial, parallel);
    assert_eq!(counts, other);
    for (a, b) in a.iter().zip(&b) {
        if let (Some(Ok(a)), Some(Ok(b))) = (a, b) {
            assert_eq!(a, b);
        }
    }
}

#[test]
fn termination_uses_the_shared_vocabulary() {
    const fn shared(termination: SdeTermination) -> bool {
        match termination {
            SdeTermination::ReachedEnd | SdeTermination::TerminalEvent => true,
            SdeTermination::BudgetLimit(_)
            | SdeTermination::StepSizeLimit
            | SdeTermination::ExecutionStopped(_) => false,
            _ => false,
        }
    }
    assert!(shared(SdeTermination::ReachedEnd));
    assert!(SdeTermination::TerminalEvent.is_success());
    assert!(!shared(SdeTermination::BudgetLimit(BudgetKind::Evaluation)));
    assert!(!SdeTermination::StepSizeLimit.is_success());
}
