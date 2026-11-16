//! Sparse/direct/operator and finite-difference method-of-lines tests.
#![allow(clippy::unwrap_used)]
use scioxide_core::*;
use scioxide_differentiate::*;
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::*;
use scioxide_sparse::SparseLuLimits;
struct Diffusion {
    n: usize,
}
impl OdeSystem<f64> for Diffusion {
    fn dimension(&self) -> usize {
        self.n
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[f64],
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(0.);
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let a = 0.01 * ((self.n + 1) * (self.n + 1)) as f64;
        for (i, value) in out.iter_mut().enumerate() {
            *value = a
                * (-2. * y[i]
                    + if i > 0 { y[i - 1] } else { 0. }
                    + if i + 1 < self.n { y[i + 1] } else { 0. });
        }
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let a = 0.01 * ((self.n + 1) * (self.n + 1)) as f64;
        for i in 0..self.n {
            for k in 0..self.n {
                j.set(
                    i,
                    k,
                    if i == k {
                        -2. * a
                    } else if i.abs_diff(k) == 1 {
                        a
                    } else {
                        0.
                    },
                )?;
            }
        }
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: f64,
        _: &[f64],
        pattern: StructuralPatternView<'_>,
        values: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let a = 0.01 * ((self.n + 1) * (self.n + 1)) as f64;
        for i in 0..self.n {
            for (k, value) in values
                .iter_mut()
                .enumerate()
                .take(pattern.row_ptr()[i + 1])
                .skip(pattern.row_ptr()[i])
            {
                *value = if pattern.col_indices()[k] == i {
                    -2. * a
                } else {
                    a
                };
            }
        }
        Ok(Some(()))
    }
    fn jacobian_product(
        &self,
        t: f64,
        _: &[f64],
        x: &[f64],
        out: &mut [f64],
        _: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.rhs(t, x, out, cx).map(Some)
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}
fn pattern(n: usize) -> StructuralPattern {
    let mut rows = vec![0];
    let mut cols = vec![];
    for row in 0..n {
        cols.extend(row.saturating_sub(1)..(row + 2).min(n));
        rows.push(cols.len());
    }
    StructuralPattern::new(n, n, rows, cols).unwrap()
}
fn run(system: &dyn OdeSystem<f64>, backend: OdeStageBackend) -> (Vec<f64>, OdeSolveReport<f64>) {
    run_with(
        system,
        backend,
        DerivativeStrategy::AnalyticOrAutomaticOrForward,
    )
}
fn run_with(
    system: &dyn OdeSystem<f64>,
    backend: OdeStageBackend,
    strategy: DerivativeStrategy,
) -> (Vec<f64>, OdeSolveReport<f64>) {
    let n = system.dimension();
    let y: Vec<_> = (0..n)
        .map(|i| (core::f64::consts::PI * (i + 1) as f64 / (n + 1) as f64).sin())
        .collect();
    let method = RosenbrockW::new(RosenbrockMethod::Ros34Pw2)
        .unwrap()
        .with_backend(backend)
        .with_derivative_strategy(strategy);
    let options = OdeOptions::new()
        .with_initial_step(0.001)
        .with_maximum_step(0.001)
        .with_error_tolerance(ErrorTolerance::scalar(1e-8, 1e-6, NormSelection::Infinity).unwrap());
    let plan = (method, options);
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(system, 0.01),
            &plan.1,
            OdeState::new(0., Vector::from_vec(y.clone())),
        )
        .unwrap();
    let report = *solution.report();
    assert!(report.is_success());
    let eigen = -4.
        * 0.01
        * ((n + 1) * (n + 1)) as f64
        * (core::f64::consts::PI / (2 * (n + 1)) as f64).sin().powi(2);
    let actual = solution.state().values();
    for (&actual, &initial) in actual.iter().zip(&y) {
        assert!((actual - initial * (0.01 * eigen).exp()).abs() < 1e-7);
    }
    (actual.to_vec(), report)
}
fn krylov() -> OdeStageBackend {
    OdeStageBackend::Krylov {
        iterations: Budget::try_from(100).unwrap(),
        restart: Budget::try_from(20).unwrap(),
        operator_scratch: 0,
        preconditioner_scratch: 0,
    }
}
#[test]
fn method_of_lines_dense_sparse_and_operator_parity() {
    let n = 128;
    let system = Diffusion { n };
    let reference = run(&system, OdeStageBackend::Dense).0;
    for backend in [
        OdeStageBackend::Sparse {
            pattern: pattern(n),
            limits: SparseLuLimits::new(8 * n, 4 * 1024 * 1024),
        },
        krylov(),
    ] {
        let actual = run(&system, backend).0;
        for (&a, &b) in actual.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-8);
        }
    }
}
/// The diffusion right-hand side without any derivative provider.
struct RhsOnly(Diffusion);
impl OdeSystem<f64> for RhsOnly {
    fn dimension(&self) -> usize {
        self.0.n
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.0.rhs(t, y, out, cx)
    }
}
#[test]
fn finite_difference_jacobians_compose_with_every_stage_backend() {
    let n = 8;
    let reference = run(&Diffusion { n }, OdeStageBackend::Dense).0;
    let system = RhsOnly(Diffusion { n });
    let sparse = || OdeStageBackend::Sparse {
        pattern: pattern(n),
        limits: SparseLuLimits::new(8 * n, 1024 * 1024),
    };
    for strategy in [
        DerivativeStrategy::Forward,
        DerivativeStrategy::Central,
        DerivativeStrategy::AnalyticOrAutomaticOrForward,
    ] {
        for backend in [OdeStageBackend::Dense, sparse(), krylov()] {
            let krylov = matches!(backend, OdeStageBackend::Krylov { .. });
            let (actual, report) = run_with(&system, backend, strategy);
            // Difference probes of the colored sparse Jacobian and of the Krylov
            // products are right-hand-side evaluations charged to the solve.
            let calls = report.accounting().total().unwrap();
            assert!(calls.provider(EvaluationRole::RightHandSide) > report.accepted_steps());
            if !krylov {
                assert!(calls.provider(EvaluationRole::Factorization) > 0);
            }
            for (&a, &b) in actual.iter().zip(&reference) {
                assert!((a - b).abs() < 1e-7, "{strategy:?}: {a} vs {b}");
            }
        }
    }
}
#[test]
fn analytic_only_strategies_refuse_rhs_only_systems_on_every_backend() {
    let n = 8;
    let system = RhsOnly(Diffusion { n });
    for backend in [
        OdeStageBackend::Dense,
        OdeStageBackend::Sparse {
            pattern: pattern(n),
            limits: SparseLuLimits::new(8 * n, 1024 * 1024),
        },
        krylov(),
    ] {
        for strategy in [DerivativeStrategy::Analytic, DerivativeStrategy::Automatic] {
            let method = RosenbrockW::new(RosenbrockMethod::Ros34Pw2)
                .unwrap()
                .with_backend(backend.clone())
                .with_derivative_strategy(strategy);
            let plan = (method, OdeOptions::new().with_initial_step(1e-3));
            let problem = OdeProblem::new(&system, 1e-3);
            let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
            let mut state = OdeState::new(0., Vector::from_vec(vec![1.; n]));
            let error = plan
                .0
                .advance(
                    &problem,
                    &plan.1,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::UnsupportedConfiguration,
                    ..
                }
            ));
            assert_eq!(state.time(), 0.);
            assert_eq!(state.values(), [1.; 8]);
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::Failed
            );
        }
    }
}
/// Diffusion with an analytic sparse Jacobian that counts its evaluations.
struct CountedSparse {
    inner: Diffusion,
    calls: std::cell::Cell<usize>,
}
impl OdeSystem<f64> for CountedSparse {
    fn dimension(&self) -> usize {
        self.inner.n
    }
    fn time_derivative(
        &self,
        time: f64,
        state: &[f64],
        output: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.inner.time_derivative(time, state, output, cx)
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.inner.rhs(t, y, out, cx)
    }
    fn sparse_jacobian(
        &self,
        t: f64,
        y: &[f64],
        pattern: StructuralPatternView<'_>,
        values: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        self.inner.sparse_jacobian(t, y, pattern, values, cx)
    }
}
#[test]
fn sparse_stages_follow_the_configured_strategy() {
    let n = 8;
    let backend = OdeStageBackend::Sparse {
        pattern: pattern(n),
        limits: SparseLuLimits::new(8 * n, 1024 * 1024),
    };
    let reference = run(&Diffusion { n }, OdeStageBackend::Dense).0;
    for (strategy, analytic) in [
        (DerivativeStrategy::AnalyticOrAutomaticOrForward, true),
        (DerivativeStrategy::Analytic, true),
        (DerivativeStrategy::Forward, false),
        (DerivativeStrategy::Central, false),
    ] {
        let system = CountedSparse {
            inner: Diffusion { n },
            calls: std::cell::Cell::new(0),
        };
        let (actual, _) = run_with(&system, backend.clone(), strategy);
        assert_eq!(system.calls.get() > 0, analytic, "{strategy:?}");
        for (&a, &b) in actual.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
#[test]
fn nested_krylov_budget_returns_the_accepted_state() {
    let system = Diffusion { n: 8 };
    let method = RosenbrockW::new(RosenbrockMethod::Ros2)
        .unwrap()
        .with_backend(krylov());
    // The zero analytic time derivative is admitted as one provider evaluation.
    let options = OdeOptions::new()
        .with_initial_step(0.001)
        .with_max_evaluations(Budget::try_from(5).unwrap());
    let plan = (method, options);
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&system, 0.1),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![1.; 8])),
        )
        .unwrap();
    let report = solution.report();
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Evaluation))
    );
    assert_eq!(report.final_time(), 0.);
    assert_eq!(solution.state().time(), 0.);
    assert_eq!(solution.state().values(), [1.; 8]);
}

/// `inner` with one caller-scheduled step limit per proposal.
struct Scheduled<'a, Sys> {
    inner: &'a Sys,
    limits: &'a [f64],
    proposals: core::cell::Cell<usize>,
}
impl<Sys: OdeSystem<f64>> OdeSystem<f64> for Scheduled<'_, Sys> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn step_limit(
        &self,
        _: f64,
        _: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<f64>, SciOxideError> {
        let index = self.proposals.get();
        self.proposals.set(index + 1);
        Ok(self.limits.get(index).copied())
    }
    fn time_derivative(
        &self,
        time: f64,
        state: &[f64],
        output: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.inner.time_derivative(time, state, output, cx)
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.inner.rhs(t, y, out, cx)
    }
    fn jacobian(
        &self,
        t: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.inner.jacobian(t, y, j, cx)
    }
    fn sparse_jacobian(
        &self,
        t: f64,
        y: &[f64],
        pattern: StructuralPatternView<'_>,
        values: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.inner.sparse_jacobian(t, y, pattern, values, cx)
    }
    fn jacobian_product(
        &self,
        t: f64,
        y: &[f64],
        x: &[f64],
        out: &mut [f64],
        scratch: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.inner.jacobian_product(t, y, x, out, scratch, cx)
    }
    fn jacobian_is_constant(&self) -> bool {
        self.inner.jacobian_is_constant()
    }
}

#[test]
fn bdf_sparse_and_krylov_share_the_actual_node_history() {
    // A fixed-step method keeps a reduced step, so the schedule shrinks every step;
    // the end time halves the last one.
    let steps: Vec<f64> = (0..20).map(|i| 0.002 * 0.93_f64.powi(i)).collect();
    let end = steps[..19].iter().sum::<f64>() + 0.5 * steps[19];
    let trajectory = |backend: OdeStageBackend| -> Vec<f64> {
        let diffusion = Diffusion { n: 16 };
        let system = Scheduled {
            inner: &diffusion,
            limits: &steps,
            proposals: core::cell::Cell::new(0),
        };
        let options = OdeOptions::new()
            .with_error_tolerance(ErrorTolerance::default())
            .with_initial_step(1.);
        let plan = (Bdf::new(3).unwrap().with_backend(backend), options);
        let y = (0..16)
            .map(|i| (core::f64::consts::PI * (i + 1) as f64 / 17.).sin())
            .collect();
        let solution = plan
            .0
            .solve(
                &OdeProblem::new(&system, end),
                &plan.1,
                OdeState::new(0., Vector::from_vec(y)),
            )
            .unwrap();
        assert_eq!(
            solution.report().termination(),
            Some(OdeTermination::ReachedEnd)
        );
        assert_eq!(solution.report().accepted_steps(), 20);
        solution.state().values().to_vec()
    };
    let dense = trajectory(OdeStageBackend::Dense);
    for backend in [
        OdeStageBackend::Sparse {
            pattern: pattern(16),
            limits: SparseLuLimits::new(128, 1024 * 1024),
        },
        krylov(),
    ] {
        let result = trajectory(backend);
        for (a, b) in result.iter().zip(&dense) {
            assert!((a - b).abs() < 2e-7);
        }
    }
}

#[derive(Default)]
struct Preconditioned {
    shift: std::cell::Cell<Option<f64>>,
    inverse: std::cell::Cell<f64>,
    rebuilds: std::cell::Cell<usize>,
    calls: std::cell::Cell<usize>,
}
impl Preconditioner<f64> for Preconditioned {
    fn dimension(&self) -> usize {
        1
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn precondition_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        y[0] = self.inverse.get() * x[0];
        Ok(())
    }
}
impl OdeSystem<f64> for Preconditioned {
    fn dimension(&self) -> usize {
        1
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[f64],
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(0.);
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = -10. * y[0];
        Ok(())
    }
    fn step_limit(
        &self,
        t: f64,
        _: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<f64>, SciOxideError> {
        // Three steps of 2^-10, then one of 2^-9: every endpoint is exact.
        Ok(Some(if t < 3. * SMALL { SMALL } else { 2. * SMALL }))
    }
    fn jacobian_product(
        &self,
        t: f64,
        _: &[f64],
        x: &[f64],
        out: &mut [f64],
        _: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.rhs(t, x, out, cx).map(Some)
    }
    fn stage_preconditioner(
        &self,
        _: f64,
        _: &[f64],
        shift: f64,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<&dyn Preconditioner<f64>>, SciOxideError> {
        if self.shift.get() != Some(shift) {
            self.shift.set(Some(shift));
            self.inverse.set(1. / (1. + 10. * shift));
            self.rebuilds.set(self.rebuilds.get() + 1);
        }
        Ok(Some(self))
    }
}
const SMALL: f64 = 1. / 1024.;
#[test]
fn injected_preconditioner_reuses_its_shift_epoch() {
    let system = Preconditioned::default();
    let method = RosenbrockW::new(RosenbrockMethod::Ros34Pw2)
        .unwrap()
        .with_backend(krylov());
    let options = OdeOptions::new()
        .with_initial_step(SMALL)
        .with_output_schedule(OdeOutputSchedule::AcceptedSteps);
    let plan = (method, options);
    let problem = OdeProblem::new(&system, 5. * SMALL);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
    let mut observed = Vec::new();
    let mut sink = |t: f64, _: &[f64]| {
        observed.push((t, system.rebuilds.get(), system.calls.get()));
        Ok(())
    };
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut sink,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert_eq!(report.rejected_steps(), 0);
    let times: Vec<f64> = observed.iter().map(|&(t, ..)| t).collect();
    assert_eq!(times, [0., SMALL, 2. * SMALL, 3. * SMALL, 5. * SMALL]);
    let (_, rebuilds, calls) = observed[3];
    assert_eq!(rebuilds, 1);
    assert!(calls >= 12);
    assert_eq!(observed[4].1, 2);
}

/// One step of `h = 0.01` from the unit state.
fn coupled_next<M: OdeMethod<f64> + Clone>(method: M) -> Vec<f64> {
    let system = Diffusion { n: 8 };
    let options = OdeOptions::new()
        .with_error_tolerance(ErrorTolerance::default())
        .with_initial_step(0.01);
    let plan = (method, options);
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&system, 0.01),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![1.; 8])),
        )
        .unwrap();
    assert_eq!(
        solution.report().termination(),
        Some(OdeTermination::ReachedEnd)
    );
    assert_eq!(solution.report().counts().proposals(), 1);
    solution.state().values().to_vec()
}
#[test]
fn coupled_and_diagonal_implicit_operator_parity() {
    for (dense, operator) in [
        (
            coupled_next(GaussRk::new()),
            coupled_next(GaussRk::new().with_backend(krylov())),
        ),
        (
            coupled_next(RadauIia::new()),
            coupled_next(RadauIia::new().with_backend(krylov())),
        ),
        (
            coupled_next(Sdirk::new()),
            coupled_next(Sdirk::new().with_backend(krylov())),
        ),
    ] {
        for (a, b) in dense.iter().zip(&operator) {
            assert!((a - b).abs() < 2e-8);
        }
    }
}
