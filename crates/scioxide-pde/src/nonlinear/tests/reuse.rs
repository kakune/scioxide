use super::*;
use core::cell::Cell;
use scioxide_core::{
    Budget, BudgetKind, CancellationToken, ExecutionContext, SolveBudgets, SolveIdentity,
    SolveState, StructuralPatternView,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Failure {
    None,
    Missing,
    Incomplete,
    NonFinite,
    Callback,
    Cancel,
}

struct CountingProvider<'a> {
    calls: Cell<usize>,
    failure: Cell<Failure>,
    coefficient: Cell<f64>,
    cancellation: Option<&'a CancellationToken>,
}
impl CountingProvider<'_> {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
            failure: Cell::new(Failure::None),
            coefficient: Cell::new(1.),
            cancellation: None,
        }
    }
    fn derivative(
        &self,
        time: f64,
        x: &[f64],
        mut write: impl FnMut(usize, usize, f64) -> Result<(), SciOxideError>,
    ) -> Result<Option<()>, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        let failure = self.failure.get();
        if failure == Failure::Missing {
            return Ok(None);
        }
        for (row, &x) in x.iter().enumerate() {
            for col in 0..3 {
                if failure == Failure::Incomplete && row == 2 && col == 2 {
                    continue;
                }
                let value = if failure == Failure::NonFinite && row == col {
                    f64::INFINITY
                } else if row == col {
                    2. * (row + 1) as f64 * x * (self.coefficient.get() + time)
                } else {
                    0.
                };
                write(row, col, value)?;
            }
        }
        if failure == Failure::Callback {
            return Err(invalid(
                scioxide_core::argument!("fixture"),
                "injected failure",
            ));
        }
        if failure == Failure::Cancel {
            self.cancellation.unwrap().cancel();
        }
        Ok(Some(()))
    }
}

impl OdeSystem<f64> for CountingProvider<'_> {
    fn dimension(&self) -> usize {
        3
    }
    fn rhs(
        &self,
        time: f64,
        x: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for (i, (out, &x)) in out.iter_mut().zip(x).enumerate() {
            let target = (i + 1) as f64;
            *out = (self.coefficient.get() + time) * target * (x * x - target * target);
        }
        Ok(())
    }
    fn jacobian(
        &self,
        time: f64,
        x: &[f64],
        out: &mut scioxide_linalg::MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.derivative(time, x, |row, col, value| out.set(row, col, value))
    }
    fn jacobian_product(
        &self,
        time: f64,
        x: &[f64],
        direction: &[f64],
        out: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        for (i, ((out, &x), &v)) in out.iter_mut().zip(x).zip(direction).enumerate() {
            *out = 2. * (i + 1) as f64 * x * (self.coefficient.get() + time) * v;
        }
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        time: f64,
        x: &[f64],
        _: StructuralPatternView<'_>,
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.derivative(time, x, |row, col, value| {
            if row == col {
                out[row] = value;
            }
            Ok(())
        })
    }
}

fn modes() -> [PdeJacobianMode; 2] {
    [
        PdeJacobianMode::Dense,
        PdeJacobianMode::Sparse(
            StructuralPattern::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2]).unwrap(),
        ),
    ]
}

fn assert_product(actual: &[f64], expected: &[f64]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn matrix_free_products_keep_calling_the_provider() {
    let provider = CountingProvider::new();
    let adjoint_calls = Cell::new(0);
    let adjoint = |_: f64, x: &[f64], v: &[f64], out: &mut [f64], _: &mut [f64]| {
        adjoint_calls.set(adjoint_calls.get() + 1);
        for (i, ((out, &x), &v)) in out.iter_mut().zip(x).zip(v).enumerate() {
            *out = 2. * (i + 1) as f64 * x * v;
        }
        Ok(())
    };
    let plan_fields = PdeFieldLayout::new(&[3]).unwrap();
    let plan_mode = PdeJacobianMode::Operator {
        operator_scratch: 0,
    };
    let mut workspace = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
        .workspace()
        .unwrap();
    let residual = NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
        .with_adjoint(&adjoint, 0)
        .bind(&mut workspace)
        .unwrap();
    let mut out = [0.; 3];
    let mut scratch = [0.; 3];
    context(|cx| {
        for _ in 0..3 {
            residual.jvp_into_scoped(&[1.; 3], &[1.; 3], &mut out, &mut scratch, cx)?;
            assert_product(&out, &[2., 4., 6.]);
            residual.vjp_into_scoped(&[1.; 3], &[1.; 3], &mut out, &mut scratch, cx)?;
            assert_product(&out, &[2., 4., 6.]);
        }
        assert_eq!(cx.provider_count(EvaluationRole::Jacobian)?, 0);
        Ok(())
    })
    .unwrap();
    assert_eq!(provider.calls.get(), 3);
    assert_eq!(adjoint_calls.get(), 3);
}

#[test]
fn newton_assembles_once_per_linearization() {
    for mode in modes() {
        let provider = CountingProvider::new();
        let plan_fields = PdeFieldLayout::new(&[3]).unwrap();
        let plan_mode = mode;
        let mut workspace = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
            .workspace()
            .unwrap();
        let residual = NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
            .bind(&mut workspace)
            .unwrap();
        let problem = NewtonKrylovProblem::with_jacobian(&residual);
        let opts = options()
            .with_history(true)
            .with_forcing(scioxide_rootfind::NewtonKrylovForcing::Constant(1e-10));
        let result = NewtonKrylovSolver::new(opts)
            .unwrap()
            .solve(&problem, &[0.7, 1.1, 1.9])
            .unwrap();
        assert!(result.is_success(), "{:?}", result.termination());
        let linearizations = result.history().len();
        let inner_iterations: usize = result.history().iter().map(|h| h.linear_iterations()).sum();
        assert!(linearizations > 1);
        assert!(inner_iterations > linearizations);
        assert_eq!(provider.calls.get(), linearizations);
        assert_eq!(
            result
                .accounting()
                .total()
                .unwrap()
                .provider(EvaluationRole::Jacobian),
            linearizations
        );
        for (&x, expected) in result.point().iter().zip([1., 2., 3.]) {
            assert!((x - expected).abs() < 1e-9);
        }
    }
}

#[test]
fn rebinding_resets_provider_and_time() {
    for mode in modes() {
        let provider = CountingProvider::new();
        let other = CountingProvider::new();
        let plan_fields = PdeFieldLayout::new(&[3]).unwrap();
        let plan_mode = mode;
        let mut workspace = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
            .workspace()
            .unwrap();
        let mut out = [0.; 3];
        let mut scratch = [0.; 6];
        for (provider, time, expected) in
            [(&provider, 0., 2.), (&provider, 1., 4.), (&other, 2., 6.)]
        {
            let residual = NonlinearPdeProblem::new(provider, time, &plan_fields, &plan_mode)
                .bind(&mut workspace)
                .unwrap();
            residual
                .jvp_into(&[1.; 3], &[1.; 3], &mut out, &mut scratch)
                .unwrap();
            assert!((out[0] - expected).abs() < 1e-14);
        }
        assert_eq!(provider.calls.get(), 2);
        assert_eq!(other.calls.get(), 1);
        other.coefficient.set(3.);
        let residual = NonlinearPdeProblem::new(&other, 2., &plan_fields, &plan_mode)
            .bind(&mut workspace)
            .unwrap();
        residual
            .jvp_into(&[1.; 3], &[1.; 3], &mut out, &mut scratch)
            .unwrap();
        assert_product(&out, &[10., 20., 30.]);
        assert_eq!(other.calls.get(), 2);
    }
}

#[test]
fn failed_assembly_never_reuses_partial_or_stale_values() {
    for mode in modes() {
        let provider = CountingProvider::new();
        let plan_fields = PdeFieldLayout::new(&[3]).unwrap();
        let plan_mode = mode;
        let mut workspace = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
            .workspace()
            .unwrap();
        let residual = NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
            .bind(&mut workspace)
            .unwrap();
        let mut out = [0.; 3];
        let mut scratch = [0.; 6];
        for failure in [
            Failure::Missing,
            Failure::Incomplete,
            Failure::NonFinite,
            Failure::Callback,
        ] {
            provider.failure.set(Failure::None);
            residual
                .jvp_into(&[1.; 3], &[1.; 3], &mut out, &mut scratch)
                .unwrap();
            provider.failure.set(failure);
            out.fill(77.);
            assert!(
                residual
                    .vjp_into(&[2.; 3], &[1.; 3], &mut out, &mut scratch)
                    .is_err()
            );
            assert_eq!(out, [77.; 3]);
            assert!(!residual.workspace.borrow().valid);
            let calls = provider.calls.get();
            provider.failure.set(Failure::None);
            residual
                .jvp_into(&[2.; 3], &[1.; 3], &mut out, &mut scratch)
                .unwrap();
            assert_eq!(provider.calls.get(), calls + 1);
            assert_product(&out, &[4., 8., 12.]);
        }
    }
}

#[test]
fn assembly_budget_and_cancellation_do_not_publish_cached_state() {
    for mode in modes() {
        let cancellation = CancellationToken::new();
        let mut provider = CountingProvider::new();
        provider.cancellation = Some(&cancellation);
        let plan_fields = PdeFieldLayout::new(&[3]).unwrap();
        let plan_mode = mode;
        let mut workspace = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
            .workspace()
            .unwrap();
        let residual = NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
            .bind(&mut workspace)
            .unwrap();
        let mut scratch = [0.; 6];
        let mut out = [0.; 3];
        residual
            .jvp_into(&[1.; 3], &[1.; 3], &mut out, &mut scratch)
            .unwrap();
        let identity = SolveIdentity::new(
            scioxide_core::ProblemToken::new(2352),
            scioxide_core::algorithm_id!("test", "pde-assembly", 1),
            0,
            0,
        );
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
        let mut cx = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        cx.record_provider(EvaluationRole::Residual)
            .unwrap()
            .require()
            .unwrap();
        // A reused matrix consumes no nested assembly admission.
        residual
            .jvp_into_scoped(&[1.; 3], &[1.; 3], &mut out, &mut scratch, &mut cx)
            .unwrap();
        out.fill(77.);
        assert!(
            residual
                .jvp_into_scoped(&[2.; 3], &[1.; 3], &mut out, &mut scratch, &mut cx)
                .is_err()
        );
        assert_eq!(provider.calls.get(), 1);
        assert_eq!(out, [77.; 3]);
        assert!(!residual.workspace.borrow().valid);
        provider.failure.set(Failure::Cancel);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut cx = state
            .begin_solve(
                ExecutionContext::sequential().with_cancellation(&cancellation),
                identity,
                &mut budgets,
            )
            .unwrap();
        assert!(
            residual
                .jvp_into_scoped(&[2.; 3], &[1.; 3], &mut out, &mut scratch, &mut cx)
                .is_err()
        );
        assert_eq!(out, [77.; 3]);
        assert!(!residual.workspace.borrow().valid);
        cancellation.reset();
        provider.failure.set(Failure::None);
        residual
            .jvp_into_scoped(&[2.; 3], &[1.; 3], &mut out, &mut scratch, &mut cx)
            .unwrap();
        assert_eq!(provider.calls.get(), 3);
        assert_product(&out, &[4., 8., 12.]);
    }
}
