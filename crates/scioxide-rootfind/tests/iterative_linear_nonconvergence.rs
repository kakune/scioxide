//! Regression tests for nested iterative linear-solver outcomes.

use scioxide_core::LinearOperator;
use scioxide_core::SciOxideError;
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    IterativeLinearSolver, IterativeSolveReport, IterativeWorkspace, LinalgExecutionContext,
    VectorView, VectorViewMut,
};
use scioxide_rootfind::{NewtonSystem, SystemRootSolver};

struct AffineSystem;

impl scioxide_core::ResidualOperator<f64> for AffineSystem {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        2
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(
        &self,
        x: &[f64],
        out: &mut [f64],
        _scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[0] = x[0] - 1.0;
        out[1] = x[1] - 2.0;
        Ok(())
    }
}
impl scioxide_rootfind::DenseJacobian<f64> for AffineSystem {
    fn jacobian_into(
        &self,
        _x: &[f64],
        out: &mut [f64],
        _scratch: &mut [f64],
    ) -> Result<bool, SciOxideError> {
        out.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        Ok(true)
    }
}

struct NonConvergingLinearSolver;

impl IterativeLinearSolver<f64> for NonConvergingLinearSolver {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "NonConvergingLinearSolver", 1)
    }

    fn solve_into(
        &self,
        _a: &dyn LinearOperator<f64>,
        b: VectorView<'_, f64>,
        _preconditioning: Preconditioning<'_, f64>,
        mut x: VectorViewMut<'_, f64>,
        _workspace: &mut IterativeWorkspace<f64>,
        _context: LinalgExecutionContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        b.copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (x).reborrow(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok(IterativeSolveReport::new(
            scioxide_linalg::LinearSolveTermination::IterationLimit,
            7,
            Some(scioxide_core::ResidualReport::unscaled(
                0.25,
                scioxide_core::NormSelection::L2,
            )?),
            scioxide_core::PartialResultTrust::ResidualChecked,
        ))
    }
}

struct StructurallyFailingLinearSolver;

impl IterativeLinearSolver<f64> for StructurallyFailingLinearSolver {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "StructurallyFailingLinearSolver", 1)
    }

    fn solve_into(
        &self,
        _a: &dyn LinearOperator<f64>,
        _b: VectorView<'_, f64>,
        _preconditioning: Preconditioning<'_, f64>,
        _x: VectorViewMut<'_, f64>,
        _workspace: &mut IterativeWorkspace<f64>,
        _context: LinalgExecutionContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(2),
            scioxide_core::Dimensions::one(1),
        ))
    }
}

#[test]
fn system_solvers_reject_unconverged_injected_linear_results() {
    fn check(solver: &impl SystemRootSolver<f64>, initial: &[f64]) {
        let expected_detail = "nested linear solve did not converge";
        let result = solver.solve(&AffineSystem, initial);

        let error = result.unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, detail, .. }
                if diagnostics.counts().own().iterations() == 0
                    && diagnostics.nested_levels().next().is_some_and(|nested| {
                        nested.algorithm().method().ends_with("NonConvergingLinearSolver")
                            && nested.counts().own().iterations() == 7
                            && nested.residual().map(scioxide_core::ResidualReport::raw_norm)
                                == Some(0.25)
                    })
                    && detail == expected_detail
        ));
    }
    check(
        &NewtonSystem::new(
            (scioxide_rootfind::NewtonSystemOptions::default()
                .with_linear_solver(NonConvergingLinearSolver))
            .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap(),
        &[0.0, 0.0],
    );
}

#[test]
fn system_solver_preserves_injected_structural_failure() {
    let solver = NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default()
            .with_linear_solver(StructurallyFailingLinearSolver))
        .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap();
    let result = solver.solve(&AffineSystem, &[0.0, 0.0]);

    let error = result.unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { expected, actual, .. }
            if expected == [2].as_slice() && actual == [1].as_slice()
    ));
}

#[test]
fn trust_region_uses_its_qr_factor_for_the_direct_step() {
    use scioxide_core::EvaluationRole;
    let plan = scioxide_rootfind::TrustRegionSystem::new(
        scioxide_rootfind::TrustRegionSystemOptions::default()
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap();
    let result = plan.solve(&AffineSystem, &[0.0, 0.0]).unwrap();
    assert!(result.is_success());
    assert!(result.residual_norm().unwrap() < 1e-12);
    assert_eq!(
        result
            .accounting()
            .own()
            .provider(EvaluationRole::Factorization),
        1
    );
    assert!(
        result
            .accounting()
            .own()
            .provider(EvaluationRole::LinearSolve)
            > 0
    );
    assert_eq!(result.accounting().nested().total_providers().unwrap(), 0);
}
