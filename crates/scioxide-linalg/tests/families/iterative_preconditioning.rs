//! Borrowed preconditioning, object-safe forwarding, and shareable plans.
#![allow(clippy::unwrap_used)]

use core::cell::Cell;

use scioxide_core::{
    AttemptOutcome, BlockVectorView, BlockVectorViewMut, Budget, ExecutionContext,
    InvalidInputReason, Preconditioner, ProblemToken, ResidualTolerance, SciOxideError,
    SciOxideErrorView, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::*;

fn tolerance() -> ResidualTolerance<f64> {
    ResidualTolerance::new(1e-11, 0.0).unwrap()
}

fn limit() -> Budget {
    Budget::try_from(100).unwrap()
}

// Deliberately neither Clone nor Sync: borrowing a provider must not change a plan's traits.
struct Inverse {
    calls: Cell<usize>,
    scale: f64,
    dimension: usize,
    fail: bool,
}

impl Inverse {
    fn new(scale: f64) -> Self {
        Self {
            calls: Cell::new(0),
            scale,
            dimension: 3,
            fail: false,
        }
    }
}

impl Preconditioner<f64> for Inverse {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(5)
    }
    fn precondition_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(scratch.len(), 5);
        if self.fail {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "inverse failure",
            ));
        }
        for ((y, x), d) in y.iter_mut().zip(x).zip([2.0, 3.0, 4.0]) {
            *y = self.scale * x / d;
        }
        Ok(())
    }
}

fn matrix() -> Matrix<f64> {
    Matrix::from_vec(3, 3, vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]).unwrap()
}

use crate::support::comparisons::slices_within as close;
macro_rules! scalar_contract {
    ($test:ident, $solver:expr) => {
        #[test]
        fn $test() {
            let plan = $solver;
            let matrix = matrix();
            let a = matrix.view().try_finite().unwrap();
            let b = [2.0, 6.0, 12.0];
            let inverse = Inverse::new(1.0);
            let p = Preconditioning::Inverse(&inverse);
            let owned = plan.solve_with_options(&a, b.as_slice(), p, None).unwrap();
            assert!(owned.is_success());
            assert!(inverse.calls.get() > 0);
            let solver: &dyn IterativeLinearSolver<f64> = &plan;
            inverse.calls.set(0);
            let dynamic = solver
                .solve_with_options(&a, VectorView::from_slice(&b), p, None)
                .unwrap();
            assert!(dynamic.is_success());
            assert!(inverse.calls.get() > 0);
            close(dynamic.solution().as_slice(), owned.solution().as_slice());

            let request = IterativeWorkspaceRequest::operator(&a)
                .unwrap()
                .with_preconditioner(p)
                .unwrap();
            let mut workspace = solver.workspace(&request).unwrap();
            // The same immutable plan and workspace accept a different inverse on the next call.
            for inverse in [Inverse::new(1.0), Inverse::new(0.5)] {
                let mut x = [0.0; 3];
                let report = solver
                    .solve_into(
                        &a,
                        VectorView::from_slice(&b),
                        Preconditioning::Inverse(&inverse),
                        VectorViewMut::from_slice(&mut x),
                        &mut workspace,
                        ExecutionContext::sequential().into(),
                    )
                    .unwrap();
                assert!(report.is_success());
                assert!(inverse.calls.get() > 0);
                close(&x, &[1.0, 2.0, 3.0]);
            }

            let request = request.with_right_hand_sides(2);
            let mut workspace = solver.workspace(&request).unwrap();
            let block = [2.0, 6.0, 12.0, -2.0, -6.0, -12.0];
            let mut x = [0.0; 6];
            solver
                .solve_block_into(
                    &a,
                    BlockVectorView::new(&block, 3, 2).unwrap(),
                    p,
                    BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                    &mut workspace,
                    ExecutionContext::sequential().into(),
                )
                .unwrap();
            close(&x, &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);

            let mut x = [7.0; 3];
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    SolveIdentity::new(ProblemToken::new(1218), solver.algorithm_id(), 0, 0),
                    &mut budgets,
                )
                .unwrap();
            let before = context.counts();
            let invalid = Inverse {
                dimension: 2,
                ..Inverse::new(1.0)
            };
            let error = solver
                .solve_into(
                    &a,
                    VectorView::from_slice(&b),
                    Preconditioning::Inverse(&invalid),
                    VectorViewMut::from_slice(&mut x),
                    &mut workspace,
                    (&mut context).into(),
                )
                .unwrap_err();
            assert!(matches!(
                error.view(),
                SciOxideErrorView::DimensionMismatch { .. }
            ));
            assert_eq!(invalid.calls.get(), 0);
            assert_eq!(x, [7.0; 3]);

            assert_eq!(context.counts(), before);
            let failing = Inverse {
                fail: true,
                ..Inverse::new(1.0)
            };
            let error = solver
                .solve_into(
                    &a,
                    VectorView::from_slice(&b),
                    Preconditioning::Inverse(&failing),
                    VectorViewMut::from_slice(&mut x),
                    &mut workspace,
                    ExecutionContext::sequential().into(),
                )
                .unwrap_err();
            assert!(matches!(error.view(), SciOxideErrorView::NonFinite { .. }));
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::Failed
            );
            assert_eq!(x, [7.0; 3]);
        }
    };
}

scalar_contract!(gmres, GmresSolver::new(tolerance(), limit()));
scalar_contract!(cg, CgSolver::new(tolerance(), limit()));
scalar_contract!(minres, MinresSolver::new(tolerance(), limit()));
scalar_contract!(
    richardson,
    RichardsonSolver::new(tolerance(), limit(), RelaxationWeight::one())
);
scalar_contract!(bicgstab, BicgstabSolver::new(tolerance(), limit()));
scalar_contract!(cgs, CgsSolver::new(tolerance(), limit()));
scalar_contract!(tfqmr, TfqmrSolver::new(tolerance(), limit()));

scalar_contract!(lgmres, LgmresSolver::new(tolerance(), limit()));
scalar_contract!(gcrot, GcrotSolver::new(tolerance(), limit()));

struct AdjointInverse {
    forward: Cell<usize>,
    adjoint: Cell<usize>,
}
impl scioxide_core::LinearOperator<f64> for AdjointInverse {
    fn rows(&self) -> usize {
        3
    }
    fn cols(&self) -> usize {
        3
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(2)
    }
    fn apply_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), 2);
        self.forward.set(self.forward.get() + 1);
        for ((y, x), d) in y.iter_mut().zip(x).zip([2.0, 3.0, 4.0]) {
            *y = x / d;
        }
        Ok(())
    }
}
impl scioxide_core::AdjointOperator<f64> for AdjointInverse {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(7)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), 7);
        self.adjoint.set(self.adjoint.get() + 1);
        for ((y, x), d) in y.iter_mut().zip(x).zip([2.0, 3.0, 4.0]) {
            *y = x / d;
        }
        Ok(())
    }
}

#[test]
fn adjoint_inverse_reserves_both_orientations() {
    let matrix = matrix();
    let a = matrix.view().try_finite().unwrap();
    let inverse = AdjointInverse {
        forward: Cell::new(0),
        adjoint: Cell::new(0),
    };
    let p = Preconditioning::InverseAdjoint(&inverse);
    let request = IterativeWorkspaceRequest::adjoint(&a)
        .unwrap()
        .with_preconditioner(p)
        .unwrap();
    macro_rules! check {
        ($solver:ident) => {{
            let plan = $solver::new(tolerance(), limit());
            let mut workspace = plan.workspace(&request).unwrap();
            let mut x = [0.0; 3];
            inverse.forward.set(0);
            inverse.adjoint.set(0);
            let report = plan
                .solve_into(
                    &a,
                    &[2.0, 6.0, 12.0][..],
                    p,
                    VectorViewMut::from_slice(&mut x),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.is_success());
            assert!(inverse.forward.get() > 0);
            assert!(inverse.adjoint.get() > 0);
            close(&x, &[1.0, 2.0, 3.0]);
            let before = x;
            let forward_only = Inverse::new(1.0);
            let error = plan
                .solve_into(
                    &a,
                    &[2.0, 6.0, 12.0][..],
                    Preconditioning::Inverse(&forward_only),
                    VectorViewMut::from_slice(&mut x),
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
            assert_eq!(forward_only.calls.get(), 0);
            assert_eq!(x, before);
        }};
    }
    check!(BicgSolver);
    check!(QmrSolver);
}

#[test]
fn rectangular_and_matrix_splitting_methods_need_no_preconditioning_argument() {
    let matrix = matrix();
    let a = matrix.view().try_finite().unwrap();
    let b = [2.0, 6.0, 12.0];
    let results = [
        LsqrSolver::new(tolerance(), limit())
            .solve_with_options(&a, &b[..], None)
            .unwrap(),
        LsmrSolver::new(tolerance(), limit())
            .solve_with_options(&a, &b[..], None)
            .unwrap(),
        JacobiSolver::new(tolerance(), limit())
            .solve_with_options(StationaryMatrix::Dense(a), &b[..], None)
            .unwrap(),
        GaussSeidelSolver::new(tolerance(), limit())
            .solve_with_options(StationaryMatrix::Dense(a), &b[..], None)
            .unwrap(),
        SorSolver::new(tolerance(), limit(), SorWeight::new(1.0).unwrap())
            .solve_with_options(StationaryMatrix::Dense(a), &b[..], None)
            .unwrap(),
    ];
    for result in results {
        assert!(result.is_success());
        close(result.solution().as_slice(), &[1.0, 2.0, 3.0]);
    }
}

#[test]
fn block_and_recycled_calls_borrow_reserved_preconditioners() {
    let matrix = matrix();
    let a = matrix.view().try_finite().unwrap();
    let inverse = Inverse::new(1.0);
    let p = Preconditioning::Inverse(&inverse);
    let b = [2.0, 6.0, 12.0, -2.0, -6.0, -12.0];
    let request = IterativeWorkspaceRequest::operator(&a)
        .unwrap()
        .with_preconditioner(p)
        .unwrap();
    macro_rules! block {
        ($solver:ident) => {{
            let plan = $solver::new(tolerance(), limit());
            let mut workspace = plan.workspace(&request.with_right_hand_sides(2)).unwrap();
            let mut x = [0.0; 6];
            let report = plan
                .solve_into(
                    &a,
                    BlockVectorView::new(&b, 3, 2).unwrap(),
                    p,
                    BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.is_success());
            close(&x, &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
        }};
    }
    block!(BlockGmresSolver);
    block!(BlockLgmresSolver);
    block!(BlockGcrotSolver);
    macro_rules! recycled {
        ($solver:ident, $columns:expr, $rhs:expr, $output:expr) => {{
            let plan = $solver::new(tolerance(), limit());
            let request = if $columns == 1 {
                request
            } else {
                request.with_right_hand_sides($columns)
            };
            let mut workspace = IterativeWorkspace::new();
            let epoch = ProblemToken::new(1100);
            let mut recycle = KrylovRecycleSpace::new(3, 2, epoch).unwrap();
            workspace
                .try_reserve(
                    &plan
                        .workspace_requirements(&request.with_recycle_space(&recycle).unwrap())
                        .unwrap(),
                )
                .unwrap();
            for _ in 0..2 {
                let mut x = [0.0; 6];
                let report = plan
                    .solve_recycled_into(
                        &a,
                        $rhs(&b),
                        p,
                        epoch,
                        &mut recycle,
                        $output(&mut x),
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert!(report.is_success());
                close(
                    &x[..3 * $columns],
                    &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0][..3 * $columns],
                );
            }
        }};
    }
    fn scalar_rhs(b: &[f64; 6]) -> VectorView<'_, f64> {
        VectorView::from_slice(&b[..3])
    }
    fn scalar_out(x: &mut [f64; 6]) -> VectorViewMut<'_, f64> {
        VectorViewMut::from_slice(&mut x[..3])
    }
    fn block_rhs(b: &[f64; 6]) -> BlockVectorView<'_, f64> {
        BlockVectorView::new(b, 3, 2).unwrap()
    }
    fn block_out(x: &mut [f64; 6]) -> BlockVectorViewMut<'_, f64> {
        BlockVectorViewMut::new(x, 3, 2).unwrap()
    }
    recycled!(LgmresSolver, 1, scalar_rhs, scalar_out);
    recycled!(GcrotSolver, 1, scalar_rhs, scalar_out);
    recycled!(BlockLgmresSolver, 2, block_rhs, block_out);
    recycled!(BlockGcrotSolver, 2, block_rhs, block_out);
}

#[test]
fn multiple_rhs_preflight_replaces_the_previous_attempt() {
    let matrix = matrix();
    let a = matrix.view().try_finite().unwrap();
    let plan = GmresSolver::new(tolerance(), limit());
    let solver: &dyn IterativeLinearSolver<f64> = &plan;
    let b = [2.0, 6.0, 12.0];
    let mut x = [0.0; 3];
    let mut workspace = IterativeWorkspace::new();
    solver
        .solve_block_into(
            &a,
            BlockVectorView::new(&b, 3, 1).unwrap(),
            Preconditioning::None,
            BlockVectorViewMut::new(&mut x, 3, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential().into(),
        )
        .unwrap();
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
    let before = x;
    let inverse = Inverse {
        dimension: 2,
        ..Inverse::new(1.0)
    };
    let error = solver
        .solve_block_into(
            &a,
            BlockVectorView::new(&b, 3, 1).unwrap(),
            Preconditioning::Inverse(&inverse),
            BlockVectorViewMut::new(&mut x, 3, 1).unwrap(),
            &mut workspace,
            ExecutionContext::sequential().into(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert!(workspace.last_accepted().is_none());
    assert_eq!(inverse.calls.get(), 0);
    assert_eq!(x, before);
}

#[test]
fn every_trait_implementation_accepts_the_same_inverse_values() {
    let matrix = matrix();
    let a = matrix.view().try_finite().unwrap();
    let forward = Inverse::new(1.0);
    let adjoint = AdjointInverse {
        forward: Cell::new(0),
        adjoint: Cell::new(0),
    };
    let solvers: [&dyn IterativeLinearSolver<f64>; 9] = [
        &CgSolver::new(tolerance(), limit()),
        &MinresSolver::new(tolerance(), limit()),
        &RichardsonSolver::new(tolerance(), limit(), RelaxationWeight::one()),
        &GmresSolver::new(tolerance(), limit()),
        &BicgstabSolver::new(tolerance(), limit()),
        &CgsSolver::new(tolerance(), limit()),
        &TfqmrSolver::new(tolerance(), limit()),
        &LgmresSolver::new(tolerance(), limit()),
        &GcrotSolver::new(tolerance(), limit()),
    ];
    let b = [2.0, 6.0, 12.0, -2.0, -6.0, -12.0];
    for inverse in [
        Preconditioning::Inverse(&forward),
        Preconditioning::InverseAdjoint(&adjoint),
    ] {
        for solver in solvers {
            forward.calls.set(0);
            adjoint.forward.set(0);
            let request = IterativeWorkspaceRequest::operator(&a)
                .unwrap()
                .with_preconditioner(inverse)
                .unwrap()
                .with_right_hand_sides(2);
            let mut workspace = solver.workspace(&request).unwrap();
            let mut x = [0.; 6];
            solver
                .solve_block_into(
                    &a,
                    BlockVectorView::new(&b, 3, 2).unwrap(),
                    inverse,
                    BlockVectorViewMut::new(&mut x, 3, 2).unwrap(),
                    &mut workspace,
                    ExecutionContext::sequential().into(),
                )
                .unwrap();
            close(&x, &[1., 2., 3., -1., -2., -3.]);
            assert!(forward.calls.get() + adjoint.forward.get() > 0);
            assert_eq!(adjoint.adjoint.get(), 0);
        }
    }
}
