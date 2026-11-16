//! Recycle a nonsymmetric solve, then solve coupled dependent right-hand sides.
use scioxide_core::{
    BlockVectorView, Budget, ExecutionContext, ProblemToken, ResidualTolerance, SciOxideError,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    BlockGmresSolver, GcrotSolver, IterativeWorkspaceRequest, KrylovRecycleSpace, Matrix, Vector,
};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(3, 3, vec![4., 1., 0., 0., 3., 1., 1., 0., 2.])?;
    let a = a.view().try_finite()?;
    let b = Vector::from_vec(vec![5., 4., 3.]);
    let tolerance = ResidualTolerance::new(1e-11, 0.)?;
    let solver = GcrotSolver::new(tolerance, Budget::try_from(30)?);
    let mut workspace = scioxide_linalg::IterativeWorkspace::new();
    let operator_epoch = ProblemToken::new(1);
    let mut recycle = KrylovRecycleSpace::new(3, 3, operator_epoch)?;
    workspace.try_reserve(&solver.workspace_requirements(
        &IterativeWorkspaceRequest::operator(&a)?.with_recycle_space(&recycle)?,
    )?)?;
    for _ in 0..2 {
        let mut x = Vector::zeros(3)?;
        let report = solver.solve_recycled_into(
            &a,
            &b,
            Preconditioning::None,
            operator_epoch,
            &mut recycle,
            x.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        println!(
            "GCROT: {:?}, {} inner steps, x={:?}",
            report.termination(),
            report.iterations(),
            x.as_slice()
        );
    }
    let rhs = [5., 4., 3., 10., 8., 6., 0., 0., 0.];
    let result = BlockGmresSolver::new(tolerance, Budget::try_from(30)?).solve(
        &a,
        BlockVectorView::new(&rhs, 3, 3)?,
        Preconditioning::None,
        None,
    )?;
    println!(
        "Block GMRES: {:?}, {} deflated directions",
        result.report().termination(),
        result.report().deflated()
    );
    Ok(())
}
