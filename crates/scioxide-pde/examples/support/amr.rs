//! Shared deterministic AMR fixtures and caller-owned solver composition.
#![allow(dead_code)]
use scioxide_core::{
    Budget, ExecutionContext, LinearOperator, NumericScalar, ProblemToken, RealScalar,
    ResidualTolerance, SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveState,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{CgSolver, Vector};
use scioxide_pde::{AmrBoundaryTags, AmrLimits};
use scioxide_spatial::*;

pub fn limits() -> AmrLimits {
    AmrLimits::new(100_000, 5_000_000, 4096, 500_000_000, 1024 * 1024 * 1024)
}
pub fn topology_limits() -> AdaptationLimits {
    AdaptationLimits::new()
        .with_max_depth(12)
        .with_max_cells(20_000)
        .with_max_entities(200_000)
        .with_max_work(500_000_000)
        .with_max_storage_bytes(1024 * 1024 * 1024)
}
pub fn tensor<R: RealScalar, const D: usize>(
    n: usize,
    periodic: bool,
) -> Result<AdaptiveMesh<R, D>, SciOxideError> {
    let axis = CoordinateAxis::uniform(R::ZERO, R::ONE, n + 1)?;
    let grid = StructuredGrid::new(
        core::array::from_fn(|_| axis.clone()),
        [if periodic {
            AxisTopology::Periodic
        } else {
            AxisTopology::Bounded
        }; D],
    )?;
    AdaptiveMesh::from_structured(&CellMesh::new(grid)?, topology_limits())
}
pub fn adapt<R: RealScalar, const D: usize>(
    mesh: &AdaptiveMesh<R, D>,
    refine: &[usize],
    coarsen: &[usize],
) -> Result<AdaptationResult<R, D>, SciOxideError> {
    mesh.adapt_scoped(
        AdaptationRequest {
            source: mesh.mesh_token(),
            refine,
            coarsen,
        },
        scioxide_core::ExecutionContext::sequential(),
    )
}
pub fn tags<R: RealScalar, const D: usize>(
    mesh: &AdaptiveMesh<R, D>,
) -> Result<AmrBoundaryTags<D>, SciOxideError> {
    let entries: Vec<_> = mesh
        .interfaces()
        .iter()
        .filter(|i| i.right.is_none())
        .map(|i| (i.left_face, 0))
        .collect();
    AmrBoundaryTags::new(mesh, &entries)
}
pub fn context<T>(
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        ProblemToken::new(635),
        scioxide_core::algorithm_id!("pde", "amr", 1),
        0,
        0,
    );
    let mut context = state.begin_solve(ExecutionContext::sequential(), id, &mut budgets)?;
    run(&mut context)
}
pub fn solve<V: NumericScalar>(
    operator: &dyn LinearOperator<V>,
    rhs: &[V],
    output: &mut [V],
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    // Portable CG needs rounding headroom, while transfer consumers validate
    // projection residuals at 64ε. Keep the solve target stricter than that gate.
    let tolerance = V::Real::EPSILON * V::Real::try_from_f64_lossy(32.)?;
    let solver = CgSolver::new(
        ResidualTolerance::new(tolerance, tolerance)?,
        Budget::try_from(5000)?,
    );
    let mut work = scioxide_linalg::IterativeWorkspace::new();
    let rhs = Vector::from_vec(rhs.to_vec());
    let mut x = Vector::zeros(rhs.len())?;
    let report = solver.solve_into(
        operator,
        &rhs,
        Preconditioning::None,
        x.view_mut(),
        &mut work,
        &mut *context,
    )?;
    if !report.is_success() {
        return Err(SciOxideError::external("injected CG failed to converge"));
    }
    output.copy_from_slice(x.as_slice());
    Ok(())
}
