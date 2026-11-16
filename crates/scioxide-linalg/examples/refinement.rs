//! Reuse equilibrated f32 factors while requesting f64 solve accuracy.
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance, SciOxideError};
use scioxide_linalg::{
    EquilibrationMetadata, IterativeRefinement, LuDecomposition, Matrix, RefinementCriterion,
    RefinementFallback, RefinementOptions, RefinementProblem, RefinementSystem, Vector,
};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(2, 2, vec![4f64, 1., 2., 3.])?;
    let system = RefinementSystem::Dense(a.view());
    let scaling = EquilibrationMetadata::prepare(system)?;
    let mut low = Matrix::<f32>::zeros(2, 2)?;
    scaling.scale_matrix_into(a.view(), &mut low.view_mut())?;
    let low_factor = LuDecomposition::factor(&low)?;
    let low_inverse = low_factor.inverse_operator();
    // Fallback solves the original system, in solution/residual precision.
    let high_factor = LuDecomposition::factor(&a)?;
    let high_inverse = high_factor.inverse_operator();
    let problem =
        RefinementProblem::new(system, &low_inverse, Some(&scaling), Some(&high_inverse))?;
    let plan = IterativeRefinement::new(
        RefinementOptions::new(ResidualTolerance::new(0., 16. * f64::EPSILON)?)
            .with_criterion(RefinementCriterion::Componentwise)
            .with_fallback(RefinementFallback::OnFailure)
            .with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new()
                    .with_iteration_budget(Budget::ONE.saturating_add(11))
                    .with_byte_limit(4096),
            ),
    )?;
    let mut workspace =
        plan.workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))?;
    let b = Vector::from_vec(vec![5., 5.]);
    let mut x = Vector::zeros(2)?;
    let report = plan.refine_into(
        &problem,
        b.view(),
        x.view_mut(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?[0];
    println!("x={:?}, report={report:?}", x.as_slice());
    Ok(())
}
