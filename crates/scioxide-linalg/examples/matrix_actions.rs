//! Sparse exponential/phi actions and independently verified partial Ritz pairs.
use scioxide_core::{BlockVectorView, BlockVectorViewMut, ExecutionContext, SciOxideError};
use scioxide_linalg::{ExponentialAction, Matrix, PartialSpectrumSolver};
use scioxide_sparse::ExplicitZeroPolicy;
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::<f64>::from_vec(2, 2, vec![-1., 0.25, 0., -2.])?
        .to_sparse_csr(ExplicitZeroPolicy::Drop)?;
    let options = ExponentialAction::new(2.25).with_max_phi(1);
    let plan = options;
    let mut work = plan.workspace(&a, 1)?;
    let input = [1., 1.];
    let mut output = [0.; 2];
    for phi in [0, 1] {
        let report = plan.apply_into(
            &a,
            scioxide_linalg::PhiFunction::phi(phi, 1.),
            BlockVectorView::new(&input, 2, 1)?,
            BlockVectorViewMut::new(&mut output, 2, 1)?,
            &mut work,
            ExecutionContext::sequential_native(),
        )?;
        println!("phi_{phi}(A)b = {output:?}; {:?}", report.termination());
    }
    let spectrum_plan = PartialSpectrumSolver::new(1);
    let requirements = spectrum_plan.workspace_requirements(&a)?;
    let mut workspace = scioxide_linalg::PartialSpectrumWorkspace::new();
    workspace.try_reserve(&requirements)?;
    let mut spectrum = scioxide_linalg::PartialSpectrumBuffer::new(2, 1)?;
    spectrum_plan.solve_into(
        &a,
        input.as_slice(),
        &mut spectrum,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "Ritz values = {:?}; true residuals = {:?}",
        spectrum.eigenvalues(),
        spectrum.residuals()
    );
    Ok(())
}
