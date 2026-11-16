//! Shared SR1 and strong-Wolfe BFGS with explicit curvature diagnostics.
use scioxide_core::SciOxideError;
use scioxide_linalg::Matrix;
use scioxide_optimize::{Bfgs, HessianForm, HessianUpdate, HessianUpdateWorkspace, ObjectiveFn};

fn main() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + 4. * x[1] * x[1]))
        .with_gradient(|x: &[f64], output| {
            output.copy_from_slice(&[2. * x[0], 8. * x[1]]);
            Ok(())
        });
    let result = Bfgs::<f64>::default().minimize(&objective, &[1., -1.])?;
    println!(
        "termination: {:?}; updates: {:?}",
        result.termination(),
        result.method_diagnostics()
    );
    let mut model = Matrix::from_vec(2, 2, vec![1., 0., 0., 1.])?;
    let mut workspace = HessianUpdateWorkspace::new();
    workspace.try_reserve(2)?;
    let report = workspace.update(
        &mut model,
        &[1., 0.],
        &[-2., 0.],
        HessianForm::Hessian,
        HessianUpdate::Sr1,
    )?;
    println!("SR1: {report:?}; indefinite model: {:?}", model.as_slice());
    Ok(())
}
