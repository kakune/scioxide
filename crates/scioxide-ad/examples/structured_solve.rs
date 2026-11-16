//! Differentiate through a matrix product and a linear solve on a tape.
use scioxide_ad::{Real, Var, gemm_into, grad, solve_into};
use scioxide_core::SciOxideError;

fn main() -> Result<(), SciOxideError> {
    // f(B) = 1ᵀ·(A·B)⁻¹·c for A = [[2, 1], [1, 3]], c = [1, 2].
    let gradient = grad(
        |b| {
            let a = [2.0, 1.0, 1.0, 3.0].map(Var::constant);
            let mut m = [Var::constant(0.0); 4];
            gemm_into(&a, 2, 2, b, 2, &mut m)?;
            let mut x = [Var::constant(0.0); 2];
            solve_into(&m, 2, &[Var::constant(1.0), Var::constant(2.0)], 1, &mut x)?;
            Ok(x[0] + x[1])
        },
        &[1.0, 0.0, 0.0, 1.0],
    )?;
    // At B = I: ∂f/∂B = −Aᵀ·λ·xᵀ with x = A⁻¹c and λ = A⁻ᵀ·1.
    let x = [0.2, 0.6];
    let lambda = [0.4, 0.2];
    let at = [2.0, 1.0, 1.0, 3.0];
    for (i, row) in gradient.as_slice().chunks_exact(2).enumerate() {
        let mu = at[i * 2] * lambda[0] + at[i * 2 + 1] * lambda[1];
        for (g, xj) in row.iter().zip(x) {
            assert!((g + mu * xj).abs() < 1e-14);
        }
    }
    println!("∂f/∂B = {:?}", gradient.as_slice());
    Ok(())
}
