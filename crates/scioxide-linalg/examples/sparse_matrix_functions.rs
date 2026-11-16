//! Explicit fill and memory policy for sparse powers and matrix functions.
use scioxide_core::ExecutionContext;
use scioxide_linalg::{
    Estimation, MatrixFunction, SparseFunctionOptions, SparseFunctionPlan, SparsePowerOptions,
    SparsePowerPlan,
};
use scioxide_sparse::CsrMatrix;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix = CsrMatrix::from_raw(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 2.0, 4.0],
    )?;
    let powers = SparsePowerOptions::new()
        .with_max_entries(3)
        .with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_byte_limit(2_000_000),
        );

    let square = SparsePowerPlan::new(matrix.view(), 2, &powers)?;
    let squared = square.evaluate(matrix.view())?;
    println!("A² diagonal: {:?}", squared.value().values());

    let inverse = SparsePowerPlan::new(matrix.view(), -1, &powers)?;
    let mut workspace = inverse.workspace(())?;
    let mut output = inverse.create_output()?;
    inverse.evaluate_into(
        matrix.view(),
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!("A⁻¹ diagonal: {:?}", output.values());

    let functions = SparseFunctionPlan::new(
        matrix.view(),
        MatrixFunction::Exp,
        &SparseFunctionOptions::new()
            .with_max_entries(3)
            .with_error_estimate(Estimation::Estimate),
    )?;
    let exponential = functions.evaluate(matrix.view())?;
    println!(
        "exp(A): estimated fill {}, actual fill {}, error estimate {:?}",
        exponential.report().structural_fill(),
        exponential.report().nonzero_fill(),
        exponential.report().estimated_error()
    );
    Ok(())
}
