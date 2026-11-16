//! Match two asymmetric weighted graphs with one fixed node pair.
use scioxide_linalg::MatrixView;
use scioxide_optimize::{QapDirection, QapMethod, QapOptions, QapProblemView};

fn main() -> Result<(), scioxide_core::SciOxideError> {
    let a = [0.0, 3.0, 1.0, 8.0, 0.0, 2.0, 4.0, 6.0, 0.0];
    let b = [0.0, 5.0, 7.0, 2.0, 0.0, 9.0, 3.0, 1.0, 0.0];
    let plan = QapProblemView::new(
        MatrixView::from_slice(3, 3, &a)?,
        MatrixView::from_slice(3, 3, &b)?,
        &[(1, 2)],
    )?
    .prepare_with_limit(1_000_000)?;
    let options = QapOptions::default()
        .with_method(QapMethod::FaqThenTwoOpt)
        .with_max_workspace_bytes(1_000_000)
        .with_seed(675)
        .with_restarts(4);
    let result = plan.solve_with_options(QapDirection::Minimize, None, &options)?;
    if let Some(permutation) = result.permutation() {
        println!(
            "mapping={:?}, objective={:?}, stop={:?}",
            permutation,
            result.objective(),
            result.termination()
        );
    }
    Ok(())
}
