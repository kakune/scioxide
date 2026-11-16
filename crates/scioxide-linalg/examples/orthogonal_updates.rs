//! RQ and reusable full QR edits in original coordinates.
use scioxide_core::SciOxideError;
use scioxide_linalg::{
    LinalgWorkspace, Matrix, QrModification, QrUpdateDecomposition, QrUpdateOptions, RankPolicy,
    RqDecomposition,
};
pub fn main() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(3, 2, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0])?;
    let rq = RqDecomposition::factor(a.view())?;
    println!(
        "RQ: R={}x{}, Q={}x{}",
        rq.r_matrix().rows(),
        rq.r_matrix().cols(),
        rq.q_matrix().rows(),
        rq.q_matrix().cols()
    );
    let options = QrUpdateOptions::new(RankPolicy::DimensionScaledEpsilon);
    let mut qr = QrUpdateDecomposition::factor_with_options(a.view(), &options)?;
    let mut workspace = LinalgWorkspace::new();
    workspace.try_reserve(&QrUpdateDecomposition::<f64>::workspace_requirements(
        &scioxide_linalg::QrUpdateWorkspaceRequest::new(4, 2, &options),
    )?)?;
    let u = Matrix::from_vec(3, 1, vec![1.0, 0.0, 0.0])?;
    let v = Matrix::from_vec(2, 1, vec![0.5, -0.5])?;
    let report = qr.modify_in_place(
        QrModification::RankUpdate {
            u: u.view(),
            v: v.view(),
        },
        &options,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    println!(
        "Rank: {} -> {}",
        report.before().report().rank(),
        report.after().report().rank()
    );
    let row = Matrix::from_vec(1, 2, vec![2.0, 3.0])?;
    qr.modify_in_place(
        QrModification::InsertRows {
            index: 1,
            values: row.view(),
        },
        &options,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    qr.modify_in_place(
        QrModification::DeleteRows { index: 1, count: 1 },
        &options,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    println!(
        "Restored shape: {}x{}",
        qr.r_matrix().rows(),
        qr.r_matrix().cols()
    );
    Ok(())
}
