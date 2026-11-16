//! Deterministic dense fixtures for workspace allocation contracts.
#![allow(clippy::panic)]

use scioxide_linalg::Matrix;
pub const ORDER: usize = 64;
pub const MANY_RHS: usize = 16;
pub fn general_matrix() -> Matrix<f64> {
    Matrix::from_vec(
        ORDER,
        ORDER,
        (0..ORDER)
            .flat_map(|row| {
                (0..ORDER).map(move |col| {
                    if row == col {
                        2.0 * ORDER as f64
                    } else {
                        1.0 / (1.0 + row.abs_diff(col) as f64)
                    }
                })
            })
            .collect(),
    )
    .unwrap_or_else(|error| panic!("{error}"))
}
pub fn rhs_matrix(cols: usize) -> Matrix<f64> {
    Matrix::from_vec(
        ORDER,
        cols,
        (0..ORDER * cols)
            .map(|index| (index % 17) as f64 / 11.0 - 0.25)
            .collect(),
    )
    .unwrap_or_else(|error| panic!("{error}"))
}
