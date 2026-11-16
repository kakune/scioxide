//! Low-level kernels shared with other SciOxide crates of the same version.
//!
//! This module is public so that other SciOxide crates can call these kernels
//! directly. Its items follow different rules from the rest of the crate:
//!
//! - On error, the right-hand sides and scratch buffers you pass may be partly
//!   overwritten. The rule in [Output on error](scioxide_core#output-on-error)
//!   does not apply here.
//! - The kernels check storage sizes, non-finite values and exact zero pivots.
//!   They apply no [`RankPolicy`](crate::RankPolicy), compute no residual or
//!   condition diagnostics, and do no execution accounting.
//!
//! Applications should normally use the crate-root types instead, such as
//! [`TridiagonalLuDecomposition`](crate::TridiagonalLuDecomposition) and
//! [`CyclicTridiagonalPlan`](crate::CyclicTridiagonalPlan).

pub mod bordered_bidiagonal;
pub mod dense;
pub mod hessenberg;

/// Prepared dense partial-pivoted LU without solve accounting or diagnostics.
pub mod lu {
    pub use crate::decompose::lu::LuFactor;
}

/// Tridiagonal LU factors with adjacent row interchanges, and cyclic tridiagonal factors.
pub mod tridiagonal {
    pub use crate::structured::tridiagonal::{CyclicTridiagonalFactor, TridiagonalFactor};
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::tridiagonal::{CyclicTridiagonalFactor, TridiagonalFactor};
    #[test]
    fn compact_factors_pivot_and_reuse_rhs_storage() {
        let factor = TridiagonalFactor::factor(
            vec![2.0_f64, 1.0],
            vec![0.0, 3.0, 4.0],
            vec![1.0, 1.0],
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
        let mut rhs = [2.0, 11.0, 14.0];
        factor
            .solve_in_place(
                &mut rhs,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        for (value, expected) in rhs.into_iter().zip([1.0, 2.0, 3.0]) {
            assert!((value - expected).abs() < 1e-14);
        }
        let mut short = [17.0];
        assert!(
            factor
                .solve_in_place(
                    &mut short,
                    scioxide_core::ExecutionContext::sequential_native()
                )
                .is_err()
        );
        assert_eq!(short, [17.0]);
        assert!(
            TridiagonalFactor::factor(
                vec![1.0_f64],
                vec![1.0],
                vec![],
                scioxide_core::ExecutionContext::sequential_native()
            )
            .is_err()
        );
        assert!(
            TridiagonalFactor::factor(
                vec![],
                vec![0.0_f64],
                vec![],
                scioxide_core::ExecutionContext::sequential_native()
            )
            .is_err()
        );
        assert!(
            TridiagonalFactor::factor(
                vec![],
                vec![f64::NAN],
                vec![],
                scioxide_core::ExecutionContext::sequential_native()
            )
            .is_err()
        );
        let empty = TridiagonalFactor::<f64>::factor(
            vec![],
            vec![],
            vec![],
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
        empty
            .solve_in_place(
                &mut [],
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
    }
    #[test]
    fn dominant_real_factors_retain_small_and_large_rhs_behavior() {
        for scale in [1.0_f64, 1e-100, 1e-200] {
            let factor = TridiagonalFactor::factor(
                vec![scale; 2],
                vec![4.0 * scale; 3],
                vec![scale; 2],
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
            let mut rhs = [5.0 * scale, 6.0 * scale, 5.0 * scale];
            factor
                .solve_in_place(
                    &mut rhs,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert!(rhs.into_iter().all(|v| (v - 1.0).abs() < 2e-14));
        }
        let factor = TridiagonalFactor::factor(
            vec![1.0_f64; 2],
            vec![4.0; 3],
            vec![1.0; 2],
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
        let x = f64::MAX / 16.0;
        let mut rhs = [5.0 * x, 6.0 * x, 5.0 * x];
        factor
            .solve_in_place(
                &mut rhs,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(rhs.into_iter().all(|v| (v / x - 1.0).abs() < 2e-14));
        let mut rhs = [f64::from_bits(1), 0.0, -f64::from_bits(1)];
        factor
            .solve_in_place(
                &mut rhs,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(rhs.into_iter().all(f64::is_finite));
    }

    #[test]
    fn compact_cyclic_factor_preserves_the_full_system() {
        let factor = CyclicTridiagonalFactor::factor(
            vec![-1.0_f32; 2],
            vec![4.0; 3],
            vec![-1.0; 2],
            -1.0,
            -1.0,
        )
        .unwrap();
        let mut rhs = [2.0_f32; 3];
        factor.solve_in_place(&mut rhs).unwrap();
        assert!(rhs.into_iter().all(|v| (v - 1.0).abs() < 2e-6));
        assert!(
            CyclicTridiagonalFactor::factor(vec![0.0_f64; 2], vec![0.0; 3], vec![0.0; 2], 1.0, 1.0)
                .is_err()
        );
    }
}
