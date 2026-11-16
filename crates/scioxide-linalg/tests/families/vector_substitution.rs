//! Standard vector substitution retains finite results across binary frames.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, SciOxideErrorView};
use scioxide_core::{DiagonalKind, Triangle};
use scioxide_linalg::{
    BandedMatrix, LinalgExecutionContext, LinalgWorkspace, MatrixView, PackedKind, PackedMatrix,
    StructuredOperation, TriangularPlan, VectorView, VectorViewMut,
};

#[test]
fn storage_operations_and_ignored_unit_diagonals_agree() {
    let n = 17;
    for triangle in [Triangle::Lower, Triangle::Upper] {
        for diagonal in [DiagonalKind::Unit, DiagonalKind::Explicit] {
            let mut dense = vec![Complex::new(f64::NAN, f64::NAN); n * n];
            let mut packed = Vec::new();
            let mut band = vec![Complex::new(f64::NAN, f64::NAN); n * n];
            for j in 0..n {
                for i in 0..n {
                    if (triangle == Triangle::Lower && i >= j)
                        || (triangle == Triangle::Upper && i <= j)
                    {
                        let a = if i == j {
                            if diagonal == DiagonalKind::Unit {
                                Complex::new(f64::NAN, 0.0)
                            } else {
                                Complex::new(4.0, 0.5)
                            }
                        } else {
                            Complex::new(((i + 3 * j) % 7) as f64 / 128.0, 1.0 / 64.0)
                        };
                        dense[i * n + j] = a;
                        packed.push(a);
                        let upper = if triangle == Triangle::Upper {
                            n - 1
                        } else {
                            0
                        };
                        band[j * n + upper + i - j] = a;
                    }
                }
            }
            let matrix = MatrixView::from_slice(n, n, &dense).unwrap();
            let packed = PackedMatrix::new(n, triangle, PackedKind::Triangular, packed).unwrap();
            let (lower, upper) = if triangle == Triangle::Lower {
                (n - 1, 0)
            } else {
                (0, n - 1)
            };
            let band = BandedMatrix::new(n, n, lower, upper, n, band).unwrap();
            let plans = [
                TriangularPlan::from_dense(matrix, triangle, diagonal).unwrap(),
                TriangularPlan::from_packed(packed.view(), diagonal).unwrap(),
                TriangularPlan::from_band(band.view(), triangle, diagonal).unwrap(),
            ];
            let expected: Vec<_> = (0..n)
                .map(|i| Complex::new((i + 1) as f64 / 16.0, -0.25))
                .collect();
            for op in [
                StructuredOperation::Normal,
                StructuredOperation::Transpose,
                StructuredOperation::Adjoint,
            ] {
                let rhs: Vec<_> = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                let (r, c) = if op == StructuredOperation::Normal {
                                    (i, j)
                                } else {
                                    (j, i)
                                };
                                let a = if r == c && diagonal == DiagonalKind::Unit {
                                    Complex::new(1.0, 0.0)
                                } else if (triangle == Triangle::Lower && r < c)
                                    || (triangle == Triangle::Upper && r > c)
                                {
                                    Complex::new(0.0, 0.0)
                                } else {
                                    dense[r * n + c]
                                };
                                (if op == StructuredOperation::Adjoint {
                                    a.conj()
                                } else {
                                    a
                                }) * expected[j]
                            })
                            .fold(Complex::new(0.0, 0.0), |a, b| a + b)
                    })
                    .collect();
                for plan in &plans {
                    let actual = plan.solve(op, VectorView::from_slice(&rhs)).unwrap();
                    for (&a, &b) in actual.as_slice().iter().zip(&expected) {
                        assert!((a - b).norm() < 2e-14);
                    }
                }
            }
        }
    }
}

#[test]
fn singular_pivots_and_unrepresentable_results_are_transactional() {
    let singular = [1.0, 0.0, 2.0, 0.0];
    assert!(matches!(
        TriangularPlan::from_dense(
            MatrixView::from_slice(2, 2, &singular).unwrap(),
            Triangle::Lower,
            DiagonalKind::Explicit
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::Singular { .. }
    ));
    let plan = TriangularPlan::from_dense(
        MatrixView::from_slice(1, 1, &[f64::MIN_POSITIVE]).unwrap(),
        Triangle::Lower,
        DiagonalKind::Explicit,
    )
    .unwrap();
    let mut output = [19.0];
    assert!(
        plan.solve_into(
            StructuredOperation::Normal,
            VectorView::from_slice(&[f64::MAX]),
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            VectorViewMut::from_slice(&mut output),
            &mut LinalgWorkspace::from_scalar_slice(&mut [0.0]),
            LinalgExecutionContext::sequential()
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })
        .is_err()
    );
    assert_eq!(output, [19.0]);
}
