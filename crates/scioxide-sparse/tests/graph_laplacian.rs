#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::AlgebraicScalar;
use scioxide_core::{AdjointOperator, Complex, LinearOperator, NumericScalar, RealScalar};
use scioxide_sparse::GraphPlan;
use scioxide_sparse::*;
fn check<T: NumericScalar>() {
    type D = LaplacianDegree;
    type N = LaplacianNormalization;
    let s = |v| T::Real::try_from_f64_exact(v).unwrap();
    for direction in [GraphDirection::Directed, GraphDirection::Undirected] {
        let mut b = SparseBuilder::new(4, 4);
        for (u, v, w) in [
            (0, 0, 99.0),
            (0, 1, 4.0),
            (1, 0, 1.0),
            (1, 2, 2.0),
            (2, 0, 3.0),
        ] {
            b.push(u, v, s(w)).unwrap();
        }
        let matrix = b
            .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            .unwrap();
        let graph = GraphPlan::new(matrix.view(), direction).unwrap();
        let mut adjacency = [[0.0; 4]; 4];
        adjacency[0][1] = 4.0;
        adjacency[1][0] = 1.0;
        adjacency[1][2] = 2.0;
        adjacency[2][0] = 3.0;
        if direction == GraphDirection::Undirected {
            adjacency = [
                [0.0, 1.0, 3.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],
                [3.0, 2.0, 0.0, 0.0],
                [0.0; 4],
            ];
        }
        for degree in [D::Out, D::In] {
            for normalization in [N::Unnormalized, N::Symmetric] {
                let lap = graph.laplacian::<T>(degree, normalization).unwrap();
                let mut expected_degree = [0.0; 4];
                for (u, row) in adjacency.iter().enumerate() {
                    for (v, &w) in row.iter().enumerate() {
                        expected_degree[if degree == D::Out { u } else { v }] += w;
                    }
                }
                let tolerance = s(32.0) * T::Real::EPSILON;
                for (u, row) in adjacency.iter().enumerate() {
                    for (v, &weight) in row.iter().enumerate() {
                        let expected = if u == v {
                            if normalization == N::Symmetric && expected_degree[u] > 0.0 {
                                1.0
                            } else {
                                expected_degree[u]
                            }
                        } else if normalization == N::Unnormalized {
                            -weight
                        } else {
                            -weight
                                / (if expected_degree[u] > 0.0 {
                                    expected_degree[u].sqrt()
                                } else {
                                    1.0
                                })
                                / (if expected_degree[v] > 0.0 {
                                    expected_degree[v].sqrt()
                                } else {
                                    1.0
                                })
                        };
                        assert!(
                            (lap.matrix().get(u, v).unwrap()
                                - T::from_real(T::Real::try_from_f64_lossy(expected).unwrap()))
                            .abs()
                                <= tolerance
                        );
                    }
                }
                let null: Vec<_> = lap
                    .degrees()
                    .iter()
                    .map(|&d| {
                        T::from_real(if normalization == N::Symmetric && d > T::Real::ZERO {
                            d.sqrt()
                        } else {
                            T::Real::ONE
                        })
                    })
                    .collect();
                let mut output = vec![T::ZERO; 4];
                let mut scratch = output.clone();
                if degree == D::Out {
                    lap.apply_into(&null, &mut output, &mut scratch).unwrap();
                } else {
                    lap.apply_adjoint_into(&null, &mut output, &mut scratch)
                        .unwrap();
                }
                assert!(output.iter().all(|v| v.abs() <= tolerance * s(8.0)));
            }
        }
    }
}
#[test]
fn real_and_complex_nullspaces() {
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}
#[test]
fn empty_negative_zero_degree_and_extreme_scales() {
    for weights in [[f64::from_bits(1), f64::MAX], [1.0, 0.0]] {
        let mut b = SparseBuilder::new(3, 3);
        b.push(0, 1, weights[0]).unwrap();
        b.push(1, 2, weights[1]).unwrap();
        let m = b
            .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            .unwrap();
        let g = GraphPlan::new(m.view(), GraphDirection::Directed).unwrap();
        let lap = g
            .laplacian::<f64>(LaplacianDegree::Out, LaplacianNormalization::Symmetric)
            .unwrap();
        assert!(lap.matrix().get(0, 1).unwrap() < 0.0);
    }
    let m = SparseBuilder::<f64>::new(0, 0)
        .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let g = GraphPlan::new(m.view(), GraphDirection::Directed).unwrap();
    assert_eq!(
        g.laplacian::<f64>(LaplacianDegree::In, LaplacianNormalization::Symmetric)
            .unwrap()
            .rows(),
        0
    );
    let mut b = SparseBuilder::<f64>::new(1, 1);
    b.push(0, 0, -1.0).unwrap();
    let m = b
        .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    assert!(
        GraphPlan::new(m.view(), GraphDirection::Directed)
            .unwrap()
            .laplacian::<f64>(LaplacianDegree::Out, LaplacianNormalization::Unnormalized)
            .is_err()
    );
    // Two finite maximal weights into one degree overflow; In degrees do not.
    let mut b = SparseBuilder::<f64>::new(3, 3);
    b.push(0, 1, f64::MAX).unwrap();
    b.push(0, 2, f64::MAX).unwrap();
    let m = b
        .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let g = GraphPlan::new(m.view(), GraphDirection::Directed).unwrap();
    for normalization in [
        LaplacianNormalization::Unnormalized,
        LaplacianNormalization::Symmetric,
    ] {
        let error = g
            .laplacian::<f64>(LaplacianDegree::Out, normalization)
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
        let lap = g
            .laplacian::<f64>(LaplacianDegree::In, normalization)
            .unwrap();
        assert_eq!(lap.degrees(), &[0.0, f64::MAX, f64::MAX]);
    }
}
