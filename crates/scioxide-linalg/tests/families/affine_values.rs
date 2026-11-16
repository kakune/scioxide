//! Canonical owned and borrowed affine vector adapters.
#![allow(clippy::unwrap_used)]
use scioxide_core::{AffineValue, Complex, affine_combination_into, write_value};
use scioxide_linalg::{
    CyclicTridiagonalPlan, StructuredOperation, TridiagonalLuDecomposition, TridiagonalMatrixView,
    VectorView, VectorViewMut,
};

#[test]
fn conjugated_and_strided_views_preserve_logical_component_order() {
    let data = [
        Complex::new(1.0_f64, 2.0),
        Complex::new(99.0, 99.0),
        Complex::new(3.0, 4.0),
    ];
    let left = VectorView::from_slice_with_stride(&data, 2, 2).unwrap();
    let right = left.conjugate();
    let mut storage = [
        Complex::new(0.0, 0.0),
        Complex::new(99.0, 99.0),
        Complex::new(0.0, 0.0),
    ];
    let mut output = VectorViewMut::from_slice_with_stride(&mut storage, 2, 2).unwrap();
    affine_combination_into(&[(0.5, &left), (0.5, &right)], &mut output, &mut [0.0; 4]).unwrap();
    assert_eq!(
        storage,
        [
            Complex::new(1.0, 0.0),
            Complex::new(99.0, 99.0),
            Complex::new(3.0, 0.0)
        ]
    );
    let owned = right.try_to_owned().unwrap();
    assert_eq!(
        owned.as_slice(),
        &[Complex::new(1.0, -2.0), Complex::new(3.0, -4.0)]
    );
}

#[test]
fn broadcast_component_overflow_is_a_checked_shape_error() {
    let data = [Complex::new(1.0_f32, 2.0)];
    let view = VectorView::from_slice_with_stride(&data, usize::MAX, 0).unwrap();
    assert!(matches!(
        (view.dimension()).as_ref().err().unwrap().view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
    let mut output = Complex::new(7.0, 8.0);
    assert!(matches!(
        (affine_combination_into(&[(1.0, &view)], &mut output, &mut [0.0; 2]))
            .as_ref()
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
    assert_eq!(output, Complex::new(7.0, 8.0));

    assert!(write_value(&[1.0_f32], &mut output).is_err());
    assert_eq!(output, Complex::new(7.0, 8.0));
}

#[test]
fn canonical_cyclic_solver_preserves_real_spline_regressions() {
    for scale in [1e-200_f64, 1.0, 1e200] {
        let lower = [scale, scale];
        let main = [0.0, 2.0 * scale, 2.0 * scale];
        let upper = lower;
        let rhs = [3.5 * scale, 8.0 * scale, 8.5 * scale];
        let plan = CyclicTridiagonalPlan::new(
            TridiagonalMatrixView::new(&lower, &main, &upper).unwrap(),
            0.5 * scale,
            0.5 * scale,
            &scioxide_linalg::LinalgExecutionOptions::new(),
        )
        .unwrap();
        let output = plan.solve(StructuredOperation::Normal, &rhs[..]).unwrap();
        for (&actual, expected) in output.as_slice().iter().zip([1.0, 2.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-11);
        }
    }
}

#[test]
fn owned_tridiagonal_and_cyclic_solves_cover_both_complex_precisions() {
    macro_rules! check {
        ($s:ty, $tol:expr) => {{
            let c = |re, im| Complex::<$s>::new(re, im);
            let lower = [c(1.0, 0.5), c(0.5, -0.25)];
            let main = [c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)];
            let upper = [c(0.25, -0.5), c(0.75, 0.25)];
            let corners = [c(0.125, 0.125), c(-0.25, 0.125)];
            let expected = [c(1.0, 1.0), c(2.0, -1.0), c(-1.0, 0.5)];
            let ordinary = [
                main[0] * expected[0] + upper[0] * expected[1],
                lower[0] * expected[0] + main[1] * expected[1] + upper[1] * expected[2],
                lower[1] * expected[1] + main[2] * expected[2],
            ];
            let view = TridiagonalMatrixView::new(&lower, &main, &upper).unwrap();
            let factor = TridiagonalLuDecomposition::factor(view).unwrap();
            let actual = factor
                .solve(StructuredOperation::Normal, &ordinary[..])
                .unwrap();
            for (value, expected) in actual.as_slice().iter().zip(expected) {
                assert!(
                    (value.re - expected.re).abs() < $tol && (value.im - expected.im).abs() < $tol
                );
            }
            let mut cyclic = ordinary;
            cyclic[0] += corners[0] * expected[2];
            cyclic[2] += corners[1] * expected[0];
            let actual = CyclicTridiagonalPlan::new(
                view,
                corners[0],
                corners[1],
                &scioxide_linalg::LinalgExecutionOptions::new(),
            )
            .unwrap()
            .solve(StructuredOperation::Normal, &cyclic[..])
            .unwrap();
            for (value, expected) in actual.as_slice().iter().zip(expected) {
                assert!(
                    (value.re - expected.re).abs() < $tol && (value.im - expected.im).abs() < $tol
                );
            }
        }};
    }
    check!(f32, 2e-6);
    check!(f64, 2e-14);
}

#[test]
fn owned_solves_retain_empty_and_error_contracts() {
    let factor = TridiagonalLuDecomposition::<f64>::factor(
        TridiagonalMatrixView::new(&[], &[], &[]).unwrap(),
    )
    .unwrap();
    assert!(
        factor
            .solve(StructuredOperation::Normal, &[][..])
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        factor
            .solve(StructuredOperation::Normal, &[1.0][..])
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    let factor =
        TridiagonalLuDecomposition::factor(TridiagonalMatrixView::new(&[], &[1.0], &[]).unwrap())
            .unwrap();
    assert!(matches!(
        factor
            .solve(StructuredOperation::Normal, &[f64::NAN][..])
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
}
