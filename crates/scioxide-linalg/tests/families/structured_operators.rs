#![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_range_loop)]
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AdjointOperator, Complex, ExecutionContext, LinearOperator, NumericScalar, RealScalar,
    SciOxideErrorKind,
};
use scioxide_linalg::*;
use scioxide_sparse::CsrMatrix;
/// Execution options whose byte limit admits exactly `scalars` values of `T`.
fn cap<T>(scalars: usize) -> LinalgExecutionOptions {
    LinalgExecutionOptions::new().with_byte_limit(scalars * core::mem::size_of::<T>())
}
fn request(order: usize, right_hand_sides: usize) -> StructuredWorkspaceRequest {
    StructuredWorkspaceRequest::solve_matrix(
        order,
        right_hand_sides,
        &LinalgExecutionOptions::new(),
    )
}
fn solve_scratch<T: NumericScalar>(
    plan: &StructuredSolvePlan<'_, T>,
    right_hand_sides: usize,
) -> Vec<T> {
    let len = plan
        .workspace_requirements(&request(plan.order(), right_hand_sides))
        .unwrap()
        .scalar_len()
        .unwrap();
    vec![T::ZERO; len]
}
fn scalar_scratch<T: NumericScalar>(
    _op: &dyn LinearOperator<T>,
    request: OperatorWorkspaceRequest<T>,
) -> Vec<T> {
    let len = LinalgWorkspace::workspace_requirements(&request)
        .unwrap()
        .scalar_len()
        .unwrap();
    vec![T::ZERO; len]
}
fn scalar<T: NumericScalar>(x: f64, y: f64) -> T {
    T::from_components(
        T::Real::try_from_f64_exact(x).unwrap(),
        T::Real::try_from_f64_exact(y).unwrap(),
    )
}
fn close<T: NumericScalar>(x: T, y: T) {
    crate::support::comparisons::within(
        x,
        y,
        T::Real::try_from_f64_exact(0.0001220703125).unwrap() * (T::Real::ONE + y.abs()),
    );
}
fn compact<T: NumericScalar>() {
    let c = [scalar::<T>(2., 0.), scalar(3., 1.), scalar(4., -1.)];
    let r = [c[0], scalar(5., 2.)];
    let a = StructuredMatrixView::toeplitz(&c, &r).unwrap();
    let expected = [c[0], r[1], c[1], c[0], c[2], c[1]];
    let dense = a.materialize_with_options(&cap::<T>(6)).unwrap();
    assert_eq!(dense.as_slice(), expected);
    let owned = StructuredMatrix::from_view_with_options(a, &cap::<T>(5)).unwrap();
    assert_eq!(
        owned.view().materialize_with_options(&cap::<T>(6)).unwrap(),
        dense
    );
    assert!(a.materialize_with_options(&cap::<T>(5)).is_err());
    let x = [scalar::<T>(1., 1.), scalar(-2., 0.)];
    let y = [scalar::<T>(2., 0.), scalar(0., 1.), scalar(-1., 1.)];
    let mut ax = [T::ZERO; 3];
    let mut ahy = [T::ZERO; 2];
    a.apply_into(&x, &mut ax, &mut [T::ZERO; 3]).unwrap();
    a.apply_adjoint_into(&y, &mut ahy, &mut [T::ZERO; 2])
        .unwrap();
    for i in 0..3 {
        close(ax[i], expected[2 * i] * x[0] + expected[2 * i + 1] * x[1]);
    }
    close(
        ax.iter().zip(y).fold(T::ZERO, |s, (a, b)| s + a.conj() * b),
        x.iter()
            .zip(ahy)
            .fold(T::ZERO, |s, (a, b)| s + a.conj() * b),
    );
    let hr = [c[2], r[1]];
    let h = StructuredMatrixView::hankel(&c, &hr).unwrap();
    assert_eq!(
        h.materialize_with_options(&cap::<T>(6)).unwrap().as_slice(),
        [c[0], c[1], c[1], c[2], c[2], hr[1]]
    );
    let circ = StructuredMatrixView::circulant(&c)
        .unwrap()
        .materialize_with_options(&cap::<T>(9))
        .unwrap();
    assert_eq!(
        circ.as_slice(),
        [c[0], c[2], c[1], c[1], c[0], c[2], c[2], c[1], c[0]]
    );
}
#[test]
fn compact_domains() {
    compact::<f32>();
    compact::<f64>();
    compact::<Complex<f32>>();
    compact::<Complex<f64>>();
}
#[test]
fn canonical_identities() {
    for n in 0..7 {
        let h = StructuredMatrixView::<f64>::hilbert(n, scioxide_linalg::MatrixForm::Direct)
            .unwrap()
            .materialize_with_options(&cap::<f64>(n * n))
            .unwrap();
        let inv = StructuredMatrixView::<f64>::hilbert(n, scioxide_linalg::MatrixForm::Inverse)
            .unwrap()
            .materialize_with_options(&cap::<f64>(n * n))
            .unwrap();
        let product = h.view().mul_mat(&inv).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert!((product.view().get(i, j).unwrap() - f64::from(i == j)).abs() < 1e-7);
            }
        }
        for kind in [PascalKind::Lower, PascalKind::Upper, PascalKind::Symmetric] {
            let a =
                StructuredMatrixView::<f64>::pascal(n, kind, scioxide_linalg::MatrixForm::Direct)
                    .unwrap()
                    .materialize_with_options(&cap::<f64>(n * n))
                    .unwrap();
            let b =
                StructuredMatrixView::<f64>::pascal(n, kind, scioxide_linalg::MatrixForm::Inverse)
                    .unwrap()
                    .materialize_with_options(&cap::<f64>(n * n))
                    .unwrap();
            assert_eq!(a.view().mul_mat(&b).unwrap(), Matrix::identity(n).unwrap());
        }
        for full in [HelmertKind::Reduced, HelmertKind::Full] {
            let a = StructuredMatrixView::<f64>::helmert(n, full)
                .unwrap()
                .materialize_with_options(&cap::<f64>(n * n))
                .unwrap();
            let p = a
                .view()
                .mul_mat(&a.view().transpose().try_to_matrix().unwrap())
                .unwrap();
            for i in 0..p.rows() {
                for j in 0..p.cols() {
                    close(p.view().get(i, j).unwrap(), f64::from(i == j));
                }
            }
        }
        let a = StructuredMatrixView::<Complex<f64>>::dft(
            n,
            DftDirection::Forward,
            DftNormalization::Unitary,
        )
        .unwrap()
        .materialize_with_options(&cap::<Complex<f64>>(n * n))
        .unwrap();
        let p = a
            .view()
            .mul_mat(&a.view().adjoint().try_to_matrix().unwrap())
            .unwrap();
        for i in 0..n {
            for j in 0..n {
                close(
                    p.view().get(i, j).unwrap(),
                    Complex::new(f64::from(i == j), 0.),
                );
            }
        }
    }
    for n in [0, 1, 2, 4, 8] {
        let h = StructuredMatrixView::<f64>::hadamard(n)
            .unwrap()
            .materialize_with_options(&cap::<f64>(n * n))
            .unwrap();
        assert_eq!(
            h.view().mul_mat(&h).unwrap(),
            Matrix::identity(n)
                .unwrap()
                .view()
                .try_scale(n as f64)
                .unwrap()
        );
    }
    let coefficients = [105., -176., 86., -16., 1.];
    let f = StructuredMatrixView::companion(&coefficients, scioxide_linalg::CompanionKind::Fiedler)
        .unwrap()
        .materialize_with_options(&cap::<f64>(16))
        .unwrap();
    assert_eq!(
        f.as_slice(),
        [
            16., -86., 1., 0., 1., 0., 0., 0., 0., 176., 0., -105., 0., 1., 0., 0.
        ]
    );
}
#[test]
fn convolution_and_failures() {
    let kernel = [2., -1., 3.];
    let x = Vector::from_vec(vec![1., 2., -1., 3.]);
    for (mode, expected) in [
        (ConvolutionMode::Full, vec![2., 3., -1., 13., -6., 9.]),
        (ConvolutionMode::Same, vec![3., -1., 13., -6.]),
        (ConvolutionMode::Valid, vec![-1., 13.]),
    ] {
        let a = StructuredMatrixView::convolution(&kernel, 4, mode)
            .unwrap()
            .materialize_with_options(&cap::<f64>(24))
            .unwrap();
        assert_eq!(a.view().mul_vec(&x).unwrap().as_slice(), expected);
    }
    assert!(StructuredMatrixView::toeplitz(&[1.], &[2.]).is_err());
    assert!(StructuredMatrixView::circulant(&[f64::NAN]).is_err());
    assert!(StructuredMatrixView::<f64>::hadamard(3).is_err());
    assert!(
        StructuredMatrixView::<f64>::hilbert(usize::MAX, scioxide_linalg::MatrixForm::Direct)
            .is_err()
    );
    assert!(
        StructuredMatrixView::companion(&[1., 0.], scioxide_linalg::CompanionKind::Frobenius)
            .is_err()
    );
    let a = StructuredMatrixView::toeplitz(&[1., 2.], &[]).unwrap();
    assert_eq!((a.rows(), a.cols()), (2, 0));
    let mut out = [1., 2.];
    a.apply_into(&[], &mut out, &mut [0.; 2]).unwrap();
    assert_eq!(out, [0., 0.]);
    let a = StructuredMatrixView::<f32>::hilbert(40, scioxide_linalg::MatrixForm::Inverse).unwrap();
    assert!(a.materialize().is_err());
}
fn solves<T: NumericScalar>() {
    for n in [0, 1, 2, 3, 7, 16] {
        let mut c: Vec<T> = (0..n).map(|i| scalar((i % 3) as f64 - 1., 0.25)).collect();
        let mut r: Vec<T> = (0..n).map(|i| scalar((i % 2) as f64, -0.5)).collect();
        if n > 0 {
            c[0] = scalar(8., 0.);
            r[0] = c[0];
        }
        let last: Vec<T> = (0..n)
            .map(|i| if i == 0 { c[n - 1] } else { r[i] })
            .collect();
        for a in [
            StructuredMatrixView::toeplitz(&c, &r).unwrap(),
            StructuredMatrixView::hankel(&c, &last).unwrap(),
            StructuredMatrixView::circulant(&c).unwrap(),
        ] {
            for (op, execution) in [
                StructuredOperation::Normal,
                StructuredOperation::Transpose,
                StructuredOperation::Adjoint,
            ]
            .into_iter()
            .flat_map(|op| {
                [
                    (op, ExecutionContext::sequential()),
                    (op, ExecutionContext::sequential_native()),
                ]
            }) {
                let x: Vec<T> = (0..n).map(|i| scalar((i + 1) as f64, 0.5)).collect();
                let b: Vec<T> = (0..n)
                    .map(|i| {
                        (0..n).fold(T::ZERO, |sum, j| {
                            sum + match op {
                                StructuredOperation::Normal => a.get(i, j).unwrap(),
                                StructuredOperation::Transpose => a.get(j, i).unwrap(),
                                StructuredOperation::Adjoint => a.get(j, i).unwrap().conj(),
                            } * x[j]
                        })
                    })
                    .collect();
                let tolerance = scioxide_core::Tolerance::new(
                    T::Real::ZERO,
                    T::Real::try_from_usize_exact(256 * n.max(1)).unwrap() * T::Real::EPSILON,
                )
                .unwrap();
                let options = StructuredSolveOptions::new().with_tolerance(tolerance);
                let plan = StructuredSolvePlan::new(a, &options).unwrap();
                let mut out = vec![T::ZERO; n];
                let mut work = plan.workspace(&request(plan.order(), 1)).unwrap();
                let error = plan
                    .solve_into(
                        op,
                        &b[..],
                        VectorViewMut::from_slice(&mut out),
                        &mut work,
                        execution,
                    )
                    .unwrap();
                assert_eq!(error.dense_lu_solves(), 0);
                assert!(error.backward_error() <= tolerance.relative());
                for (a, b) in out.into_iter().zip(x) {
                    close(a, b);
                }
            }
        }
    }
    // Nonsingular indefinite system with zero first leading principal minor.
    let c = [T::ZERO, T::ONE];
    let a = StructuredMatrixView::toeplitz(&c, &c).unwrap();
    let plan = StructuredSolvePlan::new(a, &StructuredSolveOptions::new()).unwrap();
    let mut out = [T::ZERO; 2];
    plan.solve_into(
        StructuredOperation::Normal,
        &[scalar(2., 0.), scalar(1., 0.)][..],
        VectorViewMut::from_slice(&mut out),
        &mut plan.workspace(&request(plan.order(), 1)).unwrap(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    close(out[0], T::ONE);
    close(out[1], scalar(2., 0.));
}
#[test]
fn compact_solve_domains() {
    solves::<f32>();
    solves::<f64>();
    solves::<Complex<f32>>();
    solves::<Complex<f64>>();
}
fn products<T: NumericScalar>() {
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            scalar::<T>(2., 0.),
            scalar(1., 1.),
            scalar(-1., 0.),
            scalar(3., 0.),
        ],
    )
    .unwrap();
    let b = Matrix::from_vec(
        2,
        2,
        vec![
            scalar::<T>(1., 0.),
            scalar(0., 1.),
            scalar(2., 0.),
            scalar(4., 0.),
        ],
    )
    .unwrap();
    let a_finite = a.view().try_finite().unwrap();
    let b_finite = b.view().try_finite().unwrap();
    let k = KroneckerOperator::new(&a_finite, &b_finite).unwrap();
    let mut dense = Matrix::zeros(4, 4).unwrap();
    let mut work = scalar_scratch(
        &k,
        OperatorWorkspaceRequest::materialize(&k, &LinalgExecutionOptions::new()).unwrap(),
    );
    materialize_operator_into(
        &k,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        dense.view_mut(),
        &mut LinalgWorkspace::from_scalar_slice(&mut work),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    for i in 0..4 {
        for j in 0..4 {
            close(
                dense.view().get(i, j).unwrap(),
                a.view().get(i / 2, j / 2).unwrap() * b.view().get(i % 2, j % 2).unwrap(),
            );
        }
    }
    let x = vec![
        scalar::<T>(1., 0.),
        scalar(2., 1.),
        scalar(3., 0.),
        scalar(4., 0.),
    ];
    let rhs = dense.view().mul_vec(&Vector::from_vec(x.clone())).unwrap();
    let lu_a = LuDecomposition::factor(&a).unwrap();
    let lu_b = LuDecomposition::factor(&b).unwrap();
    let ai = lu_a.inverse_operator();
    let bi = lu_b.inverse_operator();
    let mut work = vec![
        T::ZERO;
        LinalgWorkspace::<T>::workspace_requirements(
            &OperatorWorkspaceRequest::kronecker_solve(
                &k,
                [&ai, &bi],
                &LinalgExecutionOptions::new()
            )
            .unwrap()
        )
        .unwrap()
        .scalar_len()
        .unwrap()
    ];
    let mut out = vec![T::ZERO; 4];
    let report = k
        .solve_into(
            [&ai, &bi],
            scioxide_linalg::VectorView::from_slice(rhs.as_slice()),
            &scioxide_linalg::CompositeSolveOptions::new()
                .with_tolerance(scioxide_core::Tolerance::from_sqrt_epsilon()),
            scioxide_linalg::VectorViewMut::from_slice(&mut out),
            &mut LinalgWorkspace::from_scalar_slice(&mut work),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.termination().is_success());
    for (&a, &b) in out.iter().zip(&x) {
        close(a, b);
    }
    let blocks: [&(dyn AdjointOperator<T> + Sync); 2] = [
        &a.view().try_finite().unwrap(),
        &b.view().try_finite().unwrap(),
    ];
    let block = BlockDiagonalOperator::new(&blocks).unwrap();
    let mut rhs = vec![T::ZERO; 4];
    let mut work = vec![T::ZERO; block.apply_workspace_len().unwrap()];
    block.apply_into(&x, &mut rhs, &mut work).unwrap();
    let invs: [&dyn LinearOperator<T>; 2] = [&ai, &bi];
    let mut workspace = LinalgWorkspace::<T>::workspace(
        &OperatorWorkspaceRequest::block_diagonal_solve(
            &block,
            &invs,
            &LinalgExecutionOptions::new(),
        )
        .unwrap(),
    )
    .unwrap();
    let report = block
        .solve_into(
            &invs,
            scioxide_linalg::VectorView::from_slice(&rhs),
            &scioxide_linalg::CompositeSolveOptions::new()
                .with_tolerance(scioxide_core::Tolerance::from_sqrt_epsilon()),
            scioxide_linalg::VectorViewMut::from_slice(&mut out),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.termination().is_success());
    for (&a, &b) in out.iter().zip(&x) {
        close(a, b);
    }
    let kr = khatri_rao_with_options(a.view(), b.view(), &cap::<T>(8)).unwrap();
    for i in 0..4 {
        for j in 0..2 {
            close(
                kr.view().get(i, j).unwrap(),
                a.view().get(i / 2, j).unwrap() * b.view().get(i % 2, j).unwrap(),
            );
        }
    }
    for op in [
        &k as &dyn AdjointOperator<T>,
        &block as &dyn AdjointOperator<T>,
    ] {
        let mut work = vec![T::ZERO; op.apply_adjoint_workspace_len().unwrap()];
        let mut adj = vec![T::ZERO; 4];
        op.apply_adjoint_into(&x, &mut adj, &mut work).unwrap();
        let mut normwork = scalar_scratch(
            op,
            OperatorWorkspaceRequest::norm(op, &LinalgExecutionOptions::new()).unwrap(),
        );
        let one = operator_norm_scoped(
            op,
            MatrixNorm::One,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut LinalgWorkspace::from_scalar_slice(&mut normwork),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert!(one > T::Real::ZERO);
        let mut d = Matrix::zeros(4, 4).unwrap();
        materialize_operator_into(
            op,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            d.view_mut(),
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        let expected = d
            .view()
            .adjoint()
            .try_to_matrix()
            .unwrap()
            .view()
            .mul_vec(&Vector::from_vec(x.clone()))
            .unwrap();
        for (&a, &b) in adj.iter().zip(expected.as_slice()) {
            close(a, b);
        }
        let expected = (0..4)
            .map(|j| {
                (0..4).fold(T::Real::ZERO, |sum, i| {
                    sum + d.view().get(i, j).unwrap().abs()
                })
            })
            .fold(T::Real::ZERO, |a, b| a.max(b));
        close(T::from_real(one), T::from_real(expected));
    }
}
#[test]
fn composite_domains() {
    products::<f32>();
    products::<f64>();
    products::<Complex<f32>>();
    products::<Complex<f64>>();
}
#[test]
fn conditions_and_solve_failures() {
    let c = [2., 0., 0.];
    let a = StructuredMatrixView::circulant(&c).unwrap();
    let tol = scioxide_core::Tolerance::new(0., 1e-12).unwrap();
    let options = StructuredSolveOptions::new().with_tolerance(tol);
    let p = StructuredSolvePlan::new(a, &options).unwrap();
    let mut w = solve_scratch(&p, 1);
    let sequential = ExecutionContext::sequential;
    close(
        p.spectral_condition_estimate()
            .unwrap()
            .reciprocal()
            .value(),
        1.,
    );
    close(
        p.condition_estimate_scoped(
            &mut LinalgWorkspace::from_scalar_slice(&mut w),
            sequential(),
        )
        .unwrap()
        .reciprocal()
        .value(),
        1.,
    );
    let mut out = [9.; 3];
    let len = w.len();
    assert_eq!(
        p.solve_into(
            StructuredOperation::Normal,
            &[1.; 3][..],
            VectorViewMut::from_slice(&mut out),
            &mut LinalgWorkspace::from_scalar_slice(&mut w[..len - 1]),
            sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(out, [9.; 3]);
    assert!(
        p.solve_into(
            StructuredOperation::Normal,
            &[f64::NAN; 3][..],
            VectorViewMut::from_slice(&mut out),
            &mut LinalgWorkspace::from_scalar_slice(&mut w),
            sequential()
        )
        .is_err()
    );
    assert_eq!(out, [9.; 3]);
    let z = [0.; 3];
    let p = StructuredSolvePlan::new(StructuredMatrixView::toeplitz(&z, &z).unwrap(), &options)
        .unwrap();
    let mut workspace = p
        .workspace(&request(p.order(), 1).with_dense_fallback())
        .unwrap();
    assert!(matches!(
        p.solve_into(
            StructuredOperation::Normal,
            &[1.; 3][..],
            VectorViewMut::from_slice(&mut out),
            &mut workspace,
            sequential()
        )
        .unwrap_err()
        .view(),
        scioxide_core::SciOxideErrorView::Singular { .. }
    ));
    assert_eq!(out, [9.; 3]);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::Failed
    );
    let c = [1.0_f64, 1. - 1e-8];
    let a = StructuredMatrixView::toeplitz(&c, &c).unwrap();
    let p = StructuredSolvePlan::new(a, &options).unwrap();
    let r = p
        .condition_estimate_scoped(
            &mut p
                .workspace(&request(p.order(), 1).with_dense_fallback())
                .unwrap(),
            sequential(),
        )
        .unwrap()
        .reciprocal()
        .value();
    assert!((r - 5e-9).abs() < 1e-15);
    let norm =
        StructuredMatrixView::<f64>::hilbert(3, scioxide_linalg::MatrixForm::Direct).unwrap();
    let mut w = LinalgWorkspace::new();
    close(
        operator_norm_scoped(
            &norm,
            MatrixNorm::Infinity,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut w,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap(),
        11. / 6.,
    );
    close(
        operator_norm_scoped(
            &norm,
            MatrixNorm::Frobenius,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut w,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap(),
        (1. + 2. / 4. + 3. / 9. + 2. / 16. + 1. / 25.).sqrt(),
    );
}

#[cfg(feature = "std")]
#[test]
fn executor_and_strided_views() {
    use scioxide_core::ThreadExecutor;
    let c = [4., 1., -1.];
    let a = StructuredMatrixView::toeplitz(&c, &c).unwrap();
    let options = StructuredSolveOptions::new()
        .with_tolerance(scioxide_core::Tolerance::new(0., 1e-12).unwrap());
    let p = StructuredSolvePlan::new(a, &options).unwrap();
    let rhs = Matrix::from_vec(3, 12, (0..36).map(|i| i as f64).collect()).unwrap();
    let mut seq = Matrix::zeros(3, 12).unwrap();
    let mut par = Matrix::zeros(3, 12).unwrap();
    let mut work = p.workspace(&request(p.order(), 12)).unwrap();
    let pool = ThreadExecutor::new(core::num::NonZeroUsize::new(3).unwrap());
    p.solve_matrix_into(
        StructuredOperation::Normal,
        rhs.view(),
        seq.view_mut(),
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    p.solve_matrix_into(
        StructuredOperation::Normal,
        rhs.view(),
        par.view_mut(),
        &mut work,
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(seq, par);
    let mut work = LinalgWorkspace::new();
    apply_operator_matrix_into(
        &a,
        rhs.view(),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        seq.view_mut(),
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    apply_operator_matrix_into(
        &a,
        rhs.view(),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        par.view_mut(),
        &mut work,
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(seq, par);
    let transposed = rhs.view().transpose().try_to_matrix().unwrap();
    apply_operator_matrix_into(
        &a,
        transposed.view().transpose(),
        &scioxide_linalg::LinalgExecutionOptions::new(),
        par.view_mut(),
        &mut work,
        ExecutionContext::new(&pool),
    )
    .unwrap();
    assert_eq!(seq, par);
}

#[test]
fn budgets_cancellation_sparse_and_transactional_products() {
    use scioxide_core::{Budget, CancellationToken};
    let c = [4., 1., -1.];
    let a = StructuredMatrixView::toeplitz(&c, &c).unwrap();
    // One iteration per RHS entry is charged against the plan's own allowance.
    let options = StructuredSolveOptions::new().with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new()
            .with_iteration_budget(Budget::try_from(4).unwrap()),
    );
    let plan = StructuredSolvePlan::new(a, &options).unwrap();
    let mut out = [99.; 3];
    let mut work = plan.workspace(&request(plan.order(), 1)).unwrap();
    plan.solve_into(
        StructuredOperation::Normal,
        &[4.; 3][..],
        VectorViewMut::from_slice(&mut out),
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let mut matrix_out = Matrix::from_vec(3, 2, vec![99.; 6]).unwrap();
    assert_eq!(
        plan.solve_matrix_into(
            StructuredOperation::Normal,
            &Matrix::from_vec(3, 2, vec![1.; 6]).unwrap(),
            matrix_out.view_mut(),
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(matrix_out.as_slice(), &[99.; 6]);
    assert_eq!(work.last_attempt().unwrap().counts().own().iterations(), 4);
    let before = out;
    let token = CancellationToken::new();
    token.cancel();
    let ex = ExecutionContext::sequential().with_cancellation(&token);
    assert!(
        plan.solve_into(
            StructuredOperation::Normal,
            &[1.; 3][..],
            VectorViewMut::from_slice(&mut out),
            &mut work,
            ex
        )
        .is_err()
    );
    assert_eq!(out, before);
    let sparse = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2., 3.]).unwrap();
    let k = KroneckerOperator::new(&sparse, &a).unwrap();
    let mut dense = Matrix::zeros(6, 6).unwrap();
    materialize_operator_into(
        &k,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        dense.view_mut(),
        &mut LinalgWorkspace::new(),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    for i in 0..6 {
        for j in 0..6 {
            close(
                dense.view().get(i, j).unwrap(),
                if i / 3 == j / 3 {
                    (if i < 3 { 2. } else { 3. }) * a.get(i % 3, j % 3).unwrap()
                } else {
                    0.
                },
            );
        }
    }
    let huge = Matrix::from_vec(1, 1, vec![f64::MAX]).unwrap();
    let two = Matrix::from_vec(1, 1, vec![2.]).unwrap();
    let huge_finite = huge.view().try_finite().unwrap();
    let two_finite = two.view().try_finite().unwrap();
    let k = KroneckerOperator::new(&huge_finite, &two_finite).unwrap();
    let mut out = [99.];
    let mut w = vec![0.; k.apply_workspace_len().unwrap()];
    assert!(k.apply_into(&[1.], &mut out, &mut w).is_err());
    assert_eq!(out, [99.]);
    let empty = Matrix::<f64>::zeros(0, 2).unwrap();
    let empty_finite = empty.view().try_finite().unwrap();
    let b = KroneckerOperator::new(&empty_finite, &a).unwrap();
    let mut w = vec![0.; b.apply_workspace_len().unwrap()];
    b.apply_into(&[1.; 6], &mut [], &mut w).unwrap();
}
fn constructor_scalar_coverage<T: NumericScalar>() {
    let h = StructuredMatrixView::<T>::hilbert(2, scioxide_linalg::MatrixForm::Inverse)
        .unwrap()
        .materialize_with_options(&cap::<T>(4))
        .unwrap();
    assert_eq!(
        h.as_slice(),
        [
            scalar::<T>(4., 0.),
            scalar(-6., 0.),
            scalar(-6., 0.),
            scalar(12., 0.)
        ]
    );
    for kind in [PascalKind::Lower, PascalKind::Upper, PascalKind::Symmetric] {
        let p = StructuredMatrixView::<T>::pascal(3, kind, scioxide_linalg::MatrixForm::Direct)
            .unwrap()
            .materialize_with_options(&cap::<T>(9))
            .unwrap();
        let inv = StructuredMatrixView::<T>::pascal(3, kind, scioxide_linalg::MatrixForm::Inverse)
            .unwrap()
            .materialize_with_options(&cap::<T>(9))
            .unwrap();
        assert_eq!(
            p.view().mul_mat(&inv).unwrap(),
            Matrix::identity(3).unwrap()
        );
    }
    let f = [scalar::<T>(1., 0.), scalar(2., 0.), scalar(4., 0.)];
    let s = [scalar::<T>(0.5, 0.), scalar(0.25, 0.)];
    let l = StructuredMatrixView::leslie(&f, &s)
        .unwrap()
        .materialize_with_options(&cap::<T>(9))
        .unwrap();
    assert_eq!(
        l.as_slice(),
        [
            f[0],
            f[1],
            f[2],
            s[0],
            T::ZERO,
            T::ZERO,
            T::ZERO,
            s[1],
            T::ZERO
        ]
    );
    let d = StructuredMatrixView::fiedler(&f)
        .unwrap()
        .materialize_with_options(&cap::<T>(9))
        .unwrap();
    assert_eq!(
        d.as_slice(),
        [
            T::ZERO,
            T::ONE,
            scalar(3., 0.),
            T::ONE,
            T::ZERO,
            scalar(2., 0.),
            scalar(3., 0.),
            scalar(2., 0.),
            T::ZERO
        ]
    );
    let coefficients = [scalar::<T>(2., 0.), scalar(-3., 0.), T::ONE];
    for fiedler in [CompanionKind::Frobenius, CompanionKind::Fiedler] {
        let a = StructuredMatrixView::companion(&coefficients, fiedler)
            .unwrap()
            .materialize_with_options(&cap::<T>(4))
            .unwrap();
        let polynomial = a
            .view()
            .mul_mat(&a)
            .unwrap()
            .view()
            .try_sub(&a.view().try_scale(scalar(3., 0.)).unwrap())
            .unwrap()
            .view()
            .try_add(
                &Matrix::identity(2)
                    .unwrap()
                    .view()
                    .try_scale(scalar(2., 0.))
                    .unwrap(),
            )
            .unwrap();
        assert!(polynomial.as_slice().iter().all(|&v| v == T::ZERO));
    }
    let h = StructuredMatrixView::<T>::helmert(3, scioxide_linalg::HelmertKind::Full)
        .unwrap()
        .materialize_with_options(&cap::<T>(9))
        .unwrap();
    let p = h
        .view()
        .mul_mat(&h.view().adjoint().try_to_matrix().unwrap())
        .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            close(
                p.view().get(i, j).unwrap(),
                if i == j { T::ONE } else { T::ZERO },
            );
        }
    }
}
#[test]
fn canonical_scalar_domains() {
    constructor_scalar_coverage::<f32>();
    constructor_scalar_coverage::<f64>();
    constructor_scalar_coverage::<Complex<f32>>();
    constructor_scalar_coverage::<Complex<f64>>();
    let f = StructuredMatrixView::<Complex<f32>>::dft(
        7,
        DftDirection::Forward,
        DftNormalization::Unitary,
    )
    .unwrap()
    .materialize_with_options(&cap::<Complex<f32>>(49))
    .unwrap();
    let p = f
        .view()
        .mul_mat(&f.view().adjoint().try_to_matrix().unwrap())
        .unwrap();
    for i in 0..7 {
        for j in 0..7 {
            close(
                p.view().get(i, j).unwrap(),
                Complex::new(f32::from(i == j), 0.),
            );
        }
    }
}

use crate::support::scoped_adjoint;
#[test]
fn operator_compositions_forward_both_scoped_orientations() {
    use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let a = scoped_adjoint::ScopedDiagonal::new();
    let b = scoped_adjoint::ScopedDiagonal::new();
    let k = KroneckerOperator::new(&a, &b).unwrap();
    let blocks: [&(dyn AdjointOperator<f64> + Sync); 2] = [&a, &b];
    let d = BlockDiagonalOperator::new(&blocks).unwrap();
    for op in [&k as &dyn AdjointOperator<f64>, &d] {
        for adjoint in [false, true] {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let id = SolveIdentity::new(
                ProblemToken::new(1),
                scioxide_core::algorithm_id!("test", "composition", 1),
                0,
                0,
            );
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
                .unwrap();
            let n = op.rows();
            let mut scratch = vec![
                0.0;
                if adjoint {
                    op.apply_adjoint_workspace_len()
                } else {
                    op.apply_workspace_len()
                }
                .unwrap()
            ];
            let mut output = vec![99.0; n];
            if adjoint {
                op.apply_adjoint_into_scoped(
                    &vec![1.0; n],
                    &mut output,
                    &mut scratch,
                    &mut scioxide_core::OperatorContext::new(&mut context),
                )
                .and_then(scioxide_core::Invocation::require)
                .unwrap();
            } else {
                op.apply_into_scoped(
                    &vec![1.0; n],
                    &mut output,
                    &mut scratch,
                    &mut scioxide_core::OperatorContext::new(&mut context),
                )
                .and_then(scioxide_core::Invocation::require)
                .unwrap();
            }
            let expected = if n == 9 {
                vec![4.0, 6.0, 8.0, 6.0, 9.0, 12.0, 8.0, 12.0, 16.0]
            } else {
                vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]
            };
            assert!(
                output
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (*a - b).abs() < 1e-12)
            );
            assert_eq!(
                context
                    .counts()
                    .nested()
                    .provider(scioxide_core::EvaluationRole::Function),
                if n == 9 { 6 } else { 2 }
            );
        }
    }
}

#[test]
fn gko_rejection_requires_declared_dense_fallback() {
    let column = [2.0_f64, 0.1875, -0.25];
    let matrix = StructuredMatrixView::toeplitz(&column, &column).unwrap();
    let rhs = [4.0, -6.0 / 11.0, 14.0 / 13.0];
    let options = StructuredSolveOptions::new()
        .with_tolerance(scioxide_core::Tolerance::new(0., 3e-17).unwrap());
    let plan = StructuredSolvePlan::new(matrix, &options).unwrap();
    let basic = request(3, 1);
    let full = basic.with_dense_fallback();
    let requirements = plan.workspace_requirements(&full).unwrap();
    // The byte limit covers five plan vectors plus the reserved workspace.
    let bytes = 5 * column.len() * core::mem::size_of::<f64>()
        + requirements.required_bytes::<f64>().unwrap();
    {
        let bounded =
            options.with_execution_options(options.execution_options().with_byte_limit(bytes));
        let plan = StructuredSolvePlan::new(matrix, &bounded).unwrap();
        let mut workspace = plan.workspace(&full).unwrap();
        let mut output = [9.; 3];
        let report = plan
            .solve_into(
                StructuredOperation::Normal,
                &rhs[..],
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.gko_solves(), 0);
        assert_eq!(report.dense_lu_solves(), 1);
        assert!(report.backward_error() <= 3e-17);
    }
    {
        let bounded =
            options.with_execution_options(options.execution_options().with_byte_limit(bytes - 1));
        let plan = StructuredSolvePlan::new(matrix, &bounded).unwrap();
        assert_eq!(
            plan.workspace_requirements(&full).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        let mut workspace = plan.workspace(&basic).unwrap();
        let mut output = [9.; 3];
        let report = plan
            .solve_into(
                StructuredOperation::Normal,
                &rhs[..],
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            DirectSolveTermination::ResidualRejected
        );
        assert!(report.backward_error() > 3e-17);
        assert_eq!(output, [9.; 3]);
    }
    // Extra scalar storage alone does not declare the LU's other arenas.
    let mut storage = vec![0.; requirements.scalar_len().unwrap()];
    let mut output = [9.; 3];
    assert_eq!(
        plan.solve_into(
            StructuredOperation::Normal,
            &rhs[..],
            VectorViewMut::from_slice(&mut output),
            &mut LinalgWorkspace::from_scalar_slice(&mut storage),
            ExecutionContext::sequential()
        )
        .unwrap()
        .termination(),
        DirectSolveTermination::ResidualRejected
    );
    assert_eq!(output, [9.; 3]);

    // Capacity retained from unrelated LU work must not enable a fallback.
    let execution = LinalgExecutionOptions::new();
    let unrelated = scioxide_linalg::LuDecomposition::<f64>::workspace_requirements(
        &scioxide_linalg::LuWorkspaceRequest::factor(16, &execution),
    )
    .unwrap()
    .max(
        &scioxide_linalg::LuDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::LuWorkspaceRequest::solve_matrix(16, 1, &execution),
        )
        .unwrap(),
    )
    .max(&plan.workspace_requirements(&basic).unwrap());
    let mut workspace = LinalgWorkspace::new();
    workspace.try_reserve(&unrelated).unwrap();
    assert_eq!(
        plan.solve_into(
            StructuredOperation::Normal,
            &rhs[..],
            VectorViewMut::from_slice(&mut output),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap()
        .termination(),
        DirectSolveTermination::ResidualRejected
    );
    assert_eq!(output, [9.; 3]);
    {
        workspace.try_reserve(&requirements).unwrap();
        let report = plan
            .solve_into(
                StructuredOperation::Normal,
                &rhs[..],
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.dense_lu_solves(), 1);
    };
}

#[test]
fn gko_condition_estimation_is_allocation_free() {
    let column = [0.0_f64, 1.0, 0.0, 0.0];
    let plan = StructuredSolvePlan::new(
        StructuredMatrixView::toeplitz(&column, &column).unwrap(),
        &StructuredSolveOptions::new(),
    )
    .unwrap();
    let mut workspace = plan.workspace(&request(4, 1)).unwrap();
    let estimate = plan
        .condition_estimate_scoped(&mut workspace, ExecutionContext::sequential())
        .unwrap();
    close(estimate.reciprocal().value(), 0.25);
}

#[test]
fn gko_default_limits_and_borrowed_capacity_are_exact() {
    let column = [3.0_f64, 0.5, -0.25];
    let row = [3.0, -0.75, 0.125];
    let matrix = StructuredMatrixView::toeplitz(&column, &row).unwrap();
    let options = StructuredSolveOptions::new();
    let plan = StructuredSolvePlan::new(matrix, &options).unwrap();
    let request = request(3, 1);
    let requirements = plan.workspace_requirements(&request).unwrap();
    let count = requirements.scalar_len().unwrap();
    // Two coefficient and four phase vectors plus the borrowed workspace.
    let bytes = 6 * column.len() * core::mem::size_of::<f64>()
        + requirements.required_bytes::<f64>().unwrap();
    {
        let bounded =
            options.with_execution_options(options.execution_options().with_byte_limit(bytes));
        let plan = StructuredSolvePlan::new(matrix, &bounded).unwrap();
        let mut storage = vec![0.; count];
        let mut workspace = LinalgWorkspace::from_scalar_slice(&mut storage);
        let mut output = [91.; 3];
        let report = plan
            .solve_into(
                StructuredOperation::Normal,
                &column[..],
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.gko_solves(), 1);
        for (actual, expected) in output.into_iter().zip([1., 0., 0.]) {
            close(actual, expected);
        }
    }
    {
        let bounded =
            options.with_execution_options(options.execution_options().with_byte_limit(bytes - 1));
        let plan = StructuredSolvePlan::new(matrix, &bounded).unwrap();
        assert_eq!(
            plan.workspace_requirements(&request).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        let mut output = [91.; 3];
        assert_eq!(
            plan.solve_into(
                StructuredOperation::Normal,
                &column[..],
                VectorViewMut::from_slice(&mut output),
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(output, [91.; 3]);
    }
    let mut short = vec![0.; count - 1];
    let mut output = [91.; 3];
    assert_eq!(
        plan.solve_into(
            StructuredOperation::Normal,
            &column[..],
            VectorViewMut::from_slice(&mut output),
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [91.; 3]);
}

fn kms<T: NumericScalar>(powers: [i32; 2]) {
    for (case, power) in powers.into_iter().enumerate() {
        let rho = T::Real::ONE - T::Real::ONE.scalbn(-power);
        for n in [8, 32] {
            let column: Vec<T> = (0..n).map(|i| T::from_real(rho.powi(i as i32))).collect();
            let options = StructuredSolveOptions::new();
            let plan = StructuredSolvePlan::new(
                StructuredMatrixView::toeplitz(&column, &column).unwrap(),
                &options,
            )
            .unwrap();
            let mut rhs = vec![T::ZERO; n];
            rhs[n / 2] = T::ONE;
            let mut output = vec![T::ZERO; n];
            let result = plan.solve_into(
                StructuredOperation::Normal,
                &rhs[..],
                VectorViewMut::from_slice(&mut output),
                &mut plan
                    .workspace(&request(plan.order(), 1).with_dense_fallback())
                    .unwrap(),
                ExecutionContext::sequential(),
            );
            assert!(result.is_ok(), "KMS n={n}, power={power}: {result:?}");
            let report = result.unwrap();
            assert_eq!(report.levinson_solves(), usize::from(case == 0));
            assert_eq!(report.gko_solves(), usize::from(case == 1));
            assert_eq!(report.dense_lu_solves(), 0);
            assert!(
                options
                    .tolerance()
                    .is_residual_ok(report.backward_error(), T::Real::ONE)
                    .unwrap()
            );
            // KMS has a known tridiagonal inverse. Forward sensitivity grows as
            // 1/(1-rho); backward acceptance above retains its tighter bound.
            let denominator = (T::Real::ONE - rho) * (T::Real::ONE + rho);
            let max_inverse = (T::Real::ONE + rho * rho) / denominator;
            let forward_bound = T::Real::try_from_usize_exact(64 * n).unwrap() * T::Real::EPSILON
                / (T::Real::ONE - rho);
            for (i, &x) in output.iter().enumerate() {
                let exact = if i == n / 2 {
                    max_inverse
                } else if i.abs_diff(n / 2) == 1 {
                    -rho / denominator
                } else {
                    T::Real::ZERO
                };
                assert!((x.real_component() - exact).abs() / max_inverse <= forward_bound);
            }
        }
    }
}

#[test]
fn ill_conditioned_kac_murdock_szego() {
    kms::<f32>([10, 20]);
    kms::<f64>([20, 40]);
}

#[test]
fn structured_reports_cover_empty_blocks_and_complex_hankel_circulant() {
    let empty = StructuredSolvePlan::new(
        StructuredMatrixView::<f64>::toeplitz(&[], &[]).unwrap(),
        &StructuredSolveOptions::new(),
    )
    .unwrap();
    let report = empty
        .solve_into(
            StructuredOperation::Normal,
            &[][..],
            VectorViewMut::from_slice(&mut []),
            &mut empty.workspace(&request(empty.order(), 1)).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.backward_error(), 0.0);
    assert_eq!(
        report.levinson_solves()
            + report.gko_solves()
            + report.dense_lu_solves()
            + report.circulant_solves(),
        0
    );
    for matrix in [
        StructuredMatrixView::toeplitz(&[4., 0.5, 0.25], &[4., 0.5, 0.25]).unwrap(),
        StructuredMatrixView::hankel(&[0.25, 0.5, 4.], &[4., 0.5, 0.25]).unwrap(),
        StructuredMatrixView::circulant(&[4., 0.5, 0.25]).unwrap(),
    ] {
        let plan = StructuredSolvePlan::new(matrix, &StructuredSolveOptions::new()).unwrap();
        let rhs = Matrix::from_vec(3, 2, vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let mut output = Matrix::zeros(3, 2).unwrap();
        let report = plan
            .solve_matrix_into(
                StructuredOperation::Normal,
                &rhs,
                output.view_mut(),
                &mut plan.workspace(&request(plan.order(), 2)).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            report.levinson_solves()
                + report.gko_solves()
                + report.dense_lu_solves()
                + report.circulant_solves(),
            2
        );
        let mut maximum = 0.0_f64;
        for j in 0..2 {
            let mut x = [0.; 3];
            let single = plan
                .solve_into(
                    StructuredOperation::Normal,
                    &[
                        rhs.view().get(0, j).unwrap(),
                        rhs.view().get(1, j).unwrap(),
                        rhs.view().get(2, j).unwrap(),
                    ][..],
                    VectorViewMut::from_slice(&mut x),
                    &mut plan.workspace(&request(plan.order(), 1)).unwrap(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            maximum = maximum.max(single.backward_error());
            assert_eq!(report.levinson_solves(), 2 * single.levinson_solves());
            assert_eq!(report.gko_solves(), 2 * single.gko_solves());
            assert_eq!(report.dense_lu_solves(), 2 * single.dense_lu_solves());
            assert_eq!(report.circulant_solves(), 2 * single.circulant_solves());
        }
        assert_eq!(report.backward_error(), maximum);
        let rhs = Matrix::zeros(3, 0).unwrap();
        let mut output = Matrix::zeros(3, 0).unwrap();
        let report = plan
            .solve_matrix_into(
                StructuredOperation::Normal,
                &rhs,
                output.view_mut(),
                &mut plan.workspace(&request(plan.order(), 0)).unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.backward_error(), 0.0);
        assert_eq!(
            report.levinson_solves()
                + report.gko_solves()
                + report.dense_lu_solves()
                + report.circulant_solves(),
            0
        );
    }
    let c = [Complex::new(4., 0.), Complex::new(0.5, 0.)];
    let plan = StructuredSolvePlan::new(
        StructuredMatrixView::toeplitz(&c, &c).unwrap(),
        &StructuredSolveOptions::new(),
    )
    .unwrap();
    let mut out = [Complex::new(0., 0.); 2];
    let report = plan
        .solve_into(
            StructuredOperation::Normal,
            &c[..],
            VectorViewMut::from_slice(&mut out),
            &mut plan.workspace(&request(plan.order(), 1)).unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.levinson_solves(), 1);
    close(out[0], Complex::new(1., 0.));
    close(out[1], Complex::new(0., 0.));
}

#[test]
fn structured_binary_extremes_and_failed_publication() {
    for power in [-1070, -1000, 1000] {
        for base in [[4.0_f64, 1.0], [0.0, 1.0]] {
            let c = base.map(|v| v.scalbn(power));
            let b = [
                (base[0] + 2. * base[1]).scalbn(power),
                (base[1] + 2. * base[0]).scalbn(power),
            ];
            let plan = StructuredSolvePlan::new(
                StructuredMatrixView::toeplitz(&c, &c).unwrap(),
                &StructuredSolveOptions::new(),
            )
            .unwrap();
            let mut out = [99.; 2];
            let report = plan
                .solve_into(
                    StructuredOperation::Normal,
                    &b[..],
                    VectorViewMut::from_slice(&mut out),
                    &mut plan
                        .workspace(&request(plan.order(), 1).with_dense_fallback())
                        .unwrap(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.backward_error() < 1e-12);
            close(out[0], 1.);
            close(out[1], 2.);
        }
    }
    let c = [f64::MIN_POSITIVE];
    let plan = StructuredSolvePlan::new(
        StructuredMatrixView::toeplitz(&c, &c).unwrap(),
        &StructuredSolveOptions::new(),
    )
    .unwrap();
    let mut out = [99.];
    assert!(
        plan.solve_into(
            StructuredOperation::Normal,
            &[f64::MAX][..],
            VectorViewMut::from_slice(&mut out),
            &mut plan
                .workspace(&request(plan.order(), 1).with_dense_fallback())
                .unwrap(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out, [99.]);
    let rhs = Matrix::from_vec(1, 2, vec![f64::MIN_POSITIVE, f64::MAX]).unwrap();
    let mut out = Matrix::from_vec(1, 2, vec![99., 99.]).unwrap();
    assert!(
        plan.solve_matrix_into(
            StructuredOperation::Normal,
            &rhs,
            out.view_mut(),
            &mut plan
                .workspace(&request(plan.order(), 2).with_dense_fallback())
                .unwrap(),
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(out.as_slice(), &[99., 99.]);
}

#[test]
fn residual_rejection_retries_lu_and_aggregates_mixed_paths() {
    let c = [2.0_f64, 0.1875, -0.25];
    let rhs = [4.0, -6.0 / 11.0, 14.0 / 13.0];
    // This SPD system completes Levinson but its rounded candidate misses the
    // strict residual tolerance. LU gives an independently accepted result.
    let options = StructuredSolveOptions::new()
        .with_tolerance(scioxide_core::Tolerance::new(0., 3e-17).unwrap());
    let plan = StructuredSolvePlan::new(StructuredMatrixView::toeplitz(&c, &c).unwrap(), &options)
        .unwrap();
    let mut out = [0.; 3];
    let report = plan
        .solve_into(
            StructuredOperation::Normal,
            &rhs[..],
            VectorViewMut::from_slice(&mut out),
            &mut plan
                .workspace(&request(plan.order(), 1).with_dense_fallback())
                .unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.levinson_solves(), 0);
    assert_eq!(report.dense_lu_solves(), 1);
    assert!(report.backward_error() <= 3e-17);
    let rhs = Matrix::from_vec(3, 2, vec![rhs[0], 0., rhs[1], 0., rhs[2], 0.]).unwrap();
    let mut out = Matrix::zeros(3, 2).unwrap();
    let report = plan
        .solve_matrix_into(
            StructuredOperation::Normal,
            &rhs,
            out.view_mut(),
            &mut plan
                .workspace(&request(plan.order(), 2).with_dense_fallback())
                .unwrap(),
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.levinson_solves(), 1);
    assert_eq!(report.dense_lu_solves(), 1);
    #[cfg(feature = "std")]
    {
        let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
        let mut parallel = Matrix::zeros(3, 2).unwrap();
        let parallel_report = plan
            .solve_matrix_into(
                StructuredOperation::Normal,
                &rhs,
                parallel.view_mut(),
                &mut plan
                    .workspace(&request(plan.order(), 2).with_dense_fallback())
                    .unwrap(),
                ExecutionContext::new(&pool),
            )
            .unwrap();
        assert_eq!(parallel, out);
        assert_eq!(parallel_report, report);
    }
    let rejected = StructuredSolvePlan::new(
        StructuredMatrixView::toeplitz(&c, &c).unwrap(),
        &options.with_tolerance(scioxide_core::Tolerance::new(0., 0.).unwrap()),
    )
    .unwrap();
    let before = out.clone();
    for request in [
        request(rejected.order(), 2),
        request(rejected.order(), 2).with_dense_fallback(),
    ] {
        let mut workspace = rejected.workspace(&request).unwrap();
        let report = rejected
            .solve_matrix_into(
                StructuredOperation::Normal,
                &rhs,
                out.view_mut(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            DirectSolveTermination::ResidualRejected
        );
        assert!(report.backward_error() > 0.);
        assert_eq!(report.levinson_solves(), 1);
        assert_eq!(out, before);
    }
    // Condition estimates must not consume rejected inverse columns.
    assert_eq!(
        rejected.condition_estimate().unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
}

#[test]
fn exact_operator_utilities_forward_scope_and_preserve_strided_padding() {
    use scioxide_core::{EvaluationRole, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let op = scoped_adjoint::ScopedDiagonal::new();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        ProblemToken::new(1101),
        scioxide_core::algorithm_id!("test", "operator_utilities", 1),
        0,
        0,
    );
    let mut parent = state
        .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
        .unwrap();
    let mut workspace = LinalgWorkspace::new();
    assert_eq!(
        operator_norm_scoped(
            &op,
            MatrixNorm::One,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut workspace,
            &mut parent
        )
        .unwrap(),
        4.0
    );
    let mut output = [99.0; 18];
    materialize_operator_into(
        &op,
        &scioxide_linalg::LinalgExecutionOptions::new(),
        MatrixViewMut::from_slice_with_strides(3, 3, &mut output, 6, 2).unwrap(),
        &mut workspace,
        &mut parent,
    )
    .unwrap();
    for (i, pair) in output.chunks_exact(2).enumerate() {
        assert_eq!(
            pair[0],
            if i % 4 == 0 {
                2.0 + (i / 4) as f64
            } else {
                0.0
            }
        );
        assert_eq!(pair[1], 99.0);
    }
    assert_eq!(parent.counts().own().iterations(), 0);
    assert_eq!(parent.counts().nested().iterations(), 18);
    assert_eq!(
        parent.counts().nested().provider(EvaluationRole::Function),
        6
    );
    assert_eq!(
        workspace
            .last_attempt()
            .unwrap()
            .counts()
            .own()
            .iterations(),
        3
    );
    assert_eq!(operator_norm(&op, MatrixNorm::Infinity).unwrap(), 4.0);
}

#[test]
fn operator_utilities_reject_late_cancellation_and_provider_failure() {
    use scioxide_core::{CancellationToken, SciOxideError};
    struct Stops<'a> {
        token: &'a CancellationToken,
        fail: bool,
    }
    impl LinearOperator<f64> for Stops<'_> {
        fn rows(&self) -> usize {
            2
        }
        fn cols(&self) -> usize {
            2
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
            y.copy_from_slice(x);
            if x[1] != 0.0 {
                if self.fail {
                    return Err(SciOxideError::singular(None));
                }
                self.token.cancel();
            }
            Ok(())
        }
    }
    for fail in [false, true] {
        for materialize in [false, true] {
            let token = CancellationToken::new();
            let op = Stops {
                token: &token,
                fail,
            };
            let mut output = Matrix::from_vec(2, 2, vec![91.0; 4]).unwrap();
            let mut workspace = LinalgWorkspace::new();
            let cx = ExecutionContext::sequential().with_cancellation(&token);
            let result = if materialize {
                materialize_operator_into(
                    &op,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    output.view_mut(),
                    &mut workspace,
                    cx,
                )
            } else {
                operator_norm_scoped(
                    &op,
                    MatrixNorm::One,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    &mut workspace,
                    cx,
                )
                .map(|_| ())
            };
            assert!(result.is_err());
            assert_eq!(output.as_slice(), &[91.0; 4]);
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                if fail {
                    scioxide_core::AttemptOutcome::Failed
                } else {
                    scioxide_core::AttemptOutcome::Stopped
                }
            );
        }
    }
}

#[test]
fn large_kms_accepts_subnormal_coefficients_and_meets_backward_error() {
    for rho in [0.3_f64, 0.9] {
        for n in [1_000, 10_000] {
            let column: Vec<_> = (0..n).map(|i| rho.powi(i as i32)).collect();
            if rho == 0.3 || n == 10_000 {
                assert!(column.iter().any(|x| x.is_subnormal()));
            }
            let options = StructuredSolveOptions::new()
                .with_tolerance(scioxide_core::Tolerance::new(0.0, 1e-14).unwrap());
            let plan = StructuredSolvePlan::new(
                StructuredMatrixView::toeplitz(&column, &column).unwrap(),
                &options,
            )
            .unwrap();
            let rhs: Vec<_> = (0..n)
                .map(|i| ((17 * i % 37) as f64 - 18.0) / 19.0)
                .collect();
            let mut x = vec![0.0; n];
            let report = plan
                .solve_into(
                    StructuredOperation::Normal,
                    &rhs[..],
                    VectorViewMut::from_slice(&mut x),
                    &mut plan.workspace(&request(plan.order(), 1)).unwrap(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.backward_error() <= 1e-14);
            // The KMS inverse is independently tridiagonal. Multiplication by
            // KMS uses two geometric recurrences, without dense storage.
            let denominator = (1.0 - rho) * (1.0 + rho);
            let mut left = vec![0.0; n];
            let mut norm_x = 0.0_f64;
            let mut norm_b = 0.0_f64;
            for i in 0..n {
                let diagonal = if i == 0 || i + 1 == n {
                    1.0
                } else {
                    1.0 + rho * rho
                };
                let exact = (diagonal * rhs[i]
                    - if i > 0 { rho * rhs[i - 1] } else { 0.0 }
                    - if i + 1 < n { rho * rhs[i + 1] } else { 0.0 })
                    / denominator;
                assert!((x[i] - exact).abs() <= 1e-12 * (1.0 + exact.abs()));
                norm_x = norm_x.max(x[i].abs());
                norm_b = norm_b.max(rhs[i].abs());
                if i > 0 {
                    left[i] = rho * (left[i - 1] + x[i - 1]);
                }
            }
            let mut right = 0.0;
            let mut residual = 0.0_f64;
            for i in (0..n).rev() {
                residual = residual.max((left[i] + x[i] + right - rhs[i]).abs());
                right = rho * (right + x[i]);
            }
            assert!(residual / (((1.0 + rho) / (1.0 - rho)) * norm_x + norm_b) <= 1e-14);
        }
    }
}

#[test]
fn materialization_caps_report_resource_limit() {
    let column = [1.0_f64, 2.0, 3.0];
    let row = [1.0_f64, 4.0];
    let a = StructuredMatrixView::toeplitz(&column, &row).unwrap();
    let cap = cap::<f64>;
    let dense = a.materialize_with_options(&cap(6)).unwrap();
    assert_eq!(dense.as_slice(), &[1.0, 4.0, 2.0, 1.0, 3.0, 2.0]);
    assert_eq!(
        a.materialize_with_options(&cap(5)).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let owned = StructuredMatrix::from_view_with_options(a, &cap(5)).unwrap();
    assert_eq!(owned.view().materialize().unwrap(), dense);
    assert_eq!(
        StructuredMatrix::from_view_with_options(a, &cap(4))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    // An empty matrix retains no coefficients; canonical patterns have none.
    let empty = StructuredMatrixView::<f64>::toeplitz(&[], &row).unwrap();
    assert!(StructuredMatrix::from_view_with_options(empty, &cap(0)).is_ok());
    let hilbert = StructuredMatrixView::<f64>::hilbert(2, MatrixForm::Direct).unwrap();
    assert_eq!(
        StructuredMatrix::from_view_with_options(hilbert, &cap(9))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn owned_structured_storage_rejects_time_limits_without_a_clock() {
    let column = [1.0_f64, 2.0, 3.0];
    let row = [1.0_f64, 4.0];
    let a = StructuredMatrixView::toeplitz(&column, &row).unwrap();
    let timed = LinalgExecutionOptions::new().with_time_limit(core::time::Duration::from_secs(1));
    let factor = Matrix::from_vec(2, 1, vec![2.0_f64, 3.0]).unwrap();
    for kind in [
        a.materialize_with_options(&timed).unwrap_err().kind(),
        StructuredMatrix::from_view_with_options(a, &timed)
            .unwrap_err()
            .kind(),
        khatri_rao_with_options(factor.view(), factor.view(), &timed)
            .unwrap_err()
            .kind(),
    ] {
        assert_eq!(kind, SciOxideErrorKind::CapabilityUnavailable);
    }
}

#[test]
fn dft_direction_selects_conjugate_matrices() {
    let forward =
        StructuredMatrixView::<Complex<f64>>::dft(4, DftDirection::Forward, DftNormalization::None)
            .unwrap();
    let inverse = StructuredMatrixView::<Complex<f64>>::dft(
        4,
        DftDirection::Inverse,
        DftNormalization::ByOrder,
    )
    .unwrap();
    assert_eq!(forward.get(1, 1).unwrap(), Complex::new(0.0, -1.0));
    for i in 0..4 {
        for j in 0..4 {
            close(
                inverse.get(i, j).unwrap() * Complex::new(4.0, 0.0),
                forward.get(i, j).unwrap().conj(),
            );
        }
    }
}

#[test]
fn composite_shape_errors_are_dimension_mismatches() {
    let two = Matrix::from_vec(1, 1, vec![2.0]).unwrap();
    let rectangular = Matrix::from_vec(1, 2, vec![1.0, 2.0]).unwrap();
    let (two, rectangular) = (
        two.view().try_finite().unwrap(),
        rectangular.view().try_finite().unwrap(),
    );
    let blocks: [&(dyn AdjointOperator<f64> + Sync); 2] = [&two, &rectangular];
    let operator = BlockDiagonalOperator::new(&blocks).unwrap();
    let kind = |result: Result<OperatorWorkspaceRequest<f64>, scioxide_core::SciOxideError>| {
        result.unwrap_err().kind()
    };
    let none = LinalgExecutionOptions::new();
    assert_eq!(
        kind(OperatorWorkspaceRequest::block_diagonal_solve(
            &operator,
            &[&two],
            &none
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        kind(OperatorWorkspaceRequest::block_diagonal_solve(
            &operator,
            &[&two, &two],
            &none
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    let square: [&(dyn AdjointOperator<f64> + Sync); 1] = [&two];
    let square = BlockDiagonalOperator::new(&square).unwrap();
    assert_eq!(
        kind(OperatorWorkspaceRequest::block_diagonal_solve(
            &square,
            &[&rectangular],
            &none
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        square
            .apply_into(&[1.0, 2.0], &mut [0.0], &mut [0.0; 2])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let product = KroneckerOperator::new(&rectangular, &two).unwrap();
    assert_eq!(
        kind(OperatorWorkspaceRequest::kronecker_solve(
            &product,
            [&rectangular, &two],
            &none
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        product
            .apply_into(&[1.0], &mut [0.0], &mut [0.0; 8])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let circulant = StructuredMatrixView::circulant(&[1.0_f64, 2.0]).unwrap();
    assert_eq!(
        circulant
            .apply_into(&[1.0], &mut [0.0; 2], &mut [0.0; 2])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        StructuredMatrixView::leslie(&[1.0_f64, 2.0], &[0.5, 0.5])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
}

fn every_operation<T: NumericScalar>(matrix: StructuredMatrixView<'_, T>) {
    use StructuredOperation::{Adjoint, Normal, Transpose};
    let n = matrix.rows();
    let dense = matrix.materialize().unwrap();
    let plan = StructuredSolvePlan::new(matrix, &StructuredSolveOptions::new()).unwrap();
    let x: Vec<T> = (0..n)
        .map(|i| scalar(1.0 + i as f64, 0.5 - 0.25 * i as f64))
        .collect();
    for op in [Normal, Transpose, Adjoint] {
        let entry = |i: usize, j: usize| match op {
            Normal => dense.view().get(i, j).unwrap(),
            Transpose => dense.view().get(j, i).unwrap(),
            Adjoint => dense.view().get(j, i).unwrap().conj(),
        };
        let b: Vec<T> = (0..n)
            .map(|i| (0..n).fold(T::ZERO, |sum, j| sum + entry(i, j) * x[j]))
            .collect();
        let mut out = vec![T::ZERO; n];
        plan.solve_into(
            op,
            &b[..],
            VectorViewMut::from_slice(&mut out),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        for (&actual, &expected) in out.iter().zip(&x) {
            close(actual, expected);
        }
    }
}

fn operation_per_solve<T: NumericScalar>() {
    let c = [
        scalar::<T>(4.0, 1.0),
        scalar(1.0, -0.5),
        scalar(-1.0, 0.25),
        scalar(0.5, 0.0),
    ];
    let r = [
        c[0],
        scalar(2.0, 0.5),
        scalar(0.5, -1.0),
        scalar(-0.25, 0.0),
    ];
    let symmetric = [c[0], c[1], c[2], c[3]];
    every_operation(StructuredMatrixView::toeplitz(&c, &r).unwrap());
    every_operation(StructuredMatrixView::toeplitz(&symmetric, &symmetric).unwrap());
    every_operation(StructuredMatrixView::circulant(&c).unwrap());
    // A large middle anti-diagonal keeps the Hankel system well conditioned.
    let column = [
        scalar::<T>(0.5, 0.25),
        scalar(-1.0, 0.0),
        scalar(1.0, -0.5),
        c[0],
    ];
    let last_row = [c[0], scalar(1.0, 0.5), scalar(-0.5, 0.0), scalar(0.25, 1.0)];
    every_operation(StructuredMatrixView::hankel(&column, &last_row).unwrap());
}

#[test]
fn one_compact_plan_solves_every_operation() {
    operation_per_solve::<f64>();
    operation_per_solve::<Complex<f64>>();
}

#[test]
fn default_request_reserves_one_packed_gko_factor() {
    let n = 10_000;
    let mut column = vec![0_f64; n];
    column[0] = 1.;
    let plan = StructuredSolvePlan::new(
        StructuredMatrixView::toeplitz(&column, &column).unwrap(),
        &StructuredSolveOptions::new(),
    )
    .unwrap();
    let factor = n * (n + 1) + 14 * n; // Complex upper triangle and seven complex vectors.
    for columns in [1, 3] {
        assert_eq!(
            plan.workspace_requirements(&request(n, columns))
                .unwrap()
                .scalar_len()
                .unwrap(),
            factor + columns * (3 * n + 2)
        );
    }
    assert_eq!(
        plan.workspace_requirements(&request(n, 0))
            .unwrap()
            .scalar_len()
            .unwrap(),
        0
    );
    let column = [1_f64, 2., 0.];
    let plan = StructuredSolvePlan::new(
        StructuredMatrixView::toeplitz(&column, &column).unwrap(),
        &StructuredSolveOptions::new(),
    )
    .unwrap();
    let basic = request(3, 1);
    assert_eq!(
        plan.workspace_requirements(&basic)
            .unwrap()
            .scalar_len()
            .unwrap(),
        65
    );
    assert_eq!(
        plan.workspace_requirements(&basic.with_dense_fallback())
            .unwrap()
            .scalar_len()
            .unwrap(),
        101
    );
    for request in [basic, basic.with_dense_fallback()] {
        let mut workspace = plan.workspace(&request).unwrap();
        let mut output = [91.; 3];
        let report = plan
            .solve_into(
                StructuredOperation::Normal,
                &[3., 5., 3.][..],
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.levinson_solves(), 0);
        assert_eq!(report.gko_solves(), 1);
        assert_eq!(report.dense_lu_solves(), 0);
        for value in output {
            close(value, 1.);
        }
    }
}

#[test]
fn hermitian_levinson_preserves_complex_operations() {
    fn check<T: NumericScalar>() {
        let column = [scalar::<T>(4., 0.), scalar(0.5, 0.75), scalar(-0.25, 0.125)];
        let row = column.map(NumericScalar::conj);
        let matrix = StructuredMatrixView::toeplitz(&column, &row).unwrap();
        let plan = StructuredSolvePlan::new(matrix, &StructuredSolveOptions::new()).unwrap();
        assert_eq!(
            plan.workspace_requirements(&request(3, 1))
                .unwrap()
                .scalar_len()
                .unwrap(),
            38
        );
        let expected = [scalar::<T>(1., 2.), scalar(-2., 0.5), scalar(0.25, -1.)];
        let mut workspace = plan.workspace(&request(3, 1)).unwrap();
        for op in [
            StructuredOperation::Normal,
            StructuredOperation::Transpose,
            StructuredOperation::Adjoint,
        ] {
            let rhs: Vec<_> = (0..3)
                .map(|i| {
                    (0..3).fold(T::ZERO, |sum, j| {
                        let a = match op {
                            StructuredOperation::Normal => matrix.get(i, j).unwrap(),
                            StructuredOperation::Transpose => matrix.get(j, i).unwrap(),
                            StructuredOperation::Adjoint => matrix.get(j, i).unwrap().conj(),
                        };
                        sum + a * expected[j]
                    })
                })
                .collect();
            let mut output = [T::ZERO; 3];
            let report = plan
                .solve_into(
                    op,
                    &rhs[..],
                    VectorViewMut::from_slice(&mut output),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.levinson_solves(), 1);
            for (actual, wanted) in output.into_iter().zip(expected) {
                close(actual, wanted);
            }
        }
    }
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}

/// Square operator whose every application returns NaN.
struct NanOperator;
impl LinearOperator<f64> for NanOperator {
    fn rows(&self) -> usize {
        2
    }
    fn cols(&self) -> usize {
        2
    }
    fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn apply_into(
        &self,
        _: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        y.fill(f64::NAN);
        Ok(())
    }
}
impl AdjointOperator<f64> for NanOperator {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        self.apply_into(x, y, w)
    }
}

#[test]
fn nonfinite_operator_output_has_callback_origin() {
    let callback = |error: scioxide_core::SciOxideError| {
        assert!(
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Callback,
                    ..
                }
            ),
            "{error:?}"
        );
    };
    let options = LinalgExecutionOptions::new();
    let mut w = LinalgWorkspace::new();
    let cx = ExecutionContext::sequential;
    callback(
        operator_norm_scoped(&NanOperator, MatrixNorm::One, &options, &mut w, cx()).unwrap_err(),
    );
    let mut dense = Matrix::<f64>::zeros(2, 2).unwrap();
    callback(
        materialize_operator_into(&NanOperator, &options, dense.view_mut(), &mut w, cx())
            .unwrap_err(),
    );
    let input = Matrix::<f64>::identity(2).unwrap();
    callback(
        apply_operator_matrix_into(
            &NanOperator,
            &input,
            &options,
            dense.view_mut(),
            &mut w,
            cx(),
        )
        .unwrap_err(),
    );
    assert_eq!(dense.as_slice(), &[0.0; 4]);
    // Composite products check each inner provider's output at its boundary.
    let blocks: [&(dyn AdjointOperator<f64> + Sync); 1] = [&NanOperator];
    let block = BlockDiagonalOperator::new(&blocks).unwrap();
    let kronecker = KroneckerOperator::new(&NanOperator, &NanOperator).unwrap();
    for composite in [&block as &dyn LinearOperator<f64>, &kronecker] {
        let mut y = vec![0.0; composite.rows()];
        let mut scratch = vec![0.0; composite.apply_workspace_len().unwrap()];
        callback(
            composite
                .apply_into(&vec![1.0; composite.cols()], &mut y, &mut scratch)
                .unwrap_err(),
        );
    }
}

#[test]
fn symmetric_pascal_inverse_entries_match_integer_sums() {
    // inv(P)ᵢⱼ = (−1)^(i+j) Σ_{k ≥ max(i, j)} C(k, i) C(k, j) for P of order n.
    fn binomial(n: u128, k: u128) -> u128 {
        (0..k).fold(1, |value, j| value * (n - j) / (j + 1))
    }
    let n = 24_usize;
    let inverse = StructuredMatrixView::<f64>::pascal(
        n,
        PascalKind::Symmetric,
        scioxide_linalg::MatrixForm::Inverse,
    )
    .unwrap();
    for (i, j) in [(0, 0), (1, 1), (0, 1), (5, 3), (11, 17), (23, 23), (22, 23)] {
        let sum: u128 = (i.max(j)..n)
            .map(|k| binomial(k as u128, i as u128) * binomial(k as u128, j as u128))
            .sum();
        let expected = if (i + j) % 2 == 0 {
            sum as f64
        } else {
            -(sum as f64)
        };
        let actual = inverse.get(i, j).unwrap();
        assert!((actual - expected).abs() <= 64.0 * f64::EPSILON * expected.abs());
    }
}
