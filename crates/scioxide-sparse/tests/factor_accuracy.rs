//! Arithmetic tiers retain range, projection, allocation and publication contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Accuracy, Complex, ExecutionContext, ProductSum, SciOxideErrorKind};
use scioxide_sparse::*;

/// Solves with `solve_into` and returns the solution, for configured options.
macro_rules! solve {
    ($factor:expr, $rhs:expr, $operation:expr, $options:expr) => {{
        let factor = &$factor;
        let rhs = $rhs;
        let mut output = rhs.to_vec();
        factor
            .solve_into(
                &rhs[..],
                $operation,
                $options,
                &mut output,
                &mut factor.workspace(1).unwrap(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .map(|()| output)
    }};
}

fn lu_plan(a: &CsrMatrix<f64>) -> SparseLuSymbolic {
    SparseLuSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::Natural,
        SparseLuLimits::new(a.rows() * a.rows(), usize::MAX),
    )
    .unwrap()
}
fn symmetric_plan<S: scioxide_core::NumericScalar>(a: &CsrMatrix<S>) -> SparseSymmetricSymbolic {
    SparseSymmetricSymbolic::analyze(
        a.structural_pattern(),
        SparseOrdering::Natural,
        None,
        SparseSymmetricLimits::new(a.rows() * a.rows(), a.rows() * a.rows(), usize::MAX),
    )
    .unwrap()
}
fn incomplete_plan(
    a: &CsrMatrix<f64>,
    method: IncompleteMethod,
    block: usize,
) -> IncompleteSymbolic {
    IncompleteSymbolic::analyze(
        a.structural_pattern(),
        method,
        block,
        IncompleteLimits::new(a.rows() * a.rows(), usize::MAX),
    )
    .unwrap()
}
fn incomplete_options() -> IncompleteOptions<f64> {
    IncompleteOptions::new()
}
fn close(actual: &[f64], expected: &[f64]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 128. * f64::EPSILON * (1. + expected.abs()),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn extreme_diagonals_in_both_factor_and_solve_tiers() {
    assert_eq!(SparseSolveOptions::new().accuracy(), Accuracy::Standard);
    for scale in [f64::from_bits(1), 1e-300, 1e300, f64::MAX] {
        let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![scale; 2]).unwrap();
        for factor_accuracy in [Accuracy::Standard, Accuracy::Exact] {
            let options = SparseSolveOptions::new().with_accuracy(factor_accuracy);
            let lu = SparseLu::factor_scoped(
                lu_plan(&a),
                a.view(),
                options,
                ExecutionContext::sequential(),
            )
            .unwrap();
            let chol = SparseCholesky::factor_scoped(
                symmetric_plan(&a),
                a.view(),
                options,
                ExecutionContext::sequential(),
            )
            .unwrap();
            let ldl = SparseLdl::factor_scoped(
                symmetric_plan(&a),
                a.view(),
                options,
                ExecutionContext::sequential(),
            )
            .unwrap();
            for solve_accuracy in [Accuracy::Standard, Accuracy::Exact] {
                let solve = options.with_accuracy(solve_accuracy);
                for operation in [
                    SparseOperation::Normal,
                    SparseOperation::Transpose,
                    SparseOperation::Adjoint,
                ] {
                    close(
                        &solve!(lu, &[scale; 2], operation, solve).unwrap(),
                        &[1.; 2],
                    );
                    close(
                        &solve!(chol, &[scale; 2], operation, solve).unwrap(),
                        &[1.; 2],
                    );
                    close(
                        &solve!(ldl, &[scale; 2], operation, solve).unwrap(),
                        &[1.; 2],
                    );
                }
            }
            for method in [
                IncompleteMethod::Ilu { level: 0 },
                IncompleteMethod::Ilut,
                IncompleteMethod::Ic0,
                IncompleteMethod::Ict,
            ] {
                for block in [1, 2] {
                    let factor = IncompleteFactor::factor_scoped(
                        incomplete_plan(&a, method, block),
                        a.view(),
                        incomplete_options(),
                        options,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
                        close(
                            &solve!(
                                factor,
                                &[scale; 2],
                                SparseOperation::Normal,
                                options.with_accuracy(accuracy)
                            )
                            .unwrap(),
                            &[1.; 2],
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn accuracy_is_selected_independently_for_each_operation() {
    let b = 1. + 2_f64.powi(-27);
    let c = 1. - 2_f64.powi(-27);
    let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1., b, 1.]).unwrap();
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let options = SparseSolveOptions::new().with_accuracy(accuracy);
        let lu = SparseLu::factor_scoped(
            lu_plan(&a),
            a.view(),
            options,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let standard = solve!(
            lu,
            &[1., c],
            SparseOperation::Normal,
            options.with_accuracy(Accuracy::Standard)
        )
        .unwrap();
        let exact = solve!(
            lu,
            &[1., c],
            SparseOperation::Normal,
            options.with_accuracy(Accuracy::Exact)
        )
        .unwrap();
        assert_eq!(standard[0], 0.);
        assert_eq!(exact[0], 2_f64.powi(-54));
    }
}

#[test]
fn strict_projection_and_overflowing_complex_magnitudes_keep_their_errors() {
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let options = SparseSolveOptions::new().with_accuracy(accuracy);
        let z = Complex::new(f64::MAX, f64::MAX);
        let a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![z]).unwrap();
        let plan = SparseLuSymbolic::analyze(
            a.structural_pattern(),
            SparseOrdering::Natural,
            SparseLuLimits::new(1, usize::MAX),
        )
        .unwrap();
        let factor =
            SparseLu::factor_scoped(plan, a.view(), options, ExecutionContext::sequential())
                .unwrap();
        let solution = solve!(factor, &[z], SparseOperation::Normal, options).unwrap();
        assert_eq!(solution, [Complex::new(1., 0.)]);

        let off = Complex::new(1., f64::from_bits(1));
        let a = CsrMatrix::from_raw(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![Complex::new(4., 0.), off.conj(), off, Complex::new(4., 0.)],
        )
        .unwrap();
        let error = SparseCholesky::factor_scoped(
            symmetric_plan(&a),
            a.view(),
            options,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);

        let a = CsrMatrix::from_raw(
            2,
            2,
            vec![0, 1, 3],
            vec![0, 0, 1],
            vec![f64::MAX, f64::from_bits(1), 1.],
        )
        .unwrap();
        let error = SparseLu::factor_scoped(
            lu_plan(&a),
            a.view(),
            options,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    }
}

#[test]
fn standard_residual_obeys_the_componentwise_backward_error_envelope() {
    let n = 12;
    let mut builder = SparseBuilder::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let value = if i == j {
                8.
            } else {
                ((i * 7 + j * 3) % 11) as f64 / 16. - 0.3
            };
            builder.push(i, j, value).unwrap();
        }
    }
    let a = builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let rhs: Vec<_> = (0..n).map(|i| i as f64 / 7. - 0.4).collect();
    let factor = SparseLu::factor(lu_plan(&a), a.view()).unwrap();
    let x = factor.solve(&rhs).unwrap();
    let k = (8 * n + 8) as f64;
    let gamma = k * f64::EPSILON / (1. - k * f64::EPSILON);
    for (i, &b) in rhs.iter().enumerate() {
        let mut residual = ProductSum::new();
        residual.add_product(b, 1.).unwrap();
        let mut scale = b.abs();
        for position in a.row_ptr()[i]..a.row_ptr()[i + 1] {
            let coefficient = a.values()[position];
            let value = x[a.col_indices()[position]];
            residual.subtract_product(coefficient, value).unwrap();
            scale += coefficient.abs() * value.abs();
        }
        assert!(residual.finish().unwrap().abs() <= gamma * scale);
    }
}

#[test]
fn ordinary_ldl_keeps_exact_cancellation_and_real_hermitian_diagonals() {
    let c = |re: f64, im: f64| Complex::new(re, im);
    let near = |actual: Complex<f64>, expected: Complex<f64>| {
        assert!(
            (actual - expected).norm() <= 64. * f64::EPSILON * (1. + expected.norm()),
            "{actual:?} != {expected:?}"
        );
    };
    // [[2, 2i, 0], [-2i, 2, 1+i], [0, 1-i, 2]]: the first Schur diagonal cancels
    // exactly to zero, so a 1x1 pivot follows with D = [2, 2, -1].
    let a = CsrMatrix::from_raw(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![
            c(2., 0.),
            c(0., 2.),
            c(0., -2.),
            c(2., 0.),
            c(1., 1.),
            c(1., -1.),
            c(2., 0.),
        ],
    )
    .unwrap();
    let solution = [c(1., -1.), c(0.5, 2.), c(-3., 0.25)];
    let rhs: Vec<_> = (0..3)
        .map(|row| {
            (a.row_ptr()[row]..a.row_ptr()[row + 1]).fold(c(0., 0.), |sum, position| {
                sum + a.values()[position] * solution[a.col_indices()[position]]
            })
        })
        .collect();
    // [[0, 1+i], [1-i, 0]] needs a 2x2 pivot whose second Cramer numerator
    // cancels to zero for the right-hand side [0, 1-i] with solution [1, 0].
    let saddle =
        CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![1, 0], vec![c(1., 1.), c(1., -1.)]).unwrap();
    for accuracy in [Accuracy::Standard, Accuracy::Exact] {
        let options = SparseSolveOptions::new().with_accuracy(accuracy);
        let factor = SparseLdl::factor_scoped(
            symmetric_plan(&a),
            a.view(),
            options,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(factor.inertia().unwrap(), LdlInertia::new(2, 1, 0).unwrap());
        assert!(factor.diagonal().iter().all(|d| d.im == 0.));
        let mut diagonal: Vec<_> = factor.diagonal().iter().map(|d| d.re).collect();
        diagonal.sort_by(f64::total_cmp);
        close(&diagonal, &[-1., 2., 2.]);
        let x = solve!(factor, &rhs, SparseOperation::Normal, options).unwrap();
        for (&actual, &expected) in x.iter().zip(&solution) {
            near(actual, expected);
        }

        let factor = SparseLdl::factor_scoped(
            symmetric_plan(&saddle),
            saddle.view(),
            options,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(factor.inertia().unwrap(), LdlInertia::new(1, 1, 0).unwrap());
        let x = solve!(
            factor,
            &[c(0., 0.), c(1., -1.)],
            SparseOperation::Normal,
            options
        )
        .unwrap();
        near(x[0], c(1., 0.));
        near(x[1], c(0., 0.));
    }
}
