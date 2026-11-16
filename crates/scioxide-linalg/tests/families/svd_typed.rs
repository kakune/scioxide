//! Public contracts for typed SVD jobs.

#![allow(clippy::panic)]

use crate::support::metrics as decomposition_metrics;

use crate::support::svd_cases;

use crate::support::bidiagonal;
use scioxide_core::{AlgebraicScalar, Complex};
use scioxide_core::{Budget, ExecutionPolicy, NumericScalar, RealScalar};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, Matrix, SvdValues, ThinSvdDecomposition,
};

fn graded_bidiagonal_values<S: NumericScalar>() {
    let cast = |x| S::Real::try_from_f64_lossy(x).unwrap_or_else(|error| panic!("{error}"));
    for case in bidiagonal::cases::<S::Real>() {
        let diagonal = case.diagonal;
        let off = case.off_diagonal;
        let expected = case.expected;
        let n = diagonal.len();
        let bound = cast(128.0 * n as f64) * S::Real::EPSILON;
        for conjugate in [false, true] {
            let phase = |i: usize| match i % 4 {
                0 => S::ONE,
                1 if !S::IS_REAL => S::from_components(S::Real::ZERO, S::Real::ONE),
                _ => -S::ONE,
            };
            let mut a = Matrix::<S>::zeros(n, n).unwrap_or_else(|error| panic!("{error}"));
            for i in 0..n {
                a.view_mut()
                    .set(i, i, S::from_real(diagonal[i]))
                    .unwrap_or_else(|error| panic!("{error}"));
                if i + 1 < n {
                    a.view_mut()
                        .set(i, i + 1, phase(i) * S::from_real(off[i]))
                        .unwrap_or_else(|error| panic!("{error}"));
                }
            }
            let view = if conjugate {
                a.view().conjugate()
            } else {
                a.view()
            };
            let result = SvdValues::compute(view).unwrap_or_else(|error| panic!("{error}"));
            for (&actual, &reference) in result.singular_values().iter().zip(&expected) {
                let allowance = bound * reference + S::Real::MIN_SUBNORMAL;
                assert!(
                    (actual - reference).abs() <= allowance,
                    "case {} conjugate={conjugate}: {actual:?} != {reference:?}, allowance {allowance:?}",
                    case.name
                );
                if reference > S::Real::ZERO {
                    assert!(actual > S::Real::ZERO);
                }
            }
            assert!(result.singular_values().windows(2).all(|v| v[0] >= v[1]));
            assert!(
                result
                    .singular_values()
                    .iter()
                    .all(|value| !value.into_f64().is_sign_negative())
            );
        }
    }
}

#[test]
fn graded_bidiagonals_preserve_relative_accuracy_for_real_and_complex_values() {
    graded_bidiagonal_values::<f32>();
    graded_bidiagonal_values::<f64>();
    graded_bidiagonal_values::<Complex<f32>>();
    graded_bidiagonal_values::<Complex<f64>>();
}

/// Vector jobs have a normwise backward-error contract. Weyl's inequality
/// gives absolute value error bounded by that perturbation, even for zeros
/// and clusters; the relative-accuracy dqds test above remains separate.
fn reference_vectors<S: NumericScalar>(sweep: bool) {
    let real = |x| S::Real::try_from_f64_lossy(x).unwrap_or_else(|error| panic!("{error}"));
    for case in bidiagonal::cases::<S::Real>() {
        let n = case.diagonal.len();
        if sweep && !(case.name == "zero_leaf_40" || case.name == "cluster_boundary_65") {
            continue;
        }
        let sizes = if sweep {
            vec![2 * n - 1, 2 * n, 2 * n + 1, 4 * n]
        } else {
            vec![n]
        };
        for rows in sizes {
            let mut a = Matrix::<S>::zeros(rows, n).unwrap_or_else(|error| panic!("{error}"));
            for i in 0..n {
                a.as_mut_slice()[i * n + i] = S::from_real(case.diagonal[i]);
                if i + 1 < n {
                    let phase = if !S::IS_REAL && i % 2 == 0 {
                        S::from_components(S::Real::ZERO, S::Real::ONE)
                    } else {
                        S::ONE
                    };
                    a.as_mut_slice()[i * n + i + 1] = phase * S::from_real(case.off_diagonal[i]);
                }
            }
            if sweep {
                // Mix zero-padded rows by orthogonal rotations. The added
                // destination rounding is covered by the same normwise bound.
                for i in 0..rows - n {
                    let top = i % n;
                    for j in 0..n {
                        let value = a.as_slice()[top * n + j];
                        a.as_mut_slice()[(n + i) * n + j] = value * S::from_real(real(0.8));
                        a.as_mut_slice()[top * n + j] = value * S::from_real(real(0.6));
                    }
                }
            }
            for wide in [false, true] {
                let a = if wide {
                    a.view()
                        .adjoint()
                        .try_to_matrix()
                        .unwrap_or_else(|e| panic!("{e}"))
                } else {
                    a.clone()
                };
                let dimension = a.rows().max(a.cols());
                let nu = real(64.0 * dimension as f64) * S::Real::EPSILON * S::Real::HALF;
                let bound = nu / (S::Real::ONE - nu);
                let norm = decomposition_metrics::matrix_norm(&a);
                let thin = ThinSvdDecomposition::compute(&a).unwrap_or_else(|e| panic!("{e}"));
                let values = SvdValues::compute(&a).unwrap_or_else(|e| panic!("{e}"));
                for spectrum in [thin.singular_values(), values.singular_values()] {
                    for (&actual, &reference) in spectrum.iter().zip(&case.expected) {
                        assert!(
                            (actual - reference).abs() <= bound * norm + S::Real::MIN_SUBNORMAL,
                            "{} {dimension} wide={wide}: {actual:?} != {reference:?}",
                            case.name
                        );
                    }
                }
                let residual = decomposition_metrics::svd_residuals(
                    &a,
                    thin.u_matrix(),
                    thin.singular_values(),
                    thin.vt_matrix(),
                );
                assert!(residual.maximum() <= bound, "{}: {residual:?}", case.name);
                let triplets = decomposition_metrics::singular_triplet_residual(
                    &a,
                    thin.u_matrix(),
                    thin.singular_values(),
                    thin.vt_matrix(),
                );
                assert!(triplets <= bound, "{} triplets: {triplets:?}", case.name);
                if !sweep {
                    let full = scioxide_linalg::SvdDecomposition::compute(&a)
                        .unwrap_or_else(|e| panic!("{e}"));
                    for (&actual, &reference) in full.singular_values().iter().zip(&case.expected) {
                        assert!(
                            (actual - reference).abs() <= bound * norm + S::Real::MIN_SUBNORMAL
                        );
                    }
                    let residual = decomposition_metrics::svd_residuals(
                        &a,
                        full.u_matrix(),
                        full.singular_values(),
                        full.vt_matrix(),
                    );
                    assert!(residual.maximum() <= bound, "{}: {residual:?}", case.name);
                }
            }
        }
    }
}

#[test]
fn independent_bidiagonal_vector_references() {
    reference_vectors::<f32>(false);
    reference_vectors::<f64>(false);
    reference_vectors::<Complex<f32>>(false);
    reference_vectors::<Complex<f64>>(false);
}

#[test]
fn independent_tall_wide_references_across_qr_crossover() {
    reference_vectors::<f32>(true);
    reference_vectors::<f64>(true);
    reference_vectors::<Complex<f32>>(true);
    reference_vectors::<Complex<f64>>(true);
}

#[test]
fn zero_order_jobs_do_not_walk_the_other_extent() -> Result<(), scioxide_core::SciOxideError> {
    for (rows, cols) in [(usize::MAX, 0), (0, usize::MAX)] {
        let view =
            scioxide_linalg::MatrixView::<f64>::from_slice_with_strides(rows, cols, &[], 0, 0)?;
        let values = SvdValues::compute(view)?;
        let thin = ThinSvdDecomposition::compute(view)?;
        assert_eq!((values.rows(), values.cols()), (rows, cols));
        assert_eq!((thin.rows(), thin.cols()), (rows, cols));
        assert!(values.singular_values().is_empty());
        assert!(thin.singular_values().is_empty());
    }
    Ok(())
}

#[test]
fn values_preserve_connected_and_isolated_values_below_the_squaring_range() {
    fn check<R: RealScalar>() {
        let tiny = R::MIN_POSITIVE;
        let phi = (R::ONE + (R::TWO * R::TWO + R::ONE).sqrt()) / R::TWO;
        let a = Matrix::from_vec(
            3,
            3,
            vec![
                R::ONE,
                tiny,
                R::ZERO,
                R::ZERO,
                tiny,
                tiny,
                R::ZERO,
                R::ZERO,
                tiny,
            ],
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let result = SvdValues::compute(&a).unwrap_or_else(|error| panic!("{error}"));
        for (&actual, reference) in
            result
                .singular_values()
                .iter()
                .zip([R::ONE, phi * tiny, tiny / phi])
        {
            let allowance = R::TEN * R::EPSILON * reference + R::MIN_SUBNORMAL;
            assert!(
                (actual - reference).abs() <= allowance,
                "{actual:?} != {reference:?}"
            );
        }
        let a = Matrix::from_vec(
            3,
            3,
            vec![
                R::HALF,
                R::ZERO,
                R::ZERO,
                R::ZERO,
                R::MIN_SUBNORMAL,
                R::ZERO,
                R::ZERO,
                R::ZERO,
                tiny,
            ],
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let result = SvdValues::compute(&a).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(result.singular_values(), &[R::HALF, tiny, R::MIN_SUBNORMAL]);
        // Every squared input fits the dqds frame, but the final qd product
        // does not. The large values round to one; the determinant gives the
        // small value t^3 up to an O(t^2) relative correction below one ULP.
        let t = R::ONE.scalbn(if R::MANTISSA_DIGITS == 24 { -45 } else { -350 });
        let a = Matrix::from_vec(
            3,
            3,
            vec![t, R::ONE, R::ZERO, R::ZERO, t, R::ONE, R::ZERO, R::ZERO, t],
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let result = SvdValues::compute(&a).unwrap_or_else(|error| panic!("{error}"));
        assert!(result.singular_values()[2] > R::ZERO);
        for (&actual, reference) in result
            .singular_values()
            .iter()
            .zip([R::ONE, R::ONE, t * t * t])
        {
            assert!(
                (actual - reference).abs() <= R::TEN * R::EPSILON * reference + R::MIN_SUBNORMAL,
                "{actual:?} != {reference:?}"
            );
        }
        let mut output = result.clone();
        let Err(error) = SvdValues::compute_into(
            &a,
            &LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
            &mut output,
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        ) else {
            panic!("range recovery must share the dqds budget");
        };
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!(output.singular_values(), result.singular_values());
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn tiny_cluster_recovery_preserves_outputs_and_iteration_limits()
-> Result<(), scioxide_core::SciOxideError> {
    fn check<R: RealScalar>(tiny: R) -> Result<(), scioxide_core::SciOxideError> {
        let coupling = R::try_from_f64_lossy(1e-3)?;
        let a = Matrix::from_vec(
            4,
            4,
            vec![
                R::ONE,
                coupling,
                R::ZERO,
                R::ZERO,
                R::ZERO,
                R::ONE,
                coupling,
                R::ZERO,
                R::ZERO,
                R::ZERO,
                R::ONE,
                coupling,
                R::ZERO,
                R::ZERO,
                R::ZERO,
                tiny,
            ],
        )?;
        let options = LinalgExecutionOptions::new();
        let mut output = SvdValues::compute(&a)?;
        let saved = output.singular_values().to_vec();
        let mut workspace = LinalgWorkspace::new();
        let error = SvdValues::compute_into(
            &a,
            &options.with_iteration_budget(Budget::ONE),
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .err()
        .unwrap_or_else(|| panic!("a one-transform budget must be exhausted"));
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!(output.singular_values(), saved);
        SvdValues::compute_into(
            &a,
            &options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        assert_eq!(output.singular_values(), saved);
        Ok(())
    }
    check(1e-30_f32)?;
    check(1e-300_f64)
}

#[test]
fn typed_jobs_accept_strided_conjugated_input_without_reading_padding()
-> Result<(), scioxide_core::SciOxideError> {
    use scioxide_linalg::{MatrixView, SvdDecomposition};
    let mut storage = [Complex::new(f64::NAN, f64::NAN); 10];
    storage[0] = Complex::new(3.0, 4.0);
    storage[2] = Complex::new(0.0, 0.0);
    storage[5] = Complex::new(0.0, 0.0);
    storage[7] = Complex::new(0.0, 2.0);
    let view = MatrixView::from_slice_with_strides(2, 2, &storage, 5, 2)?.conjugate();
    let values = SvdValues::compute(view)?;
    let thin = ThinSvdDecomposition::compute(view)?;
    let full = SvdDecomposition::compute(view)?;
    for spectrum in [
        values.singular_values(),
        thin.singular_values(),
        full.singular_values(),
    ] {
        for (&actual, expected) in spectrum.iter().zip([5.0, 2.0]) {
            assert!((actual - expected).abs() < 1e-13);
        }
    }
    let matrix = view.try_to_matrix()?;
    for (u, spectrum, vt) in [
        (thin.u_matrix(), thin.singular_values(), thin.vt_matrix()),
        (full.u_matrix(), full.singular_values(), full.vt_matrix()),
    ] {
        assert!(decomposition_metrics::svd_residuals(&matrix, u, spectrum, vt).maximum() < 1e-13);
    }
    Ok(())
}

#[test]
fn svd_scope_charges_sweeps_only_after_preflight() -> Result<(), scioxide_core::SciOxideError> {
    use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let valid = Matrix::from_vec(3, 3, vec![4.0_f64, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0])?;
    let invalid = Matrix::from_vec(1, 1, vec![f64::NAN])?;
    let options = LinalgExecutionOptions::new();
    let mut output = SvdValues::compute(&valid)?;
    let expected = output.singular_values().to_vec();
    let mut workspace = LinalgWorkspace::new();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "svd_parent", 1),
        0,
        0,
    );
    let mut parent = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    let counts = parent.counts();
    assert!(
        SvdValues::compute_into(&invalid, &options, &mut output, &mut workspace, &mut parent)
            .is_err()
    );
    assert_eq!(parent.counts(), counts);
    assert_eq!(output.singular_values(), expected);
    SvdValues::compute_into(&valid, &options, &mut output, &mut workspace, &mut parent)?;
    assert_eq!(parent.counts().own().iterations(), 0);
    assert!(parent.counts().nested().iterations() > 0);
    for (&actual, expected) in output.singular_values().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-13);
    }
    Ok(())
}

fn assert_typed_fixture<S: NumericScalar>(matrix: Matrix<S>, tolerance: f64) {
    let order = matrix.rows().min(matrix.cols());
    let values =
        scioxide_linalg::SvdValues::compute(&matrix).unwrap_or_else(|error| panic!("{error}"));
    let thin = scioxide_linalg::ThinSvdDecomposition::compute(&matrix)
        .unwrap_or_else(|error| panic!("{error}"));
    let full = scioxide_linalg::SvdDecomposition::compute(&matrix)
        .unwrap_or_else(|error| panic!("{error}"));

    for spectrum in [
        values.singular_values(),
        thin.singular_values(),
        full.singular_values(),
    ] {
        assert!(spectrum.iter().all(|&value| value >= S::Real::ZERO));
        assert!(
            spectrum
                .iter()
                .all(|value| !value.into_f64().is_sign_negative())
        );
        assert!(spectrum.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    assert_eq!(values.singular_values().len(), order);
    assert_eq!(
        (thin.u_matrix().rows(), thin.u_matrix().cols()),
        (matrix.rows(), order)
    );
    assert_eq!(
        (thin.vt_matrix().rows(), thin.vt_matrix().cols()),
        (order, matrix.cols())
    );
    assert_eq!(
        (full.u_matrix().rows(), full.u_matrix().cols()),
        (matrix.rows(), matrix.rows())
    );
    assert_eq!(
        (full.vt_matrix().rows(), full.vt_matrix().cols()),
        (matrix.cols(), matrix.cols())
    );
    // Different bidiagonal algorithms round independently. Compare in the
    // spectrum's frame, including the extreme input scales below.
    let scale = values
        .singular_values()
        .first()
        .copied()
        .filter(|&x| x > S::Real::ZERO)
        .unwrap_or(S::Real::ONE);
    values
        .singular_values()
        .iter()
        .zip(thin.singular_values())
        .zip(full.singular_values())
        .for_each(|((&value, &thin_value), &full_value)| {
            let value_error = (value / scale - thin_value / scale)
                .abs()
                .try_to_f64_exact()
                .unwrap_or(f64::INFINITY);
            let full_error = (value / scale - full_value / scale)
                .abs()
                .try_to_f64_exact()
                .unwrap_or(f64::INFINITY);
            assert!(
                value_error <= tolerance,
                "values={value:?}, thin={thin_value:?}, error={value_error}, tolerance={tolerance}"
            );
            assert!(full_error <= tolerance);
        });

    let thin_residuals = decomposition_metrics::svd_residuals(
        &matrix,
        thin.u_matrix(),
        thin.singular_values(),
        thin.vt_matrix(),
    );
    assert!(
        decomposition_metrics::as_f64(thin_residuals.maximum()) <= tolerance,
        "{}x{} thin residuals = {thin_residuals:?}",
        matrix.rows(),
        matrix.cols()
    );
    let full_residuals = decomposition_metrics::svd_residuals(
        &matrix,
        full.u_matrix(),
        full.singular_values(),
        full.vt_matrix(),
    );
    assert!(
        decomposition_metrics::as_f64(full_residuals.maximum()) <= tolerance,
        "{}x{} full residuals = {full_residuals:?}",
        matrix.rows(),
        matrix.cols()
    );
}

/// Bidiagonal with `dᵢ = g^(i−n+1)` and `eᵢ = g^(i−n+3/2)` above (or, if
/// `lower`, below) the diagonal: the entries grow from the top to the bottom.
fn upward_graded_bidiagonal<S: NumericScalar>(
    grading: f64,
    order: usize,
    lower: bool,
) -> Matrix<S> {
    let power = |exponent: f64| {
        S::from_real(
            S::Real::try_from_f64_lossy(grading.powf(exponent))
                .unwrap_or_else(|error| panic!("{error}")),
        )
    };
    let last = order as f64 - 1.0;
    Matrix::from_vec(
        order,
        order,
        (0..order * order)
            .map(|index| {
                let (row, column) = (index / order, index % order);
                let (row, column) = if lower { (column, row) } else { (row, column) };
                if column == row {
                    power(row as f64 - last)
                } else if column == row + 1 {
                    power(row as f64 - last + 0.5)
                } else {
                    S::ZERO
                }
            })
            .collect(),
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

#[test]
fn upward_graded_bidiagonals_converge_with_vectors() {
    // The bulge is chased from the large end, in the QR solver and in the
    // QR leaves of divide and conquer (order above 64).
    for (grading, order) in [(1e4, 29), (1e3, 38), (1e2, 56), (1e3, 80), (10.0, 200)] {
        let bound = decomposition_metrics::invariant_bound::<f64>(order).into_f64();
        for lower in [false, true] {
            assert_typed_fixture(
                upward_graded_bidiagonal::<f64>(grading, order, lower),
                bound,
            );
        }
    }
    let bound = decomposition_metrics::invariant_bound::<f32>(18).into_f64();
    assert_typed_fixture(upward_graded_bidiagonal::<f32>(1e2, 18, false), bound);
    assert_typed_fixture(
        upward_graded_bidiagonal::<Complex<f64>>(1e4, 29, false),
        decomposition_metrics::invariant_bound::<f64>(29).into_f64(),
    );
}

#[test]
fn zero_diagonals_converge_in_every_vector_block_position() {
    fn check<S: NumericScalar>() {
        svd_cases::zero_diagonals::<S>(|_, matrix| {
            let bound =
                decomposition_metrics::invariant_bound::<S::Real>(matrix.rows().max(matrix.cols()));
            assert_typed_fixture(matrix, bound.into_f64());
        });
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
    for (rows, cols, values) in [
        (2, 2, vec![0.0, 1.0, 0.0, 1.0]),
        (3, 3, vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0]),
        (3, 2, vec![0.0, 1.0, 0.0, 3.0, 0.0, 5.0]),
        (2, 3, vec![0.0, 0.0, 0.0, 1.0, 3.0, 5.0]),
    ] {
        assert_typed_fixture(
            Matrix::<f64>::from_vec(rows, cols, values).unwrap_or_else(|error| panic!("{error}")),
            1e-12,
        );
    }
}

#[test]
fn singular_zero_sign_and_normalized_subnormal_floor() -> Result<(), scioxide_core::SciOxideError> {
    for value in [0.0_f32, -0.0] {
        assert_typed_fixture(Matrix::from_vec(2, 2, vec![value; 4])?, 1e-5);
    }
    let scale = 2.0_f32.powi(60);
    for (small, expected) in [
        (2.0_f32.powi(-88), 2.0_f32.powi(-88)),
        (2.0_f32.powi(-90), 0.0),
    ] {
        let matrix = Matrix::from_vec(2, 2, vec![scale, 0.0, 0.0, small])?;
        let result = SvdValues::compute(&matrix)?;
        // Both inputs are normal, but packing divides by 2^61. Its
        // subnormal quantum restores to 2^61 * 2^-149 = 2^-88.
        assert_eq!(result.singular_values(), &[scale, expected]);
    }
    Ok(())
}

#[test]
fn typed_jobs_cover_real_and_complex_tall_and_wide_inputs() {
    assert_typed_fixture(
        Matrix::from_vec(
            5,
            3,
            vec![
                1.0_f64, 2.0, 3.0, 4.0, -2.0, 1.0, 3.0, 5.0, 2.0, -1.0, 2.0, 4.0, 0.5, -3.0, 2.0,
            ],
        )
        .unwrap_or_else(|error| panic!("{error}")),
        5.0e-11,
    );
    assert_typed_fixture(
        Matrix::from_vec(
            3,
            5,
            (0..15)
                .map(|index| (index % 7) as f32 / 3.0 - 1.0)
                .collect(),
        )
        .unwrap_or_else(|error| panic!("{error}")),
        5.0e-4,
    );
    assert_typed_fixture(
        Matrix::from_vec(
            4,
            3,
            (0..12)
                .map(|index| Complex::new((index % 5) as f64 - 2.0, (index % 3) as f64 - 1.0))
                .collect(),
        )
        .unwrap_or_else(|error| panic!("{error}")),
        1.0e-10,
    );
    assert_typed_fixture(
        Matrix::from_vec(
            3,
            4,
            (0..12)
                .map(|index| {
                    Complex::new(
                        (index % 5) as f32 / 2.0 - 1.0,
                        (index % 4) as f32 / 3.0 - 0.5,
                    )
                })
                .collect(),
        )
        .unwrap_or_else(|error| panic!("{error}")),
        8.0e-4,
    );
}

#[test]
fn zero_singleton_rank_deficient_and_extreme_scales_are_supported() {
    for matrix in [
        Matrix::<f64>::zeros(0, 3).unwrap_or_else(|error| panic!("{error}")),
        Matrix::<f64>::zeros(3, 0).unwrap_or_else(|error| panic!("{error}")),
    ] {
        assert_typed_fixture(matrix, 1.0e-12);
    }
    assert_typed_fixture(
        Matrix::from_vec(1, 1, vec![-7.0_f64]).unwrap_or_else(|error| panic!("{error}")),
        1.0e-12,
    );
    assert_typed_fixture(
        Matrix::from_vec(
            4,
            3,
            vec![
                1.0_f64, 2.0, 3.0, 2.0, 4.0, 6.0, -1.0, -2.0, -3.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap_or_else(|error| panic!("{error}")),
        1.0e-10,
    );
    for scale in [1.0e-200_f64, 1.0e200_f64] {
        assert_typed_fixture(
            Matrix::from_vec(
                3,
                2,
                vec![
                    scale,
                    0.5 * scale,
                    -0.25 * scale,
                    scale,
                    0.75 * scale,
                    -scale,
                ],
            )
            .unwrap_or_else(|error| panic!("{error}")),
            1.0e-10,
        );
    }
    for scale in [1.0e-30_f32, 1.0e30_f32] {
        assert_typed_fixture(
            Matrix::from_vec(2, 2, vec![scale, 0.25 * scale, -0.5 * scale, scale])
                .unwrap_or_else(|error| panic!("{error}")),
            5.0e-4,
        );
    }
}

#[test]
fn repeated_singular_values_are_checked_as_an_invariant_subspace() {
    let matrix = Matrix::from_vec(3, 3, vec![3.0_f64, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        .unwrap_or_else(|error| panic!("{error}"));
    let decomposition = scioxide_linalg::ThinSvdDecomposition::compute(&matrix)
        .unwrap_or_else(|error| panic!("{error}"));
    let u = decomposition.u_matrix();
    for row in 0..3 {
        for column in 0..3 {
            let projector = (0..2).fold(0.0, |sum, index| {
                sum + u.as_slice()[row * u.cols() + index] * u.as_slice()[column * u.cols() + index]
            });
            let expected = if row == column && row < 2 { 1.0 } else { 0.0 };
            assert!((projector - expected).abs() < 1.0e-12);
        }
    }
}

#[test]
fn simple_singular_vectors_have_deterministic_phase() {
    let matrix = Matrix::from_vec(
        3,
        2,
        vec![
            Complex::new(2.0_f64, 1.0),
            Complex::new(-0.5, 0.25),
            Complex::new(1.0, -2.0),
            Complex::new(3.0, 0.5),
            Complex::new(-1.5, 1.0),
            Complex::new(0.75, -0.25),
        ],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let first = scioxide_linalg::ThinSvdDecomposition::compute(&matrix)
        .unwrap_or_else(|error| panic!("{error}"));
    let second = scioxide_linalg::ThinSvdDecomposition::compute(&matrix)
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(first.singular_values(), second.singular_values());
    assert_eq!(first.u_matrix().as_slice(), second.u_matrix().as_slice());
    assert_eq!(first.vt_matrix().as_slice(), second.vt_matrix().as_slice());

    let column_vector = Matrix::from_vec(
        2,
        1,
        vec![Complex::new(0.3_f64, 0.7), Complex::new(0.1, -0.2)],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let third = scioxide_linalg::ThinSvdDecomposition::compute(&column_vector)
        .unwrap_or_else(|error| panic!("{error}"));
    for svd in [&first, &third] {
        let u = svd.u_matrix();
        for column in 0..u.cols() {
            // The first entry of largest magnitude is exactly real and positive.
            let magnitude = |row: usize| u.as_slice()[row * u.cols() + column].norm();
            let largest = (0..u.rows()).map(magnitude).fold(0.0, f64::max);
            let pivot = (0..u.rows())
                .find(|&row| magnitude(row) == largest)
                .unwrap_or(0);
            let value = u.as_slice()[pivot * u.cols() + column];
            assert_eq!(value.im, 0.0);
            assert!(value.re > 0.0);
        }
        let original = if u.rows() == 2 {
            &column_vector
        } else {
            &matrix
        };
        let vt = svd.vt_matrix();
        for (index, &entry) in original.as_slice().iter().enumerate() {
            let (row, column) = (index / vt.cols(), index % vt.cols());
            let product: Complex<f64> = (0..u.cols())
                .map(|k| {
                    u.as_slice()[row * u.cols() + k]
                        * svd.singular_values()[k]
                        * vt.as_slice()[k * vt.cols() + column]
                })
                .sum();
            assert!((product - entry).norm() <= 1.0e-14);
        }
    }
}

#[test]
fn preflight_and_budget_failures_preserve_caller_outputs() {
    let valid = Matrix::from_vec(
        4,
        4,
        vec![
            4.0_f64, 1.0, 2.0, 0.0, 1.0, 3.0, 0.5, 1.0, 2.0, 0.5, 5.0, -1.0, 0.0, 1.0, -1.0, 2.0,
        ],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let options = LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
    let mut output = ThinSvdDecomposition::empty();
    let mut workspace = LinalgWorkspace::new();
    scioxide_linalg::ThinSvdDecomposition::compute_into(
        &valid,
        &options,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let prior_u = output.u_matrix().as_slice().to_vec();
    let prior_values = output.singular_values().to_vec();
    let prior_vt = output.vt_matrix().as_slice().to_vec();

    let non_finite = Matrix::from_vec(2, 2, vec![1.0, f64::NAN, 0.0, 1.0])
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &non_finite,
            &options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output.u_matrix().as_slice(), prior_u);
    assert_eq!(output.singular_values(), prior_values);
    assert_eq!(output.vt_matrix().as_slice(), prior_vt);

    let constrained = options.with_iteration_budget(Budget::ONE);
    assert!(
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &valid,
            &constrained,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output.u_matrix().as_slice(), prior_u);
    assert_eq!(output.singular_values(), prior_values);
    assert_eq!(output.vt_matrix().as_slice(), prior_vt);

    let mut values = SvdValues::empty();
    scioxide_linalg::SvdValues::compute_into(
        &valid,
        &options,
        &mut values,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let prior_values = values.singular_values().to_vec();
    assert!(
        scioxide_linalg::SvdValues::compute_into(
            &non_finite,
            &options,
            &mut values,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(values.singular_values(), prior_values);

    let mut full = scioxide_linalg::SvdDecomposition::compute(&valid)
        .unwrap_or_else(|error| panic!("{error}"));
    let prior_u = full.u_matrix().as_slice().to_vec();
    let prior_values = full.singular_values().to_vec();
    let prior_vt = full.vt_matrix().as_slice().to_vec();
    assert!(
        scioxide_linalg::SvdDecomposition::compute_into(
            &non_finite,
            &options,
            &mut full,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(full.u_matrix().as_slice(), prior_u);
    assert_eq!(full.singular_values(), prior_values);
    assert_eq!(full.vt_matrix().as_slice(), prior_vt);
}

#[test]
fn damped_filter_combines_exponents_before_restoring() {
    use scioxide_linalg::{RankPolicy, Vector};
    for (sigma, rhs, damping, expected) in [
        (1e-150, 1e-150, 1e-300, 0.5),
        (1e150, 1e150, 1e300, 0.5),
        (1e-300, 1e300, 1e300, 1e-300),
    ] {
        let a = Matrix::from_vec(1, 1, vec![sigma]).unwrap();
        let factor = scioxide_linalg::ThinSvdDecomposition::compute(&a).unwrap();
        let mut output = Vector::from_vec(vec![7.0]);
        let mut scratch = scioxide_linalg::LinalgWorkspace::new();
        factor
            .solve_into(
                &Vector::from_vec(vec![rhs]),
                RankPolicy::DimensionScaledEpsilon,
                damping,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                output.view_mut(),
                &mut scratch,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!((output.as_slice()[0] / expected - 1.0).abs() < 1e-12);
    }
}

#[allow(clippy::unwrap_used)]
fn dense_spectrum<S: NumericScalar>(rows: usize, cols: usize, values: &[f64]) -> Matrix<S> {
    let cosine = |row: usize, mode: usize, n: usize| {
        let scale = if mode == 0 { 1.0 } else { 2.0 };
        (scale / n as f64).sqrt()
            * (core::f64::consts::PI * (row as f64 + 0.5) * mode as f64 / n as f64).cos()
    };
    Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|i| {
                let row = i / cols;
                let col = i % cols;
                let value: f64 = values
                    .iter()
                    .enumerate()
                    .map(|(k, &s)| cosine(row, k, rows) * s * cosine(col, k, cols))
                    .sum();
                let angle = if core::mem::size_of::<S>() > core::mem::size_of::<S::Real>() {
                    (row * 3 + col * 7) as f64 * 0.17
                } else {
                    0.0
                };
                S::from_components(
                    S::Real::try_from_f64_lossy(value * angle.cos()).unwrap(),
                    S::Real::try_from_f64_lossy(value * angle.sin()).unwrap(),
                )
            })
            .collect(),
    )
    .unwrap()
}

#[allow(clippy::unwrap_used)]
fn large_spectra<S: NumericScalar>() {
    for (rows, cols) in [(39, 33), (33, 39), (65, 65)] {
        let n = rows.min(cols);
        let tolerance = 512.0 * rows.max(cols) as f64 * S::Real::EPSILON.into_f64();
        for fixture in 0..3 {
            let values: Vec<_> = (0..n)
                .map(|i| match fixture {
                    0 => 1.0 + (i % 3) as f64 * 32.0 * S::Real::EPSILON.into_f64(),
                    1 => 2.0_f64.powi(-(i as i32)),
                    _ if i >= n / 2 => 0.0,
                    _ => 1.0 + i as f64 / n as f64,
                })
                .collect();
            let matrix = dense_spectrum::<S>(rows, cols, &values);
            assert_typed_fixture(matrix.clone(), tolerance);
            let actual = scioxide_linalg::SvdValues::compute(&matrix).unwrap();
            let mut expected = values;
            expected.sort_by(|a, b| b.total_cmp(a));
            for (&a, b) in actual.singular_values().iter().zip(expected) {
                assert!((a.into_f64() - b).abs() <= tolerance);
            }
        }
    }
}

#[test]
fn large_dense_clustered_graded_and_rank_deficient_spectra() {
    large_spectra::<f32>();
    large_spectra::<f64>();
    large_spectra::<Complex<f32>>();
    large_spectra::<Complex<f64>>();
}

#[test]
fn large_vector_jobs_preserve_output_after_failure_and_retry() {
    use scioxide_core::{CancellationToken, ExecutionContext};

    let a = dense_spectrum::<f64>(
        65,
        65,
        &(0..65).map(|i| 1.0 + i as f64 / 65.0).collect::<Vec<_>>(),
    );
    let options = LinalgExecutionOptions::new();
    let mut workspace = LinalgWorkspace::new();
    let mut thin = scioxide_linalg::ThinSvdDecomposition::compute(&a).unwrap();
    let original = thin.clone();
    assert!(
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &a,
            &(options.with_iteration_budget(Budget::ONE)),
            &mut thin,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &a,
            &(options),
            &mut thin,
            &mut workspace,
            ExecutionContext::sequential_native().with_cancellation(&token)
        )
        .is_err()
    );
    let overflow = Matrix::from_vec(33, 33, vec![f64::MAX; 33 * 33]).unwrap();
    // The largest singular value 33 MAX exceeds the range: overflow, not NaN.
    assert_eq!(
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &overflow,
            &(options),
            &mut thin,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(thin.u_matrix(), original.u_matrix());
    assert_eq!(thin.vt_matrix(), original.vt_matrix());
    assert_eq!(thin.singular_values(), original.singular_values());
    scioxide_linalg::ThinSvdDecomposition::compute_into(
        &a,
        &(options),
        &mut thin,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(thin.u_matrix(), original.u_matrix());
    let mut full = scioxide_linalg::SvdDecomposition::compute(&a).unwrap();
    let old_u = full.u_matrix().clone();
    let old_vt = full.vt_matrix().clone();
    let old_values = full.singular_values().to_vec();
    assert!(
        scioxide_linalg::SvdDecomposition::compute_into(
            &a,
            &(options.with_iteration_budget(Budget::ONE)),
            &mut full,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert_eq!(full.u_matrix(), &old_u);
    assert_eq!(full.vt_matrix(), &old_vt);
    assert_eq!(full.singular_values(), old_values);
    scioxide_linalg::SvdDecomposition::compute_into(
        &a,
        &(options),
        &mut full,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
}

#[cfg(feature = "std")]
#[test]
fn values_agree_across_scalar_and_native_panels() {
    fn check<S: NumericScalar>() {
        let processor = scioxide_core::StdProcessor;
        for n in [31, 33, 65] {
            let a = dense_spectrum::<S>(
                n + 3,
                n,
                &(0..n)
                    .map(|i| 1.0 + i as f64 / n as f64)
                    .collect::<Vec<_>>(),
            );
            let options = LinalgExecutionOptions::new();
            let scalar = SvdValues::compute(&a).unwrap_or_else(|error| panic!("{error}"));
            let mut native = SvdValues::empty();
            SvdValues::compute_into(
                &a,
                &options,
                &mut native,
                &mut LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential().with_processor(&processor),
            )
            .unwrap_or_else(|error| panic!("{error}"));
            let bound = decomposition_metrics::invariant_bound::<S::Real>(n);
            for (&a, &b) in scalar
                .singular_values()
                .iter()
                .zip(native.singular_values())
            {
                assert!((a - b).abs() <= bound * a, "{a:?} != {b:?}");
            }
        }
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}

#[cfg(feature = "std")]
#[test]
fn scalar_and_native_backends_agree_across_the_crossover() {
    use scioxide_core::{ExecutionContext, StdProcessor};

    let processor = StdProcessor;
    let native = ExecutionContext::sequential().with_processor(&processor);
    for n in [31, 32, 33, 65] {
        let a = dense_spectrum::<Complex<f64>>(
            n + 3,
            n,
            &(0..n)
                .map(|i| 1.0 + i as f64 / n as f64)
                .collect::<Vec<_>>(),
        );
        let scalar = scioxide_linalg::ThinSvdDecomposition::compute(&a).unwrap();
        let mut detected = ThinSvdDecomposition::empty();
        scioxide_linalg::ThinSvdDecomposition::compute_into(
            &a,
            &(LinalgExecutionOptions::new()),
            &mut detected,
            &mut LinalgWorkspace::new(),
            native,
        )
        .unwrap();
        let tolerance = decomposition_metrics::invariant_bound::<f64>(n + 3);
        let residual = decomposition_metrics::svd_residuals(
            &a,
            detected.u_matrix(),
            detected.singular_values(),
            detected.vt_matrix(),
        );
        assert!(residual.maximum() <= tolerance);
        assert!(
            decomposition_metrics::real_spectrum_difference(
                scalar.singular_values(),
                detected.singular_values()
            ) <= tolerance
        );
    }
}

#[test]
fn large_diagonal_preserves_isolated_tiny_singular_values() {
    let n = 65;
    let mut values: Vec<_> = (0..n).map(|i| 0.25_f64.scalbn(-(i as i32) * 15)).collect();
    values[n / 2] = f64::from_bits(1);
    let a = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| if i / n == i % n { values[i / n] } else { 0.0 })
            .collect(),
    )
    .unwrap();
    values.sort_by(|a, b| b.total_cmp(a));
    let thin = scioxide_linalg::ThinSvdDecomposition::compute(&a).unwrap();
    for (&actual, &expected) in thin.singular_values().iter().zip(&values) {
        assert!(actual > 0.0);
        assert!(
            (actual - expected).abs()
                <= (512.0 * n as f64 * f64::EPSILON * expected).max(f64::from_bits(1)),
            "{actual} != {expected}"
        );
    }
    let residuals = decomposition_metrics::svd_residuals(
        &a,
        thin.u_matrix(),
        thin.singular_values(),
        thin.vt_matrix(),
    );
    assert!(residuals.maximum() <= decomposition_metrics::invariant_bound(n));
}

#[test]
#[allow(clippy::unwrap_used)]
fn parallel_policy_carves_divide_and_conquer_within_its_reservation() {
    use scioxide_core::ExecutionContext;
    use scioxide_linalg::SvdDecomposition;
    // Order 72 runs divide and conquer with two packed GEMM row partitions; a
    // sequential context reserves one worker although the policy allows two.
    let n = 72;
    let values: Vec<_> = (0..n).map(|i| 1.0 + i as f64 / n as f64).collect();
    let a = dense_spectrum::<f64>(n, n, &values);
    let options = LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Parallel(
        core::num::NonZeroUsize::new(2).unwrap(),
    ));
    let tolerance = decomposition_metrics::invariant_bound::<f64>(n);
    let mut thin = ThinSvdDecomposition::empty();
    let mut full = SvdDecomposition::empty();
    let mut workspace = LinalgWorkspace::new();
    ThinSvdDecomposition::compute_into(
        &a,
        &options,
        &mut thin,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    SvdDecomposition::compute_into(
        &a,
        &options,
        &mut full,
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    for (u, spectrum, vt) in [
        (thin.u_matrix(), thin.singular_values(), thin.vt_matrix()),
        (full.u_matrix(), full.singular_values(), full.vt_matrix()),
    ] {
        assert!(decomposition_metrics::svd_residuals(&a, u, spectrum, vt).maximum() <= tolerance);
    }
}

#[test]
fn divide_and_conquer_leaves_share_the_iteration_budget() {
    use scioxide_linalg::{LinalgWorkspace, ThinSvdDecomposition};
    // Two dense 32 x 32 blocks split by a zero row and column give two
    // independent divide-and-conquer leaves.
    let n = 65;
    let mut a = Matrix::<f64>::zeros(n, n).unwrap();
    for block in [0, 33] {
        for i in 0..32 {
            for j in 0..32 {
                let value = ((7 * i + 3 * j + block) as f64).sin() + if i == j { 2.0 } else { 0.0 };
                a.view_mut().set(block + i, block + j, value).unwrap();
            }
        }
    }
    let mut workspace = LinalgWorkspace::new();
    let mut output = ThinSvdDecomposition::empty();
    ThinSvdDecomposition::compute_into(
        &a,
        &LinalgExecutionOptions::new(),
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let used = workspace
        .last_attempt()
        .unwrap()
        .counts()
        .total()
        .unwrap()
        .iterations();
    assert!(used > 2);
    // Every budget either stops the attempt or bounds all leaf and merge iterations.
    for budget in 1..=used {
        let options =
            LinalgExecutionOptions::new().with_iteration_budget(Budget::try_from(budget).unwrap());
        match ThinSvdDecomposition::compute_into(
            &a,
            &options,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        ) {
            Ok(()) => assert!(
                workspace
                    .last_attempt()
                    .unwrap()
                    .counts()
                    .own()
                    .iterations()
                    <= budget,
                "budget {budget}"
            ),
            Err(error) => assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::ConvergenceFailure
            ),
        }
    }
}
