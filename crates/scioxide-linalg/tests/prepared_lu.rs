//! Prepared dense backend: pivoting, range safety, capacity and allocation contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, ExecutionContext, NumericScalar, RealScalar};
use scioxide_linalg::{MatrixView, backend::lu::LuFactor};

fn solve_cases<T: NumericScalar>() {
    let real = |value| T::Real::try_from_f64_lossy(value).unwrap();
    for n in [0, 1, 2, 16, 65] {
        let mut matrix = vec![T::ZERO; n * n];
        let expected: Vec<_> = (0..n)
            .map(|i| T::from_components(real(1.0 + i as f64 / n as f64), real(0.25)))
            .collect();
        for row in 0..n {
            // Reverse the dominant diagonal to require row interchanges.
            matrix[row * n + n - 1 - row] = T::from_real(real(4.0));
            if row + 1 < n {
                matrix[row * n + n - 2 - row] = T::from_components(real(-0.5), real(0.25));
            }
        }
        let rhs: Vec<_> = matrix
            .chunks(n.max(1))
            .map(|row| {
                row.iter()
                    .zip(&expected)
                    .fold(T::ZERO, |sum, (&a, &x)| sum + a * x)
            })
            .collect();
        for execution in [
            ExecutionContext::sequential(),
            ExecutionContext::sequential_native(),
        ] {
            let mut factor = LuFactor::new(n).unwrap();
            let mut output = vec![T::ZERO; n];
            let matrix = MatrixView::from_slice(n, n, &matrix).unwrap();
            {
                factor.factor(matrix, execution).unwrap();
                factor.solve_into(&rhs, &mut output, execution).unwrap();
            };
            for (&actual, &expected) in output.iter().zip(&expected) {
                assert!((actual - expected).abs() < real(32.0) * T::Real::EPSILON);
            }
            let before = factor.retained_bytes().unwrap();
            factor.try_reserve(n + 1).unwrap();
            assert!(factor.retained_bytes().unwrap() >= before);
            factor.solve_into(&rhs, &mut output, execution).unwrap();
        }
    }
}

#[test]
fn prepared_factors_solve_real_and_complex_pivoted_systems_without_allocation() {
    solve_cases::<f32>();
    solve_cases::<f64>();
    solve_cases::<Complex<f32>>();
    solve_cases::<Complex<f64>>();
}

#[test]
fn invalid_inputs_and_failed_refactors_do_not_publish_solutions() {
    let execution = ExecutionContext::sequential();
    let mut factor = LuFactor::<f64>::new(2).unwrap();
    let mut output = [17.0; 2];
    assert!(
        factor
            .solve_into(&[1.0; 2], &mut output, execution)
            .is_err()
    );
    for values in [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, f64::NAN, 0.0, 1.0],
        [1.0, 0.0, f64::INFINITY, 1.0],
        [1.0, 0.0, f64::NEG_INFINITY, 1.0],
    ] {
        factor
            .factor(
                MatrixView::from_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]).unwrap(),
                execution,
            )
            .unwrap();
        assert!(
            factor
                .factor(MatrixView::from_slice(2, 2, &values).unwrap(), execution)
                .is_err()
        );
        assert!(
            factor
                .solve_into(&[1.0; 2], &mut output, execution)
                .is_err()
        );
        assert_eq!(output, [17.0; 2]);
    }
    assert!(
        factor
            .factor(
                MatrixView::from_slice(1, 2, &[1.0, 1.0]).unwrap(),
                execution
            )
            .is_err()
    );
    assert!(
        factor
            .factor(MatrixView::from_slice(3, 3, &[1.0; 9]).unwrap(), execution)
            .is_err()
    );
    factor
        .factor(
            MatrixView::from_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]).unwrap(),
            execution,
        )
        .unwrap();
    assert!(factor.solve_into(&[1.0], &mut output, execution).is_err());
    assert!(
        factor
            .solve_into(&[f64::NAN; 2], &mut output, execution)
            .is_err()
    );
    assert_eq!(output, [17.0; 2]);
}

#[test]
fn equilibration_handles_extreme_scales_and_overflow_is_transactional() {
    let execution = ExecutionContext::sequential();
    let mut factor = LuFactor::<f64>::new(2).unwrap();
    for scale in [1e-300, 1e300] {
        factor
            .factor(
                MatrixView::from_slice(2, 2, &[2.0 * scale, scale, scale, 2.0 * scale]).unwrap(),
                execution,
            )
            .unwrap();
        let mut output = [0.0; 2];
        factor
            .solve_into(&[3.0 * scale; 2], &mut output, execution)
            .unwrap();
        assert!(output.iter().all(|&x| (x - 1.0).abs() < 1e-14));
    }
    factor
        .factor(MatrixView::from_slice(1, 1, &[1e-300]).unwrap(), execution)
        .unwrap();
    let mut output = [17.0];
    assert!(factor.solve_into(&[1e300], &mut output, execution).is_err());
    assert_eq!(output, [17.0]);
}
