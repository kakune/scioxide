//! Real-coordinate affine Newton interpolation and prepared calculus.
#![allow(clippy::unwrap_used)]
mod support;

use support::assert_error;

use scioxide_core::{Complex, SciOxideError};
use scioxide_interp::{Differentiable, ExtrapolationPolicy, Interpolant, NewtonInterpolator};

#[test]
fn analytic_complex_polynomial_calculus_in_both_precisions() {
    macro_rules! check {
        ($s:ty, $tol:expr) => {{
            let polynomial = NewtonInterpolator::new(
                vec![-1.0 as $s, 0.0 as $s],
                vec![
                    Complex::new(1.0 as $s, -1.0 as $s),
                    Complex::new(0.0 as $s, 1.0 as $s),
                ],
            )
            .unwrap()
            .insert_point(1.0 as $s, Complex::new(1.0 as $s, 3.0 as $s))
            .unwrap();
            let value = polynomial
                .evaluate([0.5 as $s], ExtrapolationPolicy::Reject)
                .unwrap();
            assert!((value.re - 0.25 as $s).abs() <= $tol);
            assert!((value.im - 2.0 as $s).abs() <= $tol);
            let [slope] = polynomial
                .gradient([0.5 as $s], ExtrapolationPolicy::Reject)
                .unwrap();
            assert!((slope.re - 1.0 as $s).abs() <= $tol);
            assert!((slope.im - 2.0 as $s).abs() <= $tol);
            let integral = polynomial
                .prepare_antiderivative()
                .unwrap()
                .integral(-1.0 as $s, 1.0 as $s, ExtrapolationPolicy::Reject)
                .unwrap();
            assert!((integral.re - (2.0 as $s / 3.0 as $s)).abs() <= $tol);
            assert!((integral.im - 2.0 as $s).abs() <= $tol);
        }};
    }
    check!(f32, 2e-6);
    check!(f64, 2e-14);
}

#[test]
fn dynamic_and_borrowed_queries_have_no_hidden_allocations() {
    let polynomial = NewtonInterpolator::new(
        vec![-1.0_f64, 0.0, 1.0],
        vec![vec![1.0, -1.0], vec![0.0, 1.0], vec![1.0, 3.0]],
    )
    .unwrap();
    let prepared = polynomial.prepare_antiderivative().unwrap();
    let mut output = [0.0; 2];
    let mut scratch = [0.0; 2];
    let mut workspace = polynomial.workspace().unwrap();
    {
        polynomial
            .evaluate_into(
                [0.5],
                ExtrapolationPolicy::Reject,
                &mut output[..],
                &mut workspace,
            )
            .unwrap();
        assert_eq!(output, [0.25, 2.0]);
        polynomial
            .gradient_into(
                [0.5],
                ExtrapolationPolicy::Reject,
                &mut output[..],
                &mut workspace,
            )
            .unwrap();
        assert_eq!(output, [1.0, 2.0]);
        prepared
            .integral_into(
                -1.0,
                1.0,
                ExtrapolationPolicy::Reject,
                &mut output[..],
                &mut scratch,
            )
            .unwrap();
    };
    assert!((output[0] - 2.0 / 3.0).abs() < 2e-14);
    assert!((output[1] - 2.0).abs() < 2e-14);
}

#[test]
fn fixed_vectors_and_empty_vectors_remain_allocation_free() {
    let polynomial =
        NewtonInterpolator::new(vec![0.0_f32, 1.0], vec![[1.0, -1.0], [3.0, 1.0]]).unwrap();
    assert_eq!(
        polynomial
            .evaluate([0.5], ExtrapolationPolicy::Reject)
            .unwrap(),
        [2.0, 0.0]
    );
    let empty =
        NewtonInterpolator::new(vec![0.0_f64, 1.0], vec![Vec::<f64>::new(), Vec::new()]).unwrap();
    let prepared = empty.prepare_antiderivative().unwrap();
    let mut workspace = empty.workspace().unwrap();
    {
        empty
            .evaluate_into(
                [0.5],
                ExtrapolationPolicy::Reject,
                &mut [] as &mut [f64],
                &mut workspace,
            )
            .unwrap();
        prepared
            .integral_into(
                0.0,
                1.0,
                ExtrapolationPolicy::Reject,
                &mut [] as &mut [f64],
                &mut [],
            )
            .unwrap();
    };
}

#[test]
fn invalid_shapes_coordinates_and_point_insertion_are_transactional() {
    assert_error(
        &NewtonInterpolator::new(vec![0.0_f64, 1.0], vec![vec![1.0], vec![1.0, 2.0]]),
        scioxide_core::SciOxideErrorKind::DimensionMismatch,
    );
    let polynomial =
        NewtonInterpolator::new(vec![0.0_f64, 1.0], vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    assert!(polynomial.insert_point(2.0, vec![1.0]).is_err());
    assert_eq!(polynomial.nodes(), &[0.0, 1.0]);
    let mut workspace = polynomial.workspace().unwrap();
    let mut output = [7.0, 8.0];
    for point in [f64::NAN, f64::INFINITY, 2.0] {
        assert!(
            polynomial
                .evaluate_into(
                    [point],
                    ExtrapolationPolicy::Reject,
                    &mut output,
                    &mut workspace
                )
                .is_err()
        );
        assert_eq!(output, [7.0, 8.0]);
    }
    assert!(
        polynomial
            .evaluate_into(
                [0.5],
                ExtrapolationPolicy::Reject,
                &mut output[..1],
                &mut workspace
            )
            .is_err()
    );
    assert_eq!(output, [7.0, 8.0]);
    // A workspace prepared for a smaller model is a dimension mismatch.
    let wide =
        NewtonInterpolator::new(vec![0.0_f64, 1.0], vec![vec![1.0; 20], vec![2.0; 20]]).unwrap();
    let mut wide_output = [0.0; 20];
    assert_eq!(
        wide.evaluate_into(
            [0.5],
            ExtrapolationPolicy::Reject,
            &mut wide_output,
            &mut workspace
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(wide_output, [0.0; 20]);
    let prepared = polynomial.prepare_antiderivative().unwrap();
    assert!(
        prepared
            .integral_into(
                1.0,
                0.0,
                ExtrapolationPolicy::Reject,
                &mut output,
                &mut [0.0; 2]
            )
            .is_err()
    );
    assert_eq!(output, [7.0, 8.0]);
}

#[test]
fn finite_extrapolation_survives_an_overflowing_coordinate_difference() {
    let polynomial = NewtonInterpolator::new(
        vec![f64::MAX / 2.0, f64::MAX],
        vec![[f64::MAX / 2.0, 1.0], [f64::MAX, 1.0]],
    )
    .unwrap();
    let value = polynomial
        .evaluate([-f64::MAX], ExtrapolationPolicy::MethodDefined)
        .unwrap();
    assert!((value[0] / f64::MAX + 1.0).abs() <= 2.0 * f64::EPSILON);
    assert_eq!(value[1], 1.0);
}

#[test]
fn externally_changed_custom_shapes_cannot_corrupt_prepared_queries() {
    use scioxide_core::AffineValue;
    use std::sync::{Arc, Mutex};
    struct Shared(Arc<Mutex<Vec<f64>>>);
    impl AffineValue for Shared {
        type Scale = f64;
        type Owned = Vec<f64>;
        fn dimension(&self) -> Result<usize, SciOxideError> {
            Ok(self.0.lock().unwrap().len())
        }
        fn component(&self, index: usize) -> Result<f64, SciOxideError> {
            self.0.lock().unwrap().as_slice().component(index)
        }
        fn try_to_owned(&self) -> Result<Vec<f64>, SciOxideError> {
            scioxide_core::backend::try_copy_slice(
                &self.0.lock().unwrap(),
                scioxide_core::quantity!("shared_test_value"),
            )
        }
    }
    let shared = Arc::new(Mutex::new(vec![3.0]));
    let polynomial =
        NewtonInterpolator::new(vec![0.0_f64], vec![Shared(Arc::clone(&shared))]).unwrap();
    shared.lock().unwrap().push(4.0);
    assert_error(
        &polynomial.evaluate([0.0], ExtrapolationPolicy::Reject),
        scioxide_core::SciOxideErrorKind::DimensionMismatch,
    );
    assert_error(
        &polynomial.prepare_antiderivative(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch,
    );
    let mut output = [0.0];
    let mut workspace = polynomial.workspace().unwrap();
    polynomial
        .evaluate_into(
            [0.0],
            ExtrapolationPolicy::Reject,
            &mut output,
            &mut workspace,
        )
        .unwrap();
    assert_eq!(output, [3.0]);
}
