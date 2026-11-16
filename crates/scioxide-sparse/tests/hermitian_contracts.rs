//! External consumers of the documented implementation boundary.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, NonFiniteOrigin, SciOxideErrorKind};
use scioxide_sparse::backend::hermitian;

macro_rules! hermitian_contract {
    ($name:ident, $scalar:ty) => {
        #[test]
        fn $name() {
            {
                let a = Complex::new(2.0 as $scalar, 0.0);
                let b = Complex::new(0.0 as $scalar, 1.0);
                let c = Complex::new(3.0 as $scalar, 0.0);
                assert_eq!(
                    hermitian::determinant(a, b, c).unwrap().round().unwrap(),
                    5.0
                );
                // Multiplying [2 -i; i 3] by [1+2i; 3-i] gives [1+i; 7-2i].
                let actual = hermitian::solve_block(
                    a,
                    b,
                    c,
                    Complex::new(1.0, 1.0),
                    Complex::new(7.0, -2.0),
                )
                .unwrap();
                for (actual, expected) in actual
                    .iter()
                    .zip([Complex::new(1.0, 2.0), Complex::new(3.0, -1.0)])
                {
                    let actual = actual.round().unwrap();
                    assert!((actual.re - expected.re).abs() <= 8.0 * <$scalar>::EPSILON);
                    assert!((actual.im - expected.im).abs() <= 8.0 * <$scalar>::EPSILON);
                }
                let values = hermitian::block_values(a, b, c).unwrap();
                let expected = [
                    2.5 + (1.25 as $scalar).sqrt(),
                    2.5 - (1.25 as $scalar).sqrt(),
                ];
                for (actual, expected) in values.iter().zip(expected) {
                    assert!(
                        (actual.round().unwrap() - expected).abs() <= 16.0 * <$scalar>::EPSILON
                    );
                }
                let bad = Complex::new(2.0, 1.0);
                assert_eq!(
                    hermitian::determinant(bad, b, c).unwrap_err().kind(),
                    SciOxideErrorKind::InvalidInput
                );
                assert_eq!(
                    hermitian::block_values(a, b, bad).unwrap_err().kind(),
                    SciOxideErrorKind::InvalidInput
                );
                assert_eq!(
                    hermitian::solve_block(bad, b, c, a, c).unwrap_err().kind(),
                    SciOxideErrorKind::InvalidInput
                );
                for value in [<$scalar>::NAN, <$scalar>::INFINITY] {
                    let bad = Complex::new(2.0, value);
                    for error in [
                        hermitian::determinant(bad, b, c).unwrap_err(),
                        hermitian::block_values(a, b, bad).unwrap_err(),
                    ] {
                        assert_eq!(
                            match error.view() {
                                scioxide_core::SciOxideErrorView::NonFinite { origin, .. } =>
                                    Some(origin),
                                _ => None,
                            }
                            .unwrap(),
                            NonFiniteOrigin::Input
                        );
                    }
                }
                assert_eq!(
                    hermitian::solve_block(0.0 as $scalar, 0.0, 0.0, 1.0, 1.0)
                        .unwrap_err()
                        .kind(),
                    SciOxideErrorKind::Singular
                );
                let tiny = <$scalar>::from_bits(1);
                let det = hermitian::determinant(tiny, 0.0, tiny).unwrap();
                assert!(!det.is_zero());
                assert!(det.exponent() < <$scalar>::MIN_EXP as i64);
            };
        }
    };
}
hermitian_contract!(hermitian_f32, f32);
hermitian_contract!(hermitian_f64, f64);
