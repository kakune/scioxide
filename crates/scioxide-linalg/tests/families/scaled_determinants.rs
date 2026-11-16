//! Component-preserving determinant representation and allocation regressions.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, NumericalIssue, ScaledValue, SciOxideErrorView};
use scioxide_linalg::{LuDecomposition, Matrix};

macro_rules! width {
    ($name:ident, $s:ty) => {
        #[test]
        fn $name() {
            // Unit-size rows require no equilibration. The old determinant's
            // [0.5,1) shared frame rounded away the imaginary minimum quantum.
            let tiny = <$s>::from_bits(1);
            let z = Complex::new(1.0 as $s, tiny);
            let matrix = Matrix::from_vec(
                2,
                2,
                vec![
                    z,
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(1.0, 0.0),
                ],
            )
            .unwrap();
            let factor = LuDecomposition::factor(&matrix).unwrap();
            {
                let scaled: ScaledValue<Complex<$s>> = factor.scaled_determinant().unwrap();
                let value = scaled.try_value().unwrap();
                assert_eq!(value.re.to_bits(), (1.0 as $s).to_bits());
                assert_eq!(value.im.to_bits(), tiny.to_bits());
                let frame = scaled.try_projective().unwrap();
                assert_eq!(
                    frame
                        .into_scaled()
                        .unwrap()
                        .try_value()
                        .unwrap()
                        .im
                        .to_bits(),
                    tiny.to_bits()
                );
            };
            let matrix = Matrix::from_vec(2, 2, vec![<$s>::MAX, 0.0, 0.0, 2.0]).unwrap();
            let factor = LuDecomposition::factor(&matrix).unwrap();
            {
                let scaled = factor.scaled_determinant().unwrap();
                assert!(matches!(
                    scaled.try_value().unwrap_err().view(),
                    SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow,
                        ..
                    }
                ));
                assert!(factor.slogdet().unwrap().log_abs().is_finite());
            };
        }
    };
}
width!(binary32_determinant, f32);
width!(binary64_determinant, f64);
