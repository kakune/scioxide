//! Value representations: extended-range, shared-exponent and logarithmic.
//!
//! `ScaledValue` stores each component with its own exponent, `ProjectiveValue`
//! uses one exponent for both components, and `LogValue` stores a phase and the
//! logarithm of the magnitude. `round` allows gradual underflow; `try_value`
//! rejects a nonzero component that would round to zero.

mod logarithmic;
mod projective;
pub use logarithmic::LogValue;
pub use projective::ProjectiveValue;

/// A function value together with its derivative.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ValueAndDerivative<V, D> {
    value: V,
    derivative: D,
}

impl<V, D> ValueAndDerivative<V, D> {
    /// Pairs a value with its derivative.
    pub const fn new(value: V, derivative: D) -> Self {
        Self { value, derivative }
    }

    /// Returns the function value.
    pub const fn value(&self) -> &V {
        &self.value
    }

    /// Returns the derivative.
    pub const fn derivative(&self) -> &D {
        &self.derivative
    }

    /// Returns the value and the derivative.
    pub fn into_parts(self) -> (V, D) {
        (self.value, self.derivative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Complex64;
    use crate::RealScalar;
    use crate::ScaledValue;

    #[test]
    fn scaled_values_round_trip_real_and_complex_values() {
        let real = ScaledValue::from_value(-12.0_f64).unwrap();
        assert_eq!(real.real().significand(), -1.5);
        assert_eq!(real.real().exponent(), 3);
        assert_eq!(real.try_value().unwrap(), -12.0);

        let complex = ScaledValue::from_value(Complex64::new(3.0, 4.0)).unwrap();
        assert_eq!(complex.try_value().unwrap(), Complex64::new(3.0, 4.0));
    }

    #[test]
    fn scaled_values_preserve_representable_ieee_boundary_components() {
        for value in [f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX, -f64::MAX] {
            assert_eq!(
                ScaledValue::from_value(value).unwrap().try_value().unwrap(),
                value
            );
            let complex = Complex64::new(value, -value);
            let scaled = ScaledValue::from_value(complex).unwrap();
            assert!((1.0..2.0).contains(&scaled.real().significand().abs()));
            assert_eq!(scaled.try_value().unwrap(), complex);
        }
        for value in [f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX, -f32::MAX] {
            assert_eq!(
                ScaledValue::from_value(value).unwrap().try_value().unwrap(),
                value
            );
            let complex = crate::Complex32::new(value, -value);
            assert_eq!(
                ScaledValue::from_value(complex)
                    .unwrap()
                    .try_value()
                    .unwrap(),
                complex
            );
        }
        let mixed = Complex64::new(f64::MIN_POSITIVE, f64::from_bits(1));
        assert_eq!(
            ScaledValue::from_value(mixed).unwrap().try_value().unwrap(),
            mixed
        );
        assert!(
            ScaledValue::from_parts(Complex64::new(0.5, 0.0), 2048)
                .unwrap()
                .try_value()
                .is_err()
        );
        assert!(
            ScaledValue::from_parts(Complex64::new(0.5, 0.0), -2048)
                .unwrap()
                .try_value()
                .is_err()
        );
    }

    #[test]
    fn real_scaled_constructors_preserve_normalization_at_ieee_boundaries() {
        for value in [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            1.0_f64.next_down(),
            f64::MAX,
        ] {
            for value in [value, -value] {
                let scaled = ScaledValue::from_value(value).unwrap();
                assert!((1.0..2.0).contains(&scaled.real().significand().abs()));
                assert_eq!(
                    scaled
                        .real()
                        .significand()
                        .scalbn(i32::try_from(scaled.real().exponent()).unwrap()),
                    value
                );
            }
        }
        for value in [
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.0_f32.next_down(),
            f32::MAX,
        ] {
            let scaled = ScaledValue::from_value(value).unwrap();
            assert!((1.0..2.0).contains(&scaled.real().significand()));
            assert_eq!(
                scaled
                    .real()
                    .significand()
                    .scalbn(i32::try_from(scaled.real().exponent()).unwrap()),
                value
            );
        }
        for logarithm in [-1000.0_f64, -1e10, -1e18] {
            let scaled = ScaledValue::from_log(1.0, logarithm).unwrap();
            assert!((1.0..2.0).contains(&scaled.real().significand()));
            let scaled = ScaledValue::from_log(1.0_f32, logarithm as f32).unwrap();
            assert!((1.0..2.0).contains(&scaled.real().significand()));
        }
    }

    #[test]
    fn logarithmic_values_preserve_sign_and_phase() {
        let real = LogValue::from_value(-12.0_f64).unwrap();
        assert_eq!(real.sign(), -1.0);
        assert!((real.try_value().unwrap() + 12.0).abs() < 2.0e-14);

        let complex = LogValue::from_value(Complex64::new(3.0, 4.0)).unwrap();
        let restored = complex.try_value().unwrap();
        assert!((restored.re - 3.0).abs() < 1e-14);
        assert!((restored.im - 4.0).abs() < 1e-14);
    }
}
