//! Compare component-scaled results in an oracle's binary64 reference frame.
//! This does not convert the result to a shared exponent in its source width.
use scioxide_core::{Complex64, NumericScalar, RealScalar, ScaledValue};

pub fn at_exponent<T: NumericScalar>(value: &ScaledValue<T>, exponent: i64) -> Complex64 {
    let project = |component: scioxide_core::Scaled<T::Real>| {
        let shift = (i128::from(component.exponent()) - i128::from(exponent))
            .clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32;
        component.significand().into_f64().scalbn(shift)
    };
    Complex64::new(project(value.real()), project(value.imaginary()))
}

/// A scaled-coordinate error expressed in the same binary64 reference frame.
#[allow(dead_code)]
pub fn error_at_exponent<T: NumericScalar>(
    value: &ScaledValue<T>,
    error: f64,
    exponent: i64,
) -> f64 {
    let shift = (i128::from(value.scale_exponent()) - i128::from(exponent))
        .clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32;
    error.scalbn(shift)
}
