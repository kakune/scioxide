//! Newton, Halley and secant corrections for real and complex iterations.
//!
//! Each formula is generic over [`NumericScalar`]. Ordinary arithmetic forms
//! the candidate while the step stays in the normal range; otherwise
//! [`ScaledValue`] arithmetic with an extended exponent range, or a correctly
//! rounded sum of products, finds a representable candidate.

use scioxide_core::backend::scaled::scaled_magnitude_le_tolerance;
use scioxide_core::{
    AlgebraicScalar, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError, Tolerance,
    checked_sum_of_products,
};

use super::Sample;

/// Largest component magnitude; the absolute value of a real scalar.
///
/// The real case avoids a `max` with the zero imaginary part, which would sit
/// on every iteration's dependency chain.
#[inline(always)]
fn component_scale<T: NumericScalar>(value: T) -> T::Real {
    let real = value.real_component().abs();
    if T::IS_REAL {
        real
    } else {
        real.max(value.imaginary_component().abs())
    }
}

/// Whether the largest component is normal, which implies finite components.
#[inline(always)]
fn normal<T: NumericScalar>(value: T) -> bool {
    component_scale(value).is_normal()
}

/// Divide every component by a positive real `scale`.
#[inline(always)]
fn divide<T: NumericScalar>(value: T, scale: T::Real) -> T {
    T::from_components(
        value.real_component() / scale,
        value.imaginary_component() / scale,
    )
}

/// Newton (`second = None`) or Halley correction `x - f/f'` at `x`, with
/// Halley's factor `1 / (1 - f f''/(2 f'^2))` applied only while that ratio
/// has magnitude below one.
///
/// Returns `Ok(None)` for a zero derivative. Steps outside the normal range are
/// recomputed in extended range, so a representable candidate is returned
/// whenever one exists.
///
/// # Errors
/// Returns overflow when the candidate itself is not representable.
#[inline(always)]
pub(crate) fn newton<T: NumericScalar>(
    x: T,
    value: T,
    first: T,
    second: Option<T>,
) -> Result<Option<T>, SciOxideError> {
    if first == T::ZERO {
        return Ok(None);
    }
    let mut step = value / first;
    if normal(step) {
        if let Some(second) = second {
            let ratio = second / first;
            let half = step * T::from_real(<T::Real as RealScalar>::HALF);
            let adjustment = half * ratio;
            if ((!normal(ratio) || !normal(half)) && second != T::ZERO)
                || !adjustment.has_finite_components()
            {
                return newton_scaled(x, value, first, Some(second));
            }
            if adjustment.abs() < <T::Real as AlgebraicScalar>::ONE {
                step = step / (T::ONE - adjustment);
            }
        }
        let candidate = x - step;
        if normal(step) && candidate.has_finite_components() {
            return Ok(Some(candidate));
        }
    }
    newton_scaled(x, value, first, second)
}

#[cold]
#[inline(never)]
fn newton_scaled<T: NumericScalar>(
    x: T,
    value: T,
    first: T,
    second: Option<T>,
) -> Result<Option<T>, SciOxideError> {
    let derivative = ScaledValue::from_value(first)?;
    if derivative.is_zero() {
        return Ok(None);
    }
    let mut step = ScaledValue::from_value(value)?.div(derivative)?;
    if let Some(second) = second {
        let adjustment = step
            .mul(ScaledValue::from_value(second)?)?
            .div(derivative)?
            .mul(ScaledValue::HALF)?;
        if adjustment.magnitude()?.abs_cmp(Scaled::ONE).is_lt() {
            step = step.div(ScaledValue::ONE.sub(adjustment)?)?;
        }
    }
    ScaledValue::from_value(x)?.sub(step)?.round().map(Some)
}

/// Secant candidate through `previous` and `current`.
///
/// Real values are first divided by their larger magnitude, so one of them is
/// exactly `±1` and their difference cannot overflow. The complex quotient
/// already rescales to avoid overflow, so complex values interpolate directly;
/// a complex difference that overflows takes the extended-range path.
///
/// Returns `Ok(None)` when the two values are equal or both zero. Steps
/// outside the normal range retry before projecting the interpolation weights.
///
/// # Errors
/// Returns non-finite or overflow errors when no representable candidate exists.
#[inline(always)]
pub(crate) fn secant<T: NumericScalar>(
    previous: Sample<T>,
    current: Sample<T>,
) -> Result<Option<T>, SciOxideError> {
    let (previous_value, current_value) = if T::IS_REAL {
        let scale = component_scale(previous.value).max(component_scale(current.value));
        if scale == <T::Real as AlgebraicScalar>::ZERO {
            return Ok(None);
        }
        let a = divide(previous.value, scale);
        let b = divide(current.value, scale);
        if (previous.value != T::ZERO && !normal(a)) || (current.value != T::ZERO && !normal(b)) {
            return secant_scaled(previous, current);
        }
        (a, b)
    } else {
        (previous.value, current.value)
    };
    let denominator = current_value - previous_value;
    if denominator == T::ZERO {
        return Ok(None);
    }
    let displacement = current.x - previous.x;
    if denominator.has_finite_components() {
        let weight = current_value / denominator;
        let step = weight * displacement;
        let candidate = current.x - step;
        let cancellation = candidate.has_finite_components()
            && step.has_finite_components()
            && component_scale(candidate)
                <= <T::Real as RealScalar>::SQRT_EPSILON
                    * component_scale(current.x).max(component_scale(step));
        if displacement.has_finite_components()
            && normal(step)
            && candidate.has_finite_components()
            && !cancellation
        {
            return Ok(Some(candidate));
        }
        if cancellation {
            return secant_scaled(previous, current);
        }
        if T::IS_REAL {
            if current_value != T::ZERO && !normal(weight) {
                return secant_scaled(previous, current);
            }
            return sum_of_products(
                [weight, -(previous_value / denominator)],
                [previous.x, current.x],
            )
            .map(Some);
        }
    }
    secant_scaled(previous, current)
}

/// Secant intersection `(x₀ f₁ − x₁ f₀)/(f₁ − f₀)` with extended products.
/// Keeping the function values until after the coordinate products prevents
/// a small interpolation weight from vanishing before a large multiplication.
#[cold]
#[inline(never)]
fn secant_scaled<T: NumericScalar>(
    previous: Sample<T>,
    current: Sample<T>,
) -> Result<Option<T>, SciOxideError> {
    let previous_value = ScaledValue::from_value(previous.value)?;
    let current_value = ScaledValue::from_value(current.value)?;
    let denominator = current_value.sub(previous_value)?;
    if denominator.is_zero() {
        return Ok(None);
    }
    ScaledValue::from_value(previous.x)?
        .difference_of_products(
            current_value,
            ScaledValue::from_value(current.x)?,
            previous_value,
        )?
        .div(denominator)?
        .round()
        .map(Some)
}

/// Correctly rounded `a x + b y`, componentwise for complex scalars.
fn sum_of_products<T: NumericScalar>([a, b]: [T; 2], [x, y]: [T; 2]) -> Result<T, SciOxideError> {
    let re = NumericScalar::real_component;
    let im = NumericScalar::imaginary_component;
    if T::IS_REAL {
        return checked_sum_of_products(&[re(a), re(b)], &[re(x), re(y)]).map(T::from_real);
    }
    let real = checked_sum_of_products(
        &[re(a), -im(a), re(b), -im(b)],
        &[re(x), im(x), re(y), im(y)],
    )?;
    let imaginary =
        checked_sum_of_products(&[re(a), im(a), re(b), im(b)], &[im(x), re(x), im(y), re(y)])?;
    Ok(T::from_components(real, imaginary))
}

/// Whether finite reals `a` and `b` lie within `tolerance` at the reference
/// `max(|a|, |b|)`. A representable threshold compares without forming
/// `a − b`, as [`Tolerance::is_converged`] does; a threshold beyond the finite
/// range compares in extended range, as complex magnitudes do.
///
/// # Errors
/// Returns errors for non-finite points.
#[inline(always)]
pub(crate) fn within<S: RealScalar>(
    tolerance: &Tolerance<S>,
    a: S,
    b: S,
) -> Result<bool, SciOxideError> {
    let scale = a.abs().max(b.abs());
    let threshold = tolerance.absolute() + tolerance.relative() * scale;
    if threshold.is_finite() {
        return Ok(scioxide_core::backend::absolute_difference_le(
            a, b, threshold,
        ));
    }
    scaled_magnitude_le_tolerance(
        ScaledValue::from_value(b)?
            .sub(ScaledValue::from_value(a)?)?
            .magnitude()?,
        tolerance,
        Scaled::from_value(scale)?,
    )
}

/// Whether the move from `previous` to `next` meets `tolerance` in coordinate
/// magnitude. A zero tolerance needs only equality, with no threshold
/// construction; magnitudes or thresholds outside the normal range use
/// extended arithmetic for real and complex points alike.
///
/// # Errors
/// Returns errors for non-finite points.
#[inline(always)]
pub(crate) fn small_step<T: NumericScalar>(
    tolerance: &Tolerance<T::Real>,
    previous: T,
    next: T,
) -> Result<bool, SciOxideError> {
    let zero = <T::Real as AlgebraicScalar>::ZERO;
    if tolerance.absolute() == zero && tolerance.relative() == zero {
        return Ok(previous == next);
    }
    if T::IS_REAL {
        return within(tolerance, previous.real_component(), next.real_component());
    }
    let change = (next - previous).abs();
    let reference = previous.abs().max(next.abs());
    let relative = tolerance.relative() * reference;
    let threshold = tolerance.absolute() + relative;
    if change.is_finite()
        && reference.is_finite()
        && threshold.is_finite()
        && (relative.is_normal() || tolerance.relative() == zero)
        && (change.is_normal() || next == previous)
    {
        return Ok(change <= threshold);
    }
    let previous = ScaledValue::from_value(previous)?;
    let next = ScaledValue::from_value(next)?;
    let reference = previous.magnitude()?;
    let next_magnitude = next.magnitude()?;
    scaled_magnitude_le_tolerance(
        next.sub(previous)?.magnitude()?,
        tolerance,
        if reference.abs_le(next_magnitude) {
            next_magnitude
        } else {
            reference
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    fn sample<T: NumericScalar>(x: T, value: T) -> Sample<T> {
        Sample::checked(x, value).unwrap().0
    }

    #[test]
    fn real_and_embedded_complex_corrections_agree() {
        for (x, value, first, second) in [
            (1.5_f64, 0.25, 3.0, None),
            (1.5, 0.25, 3.0, Some(2.0)),
            (f64::MAX, f64::MAX / 2.0, 0.5, Some(0.0)),
            (1e-300, 1e-300, 1e300, None),
        ] {
            let real = newton(x, value, first, second).unwrap().unwrap();
            let complex = newton(
                Complex::new(x, 0.0),
                Complex::new(value, 0.0),
                Complex::new(first, 0.0),
                second.map(|s| Complex::new(s, 0.0)),
            )
            .unwrap()
            .unwrap();
            assert_eq!(complex, Complex::new(real, 0.0));
        }
        assert_eq!(newton(1.0_f64, 1.0, 0.0, None).unwrap(), None);
        let real = secant(sample(0.0_f64, -1.0), sample(1.0, 3.0)).unwrap();
        let complex = secant(
            sample(Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)),
            sample(Complex::new(1.0, 0.0), Complex::new(3.0, 0.0)),
        )
        .unwrap();
        assert_eq!(real, Some(0.25));
        assert_eq!(complex, Some(Complex::new(0.25, 0.0)));
        assert_eq!(
            secant(sample(0.0_f64, 1.0), sample(1.0, 1.0)).unwrap(),
            None
        );
    }

    #[test]
    fn extreme_secant_steps_round_once() {
        let maximum = f64::MAX;
        let candidate = secant(
            sample(maximum / 2.0, 0.75 * maximum),
            sample(maximum, maximum),
        )
        .unwrap();
        assert_eq!(candidate, Some(-maximum));
    }

    #[test]
    fn secant_preserves_small_values_before_large_coordinate_products() {
        for points in [
            [sample(0.0_f64, 1e-300), sample(1e300, 1.0)],
            [sample(1e300, 1.0), sample(0.0_f64, 1e-300)],
        ] {
            let candidate = secant(points[0], points[1]).unwrap().unwrap();
            assert!((candidate + 1.0).abs() < 4.0 * f64::EPSILON);
        }
        for (small, large) in [(1e-300_f64, 1e300), (1e-30, 1e30)] {
            let candidate = secant(sample(large, large), sample(0.0, small))
                .unwrap()
                .unwrap();
            assert!((candidate / small + 1.0).abs() < 4.0 * f64::EPSILON);
            let complex = secant(
                sample(Complex::new(large, 0.0), Complex::new(large, 0.0)),
                sample(Complex::new(0.0, 0.0), Complex::new(small, 0.0)),
            )
            .unwrap()
            .unwrap();
            assert!((complex.re / small + 1.0).abs() < 4.0 * f64::EPSILON);
            assert_eq!(complex.im, 0.0);
        }
    }

    #[test]
    fn complex_secant_recovers_an_overflowing_difference() {
        let maximum = f64::MAX;
        let real = secant(sample(0.0_f64, -0.75 * maximum), sample(1.0, maximum))
            .unwrap()
            .unwrap();
        let complex = secant(
            sample(Complex::new(0.0, 0.0), Complex::new(-0.75 * maximum, 0.0)),
            sample(Complex::new(1.0, 0.0), Complex::new(maximum, 0.0)),
        )
        .unwrap()
        .unwrap();
        assert!((real - 3.0 / 7.0).abs() <= 2.0 * f64::EPSILON);
        assert!((complex.re - real).abs() <= 2.0 * f64::EPSILON);
        assert_eq!(complex.im, 0.0);
        // Ordinary complex values interpolate directly.
        let direct = secant(
            sample(Complex::new(0.0, 0.0), Complex::new(-1.0, 1.0)),
            sample(Complex::new(1.0, 0.0), Complex::new(3.0, 1.0)),
        )
        .unwrap();
        assert_eq!(direct, Some(Complex::new(0.25, -0.25)));
    }

    #[test]
    fn zero_tolerance_small_steps_require_equality() {
        let zero = Tolerance::new(0.0, 0.0).unwrap();
        assert!(small_step(&zero, 1.0_f64, 1.0).unwrap());
        assert!(!small_step(&zero, 1.0_f64, 1.0 + f64::EPSILON).unwrap());
        let z = Complex::new(1.0, 1.0);
        assert!(small_step(&zero, z, z).unwrap());
        let loose = Tolerance::new(1e-3, 0.0).unwrap();
        assert!(small_step(&loose, z, z + Complex::new(0.0, 1e-4)).unwrap());
        assert!(!small_step(&loose, z, z + Complex::new(0.0, 1e-2)).unwrap());
    }
}
