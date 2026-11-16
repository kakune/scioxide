//! Vector kernels of the nonlinear solvers, with extended-range fallback.
//!
//! Inputs have already been checked for finiteness. The kernels check the
//! arithmetic range before the caller accepts a trial. Real problems stay real
//! on the extended-range paths.
use core::cmp::Ordering;
use core::ops::Neg;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::standard::product_needs_retry;
use scioxide_core::{CompensatedSum, NumericScalar, RealScalar, ScaledNorm, SciOxideError};
use scioxide_core::{Scaled, ScaledAccumulator, ScaledValue};
pub(crate) fn scaled<T: NumericScalar>(x: T) -> Result<ScaledValue<T>, SciOxideError> {
    ScaledValue::from_value(x)
}
pub(crate) fn rounded<T: NumericScalar>(x: ScaledValue<T>) -> Result<T, SciOxideError> {
    x.round()
}

/// Native affine update; products at risk of overflow or underflow are
/// recomputed in extended range before the output is written.
#[inline]
pub(crate) fn affine<T: NumericScalar>(x: T, a: T, y: T) -> Result<T, SciOxideError> {
    let product = a * y;
    let value = x + product;
    if value.has_finite_components() && !product_needs_retry(a, y, product) {
        Ok(value)
    } else {
        affine_range(x, a, y)
    }
}

#[cold]
fn affine_range<T: NumericScalar>(x: T, a: T, y: T) -> Result<T, SciOxideError> {
    if T::IS_REAL {
        let value = Scaled::from_value(x.real_component())?.add(
            Scaled::from_value(a.real_component())?.mul(Scaled::from_value(y.real_component())?)?,
        )?;
        Ok(T::from_real(value.round()?))
    } else {
        rounded(scaled(x)?.add(scaled(a)?.mul(scaled(y)?)?)?)
    }
}

/// How [`affine_into`] rounds a damped real coordinate `x + alpha·d`.
///
/// A full step (`alpha == 1`) is the exact sum `x + d` under either choice, and
/// complex coordinates always round the product and then the sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StepRounding {
    /// Round the product and then the sum. The fixed-point and mixing updates
    /// use this vectorisable form.
    Separate,
    /// Round once with a fused multiply-add. The dense Newton, Broyden and
    /// trust-region iterates depend on this single rounding of damped
    /// line-search steps.
    Fused,
}

/// The one slice update `output = point + alpha·direction`, rounded as the
/// caller's [`StepRounding`] states.
///
/// The ordinary-range screen is aggregated so the full pass has no
/// per-coordinate error branch; a coordinate outside the native range replays
/// in extended range, and a result that does not fit is an overflow error.
pub(crate) fn affine_into<T: NumericScalar>(
    output: &mut [T],
    point: &[T],
    alpha: T,
    direction: &[T],
    rounding: StepRounding,
) -> Result<(), SciOxideError> {
    let fused = T::IS_REAL && rounding == StepRounding::Fused && alpha != T::ONE;
    let mut ordinary = true;
    if alpha == T::ONE {
        // The product `1·d` is exact, so this equals both roundings.
        for ((out, &x), &d) in output.iter_mut().zip(point).zip(direction) {
            *out = x + d;
            ordinary &= out.has_finite_components();
        }
    } else if fused {
        for ((out, &x), &d) in output.iter_mut().zip(point).zip(direction) {
            *out = T::from_real(
                alpha
                    .real_component()
                    .mul_add(d.real_component(), x.real_component()),
            );
            ordinary &= out.has_finite_components();
        }
    } else {
        for ((out, &x), &d) in output.iter_mut().zip(point).zip(direction) {
            let product = alpha * d;
            *out = x + product;
            ordinary &= out.has_finite_components() && !product_needs_retry(alpha, d, product);
        }
    }
    if !ordinary {
        for ((out, &x), &d) in output.iter_mut().zip(point).zip(direction) {
            *out = if !fused {
                affine(x, alpha, d)?
            } else if out.has_finite_components() {
                *out
            } else {
                affine_range(x, alpha, d)?
            };
        }
    }
    Ok(())
}

/// History coefficients may legitimately exceed the scalar exponent range.
pub(crate) fn affine_scaled<T: NumericScalar>(
    x: T,
    a: ScaledValue<T>,
    y: T,
) -> Result<T, SciOxideError> {
    if let Some(a) = native_coefficient(a) {
        affine(x, a, y)
    } else {
        rounded(scaled(x)?.add(a.mul(scaled(y)?)?)?)
    }
}

/// Subnormal coefficient projection can lose bits that a large factor restores.
pub(crate) fn native_coefficient<T: NumericScalar>(a: ScaledValue<T>) -> Option<T> {
    let value = a.try_value().ok()?;
    let real = value.real_component();
    let imaginary = value.imaginary_component();
    ((real.is_normal() || a.real().is_zero()) && (imaginary.is_normal() || a.imaginary().is_zero()))
        .then_some(value)
}

#[inline]
pub(crate) fn difference<T: NumericScalar>(x: T, y: T) -> Result<T, SciOxideError> {
    let value = x - y;
    if value.has_finite_components() {
        Ok(value)
    } else {
        rounded(scaled(x)?.sub(scaled(y)?)?)
    }
}

/// Hermitian product with native compensated component sums. Range hazards
/// replay the original products; a finite final projection is optional.
#[inline]
pub(crate) fn dot<T: NumericScalar>(x: &[T], y: &[T]) -> Result<ScaledValue<T>, SciOxideError> {
    if let Some(value) = native_dot(x, y) {
        return scaled(value);
    }
    dot_range(x, y)
}

#[inline]
fn native_dot<T: NumericScalar>(x: &[T], y: &[T]) -> Option<T> {
    let mut real = CompensatedSum::ZERO;
    let mut imaginary = CompensatedSum::ZERO;
    for (&a, &b) in x.iter().zip(y) {
        let a = a.conj();
        let product = a * b;
        if product_needs_retry(a, b, product) {
            return None;
        }
        real.add(product.real_component()).ok()?;
        if !T::IS_REAL {
            imaginary.add(product.imaginary_component()).ok()?;
        }
    }
    Some(T::from_components(
        real.total().ok()?,
        imaginary.total().ok()?,
    ))
}

#[cold]
fn dot_range<T: NumericScalar>(x: &[T], y: &[T]) -> Result<ScaledValue<T>, SciOxideError> {
    let mut real = ScaledAccumulator::ZERO;
    let mut imaginary = ScaledAccumulator::ZERO;
    for (&a, &b) in x.iter().zip(y) {
        let ar = Scaled::from_value(a.real_component())?;
        let br = Scaled::from_value(b.real_component())?;
        real.add_product(ar, br)?;
        if !T::IS_REAL {
            let ai = Scaled::from_value(a.imaginary_component())?;
            let bi = Scaled::from_value(b.imaginary_component())?;
            real.add_product(ai, bi)?;
            imaginary.add_product(ar, bi)?;
            imaginary.add_product(ai.neg(), br)?;
        }
    }
    Ok(ScaledValue::from_components(
        real.total()?,
        imaginary.total()?,
    ))
}

#[inline]
pub(crate) fn norm<T: NumericScalar>(x: &[T]) -> Result<T::Real, SciOxideError> {
    norm_iter(x.iter().copied())
}

/// Euclidean norm from four square-sum lanes. When the sum is not a normal
/// number, the norm is recomputed with the scaled sum of squares of
/// `ScaledNorm`, which avoids overflow and underflow; real values stay real.
#[inline]
pub(crate) fn norm_iter<T: NumericScalar>(
    values: impl Iterator<Item = T> + Clone,
) -> Result<T::Real, SciOxideError> {
    let mut sums = [T::Real::ZERO; 4];
    let mut nonzero = false;
    for (i, value) in values.clone().enumerate() {
        let real = value.real_component();
        nonzero |= real != T::Real::ZERO;
        let mut square = real * real;
        if !T::IS_REAL {
            let imaginary = value.imaginary_component();
            nonzero |= imaginary != T::Real::ZERO;
            square += imaginary * imaginary;
        }
        sums[i % 4] += square;
    }
    let sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    if sum.is_normal() || (sum == T::Real::ZERO && !nonzero) {
        Ok(sum.sqrt())
    } else {
        norm_range(values)
    }
}

#[cold]
fn norm_range<T: NumericScalar>(values: impl Iterator<Item = T>) -> Result<T::Real, SciOxideError> {
    let mut sum = ScaledNorm::new();
    for value in values {
        sum.add(value.real_component())?;
        if !T::IS_REAL {
            sum.add(value.imaginary_component())?;
        }
    }
    sum.finish()
}

/// Largest coordinate move of `direction` relative to `max(1, |xᵢ|)`.
///
/// A trial `x + λd` moves no coordinate by more than `λ` times this value in
/// those units, so backtracking compares it with a relative step floor.
pub(crate) fn relative_length<T: NumericScalar>(point: &[T], direction: &[T]) -> T::Real {
    point
        .iter()
        .zip(direction)
        .fold(T::Real::ZERO, |largest, (&x, &d)| {
            largest.max(d.abs() / x.abs().max(T::Real::ONE))
        })
}

/// Safeguarded Armijo backtracking on `φ(λ) = ½‖F(x + λd)‖²`
/// (Dennis–Schnabel, Algorithm A6.3.1).
///
/// The first trial is `λ = 1`. A rejected trial moves to the minimizer of the
/// quadratic interpolant of `φ(0)`, `φ'(0)` and `φ(λ)`, and later rejections to the
/// minimizer of the cubic through the last two trials; each new step is clamped to
/// `[0.1, 0.5]·λ`. Native comparisons accept a step only outside a
/// conservative rounding envelope. Other decisions and the interpolation use
/// extended range, so residuals near the limits of the floating-point range
/// never overflow the merit test.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Backtracking<R: RealScalar> {
    /// `φ'(0)`, strictly negative.
    slope: Scaled<R>,
    /// Current trial step.
    step: R,
    /// Previous rejected step and its curvature excess `φ(λ) − φ(0) − λφ'(0)`.
    previous: Option<(R, Scaled<R>)>,
    /// Nonmonotone allowance `C − φ(0) ≥ 0` above the current merit.
    allowance: Scaled<R>,
}

impl<R: RealScalar> Backtracking<R> {
    /// Start at the full step for a strictly negative merit slope `φ'(0)`.
    pub(crate) const fn new(slope: Scaled<R>) -> Self {
        Self {
            slope,
            step: R::ONE,
            previous: None,
            allowance: Scaled::ZERO,
        }
    }

    /// Nonmonotone search accepting `φ(λ) ≤ C + cλφ'(0)` for a reference merit
    /// `C = φ(0) + allowance`; a negative allowance is treated as zero.
    pub(crate) fn with_allowance(slope: Scaled<R>, allowance: Scaled<R>) -> Self {
        Self {
            allowance: if allowance.is_sign_negative() {
                Scaled::ZERO
            } else {
                allowance
            },
            ..Self::new(slope)
        }
    }

    /// Current trial step.
    pub(crate) const fn step(&self) -> R {
        self.step
    }

    /// Reject an unevaluable trial without using it in merit interpolation.
    pub(crate) fn reject_domain(&mut self) {
        self.step *= R::HALF;
        self.previous = None;
    }

    /// Accept the trial residual by the Armijo test `φ(λ) ≤ C + cλφ'(0)`, where
    /// `C ≥ φ(0)` is the reference merit, or reject it and select the next step.
    pub(crate) fn assess<T: NumericScalar<Real = R>>(
        &mut self,
        current: &[T],
        trial: &[T],
        c: R,
    ) -> Result<bool, SciOxideError> {
        if self.native_accept(current, trial, c) {
            return Ok(true);
        }
        let mut change = ScaledAccumulator::ZERO;
        for (&trial, &current) in trial.iter().zip(current) {
            for (t, f) in [
                (trial.real_component(), current.real_component()),
                (trial.imaginary_component(), current.imaginary_component()),
            ] {
                change.add_scaled(Scaled::difference_of_products(
                    Scaled::from_value(t)?,
                    Scaled::from_value(t)?,
                    Scaled::from_value(f)?,
                    Scaled::from_value(f)?,
                )?)?;
            }
        }
        let change = change.total()?.shifted(-1)?;
        let step = Scaled::from_value(self.step)?;
        let decrease = self.slope.mul(step)?;
        if change.cmp_value(decrease.mul(Scaled::from_value(c)?)?.add(self.allowance)?)
            != Ordering::Greater
        {
            return Ok(true);
        }
        let excess = change.sub(decrease)?;
        let upper = self.step * R::HALF;
        let lower = self.step / (R::TWO * R::TWO + R::ONE) * R::HALF;
        let candidate = match self.previous {
            None => Some(
                self.slope
                    .neg()
                    .mul(step.square()?)?
                    .div(excess.shifted(1)?)?,
            ),
            Some((previous, previous_excess)) => {
                self.cubic(step, excess, previous, previous_excess)?
            }
        };
        let next = match candidate {
            Some(value) if value.cmp_value(Scaled::from_value(upper)?) == Ordering::Greater => {
                upper
            }
            Some(value) if value.cmp_value(Scaled::from_value(lower)?) != Ordering::Less => {
                value.round()?
            }
            Some(_) => lower,
            None => upper,
        };
        self.previous = Some((self.step, excess));
        self.step = next;
        Ok(false)
    }

    /// Accepts only when the native merit comparison is outside its rounding
    /// envelope. Rejections and close decisions use the scaled
    /// difference-of-products path, which also drives the interpolation.
    fn native_accept<T: NumericScalar<Real = R>>(&self, current: &[T], trial: &[T], c: R) -> bool {
        let (Ok(slope), Ok(allowance), Ok(length)) = (
            self.slope.try_value(),
            self.allowance.try_value(),
            R::try_from_usize_exact(current.len()),
        ) else {
            return false;
        };
        let decrease = slope * self.step * c;
        if !decrease.is_normal() {
            return false;
        }
        let mut before = [R::ZERO; 4];
        let mut after = [R::ZERO; 4];
        for (i, (&f, &t)) in current.iter().zip(trial).enumerate() {
            for (f, t) in [
                (f.real_component(), t.real_component()),
                (f.imaginary_component(), t.imaginary_component()),
            ]
            .into_iter()
            .take(if T::IS_REAL { 1 } else { 2 })
            {
                let (ff, tt) = (f * f, t * t);
                if (f != R::ZERO && ff < R::MIN_POSITIVE) || (t != R::ZERO && tt < R::MIN_POSITIVE)
                {
                    return false;
                }
                before[i % 4] += ff;
                after[i % 4] += tt;
            }
        }
        let before = (before[0] + before[1]) + (before[2] + before[3]);
        let after = (after[0] + after[1]) + (after[2] + after[3]);
        let mass = before + after;
        let change = (after - before) * R::HALF;
        let threshold = decrease + allowance;
        let eight = R::TWO * R::TWO * R::TWO;
        // Covers products, both sums, subtraction and the scalar RHS. A loose
        // bound intentionally sends near-boundary cases to the canonical test.
        let relative = (eight * length + eight) * R::EPSILON;
        let margin = relative * mass + eight * R::EPSILON * (decrease.abs() + allowance);
        mass.is_finite() && margin.is_finite() && relative < R::HALF && change < threshold - margin
    }

    /// Minimizer of the cubic interpolant through the last two rejected trials,
    /// or `None` when the cubic has no real minimizer.
    fn cubic(
        &self,
        step: Scaled<R>,
        excess: Scaled<R>,
        previous: R,
        previous_excess: Scaled<R>,
    ) -> Result<Option<Scaled<R>>, SciOxideError> {
        let previous_step = Scaled::from_value(previous)?;
        let width = Scaled::from_value(self.step - previous)?;
        let current = excess.div(step.square()?)?;
        let earlier = previous_excess.div(previous_step.square()?)?;
        let a = current.sub(earlier)?.div(width)?;
        let b = earlier
            .mul(step)?
            .sub(current.mul(previous_step)?)?
            .div(width)?;
        if a.is_zero() {
            return if b.is_zero() {
                Ok(None)
            } else {
                Ok(Some(self.slope.neg().div(b.shifted(1)?)?))
            };
        }
        let three = Scaled::from_value(R::TWO + R::ONE)?;
        let discriminant = b.square()?.sub(three.mul(a)?.mul(self.slope)?)?;
        if discriminant.is_sign_negative() {
            return Ok(None);
        }
        let root = discriminant.sqrt()?;
        // Use the cancellation-free form of `(-b + √disc)/(3a)`.
        Ok(Some(if b.is_sign_negative() {
            root.sub(b)?.div(three.mul(a)?)?
        } else {
            self.slope.neg().div(b.add(root)?)?
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! vector_cases {
        ($name:ident, $r:ty, $large:expr) => {
            mod $name {
                use super::*;
                use scioxide_core::Complex;

                #[test]
                fn native_and_extreme_norms_preserve_range_and_nonzero_values() {
                    assert_eq!(norm(&[3.0 as $r, 4.0]).unwrap(), 5.0);
                    assert_eq!(norm::<$r>(&[]).unwrap(), 0.0);
                    assert!(!norm(&[-0.0 as $r]).unwrap().is_sign_negative());
                    let tiny = <$r>::from_bits(1);
                    assert_eq!(norm(&[tiny]).unwrap(), tiny);
                    let large = <$r>::MAX / 4.0;
                    let measured = norm(&[large, large]).unwrap();
                    assert!((measured / large - 2.0.sqrt()).abs() < 8.0 * <$r>::EPSILON);
                    let complex = norm(&[Complex::new(large, large); 2]).unwrap();
                    assert!((complex / large - 2.0).abs() < 8.0 * <$r>::EPSILON);
                    assert!(norm(&[<$r>::MAX; 4]).is_err());
                    assert_eq!(
                        norm(&[<$r>::NAN]).unwrap_err().kind(),
                        scioxide_core::SciOxideErrorKind::NonFinite
                    );
                }

                #[test]
                fn affine_replays_overflow_and_reports_genuine_overflow() {
                    let max = <$r>::MAX;
                    let value = affine(-max, max, 2.0).unwrap();
                    assert!((value / max - 1.0).abs() < 4.0 * <$r>::EPSILON);
                    let value = affine(
                        Complex::new(-max, -max),
                        Complex::new(max, max),
                        Complex::new(2.0, 0.0),
                    )
                    .unwrap();
                    assert!((value.re / max - 1.0).abs() < 4.0 * <$r>::EPSILON);
                    assert!((value.im / max - 1.0).abs() < 4.0 * <$r>::EPSILON);
                    assert!(affine(0.0, max, 2.0).is_err());
                    assert!(difference(max, -max).is_err());
                    let tiny = <$r>::from_bits(1);
                    assert_eq!(affine(tiny, 0.5, tiny).unwrap(), 2.0 * tiny);
                    let coefficient = ScaledValue::<$r>::from_components(
                        Scaled::from_value(tiny)
                            .unwrap()
                            .mul(Scaled::from_value(1.5).unwrap())
                            .unwrap(),
                        Scaled::ZERO,
                    );
                    let multiplier = (1.0 as $r).scalbn(100);
                    let value = affine_scaled(0.0, coefficient, multiplier).unwrap();
                    let expected = tiny.scalbn(100) * 1.5;
                    assert!((value / expected - 1.0).abs() < 4.0 * <$r>::EPSILON);
                }

                #[test]
                fn compensated_products_keep_cancellation_and_extended_range() {
                    let value = dot(&[$large as $r, 1.0, -$large], &[1.0; 3])
                        .unwrap()
                        .round()
                        .unwrap();
                    assert_eq!(value, 1.0);
                    let max = <$r>::MAX;
                    assert_eq!(
                        dot(&[max, max], &[2.0, -2.0]).unwrap().round().unwrap(),
                        0.0
                    );
                    assert!(dot(&[max], &[max]).unwrap().round().is_err());
                    let value = dot(
                        &[Complex::new(1.0 as $r, 2.0), Complex::new(3.0, -4.0)],
                        &[Complex::new(5.0, -6.0), Complex::new(-7.0, 8.0)],
                    )
                    .unwrap()
                    .round()
                    .unwrap();
                    assert_eq!(value, Complex::new(-60.0, -20.0));
                    let value = dot(
                        &[Complex::new(max, 0.0); 2],
                        &[Complex::new(2.0, 0.0), Complex::new(-2.0, 0.0)],
                    )
                    .unwrap()
                    .round()
                    .unwrap();
                    assert_eq!(value, Complex::new(0.0, 0.0));
                }
            }
        };
    }
    vector_cases!(single, f32, 1e30);
    vector_cases!(double, f64, 1e300);

    fn slope(value: f64) -> Scaled<f64> {
        Scaled::from_value(value).unwrap()
    }

    #[test]
    fn quadratic_backtrack_hits_the_minimizer_of_a_quadratic_merit() {
        // F(x) = x at x = 1 along d = -2: φ(λ) = ½(1 − 2λ)², φ'(0) = −2.
        let mut search = Backtracking::new(slope(-2.0));
        assert!(!search.assess(&[1.0], &[-1.0], 1e-4).unwrap());
        assert_eq!(search.step(), 0.5);
        assert!(search.assess(&[1.0], &[0.0], 1e-4).unwrap());
    }

    #[test]
    fn interpolated_steps_are_clamped_and_cubic_follows_quadratic() {
        // A huge rejected merit puts the quadratic minimizer below 0.1λ.
        let mut search = Backtracking::new(slope(-1.0));
        assert!(!search.assess(&[1.0], &[1e6], 1e-4).unwrap());
        assert_eq!(search.step(), 0.1);
        // A mild rejection puts it above 0.5λ.
        let mut mild = Backtracking::new(slope(-1.0));
        assert!(!mild.assess(&[1.0], &[1.0], 1e-4).unwrap());
        assert_eq!(mild.step(), 0.5);
        // The second rejection interpolates a cubic within [0.1, 0.5]·λ.
        assert!(!search.assess(&[1.0], &[2.0], 1e-4).unwrap());
        assert!((0.01..=0.05).contains(&search.step()), "{}", search.step());
    }

    #[test]
    fn allowance_accepts_bounded_increases_and_extremes_stay_finite() {
        let mut monotone = Backtracking::new(slope(-1e-8));
        assert!(!monotone.assess(&[1.0], &[1.5], 1e-4).unwrap());
        let mut nonmonotone = Backtracking::with_allowance(slope(-1e-8), slope(1.0));
        assert!(nonmonotone.assess(&[1.0], &[1.5], 1e-4).unwrap());
        let negative = Backtracking::with_allowance(slope(-1.0), slope(-1.0));
        assert_eq!(negative.allowance.cmp_value(Scaled::ZERO), Ordering::Equal);
        // Merits near the exponent limits neither overflow nor underflow.
        let mut huge = Backtracking::new(slope(-1e300).mul(slope(1e300)).unwrap());
        assert!(!huge.assess(&[1e300], &[1e301], 1e-4).unwrap());
        assert!(huge.step() > 0.0 && huge.step() < 1.0);
        assert!(relative_length(&[2.0, 0.5], &[1.0, -3.0]) == 3.0);
    }
}
