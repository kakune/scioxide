//! Exact sums of real and complex pair and triple products.

use super::RangeAccumulator;
use super::accumulator::ExactTerm;
use crate::scalar::{FiniteParts, finite_parts};
use crate::{NumericScalar, Scaled, SciOxideError};

/// Exact accumulator for sums of real or complex pair and triple products.
///
/// Complex products are expanded into real and imaginary parts and added
/// without forming any floating-point product. [`Self::finish`] rounds each
/// component once, so each component of the result is correctly rounded. A
/// failed update or merge leaves the state unchanged.
#[derive(Debug, Clone)]
#[must_use]
pub struct ProductSum<S: NumericScalar> {
    real: RangeAccumulator<S::Real>,
    imaginary: RangeAccumulator<S::Real>,
}

impl<S: NumericScalar> Default for ProductSum<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: NumericScalar> ProductSum<S> {
    /// Returns an empty sum, whose value is `+0`.
    pub const fn new() -> Self {
        Self {
            real: RangeAccumulator::new(),
            imaginary: RangeAccumulator::new(),
        }
    }

    fn update_product(&mut self, left: S, right: S, subtract: bool) -> Result<(), SciOxideError> {
        if matches!(
            S::kind(),
            crate::backend::ScalarKind::F32 | crate::backend::ScalarKind::F64
        ) {
            return self.real.update_product(
                left.real_component(),
                right.real_component(),
                subtract,
            );
        }
        let left =
            Self::finite_components(left, "left product factor must have finite components")?;
        let right =
            Self::finite_components(right, "right product factor must have finite components")?;
        self.update_finite_product(left, right, subtract)
    }

    fn finite_components(
        value: S,
        detail: &'static str,
    ) -> Result<[FiniteParts; 2], SciOxideError> {
        Ok([
            finite_parts(value.real_component(), detail)?,
            finite_parts(value.imaginary_component(), detail)?,
        ])
    }

    fn update_finite_product(
        &mut self,
        [left_real, left_imaginary]: [FiniteParts; 2],
        [right_real, right_imaginary]: [FiniteParts; 2],
        subtract: bool,
    ) -> Result<(), SciOxideError> {
        let real = [
            ExactTerm::pair(left_real, right_real, subtract),
            ExactTerm::pair(left_imaginary, right_imaginary, !subtract),
        ];
        let imaginary = [
            ExactTerm::pair(left_real, right_imaginary, subtract),
            ExactTerm::pair(left_imaginary, right_real, subtract),
        ];
        self.real.check_terms(&real)?;
        self.imaginary.check_terms(&imaginary)?;
        self.real.commit_terms(&real);
        self.imaginary.commit_terms(&imaginary);
        Ok(())
    }

    /// Adds the exact product `left · right`.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a factor has a NaN
    ///   or infinite component.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   if the accumulator capacity is exceeded.
    pub fn add_product(&mut self, left: S, right: S) -> Result<(), SciOxideError> {
        self.update_product(left, right, false)
    }

    /// Subtracts the exact product `left · right`.
    ///
    /// # Errors
    ///
    /// The same as [`Self::add_product`].
    pub fn subtract_product(&mut self, left: S, right: S) -> Result<(), SciOxideError> {
        self.update_product(left, right, true)
    }

    fn update_triple_product(
        &mut self,
        first: S,
        second: S,
        third: S,
        subtract: bool,
    ) -> Result<(), SciOxideError> {
        if matches!(
            S::kind(),
            crate::backend::ScalarKind::F32 | crate::backend::ScalarKind::F64
        ) {
            return self.real.update_triple_product(
                first.real_component(),
                second.real_component(),
                third.real_component(),
                subtract,
            );
        }
        let [ar, ai] =
            Self::finite_components(first, "triple-product factors must have finite components")?;
        let second =
            Self::finite_components(second, "triple-product factors must have finite components")?;
        let third =
            Self::finite_components(third, "triple-product factors must have finite components")?;
        if ar.is_one() && ai.significand() == 0 {
            return self.update_finite_product(second, third, subtract);
        }
        let [br, bi] = second;
        let [cr, ci] = third;
        let real = [
            ExactTerm::triple(ar, br, cr, subtract),
            ExactTerm::triple(ar, bi, ci, !subtract),
            ExactTerm::triple(ai, br, ci, !subtract),
            ExactTerm::triple(ai, bi, cr, !subtract),
        ];
        let imaginary = [
            ExactTerm::triple(ar, br, ci, subtract),
            ExactTerm::triple(ar, bi, cr, subtract),
            ExactTerm::triple(ai, br, cr, subtract),
            ExactTerm::triple(ai, bi, ci, !subtract),
        ];
        self.real.check_terms(&real)?;
        self.imaginary.check_terms(&imaginary)?;
        self.real.commit_terms(&real);
        self.imaginary.commit_terms(&imaginary);
        Ok(())
    }

    /// Adds the exact product `first · second · third`.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a factor has a NaN
    ///   or infinite component.
    /// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    ///   if the accumulator capacity is exceeded.
    pub fn add_triple_product(
        &mut self,
        first: S,
        second: S,
        third: S,
    ) -> Result<(), SciOxideError> {
        self.update_triple_product(first, second, third, false)
    }

    /// Subtracts the exact product `first · second · third`.
    ///
    /// # Errors
    ///
    /// The same as [`Self::add_triple_product`].
    pub fn subtract_triple_product(
        &mut self,
        first: S,
        second: S,
        third: S,
    ) -> Result<(), SciOxideError> {
        self.update_triple_product(first, second, third, true)
    }

    /// Adds the exact value of `other` to this sum.
    ///
    /// On error this sum is unchanged.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability) if
    /// repeated merges exceed the accumulator capacity. A reduction over any
    /// input that fits in memory does not.
    pub fn merge(&mut self, other: &Self) -> Result<(), SciOxideError> {
        self.real.check_merge(&other.real)?;
        self.imaginary.check_merge(&other.imaginary)?;
        self.real.commit_merge(&other.real);
        self.imaginary.commit_merge(&other.imaginary);
        Ok(())
    }

    pub(crate) fn scaled_components(&self) -> Result<[Scaled<S::Real>; 2], SciOxideError> {
        Ok([self.real.finish_scaled()?, self.imaginary.finish_scaled()?])
    }

    /// Returns the exact sum with each component rounded once to nearest, ties
    /// to even.
    ///
    /// # Errors
    ///
    /// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
    /// ([`Overflow`](crate::NumericalIssue::Overflow)) if either rounded
    /// component is outside the finite range.
    pub fn finish(&self) -> Result<S, SciOxideError> {
        if matches!(
            S::kind(),
            crate::backend::ScalarKind::F32 | crate::backend::ScalarKind::F64
        ) {
            return Ok(S::from_real(self.real.finish()?));
        }
        Ok(S::from_components(
            self.real.finish()?,
            self.imaginary.finish()?,
        ))
    }
}

#[cfg(test)]
mod tests {

    use super::super::accumulator::ACCUMULATOR_LIMBS;
    use super::*;
    use crate::{Complex, NonFiniteOrigin};

    #[test]
    fn imaginary_merge_failure_preserves_both_components() {
        let mut sum = ProductSum::<Complex<f64>>::new();
        sum.real.add(1.0).unwrap();
        sum.imaginary.positive.words.fill(u64::MAX);
        sum.imaginary.positive.lo = 0;
        sum.imaginary.positive.hi = ACCUMULATOR_LIMBS;
        let previous = sum.clone();
        let mut other = ProductSum::new();
        other.real.add(2.0).unwrap();
        other.imaginary.add(1.0).unwrap();
        assert!(sum.merge(&other).is_err());
        assert_eq!(sum.real, previous.real);
        assert_eq!(sum.imaginary, previous.imaginary);
    }

    #[test]
    fn invalid_complex_components_never_commit_product_terms() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for component in 0..6 {
                let mut values = [Complex::new(0.0, -0.0); 3];
                if component % 2 == 0 {
                    values[component / 2].re = invalid;
                } else {
                    values[component / 2].im = invalid;
                }
                let mut sum = ProductSum::new();
                sum.add_product(Complex::new(1.0, 2.0), Complex::new(3.0, 4.0))
                    .unwrap();
                let previous = sum.clone();
                for subtract in [false, true] {
                    if component < 4 {
                        let error = sum
                            .update_product(values[0], values[1], subtract)
                            .unwrap_err();
                        assert!(matches!(
                            error.view(),
                            crate::SciOxideErrorView::NonFinite {
                                origin: NonFiniteOrigin::Input,
                                ..
                            }
                        ));
                    }
                    let error = sum
                        .update_triple_product(values[0], values[1], values[2], subtract)
                        .unwrap_err();
                    assert!(matches!(
                        error.view(),
                        crate::SciOxideErrorView::NonFinite {
                            origin: NonFiniteOrigin::Input,
                            ..
                        }
                    ));
                    assert_eq!(sum.real, previous.real);
                    assert_eq!(sum.imaginary, previous.imaginary);
                }
            }
        }
    }

    #[test]
    fn capacity_preflight_preserves_every_component() {
        let mut sum = ProductSum::<Complex<f64>>::new();
        sum.real.positive.words.fill(u64::MAX);
        sum.real.positive.lo = 0;
        sum.real.positive.hi = ACCUMULATOR_LIMBS;
        // Room for one real unit, but (1+i)(1-i) adds two units.
        sum.real.positive.words[50] -= 1 << 22;
        let previous = sum.clone();
        assert!(
            sum.add_product(Complex::new(1.0, 1.0), Complex::new(1.0, -1.0))
                .is_err()
        );
        assert_eq!(sum.real, previous.real);
        assert_eq!(sum.imaginary, previous.imaginary);
        assert!(
            sum.add_triple_product(
                Complex::new(1.0, 1.0),
                Complex::new(1.0, -1.0),
                Complex::new(2.0, 0.0)
            )
            .is_err()
        );
        assert_eq!(sum.real, previous.real);
        assert_eq!(sum.imaginary, previous.imaginary);
        assert!(sum.merge(&previous).is_err());
        assert_eq!(sum.real, previous.real);
    }

    #[test]
    fn complex_product_sum_avoids_intermediate_overflow() {
        let maximum = f64::MAX;
        let left = [
            Complex::new(maximum, maximum),
            Complex::new(-maximum, -maximum),
        ];
        let right = [Complex::new(1.0, 1.0), Complex::new(1.0, 1.0)];
        let mut sum = ProductSum::new();
        left.into_iter()
            .zip(right)
            .try_for_each(|(left, right)| sum.add_product(left, right))
            .unwrap();

        assert_eq!(sum.finish().unwrap(), Complex::new(0.0, 0.0));
    }

    #[test]
    fn product_sum_names_the_subtracted_product() {
        let left = Complex::new(f32::MAX, f32::MAX);
        let right = Complex::new(1.0_f32, 1.0);
        let mut difference = ProductSum::new();
        difference.add_product(left, right).unwrap();
        difference.subtract_product(left, right).unwrap();
        assert_eq!(difference.finish().unwrap(), Complex::new(0.0, 0.0));
    }
}
