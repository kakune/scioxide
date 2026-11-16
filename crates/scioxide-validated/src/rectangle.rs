//! Shared component interval algebra for affine and linear proofs.
use crate::{DecoratedInterval as D, Decoration};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, checked_midpoint};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Box2<S: RealScalar> {
    pub(crate) re: D<S>,
    pub(crate) im: D<S>,
}

impl<S: RealScalar> D<S> {
    pub(crate) fn with_weaker(self, decoration: Decoration) -> Result<Self, SciOxideError> {
        Self::with_decoration(self.set(), self.decoration().min(decoration))
    }
}
pub(crate) fn constant<S: RealScalar>(x: f64) -> Result<D<S>, SciOxideError> {
    D::result(narrow(x, false)?, narrow(x, true)?, Decoration::Com)
}
/// Round a binary64 endpoint outward in the destination scalar.
pub(crate) fn narrow<S: RealScalar>(x: f64, up: bool) -> Result<S, SciOxideError> {
    if x == f64::INFINITY {
        return Ok(S::INFINITY);
    }
    if x == f64::NEG_INFINITY {
        return Ok(-S::INFINITY);
    }
    let maximum = S::MAX.into_f64();
    if x > maximum {
        return Ok(if up { S::INFINITY } else { S::MAX });
    }
    if x < -maximum {
        return Ok(if up { -S::MAX } else { -S::INFINITY });
    }
    let rounded = S::try_from_f64_lossy(x)?;
    Ok(if up && rounded.into_f64() < x {
        rounded.next_after(S::INFINITY)
    } else if !up && rounded.into_f64() > x {
        rounded.next_after(-S::INFINITY)
    } else {
        rounded
    })
}
pub(crate) fn magnitude<S: RealScalar>(x: D<S>) -> S {
    x.bounds()
        .map_or(S::INFINITY, |(a, b)| a.abs().max(b.abs()))
}
impl<S: RealScalar> Box2<S> {
    pub(crate) fn zero() -> Result<Self, SciOxideError> {
        Self::real(constant(0.0)?)
    }
    pub(crate) fn one() -> Result<Self, SciOxideError> {
        Self::real(constant(1.0)?)
    }
    pub(crate) fn constant(x: f64) -> Result<Self, SciOxideError> {
        Self::real(constant(x)?)
    }
    pub(crate) fn real(re: D<S>) -> Result<Self, SciOxideError> {
        Ok(Self {
            re,
            im: constant(0.0)?,
        })
    }
    pub(crate) fn point<T: NumericScalar<Real = S>>(x: T) -> Result<Self, SciOxideError> {
        Ok(Self {
            re: D::point(x.real_component())?,
            im: D::point(x.imaginary_component())?,
        })
    }
    pub(crate) fn symmetric(real: S, imaginary: S) -> Result<Self, SciOxideError> {
        Ok(Self {
            re: D::result(-real, real, Decoration::Com)?,
            im: D::result(-imaginary, imaginary, Decoration::Com)?,
        })
    }
    pub(crate) fn radii(self) -> Result<(S, S), SciOxideError> {
        Ok((magnitude(self.re), magnitude(self.im)))
    }
    pub(crate) fn im_is_zero(self) -> bool {
        self.im
            .bounds()
            .is_some_and(|(a, b)| a == S::ZERO && b == S::ZERO)
    }
    pub(crate) fn is_zero(self) -> bool {
        self.im_is_zero()
            && self
                .re
                .bounds()
                .is_some_and(|(a, b)| a == S::ZERO && b == S::ZERO)
    }
    pub(crate) fn empty(self) -> bool {
        self.re.set().is_empty() || self.im.set().is_empty()
    }
    pub(crate) fn finite(self) -> bool {
        self.re.set().is_finite() && self.im.set().is_finite() && !self.empty()
    }
    pub(crate) fn contains_zero(self) -> bool {
        self.re.contains(S::ZERO) && self.im.contains(S::ZERO)
    }
    pub(crate) fn decorate(self, decoration: Decoration) -> Result<Self, SciOxideError> {
        Ok(Self {
            re: self.re.with_weaker(decoration)?,
            im: self.im.with_weaker(decoration)?,
        })
    }
    pub(crate) fn decoration(self) -> Decoration {
        self.re.decoration().min(self.im.decoration())
    }
    pub(crate) fn add(self, b: Self) -> Result<Self, SciOxideError> {
        Ok(Self {
            re: self.re.add_admitted(&b.re)?,
            im: self.im.add_admitted(&b.im)?,
        })
    }
    pub(crate) fn neg(self) -> Result<Self, SciOxideError> {
        Ok(Self {
            re: self.re.neg_admitted()?,
            im: self.im.neg_admitted()?,
        })
    }
    pub(crate) fn conj(self) -> Result<Self, SciOxideError> {
        Ok(Self {
            re: self.re,
            im: self.im.neg_admitted()?,
        })
    }
    pub(crate) fn sub(self, b: Self) -> Result<Self, SciOxideError> {
        self.add(b.neg()?)
    }
    pub(crate) fn mul(self, b: Self) -> Result<Self, SciOxideError> {
        Ok(Self {
            re: self
                .re
                .mul_admitted(&b.re)?
                .sub_admitted(&self.im.mul_admitted(&b.im)?)?,
            im: self
                .re
                .mul_admitted(&b.im)?
                .add_admitted(&self.im.mul_admitted(&b.re)?)?,
        })
    }
    pub(crate) fn square(self) -> Result<Self, SciOxideError> {
        Ok(Self {
            re: self
                .re
                .square_admitted()?
                .sub_admitted(&self.im.square_admitted()?)?,
            im: self
                .re
                .mul_admitted(&self.im)?
                .mul_admitted(&constant(2.0)?)?,
        })
    }
    pub(crate) fn scale(self, x: f64) -> Result<Self, SciOxideError> {
        self.mul(Self::constant(x)?)
    }
    /// Enclose `1/z = conj(z)/|z|²` over the box.
    ///
    /// A complex box whose largest endpoint magnitude `m` satisfies `2^(e−1) ≤ m < 2^e`
    /// with `|e|` above a quarter of the largest finite exponent is scaled first:
    /// `1/z = 2^−e · conj(w)/|w|²` with `w = 2^−e z`, so the squared modulus neither
    /// overflows nor underflows. Boxes containing zero or with infinite endpoints, and
    /// boxes in the central exponent range, use the direct formula.
    pub(crate) fn reciprocal(self, complex: bool) -> Result<Self, SciOxideError> {
        if !complex {
            return Self::real(self.re.reciprocal_admitted()?);
        }
        let largest = magnitude(self.re).max(magnitude(self.im));
        if !largest.is_finite() || self.contains_zero() {
            return self.conjugate_over_norm();
        }
        let exponent = scioxide_core::backend::standard::exponent(largest)?;
        if exponent.abs() <= scioxide_core::backend::standard::exponent(S::MAX)? / 4 {
            return self.conjugate_over_norm();
        }
        crate::interval_math::scale(
            crate::interval_math::scale(self, -exponent)?.conjugate_over_norm()?,
            -exponent,
        )
    }
    fn conjugate_over_norm(self) -> Result<Self, SciOxideError> {
        let denominator = self
            .re
            .square_admitted()?
            .add_admitted(&self.im.square_admitted()?)?;
        Ok(Self {
            re: self.re.div_admitted(&denominator)?,
            im: self.im.neg_admitted()?.div_admitted(&denominator)?,
        })
    }
    pub(crate) fn div(self, b: Self, complex: bool) -> Result<Self, SciOxideError> {
        if !complex {
            return Self::real(self.re.div_admitted(&b.re)?);
        }
        self.mul(b.reciprocal(true)?)
    }
    pub(crate) fn powi(self, exponent: i32, complex: bool) -> Result<Self, SciOxideError> {
        if !complex {
            return Self::real(self.re.powi_admitted(exponent)?.hull());
        }
        let mut power = exponent.unsigned_abs();
        let mut base = if exponent < 0 {
            self.reciprocal(true)?
        } else {
            self
        };
        let mut result = Self::one()?;
        while power != 0 {
            if power & 1 != 0 {
                result = result.mul(base)?;
            }
            power >>= 1;
            if power != 0 {
                base = base.square()?;
            }
        }
        Ok(result)
    }
    pub(crate) fn times_i(self) -> Result<Self, SciOxideError> {
        Ok(Self {
            re: self.im.neg_admitted()?,
            im: self.re,
        })
    }
    pub(crate) fn modulus(self) -> Result<D<S>, SciOxideError> {
        Ok(self.re.hypot_admitted(&self.im)?.hull())
    }
    /// Extract a finite componentwise midpoint without constructing a residual.
    pub(crate) fn center<T: NumericScalar<Real = S>>(self) -> Result<Option<T>, SciOxideError> {
        if !self.finite() {
            return Ok(None);
        }
        let (Some((rl, rh)), Some((il, ih))) = (self.re.bounds(), self.im.bounds()) else {
            return Ok(None);
        };
        Ok(Some(T::from_components(
            checked_midpoint(rl, rh)?,
            checked_midpoint(il, ih)?,
        )))
    }
    /// Compute a finite center and symmetric residual box, returning `None` for
    /// unbounded or empty boxes.
    pub(crate) fn midpoint<T: NumericScalar<Real = S>>(
        self,
    ) -> Result<Option<(T, Self)>, SciOxideError> {
        let Some(center) = self.center::<T>()? else {
            return Ok(None);
        };
        let delta = self.sub(Self::point(center)?)?;
        let (r, i) = delta.radii()?;
        Ok(Some((center, Self::symmetric(r, i)?)))
    }
    pub(crate) fn symmetric_hull(self) -> Result<Self, SciOxideError> {
        let (r, i) = self.radii()?;
        Self::symmetric(r, i)
    }
    pub(crate) fn hull(self, b: Self) -> Self {
        Self {
            re: self.re.hull(&b.re),
            im: self.im.hull(&b.im),
        }
    }
    /// Componentwise set intersection, or `None` when a component is empty.
    pub(crate) fn intersection(self, b: Self) -> Option<Self> {
        let re = self.re.set().intersection(&b.re.set());
        let im = self.im.set().intersection(&b.im.set());
        (!re.is_empty() && !im.is_empty()).then(|| Self {
            re: D::from(re),
            im: D::from(im),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    #[test]
    fn directed_constants_cover_conversion_boundaries() {
        for x in [
            -0.0,
            0.0,
            0.1,
            -0.1,
            f64::from(f32::MAX),
            -f64::from(f32::MAX),
            f64::from_bits(1),
            -f64::from_bits(1),
        ] {
            let (lo, hi) = constant::<f32>(x).unwrap().bounds().unwrap();
            assert!(f64::from(lo) <= x && x <= f64::from(hi));
            if x == 0.0 {
                assert_eq!(f64::from(lo).to_bits(), (-0.0_f64).to_bits());
                assert_eq!(f64::from(hi).to_bits(), 0.0_f64.to_bits());
                assert_eq!(
                    f64::from(narrow::<f32>(x, false).unwrap()).to_bits(),
                    x.to_bits()
                );
            }
        }
        assert_eq!(narrow::<f32>(f64::MAX, false).unwrap(), f32::MAX);
        assert_eq!(narrow::<f32>(f64::MAX, true).unwrap(), f32::INFINITY);
        assert_eq!(narrow::<f32>(-f64::MAX, false).unwrap(), f32::NEG_INFINITY);
        assert_eq!(narrow::<f32>(-f64::MAX, true).unwrap(), -f32::MAX);
        assert_eq!(narrow::<f32>(f64::INFINITY, false).unwrap(), f32::INFINITY);
        assert_eq!(
            narrow::<f32>(f64::NEG_INFINITY, true).unwrap(),
            f32::NEG_INFINITY
        );
        assert!(narrow::<f32>(f64::NAN, true).is_err());
    }

    fn scaled_reciprocal<S: RealScalar>(exponents: &[i32]) {
        // 1/(2^k (1 + i)) = 2^(−k−1) (1 − i) is exact at every listed scale.
        for &k in exponents {
            let z = Box2::point(Complex::new(S::ONE.scalbn(k), S::ONE.scalbn(k))).unwrap();
            let r = z.reciprocal(true).unwrap();
            let value = S::ONE.scalbn(-k - 1);
            for (part, expected) in [(r.re, value), (r.im, -value)] {
                let (lo, hi) = part.bounds().unwrap();
                assert!(lo <= expected && expected <= hi, "{k}: {r:?}");
                assert!(
                    hi - lo <= S::EPSILON * (S::TWO + S::TWO) * value,
                    "{k}: {r:?}"
                );
            }
        }
        // Boxes containing zero or with infinite endpoints keep the direct formula.
        let wide = Box2 {
            re: D::new(-S::ONE, S::ONE).unwrap(),
            im: D::point(S::ZERO).unwrap(),
        };
        assert!(!wide.reciprocal(true).unwrap().finite());
        let unbounded = Box2::real(D::from(
            scioxide_core::ExtendedInterval::new(Some(S::ONE), None).unwrap(),
        ))
        .unwrap();
        let r = unbounded.reciprocal(true).unwrap();
        assert!(r.re.contains(S::ZERO) && r.re.contains(S::ONE));
    }

    #[test]
    fn complex_reciprocals_scale_extreme_boxes() {
        scaled_reciprocal::<f32>(&[-100, -60, -33, 0, 33, 60, 100]);
        scaled_reciprocal::<f64>(&[-900, -600, -257, 0, 257, 600, 900]);
    }

    #[test]
    fn centers_cover_extremes_and_absent_finite_midpoints() {
        let adjacent = 1.0_f64.next_after(f64::INFINITY);
        for (lo, hi, expected) in [
            (-f64::MAX, f64::MAX, 0.0),
            (1.0, adjacent, 1.0),
            (0.0, f64::from_bits(1), 0.0),
            (f64::MAX, f64::MAX, f64::MAX),
        ] {
            let value = Box2::real(D::new(lo, hi).unwrap()).unwrap();
            let center = value.center::<Complex<f64>>().unwrap().unwrap();
            assert!((center.re - expected).abs() <= f64::EPSILON);
            assert_eq!(center.im, 0.0);
            let (same, residual) = value.midpoint::<Complex<f64>>().unwrap().unwrap();
            assert_eq!(center, same);
            let reconstructed = residual.add(Box2::point(center).unwrap()).unwrap();
            assert!(reconstructed.re.contains_interval(&value.re));
        }
        for re in [D::<f64>::empty(), D::entire()] {
            let value = Box2::real(re).unwrap();
            assert!(value.center::<f64>().unwrap().is_none());
            assert!(value.midpoint::<f64>().unwrap().is_none());
        }
    }
}
