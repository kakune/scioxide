//! Enclosures of `hypot`, `atan2`, integer and real powers, and logarithms to
//! a given base.
use super::{
    ElementaryEnclosure as E, EnclosureStatus as Status,
    kernel::{self, B, bounds, point, range},
};
use crate::DecoratedInterval as D;
use scioxide_core::{RealScalar, SciOxideError};

fn hypot_point(x: f64, y: f64) -> Result<B, SciOxideError> {
    // Either zero sign takes the same explicit zero result below.
    let maximum = x.max(y);
    if maximum == 0.0 {
        return point(0.0);
    }
    if maximum == f64::INFINITY {
        return range(f64::MAX, f64::INFINITY);
    }
    let k = scioxide_core::backend::standard::exponent(maximum)? - 1;
    let x = kernel::scale(point(x)?, -k)?;
    let y = kernel::scale(point(y)?, -k)?;
    kernel::scale(
        x.square_admitted()?
            .add_admitted(&y.square_admitted()?)?
            .sqrt_admitted()?,
        k,
    )
}

fn angle(y: f64, x: f64) -> Result<Option<B>, SciOxideError> {
    if x == 0.0 && y == 0.0 {
        return Ok(None);
    }
    let pi = kernel::pi()?;
    let half = kernel::scale(pi, -1)?;
    let value = if y == 0.0 {
        if x < 0.0 { pi } else { point(0.0)? }
    } else if x == 0.0 {
        half
    } else if x.is_infinite() && y.is_infinite() {
        if x > 0.0 {
            range(0.0, bounds(half)?.1)?
        } else {
            range(bounds(half)?.0, bounds(pi)?.1)?
        }
    } else if x.is_infinite() {
        if x > 0.0 { point(0.0)? } else { pi }
    } else if y.is_infinite() {
        half
    } else {
        let (a, b) = bounds(point(y.abs())?.div_admitted(&point(x.abs())?)?)?;
        let acute = range(bounds(kernel::atan(a)?)?.0, bounds(kernel::atan(b)?)?.1)?;
        if x < 0.0 {
            pi.sub_admitted(&acute)?
        } else {
            acute
        }
    };
    Ok(Some(if y.is_sign_negative() {
        value.neg_admitted()?
    } else {
        value
    }))
}
fn angle_branch(a: f64, b: f64, c: f64, d: f64) -> Result<(f64, f64), SciOxideError> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (y, x) in [(a, c), (a, d), (b, c), (b, d)] {
        if let Some(value) = angle(y, x)? {
            let (l, h) = bounds(value)?;
            lo = lo.min(l);
            hi = hi.max(h);
        }
    }
    Ok((lo, hi))
}
fn integer_power(mut x: B, mut n: u32) -> Result<B, SciOxideError> {
    let mut result = point(1.0)?;
    while n != 0 {
        if n & 1 != 0 {
            result = result.mul_admitted(&x)?;
        }
        n >>= 1;
        if n != 0 {
            x = x.square_admitted()?;
        }
    }
    Ok(result)
}
fn power_range(a: f64, b: f64, c: f64, d: f64) -> Result<(f64, f64), SciOxideError> {
    let logs = range(bounds(kernel::ln(a)?)?.0, bounds(kernel::ln(b)?)?.1)?;
    bounds(kernel::exp_interval(logs.mul_admitted(&range(c, d)?)?)?)
}
fn include(hull: &mut Option<(f64, f64)>, value: (f64, f64), gaps: &mut bool) {
    *hull = Some(match *hull {
        Some((a, b)) => {
            *gaps |= value.0 > b || value.1 < a;
            (a.min(value.0), b.max(value.1))
        }
        None => value,
    });
}
impl<S: RealScalar> D<S> {
    /// Encloses `{√(x² + y²) : x ∈ self, y ∈ other}`.
    ///
    /// The operands are scaled by powers of two, so the result neither
    /// overflows nor underflows unless the exact value does.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - `NumericalInstability` if an internal enclosure step breaks down.
    pub fn hypot(&self, other: &Self) -> Result<E<S>, SciOxideError> {
        crate::environment::check()?;
        self.hypot_admitted(other)
    }

    /// Enclosure kernel; the public caller checks the floating-point environment.
    pub(crate) fn hypot_admitted(&self, other: &Self) -> Result<E<S>, SciOxideError> {
        let x = self.abs_admitted()?;
        let y = other.abs_admitted()?;
        let Some(((a, b), (c, d))) = x.wide_bounds().zip(y.wide_bounds()) else {
            return Ok(E::empty(Status::EmptyInput));
        };
        E::one(
            range(bounds(hypot_point(a, c)?)?.0, bounds(hypot_point(b, d)?)?.1)?,
            Status::Defined,
            self.decoration().min(other.decoration()),
        )
    }
    /// Encloses the angles `{atan2(y, x) : y ∈ self, x ∈ other, (x, y) ≠ (0, 0)}`
    /// of the points `(x, y)`, with values in `(−π, π]`.
    ///
    /// The origin is excluded from the domain and reported by the status. The
    /// negative real axis has angle `+π`, whatever the signs of zero ends. When
    /// the box `other × self` crosses the negative real axis, the result has two
    /// pieces, one near `−π` and one near `+π`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - `NumericalInstability` if an internal enclosure step breaks down.
    pub fn atan2(&self, other: &Self) -> Result<E<S>, SciOxideError> {
        crate::environment::check()?;
        self.atan2_admitted(other)
    }

    /// Enclosure kernel; the public caller checks the floating-point environment.
    pub(crate) fn atan2_admitted(&self, other: &Self) -> Result<E<S>, SciOxideError> {
        let Some(((a, b), (c, d))) = self.wide_bounds().zip(other.wide_bounds()) else {
            return Ok(E::empty(Status::EmptyInput));
        };
        if a == 0.0 && b == 0.0 && c == 0.0 && d == 0.0 {
            return Ok(E::empty(Status::Undefined));
        }
        let status = if a <= 0.0 && b >= 0.0 && c <= 0.0 && d >= 0.0 {
            Status::PartiallyDefined
        } else {
            Status::Defined
        };
        let mut pieces = [(0.0, 0.0); 2];
        let mut len = 0;
        if a < 0.0 {
            pieces[len] = angle_branch(a, if b < 0.0 { b } else { -0.0 }, c, d)?;
            len += 1;
        }
        if b > 0.0 {
            pieces[len] = angle_branch(if a > 0.0 { a } else { 0.0 }, b, c, d)?;
            len += 1;
        } else if b == 0.0 {
            if c < 0.0 {
                pieces[len] = bounds(kernel::pi()?)?;
                len += 1;
            }
            if a == 0.0 && d > 0.0 {
                pieces[len] = (0.0, 0.0);
                len += 1;
            }
        }
        let decoration = self.decoration().min(other.decoration());
        // The principal angle is discontinuous across the negative real axis.
        let decoration = if c < 0.0 && a <= 0.0 && b >= 0.0 {
            decoration.min(crate::Decoration::Def)
        } else {
            decoration
        };
        E::new(pieces, len, status, decoration, false)
    }
    /// Encloses `{xⁿ : x ∈ self}` for an integer `n`, with `0⁰ = 1`.
    ///
    /// For `n < 0` zero is excluded from the domain, and an odd `n` gives two
    /// pieces when `self` contains both signs. Each endpoint takes at most 32
    /// steps of binary exponentiation for any `i32` exponent.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - `NumericalInstability` if an internal enclosure step breaks down.
    pub fn powi(&self, n: i32) -> Result<E<S>, SciOxideError> {
        crate::environment::check()?;
        self.powi_admitted(n)
    }

    /// Enclosure kernel; the public caller checks the floating-point environment.
    pub(crate) fn powi_admitted(&self, n: i32) -> Result<E<S>, SciOxideError> {
        let Some((a, b)) = self.wide_bounds() else {
            return Ok(E::empty(Status::EmptyInput));
        };
        if n == 0 {
            return E::one(point(1.0)?, Status::Defined, self.decoration());
        }
        if a == 0.0 && b == 0.0 && n < 0 {
            return Ok(E::empty(Status::Undefined));
        }
        let status = if n < 0 && a <= 0.0 && b >= 0.0 {
            Status::PartiallyDefined
        } else {
            Status::Defined
        };
        let magnitude = |lo: f64, hi: f64| -> Result<(f64, f64), SciOxideError> {
            let x = range(lo, hi)?;
            let x = if n < 0 { x.reciprocal_admitted()? } else { x };
            // The lower bound is finite: lo >= 0 here, and for n < 0 the
            // reciprocal rounds 1/hi down to at most MAX because hi > 0.
            let (lo, hi) = bounds(x)?;
            let lower = bounds(integer_power(point(lo)?, n.unsigned_abs())?)?.0;
            let upper = if hi == f64::INFINITY {
                f64::INFINITY
            } else {
                bounds(integer_power(point(hi)?, n.unsigned_abs())?)?.1
            };
            Ok((lower, upper))
        };
        let mut pieces = [(0.0, 0.0); 2];
        let mut len = 0;
        if a < 0.0 {
            let (lo, hi) = magnitude(if b < 0.0 { -b } else { 0.0 }, -a)?;
            pieces[len] = if n & 1 != 0 { (-hi, -lo) } else { (lo, hi) };
            len += 1;
        }
        if b > 0.0 {
            pieces[len] = magnitude(if a > 0.0 { a } else { 0.0 }, b)?;
            len += 1;
        }
        if len == 0 {
            pieces[0] = (0.0, 0.0);
            len = 1;
        }
        E::new(pieces, len, status, self.decoration(), false)
    }
    /// Encloses `{xʸ : x ∈ self, y ∈ exponent}` over the real domain of the
    /// power: `x > 0` with any `y`, `x < 0` with integer `y`, and `x = 0` with
    /// `y ≥ 0`, where `0⁰ = 1`.
    ///
    /// Pairs outside this domain are excluded and reported by the status. A
    /// negative base with many integer exponents gives one piece for each sign
    /// of the result, with
    /// [`ElementaryEnclosure::gaps_filled`](crate::ElementaryEnclosure::gaps_filled)
    /// set. The cost does not depend on the number of integers in `exponent`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - `NumericalInstability` if an internal enclosure step breaks down.
    pub fn powf(&self, exponent: &Self) -> Result<E<S>, SciOxideError> {
        crate::environment::check()?;
        self.powf_admitted(exponent)
    }

    /// Enclosure kernel; the public caller checks the floating-point environment.
    pub(crate) fn powf_admitted(&self, exponent: &Self) -> Result<E<S>, SciOxideError> {
        let Some(((a, b), (c, d))) = self.wide_bounds().zip(exponent.wide_bounds()) else {
            return Ok(E::empty(Status::EmptyInput));
        };
        let zero = a <= 0.0 && b >= 0.0;
        let decoration = self.decoration().min(exponent.decoration());
        // The joint real power extension with 0^0=1 is discontinuous at
        // (0,0), even though the fixed-exponent unary powi(0) is constant.
        let decoration = if zero && c <= 0.0 && d >= 0.0 {
            decoration.min(crate::Decoration::Def)
        } else {
            decoration
        };
        if a == 0.0 && b == 0.0 {
            if d < 0.0 {
                return Ok(E::empty(Status::Undefined));
            }
            let mut pieces = [(0.0, 0.0); 2];
            let mut len = usize::from(d > 0.0);
            if c <= 0.0 {
                pieces[len] = (1.0, 1.0);
                len += 1;
            }
            return E::new(
                pieces,
                len,
                if c < 0.0 {
                    Status::PartiallyDefined
                } else {
                    Status::Defined
                },
                decoration,
                false,
            );
        }
        if c == d
            && let Some(n) = super::integer_exponent(c)
        {
            let mut result = self.powi_admitted(n)?;
            for piece in &mut result.pieces[..result.len] {
                piece.decoration = piece.decoration.min(decoration);
            }
            return Ok(result);
        }
        let mut positive = None;
        let mut negative = None;
        let mut gaps = false;
        if b > 0.0 {
            include(
                &mut positive,
                power_range(if a > 0.0 { a } else { 0.0 }, b, c, d)?,
                &mut gaps,
            );
        }
        if zero && d > 0.0 {
            include(&mut positive, (0.0, 0.0), &mut gaps);
        }
        if zero && c <= 0.0 && d >= 0.0 {
            include(&mut positive, (1.0, 1.0), &mut gaps);
        }
        let first = RealScalar::ceil(c);
        let last = RealScalar::floor(d);
        if a < 0.0 && first <= last {
            let magnitude = power_range(if b < 0.0 { -b } else { 0.0 }, -a, first, last)?;
            let singleton = first == last;
            let even = !singleton || first % 2.0 == 0.0;
            let odd = !singleton || first % 2.0 != 0.0;
            if even {
                include(&mut positive, magnitude, &mut gaps);
            }
            if odd {
                include(&mut negative, (-magnitude.1, -magnitude.0), &mut gaps);
            }
            // More than one exponent per parity can leave arbitrarily many
            // gaps. Conservatively flag this even when ranges happen to overlap.
            gaps |= last - first >= 2.0;
        }
        if positive.is_none() && negative.is_none() {
            return Ok(E::empty(Status::Undefined));
        }
        let partial = (a < 0.0 && (c != d || c != RealScalar::floor(c))) || (zero && c < 0.0);
        let mut pieces = [(0.0, 0.0); 2];
        let mut len = 0;
        for piece in [negative, positive].into_iter().flatten() {
            pieces[len] = piece;
            len += 1;
        }
        E::new(
            pieces,
            len,
            if partial {
                Status::PartiallyDefined
            } else {
                Status::Defined
            },
            decoration,
            gaps,
        )
    }
    /// Encloses `{ln x / ln b : x ∈ self, b ∈ base, x > 0, b > 0, b ≠ 1}`, the
    /// logarithm of `x` to base `b`.
    ///
    /// Pairs outside this domain are excluded and reported by the status. A
    /// base interval that contains 1 gives separate pieces for `b < 1` and
    /// `b > 1`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - `NumericalInstability` if an internal enclosure step breaks down.
    pub fn log(&self, base: &Self) -> Result<E<S>, SciOxideError> {
        crate::environment::check()?;
        self.log_admitted(base)
    }

    /// Enclosure kernel; the public caller checks the floating-point environment.
    pub(crate) fn log_admitted(&self, base: &Self) -> Result<E<S>, SciOxideError> {
        let Some(((a, b), (c, d))) = self.wide_bounds().zip(base.wide_bounds()) else {
            return Ok(E::empty(Status::EmptyInput));
        };
        if b <= 0.0 || d <= 0.0 || (c == 1.0 && d == 1.0) {
            return Ok(E::empty(Status::Undefined));
        }
        let numerator = range(
            bounds(kernel::ln(if a > 0.0 { a } else { 0.0 })?)?.0,
            bounds(kernel::ln(b)?)?.1,
        )?;
        let mut pieces = [(0.0, 0.0); 2];
        let mut len = 0;
        if c < 1.0 {
            let denominator = range(
                bounds(kernel::ln(if c > 0.0 { c } else { 0.0 })?)?.0,
                if d >= 1.0 {
                    -0.0
                } else {
                    bounds(kernel::ln(d)?)?.1
                },
            )?;
            pieces[len] = bounds(numerator.div_admitted(&denominator)?)?;
            len += 1;
        }
        if d > 1.0 {
            let denominator = range(
                if c <= 1.0 {
                    0.0
                } else {
                    bounds(kernel::ln(c)?)?.0
                },
                bounds(kernel::ln(d)?)?.1,
            )?;
            pieces[len] = bounds(numerator.div_admitted(&denominator)?)?;
            len += 1;
        }
        let partial = a <= 0.0 || c <= 0.0 || (c <= 1.0 && d >= 1.0);
        E::new(
            pieces,
            len,
            if partial {
                Status::PartiallyDefined
            } else {
                Status::Defined
            },
            self.decoration().min(base.decoration()),
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_zero_branches<S: RealScalar>() {
        let scalar = |x| S::try_from_f64_lossy(x).unwrap();
        let interval = |a, b| D::new(scalar(a), scalar(b)).unwrap();
        for zero in [-0.0, 0.0] {
            let angles = interval(zero, 1.0).atan2(&interval(-2.0, -1.0)).unwrap();
            assert_eq!(angles.status(), Status::Defined);
            assert_eq!(angles.pieces().len(), 1);
            assert!(angles.pieces()[0].contains_interval(&interval(
                3.0 * core::f64::consts::FRAC_PI_4,
                core::f64::consts::PI,
            )));
            assert!(!angles.contains(S::ZERO));
            assert!(!angles.contains(scalar(-core::f64::consts::PI)));

            let below = interval(-1.0, zero).atan2(&interval(-2.0, -1.0)).unwrap();
            assert_eq!(below.pieces().len(), 2);
            assert!(below.contains(scalar(-core::f64::consts::PI)));
            assert!(below.contains(scalar(core::f64::consts::PI)));
            assert!(!below.contains(S::ZERO));
            let origin = interval(zero, zero).atan2(&interval(zero, zero)).unwrap();
            assert_eq!(origin.status(), Status::Undefined);
            assert!(origin.pieces().is_empty());

            let positive = interval(zero, 1.0);
            let negative = interval(-1.0, zero);
            let zero_base = interval(zero, zero);
            let positive_limit =
                D::from(scioxide_core::ExtendedInterval::new(Some(S::ONE), None).unwrap());
            let negative_limit =
                D::from(scioxide_core::ExtendedInterval::new(None, Some(-S::ONE)).unwrap());
            for n in [-3, -1] {
                let exponent = interval(f64::from(n), f64::from(n));
                for (base, expected) in [(positive, positive_limit), (negative, negative_limit)] {
                    for result in [base.powi(n).unwrap(), base.powf(&exponent).unwrap()] {
                        assert_eq!(result.status(), Status::PartiallyDefined);
                        assert_eq!(result.pieces().len(), 1);
                        assert!(result.pieces()[0].contains_interval(&expected));
                        assert!(!result.contains(S::ZERO));
                    }
                }
                assert_eq!(zero_base.powi(n).unwrap().status(), Status::Undefined);
                assert_eq!(
                    zero_base.powf(&exponent).unwrap().status(),
                    Status::Undefined
                );
            }
            let fractional = interval(-0.5, -0.5);
            let power = positive.powf(&fractional).unwrap();
            assert_eq!(power.status(), Status::PartiallyDefined);
            assert!(power.hull().contains_interval(&positive_limit));
            assert_eq!(
                negative.powf(&fractional).unwrap().status(),
                Status::Undefined
            );
            assert_eq!(
                zero_base.powf(&fractional).unwrap().status(),
                Status::Undefined
            );
            let power = negative.powf(&interval(-3.5, -2.5)).unwrap();
            assert_eq!(power.status(), Status::PartiallyDefined);
            assert!(power.hull().contains_interval(&negative_limit));
        }
    }

    #[test]
    fn signed_zero_atan2_and_power_branches_enclose_the_real_image() {
        signed_zero_branches::<f32>();
        signed_zero_branches::<f64>();
    }
}
