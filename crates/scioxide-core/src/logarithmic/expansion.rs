//! Fixed four-component arithmetic for logarithmic reconstruction.
//!
//! Components are nonoverlapping and ordered by magnitude. TwoSum insertion
//! keeps widely separated residuals (including subnormals); this is not a
//! fixed absolute grid. The bounded exponential reduction uses |r| <= ln(2)/512,
//! a Taylor polynomial through degree 24, and eight exact-power squarings.
//! Logarithms use three residual Newton corrections. See Hida, Li and Bailey,
//! "Library for Double-Double and Quad-Double Arithmetic", sections 2 and 5.

use crate::{RangeAccumulator, RealScalar, SciOxideError};

#[derive(Clone, Copy, Debug)]
pub(super) struct Expansion([f64; 4]);

impl Expansion {
    pub(super) const ZERO: Self = Self([0.0; 4]);
    pub(super) const ONE: Self = Self([1.0, 0.0, 0.0, 0.0]);
    // Successive nearest-even residuals of ln(2), independently reproduced by
    // the MPFR logarithmic fixture generator. Error is less than 2^-217.
    const LN_2: Self = Self([
        core::f64::consts::LN_2,
        2.319_046_813_846_299_6e-17,
        5.707_708_438_416_212e-34,
        -3.582_432_210_601_811_4e-50,
    ]);

    pub(super) fn from(value: f64) -> Self {
        Self([value, 0.0, 0.0, 0.0])
    }

    pub(super) fn leading(self) -> f64 {
        self.0[0]
    }

    #[cfg(test)]
    pub(super) fn absolute_upper(self) -> f64 {
        self.0
            .into_iter()
            .fold(0.0, |bound, part| super::add_up(bound, part.abs()))
    }

    pub(super) fn rounded(self) -> f64 {
        if self.0[1..].iter().all(|&part| part == 0.0) {
            return self.0[0];
        }
        self.0[0] + (self.0[1] + (self.0[2] + self.0[3]))
    }

    pub(super) fn round_to<S: RealScalar>(self) -> Result<S, SciOxideError> {
        if self.leading() == 0.0 {
            return Ok(if self.leading().is_sign_negative() {
                S::NEG_ZERO
            } else {
                S::ZERO
            });
        }
        let mut sum = RangeAccumulator::<f64>::new();
        for part in self.0 {
            sum.add(part)?;
        }
        sum.finish_to()
    }

    fn insert(mut self, mut value: f64) -> Self {
        for part in &mut self.0 {
            let (hi, lo) = crate::arithmetic::two_sum(*part, value);
            *part = hi;
            value = lo;
        }
        // Compress from the least significant component. This also moves a
        // surviving low component to the front after exact cancellation.
        let mut parts = [0.0; 4];
        let mut q = self.0[3];
        let mut end = 3;
        for &part in self.0[..3].iter().rev() {
            let (hi, lo) = crate::arithmetic::two_sum(part, q);
            if lo != 0.0 {
                parts[end] = lo;
                end -= 1;
            }
            q = hi;
        }
        parts[end] = q;
        parts.copy_within(end.., 0);
        parts[4 - end..].fill(0.0);
        Self(parts)
    }

    pub(super) fn add(self, rhs: Self) -> Self {
        rhs.0.into_iter().fold(self, Self::insert)
    }

    pub(super) fn neg(self) -> Self {
        Self(self.0.map(|x| -x))
    }

    pub(super) fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    pub(super) fn scale(self, exponent: i32) -> Self {
        Self(self.0.map(|x| RealScalar::scalbn(x, exponent)))
    }

    pub(super) fn mul(self, rhs: Self) -> Self {
        let mut result = Self::ZERO;
        // Insert small products first so their sum can carry into the retained
        // components. Factors here are bounded; external weights are instead
        // multiplied exactly by RangeAccumulator::add_scaled_product.
        for &a in self.0.iter().rev() {
            for &b in rhs.0.iter().rev() {
                if a == 0.0 || b == 0.0 {
                    continue;
                }
                let product = a * b;
                let residual = RealScalar::mul_add(a, b, -product);
                result = result.insert(residual).insert(product);
            }
        }
        result
    }

    pub(super) fn div(self, rhs: Self) -> Self {
        let mut residual = self;
        let mut result = Self::ZERO;
        for _ in 0..4 {
            let q = Self::from(residual.leading() / rhs.leading());
            result = result.add(q);
            residual = residual.sub(rhs.mul(q));
        }
        result
    }

    pub(super) fn log_two(exponent: i32) -> Self {
        Self::LN_2.mul(Self::from(f64::from(exponent)))
    }

    /// Mantissa and separate binary exponent; the caller bounds x to [-1200, 1].
    pub(super) fn exp_scaled(self) -> (Self, i32) {
        if self.leading().abs() < 256.0 * f64::MIN_POSITIVE {
            return (Self::ONE.add(self), 0);
        }
        let exponent = RealScalar::round(self.leading() * core::f64::consts::LOG2_E) as i32;
        let r = self.sub(Self::log_two(exponent)).scale(-8);
        let mut term = r;
        let mut correction = r;
        for degree in 2..=24 {
            term = term.mul(r).div(Self::from(f64::from(degree)));
            correction = correction.add(term);
        }
        // expm1(2r) = expm1(r) * (2 + expm1(r)); retaining the correction
        // separately avoids rounding exp(tiny) to one before cancellation.
        for _ in 0..8 {
            correction = correction.mul(Self::from(2.0).add(correction));
        }
        (Self::ONE.add(correction), exponent)
    }

    pub(super) fn exp(self) -> Self {
        if self.leading() < -1200.0 {
            return Self::ZERO;
        }
        let (value, exponent) = self.exp_scaled();
        value.scale(exponent)
    }

    /// Log of a positive expansion with leading component in [1/2, 4].
    pub(super) fn ln(self) -> Self {
        let mut log = Self::from(RealScalar::ln(self.leading()));
        for _ in 0..3 {
            let residual = self.mul(log.neg().exp()).sub(Self::ONE);
            log = log.add(residual);
        }
        log
    }

    pub(super) fn sqrt(self) -> Self {
        let mut root = Self::from(RealScalar::sqrt(self.leading()));
        for _ in 0..3 {
            root = root.add(self.div(root)).scale(-1);
        }
        root
    }

    pub(super) fn accumulate(
        self,
        sum: &mut RangeAccumulator<f64>,
        weight: f64,
        exponent: i32,
    ) -> Result<(), SciOxideError> {
        for part in self.0 {
            sum.add_scaled_product(weight, part, exponent)?;
        }
        Ok(())
    }

    pub(super) fn accumulate_weight(
        self,
        sum: &mut RangeAccumulator<f64>,
        weight: Self,
        exponent: i32,
    ) -> Result<(), SciOxideError> {
        for part in weight.0 {
            self.accumulate(sum, part, exponent)?;
        }
        Ok(())
    }

    // Accumulate 2*w*exp(x) around the exact unit, keeping the constant,
    // linear and quadratic terms separate. For |x|<=2^-27 the residual
    // coefficient 2*(exp(x)-1-x)/x² has a rapidly convergent series. Binary
    // scaling of x² precedes multiplication, so subnormal curvature survives.
    pub(super) fn accumulate_centered(
        sum: &mut RangeAccumulator<f64>,
        weight: Self,
        x: f64,
    ) -> Result<(), SciOxideError> {
        if weight.leading() == 0.0 {
            return Ok(());
        }
        let mut coefficient = Self::ONE;
        if x.abs() >= RealScalar::scalbn(1.0_f64, -600) {
            let mut term = Self::ONE;
            for degree in 3..=10 {
                term = term.mul(Self::from(x)).div(Self::from(f64::from(degree)));
                coefficient = coefficient.add(term);
            }
        }
        let scaled = crate::Scaled::from_value(x)?;
        let mantissa = Self::from(scaled.significand());
        let quadratic = mantissa.mul(mantissa).mul(coefficient);
        let exponent = 2 * scaled.exponent() as i32;
        for part in weight.0 {
            sum.add_product(2.0, part)?;
            sum.add_triple_product(2.0, part, x)?;
            quadratic.accumulate(sum, part, exponent)?;
        }
        Ok(())
    }

    pub(super) fn from_sum(sum: &RangeAccumulator<f64>) -> (Self, i32) {
        let (parts, exponent) = sum.expansion();
        (Self(parts), exponent)
    }

    // Subtract the exact unit BEFORE extracting components, so a residual
    // below the IEEE grid can still determine the final zero sign.
    pub(super) fn small_real_log(
        anchor: f64,
        sum: &RangeAccumulator<f64>,
        scale: i32,
    ) -> Result<Option<Self>, SciOxideError> {
        if anchor.abs() >= RealScalar::scalbn(1.0_f64, -500) {
            return Ok(None);
        }
        let negative = sum.sign().is_lt();
        let mut residual = sum.clone();
        let unit = RealScalar::scalbn(1.0_f64, -scale);
        residual.add(if negative { unit } else { -unit })?;
        let (mut delta, exponent) = Self::from_sum(&residual);
        let exponent = exponent + scale;
        if delta.leading() == 0.0 {
            return Ok(Some(Self::from(anchor + 0.0)));
        }
        if exponent >= -27 {
            return Ok(None);
        }
        if negative {
            delta = delta.neg();
        }
        if exponent < -1075 && anchor == 0.0 {
            return Ok(Some(Self::from(0.0_f64.copysign(delta.leading()))));
        }
        if exponent < -1022 && anchor == 0.0 {
            // Round on the destination subnormal grid, keeping low components
            // in units of MIN_SUBNORMAL. At an exact midpoint log1p(d)<d
            // for either sign, including when d² itself cannot fit binary64.
            let units = delta.scale(exponent + 1074);
            let correction = units.mul(delta.scale(exponent)).scale(-1);
            let logarithm = units.sub(correction);
            let mut floor = RealScalar::floor(logarithm.leading());
            let mut fraction = logarithm.sub(Self::from(floor));
            if fraction.leading() < 0.0 {
                floor -= 1.0;
                fraction = fraction.add(Self::ONE);
            }
            if fraction.sub(Self::from(0.5)).leading() > 0.0 {
                floor += 1.0;
            }
            return Ok(Some(Self::from(if floor == 0.0 {
                0.0_f64.copysign(delta.leading())
            } else {
                RealScalar::scalbn(floor, -1074)
            })));
        }
        let raised = delta.scale(exponent + 512);
        let ordinary = delta.scale(exponent);
        // log1p(d)/d = 1-d/2+d²/3-... . For |d|<2^-27 the
        // remainder after ten terms is <2^-270 relative to d. Factoring
        // d keeps both a tiny positive correction and 1-d's borrow intact.
        let mut factor = Self::ONE;
        let mut term = Self::ONE;
        for degree in 2..=10 {
            term = term.mul(ordinary.neg());
            factor = factor.add(term.div(Self::from(f64::from(degree))));
        }
        let logarithm = raised.mul(factor);
        let combined = Self::from(RealScalar::scalbn(anchor, 512)).add(logarithm);
        Ok(Some(combined.scale(-512)))
    }

    /// Reconstruct only after the complete logarithmic correction is available.
    pub(super) fn reconstruct(anchor: Self, magnitude: Self, exponent: i32) -> Self {
        let correction = magnitude.ln().add(Self::log_two(exponent));
        // If the anchor is this large, every possible window correction is
        // below half an ULP. Avoid overflowing TwoSum's intermediate at MAX.
        if correction.leading() == 0.0 || anchor.leading().abs() > 73_786_976_294_838_206_464.0 {
            anchor
        } else {
            anchor.add(correction)
        }
    }
}
