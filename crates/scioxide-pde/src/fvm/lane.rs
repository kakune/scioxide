//! Forward-mode lanes for the nonlinear finite-volume face laws.
//!
//! The FVM operator depends nonlinearly on the sampled diffusion tensors and
//! Robin coefficients (harmonic face elimination, boundary-constrained gradient
//! reconstruction). One generic implementation of those laws runs either on the
//! real scalar itself, which is the primal preparation, or on a [`Tangent`] pair,
//! which propagates one exact directional derivative. Branches, pivots and
//! finiteness checks always act on the primal value, so both lanes take the same
//! branch and the derivative is that of the selected smooth expression.

use core::ops::{Add, Div, Mul, Neg, Sub};

use scioxide_core::{RealScalar, SciOxideError};

/// Arithmetic of one finite-volume lane.
pub(crate) trait Lane:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Real scalar of geometry and primal values.
    type Real: RealScalar + scioxide_core::NumericScalar<Real = Self::Real>;
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// A value with no derivative.
    fn constant(value: Self::Real) -> Self;
    /// The primal value that selects branches.
    fn primal(self) -> Self::Real;
    /// Whether every component is finite.
    fn all_finite(self) -> bool;
    /// Whether every component is exactly zero.
    fn exactly_zero(self) -> bool;
    /// `scale · self`, rejecting an unrepresentable product of finite operands.
    fn scaled(self, scale: Self::Real) -> Result<Self, SciOxideError>;
    /// The operand with the larger primal value (`self` on ties).
    fn larger(self, other: Self) -> Self {
        if self.primal() >= other.primal() {
            self
        } else {
            other
        }
    }
    /// The operand with the smaller primal value (`self` on ties).
    fn smaller(self, other: Self) -> Self {
        if self.primal() <= other.primal() {
            self
        } else {
            other
        }
    }
}

impl<R: RealScalar> Lane for R {
    type Real = R;
    const ZERO: Self = <R as scioxide_core::AlgebraicScalar>::ZERO;
    const ONE: Self = <R as scioxide_core::AlgebraicScalar>::ONE;
    #[inline]
    fn constant(value: R) -> Self {
        value
    }
    #[inline]
    fn primal(self) -> R {
        self
    }
    #[inline]
    fn all_finite(self) -> bool {
        RealScalar::is_finite(self)
    }
    #[inline]
    fn exactly_zero(self) -> bool {
        self == <R as scioxide_core::AlgebraicScalar>::ZERO
    }
    #[inline]
    fn scaled(self, scale: R) -> Result<Self, SciOxideError> {
        scioxide_core::checked_float_product(scale, self)
    }
}

/// A primal value and one directional derivative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tangent<R> {
    pub(crate) value: R,
    pub(crate) derivative: R,
}

impl<R: RealScalar> Add for Tangent<R> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            derivative: self.derivative + other.derivative,
        }
    }
}

impl<R: RealScalar> Sub for Tangent<R> {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            derivative: self.derivative - other.derivative,
        }
    }
}

impl<R: RealScalar> Mul for Tangent<R> {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            derivative: self.derivative * other.value + self.value * other.derivative,
        }
    }
}

impl<R: RealScalar> Div for Tangent<R> {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        Self {
            value,
            derivative: (self.derivative - value * other.derivative) / other.value,
        }
    }
}

impl<R: RealScalar> Neg for Tangent<R> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivative: -self.derivative,
        }
    }
}

impl<R: RealScalar> Lane for Tangent<R> {
    type Real = R;
    const ZERO: Self = Self {
        value: <R as scioxide_core::AlgebraicScalar>::ZERO,
        derivative: <R as scioxide_core::AlgebraicScalar>::ZERO,
    };
    const ONE: Self = Self {
        value: <R as scioxide_core::AlgebraicScalar>::ONE,
        derivative: <R as scioxide_core::AlgebraicScalar>::ZERO,
    };
    #[inline]
    fn constant(value: R) -> Self {
        Self {
            value,
            derivative: <R as scioxide_core::AlgebraicScalar>::ZERO,
        }
    }
    #[inline]
    fn primal(self) -> R {
        self.value
    }
    #[inline]
    fn all_finite(self) -> bool {
        self.value.is_finite() && self.derivative.is_finite()
    }
    #[inline]
    fn exactly_zero(self) -> bool {
        self.value == R::ZERO && self.derivative == R::ZERO
    }
    #[inline]
    fn scaled(self, scale: R) -> Result<Self, SciOxideError> {
        Ok(Self {
            value: scioxide_core::checked_float_product(scale, self.value)?,
            derivative: scioxide_core::checked_float_product(scale, self.derivative)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Lane, Tangent};

    #[test]
    fn tangent_arithmetic_differentiates_rational_expressions() {
        // f(x) = x·x/(1 + x) at x = 2: f = 4/3, f' = (2x(1+x) - x²)/(1+x)² = 8/9.
        let x = Tangent {
            value: 2.0_f64,
            derivative: 1.0,
        };
        let f = x * x / (Tangent::ONE + x);
        assert!((f.value - 4.0 / 3.0).abs() < 1e-15);
        assert!((f.derivative - 8.0 / 9.0).abs() < 1e-15);
        let g = -(x - Tangent::constant(1.0));
        assert_eq!((g.value, g.derivative), (-1.0, -1.0));
        assert_eq!(x.larger(Tangent::constant(3.0)).derivative, 0.0);
        assert_eq!(x.smaller(Tangent::constant(3.0)).derivative, 1.0);
        assert!(x.scaled(f64::MAX).is_err());
        assert!(Tangent::<f64>::ZERO.exactly_zero() && !x.exactly_zero());
    }
}
