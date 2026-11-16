// Bounded rational reference field shared by tests and runnable examples.
use core::ops::{Add, Div, Mul, Neg, Sub};
use scioxide_poly::{ExactCoefficient, ExactField};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Q(pub(crate) i128, pub(crate) i128);
impl Q {
    pub(crate) fn new(a: i128, b: i128) -> Self {
        assert_ne!(b, 0);
        let (mut x, mut y) = (a.abs(), b.abs());
        while y != 0 {
            (x, y) = (y, x % y);
        }
        let g = x * b.signum();
        Self(a / g, b / g)
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl Add for Q {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::new(self.0 * b.1 + b.0 * self.1, self.1 * b.1)
    }
}
impl Neg for Q {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0, self.1)
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl Sub for Q {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
impl Mul for Q {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self::new(self.0 * b.0, self.1 * b.1)
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div for Q {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        Self::new(self.0 * b.1, self.1 * b.0)
    }
}
impl ExactCoefficient for Q {
    fn zero() -> Self {
        Self(0, 1)
    }
    fn one() -> Self {
        Self(1, 1)
    }
    fn from_usize(n: usize) -> Self {
        Self(n as i128, 1)
    }
}
impl ExactField for Q {}
