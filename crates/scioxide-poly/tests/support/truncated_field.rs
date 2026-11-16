//! Deliberately invalid field division to exercise bounded breakdown detection.
use core::ops::{Add, Div, Mul, Neg, Sub};
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_poly::{ExactCoefficient, ExactField};
pub(crate) static DIVISIONS: AtomicUsize = AtomicUsize::new(0);
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Z(pub(crate) i64);
impl Add for Z {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self(self.0 + b.0)
    }
}
impl Sub for Z {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self(self.0 - b.0)
    }
}
impl Mul for Z {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self(self.0 * b.0)
    }
}
impl Neg for Z {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl Div for Z {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        // A broken implementation of long division fails this bound instead
        // of hanging the test process. No timing or external timeout is used.
        assert!(DIVISIONS.fetch_add(1, Ordering::Relaxed) < 8);
        Self(self.0 / b.0)
    }
}
impl ExactCoefficient for Z {
    fn zero() -> Self {
        Self(0)
    }
    fn one() -> Self {
        Self(1)
    }
    fn from_usize(n: usize) -> Self {
        Self(n as i64)
    }
}
impl ExactField for Z {}
