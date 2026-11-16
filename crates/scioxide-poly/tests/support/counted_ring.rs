//! Ring-only coefficients that count clones during prepared publication.
use crate::test_prime_field::Fp;
use core::ops::{Add, Mul, Neg, Sub};
use scioxide_poly::ExactCoefficient;
use std::cell::Cell;
std::thread_local! { pub(crate) static CLONES: Cell<usize> = const { Cell::new(0) }; }
#[derive(Debug, PartialEq)]
pub(crate) struct Ring(pub(crate) Fp<65537>);
impl Clone for Ring {
    fn clone(&self) -> Self {
        CLONES.with(|n| n.set(n.get() + 1));
        Self(self.0)
    }
}
impl Add for Ring {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self(self.0 + b.0)
    }
}
impl Sub for Ring {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self(self.0 - b.0)
    }
}
impl Mul for Ring {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self(self.0 * b.0)
    }
}
impl Neg for Ring {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl ExactCoefficient for Ring {
    fn zero() -> Self {
        Self(Fp::<65537>(0))
    }
    fn one() -> Self {
        Self(Fp::<65537>(1))
    }
    fn from_usize(n: usize) -> Self {
        Self(Fp::<65537>::from_usize(n))
    }
}
