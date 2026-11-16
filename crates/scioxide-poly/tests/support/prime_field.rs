//! Bounded prime fields and field-operation counts for independent tests.
#![allow(dead_code, clippy::suspicious_arithmetic_impl)]
use core::ops::{Add, Div, Mul, Neg, Sub};
use scioxide_poly::{ExactCoefficient, ExactField};
use std::cell::Cell;
std::thread_local! { static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) }; }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fp<const P: u64>(pub(crate) u64);
impl<const P: u64> Add for Fp<P> {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self((self.0 + b.0) % P)
    }
}
impl<const P: u64> Sub for Fp<P> {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self((self.0 + P - b.0) % P)
    }
}
impl<const P: u64> Neg for Fp<P> {
    type Output = Self;
    fn neg(self) -> Self {
        Self((P - self.0) % P)
    }
}
impl<const P: u64> Mul for Fp<P> {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        COUNTS.with(|c| {
            let (m, d) = c.get();
            c.set((m + 1, d));
        });
        Self(self.0 * b.0 % P)
    }
}
impl<const P: u64> Div for Fp<P> {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        assert_ne!(b.0, 0);
        COUNTS.with(|c| {
            let (m, d) = c.get();
            c.set((m, d + 1));
        });
        let (mut n, mut power, mut inverse) = (P - 2, b.0, 1);
        while n != 0 {
            if n & 1 != 0 {
                inverse = inverse * power % P;
            }
            power = power * power % P;
            n >>= 1;
        }
        Self(self.0 * inverse % P)
    }
}
impl<const P: u64> ExactCoefficient for Fp<P> {
    fn zero() -> Self {
        Self(0)
    }
    fn one() -> Self {
        Self(1)
    }
    fn from_usize(n: usize) -> Self {
        Self((n % P as usize) as u64)
    }
}
impl<const P: u64> ExactField for Fp<P> {}
pub(crate) fn measured<T>(operation: impl FnOnce() -> T) -> (T, (usize, usize)) {
    COUNTS.with(|c| c.set((0, 0)));
    let result = operation();
    (result, COUNTS.with(Cell::get))
}
