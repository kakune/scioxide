//! Bit contracts at the exponent limits, plus fixed-storage primitive arithmetic.
#![cfg(test)]

use scioxide_core::RealScalar;
use scioxide_validated::{DecoratedInterval as D, Decoration};

fn bounds<S: RealScalar>(x: D<S>, lo: S, hi: S) {
    let (a, b) = x.set().bounds().unwrap();
    assert_eq!(
        a.unwrap_or(-S::INFINITY).into_f64().to_bits(),
        lo.into_f64().to_bits()
    );
    assert_eq!(
        b.unwrap_or(S::INFINITY).into_f64().to_bits(),
        hi.into_f64().to_bits()
    );
}
fn boundaries<S: RealScalar>() {
    let eta = S::MIN_SUBNORMAL;
    let zero = D::point(S::ZERO).unwrap();
    let tiny = D::point(eta).unwrap();
    let maximum = D::point(S::MAX).unwrap();
    let two = D::point(S::TWO).unwrap();
    let half = D::point(S::ONE / S::TWO).unwrap();
    {
        bounds(tiny.add(&tiny).unwrap(), eta * S::TWO, eta * S::TWO);
        bounds(zero.sub(&tiny).unwrap(), -eta, -eta);
        bounds(
            D::point(S::MIN_POSITIVE).unwrap().mul(&half).unwrap(),
            S::MIN_POSITIVE / S::TWO,
            S::MIN_POSITIVE / S::TWO,
        );
        bounds(tiny.mul(&half).unwrap(), -S::ZERO, eta);
        bounds(D::point(-eta).unwrap().mul(&half).unwrap(), -eta, S::ZERO);
        bounds(tiny.div(&two).unwrap(), -S::ZERO, eta);
        bounds(D::point(-eta).unwrap().div(&two).unwrap(), -eta, S::ZERO);
        bounds(maximum.add(&maximum).unwrap(), S::MAX, S::INFINITY);
        bounds(
            D::point(-S::MAX).unwrap().sub(&maximum).unwrap(),
            -S::INFINITY,
            -S::MAX,
        );
        bounds(maximum.mul(&two).unwrap(), S::MAX, S::INFINITY);
        bounds(
            D::point(-S::MAX).unwrap().mul(&two).unwrap(),
            -S::INFINITY,
            -S::MAX,
        );
        bounds(maximum.div(&half).unwrap(), S::MAX, S::INFINITY);
        bounds(
            D::point(-S::MAX).unwrap().div(&half).unwrap(),
            -S::INFINITY,
            -S::MAX,
        );
        bounds(maximum.square().unwrap(), S::MAX, S::INFINITY);
        bounds(tiny.square().unwrap(), -S::ZERO, eta);
        let power = if S::MANTISSA_DIGITS == 24 { -74 } else { -537 };
        let squared = if S::MANTISSA_DIGITS == 24 {
            eta * S::TWO
        } else {
            eta
        };
        bounds(
            D::point(S::ONE.scalbn(power)).unwrap().square().unwrap(),
            squared,
            squared,
        );
        for x in [
            zero.add(&zero).unwrap(),
            zero.sub(&zero).unwrap(),
            zero.mul(&maximum).unwrap(),
            zero.div(&two).unwrap(),
            zero.sqrt().unwrap(),
            zero.square().unwrap(),
            zero.neg().unwrap(),
        ] {
            bounds(x, -S::ZERO, S::ZERO);
        }
        for x in [tiny.sqrt().unwrap(), maximum.sqrt().unwrap()] {
            let (lo, hi) = x.set().bounds().unwrap();
            assert!(lo.unwrap().is_normal() && hi.unwrap().is_normal());
            assert_eq!(x.decoration(), Decoration::Com);
        }
    };
}
#[test]
fn exponent_boundaries_f32() {
    boundaries::<f32>();
}
#[test]
fn exponent_boundaries_f64() {
    boundaries::<f64>();
}
