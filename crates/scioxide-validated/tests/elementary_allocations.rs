//! Fixed-storage guarantees.
#![cfg(test)]

use scioxide_core::{RealScalar, SciOxideError};
use scioxide_validated::{DecoratedInterval as D, ElementaryEnclosure as E};

fn check<S: RealScalar>() {
    type Unary<S> = fn(&D<S>) -> Result<E<S>, SciOxideError>;
    let unary: [Unary<S>; 21] = [
        D::exp,
        D::exp2,
        D::exp10,
        D::exp_m1,
        D::ln,
        D::log2,
        D::log10,
        D::ln_1p,
        D::sin,
        D::cos,
        D::tan,
        D::asin,
        D::acos,
        D::atan,
        D::sinh,
        D::cosh,
        D::tanh,
        D::asinh,
        D::acosh,
        D::atanh,
        D::cbrt,
    ];
    for (a, b) in [
        (S::ONE, S::TWO),
        (-S::MAX, S::MAX),
        (S::ZERO, S::ZERO.next_after(S::ONE)),
    ] {
        let x = D::new(a, b).unwrap();
        for operation in unary {
            let _ = operation(&x).unwrap();
        }
        (|| -> Result<(), SciOxideError> {
            let _ = x.hypot(&x)?;
            let _ = x.atan2(&x)?;
            let _ = x.powi(i32::MIN)?;
            let _ = x.powf(&x)?;
            let _ = x.log(&x)?;
            Ok(())
        })()
        .unwrap();
    }
}
#[test]
fn fixed_storage_f32() {
    check::<f32>();
}
#[test]
fn fixed_storage_f64() {
    check::<f64>();
}
