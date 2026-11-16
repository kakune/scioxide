use super::*;
use crate::util::error_kind;
use crate::{SignalExecutionOptions, SignalLayout};

use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NdView, NdViewMut, ResidualTolerance, SciOxideErrorKind,
    StridedLayout,
};

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "{actual:.17e} != {expected:.17e}, tolerance {tol}"
    );
}
fn cheb(n: usize, x: f64) -> f64 {
    let (mut a, mut b) = (1., x);
    if n == 0 {
        return a;
    }
    for _ in 1..n {
        (a, b) = (b, 2. * x * b - a);
    }
    b
}

mod design;
mod filtering;
