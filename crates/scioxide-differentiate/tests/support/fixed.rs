// Test scaffolding for cached-base numerical regressions. No numerical kernels.
use scioxide_core::{Bounds, RealScalar, SciOxideError};
use scioxide_differentiate::*;

pub(crate) fn gradient<S: RealScalar, F: FnMut(&[S]) -> Result<S, SciOxideError>>(
    options: &FiniteDifference<S>,
    bounds: Option<&Bounds<S>>,
    x: &[S],
    base: Option<S>,
    mut function: F,
) -> Result<DifferentiationResult<S>, SciOxideError> {
    options.gradient(
        |x: &[S], y: &mut [S]| {
            y[0] = function(x)?;
            Ok(())
        },
        x,
        bounds,
        base,
    )
}

pub(crate) fn jacobian<S: RealScalar, F: FnMut(&[S], &mut [S]) -> Result<(), SciOxideError>>(
    options: &FiniteDifference<S>,
    bounds: Option<&Bounds<S>>,
    x: &[S],
    outputs: usize,
    base: Option<&[S]>,
    function: F,
) -> Result<DifferentiationResult<S>, SciOxideError> {
    options.jacobian(function, x, outputs, bounds, base)
}

pub(crate) fn hessian<S: RealScalar, F: FnMut(&[S]) -> Result<S, SciOxideError>>(
    options: &FiniteDifference<S>,
    bounds: Option<&Bounds<S>>,
    x: &[S],
    base: Option<S>,
    mut function: F,
) -> Result<DifferentiationResult<S>, SciOxideError> {
    options.hessian(
        |x: &[S], y: &mut [S]| {
            y[0] = function(x)?;
            Ok(())
        },
        x,
        bounds,
        base,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn jvp<S: RealScalar, F: FnMut(&[S], &mut [S]) -> Result<(), SciOxideError>>(
    options: &FiniteDifference<S>,
    bounds: Option<&Bounds<S>>,
    policy: ZeroDirectionPolicy,
    x: &[S],
    direction: &[S],
    outputs: usize,
    base: Option<&[S]>,
    function: F,
) -> Result<DifferentiationResult<S>, SciOxideError> {
    options
        .clone()
        .with_zero_direction(policy)
        .jvp(function, x, direction, outputs, bounds, base)
}
