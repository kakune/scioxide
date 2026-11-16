//! Measures use exact determinant accumulation followed by rounded magnitudes.
//! A valid mesh can still have a measure that is not representable.

use crate::{Point, predicates::determinant_magnitude};
use scioxide_core::{
    NumericalIssue, RealScalar, ScaledNorm, SciOxideError, checked_difference, checked_scalbn,
};

fn finish<S: RealScalar>(mantissa: f64, exponent: i32) -> Result<S, SciOxideError> {
    // Round in the destination format after rescaling in binary64 when it fits;
    // this also preserves values subnormal in f32 but normal in binary64.
    let value = checked_scalbn(mantissa, exponent)?;
    // `value` is finite and nonnegative. The only possible narrowing failure
    // is destination overflow; report the numerical operation, not conversion.
    let result = S::try_from_f64_lossy(value).map_err(|_| {
        SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            "positive measure exceeds scalar range",
        )
    })?;
    if result == S::ZERO {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Underflow,
            "positive measure rounds to zero",
        ));
    }
    Ok(result)
}

pub(crate) fn cell<S: RealScalar, const D: usize>(
    points: &[Point<S, D>],
    vertices: &[usize],
) -> Result<S, SciOxideError> {
    if D == 1 {
        return length(points, vertices);
    }
    let mut coordinates = [[0.0; 3]; 5];
    for (out, &vertex) in coordinates.iter_mut().zip(vertices) {
        for (out, &value) in out.iter_mut().zip(points[vertex].as_array()) {
            *out = value.try_to_f64_exact()?;
        }
    }
    // The homogeneous orientation determinant equals D! times the signed
    // measure. It is accumulated exactly and rounded once, so a sliver
    // cell's volume is not lost to cancellation among rounded differences.
    let (mantissa, exponent) = determinant_magnitude(&coordinates, D, D + 1);
    finish(mantissa / if D == 2 { 2.0 } else { 6.0 }, exponent)
}

pub(super) fn length<S: RealScalar, const D: usize>(
    points: &[Point<S, D>],
    vertices: &[usize],
) -> Result<S, SciOxideError> {
    // Differences are checked for overflow. ScaledNorm then avoids overflow
    // and underflow in the sum of squares.
    let mut norm = ScaledNorm::<f64>::new();
    for (&a, &b) in points[vertices[0]]
        .as_array()
        .iter()
        .zip(points[vertices[1]].as_array())
    {
        norm.add(checked_difference(
            a.try_to_f64_exact()?,
            b.try_to_f64_exact()?,
        )?)?;
    }
    finish(norm.finish()?, 0)
}

pub(crate) fn face<S: RealScalar, const D: usize>(
    points: &[Point<S, D>],
    vertices: &[usize],
) -> Result<S, SciOxideError> {
    if D == 1 {
        // A vertex face has unit zero-dimensional measure.
        return Ok(S::ONE);
    }
    if D == 2 {
        return length(points, vertices);
    }
    // Triangle area is half the norm of the cross product (b-a) x (c-a). Its
    // components are the exact 2D orientation determinants of the triangle
    // projected onto the yz, zx and xy planes.
    let mut components = [(0.0, 0); 3];
    for (component, axes) in components.iter_mut().zip([[1, 2], [2, 0], [0, 1]]) {
        let mut projected = [[0.0; 3]; 5];
        for (out, &vertex) in projected.iter_mut().zip(vertices) {
            for (out, axis) in out.iter_mut().zip(axes) {
                *out = points[vertex].as_array()[axis].try_to_f64_exact()?;
            }
        }
        *component = determinant_magnitude(&projected, 2, 3);
    }
    // The component exponents can lie outside the f64 range. Rescaling by
    // the largest one keeps each summand within [0, 2]. A component far
    // below the largest may flush to zero, which is below the final rounding.
    let exponent = components
        .iter()
        .filter(|(m, _)| *m != 0.0)
        .map(|(_, e)| *e)
        .max()
        .unwrap_or(0);
    let mut norm = ScaledNorm::<f64>::new();
    for (mantissa, power) in components {
        norm.add(mantissa.scalbn(power - exponent))?;
    }
    finish(norm.finish()? / 2.0, exponent)
}
