//! Native fit kernels with extended-range replay from the original operands.
use super::*;

/// Tiny products retry before normalization or row weighting can amplify lost
/// bits. Complex cross-products use core's component-aware range guard.
#[inline]
pub(super) fn native_product<C: NumericScalar>(value: C, factor: C::Real) -> Option<C> {
    if factor == C::Real::ONE {
        return Some(value);
    }
    let product = C::from_components(
        value.real_component() * factor,
        value.imaginary_component() * factor,
    );
    (!standard::product_needs_retry(value, C::from_real(factor), product)).then_some(product)
}

fn add_norm<C: NumericScalar>(
    norm: &mut ScaledNorm<C::Real>,
    value: C,
) -> Result<(), SciOxideError> {
    norm.add(value.real_component())?;
    if !C::IS_REAL {
        norm.add(value.imaginary_component())?;
    }
    Ok(())
}

/// Row-contiguous weighting and four-lane scaled sums of squares. Temporary
/// state is per column; any range loss discards the candidate for scaled replay.
pub(super) fn native_columns<C: NumericScalar>(
    design: &Matrix<C>,
    roots: &[C::Real],
    weighted: &mut Matrix<C>,
    scales: &mut [Scaled<C::Real>],
) -> Result<bool, SciOxideError> {
    let cols = design.cols();
    let mut maxima = try_filled(
        cols,
        C::Real::ZERO,
        scioxide_core::quantity!("fit_column_maxima"),
    )?;
    for ((row, out), &root) in design
        .as_slice()
        .chunks_exact(cols)
        .zip(weighted.as_mut_slice().chunks_exact_mut(cols))
        .zip(roots)
    {
        if root == C::Real::ZERO {
            out.fill(C::ZERO);
            continue;
        }
        if root == C::Real::ONE {
            out.copy_from_slice(row);
        } else {
            for (out, &value) in out.iter_mut().zip(row) {
                let Some(value) = native_product(value, root) else {
                    return Ok(false);
                };
                *out = value;
            }
        }
        for (maximum, &value) in maxima.iter_mut().zip(out.iter()) {
            *maximum = maximum.max(standard::magnitude(value));
        }
    }
    let mut inverses = try_filled(
        cols,
        C::Real::ONE,
        scioxide_core::quantity!("fit_column_inverses"),
    )?;
    for (&maximum, inverse) in maxima.iter().zip(&mut inverses) {
        if maximum != C::Real::ZERO {
            *inverse = C::Real::ONE / maximum;
        }
        if !inverse.is_finite() || *inverse < C::Real::MIN_POSITIVE {
            return Ok(false);
        }
    }
    let length =
        scioxide_core::backend::checked_mul(cols, 4, scioxide_core::quantity!("fit_norm_lanes"))?;
    let mut squares = try_filled(
        length,
        C::Real::ZERO,
        scioxide_core::quantity!("fit_norm_lanes"),
    )?;
    for (index, row) in weighted.as_slice().chunks_exact(cols).enumerate() {
        let lane = &mut squares[(index % 4) * cols..][..cols];
        for ((&value, &inverse), sum) in row.iter().zip(&inverses).zip(lane) {
            let re = value.real_component() * inverse;
            let im = value.imaginary_component() * inverse;
            *sum += re * re;
            if !C::IS_REAL {
                *sum += im * im;
            }
        }
    }
    // The normalized maximum contributes one. Underflowed squared ratios are
    // negligible beside that contribution, as in the LASSQ recurrence.
    for (column, ((&maximum, inverse), scale)) in
        maxima.iter().zip(&mut inverses).zip(scales).enumerate()
    {
        let norm = if maximum == C::Real::ZERO {
            C::Real::ONE
        } else {
            let square = ((squares[column] + squares[cols + column])
                + (squares[2 * cols + column] + squares[3 * cols + column]))
                .max(C::Real::ONE);
            let Ok(norm) = ScaledNorm::from_parts(maximum, square)?.finish() else {
                return Ok(false);
            };
            norm
        };
        *inverse = C::Real::ONE / norm;
        if !inverse.is_finite() || *inverse < C::Real::MIN_POSITIVE {
            return Ok(false);
        }
        *scale = Scaled::from_value(norm)?;
    }
    let mut retry = false;
    for row in weighted.as_mut_slice().chunks_exact_mut(cols) {
        for (value, &inverse) in row.iter_mut().zip(&inverses) {
            let candidate = C::from_components(
                value.real_component() * inverse,
                value.imaginary_component() * inverse,
            );
            retry |= standard::product_needs_retry(*value, C::from_real(inverse), candidate);
            *value = candidate;
        }
    }
    Ok(!retry)
}

#[cold]
pub(super) fn extended_column<C: NumericScalar>(
    design: &Matrix<C>,
    roots: &[C::Real],
    column: usize,
    weighted: &mut Matrix<C>,
) -> Result<Scaled<C::Real>, SciOxideError> {
    let cols = design.cols();
    let mut square = Scaled::ZERO;
    for (row, &root) in design.as_slice().chunks_exact(cols).zip(roots) {
        let value = ScaledValue::from_value(row[column])?
            .magnitude()?
            .mul(Scaled::from_value(root)?)?;
        square = square.add(value.mul(value)?)?;
    }
    let scale = if square.is_zero() {
        Scaled::ONE
    } else {
        square.sqrt()?
    };
    for ((row, out), &root) in design
        .as_slice()
        .chunks_exact(cols)
        .zip(weighted.as_mut_slice().chunks_exact_mut(cols))
        .zip(roots)
    {
        out[column] = ScaledValue::from_value(row[column])?
            .mul_real(Scaled::from_value(root)?.div(scale)?)?
            .round()?;
    }
    Ok(scale)
}

pub(super) fn native_rhs<C: NumericScalar>(
    values: &[C],
    roots: &[C::Real],
    rhs: &mut [C],
) -> Option<C::Real> {
    let mut scale = C::Real::ZERO;
    for ((out, &value), &root) in rhs.iter_mut().zip(values).zip(roots) {
        *out = native_product(value, root)?;
        scale = scale.max(out.abs());
    }
    if scale == C::Real::ZERO {
        scale = C::Real::ONE;
    }
    let inverse = C::Real::ONE / scale;
    if !scale.is_finite() || !inverse.is_finite() || inverse < C::Real::MIN_POSITIVE {
        return None;
    }
    for out in rhs {
        *out = native_product(*out, inverse)?;
    }
    Some(scale)
}

#[cold]
pub(super) fn extended_rhs<C: NumericScalar>(
    values: &[C],
    roots: &[C::Real],
    rhs: &mut [C],
) -> Result<Scaled<C::Real>, SciOxideError> {
    let mut scale = Scaled::ZERO;
    for (&value, &root) in values.iter().zip(roots) {
        let magnitude = ScaledValue::from_value(value)?
            .magnitude()?
            .mul(Scaled::from_value(root)?)?;
        if !magnitude.abs_le(scale) {
            scale = magnitude;
        }
    }
    if scale.is_zero() {
        scale = Scaled::ONE;
    }
    for ((out, &value), &root) in rhs.iter_mut().zip(values).zip(roots) {
        *out = ScaledValue::from_value(value)?
            .mul_real(Scaled::from_value(root)?.div(scale)?)?
            .round()?;
    }
    Ok(scale)
}

pub(super) fn native_residual<C: NumericScalar>(
    design: &Matrix<C>,
    values: &[C],
    roots: &[C::Real],
    coefficients: &[C],
    context: ExecutionContext<'_>,
) -> Result<Option<C::Real>, SciOxideError> {
    let mut norm = ScaledNorm::new();
    for ((row, &value), &root) in design
        .as_slice()
        .chunks_exact(coefficients.len())
        .zip(values)
        .zip(roots)
    {
        context.check()?;
        if root == C::Real::ZERO {
            continue;
        }
        let mut predicted = C::ZERO;
        for (&basis, &coefficient) in row.iter().zip(coefficients) {
            let product = basis * coefficient;
            if standard::product_needs_retry(basis, coefficient, product) {
                return Ok(None);
            }
            predicted = predicted + product;
        }
        let residual = predicted - value;
        if !residual.has_finite_components() {
            return Ok(None);
        }
        let Some(residual) = native_product(residual, root) else {
            return Ok(None);
        };
        add_norm(&mut norm, residual)?;
    }
    // An overflowing native norm can be a boundary-rounding artifact. Let
    // extended replay decide final representability before reporting failure.
    Ok(norm.finish().ok())
}

#[cold]
pub(super) fn extended_residual<C: NumericScalar>(
    design: &Matrix<C>,
    values: &[C],
    roots: &[C::Real],
    coefficients: &[C],
    context: ExecutionContext<'_>,
) -> Result<C::Real, SciOxideError> {
    let mut square = Scaled::ZERO;
    for ((row, &value), &root) in design
        .as_slice()
        .chunks_exact(coefficients.len())
        .zip(values)
        .zip(roots)
    {
        context.check()?;
        if root == C::Real::ZERO {
            continue;
        }
        let predicted = row.iter().zip(coefficients).try_fold(
            ScaledValue::ZERO,
            |sum, (&basis, &coefficient)| {
                sum.add(ScaledValue::from_value(basis)?.mul(ScaledValue::from_value(coefficient)?)?)
            },
        )?;
        let residual = predicted
            .sub(ScaledValue::from_value(value)?)?
            .magnitude()?
            .mul(Scaled::from_value(root)?)?;
        square = square.add(residual.mul(residual)?)?;
    }
    square.sqrt()?.round()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    fn residual_replay<C: NumericScalar>() {
        let context = ExecutionContext::sequential_native();
        let tiny_root = C::Real::MIN_POSITIVE.sqrt();
        let part = tiny_root * C::Real::EPSILON;
        let a = C::from_components(part, part);
        let b = C::from_components(part, -part);
        let design = Matrix::from_vec(1, 1, alloc::vec![a]).unwrap();
        let roots = [C::Real::ONE / tiny_root];
        // The product rounds to zero before the weight would recover it.
        assert_eq!(a * b, C::ZERO);
        assert_eq!(
            native_residual(&design, &[C::ZERO], &roots, &[b], context).unwrap(),
            None
        );
        let residual = extended_residual(&design, &[C::ZERO], &roots, &[b], context).unwrap();
        let components = if C::IS_REAL {
            C::Real::ONE
        } else {
            C::Real::ONE + C::Real::ONE
        };
        let expected = tiny_root * C::Real::EPSILON * C::Real::EPSILON * components;
        assert!((residual / expected - C::Real::ONE).abs() <= C::Real::EPSILON * components);

        let maximum = C::from_components(C::Real::MAX, C::Real::MAX);
        let quarter = C::Real::ONE / C::Real::try_from_usize_exact(4).unwrap();
        let design = Matrix::from_vec(1, 3, alloc::vec![maximum, maximum, -maximum]).unwrap();
        assert_eq!(
            native_residual(&design, &[C::ZERO], &[quarter], &[C::ONE; 3], context).unwrap(),
            None
        );
        let residual =
            extended_residual(&design, &[C::ZERO], &[quarter], &[C::ONE; 3], context).unwrap();
        let expected = C::Real::MAX * quarter * components.sqrt();
        assert!((residual / expected - C::Real::ONE).abs() <= C::Real::EPSILON * components);
    }

    #[test]
    fn residual_replay_recovers_weighted_underflow_and_cancelling_overflow() {
        residual_replay::<f32>();
        residual_replay::<f64>();
        residual_replay::<Complex<f32>>();
        residual_replay::<Complex<f64>>();
    }
}
