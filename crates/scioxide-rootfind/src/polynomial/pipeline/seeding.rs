//! Preprocessing, closed-form roots and starting points of polynomial root solves.
use super::*;
/// Prepare the coefficient input `argument`: deflate exact zero roots, then
/// apply lossless binary variable and coefficient scaling where it preserves
/// every component.
pub(crate) fn prepare<'a, S: RealScalar>(
    input: &[impl NumericScalar<Real = S>],
    basis: PolynomialRootBasisRef<'a, S>,
    output: &mut [Complex<S>],
    argument: Argument,
) -> Result<Prepared<'a, S>, SciOxideError> {
    let monomial = basis.is_monomial();
    let zeros = if monomial {
        input
            .iter()
            .take(input.len() - 1)
            .take_while(|c| c.real_component().is_zero() && c.imaginary_component().is_zero())
            .count()
    } else {
        0
    };
    let input = &input[zeros..];
    let degree = input.len() - 1;
    let real = input.iter().all(|c| c.imaginary_component().is_zero());
    let identity = |output: &mut [Complex<S>]| {
        for (out, c) in output.iter_mut().zip(input) {
            *out = Complex::new(c.real_component(), c.imaginary_component());
        }
        Prepared {
            zeros,
            degree,
            scale: S::ONE,
            basis,
            real,
            companion_safe: false,
        }
    };
    // Analytic formulas work in the original coordinate and coefficient range.
    if monomial && degree <= 2 {
        return Ok(identity(output));
    }
    let magnitude_log = |re: S, im: S| {
        let m = re.abs().max(im.abs());
        if m.is_zero() {
            S::NEG_INFINITY
        } else {
            m.log2() + ((re / m).powi(2) + (im / m).powi(2)).log2() * S::HALF
        }
    };
    let lead = input[degree];
    let lead_log = magnitude_log(lead.real_component(), lead.imaginary_component());
    let variable_exponent = if monomial && degree > 0 {
        input
            .iter()
            .take(degree)
            .enumerate()
            .filter(|(_, c)| !c.abs().is_zero())
            .try_fold(S::NEG_INFINITY, |m, (i, c)| {
                Ok::<_, SciOxideError>(m.max(
                    (magnitude_log(c.real_component(), c.imaginary_component()) - lead_log)
                        / S::try_from_usize_exact(degree - i)?,
                ))
            })?
            .ceil()
    } else {
        S::ZERO
    };
    let variable_exponent = if variable_exponent.is_finite() {
        variable_exponent.try_to_f64_exact()? as i64
    } else {
        0
    };
    let scaled = Scaled::from_value(S::ONE)?.shifted(variable_exponent)?;
    if !scaled.abs_le(Scaled::from_value(S::MAX)?) {
        return Ok(identity(output));
    }
    let scale = scaled.round()?;
    if scale.is_zero() {
        return Ok(identity(output));
    }
    let common = input
        .iter()
        .enumerate()
        .try_fold(S::NEG_INFINITY, |m, (i, c)| {
            Ok::<_, SciOxideError>(m.max(
                magnitude_log(c.real_component(), c.imaginary_component())
                    + S::try_from_usize_exact(i)?
                        * S::try_from_f64_lossy(variable_exponent as f64)?,
            ))
        })?
        .ceil()
        .try_to_f64_exact()? as i64;
    for (i, c) in input.iter().enumerate() {
        let shift = variable_exponent
            .checked_mul(i64::try_from(i).map_err(|_| {
                invalid(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "degree exceeds scaling range",
                )
            })?)
            .and_then(|e| e.checked_sub(common))
            .ok_or_else(|| {
                invalid(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "scaling exponent overflow",
                )
            })?;
        output[i] = Complex::new(
            Scaled::from_value(c.real_component())?
                .shifted(shift)?
                .round()?,
            Scaled::from_value(c.imaginary_component())?
                .shifted(shift)?
                .round()?,
        );
        if (!c.real_component().is_zero() && output[i].re.is_zero())
            || (!c.imaginary_component().is_zero() && output[i].im.is_zero())
        {
            return Ok(identity(output));
        }
        for (original, scaled) in [
            (c.real_component(), output[i].re),
            (c.imaginary_component(), output[i].im),
        ] {
            let restored = Scaled::from_value(scaled)?.shifted(-shift)?;
            if !restored.abs_le(Scaled::from_value(S::MAX)?) || restored.round()? != original {
                return Ok(identity(output));
            }
        }
    }
    Ok(Prepared {
        zeros,
        degree,
        scale,
        basis,
        real,

        companion_safe: true,
    })
}

// Upper Newton polygon of log coefficient magnitudes. Each edge supplies its
// horizontal length in starts on the corresponding modulus circle (Bini 1996).
pub(crate) fn seed_radii<S: RealScalar>(
    coefficients: &[Complex<S>],
    basis: PolynomialRootBasisRef<'_, S>,
    radii: &mut [S],
    hull: &mut [usize],
) -> Result<(), SciOxideError> {
    if !basis.is_monomial() {
        radii.fill(S::ONE);
        return Ok(());
    }
    let log = |i: usize| {
        let c = coefficients[i];
        let m = c.re.abs().max(c.im.abs());
        m.log2() + ((c.re / m).powi(2) + (c.im / m).powi(2)).log2() * S::HALF
    };
    let mut count = 0;
    for (i, _) in coefficients
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.norm().is_zero())
    {
        while count >= 2 {
            let a = hull[count - 2];
            let b = hull[count - 1];
            if (log(b) - log(a)) * S::try_from_usize_exact(i - b)?
                > (log(i) - log(b)) * S::try_from_usize_exact(b - a)?
            {
                break;
            }
            count -= 1;
        }
        hull[count] = i;
        count += 1;
    }
    for pair in hull[..count].windows(2) {
        let a = pair[0];
        let b = pair[1];
        let radius = ((log(a) - log(b)) / S::try_from_usize_exact(b - a)?).exp2();
        if !radius.is_finite() || radius.is_zero() {
            return Err(SciOxideError::numerical_instability(
                if radius.is_zero() {
                    NumericalIssue::Underflow
                } else {
                    NumericalIssue::Overflow
                },
                "seed radius is outside the scalar range",
            ));
        }
        radii[a..b].fill(radius);
    }
    Ok(())
}
pub(crate) fn initialize<S: RealScalar>(
    roots: &mut [Complex<S>],
    radii: &[S],
    start: usize,
) -> Result<(), SciOxideError> {
    let mut begin = 0;
    while begin < roots.len() {
        let end = (begin + 1..roots.len())
            .find(|&i| radii[i] != radii[begin])
            .unwrap_or(roots.len());
        let count = S::try_from_usize_exact(end - begin)?;
        let phase = S::try_from_usize_exact(start % 4)? / S::try_from_usize_exact(4)?;
        let factor = match start % 3 {
            0 => S::ONE,
            1 => S::ONE + S::HALF,
            _ => S::ONE / (S::ONE + S::HALF),
        };
        for (i, root) in roots[begin..end].iter_mut().enumerate() {
            let angle = S::TAU * (S::try_from_usize_exact(i)? + S::HALF * S::HALF + phase) / count;
            let radius = radii[begin] * factor;
            *root = Complex::new(radius * angle.cos(), radius * angle.sin());
        }
        begin = end;
    }
    Ok(())
}
/// Closed-form roots of degree one or two.
pub(crate) fn analytic<S: RealScalar>(
    c: &[Complex<S>],
    roots: &mut [Complex<S>],
) -> Result<(), SciOxideError> {
    if roots.len() == 1 {
        roots[0] = scioxide_core::ScaledValue::from_value(c[0])?
            .neg()
            .div(scioxide_core::ScaledValue::from_value(c[1])?)?
            .round()?;
    } else if c.iter().all(|c| c.im.is_zero()) {
        let r = checked_quadratic_roots(c[2].re, c[1].re, c[0].re)?;
        roots.copy_from_slice(&r);
    } else {
        let a = scioxide_core::ScaledValue::from_value(c[2])?;
        let b = scioxide_core::ScaledValue::from_value(c[1])?;
        let c = scioxide_core::ScaledValue::from_value(c[0])?;
        let four = Scaled::from_value(S::TWO * S::TWO)?;
        let discriminant = b.mul(b)?.sub(a.mul(c)?.mul_real(four)?)?.sqrt()?;
        let plus = b.neg().add(discriminant)?;
        let minus = b.neg().sub(discriminant)?;
        let q = if plus.magnitude()?.abs_le(minus.magnitude()?) {
            minus
        } else {
            plus
        }
        .mul_real(Scaled::from_value(S::HALF)?)?;
        if q.is_zero() {
            roots.fill(
                b.neg()
                    .div(a.mul_real(Scaled::from_value(S::TWO)?)?)?
                    .round()?,
            );
        } else {
            roots[0] = q.div(a)?.round()?;
            roots[1] = c.div(q)?.round()?;
        }
    }
    for &root in roots.iter() {
        finite_complex(root)?;
    }
    Ok(())
}
pub(crate) fn publish_candidates<S: RealScalar>(
    p: Prepared<'_, S>,
    w: &mut PolynomialRootWorkspace<S>,
) -> Result<(), SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    for (i, &root) in w.current[..p.degree].iter().enumerate() {
        let root = root * Complex::new(p.scale, S::ZERO);
        finite_complex(root)?;
        if w.candidates[p.zeros + i] != root {
            w.scratch_diagnostics[p.zeros + i] = PolynomialRootDiagnostic::unevaluated();
        }
        w.scratch_diagnostics[p.zeros + i].available = true;
        w.candidates[p.zeros + i] = root;
    }
    Ok(())
}
pub(crate) fn deterministic_perturbation<S: RealScalar>(
    index: usize,
    other_index: usize,
    start: usize,
    root: Complex<S>,
) -> Complex<S> {
    let code = index
        .wrapping_mul(17)
        .wrapping_add(other_index.wrapping_mul(31))
        .wrapping_add(start.wrapping_mul(43));
    let quadrant = code % 4;
    let magnitude = (S::SQRT_EPSILON * root.re.abs().max(root.im.abs())).max(S::MIN_SUBNORMAL);
    match quadrant {
        0 => Complex::new(magnitude, magnitude),
        1 => Complex::new(-magnitude, magnitude),
        2 => Complex::new(-magnitude, -magnitude),
        _ => Complex::new(magnitude, -magnitude),
    }
}

pub(crate) fn finite_complex<S: RealScalar>(value: Complex<S>) -> Result<(), SciOxideError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            "complex arithmetic produced a non-finite value",
        ))
    }
}

pub(crate) fn real_zero<S: RealScalar>() -> S {
    <S as scioxide_core::AlgebraicScalar>::ZERO
}

pub(crate) fn real_one<S: RealScalar>() -> S {
    <S as scioxide_core::AlgebraicScalar>::ONE
}

/// Resize `quantity` storage.
pub(crate) fn resize<T: Clone>(
    values: &mut Vec<T>,
    len: usize,
    value: T,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    if values.len() < len {
        try_reserve(values, len - values.len(), quantity)?;
        values.resize(len, value);
    } else {
        values.truncate(len);
    }
    Ok(())
}
