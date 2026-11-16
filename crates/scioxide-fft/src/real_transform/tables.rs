//! Length contracts, phase tables and normalization factors.
use super::*;

pub(super) const fn dual_type(ty: KernelType) -> KernelType {
    match ty {
        KernelType::SineOne => KernelType::SineOne,
        KernelType::CosineOne => KernelType::CosineOne,
        KernelType::Two => KernelType::Three,
        KernelType::Three => KernelType::Two,
        KernelType::Four => KernelType::Four,
    }
}

pub(super) fn reciprocal_factor<S: scioxide_core::RealScalar>(
    n: usize,
    ty: KernelType,
    length: Argument,
) -> Result<S, SciOxideError> {
    match ty {
        KernelType::SineOne => Ok(S::TWO
            / scalar_from_usize_rounded::<S>(checked_add(
                n,
                1,
                scioxide_core::quantity!("dst_i_round_trip_factor"),
            )?)?),
        KernelType::CosineOne => {
            let n_minus_one = n.checked_sub(1).ok_or_else(|| {
                SciOxideError::invalid_input(
                    length,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "DCT-I requires input length >= 2",
                )
            })?;
            let factor = checked_mul(
                2,
                n_minus_one,
                scioxide_core::quantity!("dct_i_round_trip_factor"),
            )?;
            Ok(scalar_from_usize_rounded::<S>(factor)?.recip())
        }
        KernelType::Two | KernelType::Three => Ok(scalar_from_usize_rounded::<S>(n)?.recip()),
        KernelType::Four => Ok(S::TWO / scalar_from_usize_rounded::<S>(n)?),
    }
}

// Rejects an unsupported `length` argument.
pub(super) fn validate_length(
    n: usize,
    ty: KernelType,
    length: Argument,
) -> Result<(), SciOxideError> {
    crate::validation::nonzero(n, length)?;
    if ty == KernelType::CosineOne && n == 1 {
        return Err(SciOxideError::invalid_input(
            length,
            scioxide_core::InvalidInputReason::OutOfRange,
            "DCT-I requires length >= 2",
        ));
    }
    Ok(())
}

pub(super) fn real_len(n: usize, ty: KernelType) -> Result<usize, SciOxideError> {
    match ty {
        KernelType::CosineOne => {
            checked_mul(2, n - 1, scioxide_core::quantity!("dct_i_real_fft_length"))
        }
        KernelType::SineOne => checked_mul(
            2,
            checked_add(n, 1, scioxide_core::quantity!("dst_i_extension_length"))?,
            scioxide_core::quantity!("dst_i_real_fft_length"),
        ),
        _ => Ok(n),
    }
}

pub(super) fn phases<S: scioxide_core::RealScalar>(
    n: usize,
    ty: KernelType,
    length: Argument,
) -> Result<Vec<Complex<S>>, SciOxideError> {
    let count = match ty {
        KernelType::Two | KernelType::Three => n / 2 + 1,
        KernelType::Four if n.is_multiple_of(2) => n / 2,
        _ => 0,
    };
    let mut phases = try_with_capacity(count, scioxide_core::quantity!("real_transform_phases"))?;
    for k in 0..count {
        let (numerator, denominator) = if ty == KernelType::Four {
            (8 * k as u128 + 1, if n == 2 { 16 } else { 16 * n as u128 })
        } else {
            (k as u128, 4 * n as u128)
        };
        phases.push(reduced_root(numerator, denominator, false, length)?);
    }
    Ok(phases)
}

pub(super) fn scaling<S: scioxide_core::RealScalar>(
    n: usize,
    configured_type: KernelType,
    raw_type: KernelType,
    normalization: Normalization,
    inverse: bool,
    length: Argument,
) -> Result<Scaling<S>, SciOxideError> {
    let mut scaling =
        raw_scaling::<S>(n, configured_type, raw_type, normalization, inverse, length)?;
    if raw_type == KernelType::Four && !n.is_multiple_of(2) {
        // The odd reduction has a common 1/sqrt(2) factor. Fold it here;
        // orthonormal output then uses 1/sqrt(n) with one rounding.
        scaling.output_scale = if normalization == Normalization::Orthonormal {
            scalar_from_usize_rounded::<S>(n)?.sqrt().recip()
        } else {
            scaling.output_scale * S::try_from_f64_lossy(core::f64::consts::FRAC_1_SQRT_2)?
        };
    }
    scaling.output_single = scaling.output_scale * scaling.output_first * scaling.output_last;
    scaling.output_first = scaling.output_scale * scaling.output_first;
    scaling.output_last = scaling.output_scale * scaling.output_last;
    if [
        scaling.output_scale,
        scaling.output_first,
        scaling.output_last,
        scaling.output_single,
    ]
    .iter()
    .any(|v| !v.is_finite() || *v == S::ZERO)
    {
        return Err(crate::validation::numerical(
            scioxide_core::NumericalIssue::Overflow,
            "real-transform scale is unrepresentable",
        ));
    }
    Ok(scaling)
}

pub(super) fn raw_scaling<S: scioxide_core::RealScalar>(
    n: usize,
    configured_type: KernelType,
    raw_type: KernelType,
    normalization: Normalization,
    inverse: bool,
    length: Argument,
) -> Result<Scaling<S>, SciOxideError> {
    let one = S::ONE;
    if normalization != Normalization::Orthonormal {
        let output_scale = if (!inverse && normalization == Normalization::Forward)
            || (inverse && normalization == Normalization::Inverse)
        {
            reciprocal_factor(n, configured_type, length)?
        } else {
            one
        };
        return Ok(Scaling {
            input_first: one,
            input_last: one,
            output_scale,
            output_first: one,
            output_last: one,
            output_single: S::ONE,
        });
    }

    // Orthonormal forms are the raw transforms with endpoint weights: the
    // sqrt(2) factors on the first/last input or output make the raw cosine
    // matrix orthogonal once multiplied by `output_scale`. Because
    // the weights depend on the raw type actually executed, the inverse of an
    // orthonormal DCT-II uses the DCT-III weights and vice versa.
    let two = S::TWO;
    let sqrt_two = two.sqrt();
    let n_scalar = scalar_from_usize_rounded::<S>(n)?;
    match raw_type {
        KernelType::SineOne => Ok(Scaling {
            input_first: one,
            input_last: one,
            output_scale: (two
                / scalar_from_usize_rounded::<S>(checked_add(
                    n,
                    1,
                    scioxide_core::quantity!("dst_i_scale"),
                )?)?)
            .sqrt(),
            output_first: one,
            output_last: one,
            output_single: one,
        }),
        KernelType::CosineOne => {
            let denominator = checked_mul(
                2,
                n - 1,
                scioxide_core::quantity!("orthonormal_dct_i_scale"),
            )?;
            Ok(Scaling {
                input_first: sqrt_two,
                input_last: sqrt_two,
                output_scale: scalar_from_usize_rounded::<S>(denominator)?.sqrt().recip(),
                output_first: sqrt_two.recip(),
                output_last: sqrt_two.recip(),
                output_single: S::ONE,
            })
        }
        KernelType::Two => Ok(Scaling {
            input_first: one,
            input_last: one,
            output_scale: (two / n_scalar).sqrt(),
            output_first: sqrt_two.recip(),
            output_last: one,
            output_single: S::ONE,
        }),
        KernelType::Three => Ok(Scaling {
            input_first: sqrt_two,
            input_last: one,
            output_scale: (two * n_scalar).sqrt().recip(),
            output_first: one,
            output_last: one,
            output_single: S::ONE,
        }),
        KernelType::Four => Ok(Scaling {
            input_first: one,
            input_last: one,
            output_scale: (two / n_scalar).sqrt(),
            output_first: one,
            output_last: one,
            output_single: S::ONE,
        }),
    }
}
