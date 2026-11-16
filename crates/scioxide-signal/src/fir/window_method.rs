use super::*;

/// Designs a linear-phase FIR filter of `n` taps by the window method.
///
/// The ideal response has the constant amplitude `D_b` on each band `b` and
/// zero in the gaps between bands. Its impulse response, centered at
/// `c = (n − 1)/2`, is truncated to `n` taps and multiplied by the symmetric
/// window `w` of length `n`:
///
/// ```text
/// h[i] = w[i] Σ_b D_b ∫_{l_b}^{u_b} cos(π (c − i) f) df    (even symmetry)
/// h[i] = w[i] Σ_b D_b ∫_{l_b}^{u_b} sin(π (c − i) f) df    (odd symmetry)
/// ```
///
/// where `[l_b, u_b]` are the band edges in multiples of Nyquist. The band
/// integrals are evaluated in closed form, so the design costs `O(n B)` for `B`
/// bands plus the window. Band weights do not affect this method. All four
/// linear-phase types are supported (see [`FirSymmetry`]). The optional
/// `normalization` then rescales the taps. The design is direct and reports
/// [`FirTermination::Completed`].
///
/// # Errors
///
/// - `InvalidInput` if `n` is zero, if `n = 1` with odd symmetry, or if the
///   bands are invalid (see [`FirBand`]).
/// - `InvalidInput` (`UnsupportedConfiguration`) if a band has
///   `desired_lower ≠ desired_upper`.
/// - `InvalidInput` if the desired amplitude is nonzero where the symmetry
///   forces a zero, or if the window parameters are invalid (see
///   [`WindowKind`](crate::WindowKind)).
/// - `InvalidInput` or `DomainError` from `normalization`, as described on
///   [`FirNormalization`].
/// - `NumericalInstability` if a tap overflows.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{FirBand, FirNormalization, FirSymmetry, WindowKind, design_fir_window};
/// // Low-pass filter with cutoff 0.25 × Nyquist and unit gain at DC.
/// let design = design_fir_window(
///     21,
///     FirSymmetry::Even,
///     &[FirBand::new(0.0_f64, 0.25, 1.0, 1.0)],
///     WindowKind::Hamming,
///     FirNormalization::At {
///         frequency: 0.0,
///         gain: 1.0,
///     },
/// )?;
/// assert!((design.response(0.0)?.norm() - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd
/// ed., Pearson, 2010, ch. 7.
pub fn design_fir_window<S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    bands: &[FirBand<S>],
    window: crate::WindowKind<'_>,
    normalization: FirNormalization<S>,
) -> Result<FirDesign<S>, SciOxideError> {
    validate(n, symmetry, bands)?;
    if bands.iter().any(|b| b.desired_lower != b.desired_upper) {
        return Err(unsupported(
            scioxide_core::argument!("bands"),
            "window design requires constant desired bands",
        ));
    }
    let w = crate::window::<S>(
        window,
        n,
        crate::WindowSymmetry::Symmetric,
        crate::WindowNormalization::Native,
    )?;
    let mut result = from_amplitudes(n, symmetry, &[])?;
    let two = S::TWO;
    let center = scalar::<S>(n - 1)? / two;
    for (i, &weight) in w.iter().enumerate().take(n.div_ceil(2)) {
        let k = center - scalar::<S>(i)?;
        let mut sum = S::ZERO;
        for b in bands {
            let integral = if symmetry == FirSymmetry::Even {
                cosine_integral(k, b.lower, b.upper)?
            } else {
                sine_integral(k, b.lower, b.upper)?
            };
            sum += b.desired_lower * integral;
        }
        if !sum.is_finite() {
            let mut exact = scioxide_core::RangeAccumulator::new();
            for b in bands {
                let integral = if symmetry == FirSymmetry::Even {
                    cosine_integral(k, b.lower, b.upper)?
                } else {
                    sine_integral(k, b.lower, b.upper)?
                };
                exact.add_product(b.desired_lower, integral)?;
            }
            sum = exact.finish()?;
        }
        let h = finite(sum * weight)?;
        result.taps[i] = h;
        result.taps[n - 1 - i] = if symmetry == FirSymmetry::Even { h } else { -h };
    }
    normalize(&mut result, normalization)?;
    Ok(result)
}
