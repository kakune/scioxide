use super::*;

/// Designs a linear-phase FIR filter of `n` taps by frequency sampling.
///
/// The desired zero-phase amplitude `A(f)` is the piecewise-linear
/// interpolant of the knots `(frequencies[j], amplitudes[j])` on `[0, 1]`, in
/// multiples of Nyquist. With `N = mesh_size` and `c = (n − 1)/2`, the design
/// samples the linear-phase response
///
/// ```text
/// H_k = e^(−iπ f_k c) A(f_k)      (even symmetry)
/// H_k = i e^(−iπ f_k c) A(f_k)    (odd symmetry),    f_k = 2k/N,  k = 0, …, N/2
/// ```
///
/// takes the real inverse DFT of length `N`, keeps the first `n` samples,
/// averages each with its reflection `±h[n − 1 − i]` to impose `symmetry`, and
/// multiplies the result by the optional symmetric `window` (rectangular if
/// `None`). The optional `normalization` then rescales the taps. The design
/// costs `O(N log N)`, is direct and reports [`FirTermination::Completed`].
///
/// # Errors
///
/// - `InvalidInput` if `n` is zero or `n = 1` with odd symmetry.
/// - `InvalidInput` if `frequencies` has fewer than two knots, does not start
///   at 0 and end at 1, or does not increase strictly.
/// - `DimensionMismatch` if `amplitudes` and `frequencies` differ in length.
/// - `InvalidInput` if `mesh_size` is odd, smaller than `2n` or larger than
///   2⁵².
/// - `InvalidInput` if an end amplitude is nonzero where the symmetry forces a
///   zero (see [`FirSymmetry`]), or if the window parameters are invalid.
/// - `InvalidInput` or `DomainError` from `normalization`, as described on
///   [`FirNormalization`].
/// - `NonFinite` if a knot or amplitude is NaN or infinite.
/// - `NumericalInstability` if a tap overflows.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{FirNormalization, FirSymmetry, WindowKind, design_fir_frequency_sampling};
/// // Low-pass filter: pass band [0, 0.3], transition to 0.5, stop band [0.5, 1].
/// let design = design_fir_frequency_sampling(
///     31,
///     FirSymmetry::Even,
///     &[0.0_f64, 0.3, 0.5, 1.0],
///     &[1.0, 1.0, 0.0, 0.0],
///     64,
///     Some(WindowKind::Hamming),
///     FirNormalization::At {
///         frequency: 0.0,
///         gain: 1.0,
///     },
/// )?;
/// assert!((design.response(0.0)?.norm() - 1.0).abs() < 1e-12);
/// assert!(design.response(0.8)?.norm() < 1e-2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// L. R. Rabiner, B. Gold and C. A. McGonegal, "An approach to the
/// approximation problem for nonrecursive digital filters," *IEEE Trans. Audio
/// Electroacoust.* 18(2), 83–106, 1970.
pub fn design_fir_frequency_sampling<S: RealScalar>(
    n: usize,
    symmetry: FirSymmetry,
    frequencies: &[S],
    amplitudes: &[S],
    mesh_size: usize,
    window: Option<crate::WindowKind<'_>>,
    normalization: FirNormalization<S>,
) -> Result<FirDesign<S>, SciOxideError> {
    let execution = ExecutionContext::sequential_native();
    use scioxide_fft::{FftDirection, FftInput, FftLayout, FftPlan, FftPlanSpec, Normalization};
    dimensions(n, symmetry)?;
    check_real_slice_finite(frequencies)?;
    check_real_slice_finite(amplitudes)?;
    if frequencies.len() < 2
        || frequencies[0] != S::ZERO
        || frequencies[frequencies.len() - 1] != S::ONE
        || frequencies.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(invalid(
            scioxide_core::argument!("frequencies"),
            "knots must increase strictly from 0 to 1",
        ));
    }
    exact_len(frequencies.len(), amplitudes.len())?;
    if !mesh_size.is_multiple_of(2)
        || mesh_size < checked_mul(n, 2, scioxide_core::quantity!("fir_fft_mesh"))?
    {
        return Err(invalid(
            scioxide_core::argument!("mesh_size"),
            "mesh size must be even and at least twice the tap count",
        ));
    }
    for i in [0, frequencies.len() - 1] {
        if forced_zero(n, symmetry, frequencies[i]) && amplitudes[i] != S::ZERO {
            return Err(inconsistent(
                scioxide_core::argument!("amplitudes"),
                "endpoint amplitude must vanish where the symmetry forces a zero",
            ));
        }
    }
    crate::windows::count(mesh_size, scioxide_core::argument!("mesh_size"))?;
    let plan = FftPlan::build(
        FftPlanSpec::new(
            mesh_size,
            FftDirection::Inverse,
            Normalization::Inverse,
            FftLayout::Real,
        ),
        execution,
    )?;
    let z = Complex::new(S::ZERO, S::ZERO);
    let mut spectrum = try_filled(
        mesh_size / 2 + 1,
        z,
        scioxide_core::quantity!("fir_sampling_spectrum"),
    )?;
    let mut output = try_filled(
        mesh_size,
        S::ZERO,
        scioxide_core::quantity!("fir_sampling_inverse"),
    )?;
    let mut scratch = plan.workspace(())?;
    let half_mesh = scalar::<S>(mesh_size / 2)?;
    let center = scalar::<S>(n - 1)? / (S::TWO);
    let mut segment = 0;
    for (k, bin) in spectrum.iter_mut().enumerate() {
        let f = S::try_from_f64_lossy(k as f64)? / half_mesh;
        while segment + 2 < frequencies.len() && f > frequencies[segment + 1] {
            segment += 1;
        }
        if amplitudes[segment] == S::ZERO && amplitudes[segment + 1] == S::ZERO {
            if segment + 2 == frequencies.len() {
                break;
            }
            continue;
        }
        let t = (f - frequencies[segment]) / (frequencies[segment + 1] - frequencies[segment]);
        let a = finite(amplitudes[segment] * (S::ONE - t) + amplitudes[segment + 1] * t)?;
        if a == S::ZERO {
            continue;
        }
        let (sin, cos) = (S::PI * f * center).sin_cos();
        let v = if symmetry == FirSymmetry::Even {
            Complex::new(a * cos, -a * sin)
        } else {
            Complex::new(a * sin, a * cos)
        };
        *bin = if k == 0 || k == mesh_size / 2 {
            Complex::new(v.re, S::ZERO)
        } else {
            v
        };
    }
    plan.execute_into(
        FftInput::hermitian(mesh_size, &spectrum),
        &mut output,
        &mut scratch,
        execution,
    )?;
    let weights = crate::window::<S>(
        window.unwrap_or(crate::WindowKind::Boxcar),
        n,
        crate::WindowSymmetry::Symmetric,
        crate::WindowNormalization::Native,
    )?;
    let mut result = from_amplitudes(n, symmetry, &[])?;
    let two = S::TWO;
    for i in 0..n.div_ceil(2) {
        let reflected = if symmetry == FirSymmetry::Even {
            output[n - 1 - i]
        } else {
            -output[n - 1 - i]
        };
        let h = finite((output[i] / two + reflected / two) * weights[i])?;
        result.taps[i] = h;
        result.taps[n - 1 - i] = if symmetry == FirSymmetry::Even { h } else { -h };
    }
    normalize(&mut result, normalization)?;
    Ok(result)
}
