//! Window functions and window metrics.

use crate::util::window_finite as finite;
use crate::util::{check_real_slice_finite, empty, inconsistent, invalid, unsupported};
use alloc::vec::Vec;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{Argument, Complex, ExecutionContext, RealScalar, SciOxideError};
use scioxide_fft::{FftDirection, FftLayout, FftPlan, FftPlanSpec, Normalization};

/// Sampling convention of a window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindowSymmetry {
    /// Samples the symmetric window of the requested length, including both
    /// end points. This is the usual choice for FIR filter design.
    #[default]
    Symmetric,
    /// Samples the symmetric window one sample longer and drops its last
    /// sample, so the window is periodic with the requested length. This is
    /// the usual choice for spectral analysis.
    Periodic,
}

/// Scaling of window samples, applied after a periodic window drops its last
/// sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindowNormalization {
    /// Keeps the scaling of the window's definition. Symmetric DPSS tapers have
    /// unit energy.
    #[default]
    Native,
    /// Divides by the largest absolute sample.
    UnitPeak,
    /// Divides by the Euclidean norm of the returned samples.
    UnitEnergy,
}

/// Window function and its parameters.
///
/// The formulas below use the sampled length `N` (the requested length, plus
/// one for [`WindowSymmetry::Periodic`]), the sample index `i = 0, …, N − 1`,
/// the center `c = (N − 1)/2` and the centered coordinate
/// `z = 2i/(N − 1) − 1 ∈ [−1, 1]`. A window of length 0 is empty, and a
/// window of length 1 is `[1]`. Parameters are `f64` for every output type, and
/// the samples are evaluated in `f64`.
///
/// The cosine-sum windows are `w = Σⱼ (−1)ʲ aⱼ cos(2πji/(N − 1))` with the
/// coefficients `aⱼ` listed on each variant.
///
/// Invalid parameters are `InvalidInput`, and NaN or infinite parameters are
/// `NonFinite`. Each variant states its own conditions.
#[derive(Clone, Copy, Debug)]
pub enum WindowKind<'a> {
    /// Rectangular window, `w = 1`.
    Boxcar,
    /// Triangular window with nonzero end points,
    /// `w = 1 − 2|i − c|/L` with `L = N` for even `N` and `L = N + 1` for odd `N`.
    Triang,
    /// Parzen (de la Vallée Poussin) window, a piecewise cubic.
    Parzen,
    /// Bohman window, `w = (1 − |z|) cos(πz) + sin(π|z|)/π`.
    Bohman,
    /// Blackman window, a cosine sum with `a = [0.42, 0.5, 0.08]`.
    Blackman,
    /// Nuttall four-term minimum-sidelobe window, a cosine sum with
    /// `a = [0.3635819, 0.4891775, 0.1365995, 0.0106411]`.
    Nuttall,
    /// Four-term Blackman–Harris window, a cosine sum with
    /// `a = [0.35875, 0.48829, 0.14128, 0.01168]`.
    BlackmanHarris,
    /// Flat-top window, a cosine sum with
    /// `a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]`.
    FlatTop,
    /// Bartlett window, a triangle with zero end points, `w = 1 − |z|`.
    Bartlett,
    /// Bartlett–Hann window,
    /// `w = 0.62 − 0.48 |x − 1/2| − 0.38 cos(2πx)` with `x = i/(N − 1)`.
    Barthann,
    /// Hamming window, a cosine sum with `a = [0.54, 0.46]`.
    Hamming,
    /// Hann window, a cosine sum with `a = [0.5, 0.5]`.
    Hann,
    /// Kaiser window, `w = I₀(β √(1 − z²)) / I₀(β)`, where `I₀` is the
    /// modified Bessel function of order zero.
    ///
    /// A negative `beta` is `InvalidInput`.
    Kaiser {
        /// Nonnegative shape parameter `β`.
        beta: f64,
    },
    /// Kaiser–Bessel-derived window of even length `N`.
    ///
    /// With `v` the Kaiser window of length `N/2 + 1` and parameter `β`,
    /// `w[i] = √(Σ_{j ≤ i} v[j] / Σ_{j ≤ N/2} v[j])` for `i < N/2`, and the
    /// second half mirrors the first. A negative `beta`, an odd length or
    /// periodic sampling is `InvalidInput`.
    KaiserBesselDerived {
        /// Nonnegative shape parameter `β`.
        beta: f64,
    },
    /// Gaussian window, `w = exp(−((i − c)/σ)²/2)`.
    ///
    /// A `sigma` that is not positive is `InvalidInput`.
    Gaussian {
        /// Positive standard deviation `σ` in samples.
        sigma: f64,
    },
    /// Cosine-sum window with the given coefficients `aⱼ`, written with
    /// positive signs as for Hann, `a = [0.5, 0.5]`.
    ///
    /// An empty slice is `InvalidInput`.
    GeneralCosine(&'a [f64]),
    /// Generalized Gaussian window, `w = exp(−|(i − c)/σ|^(2p)/2)`.
    ///
    /// A `power` or `sigma` that is not positive is `InvalidInput`.
    GeneralGaussian {
        /// Positive exponent parameter `p`.
        power: f64,
        /// Positive scale `σ` in samples.
        sigma: f64,
    },
    /// Generalized Hamming window, `w = α − (1 − α) cos(2πi/(N − 1))`.
    GeneralHamming {
        /// Coefficient `α`.
        alpha: f64,
    },
    /// Dolph–Chebyshev window, whose sidelobes all lie `attenuation` dB below
    /// the main lobe. The samples are scaled to unit peak and are computed
    /// with an FFT of length `N`.
    ///
    /// An `attenuation` that is not positive is `InvalidInput`.
    Chebyshev {
        /// Positive sidelobe attenuation in dB.
        attenuation: f64,
    },
    /// Cosine (sine) window, `w = sin(π(i + 1/2)/N)`.
    Cosine,
    /// Exponential window, `w = exp(−|i − c₀|/τ)` with the center `c₀`.
    ///
    /// A `tau` that is not positive is `InvalidInput`, and so is an explicit
    /// `center` with [`WindowSymmetry::Symmetric`].
    Exponential {
        /// Positive decay scale `τ` in samples.
        tau: f64,
        /// Center `c₀` as a sample coordinate, or `None` for `c`. An explicit
        /// center requires [`WindowSymmetry::Periodic`].
        center: Option<f64>,
    },
    /// Tukey (tapered cosine) window, flat in the middle with Hann tapers over
    /// the fraction `α` of the window. `α ≤ 0` gives the rectangular window and
    /// `α ≥ 1` the Hann window.
    Tukey {
        /// Taper fraction `α`.
        alpha: f64,
    },
    /// Taylor window, whose sidelobes next to the main lobe stay near
    /// `sidelobe_level` dB below it and whose farther sidelobes decay.
    ///
    /// The samples are `w = (1 + 2 Σ_{m=1}^{nbar−1} F_m cos(2πm(i − c)/N)) / d`
    /// with the Taylor coefficients `F_m`, where `d = 1 + 2 Σ F_m` if
    /// `normalize` is set and `d = 1` otherwise. A zero `nbar` or a
    /// `sidelobe_level` that is not positive is `InvalidInput`.
    Taylor {
        /// Number of nearly constant-level sidelobes, at least one.
        nbar: usize,
        /// Positive sidelobe suppression in dB.
        sidelobe_level: f64,
        /// Scales the continuous window to one at its center when set.
        normalize: bool,
    },
    /// Lanczos window, `w = sinc(z) = sin(πz)/(πz)`.
    Lanczos,
}

/// Sampled window: holds the samples of one window function.
///
/// Most windows cost `O(N)` to prepare, and symmetric windows evaluate one
/// half and mirror it. The Taylor window costs `O(N nbar + nbar²)` and the
/// Dolph–Chebyshev window `O(N log N)`. A
/// [`WindowNormalization`] other than `Native` is computed from the returned
/// samples, after a periodic window drops its last sample.
///
/// # Accuracy
///
/// Cosine-sum windows are evaluated in `f64` from direct harmonic phases. For
/// an `m`-term sum with coefficients `aⱼ`, `A = Σ|aⱼ|`, `J = Σ j|aⱼ|` and
/// `u = 2⁻⁵³`, the absolute error of a sample is at most about
/// `2m·u·A + 10π·u·J` plus the error of the cosine evaluations. Near a zero of
/// the window this gives no relative accuracy.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{Window, WindowKind, WindowNormalization, WindowSymmetry};
/// // The symmetric Hann window of length 5 is [0, 0.5, 1, 0.5, 0].
/// let window = Window::<f64>::new(
///     WindowKind::Hann,
///     5,
///     WindowSymmetry::Symmetric,
///     WindowNormalization::Native,
/// )?;
/// assert_eq!(window.samples().len(), 5);
/// assert!((window.samples()[2] - 1.0).abs() < 1e-15 && window.samples()[0].abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - F. J. Harris, "On the use of windows for harmonic analysis with the
///   discrete Fourier transform," *Proc. IEEE* 66(1), 51–83, 1978.
/// - A. H. Nuttall, "Some windows with very good sidelobe behavior," *IEEE
///   Trans. Acoust. Speech Signal Process.* 29(1), 84–91, 1981.
/// - J. F. Kaiser, "Nonrecursive digital filter design using the I₀-sinh
///   window function," *Proc. IEEE Int. Symp. Circuits and Systems*, 20–23,
///   1974.
#[derive(Clone, Debug)]
pub struct Window<S: RealScalar> {
    samples: Vec<S>,
}

impl<S: RealScalar> Window<S> {
    /// Prepares `length` samples of the window `kind`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` or `NonFinite` if the parameters of `kind` are invalid,
    ///   as stated on its [`WindowKind`] variant.
    /// - `InvalidInput` if the sampled length exceeds 2⁵², beyond which sample
    ///   coordinates are not exact in `f64`.
    /// - `DomainError` if `normalization` is `UnitPeak` or `UnitEnergy` and all
    ///   samples are zero.
    /// - `NonFinite` if a sample computation does not stay finite.
    pub fn new(
        kind: WindowKind<'_>,
        length: usize,
        symmetry: WindowSymmetry,
        normalization: WindowNormalization,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            kind,
            length,
            symmetry,
            normalization,
            ExecutionContext::sequential_native(),
        )
    }

    /// Prepares the window like [`new`](Self::new), with the execution context
    /// `cx`.
    ///
    /// Preparation checks cancellation and the deadline of `cx` between
    /// samples, and the Dolph–Chebyshev FFT runs with `cx`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`new`](Self::new), and `ExecutionStopped` on
    /// cancellation or deadline expiry.
    pub fn build(
        kind: WindowKind<'_>,
        length: usize,
        symmetry: WindowSymmetry,
        normalization: WindowNormalization,
        cx: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        validate(kind, length, symmetry)?;
        cx.check()?;
        let extended = extended_len(length, symmetry)?;
        if normalization == WindowNormalization::Native {
            let mut samples = prepare::<S>(kind, extended, cx)?;
            samples.truncate(length);
            cx.check()?;
            return Ok(Self { samples });
        }
        let mut samples = prepare::<f64>(kind, extended, cx)?;
        samples.truncate(length);
        let (peak, energy) = normalization_scale(
            &samples,
            normalization,
            scioxide_core::argument!("normalization"),
        )?;
        let mut converted = try_with_capacity(length, scioxide_core::quantity!("window_samples"))?;
        for value in samples {
            converted.push(S::try_from_f64_lossy(value / peak / energy)?);
        }
        cx.check()?;
        Ok(Self { samples: converted })
    }

    /// Returns the samples in order.
    pub fn samples(&self) -> &[S] {
        &self.samples
    }

    /// Returns the samples, consuming the window.
    pub fn into_samples(self) -> Vec<S> {
        self.samples
    }
}

/// Computes `length` samples of the window `kind`.
///
/// This is [`Window::new`] followed by [`Window::into_samples`].
///
/// # Errors
///
/// Returns the errors of [`Window::new`].
///
/// # Examples
///
/// ```
/// use scioxide_signal::{WindowKind, WindowNormalization, WindowSymmetry, window};
/// // The periodic Hann window of length 4 drops the last sample of the
/// // symmetric window [0, 0.5, 1, 0.5, 0].
/// let hann = window::<f64>(
///     WindowKind::Hann,
///     4,
///     WindowSymmetry::Periodic,
///     WindowNormalization::Native,
/// )?;
/// assert!(hann.iter().zip([0.0, 0.5, 1.0, 0.5]).all(|(w, e)| (w - e).abs() < 1e-15));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn window<S: RealScalar>(
    kind: WindowKind<'_>,
    length: usize,
    symmetry: WindowSymmetry,
    normalization: WindowNormalization,
) -> Result<Vec<S>, SciOxideError> {
    Ok(Window::new(kind, length, symmetry, normalization)?.into_samples())
}

/// Coherent gain and equivalent noise bandwidth of window samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowMetrics<S: RealScalar> {
    coherent_gain: S,
    equivalent_noise_bandwidth: S,
}

impl<S: RealScalar> WindowMetrics<S> {
    /// Returns the coherent gain `Σᵢ w[i] / N`, the mean sample with its sign.
    pub const fn coherent_gain(self) -> S {
        self.coherent_gain
    }
    /// Returns the equivalent noise bandwidth `N Σᵢ w[i]² / (Σᵢ w[i])²` in DFT
    /// bins.
    pub const fn equivalent_noise_bandwidth(self) -> S {
        self.equivalent_noise_bandwidth
    }
}

/// Computes the coherent gain and equivalent noise bandwidth of the `N` window
/// samples `w = samples`.
///
/// See [`WindowMetrics`] for the definitions. The sums are computed in `f64`
/// with compensated summation after scaling by the largest `|w[i]|`.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `samples` is empty.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `DomainError` if the samples sum to zero, which leaves the equivalent
///   noise bandwidth undefined.
///
/// # Examples
///
/// ```
/// use scioxide_signal::window_metrics;
/// // Periodic Hann window of length 4: gain 2/4, bandwidth 4 · 1.5 / 2² bins.
/// let metrics = window_metrics(&[0.0_f64, 0.5, 1.0, 0.5])?;
/// assert!((metrics.coherent_gain() - 0.5).abs() < 1e-15);
/// assert!((metrics.equivalent_noise_bandwidth() - 1.5).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// F. J. Harris, "On the use of windows for harmonic analysis with the
/// discrete Fourier transform," *Proc. IEEE* 66(1), 51–83, 1978.
pub fn window_metrics<S: RealScalar>(samples: &[S]) -> Result<WindowMetrics<S>, SciOxideError> {
    const SAMPLES: Argument = scioxide_core::argument!("samples");
    if samples.is_empty() {
        return Err(empty(SAMPLES, "window metrics require nonempty samples"));
    }
    check_real_slice_finite(samples)?;
    let scale = samples
        .iter()
        .fold(0.0_f64, |scale, sample| scale.max(sample.into_f64().abs()));
    let zero_sum = || {
        SciOxideError::domain_error(
            SAMPLES,
            scioxide_core::DomainReason::Undefined,
            "ENBW is undefined for a zero-sum window",
        )
    };
    if scale == 0.0 {
        return Err(zero_sum());
    }
    let sum = compensated(samples.iter().map(|x| x.into_f64() / scale));
    if sum == 0.0 {
        return Err(zero_sum());
    }
    let energy = compensated(samples.iter().map(|x| (x.into_f64() / scale).powi(2)));
    let count = count(samples.len(), SAMPLES)?;
    Ok(WindowMetrics {
        coherent_gain: S::try_from_f64_lossy((sum / count) * scale)?,
        equivalent_noise_bandwidth: S::try_from_f64_lossy((energy / sum) * (count / sum))?,
    })
}

/// Exact `f64` sample coordinate `n`; an out-of-range count is attributed to
/// `argument`, the caller parameter it derives from.
pub(crate) fn count(n: usize, argument: Argument) -> Result<f64, SciOxideError> {
    // Integer sample coordinates must remain distinguishable in the working precision.
    let x = scioxide_core::backend::scalar_from_u128_rounded::<f64>(n as u128)?;
    if x > 4_503_599_627_370_496.0 {
        return Err(invalid(
            argument,
            "window length exceeds exact sample-coordinate range",
        ));
    }
    Ok(x)
}
/// Sampled length before periodic truncation of the caller's `length`.
pub(crate) fn extended_len(n: usize, symmetry: WindowSymmetry) -> Result<usize, SciOxideError> {
    let length = scioxide_core::argument!("length");
    let extended = if n > 1 && symmetry == WindowSymmetry::Periodic {
        n.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("window_extended_length"))
        })?
    } else {
        n
    };
    count(extended, length)?;
    Ok(extended)
}
pub(crate) fn compensated(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut correction) = (0.0, 0.0);
    for value in values {
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}
/// Apply `normalization`; an all-zero window is
/// attributed to `argument`, the caller parameter that requested scaling.
pub(crate) fn normalize(
    values: &mut [f64],
    normalization: WindowNormalization,
    argument: Argument,
) -> Result<(), SciOxideError> {
    let (peak, energy) = normalization_scale(values, normalization, argument)?;
    for value in values {
        *value = finite(*value)? / peak / energy;
    }
    Ok(())
}

/// One reduction obtains scaling; one application pass preserves extreme ranges.
/// The norm's scale is kept separate so even an overflowing Euclidean norm
/// can normalize finite samples. Native samples require no reduction.
fn normalization_scale(
    values: &[f64],
    normalization: WindowNormalization,
    argument: Argument,
) -> Result<(f64, f64), SciOxideError> {
    if values.is_empty() || normalization == WindowNormalization::Native {
        return Ok((1.0, 1.0));
    }
    let (peak, energy) = if normalization == WindowNormalization::UnitEnergy {
        let mut norm = scioxide_core::ScaledNorm::new();
        for &value in values {
            norm.add_magnitude(finite(value)?.abs())?;
        }
        (norm.scale(), norm.sum_squares().sqrt())
    } else {
        let mut peak = 0.0_f64;
        for &value in values {
            peak = peak.max(finite(value)?.abs());
        }
        (peak, 1.0)
    };
    if peak == 0.0 {
        return Err(SciOxideError::domain_error(
            argument,
            scioxide_core::DomainReason::Undefined,
            "cannot normalize an all-zero window",
        ));
    }
    Ok((peak, energy))
}
/// Validate window parameters.
fn validate(
    kind: WindowKind<'_>,
    length: usize,
    symmetry: WindowSymmetry,
) -> Result<(), SciOxideError> {
    use WindowKind::*;
    let kind_argument = scioxide_core::argument!("kind");
    match kind {
        Kaiser { beta } | KaiserBesselDerived { beta } => {
            check_real_slice_finite(&[beta])?;
            if beta < 0.0 {
                return Err(invalid(kind_argument, "Kaiser beta must be nonnegative"));
            }
        }
        Gaussian { sigma } => window_positive(sigma)?,
        GeneralGaussian { power, sigma } => {
            window_positive(power)?;
            window_positive(sigma)?;
        }
        GeneralCosine(a) => {
            if a.is_empty() {
                return Err(empty(kind_argument, "cosine coefficients must be nonempty"));
            }
            check_real_slice_finite(a)?;
            count(a.len(), kind_argument)?;
        }
        GeneralHamming { alpha } | Tukey { alpha } => {
            check_real_slice_finite(&[alpha])?;
        }
        Chebyshev { attenuation } => window_positive(attenuation)?,
        Exponential { tau, center } => {
            window_positive(tau)?;
            if let Some(center) = center {
                check_real_slice_finite(&[center])?;
                if symmetry == WindowSymmetry::Symmetric {
                    return Err(unsupported(
                        scioxide_core::argument!("symmetry"),
                        "explicit exponential center requires periodic sampling",
                    ));
                }
            }
        }
        Taylor {
            nbar,
            sidelobe_level,
            ..
        } => {
            window_positive(sidelobe_level)?;
            if nbar == 0 {
                return Err(invalid(kind_argument, "Taylor nbar must be positive"));
            }
            count(nbar, kind_argument)?;
        }
        _ => {}
    }
    if matches!(kind, KaiserBesselDerived { .. }) {
        if symmetry != WindowSymmetry::Symmetric {
            return Err(unsupported(
                scioxide_core::argument!("symmetry"),
                "Kaiser-Bessel-derived requires symmetric sampling",
            ));
        }
        if !length.is_multiple_of(2) {
            return Err(inconsistent(
                scioxide_core::argument!("length"),
                "Kaiser-Bessel-derived requires even length",
            ));
        }
    }
    Ok(())
}
// For m coefficients the ordinary product sum has absolute rounding error
// gamma(2*m)*sum(abs(a)), plus subnormal rounding. Direct harmonic phases
// avoid accumulated phase drift; phase/cosine errors add sum(abs(a_j)*delta_j).
fn cosine_sum(a: &[f64], phase: f64) -> f64 {
    a.iter()
        .enumerate()
        .map(|(j, &coefficient)| coefficient * (j as f64 * phase).cos())
        .sum()
}
fn kaiser_ratio(argument: f64, reference: f64, denominator: f64) -> Result<f64, SciOxideError> {
    Ok(
        scioxide_special::bessel_i_scaled(0.0, argument)? / denominator
            * (argument - reference).exp(),
    )
}

fn convert_samples<S: RealScalar>(values: Vec<f64>) -> Result<Vec<S>, SciOxideError> {
    let mut result = try_with_capacity(values.len(), scioxide_core::quantity!("window_samples"))?;
    for value in values {
        result.push(S::try_from_f64_lossy(value)?);
    }
    Ok(result)
}

/// Evaluate immutable sample coordinates, reflecting only symmetric families.
/// The length check makes all subsequent integer-to-`f64` casts exact.
fn sample_window<S: RealScalar>(
    length: usize,
    symmetric: bool,
    context: ExecutionContext<'_>,
    sample: impl Fn(usize) -> Result<f64, SciOxideError>,
) -> Result<Vec<S>, SciOxideError> {
    let mut result = try_filled(
        length,
        S::ZERO,
        scioxide_core::quantity!("window_preparation"),
    )?;
    let evaluated = if symmetric {
        length.div_ceil(2)
    } else {
        length
    };
    for i in 0..evaluated {
        context.check()?;
        let value = S::try_from_f64_lossy(finite(sample(i)?)?)?;
        result[i] = value;
        if symmetric {
            result[length - 1 - i] = value;
        }
    }
    Ok(result)
}

fn sample_formula<S: RealScalar>(
    length: usize,
    context: ExecutionContext<'_>,
    formula: impl Fn(usize) -> f64,
) -> Result<Vec<S>, SciOxideError> {
    sample_window(length, true, context, |i| Ok(formula(i)))
}

/// Sample the (extended) window, dispatching once before the numerical loop.
fn prepare<S: RealScalar>(
    kind: WindowKind<'_>,
    length: usize,
    context: ExecutionContext<'_>,
) -> Result<Vec<S>, SciOxideError> {
    use WindowKind::*;
    let n = count(length, scioxide_core::argument!("length"))?;
    if length <= 1 {
        return try_filled(
            length,
            S::ONE,
            scioxide_core::quantity!("window_preparation"),
        );
    }
    let center = (n - 1.0) * 0.5;
    let inverse_span = 1.0 / (n - 1.0);
    let z = |i: usize| 2.0 * (i as f64 * inverse_span) - 1.0;
    let phase = |i| core::f64::consts::PI * z(i);
    let cosine = |a: &[f64]| sample_formula::<S>(length, context, |i| cosine_sum(a, phase(i)));
    match kind {
        Boxcar => try_filled(
            length,
            S::ONE,
            scioxide_core::quantity!("window_preparation"),
        ),
        Triang => {
            let scale = 2.0 / if length.is_multiple_of(2) { n } else { n + 1.0 };
            sample_formula(length, context, |i| {
                1.0 - ((i as f64 - center) * scale).abs()
            })
        }
        Bartlett => sample_formula(length, context, |i| 1.0 - z(i).abs()),
        Parzen => sample_formula(length, context, |i| {
            let centered = i as f64 - center;
            let a = (2.0 * centered / n).abs();
            if centered.abs() <= (n - 1.0) * 0.25 {
                1.0 - 6.0 * a * a + 6.0 * a * a * a
            } else {
                2.0 * (1.0 - a).powi(3)
            }
        }),
        Bohman => sample_formula(length, context, |i| {
            if i == 0 {
                0.0
            } else {
                let phase = phase(i);
                (1.0 - z(i).abs()) * phase.cos() + phase.abs().sin() / core::f64::consts::PI
            }
        }),
        Blackman => cosine(&[0.42, 0.5, 0.08]),
        Nuttall => cosine(&[0.3635819, 0.4891775, 0.1365995, 0.0106411]),
        BlackmanHarris => cosine(&[0.35875, 0.48829, 0.14128, 0.01168]),
        FlatTop => cosine(&[
            0.21557895,
            0.41663158,
            0.277263158,
            0.083578947,
            0.006947368,
        ]),
        Barthann => sample_formula(length, context, |i| {
            let x = i as f64 * inverse_span;
            0.62 - 0.48 * (x - 0.5).abs() - 0.38 * (2.0 * core::f64::consts::PI * x).cos()
        }),
        Hamming => cosine(&[0.54, 0.46]),
        Hann => cosine(&[0.5, 0.5]),
        GeneralHamming { alpha } => cosine(&[alpha, 1.0 - alpha]),
        GeneralCosine(a) => cosine(a),
        Kaiser { beta } => {
            let denominator = scioxide_special::bessel_i_scaled(0.0, beta)?;
            sample_window(length, true, context, |i| {
                let z = z(i);
                kaiser_ratio(beta * (1.0 - z * z).max(0.0).sqrt(), beta, denominator)
            })
        }
        KaiserBesselDerived { beta } => {
            let half = length / 2;
            let h = half as f64;
            let nearest = 2.0 * (half / 2) as f64 / h - 1.0;
            let reference = beta * (1.0 - nearest * nearest).sqrt();
            let denominator = scioxide_special::bessel_i_scaled(0.0, reference)?;
            let mut result =
                try_filled(length, 0.0, scioxide_core::quantity!("window_preparation"))?;
            let mut cumulative = 0.0;
            for (i, output) in result.iter_mut().enumerate().take(half + 1) {
                context.check()?;
                let x = 2.0 * i.min(half - i) as f64 / h - 1.0;
                let argument = beta * (1.0 - x * x).max(0.0).sqrt();
                cumulative += kaiser_ratio(argument, reference, denominator)?;
                if i < half {
                    *output = cumulative;
                }
            }
            for i in 0..half {
                let value = finite((result[i] / cumulative).sqrt())?;
                result[i] = value;
                result[length - 1 - i] = value;
            }
            convert_samples(result)
        }
        Gaussian { sigma } => sample_formula(length, context, |i| {
            (-0.5 * ((i as f64 - center) / sigma).powi(2)).exp()
        }),
        GeneralGaussian { power, sigma } => sample_formula(length, context, |i| {
            (-0.5 * ((i as f64 - center) / sigma).abs().powf(2.0 * power)).exp()
        }),
        Cosine => {
            let step = core::f64::consts::PI / n;
            sample_formula(length, context, |i| (step * (i as f64 + 0.5)).sin())
        }
        Exponential {
            tau,
            center: explicit,
        } => {
            let center = explicit.unwrap_or(center);
            sample_window(length, explicit.is_none(), context, |i| {
                Ok((-(i as f64 - center).abs() / tau).exp())
            })
        }
        Tukey { alpha } if alpha <= 0.0 => try_filled(
            length,
            S::ONE,
            scioxide_core::quantity!("window_preparation"),
        ),
        Tukey { alpha } if alpha >= 1.0 => cosine(&[0.5, 0.5]),
        Tukey { alpha } => {
            let edge = (n - 1.0) * alpha * 0.5;
            sample_formula(length, context, |i| {
                if i == 0 {
                    0.0
                } else if i as f64 >= edge {
                    1.0
                } else {
                    0.5 * (1.0 - (core::f64::consts::PI * (i as f64 / edge)).cos())
                }
            })
        }
        Taylor {
            nbar,
            sidelobe_level,
            normalize,
        } => {
            let coefficients = taylor_coefficients(nbar, sidelobe_level, context)?;
            let denominator = if normalize {
                1.0 + 2.0 * compensated(coefficients.iter().copied())
            } else {
                1.0
            };
            let step = 2.0 * core::f64::consts::PI / n;
            sample_formula(length, context, |i| {
                (1.0 + 2.0 * cosine_sum(&coefficients, step * (i as f64 - center))) / denominator
            })
        }
        Lanczos => sample_window(length, true, context, |i| scioxide_core::sincpi(z(i))),
        Chebyshev { attenuation } => convert_samples(chebyshev(length, attenuation, context)?),
    }
}
fn taylor_coefficients(
    nbar: usize,
    level: f64,
    context: ExecutionContext<'_>,
) -> Result<Vec<f64>, SciOxideError> {
    let count = |n| count(n, scioxide_core::argument!("kind"));
    let a = acosh_exp(level * core::f64::consts::LN_10 / 20.0) / core::f64::consts::PI;
    let n = count(nbar)?;
    let s2 = n * n / (a * a + (n - 0.5).powi(2));
    let mut coefficients = try_filled(nbar, 0.0, scioxide_core::quantity!("taylor_coefficients"))?;
    for (m, coefficient) in coefficients.iter_mut().enumerate().skip(1) {
        context.check()?;
        let m2 = count(m)?.powi(2);
        let mut product = if m.is_multiple_of(2) { -0.5 } else { 0.5 };
        for j in 1..nbar {
            let jf = count(j)?;
            product *= 1.0 - m2 / s2 / (a * a + (jf - 0.5).powi(2));
            if j != m {
                product /= 1.0 - m2 / (jf * jf);
            }
        }
        *coefficient = finite(product)?;
    }
    Ok(coefficients)
}
fn acosh_exp(x: f64) -> f64 {
    x + (1.0 + (-(-2.0 * x).exp_m1()).sqrt()).ln()
}
fn log_cosh(x: f64) -> f64 {
    x.abs() + (-2.0 * x.abs()).exp().ln_1p() - core::f64::consts::LN_2
}
fn chebyshev(
    length: usize,
    attenuation: f64,
    context: ExecutionContext<'_>,
) -> Result<Vec<f64>, SciOxideError> {
    let count = |n| count(n, scioxide_core::argument!("length"));
    let n = count(length)?;
    let order = n - 1.0;
    let log_peak = attenuation * core::f64::consts::LN_10 / 20.0;
    let beta = finite((acosh_exp(log_peak) / order).cosh())?;
    let mut spectrum = try_filled(
        length,
        Complex::new(0.0, 0.0),
        scioxide_core::quantity!("chebyshev_spectrum"),
    )?;
    let inverse_n = 1.0 / n;
    for k in 0..=length / 2 {
        context.check()?;
        let angle = core::f64::consts::PI * k as f64 * inverse_n;
        let x = beta * angle.cos();
        let value = if x.abs() <= 1.0 {
            (order * x.acos()).cos() * (-log_peak).exp()
        } else {
            let sign = if x < -1.0 && length.is_multiple_of(2) {
                -1.0
            } else {
                1.0
            };
            sign * (log_cosh(order * x.abs().acosh()) - log_peak).exp()
        };
        let bin = if length.is_multiple_of(2) {
            Complex::new(
                value * angle.cos(),
                if k == 0 || k == length / 2 {
                    0.0
                } else {
                    value * angle.sin()
                },
            )
        } else {
            Complex::new(value, 0.0)
        };
        spectrum[k] = bin;
        if k > 0 && k < length - k {
            spectrum[length - k] = Complex::new(bin.re, -bin.im);
        }
    }
    let plan = FftPlan::build(
        FftPlanSpec::new(
            length,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Complex,
        ),
        context,
    )?;
    let mut scratch = plan.workspace(())?;
    plan.execute_in_place(&mut spectrum, &mut scratch, context)?;
    let mut result = try_filled(length, 0.0, scioxide_core::quantity!("chebyshev_samples"))?;
    let midpoint = length / 2;
    for (i, value) in result.iter_mut().enumerate() {
        let k = if length.is_multiple_of(2) {
            if i < midpoint {
                midpoint - i
            } else {
                i - midpoint + 1
            }
        } else {
            i.abs_diff(midpoint)
        };
        *value = spectrum[k].re;
    }
    normalize(
        &mut result,
        WindowNormalization::UnitPeak,
        scioxide_core::argument!("kind"),
    )?;
    Ok(result)
}

fn window_positive(x: f64) -> Result<(), SciOxideError> {
    crate::util::positive(x, scioxide_core::argument!("kind"), || {
        SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "input contains a non-finite value",
        )
    })
    .map(|_| ())
}
