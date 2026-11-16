//! Analytic waveforms sampled at given times.
use crate::util::{
    capacity, check_real_slice_finite, finite, invalid, scalar_from_usize_rounded as scalar,
};
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{Argument, ExecutionContext, RealScalar, SciOxideError};
/// Instantaneous frequency `f(t)` of a [`Waveform::Chirp`].
///
/// `f₀` is the initial frequency, `f₁` the final frequency and `T` the
/// duration, so `f(0) = f₀` and `f(T) = f₁`. Times outside `[0, T]` follow the
/// same law.
#[derive(Clone, Copy, Debug)]
pub enum ChirpLaw {
    /// `f(t) = f₀ + (f₁ − f₀) t / T`.
    Linear,
    /// A parabola in `t` with its vertex at `t = 0` or at `t = T`.
    ///
    /// The vertex at `t = 0` gives `f(t) = f₀ + (f₁ − f₀) (t / T)²`. The vertex
    /// at `t = T` gives `f(t) = f₀ + (f₁ − f₀) (1 − (1 − t / T)²)`.
    Quadratic {
        /// Puts the vertex at `t = T` instead of `t = 0`.
        vertex_at_end: bool,
    },
    /// `f(t) = f₀ (f₁ / f₀)^(t / T)`. `f₀` and `f₁` must be nonzero with the same sign.
    Logarithmic,
    /// `f(t) = f₀ / (1 + (f₀ / f₁ − 1) t / T)`. `f₀` and `f₁` must be nonzero
    /// with the same sign. Times where the denominator is zero or negative are
    /// a `DomainError` (`Pole`).
    Hyperbolic,
}
/// Analytic waveform with closed-form samples.
///
/// [`Waveform::sample`], [`waveform`], [`waveform_into`] and
/// [`waveform_grid_into`] evaluate it. Frequencies are in cycles per time unit
/// and phases in radians (see the [crate conventions](crate#conventions)).
/// Every parameter must be finite. For the periodic shapes,
/// `u(t) = frac(f t + φ / (2π))` is the position in the period, where `f` is
/// the frequency, `φ` the phase and `frac` the fractional part.
#[derive(Clone, Copy, Debug)]
pub enum Waveform<'a, S: RealScalar> {
    /// `sin(2π f t + φ)`.
    Sine {
        /// Frequency `f`, in cycles per time unit.
        frequency: S,
        /// Phase `φ` at `t = 0`, in radians.
        phase: S,
    },
    /// `cos(2π f t + φ)`.
    Cosine {
        /// Frequency `f`, in cycles per time unit.
        frequency: S,
        /// Phase `φ` at `t = 0`, in radians.
        phase: S,
    },
    /// Square wave: `+1` for `u(t) < duty` and `−1` otherwise.
    ///
    /// A duty cycle of 1 gives the constant `+1`.
    Square {
        /// Frequency `f`, in cycles per time unit.
        frequency: S,
        /// Phase `φ` at `t = 0`, in radians.
        phase: S,
        /// Fraction of the period at `+1`, in `[0, 1]`.
        duty: S,
    },
    /// Sawtooth wave that rises from `−1` to `+1` for `u(t) < width` and falls back to `−1` after.
    ///
    /// ```text
    /// s(t) = 2 u / w − 1                    for u < w
    /// s(t) = 1 − 2 (u − w) / (1 − w)        for u ≥ w
    /// ```
    ///
    /// with `u = u(t)` and `w = width`. A width of 1 gives a rising ramp, 0 a
    /// falling ramp and 1/2 a triangle wave.
    Sawtooth {
        /// Frequency `f`, in cycles per time unit.
        frequency: S,
        /// Phase `φ` at `t = 0`, in radians.
        phase: S,
        /// Rising fraction `w` of the period, in `[0, 1]`.
        width: S,
    },
    /// Gaussian-modulated cosine `exp(−(t / σ)² / 2) cos(2π f t + φ)`.
    ///
    /// The value is exactly zero for `|t / σ| > 64`.
    GaussianPulse {
        /// Carrier frequency `f`, in cycles per time unit.
        frequency: S,
        /// Carrier phase `φ`, in radians.
        phase: S,
        /// Standard deviation `σ` of the envelope, in time units. It must be positive.
        sigma: S,
    },
    /// Swept-frequency cosine `cos(2π ∫₀ᵗ f(s) ds + φ)`, with `f` given by [`ChirpLaw`].
    Chirp {
        /// Frequency `f₀` at `t = 0`, in cycles per time unit.
        initial: S,
        /// Frequency `f₁` at `t = T`, in cycles per time unit.
        final_frequency: S,
        /// Duration `T`. It must be positive.
        duration: S,
        /// Phase `φ` at `t = 0`, in radians.
        phase: S,
        /// Law of the instantaneous frequency.
        law: ChirpLaw,
    },
    /// Swept-frequency cosine with a polynomial instantaneous frequency.
    ///
    /// With `f(t) = a₀ + a₁ t + … + a_d t^d`, the value is
    /// `cos(2π Σₖ aₖ t^(k+1) / (k + 1) + φ)`.
    PolynomialSweep {
        /// Coefficients `a₀, a₁, …` in ascending powers of `t`.
        coefficients: &'a [S],
        /// Phase `φ` at `t = 0`, in radians.
        phase: S,
    },
}
impl<S: RealScalar> Waveform<'_, S> {
    /// Validates the parameters; failures name `argument`.
    fn validate(&self, argument: Argument) -> Result<(), SciOxideError> {
        match *self {
            Self::Sine { frequency, phase } | Self::Cosine { frequency, phase } => {
                check_real_slice_finite(&[frequency, phase])
            }
            Self::Square {
                frequency,
                phase,
                duty,
            }
            | Self::Sawtooth {
                frequency,
                phase,
                width: duty,
            } => {
                check_real_slice_finite(&[frequency, phase, duty])?;
                if duty < S::ZERO || duty > S::ONE {
                    return Err(invalid(
                        argument,
                        "duty cycle or sawtooth width must lie in [0, 1]",
                    ));
                }
                Ok(())
            }
            Self::GaussianPulse {
                frequency,
                phase,
                sigma,
            } => {
                check_real_slice_finite(&[frequency, phase, sigma])?;
                if sigma <= S::ZERO {
                    return Err(invalid(argument, "Gaussian pulse sigma must be positive"));
                }
                Ok(())
            }
            Self::Chirp {
                initial,
                final_frequency,
                duration,
                phase,
                law,
            } => {
                check_real_slice_finite(&[initial, final_frequency, duration, phase])?;
                if duration <= S::ZERO {
                    return Err(invalid(argument, "chirp duration must be positive"));
                }
                if matches!(law, ChirpLaw::Logarithmic | ChirpLaw::Hyperbolic)
                    && (initial == S::ZERO
                        || final_frequency == S::ZERO
                        || initial.is_sign_negative() != final_frequency.is_sign_negative())
                {
                    return Err(SciOxideError::domain_error(
                        argument,
                        scioxide_core::DomainReason::OutsideDomain,
                        "logarithmic and hyperbolic chirps need nonzero same-sign frequencies",
                    ));
                }
                Ok(())
            }
            Self::PolynomialSweep {
                coefficients,
                phase,
            } => {
                check_real_slice_finite(coefficients)?;
                check_real_slice_finite(&[phase])
            }
        }
    }
    /// Evaluates at `t`. A hyperbolic chirp past its frequency pole is a domain
    /// error naming `argument`.
    fn value(&self, t: S, argument: Argument) -> Result<S, SciOxideError> {
        let finite = |value: S| finite(value);
        let angle = |frequency: S, phase: S| finite(finite(frequency * t)? * S::TAU + phase);
        let unit_phase = |frequency: S, phase: S| -> Result<S, SciOxideError> {
            let cycles = finite(frequency * t + phase / S::TAU)?;
            Ok(cycles - cycles.floor())
        };
        finite(match *self {
            Self::Sine { frequency, phase } => angle(frequency, phase)?.sin(),
            Self::Cosine { frequency, phase } => angle(frequency, phase)?.cos(),
            Self::Square {
                frequency,
                phase,
                duty,
            } => {
                if duty == S::ONE || unit_phase(frequency, phase)? < duty {
                    S::ONE
                } else {
                    -S::ONE
                }
            }
            Self::Sawtooth {
                frequency,
                phase,
                width,
            } => {
                let u = unit_phase(frequency, phase)?;
                let two = S::TWO;
                if width == S::ONE || u < width {
                    two * (u / width) - S::ONE
                } else {
                    S::ONE - two * ((u - width) / (S::ONE - width))
                }
            }
            Self::GaussianPulse {
                frequency,
                phase,
                sigma,
            } => {
                let q = t / sigma;
                if !q.is_finite() || q.abs() > scalar::<S>(64)? {
                    S::ZERO
                } else {
                    (-S::HALF * q * q).exp() * angle(frequency, phase)?.cos()
                }
            }
            Self::Chirp {
                initial,
                final_frequency,
                duration,
                phase,
                law,
            } => {
                let u = finite(t / duration)?;
                let delta = finite(final_frequency - initial)?;
                let cycles = match law {
                    ChirpLaw::Linear => finite(t * (initial + delta * u * S::HALF))?,
                    ChirpLaw::Quadratic { vertex_at_end } => {
                        let shape = if vertex_at_end {
                            u - u * u / scalar::<S>(3)?
                        } else {
                            u * u / scalar::<S>(3)?
                        };
                        finite(t * (initial + delta * shape))?
                    }
                    ChirpLaw::Logarithmic => {
                        let log_ratio = final_frequency.abs().ln() - initial.abs().ln();
                        if log_ratio == S::ZERO {
                            finite(initial * t)?
                        } else {
                            finite(initial * duration * (log_ratio * u).exp_m1() / log_ratio)?
                        }
                    }
                    ChirpLaw::Hyperbolic => {
                        // f(t)=f0/(1 + (f0/f1 - 1)*t/T).
                        let b = finite(initial / final_frequency - S::ONE)?;
                        let z = finite(b * u)?;
                        if z <= -S::ONE {
                            return Err(SciOxideError::domain_error(
                                argument,
                                scioxide_core::DomainReason::Pole,
                                "hyperbolic chirp crosses its frequency pole",
                            ));
                        }
                        if b == S::ZERO {
                            finite(initial * t)?
                        } else {
                            finite(initial * duration * z.ln_1p() / b)?
                        }
                    }
                };
                finite(S::TAU * cycles + phase)?.cos()
            }
            Self::PolynomialSweep {
                coefficients,
                phase,
            } => {
                let mut integral = S::ZERO;
                for (i, &a) in coefficients.iter().enumerate().rev() {
                    integral = finite(integral * t + a / scalar::<S>(i + 1)?)?;
                }
                finite(S::TAU * finite(integral * t)? + phase)?.cos()
            }
        })
    }
    /// Evaluates the waveform at `time`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `time` or a parameter is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `duty` or `width` is outside `[0, 1]`,
    ///   or `sigma` or `duration` is not positive.
    /// - `DomainError` (`OutsideDomain`) if a logarithmic or hyperbolic chirp
    ///   has a zero frequency or frequencies of opposite sign.
    /// - `DomainError` (`Pole`) if `time` is at or past the frequency pole of a
    ///   hyperbolic chirp.
    /// - `NumericalInstability` (`Overflow`) if the phase or the value overflows.
    pub fn sample(&self, time: S) -> Result<S, SciOxideError> {
        let argument = scioxide_core::argument!("self");
        self.validate(argument)?;
        check_real_slice_finite(&[time])?;
        self.value(time, argument)
    }
}
/// Evaluates `waveform` at each of `times` and returns the values.
///
/// Entry `k` of the result is the value at `times[k]`, as from
/// [`Waveform::sample`]. The times need not be sorted or uniform; for a uniform
/// grid generated in chunks, use [`waveform_grid_into`]. The cost is one
/// closed-form evaluation per time. This allocates the result, calls
/// [`waveform_into`] and runs with [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - The errors of [`Waveform::sample`] for each time.
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{Waveform, waveform};
/// let sine = Waveform::Sine {
///     frequency: 2.0_f64,
///     phase: 0.0,
/// };
/// // sin(2π·2t) at t = 1/8 and t = 3/8 is 1 and −1.
/// let values = waveform(sine, &[0.125, 0.375])?;
/// assert!((values[0] - 1.0).abs() < 1e-12 && (values[1] + 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn waveform<S: RealScalar>(
    waveform: Waveform<'_, S>,
    times: &[S],
) -> Result<Vec<S>, SciOxideError> {
    let mut output = try_filled(
        times.len(),
        S::ZERO,
        scioxide_core::quantity!("waveform_output"),
    )?;
    waveform_into(
        waveform,
        times,
        &mut output,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
/// Evaluates `waveform` at each of `times` and writes the values to `output`.
///
/// `output[k]` receives the value at `times[k]`. `output` must hold at least
/// `times.len()` values, and only that prefix is written. Each sample is
/// evaluated twice, once to check it before any output is written.
///
/// # Errors
///
/// - The errors of [`Waveform::sample`] for each time.
/// - `DimensionMismatch` if `output` is shorter than `times`.
/// - `ExecutionStopped` on cancellation or a passed deadline.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{Waveform, waveform_into};
/// let mut output = [0.0; 2];
/// waveform_into(
///     Waveform::Cosine {
///         frequency: 1.0_f64,
///         phase: 0.0,
///     },
///     &[0.0, 0.5],
///     &mut output,
///     ExecutionContext::sequential_native(),
/// )?;
/// // cos(0) = 1 and cos(π) = −1.
/// assert!((output[0] - 1.0).abs() < 1e-12 && (output[1] + 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn waveform_into<S: RealScalar>(
    waveform: Waveform<'_, S>,
    times: &[S],
    output: &mut [S],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let argument = scioxide_core::argument!("waveform");
    waveform.validate(argument)?;
    check_real_slice_finite(times)?;
    capacity(times.len(), output.len())?;
    context.check()?;
    times
        .iter()
        .try_for_each(|&t| waveform.value(t, argument).map(|_| ()))?;
    context.check()?;
    for (y, &t) in output.iter_mut().zip(times) {
        *y = waveform.value(t, argument)?;
    }
    Ok(())
}
/// Evaluates `waveform` on the uniform grid `tᵢ = origin + (start_index + i) · step`.
///
/// `output[i]` receives the value at `tᵢ` for `i` in `0..output.len()`. Because
/// the grid uses the absolute index `start_index + i`, a long signal generated
/// in chunks equals the signal generated in one call. Each sample is evaluated
/// twice, once to check it before any output is written.
///
/// # Errors
///
/// - The errors of [`Waveform::sample`] for each grid time.
/// - `NonFinite` if `origin` or `step` is NaN or infinite.
/// - `SizeOverflow` if `start_index + output.len()` overflows `usize`.
/// - `ExecutionStopped` on cancellation or a passed deadline.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{Waveform, waveform_grid_into};
/// let cosine = Waveform::Cosine {
///     frequency: 0.25_f64,
///     phase: 0.0,
/// };
/// let (mut samples, cx) = ([0.0; 4], ExecutionContext::sequential_native());
/// let (head, tail) = samples.split_at_mut(2);
/// waveform_grid_into(cosine, 0.0, 1.0, 0, head, cx)?;
/// waveform_grid_into(cosine, 0.0, 1.0, 2, tail, cx)?;
/// // Two chunks give cos(πk/2) for k = 0, 1, 2, 3.
/// for (s, e) in samples.iter().zip([1.0, 0.0, -1.0, 0.0]) {
///     assert!((s - e).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn waveform_grid_into<S: RealScalar>(
    waveform: Waveform<'_, S>,
    origin: S,
    step: S,
    start_index: usize,
    output: &mut [S],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let argument = scioxide_core::argument!("waveform");
    waveform.validate(argument)?;
    check_real_slice_finite(&[origin, step])?;
    start_index
        .checked_add(output.len())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("waveform_index")))?;
    context.check()?;
    let sample = |i: usize| -> Result<S, SciOxideError> {
        waveform.value(
            finite(origin + scalar::<S>(start_index + i)? * step)?,
            argument,
        )
    };
    (0..output.len()).try_for_each(|i| sample(i).map(|_| ()))?;
    context.check()?;
    for (i, y) in output.iter_mut().enumerate() {
        *y = sample(i)?;
    }
    Ok(())
}
/// Writes a block of samples of the unit impulse at index `impulse_index`.
///
/// The block holds samples `start_index..start_index + output.len()`:
/// `output[i]` is 1 if `start_index + i == impulse_index` and 0 otherwise, so a
/// block that does not contain the impulse is all zeros.
///
/// # Errors
///
/// - `SizeOverflow` if `start_index + output.len()` overflows `usize`.
/// - `ExecutionStopped` on cancellation or a passed deadline.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::unit_impulse_into;
/// let (mut first, mut second) = ([9.0_f64; 3], [9.0; 3]);
/// unit_impulse_into(4, 0, &mut first, ExecutionContext::sequential_native())?;
/// unit_impulse_into(4, 3, &mut second, ExecutionContext::sequential_native())?;
/// assert_eq!((first, second), ([0.0; 3], [0.0, 1.0, 0.0]));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn unit_impulse_into<S: RealScalar>(
    impulse_index: usize,
    start_index: usize,
    output: &mut [S],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    start_index
        .checked_add(output.len())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("impulse_index")))?;
    context.check()?;
    output.iter_mut().enumerate().for_each(|(i, y)| {
        *y = if start_index + i == impulse_index {
            S::ONE
        } else {
            S::ZERO
        }
    });
    Ok(())
}
