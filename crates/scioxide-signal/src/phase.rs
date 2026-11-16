//! Vector strength of event times.
use crate::util::{
    capacity, check_real_slice_finite, empty, finite, invalid, scalar_from_usize_rounded as scalar,
};
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};
/// Vector strength and mean phase of event times at one period, from
/// [`vector_strength`] or [`vector_strength_into`].
///
/// `Default` gives zero strength and no phase, for initializing output buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorStrength<S: RealScalar> {
    strength: S,
    phase: Option<S>,
}
impl<S: RealScalar> Default for VectorStrength<S> {
    fn default() -> Self {
        Self {
            strength: S::ZERO,
            phase: None,
        }
    }
}
impl<S: RealScalar> VectorStrength<S> {
    /// Returns the vector strength `|R|`, in `[0, 1]`.
    pub fn strength(&self) -> S {
        self.strength
    }
    /// Returns the mean phase `arg R` in radians, in `[−π, π]`, or `None` when `R` is exactly zero.
    pub fn phase(&self) -> Option<S> {
        self.phase
    }
}
/// Computes the vector strength and mean phase of event times at one period.
///
/// For event times `t₁, …, t_N` and the period `T = period`, the mean resultant
/// is
///
/// ```text
/// R = (1/N) Σₖ exp(2πi (tₖ mod T) / T)
/// ```
///
/// The strength `|R|` is 1 when all events share one phase and near 0 when the
/// phases are spread uniformly, and the mean phase is `arg R`. Events need not
/// be sorted. The sums are compensated, and the cost is `O(N)`. This is
/// [`vector_strength_into`] with the single period `period`, run with
/// [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `events` is empty.
/// - `InvalidInput` (`OutOfRange`) if `period` is finite and not positive.
/// - `NonFinite` if an event time or `period` is NaN or infinite.
///
/// # Examples
///
/// ```
/// use core::f64::consts::FRAC_PI_2;
/// // Events a quarter period after each cycle start lock at phase π/2.
/// let locked = scioxide_signal::vector_strength(&[0.25_f64, 1.25, 2.25], 1.0)?;
/// let phase = locked.phase().unwrap_or(f64::NAN);
/// assert!((locked.strength() - 1.0).abs() < 1e-12 && (phase - FRAC_PI_2).abs() < 1e-12);
/// // Phases spread evenly over the period cancel.
/// let spread = scioxide_signal::vector_strength(&[0.0_f64, 0.25, 0.5, 0.75], 1.0)?;
/// assert!(spread.strength() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. M. Goldberg and P. B. Brown, "Response of binaural neurons of dog
/// superior olivary complex to dichotic tonal stimuli: some physiological
/// mechanisms of sound localization", *J. Neurophysiol.* 32(4), 1969, 613–636.
pub fn vector_strength<S: RealScalar>(
    events: &[S],
    period: S,
) -> Result<VectorStrength<S>, SciOxideError> {
    if period.is_finite() && period <= S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("period"),
            "periods must be positive",
        ));
    }
    let mut output = [VectorStrength::default()];
    vector_strength_into(
        events,
        &[period],
        &mut output,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output[0])
}
/// Computes the vector strength of event times for each period in `periods`.
///
/// For event times `t₁, …, t_N` and a period `T`, the mean resultant is
///
/// ```text
/// R = (1/N) Σₖ exp(2πi (tₖ mod T) / T)
/// ```
///
/// `|R|` is 1 when all events share one phase and near 0 when the phases are
/// spread uniformly. `output[j]` receives the result for `periods[j]`.
/// Events need not be sorted. Each time is reduced modulo `T` before it is
/// converted to an angle, so phase lost in the stored time itself cannot be
/// recovered. The sums are compensated. The cost is O(N P) for `P` periods,
/// and execution is sequential. Every result is checked before a second pass
/// writes the output, so errors leave the entire destination unchanged.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `events` is empty.
/// - `InvalidInput` (`OutOfRange`) if a period is not positive.
/// - `NonFinite` if an event time or a period is NaN or infinite.
/// - `DimensionMismatch` if `output` is shorter than `periods`.
/// - `ExecutionStopped` on cancellation or a passed deadline.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{VectorStrength, vector_strength_into};
/// let mut output = [VectorStrength::default(); 1];
/// vector_strength_into(
///     &[0.0_f64, 1.0, 2.0],
///     &[1.0],
///     &mut output,
///     ExecutionContext::sequential_native(),
/// )?;
/// // Events exactly one period apart are perfectly phase-locked.
/// assert!((output[0].strength() - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. M. Goldberg and P. B. Brown, "Response of binaural neurons of dog
/// superior olivary complex to dichotic tonal stimuli: some physiological
/// mechanisms of sound localization", *J. Neurophysiol.* 32(4), 1969, 613–636.
pub fn vector_strength_into<S: RealScalar>(
    events: &[S],
    periods: &[S],
    output: &mut [VectorStrength<S>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if events.is_empty() {
        return Err(empty(
            scioxide_core::argument!("events"),
            "at least one event time is required",
        ));
    }
    check_real_slice_finite(events)?;
    check_real_slice_finite(periods)?;
    if periods.iter().any(|&p| p <= S::ZERO) {
        return Err(invalid(
            scioxide_core::argument!("periods"),
            "periods must be positive",
        ));
    }
    capacity(periods.len(), output.len())?;
    let count = scalar::<S>(events.len())?;
    context.check()?;
    for &period in periods {
        resultant(events, period, count, || context.check())?;
    }
    context.check()?;
    for (&period, out) in periods.iter().zip(output.iter_mut()) {
        *out = resultant(events, period, count, || Ok(()))?;
    }
    Ok(())
}

/// Compensated mean resultant of finite times at a finite positive period.
fn resultant<S: RealScalar>(
    events: &[S],
    period: S,
    count: S,
    mut check: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<VectorStrength<S>, SciOxideError> {
    let mut real = scioxide_core::CompensatedSum::ZERO;
    let mut imaginary = scioxide_core::CompensatedSum::ZERO;
    for (i, &time) in events.iter().enumerate() {
        if i.is_multiple_of(256) {
            check()?;
        }
        let angle = (time % period) / period * S::TAU;
        let (sin, cos) = angle.sin_cos();
        real.add(cos / count)?;
        imaginary.add(sin / count)?;
    }
    let re = real.total()?;
    let im = imaginary.total()?;
    let strength = finite(re.hypot(im))?.min(S::ONE);
    Ok(VectorStrength {
        strength,
        phase: if strength == S::ZERO {
            None
        } else {
            Some(im.atan2(re))
        },
    })
}
