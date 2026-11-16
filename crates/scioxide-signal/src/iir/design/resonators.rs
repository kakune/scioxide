use super::*;

/// Choice between rejecting and passing the center frequencies of
/// [`design_notch_peak`] and [`design_comb`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResonanceKind {
    /// Zeros at the center frequencies.
    Notch,
    /// Resonant peaks with unit gain at the center frequencies.
    Peak,
}

/// Designs a second-order digital notch or peak filter.
///
/// `frequency` is the center frequency `f₀` in multiples of Nyquist, and
/// `quality` is `Q = f₀/Δf`, where `Δf` is the −3 dB bandwidth. With
/// `β = tan(πΔf/2)` and `g = 1/(1 + β)`, the filter is
///
/// ```text
/// notch: H(z) = g (1 − 2 cos(πf₀) z⁻¹ + z⁻²) / D(z)
/// peak:  H(z) = (1 − g) (1 − z⁻²) / D(z)
/// D(z) = 1 − 2g cos(πf₀) z⁻¹ + (2g − 1) z⁻²
/// ```
///
/// The notch has a zero at `f₀` and unit gain at DC and Nyquist. The peak has
/// unit gain at `f₀` and zeros at DC and Nyquist.
///
/// # Errors
///
/// - `NonFinite` if `frequency` or `quality` is NaN or infinite.
/// - `InvalidInput` if `frequency` is not in `(0, 1)`, if `quality` is not
///   positive, if `frequency/quality` is not below one, or if the rounded
///   poles are not strictly stable.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{ResonanceKind, design_notch_peak};
/// // Notch at 0.3 times Nyquist with Q = 20.
/// let notch = design_notch_peak(0.3_f64, 20.0, ResonanceKind::Notch)?;
/// assert!(notch.response(0.3)?.norm() < 1e-12);
/// assert!((notch.response(0.0)?.norm() - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. J. Orfanidis, *Introduction to Signal Processing*, Prentice Hall, 1996,
/// ch. 11.
pub fn design_notch_peak<S: RealScalar>(
    frequency: S,
    quality: S,
    kind: ResonanceKind,
) -> Result<SecondOrderSections<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    positive_parameter(frequency, scioxide_core::argument!("frequency"))?;
    positive_parameter(quality, scioxide_core::argument!("quality"))?;
    if frequency >= S::ONE {
        return Err(invalid(
            scioxide_core::argument!("frequency"),
            "frequency must lie below Nyquist",
        ));
    }
    let bandwidth = frequency / quality;
    if bandwidth >= S::ONE {
        return Err(invalid(
            scioxide_core::argument!("quality"),
            "bandwidth frequency / quality must lie below Nyquist",
        ));
    }
    let beta = (S::PI * bandwidth * S::HALF).tan();
    let gain = S::ONE / (S::ONE + beta);
    let two = S::TWO;
    let cos = (S::PI * frequency).cos();
    let b = if kind == ResonanceKind::Peak {
        [S::ONE - gain, S::ZERO, gain - S::ONE]
    } else {
        [gain, -two * gain * cos, gain]
    };
    SecondOrderSections::new(
        IirDomain::Digital,
        &[[
            b[0],
            b[1],
            b[2],
            S::ONE,
            -two * gain * cos,
            two * gain - S::ONE,
        ]],
    )
}
/// Designs a digital comb filter with integer delay `D` and feedback `α`.
///
/// `D` is `delay` and `α` is `feedback`:
///
/// ```text
/// notch: H(z) = (1 + α)/2 · (1 − z⁻ᴰ) / (1 − α z⁻ᴰ)
/// peak:  H(z) = (1 − α) / (1 − α z⁻ᴰ)
/// ```
///
/// The harmonics are the frequencies `2k/D`, `k = 0, 1, …`, in multiples of
/// Nyquist. The notch comb has zeros at the harmonics and unit gain halfway
/// between them. The peak comb has unit gain at the harmonics. For
/// `0 < α < 1`, the notches or peaks narrow as `α` approaches one.
///
/// # Errors
///
/// - `NonFinite` if `feedback` is NaN or infinite.
/// - `InvalidInput` if `delay` is zero or `|feedback| ≥ 1`.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{ResonanceKind, design_comb};
/// let notch = design_comb(4, 0.5_f64, ResonanceKind::Notch)?;
/// assert!(notch.response(0.5)?.norm() < 1e-14);
/// assert!((notch.response(0.25)?.norm() - 1.0).abs() < 1e-14);
/// let peak = design_comb(4, 0.5_f64, ResonanceKind::Peak)?;
/// assert!((peak.response(0.5)?.norm() - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn design_comb<S: RealScalar>(
    delay: usize,
    feedback: S,
    kind: ResonanceKind,
) -> Result<TransferFunction<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    finite_parameter(feedback)?;
    if delay == 0 {
        return Err(invalid(
            scioxide_core::argument!("delay"),
            "delay must be positive",
        ));
    }
    if feedback.abs() >= S::ONE {
        return Err(invalid(
            scioxide_core::argument!("feedback"),
            "feedback magnitude must be below one",
        ));
    }
    let mut b = zeros(add(delay, 1)?)?;
    let mut a = zeros(b.len())?;
    a[0] = S::ONE;
    a[delay] = -feedback;
    if kind == ResonanceKind::Peak {
        b[0] = S::ONE - feedback;
    } else {
        b[0] = (S::ONE + feedback) * S::HALF;
        b[delay] = -b[0];
    }
    TransferFunction::new(IirDomain::Digital, &b, &a)
}
/// Designs a complex gammatone filter as `order` identical first-order sections.
///
/// The filter approximates the analytic gammatone of order `n` (`order`),
/// center frequency `f₀` (`frequency`) and bandwidth parameter `b`
/// (`bandwidth`).
/// With time `t` in samples and `f₀`, `b` in multiples of Nyquist, the
/// gammatone is `tⁿ⁻¹ e^(−πbt) e^(iπf₀t)`. Each section is
///
/// ```text
/// H₁(z) = (1 − r) / (1 − p z⁻¹),   p = r exp(iπf₀),   r = exp(−πb)
/// ```
///
/// so the cascade `H = H₁ⁿ` has unit gain at `f₀`. Its impulse response is
/// `(1 − r)ⁿ C(k + n − 1, n − 1) pᵏ` at sample `k ≥ 0`, where `C` is the
/// binomial coefficient.
///
/// # Errors
///
/// - `NonFinite` if `frequency` or `bandwidth` is NaN or infinite.
/// - `InvalidInput` if `order` is zero, if `frequency` is not in `[0, 1)`, or
///   if `bandwidth` is not positive.
/// - `NumericalInstability` (`LossOfSignificance`) if the pole radius `r`
///   rounds to zero or one.
///
/// # Examples
///
/// ```
/// use scioxide_signal::design_gammatone;
/// let gammatone = design_gammatone(4, 0.25_f64, 0.05)?;
/// assert_eq!(gammatone.sections().len(), 4);
/// assert!((gammatone.response(0.25)?.norm() - 1.0).abs() < 1e-12);
/// assert!(gammatone.response(0.75)?.norm() < 1e-3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// V. Hohmann, "Frequency analysis and synthesis using a Gammatone
/// filterbank", *Acta Acustica united with Acustica* 88 (2002), 433–442.
pub fn design_gammatone<S: RealScalar>(
    order: usize,
    frequency: S,
    bandwidth: S,
) -> Result<SecondOrderSections<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    finite_parameter(frequency)?;
    positive_parameter(bandwidth, scioxide_core::argument!("bandwidth"))?;
    if order == 0 {
        return Err(invalid(
            scioxide_core::argument!("order"),
            "order must be positive",
        ));
    }
    if frequency < S::ZERO || frequency >= S::ONE {
        return Err(invalid(
            scioxide_core::argument!("frequency"),
            "frequency must lie in [0, 1)",
        ));
    }
    let radius = (-S::PI * bandwidth).exp();
    if radius == S::ZERO || radius >= S::ONE {
        return Err(failure("gammatone pole rounding"));
    }
    let p = Complex::from_polar(radius, S::PI * frequency);
    let mut rows = Vec::new();
    rows.try_reserve_exact(order).map_err(|_| {
        SciOxideError::allocation_failure(scioxide_core::quantity!("gammatone"), order)
    })?;
    for _ in 0..order {
        rows.push([
            c(S::ONE - radius),
            c(S::ZERO),
            c(S::ZERO),
            c(S::ONE),
            -p,
            c(S::ZERO),
        ]);
    }
    SecondOrderSections::new(IirDomain::Digital, &rows)
}

/// Designs a real gammatone filter of order `2·order` as real second-order
/// sections.
///
/// The filter is `(H + H̄)/2`, rescaled to unit gain at `frequency`, where `H`
/// is the complex filter of [`design_gammatone`] and `H̄` the filter with
/// conjugate coefficients. Its impulse response is therefore proportional to
/// the real part of that of `H`. It has `order`-fold poles at `p` and `p̄` and
/// `order` zeros, which come from the polynomial root solver and are checked
/// as in [`TransferFunction::to_zpk`]. The root solve runs in a fresh root
/// scope; [`design_real_gammatone_scoped`] runs it inside the caller's solve.
/// At `frequency = 0` the filter of [`design_gammatone`] is already real, and
/// the result is its `order` first-order sections.
///
/// # Errors
///
/// - The errors of [`design_gammatone`].
/// - `NumericalInstability` if no computed set of zeros passes the
///   backward-error test, if the response vanishes at `frequency`, or if the
///   gain is above the finite range or nonzero and below the normal range.
/// - `DomainError` (`OutsideDomain`) if the section coefficients are not real
///   to within `128 ε`.
///
/// # Examples
///
/// ```
/// use scioxide_signal::design_real_gammatone;
/// let gammatone = design_real_gammatone(2, 0.25_f64, 0.05)?;
/// assert_eq!(gammatone.sections().len(), 2);
/// assert!((gammatone.response(0.25)?.norm() - 1.0).abs() < 1e-12);
/// // Real coefficients give a conjugate-symmetric response.
/// let (positive, negative) = (gammatone.response(0.4)?, gammatone.response(-0.4)?);
/// assert!((positive - negative.conj()).norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn design_real_gammatone<S>(
    order: usize,
    frequency: S,
    bandwidth: S,
) -> Result<SecondOrderSections<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    design_real_gammatone_scoped(
        order,
        frequency,
        bandwidth,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Designs a real gammatone filter as [`design_real_gammatone`] does, as one
/// epoch of `scope`.
///
/// The root solve of the zeros counts toward the limits of `scope` and
/// observes its cancellation and deadline, as described in
/// [scoped designs](crate#scoped-designs).
///
/// # Errors
///
/// - The errors of [`design_real_gammatone`].
/// - `ConvergenceFailure` if the root solve exhausts the limits of `scope`.
/// - `ExecutionStopped` on cancellation or a passed deadline.
pub fn design_real_gammatone_scoped<'scope, 'capability, S>(
    order: usize,
    frequency: S,
    bandwidth: S,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<SecondOrderSections<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    run_design(
        scope,
        scioxide_core::algorithm_id!("signal", "real_gammatone", 1),
        |context| real_gammatone(order, frequency, bandwidth, context),
    )
}
fn real_gammatone<S>(
    order: usize,
    frequency: S,
    bandwidth: S,
    context: &mut SolveContext<'_, '_>,
) -> Result<SecondOrderSections<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    let analytic = design_gammatone(order, frequency, bandwidth)?;
    if frequency == S::ZERO {
        return analytic.to_real(scioxide_core::ResidualTolerance::default());
    }
    let pole = -analytic.sections[0][4];
    let p = zeros::<Complex<S>>(order)?;
    let mut poles = p;
    poles.fill(pole);
    let denominator = expand(&poles)?;
    let mut numerator = zeros(order + 1)?;
    for (i, value) in denominator.iter().enumerate() {
        numerator[i] = value.re;
    }
    numerator.reverse();
    let z = roots(&numerator, Some(context))?;
    poles.try_reserve_exact(order).map_err(|_| {
        SciOxideError::allocation_failure(scioxide_core::quantity!("gammatone_poles"), order)
    })?;
    poles.resize(2 * order, pole.conj());
    // Sum of conjugate rational functions has numerator degree order and denominator 2*order.
    // Multiply by z^order to express the causal q=z^-1 coefficients without an unintended delay.
    let mut z = z;
    z.try_reserve_exact(order).map_err(|_| {
        SciOxideError::allocation_failure(scioxide_core::quantity!("gammatone_zeros"), order)
    })?;
    z.resize(z.len() + order, c(S::ZERO));
    let mut model = PoleZeroGain::new(IirDomain::Digital, &z, &poles, c(S::ONE))?;
    let magnitude = model.response(frequency)?.norm();
    if magnitude == S::ZERO {
        return Err(failure("gammatone center response vanished"));
    }
    model.gain = ZpkGain::new(c(S::ONE))?.unscale(magnitude)?.finish()?;
    model
        .to_sos()?
        .to_real(scioxide_core::ResidualTolerance::new(
            S::EPSILON * number::<S>(128.)?,
            S::EPSILON * number::<S>(128.)?,
        )?)
}
