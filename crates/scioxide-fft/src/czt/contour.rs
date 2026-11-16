//! Geometric contours and the evaluation of their points.
use crate::chirp::phase;
use crate::validation::invalid;
use scioxide_core::{Argument, Complex, RealScalar, Scaled, ScaledValue, SciOxideError};

/// Describes the geometric contour on which a [`CztPlan`](crate::CztPlan)
/// evaluates the z-transform.
///
/// The points are
///
/// ```text
/// z_k = exp(ρ₀ + k ρ + 2πi (t₀ + k t)),   k = 0, 1, …,
/// ```
///
/// where `ρ₀` is `log_radius`, `ρ` is `log_radius_step`, `t₀` is `start_turns`
/// and `t` is `step_turns`; one turn is 2π radians. The contour is a circle
/// when `ρ = 0` and a spiral otherwise. Storing logarithms lets the radii lie
/// outside the range of `S`.
#[derive(Debug, Clone, Copy)]
pub struct CztContour<S: RealScalar> {
    log_radius: S,
    log_radius_step: S,
    start_turns: f64,
    step_turns: f64,
    step_denominator: Option<usize>,
}
impl<S: RealScalar> CztContour<S> {
    /// Creates a contour from its log-radius, log-radius step, initial phase and
    /// phase step in turns.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    pub fn new(
        log_radius: S,
        log_radius_step: S,
        start_turns: S,
        step_turns: S,
    ) -> Result<Self, SciOxideError> {
        if [log_radius, log_radius_step, start_turns, step_turns]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(crate::validation::nonfinite());
        }
        Ok(Self {
            log_radius,
            log_radius_step,
            start_turns: start_turns.try_to_f64_exact()?,
            step_turns: step_turns.try_to_f64_exact()?,
            step_denominator: None,
        })
    }
    /// Creates the contour `z_k = exp(2πi k / points)` of the DFT of length
    /// `points`.
    ///
    /// The phases `k/points` are reduced in exact integer arithmetic.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `points` is zero.
    pub fn dft(points: usize) -> Result<Self, SciOxideError> {
        if points == 0 {
            return Err(invalid(
                scioxide_core::InvalidInputReason::Empty,
                scioxide_core::argument!("points"),
                "DFT contour requires positive points",
            ));
        }
        Ok(Self {
            log_radius: S::ZERO,
            log_radius_step: S::ZERO,
            start_turns: 0.0,
            step_turns: 1.0 / points as f64,
            step_denominator: Some(points),
        })
    }
    /// Creates the unit-circle contour that samples the frequency band from
    /// `start` to `end` at `m` frequencies.
    ///
    /// The frequencies are `f_k = start + k Δ` and the points are
    /// `z_k = exp(2πi f_k / sample_rate)`. With `endpoint` the band includes
    /// `end` and `Δ = (end − start)/(m − 1)`; otherwise `Δ = (end − start)/m`
    /// and `end` is excluded. A single point samples `start`. Negative and
    /// descending bands are valid. Finite normalized endpoints may span more
    /// than the scalar range when their phase step remains representable.
    /// A band of exactly one full turn has its
    /// phases reduced in exact integer arithmetic, as for [`Self::dft`].
    /// [`zoom_fft`](crate::zoom_fft) takes the same arguments.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `m` is zero.
    /// - `NonFinite` if `start`, `end` or `sample_rate` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `sample_rate` is not positive.
    /// - `NumericalInstability` (`Overflow`) if the normalized start or phase
    ///   step overflows binary64 arithmetic.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::f64::consts::TAU;
    /// use scioxide_core::Complex;
    /// use scioxide_fft::CztContour;
    ///
    /// // 32 frequencies in [0.1, 0.2) at unit sample rate: f_k = 0.1 + k/320.
    /// let contour = CztContour::zoom(0.1_f64, 0.2, 32, 1.0, false)?;
    /// let expected = Complex::from_polar(1.0, TAU * (0.1 + 1.0 / 320.0));
    /// assert!((contour.point(1)? - expected).norm() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn zoom(
        start: S,
        end: S,
        m: usize,
        sample_rate: S,
        endpoint: bool,
    ) -> Result<Self, SciOxideError> {
        if m == 0 {
            return Err(invalid(
                scioxide_core::InvalidInputReason::Empty,
                scioxide_core::argument!("m"),
                "zoom contour requires positive points",
            ));
        }
        if !start.is_finite() || !end.is_finite() || !sample_rate.is_finite() {
            return Err(crate::validation::nonfinite());
        }
        if sample_rate <= S::ZERO {
            return Err(invalid(
                scioxide_core::InvalidInputReason::OutOfRange,
                scioxide_core::argument!("sample_rate"),
                "sample rate must be positive",
            ));
        }
        let start = start.try_to_f64_exact()? / sample_rate.try_to_f64_exact()?;
        let end = end.try_to_f64_exact()? / sample_rate.try_to_f64_exact()?;
        let denominator = if endpoint && m > 1 { m - 1 } else { m };
        let span = end - start;
        let step = if m == 1 {
            0.0
        } else if !span.is_finite() && start.is_finite() && end.is_finite() {
            Scaled::from_value(end)?
                .sub(Scaled::from_value(start)?)?
                .div(Scaled::from_value(denominator as f64)?)?
                .round()?
        } else {
            span / denominator as f64
        };
        if !start.is_finite() || !step.is_finite() {
            return Err(crate::validation::numerical(
                scioxide_core::NumericalIssue::Overflow,
                "normalized zoom frequencies overflow",
            ));
        }
        // A zoom spanning exactly one full turn has a rational step
        // 1 / denominator, so it is given an exact integer denominator and
        // its phases are reduced exactly, as for `CztContour::dft`.
        let contour = CztContour {
            log_radius: S::ZERO,
            log_radius_step: S::ZERO,
            start_turns: start,
            step_turns: step,
            step_denominator: if m > 1 && span == 1.0 {
                Some(denominator)
            } else {
                None
            },
        };
        Ok(contour)
    }
    // Returns exp(+-2*pi*i * step * index), or with `half` the square root
    // step/2 needed by the Bluestein chirps (index is then j^2). A rational
    // DFT-style step keeps an exact integer denominator so the phase is
    // reduced in integers; otherwise the binary turn count is reduced by
    // `turns_product`. Neither path ever forms step * index in floating point.
    // Failures are attributed to `argument`.
    pub(super) fn step_phase(
        self,
        index: u128,
        half: bool,
        inverse: bool,
        argument: Argument,
    ) -> Result<Complex<S>, SciOxideError> {
        if let Some(denominator) = self.step_denominator {
            crate::numeric::reduced_root(
                index,
                denominator as u128 * if half { 2 } else { 1 },
                !inverse,
                argument,
            )
        } else {
            phase(
                self.step_turns * if half { 0.5 } else { 1.0 } * if inverse { -1.0 } else { 1.0 },
                index,
            )
        }
    }
    /// Creates the contour `z_k = a · w^(−k)` from its first point `a` and the
    /// ratio `w` between successive points.
    ///
    /// [`czt`](crate::czt) takes the same `(w, a)` arguments. A computed
    /// log-modulus of magnitude at most `2ε`, with `ε` the machine epsilon of
    /// `S`, is set to zero, so `w = exp(−2πi/n)` rounded to `S` still gives a
    /// unit-circle contour. Use [`Self::new`] for a contour whose log-radius
    /// is that small but nonzero, or to give the phases exactly in turns.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a component of `w` or `a` is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if `w` or `a` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Complex, ExecutionContext};
    /// use scioxide_fft::{CztContour, CztPlan, CztPlanSpec, Normalization, czt};
    ///
    /// // Three points z_k = 0.9 exp(0.5ik) on the circle |z| = 0.9.
    /// let (w, a) = (Complex::from_polar(1.0_f64, -0.5), Complex::new(0.9, 0.0));
    /// let x = [Complex::new(1.0, 0.0), Complex::new(-2.0, 0.5)];
    /// let expected = czt(&x, 3, w, a, Normalization::Unscaled)?;
    /// let spec = CztPlanSpec::new(2, 3, CztContour::from_wa(w, a)?, Normalization::Unscaled);
    /// let plan = CztPlan::new(spec)?;
    /// let mut output = [Complex::new(0.0, 0.0); 3];
    /// let cx = ExecutionContext::sequential_native();
    /// plan.execute_into(&x[..], &mut output[..], &mut plan.workspace(())?, cx)?;
    /// for (&p, &q) in output.iter().zip(&expected) {
    ///     assert!((p - q).norm() < 1e-14);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_wa(w: Complex<S>, a: Complex<S>) -> Result<Self, SciOxideError> {
        // Contour points are z_k = a * w^(-k), so the log-radius and turn
        // steps are the negated log|w| and arg(w) / (2*pi). log|z| is formed
        // as ln(scale) + ln(|z / scale|^2) / 2 so |z|^2 cannot overflow or
        // underflow for extreme a or w.
        let polar = |z: Complex<S>, argument: Argument| -> Result<(S, S), SciOxideError> {
            if !z.re.is_finite() || !z.im.is_finite() {
                return Err(crate::validation::nonfinite());
            }
            let scale = z.re.abs().max(z.im.abs());
            if scale.is_zero() {
                return Err(invalid(
                    scioxide_core::InvalidInputReason::OutOfRange,
                    argument,
                    "a and w must be nonzero",
                ));
            }
            let re = z.re / scale;
            let im = z.im / scale;
            let log = scale.ln() + (re * re + im * im).ln() / S::TWO;
            // A rounded unit point contributes at most u to its log modulus;
            // polar evaluation adds about 1.7u (the final cancellation is
            // exact by Sterbenz). The 4u window covers both widths, while
            // genuinely off-circle radii about 6u and beyond remain explicit.
            let log = if log.abs() <= S::TWO * S::EPSILON {
                S::ZERO
            } else {
                log
            };
            Ok((log, z.im.atan2(z.re) / S::TAU))
        };
        let (ar, ap) = polar(a, scioxide_core::argument!("a"))?;
        let (wr, wp) = polar(w, scioxide_core::argument!("w"))?;
        Self::new(ar, if wr.is_zero() { S::ZERO } else { -wr }, ap, -wp)
    }
    /// Returns the log-radius `ρ₀ = ln |z₀|`.
    pub const fn log_radius(self) -> S {
        self.log_radius
    }
    /// Returns the log-radius step `ρ = ln |z_(k+1) / z_k|`.
    pub const fn log_radius_step(self) -> S {
        self.log_radius_step
    }
    /// Returns the initial phase `t₀` in turns.
    pub const fn start_turns(self) -> f64 {
        self.start_turns
    }
    /// Returns the phase step `t` in turns.
    pub const fn step_turns(self) -> f64 {
        self.step_turns
    }
    /// Returns the contour point `z_index`.
    ///
    /// Plans do not need the points to be representable; this method does.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if `z_index` is not representable in `S`.
    pub fn point(self, index: usize) -> Result<Complex<S>, SciOxideError> {
        let magnitude = radial::<S>(
            self.log_radius.try_to_f64_exact()?
                + self.log_radius_step.try_to_f64_exact()? * index as f64,
        )?;
        let unit = phase::<S>(self.start_turns, 1)?
            * self.step_phase(
                index as u128,
                false,
                false,
                scioxide_core::argument!("self"),
            )?;
        ScaledValue::from_components(
            magnitude.mul(Scaled::from_value(unit.re)?)?,
            magnitude.mul(Scaled::from_value(unit.im)?)?,
        )
        .round()
    }
}

pub(super) fn radial<S: RealScalar>(log: f64) -> Result<Scaled<S>, SciOxideError> {
    // exp(log) = 2^e * exp(log - e ln 2) with the mantissa in [1, 2), so only
    // a bounded value is converted to S and the binary exponent is carried
    // separately.
    let exponent = (log / core::f64::consts::LN_2).floor();
    if !exponent.is_finite() || exponent.abs() >= (i64::MAX / 4) as f64 {
        return Err(crate::validation::numerical(
            scioxide_core::NumericalIssue::Overflow,
            "contour exceeds extended exponent range",
        ));
    }
    let mantissa = (log - exponent * core::f64::consts::LN_2).exp();
    Scaled::from_value(S::try_from_f64_lossy(mantissa)?)?.shifted(exponent as i64)
}
