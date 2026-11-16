use super::*;

/// Filter family for [`select_iir_order`].
///
/// A Bessel filter is specified by its group delay rather than by band edges,
/// so it has no order equation. Choose its order directly.
#[derive(Clone, Copy, Debug)]
pub enum IirOrderFamily {
    /// Maximally flat magnitude.
    Butterworth,
    /// Equiripple passband.
    ChebyshevI,
    /// Equiripple stopband.
    ChebyshevII,
    /// Equiripple passband and stopband.
    Elliptic,
}
/// Outcome of [`select_iir_order`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IirOrderTermination {
    /// The selected order meets the specification.
    Selected,
    /// The required order exceeds [`IirOrderOptions::maximum_order`]. The
    /// report holds the maximum order and a positive order deficit.
    OrderLimit,
}
/// Order, family and design band chosen by [`select_iir_order`], with
/// diagnostics.
///
/// Pass [`order`](IirOrderReport::order), [`family`](IirOrderReport::family)
/// and [`band`](IirOrderReport::band) to [`design_iir`] to design the filter.
/// After [`IirOrderTermination::OrderLimit`] the report holds
/// [`IirOrderOptions::maximum_order`] and its band, and a filter of that order
/// does not meet the specification.
#[derive(Clone, Copy, Debug)]
pub struct IirOrderReport<S: RealScalar> {
    order: usize,
    family: IirFamily<S>,
    band: IirBand<S>,
    termination: IirOrderTermination,
    counts: scioxide_core::SolveCounts,
    deficit: scioxide_core::ResidualReport<S>,
}
impl<S: RealScalar> IirOrderReport<S> {
    /// Returns the selected prototype order.
    ///
    /// Bandpass and bandstop designs have twice this order.
    pub const fn order(self) -> usize {
        self.order
    }
    /// Returns the prototype family of the specification: the
    /// [`IirOrderFamily`] with the ripple and attenuation passed to the
    /// selection, as far as the family uses them.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{IirBand, IirDomain, IirOrderFamily, design_iir, select_iir_order};
    /// let (pass, stop) = (IirBand::Lowpass(0.2_f64), IirBand::Lowpass(0.3));
    /// let family = IirOrderFamily::Elliptic;
    /// let report = select_iir_order(family, pass, stop, 1.0, 60.0, IirDomain::Digital)?;
    /// let (order, band) = (report.order(), report.band());
    /// let design = design_iir(order, report.family(), band, IirDomain::Digital)?;
    /// // The design meets the specification: at most 1 dB of loss at the pass edge.
    /// let gain = design.zpk().response(0.2)?.norm();
    /// assert!(20.0 * gain.log10() >= -1.0 - 1e-9);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn family(self) -> IirFamily<S> {
        self.family
    }
    /// Returns the band edges to design with, in the units of the input edges.
    ///
    /// They are the half-power edges for Butterworth, the edges where the
    /// stopband begins for Chebyshev II, and the passband edges for Chebyshev
    /// I and elliptic filters.
    pub const fn band(self) -> IirBand<S> {
        self.band
    }
    /// Returns the outcome of the selection.
    pub const fn termination(self) -> IirOrderTermination {
        self.termination
    }
    /// Returns whether the selected order meets the specification.
    pub fn is_success(self) -> bool {
        self.termination == IirOrderTermination::Selected
    }
    /// Returns the work counts of the selection.
    ///
    /// Only the elliptic order equation does work, in its complete elliptic
    /// integrals. The closed-form equations record no iterations.
    pub const fn counts(self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the order deficit `max(n* − n, 0)`.
    ///
    /// `n*` is the real solution of the order equation and `n` the selected
    /// prototype order. The deficit is positive only after
    /// [`IirOrderTermination::OrderLimit`].
    pub const fn order_deficit(self) -> scioxide_core::ResidualReport<S> {
        self.deficit
    }
    /// Returns how far the report can be used: `ResidualChecked` after
    /// [`IirOrderTermination::Selected`], `DiagnosticOnly` otherwise.
    pub fn trust(self) -> scioxide_core::PartialResultTrust {
        if self.is_success() {
            scioxide_core::PartialResultTrust::ResidualChecked
        } else {
            scioxide_core::PartialResultTrust::DiagnosticOnly
        }
    }
}
/// Order limit of [`select_iir_order_with_options`] and
/// [`select_iir_order_scoped`].
///
/// The default maximum order is `usize::MAX`, which sets no limit. The limit
/// is only compared with the required order: it changes neither the cost nor
/// an order that it does not cut off.
///
/// ```
/// use scioxide_signal::IirOrderOptions;
/// let options = IirOrderOptions::new().with_maximum_order(8);
/// assert_eq!(options.maximum_order(), 8);
/// assert_eq!(IirOrderOptions::default().maximum_order(), usize::MAX);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IirOrderOptions {
    maximum_order: usize,
}
impl IirOrderOptions {
    /// Creates the default options, without an order limit.
    pub const fn new() -> Self {
        Self {
            maximum_order: usize::MAX,
        }
    }
    /// Sets the largest prototype order to report. It must be positive.
    pub const fn with_maximum_order(mut self, maximum_order: usize) -> Self {
        self.maximum_order = maximum_order;
        self
    }
    /// Returns the largest prototype order to report.
    pub const fn maximum_order(self) -> usize {
        self.maximum_order
    }
}
impl Default for IirOrderOptions {
    fn default() -> Self {
        Self::new()
    }
}
/// Selects the smallest filter order that meets a band-edge specification.
///
/// The specification allows at most `ripple` dB of loss in the passband
/// `pass` and requires at least `attenuation` dB in the stopband `stop`. Both
/// bands are the same variant of [`IirBand`]. The stop edges lie above a
/// lowpass pass edge, below a highpass pass edge, outside the bandpass pass
/// edges or inside the bandstop pass edges. Edges are in the units of
/// [`design_iir`], and digital edges are prewarped as there. For a bandstop
/// band, one pass edge moves inward to balance the two stop edges, which
/// lowers the order.
///
/// The bands give the stop edge `Ω > 1` of the equivalent lowpass prototype
/// with passband edge one. For a bandpass or bandstop band, `Ω` is the smaller
/// of the values of the two stop edges. The order is the smallest positive
/// integer `n` not below the real solution `n*` of
///
/// ```text
/// Butterworth:     n* = log(εₛ²/εₚ²) / (2 log Ω)
/// Chebyshev I, II: n* = acosh(εₛ/εₚ) / acosh(Ω)
/// elliptic:        n* = K(1/Ω²) K′(εₚ²/εₛ²) / (K′(1/Ω²) K(εₚ²/εₛ²))
/// ```
///
/// where `εₚ² = 10^(ripple/10) − 1`, `εₛ² = 10^(attenuation/10) − 1`, `K(m)`
/// is the complete elliptic integral of the first kind with parameter `m`,
/// and `K′(m) = K(1 − m)`. The equations cost `O(1)` operations, and the
/// elliptic integrals are evaluated in a fresh root scope.
///
/// The selection has no order limit. [`select_iir_order_with_options`] reports
/// an order above a configured limit in its result, and
/// [`select_iir_order_scoped`] runs the evaluations inside the caller's solve.
///
/// # Errors
///
/// - `InvalidInput` if `attenuation` does not exceed `ripple`, if ripple,
///   attenuation or an edge is not positive, if a digital edge is not below
///   one, if band edges are not increasing, or if the bands differ in variant
///   or their edges are not ordered as described above.
/// - `NonFinite` if an edge, ripple or attenuation is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if the order equation overflows.
/// - `ConvergenceFailure` (`TruncationLimit`) if the required order does not
///   fit in `usize`.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{IirBand, IirDomain, IirOrderFamily, select_iir_order};
/// // 1 dB up to 0.2 and 80 dB from 0.3 times Nyquist need a Butterworth order of 22.
/// let (pass, stop) = (IirBand::Lowpass(0.2_f64), IirBand::Lowpass(0.3));
/// let family = IirOrderFamily::Butterworth;
/// let report = select_iir_order(family, pass, stop, 1., 80., IirDomain::Digital)?;
/// assert_eq!(report.order(), 22);
/// assert!(report.order_deficit().raw_norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// T. W. Parks and C. S. Burrus, *Digital Filter Design*, Wiley, 1987.
pub fn select_iir_order<S: RealScalar>(
    family: IirOrderFamily,
    pass: IirBand<S>,
    stop: IirBand<S>,
    ripple: S,
    attenuation: S,
    domain: IirDomain,
) -> Result<IirOrderReport<S>, SciOxideError> {
    select_iir_order_with_options(
        family,
        pass,
        stop,
        ripple,
        attenuation,
        domain,
        IirOrderOptions::default(),
    )
    .and_then(require_success)
}
/// Selects a filter order as [`select_iir_order`] does, up to the maximum
/// order of `options`.
///
/// An order above the maximum is not an error: the report then has
/// [`IirOrderTermination::OrderLimit`] and holds the maximum order. Check
/// [`is_success`](IirOrderReport::is_success) before designing. The cost does
/// not depend on the maximum order.
///
/// # Errors
///
/// - The errors of [`select_iir_order`], except that an order that does not
///   fit in `usize` is reported in the result.
/// - `InvalidInput` if the maximum order of `options` is zero.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{
///     IirBand, IirDomain, IirOrderFamily, IirOrderOptions, IirOrderTermination,
///     select_iir_order_with_options,
/// };
/// // 1 dB up to 0.2 and 80 dB from 0.3 times Nyquist need more than order 2.
/// let (pass, stop) = (IirBand::Lowpass(0.2_f64), IirBand::Lowpass(0.3));
/// let family = IirOrderFamily::Butterworth;
/// let options = IirOrderOptions::new().with_maximum_order(2);
/// let report =
///     select_iir_order_with_options(family, pass, stop, 1., 80., IirDomain::Digital, options)?;
/// assert_eq!(report.termination(), IirOrderTermination::OrderLimit);
/// assert_eq!(report.order(), 2);
/// assert!(report.order_deficit().raw_norm() > 0.);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn select_iir_order_with_options<S: RealScalar>(
    family: IirOrderFamily,
    pass: IirBand<S>,
    stop: IirBand<S>,
    ripple: S,
    attenuation: S,
    domain: IirDomain,
    options: IirOrderOptions,
) -> Result<IirOrderReport<S>, SciOxideError> {
    select_iir_order_scoped(
        family,
        pass,
        stop,
        ripple,
        attenuation,
        domain,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Selects a filter order as [`select_iir_order_with_options`] does, as one
/// epoch of `scope`.
///
/// The elliptic integrals count toward the limits of `scope` and observe its
/// cancellation and deadline, as described in
/// [scoped designs](crate#scoped-designs). An order above the maximum order
/// of `options` is reported in the result as by
/// [`select_iir_order_with_options`].
///
/// # Errors
///
/// - The errors of [`select_iir_order_with_options`].
/// - `ConvergenceFailure` if the evaluations exhaust the limits of `scope`.
/// - `ExecutionStopped` on cancellation or a passed deadline.
#[allow(clippy::too_many_arguments)]
pub fn select_iir_order_scoped<'scope, 'capability, S: RealScalar>(
    family: IirOrderFamily,
    pass: IirBand<S>,
    stop: IirBand<S>,
    ripple: S,
    attenuation: S,
    domain: IirDomain,
    options: IirOrderOptions,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<IirOrderReport<S>, SciOxideError> {
    let maximum_order = options.maximum_order();
    if attenuation <= ripple {
        return Err(inconsistent(
            scioxide_core::argument!("attenuation"),
            "attenuation must exceed the ripple",
        ));
    }
    if maximum_order == 0 {
        return Err(invalid(
            scioxide_core::argument!("options.maximum_order"),
            "maximum order must be positive",
        ));
    }
    run_design(
        scope,
        scioxide_core::algorithm_id!("signal", "iir_order", 1),
        |context| {
            select_order_candidate(
                family,
                pass,
                stop,
                ripple,
                attenuation,
                domain,
                maximum_order,
                context,
            )
        },
    )
}
/// Returns a selected report and maps [`IirOrderTermination::OrderLimit`] to
/// the `ConvergenceFailure` of [`select_iir_order`].
fn require_success<S: RealScalar>(
    report: IirOrderReport<S>,
) -> Result<IirOrderReport<S>, SciOxideError> {
    match report.termination() {
        IirOrderTermination::Selected => Ok(report),
        IirOrderTermination::OrderLimit => Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::TruncationLimit,
            scioxide_core::ConvergenceDiagnostics::new().with_counts(report.counts()),
            "required order exceeds the maximum order",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn select_order_candidate<S: RealScalar>(
    family: IirOrderFamily,
    pass: IirBand<S>,
    stop: IirBand<S>,
    ripple: S,
    attenuation: S,
    domain: IirDomain,
    maximum_order: usize,
    context: &mut SolveContext<'_, '_>,
) -> Result<IirOrderReport<S>, SciOxideError> {
    let mut ctx = Some(context);
    let ep = ripple_power(ripple, scioxide_core::argument!("ripple"))?;
    let es = ripple_power(attenuation, scioxide_core::argument!("attenuation"))?;
    let warp = |x: S, argument: Argument| -> Result<S, SciOxideError> {
        positive_parameter(x, argument)?;
        if domain == IirDomain::Digital {
            if x >= S::ONE {
                return Err(invalid(
                    argument,
                    "digital band edges must lie below Nyquist",
                ));
            }
            check((S::PI * x * S::HALF).tan())
        } else {
            Ok(x)
        }
    };
    let pass_argument = scioxide_core::argument!("pass");
    let stop_argument = scioxide_core::argument!("stop");
    let mut p = map_band(pass, |x| warp(x, pass_argument), pass_argument)?;
    let s = map_band(stop, |x| warp(x, stop_argument), stop_argument)?;
    // Balance the two limiting stop edges by moving only unconstrained pass edges inward.
    // This maximizes the worst transformed separation and hence minimizes prototype order.
    if let (IirBand::Bandstop(a, b), IirBand::Bandstop(c, d)) = (p, s)
        && a < c
        && b > d
    {
        let lower = a.max(c * d / b);
        p = IirBand::Bandstop(lower, c * d / lower);
    }
    let ratio = match (p, s) {
        (IirBand::Lowpass(p), IirBand::Lowpass(s)) if s > p => s / p,
        (IirBand::Highpass(p), IirBand::Highpass(s)) if s < p => p / s,
        (IirBand::Bandpass(a, b), IirBand::Bandpass(c, d)) if c < a && d > b => ((c * c - a * b)
            / (c * (b - a)))
            .abs()
            .min(((d * d - a * b) / (d * (b - a))).abs()),
        (IirBand::Bandstop(a, b), IirBand::Bandstop(c, d)) if a < c && b > d => ((c * (b - a))
            / (c * c - a * b))
            .abs()
            .min(((d * (b - a)) / (d * d - a * b)).abs()),
        _ => {
            return Err(inconsistent(
                stop_argument,
                "stop edges must lie outside the pass edges of the same band type",
            ));
        }
    };
    if ratio <= S::ONE {
        return Err(inconsistent(
            stop_argument,
            "transition ratio must exceed one",
        ));
    }
    if let Some(context) = ctx.as_ref() {
        context.execution().check()?;
    }
    let continuous = match family {
        IirOrderFamily::Butterworth => (es / ep).ln() / ((S::TWO) * ratio.ln()),
        IirOrderFamily::ChebyshevI | IirOrderFamily::ChebyshevII => {
            (es / ep).sqrt().acosh() / ratio.acosh()
        }
        IirOrderFamily::Elliptic => {
            let m = S::ONE / (ratio * ratio);
            let m1 = ep / es;
            k(m, &mut ctx)? * rf(S::ZERO, m1, S::ONE, &mut ctx)?
                / (rf(S::ZERO, m, S::ONE, &mut ctx)? * k(m1, &mut ctx)?)
        }
    };
    check(continuous)?;
    // The smallest positive integer order not below the continuous requirement.

    let (order, termination) = match continuous.ceil().max(S::ONE).try_to_usize_exact() {
        Ok(order) if order <= maximum_order => (order, IirOrderTermination::Selected),
        _ => (maximum_order, IirOrderTermination::OrderLimit),
    };
    let n = scalar::<S>(order)?;
    let scale = match family {
        IirOrderFamily::Butterworth => ep.powf(-S::HALF / n),
        IirOrderFamily::ChebyshevII => ((es / ep).sqrt().acosh() / n).cosh(),
        _ => S::ONE,
    };
    let band = match p {
        IirBand::Lowpass(x) => IirBand::Lowpass(x * scale),
        IirBand::Highpass(x) => IirBand::Highpass(x / scale),
        IirBand::Bandpass(a, b) | IirBand::Bandstop(a, b) => {
            let bw = if matches!(p, IirBand::Bandpass(..)) {
                (b - a) * scale
            } else {
                (b - a) / scale
            };
            let w = (bw * bw + (S::TWO) * (S::TWO) * a * b).sqrt();
            let lo = (w - bw) * S::HALF;
            let hi = (w + bw) * S::HALF;
            if matches!(p, IirBand::Bandpass(..)) {
                IirBand::Bandpass(lo, hi)
            } else {
                IirBand::Bandstop(lo, hi)
            }
        }
    };
    Ok(IirOrderReport {
        order,
        family: match family {
            IirOrderFamily::Butterworth => IirFamily::Butterworth,
            IirOrderFamily::ChebyshevI => IirFamily::ChebyshevI { ripple },
            IirOrderFamily::ChebyshevII => IirFamily::ChebyshevII { attenuation },
            IirOrderFamily::Elliptic => IirFamily::Elliptic {
                ripple,
                attenuation,
            },
        },
        termination,
        counts: ctx
            .as_ref()
            .map_or_else(scioxide_core::SolveCounts::new, |c| c.counts()),
        deficit: scioxide_core::ResidualReport::unscaled(
            (continuous - n).max(S::ZERO),
            scioxide_core::NormSelection::Infinity,
        )?,
        band: map_band(
            band,
            |x| {
                if domain == IirDomain::Digital {
                    Ok((S::TWO) * x.atan() / S::PI)
                } else {
                    Ok(x)
                }
            },
            pass_argument,
        )?,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{ConvergenceReason, SciOxideErrorView};

    const PASS: IirBand<f64> = IirBand::Lowpass(0.2);
    const STOP: IirBand<f64> = IirBand::Lowpass(0.3);
    const FAMILIES: [IirOrderFamily; 4] = [
        IirOrderFamily::Butterworth,
        IirOrderFamily::ChebyshevI,
        IirOrderFamily::ChebyshevII,
        IirOrderFamily::Elliptic,
    ];

    fn select(family: IirOrderFamily, options: IirOrderOptions) -> IirOrderReport<f64> {
        select_iir_order_with_options(family, PASS, STOP, 1.0, 80.0, IirDomain::Digital, options)
            .unwrap()
    }

    #[test]
    fn order_limit_is_a_truncation_limit() {
        let report = select(
            IirOrderFamily::Elliptic,
            IirOrderOptions::new().with_maximum_order(2),
        );
        assert_eq!(report.termination(), IirOrderTermination::OrderLimit);
        let error = require_success(report).unwrap_err();
        let failure = match error.view() {
            SciOxideErrorView::ConvergenceFailure {
                reason,
                diagnostics,
                detail,
            } => Some((reason, diagnostics.counts(), detail)),
            _ => None,
        };
        assert_eq!(
            failure,
            Some((
                ConvergenceReason::TruncationLimit,
                report.counts(),
                "required order exceeds the maximum order"
            ))
        );
    }

    #[test]
    fn default_selections_succeed_through_the_one_shot() {
        for family in FAMILIES {
            let configured = select(family, IirOrderOptions::default());
            let report =
                select_iir_order(family, PASS, STOP, 1.0, 80.0, IirDomain::Digital).unwrap();
            assert!(configured.is_success() && report.is_success());
            assert_eq!(report.order(), configured.order());
            assert_eq!(require_success(configured).unwrap().order(), report.order());
        }
    }
}
