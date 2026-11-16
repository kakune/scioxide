use super::*;
use crate::util::admit;
use scioxide_rootfind::{
    Brent, RootResult, ScalarFn, ScalarFunction, ScalarRootOptions, ScalarRootSolver,
};

mod bessel;
mod order;
mod prototypes;
mod resonators;
mod transforms;

pub use order::{
    IirOrderFamily, IirOrderOptions, IirOrderReport, IirOrderTermination, select_iir_order,
    select_iir_order_scoped, select_iir_order_with_options,
};
pub use prototypes::{design_iir, design_iir_scoped, iir_prototype, iir_prototype_scoped};
pub use resonators::{
    ResonanceKind, design_comb, design_gammatone, design_notch_peak, design_real_gammatone,
    design_real_gammatone_scoped,
};

/// Run one scalar design equation, keeping nested special-function accounting.
fn design_root<S: RealScalar, F: ScalarFunction<S>>(
    function: &F,
    interval: scioxide_core::Interval<S>,
    context: &mut Option<&mut SolveContext<'_, '_>>,
) -> Result<RootResult<S>, SciOxideError> {
    let options = ScalarRootOptions::default()
        .with_step_tolerance(scioxide_core::StepTolerance::new(S::ZERO, S::EPSILON)?);
    let result = if let Some(context) = context.as_deref_mut() {
        Brent.solve_scoped(function, interval, &options, context)?
    } else {
        Brent.solve(function, interval, &options)?
    };
    if result.is_success() {
        Ok(result)
    } else {
        Err(unsuccessful_root(
            result.termination(),
            result.accounting(),
            "IIR normalization root did not converge",
        ))
    }
}

/// Bounded elliptic degree equation, including its analytic endpoint limits.
struct EllipticResidual<S: RealScalar> {
    fraction: S,
}
impl<S: RealScalar> EllipticResidual<S> {
    fn value(
        &self,
        m: S,
        mut context: Option<&mut SolveContext<'_, '_>>,
    ) -> Result<S, SciOxideError> {
        let fraction = if m == S::ZERO {
            S::ZERO
        } else if m == S::ONE {
            S::ONE
        } else {
            let complete = k(m, &mut context)?;
            complete / (complete + rf(S::ZERO, m, S::ONE, &mut context)?)
        };
        Ok(fraction - self.fraction)
    }
}
impl<S: RealScalar> ScalarFunction<S> for EllipticResidual<S> {
    fn eval(&self, m: S) -> Result<S, SciOxideError> {
        self.value(m, None)
    }
    fn eval_scoped(&self, m: S, context: &mut SolveContext<'_, '_>) -> Result<S, SciOxideError> {
        self.value(m, Some(context))
    }
}

/// Normalization of the Bessel analog prototype.
///
/// The Bessel prototype of order `n` is `H(s) = θₙ(0)/θₙ(s)`, where `θₙ` is
/// the reverse Bessel polynomial. The normalizations differ by a frequency
/// scale only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BesselNormalization {
    /// Unit group delay at zero frequency.
    Delay,
    /// Unit geometric mean of the pole magnitudes, so that the high-frequency
    /// asymptote `|H(iω)| ≈ ω⁻ⁿ` matches that of the Butterworth prototype.
    Phase,
    /// Half power, `|H(i)|² = 1/2`, at `ω = 1`.
    Magnitude,
}
/// Classical analog lowpass prototype, with its characteristic frequency at
/// `ω = 1`.
///
/// Ripple and attenuation are in decibels. The gain is normalized so that the
/// largest passband magnitude is one. [`IirBand`] maps `ω = 1` to the band
/// edges of a design.
#[derive(Clone, Copy, Debug)]
pub enum IirFamily<S: RealScalar> {
    /// Maximally flat magnitude, `|H(iω)|² = 1/(1 + ω²ⁿ)` for order `n`, with
    /// half power at `ω = 1`.
    Butterworth,
    /// Chebyshev type I: equiripple passband `0 ≤ ω ≤ 1` and monotonic
    /// stopband.
    ChebyshevI {
        /// Positive peak-to-peak passband ripple in dB.
        ripple: S,
    },
    /// Chebyshev type II: monotonic passband and equiripple stopband `ω ≥ 1`.
    ChebyshevII {
        /// Positive minimum stopband attenuation in dB.
        attenuation: S,
    },
    /// Bessel: maximally flat group delay, scaled as the normalization selects.
    Bessel(BesselNormalization),
    /// Elliptic (Cauer): equiripple passband `0 ≤ ω ≤ 1` and equiripple
    /// stopband.
    Elliptic {
        /// Positive peak-to-peak passband ripple in dB.
        ripple: S,
        /// Minimum stopband attenuation in dB, greater than `ripple`.
        attenuation: S,
    },
}
/// Band of a filter designed from a lowpass prototype.
///
/// Each edge is the image of the prototype frequency `ω = 1` (see
/// [`IirFamily`]). Edges are positive and band edges increasing. Their units
/// follow the [crate conventions](crate#conventions): radians per time unit
/// for analog designs, multiples of Nyquist below one for digital designs.
/// [`PoleZeroGain::transform`] lists the frequency transformations.
#[derive(Clone, Copy, Debug)]
pub enum IirBand<S: RealScalar> {
    /// Lowpass with cutoff `ω_c`.
    Lowpass(S),
    /// Highpass with cutoff `ω_c`.
    Highpass(S),
    /// Bandpass between the edges `ω₁ < ω₂`.
    Bandpass(S, S),
    /// Bandstop between the edges `ω₁ < ω₂`.
    Bandstop(S, S),
}
/// Filter designed by [`design_iir`], in zero-pole-gain form.
///
/// To filter with it, convert it with [`to_sos`](Self::to_sos). That
/// conversion checks the rounded section poles again, and it can fail for a
/// filter very close to the stability boundary.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{IirBand, IirDomain, IirFamily, design_iir};
/// let band = IirBand::Bandpass(0.2_f64, 0.4);
/// let design = design_iir(2, IirFamily::Butterworth, band, IirDomain::Digital)?;
/// // The bandpass transformation doubles the prototype order.
/// let orders = (design.prototype_order(), design.zpk().poles().len());
/// assert_eq!(orders, (2, 4));
/// assert!(design.stability_margin() > 0.0);
/// assert_eq!(design.to_sos()?.sections().len(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct IirDesign<S: RealScalar> {
    zpk: PoleZeroGain<S>,
    order: usize,
}
impl<S: RealScalar> IirDesign<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Returns the zeros, poles and gain of the design.
    pub fn zpk(&self) -> &PoleZeroGain<S> {
        &self.zpk
    }
    /// Returns the order of the lowpass prototype.
    ///
    /// Bandpass and bandstop designs have twice this order.
    pub fn prototype_order(&self) -> usize {
        self.order
    }
    /// Returns the smallest distance of a pole to the stability boundary.
    ///
    /// The distance is defined as in [`PoleZeroGain::stability_margin`]. It
    /// measures how close the design is to instability and is not an error
    /// bound.
    pub fn stability_margin(&self) -> S {
        self.zpk.stability_margin()
    }
    /// Converts the design to real second-order sections.
    ///
    /// The zeros and poles are grouped into sections as by
    /// [`PoleZeroGain::to_sos`], and the imaginary parts of the section
    /// coefficients, rounding errors of a real design, are dropped as by
    /// [`SecondOrderSections::to_real`] with the default
    /// [`ResidualTolerance`](scioxide_core::ResidualTolerance): every coefficient
    /// `x` must satisfy `|Im x| ≤ √ε·(1 + |Re x|)`, where `ε` is the machine
    /// epsilon of `S`. Pairing the roots costs `O(n²)` comparisons for `n`
    /// poles. The sections keep the domain of the design.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a pole is not strictly stable, before or after the
    ///   section coefficients are rounded.
    /// - `NumericalInstability` if the gain per section underflows, if a
    ///   section coefficient overflows, or if zeros remain unpaired.
    /// - `DomainError` (`OutsideDomain`) if an imaginary part exceeds the
    ///   tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{IirBand, IirDomain, IirFamily, design_iir};
    /// let band = IirBand::Lowpass(0.2_f64);
    /// let design = design_iir(4, IirFamily::Butterworth, band, IirDomain::Digital)?;
    /// let sos = design.to_sos()?;
    /// assert_eq!(sos.sections().len(), 2);
    /// // The sections keep the response of the design: unit gain at DC.
    /// assert!((sos.response(0.0)? - design.zpk().response(0.0)?).norm() < 1e-14);
    /// assert!((sos.response(0.0)?.re - 1.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn to_sos(&self) -> Result<SecondOrderSections<S>, SciOxideError> {
        self.zpk
            .to_sos()?
            .to_real(scioxide_core::ResidualTolerance::default())
    }
}

fn step(context: &mut Option<&mut SolveContext<'_, '_>>) -> Result<(), SciOxideError> {
    if let Some(c) = context.as_deref_mut() {
        admit(c)?;
    }
    Ok(())
}
fn ripple_power<S: RealScalar>(db: S, argument: Argument) -> Result<S, SciOxideError> {
    positive_parameter(db, argument)?;
    crate::util::positive(
        check((db * number::<S>(core::f64::consts::LN_10 / 10.)?).exp_m1())?,
        argument,
        crate::util::output_overflow,
    )
}
fn rf<S: RealScalar>(
    x: S,
    y: S,
    z: S,
    ctx: &mut Option<&mut SolveContext<'_, '_>>,
) -> Result<S, SciOxideError> {
    if let Some(ctx) = ctx.as_deref_mut() {
        scioxide_special::carlson_rf_scoped(x, y, z, Default::default(), ctx)?.try_value()
    } else {
        scioxide_special::carlson_rf(x, y, z)
    }
}
fn k<S: RealScalar>(m: S, ctx: &mut Option<&mut SolveContext<'_, '_>>) -> Result<S, SciOxideError> {
    rf(S::ZERO, S::ONE - m, S::ONE, ctx)
}
fn jacobi<S: RealScalar>(
    u: S,
    m: S,
    ctx: &mut Option<&mut SolveContext<'_, '_>>,
) -> Result<(S, S, S), SciOxideError> {
    if let Some(ctx) = ctx.as_deref_mut() {
        Ok((
            scioxide_special::jacobi_sn_scoped(u, m, Default::default(), &mut *ctx)?.try_value()?,
            scioxide_special::jacobi_cn_scoped(u, m, Default::default(), &mut *ctx)?.try_value()?,
            scioxide_special::jacobi_dn_scoped(u, m, Default::default(), ctx)?.try_value()?,
        ))
    } else {
        Ok((
            scioxide_special::jacobi_sn(u, m)?,
            scioxide_special::jacobi_cn(u, m)?,
            scioxide_special::jacobi_dn(u, m)?,
        ))
    }
}
/// Map band edges; misordered edges are attributed to `argument`, the caller
/// parameter that supplied `band`.
fn map_band<S: RealScalar>(
    band: IirBand<S>,
    mut map: impl FnMut(S) -> Result<S, SciOxideError>,
    argument: Argument,
) -> Result<IirBand<S>, SciOxideError> {
    Ok(match band {
        IirBand::Lowpass(x) => IirBand::Lowpass(map(x)?),
        IirBand::Highpass(x) => IirBand::Highpass(map(x)?),
        IirBand::Bandpass(a, b) => {
            if a >= b {
                return Err(invalid(argument, "band edges must be increasing"));
            }
            IirBand::Bandpass(map(a)?, map(b)?)
        }
        IirBand::Bandstop(a, b) => {
            if a >= b {
                return Err(invalid(argument, "band edges must be increasing"));
            }
            IirBand::Bandstop(map(a)?, map(b)?)
        }
    })
}
