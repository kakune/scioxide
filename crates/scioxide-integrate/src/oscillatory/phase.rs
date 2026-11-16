//! Phase geometry of oscillatory integrals: regions with bounds on `g′`, and
//! phase functions with optional inverses.
use super::number;
use alloc::vec::Vec;
use scioxide_core::{Interval, RealScalar, SciOxideError};

/// A real phase function `g` that returns `g(x)` and `g′(x)` together.
///
/// Any closure `Fn(S) -> Result<(S, S), SciOxideError>` is a phase without an
/// inverse. A pair `(phase, inverse)` of such a closure and an
/// [`OscillatoryPhaseInverse`] is a phase with an inverse, which
/// [`OscillatoryMethod::Filon`](crate::OscillatoryMethod::Filon) requires.
///
/// The plan does not differentiate `g` numerically. Each call of
/// [`evaluate`](Self::evaluate) counts as one function evaluation, and the
/// results must be finite and the same for the same `x` during one call of
/// the plan.
pub trait OscillatoryPhase<S: RealScalar> {
    /// Returns `(g(x), g′(x))`, without the frequency `ω`.
    ///
    /// # Errors
    ///
    /// Errors of the phase function are returned unchanged by the plan.
    fn evaluate(&self, x: S) -> Result<(S, S), SciOxideError>;
    /// Returns the branch inverse of `g`, or `None` (the default).
    fn inverse(&self) -> Option<&dyn OscillatoryPhaseInverse<S>> {
        None
    }
}
/// The inverse of a phase `g` on an interval where `g` is strictly monotone.
///
/// Any closure `Fn(Interval<S>, S) -> Result<S, SciOxideError>` implements it.
pub trait OscillatoryPhaseInverse<S: RealScalar> {
    /// Returns the `x` in `interval` with `g(x) = phase`, to working precision.
    ///
    /// The interval selects the branch of the inverse; `g` may be increasing or
    /// decreasing on it. The plan checks that `x` lies in `interval` and that
    /// `g(x)` matches `phase` (see
    /// [`OscillatoryQuadraturePlan::integrate_into`](crate::OscillatoryQuadraturePlan::integrate_into)).
    ///
    /// # Errors
    ///
    /// Errors of the inverse are returned unchanged by the plan.
    fn invert(&self, interval: Interval<S>, phase: S) -> Result<S, SciOxideError>;
}
/// A region of the phase partition with an enclosure `[m, M]` of `g′` on it.
///
/// The enclosure must contain `g′(x)` for every `x` in the region, not only at
/// the nodes. The plan checks it at every node and fails if a node violates
/// it, but it cannot detect a violation between nodes. Every zero of `g′` (a
/// stationary point) must lie in a region created by
/// [`OscillatoryRegion::stationary`].
#[derive(Debug, Clone, Copy)]
pub struct OscillatoryRegion<S: RealScalar> {
    pub(super) interval: Interval<S>,
    pub(super) derivative: Interval<S>,
    pub(super) stationary: bool,
}
impl<S: RealScalar> OscillatoryRegion<S> {
    /// Creates a region on which `g′` lies in `derivative`, which excludes zero.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `derivative` contains zero or `interval` has zero
    ///   width.
    pub fn nonstationary(
        interval: Interval<S>,
        derivative: Interval<S>,
    ) -> Result<Self, SciOxideError> {
        if derivative.lower() <= S::ZERO && derivative.upper() >= S::ZERO {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("derivative"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "nonstationary derivative bound contains zero",
            ));
        }
        Self::new(interval, derivative, false)
    }
    /// Creates a region on which `g′` lies in `derivative` and may vanish.
    ///
    /// The region may contain stationary points of any order, including at its
    /// ends. The plan integrates it with the Gauss–Kronrod rule on panels that
    /// resolve the phase, using the bound `max(|m|, |M|)` on `|g′|`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `interval` has zero width.
    pub fn stationary(
        interval: Interval<S>,
        derivative: Interval<S>,
    ) -> Result<Self, SciOxideError> {
        Self::new(interval, derivative, true)
    }
    fn new(
        interval: Interval<S>,
        derivative: Interval<S>,
        stationary: bool,
    ) -> Result<Self, SciOxideError> {
        if interval.lower() == interval.upper() {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("interval"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "phase regions must have positive width",
            ));
        }
        Ok(Self {
            interval,
            derivative,
            stationary,
        })
    }
    /// Returns the interval of the region.
    pub const fn interval(&self) -> Interval<S> {
        self.interval
    }
    /// Returns the declared enclosure of `g′` on the region.
    pub const fn derivative_bounds(&self) -> Interval<S> {
        self.derivative
    }
    /// Returns whether the region was created by [`Self::stationary`].
    pub const fn is_stationary(&self) -> bool {
        self.stationary
    }
}
/// The interval `[a, b]`, the frequency `ω` and the partition of `[a, b]` into
/// [`OscillatoryRegion`]s of an oscillatory integral.
///
/// The plan stores no phase function; each call of
/// [`OscillatoryQuadraturePlan::integrate_into`](crate::OscillatoryQuadraturePlan::integrate_into)
/// receives it.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{OscillatoryGeometry, OscillatoryRegion};
///
/// // One nonstationary region with g′(x) = 1 on [0, 1].
/// let (interval, one) = (Interval::new(0.0_f64, 1.0)?, Interval::new(1.0, 1.0)?);
/// let regions = [OscillatoryRegion::nonstationary(interval, one)?];
/// let geometry = OscillatoryGeometry::new(interval, 100.0, &regions)?;
/// assert_eq!(geometry.frequency(), 100.0);
/// assert_eq!(geometry.regions().len(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct OscillatoryGeometry<S: RealScalar> {
    pub(super) interval: Interval<S>,
    pub(super) omega: S,
    pub(super) regions: Vec<OscillatoryRegion<S>>,
    pub(super) segments: Vec<Interval<S>>,
}
impl<S: RealScalar> OscillatoryGeometry<S> {
    /// Creates the geometry of `∫ₐᵇ f(x) cos(ω g(x)) dx` or the sine analog,
    /// with `[a, b] = interval` and `ω = omega`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `omega` is NaN or infinite.
    /// - `InvalidInput` if `regions` do not cover `interval` in ascending order
    ///   without gaps or overlaps.
    /// - `AllocationFailure` if the regions cannot be stored.
    pub fn new(
        interval: Interval<S>,
        omega: S,
        regions: &[OscillatoryRegion<S>],
    ) -> Result<Self, SciOxideError> {
        scioxide_core::validate_value(&omega)?;
        let mut end = interval.lower();
        for region in regions {
            if region.interval.lower() != end || region.interval.upper() > interval.upper() {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("regions"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "phase regions must partition the interval in ascending order",
                ));
            }
            end = region.interval.upper();
        }
        if end != interval.upper() || (interval.lower() == interval.upper() && !regions.is_empty())
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("regions"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "phase partition does not cover the interval",
            ));
        }
        let mut stored = scioxide_core::backend::try_with_capacity(
            regions.len(),
            scioxide_core::quantity!("regions"),
        )?;
        stored.extend_from_slice(regions);
        let mut segments = scioxide_core::backend::try_with_capacity(
            regions.len(),
            scioxide_core::quantity!("segments"),
        )?;
        segments.extend(regions.iter().map(|r| r.interval));
        Ok(Self {
            interval,
            omega,
            regions: stored,
            segments,
        })
    }
    /// Returns the interval `[a, b]`.
    pub const fn interval(&self) -> Interval<S> {
        self.interval
    }
    /// Returns the signed frequency `ω`.
    pub const fn frequency(&self) -> S {
        self.omega
    }
    /// Returns the regions of the partition.
    pub fn regions(&self) -> &[OscillatoryRegion<S>] {
        &self.regions
    }
    pub(super) fn ordinary_panels(
        &self,
        interval: Interval<S>,
        root: usize,
        levin: bool,
        ordinary_limit: usize,
    ) -> Result<Option<usize>, SciOxideError> {
        let declaration = self.regions[root];
        let upper = declaration
            .derivative
            .lower()
            .abs()
            .max(declaration.derivative.upper().abs());
        let lower = declaration
            .derivative
            .lower()
            .abs()
            .min(declaration.derivative.upper().abs());
        // Returns `None` when the region is oscillatory enough for the phase-aware rule, or
        // `Some(panels)` for the ordinary-rule fallback. `omega * width * |g'|` bounds the
        // number of radians the phase sweeps over the region; only a strictly non-stationary
        // region with enough guaranteed oscillation (the lower |g'| bound) goes to Levin/Filon.
        let scale = crate::kernel::checked_intermediate(self.omega.abs() * interval.width()?)?;
        // Degree-24 collocation loses rank as the homogeneous exponential becomes
        // polynomially resolved. Ordinary rules handle that regime explicitly.
        let threshold = number::<S>(if levin { 64 } else { 2 })?;
        if !declaration.stationary && scale * lower > threshold {
            return Ok(None);
        }
        // Fallback panels are doubled until each sweeps at most ~2 radians of phase (using
        // the worst-case |g'|), where a 15-point Kronrod rule resolves the oscillating factor
        // as an ordinary smooth integrand. Exceeding the budget is reported to the caller,
        // which stops with a refinement limit rather than silently under-resolving.
        let mut variation = crate::kernel::checked_intermediate(scale * upper)?;
        let mut panels = 1usize;
        while variation > number::<S>(2)? {
            panels = scioxide_core::backend::checked_mul(
                panels,
                2,
                scioxide_core::quantity!("ordinary_panels"),
            )?;
            if panels > ordinary_limit {
                return Ok(Some(panels));
            }
            variation /= number::<S>(2)?;
        }
        Ok(Some(panels))
    }
}

impl<S: RealScalar, F> OscillatoryPhase<S> for F
where
    F: Fn(S) -> Result<(S, S), SciOxideError>,
{
    fn evaluate(&self, x: S) -> Result<(S, S), SciOxideError> {
        self(x)
    }
}
impl<S: RealScalar, G> OscillatoryPhaseInverse<S> for G
where
    G: Fn(Interval<S>, S) -> Result<S, SciOxideError>,
{
    fn invert(&self, interval: Interval<S>, phase: S) -> Result<S, SciOxideError> {
        self(interval, phase)
    }
}
/// A phase closure paired with its branch inverse.
impl<S: RealScalar, F, G> OscillatoryPhase<S> for (F, G)
where
    F: Fn(S) -> Result<(S, S), SciOxideError>,
    G: OscillatoryPhaseInverse<S>,
{
    fn evaluate(&self, x: S) -> Result<(S, S), SciOxideError> {
        self.0(x)
    }
    fn inverse(&self) -> Option<&dyn OscillatoryPhaseInverse<S>> {
        Some(&self.1)
    }
}
