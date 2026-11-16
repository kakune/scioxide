//! Weighted quadrature on finite intervals, Fourier integrals over half-lines and
//! complex contour integrals.
//!
//! Regions where the weight is singular or oscillates strongly use modified
//! Clenshaw–Curtis rules with Chebyshev moments of the weight; other regions use
//! the Gauss–Kronrod rule.
use crate::kernel::number;
mod contour;
mod options;

mod chebyshev;
pub(crate) mod moments;
pub(crate) mod rule;
pub(crate) mod solve;
mod tail;

use alloc::sync::Arc;
use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Budget, Interval, RealScalar, SciOxideError,
    SolveScope,
};

use crate::SubdivisionOptions;
use crate::report::IntegrationReport;
use crate::workspace::{Family, IntegrationRequirements};

pub use contour::ContourPlan;
pub use options::{FourierTailOptions, WeightedQuadratureOptions};

pub use solve::WeightedQuadratureWorkspace;
pub use tail::{FourierDomain, FourierTailPlan, FourierTailWorkspace};

/// Selects the Fourier weight `cos(ωx)` or `sin(ωx)`.
///
/// The integrand callback returns only the amplitude `f(x)`; the plan applies
/// the weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourierWeight {
    /// The weight `cos(ωx)`.
    Cosine,
    /// The weight `sin(ωx)`.
    Sine,
}

/// Algebraic–logarithmic endpoint weight on an interval `[a, b]`.
///
/// ```text
/// w(x) = (x − a)^α (b − x)^β [ln(x − a)] [ln(b − x)],   α > −1, β > −1,
/// ```
///
/// where each bracketed logarithm is present only if selected. The weight is
/// integrable on `[a, b]`. The interval is the one passed to
/// [`WeightedQuadraturePlan::endpoint`].
///
/// # Examples
///
/// ```
/// use scioxide_integrate::EndpointWeight;
///
/// // (x − a)^(−1/2) (b − x)^(1/4) ln(x − a)
/// let weight = EndpointWeight::new(-0.5, 0.25, true, false)?;
/// assert_eq!(weight.exponents(), (-0.5, 0.25));
/// assert_eq!(weight.logarithms(), (true, false));
/// assert!(EndpointWeight::new(-1.0, 0.0, false, false).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndpointWeight<S: RealScalar> {
    alpha: S,
    beta: S,
    left_log: bool,
    right_log: bool,
}
impl<S: RealScalar> EndpointWeight<S> {
    /// Creates the weight with exponents `α = alpha` and `β = beta`.
    ///
    /// `left_log` selects the factor `ln(x − a)` and `right_log` the factor
    /// `ln(b − x)`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite.
    /// - `InvalidInput` if `alpha ≤ −1` or `beta ≤ −1`.
    pub fn new(alpha: S, beta: S, left_log: bool, right_log: bool) -> Result<Self, SciOxideError> {
        scioxide_core::validate_value(&alpha)?;
        scioxide_core::validate_value(&beta)?;
        for (exponent, argument) in [
            (alpha, scioxide_core::argument!("alpha")),
            (beta, scioxide_core::argument!("beta")),
        ] {
            if exponent <= -S::ONE {
                return Err(crate::kernel::invalid(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "endpoint exponents must exceed -1",
                ));
            }
        }
        Ok(Self {
            alpha,
            beta,
            left_log,
            right_log,
        })
    }
    /// Returns the exponents `(α, β)`.
    pub const fn exponents(&self) -> (S, S) {
        (self.alpha, self.beta)
    }
    /// Returns whether the factors `ln(x − a)` and `ln(b − x)` are present.
    pub const fn logarithms(&self) -> (bool, bool) {
        (self.left_log, self.right_log)
    }
}

enum Weight<S: RealScalar> {
    Ordinary,
    Endpoint {
        specification: EndpointWeight<S>,
        left: rule::EndpointSide<S>,
        right: rule::EndpointSide<S>,
    },
    Cauchy(S),
    Fourier {
        omega: S,
        weight: FourierWeight,
        levels: usize,
        table: FourierTable<S>,
    },
}

/// Fourier moments of every initial segment at each strongly oscillating
/// bisection depth, with the nodal weights of their even and odd orders. The
/// normalized phase `|ω| h` halves with depth, so the strong depths of a segment
/// form a prefix: segment `s` owns `depths[starts[s]..starts[s + 1]]`, and every
/// deeper level is weak. Each stored depth holds at most 69 scalars: its phase,
/// 25 pairs of half nodal weights, `m_0` with its error and 16 per-parity sums
/// and extremes.
struct FourierTable<S: RealScalar> {
    depths: Vec<rule::FourierDepth<S>>,
    starts: Vec<usize>,
}

impl<S: RealScalar> FourierTable<S> {
    /// Returns the index of the stored depth of segment `root`, or `None` at weak depths.
    fn index(&self, root: usize, depth: usize) -> Option<usize> {
        let start = self.starts[root].checked_add(depth)?;
        (start < self.starts[root + 1]).then_some(start)
    }
}

/// Prepared geometry of a weighted plan: the interval, the sorted initial
/// segments, the weight with its moment tables, the Chebyshev cosine table and
/// its top coefficient rows. Clones of a plan share it through `Arc`.
pub(crate) struct Weighting<S: RealScalar> {
    interval: Interval<S>,
    segments: Vec<Interval<S>>,
    weight: Weight<S>,
    cosines: [[S; 25]; 25],
    top: [[S; 2]; 25],
    kronrod: crate::GaussKronrod<S>,
}

impl<S: RealScalar> core::fmt::Debug for Weighting<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Weighting")
            .field("interval", &self.interval)
            .field("segments", &self.segments)
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar> Weighting<S> {
    /// Builds unweighted geometry with the `choice` Kronrod rule; the weighted
    /// constructors then replace the weight.
    fn prepare(
        interval: Interval<S>,
        points: &[S],
        choice: crate::GaussKronrodRule,
    ) -> Result<Self, SciOxideError> {
        // Endpoints plus caller breakpoints, sorted and deduplicated so that repeated points
        // or breakpoints on the boundary never produce empty segments.
        let mut sorted = try_with_capacity(
            checked_add(points.len(), 2, scioxide_core::quantity!("breakpoints"))?,
            scioxide_core::quantity!("breakpoints"),
        )?;
        sorted.push(interval.lower());
        for &p in points {
            scioxide_core::validate_value(&p)?;
            if p < interval.lower() || p > interval.upper() {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("points"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "breakpoint outside interval",
                ));
            }
            sorted.push(p);
        }
        sorted.push(interval.upper());
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        sorted.dedup();
        let mut segments = try_with_capacity(
            sorted.len().saturating_sub(1),
            scioxide_core::quantity!("segments"),
        )?;
        for p in sorted.windows(2) {
            segments.push(Interval::new(p[0], p[1])?);
        }
        let kronrod = crate::GaussKronrod::new(choice)?;
        let cosines = moments::cosines()?;
        Ok(Self {
            interval,
            segments,
            weight: Weight::Ordinary,
            top: rule::top_coefficient_rows(&cosines),
            cosines,
            kronrod,
        })
    }

    fn endpoint(
        interval: Interval<S>,
        points: &[S],
        weight: EndpointWeight<S>,
    ) -> Result<Self, SciOxideError> {
        let left = moments::endpoint(weight.alpha)?;
        let right = moments::endpoint(weight.beta)?;
        let mut prepared = Self::prepare(interval, points, crate::GaussKronrodRule::G7K15)?;
        prepared.weight = Weight::Endpoint {
            specification: weight,
            left: rule::EndpointSide::new(left, true, weight.left_log, &prepared.cosines),
            right: rule::EndpointSide::new(right, false, weight.right_log, &prepared.cosines),
        };
        // No regional endpoint rule may include both physical singularities.
        if prepared.segments.len() == 1 {
            let middle = interval.midpoint()?;
            if middle == interval.lower() || middle == interval.upper() {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("interval"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "endpoint interval cannot be split",
                ));
            }
            prepared.segments[0] = Interval::new(interval.lower(), middle)?;
            prepared.segments.try_reserve(1).map_err(|_| {
                SciOxideError::allocation_failure(scioxide_core::quantity!("segments"), 1)
            })?;
            prepared
                .segments
                .push(Interval::new(middle, interval.upper())?);
        }
        Ok(prepared)
    }

    fn principal_value(interval: Interval<S>, pole: S) -> Result<Self, SciOxideError> {
        scioxide_core::validate_value(&pole)?;
        if pole == interval.lower() || pole == interval.upper() {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("pole"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Cauchy pole cannot be an endpoint",
            ));
        }
        let mut prepared = Self::prepare(interval, &[], crate::GaussKronrodRule::G7K15)?;
        prepared.weight = Weight::Cauchy(pole);
        Ok(prepared)
    }

    pub(crate) fn fourier(
        interval: Interval<S>,
        points: &[S],
        omega: S,
        weight: FourierWeight,
        moment_levels: Budget,
    ) -> Result<Self, SciOxideError> {
        scioxide_core::validate_value(&omega)?;
        let mut prepared = Self::prepare(interval, points, crate::GaussKronrodRule::G7K15)?;
        let levels = moment_levels.get();
        // Moments depend only on omega * h, and h halves with each bisection, so they are
        // precomputed per segment for every depth up to `levels`. Following QUADPACK QAWO,
        // moments are used only when omega * h > 2; below that the oscillation is resolved
        // by the ordinary Kronrod rule. Only that strong prefix of depths is stored: a
        // finite phase above two halves to two within the scalar exponent range.
        let strong = |segment: &Interval<S>| -> Result<(S, usize), SciOxideError> {
            let phase = crate::fixed::interval_map(*segment)?
                .scaled_weight(omega.abs())?
                .round()?;
            let (mut depth, mut current) = (0, phase);
            while depth < levels && current > S::TWO {
                depth += 1;
                current *= S::HALF;
            }
            Ok((phase, depth))
        };
        let mut starts = try_with_capacity(
            checked_add(
                prepared.segments.len(),
                1,
                scioxide_core::quantity!("moment_table"),
            )?,
            scioxide_core::quantity!("moment_table"),
        )?;
        starts.push(0);
        for segment in &prepared.segments {
            let (_, depth) = strong(segment)?;
            let end = checked_add(
                starts[starts.len() - 1],
                depth,
                scioxide_core::quantity!("moment_table"),
            )?;
            starts.push(end);
        }
        let mut depths = try_with_capacity(
            starts[starts.len() - 1],
            scioxide_core::quantity!("moment_table"),
        )?;
        for segment in &prepared.segments {
            let (mut phase, depth) = strong(segment)?;
            for _ in 0..depth {
                depths.push(rule::FourierDepth::new(phase, &prepared.cosines)?);
                phase *= S::HALF;
            }
        }
        prepared.weight = Weight::Fourier {
            omega,
            weight,
            levels,
            table: FourierTable { depths, starts },
        };
        Ok(prepared)
    }

    /// Returns whether the weight is identically zero (a sine weight with `ω = 0`).
    fn vanishes(&self) -> bool {
        matches!(
            self.weight,
            Weight::Fourier { omega, weight: FourierWeight::Sine, .. } if omega == S::ZERO
        )
    }

    /// Returns the region capacity: the initial segments plus the subdivision limit.
    fn regions(&self, options: &SubdivisionOptions<S>) -> Result<usize, SciOxideError> {
        let regions = checked_add(
            self.segments.len(),
            options.max_subdivisions().get(),
            scioxide_core::quantity!("weighted_regions"),
        )?;
        // The two staging slots must stay representable as well.
        checked_add(regions, 2, scioxide_core::quantity!("regional_slots"))?;
        Ok(regions)
    }
}

/// Adaptive quadrature of `∫ₐᵇ f(x) w(x) dx` for a known weight `w` on a
/// finite interval.
///
/// The constructor selects the weight:
///
/// | Constructor | Weight `w(x)` |
/// |---|---|
/// | [`breakpoints`](Self::breakpoints) | `1`, with breakpoints of `f` |
/// | [`endpoint`](Self::endpoint) | `(x − a)^α (b − x)^β`, optionally with `ln(x − a)`, `ln(b − x)` |
/// | [`principal_value`](Self::principal_value) | `1/(x − c)`, as a Cauchy principal value |
/// | [`fourier`](Self::fourier) | `cos(ωx)` or `sin(ωx)` |
///
/// The callback returns only the amplitude `f(x)`, as a real, complex or vector
/// value (see [Integrands](crate#integrands)). The plan bisects the region with
/// the largest error estimate until the tolerance of its
/// [`WeightedQuadratureOptions`] is met or a limit is reached.
///
/// Where the weight is singular or oscillates strongly, the plan integrates the
/// weight analytically by modified Clenshaw–Curtis quadrature. It interpolates
/// `f` at the 25 Chebyshev points of the region and combines the Chebyshev
/// coefficients with the modified Chebyshev moments of the weight: the
/// integrals of `Tₖ(t) w(x(t))` over `t ∈ [−1, 1]`, where `x(t)` maps `[−1, 1]`
/// onto the region and `Tₖ` is the Chebyshev polynomial of degree `k`. The
/// difference from the nested 13-point rule estimates the error. On
/// all other regions the plan applies the 15-point Gauss–Kronrod rule to
/// `f(x) w(x)`. A region costs 25 or 15 evaluations of `f`.
///
/// The reported error is the larger of the discretization estimate and a
/// rounding floor, plus the propagated error of the moments; see
/// [Error estimates](crate#error-estimates). Clones share the prepared moments.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{EndpointWeight, WeightedQuadratureOptions, WeightedQuadraturePlan};
///
/// // ∫₀¹ x^(−1/2) dx = 2
/// let weight = EndpointWeight::new(-0.5, 0.0, false, false)?;
/// let options = WeightedQuadratureOptions::default();
/// let plan = WeightedQuadraturePlan::endpoint(Interval::new(0.0, 1.0)?, &[], weight, options)?;
/// let (value, report) = plan.integrate(|_: f64| Ok(1.0))?;
/// assert!(report.is_success());
/// assert!((value - 2.0).abs() < 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber and D. K. Kahaner,
///   *QUADPACK: A Subroutine Package for Automatic Integration*, Springer, 1983.
///   <https://doi.org/10.1007/978-3-642-61786-7>
/// - R. Piessens and M. Branders, "The evaluation and application of some
///   modified moments", *BIT* 13, 1973, 443–450.
/// - C. W. Clenshaw and A. R. Curtis, "A method for numerical integration on an
///   automatic computer", *Numer. Math.* 2, 1960, 197–205.
#[derive(Debug, Clone)]
#[must_use]
pub struct WeightedQuadraturePlan<S: RealScalar> {
    weighting: Arc<Weighting<S>>,
    options: WeightedQuadratureOptions<S>,
    extrapolation: bool,
}

impl<S: RealScalar> WeightedQuadraturePlan<S> {
    fn with(
        weighting: Weighting<S>,
        options: WeightedQuadratureOptions<S>,
    ) -> Result<Self, SciOxideError> {
        weighting.regions(options.subdivision_options())?;
        Ok(Self {
            weighting: Arc::new(weighting),
            options,
            extrapolation: false,
        })
    }

    /// Prepares `∫ₐᵇ f(x) dx` with breakpoints where `f` is singular,
    /// discontinuous or not smooth.
    ///
    /// The breakpoints split `[a, b]` into the initial segments. They may come in
    /// any order; duplicates and points equal to `a` or `b` are ignored. The
    /// Gauss–Kronrod nodes are interior, so `f` is never evaluated at a
    /// breakpoint or at `a` or `b`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a breakpoint is NaN or infinite.
    /// - `InvalidInput` if a breakpoint lies outside `[a, b]`.
    /// - `SizeOverflow` or `AllocationFailure` if the region storage cannot be
    ///   sized or reserved.
    pub fn breakpoints(
        interval: Interval<S>,
        points: &[S],
        options: WeightedQuadratureOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::with(
            Weighting::prepare(interval, points, crate::GaussKronrodRule::G7K15)?,
            options,
        )
    }

    /// Prepares `∫ₐᵇ f(x) w(x) dx` for an algebraic–logarithmic
    /// [`EndpointWeight`] `w`.
    ///
    /// `f` must be smooth on `[a, b]`, and the modified Clenshaw–Curtis rule
    /// evaluates it at `a` and `b`. Optional interior breakpoints split `[a, b]`
    /// as in [`Self::breakpoints`]. Without them the plan splits `[a, b]` at its
    /// midpoint, so that no region contains both singular endpoints.
    ///
    /// A region of width `L` that touches a singular endpoint uses moments of
    /// the unit interval scaled by `L^(α+1)` (and shifted by `ln L` for a
    /// logarithmic factor). The scaling is done in extended range, so large
    /// exponents and tiny regions neither overflow nor underflow. The plan
    /// prepares the moments of each endpoint once. A region value that
    /// underflows contributes its rounded value and adds the lost precision to
    /// the rounding floor.
    ///
    /// # Errors
    ///
    /// - The conditions of [`Self::breakpoints`].
    /// - `InvalidInput` if there are no interior breakpoints and `[a, b]` is too
    ///   narrow to split at its midpoint.
    /// - `NonFinite` if a modified moment is not representable.
    ///
    /// Integration fails with `NumericalInstability` if a region value
    /// overflows.
    pub fn endpoint(
        interval: Interval<S>,
        points: &[S],
        weight: EndpointWeight<S>,
        options: WeightedQuadratureOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::with(Weighting::endpoint(interval, points, weight)?, options)
    }

    /// Prepares the Cauchy principal value of `∫ₐᵇ f(x) / (x − c) dx` with
    /// pole `c = pole`.
    ///
    /// ```text
    /// PV ∫ₐᵇ f(x)/(x − c) dx = lim_(ε→0⁺) [ ∫ₐ^(c−ε) + ∫_(c+ε)^b ] f(x)/(x − c) dx
    /// ```
    ///
    /// `f` must be smooth near the pole. A pole outside `[a, b]` gives an
    /// ordinary integral. The plan never divides by `x − c` near the pole: a
    /// region that contains the pole, or whose center lies within 1.1
    /// half-widths of it, integrates `1/(x − c)` analytically with Chebyshev
    /// moments. Other regions apply the Gauss–Kronrod rule to `f(x)/(x − c)`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `pole` is NaN or infinite.
    /// - `InvalidInput` if `pole` equals `a` or `b`.
    /// - `SizeOverflow` or `AllocationFailure` if the region storage cannot be
    ///   sized or reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_integrate::{WeightedQuadratureOptions, WeightedQuadraturePlan};
    ///
    /// // PV ∫₀¹ dx / (x − ¼) = ln((1 − ¼) / ¼) = ln 3
    /// let options = WeightedQuadratureOptions::default();
    /// let plan = WeightedQuadraturePlan::principal_value(Interval::new(0.0, 1.0)?, 0.25, options)?;
    /// let (value, report) = plan.integrate(|_: f64| Ok(1.0))?;
    /// assert!(report.is_success());
    /// assert!((value - 3.0_f64.ln()).abs() < 1e-10);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn principal_value(
        interval: Interval<S>,
        pole: S,
        options: WeightedQuadratureOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::with(Weighting::principal_value(interval, pole)?, options)
    }

    /// Prepares `∫ₐᵇ f(x) cos(ωx) dx` or `∫ₐᵇ f(x) sin(ωx) dx` with `ω = omega`.
    ///
    /// `ω` may be negative or zero; with `ω = 0` the sine weight gives exactly
    /// zero without evaluating `f`. Optional breakpoints split `[a, b]` as in
    /// [`Self::breakpoints`].
    ///
    /// A region of half-width `h` with `|ω| h > 2` uses a modified
    /// Clenshaw–Curtis rule with the Chebyshev moments of `cos(ωx)` or `sin(ωx)`.
    /// Other regions apply the Gauss–Kronrod rule to the product. The plan
    /// prepares moments for each initial segment and each bisection depth below
    /// [`moment_levels`](WeightedQuadratureOptions::moment_levels) at which
    /// `|ω| h > 2`. Each prepared depth stores at most 69 scalars. A region with
    /// `|ω| h > 2` at depth `moment_levels` is not bisected further; if the
    /// tolerance is not met, the call ends with
    /// [`RefinementLimit`](crate::IntegrationTermination::RefinementLimit).
    ///
    /// For large `|ω| h`, the difference of the nested rules depends mostly on
    /// `f` near the region ends and misses jumps, kinks or cusps inside the
    /// region. Each moment region therefore also computes the bound
    /// `2 (|c₂₃| + |c₂₄|) L`, where `c₂₃` and `c₂₄` are the two highest Chebyshev
    /// coefficients of the interpolant of `f` and `L` is the region width. Its
    /// discretization estimate is the larger of the two.
    ///
    /// # Errors
    ///
    /// - The conditions of [`Self::breakpoints`].
    /// - `NonFinite` if `omega` is NaN or infinite, or if `ω` times the width
    ///   of an initial segment, or a moment, is not representable.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Budget, Interval};
    /// use scioxide_integrate::{FourierWeight, WeightedQuadratureOptions, WeightedQuadraturePlan};
    ///
    /// let levels = Budget::try_from(16)?;
    /// let options = WeightedQuadratureOptions::default().with_moment_levels(levels);
    /// let (interval, cosine) = (Interval::new(0.0, 1.0)?, FourierWeight::Cosine);
    /// let plan = WeightedQuadraturePlan::fourier(interval, &[], 1000.0, cosine, options)?;
    /// // Amplitudes eˣ and 1; the second integral is ∫₀¹ cos(1000x) dx = sin(1000)/1000.
    /// let (value, report) = plan.integrate(|x: f64| Ok([x.exp(), 1.0]))?;
    /// assert!(report.is_success());
    /// assert!((value[1] - 1000.0_f64.sin() / 1000.0).abs() < 1e-10);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// R. Piessens and M. Branders, "Computation of oscillating integrals",
    /// *J. Comput. Appl. Math.* 1, 1975, 153–164.
    pub fn fourier(
        interval: Interval<S>,
        points: &[S],
        omega: S,
        weight: FourierWeight,
        options: WeightedQuadratureOptions<S>,
    ) -> Result<Self, SciOxideError> {
        Self::with(
            Weighting::fourier(interval, points, omega, weight, options.moment_levels())?,
            options,
        )
    }

    /// Enables or disables extrapolation of the bisection sequence by Wynn's
    /// ε-algorithm.
    ///
    /// Extrapolation is off by default. It speeds up convergence for endpoint
    /// singularities of `f` in a [`Self::breakpoints`] plan. Each real component
    /// is extrapolated separately, and the tolerance norm combines their error
    /// estimates.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `enabled` is `true` and the plan has an endpoint,
    ///   Cauchy or Fourier weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_integrate::{WeightedQuadratureOptions, WeightedQuadraturePlan};
    ///
    /// // ∫₀¹ |x − ½|^(−1/2) dx = 2√2
    /// let options = WeightedQuadratureOptions::default();
    /// let plan = WeightedQuadraturePlan::breakpoints(Interval::new(0.0, 1.0)?, &[0.5], options)?
    ///     .with_extrapolation(true)?;
    /// let (value, report) = plan.integrate(|x: f64| Ok((x - 0.5).abs().sqrt().recip()))?;
    /// assert!(report.is_success() && (value - 8.0_f64.sqrt()).abs() < 1e-7);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// P. Wynn, "On a device for computing the eₘ(Sₙ) transformation",
    /// *Math. Tables Aids Comput.* 10, 1956, 91–96.
    pub fn with_extrapolation(mut self, enabled: bool) -> Result<Self, SciOxideError> {
        if enabled && !matches!(self.weighting.weight, Weight::Ordinary) {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("extrapolation"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "bisection extrapolation requires an ordinary breakpoint plan",
            ));
        }
        self.extrapolation = enabled;
        Ok(self)
    }

    /// Returns whether extrapolation is enabled.
    pub const fn extrapolation_enabled(&self) -> bool {
        self.extrapolation
    }

    /// Returns the options.
    pub const fn options(&self) -> &WeightedQuadratureOptions<S> {
        &self.options
    }

    /// Returns the identifier of the method, as reported in diagnostics.
    pub fn algorithm_id(&self) -> AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "WeightedQuadraturePlan", 1)
    }

    /// Returns the integration interval `[a, b]`.
    pub fn interval(&self) -> Interval<S> {
        self.weighting.interval
    }

    /// Returns the nonempty initial segments in ascending order.
    pub fn segments(&self) -> &[Interval<S>] {
        &self.weighting.segments
    }

    /// Returns the storage that a workspace needs for `components` real value
    /// components. A complex value has two components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size is not representable.
    pub fn workspace_requirements(
        &self,
        components: usize,
    ) -> Result<IntegrationRequirements, SciOxideError> {
        Ok(IntegrationRequirements::new(
            Family::Weighted,
            components,
            [
                self.weighting.regions(self.options.subdivision_options())?,
                0,
            ],
        ))
    }

    /// Creates a workspace for `components` real value components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized or
    ///   reserved.
    pub fn workspace(
        &self,
        components: usize,
    ) -> Result<WeightedQuadratureWorkspace<S>, SciOxideError> {
        WeightedQuadratureWorkspace::allocate(&self.workspace_requirements(components)?)
    }

    /// Integrates the amplitude `f` against the plan's weight and writes the
    /// result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// On an empty interval the result is exactly zero, the termination is
    /// [`DegenerateDomain`](crate::IntegrationTermination::DegenerateDomain) and
    /// `f` is not called. A value with no components gives
    /// [`FixedRuleCompleted`](crate::IntegrationTermination::FixedRuleCompleted).
    /// The samples of one region are evaluated under the execution policy of the
    /// options; the result and the evaluation count do not depend on it. See
    /// [Termination](crate#termination) for results that end before the tolerance
    /// is met.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` or a per-component tolerance does not have
    ///   the workspace's number of components, or if the workspace holds fewer
    ///   regions than the plan needs.
    /// - `NumericalInstability` if a region value of an endpoint weight
    ///   overflows.
    /// - The callback, non-finite and stop errors listed under
    ///   [Errors](crate#errors).
    pub fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        out: &mut O,
        workspace: &mut WeightedQuadratureWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        let components = workspace.components();
        let policy = self.options.subdivision_options().execution_policy();
        solve::integrate_into(
            &self.weighting,
            self.options.subdivision_options(),
            self.extrapolation,
            self.algorithm_id(),
            solve::epoch(|points: &[S], rows: &mut [S], context| {
                crate::evaluation::sample_1d(f, points, components, rows, policy, context)
            }),
            out,
            workspace,
            scope.into(),
        )
    }

    /// Integrates a real, complex or fixed-size vector amplitude `f` and returns
    /// the value with its report.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Uses a new workspace and
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::integrate_into`].
    pub fn integrate<F, V>(&self, f: F) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let f = &f;
        crate::execution::owned(
            V::default(),
            |components| self.workspace(components),
            |out, workspace, cx| self.integrate_into(f, out, workspace, cx),
        )
    }
}

/// Integrates a finite interval split at known discontinuities or singularities.
///
/// Uses adaptive Gauss–Kronrod on the pieces, without extrapolation. Unsorted
/// points are accepted; duplicates and endpoints are ignored.
/// Defaults are √ε absolute and relative tolerances and 50 subdivisions.
/// Cost and error estimation follow [`WeightedQuadraturePlan`].
///
/// # Errors
///
/// Rejects invalid bounds or points outside the interval, propagates callback and allocation errors,
/// and returns the [one-shot termination error](crate#termination) on an unsuccessful stop.
///
/// ```
/// let result = scioxide_integrate::integrate_piecewise(
///     |x: f64| Ok((x - 0.5).abs()), 0.0, 1.0, &[0.5],
/// )?;
/// assert!((result.value() - 0.25).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. Piessens et al., QUADPACK (1983), weighted adaptive quadrature.
pub fn integrate_piecewise<S, F, V>(
    f: F,
    lower: S,
    upper: S,
    points: &[S],
) -> Result<crate::Integral<V>, SciOxideError>
where
    S: scioxide_core::RealScalar,
    F: Fn(S) -> Result<V, SciOxideError> + Sync,
    V: scioxide_core::AffineValueMut<Scale = S> + Copy + Default,
{
    let interval = scioxide_core::Interval::new(lower, upper)?;
    let plan = WeightedQuadraturePlan::breakpoints(
        interval,
        points,
        WeightedQuadratureOptions::default(),
    )?;
    let mut value = V::default();
    let mut workspace = plan.workspace(value.dimension()?)?;
    let report = plan.integrate_into(
        &f,
        &mut value,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    crate::workspace::require_success(
        report.termination(),
        plan.algorithm_id(),
        workspace
            .last_attempt()
            .map_or_else(scioxide_core::SolveCounts::new, |attempt| attempt.counts()),
    )?;
    Ok(crate::Integral::from_report(value, &report))
}

/// Integrates f(x) times sin(ωx) or cos(ωx) over finite bounds.
///
/// Uses adaptive weighted quadrature with 16 moment levels.
/// Defaults are √ε absolute and relative tolerances and 50 subdivisions.
/// Cost and error estimation follow [`WeightedQuadraturePlan`].
///
/// # Errors
///
/// Rejects invalid bounds or frequency, propagates callback and allocation errors,
/// and returns the [one-shot termination error](crate#termination) on an unsuccessful stop.
///
/// ```
/// use scioxide_integrate::{integrate_fourier, FourierWeight};
/// let result = integrate_fourier(|_: f64| Ok(1.0), 0.0, 1.0, 1.0, FourierWeight::Cosine)?;
/// assert!((result.value() - 1.0_f64.sin()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. Piessens et al., QUADPACK (1983), weighted adaptive quadrature.
pub fn integrate_fourier<S, F, V>(
    f: F,
    lower: S,
    upper: S,
    omega: S,
    weight: FourierWeight,
) -> Result<crate::Integral<V>, SciOxideError>
where
    S: scioxide_core::RealScalar,
    F: Fn(S) -> Result<V, SciOxideError> + Sync,
    V: scioxide_core::AffineValueMut<Scale = S> + Copy + Default,
{
    let interval = scioxide_core::Interval::new(lower, upper)?;
    let plan = WeightedQuadraturePlan::fourier(
        interval,
        &[],
        omega,
        weight,
        WeightedQuadratureOptions::default(),
    )?;
    let mut value = V::default();
    let mut workspace = plan.workspace(value.dimension()?)?;
    let report = plan.integrate_into(
        &f,
        &mut value,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    crate::workspace::require_success(
        report.termination(),
        plan.algorithm_id(),
        workspace
            .last_attempt()
            .map_or_else(scioxide_core::SolveCounts::new, |attempt| attempt.counts()),
    )?;
    Ok(crate::Integral::from_report(value, &report))
}
