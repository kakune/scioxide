//! Quadrature of `f(x) cos(ω g(x))` and `f(x) sin(ω g(x))` on finite intervals
//! by Levin collocation or Filon–Clenshaw–Curtis rules.
use crate::adaptive::regional::RegionalPolicy;
use crate::kernel::number;
mod levin;
mod options;
mod phase;
pub use options::OscillatoryOptions;
mod rule;
#[cfg(test)]
mod tests;
use crate::weighted::solve::run_regions;
use crate::{FourierWeight, IntegrationErrorComponents, IntegrationReport};
use alloc::vec::Vec;
pub use phase::{
    OscillatoryGeometry, OscillatoryPhase, OscillatoryPhaseInverse, OscillatoryRegion,
};
use scioxide_core::RangeAccumulator;
use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Complex, NumericScalar, RealScalar, SciOxideError,
    SolveScope, check_value_shape,
};

/// Selects the rule that an [`OscillatoryQuadraturePlan`] applies on strongly
/// oscillating regions.
///
/// Both rules use 25 Chebyshev points per region and need no derivatives of
/// the amplitude `f`. The error estimate of a region is the larger of four times
/// the difference from the nested 13-point rule and the bound
/// `2 (|c₂₃| + |c₂₄|) L`. Here `c₂₃` and `c₂₄` are the two highest Chebyshev
/// coefficients of the interpolated function, and `L` is the region width, in
/// `x` for Levin and in the phase `g` for Filon. For large `ω`, the nested
/// difference depends mostly on the region ends; the second bound detects
/// jumps, kinks and cusps inside the region.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{
///     FourierWeight, OscillatoryMethod, OscillatoryGeometry, OscillatoryQuadraturePlan,
///     OscillatoryRegion,
/// };
///
/// let (interval, one) = (Interval::new(0.0, 1.0)?, Interval::new(1.0, 1.0)?);
/// let regions = [OscillatoryRegion::nonstationary(interval, one)?];
/// let geometry = OscillatoryGeometry::new(interval, 100.0, &regions)?;
/// let (method, weight) = (OscillatoryMethod::Filon, FourierWeight::Cosine);
/// let plan = OscillatoryQuadraturePlan::new(method, geometry, weight, Default::default())?;
/// let phase = (|x: f64| Ok((x, 1.0)), |_: Interval<f64>, p: f64| Ok(p));
/// let (value, report) = plan.integrate(&|_: f64| Ok(1.0), &phase)?;
/// // g(x) = x with inverse φ ↦ φ: ∫₀¹ cos(100x) dx = sin(100)/100.
/// assert!(report.is_success() && (value - 100.0_f64.sin() / 100.0).abs() < 1e-9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - D. Levin, "Procedures for computing one- and two-dimensional integrals of
///   functions with rapid irregular oscillations", *Math. Comp.* 38, 1982,
///   531–538.
/// - S. Olver, "Moment-free numerical integration of highly oscillatory
///   functions", *IMA J. Numer. Anal.* 26, 2006, 213–227.
/// - A. Iserles and S. P. Nørsett, "Efficient quadrature of highly oscillatory
///   integrals using derivatives", *Proc. R. Soc. A* 461, 2005, 1383–1399.
/// - V. Domínguez, I. G. Graham and V. P. Smyshlyaev, "Stability and error
///   estimates for Filon–Clenshaw–Curtis rules for highly oscillatory
///   integrals", *IMA J. Numer. Anal.* 31, 2011, 1253–1280.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OscillatoryMethod {
    /// Levin collocation in `x`.
    ///
    /// Solves `p′(x) + i ω g′(x) p(x) = f(x)` at the 25 Chebyshev points of the
    /// region for a polynomial `p` of degree 24. The integral of `f e^(iωg)`
    /// over the region is then the difference of `p e^(iωg)` between its ends.
    /// The rule needs only `g` and `g′`. It solves the collocation system once
    /// per region by column-pivoted Householder QR, in a form that applies to
    /// every component of `f`. A region of width `L` with
    /// `|ω| L m ≤ 64`, where `m` is the lower bound of `|g′|` declared for the
    /// region, uses the ordinary rule instead. If the collocation system is
    /// rank deficient or ill-conditioned, the region reports `L max|f|` as its
    /// error estimate and is bisected; the call does not fail.
    Levin,
    /// Filon–Clenshaw–Curtis quadrature in the phase variable `φ = g(x)`.
    ///
    /// Integrates `f(x(φ)) / |g′(x(φ))|` against `cos(ωφ)` or `sin(ωφ)` by
    /// interpolation at 25 Chebyshev points in `φ`, with Chebyshev moments of
    /// the weight computed in closed form. The rule needs the branch inverse of
    /// `g` ([`OscillatoryPhase::inverse`]). A region with `|ω| L m ≤ 2` uses
    /// the ordinary rule instead.
    Filon,
}

/// Reusable storage for [`OscillatoryQuadraturePlan`].
///
/// Create it with [`OscillatoryQuadraturePlan::workspace`]. It holds
/// `O(c r + 25 c + 25²)` scalars, where `c` is the number of real value
/// components and `r` is the number of declared regions plus the subdivision
/// limit. See [plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct OscillatoryWorkspace<S: RealScalar>
where
    Complex<S>: NumericScalar<Real = S>,
{
    requirements: crate::IntegrationRequirements,
    attempts: crate::workspace::Attempts<IntegrationReport<S>>,
    errors: Vec<IntegrationErrorComponents<S>>,
    published: bool,
    components: usize,
    regional: crate::adaptive::regional::RegionalWorkspace<S>,
    levin: Option<levin::LevinWorkspace<S>>,
    totals: Vec<[RangeAccumulator<S>; 4]>,
}

/// Adaptive quadrature of `∫ₐᵇ f(x) cos(ω g(x)) dx` or `∫ₐᵇ f(x) sin(ω g(x)) dx`
/// for a smooth phase `g`.
///
/// Construction prepares a 15-point Gauss–Kronrod rule and a 25 × 25 cosine table, using O(25²) work and storage.
///
/// The [`OscillatoryGeometry`] fixes `[a, b]`, the frequency `ω` and a
/// partition of `[a, b]` into [`OscillatoryRegion`]s with declared bounds on
/// `g′`. Each call takes the phase as an [`OscillatoryPhase`] that returns
/// `(g(x), g′(x))`, and the amplitude `f` as a real, complex or vector value
/// (see [Integrands](crate#integrands)). The plan bisects the region with the
/// largest error estimate until the tolerance is met or a limit is reached.
///
/// A nonstationary region on which the phase sweeps enough radians uses the
/// [`OscillatoryMethod`]. Such a region costs 25 amplitude and 25 phase
/// evaluations, plus 23 inverse evaluations for Filon. A stationary or weakly
/// oscillating region uses the 15-point Gauss–Kronrod rule on `2ᵏ` equal
/// panels, with `k` chosen from the declared bound on `|g′|` so that each panel
/// sweeps at most 2 radians. A region that needs more panels than
/// [`OscillatoryOptions::ordinary_panels`] is not refined; if it is a declared
/// region, the call fails as described under [Termination](crate#termination).
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{
///     FourierWeight, OscillatoryMethod, OscillatoryGeometry, OscillatoryQuadraturePlan,
///     OscillatoryRegion,
/// };
///
/// // ∫₀¹ cos(100 g(x)) dx with g(x) = x and g′(x) = 1 is sin(100)/100.
/// let (interval, one) = (Interval::new(0.0, 1.0)?, Interval::new(1.0, 1.0)?);
/// let regions = [OscillatoryRegion::nonstationary(interval, one)?];
/// let geometry = OscillatoryGeometry::new(interval, 100.0, &regions)?;
/// let (method, weight) = (OscillatoryMethod::Levin, FourierWeight::Cosine);
/// let plan = OscillatoryQuadraturePlan::new(method, geometry, weight, Default::default())?;
/// let (value, report) = plan.integrate(&|_: f64| Ok(1.0), &|x: f64| Ok((x, 1.0)))?;
/// assert!(report.is_success());
/// assert!((value - 100.0_f64.sin() / 100.0).abs() < 1e-9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - D. Levin, "Fast integration of rapidly oscillatory functions",
///   *J. Comput. Appl. Math.* 67, 1996, 95–101.
/// - L. N. G. Filon, "On a quadrature formula for trigonometric integrals",
///   *Proc. Roy. Soc. Edinburgh* 49, 1929, 38–47.
#[derive(Clone)]
pub struct OscillatoryQuadraturePlan<S: RealScalar> {
    method: OscillatoryMethod,
    phase: OscillatoryGeometry<S>,
    weight: FourierWeight,
    options: OscillatoryOptions<S>,
    kronrod: crate::GaussKronrod<S>,
    cosines: [[S; 25]; 25],
    /// Rows of the top Chebyshev coefficients of Levin amplitude samples.
    top: [[S; 2]; 25],
}

impl<S: RealScalar> OscillatoryQuadraturePlan<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Prepares the plan for `method`, the phase `geometry` and the
    /// Fourier `weight`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the region storage size is not representable.
    pub fn new(
        method: OscillatoryMethod,
        geometry: OscillatoryGeometry<S>,
        weight: FourierWeight,
        options: OscillatoryOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let cosines = crate::weighted::moments::cosines()?;
        let plan = Self {
            method,
            phase: geometry,
            weight,
            options,
            kronrod: crate::GaussKronrod::new(crate::GaussKronrodRule::G7K15)?,
            top: crate::weighted::rule::top_coefficient_rows(&cosines),
            cosines,
        };
        let _ = plan.workspace_requirements(0)?;
        Ok(plan)
    }
    /// Returns the method.
    pub const fn method(&self) -> OscillatoryMethod {
        self.method
    }
    /// Returns the phase geometry.
    pub const fn geometry(&self) -> &OscillatoryGeometry<S> {
        &self.phase
    }
    /// Returns the Fourier weight.
    pub const fn weight(&self) -> FourierWeight {
        self.weight
    }
    /// Returns the options.
    pub const fn options(&self) -> &OscillatoryOptions<S> {
        &self.options
    }
    /// Returns the identifier of the method, as reported in diagnostics.
    pub fn algorithm_id(&self) -> AlgorithmId {
        match self.method {
            OscillatoryMethod::Levin => {
                scioxide_core::algorithm_id!("integrate", "LevinQuadrature", 1)
            }
            OscillatoryMethod::Filon => {
                scioxide_core::algorithm_id!("integrate", "FilonQuadrature", 1)
            }
        }
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
    ) -> Result<crate::IntegrationRequirements, SciOxideError> {
        let regions = checked_add(
            self.phase.segments.len(),
            self.options.subdivision_options().max_subdivisions().get(),
            scioxide_core::quantity!("oscillatory_regions"),
        )?;
        Ok(crate::IntegrationRequirements::new(
            crate::workspace::Family::Oscillatory,
            components,
            [
                regions,
                usize::from(self.method == OscillatoryMethod::Levin),
            ],
        ))
    }
    /// Creates a workspace for `components` real value components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized or
    ///   reserved.
    pub fn workspace(&self, components: usize) -> Result<OscillatoryWorkspace<S>, SciOxideError> {
        OscillatoryWorkspace::allocate(&self.workspace_requirements(components)?)
    }
    /// Integrates `f(x)` times `cos(ω g(x))` or `sin(ω g(x))` and writes the
    /// result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `phase` returns `(g(x), g′(x))` and, for Filon, the branch inverse. A
    /// sine weight with `ω = 0` gives exactly zero without any callback. The
    /// amplitude samples of one region are evaluated under the execution policy
    /// of the options; the phase is evaluated sequentially. See
    /// [Termination](crate#termination) for results that end before the
    /// tolerance is met.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` or a per-component tolerance does not have
    ///   the workspace's number of components.
    /// - `InvalidInput` if the workspace is smaller than this plan requires or
    ///   belongs to another plan type, or if the method is Filon and `phase` has
    ///   no inverse. These are checked before any callback.
    /// - `InvalidInput` if `g′` at a node lies outside the bounds declared for
    ///   its region, or is zero in a nonstationary region.
    /// - `InvalidInput` for Filon if `g` is not monotone between the ends of a
    ///   region as declared, if the inverse returns a point outside the region,
    ///   or if `|g(x) − φ|` exceeds `64 ε max(1, |g(x)|, |φ|)` for the inverse
    ///   `x` of a phase value `φ`.
    /// - `NonFinite` if the phase or its inverse returns NaN or infinity.
    /// - The callback, non-finite and stop errors listed under
    ///   [Errors](crate#errors).
    pub fn integrate_into<'scope, 'capability, P, F, V, O>(
        &self,
        amplitude: F,
        phase: &P,
        out: &mut O,
        workspace: &mut OscillatoryWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        P: OscillatoryPhase<S> + ?Sized,
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let amplitude = &amplitude;
        workspace.published = false;
        let components = workspace.components;
        let provider = rule::Phase {
            evaluate: &|x| phase.evaluate(x),
            inverse: phase.inverse(),
        };
        let options = self.options.subdivision_options();
        if let Err(error) = check_value_shape(out, components)
            .and_then(|()| crate::workspace::check_tolerance(options.tolerance(), components))
            .and_then(|()| {
                if workspace.check(&self.workspace_requirements(components)?)?
                    == scioxide_core::Capacity::NeedsGrowth
                {
                    return Err(crate::kernel::invalid(
                        scioxide_core::argument!("workspace"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "insufficient oscillatory storage",
                    ));
                }
                if self.method == OscillatoryMethod::Filon && provider.inverse.is_none() {
                    return Err(crate::kernel::invalid(
                        scioxide_core::argument!("phase"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "Filon requires a phase provider with a branch inverse",
                    ));
                }
                Ok(())
            })
        {
            return workspace.attempts.reject(error);
        }
        let mut attempts = core::mem::take(&mut workspace.attempts);
        let mut errors = core::mem::take(&mut workspace.errors);
        let result = attempts.run(
            scope.into(),
            crate::workspace::identity(self.algorithm_id(), components),
            options.work_allowance(),
            |context| {
                let mut kernel = rule::Kernel {
                    plan: self,
                    phase: provider,
                    sampler: &mut |points, rows, context| {
                        crate::evaluation::sample_1d(
                            amplitude,
                            points,
                            components,
                            rows,
                            options.execution_policy(),
                            context,
                        )
                        .map(|()| points.len())
                    },
                    levin: workspace.levin.as_mut(),
                    totals: &mut workspace.totals,
                };
                // Each declared phase region is one root segment, so its derivative bounds
                // and stationarity flag follow every child produced by bisection. A sine
                // weight with zero frequency vanishes identically and is an exact zero.
                let mut allowance = crate::adaptive::regional::OptionAllowance::new(
                    crate::RegionalProgress {
                        termination: crate::IntegrationTermination::ToleranceReached,
                        measured: false,
                        num_evals: 0,
                        subdivisions: 0,
                    },
                    options,
                );
                let tolerance = options.tolerance();
                let scheduled = run_regions(
                    &self.phase.segments,
                    S::ZERO,
                    self.phase.omega == S::ZERO && self.weight == FourierWeight::Sine,
                    &mut kernel,
                    &mut workspace.regional,
                    context,
                    &RegionalPolicy {
                        extrapolation: false,
                        norm: &|errors, scales| {
                            crate::kernel::diagnostic(tolerance.normalized_norm(errors, scales))
                        },
                        refine: true,
                        offsets: None,
                    },
                    &mut allowance,
                )?;
                let progress = crate::weighted::solve::finish_controlled(
                    scheduled,
                    allowance,
                    &workspace.regional,
                    &mut errors,
                    out,
                    context,
                    true,
                )?;
                if !progress.measured {
                    return Err(crate::workspace::unpublished(
                        progress.termination,
                        self.algorithm_id(),
                        context.counts(),
                    ));
                }
                Ok(IntegrationReport::new(
                    progress.termination,
                    progress.num_evals,
                    progress.subdivisions,
                    IntegrationErrorComponents::envelope(errors.iter()),
                ))
            },
        );
        workspace.attempts = attempts;
        workspace.errors = errors;
        workspace.published = result.is_ok();
        result
    }
    /// Integrates a real, complex or fixed-size vector amplitude and returns the
    /// value with its report.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Uses a new workspace and
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::integrate_into`].
    pub fn integrate<P, F, V>(
        &self,
        amplitude: F,
        phase: &P,
    ) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        P: OscillatoryPhase<S> + ?Sized,
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let amplitude = &amplitude;
        crate::execution::owned(
            V::default(),
            |components| self.workspace(components),
            |out, ws, cx| self.integrate_into(amplitude, phase, out, ws, cx),
        )
    }
}

impl<S: RealScalar> OscillatoryWorkspace<S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    fn allocate(requirements: &crate::IntegrationRequirements) -> Result<Self, SciOxideError> {
        let components = requirements.components();
        let [regions, levin] = requirements.units();
        Ok(Self {
            requirements: *requirements,
            attempts: crate::workspace::Attempts::default(),
            errors: try_filled(
                components,
                IntegrationErrorComponents::discretization_only(S::ZERO),
                scioxide_core::quantity!("oscillatory_errors"),
            )?,
            published: false,
            components,
            regional: crate::adaptive::regional::RegionalWorkspace::new(components, regions, 25)?,
            levin: if levin != 0 {
                Some(levin::LevinWorkspace::new()?)
            } else {
                None
            },
            totals: try_filled(
                components,
                core::array::from_fn(|_| RangeAccumulator::new()),
                scioxide_core::quantity!("component_totals"),
            )?,
        })
    }

    /// Returns the error estimate of each component of the last result, or
    /// `None` if the last call failed.
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        self.published.then_some(&self.errors)
    }
}
crate::workspace::workspace_protocol!(OscillatoryWorkspace, IntegrationReport<S>);
