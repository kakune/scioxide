//! Fourier integrals over half-lines and the real line, summed cycle by cycle.
use super::{
    FourierTailOptions, FourierWeight, WeightedQuadraturePlan, WeightedQuadratureWorkspace, solve,
};
use crate::adaptive::extrapolation::EpsilonTable;
use crate::adaptive::regional::{EpochAllowance, RegionalPolicy, Scheduled};
use crate::workspace::{
    Attempts, Family, IntegrationRequirements, Outcome, check_tolerance, workspace_protocol,
};
use crate::{
    ErrorEstimate, ErrorReliability, IntegrationErrorComponents, IntegrationReport,
    IntegrationTermination, TailDiagnostics, TailStatus,
};
use alloc::vec::Vec;
use scioxide_core::PromotingCompensatedSum;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, ErrorTolerance, Interval, PartialResultTrust,
    RealScalar, SciOxideError, SolveContext, SolveScope, check_value_shape, write_value,
};
fn empty_errors<S: RealScalar>() -> IntegrationErrorComponents<S> {
    IntegrationErrorComponents::default()
}
use super::solve::Sampler;
struct TailProgress {
    termination: IntegrationTermination,
    measured: bool,
    subdivisions: usize,
}

use crate::kernel::number;
/// Selects the domain of a [`FourierTailPlan`]: a half-line or the real line.
///
/// The integral over each infinite direction must converge on its own.
#[derive(Debug, Clone, Copy)]
pub enum FourierDomain<S: RealScalar> {
    /// The half-line `[lower, +∞)`.
    Upper {
        /// The finite left endpoint.
        lower: S,
    },
    /// The half-line `(−∞, upper]`.
    Lower {
        /// The finite right endpoint.
        upper: S,
    },
    /// The real line, as the sum of the integrals over `(−∞, 0]` and `[0, +∞)`.
    ///
    /// Both must converge; this is not the symmetric principal value
    /// `lim_(R→∞) ∫_(−R)^R`.
    RealLine,
}

/// Two recent values of one cycle sequence (masses or sampled amplitude
/// maxima) with their uncertainties, oldest first.
#[derive(Clone, Copy)]
struct Recent<S: RealScalar> {
    values: [S; 2],
    uncertainties: [S; 2],
}
impl<S: RealScalar> Recent<S> {
    const EMPTY: Self = Self {
        values: [S::ZERO; 2],
        uncertainties: [S::ZERO; 2],
    };

    /// Whether `current` continues a decay to zero; `curved` adds the curvature
    /// test once the preceding decrease also counted.
    fn decays(self, current: S, uncertainty: S, curved: bool, position: S, length: S) -> bool {
        let [older, previous] = self.values;
        let [older_uncertainty, previous_uncertainty] = self.uncertainties;
        let decrease = previous - current;
        decrease > previous_uncertainty + uncertainty
            && length * (current / decrease) <= S::from_f64_ieee(5.0) * position
            && (!curved || {
                let earlier = older - previous;
                let noise = older_uncertainty + S::TWO * previous_uncertainty + uncertainty;
                (earlier - decrease - noise) / earlier
                    <= S::from_f64_ieee(4.5) * (decrease / previous)
            })
    }

    fn push(&mut self, value: S, uncertainty: S) {
        self.values = [self.values[1], value];
        self.uncertainties = [self.uncertainties[1], uncertainty];
    }
}

/// Recent cycle masses and sampled amplitude maxima, with the current run of
/// decreases that indicate decay to zero.
///
/// Decreasing masses alone do not establish convergence: masses decreasing to a
/// nonzero limit `L` make every alternating partial sum oscillate by about `L`,
/// while Wynn's epsilon still returns their Abel mean with a tiny error. A cosine
/// cycle even integrates a constant amplitude part to zero, so only the amplitude
/// itself shows that it fails to vanish. A cycle therefore counts only when both
/// its mass `m_k` (uncertainty: its error estimate) and its independently sampled
/// amplitude maximum (uncertainty: rounding) decrease, each with
/// `d_k = v_(k-1) - v_k`,
///
/// - resolved beyond both values' uncertainties;
/// - consistent with algebraic decay `|x|^(-p)`, `p >= 1/5`, in the distance
///   `x_k` of the cycle center from the origin: `length * v_k <= 5 |x_k| d_k`; and
/// - after a counted decrease, with resolved curvature at most that of a
///   translated power law `(x - x0)^(-p)`, `p >= 2/7`:
///   `v_(k-1) (d_(k-1) - d_k - u) <= (9/2) d_(k-1) d_k`, where `u` bounds the
///   curvature's uncertainty. A power law gives `(1 + p)/p` here.
///
/// For `v_k -> L != 0` both ratios grow without bound (like the cycle index for
/// algebraic approach, exponentially for exponential approach), so such sequences
/// never establish decay; a translated power law, a geometric sequence and
/// `ln(x)/sqrt(x)` pass both tests. Amplitudes decaying more slowly than
/// `|x|^(-2/7)`, and limits below the values' resolution (such as
/// `1e-6 + 1/(1 + x)` within a few cycles), remain heuristically indistinguishable.
#[derive(Clone, Copy)]
struct CycleDecay<S: RealScalar> {
    masses: Recent<S>,
    amplitudes: Recent<S>,
    ratios: [S; 3],
    decreases: usize,
    zeros: usize,
}
impl<S: RealScalar> CycleDecay<S> {
    const EMPTY: Self = Self {
        masses: Recent::EMPTY,
        amplitudes: Recent::EMPTY,
        ratios: [S::ZERO; 3],
        decreases: 0,
        zeros: 0,
    };

    /// Record the magnitude, uncertainty and sampled amplitude maximum of one
    /// cycle whose center lies `position` from the origin.
    fn observe(&mut self, mass: S, amplitude: S, error: S, position: S, length: S, guard: S) {
        // Independently sampled amplitude maxima must shrink as well as cycle
        // magnitudes. A regional roundoff model cannot supply this evidence:
        // centered contractions can shrink for divergent cos(x)*sqrt(x).
        let curved = self.decreases != 0;
        let (error, rounding) = (error + guard * mass, guard * amplitude);
        let shrinking = self.masses.decays(mass, error, curved, position, length)
            && self
                .amplitudes
                .decays(amplitude, rounding, curved, position, length);
        self.decreases = if shrinking {
            (self.decreases + 1).min(3)
        } else {
            0
        };
        self.zeros = if mass == S::ZERO && amplitude == S::ZERO {
            (self.zeros + 1).min(4)
        } else {
            0
        };
        let previous = self.masses.values[1];
        self.ratios.rotate_left(1);
        self.ratios[2] = if shrinking && previous > S::ZERO {
            mass / previous
        } else {
            S::ZERO
        };
        self.masses.push(mass, error);
        self.amplitudes.push(amplitude, rounding);
    }

    fn established(self) -> bool {
        self.decreases == 3 || self.zeros == 4
    }

    fn remainder(self) -> S {
        self.masses.values[1] / (S::ONE - self.ratios.into_iter().fold(S::ZERO, S::max))
    }
}

struct TailStorage<S: RealScalar> {
    finite: WeightedQuadratureWorkspace<S>,
    components: usize,
    cycles: usize,
    extrapolated: EpsilonTable<S>,
    partials: Vec<S>,
    sums: Vec<PromotingCompensatedSum<S>>,
    error_sums: Vec<[S; 3]>,
    decay: Vec<CycleDecay<S>>,
    values: Vec<S>,
    errors: Vec<IntegrationErrorComponents<S>>,
    total_values: Vec<S>,
    total_errors: Vec<IntegrationErrorComponents<S>>,
    normalized: Vec<S>,
    scales: Vec<S>,
}

#[derive(Debug, Clone)]
struct FourierHalf<S: RealScalar> {
    plan: WeightedQuadraturePlan<S>,
    first: WeightedQuadraturePlan<S>,
    reflect: bool,
}

/// Computes `∫ f(x) cos(ωx) dx` or `∫ f(x) sin(ωx) dx` over a half-line or the
/// real line.
///
/// The callback returns only the amplitude `f(x)`, as a real, complex or vector
/// value (see [Integrands](crate#integrands)); `f` must decay to zero. The plan
/// splits each half-line `[x₀, ∞)` into cycles of length
///
/// ```text
/// L = (2 ⌊|ω|⌋ + 1) π / |ω|,
/// ```
///
/// an odd number of half-periods, so that successive cycle integrals alternate
/// in sign. It integrates each cycle with the rule of
/// [`WeightedQuadraturePlan::fourier`], with tolerance share `0.05 · 0.9ᵏ` for
/// cycle `k` (so the shares sum to half of the tolerance), and extrapolates the
/// partial sums with Wynn's ε-algorithm. A half-line `(−∞, x₀]` is reflected
/// to `[−x₀, ∞)`.
///
/// The tail estimate is heuristic. The plan accepts it only after at least six
/// cycles, once the cycle integrals and the sampled maxima of `|f|` both
/// decrease over three consecutive cycles, at least as fast as `|x|^(−1/5)`.
/// When the estimate meets the tolerance, the termination is
/// [`HeuristicToleranceReached`](IntegrationTermination::HeuristicToleranceReached).
/// An amplitude that tends to a nonzero limit, or decays more slowly than about
/// `|x|^(−2/7)`, ends with an unsuccessful termination, such as
/// [`RefinementLimit`](IntegrationTermination::RefinementLimit) after
/// [`max_cycles`](FourierTailOptions::max_cycles) cycles. A nonzero limit
/// smaller than the resolution of the cycle values cannot be detected.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::{FourierDomain, FourierTailOptions, FourierTailPlan, FourierWeight};
///
/// // ∫₀^∞ e^(−x) sin(x) dx = 1/2
/// let domain = FourierDomain::Upper { lower: 0.0 };
/// let options = FourierTailOptions::default();
/// let plan = FourierTailPlan::<f64>::new(domain, 1.0, FourierWeight::Sine, options)?;
/// let (value, report) = plan.integrate(|x: f64| Ok((-x).exp()))?;
/// assert!(report.is_success());
/// assert!((value - 0.5).abs() < 1e-8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber and D. K. Kahaner,
///   *QUADPACK: A Subroutine Package for Automatic Integration*, Springer, 1983.
///   <https://doi.org/10.1007/978-3-642-61786-7>
/// - I. M. Longman, "Note on a method for computing infinite integrals of
///   oscillatory functions", *Proc. Cambridge Philos. Soc.* 52, 1956, 764–768.
/// - P. Wynn, "On a device for computing the eₘ(Sₙ) transformation",
///   *Math. Tables Aids Comput.* 10, 1956, 91–96.
#[derive(Debug, Clone)]
pub struct FourierTailPlan<S: RealScalar> {
    halves: Vec<FourierHalf<S>>,
    length: S,
    cycles: usize,
    weight: FourierWeight,
    options: FourierTailOptions<S>,
}
/// Reusable storage for [`FourierTailPlan`].
///
/// Create it with [`FourierTailPlan::workspace`]. It holds the regions of one
/// cycle, the extrapolation table and the cycle history of each value
/// component; see [plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct FourierTailWorkspace<S: RealScalar> {
    requirements: IntegrationRequirements,
    attempts: Attempts<IntegrationReport<S>>,
    storage: TailStorage<S>,
    published: bool,
}
workspace_protocol!(FourierTailWorkspace, IntegrationReport<S>);
impl<S: RealScalar> FourierTailWorkspace<S> {
    fn allocate(requirements: &IntegrationRequirements) -> Result<Self, SciOxideError> {
        let [regions, cycles] = requirements.units();
        Ok(Self {
            requirements: *requirements,
            attempts: Attempts::default(),
            storage: TailStorage::allocate(requirements.components(), regions, cycles)?,
            published: false,
        })
    }
    /// Returns the error estimate of each component of the last result, or
    /// `None` if the last call failed.
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        self.published.then_some(&self.storage.total_errors)
    }
}
impl<S: RealScalar> TailStorage<S> {
    fn allocate(components: usize, regions: usize, cycles: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            finite: WeightedQuadratureWorkspace::allocate(&IntegrationRequirements::new(
                Family::Weighted,
                components,
                [regions, 0],
            ))?,
            components,
            cycles,
            extrapolated: EpsilonTable::new(components)?,
            partials: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("cycle_partials"),
            )?,
            sums: try_filled(
                components,
                PromotingCompensatedSum::ZERO,
                scioxide_core::quantity!("component_sums"),
            )?,
            error_sums: try_filled(
                components,
                [S::ZERO; 3],
                scioxide_core::quantity!("error_sums"),
            )?,
            decay: try_filled(
                components,
                CycleDecay::EMPTY,
                scioxide_core::quantity!("cycle_decay"),
            )?,
            values: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("component_values"),
            )?,
            errors: try_filled(
                components,
                empty_errors(),
                scioxide_core::quantity!("component_errors"),
            )?,
            total_values: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("total_values"),
            )?,
            total_errors: try_filled(
                components,
                empty_errors(),
                scioxide_core::quantity!("total_errors"),
            )?,
            normalized: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("normalized_errors"),
            )?,
            scales: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("component_scales"),
            )?,
        })
    }
}
impl<S: RealScalar> FourierTailPlan<S> {
    /// Prepares the integral of `f(x)` times `weight` with frequency `ω = omega`
    /// over `domain`.
    ///
    /// `ω` may be negative. For [`FourierDomain::Upper`] and
    /// [`FourierDomain::Lower`], the first cycle has a breakpoint at distance
    /// `min(2/|ω|, L/4)` from the finite endpoint, so an integrable singularity
    /// of `f` there is resolved by Gauss–Kronrod nodes.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `omega` or the finite endpoint is NaN or infinite, or if
    ///   the cycle length `L` is not representable.
    /// - `InvalidInput` if `ω = 0`, if
    ///   [`max_cycles`](FourierTailOptions::max_cycles) is less than six, or if
    ///   the first cycle is too short to be represented at the endpoint.
    /// - The conditions of [`WeightedQuadraturePlan::fourier`].
    pub fn new(
        domain: FourierDomain<S>,
        omega: S,
        weight: FourierWeight,
        options: FourierTailOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let max_cycles = options.max_cycles();
        scioxide_core::validate_value(&omega)?;
        for (invalid_value, argument) in [
            (omega == S::ZERO, scioxide_core::argument!("omega")),
            (max_cycles.get() < 6, scioxide_core::argument!("max_cycles")),
        ] {
            if invalid_value {
                return Err(crate::kernel::invalid(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "infinite Fourier requires nonzero frequency and at least six cycles",
                ));
            }
        }
        // QUADPACK QAWF cycle length (2 floor(|omega|) + 1) pi / |omega|: an odd number of
        // half-periods, so successive cycle integrals of a slowly varying amplitude
        // alternate in sign, which is the regime where Wynn's epsilon algorithm accelerates
        // the partial sums best. The floor term keeps cycles at least ~2 units long.
        let frequency = omega.abs();
        let length = crate::kernel::checked_intermediate(
            (number::<S>(2)? * frequency.floor() + S::ONE) * (S::PI / frequency),
        )?;
        let mut halves = try_with_capacity(2, scioxide_core::quantity!("halves"))?;
        let mut add = |anchor: S, reflect: bool| -> Result<(), SciOxideError> {
            scioxide_core::validate_value(&anchor)?;
            let end = crate::kernel::checked_intermediate(anchor + length)?;
            if end <= anchor {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("omega"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "unrepresentable Fourier cycle",
                ));
            }
            let plan = WeightedQuadraturePlan::fourier(
                Interval::new(anchor, end)?,
                &[],
                omega,
                weight,
                options.quadrature_options().clone(),
            )?;
            // Keep the finite anchor open even for an integrable singular
            // amplitude. The short first segment uses direct weighted Kronrod
            // nodes; the rest of the cycle retains modified moments.
            let first_width = (S::TWO / frequency).min(length * S::HALF * S::HALF);
            let first = WeightedQuadraturePlan::fourier(
                Interval::new(anchor, end)?,
                &[anchor + first_width],
                omega,
                weight,
                options.quadrature_options().clone(),
            )?;
            halves.push(FourierHalf {
                plan,
                first,
                reflect,
            });
            Ok(())
        };
        // Everything is integrated on upper half-lines. A lower half-line is reflected by
        // x -> -x (the sampler negates points; sine values change sign because sin is odd),
        // and the whole line is split at zero into two reflected/unreflected halves.
        match domain {
            FourierDomain::Upper { lower } => add(lower, false)?,
            FourierDomain::Lower { upper } => add(-upper, true)?,
            FourierDomain::RealLine => {
                add(S::ZERO, false)?;
                add(S::ZERO, true)?;
            }
        }
        Ok(Self {
            halves,
            length,
            cycles: max_cycles.get(),
            weight,
            options,
        })
    }

    /// Returns the options.
    pub const fn options(&self) -> &FourierTailOptions<S> {
        &self.options
    }
    /// Returns the identifier of the method, as reported in diagnostics.
    pub fn algorithm_id(&self) -> AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "FourierTail", 1)
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
        let requirements = self.halves[0].first.workspace_requirements(components)?;
        Ok(IntegrationRequirements::new(
            Family::FourierTail,
            components,
            [requirements.units()[0], self.cycles],
        ))
    }
    /// Creates a workspace for `components` real value components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized or
    ///   reserved.
    pub fn workspace(&self, components: usize) -> Result<FourierTailWorkspace<S>, SciOxideError> {
        FourierTailWorkspace::allocate(&self.workspace_requirements(components)?)
    }
    /// Integrates the amplitude `f` against the plan's Fourier weight and writes
    /// the result to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// The samples of one region are evaluated under the execution policy of the
    /// options. See [Termination](crate#termination) for results that end before
    /// the tolerance is met.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` or a per-component tolerance does not have
    ///   the workspace's number of components.
    /// - `InvalidInput` if the workspace is smaller than this plan requires or
    ///   belongs to another plan type.
    /// - The callback, non-finite and stop errors listed under
    ///   [Errors](crate#errors).
    pub fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        f: F,
        out: &mut O,
        workspace: &mut FourierTailWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        workspace.published = false;
        let components = workspace.requirements.components();
        if let Err(error) = check_value_shape(out, components)
            .and_then(|()| {
                check_tolerance(self.options.subdivision_options().tolerance(), components)
            })
            .and_then(|()| {
                if workspace.check(&self.workspace_requirements(components)?)?
                    == scioxide_core::Capacity::NeedsGrowth
                {
                    return Err(crate::kernel::invalid(
                        scioxide_core::argument!("workspace"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "insufficient Fourier storage",
                    ));
                }
                Ok(())
            })
        {
            return workspace.attempts.reject(error);
        }
        let FourierTailWorkspace {
            attempts,
            storage,
            published,
            ..
        } = workspace;
        let policy = self.options.subdivision_options().execution_policy();
        let report = attempts.integrate(
            scope.into(),
            self.algorithm_id(),
            components,
            self.options.subdivision_options().allowance(),
            |cx| {
                let progress = self.run(
                    self.options.subdivision_options().tolerance(),
                    out,
                    storage,
                    cx,
                    &mut |points, rows, context| {
                        let result = crate::evaluation::sample_1d(
                            f, points, components, rows, policy, context,
                        );
                        result.map(|()| points.len())
                    },
                )?;
                if progress.measured {
                    Ok(Outcome::Published {
                        termination: progress.termination,
                        subdivisions: progress.subdivisions,
                        errors: IntegrationErrorComponents::envelope(storage.total_errors.iter()),
                    })
                } else {
                    Ok(Outcome::Unpublished(progress.termination))
                }
            },
        )?;
        *published = true;
        Ok(report)
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
        V: AffineValueMut<Scale = S> + Default,
    {
        let f = &f;
        crate::execution::owned(
            V::default(),
            |components| self.workspace(components),
            |out, workspace, cx| self.integrate_into(f, out, workspace, cx),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn run<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        tolerance: &ErrorTolerance<S>,
        output: &mut O,
        ws: &mut TailStorage<S>,
        context: &mut SolveContext<'_, '_>,
        sampler: &mut Sampler<'_, S>,
    ) -> Result<TailProgress, SciOxideError> {
        let d = ws.components;
        let mass_guard = number::<S>(100)? * S::EPSILON;
        check_value_shape(output, d)?;
        for (mismatch, argument) in [(ws.cycles < self.cycles, scioxide_core::argument!("ws"))] {
            if mismatch {
                return Err(crate::kernel::invalid(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "Fourier workspace shape or cycle capacity mismatch",
                ));
            }
        }
        if tolerance
            .absolute()
            .as_vector()
            .is_some_and(|a| a.len() != d)
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("tolerance.absolute"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Fourier tolerance shape mismatch",
            ));
        }
        let mut report = TailProgress {
            termination: IntegrationTermination::RefinementLimit,
            measured: false,
            subdivisions: 0,
        };
        ws.total_values.fill(S::ZERO);
        ws.total_errors.fill(empty_errors());
        // A summable local error allowance leaves half the requested tolerance
        // for tail/extrapolation. The minimum absolute component allowance is
        // conservative and keeps construction allocation-free for vector policy.
        let mut absolute = S::MAX;
        for c in 0..d {
            absolute = absolute.min(tolerance.absolute().component(c, d)?);
        }
        if d == 0 {
            absolute = S::ZERO;
        }
        let mut all_truncated = true;
        for FourierHalf {
            plan: regular,
            first,
            reflect,
        } in &self.halves
        {
            ws.sums.fill(PromotingCompensatedSum::ZERO);
            ws.error_sums.fill([S::ZERO; 3]);
            ws.extrapolated.reset();
            ws.decay.fill(CycleDecay::EMPTY);
            let mut half_measured = false;
            let mut truncated = false;
            // Cycle k gets tolerance share 0.05 * 0.9^k per half; the geometric series sums
            // to 0.5, i.e. half the requested tolerance, as in QAWF's eps (1 - p) p^k scheme.
            let mut fraction = S::try_from_f64_lossy(0.05)? / number::<S>(self.halves.len())?;
            for cycle in 0..self.cycles {
                let plan = if cycle == 0 { first } else { regular };
                let offset =
                    crate::kernel::checked_intermediate(number::<S>(cycle)? * self.length)?;
                let cycle_tolerance = ErrorTolerance::scalar(
                    absolute * fraction,
                    tolerance.relative() * fraction,
                    tolerance.norm(),
                )?;
                fraction *= S::try_from_f64_lossy(0.9)?;
                // Reuse the scale scratch until the cycle's convergence test.
                // Keep physical amplitude evidence separate from error diagnostics.
                let amplitudes = &mut ws.scales;
                amplitudes.fill(S::ZERO);
                let mut reflected =
                    |points: &[S], values: &mut [S], ctx: &mut SolveContext<'_, '_>| {
                        if *reflect {
                            let mut xs = [S::ZERO; 25];
                            for (x, &p) in xs.iter_mut().zip(points) {
                                *x = -p;
                            }
                            sampler(&xs[..points.len()], values, ctx)?;
                            if self.weight == FourierWeight::Sine {
                                for v in values.iter_mut() {
                                    *v = -*v;
                                }
                            }
                        } else {
                            sampler(points, values, ctx)?;
                        }
                        for (c, amplitude) in amplitudes.iter_mut().enumerate() {
                            *amplitude = values
                                .chunks_exact(d)
                                .fold(*amplitude, |maximum, row| maximum.max(row[c].abs()));
                        }
                        Ok(points.len())
                    };
                let mut sample = solve::epoch(
                    |points: &[S], rows: &mut [S], cx: &mut SolveContext<'_, '_>| {
                        reflected(points, rows, cx).map(|_| ())
                    },
                );
                let mut kernel = solve::WeightedKernel::new(&plan.weighting, &mut sample, 1);
                let scheduled = solve::run_regions(
                    plan.segments(),
                    offset,
                    false,
                    &mut kernel,
                    &mut ws.finite.regional,
                    context,
                    &RegionalPolicy {
                        extrapolation: false,
                        norm: &|e, v| cycle_tolerance.normalized_norm(e, v),
                        refine: true,
                        offsets: None,
                    },
                    &mut EpochAllowance,
                )?;
                if let Scheduled::Measured { subdivisions, .. } = scheduled {
                    report.subdivisions += subdivisions;
                }
                if !scheduled.is_measured() {
                    report.termination = scheduled.termination();
                    break;
                }
                half_measured = true;
                // Distance of the cycle centre from the origin, in the (reflected)
                // integration variable, for the algebraic-decay test.
                let position = crate::kernel::checked_intermediate(
                    regular.interval().lower() + offset + self.length * S::HALF,
                )?
                .abs();
                for c in 0..d {
                    ws.sums[c].add_value(ws.finite.regional.values[c])?;
                    let sum = ws.sums[c].round()?;
                    ws.partials[c] = sum;
                    let parts = ws.finite.regional.parts[c];
                    ws.error_sums[c][0] =
                        crate::kernel::checked_intermediate(ws.error_sums[c][0] + parts[1])?;
                    ws.error_sums[c][1] =
                        crate::kernel::checked_intermediate(ws.error_sums[c][1] + parts[2])?;
                    ws.error_sums[c][2] =
                        crate::kernel::checked_intermediate(ws.error_sums[c][2] + parts[3])?;
                    ws.decay[c].observe(
                        ws.finite.regional.values[c].abs(),
                        ws.scales[c],
                        ws.finite.regional.errors[c],
                        position,
                        self.length,
                        mass_guard,
                    );
                }
                ws.extrapolated
                    .observe(&ws.partials, |c| ws.error_sums[c][1])?;
                for c in 0..d {
                    // Acceleration cannot replace ordinary convergence: a
                    // transient followed by constant alternating masses, or
                    // masses decreasing to a nonzero limit, fails the three
                    // recent decay-to-zero decreases of `CycleDecay`.
                    let evidence = cycle >= 5 && ws.decay[c].established();
                    let candidate = ws.extrapolated.candidate(c);
                    let tail = if evidence {
                        ws.decay[c].remainder()
                    } else {
                        S::ZERO
                    };
                    let accelerated =
                        evidence && candidate.is_some_and(|estimate| estimate.error < tail);
                    ws.values[c] = if accelerated {
                        candidate.map_or(ws.partials[c], |estimate| estimate.value)
                    } else {
                        ws.partials[c]
                    };
                    let mut error =
                        IntegrationErrorComponents::discretization_only(ws.error_sums[c][0]);
                    error.roundoff = Some(ws.error_sums[c][1]);
                    error.moments = Some(ErrorEstimate {
                        magnitude: ws.error_sums[c][2],
                        reliability: ErrorReliability::Estimated,
                    });
                    error.extrapolation = accelerated.then_some(ErrorEstimate {
                        magnitude: candidate.map_or(S::ZERO, |estimate| estimate.error),
                        reliability: ErrorReliability::Estimated,
                    });
                    error.tail = evidence.then_some(ErrorEstimate {
                        magnitude: if accelerated { S::ZERO } else { tail },
                        reliability: ErrorReliability::Heuristic,
                    });
                    error.tail_diagnostics = Some(TailDiagnostics {
                        extent: offset + self.length,
                        shells: cycle + 1,
                        status: TailStatus::Open,
                    });
                    ws.normalized[c] = error.total().unwrap_or(S::MAX);
                    ws.scales[c] = ws.values[c].abs();
                    ws.errors[c] = error;
                }
                // A cycle that stopped by stagnation only because its discretization error hit
                // the roundoff/moment floor is as good as it can get and does not abort the run.
                let roundoff_limited = scheduled.termination()
                    == IntegrationTermination::Stagnation
                    && ws
                        .finite
                        .regional
                        .parts
                        .iter()
                        .all(|parts| parts[1] <= parts[2] + parts[3]);
                if scheduled.termination() != IntegrationTermination::ToleranceReached
                    && !roundoff_limited
                {
                    report.termination = scheduled.termination();
                    break;
                }
                // Divide global error allowance across directions; every cycle's
                // local errors still accumulate without resetting the tolerance.
                for e in &mut ws.normalized {
                    *e *= number::<S>(self.halves.len())?;
                }
                if ws.errors.iter().all(|error| error.total().is_some())
                    && tolerance.is_satisfied(&ws.normalized, &ws.scales)?
                {
                    truncated = true;
                    for e in &mut ws.errors {
                        if let Some(t) = &mut e.tail_diagnostics {
                            t.status = TailStatus::HeuristicallyTruncated;
                        }
                    }
                    break;
                }
            }
            all_truncated &= truncated;
            if !half_measured {
                report.measured = false;
                return Ok(report);
            }
            for c in 0..d {
                ws.total_values[c] =
                    crate::kernel::checked_intermediate(ws.total_values[c] + ws.values[c])?;
                if self.halves.len() == 1 || !report.measured {
                    ws.total_errors[c] = ws.errors[c];
                } else {
                    merge_error(&mut ws.total_errors[c], ws.errors[c])?;
                }
            }
            report.measured = true;
        }
        if all_truncated {
            for c in 0..d {
                ws.normalized[c] = ws.total_errors[c].total().ok_or_else(|| {
                    crate::kernel::invalid(
                        scioxide_core::argument!("function"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "missing completed Fourier error",
                    )
                })?;
                ws.scales[c] = ws.total_values[c].abs();
            }
            report.termination = if tolerance.is_satisfied(&ws.normalized, &ws.scales)? {
                IntegrationTermination::HeuristicToleranceReached
            } else {
                IntegrationTermination::Stagnation
            };
        }
        if report.measured {
            context.commit_accepted(PartialResultTrust::Admissible)?;
            write_value(&ws.total_values, output)?;
        }
        Ok(report)
    }
}

// Combines the two half-line error reports for the whole line: magnitudes add, a missing
// component on either side makes the combined component missing (so `total` refuses to
// claim a bound), heuristic reliability is contagious, and a non-truncated tail status wins.
fn merge_error<S: RealScalar>(
    left: &mut IntegrationErrorComponents<S>,
    right: IntegrationErrorComponents<S>,
) -> Result<(), SciOxideError> {
    for (a, b) in [
        (&mut left.discretization, right.discretization),
        (&mut left.moments, right.moments),
        (&mut left.extrapolation, right.extrapolation),
        (&mut left.tail, right.tail),
    ] {
        *a = match (*a, b) {
            (Some(x), Some(y)) => Some(ErrorEstimate {
                magnitude: crate::kernel::checked_intermediate(x.magnitude + y.magnitude)?,
                reliability: if x.reliability == ErrorReliability::Heuristic
                    || y.reliability == ErrorReliability::Heuristic
                {
                    ErrorReliability::Heuristic
                } else {
                    ErrorReliability::Estimated
                },
            }),
            _ => None,
        };
    }
    left.roundoff = match (left.roundoff, right.roundoff) {
        (Some(a), Some(b)) => Some(crate::kernel::checked_intermediate(a + b)?),
        _ => None,
    };
    if let (Some(a), Some(b)) = (&mut left.tail_diagnostics, right.tail_diagnostics) {
        a.extent = a.extent.max(b.extent);
        a.shells += b.shells;
        if b.status != TailStatus::HeuristicallyTruncated {
            a.status = b.status;
        }
    }
    Ok(())
}
