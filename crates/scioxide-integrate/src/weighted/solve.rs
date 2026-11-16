//! Regional measurement of weighted rules and the execution of one weighted integration.
use alloc::{sync::Arc, vec::Vec};

use scioxide_core::backend::try_filled;
use scioxide_core::{
    AffineValueMut, AlgorithmId, Interval, PartialResultTrust, RealScalar, SciOxideError,
    SolveContext, SolveScope, check_value_shape, write_value,
};

use super::rule::{Rule, Stencil};
use super::{Weight, Weighting};
use crate::SubdivisionOptions;
use crate::adaptive::regional::{
    Allowance, EpochAllowance, Preparation, Region, RegionalKernel, RegionalPolicy,
    RegionalWorkspace, Scheduled, schedule,
};
use crate::report::{IntegrationErrorComponents, IntegrationReport, IntegrationTermination};
use crate::workspace::{Attempts, IntegrationRequirements, Outcome, workspace_protocol};

/// Callback rows of the largest regional rule of weighted and oscillatory plans.
pub(crate) const REGION_NODES: usize = 25;

/// Reusable storage for [`crate::WeightedQuadraturePlan`] and [`crate::ContourPlan`].
///
/// Create it with the plan's `workspace` method. It holds `O(c r)` scalars,
/// where `c` is the number of real value components and `r` is the number of
/// initial segments plus the subdivision limit. One workspace serves every plan
/// whose requirements it meets; see
/// [plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// use scioxide_integrate::{WeightedQuadratureOptions, WeightedQuadraturePlan};
///
/// // PV ∫₀¹ dx / (x − ½) = ln(½ / ½) = 0
/// let options = WeightedQuadratureOptions::default();
/// let plan = WeightedQuadraturePlan::principal_value(Interval::new(0.0, 1.0)?, 0.5, options)?;
/// let mut workspace = plan.workspace(1)?;
/// let mut value = 0.0;
/// let cx = ExecutionContext::sequential_native();
/// let report = plan.integrate_into(|_: f64| Ok(1.0), &mut value, &mut workspace, cx)?;
/// assert!(report.is_success());
/// assert!(value.abs() < 1e-12);
/// assert!(workspace.component_errors().is_some());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct WeightedQuadratureWorkspace<S: RealScalar> {
    pub(crate) requirements: IntegrationRequirements,
    pub(crate) attempts: Attempts<IntegrationReport<S>>,
    pub(crate) regional: RegionalWorkspace<S>,
    pub(crate) errors: Vec<IntegrationErrorComponents<S>>,
    pub(crate) published_errors: bool,
}

impl<S: RealScalar> WeightedQuadratureWorkspace<S> {
    pub(crate) fn allocate(requirements: &IntegrationRequirements) -> Result<Self, SciOxideError> {
        let components = requirements.components();
        let [regions, _] = requirements.units();
        Ok(Self {
            requirements: *requirements,
            attempts: Attempts::default(),
            regional: RegionalWorkspace::new(components, regions, REGION_NODES)?,
            errors: try_filled(
                components,
                IntegrationErrorComponents::default(),
                scioxide_core::quantity!("weighted_errors"),
            )?,
            published_errors: false,
        })
    }

    pub(crate) const fn components(&self) -> usize {
        self.requirements.components()
    }

    /// Returns the error estimate of each component of the last result.
    ///
    /// Returns `None` if the last call failed or estimated no error, as on an
    /// empty interval.
    #[must_use]
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        crate::execution::component_errors(self.published_errors, self.errors.as_slice())
    }
}

workspace_protocol!(WeightedQuadratureWorkspace, IntegrationReport<S>);

/// Evaluation of one prepared region's callback rows under an allowance.
pub(crate) trait RegionSampler<S: RealScalar> {
    type Allowance: Allowance;
    fn sample(
        &mut self,
        points: &[S],
        rows: &mut [S],
        context: &mut SolveContext<'_, '_>,
        allowance: &mut Self::Allowance,
    ) -> Result<(), SciOxideError>;
}

/// Row sampler of a call that runs in its own epoch.
pub(crate) struct Epoch<F>(F);

/// Wrap a row-filling closure; every configured limit is the epoch's allowance.
pub(crate) fn epoch<S, F>(sample: F) -> Epoch<F>
where
    S: RealScalar,
    F: FnMut(&[S], &mut [S], &mut SolveContext<'_, '_>) -> Result<(), SciOxideError>,
{
    Epoch(sample)
}

impl<S, F> RegionSampler<S> for Epoch<F>
where
    S: RealScalar,
    F: FnMut(&[S], &mut [S], &mut SolveContext<'_, '_>) -> Result<(), SciOxideError>,
{
    type Allowance = EpochAllowance;
    fn sample(
        &mut self,
        points: &[S],
        rows: &mut [S],
        context: &mut SolveContext<'_, '_>,
        _allowance: &mut EpochAllowance,
    ) -> Result<(), SciOxideError> {
        (self.0)(points, rows, context)
    }
}

/// Weighted regional rules over a prepared geometry and one row sampler.
pub(crate) struct WeightedKernel<'a, S: RealScalar, P> {
    weighting: &'a Weighting<S>,
    sampler: P,
    /// Provider records charged per callback sample.
    providers_per_sample: usize,
}

impl<'a, S: RealScalar, P> WeightedKernel<'a, S, P> {
    pub(crate) fn new(
        weighting: &'a Weighting<S>,
        sampler: P,
        providers_per_sample: usize,
    ) -> Self {
        Self {
            weighting,
            sampler,
            providers_per_sample,
        }
    }
}

impl<S: RealScalar, P: RegionSampler<S>> RegionalKernel<S> for WeightedKernel<'_, S, P> {
    const TRANSACTIONAL_SPLIT: bool = true;
    fn refine(
        &mut self,
        region: Region<S>,
        _slot: usize,
        _estimates: &[[S; 4]],
    ) -> Result<crate::adaptive::regional::Refinement<S, Self::Prepared>, SciOxideError> {
        crate::adaptive::regional::bisect(self, region)
    }
    type Prepared = Rule<S>;
    type Allowance = P::Allowance;
    #[inline(always)]
    fn prepare(&mut self, region: Region<S>) -> Result<Preparation<Self::Prepared>, SciOxideError> {
        self.weighting.rule(
            region.geometry,
            region.root,
            region.depth,
            self.providers_per_sample,
        )
    }

    fn midpoint(&self, region: Region<S>) -> Result<Option<S>, SciOxideError> {
        if matches!(&self.weighting.weight, Weight::Fourier { omega, levels, .. }
            if region.depth + 1 >= *levels
                && crate::fixed::interval_map(region.geometry)?.scaled_weight(omega.abs())?
                    .shifted(-1)?.cmp_value(scioxide_core::Scaled::from_value(S::TWO)?) == core::cmp::Ordering::Greater)
        {
            return Ok(None);
        }
        // A Cauchy pole must never become a region endpoint (the principal-value rule
        // needs it strictly inside). When the region contains the pole, split at the
        // midpoint of the pole's larger side instead: the pole stays interior to one child
        // and the other child becomes pole-free and regular.
        let mut middle = region.geometry.midpoint()?;
        if let Weight::Cauchy(pole) = self.weighting.weight
            && pole > region.geometry.lower()
            && pole < region.geometry.upper()
        {
            middle = if pole <= middle {
                Interval::new(pole, region.geometry.upper())?.midpoint()?
            } else {
                Interval::new(region.geometry.lower(), pole)?.midpoint()?
            };
        }
        Ok(Some(middle))
    }

    fn measure(
        &mut self,
        _region: Region<S>,
        prepared: &Self::Prepared,
        _slot: usize,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        context: &mut SolveContext<'_, '_>,
        allowance: &mut Self::Allowance,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        let d = estimates.len();
        let rule = prepared;
        // Nodes, sample factors and the moment functional are formed in place
        // once per region, before its callbacks, and serve every component.
        let mut stencil = Stencil::unformed();
        rule.stencil(self.weighting, &mut stencil)?;
        let mut general = None;
        let functional = if rule.has_moments() {
            Some(rule.functional(self.weighting, &mut general)?)
        } else {
            None
        };
        let functional = functional.as_ref();
        self.sampler.sample(
            &stencil.points[..stencil.count],
            &mut samples[..stencil.count * d],
            context,
            allowance,
        )?;
        if let [estimate] = estimates {
            *estimate = stencil.estimate(self.weighting, functional, samples, 0, 1)?;
        } else {
            for (c, estimate) in estimates.iter_mut().enumerate() {
                *estimate = stencil.estimate(self.weighting, functional, samples, c, d)?;
            }
        }
        Ok(None)
    }
}

/// Place `segments`, shifted by `offset`, as the initial regions.
pub(crate) fn place<S: RealScalar>(
    segments: &[Interval<S>],
    offset: S,
    regional: &mut RegionalWorkspace<S>,
) -> Result<(), SciOxideError> {
    if segments.len() > regional.capacity {
        crate::kernel::shape(segments.len(), regional.capacity)?;
    }
    for (root, (region, &interval)) in regional.regions.iter_mut().zip(segments).enumerate() {
        *region = Region::new(
            if offset == S::ZERO {
                interval
            } else {
                Interval::new(
                    crate::kernel::checked_intermediate(interval.lower() + offset)?,
                    crate::kernel::checked_intermediate(interval.upper() + offset)?,
                )?
            },
            root,
        );
    }
    Ok(())
}

/// Commits the workspace's complete-domain estimate and writes it to `out`.
/// Records the error components of the measured
/// `[value, discretization, roundoff, moments]` totals in `errors` and returns
/// their combined estimate.
pub(crate) fn publish<S, O>(
    regional: &RegionalWorkspace<S>,
    errors: &mut [IntegrationErrorComponents<S>],
    out: &mut O,
    context: &mut SolveContext<'_, '_>,
) -> Result<Option<IntegrationErrorComponents<S>>, SciOxideError>
where
    S: RealScalar,
    O: AffineValueMut<Scale = S> + ?Sized,
{
    publish_estimate(regional, errors, out, context, true)
}

/// Exact zero totals of a call that evaluates no callback.
pub(crate) fn clear<S: RealScalar>(regional: &mut RegionalWorkspace<S>) {
    regional.clear_extrapolation();
    regional.values.fill(S::ZERO);
    regional.parts.fill([S::ZERO; 4]);
}

/// Checks shapes and storage, then runs one weighted integration into `out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_into<S, O, P>(
    weighting: &Arc<Weighting<S>>,
    options: &SubdivisionOptions<S>,
    extrapolation: bool,
    algorithm: AlgorithmId,
    sampler: P,
    out: &mut O,
    workspace: &mut WeightedQuadratureWorkspace<S>,
    scope: SolveScope<'_, '_>,
) -> Result<IntegrationReport<S>, SciOxideError>
where
    S: RealScalar,
    O: AffineValueMut<Scale = S> + ?Sized,
    P: RegionSampler<S, Allowance = EpochAllowance>,
{
    let components = workspace.components();
    crate::execution::preflight(
        &mut workspace.attempts,
        &mut workspace.published_errors,
        out,
        components,
        Some(options.tolerance()),
        || {
            let regions = weighting.regions(options)?;
            let [reserved, _] = workspace.requirements.units();
            if regions > reserved {
                crate::kernel::shape(regions, reserved)
            } else {
                Ok(())
            }
        },
    )?;
    let WeightedQuadratureWorkspace {
        attempts,
        regional,
        errors,
        published_errors,
        ..
    } = workspace;
    let segments = &weighting.segments;
    let tolerance = options.tolerance();
    crate::execution::integrate(
        attempts,
        published_errors,
        scope,
        algorithm,
        components,
        options.allowance(),
        |cx| {
            let policy = RegionalPolicy {
                extrapolation,
                norm: &|errors, scales| tolerance.normalized_norm(errors, scales),
                refine: true,
                offsets: None,
            };
            let scheduled = if matches!(weighting.weight, Weight::Ordinary) {
                let mut ordinary =
                    crate::adaptive::kronrod::KronrodKernel::<_, _, REGION_NODES>::new(
                        &weighting.kronrod,
                        sampler,
                        1,
                    );
                run_regions(
                    segments,
                    S::ZERO,
                    weighting.vanishes(),
                    &mut ordinary,
                    regional,
                    cx,
                    &policy,
                    &mut EpochAllowance,
                )?
            } else {
                let mut kernel = WeightedKernel::new(weighting, sampler, 1);
                run_regions(
                    segments,
                    S::ZERO,
                    weighting.vanishes(),
                    &mut kernel,
                    regional,
                    cx,
                    &policy,
                    &mut EpochAllowance,
                )?
            };
            match scheduled {
                Scheduled::Unmeasured(stop) => Ok(Outcome::Unpublished(stop)),
                Scheduled::Measured {
                    termination,
                    subdivisions,
                } => Ok(Outcome::Published {
                    termination,
                    subdivisions,
                    errors: {
                        let evidence = publish(regional, errors, out, cx)?;
                        if segments.is_empty() || components == 0 {
                            None
                        } else {
                            evidence
                        }
                    },
                }),
            }
        },
    )
}

pub(crate) type Sampler<'a, S> =
    dyn FnMut(&[S], &mut [S], &mut SolveContext<'_, '_>) -> Result<usize, SciOxideError> + 'a;

impl<S: RealScalar, P: RegionSampler<S>> RegionSampler<S> for &mut P {
    type Allowance = P::Allowance;
    fn sample(
        &mut self,
        points: &[S],
        rows: &mut [S],
        context: &mut SolveContext<'_, '_>,
        allowance: &mut Self::Allowance,
    ) -> Result<(), SciOxideError> {
        (**self).sample(points, rows, context, allowance)
    }
}

use crate::RegionalProgress;
use crate::adaptive::regional::OptionAllowance;
pub(crate) struct BorrowedSampler<'a, 'b, S: RealScalar>(pub(crate) &'a mut Sampler<'b, S>);
impl<S: RealScalar> RegionSampler<S> for BorrowedSampler<'_, '_, S> {
    type Allowance = OptionAllowance;
    fn sample(
        &mut self,
        points: &[S],
        rows: &mut [S],
        cx: &mut SolveContext<'_, '_>,
        allowance: &mut OptionAllowance,
    ) -> Result<(), SciOxideError> {
        allowance.report.num_evals += (self.0)(points, rows, cx)?;
        Ok(())
    }
}
impl<S: RealScalar> super::Weighting<S> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_controlled<O: AffineValueMut<Scale = S> + ?Sized>(
        self: &Arc<Self>,
        options: &SubdivisionOptions<S>,
        output: &mut O,
        errors: &mut [IntegrationErrorComponents<S>],
        ws: &mut WeightedQuadratureWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
        sampler: &mut Sampler<'_, S>,
        offset: S,
        commit: bool,
        providers_per_sample: usize,
        prior_evaluations: usize,
        policy: &RegionalPolicy<'_, S>,
    ) -> Result<RegionalProgress, SciOxideError> {
        let mut allowance = OptionAllowance::new(
            RegionalProgress {
                termination: IntegrationTermination::ToleranceReached,
                measured: false,
                num_evals: prior_evaluations,
                subdivisions: 0,
            },
            options,
        );
        let scheduled = if matches!(self.weight, Weight::Ordinary) {
            let mut kernel = crate::adaptive::kronrod::KronrodKernel::<_, _, REGION_NODES>::new(
                &self.kronrod,
                BorrowedSampler(sampler),
                providers_per_sample,
            );
            run_regions(
                &self.segments,
                offset,
                self.vanishes(),
                &mut kernel,
                &mut ws.regional,
                context,
                policy,
                &mut allowance,
            )?
        } else {
            let mut kernel =
                WeightedKernel::new(self, BorrowedSampler(sampler), providers_per_sample);
            run_regions(
                &self.segments,
                offset,
                self.vanishes(),
                &mut kernel,
                &mut ws.regional,
                context,
                policy,
                &mut allowance,
            )?
        };
        finish_controlled(
            scheduled,
            allowance,
            &ws.regional,
            errors,
            output,
            context,
            commit,
        )
    }
}
/// The sole regional execution path: initialize geometry and totals, then schedule.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_regions<S: RealScalar, K: RegionalKernel<S>>(
    segments: &[Interval<S>],
    offset: S,
    exact_zero: bool,
    kernel: &mut K,
    ws: &mut RegionalWorkspace<S>,
    context: &mut SolveContext<'_, '_>,
    policy: &RegionalPolicy<'_, S>,
    allowance: &mut K::Allowance,
) -> Result<Scheduled, SciOxideError> {
    if let Some(stop) = context.stop_reason()? {
        return Ok(Scheduled::Unmeasured(
            IntegrationTermination::ExecutionStopped(stop),
        ));
    }
    ws.live = 0;
    if segments.is_empty() || ws.components == 0 || exact_zero {
        clear(ws);
        if let Some((offsets, _)) = policy.offsets {
            ws.values.copy_from_slice(offsets);
        }
        return Ok(Scheduled::Measured {
            termination: if segments.is_empty() {
                IntegrationTermination::DegenerateDomain
            } else if ws.components == 0 {
                IntegrationTermination::FixedRuleCompleted
            } else {
                IntegrationTermination::ToleranceReached
            },
            subdivisions: 0,
        });
    }
    place(segments, offset, ws)?;
    schedule(segments.len(), kernel, policy, ws, context, allowance)
}

/// Writes one measured regional solve to the output and keeps its integrand
/// evaluation count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_controlled<S: RealScalar, O: AffineValueMut<Scale = S> + ?Sized>(
    scheduled: Scheduled,
    allowance: OptionAllowance,
    ws: &RegionalWorkspace<S>,
    errors: &mut [IntegrationErrorComponents<S>],
    output: &mut O,
    context: &mut SolveContext<'_, '_>,
    commit: bool,
) -> Result<RegionalProgress, SciOxideError> {
    let mut report = allowance.report;
    report.termination = scheduled.termination();
    report.measured = scheduled.is_measured();
    if let Scheduled::Measured { subdivisions, .. } = scheduled {
        report.subdivisions = subdivisions;
        publish_estimate(ws, errors, output, context, commit)?;
    }
    Ok(report)
}

/// Stores the component error estimates and optionally commits a complete estimate.
pub(crate) fn publish_estimate<S: RealScalar, O: AffineValueMut<Scale = S> + ?Sized>(
    ws: &RegionalWorkspace<S>,
    errors: &mut [IntegrationErrorComponents<S>],
    out: &mut O,
    cx: &mut SolveContext<'_, '_>,
    commit: bool,
) -> Result<Option<IntegrationErrorComponents<S>>, SciOxideError> {
    crate::kernel::shape(ws.components, errors.len())?;
    check_value_shape(out, ws.components)?;
    for (c, error) in errors.iter_mut().enumerate() {
        *error = ws.evidence(c);
    }
    if commit {
        cx.commit_accepted(PartialResultTrust::Admissible)?;
    }
    write_value(&ws.values, out)?;
    Ok(IntegrationErrorComponents::envelope(errors.iter()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::SolveBudgets;

    #[test]
    fn initial_preflight_and_paired_children_keep_their_own_stencils() {
        fn check<S: RealScalar>() -> Result<(), SciOxideError> {
            let segments: Vec<_> = (0..20)
                .map(|i| {
                    Interval::new(
                        S::try_from_usize_exact(i).unwrap(),
                        S::try_from_usize_exact(i + 1).unwrap(),
                    )
                    .unwrap()
                })
                .collect();
            let weighting = Arc::new(Weighting::prepare(
                Interval::new(S::ZERO, S::try_from_usize_exact(20)?)?,
                &segments
                    .iter()
                    .skip(1)
                    .map(|interval| interval.lower())
                    .collect::<Vec<_>>(),
                crate::GaussKronrodRule::G7K15,
            )?);
            let sampler = epoch(
                |points: &[S], rows: &mut [S], cx: &mut SolveContext<'_, '_>| {
                    crate::evaluation::sample_1d(
                        &|x: S| Ok(x),
                        points,
                        1,
                        rows,
                        scioxide_core::ExecutionPolicy::Sequential,
                        cx,
                    )
                },
            );
            let mut kernel = WeightedKernel::new(&weighting, sampler, 1);
            let missing = || {
                crate::kernel::invalid(
                    scioxide_core::argument!("rule"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected a prepared test stencil",
                )
            };
            let Preparation::Ready(first) = kernel.prepare(Region::new(segments[0], 0))? else {
                return Err(missing());
            };
            // Every payload survives preparations beyond the old 16-slot capacity.
            for (root, &interval) in segments.iter().enumerate().skip(1) {
                kernel.prepare(Region::new(interval, root))?;
            }
            let mut state = scioxide_core::SolveState::new();
            scioxide_core::SolveScope::from(scioxide_core::ExecutionContext::sequential_native())
                .run(
                    &mut state,
                    crate::workspace::identity(
                        scioxide_core::algorithm_id!("test", "stencils", 1),
                        1,
                    ),
                    SolveBudgets::unbounded(),
                    |cx| {
                        let mut rows = [S::ZERO; 15];
                        let mut estimates = [[S::ZERO; 4]; 1];
                        let mut allowance = EpochAllowance;
                        assert!(crate::workspace::admit(cx, first.providers, false)?.is_none());
                        kernel.measure(
                            Region::new(segments[0], 0),
                            &first.rule,
                            0,
                            &mut rows,
                            &mut estimates,
                            cx,
                            &mut allowance,
                        )?;
                        assert!(
                            (estimates[0][0] - S::HALF).abs()
                                <= S::try_from_usize_exact(64)? * S::EPSILON
                        );
                        let children = [
                            Interval::new(S::ZERO, S::HALF)?,
                            Interval::new(S::HALF, S::ONE)?,
                        ];
                        let regions = children.map(|interval| Region {
                            geometry: interval,
                            depth: 1,
                            ..Region::new(interval, 0)
                        });
                        let Preparation::Ready(left) = kernel.prepare(regions[0])? else {
                            return Err(missing());
                        };
                        let Preparation::Ready(right) = kernel.prepare(regions[1])? else {
                            return Err(missing());
                        };
                        for (region, prepared, expected) in [
                            (regions[0], left, S::HALF * S::HALF * S::HALF),
                            (regions[1], right, S::HALF - S::HALF * S::HALF * S::HALF),
                        ] {
                            assert!(
                                crate::workspace::admit(cx, prepared.providers, false)?.is_none()
                            );
                            kernel.measure(
                                region,
                                &prepared.rule,
                                0,
                                &mut rows,
                                &mut estimates,
                                cx,
                                &mut allowance,
                            )?;
                            assert!(
                                (estimates[0][0] - expected).abs()
                                    <= S::try_from_usize_exact(64)? * S::EPSILON
                            );
                        }
                        assert_eq!(
                            cx.provider_count(scioxide_core::EvaluationRole::Function)?,
                            45
                        );
                        Ok(())
                    },
                )
        }
        check::<f32>().unwrap();
        check::<f64>().unwrap();
    }
}
