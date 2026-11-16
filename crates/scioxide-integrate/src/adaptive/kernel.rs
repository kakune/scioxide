//! Dispatch and workspace sizes of the adaptive integrator.
use super::regional::{self, EpochAllowance, Region, RegionalPolicy, RegionalWorkspace, Scheduled};
use super::{AdaptiveRule, AdaptiveWorkspace, SubdivisionOptions};
use crate::report::{IntegrationErrorComponents, IntegrationReport, IntegrationTermination};
use crate::workspace::{Family, IntegrationRequirements, Outcome};
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Interval, PartialResultTrust, RealScalar,
    SciOxideError, SolveContext, SolveScope, write_value,
};
const SIMPSON_STENCIL: usize = 9;
const SIMPSON_EXPLORATION_DEPTH: usize = 4;
impl AdaptiveRule {
    const fn stencil(self) -> usize {
        match self {
            Self::GaussKronrod => 0,
            Self::Simpson => SIMPSON_STENCIL,
        }
    }

    /// Checks the Simpson subdivision minimum and that the region storage for
    /// `options` is representable.
    pub(crate) fn validate<S: RealScalar>(
        self,
        options: &SubdivisionOptions<S>,
    ) -> Result<(), SciOxideError> {
        if self == Self::Simpson
            && options.max_subdivisions().get() < (1 << SIMPSON_EXPLORATION_DEPTH) - 1
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("options.max_subdivisions"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Simpson requires at least 15 subdivisions for exploration",
            ));
        }
        self.requirements(options, 0).map(|_| ())
    }

    /// Workspace units: regions, and stored samples per region.
    pub(crate) fn requirements<S: RealScalar>(
        self,
        options: &SubdivisionOptions<S>,
        components: usize,
    ) -> Result<IntegrationRequirements, SciOxideError> {
        let regions = checked_add(
            options.max_subdivisions().get(),
            1,
            scioxide_core::quantity!("adaptive_regions"),
        )?;
        // The staging slots and retained stencils must also stay representable.
        checked_mul(
            checked_add(regions, 2, scioxide_core::quantity!("adaptive_slots"))?,
            self.stencil().max(1),
            scioxide_core::quantity!("retained_samples"),
        )?;
        Ok(IntegrationRequirements::new(
            Family::Adaptive,
            components,
            [regions, self.stencil()],
        ))
    }
}

/// Rule coefficients, borrowed separately from the scheduler state.
#[derive(Debug)]
pub(crate) struct PreparedRules<S: RealScalar> {
    kronrod: crate::GaussKronrod<S>,
    simpson: super::simpson::SimpsonKernel<S>,
}
impl<S: RealScalar> PreparedRules<S> {
    pub(crate) fn new() -> Result<Self, SciOxideError> {
        Ok(Self {
            kronrod: crate::GaussKronrod::new(crate::GaussKronrodRule::G7K15)?,
            simpson: super::simpson::SimpsonKernel::prepare()?,
        })
    }
}

/// Validates shapes, then runs one adaptive integration into `out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_into<S, F, V, O>(
    method: AdaptiveRule,
    algorithm: AlgorithmId,
    options: &SubdivisionOptions<S>,
    extrapolation: bool,
    function: &F,
    interval: Interval<S>,
    out: &mut O,
    workspace: &mut AdaptiveWorkspace<S>,
    scope: SolveScope<'_, '_>,
) -> Result<IntegrationReport<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
    V: AffineValue<Scale = S>,
    O: AffineValueMut<Scale = S> + ?Sized,
{
    let components = workspace.requirements.components();
    crate::execution::preflight(
        &mut workspace.attempts,
        &mut workspace.published_errors,
        out,
        components,
        Some(options.tolerance()),
        || {
            Ok(()).and_then(|()| {
                // Too few regions or no retained stencils for this method.
                let [regions, stencil] = method.requirements(options, components)?.units();
                let [reserved_regions, reserved_stencil] = workspace.requirements.units();
                if regions > reserved_regions {
                    crate::kernel::shape(regions, reserved_regions)
                } else if stencil > reserved_stencil {
                    crate::kernel::shape(stencil, reserved_stencil)
                } else {
                    Ok(())
                }
            })
        },
    )?;
    let AdaptiveWorkspace {
        attempts,
        regional,
        preparation,
        retained,
        errors,
        published_errors,
        ..
    } = workspace;
    crate::execution::integrate(
        attempts,
        published_errors,
        scope,
        algorithm,
        components,
        options.allowance(),
        |cx| {
            if components == 0 || interval.lower() == interval.upper() {
                return publish_zero(components, regional, out, cx);
            }
            regional.regions[0] = Region::new(interval, 0);
            schedule_into::<S, F, V, O>(
                method,
                options,
                extrapolation,
                function,
                1,
                regional,
                preparation,
                retained,
                errors,
                out,
                cx,
            )
        },
    )
}

/// Writes the exact zero of an empty value or a zero-width domain.
pub(crate) fn publish_zero<S, O>(
    components: usize,
    regional: &mut RegionalWorkspace<S>,
    out: &mut O,
    cx: &mut SolveContext<'_, '_>,
) -> Result<Outcome<S>, SciOxideError>
where
    S: RealScalar,
    O: AffineValueMut<Scale = S> + ?Sized,
{
    regional.clear_extrapolation();
    crate::execution::zero(
        &mut regional.values,
        out,
        if components == 0 {
            IntegrationTermination::FixedRuleCompleted
        } else {
            IntegrationTermination::DegenerateDomain
        },
        cx,
    )
}

/// Refines the `initial` regions already placed in `regional`, then writes the
/// estimate to `out` and the per-component error estimates to `errors`.
///
/// The initial regions are charged to the budget as one unit before any is
/// evaluated. One-dimensional adaptive cubature places its breakpoint pieces here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn schedule_into<S, F, V, O>(
    method: AdaptiveRule,
    options: &SubdivisionOptions<S>,
    extrapolation: bool,
    function: &F,
    initial: usize,
    regional: &mut RegionalWorkspace<S>,
    preparation: &PreparedRules<S>,
    retained: &mut [S],
    errors: &mut [IntegrationErrorComponents<S>],
    out: &mut O,
    cx: &mut SolveContext<'_, '_>,
) -> Result<Outcome<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
    V: AffineValue<Scale = S>,
    O: AffineValueMut<Scale = S> + ?Sized,
{
    let tolerance = options.tolerance();
    let policy = RegionalPolicy {
        extrapolation,
        norm: &|errors, scales| tolerance.normalized_norm(errors, scales),
        refine: true,
        offsets: None,
    };
    let components = regional.components;
    let scheduled = match method {
        AdaptiveRule::GaussKronrod => {
            if components == 1
                && options.execution_policy() == scioxide_core::ExecutionPolicy::Sequential
            {
                // This sampler's component count and policy are literal in its
                // monomorphization, not captured runtime values.
                let sampler = crate::weighted::solve::epoch({
                    #[inline(always)]
                    |points: &[S], rows: &mut [S], cx: &mut SolveContext<'_, '_>| {
                        crate::evaluation::sample_scalar_1d::<S, V, F, 0>(
                            function, points, rows, cx,
                        )
                        .map(|_| ())
                    }
                });
                let mut kernel = super::kronrod::KronrodKernel::<_, _, 15, 1>::new(
                    &preparation.kronrod,
                    sampler,
                    1,
                );
                regional::schedule(
                    initial,
                    &mut kernel,
                    &policy,
                    regional,
                    cx,
                    &mut EpochAllowance,
                )?
            } else {
                let sampler = crate::weighted::solve::epoch(
                    |points: &[S], rows: &mut [S], cx: &mut SolveContext<'_, '_>| {
                        crate::evaluation::sample_1d(
                            function,
                            points,
                            components,
                            rows,
                            options.execution_policy(),
                            cx,
                        )
                    },
                );
                let mut kernel = super::kronrod::KronrodKernel::<_, _, 15>::new(
                    &preparation.kronrod,
                    sampler,
                    1,
                );
                regional::schedule(
                    initial,
                    &mut kernel,
                    &policy,
                    regional,
                    cx,
                    &mut EpochAllowance,
                )?
            }
        }
        AdaptiveRule::Simpson => {
            let mut kernel = super::simpson_regional::SimpsonRegionalKernel::new(
                &preparation.simpson,
                function,
                retained,
                regional.components,
                options.execution_policy(),
            );
            regional::schedule(
                initial,
                &mut kernel,
                &policy,
                regional,
                cx,
                &mut EpochAllowance,
            )?
        }
    };
    match scheduled {
        Scheduled::Unmeasured(stop) => Ok(Outcome::Unpublished(stop)),
        Scheduled::Measured {
            termination,
            subdivisions,
        } => {
            for (c, error) in errors.iter_mut().enumerate() {
                *error = regional.evidence(c);
            }
            cx.commit_accepted(PartialResultTrust::Admissible)?;
            write_value(&regional.values, out)?;
            Ok(Outcome::Published {
                termination,
                subdivisions,
                errors: IntegrationErrorComponents::envelope(errors.iter()),
            })
        }
    }
}
