//! Gauss–Kronrod regions shared by the adaptive, breakpoint and sensitivity plans.
use super::regional::{self, Preparation, Prepared, Refinement, Region, RegionalKernel};
use crate::gauss::kronrod::MAX_NODES;
use crate::weighted::solve::RegionSampler;
use crate::{GaussKronrod, IntegrationTermination};
use alloc::vec::Vec;
use scioxide_core::{Interval, RealScalar, SciOxideError, SolveContext};

/// Nodes of one region, owned so that sibling children cannot overwrite each other.
#[derive(Clone, Copy)]
pub(crate) struct Stencil<S: RealScalar, const N: usize> {
    pub(crate) points: [S; N],
    pub(crate) count: usize,
    pub(crate) scale: scioxide_core::PromotingProduct<S>,
}

/// Maps and checks every node of a region before any callback is charged.
// Out of line: inlined into the kernel's preparation, the returned stencil is
// moved once more per region.
#[inline(never)]
pub(crate) fn prepare<S: RealScalar, const N: usize>(
    rule: &GaussKronrod<S>,
    interval: Interval<S>,
) -> Result<Preparation<Stencil<S, N>>, SciOxideError> {
    let Some(map) = crate::kernel::AffineMap1D::from_interval(interval)? else {
        return Ok(Preparation::Stop(IntegrationTermination::GeometryFailure));
    };
    let rule = rule.rule();
    let count = rule.evaluation_cost();
    if count > N {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(count),
            scioxide_core::Dimensions::one(N),
        ));
    }
    let mut points = [S::ZERO; N];
    let admitted = rule.points(map, &mut points[..count])?;
    if !admitted
        && !crate::kernel::mapped_geometry(points[..count].iter().copied(), interval, false)
    {
        return Ok(Preparation::Stop(IntegrationTermination::GeometryFailure));
    }
    Ok(Preparation::Ready(Prepared {
        rule: Stencil {
            points,
            count,
            scale: map.half_weight()?,
        },
        evaluations: count,
        providers: count,
    }))
}

pub(crate) struct KronrodKernel<'a, S: RealScalar, P, const N: usize, const WIDTH: usize = 0> {
    rule: &'a GaussKronrod<S>,
    sampler: P,
    providers_per_sample: usize,
}
impl<'a, S: RealScalar, P, const N: usize, const WIDTH: usize> KronrodKernel<'a, S, P, N, WIDTH> {
    pub(crate) fn new(rule: &'a GaussKronrod<S>, sampler: P, providers_per_sample: usize) -> Self {
        Self {
            rule,
            sampler,
            providers_per_sample,
        }
    }
}
impl<S: RealScalar, P: RegionSampler<S>, const N: usize, const WIDTH: usize> RegionalKernel<S>
    for KronrodKernel<'_, S, P, N, WIDTH>
{
    const TRANSACTIONAL_SPLIT: bool = true;
    type Prepared = Stencil<S, N>;
    type Allowance = P::Allowance;
    fn prepare(&mut self, region: Region<S>) -> Result<Preparation<Self::Prepared>, SciOxideError> {
        let mut prepared = prepare::<S, N>(self.rule, region.geometry)?;
        if let Preparation::Ready(ready) = &mut prepared {
            ready.providers = scioxide_core::backend::checked_mul(
                ready.evaluations,
                self.providers_per_sample,
                scioxide_core::quantity!("regional_providers"),
            )?;
        }
        Ok(prepared)
    }
    fn midpoint(&self, region: Region<S>) -> Result<Option<S>, SciOxideError> {
        region.geometry.midpoint().map(Some)
    }
    fn refine(
        &mut self,
        region: Region<S>,
        _slot: usize,
        _estimates: &[[S; 4]],
    ) -> Result<Refinement<S, Self::Prepared>, SciOxideError> {
        regional::bisect(self, region)
    }
    fn measure(
        &mut self,
        _region: Region<S>,
        stencil: &Self::Prepared,
        _slot: usize,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        cx: &mut SolveContext<'_, '_>,
        allowance: &mut Self::Allowance,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        let d = if WIDTH == 0 { estimates.len() } else { WIDTH };
        let rows = &mut samples[..stencil.count * d];
        self.sampler
            .sample(&stencil.points[..stencil.count], rows, cx, allowance)?;
        let rule = self.rule.rule();
        if WIDTH == 1 {
            // Only the factory's scalar/sequential branch constructs WIDTH=1;
            // its workspace contributes exactly one estimate to every region.
            let estimate = &mut estimates[0];
            let e = if stencil.count == 15 {
                rule.apply_values::<7>(stencil.scale, rows)?
            } else {
                rule.apply_values::<0>(stencil.scale, rows)?
            };
            *estimate = [e.value(), e.error(), e.roundoff_floor(), S::ZERO];
        } else if let [estimate] = estimates {
            let e = if stencil.count == 15 {
                rule.apply_values::<7>(stencil.scale, rows)?
            } else {
                rule.apply_values::<0>(stencil.scale, rows)?
            };
            *estimate = [e.value(), e.error(), e.roundoff_floor(), S::ZERO];
        } else {
            for (c, estimate) in estimates.iter_mut().enumerate() {
                let e = if stencil.count == 15 {
                    rule.apply_samples::<7>(stencil.scale, |i| Ok(rows[i * d + c]))?
                } else {
                    rule.apply_samples::<0>(stencil.scale, |i| Ok(rows[i * d + c]))?
                };
                *estimate = [e.value(), e.error(), e.roundoff_floor(), S::ZERO];
            }
        }
        Ok(None)
    }
}

/// Unweighted Gauss–Kronrod partition of an interval into segments.
#[derive(Debug)]
pub(crate) struct KronrodPartition<S: RealScalar> {
    interval: Interval<S>,
    segments: Vec<Interval<S>>,
    rule: GaussKronrod<S>,
}
impl<S: RealScalar> KronrodPartition<S> {
    pub(crate) fn new(
        interval: Interval<S>,
        segments: &[Interval<S>],
        rule: crate::GaussKronrodRule,
    ) -> Result<Self, SciOxideError> {
        let mut stored = scioxide_core::backend::try_with_capacity(
            segments.len(),
            scioxide_core::quantity!("segments"),
        )?;
        stored.extend_from_slice(segments);
        let partition = Self {
            interval,
            segments: stored,
            rule: GaussKronrod::new(rule)?,
        };
        partition.validate_sensitivity_geometry()?;
        Ok(partition)
    }
    pub(crate) const fn interval(&self) -> Interval<S> {
        self.interval
    }
    pub(crate) fn segments(&self) -> &[Interval<S>] {
        &self.segments
    }
    fn validate_sensitivity_geometry(&self) -> Result<(), SciOxideError> {
        for &interval in &self.segments {
            if matches!(
                prepare::<S, MAX_NODES>(&self.rule, interval)?,
                Preparation::Stop(_)
            ) {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("interval"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "regional geometry cannot represent distinct quadrature nodes",
                ));
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_controlled<O: scioxide_core::AffineValueMut<Scale = S> + ?Sized>(
        &self,
        options: &crate::SubdivisionOptions<S>,
        output: &mut O,
        errors: &mut [crate::IntegrationErrorComponents<S>],
        ws: &mut super::regional::RegionalWorkspace<S>,
        cx: &mut SolveContext<'_, '_>,
        sampler: &mut crate::weighted::solve::Sampler<'_, S>,
        prior_evaluations: usize,
        policy: &super::regional::RegionalPolicy<'_, S>,
    ) -> Result<crate::RegionalProgress, SciOxideError> {
        let mut allowance = super::regional::OptionAllowance::new(
            crate::RegionalProgress {
                termination: IntegrationTermination::ToleranceReached,
                measured: false,
                num_evals: prior_evaluations,
                subdivisions: 0,
            },
            options,
        );
        let mut kernel = KronrodKernel::<_, _, MAX_NODES>::new(
            &self.rule,
            crate::weighted::solve::BorrowedSampler(sampler),
            2,
        );
        let scheduled = crate::weighted::solve::run_regions(
            &self.segments,
            S::ZERO,
            false,
            &mut kernel,
            ws,
            cx,
            policy,
            &mut allowance,
        )?;
        crate::weighted::solve::finish_controlled(
            scheduled, allowance, ws, errors, output, cx, false,
        )
    }
}
