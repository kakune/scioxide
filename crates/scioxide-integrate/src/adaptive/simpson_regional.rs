//! Simpson regions that reuse the stored nine samples of their parent.
use super::regional::{EpochAllowance, Preparation, Prepared, Refinement, Region, RegionalKernel};
use crate::IntegrationTermination;
use crate::geometry::IntervalGeometry;
use crate::kernel::{WITNESS_FRACTION, WITNESS_WINDOW, WitnessInterpolant, witness_error};
use core::marker::PhantomData;
use scioxide_core::{
    AffineValue, ExecutionPolicy, Interval, RealScalar, Scaled, SciOxideError, SolveContext,
};
const SIMPSON_STENCIL: usize = 9;
const SIMPSON_EXPLORATION_DEPTH: usize = 4;
#[derive(Debug, Clone, Copy)]
pub(crate) struct Stencil<S: RealScalar> {
    points: [S; 17],
    reuse: Option<(usize, usize)>,
    /// The off-grid check point, sampled once with the first region.
    witness: Option<S>,
}
/// Simpson regional kernel. The sample buffer holds the nine samples of each
/// region slot, followed by one row with the check-point sample of every component.
pub(crate) struct SimpsonRegionalKernel<'a, S: RealScalar, V, F: ?Sized> {
    simpson: &'a super::simpson::SimpsonKernel<S>,
    function: &'a F,
    retained: &'a mut [S],
    components: usize,
    policy: ExecutionPolicy,
    /// The check point, once its sample is stored.
    witness: Option<S>,
    /// Depth every region reaches before convergence can be accepted.
    exploration_depth: usize,
    value: PhantomData<fn() -> V>,
}

impl<'a, S: RealScalar, V, F: ?Sized> SimpsonRegionalKernel<'a, S, V, F> {
    pub(crate) fn new(
        simpson: &'a super::simpson::SimpsonKernel<S>,
        function: &'a F,
        retained: &'a mut [S],
        components: usize,
        policy: ExecutionPolicy,
    ) -> Self {
        Self {
            simpson,
            function,
            retained,
            components,
            policy,
            witness: None,
            exploration_depth: SIMPSON_EXPLORATION_DEPTH,
            value: PhantomData,
        }
    }

    /// Compares the check-point sample with the nine samples of the region that
    /// contains it. A disagreement raises the region's error and, while all
    /// regions are still bisected uniformly, extends that uniform depth two
    /// levels below this region: every region at this depth has the same
    /// suspect spacing, and the new depth is set before the next depth exists.
    fn check_witness(
        &mut self,
        region: Region<S>,
        geometry: &IntervalGeometry<S>,
        base: usize,
        estimates: &mut [[S; 4]],
    ) -> Result<(), SciOxideError> {
        let Some(point) = self.witness else {
            return Ok(());
        };
        let lower = region.geometry.lower();
        if point < lower || point > region.geometry.upper() {
            return Ok(());
        }
        let panels = S::from_f64_ieee((SIMPSON_STENCIL - 1) as f64);
        let position = relative_position(point, lower, geometry)?.clamp(S::ZERO, S::ONE) * panels;
        // The eight-node window most nearly centred on the witness panel; the
        // position is nonnegative, so truncation is its floor.
        let panel = position.into_f64() as usize;
        let first = panel
            .saturating_sub(WITNESS_WINDOW / 2 - 1)
            .min(SIMPSON_STENCIL - WITNESS_WINDOW);
        let interpolant = WitnessInterpolant::at(position - S::from_f64_ieee(first as f64));
        let d = self.components;
        let row = self.retained.len() - d;
        let mut missing = false;
        for (c, estimate) in estimates.iter_mut().enumerate() {
            let window = core::array::from_fn(|i| self.retained[(base + first + i) * d + c]);
            let excess = interpolant.excess(&window, self.retained[row + c])?;
            if excess > S::ZERO {
                missing = true;
                estimate[1] = estimate[1].max(witness_error(excess, *geometry)?);
            }
        }
        if missing && region.depth <= self.exploration_depth {
            self.exploration_depth = self.exploration_depth.max(region.depth + 2);
        }
        Ok(())
    }
}

/// `(point - lower) / span`, rounded once. Native arithmetic is used when the
/// difference and quotient are zero or normal; otherwise an extended exponent
/// range gives the same rounding.
#[inline]
fn relative_position<S: RealScalar>(
    point: S,
    lower: S,
    geometry: &IntervalGeometry<S>,
) -> Result<S, SciOxideError> {
    let ordinary = |value: S| value == S::ZERO || value.abs() >= S::MIN_POSITIVE;
    if let Some(span) = geometry.native_span() {
        let offset = point - lower;
        let fraction = offset / span;
        if ordinary(offset) && fraction.is_finite() && ordinary(fraction) {
            return Ok(fraction);
        }
    }
    Scaled::from_value(point)?
        .sub(Scaled::from_value(lower)?)?
        .div(geometry.span()?)?
        .round()
}

impl<S, V, F> RegionalKernel<S> for SimpsonRegionalKernel<'_, S, V, F>
where
    S: RealScalar,
    V: AffineValue<Scale = S>,
    F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
{
    const TRANSACTIONAL_SPLIT: bool = true;
    type Prepared = Stencil<S>;
    type Allowance = EpochAllowance;

    fn prepare(&mut self, region: Region<S>) -> Result<Preparation<Self::Prepared>, SciOxideError> {
        let interval = region.geometry;
        let points = super::simpson::simpson_grid(interval, 8)?;
        let count = SIMPSON_STENCIL;
        let valid = crate::kernel::mapped_geometry(points[..count].iter().copied(), interval, true);
        if !valid {
            return Ok(Preparation::Stop(IntegrationTermination::GeometryFailure));
        }
        // The root stencil also samples the witness once, after its nine nodes.
        let witness = match self.witness {
            None => IntervalGeometry::prepare(interval)?
                .map(|geometry| {
                    geometry.point_at_fraction(S::try_from_f64_lossy(WITNESS_FRACTION)?)
                })
                .transpose()?,
            Some(_) => None,
        };
        let count = count + usize::from(witness.is_some());
        let stencil = Stencil {
            points,
            reuse: None,
            witness,
        };
        Ok(Preparation::Ready(Prepared {
            rule: stencil,
            evaluations: count,
            providers: count,
        }))
    }

    fn exploration(&self, depth: usize) -> usize {
        self.exploration_depth.saturating_sub(depth)
    }

    fn convergence_evidence(&self, parts: &[[S; 4]]) -> bool {
        // The shared floor is strictly positive for every nonzero mass,
        // including subnormals; zero components need no individual evidence.
        parts.iter().any(|part| part[2] > S::ZERO)
    }

    fn midpoint(&self, region: Region<S>) -> Result<Option<S>, SciOxideError> {
        region.geometry.midpoint().map(Some)
    }

    fn refine(
        &mut self,
        region: Region<S>,
        slot: usize,
        _estimates: &[[S; 4]],
    ) -> Result<Refinement<S, Self::Prepared>, SciOxideError> {
        let points = super::simpson::simpson_grid(region.geometry, 16)?;
        if !crate::kernel::mapped_geometry(points.iter().copied(), region.geometry, true) {
            return Ok(Refinement::Stop(IntegrationTermination::Stagnation));
        }
        let children = [
            Interval::new(region.geometry.lower(), points[8])?,
            Interval::new(points[8], region.geometry.upper())?,
        ];
        let prepare_child = |side: usize| -> Result<_, SciOxideError> {
            let mut child = [S::ZERO; 17];
            child[..SIMPSON_STENCIL].copy_from_slice(&points[side * 8..side * 8 + 9]);
            let region = Region {
                geometry: children[side],
                depth: region.depth + 1,
                ..region
            };
            let rule = Stencil {
                points: child,
                reuse: Some((slot, side)),
                witness: None,
            };
            Ok((
                region,
                Prepared {
                    rule,
                    evaluations: 4,
                    providers: 4,
                },
            ))
        };
        Ok(Refinement::Split([prepare_child(0)?, prepare_child(1)?]))
    }

    fn measure(
        &mut self,
        region: Region<S>,
        stencil: &Self::Prepared,
        slot: usize,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        context: &mut SolveContext<'_, '_>,
        _allowance: &mut EpochAllowance,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        let d = self.components;
        // Even child nodes repeat the parent's retained samples; only the
        // remaining nodes are evaluated, in stencil order.
        let base = slot * SIMPSON_STENCIL;
        let mut points = [S::ZERO; SIMPSON_STENCIL + 1];
        let mut nodes = [0; SIMPSON_STENCIL];
        let mut count = 0;
        for (i, &point) in stencil.points[..SIMPSON_STENCIL].iter().enumerate() {
            match stencil.reuse.filter(|_| i % 2 == 0) {
                Some((parent, side)) => {
                    // Staging slots never alias a live parent slot.
                    let source = (parent * SIMPSON_STENCIL + side * 4 + i / 2) * d;
                    self.retained
                        .copy_within(source..source + d, (base + i) * d);
                }
                None => {
                    points[count] = point;
                    nodes[count] = i;
                    count += 1;
                }
            }
        }
        let sampled = count + usize::from(stencil.witness.is_some());
        if let Some(witness) = stencil.witness {
            points[count] = witness;
        }
        let rows = &mut samples[..sampled * d];
        crate::evaluation::sample_1d(
            self.function,
            &points[..sampled],
            d,
            rows,
            self.policy,
            context,
        )?;
        for (row, &i) in rows.chunks_exact(d).zip(&nodes[..count]) {
            self.retained[(base + i) * d..(base + i + 1) * d].copy_from_slice(row);
        }
        if stencil.witness.is_some() {
            let row = self.retained.len() - d;
            self.retained[row..].copy_from_slice(&rows[count * d..sampled * d]);
            self.witness = stencil.witness;
        }
        let geometry = IntervalGeometry::prepare(region.geometry)?.ok_or_else(|| {
            SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "adaptive kernel received a degenerate interval",
            )
        })?;
        let kernel = &self.simpson;
        for (c, estimate) in estimates.iter_mut().enumerate() {
            let values = core::array::from_fn(|i| self.retained[(base + i) * d + c]);
            let e = kernel.estimate(&geometry, values)?;
            *estimate = [e.value, e.error, e.roundoff_floor, S::ZERO];
        }
        self.check_witness(region, &geometry, base, estimates)?;
        Ok(None)
    }

    fn commit(&mut self, source: usize, destination: usize) {
        let stride = SIMPSON_STENCIL * self.components;
        self.retained
            .copy_within(source * stride..(source + 1) * stride, destination * stride);
    }
}
