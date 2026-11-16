//! Region scheduler shared by the adaptive rules: priorities, totals and
//! budget charging. A bisection is evaluated completely or not at all.
use super::SubdivisionOptions;
use crate::IntegrationTermination;
use crate::execution::RegionalProgress;
use alloc::{collections::BinaryHeap, vec::Vec};
use core::cmp::Ordering;
use scioxide_core::backend::{RealValueSum, checked_add, checked_mul, try_filled};
use scioxide_core::{Budget, BudgetKind, Interval, RealScalar, SciOxideError, SolveContext};

/// Max-heap key of a region; it does not depend on the changing global totals.
#[derive(Clone, Copy)]
pub(crate) struct Priority<S: RealScalar> {
    pub(crate) score: S,
    pub(crate) exploration: usize,
    pub(crate) slot: usize,
    pub(crate) sequence: usize,
}
impl<S: RealScalar> PartialEq for Priority<S> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<S: RealScalar> Eq for Priority<S> {}
impl<S: RealScalar> PartialOrd for Priority<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<S: RealScalar> Ord for Priority<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.exploration
            .cmp(&other.exploration)
            .then_with(|| {
                self.score
                    .partial_cmp(&other.score)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

/// Adds or subtracts region estimates in the totals without rescanning all regions.
pub(crate) fn accumulate<S: RealScalar, const N: usize>(
    totals: &mut [[RealValueSum<S>; N]],
    estimates: impl Iterator<Item = [S; N]>,
    subtract: bool,
) -> Result<(), SciOxideError> {
    for (sums, values) in totals.iter_mut().zip(estimates) {
        for (sum, value) in sums.iter_mut().zip(values) {
            sum.add_value(if subtract { -value } else { value })?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct Region<S: RealScalar, G: Copy = Interval<S>> {
    pub(crate) geometry: G,
    pub(crate) root: usize,
    pub(crate) depth: usize,
    /// Fraction of the outer target allocated to nested uncertainty.
    pub(crate) factor: S,
}
impl<S: RealScalar, G: Copy> Region<S, G> {
    pub(crate) fn new(geometry: G, root: usize) -> Self {
        Self {
            geometry,
            root,
            depth: 0,
            factor: S::HALF,
        }
    }
}
impl<S: RealScalar> Region<S> {
    fn child(self, interval: Interval<S>) -> Self {
        Self {
            geometry: interval,
            depth: self.depth + 1,
            ..self
        }
    }
}

/// Result of preparing a region. Geometry and refinement limits are
/// terminations, not invalid-input errors.
pub(crate) enum Preparation<P> {
    Ready(Prepared<P>),
    Stop(IntegrationTermination),
}
pub(crate) struct Prepared<P> {
    pub(crate) rule: P,
    /// Whole-value integrand callbacks of the prepared rule.
    pub(crate) evaluations: usize,
    /// Evaluations charged to the solve context, including auxiliary callbacks.
    pub(crate) providers: usize,
}
pub(crate) enum Refinement<S: RealScalar, P, G: Copy = Interval<S>> {
    Split([(Region<S, G>, Prepared<P>); 2]),
    Tighten(Region<S, G>, Prepared<P>),
    Stop(IntegrationTermination),
}

/// How a regional solve charges complete work units and bisections to its budget.
pub(crate) trait Allowance {
    /// Charges one complete unit of `calls` callbacks, counted as `providers`
    /// evaluations, before any of its callbacks runs.
    fn admit(
        &mut self,
        context: &mut SolveContext<'_, '_>,
        calls: usize,
        providers: usize,
        split: bool,
    ) -> Result<Option<IntegrationTermination>, SciOxideError>;

    /// Whether the budget allows one more bisection.
    fn can_subdivide(&self, context: &SolveContext<'_, '_>) -> Result<bool, SciOxideError>;
}

/// Charges the solve context, which holds every configured and inherited limit.
pub(crate) struct EpochAllowance;

impl Allowance for EpochAllowance {
    fn admit(
        &mut self,
        context: &mut SolveContext<'_, '_>,
        _calls: usize,
        providers: usize,
        split: bool,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        crate::workspace::admit(context, providers, split)
    }

    fn can_subdivide(&self, context: &SolveContext<'_, '_>) -> Result<bool, SciOxideError> {
        Ok(context
            .can_consume(BudgetKind::Subdivision, 1)?
            .is_granted())
    }
}

/// Option limits counted in whole callbacks, for methods that run inside the
/// caller's solve context and report a [`RegionalProgress`].
pub(crate) struct OptionAllowance {
    pub(crate) report: RegionalProgress,
    max_evaluations: Option<Budget>,
    max_subdivisions: usize,
}

impl OptionAllowance {
    pub(crate) fn new<S: RealScalar>(
        report: RegionalProgress,
        options: &SubdivisionOptions<S>,
    ) -> Self {
        Self {
            report,
            max_evaluations: options.max_evaluations(),
            max_subdivisions: options.max_subdivisions().get(),
        }
    }
}

impl Allowance for OptionAllowance {
    fn admit(
        &mut self,
        context: &mut SolveContext<'_, '_>,
        calls: usize,
        providers: usize,
        split: bool,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        if self
            .max_evaluations
            .is_some_and(|limit| calls > limit.get().saturating_sub(self.report.num_evals))
        {
            return Ok(Some(IntegrationTermination::BudgetLimit(
                BudgetKind::Evaluation,
            )));
        }
        if split && self.report.subdivisions >= self.max_subdivisions {
            return Ok(Some(IntegrationTermination::BudgetLimit(
                BudgetKind::Subdivision,
            )));
        }
        let stop = crate::workspace::admit(context, providers, split)?;
        if stop.is_none() && split {
            self.report.subdivisions += 1;
        }
        Ok(stop)
    }

    fn can_subdivide(&self, _context: &SolveContext<'_, '_>) -> Result<bool, SciOxideError> {
        Ok(self.report.subdivisions < self.max_subdivisions)
    }
}

pub(crate) trait RegionalKernel<S: RealScalar, G: Copy = Interval<S>> {
    /// Refinement returns a complete two-child split or a stop, never a tightening.
    const TRANSACTIONAL_SPLIT: bool = false;
    /// Whether the slow-progress counters apply. They assume the asymptotic
    /// error reduction of one-dimensional bisection (Piessens et al., 1983).
    /// Other rules still stop at the rounding-error floor and at geometry limits.
    const SLOW_PROGRESS: bool = true;
    type Prepared;
    type Allowance: Allowance;
    fn prepare(
        &mut self,
        region: Region<S, G>,
    ) -> Result<Preparation<Self::Prepared>, SciOxideError>;
    fn midpoint(&self, _region: Region<S, G>) -> Result<Option<S>, SciOxideError> {
        Ok(None)
    }
    fn exploration(&self, _depth: usize) -> usize {
        0
    }
    fn convergence_evidence(&self, _parts: &[[S; 4]]) -> bool {
        true
    }
    fn refine(
        &mut self,
        region: Region<S, G>,
        slot: usize,
        estimates: &[[S; 4]],
    ) -> Result<Refinement<S, Self::Prepared, G>, SciOxideError>;

    /// Measurement may stop while recursively evaluating an inner axis.
    #[allow(clippy::too_many_arguments)]
    fn measure(
        &mut self,
        region: Region<S, G>,
        prepared: &Self::Prepared,
        slot: usize,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        context: &mut SolveContext<'_, '_>,
        allowance: &mut Self::Allowance,
    ) -> Result<Option<IntegrationTermination>, SciOxideError>;
    /// Adjust complete children before totals, priorities and stopping are updated.
    fn refinement_estimates(
        &self,
        _parent: [S; 4],
        _left: &mut [S; 4],
        _right: &mut [S; 4],
    ) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn commit(&mut self, _source: usize, _destination: usize) {}
}
pub(crate) fn bisect<S: RealScalar, K: RegionalKernel<S> + ?Sized>(
    kernel: &mut K,
    region: Region<S>,
) -> Result<Refinement<S, K::Prepared>, SciOxideError> {
    let Some(midpoint) = kernel.midpoint(region)? else {
        return Ok(Refinement::Stop(IntegrationTermination::RefinementLimit));
    };
    if midpoint <= region.geometry.lower() || midpoint >= region.geometry.upper() {
        return Ok(Refinement::Stop(IntegrationTermination::Stagnation));
    }
    let left = region.child(Interval::new(region.geometry.lower(), midpoint)?);
    let right = region.child(Interval::new(midpoint, region.geometry.upper())?);
    match (kernel.prepare(left)?, kernel.prepare(right)?) {
        (Preparation::Ready(left_rule), Preparation::Ready(right_rule)) => {
            Ok(Refinement::Split([(left, left_rule), (right, right_rule)]))
        }
        (Preparation::Stop(stop), _) | (_, Preparation::Stop(stop)) => {
            Ok(Refinement::Stop(refinement_stop(stop)))
        }
    }
}
fn refinement_stop(stop: IntegrationTermination) -> IntegrationTermination {
    if stop == IntegrationTermination::GeometryFailure {
        IntegrationTermination::Stagnation
    } else {
        stop
    }
}

type RegionalNorm<'a, S> = &'a dyn Fn(&[S], &[S]) -> Result<S, SciOxideError>;
/// Error control of one regional solve; the norm comes from the plan's tolerance.
pub(crate) struct RegionalPolicy<'a, S> {
    pub(crate) extrapolation: bool,
    pub(crate) norm: RegionalNorm<'a, S>,
    pub(crate) refine: bool,
    pub(crate) offsets: Option<(&'a [S], &'a [S])>,
}

/// Outcome of one regional solve.
#[derive(Clone, Copy)]
pub(crate) enum Scheduled {
    /// The workspace holds a complete-domain estimate.
    Measured {
        termination: IntegrationTermination,
        subdivisions: usize,
    },
    /// The solve stopped before every initial region was measured.
    Unmeasured(IntegrationTermination),
}

impl Scheduled {
    pub(crate) const fn termination(self) -> IntegrationTermination {
        match self {
            Self::Measured { termination, .. } | Self::Unmeasured(termination) => termination,
        }
    }

    pub(crate) const fn is_measured(self) -> bool {
        matches!(self, Self::Measured { .. })
    }
}

/// Scheduler storage. Each estimate is [value, discretization, roundoff floor,
/// additional uncertainty]. Slots `capacity` and `capacity + 1` hold new children.
pub(crate) struct RegionalWorkspace<S: RealScalar, G: Copy = Interval<S>> {
    pub(crate) components: usize,
    pub(crate) capacity: usize,
    pub(crate) live: usize,
    pub(crate) regions: Vec<Region<S, G>>,
    pub(crate) estimates: Vec<[S; 4]>,
    pub(crate) samples: Vec<S>,
    pub(crate) values: Vec<S>,
    pub(crate) errors: Vec<S>,
    pub(crate) scales: Vec<S>,
    pub(crate) parts: Vec<[S; 4]>,
    queue: BinaryHeap<Priority<S>>,
    totals: Vec<[RealValueSum<S>; 4]>,
    // Lazy heap entries carry a unique sequence stamp. Every region version
    // enters each heap at most once, so twice the live capacity bounds storage.
    large_queue: BinaryHeap<Priority<S>>,
    pending_queue: BinaryHeap<Priority<S>>,
    stamps: Vec<usize>,
    large_totals: Vec<RealValueSum<S>>,
    large_errors: Vec<S>,
    selected: Vec<Option<S>>,
    extrapolated: super::extrapolation::Extrapolation<S>,
    level: usize,
}
impl<S: RealScalar, G: Copy> core::fmt::Debug for RegionalWorkspace<S, G> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RegionalWorkspace")
            .field("components", &self.components)
            .field("capacity", &self.capacity)
            .field("live", &self.live)
            .finish_non_exhaustive()
    }
}
impl<S: RealScalar> RegionalWorkspace<S> {
    pub(crate) fn new(
        components: usize,
        capacity: usize,
        nodes: usize,
    ) -> Result<Self, SciOxideError> {
        Self::allocate(
            components,
            capacity,
            nodes,
            Interval::new(S::ZERO, S::ZERO)?,
            true,
        )
    }
}
impl<S: RealScalar, G: Copy> RegionalWorkspace<S, G> {
    /// Creates storage for scheduling without extrapolation.
    pub(crate) fn with_geometry(
        components: usize,
        capacity: usize,
        nodes: usize,
        geometry: G,
    ) -> Result<Self, SciOxideError> {
        Self::allocate(components, capacity, nodes, geometry, false)
    }

    fn allocate(
        components: usize,
        capacity: usize,
        nodes: usize,
        geometry: G,
        extrapolation: bool,
    ) -> Result<Self, SciOxideError> {
        let extra_components = if extrapolation { components } else { 0 };
        let extra_slots = if extrapolation {
            checked_add(capacity, 2, scioxide_core::quantity!("regional_slots"))?
        } else {
            0
        };
        let slots = checked_add(capacity, 2, scioxide_core::quantity!("regional_slots"))?;
        let mut queue = BinaryHeap::new();
        let heap_capacity = checked_mul(capacity, 2, scioxide_core::quantity!("regional_queue"))?;
        queue.try_reserve(heap_capacity).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("regional_queue"), capacity)
        })?;
        let mut large_queue = BinaryHeap::new();
        let mut pending_queue = BinaryHeap::new();
        for heap in [&mut large_queue, &mut pending_queue] {
            heap.try_reserve(if extrapolation { heap_capacity } else { 0 })
                .map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("regional_queue"),
                        heap_capacity,
                    )
                })?;
        }
        Ok(Self {
            components,
            capacity,
            live: 0,
            queue,
            large_queue,
            pending_queue,
            stamps: try_filled(extra_slots, 0, scioxide_core::quantity!("regional_stamps"))?,
            large_totals: try_filled(
                extra_components,
                RealValueSum::new(),
                scioxide_core::quantity!("regional_large_totals"),
            )?,
            large_errors: try_filled(
                extra_components,
                S::ZERO,
                scioxide_core::quantity!("regional_large_errors"),
            )?,
            selected: try_filled(
                extra_components,
                None,
                scioxide_core::quantity!("regional_extrapolation"),
            )?,
            extrapolated: super::extrapolation::Extrapolation::new(extra_components)?,
            level: 0,
            regions: try_filled(
                slots,
                Region::new(geometry, 0),
                scioxide_core::quantity!("regions"),
            )?,
            estimates: try_filled(
                checked_mul(
                    slots,
                    components,
                    scioxide_core::quantity!("regional_estimates"),
                )?,
                [S::ZERO; 4],
                scioxide_core::quantity!("regional_estimates"),
            )?,
            samples: try_filled(
                checked_mul(
                    nodes,
                    components,
                    scioxide_core::quantity!("regional_samples"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("regional_samples"),
            )?,
            values: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("regional_values"),
            )?,
            errors: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("regional_errors"),
            )?,
            scales: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("regional_scales"),
            )?,
            parts: try_filled(
                components,
                [S::ZERO; 4],
                scioxide_core::quantity!("regional_diagnostics"),
            )?,
            totals: try_filled(
                components,
                core::array::from_fn(|_| RealValueSum::new()),
                scioxide_core::quantity!("regional_totals"),
            )?,
        })
    }
    fn add(&mut self, slot: usize, subtract: bool) -> Result<(), SciOxideError> {
        accumulate(
            &mut self.totals,
            self.estimates[slot * self.components..(slot + 1) * self.components]
                .iter()
                .copied(),
            subtract,
        )
    }

    #[inline(always)]
    fn copy_estimate(&mut self, source: usize, destination: usize) {
        if self.components == 1 {
            self.estimates[destination] = self.estimates[source];
        } else {
            self.estimates.copy_within(
                source * self.components..(source + 1) * self.components,
                destination * self.components,
            );
        }
    }
    fn finish(&mut self, policy: &RegionalPolicy<'_, S>) -> Result<(), SciOxideError> {
        for (c, sums) in self.totals.iter_mut().enumerate() {
            let mut parts = [S::ZERO; 4];
            for (value, sum) in parts.iter_mut().zip(sums) {
                *value = sum.finish()?;
            }
            self.values[c] = parts[0];
            self.scales[c] = parts[0].abs();
            self.errors[c] =
                crate::kernel::checked_intermediate(parts[1].max(parts[2]) + parts[3])?;
            if let Some((_, floors)) = policy.offsets {
                self.errors[c] = crate::kernel::checked_intermediate(self.errors[c] + floors[c])?;
            }
            self.parts[c] = parts;
        }
        Ok(())
    }
    /// A two-region total is one correctly rounded addition per scalar part.
    fn finish_pair(&mut self, left: usize, right: usize) -> Result<(), SciOxideError> {
        for c in 0..self.components {
            let mut parts = [S::ZERO; 4];
            for (p, value) in parts.iter_mut().enumerate() {
                let sum = &mut self.totals[c][p];
                sum.add_value(self.estimates[left * self.components + c][p])?;
                sum.add_value(self.estimates[right * self.components + c][p])?;
                *value = sum.finish()?;
            }
            self.parts[c] = parts;
            self.values[c] = parts[0];
            self.scales[c] = parts[0].abs();
            self.errors[c] =
                crate::kernel::checked_intermediate(parts[1].max(parts[2]) + parts[3])?;
        }
        Ok(())
    }

    fn enqueue(
        &mut self,
        slot: usize,
        sequence: usize,
        exploration: usize,
        policy: &RegionalPolicy<'_, S>,
    ) -> Result<(), SciOxideError> {
        // Scalar QAG priorities need only an absolute error contribution.
        // Keep global error/scale scratch intact while constructing the key.
        let score = if self.components == 1 {
            let parts = self.estimates[slot];
            crate::kernel::diagnostic(crate::kernel::checked_intermediate(
                parts[1].max(parts[2]) + parts[3],
            ))?
            .abs()
        } else {
            for (c, e) in self.estimates[slot * self.components..(slot + 1) * self.components]
                .iter()
                .enumerate()
            {
                self.errors[c] = crate::kernel::diagnostic(crate::kernel::checked_intermediate(
                    e[1].max(e[2]) + e[3],
                ))?;
                self.scales[c] = e[0].abs();
            }
            crate::kernel::diagnostic((policy.norm)(&self.errors, &self.scales))?
        };
        self.queue.push(Priority {
            slot,
            sequence,
            exploration,
            score,
        });
        if policy.extrapolation {
            self.stamps[slot] = sequence;
            let priority = Priority {
                slot,
                sequence,
                exploration: 0,
                score,
            };
            if self.regions[slot].depth <= self.level {
                self.large_queue.push(priority);
                self.update_large(slot, false)?;
            } else {
                self.pending_queue.push(Priority {
                    exploration: usize::MAX - self.regions[slot].depth,
                    ..priority
                });
            }
        }
        Ok(())
    }

    fn update_large(&mut self, slot: usize, subtract: bool) -> Result<(), SciOxideError> {
        for (sum, parts) in self
            .large_totals
            .iter_mut()
            .zip(&self.estimates[slot * self.components..(slot + 1) * self.components])
        {
            let error = crate::kernel::checked_intermediate(parts[1].max(parts[2]) + parts[3])?;
            sum.add_value(if subtract { -error } else { error })?;
        }
        Ok(())
    }

    fn advance_level(&mut self) -> Result<(), SciOxideError> {
        self.level += 1;
        loop {
            discard_stale(&mut self.pending_queue, &self.stamps);
            let Some(priority) = self.pending_queue.peek() else {
                break;
            };
            if self.regions[priority.slot].depth > self.level {
                break;
            }
            let priority = self.pending_queue.pop().ok_or_else(empty_queue)?;
            self.update_large(priority.slot, false)?;
            self.large_queue.push(Priority {
                exploration: 0,
                ..priority
            });
        }
        Ok(())
    }

    pub(crate) fn measured_regions(&self) -> impl Iterator<Item = G> + '_ {
        self.regions[..self.live]
            .iter()
            .map(|region| region.geometry)
    }

    /// Error estimate of component `c`, matching the value written to the output.
    pub(crate) fn evidence(&self, c: usize) -> crate::IntegrationErrorComponents<S> {
        let parts = self.parts[c];
        let mut error = crate::IntegrationErrorComponents::discretization_only(parts[1])
            .with_roundoff(parts[2]);
        if parts[3] != S::ZERO {
            error.moments = Some(crate::ErrorEstimate::new(
                parts[3],
                crate::ErrorReliability::Estimated,
            ));
        }
        error.extrapolation = self.selected.get(c).copied().flatten().map(|magnitude| {
            crate::ErrorEstimate::new(magnitude, crate::ErrorReliability::Estimated)
        });
        error
    }

    pub(crate) fn clear_extrapolation(&mut self) {
        self.selected.fill(None);
        self.extrapolated.reset();
    }

    fn candidate_norm(
        &mut self,
        policy: &RegionalPolicy<'_, S>,
    ) -> Result<Option<S>, SciOxideError> {
        for c in 0..self.components {
            if let Some(candidate) = self.extrapolated.candidate(c) {
                let total = candidate.total()?;
                if total < self.errors[c] {
                    self.errors[c] = total;
                    self.scales[c] = candidate.value.abs();
                }
            }
        }
        match (policy.norm)(&self.errors, &self.scales) {
            Ok(norm) => Ok(Some(norm)),
            // A candidate at zero can have an unbounded relative-only error
            // even though the direct estimate's normalization is representable.
            // Such a candidate is unusable; it must not discard the direct result.
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    }
                ) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn select_candidates(&mut self) -> Result<(), SciOxideError> {
        for c in 0..self.components {
            if let Some(candidate) = self.extrapolated.candidate(c) {
                let total = candidate.total()?;
                if total < self.errors[c] {
                    self.values[c] = candidate.value;
                    // Epsilon replaces small-region truncation uncertainty.
                    // The larger-region correction and arithmetic floor remain.
                    self.parts[c] = [self.values[c], candidate.larger, candidate.floor, S::ZERO];
                    self.selected[c] = Some(candidate.error);
                }
            }
        }
        Ok(())
    }
}

fn empty_queue() -> SciOxideError {
    crate::kernel::invalid(
        scioxide_core::argument!("workspace"),
        scioxide_core::InvalidInputReason::Inconsistent,
        "empty regional queue",
    )
}

fn discard_stale<S: RealScalar>(queue: &mut BinaryHeap<Priority<S>>, stamps: &[usize]) {
    while queue
        .peek()
        .is_some_and(|entry| stamps[entry.slot] != entry.sequence)
    {
        queue.pop();
    }
}

/// Sums the callbacks and evaluations of prepared rules, checking overflow.
fn costs<'a, P: 'a>(
    prepared: impl IntoIterator<Item = &'a Prepared<P>>,
) -> Result<(usize, usize), SciOxideError> {
    prepared
        .into_iter()
        .try_fold((0, 0), |(calls, providers), rule| {
            Ok((
                checked_add(
                    calls,
                    rule.evaluations,
                    scioxide_core::quantity!("split_evaluations"),
                )?,
                checked_add(
                    providers,
                    rule.providers,
                    scioxide_core::quantity!("split_providers"),
                )?,
            ))
        })
}

/// Runs the adaptive loop for ordinary, weighted, oscillatory, nested and
/// multidimensional rules.
///
/// The caller places `initial` regions in the workspace and sizes it for one more
/// region per bisection. The initial regions are charged as one unit before any is
/// evaluated, and each bisection as one unit, so a stop after the initial regions
/// still leaves an estimate over the whole domain.
pub(crate) fn schedule<S: RealScalar, G: Copy, K: RegionalKernel<S, G>>(
    initial: usize,
    kernel: &mut K,
    policy: &RegionalPolicy<'_, S>,
    ws: &mut RegionalWorkspace<S, G>,
    context: &mut SolveContext<'_, '_>,
    allowance: &mut K::Allowance,
) -> Result<Scheduled, SciOxideError> {
    if initial > ws.capacity {
        crate::kernel::shape(initial, ws.capacity)?;
    }
    let value_change_threshold = S::try_from_f64_lossy(1e-5)?;
    let error_retention_threshold = S::try_from_f64_lossy(0.99)?;
    ws.live = 0;
    ws.queue.clear();
    ws.large_queue.clear();
    ws.pending_queue.clear();
    ws.level = 0;
    ws.clear_extrapolation();
    for sum in &mut ws.large_totals {
        sum.clear();
    }
    for sums in &mut ws.totals {
        for sum in sums {
            sum.clear();
        }
    }
    if let Some((offsets, _)) = policy.offsets {
        for (sums, &offset) in ws.totals.iter_mut().zip(offsets) {
            sums[0].add_value(offset)?;
        }
    }
    let mut calls = 0;
    let mut providers = 0;
    // Retain the first and last prepared rules across admission. Solves with
    // one or two initial regions then prepare every initial rule just once,
    // without allocating an array of prepared rules for longer initializations.
    let mut retained = [None, None];
    for (slot, &region) in ws.regions[..initial].iter().enumerate() {
        match kernel.prepare(region)? {
            Preparation::Ready(rule) => {
                calls = checked_add(
                    calls,
                    rule.evaluations,
                    scioxide_core::quantity!("initial_evaluations"),
                )?;
                providers = checked_add(
                    providers,
                    rule.providers,
                    scioxide_core::quantity!("initial_providers"),
                )?;
                if slot == 0 {
                    retained[0] = Some(rule);
                } else if slot + 1 == initial {
                    retained[1] = Some(rule);
                }
            }
            Preparation::Stop(stop) => return Ok(Scheduled::Unmeasured(stop)),
        }
    }
    if let Some(stop) = allowance.admit(context, calls, providers, false)? {
        return Ok(Scheduled::Unmeasured(stop));
    }
    let mut initial_live = initial;
    let mut slow = 0;
    let mut growth = 0;
    let mut refinements = 0;
    let mut sequence = initial;
    for slot in 0..initial {
        // Payloads are measured in place; only middle regions are prepared again.
        let held = match slot {
            0 => retained[0].as_ref(),
            _ if slot + 1 == initial => retained[1].as_ref(),
            _ => None,
        };
        let prepared;
        let rule = match held {
            Some(rule) => rule,
            None => match kernel.prepare(ws.regions[slot])? {
                Preparation::Ready(rule) => {
                    prepared = rule;
                    &prepared
                }
                Preparation::Stop(stop) => return Ok(Scheduled::Unmeasured(stop)),
            },
        };
        if let Some(stop) = kernel.measure(
            ws.regions[slot],
            &rule.rule,
            slot,
            &mut ws.samples,
            &mut ws.estimates[slot * ws.components..(slot + 1) * ws.components],
            context,
            allowance,
        )? {
            return Ok(Scheduled::Unmeasured(stop));
        }
        // A single complete regional estimate needs neither an exact sum of
        // several regions nor a priority queue until it actually needs a split.
        // Copying its scalar parts is exactly the one-term exact total.
        if initial == 1 && !policy.extrapolation && policy.offsets.is_none() {
            for (c, &parts) in ws.estimates[..ws.components].iter().enumerate() {
                ws.parts[c] = parts;
                ws.values[c] = parts[0];
                ws.scales[c] = parts[0].abs();
                ws.errors[c] =
                    crate::kernel::checked_intermediate(parts[1].max(parts[2]) + parts[3])?;
            }
            let completed = !policy.refine
                || kernel.convergence_evidence(&ws.parts)
                    && kernel.exploration(ws.regions[0].depth) == 0
                    && crate::kernel::diagnostic((policy.norm)(&ws.errors, &ws.scales))? <= S::ONE;
            if completed {
                ws.live = 1;
                return Ok(Scheduled::Measured {
                    termination: if policy.refine {
                        IntegrationTermination::ToleranceReached
                    } else {
                        IntegrationTermination::FixedRuleCompleted
                    },
                    subdivisions: 0,
                });
            }
        }
        if initial == 1
            && !policy.extrapolation
            && policy.offsets.is_none()
            && policy.refine
            && K::TRANSACTIONAL_SPLIT
            && kernel.exploration(ws.regions[0].depth) == 0
        {
            ws.live = 1;
            if ws.parts.iter().all(|e| e[1] <= e[2] && e[3] == S::ZERO) {
                return Ok(Scheduled::Measured {
                    termination: IntegrationTermination::Stagnation,
                    subdivisions: 0,
                });
            }
            if !allowance.can_subdivide(context)? {
                return Ok(Scheduled::Measured {
                    termination: IntegrationTermination::BudgetLimit(BudgetKind::Subdivision),
                    subdivisions: 0,
                });
            }
            if let Some(stop) = context.stop_reason()? {
                return Ok(Scheduled::Measured {
                    termination: IntegrationTermination::ExecutionStopped(stop),
                    subdivisions: 0,
                });
            }
            let children = match kernel.refine(ws.regions[0], 0, &ws.estimates[..ws.components])? {
                Refinement::Split(children) => children,
                Refinement::Stop(stop) => {
                    return Ok(Scheduled::Measured {
                        termination: stop,
                        subdivisions: 0,
                    });
                }
                Refinement::Tighten(..) => {
                    return Err(crate::kernel::invalid(
                        scioxide_core::argument!("refinement"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "transactional split kernel returned a tightening",
                    ));
                }
            };
            if ws.capacity < 2 {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("workspace"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "regional capacity is below the admitted subdivisions",
                ));
            }
            let (calls, providers) = costs(children.iter().map(|(_, prepared)| prepared))?;
            if let Some(stop) = allowance.admit(context, calls, providers, true)? {
                return Ok(Scheduled::Measured {
                    termination: stop,
                    subdivisions: 0,
                });
            }
            let first = ws.capacity;
            let second = first + 1;
            refinements = 1;
            for (child, &(region, ref prepared)) in children.iter().enumerate() {
                let destination = first + child;
                ws.regions[destination] = region;
                if let Some(stop) = kernel.measure(
                    region,
                    &prepared.rule,
                    destination,
                    &mut ws.samples,
                    &mut ws.estimates
                        [destination * ws.components..(destination + 1) * ws.components],
                    context,
                    allowance,
                )? {
                    return Ok(Scheduled::Measured {
                        termination: refinement_stop(stop),
                        subdivisions: 1,
                    });
                }
            }
            let (live, staged) = ws.estimates.split_at_mut(first * ws.components);
            let (left, right) = staged.split_at_mut(ws.components);
            for ((&parent, left), right) in live[..ws.components].iter().zip(left).zip(right) {
                kernel.refinement_estimates(parent, left, right)?;
            }
            let mut no_progress = false;
            for c in 0..ws.components {
                let old = ws.estimates[c];
                let left = ws.estimates[first * ws.components + c];
                let right = ws.estimates[second * ws.components + c];
                let value = left[0] + right[0];
                no_progress |= old[1] > old[2]
                    && (value - old[0]).abs() <= value_change_threshold * value.abs()
                    && left[1] + right[1] >= error_retention_threshold * old[1];
            }
            slow = usize::from(no_progress);
            ws.finish_pair(first, second)?;
            for (source, destination) in [(first, 0), (second, 1)] {
                ws.copy_estimate(source, destination);
                ws.regions[destination] = ws.regions[source];
                kernel.commit(source, destination);
            }
            ws.live = 2;
            let satisfied =
                crate::kernel::diagnostic((policy.norm)(&ws.errors, &ws.scales))? <= S::ONE;
            let explored = ws.regions[..2]
                .iter()
                .all(|region| kernel.exploration(region.depth) == 0);
            if explored && satisfied && kernel.convergence_evidence(&ws.parts) {
                return Ok(Scheduled::Measured {
                    termination: IntegrationTermination::ToleranceReached,
                    subdivisions: 1,
                });
            }
            if !satisfied && ws.parts.iter().all(|e| e[1] <= e[2] && e[3] == S::ZERO) {
                return Ok(Scheduled::Measured {
                    termination: IntegrationTermination::Stagnation,
                    subdivisions: 1,
                });
            }
            for child in 0..2 {
                ws.enqueue(
                    child,
                    sequence,
                    kernel.exploration(ws.regions[child].depth),
                    policy,
                )?;
                sequence += 1;
            }
            initial_live = 2;
            break;
        }
        ws.add(slot, false)?;
        ws.enqueue(
            slot,
            slot,
            kernel.exploration(ws.regions[slot].depth),
            policy,
        )?;
    }
    ws.live = initial_live;
    if policy.extrapolation {
        ws.finish(policy)?;
        ws.large_errors.copy_from_slice(&ws.errors);
        ws.extrapolated
            .observe(&ws.values, &ws.parts, &ws.large_errors)?;
    }
    let mut final_stop = IntegrationTermination::Stagnation;
    let termination = loop {
        ws.finish(policy)?;
        if policy.extrapolation {
            discard_stale(&mut ws.queue, &ws.stamps);
        }
        let explored = ws.queue.peek().is_none_or(|p| p.exploration == 0);
        if !policy.refine {
            break IntegrationTermination::FixedRuleCompleted;
        }
        let satisfied = crate::kernel::diagnostic((policy.norm)(&ws.errors, &ws.scales))? <= S::ONE;
        if explored && satisfied && kernel.convergence_evidence(&ws.parts) {
            break IntegrationTermination::ToleranceReached;
        }
        if ws.queue.is_empty() {
            break final_stop;
        }
        if policy.extrapolation {
            discard_stale(&mut ws.large_queue, &ws.stamps);
            let small = ws
                .queue
                .peek()
                .is_some_and(|entry| ws.regions[entry.slot].depth > ws.level);
            if small {
                for (error, sum) in ws.large_errors.iter_mut().zip(&mut ws.large_totals) {
                    *error = sum.finish()?.max(S::ZERO);
                }
                let large_satisfied = (policy.norm)(&ws.large_errors, &ws.scales)? <= S::ONE;
                if large_satisfied || ws.large_queue.is_empty() {
                    if ws
                        .extrapolated
                        .observe(&ws.values, &ws.parts, &ws.large_errors)?
                    {
                        ws.extrapolated.reset();
                        break IntegrationTermination::ProbablyDivergent;
                    }
                    if ws.extrapolated.stagnated(&ws.errors)? {
                        break IntegrationTermination::Stagnation;
                    }
                    if ws
                        .candidate_norm(policy)?
                        .is_some_and(|norm| norm <= S::ONE)
                    {
                        break IntegrationTermination::ToleranceReached;
                    }
                    // Restore the direct error and scale after comparing candidates.
                    ws.finish(policy)?;
                    ws.advance_level()?;
                }
            }
        }
        // Additional moment/inner uncertainty must not be mistaken for a roundoff floor.
        if (K::SLOW_PROGRESS && explored && (slow >= 6 || growth >= 20))
            || (!satisfied && ws.parts.iter().all(|e| e[1] <= e[2] && e[3] == S::ZERO))
        {
            break IntegrationTermination::Stagnation;
        }
        if !allowance.can_subdivide(context)? {
            break IntegrationTermination::BudgetLimit(BudgetKind::Subdivision);
        }
        if let Some(stop) = context.stop_reason()? {
            break IntegrationTermination::ExecutionStopped(stop);
        }
        let mut parent = ws
            .queue
            .peek()
            .ok_or_else(|| {
                crate::kernel::invalid(
                    scioxide_core::argument!("workspace"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "empty regional queue",
                )
            })?
            .slot;
        if policy.extrapolation && ws.regions[parent].depth > ws.level {
            discard_stale(&mut ws.large_queue, &ws.stamps);
            if let Some(entry) = ws.large_queue.peek() {
                parent = entry.slot;
            }
        }
        let first = ws.capacity;
        let second = first + 1;
        let refinement = kernel.refine(
            ws.regions[parent],
            parent,
            &ws.estimates[parent * ws.components..(parent + 1) * ws.components],
        )?;
        // Children are measured in place from the refinement that prepared them.
        let (children, count) = match &refinement {
            Refinement::Split([(left, left_rule), (right, right_rule)]) => {
                ([Some((*left, left_rule)), Some((*right, right_rule))], 2)
            }
            Refinement::Tighten(region, prepared) => ([Some((*region, prepared)), None], 1),
            &Refinement::Stop(
                stop @ (IntegrationTermination::RefinementLimit
                | IntegrationTermination::Stagnation),
            ) => {
                // Final regions retain their contribution and uncertainty, while
                // the remaining heap continues to refine. Blocked uncertainty
                // remains a large-region error at every extrapolation level.
                if stop == IntegrationTermination::RefinementLimit {
                    final_stop = stop;
                }
                if policy.extrapolation {
                    if ws.regions[parent].depth > ws.level {
                        ws.update_large(parent, false)?;
                    }
                    ws.stamps[parent] = usize::MAX;
                    ws.clear_extrapolation();
                } else {
                    ws.queue.pop();
                }
                continue;
            }
            &Refinement::Stop(stop) => break stop,
        };
        if count == 2 && ws.live >= ws.capacity {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "regional capacity is below the admitted subdivisions",
            ));
        }
        let (calls, providers) = costs(children.iter().flatten().map(|&(_, prepared)| prepared))?;
        if let Some(stop) = allowance.admit(context, calls, providers, true)? {
            break stop;
        }
        refinements += 1;
        let mut stopped = None;
        for (child, (region, prepared)) in children.into_iter().flatten().enumerate() {
            let slot = first + child;
            ws.regions[slot] = region;
            stopped = kernel.measure(
                region,
                &prepared.rule,
                slot,
                &mut ws.samples,
                &mut ws.estimates[slot * ws.components..(slot + 1) * ws.components],
                context,
                allowance,
            )?;
            if stopped.is_some() {
                break;
            }
        }
        if let Some(stop) = stopped {
            break refinement_stop(stop);
        }
        if count == 2 {
            let (live, staged) = ws.estimates.split_at_mut(first * ws.components);
            let (left, right) = staged.split_at_mut(ws.components);
            for ((&old, left), right) in live[parent * ws.components..(parent + 1) * ws.components]
                .iter()
                .zip(left)
                .zip(right)
            {
                kernel.refinement_estimates(old, left, right)?;
            }
        }
        let mut no_progress = false;
        let mut increased = false;
        for c in 0..ws.components {
            let old = ws.estimates[parent * ws.components + c];
            let left = ws.estimates[first * ws.components + c];
            let right = if count == 2 {
                ws.estimates[second * ws.components + c]
            } else {
                [S::ZERO; 4]
            };
            let value = left[0] + right[0];
            let error = left[1] + right[1];
            no_progress |= old[1] > old[2]
                && (value - old[0]).abs() <= value_change_threshold * value.abs()
                && error >= error_retention_threshold * old[1];
            increased |= error > old[1];
        }
        slow += usize::from(explored && no_progress && count == 2);
        growth += usize::from(explored && increased && count == 2 && refinements > 10);
        if policy.extrapolation && ws.regions[parent].depth <= ws.level {
            ws.update_large(parent, true)?;
        }
        ws.add(parent, true)?;
        ws.add(first, false)?;
        if count == 2 {
            ws.add(second, false)?;
        }
        if policy.extrapolation {
            // Invalidate the parent in every heap before reusing its slot.
            ws.stamps[parent] = usize::MAX;
        } else {
            ws.queue.pop();
        }
        for (source, destination) in [(first, parent), (second, ws.live)].into_iter().take(count) {
            ws.copy_estimate(source, destination);
            ws.regions[destination] = ws.regions[source];
            kernel.commit(source, destination);
            ws.enqueue(
                destination,
                sequence,
                kernel.exploration(ws.regions[destination].depth),
                policy,
            )?;
            sequence += 1;
        }
        if count == 2 {
            ws.live += 1;
        }
    };
    // candidate_norm uses error/scale scratch, so recover direct totals before
    // choosing the value and its corresponding diagnostics for publication.
    if policy.extrapolation {
        ws.finish(policy)?;
        let direct = (policy.norm)(&ws.errors, &ws.scales)?;
        let accelerated = ws.candidate_norm(policy)?;
        ws.finish(policy)?;
        // A smaller absolute error can still be worse under relative tolerance
        // when extrapolation changes the result's scale, especially near zero.
        if accelerated.is_some_and(|norm| norm <= direct) {
            ws.select_candidates()?;
        }
    }
    Ok(Scheduled::Measured {
        termination,
        subdivisions: refinements,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn scalar_priorities_reduce_absolute_error_and_vectors_keep_normalization() {
        fn check<S: RealScalar>() {
            let tolerance = scioxide_core::ErrorTolerance::scalar(
                S::ONE,
                S::ONE,
                scioxide_core::NormSelection::Infinity,
            )
            .unwrap();
            let policy = RegionalPolicy {
                extrapolation: false,
                refine: true,
                offsets: None,
                norm: &|errors, scales| tolerance.normalized_norm(errors, scales),
            };
            for components in [1, 2] {
                let mut workspace = RegionalWorkspace::<S>::new(components, 2, 15).unwrap();
                for c in 0..components {
                    workspace.estimates[c] = [
                        S::try_from_usize_exact(1000).unwrap(),
                        S::TWO,
                        S::ZERO,
                        S::ZERO,
                    ];
                    workspace.estimates[components + c] = [S::ONE, S::ONE, S::ZERO, S::ZERO];
                }
                workspace.enqueue(0, 0, 0, &policy).unwrap();
                workspace.enqueue(1, 1, 0, &policy).unwrap();
                assert_eq!(
                    workspace.queue.pop().unwrap().slot,
                    usize::from(components > 1)
                );
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
