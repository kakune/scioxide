//! Nested transformed-trapezoid refinement with heuristic tail shells.

use crate::LevelOptions;

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    AffineValue, AffineValueMut, ErrorTolerance, ExecutionPolicy, PartialResultTrust, RealScalar,
    SciOxideError, SolveContext, write_value,
};

use super::transform::{Geometry, PairPreparation, PreparedMap, WeightedPoint};
use super::{DoubleExponentialDomain, DoubleExponentialPoint};
use crate::evaluation::{
    contract_scalar_rows, sample_rows, sample_scalar, scalar_value, write_row,
};
use crate::kernel::{checked_intermediate, roundoff_floor};
use crate::report::{
    ErrorEstimate, ErrorReliability, IntegrationErrorComponents, IntegrationTermination,
    TailDiagnostics, TailStatus,
};
use crate::workspace::{Outcome, admit};
use scioxide_core::PromotingCompensatedSum;

/// Largest supported number of refinement levels. Each level halves the step size.
pub(super) const MAX_LEVELS: usize = 15;
/// Consecutive decreasing or zero shells required before heuristic truncation.
const REQUIRED_TAIL_SHELLS: usize = 3;
// All three transforms saturate before |t| = 8 in either supported IEEE width.
pub(super) const MAX_TAIL_SHELLS: usize = 8;
/// Three successive shrinking differences, each reduced less than fourfold, indicate
/// unresolved features rather than the rapidly accelerating convergence of the DE rule.
const MAX_SLOW_DIFFERENCES: usize = 3;
/// Three refinements protect against premature agreement on a coarse grid.
pub(super) const MIN_CONVERGENCE_LEVEL: usize = 3;

/// Tail-shell test: whether the last shells decrease, and the estimated
/// remainder. Absolute masses prevent cancellation between the two tail sides.
// Shell k holds the current mesh h times the absolute weighted samples with |t| in
// (k, k+1]; it is halved whenever the mesh halves, so it never grows like 2^level.
// Use the largest of the last two decay ratios as a heuristic geometric envelope.
// Three all-zero shells alone are insufficient convergence evidence in `estimate`.
fn tail_shells<S: RealScalar>(masses: &[S], limit: S) -> (bool, S) {
    let last = masses.last().copied().unwrap_or(S::ZERO);
    if masses.len() < REQUIRED_TAIL_SHELLS {
        return (false, last);
    }
    let recent = &masses[masses.len() - REQUIRED_TAIL_SHELLS..];
    let mut ratio = S::ZERO;
    for pair in recent.windows(2) {
        if pair[1] > pair[0] || (pair[0] != S::ZERO && pair[1] == pair[0]) {
            return (false, last);
        }
        if pair[0] > S::ZERO {
            ratio = ratio.max(pair[1] / pair[0]);
        }
    }
    // Include the entire last observed shell as well as its geometric continuation.
    // This is deliberately more conservative than last * ratio / (1 - ratio).
    let remainder = saturating(last / (S::ONE - ratio));
    (remainder <= limit, remainder)
}

/// `h * sum` on the dyadic mesh `h = 2^-level`. Raw nested sums grow like
/// `2^level` times the integral, so a promoted or overflowing sum is scaled
/// exactly by `h` before its single rounding; ordinary sums keep the native
/// product, which is exact for normal results.
#[inline]
fn mesh_total<S: RealScalar>(
    sum: &PromotingCompensatedSum<S>,
    h: S,
    level: usize,
) -> Result<S, SciOxideError> {
    if sum.native().is_some()
        && let Ok(total) = sum.round()
    {
        let value = total * h;
        if value.is_finite() && (total == S::ZERO || value.abs() >= S::MIN_POSITIVE) {
            return Ok(value);
        }
    }
    // Levels never exceed `MAX_LEVELS`, so the shift is exact and small.
    sum.total()?.shifted(-(level as i64))?.round()
}

/// Nonnegative error diagnostics saturate at the scalar maximum.
#[inline]
fn saturating<S: RealScalar>(value: S) -> S {
    if value.is_finite() { value } else { S::MAX }
}

/// Whether a nonnegative absolute sum holds any positive mass, without rounding
/// a promoted total into the scalar range.
#[inline]
fn positive<S: RealScalar>(sum: &PromotingCompensatedSum<S>) -> Result<bool, SciOxideError> {
    match sum.native() {
        Some(native) => Ok(native.total()? > S::ZERO),
        None => Ok(!sum.total()?.is_zero()),
    }
}

/// Shared arithmetic floor of an absolute sum whose total may exceed the scalar range.
#[inline]
fn sum_floor<S: RealScalar>(sum: &PromotingCompensatedSum<S>) -> Result<S, SciOxideError> {
    if sum.native().is_some()
        && let Ok(total) = sum.round()
    {
        return roundoff_floor(total);
    }
    crate::kernel::scaled_roundoff_floor(sum.total()?)
}

fn shell_envelope_supported<S: RealScalar>(masses: &[S]) -> bool {
    masses.len() >= REQUIRED_TAIL_SHELLS
        && masses[masses.len() - REQUIRED_TAIL_SHELLS..]
            .windows(2)
            .all(|pair| pair[1] < pair[0] || pair[0] == S::ZERO)
}

/// Compact local address within the at-most-eight shells and two tail sides.
#[derive(Debug, Clone, Copy)]
struct ShellTarget {
    shell: u8,
    side: u8,
    boundary: bool,
    witness: bool,
}
impl ShellTarget {
    fn new(shell: usize, side: usize, boundary: bool, witness: bool) -> Self {
        Self {
            shell: shell as u8,
            side: side as u8,
            boundary,
            witness,
        }
    }
}

/// Where one staged sample contributes: its transform weight, segment and tail shell.
#[derive(Debug, Clone, Copy)]
struct Target<S: RealScalar> {
    weight: S,
    segment: usize,
    shell: Option<ShellTarget>,
}

/// Raw nested sums of every segment and component, tail shells and one block of
/// staged samples. The sums carry every previous level without storing samples.
#[derive(Debug)]
pub(super) struct State<S: RealScalar> {
    components: usize,
    segments: usize,
    segment_divisor: S,

    raw: Vec<PromotingCompensatedSum<S>>,
    absolute: Vec<PromotingCompensatedSum<S>>,
    shells: Vec<[S; MAX_TAIL_SHELLS]>,
    extents: Vec<usize>,
    complete_shells: Vec<usize>,
    saturated: Vec<bool>,
    stagnated: Vec<bool>,
    last_difference: Vec<S>,
    slow: Vec<usize>,
    values: Vec<S>,
    previous: Vec<S>,
    errors: Vec<S>,
    scales: Vec<S>,
    thresholds: Vec<S>,
    extensions: Vec<(bool, bool)>,
    cuts: Vec<[usize; 2]>,
    sampled: Vec<[[usize; MAX_TAIL_SHELLS]; 2]>,
    omitted: Vec<[S; 2]>,
    witnesses: Vec<[S; 2]>,
    edges: Vec<[[S; MAX_TAIL_SHELLS]; 2]>,
    outer_samples: Vec<[[S; MAX_TAIL_SHELLS]; 2]>,
    pub(super) diagnostics: Vec<IntegrationErrorComponents<S>>,
    batch: Vec<DoubleExponentialPoint<S>>,
    targets: Vec<Target<S>>,
    rows: Vec<S>,
    pending: usize,
    /// Current transformed mesh; shell masses are stored times this mesh.
    mesh: S,
}

impl<S: RealScalar> State<S> {
    pub(super) fn allocate(
        components: usize,
        segments: usize,
        block: usize,
    ) -> Result<Self, SciOxideError> {
        let n = checked_mul(
            components,
            segments,
            scioxide_core::quantity!("segment_components"),
        )?;
        let filled = |quantity| try_filled(components, S::ZERO, quantity);
        Ok(Self {
            components,
            segments,
            segment_divisor: S::ONE,
            raw: try_filled(
                n,
                PromotingCompensatedSum::ZERO,
                scioxide_core::quantity!("raw_sums"),
            )?,
            absolute: try_filled(
                n,
                PromotingCompensatedSum::ZERO,
                scioxide_core::quantity!("absolute_sums"),
            )?,
            shells: try_filled(
                checked_mul(n, 2, scioxide_core::quantity!("side_shells"))?,
                [S::ZERO; MAX_TAIL_SHELLS],
                scioxide_core::quantity!("tail_shells"),
            )?,
            extents: try_filled(segments, 0, scioxide_core::quantity!("segment_extents"))?,
            complete_shells: try_filled(segments, 0, scioxide_core::quantity!("complete_shells"))?,
            saturated: try_filled(
                segments,
                false,
                scioxide_core::quantity!("saturated_segments"),
            )?,
            stagnated: try_filled(
                segments,
                false,
                scioxide_core::quantity!("stagnated_segments"),
            )?,
            last_difference: try_filled(
                components,
                S::INFINITY,
                scioxide_core::quantity!("last_differences"),
            )?,
            slow: try_filled(components, 0, scioxide_core::quantity!("slow_counts"))?,
            values: filled(scioxide_core::quantity!("component_values"))?,
            previous: filled(scioxide_core::quantity!("previous_values"))?,
            errors: filled(scioxide_core::quantity!("component_errors"))?,
            scales: filled(scioxide_core::quantity!("component_scales"))?,
            thresholds: try_filled(n, S::ZERO, scioxide_core::quantity!("tail_thresholds"))?,
            extensions: try_filled(
                segments,
                (false, false),
                scioxide_core::quantity!("tail_decisions"),
            )?,
            cuts: try_filled(
                segments,
                [MAX_TAIL_SHELLS; 2],
                scioxide_core::quantity!("tail_cuts"),
            )?,
            sampled: try_filled(
                segments,
                [[0; MAX_TAIL_SHELLS]; 2],
                scioxide_core::quantity!("sampled_levels"),
            )?,
            omitted: try_filled(n, [S::ZERO; 2], scioxide_core::quantity!("omitted_tails"))?,
            witnesses: try_filled(n, [S::ZERO; 2], scioxide_core::quantity!("tail_witnesses"))?,
            edges: try_filled(
                n,
                [[S::ZERO; MAX_TAIL_SHELLS]; 2],
                scioxide_core::quantity!("tail_edges"),
            )?,
            outer_samples: try_filled(
                n,
                [[S::ZERO; MAX_TAIL_SHELLS]; 2],
                scioxide_core::quantity!("tail_outer_samples"),
            )?,
            diagnostics: try_filled(
                components,
                IntegrationErrorComponents::default(),
                scioxide_core::quantity!("component_diagnostics"),
            )?,
            batch: try_filled(
                block,
                DoubleExponentialPoint {
                    coordinate: S::ZERO,
                    endpoint: None,
                },
                scioxide_core::quantity!("staged_points"),
            )?,
            targets: try_filled(
                block,
                Target {
                    weight: S::ZERO,
                    segment: 0,
                    shell: None,
                },
                scioxide_core::quantity!("staged_targets"),
            )?,
            rows: try_filled(
                checked_mul(block, components, scioxide_core::quantity!("staged_rows"))?,
                S::ZERO,
                scioxide_core::quantity!("staged_rows"),
            )?,
            pending: 0,
            mesh: S::ONE,
        })
    }

    /// Validates `domain` and stages its sorted distinct breakpoints before the
    /// call starts. Repeated singular points or ones on an endpoint add no segment.
    pub(super) fn stage(
        &mut self,
        domain: DoubleExponentialDomain<'_, S>,
    ) -> Result<(), SciOxideError> {
        let requested = match domain {
            DoubleExponentialDomain::Finite {
                singular_points, ..
            } => singular_points.len() + 1,
            _ => 1,
        };
        if requested > self.segments {
            crate::kernel::shape(requested, self.segments)?;
        }
        Ok(())
    }

    /// Stages one sample, evaluating a full block first.
    fn push<F, V>(
        &mut self,
        function: &F,
        point: WeightedPoint<S>,
        segment: usize,
        shell: Option<ShellTarget>,
        policy: ExecutionPolicy,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: Fn(DoubleExponentialPoint<S>) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        if self.pending == self.batch.len() {
            self.flush(function, policy, context)?;
        }
        self.batch[self.pending] = DoubleExponentialPoint {
            coordinate: point.point,
            endpoint: point.endpoint,
        };
        self.targets[self.pending] = Target {
            weight: point.weight,
            segment,
            shell,
        };
        self.pending += 1;
        Ok(())
    }

    /// Evaluates the staged samples and adds them, in staging order, to the raw sums.
    // Sums are stored without the mesh factor h: halving h adds new points to the same
    // raw sum, and `h * raw` is the trapezoidal value of the current level. Samples are
    // therefore never stored, only the running raw, L1 (roundoff floor) and per-shell sums.
    fn flush<F, V>(
        &mut self,
        function: &F,
        policy: ExecutionPolicy,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: Fn(DoubleExponentialPoint<S>) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        let count = core::mem::take(&mut self.pending);
        if count == 0 {
            return Ok(());
        }
        let d = self.components;
        let rows = &mut self.rows[..count * d];
        if d == 1 && policy == ExecutionPolicy::Sequential {
            sample_scalar::<S, _, 0, false>(
                &self.batch[..count],
                rows,
                context,
                |point| scalar_value(&function(*point)?),
                |_| S::ONE,
            )?;
        } else {
            sample_rows(count, d, rows, policy, context, &|index, row| {
                write_row(&function(self.batch[index])?, row)
            })?;
        }
        let mut begin = 0;
        while begin < count {
            let segment = self.targets[begin].segment;
            let end = self.targets[begin..count]
                .iter()
                .position(|t| t.segment != segment)
                .map_or(count, |offset| begin + offset);
            for c in 0..d {
                let mut column = [S::ZERO; crate::evaluation::POINT_BLOCK];
                let values = if d == 1 {
                    &rows[begin..end]
                } else {
                    for (slot, row) in column
                        .iter_mut()
                        .zip(rows[begin * d..end * d].chunks_exact(d))
                    {
                        *slot = row[c];
                    }
                    &column[..end - begin]
                };
                let [raw, absolute] = contract_scalar_rows::<S, _, 2, false, _>(
                    &self.targets[begin..end],
                    values,
                    |target| target.weight,
                    |sums| {
                        let mut result = [PromotingCompensatedSum::ZERO; 2];
                        for (slot, value) in result.iter_mut().zip(sums) {
                            *slot = PromotingCompensatedSum::from_compensated(
                                scioxide_core::CompensatedSum::from_parts(value, S::ZERO)?,
                            );
                        }
                        Ok(result)
                    },
                    Ok,
                )?;
                self.raw[segment * d + c].merge(raw)?;
                self.absolute[segment * d + c].merge(absolute)?;
            }
            begin = end;
        }
        for (row, target) in rows.chunks_exact(d).zip(&self.targets[..count]) {
            for (c, &value) in row.iter().enumerate() {
                let index = target.segment * d + c;
                if let Some(location) = target.shell {
                    // Keep a raw contribution through dyadic scaling before
                    // rounding; edge and shell diagnostics saturate on overflow.
                    let weighted = value.abs() * target.weight;
                    let mass = weighted * self.mesh;
                    let (weighted, mass) = if value == S::ZERO
                        || target.weight == S::ZERO
                        || weighted.is_normal() && mass.is_normal()
                    {
                        (weighted, mass)
                    } else {
                        let product = scioxide_core::Scaled::from_value(value.abs())?
                            .mul(scioxide_core::Scaled::from_value(target.weight)?)?;
                        (
                            crate::kernel::diagnostic(product.round())?,
                            crate::kernel::diagnostic(
                                product
                                    .mul(scioxide_core::Scaled::from_value(self.mesh)?)?
                                    .round(),
                            )?,
                        )
                    };
                    let shell = usize::from(location.shell);
                    let side = usize::from(location.side);
                    if location.boundary {
                        self.edges[index][side][shell] = weighted;
                    }
                    if location.witness {
                        self.outer_samples[index][side][shell] =
                            self.outer_samples[index][side][shell].max(weighted);
                    }
                    self.shells[2 * index + side][shell] =
                        saturating(self.shells[2 * index + side][shell] + mass);
                }
            }
        }
        Ok(())
    }

    /// Integrates over the staged domain inside the solve context and writes the
    /// estimate to `out`.
    pub(super) fn run<F, V, O>(
        &mut self,
        plan: &super::DoubleExponentialIntegrator<S>,
        function: &F,
        out: &mut O,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Outcome<S>, SciOxideError>
    where
        F: Fn(DoubleExponentialPoint<S>) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let options = &plan.options;
        let basis = &plan.basis;
        let geometry = plan.geometry.as_ref();
        let domain = plan.domain();
        let degenerate = matches!(
            domain,
            DoubleExponentialDomain::Finite { interval, .. } if interval.lower() == interval.upper()
        );
        if self.components == 0 || degenerate {
            return crate::execution::zero(
                &mut self.values,
                out,
                if degenerate {
                    IntegrationTermination::DegenerateDomain
                } else {
                    IntegrationTermination::FixedRuleCompleted
                },
                context,
            );
        }
        let maps = match geometry {
            Geometry::Ready(maps) => maps.as_slice(),
            Geometry::Unavailable => {
                return Ok(Outcome::Unpublished(
                    IntegrationTermination::GeometryFailure,
                ));
            }
            Geometry::Failed(error) => return Err(error.clone()),
        };
        let termination = match self.refine(options, basis, maps, function, context)? {
            Ok(termination) => termination,
            Err(stop) => return Ok(Outcome::Unpublished(stop)),
        };
        context.commit_accepted(PartialResultTrust::Admissible)?;
        write_value(&self.values, out)?;
        Ok(Outcome::Published {
            termination,
            subdivisions: 0,
            errors: IntegrationErrorComponents::envelope(&self.diagnostics),
        })
    }

    fn segment_threshold(
        &self,
        tolerance: &ErrorTolerance<S>,
        component: usize,
        current: S,
    ) -> Result<S, SciOxideError> {
        Ok(tolerance.threshold(component, self.components, current.abs())? / self.segment_divisor)
    }

    /// Recomputes every component estimate at mesh `h`; `true` when every segment's
    /// tail is truncated, some mass was observed and every guarded change is at
    /// most a quarter of its component's sampled absolute mass.
    fn estimate(
        &mut self,
        h: S,
        level: usize,
        tolerance: &ErrorTolerance<S>,
        segments: usize,
    ) -> Result<bool, SciOxideError> {
        let d = self.components;
        self.extensions.fill((false, false));
        let mut all_truncated = true;
        let mut observed_mass = false;
        let mut resolved = true;
        for c in 0..d {
            let mut value = PromotingCompensatedSum::ZERO;
            let mut l1 = PromotingCompensatedSum::ZERO;
            let mut tail = PromotingCompensatedSum::ZERO;
            let mut status = TailStatus::HeuristicallyTruncated;
            let mut extent = 0;
            for segment in 0..segments {
                let index = segment * d + c;
                let current = mesh_total(&self.raw[index], h, level)?;
                value.add_value(current)?;
                let sampled = positive(&self.absolute[index])?;
                self.extensions[segment].0 |= sampled;
                observed_mass |= sampled;
                l1.add_scaled(self.absolute[index].total()?.shifted(-(level as i64))?)?;
                // Each segment gets an equal share of the tolerance so that the summed tail
                // estimates of all segments stay within the component's threshold.
                let threshold = self.segment_threshold(tolerance, c, current)?;
                self.thresholds[index] = threshold;
                let mut truncated = true;
                for side in 0..2 {
                    let masses = &self.shells[2 * index + side][..self.complete_shells[segment]];
                    let frozen = self.cuts[segment][side] < MAX_TAIL_SHELLS;
                    let (accepted, mut error) = if frozen {
                        let error = self.omitted[index][side];
                        (error <= threshold * S::HALF, error)
                    } else {
                        tail_shells(masses, threshold * S::HALF)
                    };
                    if !frozen {
                        self.extensions[segment].1 |= !accepted;
                    }
                    if !frozen && self.saturated[segment] && !shell_envelope_supported(masses) {
                        // Unknown endpoint mass cannot be inferred from an empty
                        // or growing shell sequence. Keep a finite, uninformative
                        // error with headroom for the other additive diagnostics.
                        error = S::MAX / (S::try_from_usize_exact(segments)? * S::TWO * S::TWO);
                    }
                    truncated &= accepted;
                    tail.add_value(error)?;
                }
                all_truncated &= !self.stagnated[segment] && truncated;
                extent = extent.max(self.extents[segment]);
                // Status precedence across segments: Stagnated > Saturated > Open > Truncated,
                // so the combined status reports the least trustworthy segment.
                if self.stagnated[segment] {
                    status = TailStatus::Stagnated;
                } else if self.saturated[segment] && status != TailStatus::Stagnated {
                    status = TailStatus::Saturated;
                } else if !truncated
                    && status != TailStatus::Saturated
                    && status != TailStatus::Stagnated
                {
                    status = TailStatus::Open;
                }
            }
            let current = value.round()?;
            let floor = sum_floor(&l1)?;
            let tail = tail.round()?;
            let difference = (current - self.previous[c]).abs();
            // A kink can make successive differences accidentally small before
            // asymptotic DE convergence. Guard the full change, as in the other
            // refinement families; this remains an empirical estimate. Changes
            // of results near the scalar maximum saturate rather than fail.
            let guard = S::try_from_usize_exact(crate::kernel::EXTRAPOLATION_ERROR_GUARD)?;
            let guarded = saturating(guard * difference);
            // An absolute tolerance alone can accept the negligible sampled
            // flank of a feature narrower than the transformed mesh. As in
            // Romberg, the guarded change must also resolve the sampled mass.
            resolved &= l1.round().map_or(true, |mass| guarded <= mass / guard);
            self.values[c] = current;
            self.scales[c] = current.abs();
            self.errors[c] = saturating(guarded.max(floor) + tail);
            self.diagnostics[c] = IntegrationErrorComponents {
                discretization: (level > 0).then_some(ErrorEstimate {
                    magnitude: guarded,
                    reliability: ErrorReliability::Estimated,
                }),
                roundoff: Some(floor),
                inner: None,
                moments: None,
                extrapolation: None,
                tail: (status != TailStatus::Stagnated
                    && self.extents[..segments].iter().all(|&n| n > 0))
                .then_some(ErrorEstimate {
                    magnitude: tail,
                    reliability: ErrorReliability::Heuristic,
                }),
                tail_diagnostics: Some(TailDiagnostics {
                    extent: S::try_from_usize_exact(extent)?,
                    shells: extent,
                    status,
                }),
            };
        }
        Ok(all_truncated && observed_mass && resolved)
    }

    // Tail decisions are per segment: one unresolved segment must not extend every
    // already-resolved segment of a singular integral.
    fn needs_extension(&self, segment: usize) -> bool {
        let (observed, unresolved) = self.extensions[segment];
        observed && unresolved
    }

    /// Freezes fully observed tails with a sixteenfold tolerance margin, keeping
    /// their remainder fixed when the mesh shrinks. Each component's adjacent
    /// inner shell is checked to reopen an underestimated tail.
    fn freeze_tails(&mut self, level: usize, maps: &[PreparedMap<S>]) -> Result<(), SciOxideError> {
        let d = self.components;
        let margin = S::from_f64_ieee(1.0 / 16.0);
        for (segment, map) in maps.iter().enumerate() {
            if self.cuts[segment] != [MAX_TAIL_SHELLS; 2]
                || self.saturated[segment]
                || self.stagnated[segment]
            {
                continue;
            }
            let mut observed = false;
            for sum in &self.absolute[segment * d..(segment + 1) * d] {
                observed |= positive(sum)?;
            }
            if !observed {
                continue;
            }
            let mut cuts = [MAX_TAIL_SHELLS; 2];
            for (side, cut) in cuts.iter_mut().enumerate() {
                for shells in REQUIRED_TAIL_SHELLS..=self.complete_shells[segment] {
                    if (0..shells).any(|shell| map.admission(level, shell, false).partial) {
                        break;
                    }
                    let mut accepted = true;
                    for c in 0..d {
                        let index = segment * d + c;
                        let (supported, remainder) = tail_shells(
                            &self.shells[2 * index + side][..shells],
                            self.thresholds[index] * margin,
                        );
                        // Overflow of a proposed pruning bound rejects pruning;
                        // it is not an overflow of the integral or its samples.
                        let edge_bound =
                            self.edges[index][side][shells - 2] * S::from_f64_ieee(4.0);
                        accepted &= supported
                            && edge_bound.is_finite()
                            && remainder.max(edge_bound) <= self.thresholds[index] * margin;
                    }
                    if accepted {
                        *cut = shells - 1;
                        break;
                    }
                }
            }
            if cuts.contains(&MAX_TAIL_SHELLS) {
                continue;
            }
            self.cuts[segment] = cuts;
            for c in 0..d {
                let index = segment * d + c;
                for (side, &cut) in cuts.iter().enumerate() {
                    self.omitted[index][side] = tail_shells(
                        &self.shells[2 * index + side][..=cut],
                        self.thresholds[index] * margin,
                    )
                    .1;
                    self.omitted[index][side] =
                        self.omitted[index][side].max(checked_intermediate(
                            self.edges[index][side][cut - 1] * S::from_f64_ieee(4.0),
                        )?);
                    self.witnesses[index][side] = self.edges[index][side][cut - 1];
                }
            }
        }
        Ok(())
    }

    fn thaw_tails(&mut self, segments: usize) -> Result<bool, SciOxideError> {
        let mut thawed = false;
        for segment in 0..segments {
            if self.cuts[segment] == [MAX_TAIL_SHELLS; 2] {
                continue;
            }
            let mut reopen = false;
            for c in 0..self.components {
                let index = segment * self.components + c;
                for side in 0..2 {
                    let cut = self.cuts[segment][side];
                    let witness = self.outer_samples[index][side][cut - 1];
                    reopen |= (witness * S::from_f64_ieee(0.25) > self.witnesses[index][side]
                        && witness > self.thresholds[index] * S::from_f64_ieee(1.0 / 16.0))
                        || self.omitted[index][side] > self.thresholds[index] * S::HALF;
                }
            }
            if reopen {
                self.cuts[segment] = [MAX_TAIL_SHELLS; 2];
                thawed = true;
            }
        }
        Ok(thawed)
    }

    /// Clears the tail estimates and marks the tails open, when a reopened tail
    /// cannot be sampled.
    fn unresolved_tail(&mut self) {
        for component in &mut self.diagnostics {
            component.tail = None;
            if let Some(tail) = &mut component.tail_diagnostics {
                tail.status = TailStatus::Open;
            }
        }
    }

    fn strides(&self, segment: usize, shell: usize, level: usize) -> [Option<usize>; 2] {
        core::array::from_fn(|side| {
            (shell < self.cuts[segment][side] && self.sampled[segment][side][shell] < level)
                .then(|| 1usize << (level - self.sampled[segment][side][shell]))
        })
    }

    /// Only reopened tails need per-index checks, to exclude older samples.
    fn refinement_cost(
        &self,
        maps: &[PreparedMap<S>],
        level: usize,
    ) -> Result<Option<usize>, SciOxideError> {
        let count = 1usize << level;
        let mut cost = 0;
        for (segment, map) in maps.iter().enumerate() {
            for shell in 0..self.extents[segment] {
                let strides = self.strides(segment, shell, level);
                if strides.iter().all(Option::is_none) {
                    continue;
                }
                if strides.iter().flatten().all(|&stride| stride == 2) {
                    let admission = map.admission(level, shell, true);
                    if admission.failed {
                        return Ok(None);
                    }
                    cost = checked_add(
                        cost,
                        admission.evaluations / 2 * strides.iter().flatten().count(),
                        scioxide_core::quantity!("level_evaluations"),
                    )?;
                } else {
                    for j in 1..=count {
                        let sides = strides
                            .iter()
                            .filter(|s| s.is_some_and(|stride| j % stride != 0))
                            .count();
                        if sides == 0 {
                            continue;
                        }
                        let Some(evaluations) = map.evaluations(level, shell * count + j) else {
                            return Ok(None);
                        };
                        cost = checked_add(
                            cost,
                            evaluations / 2 * sides,
                            scioxide_core::quantity!("level_evaluations"),
                        )?;
                    }
                }
            }
        }
        Ok(Some(cost))
    }

    fn sample_refinement<F, V>(
        &mut self,
        maps: &[PreparedMap<S>],
        basis: &super::transform::BasisTable<S>,
        level: usize,
        function: &F,
        policy: ExecutionPolicy,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: Fn(DoubleExponentialPoint<S>) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        let count = 1usize << level;
        for (segment, map) in maps.iter().enumerate() {
            for shell in 0..self.extents[segment] {
                let strides = self.strides(segment, shell, level);
                if strides.iter().all(Option::is_none) {
                    continue;
                }
                let mut sample = |j: usize, sides: [bool; 2]| -> Result<(), SciOxideError> {
                    let PairPreparation::Complete(pair) = map.pair(level, shell * count + j, basis)
                    else {
                        return Err(crate::kernel::invalid(
                            scioxide_core::argument!("workspace"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "prepared pair changed",
                        ));
                    };
                    for (side, point) in [pair.negative, pair.positive].into_iter().enumerate() {
                        if sides[side]
                            && let Some(point) = point
                        {
                            self.push(
                                function,
                                point,
                                segment,
                                Some(ShellTarget::new(shell, side, false, j + 1 == count)),
                                policy,
                                context,
                            )?;
                        }
                    }
                    Ok(())
                };
                if strides.iter().flatten().all(|&stride| stride == 2) {
                    let sides = strides.map(|stride| stride.is_some());
                    for j in (1..=count).step_by(2) {
                        sample(j, sides)?;
                    }
                } else {
                    for j in 1..=count {
                        let sides = strides.map(|s| s.is_some_and(|stride| j % stride != 0));
                        if sides[0] || sides[1] {
                            sample(j, sides)?;
                        }
                    }
                }
            }
        }
        self.flush(function, policy, context)?;
        for segment in 0..maps.len() {
            for side in 0..2 {
                for shell in 0..self.extents[segment].min(self.cuts[segment][side]) {
                    self.sampled[segment][side][shell] = level;
                }
            }
        }
        Ok(())
    }

    /// The nested refinement: `Err(stop)` before every segment center is
    /// evaluated, otherwise the termination of the returned estimate.
    #[allow(clippy::too_many_lines)]
    fn refine<F, V>(
        &mut self,
        options: &LevelOptions<S>,
        basis: &super::transform::BasisTable<S>,
        maps: &[PreparedMap<S>],
        function: &F,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Result<IntegrationTermination, IntegrationTermination>, SciOxideError>
    where
        F: Fn(DoubleExponentialPoint<S>) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        let tolerance = options.tolerance();
        let policy = options.execution_policy();
        let d = self.components;
        let centers = maps.len();
        self.segment_divisor = S::try_from_usize_exact(centers)?;
        self.values.fill(S::ZERO);
        self.previous.fill(S::ZERO);
        self.extents.fill(0);
        self.complete_shells.fill(0);
        self.saturated.fill(false);
        self.stagnated.fill(false);
        self.last_difference.fill(S::INFINITY);
        self.slow.fill(0);
        self.raw.fill(PromotingCompensatedSum::ZERO);
        self.absolute.fill(PromotingCompensatedSum::ZERO);
        self.shells.fill([S::ZERO; MAX_TAIL_SHELLS]);
        self.pending = 0;
        self.mesh = S::ONE;
        self.cuts.fill([MAX_TAIL_SHELLS; 2]);
        self.sampled.fill([[0; MAX_TAIL_SHELLS]; 2]);
        self.omitted.fill([S::ZERO; 2]);
        self.witnesses.fill([S::ZERO; 2]);
        self.edges.fill([[S::ZERO; MAX_TAIL_SHELLS]; 2]);
        // Phase 1 (h = 1): evaluate every segment's centre, then walk outward one unit shell
        // at a time until the tail heuristic accepts, the transform saturates, or
        // MAX_TAIL_SHELLS is reached. Phase 2 halves h, evaluates new odd nodes, and
        // extends individual tails when the finer shell estimates demand it.
        if let Some(stop) = admit(context, centers, false)? {
            return Ok(Err(stop));
        }
        for (segment, map) in maps.iter().enumerate() {
            let center = map.center()?;
            self.push(function, center, segment, None, policy, context)?;
        }
        self.flush(function, policy, context)?;
        let mut termination = IntegrationTermination::RefinementLimit;
        let mut stopped = false;
        'tails: for (segment, map) in maps.iter().enumerate() {
            let mut shell = 0;
            while shell < MAX_TAIL_SHELLS {
                // No tail can be accepted before the mandatory evidence window.
                // Sample that window in one dispatch, capped by inherited budgets.
                let available = context
                    .budget_report(scioxide_core::BudgetKind::Evaluation)
                    .remaining()
                    .unwrap_or(usize::MAX);
                let wanted = if shell < REQUIRED_TAIL_SHELLS {
                    (REQUIRED_TAIL_SHELLS - shell).min((available / 2).max(1))
                } else {
                    1
                };
                let mut prepared = [None; REQUIRED_TAIL_SHELLS];
                let mut count = 0;
                for slot in prepared.iter_mut().take(wanted) {
                    let PairPreparation::Complete(pair) = map.pair(0, shell + count + 1, basis)
                    else {
                        self.saturated[segment] = true;
                        break;
                    };
                    *slot = Some(pair);
                    count += 1;
                }
                if count == 0 {
                    break;
                }
                if let Some(stop) = admit(context, 2 * count, false)? {
                    termination = stop;
                    stopped = true;
                    break 'tails;
                }
                for c in 0..d {
                    self.previous[c] = self.raw[segment * d + c].round()?;
                }
                for (offset, pair) in prepared.into_iter().take(count).flatten().enumerate() {
                    for (side, point) in [pair.negative, pair.positive]
                        .into_iter()
                        .enumerate()
                        .filter_map(|(side, point)| point.map(|point| (side, point)))
                    {
                        self.push(
                            function,
                            point,
                            segment,
                            Some(ShellTarget::new(shell + offset, side, true, false)),
                            policy,
                            context,
                        )?;
                    }
                }
                self.flush(function, policy, context)?;
                shell += count;
                self.extents[segment] = shell;
                self.complete_shells[segment] = shell;
                let mut truncated = true;
                let mut stagnated = false;
                for c in 0..d {
                    let index = segment * d + c;
                    let current = self.raw[index].round()?;
                    let limit = self.segment_threshold(tolerance, c, current)?;
                    for side in 0..2 {
                        truncated &=
                            tail_shells(&self.shells[2 * index + side][..shell], limit * S::HALF).0;
                    }
                    let mass =
                        |shell| self.shells[2 * index][shell] + self.shells[2 * index + 1][shell];
                    stagnated |= shell >= 2
                        && current == self.previous[c]
                        && mass(shell - 1) >= mass(shell - 2)
                        && mass(shell - 1) > limit.max(sum_floor(&self.absolute[index])?);
                }
                if stagnated {
                    self.stagnated[segment] = true;
                    termination = IntegrationTermination::Stagnation;
                    stopped = true;
                    break 'tails;
                }
                if truncated || self.saturated[segment] {
                    break;
                }
                if shell == MAX_TAIL_SHELLS {
                    self.saturated[segment] = true;
                }
            }
        }
        self.estimate(S::ONE, 0, tolerance, centers)?;
        if stopped {
            return Ok(Ok(termination));
        }
        let mut h = S::ONE;
        'levels: for level in 1..options.max_levels().get() {
            // New points are t = (2i + 1) * new_h, the odd multiples that did not exist at
            // mesh h; `per_shell` of them fall in each unit shell, so point i belongs to
            // shell `i / per_shell`. The whole level is prepared and validated first so
            // that it is admitted (budget and cost) as one atomic unit.
            let new_h = h * S::HALF;
            self.freeze_tails(level - 1, maps)?;
            let Some(cost) = self.refinement_cost(maps, level)? else {
                termination = IntegrationTermination::Stagnation;
                break;
            };
            if cost == 0 {
                termination = IntegrationTermination::Stagnation;
                break;
            }
            if let Some(stop) = admit(context, cost, false)? {
                termination = stop;
                break;
            }
            self.previous.copy_from_slice(&self.values);
            self.outer_samples.fill([[S::ZERO; MAX_TAIL_SHELLS]; 2]);
            // Shell masses follow the trapezoidal mesh: halving it halves the
            // weight of every retained sample before the new nodes are added.
            self.mesh = new_h;
            for masses in &mut self.shells {
                for mass in masses.iter_mut() {
                    *mass *= S::HALF;
                }
            }
            self.sample_refinement(maps, basis, level, function, policy, context)?;
            h = new_h;
            let mut truncated = self.estimate(h, level, tolerance, centers)?;
            if self.thaw_tails(centers)? {
                let Some(cost) = self.refinement_cost(maps, level)? else {
                    self.unresolved_tail();
                    termination = IntegrationTermination::Stagnation;
                    break;
                };
                if let Some(stop) = admit(context, cost, false)? {
                    self.unresolved_tail();
                    termination = stop;
                    break;
                }
                self.sample_refinement(maps, basis, level, function, policy, context)?;
                truncated = self.estimate(h, level, tolerance, centers)?;
            }
            // A coarse shell may look negligible until its interior is sampled. Extend
            // on the current mesh instead of treating that discovery as stagnation.
            for (segment, map) in maps.iter().enumerate() {
                while self.extents[segment] < MAX_TAIL_SHELLS {
                    if self.extents[segment] > self.complete_shells[segment] {
                        break;
                    }
                    if !self.needs_extension(segment) {
                        break;
                    }
                    let shell = self.extents[segment];
                    let count = 1usize << level;
                    let admission = map.admission(level, shell, false);
                    let cost = admission.evaluations;
                    let partial = admission.partial;
                    if admission.failed && !map.finite() {
                        self.saturated[segment] = true;
                        break;
                    }
                    if let Some(stop) = admit(context, cost, false)? {
                        termination = stop;
                        break 'levels;
                    }
                    for index in 1..=count {
                        if let PairPreparation::Complete(pair) =
                            map.pair(level, shell * count + index, basis)
                        {
                            for (side, point) in [pair.negative, pair.positive]
                                .into_iter()
                                .enumerate()
                                .filter_map(|(side, point)| point.map(|point| (side, point)))
                            {
                                self.push(
                                    function,
                                    point,
                                    segment,
                                    Some(ShellTarget::new(
                                        shell,
                                        side,
                                        index == count,
                                        index + 1 == count,
                                    )),
                                    policy,
                                    context,
                                )?;
                            }
                        } else {
                            self.saturated[segment] = true;
                        }
                    }
                    self.flush(function, policy, context)?;
                    self.sampled[segment][0][shell] = level;
                    self.sampled[segment][1][shell] = level;
                    self.extents[segment] += 1;
                    if !partial {
                        self.complete_shells[segment] = self.extents[segment];
                    }
                    truncated = self.estimate(h, level, tolerance, centers)?;
                }
            }
            // Early levels can agree by coincidence (e.g. symmetric integrands), so no
            // decision is taken before MIN_CONVERGENCE_LEVEL. Even a met tolerance is only
            // "heuristic" because the tail term is not a bound. Finite saturation needs
            // decreasing-shell evidence; unresolved tails or all-zero samples cannot succeed.
            if level >= MIN_CONVERGENCE_LEVEL
                && truncated
                && tolerance.is_satisfied(&self.errors, &self.scales)?
            {
                termination = IntegrationTermination::HeuristicToleranceReached;
                break;
            }
            // DE differences should shrink roughly quadratically per level. Sustained
            // slow decay above roundoff leaves unresolved features outside that model;
            // unchanged values or differences at the floor exhaust arithmetic progress.
            // A growing difference is pre-asymptotic discovery, not slow decay, so it
            // restarts the count rather than extending it.
            for c in 0..d {
                let difference = (self.values[c] - self.previous[c]).abs();
                self.slow[c] = if difference > self.diagnostics[c].roundoff.unwrap_or(S::ZERO)
                    && difference < self.last_difference[c]
                    && difference >= self.last_difference[c] * S::HALF * S::HALF
                {
                    self.slow[c] + 1
                } else {
                    0
                };
                self.last_difference[c] = difference;
            }
            if level >= MIN_CONVERGENCE_LEVEL
                && self
                    .diagnostics
                    .iter()
                    .any(|part| part.roundoff.is_some_and(|floor| floor > S::ZERO))
                && (self.values == self.previous
                    || self.slow.iter().any(|&n| n >= MAX_SLOW_DIFFERENCES)
                    || self.last_difference.iter().zip(&self.diagnostics).all(
                        |(&difference, diagnostic)| {
                            difference <= diagnostic.roundoff.unwrap_or(S::ZERO)
                        },
                    ))
            {
                termination = IntegrationTermination::Stagnation;
                break;
            }
        }
        Ok(Ok(termination))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::tail_shells;

    #[test]
    fn tail_shell_growth_resets_consecutive_evidence() {
        assert!(!tail_shells(&[0.1_f64, 0.01, 0.5, 0.01], 0.1).0);
        assert!(tail_shells(&[0.1_f64, 0.01, 0.5, 0.01, 0.001, 0.0001], 0.1).0);
        assert!(!tail_shells(&[0.1_f32, 0.01, 0.5, 0.01], 0.1).0);
        assert!(tail_shells(&[0.1_f32, 0.01, 0.5, 0.01, 0.001, 0.0001], 0.1).0);
    }
}
