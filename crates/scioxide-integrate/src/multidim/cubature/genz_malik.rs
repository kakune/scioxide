//! Genz–Malik degree-7 cubature with an embedded degree-5 rule, for dimensions
//! two and above.
//!
//! Region centers and half-widths are stored in one contiguous pool. The
//! regional scheduler bisects regions along the axis with the largest fourth
//! difference. Each region costs `2^d + 2d(d−1) + 4d + 1` evaluations and a
//! split twice that. A split is charged as one unit, so every stop after the
//! initial regions leaves an estimate over the whole domain.

use alloc::vec::Vec;

use scioxide_core::backend::{RealValueSum, checked_mul, try_filled};
use scioxide_core::{
    AffineValue, AffineValueMut, Argument, Interval, PartialResultTrust, RealScalar, Scaled,
    ScaledAccumulator, SciOxideError, SolveContext, write_value,
};

use crate::SubdivisionOptions;
use crate::adaptive::regional::{
    self, EpochAllowance, Preparation, Prepared, Refinement, Region, RegionalKernel,
    RegionalPolicy, RegionalWorkspace, Scheduled,
};
use crate::geometry::{IntervalGeometry, checked_volume};
use crate::kernel::checked_intermediate;
use crate::report::{IntegrationErrorComponents, IntegrationTermination};
use crate::workspace::Outcome;

/// Multiple of `ε·volume·max_sample·Σ|w7|` in each regional roundoff floor.
const ROUNDOFF_MULTIPLE: f64 = 50.0;
/// Consecutive splits whose value change stays within the children's floors
/// before an unreachable tolerance stops refinement.
const SETTLED_SPLITS: usize = 2;

#[derive(Debug, Clone, Copy)]
struct PreparedCosts {
    corners: usize,
    region: usize,
}

fn prepared_costs(dim: usize, argument: Argument) -> Result<PreparedCosts, SciOxideError> {
    if dim < 2 {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "the Genz-Malik rule requires dimension at least two",
        ));
    }
    let shift = u32::try_from(dim).map_err(|_| {
        SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "dimension too large for cubature corner enumeration",
        )
    })?;
    let corners = 1_usize.checked_shl(shift).ok_or_else(|| {
        SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "dimension too large for cubature corner enumeration",
        )
    })?;
    // Genz-Malik degree-7 point count: 1 centre + 4d axis points (lambda2 and lambda3, both
    // signs) + 2d(d-1) edge points (+-lambda3 in two coordinates) + 2^d corners (+-lambda5).
    // A split evaluates both children from scratch, hence twice the region cost.
    let preceding = dim.checked_sub(1).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("cubature_edge_evaluation_count"))
    })?;
    let edges = dim
        .checked_mul(preceding)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("cubature_edge_evaluation_count"))
        })?;
    let axes = dim.checked_mul(4).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("cubature_axis_evaluation_count"))
    })?;
    let region = corners
        .checked_add(edges)
        .and_then(|value| value.checked_add(axes))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!(
                "cubature_region_evaluation_count"
            ))
        })?;
    region.checked_mul(2).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("cubature_split_evaluation_cost"))
    })?;
    Ok(PreparedCosts { corners, region })
}

/// Dimension-specific constants of one Genz–Malik rule.
#[derive(Debug, Clone, Copy)]
struct PreparedCubatureRule<S: RealScalar> {
    dim: usize,
    costs: PreparedCosts,
    lambda2: S,
    lambda3: S,
    lambda5: S,
    w7_center: S,
    w7_lambda2: S,
    w7_lambda3: S,
    w7_edge: S,
    w7_corner: S,
    w5_center: S,
    w5_lambda2: S,
    w5_lambda3: S,
    w5_edge: S,
    split_ratio: S,
    directional_tie_scale: S,
    absolute_weight_sum: S,
    /// `50·ε·Σ|w7|`: every sampled regional floor is at least this multiple of
    /// the region's degree-7 magnitude, so the global floor never drops below
    /// `floor_ratio·|estimate|` under refinement.
    floor_ratio: S,
}

impl<S: RealScalar> PreparedCubatureRule<S> {
    fn prepare(dim: usize, argument: Argument) -> Result<Self, SciOxideError> {
        // Genz & Malik (1980) degree-7 rule with its embedded degree-5 rule, written for a
        // region of unit volume (the caller multiplies by the physical volume). Generators
        // are lambda2 = sqrt(9/70), lambda3 = sqrt(9/10) (also used for edges) and
        // lambda5 = sqrt(9/19). The corner weight is split evenly over the 2^d corners so the
        // corner sum can be accumulated unweighted.
        let costs = prepared_costs(dim, argument)?;
        let dimension = S::try_from_usize_exact(dim)?;
        let denominator_19683 = scalar(19683.0)?;
        let w7_center = (scalar::<S>(12824.0)? - scalar::<S>(9120.0)? * dimension
            + scalar::<S>(400.0)? * dimension * dimension)
            / denominator_19683;
        let w7_lambda2 = scalar::<S>(980.0)? / scalar::<S>(6561.0)?;
        let w7_lambda3 =
            (scalar::<S>(1820.0)? - scalar::<S>(400.0)? * dimension) / denominator_19683;
        let w7_edge = scalar::<S>(200.0)? / denominator_19683;
        let corner_count = S::try_from_usize_exact(costs.corners)?;
        let w7_corner = scalar::<S>(6859.0)? / denominator_19683 / corner_count;

        let w5_center = (scalar::<S>(729.0)? - scalar::<S>(950.0)? * dimension
            + scalar::<S>(50.0)? * dimension * dimension)
            / scalar::<S>(729.0)?;
        let w5_lambda2 = scalar::<S>(245.0)? / scalar::<S>(486.0)?;
        let w5_lambda3 =
            (scalar::<S>(265.0)? - scalar::<S>(100.0)? * dimension) / scalar::<S>(1458.0)?;
        let w5_edge = scalar::<S>(25.0)? / scalar::<S>(729.0)?;
        // `split_ratio` = lambda2^2 / lambda3^2 = 1/7 cancels the second-derivative term in
        // the fourth-difference split indicator. The tie scale 10^-d treats indicators that
        // differ by less than that fraction of the embedded error as equal.
        let ten = scalar::<S>(10.0)?;
        let directional_tie_scale = (0..dim).fold(S::ONE, |scale, _| scale / ten);
        let absolute_weight_sum = w7_center.abs()
            + (w7_lambda2.abs() + w7_lambda3.abs()) * (dimension + dimension)
            + w7_edge.abs() * scalar::<S>(2.0)? * dimension * (dimension - S::ONE)
            + w7_corner.abs() * corner_count;

        Ok(Self {
            dim,
            costs,
            lambda2: scalar(0.358_568_582_800_318_1)?,
            lambda3: scalar(0.948_683_298_050_513_8)?,
            lambda5: scalar(0.688_247_201_611_685_3)?,
            w7_center,
            w7_lambda2,
            w7_lambda3,
            w7_edge,
            w7_corner,
            w5_center,
            w5_lambda2,
            w5_lambda3,
            w5_edge,
            split_ratio: scalar(1.0 / 7.0)?,
            directional_tie_scale,
            absolute_weight_sum,
            floor_ratio: absolute_weight_sum
                * <S as RealScalar>::EPSILON
                * scalar::<S>(ROUNDOFF_MULTIPLE)?,
        })
    }
}

fn scalar<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(value)
}

fn degenerate_piece() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "non-degenerate domain produced degenerate geometry",
    )
}

// Flat region storage: slot k occupies `data[2dk..2d(k+1)]` as `[centre (d) | half-widths
// (d)]`. A split rewrites the parent slot as the left child and appends the right child,
// so live regions always occupy slots 0..live without holes.
struct RegionPool<S: RealScalar> {
    dim: usize,
    data: Vec<S>,
}

#[derive(Debug, Clone, Copy)]
struct SplitGeometry<S: RealScalar> {
    axis: usize,
    child_half_width: S,
    left_center: S,
    right_center: S,
}

impl<S: RealScalar> RegionPool<S> {
    fn geometry(&self, slot: usize) -> (&[S], &[S]) {
        let start = slot * self.dim * 2;
        self.data[start..start + self.dim * 2].split_at(self.dim)
    }

    fn prepare_split(
        &self,
        slot: usize,
        axis: usize,
    ) -> Result<Option<SplitGeometry<S>>, SciOxideError> {
        let (center, half_widths) = self.geometry(slot);
        let parent_center = center[axis];
        let parent_half_width = half_widths[axis];
        let child_half_width = parent_half_width * S::HALF;
        if child_half_width == S::ZERO {
            return Ok(None);
        }
        let left_center = checked_intermediate(parent_center - child_half_width)?;
        let right_center = checked_intermediate(parent_center + child_half_width)?;
        if left_center == parent_center || right_center == parent_center {
            return Ok(None);
        }
        Ok(Some(SplitGeometry {
            axis,
            child_half_width,
            left_center,
            right_center,
        }))
    }

    fn copy(&mut self, source: usize, destination: usize) {
        let stride = self.dim * 2;
        self.data
            .copy_within(source * stride..(source + 1) * stride, destination * stride);
    }

    fn stage_split(&mut self, parent: usize, first: usize, split: SplitGeometry<S>) {
        self.copy(parent, first);
        self.copy(parent, first + 1);
        let stride = self.dim * 2;
        for (slot, center) in [(first, split.left_center), (first + 1, split.right_center)] {
            self.data[slot * stride + split.axis] = center;
            self.data[slot * stride + self.dim + split.axis] = split.child_half_width;
        }
    }
}

#[derive(Clone, Copy)]
struct AxisSamples<S: RealScalar> {
    lambda2_plus: S,
    lambda2_minus: S,
    lambda3_plus: S,
    lambda3_minus: S,
}

impl<S: RealScalar> AxisSamples<S> {
    fn zero() -> Self {
        Self {
            lambda2_plus: S::ZERO,
            lambda2_minus: S::ZERO,
            lambda3_plus: S::ZERO,
            lambda3_minus: S::ZERO,
        }
    }
}

struct CubatureScratch<S: RealScalar> {
    point: Vec<S>,
    axes: Vec<AxisSamples<S>>,
}

impl<S: RealScalar> CubatureScratch<S> {
    fn new(dim: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            point: try_filled(dim, S::ZERO, scioxide_core::quantity!("cubature_point"))?,
            axes: try_filled(
                dim,
                AxisSamples::zero(),
                scioxide_core::quantity!("cubature_axes"),
            )?,
        })
    }
}

#[derive(Clone, Copy)]
enum AxisFamily {
    Lambda2,
    Lambda3,
}

fn eval_axis_sum<S: RealScalar, F: FnMut(&[S]) -> Result<S, SciOxideError> + ?Sized>(
    evaluator: &mut F,
    center: &[S],
    half_widths: &[S],
    lambda: S,
    family: AxisFamily,
    scratch: &mut CubatureScratch<S>,
) -> Result<Scaled<S>, SciOxideError> {
    // The individual +-lambda samples are kept per axis because the split-axis selection
    // needs them, not just their sum. `scratch.point` is restored to the centre after each
    // axis so every evaluation differs from the centre in exactly one coordinate.
    let mut sum = ScaledAccumulator::ZERO;
    for axis in 0..center.len() {
        let delta = checked_intermediate(lambda * half_widths[axis])?;
        scratch.point[axis] = checked_intermediate(center[axis] + delta)?;
        let plus = evaluator(&scratch.point)?;
        scratch.point[axis] = checked_intermediate(center[axis] - delta)?;
        let minus = evaluator(&scratch.point)?;
        scratch.point[axis] = center[axis];
        match family {
            AxisFamily::Lambda2 => {
                scratch.axes[axis].lambda2_plus = plus;
                scratch.axes[axis].lambda2_minus = minus;
            }
            AxisFamily::Lambda3 => {
                scratch.axes[axis].lambda3_plus = plus;
                scratch.axes[axis].lambda3_minus = minus;
            }
        }
        sum.add_value(plus)?;
        sum.add_value(minus)?;
    }
    sum.total()
}

fn eval_edge_sum<S: RealScalar, F: FnMut(&[S]) -> Result<S, SciOxideError> + ?Sized>(
    evaluator: &mut F,
    center: &[S],
    half_widths: &[S],
    lambda: S,
    point: &mut [S],
) -> Result<Scaled<S>, SciOxideError> {
    // For each coordinate pair the four sign combinations are visited as (-,-), (+,-), (+,+),
    // (-,+), changing one coordinate per step, and both coordinates are reset afterwards.
    let mut sum = ScaledAccumulator::ZERO;
    for first in 0..center.len() - 1 {
        let first_delta = checked_intermediate(lambda * half_widths[first])?;
        point[first] = center[first] - first_delta;
        for second in first + 1..center.len() {
            let second_delta = checked_intermediate(lambda * half_widths[second])?;
            point[second] = center[second] - second_delta;
            sum.add_value(evaluator(point)?)?;
            point[first] = center[first] + first_delta;
            sum.add_value(evaluator(point)?)?;
            point[second] = center[second] + second_delta;
            sum.add_value(evaluator(point)?)?;
            point[first] = center[first] - first_delta;
            sum.add_value(evaluator(point)?)?;
            point[second] = center[second];
        }
        point[first] = center[first];
    }
    sum.total()
}

fn eval_corner_sum<S: RealScalar, F: FnMut(&[S]) -> Result<S, SciOxideError> + ?Sized>(
    evaluator: &mut F,
    center: &[S],
    half_widths: &[S],
    lambda: S,
    corners: usize,
    point: &mut [S],
) -> Result<Scaled<S>, SciOxideError> {
    for axis in 0..center.len() {
        point[axis] = checked_intermediate(center[axis] + lambda * half_widths[axis])?;
    }

    // Corners are enumerated in binary-reflected Gray code order, so consecutive corners
    // differ in one coordinate (the lowest set bit of `gray ^ next`) and each step updates a
    // single entry of `point` instead of rebuilding all d coordinates.
    let mut sum = ScaledAccumulator::ZERO;
    let mut gray = 0_usize;
    for index in 0..corners {
        sum.add_value(evaluator(point)?)?;
        if index + 1 == corners {
            break;
        }
        let next = (index + 1) ^ ((index + 1) >> 1);
        let changed = gray ^ next;
        let axis = changed.trailing_zeros() as usize;
        let delta = lambda * half_widths[axis];
        point[axis] = if next & (1_usize << axis) == 0 {
            center[axis] + delta
        } else {
            center[axis] - delta
        };
        gray = next;
    }
    sum.total()
}

// Check the ordered generator coordinates, not only movement from the centre:
// rounded corner/edge generators can coincide while the smallest still moves.
fn axis_generators_representable<S: RealScalar>(
    center: S,
    half: S,
    rule: &PreparedCubatureRule<S>,
) -> bool {
    let points = [
        center - rule.lambda3 * half,
        center - rule.lambda5 * half,
        center - rule.lambda2 * half,
        center,
        center + rule.lambda2 * half,
        center + rule.lambda5 * half,
        center + rule.lambda3 * half,
    ];
    points.iter().all(|p| p.is_finite()) && points.windows(2).all(|p| p[0] < p[1])
}
fn generator_geometry_representable<S: RealScalar>(
    center: &[S],
    half_widths: &[S],
    rule: &PreparedCubatureRule<S>,
) -> bool {
    center
        .iter()
        .zip(half_widths)
        .all(|(&center, &half)| axis_generators_representable(center, half, rule))
}
fn child_generator_geometry_representable<S: RealScalar>(
    center: &[S],
    half_widths: &[S],
    split: SplitGeometry<S>,
    rule: &PreparedCubatureRule<S>,
) -> bool {
    [split.left_center, split.right_center]
        .into_iter()
        .all(|split_center| {
            center
                .iter()
                .zip(half_widths)
                .enumerate()
                .all(|(axis, (&center, &half))| {
                    let (center, half) = if axis == split.axis {
                        (split_center, split.child_half_width)
                    } else {
                        (center, half)
                    };
                    axis_generators_representable(center, half, rule)
                })
        })
}

struct RegionEstimate<S: RealScalar> {
    value: S,
    error: S,
    roundoff: S,
    split_dim: usize,
}

fn select_split_axis<S: RealScalar>(
    prepared: &PreparedCubatureRule<S>,
    center_value: S,
    embedded_difference: S,
    half_widths: &[S],
    axes: &[AxisSamples<S>],
) -> Result<usize, SciOxideError> {
    let two = S::TWO;
    let tie_from_rule = Scaled::from_value(embedded_difference)?
        .mul(Scaled::from_value(prepared.directional_tie_scale)?)?;
    let roundoff_multiplier = scalar::<S>(64.0)? * <S as RealScalar>::EPSILON;
    // Genz-Malik fourth-difference indicator per axis:
    //     |f(+l2) + f(-l2) - 2 f(0) - (l2^2 / l3^2) (f(+l3) + f(-l3) - 2 f(0))|,
    // which cancels the second-derivative part and measures fourth-order variation along
    // the axis. Near-ties (relative to the embedded error or roundoff) prefer the widest
    // axis, which keeps regions well-shaped and avoids a systematic bias toward axis 0.
    let range = scioxide_core::backend::ShortReductionRange::new(6);
    let indicator = |samples: AxisSamples<S>| -> Result<Scaled<S>, SciOxideError> {
        if [
            center_value,
            samples.lambda2_plus,
            samples.lambda2_minus,
            samples.lambda3_plus,
            samples.lambda3_minus,
        ]
        .into_iter()
        .all(|value| range.admits(value))
        {
            let near = samples.lambda2_plus + samples.lambda2_minus - two * center_value;
            let far = samples.lambda3_plus + samples.lambda3_minus - two * center_value;
            return Scaled::from_value((near - prepared.split_ratio * far).abs());
        }
        let mut sum = scioxide_core::RangeAccumulator::new();
        sum.add(samples.lambda2_plus)?;
        sum.add(samples.lambda2_minus)?;
        sum.subtract_product(two, center_value)?;
        sum.subtract_product(prepared.split_ratio, samples.lambda3_plus)?;
        sum.subtract_product(prepared.split_ratio, samples.lambda3_minus)?;
        sum.add_triple_product(two, prepared.split_ratio, center_value)?;
        Ok(sum.finish_scaled()?.abs())
    };

    let mut best_axis = 0_usize;
    let mut best = indicator(axes[0])?;
    for axis in 1..prepared.dim {
        let candidate = indicator(axes[axis])?;
        let difference = candidate.sub(best)?;
        let scale = [candidate, best, Scaled::from_value(center_value.abs())?]
            .into_iter()
            .max_by(|a, b| a.cmp_value(*b))
            .unwrap_or(Scaled::ZERO);
        let roundoff = scale.mul(Scaled::from_value(roundoff_multiplier)?)?;
        let tie = if roundoff.abs_cmp(tie_from_rule).is_gt() {
            roundoff
        } else {
            tie_from_rule
        };
        if difference.cmp_value(tie).is_gt()
            || (difference.abs_le(tie) && half_widths[axis] > half_widths[best_axis])
        {
            best_axis = axis;
            best = candidate;
        }
    }
    Ok(best_axis)
}

fn genz_malik_evaluated<S: RealScalar, F: FnMut(&[S]) -> Result<S, SciOxideError> + ?Sized>(
    prepared: &PreparedCubatureRule<S>,
    evaluator: &mut F,
    center: &[S],
    half_widths: &[S],
    volume: Scaled<S>,
    scratch: &mut CubatureScratch<S>,
) -> Result<RegionEstimate<S>, SciOxideError> {
    let mut maximum = S::ZERO;
    let original = evaluator;
    let mut evaluator = |point: &[S]| {
        let value = original(point)?;
        maximum = maximum.max(value.abs());
        Ok(value)
    };
    scratch.point.copy_from_slice(center);
    let center_value = evaluator(&scratch.point)?;
    let axis2 = eval_axis_sum(
        &mut evaluator,
        center,
        half_widths,
        prepared.lambda2,
        AxisFamily::Lambda2,
        scratch,
    )?;
    let axis3 = eval_axis_sum(
        &mut evaluator,
        center,
        half_widths,
        prepared.lambda3,
        AxisFamily::Lambda3,
        scratch,
    )?;
    let edges = eval_edge_sum(
        &mut evaluator,
        center,
        half_widths,
        prepared.lambda3,
        &mut scratch.point,
    )?;
    let corners = eval_corner_sum(
        &mut evaluator,
        center,
        half_widths,
        prepared.lambda5,
        prepared.costs.corners,
        &mut scratch.point,
    )?;

    let mut degree7 = ScaledAccumulator::ZERO;
    degree7.add_product(
        Scaled::from_value(prepared.w7_center)?,
        Scaled::from_value(center_value)?,
    )?;
    for (weight, sum) in [
        (prepared.w7_lambda2, axis2),
        (prepared.w7_lambda3, axis3),
        (prepared.w7_edge, edges),
        (prepared.w7_corner, corners),
    ] {
        degree7.add_product(Scaled::from_value(weight)?, sum)?;
    }
    let unscaled7 = degree7.total()?;

    let mut degree5 = ScaledAccumulator::ZERO;
    degree5.add_product(
        Scaled::from_value(prepared.w5_center)?,
        Scaled::from_value(center_value)?,
    )?;
    for (weight, sum) in [
        (prepared.w5_lambda2, axis2),
        (prepared.w5_lambda3, axis3),
        (prepared.w5_edge, edges),
    ] {
        degree5.add_product(Scaled::from_value(weight)?, sum)?;
    }
    let unscaled5 = degree5.total()?;
    let embedded_difference = unscaled7.sub(unscaled5)?.abs();
    let value = volume.mul(unscaled7)?.round()?;
    // Non-asymptotic regional differences need the same fourfold guard used by
    // 1D refinement. The raw embedded difference still selects the split axis.
    let error = volume
        .mul(embedded_difference)?
        .mul(Scaled::from_value(S::TWO * S::TWO)?)?
        .round()?;
    let split_difference = crate::kernel::diagnostic(embedded_difference.round())?;
    let split_dim = select_split_axis(
        prepared,
        center_value,
        split_difference,
        half_widths,
        &scratch.axes,
    )?;
    // A conservative sampled absolute-mass envelope avoids cancellation and
    // overflow in the scale itself: apply epsilon before final projection.
    let roundoff = volume
        .mul(Scaled::from_value(maximum)?)?
        .mul(Scaled::from_value(prepared.absolute_weight_sum)?)?
        .mul(Scaled::from_value(<S as RealScalar>::EPSILON)?)?
        .mul(Scaled::from_value(scalar::<S>(ROUNDOFF_MULTIPLE)?)?)?
        .round()?;
    let roundoff = if maximum == S::ZERO {
        S::ZERO
    } else {
        roundoff.max(S::MIN_POSITIVE * <S as RealScalar>::EPSILON)
    };
    Ok(RegionEstimate {
        value,
        error,
        split_dim,
        roundoff,
    })
}

/// The prepared degree-7/5 rule and its nodes on the reference cube
/// `[-1, 1]^d`, in evaluation order.
#[derive(Debug, Clone)]
pub(super) struct GenzMalik<S: RealScalar> {
    rule: PreparedCubatureRule<S>,
    canonical: Vec<S>,
}

impl<S: RealScalar> GenzMalik<S> {
    /// Prepares the rule of `dimension >= 2`.
    ///
    /// # Errors
    ///
    /// Rejects dimensions whose corner enumeration or split cost overflows.
    pub(super) fn new(dimension: usize) -> Result<Self, SciOxideError> {
        let rule = PreparedCubatureRule::prepare(dimension, scioxide_core::argument!("bounds"))?;
        let mut canonical = try_filled(
            checked_mul(
                rule.costs.region,
                dimension,
                scioxide_core::quantity!("canonical_points"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("canonical_points"),
        )?;
        let center = try_filled(
            dimension,
            S::ZERO,
            scioxide_core::quantity!("cubature_point"),
        )?;
        let half = try_filled(
            dimension,
            S::ONE,
            scioxide_core::quantity!("cubature_point"),
        )?;
        let mut scratch = CubatureScratch::new(dimension)?;
        let mut index = 0;
        genz_malik_evaluated(
            &rule,
            &mut |point| {
                canonical[index * dimension..(index + 1) * dimension].copy_from_slice(point);
                index += 1;
                Ok(S::ZERO)
            },
            &center,
            &half,
            Scaled::from_value(S::ONE)?,
            &mut scratch,
        )?;
        Ok(Self { rule, canonical })
    }

    /// Integrates over the `cells` initial regions, whose per-axis intervals
    /// `piece(cell, axis)` returns (`None` for a collapsed breakpoint piece), and
    /// writes the estimate to `out`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn integrate<F, V, O>(
        &self,
        cells: usize,
        piece: impl Fn(usize, usize) -> Result<Option<Interval<S>>, SciOxideError>,
        options: &SubdivisionOptions<S>,
        function: &F,
        storage: &mut GenzMalikStorage<S>,
        errors: &mut [IntegrationErrorComponents<S>],
        out: &mut O,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Outcome<S>, SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let dimension = self.rule.dim;
        let GenzMalikStorage {
            pool,
            scratch,
            points,
            split_axes,
            stall,
            regional,
        } = storage;
        stall.reset();
        // Validate every initial cell before admission or callbacks.
        for cell in 0..cells {
            let offset = cell * 2 * dimension;
            for axis in 0..dimension {
                let Some(interval) = piece(cell, axis)? else {
                    return Ok(Outcome::Unpublished(
                        IntegrationTermination::GeometryFailure,
                    ));
                };
                let geometry = IntervalGeometry::prepare(interval)?.ok_or_else(degenerate_piece)?;
                pool.data[offset + axis] = interval.midpoint()?;
                pool.data[offset + dimension + axis] = geometry.half_span()?.round()?;
            }
            let (center, half) = pool.geometry(cell);
            if !generator_geometry_representable(center, half, &self.rule) {
                return Ok(Outcome::Unpublished(
                    IntegrationTermination::GeometryFailure,
                ));
            }
            let volume = checked_volume((0..dimension).map(|axis| {
                piece(cell, axis)?
                    .map(IntervalGeometry::prepare)
                    .transpose()?
                    .flatten()
                    .ok_or_else(degenerate_piece)
            }))?;
            regional.regions[cell] = Region::new(volume, cell);
        }
        let policy = RegionalPolicy {
            extrapolation: false,
            refine: true,
            offsets: None,
            norm: &|errors, scales| options.tolerance().normalized_norm(errors, scales),
        };
        let mut kernel = CubatureKernel {
            rule: self,
            function,
            options,
            pool,
            scratch,
            points,
            split_axes,
            stall,
            capacity: regional.capacity,
            components: regional.components,
        };
        let scheduled = regional::schedule(
            cells,
            &mut kernel,
            &policy,
            regional,
            cx,
            &mut EpochAllowance,
        )?;
        let Scheduled::Measured {
            termination,
            subdivisions,
        } = scheduled
        else {
            return Ok(Outcome::Unpublished(scheduled.termination()));
        };
        for (index, error) in errors.iter_mut().enumerate() {
            *error = regional.evidence(index);
            if let Some(discretization) = &mut error.discretization {
                discretization.reliability = crate::ErrorReliability::Heuristic;
            }
        }
        cx.commit_accepted(PartialResultTrust::Admissible)?;
        write_value(&regional.values, out)?;
        Ok(Outcome::Published {
            termination: if termination == IntegrationTermination::ToleranceReached {
                IntegrationTermination::HeuristicToleranceReached
            } else {
                termination
            },
            subdivisions,
            errors: IntegrationErrorComponents::envelope(errors.iter()),
        })
    }
}

struct CubatureKernel<'a, S: RealScalar, F: ?Sized> {
    rule: &'a GenzMalik<S>,
    function: &'a F,
    options: &'a SubdivisionOptions<S>,
    pool: &'a mut RegionPool<S>,
    scratch: &'a mut CubatureScratch<S>,
    points: &'a mut [S],
    split_axes: &'a mut [usize],
    stall: &'a mut Stall<S>,
    capacity: usize,
    components: usize,
}

/// Detects that refinement cannot reach a tolerance below the rounding floor.
///
/// Every sampled regional floor is at least `floor_ratio` times the region's
/// degree-7 magnitude, so the global error estimate never falls below
/// `floor_ratio·|estimate|`. When that bound alone violates the tolerance, the
/// solve can only stagnate; it stops once consecutive splits of the largest
/// regional errors change the estimate by no more than their children's floors.
struct Stall<S: RealScalar> {
    /// Exact totals of the live regional values.
    totals: Vec<RealValueSum<S>>,
    /// Per-component blocks: norm scales and floors, the staged split's parent
    /// values, and the measured first child's values and floors.
    scratch: Vec<S>,
    /// Consecutive splits whose value change stayed within the children's floors.
    settled: usize,
}

impl<S: RealScalar> Stall<S> {
    const BLOCKS: usize = 5;

    fn new(components: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            totals: try_filled(
                components,
                RealValueSum::new(),
                scioxide_core::quantity!("cubature_totals"),
            )?,
            scratch: try_filled(
                checked_mul(
                    components,
                    Self::BLOCKS,
                    scioxide_core::quantity!("cubature_stall"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("cubature_stall"),
            )?,
            settled: 0,
        })
    }

    fn reset(&mut self) {
        self.totals.iter_mut().for_each(RealValueSum::clear);
        self.settled = 0;
    }

    /// Removes the parent of a staged split from the live totals.
    fn stage(&mut self, parent: &[[S; 4]]) -> Result<(), SciOxideError> {
        let c = self.totals.len();
        let staged = &mut self.scratch[2 * c..3 * c];
        for ((total, staged), parts) in self.totals.iter_mut().zip(staged).zip(parent) {
            *staged = parts[0];
            total.add_value(-parts[0])?;
        }
        Ok(())
    }

    /// Adds a measured region. The second child of a split decides whether its
    /// guard `2·|left + right − parent|` stayed within the children's floors.
    fn observe(&mut self, estimates: &[[S; 4]], child: Option<usize>) -> Result<(), SciOxideError> {
        let c = self.totals.len();
        for (total, parts) in self.totals.iter_mut().zip(estimates) {
            total.add_value(parts[0])?;
        }
        let (_, staged) = self.scratch.split_at_mut(2 * c);
        let (parent, first) = staged.split_at_mut(c);
        let (values, floors) = first.split_at_mut(c);
        match child {
            Some(0) => {
                for ((value, floor), parts) in
                    values.iter_mut().zip(floors.iter_mut()).zip(estimates)
                {
                    *value = parts[0];
                    *floor = parts[2];
                }
            }
            Some(_) => {
                let mut settled = true;
                for (((&parent, &value), &floor), parts) in parent
                    .iter()
                    .zip(values.iter())
                    .zip(floors.iter())
                    .zip(estimates)
                {
                    let guard = Scaled::from_value(value)?
                        .add(Scaled::from_value(parts[0])?)?
                        .sub(Scaled::from_value(parent)?)?
                        .abs()
                        .mul(Scaled::from_value(S::TWO)?)?;
                    let floor = Scaled::from_value(floor)?.add(Scaled::from_value(parts[2])?)?;
                    settled &= guard.abs_cmp(floor).is_le();
                }
                self.settled = if settled { self.settled + 1 } else { 0 };
            }
            None => {}
        }
        Ok(())
    }

    /// Returns whether the floor bound of the current estimate alone violates `tolerance`.
    fn unreachable(
        &mut self,
        tolerance: &scioxide_core::ErrorTolerance<S>,
        floor_ratio: S,
    ) -> Result<bool, SciOxideError> {
        let c = self.totals.len();
        let (scales, rest) = self.scratch.split_at_mut(c);
        let floors = &mut rest[..c];
        for ((total, scale), floor) in self
            .totals
            .iter_mut()
            .zip(scales.iter_mut())
            .zip(floors.iter_mut())
        {
            *scale = total.finish()?.abs();
            *floor = floor_ratio * *scale;
        }
        match tolerance.normalized_norm(floors, scales) {
            Ok(norm) => Ok(norm > S::ONE),
            // A nonzero floor against an exactly zero threshold is unbounded.
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    }
                ) =>
            {
                Ok(true)
            }
            Err(error) => Err(error),
        }
    }
}

impl<S, F, V> RegionalKernel<S, Scaled<S>> for CubatureKernel<'_, S, F>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<V, SciOxideError> + Sync + ?Sized,
    V: AffineValue<Scale = S>,
{
    const TRANSACTIONAL_SPLIT: bool = true;
    // Axis changes and the guarded embedded difference need not reduce error
    // on every split. QUADPACK's 1D progress counters give false stalls here.
    const SLOW_PROGRESS: bool = false;
    type Prepared = ();
    type Allowance = EpochAllowance;

    fn prepare(&mut self, _region: Region<S, Scaled<S>>) -> Result<Preparation<()>, SciOxideError> {
        Ok(Preparation::Ready(Prepared {
            rule: (),
            evaluations: self.rule.rule.costs.region,
            providers: self.rule.rule.costs.region,
        }))
    }

    fn refine(
        &mut self,
        region: Region<S, Scaled<S>>,
        slot: usize,
        estimates: &[[S; 4]],
    ) -> Result<Refinement<S, (), Scaled<S>>, SciOxideError> {
        // Every remaining region is retained unrefined, so the solve stagnates.
        if self.stall.settled >= SETTLED_SPLITS
            && self
                .stall
                .unreachable(self.options.tolerance(), self.rule.rule.floor_ratio)?
        {
            return Ok(Refinement::Stop(IntegrationTermination::Stagnation));
        }
        let Some(split) = self.pool.prepare_split(slot, self.split_axes[slot])? else {
            return Ok(Refinement::Stop(IntegrationTermination::Stagnation));
        };
        let (center, half) = self.pool.geometry(slot);
        if !child_generator_geometry_representable(center, half, split, &self.rule.rule) {
            return Ok(Refinement::Stop(IntegrationTermination::Stagnation));
        }
        self.stall.stage(estimates)?;
        self.pool.stage_split(slot, self.capacity, split);
        let child = Region {
            geometry: region.geometry.shifted(-1)?,
            depth: region.depth + 1,
            ..region
        };
        Ok(Refinement::Split(core::array::from_fn(|_| {
            (
                child,
                Prepared {
                    rule: (),
                    evaluations: self.rule.rule.costs.region,
                    providers: self.rule.rule.costs.region,
                },
            )
        })))
    }

    // Fourfold guards for embedded differences and parent-to-children changes
    // are heuristic evidence, not certified bounds. Each child carries half of
    // 4*|left+right-parent| until its next refinement replaces that evidence.
    fn refinement_estimates(
        &self,
        parent: [S; 4],
        left: &mut [S; 4],
        right: &mut [S; 4],
    ) -> Result<(), SciOxideError> {
        let change = Scaled::from_value(left[0])?
            .add(Scaled::from_value(right[0])?)?
            .sub(Scaled::from_value(parent[0])?)?
            .abs()
            .mul(Scaled::from_value(S::TWO)?)?
            .round()?;
        left[1] = left[1].max(change);
        right[1] = right[1].max(change);
        Ok(())
    }

    fn commit(&mut self, source: usize, destination: usize) {
        self.pool.copy(source, destination);
        self.split_axes[destination] = self.split_axes[source];
    }

    /// Samples one region and estimates every component from the same values.
    fn measure(
        &mut self,
        region: Region<S, Scaled<S>>,
        _prepared: &(),
        slot: usize,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        cx: &mut SolveContext<'_, '_>,
        _allowance: &mut EpochAllowance,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        // Initial geometry and both refinement children already occupy the
        // scheduler's measurement slots; commit alone relocates accepted cells.
        let prepared = &self.rule.rule;
        let dimension = prepared.dim;
        let components = self.components;
        let (center, half_widths) = self.pool.geometry(slot);
        let points = &mut *self.points;
        // Canonical nodes are prepared once; mapping a region is a pure coordinate
        // pass and every component reuses the sampled rows.
        for (canonical, point) in self
            .rule
            .canonical
            .chunks_exact(dimension)
            .zip(points.chunks_exact_mut(dimension))
        {
            for (((output, &offset), &center), &half) in
                point.iter_mut().zip(canonical).zip(center).zip(half_widths)
            {
                *output = checked_intermediate(center + offset * half)?;
            }
        }
        let nodes = prepared.costs.region;
        crate::evaluation::sample_nd(
            self.function,
            points,
            dimension,
            nodes,
            components,
            &mut samples[..nodes * components],
            self.options.execution_policy(),
            cx,
        )?;
        let tolerance = self.options.tolerance();
        let mut largest_axis_score = S::ZERO;
        let mut split_dim = 0;
        for component in 0..components {
            let mut index = 0;
            let estimate = genz_malik_evaluated(
                prepared,
                &mut |_| {
                    let start = index * components;
                    index += 1;
                    Ok(samples[start + component])
                },
                center,
                half_widths,
                region.geometry,
                self.scratch,
            )?;
            estimates[component] = [estimate.value, estimate.error, estimate.roundoff, S::ZERO];
            let error = estimate.error.max(estimate.roundoff);
            // Choose the controlling component's directional indicator. Region
            // priority belongs solely to the shared scheduler.
            let score = if components == 1 {
                error
            } else {
                let threshold = tolerance.threshold(component, components, estimate.value.abs())?;
                if threshold == S::ZERO {
                    if error == S::ZERO { S::ZERO } else { S::MAX }
                } else {
                    (error / threshold).min(S::MAX)
                }
            };
            if score > largest_axis_score || component == 0 {
                largest_axis_score = score;
                split_dim = estimate.split_dim;
            }
        }
        self.split_axes[slot] = split_dim;
        // Refinement children occupy the two staging slots after the live capacity.
        self.stall
            .observe(estimates, slot.checked_sub(self.capacity))?;
        Ok(None)
    }
}

/// Region geometry, scratch and scheduler storage of the Genz–Malik rule.
/// Storage is `O(regions·(dimension + components) + nodes·(dimension + components))`.
pub(super) struct GenzMalikStorage<S: RealScalar> {
    pool: RegionPool<S>,
    scratch: CubatureScratch<S>,
    points: Vec<S>,
    split_axes: Vec<usize>,
    stall: Stall<S>,
    regional: RegionalWorkspace<S, Scaled<S>>,
}

impl<S: RealScalar> GenzMalikStorage<S> {
    pub(super) fn new(
        dimension: usize,
        components: usize,
        regions: usize,
    ) -> Result<Self, SciOxideError> {
        let costs = prepared_costs(dimension, scioxide_core::argument!("dimension"))?;
        let slots = scioxide_core::backend::checked_add(
            regions,
            2,
            scioxide_core::quantity!("regional_slots"),
        )?;
        Ok(Self {
            pool: RegionPool {
                dim: dimension,
                data: try_filled(
                    scioxide_core::backend::checked_product(
                        &[dimension, slots, 2],
                        scioxide_core::quantity!("cubature_region_capacity"),
                    )?,
                    S::ZERO,
                    scioxide_core::quantity!("cubature_region_capacity"),
                )?,
            },
            scratch: CubatureScratch::new(dimension)?,
            points: try_filled(
                checked_mul(
                    costs.region,
                    dimension,
                    scioxide_core::quantity!("region_points"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("region_points"),
            )?,
            split_axes: try_filled(slots, 0, scioxide_core::quantity!("cubature_split_axes"))?,
            stall: Stall::new(components)?,
            regional: RegionalWorkspace::with_geometry(
                components,
                regions,
                costs.region,
                Scaled::from_value(S::ZERO)?,
            )?,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use alloc::vec;

    use super::*;

    const DIMENSION: Argument = scioxide_core::argument!("dimension");

    #[test]
    fn prepared_work_costs_are_exact_and_checked() {
        for (dimension, region, split) in [(2, 17, 34), (3, 33, 66), (4, 57, 114)] {
            let costs = prepared_costs(dimension, DIMENSION).unwrap();
            assert_eq!(costs.region, region);
            assert_eq!(costs.region * 2, split);
        }
        let split_overflow_dimension = usize::BITS as usize - 1;
        assert!(matches!(
            prepared_costs(split_overflow_dimension, DIMENSION).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::SizeOverflow { context }
                if context == scioxide_core::quantity!("cubature_split_evaluation_cost")
        ));
        assert!(matches!(
            prepared_costs(usize::BITS as usize, DIMENSION)
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn volume_product_defers_representability_until_final_rounding() {
        let geometry = |widths: [f64; 2]| {
            widths.map(|width| {
                IntervalGeometry::prepare(Interval::new(0.0, width).unwrap())
                    .unwrap()
                    .unwrap()
            })
        };
        let volume = |widths| {
            checked_volume(geometry(widths).into_iter().map(Ok))
                .unwrap()
                .round()
        };
        assert!((volume([1.0e-200, 1.0e200]).unwrap() - 1.0).abs() < 1.0e-15);
        assert_eq!(volume([1.0e-200, 1.0e-200]).unwrap(), 0.0);
        assert!(matches!(
            volume([1.0e200, 1.0e200]).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn region_pool_stages_children_without_changing_the_parent() {
        let mut pool = RegionPool {
            dim: 2,
            data: vec![0.0, 3.0, 2.0, 2.0],
        };
        let split = pool.prepare_split(0, 1).unwrap().unwrap();
        pool.data.resize(12, 0.0);
        pool.stage_split(0, 1, split);
        let right = 2;
        assert_eq!(pool.geometry(0), (&[0.0, 3.0][..], &[2.0, 2.0][..]));
        assert_eq!(pool.geometry(1), (&[0.0, 2.0][..], &[2.0, 1.0][..]));
        assert_eq!(pool.geometry(right), (&[0.0, 4.0][..], &[2.0, 1.0][..]));
    }

    #[test]
    fn corner_traversal_changes_exactly_one_coordinate() {
        let mut points = Vec::new();
        let mut point = vec![0.0; 3];
        eval_corner_sum(
            &mut |point: &[f64]| {
                points.push(point.to_vec());
                Ok(0.0)
            },
            &[0.0; 3],
            &[1.0; 3],
            0.5,
            8,
            &mut point,
        )
        .unwrap();
        assert_eq!(points.len(), 8);
        for adjacent in points.windows(2) {
            let changed = adjacent[0]
                .iter()
                .zip(&adjacent[1])
                .filter(|(left, right)| left != right)
                .count();
            assert_eq!(changed, 1);
        }
    }

    #[test]
    fn split_indicator_prefers_width_only_for_numerical_ties() {
        let prepared = PreparedCubatureRule::<f64>::prepare(2, DIMENSION).unwrap();
        let tied = [AxisSamples::zero(), AxisSamples::zero()];
        assert_eq!(
            select_split_axis(&prepared, 0.0, 0.0, &[0.25, 1.0], &tied).unwrap(),
            1
        );
        let directional = [
            AxisSamples {
                lambda2_plus: 1.0,
                lambda2_minus: 1.0,
                lambda3_plus: 0.0,
                lambda3_minus: 0.0,
            },
            AxisSamples::zero(),
        ];
        assert_eq!(
            select_split_axis(&prepared, 0.0, 0.0, &[0.25, 1.0], &directional).unwrap(),
            0
        );
    }
}
