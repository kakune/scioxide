//! Circular statistics of angles and mean directions of vectors in `ℝᵈ`.
use crate::arithmetic::Compensated;
use crate::error_helpers::{check_weights, finite, invalid, mass, mass_value, number};
use crate::{KernelWeights, ObservationData, StatisticsOptions};
use alloc::vec::Vec;

use scioxide_core::Scaled;
use scioxide_core::backend::try_filled;
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

/// Period of the circular data for [`Statistic::circular`](crate::Statistic::circular).
///
/// A value `x` maps to the angle `θ = 2π (x − low) / (high − low)`, modulo `2π`,
/// and the mean is returned in `[low, high)`. The default is `[0, 2π)`, for
/// angles in radians. Use `[0, 360)` for degrees or `[0, 24)` for hours of the day.
#[derive(Clone, Copy, Debug)]
pub struct CircularOptions<S: RealScalar> {
    /// Start of the period.
    pub(crate) low: S,
    /// End of the period, greater than `low`.
    pub(crate) high: S,
}
impl<S: RealScalar> Default for CircularOptions<S> {
    fn default() -> Self {
        Self {
            low: S::ZERO,
            high: S::TAU,
        }
    }
}
/// Weighted circular mean, mean resultant length, circular variance and
/// circular standard deviation.
///
/// For angles `θᵢ` with weights `wᵢ`,
///
/// ```text
/// C̄ = Σ wᵢ cos θᵢ / Σ wᵢ,   S̄ = Σ wᵢ sin θᵢ / Σ wᵢ,   R̄ = √(C̄² + S̄²)
/// ```
///
/// The mean direction is `atan2(S̄, C̄)`. It is undefined when `R̄ ≤ 8ε`, where
/// `ε` is the machine epsilon of `S`. A scaled hypotenuse preserves nonzero
/// resultant components whose squares would underflow.
///
/// # References
///
/// K. V. Mardia and P. E. Jupp, *Directional Statistics*, Wiley, 2000.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CircularSummary<S: RealScalar> {
    /// Mean direction in `[low, high)`, or `None` when `R̄ ≤ 8ε`.
    pub mean: Option<S>,
    /// Mean resultant length `R̄`, in `[0, 1]`.
    pub resultant_length: S,
    /// Circular variance `1 − R̄`, in `[0, 1]` and without units.
    pub variance: S,
    /// Circular standard deviation `√(−2 ln R̄)`, in the units of the period, or
    /// `None` when `R̄ = 0`.
    pub standard_deviation: Option<S>,
}
/// Computes the weighted circular summary in O(n). Uses compensated `f64` sums,
/// and restarts in scaled arithmetic if a weighted term would lose range. With
/// the default period `[0, 2π)` the angles are used directly; other periods are
/// reduced modulo the period before conversion to radians. Zero-weight values
/// are skipped before they are read.
///
/// # Errors
/// Fails on zero total weight, invalid weights or period, a non-finite value,
/// or a result that cannot be represented in `S`.
pub(crate) fn circular_statistics<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    options: CircularOptions<S>,
) -> Result<CircularSummary<S>, SciOxideError> {
    check_weights(weights, values.len(), scioxide_core::argument!("weights"))?;
    let period = options.high - options.low;
    if !options.low.is_finite()
        || !options.high.is_finite()
        || !period.is_finite()
        || period <= S::ZERO
    {
        return Err(invalid(
            scioxide_core::argument!("options"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "circular period must be positive and finite",
        ));
    }
    let [s, c] = if options.low == S::ZERO && options.high == S::TAU {
        circular_components(values, weights, &|x| x)?
    } else {
        let origin = (options.low % period) / period;
        circular_components(values, weights, &|x| {
            // Divide bounded remainders before subtracting: their difference can overflow.
            ((x % period) / period - origin) * S::TAU
        })?
    };
    circular_summary(s, c, options, period)
}

fn circular_components<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    phase: &impl Fn(S) -> S,
) -> Result<[S; 2], SciOxideError> {
    let Some(weights) = weights else {
        let mut sin = Compensated::<f64>::zero();
        let mut cos = Compensated::<f64>::zero();
        for &x in values {
            if !x.is_finite() {
                return Err(invalid(
                    scioxide_core::argument!("values"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "nonfinite angle",
                ));
            }
            let (s, c) = phase(x).sin_cos();
            sin = sin.added(s.into_f64())?;
            cos = cos.added(c.into_f64())?;
        }
        return normalized_components(sin.total()?, cos.total()?, values.len() as f64);
    };
    // At most 64 count bits and exponents in [-256,256] bound total mass
    // below 2^321. Trigonometric products additionally require normal range;
    // subnormal products must not disappear before the final normalization.
    match weights {
        KernelWeights::Frequency(w) => {
            weighted_components(values, Some(weights), phase, |i| Ok(w[i] as f64))
        }
        KernelWeights::Mass(_) => weighted_components(values, Some(weights), phase, |i| {
            mass_value(Some(weights), i, scioxide_core::argument!("weights"))
        }),
    }
}

fn weighted_components<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    phase: &impl Fn(S) -> S,
    weight: impl Fn(usize) -> Result<f64, SciOxideError>,
) -> Result<[S; 2], SciOxideError> {
    let mut total = Compensated::<f64>::zero();
    let mut sin = Compensated::<f64>::zero();
    let mut cos = Compensated::<f64>::zero();
    for (i, &x) in values.iter().enumerate() {
        let w = weight(i)?;
        if w == 0.0 {
            continue;
        }
        if !x.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "nonfinite angle",
            ));
        }
        if !crate::arithmetic::native(w, 256) {
            return scaled_components(values, weights, phase);
        }
        let (s, c) = phase(x).sin_cos();
        let (s, c) = (s.into_f64(), c.into_f64());
        let (ws, wc) = (w * s, w * c);
        if (s != 0.0 && !ws.is_normal()) || (c != 0.0 && !wc.is_normal()) {
            return scaled_components(values, weights, phase);
        }
        total = total.added(w)?;
        sin = sin.added(ws)?;
        cos = cos.added(wc)?;
    }
    normalized_components(sin.total()?, cos.total()?, total.total()?)
}

fn normalized_components<S: RealScalar>(
    sin: f64,
    cos: f64,
    total: f64,
) -> Result<[S; 2], SciOxideError> {
    finish_components(
        Scaled::from_value(sin)?,
        Scaled::from_value(cos)?,
        Scaled::from_value(total)?,
    )
}

fn scaled_components<S: RealScalar>(
    values: &[S],
    weights: Option<KernelWeights<'_, S>>,
    phase: &impl Fn(S) -> S,
) -> Result<[S; 2], SciOxideError> {
    let mut total = Compensated::<Scaled<f64>>::zero();
    let mut sin = Compensated::<Scaled<f64>>::zero();
    let mut cos = Compensated::<Scaled<f64>>::zero();
    for (i, &x) in values.iter().enumerate() {
        let w = mass(weights, i, scioxide_core::argument!("weights"))?;
        if w.is_zero() {
            continue;
        }
        if !x.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "nonfinite angle",
            ));
        }
        let (s, c) = phase(x).sin_cos();
        total = total.added(w)?;
        sin = sin.added(w.mul(Scaled::from_value(s.into_f64())?)?)?;
        cos = cos.added(w.mul(Scaled::from_value(c.into_f64())?)?)?;
    }
    finish_components(sin.total()?, cos.total()?, total.total()?)
}

fn finish_components<S: RealScalar>(
    sin: Scaled<f64>,
    cos: Scaled<f64>,
    total: Scaled<f64>,
) -> Result<[S; 2], SciOxideError> {
    if total.is_zero() {
        return Err(invalid(
            scioxide_core::argument!("values"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "empty positive-mass circular sample",
        ));
    }
    Ok([sin.div(total)?.round_to()?, cos.div(total)?.round_to()?])
}

fn circular_summary<S: RealScalar>(
    s: S,
    c: S,
    options: CircularOptions<S>,
    period: S,
) -> Result<CircularSummary<S>, SciOxideError> {
    let r = finite(s.hypot(c))?.min(S::ONE);
    let undefined = r <= S::EPSILON * number(8.)?;
    let phase = s.atan2(c);
    let phase = if phase < S::ZERO {
        phase + S::TAU
    } else {
        phase
    };
    let mean = if undefined {
        None
    } else {
        let value = Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(options.low))?
            .add(
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(period))?.mul(
                    Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(phase / S::TAU))?,
                )?,
            )?
            .round_to()?;
        Some(if value >= options.high {
            options.low
        } else {
            value
        })
    };
    let sd = if r == S::ZERO {
        None
    } else {
        let angular = number::<S>(-2.)? * r.ln();
        Some(
            Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(angular.sqrt()))?
                .mul(Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(period),
                )?)?
                .div(Scaled::<f64>::from_value(
                    scioxide_core::RealScalar::into_f64(S::TAU),
                )?)?
                .round_to()?,
        )
    };
    Ok(CircularSummary {
        mean,
        resultant_length: r,
        variance: S::ONE - r,
        standard_deviation: sd,
    })
}
/// Treatment of the input vectors of
/// [`directional_statistics_into`](crate::directional_statistics_into).
#[derive(Clone, Copy, Debug)]
pub enum DirectionNormalization<S: RealScalar> {
    /// Divides each vector by its Euclidean norm, computed without overflow or
    /// underflow. A zero vector is an error.
    Normalize,
    /// Requires `|‖x‖ − 1| ≤ t` for each vector `x`, with the given tolerance
    /// `t ≥ 0`, and divides by the norm.
    RequireUnit(S),
}
/// Mean resultant length of one batch of
/// [`directional_statistics_into`](crate::directional_statistics_into).
///
/// For unit vectors `uᵢ` with weights `wᵢ`, the mean resultant vector is
/// `m = Σ wᵢ uᵢ / Σ wᵢ`. The mean direction `m / ‖m‖` is written to the output
/// when `‖m‖ > 8ε`, where `ε` is the machine epsilon of `S`; otherwise the
/// output is all zeros.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalSummary<S: RealScalar> {
    /// Mean resultant length `‖m‖`, in `[0, 1]`.
    pub resultant_length: S,
    /// Whether `‖m‖ > 8ε`, so that the mean direction is defined.
    pub direction_defined: bool,
}
/// Workspace for [`directional_statistics_into`](crate::directional_statistics_into).
///
/// See [reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct DirectionalWorkspace<S: RealScalar> {
    workers: Vec<DirectionalWorker>,
    stage: Vec<S>,
    reports: Vec<DirectionalSummary<S>>,
}

struct DirectionalWorker {
    native: Vec<Compensated<f64>>,
    extended: Vec<Compensated<Scaled<f64>>>,
}

/// Storage that [`DirectionalWorkspace`] needs for one shape of input.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalRequirements {
    dimensions: usize,
    batches: usize,
    workers: usize,
    max_bytes: usize,
}
impl DirectionalRequirements {
    /// Returns the heap bytes the workspace needs, excluding allocator overhead and the output.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn required_bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        let worker = crate::estimation::storage_bytes(&[
            (self.dimensions, core::mem::size_of::<Compensated<f64>>()),
            (
                self.dimensions,
                core::mem::size_of::<Compensated<Scaled<f64>>>(),
            ),
        ])?;
        let output =
            crate::estimation::storage_bytes(&[(self.dimensions, core::mem::size_of::<S>())])?;
        crate::estimation::storage_bytes(&[
            (self.workers, core::mem::size_of::<DirectionalWorker>()),
            (self.workers, worker),
            (self.batches, output),
            (self.batches, core::mem::size_of::<DirectionalSummary<S>>()),
        ])
    }
}
impl<S: RealScalar> DirectionalWorkspace<S> {
    /// Returns the storage needed for `view` with these options and execution context.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    /// - `ResourceLimit` if it exceeds the byte limit in `options`.
    pub fn workspace_requirements<M: crate::EmpiricalWeights>(
        view: crate::BatchView<'_, S, M>,
        options: crate::StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<DirectionalRequirements, SciOxideError> {
        let req = DirectionalRequirements {
            dimensions: view.dimensions(),
            batches: view.batches(),
            workers: crate::estimation::worker_count(options, context, view.batches())?,
            max_bytes: options.max_bytes(),
        };
        crate::error_helpers::limit_bytes(req.required_bytes::<S>()?, req.max_bytes)?;
        Ok(req)
    }
    /// Returns whether the workspace is large enough for `req`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace already holds more than the byte limit of `req`.
    pub fn check(
        &self,
        req: &DirectionalRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let bytes = crate::estimation::storage_bytes(&[
            (
                self.workers.capacity(),
                core::mem::size_of::<DirectionalWorker>(),
            ),
            (self.stage.capacity(), core::mem::size_of::<S>()),
            (
                self.reports.capacity(),
                core::mem::size_of::<DirectionalSummary<S>>(),
            ),
        ])?;
        let bytes = self.workers.iter().try_fold(bytes, |bytes, w| {
            crate::estimation::storage_bytes(&[
                (1, bytes),
                (
                    w.native.capacity(),
                    core::mem::size_of::<Compensated<f64>>(),
                ),
                (
                    w.extended.capacity(),
                    core::mem::size_of::<Compensated<Scaled<f64>>>(),
                ),
            ])
        })?;
        crate::error_helpers::limit_bytes(bytes, req.max_bytes)?;
        Ok(
            if self.workers.len() >= req.workers
                && self
                    .workers
                    .iter()
                    .take(req.workers)
                    .all(|w| w.native.len() >= req.dimensions && w.extended.len() >= req.dimensions)
                && self.stage.len() >= req.dimensions * req.batches
                && self.reports.len() >= req.batches
            {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to fit `req`, keeping storage that is large enough.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of `req`.
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn try_reserve(&mut self, req: &DirectionalRequirements) -> Result<(), SciOxideError> {
        if self.check(req)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        crate::error_helpers::limit_bytes(req.required_bytes::<S>()?, req.max_bytes)?;
        *self = Self::allocate(req)?;
        Ok(())
    }
    /// Creates a workspace that fits `req`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`try_reserve`](Self::try_reserve).
    pub fn workspace(req: &DirectionalRequirements) -> Result<Self, SciOxideError> {
        let mut ws = Self::default();
        ws.try_reserve(req)?;
        Ok(ws)
    }
}
impl<S: RealScalar> Default for DirectionalWorkspace<S> {
    fn default() -> Self {
        Self {
            workers: Vec::new(),
            stage: Vec::new(),
            reports: Vec::new(),
        }
    }
}
impl<S: RealScalar> DirectionalWorkspace<S> {
    /// Allocates storage of exactly the requested sizes.
    ///
    /// # Errors
    /// Returns size overflow or allocation failure.
    fn allocate(req: &DirectionalRequirements) -> Result<Self, SciOxideError> {
        let count = req.workers;
        let mut workers = scioxide_core::backend::try_with_capacity(
            count,
            scioxide_core::quantity!("direction_workers"),
        )?;
        for _ in 0..count {
            let native = try_filled(
                req.dimensions,
                Compensated::zero(),
                scioxide_core::quantity!("directional_native_sums"),
            )?;
            let extended = try_filled(
                req.dimensions,
                Compensated::zero(),
                scioxide_core::quantity!("directional_worker_sums"),
            )?;
            workers.push(DirectionalWorker { native, extended });
        }
        let len = req.batches.checked_mul(req.dimensions).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("direction_outputs"))
        })?;
        let stage = try_filled(len, S::ZERO, scioxide_core::quantity!("directional_stage"))?;
        let reports = try_filled(
            req.batches,
            DirectionalSummary {
                resultant_length: S::ZERO,
                direction_defined: false,
            },
            scioxide_core::quantity!("directional_reports"),
        )?;
        Ok(Self {
            workers,
            stage,
            reports,
        })
    }
}
/// Computes the mean direction and mean resultant length of each batch, in
/// O(n·d) per batch. Directions are written in batch-major order. With
/// `Normalize`, batches use compensated `f64` sums and restart in scaled
/// arithmetic when a value is out of range. `RequireUnit` always checks the
/// norms in scaled arithmetic.
///
/// # Errors
/// Fails on a zero or non-unit vector, an invalid tolerance, zero total weight,
/// output of the wrong length, a workspace that is too small, or an execution failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn directional_statistics_into<S: RealScalar>(
    view: ObservationData<'_, S>,
    normalization: DirectionNormalization<S>,
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
    workspace: &mut DirectionalWorkspace<S>,
    directions: &mut [S],
    reports: &mut [DirectionalSummary<S>],
) -> Result<(), SciOxideError> {
    let d = view.dimensions();
    let count = view.batches();
    if let DirectionNormalization::RequireUnit(t) = normalization
        && (!t.is_finite() || t < S::ZERO)
    {
        return Err(invalid(
            scioxide_core::argument!("normalization"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "unit-norm tolerance must be finite and nonnegative",
        ));
    }
    if directions.len() != count * d
        || reports.len() != count
        || workspace.stage.len() < directions.len()
        || workspace.reports.len() < count
        || workspace
            .workers
            .iter()
            .any(|w| w.native.len() < d || w.extended.len() < d)
    {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "directional storage mismatch",
        ));
    }
    // Use a combined lane (direction plus resultant) so orchestration owns disjoint output.
    crate::estimation::execute(
        options,
        context,
        count,
        d,
        &mut workspace.workers,
        &mut workspace.stage,
        |batch, worker, out| {
            let origin = view.batch_origin(batch);
            if matches!(normalization, DirectionNormalization::Normalize)
                && native_directional_batch(
                    view,
                    batch,
                    origin,
                    options,
                    &mut worker.native[..d],
                    out,
                )?
            {
                return Ok(());
            }
            let sums = &mut worker.extended[..d];
            sums[..d].fill(Compensated::zero());
            let mut total = Compensated::<Scaled<f64>>::zero();
            for i in 0..view.samples() {
                let Some(w) = directional_mass_by(view, batch, i, options, |j| {
                    view.value_at_origin(origin, i, j)
                })?
                else {
                    continue;
                };
                let w = Scaled::<f64>::from_value(w)?;
                let mut squares = Compensated::<Scaled<f64>>::zero();
                for j in 0..d {
                    squares = squares.added(
                        Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(
                            view.value_at_origin(origin, i, j),
                        ))?
                        .square()?,
                    )?;
                }
                let norm = squares.total()?.sqrt()?;
                if norm.is_zero() {
                    return Err(invalid(
                        scioxide_core::argument!("view"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "zero directional vector",
                    ));
                }
                if let DirectionNormalization::RequireUnit(t) = normalization
                    && norm
                        .sub(Scaled::<f64>::ONE)?
                        .abs_cmp(Scaled::<f64>::from_value(
                            scioxide_core::RealScalar::into_f64(t),
                        )?)
                        .is_gt()
                {
                    return Err(invalid(
                        scioxide_core::argument!("view"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "direction vector is not unit length",
                    ));
                }
                for (j, sum) in sums[..d].iter_mut().enumerate() {
                    *sum = sum.added(
                        Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(
                            view.value_at_origin(origin, i, j),
                        ))?
                        .div(norm)?
                        .mul(w)?,
                    )?;
                }
                total = total.added(w)?;
            }
            let total = total.total()?;
            if total.is_zero() {
                return Err(invalid(
                    scioxide_core::argument!("view"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "empty positive-mass directional sample",
                ));
            }
            for (slot, sum) in out.iter_mut().zip(sums) {
                *slot = sum.total()?.div(total)?.round_to()?;
            }
            Ok(())
        },
    )?;
    for (out, report) in workspace.stage[..count * d]
        .chunks_exact_mut(d)
        .zip(&mut workspace.reports)
    {
        let mut norm = Compensated::<Scaled<f64>>::zero();
        for &x in out.iter() {
            norm = norm.added(
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))?.square()?,
            )?;
        }
        let r: S = norm.total()?.sqrt()?.round_to()?;
        let defined = r > S::EPSILON * number(8.)?;
        for x in out {
            *x = if defined { *x / r } else { S::ZERO };
        }
        *report = DirectionalSummary {
            resultant_length: r.min(S::ONE),
            direction_defined: defined,
        };
    }
    directions.copy_from_slice(&workspace.stage[..count * d]);
    reports.copy_from_slice(&workspace.reports[..count]);
    Ok(())
}

fn directional_mass_by<S: RealScalar>(
    view: ObservationData<'_, S>,
    batch: usize,
    sample: usize,
    options: StatisticsOptions,
    coordinate: impl FnMut(usize) -> S,
) -> Result<Option<f64>, SciOxideError> {
    Ok(view
        .admit_count_by(
            batch,
            sample,
            options,
            scioxide_core::argument!("view"),
            coordinate,
        )?
        .map(|count| match view.weights {
            Some(KernelWeights::Mass(w)) => w[batch * view.samples() + sample].into_f64(),
            _ => count as f64,
        }))
}

// Source and mass exponents in [-256,256], with at most 64 dimension/count
// bits, keep squared norms, normalized coordinates and weighted contributions
// normal binary64. All f32 coordinates and masses fit this frame. A declined
// batch is replayed from its original observations, never from rounded sums.
fn native_directional_batch<S: RealScalar>(
    view: ObservationData<'_, S>,
    batch: usize,
    origin: usize,
    options: StatisticsOptions,
    sums: &mut [Compensated<f64>],
    out: &mut [S],
) -> Result<bool, SciOxideError> {
    sums.fill(Compensated::zero());
    let weighted = view.weights.is_some();
    let mut admitted = 0usize;
    let mut total = Compensated::<f64>::zero();
    for i in 0..view.samples() {
        let mut squares = scioxide_core::CompensatedSum::<f64>::ZERO;
        let mut native = true;
        let mut fault = None;
        // Admission already visits every retained coordinate. Accumulate its
        // squared norm in that pass; skipped observations discard this state.
        let Some(w) = directional_mass_by(view, batch, i, options, |j| {
            let x = view.value_at_origin(origin, i, j);
            let value = x.into_f64();
            native &= crate::arithmetic::native(value, 256);
            if native && let Err(error) = squares.add(value * value) {
                fault.get_or_insert(error);
            }
            x
        })?
        else {
            continue;
        };
        if !native || (weighted && !crate::arithmetic::native(w, 256)) {
            return Ok(false);
        }
        if let Some(error) = fault {
            return Err(error);
        }
        let norm = squares.total()?.sqrt();
        if norm == 0.0 {
            return Err(invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "zero directional vector",
            ));
        }
        for (j, sum) in sums.iter_mut().enumerate() {
            let component = view.value_at_origin(origin, i, j).into_f64() / norm;
            *sum = sum.added(if weighted { component * w } else { component })?;
        }
        if weighted {
            total = total.added(w)?;
        } else {
            admitted += 1;
        }
    }
    let total = if weighted {
        total.total()?
    } else {
        admitted as f64
    };
    if total == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("view"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "empty positive-mass directional sample",
        ));
    }
    let total = Scaled::<f64>::from_value(total)?;
    for (slot, sum) in out.iter_mut().zip(sums) {
        *slot = Scaled::<f64>::from_value(sum.total()?)?
            .div(total)?
            .round_to()?;
    }
    Ok(true)
}

impl<S: RealScalar> CircularOptions<S> {
    /// Returns the default period `[0, 2π)`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the start of the period. It is checked when the statistic is computed.
    pub const fn with_low(mut self, value: S) -> Self {
        self.low = value;
        self
    }
    /// Returns the start of the period.
    pub const fn low(self) -> S {
        self.low
    }
    /// Sets the end of the period. It is checked when the statistic is computed.
    pub const fn with_high(mut self, value: S) -> Self {
        self.high = value;
        self
    }
    /// Returns the end of the period.
    pub const fn high(self) -> S {
        self.high
    }
}
