//! Statistics, rolling statistics, standardization and ranks along an axis.
use crate::error_helpers::limit_bytes;
use crate::weights::WeightData;
use crate::{
    Axis, BatchView, MissingPolicy, NonFinitePolicy, RollingBoundary, RollingWindow, Statistic,
    StatisticValue, StatisticsOptions, WeightModel,
};
use alloc::vec::Vec;
use core::mem::size_of;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{Capacity, ExecutionContext, NumericScalar, RealScalar, SciOxideError};

mod geometry {
    use super::*;
    #[derive(Clone, Copy)]
    pub struct Shape {
        pub samples: usize,
        pub batches: usize,
        pub dimensions: usize,
        pub weighted: bool,
    }
    pub trait Sealed {
        fn shape(self) -> Result<Shape, SciOxideError>;
    }
    impl<T: NumericScalar, M: WeightModel> Sealed for BatchView<'_, T, M> {
        fn shape(self) -> Result<Shape, SciOxideError> {
            Ok(Shape {
                samples: self.samples(),
                batches: self.batches(),
                dimensions: self.dimensions(),
                weighted: !matches!(self.weights, WeightData::None),
            })
        }
    }
    impl Sealed for (scioxide_core::StridedLayout<'_>, Axis) {
        fn shape(self) -> Result<Shape, SciOxideError> {
            let (layout, axis) = self;
            let (samples, batches) = match axis {
                Axis::All => (layout.shape().len(), 1),
                Axis::Index(i) => (
                    layout.shape().extents().get(i).copied().ok_or_else(|| {
                        crate::error_helpers::invalid(
                            scioxide_core::argument!("axis"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "invalid sample axis",
                        )
                    })?,
                    layout.axis(i)?.len(),
                ),
            };
            Ok(Shape {
                samples,
                batches,
                dimensions: 1,
                weighted: false,
            })
        }
    }
}
/// The shape that [`StatisticsWorkspace::workspace_requirements`] sizes a
/// workspace for: a [`BatchView`], or a `(StridedLayout, Axis)` pair when the
/// values are not yet available.
///
/// A pair describes an unweighted view. This trait is sealed.
pub trait StatisticsGeometry: geometry::Sealed {}
impl<T: geometry::Sealed> StatisticsGeometry for T {}

/// Workspace size for [`StatisticsWorkspace`], returned by its requirement
/// queries.
#[derive(Clone, Copy, Debug)]
pub struct StatisticsRequirements<T: NumericScalar> {
    scalar: core::marker::PhantomData<T>,
    samples: usize,
    gathered: usize,
    outputs: usize,
    transforms: usize,
    rank_slots: usize,
    workers: usize,
    max_bytes: usize,
}
impl<T: NumericScalar> StatisticsRequirements<T> {
    /// Returns the largest number of observations one worker gathers.
    pub const fn samples(self) -> usize {
        self.samples
    }
    /// Returns the number of output values.
    pub const fn outputs(self) -> usize {
        self.outputs
    }
    /// Returns the number of workers, chosen from the execution policy and
    /// the execution context.
    pub const fn workers(self) -> usize {
        self.workers
    }
    /// Returns the heap size in bytes of a workspace with these requirements,
    /// not counting the output slice.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn required_bytes(self) -> Result<usize, SciOxideError> {
        let worker = crate::estimation::storage_bytes(&[
            (self.gathered, size_of::<T>() * 2),
            (self.gathered, size_of::<T::Real>()),
            (self.gathered, size_of::<u64>()),
            (
                1,
                crate::RawDescriptiveWorkspace::<T::Real>::required_bytes(self.samples)?,
            ),
        ])?;
        crate::estimation::storage_bytes(&[
            (self.workers, size_of::<Worker<T>>()),
            (self.workers, worker),
            (self.outputs, size_of::<StatisticValue<T>>()),
            (self.transforms, size_of::<T>()),
            (self.rank_slots, size_of::<Option<crate::Rank>>()),
        ])
    }
}

struct Worker<T: NumericScalar> {
    values: Vec<T>,
    masses: Vec<T::Real>,
    counts: Vec<u64>,
    descriptive: crate::RawDescriptiveWorkspace<T::Real>,
    standardize: crate::StandardizationWorkspace<T>,
}
/// Stores reusable scratch for [`statistics_into`], [`rolling_statistics_into`],
/// [`standardize_axes_into`], [`rank_axes_into`], [`rank_slice_into`] and
/// [`sigma_clip_into`]. Also supports its methods [`evaluate`](Self::evaluate),
/// [`trim_into`](Self::trim_into),
/// [`geometric_zscore_into`](Self::geometric_zscore_into),
/// [`obrien_transform_into`](Self::obrien_transform_into).
///
/// Size it with [`Self::workspace_requirements`] for a view, or with
/// [`Self::scalar_requirements`] for single slices, then create it with
/// [`Self::workspace`] or grow it with [`Self::try_reserve`]. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct StatisticsWorkspace<T: NumericScalar> {
    solve: crate::execution::SolveSlot<()>,
    workers: Vec<Worker<T>>,
    stage: Vec<StatisticValue<T>>,
    transforms: Vec<T>,
    ranks: Vec<Option<crate::Rank>>,
    reserved: StatisticsRequirements<T>,
}
impl<T: NumericScalar> Default for StatisticsWorkspace<T> {
    fn default() -> Self {
        Self {
            solve: Default::default(),
            workers: Vec::new(),
            stage: Vec::new(),
            transforms: Vec::new(),
            ranks: Vec::new(),
            reserved: StatisticsRequirements {
                scalar: core::marker::PhantomData,
                samples: 0,
                gathered: 0,
                outputs: 0,
                transforms: 0,
                rank_slots: 0,
                workers: 0,
                max_bytes: usize::MAX,
            },
        }
    }
}
impl<T: NumericScalar> StatisticsWorkspace<T> {
    /// Returns the workspace size for `geometry`, for [`statistics_into`]
    /// when `window` is `None` and for [`rolling_statistics_into`] otherwise.
    ///
    /// Requirements without a window also cover [`standardize_axes_into`] and
    /// [`rank_axes_into`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the view has vector observations, if the axis of a
    ///   `(StridedLayout, Axis)` pair is out of range, or if the window uses
    ///   [`RollingBoundary::ZeroPad`] with a weighted view.
    /// - `SizeOverflow` if the window width or the workspace size overflows.
    /// - `ResourceLimit` if the size exceeds the byte limit of `options`.
    pub fn workspace_requirements<G: StatisticsGeometry>(
        geometry: G,
        window: Option<RollingWindow>,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<StatisticsRequirements<T>, SciOxideError> {
        let view = geometry::Sealed::shape(geometry)?;
        if view.dimensions != 1 {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "scalar statistics require one coordinate",
            ));
        }
        if window.is_some_and(|w| w.boundary == RollingBoundary::ZeroPad) && view.weighted {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("window"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "zero padding requires unweighted samples",
            ));
        }
        let samples = window.map_or(Ok(view.samples), |w| {
            w.width().map(|n| {
                if w.boundary == RollingBoundary::Truncate {
                    n.min(view.samples)
                } else {
                    n
                }
            })
        })?;
        let elements = view.batches.checked_mul(view.samples).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_output"))
        })?;
        let outputs = if window.is_some() {
            elements
        } else {
            view.batches
        };
        let transforms = elements;
        let rank_slots = if window.is_some() { 0 } else { elements };
        let workers = crate::estimation::worker_count(options, context, view.batches)?;
        let req = StatisticsRequirements {
            scalar: core::marker::PhantomData,
            samples,
            gathered: samples,
            outputs,
            transforms,
            rank_slots,
            workers,
            max_bytes: options.max_bytes(),
        };
        limit_bytes(req.required_bytes()?, req.max_bytes)?;
        Ok(req)
    }
    /// Returns the workspace size for [`Self::evaluate`] and the scalar
    /// transforms on slices of at most `samples` observations.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the workspace size overflows.
    /// - `ResourceLimit` if the size exceeds the byte limit of `options`.
    pub fn scalar_requirements(
        samples: usize,
        options: StatisticsOptions,
    ) -> Result<StatisticsRequirements<T>, SciOxideError> {
        let req = StatisticsRequirements {
            scalar: core::marker::PhantomData,
            samples,
            gathered: 0,
            outputs: 0,
            transforms: 0,
            rank_slots: 0,
            workers: 1,
            max_bytes: options.max_bytes(),
        };
        limit_bytes(req.required_bytes()?, req.max_bytes)?;
        Ok(req)
    }
    /// Returns whether the workspace is large enough for `req`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage the workspace holds exceeds the byte
    ///   limit of `req`.
    pub fn check(&self, req: &StatisticsRequirements<T>) -> Result<Capacity, SciOxideError> {
        if req.max_bytes != usize::MAX {
            limit_bytes(self.reserved.required_bytes()?, req.max_bytes)?;
        }
        Ok(
            if (req.workers == 0
                || self.reserved.samples >= req.samples && self.reserved.gathered >= req.gathered)
                && self.reserved.outputs >= req.outputs
                && self.reserved.transforms >= req.transforms
                && self.reserved.rank_slots >= req.rank_slots
                && self.reserved.workers >= req.workers
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to satisfy `req`. A workspace that is already large
    /// enough is left as it is.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the held or required storage exceeds the byte
    ///   limit of `req`.
    /// - `SizeOverflow` if the workspace size overflows.
    /// - `AllocationFailure` if allocation fails.
    pub fn try_reserve(&mut self, req: &StatisticsRequirements<T>) -> Result<(), SciOxideError> {
        if self.check(req)? == Capacity::Sufficient {
            return Ok(());
        }
        limit_bytes(req.required_bytes()?, req.max_bytes)?;
        let mut workers =
            try_with_capacity(req.workers, scioxide_core::quantity!("statistics_workers"))?;
        for _ in 0..req.workers {
            let mut descriptive = crate::RawDescriptiveWorkspace::default();
            descriptive.reserve(req.samples)?;
            let mut standardize = crate::StandardizationWorkspace::default();
            standardize.reserve(req.gathered)?;
            workers.push(Worker {
                values: try_filled(
                    req.gathered,
                    T::ZERO,
                    scioxide_core::quantity!("statistics_values"),
                )?,
                masses: try_filled(
                    req.gathered,
                    <T::Real as scioxide_core::AlgebraicScalar>::ZERO,
                    scioxide_core::quantity!("statistics_masses"),
                )?,
                counts: try_filled(
                    req.gathered,
                    0,
                    scioxide_core::quantity!("statistics_counts"),
                )?,
                descriptive,
                standardize,
            });
        }
        let mut next = Self {
            solve: Default::default(),
            workers,
            stage: try_filled(
                req.outputs,
                StatisticValue::Location(T::ZERO),
                scioxide_core::quantity!("statistics_stage"),
            )?,
            transforms: try_filled(
                req.transforms,
                T::ZERO,
                scioxide_core::quantity!("statistics_transforms"),
            )?,
            ranks: try_filled(
                req.rank_slots,
                None,
                scioxide_core::quantity!("statistics_ranks"),
            )?,
            reserved: *req,
        };
        next.solve = core::mem::take(&mut self.solve);
        *self = next;
        Ok(())
    }
    /// Creates a workspace that satisfies `req`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::try_reserve`].
    pub fn workspace(req: &StatisticsRequirements<T>) -> Result<Self, SciOxideError> {
        let mut ws = Self::default();
        ws.try_reserve(req)?;
        Ok(ws)
    }
    /// Returns the longest slice that [`Self::evaluate`] and the scalar
    /// transforms accept.
    pub const fn capacity(&self) -> usize {
        self.reserved.samples
    }
    /// Computes one statistic of a slice, as [`statistic`](fn@crate::statistic)
    /// does, using this workspace's scratch.
    ///
    /// Statistics that need no scratch (sums, means, variances, standard
    /// deviations, extrema, geometric, harmonic and power means, circular
    /// statistics and entropy) accept any length.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of weights differs from
    ///   `values.len()`, or if the statistic needs scratch and `values` is
    ///   longer than [`Self::capacity`].
    /// - The errors of [`statistic`](fn@crate::statistic) for the observations
    ///   and the statistic.
    #[inline]
    pub fn evaluate<'a, W: crate::BatchWeights<'a, T::Real>>(
        &mut self,
        values: &[T],
        weights: W,
        request: Statistic<T, W::Model>,
    ) -> Result<StatisticValue<T>, SciOxideError> {
        let weights = crate::weights::batch_sealed::Sealed::data(weights);
        if let Some(len) = weights.len()
            && len != values.len()
        {
            return Err(SciOxideError::dimension_mismatch([values.len()], [len]));
        }
        if matches!(
            request,
            Statistic::Sum(_)
                | Statistic::Mean(_)
                | Statistic::Variance(..)
                | Statistic::StandardDeviation(..)
                | Statistic::Minimum(_)
                | Statistic::Maximum(_)
                | Statistic::GeometricMean(_)
                | Statistic::HarmonicMean(_)
                | Statistic::PowerMean(..)
                | Statistic::Circular(..)
                | Statistic::Entropy(..)
        ) {
            if let Some(worker) = self.workers.first_mut() {
                return crate::statistic::evaluate(
                    values,
                    weights,
                    request,
                    &mut worker.descriptive,
                );
            }
            return crate::statistic::evaluate(
                values,
                weights,
                request,
                &mut crate::RawDescriptiveWorkspace::default(),
            );
        }
        if self.reserved.samples < values.len() || self.workers.is_empty() {
            return Err(SciOxideError::dimension_mismatch(
                [values.len(), 1],
                [self.reserved.samples, self.workers.len()],
            ));
        }
        crate::statistic::evaluate(values, weights, request, &mut self.workers[0].descriptive)
    }
}

pub(crate) fn admit<T: NumericScalar, M: WeightModel>(
    view: BatchView<'_, T, M>,
    batch: usize,
    i: usize,
    options: StatisticsOptions,
) -> Result<Option<T>, SciOxideError> {
    admit_at_origin(view, view.batch_origin(batch), batch, i, options)
}

#[inline]
pub(crate) fn admit_at_origin<T: NumericScalar, M: WeightModel>(
    view: BatchView<'_, T, M>,
    origin: usize,
    batch: usize,
    i: usize,
    options: StatisticsOptions,
) -> Result<Option<T>, SciOxideError> {
    let index = batch * view.samples() + i;
    if view.validity.is_some_and(|mask| !mask[index]) {
        return if options.missing_policy() == MissingPolicy::Skip {
            Ok(None)
        } else {
            Err(crate::error_helpers::invalid(
                scioxide_core::argument!("view"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "missing observation",
            ))
        };
    }
    let active = if M::IS_WEIGHTED {
        match view.weights {
            WeightData::None => true,
            WeightData::Frequency(w) => w[index] != 0,
            WeightData::Mass(w) | WeightData::Reliability(w) | WeightData::Analytic(w) => {
                let mass = w[index];
                if !mass.is_finite() || mass < <T::Real as scioxide_core::AlgebraicScalar>::ZERO {
                    return Err(crate::error_helpers::invalid(
                        scioxide_core::argument!("weights"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "invalid observation mass",
                    ));
                }
                mass != <T::Real as scioxide_core::AlgebraicScalar>::ZERO
            }
        }
    } else {
        true
    };
    if !active {
        return Ok(None);
    }
    let value = view.value_at_origin(origin, i, 0);
    if !value.has_finite_components() {
        return if options.non_finite_policy() == NonFinitePolicy::Skip {
            Ok(None)
        } else {
            Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "nonfinite observation",
            ))
        };
    }
    Ok(Some(value))
}

/// Computes one statistic for each batch of a view.
///
/// `output[b]` receives the statistic of batch `b`, so `output` has
/// `view.batches()` entries. Observations marked missing by the mask, or
/// non-finite, are rejected or skipped according to `options`, and so are
/// observations with zero weight. Batches run in parallel when the execution
/// policy and context allow it; each batch gives the same result either way.
///
/// # Errors
///
/// - `InvalidInput` if the view has vector observations.
/// - `DimensionMismatch` if `output.len()` differs from `view.batches()`, or
///   the workspace is smaller than
///   [`StatisticsWorkspace::workspace_requirements`] reports.
/// - `InvalidInput` for a missing observation under
///   [`MissingPolicy::Reject`], or a negative or non-finite weight.
/// - `NonFinite` for a NaN or infinite observation under
///   [`NonFinitePolicy::Reject`].
/// - The errors of [`statistic`](fn@crate::statistic) for each batch.
/// - `ExecutionStopped` if the execution context is cancelled; see
///   [Execution contexts](scioxide_core#execution-contexts).
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, StridedLayout};
/// use scioxide_stats::{
///     Axis, BatchView, Statistic, StatisticValue, StatisticsOptions, StatisticsWorkspace,
///     statistics_into,
/// };
///
/// // Row means of the 2×3 array [[1, 2, 3], [4, 5, 6]] are 2 and 5.
/// let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let view = BatchView::new(&values, StridedLayout::new(&[2, 3], &[3, 1])?, Axis::Index(1))?;
/// let (options, context) = (StatisticsOptions::new(), ExecutionContext::sequential());
/// let req = StatisticsWorkspace::workspace_requirements(view, None, options, context)?;
/// let mut workspace = StatisticsWorkspace::workspace(&req)?;
/// let mut means = [StatisticValue::Location(0.0); 2];
/// statistics_into(view, Statistic::mean(), options, &mut means, &mut workspace, context)?;
/// assert!((means[0].scalar()? - 2.0).abs() < 1e-12 && (means[1].scalar()? - 5.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn statistics_into<T: NumericScalar, M: WeightModel>(
    view: BatchView<'_, T, M>,
    request: Statistic<T, M>,
    options: StatisticsOptions,
    output: &mut [StatisticValue<T>],
    workspace: &mut StatisticsWorkspace<T>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    run(view, request, None, options, output, workspace, context)
}

/// Computes a statistic over a rolling window around each observation.
///
/// `output` has one entry per observation, `view.samples() * view.batches()`
/// in total, in batch-major order: entry `b·n + i` is the statistic of the
/// window around observation `i` of batch `b`. [`RollingWindow`] sets the
/// window and its boundary handling. Missing, non-finite and zero-weight
/// observations are handled as in [`statistics_into`] and shorten the window.
///
/// Unweighted means, variances and standard deviations are updated
/// incrementally with compensated sums, at amortized O(1) cost per window.
/// The window is recomputed directly every `min(w, n)` windows, where `w` is
/// the window width, and whenever a running rounding-error bound exceeds
/// `64·ε` (`ε = 2⁻⁵²`) relative to the window sum (mean) or the centered sum
/// of squares (variance). Windows with a nonzero component outside
/// `2⁻⁶⁴ ≤ |x| < 2⁶⁵` are also computed directly. The unweighted median keeps
/// the window sorted and costs O(w) per window. Other statistics are computed
/// directly for each window.
///
/// # Errors
///
/// - `InvalidInput` if the view has vector observations, or the window uses
///   [`RollingBoundary::ZeroPad`] with a weighted view.
/// - `DimensionMismatch` if `output.len()` differs from the number of
///   observations, or the workspace is smaller than
///   [`StatisticsWorkspace::workspace_requirements`] reports for `window`.
/// - `InvalidInput` if a window has too few observations for the statistic,
///   such as `n ≤ ddof` for a variance.
/// - The observation and statistic errors of [`statistics_into`].
#[allow(clippy::too_many_arguments)]
pub fn rolling_statistics_into<T: NumericScalar, M: WeightModel>(
    view: BatchView<'_, T, M>,
    request: Statistic<T, M>,
    window: RollingWindow,
    options: StatisticsOptions,
    output: &mut [StatisticValue<T>],
    workspace: &mut StatisticsWorkspace<T>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    run(
        view,
        request,
        Some(window),
        options,
        output,
        workspace,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn run<T: NumericScalar, M: WeightModel>(
    view: BatchView<'_, T, M>,
    request: Statistic<T, M>,
    window: Option<RollingWindow>,
    options: StatisticsOptions,
    output: &mut [StatisticValue<T>],
    workspace: &mut StatisticsWorkspace<T>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let req = StatisticsWorkspace::workspace_requirements(view, window, options, context)?;
    if output.len() != req.outputs {
        return Err(SciOxideError::dimension_mismatch(
            [req.outputs],
            [output.len()],
        ));
    }
    if workspace.check(&req)? != Capacity::Sufficient {
        return Err(SciOxideError::dimension_mismatch(
            &[req.samples, req.outputs, req.workers],
            &[
                workspace.reserved.samples,
                workspace.reserved.outputs,
                workspace.reserved.workers,
            ],
        ));
    }
    let width = if window.is_some() { view.samples() } else { 1 };
    if width == 0 {
        context.check()?;
        return Ok(());
    }
    // Stage arithmetic in its scalar domain and publish the result enum once.
    if let Some(window) = window
        && !M::IS_WEIGHTED
    {
        let numeric = match request {
            Statistic::Mean(_) => Some(crate::ArithmeticStatistic::Mean),
            Statistic::Variance(d, _) => Some(crate::ArithmeticStatistic::Variance(d)),
            Statistic::StandardDeviation(d, _) => {
                Some(crate::ArithmeticStatistic::StandardDeviation(d))
            }
            _ => None,
        };
        if let Some(numeric) = numeric {
            crate::estimation::execute(
                options,
                context,
                view.batches(),
                width,
                &mut workspace.workers,
                &mut workspace.transforms,
                |batch, worker, out| {
                    let origin = view.batch_origin(batch);
                    // Unmasked native lanes need no repeated admission protocol
                    // on their entering/leaving reads. Prove the whole borrowed
                    // lane once, with the same source frame as rolling state.
                    let mut native_lane = view.validity.is_none();
                    if native_lane {
                        for start in (0..view.samples()).step_by(256) {
                            context.check()?;
                            native_lane = (start..start.saturating_add(256).min(view.samples()))
                                .fold(true, |admitted, i| {
                                    let value = view.value_at_origin(origin, i, 0);
                                    admitted
                                        & crate::arithmetic::native(
                                            value.real_component().into_f64(),
                                            64,
                                        )
                                        & (T::IS_REAL
                                            || crate::arithmetic::native(
                                                value.imaginary_component().into_f64(),
                                                64,
                                            ))
                                });
                            if !native_lane {
                                break;
                            }
                        }
                    }
                    crate::rolling::run(
                        view.samples(),
                        window,
                        numeric,
                        context,
                        &mut worker.values,
                        out,
                        |i| {
                            if native_lane {
                                Ok(Some(view.value_at_origin(origin, i, 0)))
                            } else {
                                admit_at_origin(view, origin, batch, i, options)
                            }
                        },
                        |values| crate::numeric_statistic(values, numeric),
                        |value| match value {
                            crate::ArithmeticValue::Location(x) => x,
                            crate::ArithmeticValue::Dispersion(x) => T::from_real(x),
                        },
                    )
                },
            )?;
            context.check()?;
            if matches!(numeric, crate::ArithmeticStatistic::Mean) {
                for (out, &value) in output.iter_mut().zip(&workspace.transforms[..req.outputs]) {
                    *out = StatisticValue::Location(value);
                }
            } else {
                for (out, &value) in output.iter_mut().zip(&workspace.transforms[..req.outputs]) {
                    *out = StatisticValue::Dispersion(value.real_component());
                }
            }
            return Ok(());
        }
    }
    crate::estimation::execute(
        options,
        context,
        view.batches(),
        width,
        &mut workspace.workers,
        &mut workspace.stage,
        |batch, worker, out| {
            let origin = view.batch_origin(batch);
            if let Some(window) = window
                && matches!(view.weights, WeightData::None)
                && let Statistic::Quantile(p, rule, _) = request
                && p.get() == 0.5
            {
                return crate::rolling_median::run(
                    view.samples(),
                    window,
                    rule.method,
                    context,
                    &mut worker.masses,
                    out,
                    |i| {
                        admit_at_origin(view, origin, batch, i, options)
                            .map(|x| x.map(NumericScalar::real_component))
                    },
                    |value| {
                        StatisticValue::Order(crate::OrderResult::Quantile(
                            crate::QuantileResult::unweighted(value, p, rule.method),
                        ))
                    },
                );
            }
            for (center, result) in out.iter_mut().enumerate() {
                context.check()?;
                let mut len = 0;
                let (start, end) = match window {
                    Some(w) if w.boundary == RollingBoundary::Truncate => (
                        center.saturating_sub(w.left),
                        center
                            .saturating_add(w.right)
                            .saturating_add(1)
                            .min(view.samples()),
                    ),
                    Some(w) => (0, w.width()?),
                    None => (0, view.samples()),
                };
                for position in start..end {
                    if position % 256 == 0 {
                        context.check()?;
                    }
                    let index = match window {
                        Some(w) if w.boundary != RollingBoundary::Truncate => {
                            w.index(center, position, view.samples())
                        }
                        _ => Some(position),
                    };
                    if let Some(i) = index {
                        let Some(value) = admit_at_origin(view, origin, batch, i, options)? else {
                            continue;
                        };
                        worker.values[len] = value;
                        let index = batch * view.samples() + i;
                        worker.counts[len] = if let WeightData::Frequency(w) = view.weights {
                            w[index]
                        } else {
                            1
                        };
                        worker.masses[len] = match view.weights {
                            WeightData::Mass(w)
                            | WeightData::Reliability(w)
                            | WeightData::Analytic(w) => w[index],
                            _ => <T::Real as scioxide_core::AlgebraicScalar>::ONE,
                        };
                    } else {
                        worker.values[len] = T::ZERO;
                        worker.counts[len] = 1;
                        worker.masses[len] = <T::Real as scioxide_core::AlgebraicScalar>::ONE;
                    }
                    len += 1;
                }
                let weights = match view.weights {
                    WeightData::None => WeightData::None,
                    WeightData::Frequency(_) => WeightData::Frequency(&worker.counts[..len]),
                    WeightData::Mass(_) => WeightData::Mass(&worker.masses[..len]),
                    WeightData::Reliability(_) => WeightData::Reliability(&worker.masses[..len]),
                    WeightData::Analytic(_) => WeightData::Analytic(&worker.masses[..len]),
                };
                *result = crate::statistic::evaluate(
                    &worker.values[..len],
                    weights,
                    request,
                    &mut worker.descriptive,
                )?;
            }
            Ok(())
        },
    )?;
    context.check()?;
    output.copy_from_slice(&workspace.stage[..req.outputs]);
    Ok(())
}

/// Computes z-scores `zᵢ = (xᵢ − x̄) / s` within each batch of a view.
///
/// `x̄` is the batch mean and `s` the standard deviation with divisor
/// `n − ddof`. For complex data, `s² = Σ |xᵢ − x̄|² / (n − ddof)`, and the real
/// and imaginary parts of `xᵢ − x̄` are both divided by `s`. `output` is in
/// batch-major order: entry `b·n + i` holds observation `i` of batch `b`.
///
/// # Errors
///
/// - `InvalidInput` if `options` skips missing or non-finite observations,
///   since every observation needs a score, or if an observation is marked
///   missing.
/// - `InvalidInput` if the batches are empty, `n ≤ ddof`, or a batch has zero
///   standard deviation.
/// - `NonFinite` for a NaN or infinite observation.
/// - `DimensionMismatch` if `output.len()` differs from the number of
///   observations, or the workspace is too small.
/// - `NumericalInstability` (`Overflow`) if a score overflows.
pub fn standardize_axes_into<T: NumericScalar>(
    view: BatchView<'_, T>,
    ddof: usize,
    options: StatisticsOptions,
    output: &mut [T],
    workspace: &mut StatisticsWorkspace<T>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let req = StatisticsWorkspace::workspace_requirements(view, None, options, context)?;
    if output.len() != req.transforms || workspace.check(&req)? != Capacity::Sufficient {
        return Err(SciOxideError::dimension_mismatch(
            [req.transforms],
            [output.len()],
        ));
    }
    if options.non_finite_policy() != NonFinitePolicy::Reject
        || options.missing_policy() != MissingPolicy::Reject
    {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("options"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "shape-preserving standardization requires strict admission",
        ));
    }
    if view.samples() == 0 && view.batches() != 0 {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("view"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty standardization lane",
        ));
    }
    if view.samples() == 0 {
        context.check()?;
        return Ok(());
    }
    crate::estimation::execute(
        options,
        context,
        view.batches(),
        view.samples(),
        &mut workspace.workers,
        &mut workspace.transforms,
        |batch, worker, out| {
            let origin = view.batch_origin(batch);
            for (i, slot) in worker.values[..view.samples()].iter_mut().enumerate() {
                *slot = admit_at_origin(view, origin, batch, i, options)?.ok_or_else(|| {
                    crate::error_helpers::invalid(
                        scioxide_core::argument!("view"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "omitted observation",
                    )
                })?;
            }
            worker.standardize.standardize_into(
                &worker.values[..view.samples()],
                &worker.values[..view.samples()],
                ddof,
                out,
            )
        },
    )?;
    context.check()?;
    output.copy_from_slice(&workspace.transforms[..req.transforms]);
    Ok(())
}

/// Computes the ranks of the observations within each batch of a view.
///
/// `output` has the shape of the input in row-major order: each rank is at the
/// position of its observation. Ranks start at 1, and `method` sets how ties
/// are ranked; [`RankMethod::Ordinal`](crate::RankMethod::Ordinal) breaks ties
/// in sample order. Observations skipped by the missing or non-finite policy
/// get `None` and are not counted.
///
/// # Errors
///
/// - `InvalidInput` if the view has vector observations, or an observation is
///   marked missing under [`MissingPolicy::Reject`].
/// - `NonFinite` for a NaN or infinite observation under
///   [`NonFinitePolicy::Reject`].
/// - `DimensionMismatch` if `output.len()` differs from the number of
///   observations, or the workspace is too small.
pub fn rank_axes_into<S: RealScalar>(
    view: BatchView<'_, S>,
    method: crate::RankMethod,
    options: StatisticsOptions,
    output: &mut [Option<crate::Rank>],
    workspace: &mut StatisticsWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let req = StatisticsWorkspace::workspace_requirements(view, None, options, context)?;
    if output.len() != req.transforms || workspace.check(&req)? != Capacity::Sufficient {
        return Err(SciOxideError::dimension_mismatch(
            [req.transforms],
            [output.len()],
        ));
    }
    if view.samples() == 0 {
        context.check()?;
        return Ok(());
    }
    crate::estimation::execute(
        options,
        context,
        view.batches(),
        view.samples(),
        &mut workspace.workers,
        &mut workspace.ranks,
        |batch, worker, out| {
            let origin = view.batch_origin(batch);
            let order = &mut worker.descriptive.order;
            order.begin(view.samples(), scioxide_core::argument!("workspace"))?;
            for i in 0..view.samples() {
                if let Some(value) = admit_at_origin(view, origin, batch, i, options)? {
                    order.push(value, S::ONE, i, scioxide_core::argument!("view"))?;
                }
            }
            order.sort(scioxide_core::argument!("view"))?;
            order.ranks[..view.samples()].fill(None);
            order.rank_sorted(method);
            out.copy_from_slice(&order.ranks[..view.samples()]);
            Ok(())
        },
    )?;
    context.check()?;
    if let Axis::Index(axis) = view.axis {
        let trailing = view.layout.shape().extents()[axis + 1..].iter().product();
        for batch in 0..view.batches() {
            for i in 0..view.samples() {
                output[crate::order_axis::logical_index(batch, i, view.samples(), trailing)] =
                    workspace.ranks[batch * view.samples() + i];
            }
        }
    } else {
        output.copy_from_slice(&workspace.ranks[..output.len()]);
    }
    Ok(())
}

/// Computes the ranks of a slice, using the workspace's scratch.
///
/// `output[i]` receives the rank of `values[i]`. Ranks start at 1, and
/// `method` sets how ties are ranked.
///
/// # Errors
///
/// - `DimensionMismatch` if `output.len()` differs from `values.len()`, or
///   `values` is longer than [`StatisticsWorkspace::capacity`].
/// - An error if a value is NaN or infinite.
pub fn rank_slice_into<S: RealScalar>(
    values: &[S],
    method: crate::RankMethod,
    output: &mut [Option<crate::Rank>],
    workspace: &mut StatisticsWorkspace<S>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if output.len() != values.len() || workspace.capacity() < values.len() {
        return Err(SciOxideError::dimension_mismatch(
            [values.len()],
            [output.len()],
        ));
    }
    context.check()?;
    let worker = workspace
        .workers
        .first_mut()
        .ok_or_else(|| SciOxideError::dimension_mismatch([1], [0]))?;
    worker.descriptive.order.load(values)?;
    worker.descriptive.order.rank_sorted(method);
    context.check()?;
    let ranks = &worker.descriptive.order.ranks[..values.len()];
    if ranks.iter().any(Option::is_none) {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "incomplete ranks",
        ));
    }
    output.copy_from_slice(ranks);
    Ok(())
}

/// Returns the ranks of a slice in a new vector.
///
/// Entry `i` is the rank of `values[i]`. Ranks start at 1, and `method` sets
/// how ties are ranked. Average ranks of ties are half-integers, which
/// [`Rank`](crate::Rank) stores exactly.
///
/// # Errors
///
/// - An error if `values` is empty or contains NaN or infinity.
/// - `AllocationFailure` if the output cannot be allocated.
pub fn rank_slice<S: RealScalar>(
    values: &[S],
    method: crate::RankMethod,
) -> Result<Vec<crate::Rank>, SciOxideError> {
    crate::order::rankdata(values, method)
}

impl<S: RealScalar> StatisticsWorkspace<S> {
    fn scalar_worker(&mut self) -> Result<&mut crate::RawDescriptiveWorkspace<S>, SciOxideError> {
        self.workers
            .first_mut()
            .map(|w| &mut w.descriptive)
            .ok_or_else(|| SciOxideError::dimension_mismatch([1], [0]))
    }
    /// Trims a sample and writes the kept observations, sorted, to the front of
    /// `output`.
    ///
    /// `⌊proportion · n⌋` observations are removed from each end chosen by
    /// `side`. Returns the number of observations written.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `proportion` is not in `[0, 1)`.
    /// - `InvalidInput` if `values` is empty, trimming removes every
    ///   observation, or `output` or the scratch is too short.
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `DimensionMismatch` if the workspace has no scratch.
    pub fn trim_into(
        &mut self,
        values: &[S],
        proportion: S,
        side: crate::TrimSide,
        output: &mut [S],
    ) -> Result<usize, SciOxideError> {
        self.scalar_worker()?
            .trim_into(values, proportion, side, output)
    }
    /// Computes geometric z-scores `zᵢ = (log xᵢ − m) / s`, where `m` and `s`
    /// are the mean and the standard deviation (divisor `n − ddof`) of the
    /// `log xᵢ`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if an observation is not positive.
    /// - `InvalidInput` if `values` is empty, `n ≤ ddof`, the logarithms have
    ///   zero standard deviation, `output.len()` differs from `values.len()`,
    ///   or the scratch is too short.
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `DimensionMismatch` if the workspace has no scratch.
    pub fn geometric_zscore_into(
        &mut self,
        values: &[S],
        ddof: usize,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.scalar_worker()?
            .geometric_zscore_into(values, ddof, output)
    }
    /// Computes O'Brien's transform of one group, used to test equality of
    /// variances.
    ///
    /// ```text
    /// rᵢ = (n (n − 1.5) (xᵢ − x̄)² − 0.5 (n − 1) s²) / ((n − 1)(n − 2))
    /// ```
    ///
    /// where `x̄` is the group mean and `s²` the unbiased sample variance. The
    /// mean of the `rᵢ` equals `s²`, so an analysis of variance of the
    /// transformed groups tests equality of the group variances. Call it once
    /// per group.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `n < 3`, `output.len()` differs from `values.len()`,
    ///   or the scratch is too short.
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `DimensionMismatch` if the workspace has no scratch.
    ///
    /// # References
    ///
    /// R. G. O'Brien, "A simple test for variance effects in experimental
    /// designs", *Psychological Bulletin* 89(3), 570–574, 1981.
    pub fn obrien_transform_into(
        &mut self,
        values: &[S],
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.scalar_worker()?.obrien_transform_into(values, output)
    }
}

/// Computes axis statistics with the [one-shot defaults](crate#one-shot-forms).
///
/// # Errors
///
/// Returns the errors of [`statistics_with_options`].
///
/// ```
/// use scioxide_core::StridedLayout;
/// use scioxide_stats::{Axis, BatchView, Statistic, statistics};
/// let values = [1.0_f64, 2.0, 3.0, 10.0, 20.0, 30.0];
/// let view = BatchView::new(&values, StridedLayout::new(&[2, 3], &[3, 1])?, Axis::Index(1))?;
/// let means = statistics(view, Statistic::mean())?;
/// assert!((means[1].scalar()? - 20.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn statistics<T: NumericScalar, M: WeightModel>(
    view: BatchView<'_, T, M>,
    request: Statistic<T, M>,
) -> Result<Vec<StatisticValue<T>>, SciOxideError> {
    statistics_with_options(view, request, StatisticsOptions::default())
}
/// Computes axis statistics into an owned vector.
///
/// Uses [`statistics_into`] with the same cost and accuracy.
///
/// # Errors
///
/// Returns the kernel's input errors, allocation failures, and memory-limit errors.
pub fn statistics_with_options<T: NumericScalar, M: WeightModel>(
    view: BatchView<'_, T, M>,
    request: Statistic<T, M>,
    options: StatisticsOptions,
) -> Result<Vec<StatisticValue<T>>, SciOxideError> {
    let context = ExecutionContext::sequential_native();
    let req = StatisticsWorkspace::workspace_requirements(view, None, options, context)?;
    let mut workspace = StatisticsWorkspace::workspace(&req)?;
    let mut output = try_filled(
        req.outputs(),
        StatisticValue::Location(T::ZERO),
        scioxide_core::quantity!("statistics_output"),
    )?;
    statistics_into(view, request, options, &mut output, &mut workspace, context)?;
    Ok(output)
}

/// Computes rolling statistics with the [one-shot defaults](crate#one-shot-forms).
///
/// # Errors
///
/// Returns the errors of [`rolling_statistics_with_options`].
pub fn rolling_statistics<T: NumericScalar, M: WeightModel>(
    view: BatchView<'_, T, M>,
    request: Statistic<T, M>,
    window: RollingWindow,
) -> Result<Vec<StatisticValue<T>>, SciOxideError> {
    rolling_statistics_with_options(view, request, window, StatisticsOptions::default())
}
/// Computes rolling statistics into an owned vector.
///
/// Uses [`rolling_statistics_into`] with the same cost and accuracy.
///
/// # Errors
///
/// Returns the kernel's input errors, allocation failures, and memory-limit errors.
pub fn rolling_statistics_with_options<T: NumericScalar, M: WeightModel>(
    view: BatchView<'_, T, M>,
    request: Statistic<T, M>,
    window: RollingWindow,
    options: StatisticsOptions,
) -> Result<Vec<StatisticValue<T>>, SciOxideError> {
    let context = ExecutionContext::sequential_native();
    let req = StatisticsWorkspace::workspace_requirements(view, Some(window), options, context)?;
    let mut workspace = StatisticsWorkspace::workspace(&req)?;
    let mut output = try_filled(
        req.outputs(),
        StatisticValue::Location(T::ZERO),
        scioxide_core::quantity!("statistics_output"),
    )?;
    rolling_statistics_into(
        view,
        request,
        window,
        options,
        &mut output,
        &mut workspace,
        context,
    )?;
    Ok(output)
}

impl<T: NumericScalar> StatisticsWorkspace<T> {
    /// Returns diagnostics of the last iterative call.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.solve.last_attempt()
    }
}

/// Removes outliers by iterative sigma clipping.
///
/// Each pass computes the mean `x̄` and the population standard deviation
/// `s` of the kept observations, then keeps those with
/// `x̄ − low·s ≤ x ≤ x̄ + high·s`. Clipping stops when a pass removes
/// nothing ([`StatsTermination::Converged`](crate::StatsTermination::Converged)),
/// after the maximum number of passes in `options`
/// ([`StatsTermination::BudgetExhausted`](crate::StatsTermination::BudgetExhausted)),
/// or when `scope` is stopped. Each call owns its iteration allowance;
/// parent iteration limits do not cap the passes. Each pass counts as one iteration; see
/// [Solve accounting](scioxide_core#solve-accounting).
///
/// The kept observations, in input order, go to the front of `output`, and
/// their input indices to the front of `indices`. The returned
/// [`SigmaClipResult`](crate::SigmaClipResult) gives their number, the
/// last bounds and the termination status.
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `low` or `high` is negative or not
///   finite.
/// - `InvalidInput` if `values` is empty, a pass removes every
///   observation, or the scratch is shorter than `values`.
/// - `NonFinite` if a value is NaN or infinite.
/// - `DimensionMismatch` if `output` or `indices` is shorter than `values`,
///   or the workspace has no scratch.
#[allow(clippy::too_many_arguments)]
pub fn sigma_clip_into<'scope, 'capability, S: RealScalar>(
    values: &[S],
    low: S,
    high: S,
    options: crate::SigmaClipOptions<S>,
    output: &mut [S],
    indices: &mut [usize],
    workspace: &mut StatisticsWorkspace<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<crate::SigmaClipResult<S>, SciOxideError> {
    workspace.solve.reject();
    if output.len() < values.len() || indices.len() < values.len() {
        return Err(SciOxideError::dimension_mismatch(
            [values.len()],
            [output.len().min(indices.len())],
        ));
    }
    crate::error_helpers::sample(values, scioxide_core::argument!("values"))?;
    if !low.is_finite() || low < S::ZERO {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("low"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "invalid clipping multiplier",
        ));
    }
    if !high.is_finite() || high < S::ZERO {
        return Err(crate::error_helpers::invalid(
            scioxide_core::argument!("high"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "invalid clipping multiplier",
        ));
    }
    workspace.scalar_worker()?;
    let mut slot = core::mem::take(&mut workspace.solve);
    let result = slot.run(
        scope.into(),
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "sigma_clip", 1),
            0,
            0,
        ),
        scioxide_core::SolveBudgets::unbounded(),
        |context| {
            workspace
                .scalar_worker()?
                .sigma_clip_into(values, low, high, options, context, output, indices)
        },
        |result| {
            (
                match result {
                    Ok(report) if report.termination().is_success() => {
                        scioxide_core::AttemptOutcome::Succeeded
                    }
                    Ok(_) => scioxide_core::AttemptOutcome::Stopped,
                    Err(_) => scioxide_core::AttemptOutcome::Failed,
                },
                None,
            )
        },
    );
    workspace.solve = slot;
    result
}
