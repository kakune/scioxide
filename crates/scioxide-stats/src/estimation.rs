//! Borrowed batched observations and the execution of independent batches.
use crate::error_helpers::invalid;
use crate::{KernelWeights, MissingPolicy, NonFinitePolicy, StatisticsOptions};
use core::num::NonZeroUsize;
use scioxide_core::Scaled;
use scioxide_core::backend::PartitionSlot;
use scioxide_core::{Argument, ExecutionContext, RealScalar, SciOxideError, StridedLayout};

/// Borrowed N-dimensional real observations.
///
/// The sample axis indexes observations and an optional coordinate axis
/// indexes their components. The remaining axes index independent batches in
/// logical row-major order. Masks and weights are in batch-major, sample-minor
/// order and apply to whole observations, independent of the strides.
#[derive(Clone, Copy)]
pub(crate) struct ObservationData<'a, S: RealScalar> {
    values: &'a [S],
    layout: StridedLayout<'a>,
    sample_axis: Option<usize>,
    coordinate_axis: Option<usize>,
    sample_stride: Option<usize>,
    coordinate_stride: usize,
    samples: usize,
    dimensions: usize,
    batches: usize,
    validity: Option<&'a [bool]>,
    pub(crate) weights: Option<KernelWeights<'a, S>>,
}
impl<'a, S: RealScalar> ObservationData<'a, S> {
    pub(crate) fn from_batch<M: crate::weights::EmpiricalWeights>(
        view: crate::BatchView<'a, S, M>,
    ) -> Self {
        Self {
            values: view.values,
            layout: view.layout,
            sample_axis: match view.axis {
                crate::Axis::All => None,
                crate::Axis::Index(i) => Some(i),
            },
            coordinate_axis: view.coordinate_axis,
            sample_stride: match view.axis {
                crate::Axis::All => None,
                crate::Axis::Index(i) => Some(view.layout.strides()[i]),
            },
            coordinate_stride: view.coordinate_axis.map_or(0, |i| view.layout.strides()[i]),
            samples: view.samples,
            dimensions: view.dimensions,
            batches: view.batches,
            validity: view.validity,
            weights: crate::statistic::legacy_weights(view.weights),
        }
    }
    /// Checks the layout and that the sample and coordinate axes differ.
    ///
    /// # Errors
    /// Fails for an invalid axis, a zero coordinate extent, size overflow, or
    /// a slice shorter than the layout.
    pub(crate) fn new(
        values: &'a [S],
        layout: StridedLayout<'a>,
        sample_axis: usize,
        coordinate_axis: Option<usize>,
    ) -> Result<Self, SciOxideError> {
        layout.validate_backing_len(values.len())?;
        let shape = layout.shape().extents();
        let samples = *shape.get(sample_axis).ok_or_else(|| {
            invalid(
                scioxide_core::argument!("sample_axis"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid sample axis",
            )
        })?;
        let dimensions = if let Some(axis) = coordinate_axis {
            if axis == sample_axis {
                return Err(invalid(
                    scioxide_core::argument!("coordinate_axis"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "sample and coordinate axes coincide",
                ));
            }
            *shape.get(axis).ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("coordinate_axis"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "invalid coordinate axis",
                )
            })?
        } else {
            1
        };
        if dimensions == 0 {
            return Err(invalid(
                scioxide_core::argument!("coordinate_axis"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "zero coordinate extent",
            ));
        }
        let batches = shape
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != sample_axis && Some(*axis) != coordinate_axis)
            .try_fold(1usize, |n, (_, extent)| {
                n.checked_mul(*extent).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("estimation_batches"))
                })
            })?;
        batches.checked_mul(samples).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("estimation_observations"))
        })?;
        Ok(Self {
            values,
            layout,
            sample_axis: Some(sample_axis),
            sample_stride: Some(layout.strides()[sample_axis]),
            coordinate_stride: coordinate_axis.map_or(0, |i| layout.strides()[i]),
            coordinate_axis,
            samples,
            dimensions,
            batches,
            validity: None,
            weights: None,
        })
    }

    /// Returns the number of observations per batch.
    pub(crate) const fn samples(self) -> usize {
        self.samples
    }
    /// Returns the number of coordinates per observation.
    pub(crate) const fn dimensions(self) -> usize {
        self.dimensions
    }
    /// Returns the number of independent batches.
    pub(crate) const fn batches(self) -> usize {
        self.batches
    }
    #[inline]
    pub(crate) fn value(self, batch: usize, sample: usize, coordinate: usize) -> S {
        self.value_at_origin(self.batch_origin(batch), sample, coordinate)
    }
    pub(crate) fn batch_origin(self, batch: usize) -> usize {
        if self.sample_axis.is_none() || self.samples == 0 {
            return 0;
        }
        let mut rest = batch;
        let mut offset = 0;
        for (axis, extent) in self.layout.shape().extents().iter().enumerate().rev() {
            if Some(axis) != self.sample_axis && Some(axis) != self.coordinate_axis {
                offset += (rest % extent) * self.layout.strides()[axis];
                rest /= extent;
            }
        }
        offset
    }
    #[inline(always)]
    pub(crate) fn value_at_origin(self, origin: usize, sample: usize, coordinate: usize) -> S {
        let mut offset = origin;
        if let Some(stride) = self.sample_stride {
            offset += sample * stride;
        } else {
            let mut rest = sample;
            for (&extent, &stride) in self
                .layout
                .shape()
                .extents()
                .iter()
                .zip(self.layout.strides())
                .rev()
            {
                offset += rest % extent * stride;
                rest /= extent;
            }
        }
        offset += coordinate * self.coordinate_stride;
        self.values[offset]
    }
    pub(crate) fn present(self, batch: usize, sample: usize) -> bool {
        self.validity
            .is_none_or(|mask| mask[batch * self.samples + sample])
    }
    pub(crate) fn admit_count(
        self,
        batch: usize,
        sample: usize,
        options: StatisticsOptions,
        argument: Argument,
    ) -> Result<Option<usize>, SciOxideError> {
        self.admit_count_by(batch, sample, options, argument, |d| {
            self.value(batch, sample, d)
        })
    }
    /// Returns the count of an observation (1 without frequency weights), or
    /// `None` if it is skipped. `coordinate` reads its components.
    #[inline(always)]
    pub(crate) fn admit_count_by(
        self,
        batch: usize,
        sample: usize,
        options: StatisticsOptions,
        argument: Argument,
        mut coordinate: impl FnMut(usize) -> S,
    ) -> Result<Option<usize>, SciOxideError> {
        let index = batch * self.samples + sample;
        if self.validity.is_some_and(|mask| !mask[index]) {
            return if options.missing_policy() == MissingPolicy::Skip {
                Ok(None)
            } else {
                Err(invalid(
                    argument,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "missing observation",
                ))
            };
        }
        let count = match self.weights {
            None => 1,
            Some(KernelWeights::Frequency(w)) => crate::weights::frequency_index(w[index])?,
            Some(KernelWeights::Mass(w)) => {
                if !w[index].is_finite() || w[index] < S::ZERO {
                    return Err(invalid(
                        argument,
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "invalid mass",
                    ));
                }
                usize::from(w[index] != S::ZERO)
            }
        };
        if count == 0 {
            return Ok(None);
        }
        if (0..self.dimensions).any(|d| !coordinate(d).is_finite()) {
            return if options.non_finite_policy() == NonFinitePolicy::Skip {
                Ok(None)
            } else {
                Err(invalid(
                    argument,
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "non-finite observation",
                ))
            };
        }
        Ok(Some(count))
    }

    pub(crate) fn admit(
        self,
        batch: usize,
        sample: usize,
        options: StatisticsOptions,
        argument: Argument,
    ) -> Result<Option<(Scaled<f64>, usize)>, SciOxideError> {
        self.admit_count(batch, sample, options, argument)?
            .map(|count| {
                let mass = match self.weights {
                    Some(KernelWeights::Mass(w)) => w[batch * self.samples + sample].into_f64(),
                    _ => count as f64,
                };
                Ok((Scaled::from_value(mass)?, count))
            })
            .transpose()
    }
}

pub(crate) fn worker_count(
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
    jobs: usize,
) -> Result<usize, SciOxideError> {
    Ok(context
        .plan(
            options.execution_policy(),
            jobs,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: NonZeroUsize::MIN.saturating_add(63),
            },
        )
        .worker_count())
}

/// Runs `jobs` independent jobs, each writing `width` results, on the
/// workers planned from `options` and `context`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute<W: Send, R: Send>(
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
    jobs: usize,
    width: usize,
    workers: &mut [W],
    results: &mut [R],
    run: impl Fn(usize, &mut W, &mut [R]) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    execute_ranges(
        options,
        context,
        jobs,
        width,
        workers,
        results,
        |range, worker, output| {
            for (i, job) in range.enumerate() {
                context.check()?;
                run(job, worker, &mut output[i * width..(i + 1) * width])?;
            }
            Ok(())
        },
    )
}

/// Runs whole partitions of the jobs, so a kernel can tile its work. The
/// callback checks for a stop request within its partition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_ranges<W: Send, R: Send>(
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
    jobs: usize,
    width: usize,
    workers: &mut [W],
    results: &mut [R],
    run: impl Fn(core::ops::Range<usize>, &mut W, &mut [R]) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    let count = worker_count(options, context, jobs)?;
    let size = jobs.checked_mul(width).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("estimation_output"))
    })?;
    if workers.len() < count || results.len() < size {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient workspace",
        ));
    }
    if jobs == 0 {
        return Ok(());
    }
    let plan = context.plan(
        options.execution_policy(),
        jobs,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: NonZeroUsize::MIN.saturating_add(63),
        },
    );
    let mut slots: [_; 64] = core::array::from_fn(|_| PartitionSlot::empty());
    let mut rest = &mut results[..size];
    for (partition, worker) in plan.partitions().zip(workers) {
        let range = partition.range();
        let (output, tail) = rest.split_at_mut(range.len() * width);
        rest = tail;
        slots[partition.worker_index()] = PartitionSlot::new((worker, output));
    }
    context.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, (worker, output), _| {
            context.check()?;
            run(partition.range(), worker, output)
        },
    )?;

    Ok(())
}

pub(crate) fn storage_bytes(parts: &[(usize, usize)]) -> Result<usize, SciOxideError> {
    parts.iter().try_fold(0usize, |total, (count, width)| {
        count
            .checked_mul(*width)
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("estimation_storage_bytes"))
            })
    })
}
