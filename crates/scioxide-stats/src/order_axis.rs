//! Borrowed observations and options for order statistics along an axis.
use crate::{MissingPolicy, NonFinitePolicy};
use core::num::NonZeroUsize;
use scioxide_core::{ExecutionPolicy, RealScalar, SciOxideError, StridedLayout};

/// Weights in logical row-major order, independent of the strides.
#[derive(Clone, Copy)]
pub(crate) enum KernelWeights<'a, S: RealScalar> {
    /// Nonnegative masses. Order statistics support the inverse empirical CDF
    /// and expectiles with masses.
    Mass(&'a [S]),
    /// Integer frequency counts. Order statistics support all nine quantile
    /// definitions with counts.
    Frequency(&'a [u64]),
}

/// Borrowed real values with a strided layout, an optional validity mask and
/// optional weights. The mask is in logical row-major order, and `true` marks
/// a present value.
#[derive(Clone, Copy)]
pub(crate) struct OrderData<'a, S: RealScalar> {
    pub(crate) values: &'a [S],
    pub(crate) layout: StridedLayout<'a>,
    pub(crate) validity: Option<&'a [bool]>,
    pub(crate) weights: Option<KernelWeights<'a, S>>,
}
impl<'a, S: RealScalar> OrderData<'a, S> {
    pub(crate) fn from_batch<M: crate::weights::EmpiricalWeights>(
        view: crate::BatchView<'a, S, M>,
    ) -> Self {
        Self {
            values: view.values,
            layout: view.layout,
            validity: view.validity,
            weights: crate::statistic::legacy_weights(view.weights),
        }
    }
}

/// Missing-value, non-finite and execution options for order statistics
/// along an axis.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OrderPolicy {
    /// The sample axis, or `None` to use all values in logical row-major order.
    pub axis: Option<usize>,
    /// Whether missing entries are rejected or skipped. Their values are not
    /// read.
    pub missing: MissingPolicy,
    /// Whether present NaN or infinite values are rejected or skipped.
    pub non_finite: NonFinitePolicy,
    /// Execution policy. Each worker processes whole lanes.
    pub execution: ExecutionPolicy,
    /// Maximum number of workers, from 1 to 64.
    pub workers: NonZeroUsize,
}

pub(crate) fn shape(
    layout: StridedLayout<'_>,
    axis: Option<usize>,
) -> Result<(usize, usize), SciOxideError> {
    match axis {
        Some(axis) => {
            let batch = layout.axis(axis)?;
            Ok((layout.shape().extents()[axis], batch.len()))
        }
        None => Ok((layout.shape().len(), 1)),
    }
}

pub(crate) fn logical_index(batch: usize, position: usize, len: usize, trailing: usize) -> usize {
    (batch / trailing) * len * trailing + batch % trailing + position * trailing
}
