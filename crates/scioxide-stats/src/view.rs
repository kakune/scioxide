//! Borrowed observations with a sample axis, validity mask and weights.
use crate::error_helpers::invalid;
use crate::weights::{BatchWeights, Unweighted, WeightData, WeightModel};
use core::marker::PhantomData;
use scioxide_core::{NumericScalar, SciOxideError, StridedLayout};

/// Selects the sample axis of a [`BatchView`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Treats all elements as one sample, in row-major order (last index
    /// fastest). A rank-zero layout is one sample of one element.
    All,
    /// Uses axis `k` as the sample axis. Each combination of indices on the
    /// other axes is an independent batch, and batches are numbered in
    /// row-major order of those indices.
    Index(usize),
}

/// Real or complex observations in a strided array, split into independent
/// batches along a sample axis.
///
/// The view borrows `values` through a [`StridedLayout`] and does not copy
/// them. [`Axis`] selects the sample axis. A view of shape `(n₀, …, n_{r−1})`
/// with sample axis `k` holds `B = Π_{j≠k} nⱼ` batches of `n = n_k`
/// observations each. With [`Axis::All`], `B = 1` and `n` is the number of
/// elements.
///
/// An optional validity mask and optional weights hold one entry per
/// observation, `n·B` in total. They are indexed in batch-major order: entry
/// `b·n + i` belongs to observation `i` of batch `b`, independent of the
/// strides of `values`. A `false` mask entry marks a missing observation, which
/// [`MissingPolicy`](crate::MissingPolicy) either rejects or skips. The weight
/// model is a type parameter `M`; see [`Weights`](crate::Weights).
///
/// [`Self::with_coordinates`] turns each observation into a vector whose
/// components run along a second axis `c`. Then `B = Π_{j≠k,c} nⱼ`, and each
/// vector observation has one mask entry and one weight.
///
/// ```
/// use scioxide_core::StridedLayout;
/// use scioxide_stats::{Axis, BatchView};
///
/// // A 2×2 row-major array; each row is a batch of two samples.
/// let values = [1.0_f64, 2.0, 3.0, 4.0];
/// let view = BatchView::new(&values, StridedLayout::new(&[2, 2], &[2, 1])?, Axis::Index(1))?;
/// assert_eq!((view.samples(), view.batches()), (2, 2));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BatchView<'a, T: NumericScalar, M: WeightModel = Unweighted> {
    pub(crate) values: &'a [T],
    pub(crate) layout: StridedLayout<'a>,
    pub(crate) axis: Axis,
    pub(crate) coordinate_axis: Option<usize>,
    sample_stride: Option<usize>,
    coordinate_stride: usize,
    pub(crate) samples: usize,
    pub(crate) dimensions: usize,
    pub(crate) batches: usize,
    pub(crate) validity: Option<&'a [bool]>,
    pub(crate) weights: WeightData<'a, T::Real>,
    model: PhantomData<M>,
}
impl<'a, T: NumericScalar> BatchView<'a, T> {
    /// Creates an unweighted view of `values` with sample axis `axis`.
    ///
    /// The layout may be a broadcast layout in which elements share offsets.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `values` is shorter than the storage span of
    ///   `layout`.
    /// - `InvalidInput` if `axis` is `Axis::Index(k)` with `k` not less than
    ///   the rank.
    /// - `SizeOverflow` if the number of batches overflows `usize`.
    pub fn new(
        values: &'a [T],
        layout: StridedLayout<'a>,
        axis: Axis,
    ) -> Result<Self, SciOxideError> {
        layout.validate_backing_len(values.len())?;
        let (samples, batches) = match axis {
            Axis::All => (layout.shape().len(), 1),
            Axis::Index(axis) => (
                layout.shape().extents().get(axis).copied().ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("axis"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "invalid sample axis",
                    )
                })?,
                layout.axis(axis)?.len(),
            ),
        };
        Ok(Self {
            values,
            layout,
            axis,
            coordinate_axis: None,
            sample_stride: match axis {
                Axis::All => None,
                Axis::Index(i) => Some(layout.strides()[i]),
            },
            coordinate_stride: 0,
            samples,
            dimensions: 1,
            batches,
            validity: None,
            weights: WeightData::None,
            model: PhantomData,
        })
    }
    /// Makes each observation a vector whose components run along `axis`.
    ///
    /// The number of batches becomes the product of the extents of the axes
    /// other than the sample and coordinate axes. Call this before
    /// [`Self::with_validity`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the sample axis or `axis` is [`Axis::All`], if the
    ///   two axes coincide, if `axis` is out of range or has extent zero, or if
    ///   a validity mask is already attached.
    /// - `SizeOverflow` if the number of observations overflows `usize`.
    pub fn with_coordinates(mut self, axis: Axis) -> Result<Self, SciOxideError> {
        let (Axis::Index(sample), Axis::Index(coordinate)) = (self.axis, axis) else {
            return Err(invalid(
                scioxide_core::argument!("axis"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "coordinates require indexed axes",
            ));
        };
        if sample == coordinate || self.validity.is_some() {
            return Err(invalid(
                scioxide_core::argument!("axis"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "coordinate axis coincides with samples or metadata is already attached",
            ));
        }
        self.dimensions = self
            .layout
            .shape()
            .extents()
            .get(coordinate)
            .copied()
            .filter(|&n| n != 0)
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("axis"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "invalid or empty coordinate axis",
                )
            })?;
        self.batches = self
            .layout
            .shape()
            .extents()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != sample && *i != coordinate)
            .try_fold(1usize, |n, (_, &extent)| {
                n.checked_mul(extent).ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("statistics_batches"))
                })
            })?;
        self.samples.checked_mul(self.batches).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_observations"))
        })?;
        self.coordinate_axis = Some(coordinate);
        self.coordinate_stride = self.layout.strides()[coordinate];
        Ok(self)
    }
}
impl<'a, T: NumericScalar, M: WeightModel> BatchView<'a, T, M> {
    /// Selects a different sample axis.
    ///
    /// An attached mask or weight slice keeps its entries and is read in the
    /// batch-major order of the new axis.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the view has vector observations or `axis` is out of
    ///   range.
    /// - `SizeOverflow` if the number of batches overflows `usize`.
    pub fn with_axis(mut self, axis: Axis) -> Result<Self, SciOxideError> {
        if self.coordinate_axis.is_some() {
            return Err(invalid(
                scioxide_core::argument!("axis"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "vector observation axes are fixed",
            ));
        }
        let (samples, batches) = match axis {
            Axis::All => (self.layout.shape().len(), 1),
            Axis::Index(i) => (
                self.layout
                    .shape()
                    .extents()
                    .get(i)
                    .copied()
                    .ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("axis"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "invalid sample axis",
                        )
                    })?,
                self.layout.axis(i)?.len(),
            ),
        };
        self.samples = samples;
        self.batches = batches;
        self.axis = axis;
        self.sample_stride = match axis {
            Axis::All => None,
            Axis::Index(i) => Some(self.layout.strides()[i]),
        };
        Ok(self)
    }
    /// Attaches a validity mask in which `true` marks a present observation.
    ///
    /// The mask has one entry per observation, in batch-major order.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `mask.len()` differs from
    ///   `self.samples() * self.batches()`.
    /// - `SizeOverflow` if that product overflows `usize`.
    pub fn with_validity(mut self, mask: &'a [bool]) -> Result<Self, SciOxideError> {
        let len = self.samples.checked_mul(self.batches).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_observations"))
        })?;
        if mask.len() != len {
            return Err(SciOxideError::dimension_mismatch([len], [mask.len()]));
        }
        self.validity = Some(mask);
        Ok(self)
    }
    /// Attaches weights, replacing any previous weights and changing the weight
    /// model to that of `weights`.
    ///
    /// Weights are in batch-major order, one per observation. Their values are
    /// checked when a statistic reads them.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of weights differs from
    ///   `self.samples() * self.batches()`.
    /// - `SizeOverflow` if that product overflows `usize`.
    pub fn with_weights<W: BatchWeights<'a, T::Real>>(
        self,
        weights: W,
    ) -> Result<BatchView<'a, T, W::Model>, SciOxideError> {
        let data = crate::weights::batch_sealed::Sealed::data(weights);
        let expected = self.samples.checked_mul(self.batches).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_observations"))
        })?;
        if data.len().is_some_and(|len| len != expected) {
            return Err(SciOxideError::dimension_mismatch(
                [expected],
                [data.len().unwrap_or(0)],
            ));
        }
        Ok(BatchView {
            values: self.values,
            layout: self.layout,
            axis: self.axis,
            coordinate_axis: self.coordinate_axis,
            sample_stride: self.sample_stride,
            coordinate_stride: self.coordinate_stride,
            samples: self.samples,
            dimensions: self.dimensions,
            batches: self.batches,
            validity: self.validity,
            weights: data,
            model: PhantomData,
        })
    }
    /// Returns the sample axis.
    pub const fn axis(self) -> Axis {
        self.axis
    }
    /// Returns the layout of the borrowed values.
    pub const fn layout(self) -> StridedLayout<'a> {
        self.layout
    }
    /// Returns the number of observations in each batch.
    pub const fn samples(self) -> usize {
        self.samples
    }
    /// Returns the number of components of each observation (one for scalar
    /// observations).
    pub const fn dimensions(self) -> usize {
        self.dimensions
    }
    /// Returns the number of independent batches.
    pub const fn batches(self) -> usize {
        self.batches
    }
    pub(crate) fn batch_origin(self, batch: usize) -> usize {
        if self.axis == Axis::All || self.samples == 0 {
            return 0;
        }
        let Axis::Index(axis) = self.axis else {
            return 0;
        };
        let mut rest = batch;
        let mut origin = 0;
        for (i, &extent) in self.layout.shape().extents().iter().enumerate().rev() {
            if i != axis && Some(i) != self.coordinate_axis {
                origin += rest % extent * self.layout.strides()[i];
                rest /= extent;
            }
        }
        origin
    }
    #[inline(always)]
    pub(crate) fn value_at_origin(self, origin: usize, sample: usize, coordinate: usize) -> T {
        self.values[self.offset_at_origin(origin, sample, coordinate)]
    }
    #[inline(always)]
    fn offset_at_origin(self, origin: usize, sample: usize, coordinate: usize) -> usize {
        if let Some(stride) = self.sample_stride {
            return origin + sample * stride + coordinate * self.coordinate_stride;
        }
        let mut rest = sample;
        let mut offset = 0;
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
        offset
    }
}
