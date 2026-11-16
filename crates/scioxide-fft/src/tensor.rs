//! Checked tensor metadata, copying and independent batch geometry.
use crate::validation::invalid;
use alloc::vec::Vec;
use scioxide_core::backend::{checked_mul, try_filled, try_with_capacity};
use scioxide_core::{Argument, NdView, SciOxideError, Shape};

// Shared N-D helpers; invalid configurations name the offending `argument`.

pub(crate) fn copy_metadata(values: &[usize]) -> Result<Vec<usize>, SciOxideError> {
    let mut result = try_with_capacity(values.len(), scioxide_core::quantity!("n_d_metadata"))?;
    result.extend_from_slice(values);
    Ok(result)
}

pub(crate) fn strides(shape: &[usize]) -> Result<Vec<usize>, SciOxideError> {
    let mut result = try_filled(shape.len(), 0, scioxide_core::quantity!("n_d_strides"))?;
    // An empty tensor addresses nothing, so all-zero strides are valid and
    // avoid overflow from the product of the remaining (possibly huge)
    // extents.
    if shape.contains(&0) {
        return Ok(result);
    }
    let mut stride = 1;
    for (extent, slot) in shape.iter().zip(result.iter_mut()).rev() {
        *slot = stride;
        stride = checked_mul(stride, *extent, scioxide_core::quantity!("n_d_shape"))?;
    }
    Ok(result)
}

// `shape_argument` and `axes_argument` name the caller parameters holding
// `shape` and `axes`.
pub(crate) fn validate_axes(
    shape: &[usize],
    axes: &[usize],
    shape_argument: Argument,
    axes_argument: Argument,
) -> Result<usize, SciOxideError> {
    let geometry = Shape::new(shape)?;
    let len = geometry.len();
    for (i, &axis) in axes.iter().enumerate() {
        geometry.extent(axis)?;
        if axes[..i].contains(&axis) {
            return Err(invalid(
                scioxide_core::InvalidInputReason::Inconsistent,
                axes_argument,
                "axes must be distinct",
            ));
        }
        if shape[axis] == 0 {
            return Err(invalid(
                scioxide_core::InvalidInputReason::Empty,
                shape_argument,
                "transformed extents must be positive",
            ));
        }
    }
    Ok(len)
}

pub(crate) fn check_shape(actual: &[usize], expected: &[usize]) -> Result<(), SciOxideError> {
    if actual != expected {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::from(expected),
            scioxide_core::Dimensions::from(actual),
        ));
    }
    Ok(())
}

/// Selected coordinates within one independent batch. Rank-sized metadata
/// keeps arbitrary layouts separate from numerical scaling; no element table.
#[derive(Debug, Clone)]
pub(crate) struct BatchGeometry {
    pub(crate) shape: Vec<usize>,
    pub(crate) batch_shape: Vec<usize>,
    pub(crate) count: usize,
    pub(crate) len: usize,
}
impl BatchGeometry {
    pub(crate) fn new(shape: &[usize], axes: &[usize]) -> Result<Self, SciOxideError> {
        let mut selected = copy_metadata(shape)?;
        let mut batches = copy_metadata(shape)?;
        for axis in 0..shape.len() {
            if axes.contains(&axis) {
                batches[axis] = 1;
            } else {
                selected[axis] = 1;
            }
        }
        let empty = shape.contains(&0);
        Ok(Self {
            count: if empty { 0 } else { batches.iter().product() },
            len: if empty { 0 } else { selected.iter().product() },
            shape: selected,
            batch_shape: batches,
        })
    }
    pub(crate) fn base(&self, batch: usize, strides: &[usize]) -> usize {
        Self::offset(batch, &self.batch_shape, strides)
    }
    fn offset(mut index: usize, shape: &[usize], strides: &[usize]) -> usize {
        let mut offset = 0;
        for (&extent, &stride) in shape.iter().zip(strides).rev() {
            if extent > 1 {
                offset += (index % extent) * stride;
                index /= extent;
            }
        }
        offset
    }
    /// Whether the batch values are `len` adjacent elements from its base.
    pub(crate) fn dense(&self, strides: &[usize]) -> bool {
        let mut width = 1;
        self.shape.iter().zip(strides).rev().all(|(&n, &stride)| {
            if n <= 1 {
                return true;
            }
            let matches = width == stride;
            width *= n;
            matches
        })
    }
    /// The batch values starting at its base, when they are adjacent.
    pub(crate) fn slice<'a, T>(&self, strides: &[usize], values: &'a [T]) -> Option<&'a [T]> {
        self.dense(strides).then(|| &values[..self.len])
    }
    /// Copy the batch values of `source` (starting at its base, with
    /// `source_strides`) to those of `target`, mapping each value.
    pub(crate) fn copy<T: Copy, U>(
        &self,
        (source, source_strides): (&[T], &[usize]),
        (target, strides): (&mut [U], &[usize]),
        map: impl Fn(T) -> U,
    ) {
        match (self.slice(source_strides, source), self.dense(strides)) {
            (Some(source), true) => {
                for (slot, &value) in target[..self.len].iter_mut().zip(source) {
                    *slot = map(value);
                }
            }
            _ => {
                for (i, o) in self.offsets(source_strides).zip(self.offsets(strides)) {
                    target[o] = map(source[i]);
                }
            }
        }
    }
    /// Apply `visit` to every batch value starting at its base.
    pub(crate) fn for_each_mut<T>(
        &self,
        strides: &[usize],
        values: &mut [T],
        mut visit: impl FnMut(&mut T),
    ) {
        let _ = self.try_for_each_mut(strides, values, |value| {
            visit(value);
            Ok::<(), core::convert::Infallible>(())
        });
    }
    /// Apply `visit` to every batch value starting at its base, in row-major order.
    pub(crate) fn try_for_each_mut<T, E>(
        &self,
        strides: &[usize],
        values: &mut [T],
        mut visit: impl FnMut(&mut T) -> Result<(), E>,
    ) -> Result<(), E> {
        if self.dense(strides) {
            values[..self.len].iter_mut().try_for_each(visit)
        } else {
            self.offsets(strides)
                .try_for_each(|offset| visit(&mut values[offset]))
        }
    }
    pub(crate) fn offsets<'a>(
        &'a self,
        strides: &'a [usize],
    ) -> impl ExactSizeIterator<Item = usize> + Clone + 'a {
        let contiguous = self.dense(strides);
        (0..self.len).map(move |index| {
            if contiguous {
                index
            } else {
                Self::offset(index, &self.shape, strides)
            }
        })
    }
}

/// Copy a borrowed tensor into its dense row-major candidate, mapping each
/// value. Inputs whose strides already are `dense` are read as one slice.
pub(crate) fn copy_in<T: Copy, U>(
    input: &NdView<'_, '_, T>,
    dense: &[usize],
    candidate: &mut [U],
    map: impl Fn(T) -> U,
) {
    if input.layout().strides() == dense {
        for (slot, &value) in candidate.iter_mut().zip(input.storage()) {
            *slot = map(value);
        }
    } else {
        for (slot, &value) in candidate.iter_mut().zip(input.iter()) {
            *slot = map(value);
        }
    }
}

/// Write a dense candidate, using contiguous output when available.
pub(crate) fn copy_out<T: Copy, U>(
    values: &[T],
    output: &mut scioxide_core::NdViewMut<'_, '_, U>,
    map: impl Fn(T) -> U,
) {
    if let Some(output) = output.contiguous_mut() {
        for (slot, &value) in output.iter_mut().zip(values) {
            *slot = map(value);
        }
    } else {
        let mut values = values.iter();
        output.for_each_mut(|out| {
            if let Some(&value) = values.next() {
                *out = map(value);
            }
        });
    }
}

/// Borrowed values starting at one batch's base in the input layout.
pub(crate) enum BatchInput<'a, S: scioxide_core::RealScalar> {
    Real(&'a [S]),
    Complex(&'a [scioxide_core::Complex<S>]),
}
impl<S: scioxide_core::RealScalar> BatchInput<'_, S> {
    #[inline]
    pub(crate) fn maximum(
        self,
        geometry: &BatchGeometry,
        strides: &[usize],
    ) -> Result<S, SciOxideError> {
        use crate::scaling::{complex_max, real_max};
        match self {
            Self::Real(values) => match geometry.slice(strides, values) {
                Some(values) => crate::scaling::real_max_slice(values),
                None => real_max(geometry.offsets(strides).map(|i| values[i])),
            },
            Self::Complex(values) => match geometry.slice(strides, values) {
                Some(values) => crate::scaling::complex_max_slice(values),
                None => complex_max(geometry.offsets(strides).map(|i| values[i])),
            },
        }
    }
}

/// Candidate values starting at one batch's base in the output layout.
pub(crate) enum BatchOutput<'a, S: scioxide_core::RealScalar> {
    Real(&'a mut [S]),
    Complex(&'a mut [scioxide_core::Complex<S>]),
}
impl<S: scioxide_core::RealScalar> BatchOutput<'_, S> {
    #[inline]
    fn restore(
        self,
        geometry: &BatchGeometry,
        strides: &[usize],
        restore: crate::scaling::BinaryScale<S>,
        normalization: S,
        check: bool,
        overflow: &'static str,
    ) -> Result<(), SciOxideError> {
        use crate::scaling::BinaryScale;
        // Select the binary arm once; finiteness accumulates without early
        // exits and is reported after the pass, as a failed batch publishes
        // nothing.
        fn real<S: scioxide_core::RealScalar>(
            values: &mut [S],
            (geometry, strides): (&BatchGeometry, &[usize]),
            restore: impl Fn(S) -> S,
        ) -> bool {
            let mut finite = true;
            geometry.for_each_mut(strides, values, |value| {
                *value = if *value == S::ZERO {
                    S::ZERO
                } else {
                    restore(*value)
                };
                finite &= value.is_finite();
            });
            finite
        }
        fn complex<S: scioxide_core::RealScalar>(
            values: &mut [scioxide_core::Complex<S>],
            (geometry, strides): (&BatchGeometry, &[usize]),
            normalization: S,
            restore: impl Fn(S) -> S,
        ) -> bool {
            let canonical = |value: S| {
                if value == S::ZERO {
                    S::ZERO
                } else {
                    restore(value)
                }
            };
            let mut finite = true;
            geometry.for_each_mut(strides, values, |value| {
                let scaled = *value * normalization;
                *value = scioxide_core::Complex::new(canonical(scaled.re), canonical(scaled.im));
                finite &= value.re.is_finite() & value.im.is_finite();
            });
            finite
        }
        let place = (geometry, strides);
        let finite = match (self, restore) {
            // Trig children already apply their axis normalizations.
            (Self::Real(values), BinaryScale::Identity) => real(values, place, |v| v),
            (Self::Real(values), BinaryScale::Multiply(factor)) => {
                real(values, place, |v| v * factor)
            }
            (Self::Real(values), BinaryScale::Extended(exponent)) => {
                real(values, place, |v| v.scalbn(exponent))
            }
            (Self::Complex(values), BinaryScale::Identity) => {
                complex(values, place, normalization, |v| v)
            }
            (Self::Complex(values), BinaryScale::Multiply(factor)) => {
                complex(values, place, normalization, |v| v * factor)
            }
            (Self::Complex(values), BinaryScale::Extended(exponent)) => {
                complex(values, place, normalization, |v| v.scalbn(exponent))
            }
        };
        if check && !finite {
            Err(crate::validation::arithmetic_overflow(overflow))
        } else {
            Ok(())
        }
    }
}

/// Family-specific storage and kernels; the driver owns batch scaling policy.
/// All methods borrow existing plan metadata and the caller's scratch.
pub(crate) trait BatchTransform<S: scioxide_core::RealScalar> {
    fn single_axis(&mut self) -> Result<(), SciOxideError>;
    /// Measure finite input. Projection and range scaling must wait for the
    /// complete batch maximum.
    fn prepare(&mut self, batch: usize) -> Result<S, SciOxideError>;
    fn transform(
        &mut self,
        batch: usize,
        maximum: S,
        down: crate::scaling::BinaryScale<S>,
        mode: crate::lanes::LaneMode,
    ) -> Result<(), SciOxideError>;
    fn output(&mut self, batch: usize) -> (BatchOutput<'_, S>, &BatchGeometry, &[usize]);
}

/// Prepared range bounds. A proven-finite single batch may defer restoration
/// to publication; independent batches restore in their own binary frames.
pub(crate) struct BatchScaling<S> {
    pub(crate) axes: usize,
    pub(crate) count: usize,
    pub(crate) headroom: i32,
    pub(crate) bounded: bool,
    pub(crate) normalization: S,
    pub(crate) output_bound: S,
    pub(crate) overflow: &'static str,
}
impl<S: scioxide_core::RealScalar> BatchScaling<S> {
    /// Run one checked child pass for a single axis, or scale and transform
    /// each independent batch. This driver writes no output.
    #[inline]
    pub(crate) fn run(
        self,
        operation: &mut impl BatchTransform<S>,
    ) -> Result<Option<crate::scaling::BinaryScale<S>>, SciOxideError> {
        use crate::lanes::LaneMode;
        use crate::scaling::{BinaryScale, range_exponent};
        if self.axes == 1 {
            operation.single_axis()?;
            return Ok(None);
        }
        let mode = if self.bounded {
            LaneMode::Bounded
        } else {
            LaneMode::Checked
        };
        let mut pending = None;
        for batch in 0..self.count {
            let maximum = operation.prepare(batch)?;
            let exponent = if self.axes == 0 {
                0
            } else {
                range_exponent(maximum) + self.headroom
            };
            operation.transform(batch, maximum, BinaryScale::new(-exponent), mode)?;
            let check = !self
                .output_bound
                .scalbn(exponent - self.headroom)
                .is_finite();
            let restore = BinaryScale::new(exponent);
            if !check && self.count == 1 {
                pending = Some(restore);
                continue;
            }
            let (output, geometry, strides) = operation.output(batch);
            output.restore(
                geometry,
                strides,
                restore,
                self.normalization,
                check,
                self.overflow,
            )?;
        }
        Ok(pending)
    }
}
