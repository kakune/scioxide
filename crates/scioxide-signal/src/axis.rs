//! Lane gathering and scattering for axis operations on N-dimensional views.
use crate::util::{axis_mismatch, capacity, exact_shape, invalid};
use scioxide_core::backend::checked_mul;
use scioxide_core::{Complex, NdView, NdViewMut, SciOxideError, Shape, StridedLayout};

/// Logical row-major staging geometry; physical input strides belong to core.
pub(crate) struct Geometry {
    pub(crate) axis: usize,
    pub(crate) inner: usize,
    pub(crate) batches: usize,
    pub(crate) output_len: usize,
}
impl Geometry {
    pub(crate) fn new(
        shape: &[usize],
        axis: usize,
        n: usize,
        m: usize,
    ) -> Result<Self, SciOxideError> {
        let shape = Shape::new(shape)?;
        if axis >= shape.rank() {
            return Err(invalid(
                scioxide_core::argument!("axis"),
                "axis must be below the shape rank",
            ));
        }
        if shape.extents()[axis] != n {
            return Err(axis_mismatch(shape.extents(), axis, n, shape.extents()));
        }
        let empty = shape
            .extents()
            .iter()
            .enumerate()
            .any(|(i, &v)| i != axis && v == 0);
        let (inner, batches) = if empty {
            (0, 0)
        } else {
            let inner = Shape::new(&shape.extents()[axis + 1..])?.len();
            let outer = Shape::new(&shape.extents()[..axis])?.len();
            (
                inner,
                checked_mul(inner, outer, scioxide_core::quantity!("signal_batches"))?,
            )
        };
        Ok(Self {
            axis,
            inner,
            batches,
            output_len: checked_mul(batches, m, scioxide_core::quantity!("signal_axis_output"))?,
        })
    }
    pub(crate) fn gather<T: Copy>(
        &self,
        input: &NdView<'_, '_, T>,
        lane: &mut [T],
        batch: usize,
    ) -> Result<(), SciOxideError> {
        for (slot, offset) in lane
            .iter_mut()
            .zip(input.layout().axis(self.axis)?.lane(batch)?.offsets())
        {
            *slot = input.storage()[offset];
        }
        Ok(())
    }
    /// Traverse a flattened lane (also the adjacent frame/bin axes of an STFT).
    pub(crate) fn gather_dense<T: Copy>(
        &self,
        input: &[T],
        lane: &mut [T],
        batch: usize,
    ) -> Result<(), SciOxideError> {
        let n = lane.len();
        let shape = [self.batches / self.inner, n, self.inner];
        let strides = [
            checked_mul(
                n,
                self.inner,
                scioxide_core::quantity!("signal_axis_stride"),
            )?,
            self.inner,
            1,
        ];
        for (slot, offset) in lane.iter_mut().zip(
            StridedLayout::new(&shape, &strides)?
                .axis(1)?
                .lane(batch)?
                .offsets(),
        ) {
            *slot = input[offset];
        }
        Ok(())
    }
    pub(crate) fn scatter<T: Copy>(
        &self,
        lane: &[T],
        batch: usize,
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        let n = lane.len();
        let shape = [self.batches / self.inner, n, self.inner];
        let strides = [
            checked_mul(
                n,
                self.inner,
                scioxide_core::quantity!("signal_axis_stride"),
            )?,
            self.inner,
            1,
        ];
        for (&value, offset) in lane.iter().zip(
            StridedLayout::new(&shape, &strides)?
                .axis(1)?
                .lane(batch)?
                .offsets(),
        ) {
            output[offset] = value;
        }
        Ok(())
    }
}
pub(crate) fn publish<T: Copy>(values: &[T], output: &mut NdViewMut<'_, '_, T>) {
    let mut values = values.iter();
    output.for_each_mut(|out| {
        if let Some(&v) = values.next() {
            *out = v;
        }
    });
}
/// Publish the real parts of staged complex values in row-major order.
pub(crate) fn publish_real<S: Copy>(values: &[Complex<S>], output: &mut NdViewMut<'_, '_, S>) {
    let mut values = values.iter();
    output.for_each_mut(|out| {
        if let Some(v) = values.next() {
            *out = v.re;
        }
    });
}
pub(crate) fn gather_blocks<T: Copy>(
    input: &NdView<'_, '_, T>,
    axis: usize,
    blocks: &mut [T],
    stride: usize,
    check: impl Fn(T) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let lanes = input.layout().axis(axis)?;
    let n = input.layout().shape().extents()[axis];
    capacity(
        checked_mul(
            lanes.len(),
            stride,
            scioxide_core::quantity!("signal_lane_blocks"),
        )?,
        blocks.len(),
    )?;
    if n == 0 {
        return Ok(());
    }
    for (batch, block) in blocks
        .chunks_exact_mut(stride)
        .take(lanes.len())
        .enumerate()
    {
        for (slot, offset) in block[..n].iter_mut().zip(lanes.lane(batch)?.offsets()) {
            let value = input.storage()[offset];
            check(value)?;
            *slot = value;
        }
    }
    Ok(())
}
pub(crate) fn publish_blocks<T: Copy>(
    output: &mut NdViewMut<'_, '_, T>,
    axis: usize,
    blocks: &[T],
    stride: usize,
    start: usize,
) -> Result<(), SciOxideError> {
    let shape = output.layout().shape();
    let n = shape.extents()[axis];
    let g = Geometry::new(shape.extents(), axis, n, n)?;
    if shape.is_empty() {
        return Ok(());
    }
    let mut index = 0;
    output.for_each_mut(|out| {
        let batch = index / (n * g.inner) * g.inner + index % g.inner;
        *out = blocks[batch * stride + start + index / g.inner % n];
        index += 1;
    });
    Ok(())
}

/// Selected-axis geometry `(axis length, element count)` of
/// matching views, validated.
pub(crate) fn shape<T>(
    input: &NdView<'_, '_, T>,
    output: &NdViewMut<'_, '_, T>,
    axis: usize,
) -> Result<(usize, usize), SciOxideError> {
    let shape = input.layout().shape();
    exact_shape(shape.extents(), output.layout().shape().extents())?;
    if axis >= shape.rank() {
        return Err(invalid(
            scioxide_core::argument!("axis"),
            "axis must be below the view rank",
        ));
    }
    let n = shape.extents()[axis];
    if shape.is_empty() {
        return Ok((n, 0));
    }
    Ok((n, shape.len()))
}
