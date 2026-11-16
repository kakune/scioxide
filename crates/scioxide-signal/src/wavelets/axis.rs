use super::*;
use crate::util::{capacity, check_axis_shape};
use scioxide_core::{NdView, NdViewMut};

pub(super) fn axis_workspace<T: NumericScalar>(
    plan: &WaveletPlan<T>,
    lanes: usize,
) -> Result<usize, SciOxideError> {
    let stride = checked_add(
        checked_add(
            plan.n,
            plan.coefficients,
            scioxide_core::quantity!("wavelet_lane"),
        )?,
        checked_mul(2, plan.width, scioxide_core::quantity!("wavelet_lane"))?,
        scioxide_core::quantity!("wavelet_lane"),
    )?;
    checked_mul(lanes, stride, scioxide_core::quantity!("wavelet_lanes"))
}
pub(super) fn axis_transform<T: NumericScalar>(
    plan: &WaveletPlan<T>,
    input: NdView<'_, '_, T>,
    mut output: NdViewMut<'_, '_, T>,
    axis: usize,
    work: &mut [T],
    inverse: bool,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let shape = input.layout().shape().extents();
    let (n, m) = if inverse {
        (plan.coefficients, plan.n)
    } else {
        (plan.n, plan.coefficients)
    };
    let geometry = crate::axis::Geometry::new(shape, axis, n, m)?;
    check_axis_shape(shape, axis, m, output.layout().shape().extents())?;
    let lanes = geometry.batches;
    let required = axis_workspace(plan, lanes)?;
    capacity(required, work.len())?;
    context.check()?;
    if n == 0 || lanes == 0 {
        return Ok(());
    }
    let stride = required / lanes;
    let blocks = &mut work[..required];
    crate::axis::gather_blocks(&input, axis, blocks, stride, |x| {
        finite(core::slice::from_ref(&x))
    })?;
    crate::util::execute_blocks(
        blocks,
        stride,
        plan.options.execution_policy,
        context,
        |chunk| {
            for block in chunk.chunks_exact_mut(stride) {
                context.check()?;
                let (x, rest) = block.split_at_mut(n);
                let (y, w) = rest.split_at_mut(m);
                plan.raw(x, y, w, inverse, context)?;
            }
            Ok(())
        },
    )?;
    context.check()?;
    crate::axis::publish_blocks(&mut output, axis, blocks, stride, n)?;
    Ok(())
}
