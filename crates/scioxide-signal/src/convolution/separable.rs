//! Separable two-dimensional FIR filtering.
use super::nd::{ConvolutionKernel, ConvolutionNdPlan, ConvolutionOptions};
use crate::util::exact_shape;
use crate::{SignalRequirements, SignalWorkspace};
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{
    ExecutionContext, NdView, NdViewMut, NumericScalar, SciOxideError, StridedLayout,
};
/// Plan for two-dimensional FIR filtering with a separable kernel.
///
/// The kernel is the outer product `h[p, q] = r[p] c[q]` of the row taps `r`,
/// applied along axis 0, and the column taps `c`, applied along axis 1. The
/// plan filters along axis 0 and then along axis 1. Both passes use the same
/// [`ConvolutionOptions`], so the result equals a
/// [`ConvolutionNdPlan`](crate::ConvolutionNdPlan) with kernel `h` and these
/// options, up to rounding. With the direct method this costs about
/// `M N (kᵣ + k_c)` products for an `M × N` array and `kᵣ` row and `k_c`
/// column taps, instead of `M N kᵣ k_c`.
///
/// ```
/// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
/// use scioxide_signal::{ConvolutionOptions, SeparableFir2dPlan};
/// let plan = SeparableFir2dPlan::new([2, 2], &[1.0_f64], &[1.0], ConvolutionOptions::default())?;
/// let mut workspace = plan.workspace(())?;
/// let mut output = [0.0; 4];
/// plan.filter_into(
///     NdView::new(&[1.0, 2.0, 3.0, 4.0], StridedLayout::new(&[2, 2], &[2, 1])?)?,
///     NdViewMut::new(&mut output, StridedLayout::new(&[2, 2], &[2, 1])?)?,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// // A unit kernel leaves the array unchanged.
/// assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SeparableFir2dPlan<T: NumericScalar> {
    rows: ConvolutionNdPlan<T>,
    columns: ConvolutionNdPlan<T>,
    intermediate_len: usize,
    output_len: usize,
}
impl<T: NumericScalar> SeparableFir2dPlan<T> {
    /// Prepares a plan for arrays of shape `shape`, using
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if an extent of `shape` is zero, `row_taps` or
    ///   `column_taps` is empty, or `options` is not a valid combination (see
    ///   [`ConvolutionNdPlan::new`]).
    /// - `NonFinite` if a tap is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the plan storage cannot be sized
    ///   or allocated.
    pub fn new(
        shape: [usize; 2],
        row_taps: &[T],
        column_taps: &[T],
        options: ConvolutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            shape,
            row_taps,
            column_taps,
            options,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepares a plan like [`new`](Self::new), planning the FFTs with `context`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`new`](Self::new), and `ExecutionStopped` if
    /// `context` is cancelled or past its deadline.
    pub fn build(
        shape: [usize; 2],
        row_taps: &[T],
        column_taps: &[T],
        options: ConvolutionOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let rows = ConvolutionNdPlan::build(
            &shape,
            NdView::new(row_taps, StridedLayout::new(&[row_taps.len(), 1], &[1, 1])?)?,
            &[0],
            options,
            context,
        )?;
        let columns = ConvolutionNdPlan::build(
            rows.output_shape(),
            NdView::new(
                column_taps,
                StridedLayout::new(&[1, column_taps.len()], &[column_taps.len(), 1])?,
            )?,
            &[1],
            options,
            context,
        )?;
        let intermediate_len = checked_mul(
            rows.output_shape()[0],
            rows.output_shape()[1],
            scioxide_core::quantity!("separable_intermediate"),
        )?;
        let output_len = checked_mul(
            columns.output_shape()[0],
            columns.output_shape()[1],
            scioxide_core::quantity!("separable_output"),
        )?;
        Ok(Self {
            rows,
            columns,
            intermediate_len,
            output_len,
        })
    }
    /// Returns the output shape, sized on each axis by the
    /// [`ConvolutionExtent`](crate::ConvolutionExtent) of the options.
    pub fn output_shape(&self) -> &[usize] {
        self.columns.output_shape()
    }
    /// Returns the workspace that one call to [`filter_into`](Self::filter_into)
    /// needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn workspace_requirements(&self, _: ()) -> Result<SignalRequirements, SciOxideError> {
        let pass = self
            .rows
            .workspace_requirements(ConvolutionKernel::Prepared)?
            .max(
                self.columns
                    .workspace_requirements(ConvolutionKernel::Prepared)?,
            );
        Ok(SignalRequirements::new(
            checked_add(
                checked_add(
                    self.intermediate_len,
                    self.output_len,
                    scioxide_core::quantity!("separable_staging"),
                )?,
                pass.values_len(),
                scioxide_core::quantity!("separable_workspace"),
            )?,
            0,
            pass.complex_len(),
        ))
    }
    /// Allocates a workspace for [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    /// - `AllocationFailure` if the allocator fails.
    pub fn workspace(&self, request: ()) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(request)?)
    }
    /// Writes the filtered `input` to `output`.
    ///
    /// `output` has shape [`output_shape`](Self::output_shape).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` does not have the prepared
    ///   shape, or `workspace` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    pub fn filter_into(
        &self,
        input: NdView<'_, '_, T>,
        mut output: NdViewMut<'_, '_, T>,
        workspace: &mut SignalWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        exact_shape(self.rows.input_shape(), input.layout().shape().extents())?;
        exact_shape(self.output_shape(), output.layout().shape().extents())?;
        let requirements = self.workspace_requirements(())?;
        let buffers = workspace.buffers(&requirements)?;
        let (intermediate, rest) = buffers.values.split_at_mut(self.intermediate_len);
        let (staged, work) = rest.split_at_mut(self.output_len);
        let ms = [self.rows.output_shape()[1], 1];
        let os = [self.output_shape()[1], 1];
        let ml = StridedLayout::new(self.rows.output_shape(), &ms)?;
        let ol = StridedLayout::new(self.output_shape(), &os)?;
        let pass = |plan: &ConvolutionNdPlan<T>| -> Result<_, SciOxideError> {
            let requirements = plan.workspace_requirements(ConvolutionKernel::Prepared)?;
            Ok((requirements.values_len(), requirements.complex_len()))
        };
        let (row_values, row_complex) = pass(&self.rows)?;
        self.rows.run_prepared(
            input,
            NdViewMut::new(intermediate, ml)?,
            &mut work[..row_values],
            &mut buffers.complex[..row_complex],
            context,
        )?;
        let (column_values, column_complex) = pass(&self.columns)?;
        self.columns.run_prepared(
            NdView::new(intermediate, ml)?,
            NdViewMut::new(staged, ol)?,
            &mut work[..column_values],
            &mut buffers.complex[..column_complex],
            context,
        )?;
        context.check()?;
        crate::axis::publish(staged, &mut output);
        Ok(())
    }
}
