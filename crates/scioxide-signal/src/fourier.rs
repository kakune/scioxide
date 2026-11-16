//! One vector/tensor FFT adapter, with packed storage for real operands.
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, RealScalar, SciOxideError,
    StridedLayout,
};
use scioxide_fft::{
    FftDirection, FftInput, FftLayout, FftNdPlan, FftNdPlanSpec, FftOutput, FftPlan, FftPlanSpec,
    Normalization,
};
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum Transform<S: RealScalar> {
    Vector(FftPlan<S>),
    Tensor(FftNdPlan<S>),
}
impl<S: RealScalar> Transform<S> {
    pub(crate) fn new(
        shape: &[usize],
        axes: &[usize],
        direction: FftDirection,
        real: bool,
        policy: ExecutionPolicy,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let layout = if real {
            FftLayout::Real
        } else {
            FftLayout::Complex
        };
        if shape.len() == 1 {
            Ok(Self::Vector(FftPlan::build(
                FftPlanSpec::new(shape[0], direction, Normalization::Inverse, layout)
                    .with_execution_policy(policy),
                context,
            )?))
        } else {
            Ok(Self::Tensor(FftNdPlan::build(
                FftNdPlanSpec::new(shape, axes, direction, Normalization::Inverse, layout)?
                    .with_execution_policy(policy),
                context,
            )?))
        }
    }
    /// Complex FFT execution storage of this transform.
    pub(crate) fn complex_work(&self) -> Result<usize, SciOxideError> {
        match self {
            Self::Vector(p) => Ok(p.workspace_requirements(())?.complex_len()),
            Self::Tensor(p) => Ok(p.workspace_requirements(())?.complex_len()),
        }
    }
    pub(crate) fn complex(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        layout: StridedLayout<'_>,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Vector(p) => p.execute_into(input, output, scratch, context),
            Self::Tensor(p) => p.execute_into(
                NdView::new(input, layout)?,
                NdViewMut::new(output, layout)?,
                scratch,
                context,
            ),
        }
    }
    pub(crate) fn complex_in_place(
        &self,
        data: &mut [Complex<S>],
        layout: StridedLayout<'_>,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Vector(p) => p.execute_in_place(data, scratch, context),
            Self::Tensor(p) => p.execute_in_place(NdViewMut::new(data, layout)?, scratch, context),
        }
    }
    pub(crate) fn forward_real(
        &self,
        input: &[S],
        output: &mut [Complex<S>],
        layouts: (StridedLayout<'_>, StridedLayout<'_>),
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Vector(p) => p.execute_into(
                input,
                FftOutput::hermitian(input.len(), output),
                scratch,
                context,
            ),
            Self::Tensor(p) => p.execute_into(
                NdView::new(input, layouts.0)?,
                NdViewMut::new(output, layouts.1)?,
                scratch,
                context,
            ),
        }
    }
    pub(crate) fn inverse_real(
        &self,
        input: &[Complex<S>],
        output: &mut [S],
        layouts: (StridedLayout<'_>, StridedLayout<'_>),
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Vector(p) => p.execute_into(
                FftInput::hermitian(output.len(), input),
                output,
                scratch,
                context,
            ),
            Self::Tensor(p) => p.execute_into(
                NdView::new(input, layouts.1)?,
                NdViewMut::new(output, layouts.0)?,
                scratch,
                context,
            ),
        }
    }
}
