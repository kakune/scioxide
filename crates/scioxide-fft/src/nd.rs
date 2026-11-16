//! Fourier transforms along selected axes of strided N-D arrays.
use crate::validation::invalid;

use crate::lanes::{LaneGeometry, LaneMode, LanePairs, Lanes, Placement};
use crate::numeric::scalar_from_usize_rounded;
use crate::real_fft::RealFftPlan;
use crate::scaling::BinaryScale;
use crate::{FftBackendStrategy, FftDirection, FftLayout, FftPlan, FftPlanSpec, Normalization};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, RealScalar, SciOxideError,
    Shape, StridedLayout,
};

use crate::tensor::{BatchGeometry, check_shape, copy_in, copy_metadata, strides, validate_axes};

/// Describes an N-D Fourier transform: shape, selected axes, direction,
/// normalization and layout.
///
/// The transform runs along each axis in `axes`, in order. Every combination of
/// indices on the other axes is an independent batch. For [`FftLayout::Real`]
/// and [`FftLayout::Hermitian`], the last selected axis of length `n` is stored
/// as `floor(n/2) + 1` packed bins; the other selected axes keep their full
/// length. Normalization uses the product of the selected axis lengths (see
/// [Normalization](crate#normalization)). The spec defaults to
/// [`ExecutionPolicy::Sequential`] and [`FftBackendStrategy::Automatic`].
#[derive(Debug, Clone)]
pub struct FftNdPlanSpec {
    shape: Vec<usize>,
    axes: Vec<usize>,
    direction: FftDirection,
    normalization: Normalization,
    layout: FftLayout,
    policy: ExecutionPolicy,
    backend: FftBackendStrategy,
}
impl FftNdPlanSpec {
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the backend strategy.
    pub const fn backend_strategy(&self) -> FftBackendStrategy {
        self.backend
    }

    /// Creates a spec for the logical array `shape`, transformed along `axes`.
    ///
    /// The shape and axes are copied. They are validated when the plan is
    /// prepared, not here.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the shape or axes cannot be copied.
    pub fn new(
        shape: &[usize],
        axes: &[usize],
        direction: FftDirection,
        normalization: Normalization,
        layout: FftLayout,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            shape: copy_metadata(shape)?,
            axes: copy_metadata(axes)?,
            direction,
            normalization,
            layout,
            policy: ExecutionPolicy::Sequential,
            backend: FftBackendStrategy::Automatic,
        })
    }
    /// Sets the execution policy.
    pub fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Sets the backend strategy.
    pub fn with_backend_strategy(mut self, backend: FftBackendStrategy) -> Self {
        self.backend = backend;
        self
    }
    /// Returns the logical shape, with the packed axis at its full length.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Returns the selected axes in order. The last is packed in real and
    /// Hermitian layouts.
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
    /// Returns the transform direction.
    pub const fn direction(&self) -> FftDirection {
        self.direction
    }
    /// Returns the normalization.
    pub const fn normalization(&self) -> Normalization {
        self.normalization
    }
    /// Returns the layout.
    pub const fn layout(&self) -> FftLayout {
        self.layout
    }
}

/// Selects real or complex input for [`FftNdPlan::execute_into`].
///
/// A complex or real [`NdView`] converts into the matching variant. The
/// variant must match the plan: complex for [`FftLayout::Complex`] and for the
/// packed side of a real or Hermitian plan, real otherwise.
pub enum FftNdInput<'d, 'l, S: RealScalar> {
    /// Full complex data, or packed bins with the plan's packed shape.
    Complex(NdView<'d, 'l, Complex<S>>),
    /// Real samples (real forward) or a real spectrum (Hermitian inverse).
    Real(NdView<'d, 'l, S>),
}
/// Selects real or complex output for [`FftNdPlan::execute_into`].
///
/// A complex or real [`NdViewMut`] converts into the matching variant. The
/// view must not alias itself; see [`NdViewMut`].
pub enum FftNdOutput<'d, 'l, S: RealScalar> {
    /// Full complex data, or packed bins with the plan's packed shape.
    Complex(NdViewMut<'d, 'l, Complex<S>>),
    /// Real samples (real inverse) or a real spectrum (Hermitian forward).
    Real(NdViewMut<'d, 'l, S>),
}
impl<'d, 'l, S: RealScalar> From<NdView<'d, 'l, Complex<S>>> for FftNdInput<'d, 'l, S> {
    fn from(view: NdView<'d, 'l, Complex<S>>) -> Self {
        Self::Complex(view)
    }
}
impl<'d, 'l, S: RealScalar> From<NdView<'d, 'l, S>> for FftNdInput<'d, 'l, S> {
    fn from(view: NdView<'d, 'l, S>) -> Self {
        Self::Real(view)
    }
}
impl<'d, 'l, S: RealScalar> From<NdViewMut<'d, 'l, Complex<S>>> for FftNdOutput<'d, 'l, S> {
    fn from(view: NdViewMut<'d, 'l, Complex<S>>) -> Self {
        Self::Complex(view)
    }
}
impl<'d, 'l, S: RealScalar> From<NdViewMut<'d, 'l, S>> for FftNdOutput<'d, 'l, S> {
    fn from(view: NdViewMut<'d, 'l, S>) -> Self {
        Self::Real(view)
    }
}

/// Computes Fourier transforms along selected axes of a strided N-D array.
///
/// The plan applies the one-dimensional transform of
/// [Conventions](crate#conventions) along each selected axis in turn. Input
/// and output are borrowed [`NdView`]s with arbitrary strides, so plans are the
/// form for strided, broadcast and padded arrays; the one-call functions such
/// as [`fftn`](crate::fftn) take dense row-major slices. Complex plans can also
/// run in place with [`Self::execute_in_place`]. For `V` elements and
/// selected axis lengths `nₐ`, the cost is `O(V Σₐ log nₐ)` operations and the
/// scratch is `O(V + maxₐ nₐ)` complex values.
///
/// Each batch (each index on the unselected axes) is scaled by its own power
/// of two before the transform and restored afterwards, so a batch with large
/// values does not cause small values in another batch to underflow. With one
/// selected axis, every lane gets the same result as a one-dimensional
/// [`FftPlan`] of the same layout. Packed-to-real input is checked as described
/// in [Hermitian input](crate#hermitian-input).
///
/// Preparation allocates the plan. Execution with a large enough workspace
/// follows [Allocation](scioxide_core#allocation) and
/// [Output on error](scioxide_core#output-on-error).
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, StridedLayout};
/// use scioxide_fft::{FftDirection, FftLayout, FftNdPlan, FftNdPlanSpec};
///
/// // Forward normalization divides by 2 · 3 = 6, so a constant array keeps its value at DC.
/// let (dir, norm) = (FftDirection::Forward, scioxide_fft::Normalization::Forward);
/// let spec = FftNdPlanSpec::new(&[2, 3], &[0, 1], dir, norm, FftLayout::Complex)?;
/// let plan = FftNdPlan::<f64>::new(spec)?;
/// let layout = StridedLayout::new(&[2, 3], &[3, 1])?;
/// let (x, mut y) = ([Complex::new(2.0, 0.0); 6], [Complex::new(0.0, 0.0); 6]);
/// let (input, output) = (NdView::new(&x, layout)?, NdViewMut::new(&mut y, layout)?);
/// let cx = ExecutionContext::sequential();
/// plan.execute_into(input, output, &mut plan.workspace(())?, cx)?;
/// assert!((y[0] - Complex::new(2.0, 0.0)).norm() < 1e-12);
/// assert!(y[1..].iter().all(|z| z.norm() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct FftNdPlan<S: RealScalar> {
    spec: FftNdPlanSpec,
    dense_strides: Vec<usize>,
    packed_shape: Vec<usize>,
    packed_strides: Vec<usize>,
    full_batches: BatchGeometry,
    packed_batches: BatchGeometry,
    endpoint_pairs: Vec<(usize, usize)>,
    len: usize,
    packed_len: usize,
    children: Vec<(LaneGeometry, FftPlan<S>)>,
    real: Option<RealAxis<S>>,
    staging_len: usize,
    scratch: crate::FftWorkspaceRequirements,
    scale: S,
    headroom: i32,
    bounded: bool,
    output_bound: S,
}
/// A finished candidate and the binary restoration, if any, still to apply
/// when it is written to the output.
type Published<'a, S> = (&'a [Complex<S>], Option<BinaryScale<S>>);

/// Candidates, strided-lane staging and child scratch of one execution.
#[derive(Debug)]
struct Work<'a, S> {
    full: &'a mut [Complex<S>],
    packed: &'a mut [Complex<S>],
    staging: &'a mut [Complex<S>],
    child: &'a mut [Complex<S>],
}

/// The packed real axis and its lanes in the full and packed candidates.
#[derive(Debug, Clone)]
struct RealAxis<S: RealScalar> {
    full: LaneGeometry,
    packed: LaneGeometry,
    plan: RealFftPlan<S>,
}

impl<S: RealScalar> FftNdPlan<S> {
    /// Returns the bytes held by the plan, its metadata and its child plans,
    /// excluding allocator overhead.
    pub fn plan_bytes(&self) -> usize {
        // Preparation checked the sum; cloning cannot increase any capacity.
        self.byte_counts().sum()
    }
    fn byte_counts(&self) -> impl Iterator<Item = usize> + '_ {
        [
            size_of::<Self>(),
            crate::memory::vector_bytes(&self.endpoint_pairs),
            crate::memory::vector_bytes(&self.children),
        ]
        .into_iter()
        .chain(
            [
                &self.spec.shape,
                &self.spec.axes,
                &self.dense_strides,
                &self.packed_shape,
                &self.packed_strides,
                &self.full_batches.shape,
                &self.full_batches.batch_shape,
                &self.packed_batches.shape,
                &self.packed_batches.batch_shape,
            ]
            .into_iter()
            .map(crate::memory::vector_bytes),
        )
        .chain(
            self.children
                .iter()
                .enumerate()
                .filter(|(i, (_, p))| {
                    !self.children[..*i]
                        .iter()
                        .any(|(_, earlier)| p.shares_core(earlier))
                })
                .map(|(_, (_, p))| p.plan_bytes() - size_of::<FftPlan<S>>()),
        )
        .chain(
            self.real
                .iter()
                .map(|axis| axis.plan.plan_bytes() - size_of::<RealFftPlan<S>>()),
        )
    }

    /// Returns the backend of each one-dimensional child plan, one per selected
    /// axis, with the packed real axis last.
    ///
    /// Axes that share a child report the same backend. A plan with no selected
    /// axes or no elements has no children.
    pub fn resolved_backends(&self) -> impl Iterator<Item = crate::FftBackend> + '_ {
        self.children
            .iter()
            .flat_map(|(_, p)| p.resolved_backends())
            .chain(
                self.real
                    .iter()
                    .flat_map(|axis| axis.plan.resolved_backends()),
            )
    }
    /// Prepares a plan with the default processor capabilities.
    ///
    /// Uses [`ExecutionContext::sequential_native`]; see [`Self::build`].
    ///
    /// # Errors
    ///
    /// The same as [`Self::build`].
    pub fn new(spec: FftNdPlanSpec) -> Result<Self, SciOxideError> {
        Self::build(spec, scioxide_core::ExecutionContext::sequential_native())
    }
    /// Prepares a plan with the given processor and provider capabilities.
    ///
    /// See [Backends and providers](crate#backends-and-providers).
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if an axis is not less than the rank of the shape.
    /// - `InvalidInput` if an axis is repeated, a selected extent is zero, or a
    ///   real or Hermitian layout has no selected axis.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if a required backend is
    ///   not available in `context`.
    /// - `ExecutionStopped` if `context` is already cancelled or past its
    ///   deadline.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the plan or
    ///   its workspace size cannot be represented or allocated.
    pub fn build<'a>(
        spec: FftNdPlanSpec,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
    ) -> Result<Self, SciOxideError> {
        let shape_argument = scioxide_core::argument!("spec.shape");
        let context = context.into();
        context.execution().check()?;
        let _ = context.resolve(spec.backend)?;
        let len = validate_axes(
            &spec.shape,
            &spec.axes,
            shape_argument,
            scioxide_core::argument!("spec.axes"),
        )?;
        if spec.layout != FftLayout::Complex && spec.axes.is_empty() {
            return Err(invalid(
                scioxide_core::InvalidInputReason::Empty,
                scioxide_core::argument!("spec.axes"),
                "packed transforms require an axis",
            ));
        }
        let dense_strides = strides(&spec.shape)?;
        let mut packed_shape = copy_metadata(&spec.shape)?;
        let packed_axis = if spec.layout == FftLayout::Complex {
            None
        } else {
            spec.axes.last().copied()
        };
        if let Some(axis) = packed_axis {
            packed_shape[axis] = crate::io::packed_len(spec.shape[axis])?;
        }
        let packed_len = Shape::new(&packed_shape)?.len();
        let packed_strides = strides(&packed_shape)?;
        let full_batches = BatchGeometry::new(&spec.shape, &spec.axes)?;
        let packed_batches = BatchGeometry::new(&packed_shape, &spec.axes)?;
        // With one selected axis every lane is its own batch, so lanes span
        // the whole tensor; otherwise they span one batch.
        let single = spec.axes.len() == 1;
        let real_forward = spec.real_forward();
        let (full_lanes, packed_lanes) = if single {
            (&spec.shape[..], &packed_shape[..])
        } else {
            (&full_batches.shape[..], &packed_batches.shape[..])
        };
        let headroom: i32 = spec
            .axes
            .iter()
            .map(|&axis| crate::scaling::growth_exponent(spec.shape[axis]))
            .sum();
        let bounded = S::ONE.scalbn(headroom).is_finite();
        let headroom = if bounded { headroom } else { 0 };
        let PreparedAxes {
            children,
            real,
            staging_len,
            child_scratch,
            factor,
        } = PreparedAxes::new(
            &spec,
            context,
            len,
            (full_lanes, &dense_strides),
            (packed_lanes, &packed_strides),
        )?;
        let (forward, inverse) = spec.normalization.fourier_scales(factor)?;
        let scale = match spec.direction {
            FftDirection::Forward => forward,
            FftDirection::Inverse => inverse,
        };
        // Scratch layout: [full-shape candidate (complex and packed-inverse
        // layouts) | packed candidate (packed layouts) | strided-lane staging |
        // shared child scratch].
        // Real forwards read their input in place and need no full candidate.
        let storage = match packed_axis {
            Some(_) if real_forward => packed_len,
            Some(_) => checked_add(len, packed_len, scioxide_core::quantity!("n_d_candidates"))?,
            None => len,
        };
        let scratch_len = checked_add(
            storage,
            checked_add(
                staging_len,
                child_scratch,
                scioxide_core::quantity!("n_d_lane_scratch"),
            )?,
            scioxide_core::quantity!("n_d_scratch"),
        )?;
        let scratch = crate::FftWorkspaceRequirements::new::<S>(scratch_len)?;
        let nested =
            |limit, child| crate::workspace::nested_byte_limit(limit, scratch.bytes(), Some(child));
        let limit = children.iter().try_fold(usize::MAX, |limit, (_, child)| {
            let child = child.lane_workspace_requirements()?;
            Ok::<_, SciOxideError>(nested(limit, (child.byte_limit(), child.bytes())))
        })?;
        let limit = real.as_ref().map_or(limit, |RealAxis { plan: child, .. }| {
            nested(
                limit,
                (
                    child.workspace_byte_limit(),
                    child.scratch_len() * size_of::<Complex<S>>(),
                ),
            )
        });
        let scratch = scratch.with_limit(limit)?;
        let output_bound = if children.iter().any(|(_, p)| {
            p.resolved_backends()
                .any(|b| b == crate::FftBackend::Provider)
        }) || real.as_ref().is_some_and(|axis| {
            axis.plan
                .resolved_backends()
                .any(|b| b == crate::FftBackend::Provider)
        }) {
            S::INFINITY
        } else {
            scalar_from_usize_rounded::<S>(factor)? * S::TWO * scale
        };
        // Single-axis children project endpoints like their 1-D equivalents.
        let endpoint_pairs = if let Some(axis) = packed_axis.filter(|_| len != 0 && !single) {
            endpoint_pairs(&spec, axis, &packed_shape, &packed_strides)?
        } else {
            Vec::new()
        };
        let plan = Self {
            spec,
            dense_strides,
            packed_shape,
            packed_strides,
            full_batches,
            packed_batches,
            endpoint_pairs,
            len,
            packed_len,
            children,
            real,
            staging_len,
            scratch,
            scale,
            // The product of axis growth, including complex rotation margin,
            // keeps every subsequent lane input within the bounded contract.
            headroom,
            bounded,
            output_bound,
        };
        plan.byte_counts().try_fold(0, |a, b| {
            checked_add(a, b, scioxide_core::quantity!("fft_plan_bytes"))
        })?;
        Ok(plan)
    }
    /// Returns the spec the plan was prepared from.
    pub fn spec(&self) -> &FftNdPlanSpec {
        &self.spec
    }
    /// Returns the shape of the packed side, with the last selected axis
    /// reduced to `floor(n/2) + 1`; for complex plans, the full shape.
    pub fn packed_shape(&self) -> &[usize] {
        &self.packed_shape
    }
    /// Borrow prepared dense layouts without allocating convenience metadata.
    pub(crate) fn dense_layout(&self) -> Result<StridedLayout<'_>, SciOxideError> {
        StridedLayout::new(&self.spec.shape, &self.dense_strides)
    }
    pub(crate) fn packed_layout(&self) -> Result<StridedLayout<'_>, SciOxideError> {
        StridedLayout::new(&self.packed_shape, &self.packed_strides)
    }

    /// Returns the workspace size that [`Self::execute_into`] and
    /// [`Self::execute_in_place`] need.
    ///
    /// # Errors
    ///
    /// None: the requirement is computed when the plan is prepared.
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<crate::FftWorkspaceRequirements, SciOxideError> {
        Ok(self.scratch)
    }

    /// Allocates a workspace that meets [`Self::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn workspace(&self, request: ()) -> Result<crate::FftWorkspace<S>, SciOxideError> {
        crate::workspace::allocate(&self.workspace_requirements(request)?)
    }

    /// Transforms `input` into `output`.
    ///
    /// Real and complex [`NdView`]s and [`NdViewMut`]s convert into
    /// [`FftNdInput`] and [`FftNdOutput`]. The views must have the plan's
    /// logical shape, or its [packed shape](Self::packed_shape) on the packed
    /// side. `scratch` must hold at least [`Self::workspace_requirements`]
    /// complex values.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the input or output
    ///   variant does not match the plan's direction and layout.
    /// - `DimensionMismatch` if a view has the wrong shape or `scratch` is too
    ///   short.
    /// - `InvalidInput` if packed input to a real or Hermitian inverse is not
    ///   Hermitian within the tolerance of [Hermitian input](crate#hermitian-input).
    ///   On the DC and Nyquist planes, bins pair across the other selected axes.
    /// - `NonFinite` if the input contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if a result overflows, or
    ///   (`Breakdown`) if a real result keeps an imaginary residual above its
    ///   rounding bound.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn execute_into<'d, 'l, 'e, 'm>(
        &self,
        input: impl Into<FftNdInput<'d, 'l, S>>,
        output: impl Into<FftNdOutput<'e, 'm, S>>,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (input, mut output) = (input.into(), output.into());
        let complex = self.spec.layout == FftLayout::Complex;
        let real_forward = self.spec.real_forward();
        match (&input, &output) {
            (FftNdInput::Complex(i), FftNdOutput::Complex(o)) if complex => {
                check_shape(i.layout().shape().extents(), &self.spec.shape)?;
                check_shape(o.layout().shape().extents(), &self.spec.shape)?;
            }
            (FftNdInput::Real(i), FftNdOutput::Complex(o)) if !complex && real_forward => {
                check_shape(i.layout().shape().extents(), &self.spec.shape)?;
                check_shape(o.layout().shape().extents(), &self.packed_shape)?;
            }
            (FftNdInput::Complex(i), FftNdOutput::Real(o)) if !complex && !real_forward => {
                check_shape(i.layout().shape().extents(), &self.packed_shape)?;
                check_shape(o.layout().shape().extents(), &self.spec.shape)?;
            }
            _ => {
                return Err(crate::io::layout_error(scioxide_core::argument!("input")));
            }
        }
        crate::validation::sufficient(scratch.len(), self.scratch.complex_len())?;
        context.check()?;
        if self.len == 0 {
            return Ok(());
        }
        let (result, pending) = self.execute_candidates(input, scratch, context)?;
        match pending {
            None => self.publish(result, &mut output, |value| value),
            Some(BinaryScale::Identity) => {
                self.publish(result, &mut output, |value| self.finish(value, |v| v));
            }
            Some(BinaryScale::Multiply(factor)) => {
                self.publish(result, &mut output, |value| {
                    self.finish(value, |v| v * factor)
                });
            }
            Some(BinaryScale::Extended(exponent)) => self.publish(result, &mut output, |value| {
                self.finish(value, |v| v.scalbn(exponent))
            }),
        }
        Ok(())
    }

    /// Write candidate `result` through `finish` into the output view.
    fn publish(
        &self,
        result: &[Complex<S>],
        output: &mut FftNdOutput<'_, '_, S>,
        finish: impl Fn(Complex<S>) -> Complex<S>,
    ) {
        match output {
            // The Hermitian inverse conjugates its packed bins. Exact zeros
            // stay positive, as after restoration.
            FftNdOutput::Complex(view) if self.spec.layout == FftLayout::Hermitian => {
                crate::tensor::copy_out(result, view, |value| {
                    let value = finish(value);
                    Complex::new(value.re, S::ZERO - value.im)
                });
            }
            FftNdOutput::Complex(view) => crate::tensor::copy_out(result, view, finish),
            FftNdOutput::Real(view) => {
                crate::tensor::copy_out(result, view, |value| finish(value).re)
            }
        }
    }

    /// Apply a restoration deferred to publication, after the plan
    /// normalization, exactly as the batch restore pass would.
    #[inline(always)]
    fn finish(&self, value: Complex<S>, restore: impl Fn(S) -> S) -> Complex<S> {
        let value = value * self.scale;
        let canonical = |v: S| if v == S::ZERO { S::ZERO } else { restore(v) };
        Complex::new(canonical(value.re), canonical(value.im))
    }

    /// Elements of the full-shape candidate: complex transforms and packed
    /// inverses write from it; real forwards read their input in place.
    fn full_len(&self) -> usize {
        if self.spec.layout != FftLayout::Complex && self.spec.real_forward() {
            0
        } else {
            self.len
        }
    }

    /// Execute into scratch candidates. Returns the finished candidate and,
    /// for a single bounded batch whose restoration is proven finite, the
    /// binary restoration still to apply after the plan normalization.
    fn execute_candidates<'a>(
        &self,
        input: FftNdInput<'_, '_, S>,
        scratch: &'a mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<Published<'a, S>, SciOxideError> {
        let packs = self.spec.layout != FftLayout::Complex;
        let (full, rest) = scratch.split_at_mut(self.full_len());
        let (packed, rest) = rest.split_at_mut(if packs { self.packed_len } else { 0 });
        let (staging, child) = rest.split_at_mut(self.staging_len);
        let mut work = Work {
            full,
            packed,
            staging,
            child,
        };
        let pending = crate::tensor::BatchScaling {
            axes: self.spec.axes.len(),
            count: self.full_batches.count,
            headroom: self.headroom,
            bounded: self.bounded,
            normalization: self.scale,
            output_bound: self.output_bound,
            overflow: "FFT output exceeds the finite scalar range",
        }
        .run(&mut FourierBatches {
            plan: self,
            input: &input,
            work: &mut work,
            context,
        })?;
        context.check()?;
        let result = if packs && self.spec.real_forward() {
            work.packed
        } else {
            work.full
        };
        Ok((result, pending))
    }

    /// One selected axis: every lane is an independent batch. Children run
    /// checked lanes across all batches with the plan normalization, exactly
    /// as the matching one-dimensional plan; no parent scaling is needed.
    fn execute_lanes(
        &self,
        input: &FftNdInput<'_, '_, S>,
        work: &mut Work<'_, S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let axis = self.spec.axes[0];
        let dense = |geometry, packed: bool| Placement {
            base: 0,
            geometry,
            shape: if packed {
                &self.packed_shape
            } else {
                &self.spec.shape
            },
            strides: if packed {
                &self.packed_strides
            } else {
                &self.dense_strides
            },
        };
        let mode = LaneMode::Checked;
        match (input, &self.real, self.children.first()) {
            // Real axes read their source lanes in place.
            (FftNdInput::Real(view), Some(real), _) => {
                let layout = view.layout();
                let strides = layout.strides();
                let source = Placement {
                    base: 0,
                    geometry: LaneGeometry::new::<S>(&self.spec.shape, strides, axis),
                    shape: &self.spec.shape,
                    strides,
                };
                real.plan.forward_lanes(
                    LanePairs::new(
                        (view.storage(), source),
                        (&mut *work.packed, dense(real.packed, true)),
                        work.staging,
                    ),
                    work.child,
                    mode,
                    (BinaryScale::Identity, self.scale),
                    context,
                )
            }
            (FftNdInput::Complex(view), Some(real), _) => {
                let layout = view.layout();
                let strides = layout.strides();
                let source = Placement {
                    base: 0,
                    geometry: LaneGeometry::new::<Complex<S>>(&self.packed_shape, strides, axis),
                    shape: &self.packed_shape,
                    strides,
                };
                let sign = if self.spec.layout == FftLayout::Hermitian {
                    -S::ONE
                } else {
                    S::ONE
                };
                real.plan.inverse_lanes(
                    LanePairs::new(
                        (view.storage(), source),
                        (&mut *work.full, dense(real.full, false)),
                        work.staging,
                    ),
                    work.child,
                    mode,
                    (sign, self.scale),
                    context,
                )
            }
            (FftNdInput::Complex(view), None, Some(&(geometry, ref plan))) => {
                copy_in(view, &self.dense_strides, work.full, |value| value);
                plan.transform_lanes(
                    Lanes::new(work.full, dense(geometry, false), work.staging),
                    work.child,
                    mode,
                    self.scale,
                    context,
                )
            }
            _ => Ok(()),
        }
    }

    /// Run every complex axis over one batch's lanes. Unit axes are skipped:
    /// the batch scan already checked finiteness, and their unscaled
    /// transform is the identity.
    fn transform_axes(
        &self,
        values: &mut [Complex<S>],
        (shape, strides): (&[usize], &[usize]),
        (staging, child): (&mut [Complex<S>], &mut [Complex<S>]),
        mode: LaneMode,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.children
            .iter()
            .filter(|(geometry, _)| geometry.len > 1)
            .try_for_each(|&(geometry, ref plan)| {
                plan.transform_lanes(
                    Lanes::new(
                        values,
                        Placement {
                            base: 0,
                            geometry,
                            shape,
                            strides,
                        },
                        staging,
                    ),
                    child,
                    mode,
                    S::ONE,
                    context,
                )
            })
    }
    fn project_endpoints(
        &self,
        packed: &mut [Complex<S>],
        maximum: S,
    ) -> Result<(), SciOxideError> {
        let projector = crate::hermitian::Projector::new(maximum);
        for &(index, mirror) in &self.endpoint_pairs {
            let projected =
                projector.project(packed[index], (index != mirror).then(|| packed[mirror]))?;
            packed[index] = projected;
            if index != mirror {
                packed[mirror] = projected.conj();
            }
        }
        Ok(())
    }

    fn clear_endpoint_residual(
        &self,
        packed: &mut [Complex<S>],
        bound: S,
    ) -> Result<(), SciOxideError> {
        for &(index, mirror) in &self.endpoint_pairs {
            for offset in [index, mirror] {
                let value = &mut packed[offset];
                if !value.is_finite() || value.im.abs() > bound {
                    return Err(crate::validation::numerical(
                        scioxide_core::NumericalIssue::Breakdown,
                        "inverse complex axes produced excessive endpoint imaginary residual",
                    ));
                }
                value.im = S::ZERO;
            }
        }
        Ok(())
    }

    /// Transforms complex `data` in place.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) for a real or Hermitian
    ///   plan.
    /// - `DimensionMismatch` if `data` has the wrong shape or `scratch` is too
    ///   short.
    /// - `NonFinite` if `data` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if a result overflows.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn execute_in_place(
        &self,
        data: NdViewMut<'_, '_, Complex<S>>,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if self.spec.layout != FftLayout::Complex {
            return Err(crate::io::layout_error(scioxide_core::argument!("self")));
        }
        check_shape(data.layout().shape().extents(), &self.spec.shape)?;
        crate::validation::sufficient(scratch.len(), self.scratch.complex_len())?;
        context.check()?;
        if self.len == 0 {
            return Ok(());
        }
        let (candidate, pending) =
            self.execute_candidates(FftNdInput::Complex(data.as_view()), scratch, context)?;
        let mut output = FftNdOutput::Complex(data);
        match pending {
            None => self.publish(candidate, &mut output, |value| value),
            Some(BinaryScale::Identity) => {
                self.publish(candidate, &mut output, |value| self.finish(value, |v| v));
            }
            Some(BinaryScale::Multiply(factor)) => {
                self.publish(candidate, &mut output, |value| {
                    self.finish(value, |v| v * factor)
                });
            }
            Some(BinaryScale::Extended(exponent)) => {
                self.publish(candidate, &mut output, |value| {
                    self.finish(value, |v| v.scalbn(exponent))
                })
            }
        }
        Ok(())
    }
}

/// Fourier storage and endpoint handling for the common tensor driver.
struct FourierBatches<'a, 'w, 'd, 'l, 'c, S: RealScalar> {
    plan: &'a FftNdPlan<S>,
    input: &'a FftNdInput<'d, 'l, S>,
    work: &'a mut Work<'w, S>,
    context: ExecutionContext<'c>,
}
impl<S: RealScalar> crate::tensor::BatchTransform<S> for FourierBatches<'_, '_, '_, '_, '_, S> {
    fn single_axis(&mut self) -> Result<(), SciOxideError> {
        self.plan.execute_lanes(self.input, self.work, self.context)
    }
    fn prepare(&mut self, batch: usize) -> Result<S, SciOxideError> {
        let plan = self.plan;
        let geometry = if plan.spec.layout == FftLayout::Complex || plan.spec.real_forward() {
            &plan.full_batches
        } else {
            &plan.packed_batches
        };
        match self.input {
            FftNdInput::Complex(view) => {
                let layout = view.layout();
                let strides = layout.strides();
                let source = &view.storage()[geometry.base(batch, strides)..];
                if plan.real.is_some() {
                    let target =
                        &mut self.work.packed[geometry.base(batch, &plan.packed_strides)..];
                    let conjugate = plan.spec.layout == FftLayout::Hermitian;
                    if let Some(source) = geometry.slice(strides, source)
                        && geometry.slice(&plan.packed_strides, target).is_some()
                    {
                        return crate::scaling::copy_complex_max(
                            source,
                            &mut target[..geometry.len],
                            conjugate,
                        );
                    }
                    // Measure original components while staging. Every input
                    // is checked before the first Hermitian projection.
                    crate::scaling::complex_max(
                        geometry
                            .offsets(strides)
                            .zip(geometry.offsets(&plan.packed_strides))
                            .map(|(from, to)| {
                                let value = source[from];
                                target[to] = if conjugate { value.conj() } else { value };
                                value
                            }),
                    )
                } else {
                    crate::tensor::BatchInput::Complex(source).maximum(geometry, strides)
                }
            }
            FftNdInput::Real(view) => {
                let layout = view.layout();
                let strides = layout.strides();
                crate::tensor::BatchInput::Real(&view.storage()[geometry.base(batch, strides)..])
                    .maximum(geometry, strides)
            }
        }
    }
    fn transform(
        &mut self,
        batch: usize,
        maximum: S,
        scaling: BinaryScale<S>,
        mode: LaneMode,
    ) -> Result<(), SciOxideError> {
        let plan = self.plan;
        let input = self.input;
        let context = self.context;
        let complex = plan.spec.layout == FftLayout::Complex;
        let real_forward = plan.spec.real_forward();
        let full_lanes = (&plan.full_batches.shape[..], &plan.dense_strides[..]);
        let packed_lanes = (&plan.packed_batches.shape[..], &plan.packed_strides[..]);
        let fbase = plan.full_batches.base(batch, &plan.dense_strides);
        let pbase = plan.packed_batches.base(batch, &plan.packed_strides);
        let full = &mut self.work.full[if complex || !real_forward { fbase } else { 0 }..];
        let packed = &mut self.work.packed[if complex { 0 } else { pbase }..];
        let (geometry, strides) = if complex || real_forward {
            (&plan.full_batches, &plan.dense_strides)
        } else {
            (&plan.packed_batches, &plan.packed_strides)
        };
        let scaled =
            |value: Complex<S>| Complex::new(scaling.apply(value.re), scaling.apply(value.im));
        match (input, &plan.real) {
            (FftNdInput::Real(view), Some(real)) => {
                let layout = view.layout();
                let input_strides = layout.strides();
                let source = Placement {
                    base: geometry.base(batch, input_strides),
                    geometry: LaneGeometry::new::<S>(
                        full_lanes.0,
                        input_strides,
                        plan.spec.axes[plan.spec.axes.len() - 1],
                    ),
                    shape: full_lanes.0,
                    strides: input_strides,
                };
                real.plan.forward_lanes(
                    LanePairs::new(
                        (view.storage(), source),
                        (
                            &mut *packed,
                            Placement::new(real.packed, packed_lanes.0, packed_lanes.1),
                        ),
                        &mut *self.work.staging,
                    ),
                    self.work.child,
                    mode,
                    (scaling, S::ONE),
                    context,
                )?;
                plan.transform_axes(
                    packed,
                    packed_lanes,
                    (self.work.staging, self.work.child),
                    mode,
                    context,
                )?;
            }
            (FftNdInput::Complex(_), Some(real)) => {
                plan.project_endpoints(packed, maximum)?;
                // Preserve the mass reducer's row-major order and rounding,
                // feeding it the projected values as they are range-scaled.
                let mass = if geometry.slice(strides, packed).is_some() {
                    crate::hermitian::component_mass(packed[..geometry.len].iter_mut().map(
                        |slot| {
                            *slot = scaled(*slot);
                            *slot
                        },
                    ))
                } else {
                    crate::hermitian::component_mass(geometry.offsets(strides).map(|offset| {
                        let value = scaled(packed[offset]);
                        packed[offset] = value;
                        value
                    }))
                };
                let length = plan.packed_batches.len / real.packed.len;
                let bound = crate::hermitian::residual_bound(length, mass)?;
                plan.transform_axes(
                    packed,
                    packed_lanes,
                    (self.work.staging, self.work.child),
                    mode,
                    context,
                )?;
                plan.clear_endpoint_residual(packed, bound)?;
                real.plan.inverse_lanes(
                    LanePairs::new(
                        (
                            &*packed,
                            Placement::new(real.packed, packed_lanes.0, packed_lanes.1),
                        ),
                        (
                            &mut *full,
                            Placement::new(real.full, full_lanes.0, full_lanes.1),
                        ),
                        &mut *self.work.staging,
                    ),
                    self.work.child,
                    mode,
                    (S::ONE, S::ONE),
                    context,
                )?;
            }
            (FftNdInput::Complex(view), None) => {
                let layout = view.layout();
                let input_strides = layout.strides();
                let base = geometry.base(batch, input_strides);
                geometry.copy(
                    (&view.storage()[base..], input_strides),
                    (full, strides),
                    scaled,
                );
                plan.transform_axes(
                    full,
                    full_lanes,
                    (self.work.staging, self.work.child),
                    mode,
                    context,
                )?;
            }
            (FftNdInput::Real(_), None) => {
                unreachable!("execution validated the input representation")
            }
        }

        Ok(())
    }
    fn output(
        &mut self,
        batch: usize,
    ) -> (crate::tensor::BatchOutput<'_, S>, &BatchGeometry, &[usize]) {
        let plan = self.plan;
        let (values, geometry, strides) =
            if plan.spec.layout != FftLayout::Complex && plan.spec.real_forward() {
                (
                    &mut *self.work.packed,
                    &plan.packed_batches,
                    &plan.packed_strides,
                )
            } else {
                (
                    &mut *self.work.full,
                    &plan.full_batches,
                    &plan.dense_strides,
                )
            };
        let base = geometry.base(batch, strides);
        (
            crate::tensor::BatchOutput::Complex(&mut values[base..]),
            geometry,
            strides,
        )
    }
}

/// Child preparation stays separate from shape validation and workspace sizing.
struct PreparedAxes<S: RealScalar> {
    children: Vec<(LaneGeometry, FftPlan<S>)>,
    real: Option<RealAxis<S>>,
    staging_len: usize,
    child_scratch: usize,
    factor: usize,
}
impl<S: RealScalar> PreparedAxes<S> {
    fn new(
        spec: &FftNdPlanSpec,
        context: crate::FftPlanningContext<'_, S>,
        len: usize,
        (full_lanes, dense_strides): (&[usize], &[usize]),
        (packed_lanes, packed_strides): (&[usize], &[usize]),
    ) -> Result<Self, SciOxideError> {
        let shape_argument = scioxide_core::argument!("spec.shape");
        let provider = scioxide_core::argument!("context");
        let packed_axis =
            (spec.layout != FftLayout::Complex).then(|| spec.axes[spec.axes.len() - 1]);
        let real_forward = spec.real_forward();
        // The Hermitian layout is the conjugate dual of the real layout:
        //   hfft(x)  = unscaled irfft(conj(x))   (already real, so no final conj)
        //   ihfft(y) = conj(unscaled rfft(y))
        // using DFT_forward(v) = conj(DFT_inverse(conj(v))). So complex
        // children run in the opposite direction, the input (Hermitian
        // forward) or output (Hermitian inverse) is conjugated, and the same
        // real/packed machinery as `FftLayout::Real` does the rest.
        let direction = if spec.layout == FftLayout::Hermitian {
            match spec.direction {
                FftDirection::Forward => FftDirection::Inverse,
                FftDirection::Inverse => FftDirection::Forward,
            }
        } else {
            spec.direction
        };
        let mut children: Vec<(LaneGeometry, FftPlan<S>)> =
            try_with_capacity(spec.axes.len(), scioxide_core::quantity!("n_d_child_plans"))?;
        let mut real = None;
        let mut staging_len = 0;
        let mut child_scratch = 0;
        // All children are unscaled; the normalization of the whole
        // transform (product of the full selected extents, including the
        // logical length of the packed axis) is applied once at commit time.
        // The packed axis is always the last selected axis and gets the real
        // child; every other axis gets a complex child.
        let mut factor = 1;
        if len != 0 {
            for &axis in &spec.axes {
                let n = spec.shape[axis];
                factor = checked_mul(factor, n, scioxide_core::quantity!("n_d_normalization"))?;
                if Some(axis) == packed_axis {
                    let child = match spec.backend {
                        FftBackendStrategy::Automatic => RealFftPlan::with_context(
                            n,
                            spec.policy,
                            context,
                            shape_argument,
                            provider,
                        )?,
                        FftBackendStrategy::Required(backend) => {
                            RealFftPlan::with_backend_and_context(
                                n,
                                spec.policy,
                                backend,
                                context,
                                shape_argument,
                                provider,
                            )?
                        }
                    };
                    child_scratch = child_scratch.max(child.lane_scratch_len());
                    let full = LaneGeometry::new::<Complex<S>>(full_lanes, dense_strides, axis);
                    let packed =
                        LaneGeometry::new::<Complex<S>>(packed_lanes, packed_strides, axis);
                    // Real forwards stage packed targets; inverses full ones.
                    // Bounded block passes stage one interleaved block.
                    let target = if real_forward { &packed } else { &full };
                    staging_len = staging_len.max(target.staging_len());
                    real = Some(RealAxis {
                        full,
                        packed,
                        plan: child,
                    });
                } else {
                    let child_spec =
                        FftPlanSpec::new(n, direction, Normalization::Unscaled, FftLayout::Complex)
                            .with_execution_policy(spec.policy)
                            .with_backend_strategy(spec.backend);
                    let child = match children.iter().find(|(_, p)| p.spec() == &child_spec) {
                        Some((_, child)) => child.clone(),
                        None => FftPlan::build(child_spec, context)?,
                    };
                    child_scratch =
                        child_scratch.max(child.lane_workspace_requirements()?.complex_len());
                    let geometry = if packed_axis.is_some() {
                        LaneGeometry::new::<Complex<S>>(packed_lanes, packed_strides, axis)
                    } else {
                        LaneGeometry::new::<Complex<S>>(full_lanes, dense_strides, axis)
                    };
                    staging_len = staging_len.max(geometry.staging_len());
                    children.push((geometry, child));
                }
            }
        }

        Ok(Self {
            children,
            real,
            staging_len,
            child_scratch,
            factor,
        })
    }
}

/// Conjugate pairs of the packed DC and Nyquist planes within one batch.
fn endpoint_pairs(
    spec: &FftNdPlanSpec,
    axis: usize,
    packed_shape: &[usize],
    packed_strides: &[usize],
) -> Result<Vec<(usize, usize)>, SciOxideError> {
    let plane = BatchGeometry::new(packed_shape, &spec.axes[..spec.axes.len() - 1])?;
    let endpoints = if spec.shape[axis].is_multiple_of(2) {
        2
    } else {
        1
    };
    let mut endpoint_pairs = try_with_capacity(
        checked_mul(
            plane.len,
            endpoints,
            scioxide_core::quantity!("fft_endpoint_pairs"),
        )?,
        scioxide_core::quantity!("fft_endpoint_pairs"),
    )?;
    for endpoint in 0..endpoints {
        let start = endpoint * (spec.shape[axis] / 2) * packed_strides[axis];
        for offset in plane.offsets(packed_strides) {
            let index = start + offset;
            let mut mirror = index;
            for &other in &spec.axes[..spec.axes.len() - 1] {
                let coordinate = index / packed_strides[other] % packed_shape[other];
                if coordinate != 0 {
                    mirror = mirror - coordinate * packed_strides[other]
                        + (packed_shape[other] - coordinate) * packed_strides[other];
                }
            }
            if index <= mirror {
                endpoint_pairs.push((index, mirror));
            }
        }
    }

    Ok(endpoint_pairs)
}

impl FftNdPlanSpec {
    /// Whether the real axis maps samples to packed bins, including Hermitian duals.
    const fn real_forward(&self) -> bool {
        matches!(
            (self.layout, self.direction),
            (FftLayout::Real, FftDirection::Forward)
                | (FftLayout::Hermitian, FftDirection::Inverse)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_axes_share_only_matching_complex_cores() -> Result<(), SciOxideError> {
        for n in [8, 23, 47] {
            for layout in [FftLayout::Complex, FftLayout::Real, FftLayout::Hermitian] {
                let plan = FftNdPlan::<f64>::new(FftNdPlanSpec::new(
                    &[n, n, n + 1, n],
                    &[0, 1, 2, 3],
                    FftDirection::Forward,
                    Normalization::Inverse,
                    layout,
                )?)?;
                assert!(plan.children[0].1.shares_core(&plan.children[1].1));
                assert!(!plan.children[0].1.shares_core(&plan.children[2].1));
                if layout == FftLayout::Complex {
                    assert!(plan.children[0].1.shares_core(&plan.children[3].1));
                }
                let cloned = plan.clone();
                assert!(cloned.children[0].1.shares_core(&plan.children[0].1));
            }
        }
        Ok(())
    }
}
