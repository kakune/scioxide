//! Multidimensional DCT and DST along selected axes.
use crate::lanes::{LaneGeometry, LaneMode, Lanes, Placement};
use crate::scaling::{BinaryScale, growth_exponent};
use crate::tensor::{BatchGeometry, check_shape, copy_in, copy_metadata, strides, validate_axes};
use crate::{
    DctType, DstType, FftBackendStrategy, FftDirection, Normalization, RealTransformKind,
    RealTransformPlan,
};
use alloc::{sync::Arc, vec::Vec};
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, RealScalar, SciOxideError,
    StridedLayout,
};

// Tensor geometry and range isolation are independent of the selected kernels.
impl<S: RealScalar> RealTransformNdPlan<S> {
    fn byte_counts(&self) -> impl Iterator<Item = usize> + '_ {
        [
            size_of::<Self>(),
            crate::memory::vector_bytes(&self.spec.types),
            crate::memory::vector_bytes(&self.children),
            crate::memory::vector_bytes(&self.axes),
        ]
        .into_iter()
        .chain(
            [
                &self.spec.shape,
                &self.spec.axes,
                &self.strides,
                &self.batches.shape,
                &self.batches.batch_shape,
            ]
            .into_iter()
            .map(crate::memory::vector_bytes),
        )
        .chain(
            self.children
                .iter()
                .enumerate()
                .filter(|(i, p)| {
                    !self.children[..*i]
                        .iter()
                        .any(|earlier| Arc::ptr_eq(p, earlier))
                })
                .map(|(_, p)| p.plan_bytes() + 2 * size_of::<usize>()),
        )
    }

    fn validate(
        &self,
        shape: &[usize],
        real: &[S],
        complex: &[Complex<S>],
    ) -> Result<(), SciOxideError> {
        check_shape(shape, &self.spec.shape)?;
        for (actual, required) in [
            (real.len(), self.scratch.real_len()),
            (complex.len(), self.scratch.complex().complex_len()),
        ] {
            if actual < required {
                return Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::one(required),
                    scioxide_core::Dimensions::one(actual),
                ));
            }
        }
        Ok(())
    }
    fn transform(
        &self,
        source: NdView<'_, '_, S>,
        real: &mut [S],
        complex: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<Option<BinaryScale<S>>, SciOxideError> {
        context.check()?;
        if self.len == 0 {
            return Ok(None);
        }
        let (candidate, rest) = real.split_at_mut(self.len);
        let staging = &mut rest[..self.staging_len];
        let pending = crate::tensor::BatchScaling {
            axes: self.spec.axes.len(),
            count: self.batches.count,
            headroom: self.headroom,
            bounded: self.bounded,
            normalization: S::ONE,
            output_bound: self.output_bound,
            overflow: "real tensor output exceeds scalar range",
        }
        .run(&mut TrigBatches {
            plan: self,
            source,
            candidate,
            staging,
            complex,
            context,
        })?;
        context.check()?;
        Ok(pending)
    }

    /// Write the output only after the complete transform and its final
    /// context check. A proven-finite restoration shares the output copy's
    /// traversal.
    fn publish(
        &self,
        values: &[S],
        output: &mut NdViewMut<'_, '_, S>,
        pending: Option<BinaryScale<S>>,
    ) {
        let canonical = |value: S, restored: S| if value == S::ZERO { S::ZERO } else { restored };
        match pending {
            None => crate::tensor::copy_out(values, output, |value| value),
            Some(BinaryScale::Identity) => {
                crate::tensor::copy_out(values, output, |value| canonical(value, value));
            }
            Some(BinaryScale::Multiply(factor)) => {
                crate::tensor::copy_out(values, output, |value| canonical(value, value * factor));
            }
            Some(BinaryScale::Extended(exponent)) => {
                crate::tensor::copy_out(values, output, |value| {
                    canonical(value, value.scalbn(exponent))
                });
            }
        }
    }
}

/// Describes a multidimensional DCT or DST plan: shape, selected axes, one
/// transform type per axis, direction and normalization.
#[derive(Debug, Clone)]
pub struct RealTransformNdPlanSpec {
    shape: Vec<usize>,
    axes: Vec<usize>,
    types: Vec<RealTransformKind>,
    direction: FftDirection,
    normalization: Normalization,
    policy: ExecutionPolicy,
    backend: FftBackendStrategy,
}
impl RealTransformNdPlanSpec {
    /// Returns the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Returns the selected axes.
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
    /// Returns the transform type of each selected axis.
    pub fn types(&self) -> &[RealTransformKind] {
        &self.types
    }
    /// Returns the direction.
    pub const fn direction(&self) -> FftDirection {
        self.direction
    }
    /// Returns the normalization.
    pub const fn normalization(&self) -> Normalization {
        self.normalization
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the backend strategy.
    pub const fn backend_strategy(&self) -> FftBackendStrategy {
        self.backend
    }

    /// Creates a spec with sequential execution and automatic backend selection.
    ///
    /// `types[i]` is the forward type on `axes[i]`. The spec copies its
    /// arguments; [`RealTransformNdPlan::new`] validates them.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the copies cannot be allocated.
    pub fn new(
        shape: &[usize],
        axes: &[usize],
        types: &[RealTransformKind],
        direction: FftDirection,
        normalization: Normalization,
    ) -> Result<Self, SciOxideError> {
        Self::from_types(shape, axes, types.iter().copied(), direction, normalization)
    }
    fn from_types(
        shape: &[usize],
        axes: &[usize],
        types: impl ExactSizeIterator<Item = RealTransformKind>,
        direction: FftDirection,
        normalization: Normalization,
    ) -> Result<Self, SciOxideError> {
        let mut kinds =
            try_with_capacity(types.len(), scioxide_core::quantity!("transform_types"))?;
        kinds.extend(types);
        Ok(Self {
            shape: copy_metadata(shape)?,
            axes: copy_metadata(axes)?,
            types: kinds,
            direction,
            normalization,
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
}
/// Computes a DCT or DST along selected axes of a real tensor, with one type
/// per axis.
///
/// The transform applies the one-dimensional transform of
/// [`RealTransformPlan`](crate::RealTransformPlan) along each selected axis in
/// turn: `types[i]` along `axes[i]`, with that axis's length, endpoint
/// weights and scale. An inverse plan applies the dual type of each forward
/// type, with the same normalization. Unselected axes are independent batches.
/// The unscaled round-trip factor is the product of the factors of the
/// selected axes; the other normalizations give an identity round trip.
///
/// With `V` elements, the work is `O(V Σᵢ log nᵢ)` over the selected axis
/// lengths `nᵢ`, and the workspace holds `V` real values plus the scratch of
/// the largest one-dimensional transform. [`RealTransformNdPlan::execute_into`]
/// takes strided input and output views, so plans are the form for strided,
/// broadcast and padded tensors; [`dctn`], [`dstn`] and their inverses take
/// dense row-major slices.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
/// use scioxide_fft::{DctType, FftDirection, Normalization};
/// use scioxide_fft::{RealTransformNdPlan, RealTransformNdPlanSpec};
/// let (kinds, norm) = ([DctType::II.into(); 2], Normalization::Orthonormal);
/// let spec = RealTransformNdPlanSpec::new(&[2, 3], &[0, 1], &kinds, FftDirection::Forward, norm)?;
/// let plan = RealTransformNdPlan::<f64>::new(spec)?;
/// let layout = StridedLayout::new(&[2, 3], &[3, 1])?;
/// let input = NdView::new(&[1.0; 6], layout)?;
/// let mut output = [0.0; 6];
/// let cx = ExecutionContext::sequential();
/// let output_view = NdViewMut::new(&mut output, layout)?;
/// plan.execute_into(input, output_view, &mut plan.workspace(())?, cx)?;
/// // Orthonormal DCT-II of ones on each axis: y[0][0] = √2 · √3, other bins 0.
/// assert!((output[0] - 6.0_f64.sqrt()).abs() < 1e-12);
/// assert!(output[1..].iter().all(|x| x.abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RealTransformNdPlan<S: RealScalar> {
    spec: RealTransformNdPlanSpec,
    strides: Vec<usize>,
    batches: BatchGeometry,
    children: Vec<Arc<RealTransformPlan<S>>>,
    axes: Vec<LaneGeometry>,
    len: usize,
    staging_len: usize,
    scratch: crate::RealTransformNdWorkspaceRequirements,
    headroom: i32,
    bounded: bool,
    output_bound: S,
}
impl<S: RealScalar> RealTransformNdPlan<S> {
    /// Returns the bytes this plan occupies, counting a one-dimensional plan
    /// shared by several axes once.
    pub fn plan_bytes(&self) -> usize {
        // Preparation checked the sum; cloning cannot increase any capacity.
        self.byte_counts().sum()
    }

    /// Returns the spec the plan was prepared from.
    pub const fn spec(&self) -> &RealTransformNdPlanSpec {
        &self.spec
    }

    /// Prepares a plan for the processor of
    /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of types differs from the number of
    ///   axes.
    /// - `OutOfBounds` if an axis is not less than the rank.
    /// - `InvalidInput` if an axis appears twice, a selected extent is zero, a
    ///   DCT-I axis has length one, or a required backend is unavailable.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the tables or
    ///   workspace size cannot be represented or reserved.
    pub fn new(spec: RealTransformNdPlanSpec) -> Result<Self, SciOxideError> {
        Self::build(spec, scioxide_core::ExecutionContext::sequential_native())
    }

    /// Prepares a plan with the processor features and provider of `context`.
    ///
    /// See [Backends and providers](crate#backends-and-providers).
    ///
    /// # Errors
    ///
    /// - The errors of [`RealTransformNdPlan::new`].
    /// - `ExecutionStopped` if the context is already cancelled or past its
    ///   deadline.
    /// - The errors of the provider, if it fails to prepare a kernel.
    pub fn build<'a>(
        spec: RealTransformNdPlanSpec,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
    ) -> Result<Self, SciOxideError> {
        let shape = &spec.shape[..];
        let axes = &spec.axes[..];
        let types = &spec.types[..];
        let backend = spec.backend;

        let shape_argument = scioxide_core::argument!("shape");
        let context = context.into();
        context.execution().check()?;
        let _ = context.resolve(backend)?;
        crate::validation::length(types.len(), axes.len())?;
        let len = validate_axes(
            shape,
            axes,
            shape_argument,
            scioxide_core::argument!("axes"),
        )?;
        // Validate type-specific lengths even for empty batches.
        for (&axis, &ty) in axes.iter().zip(types) {
            RealTransformPlan::<S>::validate_nd_length(shape[axis], ty, shape_argument)?;
        }
        let mut children: Vec<Arc<RealTransformPlan<S>>> =
            try_with_capacity(axes.len(), scioxide_core::quantity!("n_d_real_child_plans"))?;
        let dense_strides = strides(shape)?;
        let batches = BatchGeometry::new(shape, axes)?;
        let mut geometry = try_with_capacity(axes.len(), scioxide_core::quantity!("n_d_metadata"))?;
        // With one selected axis every lane is its own batch, so lanes span
        // the whole tensor; otherwise they span one batch.
        let lane_shape = if axes.len() == 1 {
            shape
        } else {
            &batches.shape[..]
        };
        let mut staging_len = 0;
        let mut complex_scratch = 0;
        if len != 0 {
            for (i, &axis) in axes.iter().enumerate() {
                // Direction, normalization, policy and capabilities are fixed
                // for this tensor; length and kind distinguish its children.
                let child = match children
                    .iter()
                    .find(|p| p.spec().length() == shape[axis] && p.spec().kind() == spec.types[i])
                {
                    Some(child) => Arc::clone(child),
                    None => {
                        let child = RealTransformPlan::with_backend_context(
                            shape[axis],
                            spec.types[i],
                            spec.direction,
                            spec.normalization,
                            spec.policy,
                            spec.backend,
                            context,
                            scioxide_core::argument!("shape"),
                        )?;
                        checked_add(
                            child.plan_bytes(),
                            2 * size_of::<usize>(),
                            scioxide_core::quantity!("fft_plan_bytes"),
                        )?;
                        Arc::new(child)
                    }
                };
                complex_scratch = complex_scratch.max(child.lane_scratch_len());
                let lanes = LaneGeometry::new::<S>(lane_shape, &dense_strides, axis);
                staging_len = staging_len
                    .max(lanes.staging_len())
                    .max(child.lane_staging_len());
                geometry.push(lanes);
                children.push(child);
            }
        }
        // Real scratch layout: [candidate tensor (len) | strided-lane staging].
        // The children run sequentially, so one complex scratch of the largest
        // child requirement is shared by all of them.
        let real_scratch = checked_add(
            len,
            staging_len,
            scioxide_core::quantity!("n_d_real_scratch"),
        )?;
        checked_mul(
            real_scratch,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("n_d_real_scratch_bytes"),
        )?;
        let requirements =
            crate::RealTransformNdWorkspaceRequirements::new::<S>(real_scratch, complex_scratch)?;
        let scratch = Self::prepare_requirements(requirements, &children)?;
        let headroom: i32 = axes
            .iter()
            .map(|&axis| {
                if shape[axis] == 1 {
                    0
                } else {
                    growth_exponent(shape[axis]) + 4
                }
            })
            .sum();
        let bounded = S::ONE.scalbn(headroom).is_finite();
        let headroom = if bounded { headroom } else { 0 };
        // Each child's bound includes its own normalization and the fixed
        // factor-four restoration of a bounded lane. Overflow of this
        // conservative product simply selects checked batch restoration.
        let output_bound = children
            .iter()
            .fold(S::ONE, |bound, child| bound * child.bounded_output_bound());
        let plan = Self {
            strides: dense_strides,
            batches,
            spec,
            children,
            axes: geometry,
            len,
            staging_len,
            scratch,
            headroom,
            bounded,
            output_bound,
        };
        plan.byte_counts().try_fold(0, |a, b| {
            checked_add(a, b, scioxide_core::quantity!("fft_plan_bytes"))
        })?;
        Ok(plan)
    }
    /// Returns the backend of each complex FFT the plan runs, in the order of
    /// the selected axes. Items can repeat.
    pub fn resolved_backends(&self) -> impl Iterator<Item = crate::FftBackend> + '_ {
        self.children.iter().flat_map(|p| p.resolved_backends())
    }
    /// Returns the real and complex scratch lengths that execution needs.
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<crate::RealTransformNdWorkspaceRequirements, SciOxideError> {
        Ok(self.scratch)
    }

    fn prepare_requirements(
        requirements: crate::RealTransformNdWorkspaceRequirements,
        children: &[Arc<RealTransformPlan<S>>],
    ) -> Result<crate::RealTransformNdWorkspaceRequirements, SciOxideError> {
        let limit = children.iter().try_fold(usize::MAX, |limit, child| {
            let child = child.lane_workspace_requirements()?;
            Ok::<_, SciOxideError>(crate::workspace::nested_byte_limit(
                limit,
                requirements.bytes(),
                [(child.byte_limit(), child.bytes())],
            ))
        })?;
        requirements.with_limit(limit)
    }
    /// Allocates a workspace of the size reported by
    /// [`RealTransformNdPlan::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the size exceeds a provider's workspace limit.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(
        &self,
        request: (),
    ) -> Result<crate::RealTransformNdWorkspace<S>, SciOxideError> {
        let mut workspace = crate::RealTransformNdWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Transforms `input` and writes the result to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the shape of `input` or `output` differs from
    ///   the plan shape, or the workspace is too small.
    /// - The shared execution errors listed under [Errors](crate#errors).
    pub fn execute_into(
        &self,
        input: NdView<'_, '_, S>,
        mut output: NdViewMut<'_, '_, S>,
        workspace: &mut crate::RealTransformNdWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let _ = workspace.check(&self.scratch)?;
        let crate::RealTransformNdWorkspace { real, complex } = workspace;
        self.validate(input.layout().shape().extents(), real, complex)?;
        check_shape(output.layout().shape().extents(), &self.spec.shape)?;
        let pending = self.transform(input, real, complex, context)?;
        self.publish(&real[..self.len], &mut output, pending);
        Ok(())
    }
    /// Transforms `data` in place.
    ///
    /// The result is computed in the workspace and copied to `data` on success.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the shape of `data` differs from the plan shape,
    ///   or the workspace is too small.
    /// - The shared execution errors listed under [Errors](crate#errors).
    pub fn execute_in_place(
        &self,
        mut data: NdViewMut<'_, '_, S>,
        workspace: &mut crate::RealTransformNdWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        // Work on a copy in `real` and write back only after the
        // whole transform succeeded, so failures leave `data` intact.
        let _ = workspace.check(&self.scratch)?;
        let crate::RealTransformNdWorkspace { real, complex } = workspace;
        self.validate(data.layout().shape().extents(), real, complex)?;
        let pending = self.transform(data.as_view(), real, complex, context)?;
        self.publish(&real[..self.len], &mut data, pending);
        Ok(())
    }
}

/// Computes the DCT of a real tensor along the selected axes, with one type
/// per axis.
///
/// `input` holds a tensor of shape `shape` in row-major order (last axis
/// fastest). `types[i]` is the DCT type on `axes[i]`; the definitions are
/// listed on [`RealTransformPlan`](crate::RealTransformPlan). Unselected axes
/// are independent batches. The result has the same shape and order. Use
/// [`RealTransformNdPlan`] for strided tensors or to transform many tensors of
/// the same shape.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`, or the
///   number of types differs from the number of axes.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` if an axis appears twice, a selected extent is zero, or a
///   type-I DCT axis has length one.
/// - The shared execution errors listed under [Errors](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_fft::{DctType, Normalization, dctn};
/// let types = [DctType::II, DctType::II];
/// let coefficients = dctn(&[1.0_f64; 6], &[2, 3], &[0, 1], &types, Normalization::Inverse)?;
/// // Under inverse normalization the forward DCT-II is unscaled: y[0][0] = Σ x = 6.
/// assert!((coefficients[0] - 6.0).abs() < 1e-12);
/// assert!(coefficients[1..].iter().all(|x| x.abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn dctn<S: RealScalar>(
    input: &[S],
    shape: &[usize],
    axes: &[usize],
    types: &[DctType],
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    real_transform(
        input,
        shape,
        axes,
        types.iter().copied().map(RealTransformKind::Dct),
        normalization,
        FftDirection::Forward,
    )
}

/// Computes the inverse DCT of a real tensor along the selected axes, with
/// one type per axis.
///
/// `types[i]` is the type of the forward DCT on `axes[i]` that this call
/// inverts; with the same normalization, [`dctn`] followed by `idctn` returns
/// the input. `input` and the result are tensors of shape `shape` in row-major
/// order, and unselected axes are independent batches.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`, or the
///   number of types differs from the number of axes.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` if an axis appears twice, a selected extent is zero, or a
///   type-I DCT axis has length one.
/// - The shared execution errors listed under [Errors](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_fft::{DctType, Normalization, dctn, idctn};
/// let (input, shape) = ([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
/// let (types, norm) = ([DctType::II, DctType::IV], Normalization::Orthonormal);
/// let coefficients = dctn(&input, &shape, &[0, 1], &types, norm)?;
/// let back = idctn(&coefficients, &shape, &[0, 1], &types, norm)?;
/// // A forward and inverse pair with the same normalization recovers the input.
/// for (a, b) in back.iter().zip(input) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn idctn<S: RealScalar>(
    input: &[S],
    shape: &[usize],
    axes: &[usize],
    types: &[DctType],
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    real_transform(
        input,
        shape,
        axes,
        types.iter().copied().map(RealTransformKind::Dct),
        normalization,
        FftDirection::Inverse,
    )
}

/// Computes the DST of a real tensor along the selected axes, with one type
/// per axis.
///
/// `input` holds a tensor of shape `shape` in row-major order (last axis
/// fastest). `types[i]` is the DST type on `axes[i]`; the definitions are
/// listed on [`RealTransformPlan`](crate::RealTransformPlan). Unselected axes
/// are independent batches. The result has the same shape and order. Use
/// [`RealTransformNdPlan`] for strided tensors or to transform many tensors of
/// the same shape.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`, or the
///   number of types differs from the number of axes.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` if an axis appears twice or a selected extent is zero.
/// - The shared execution errors listed under [Errors](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_fft::{DstType, Normalization, dstn};
/// let types = [DstType::II, DstType::II];
/// let coefficients = dstn(&[1.0_f64; 4], &[2, 2], &[0, 1], &types, Normalization::Inverse)?;
/// // Unscaled DST-II of (1, 1) is (√2, 0) on each axis, so y[0][0] = 2.
/// assert!((coefficients[0] - 2.0).abs() < 1e-12);
/// assert!(coefficients[1..].iter().all(|x| x.abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn dstn<S: RealScalar>(
    input: &[S],
    shape: &[usize],
    axes: &[usize],
    types: &[DstType],
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    real_transform(
        input,
        shape,
        axes,
        types.iter().copied().map(RealTransformKind::Dst),
        normalization,
        FftDirection::Forward,
    )
}

/// Computes the inverse DST of a real tensor along the selected axes, with
/// one type per axis.
///
/// `types[i]` is the type of the forward DST on `axes[i]` that this call
/// inverts; with the same normalization, [`dstn`] followed by `idstn` returns
/// the input. `input` and the result are tensors of shape `shape` in row-major
/// order, and unselected axes are independent batches.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`, or the
///   number of types differs from the number of axes.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` if an axis appears twice or a selected extent is zero.
/// - The shared execution errors listed under [Errors](crate#errors).
///
/// # Examples
///
/// ```
/// use scioxide_fft::{DstType, Normalization, dstn, idstn};
/// let (input, shape) = ([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
/// let (types, norm) = ([DstType::II, DstType::IV], Normalization::Orthonormal);
/// let coefficients = dstn(&input, &shape, &[0, 1], &types, norm)?;
/// let back = idstn(&coefficients, &shape, &[0, 1], &types, norm)?;
/// // A forward and inverse pair with the same normalization recovers the input.
/// for (a, b) in back.iter().zip(input) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn idstn<S: RealScalar>(
    input: &[S],
    shape: &[usize],
    axes: &[usize],
    types: &[DstType],
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    real_transform(
        input,
        shape,
        axes,
        types.iter().copied().map(RealTransformKind::Dst),
        normalization,
        FftDirection::Inverse,
    )
}

fn real_transform<S: RealScalar>(
    input: &[S],
    shape: &[usize],
    axes: &[usize],
    types: impl ExactSizeIterator<Item = RealTransformKind>,
    normalization: Normalization,
    direction: FftDirection,
) -> Result<Vec<S>, SciOxideError> {
    crate::validation::length(input.len(), scioxide_core::Shape::new(shape)?.len())?;
    let plan = RealTransformNdPlan::new(RealTransformNdPlanSpec::from_types(
        shape,
        axes,
        types,
        direction,
        normalization,
    )?)?;
    let mut output = try_filled(
        plan.len,
        S::ZERO,
        scioxide_core::quantity!("n_d_real_output"),
    )?;
    let layout = StridedLayout::new(&plan.spec.shape, &plan.strides)?;
    plan.execute_into(
        NdView::new(input, layout)?,
        NdViewMut::new(&mut output, layout)?,
        &mut plan.workspace(())?,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}

/// Trigonometric storage and child passes for the common tensor driver.
struct TrigBatches<'a, 'w, 'd, 'l, 'c, S: RealScalar> {
    plan: &'a RealTransformNdPlan<S>,
    source: NdView<'d, 'l, S>,
    candidate: &'a mut [S],
    staging: &'a mut [S],
    complex: &'w mut [Complex<S>],
    context: ExecutionContext<'c>,
}
impl<S: RealScalar> crate::tensor::BatchTransform<S> for TrigBatches<'_, '_, '_, '_, '_, S> {
    fn single_axis(&mut self) -> Result<(), SciOxideError> {
        let plan = self.plan;
        let source = &self.source;
        let candidate = &mut *self.candidate;
        let staging = &mut *self.staging;
        let complex = &mut *self.complex;
        let context = self.context;
        if let ([geometry], [child]) = (&plan.axes[..], &plan.children[..]) {
            // One selected axis: every lane is an independent batch, executed
            // by the child with exactly its one-dimensional arithmetic.
            let placement = Placement {
                base: 0,
                geometry: *geometry,
                shape: &plan.spec.shape,
                strides: &plan.strides,
            };
            if child.native_lanes() {
                let layout = source.layout();
                let strides = layout.strides();
                let source_placement = Placement::new(
                    LaneGeometry::new::<S>(&plan.spec.shape, strides, plan.spec.axes[0]),
                    &plan.spec.shape,
                    strides,
                );
                return child.transform_source_lanes(
                    crate::lanes::LanePairs::new(
                        (source.storage(), source_placement),
                        (candidate, placement),
                        staging,
                    ),
                    complex,
                    context,
                );
            }
            copy_in(source, &plan.strides, candidate, |value| value);
            child.transform_lanes(
                Lanes::new(candidate, placement, staging),
                complex,
                LaneMode::Checked,
                context,
            )
        } else {
            Ok(())
        }
    }
    fn prepare(&mut self, batch: usize) -> Result<S, SciOxideError> {
        let geometry = &self.plan.batches;
        let layout = self.source.layout();
        let strides = layout.strides();
        crate::tensor::BatchInput::Real(&self.source.storage()[geometry.base(batch, strides)..])
            .maximum(geometry, strides)
    }
    fn transform(
        &mut self,
        batch: usize,
        _maximum: S,
        scaling: BinaryScale<S>,
        mode: LaneMode,
    ) -> Result<(), SciOxideError> {
        let plan = self.plan;
        let base = plan.batches.base(batch, &plan.strides);
        let layout = self.source.layout();
        let strides = layout.strides();
        let input = &self.source.storage()[plan.batches.base(batch, strides)..];
        let values = &mut self.candidate[base..];
        let (source, target) = ((input, strides), (&mut *values, &plan.strides[..]));
        match scaling {
            BinaryScale::Identity => plan.batches.copy(source, target, |value| value),
            BinaryScale::Multiply(factor) => {
                plan.batches.copy(source, target, |value| value * factor)
            }
            BinaryScale::Extended(exponent) => {
                plan.batches
                    .copy(source, target, |value| value.scalbn(exponent));
            }
        }
        for (&geometry, child) in plan.axes.iter().zip(&plan.children) {
            let placement = Placement {
                base: 0,
                geometry,
                shape: &plan.batches.shape,
                strides: &plan.strides,
            };
            child.transform_lanes(
                Lanes::new(values, placement, &mut *self.staging),
                self.complex,
                mode,
                self.context,
            )?;
        }
        Ok(())
    }
    fn output(
        &mut self,
        batch: usize,
    ) -> (crate::tensor::BatchOutput<'_, S>, &BatchGeometry, &[usize]) {
        let geometry = &self.plan.batches;
        let strides = &self.plan.strides;
        let base = geometry.base(batch, strides);
        (
            crate::tensor::BatchOutput::Real(&mut self.candidate[base..]),
            geometry,
            strides,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deferred_restoration<S: RealScalar>() {
        let shape = [4, 8];
        let context = ExecutionContext::sequential_native();
        for backend in crate::test_support::fft_backends(context.processor()) {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                for normalization in [
                    Normalization::Unscaled,
                    Normalization::Forward,
                    Normalization::Inverse,
                    Normalization::Orthonormal,
                ] {
                    let plan = RealTransformNdPlan::<S>::new(
                        RealTransformNdPlanSpec::new(
                            &shape,
                            &[0, 1],
                            &[DctType::II.into(), DstType::IV.into()],
                            direction,
                            normalization,
                        )
                        .unwrap()
                        .with_backend_strategy(FftBackendStrategy::Required(backend)),
                    )
                    .unwrap();
                    let mut checked = plan.clone();
                    checked.output_bound = S::INFINITY;
                    for exponent in [
                        None,
                        Some(0),
                        Some(crate::scaling::range_exponent(S::MIN_SUBNORMAL) + 4),
                        Some(crate::scaling::range_exponent(S::MAX) - 9),
                        Some(crate::scaling::range_exponent(S::MAX) - 1),
                    ] {
                        for strides in [[8, 1], [19, 2]] {
                            let layout = StridedLayout::new(&shape, &strides).unwrap();
                            let sentinel = S::try_from_f64_lossy(123.0).unwrap();
                            let mut input = vec![sentinel; 72];
                            for i in 0..32 {
                                input[i / 8 * strides[0] + i % 8 * strides[1]] =
                                    exponent.map_or(S::NEG_ZERO, |e| {
                                        S::try_from_f64_lossy(((i * 7 % 17) as f64 - 8.0) / 16.0)
                                            .unwrap()
                                            .scalbn(e)
                                    });
                            }
                            for in_place in [false, true] {
                                let mut actual = if in_place {
                                    input.clone()
                                } else {
                                    vec![sentinel; 72]
                                };
                                let before = actual.clone();
                                let mut expected = actual.clone();
                                let execute = |plan: &RealTransformNdPlan<S>, output: &mut [S]| {
                                    let mut workspace = plan.workspace(()).unwrap();
                                    let output = NdViewMut::new(output, layout).unwrap();
                                    if in_place {
                                        plan.execute_in_place(output, &mut workspace, context)
                                    } else {
                                        plan.execute_into(
                                            NdView::new(&input, layout).unwrap(),
                                            output,
                                            &mut workspace,
                                            context,
                                        )
                                    }
                                };
                                let result = execute(&plan, &mut actual);
                                let reference = execute(&checked, &mut expected);
                                let failed = result.is_err();
                                assert_eq!(
                                    result.err().map(|e| e.kind()),
                                    reference.err().map(|e| e.kind())
                                );
                                for (&a, &b) in
                                    actual.iter().zip(if failed { &before } else { &expected })
                                {
                                    assert_eq!(
                                        a.try_to_f64_exact().unwrap().to_bits(),
                                        b.try_to_f64_exact().unwrap().to_bits()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn deferred_real_restoration_preserves_checked_bits_and_transactions() {
        deferred_restoration::<f32>();
        deferred_restoration::<f64>();
    }

    #[test]
    fn repeated_axes_share_only_matching_trigonometric_children() -> Result<(), SciOxideError> {
        for n in [8, 23] {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                let plan = RealTransformNdPlan::<f64>::new(RealTransformNdPlanSpec::new(
                    &[n, n, n, n + 1],
                    &[0, 1, 2, 3],
                    &[
                        DctType::II.into(),
                        DctType::II.into(),
                        DstType::II.into(),
                        DctType::II.into(),
                    ],
                    direction,
                    Normalization::Orthonormal,
                )?)?;
                assert!(Arc::ptr_eq(&plan.children[0], &plan.children[1]));
                assert!(!Arc::ptr_eq(&plan.children[0], &plan.children[2]));
                assert!(!Arc::ptr_eq(&plan.children[0], &plan.children[3]));
                let cloned = plan.clone();
                assert!(Arc::ptr_eq(&cloned.children[0], &plan.children[0]));
            }
        }
        Ok(())
    }

    #[test]
    fn cached_requirements_check_both_buffers_before_publication() -> Result<(), SciOxideError> {
        let shape = [2, 8];
        let layout = StridedLayout::new(&shape, &[8, 1])?;
        let plan = RealTransformNdPlan::<f64>::new(RealTransformNdPlanSpec::new(
            &shape,
            &[0, 1],
            &[DctType::II.into(), DstType::III.into()],
            FftDirection::Forward,
            Normalization::Orthonormal,
        )?)?;
        let requirements = plan.workspace_requirements(())?;
        let input: Vec<_> = (0..16).map(|i| f64::from(i) / 16.0).collect();
        let sentinel = [123.0; 16];
        let context = ExecutionContext::sequential();
        for short_real in [true, false] {
            let mut workspace = plan.workspace(())?;
            if short_real {
                workspace.real.truncate(requirements.real_len() - 1);
            } else {
                workspace
                    .complex
                    .truncate(requirements.complex().complex_len() - 1);
            }
            let mut output = sentinel;
            let result = plan.execute_into(
                NdView::new(&input, layout)?,
                NdViewMut::new(&mut output, layout)?,
                &mut workspace,
                context,
            );
            assert!(
                matches!(result, Err(ref e) if e.kind() == scioxide_core::SciOxideErrorKind::DimensionMismatch)
            );
            assert_eq!(output, sentinel);
            let mut inplace = sentinel;
            let result = plan.execute_in_place(
                NdViewMut::new(&mut inplace, layout)?,
                &mut workspace,
                context,
            );
            assert!(
                matches!(result, Err(ref e) if e.kind() == scioxide_core::SciOxideErrorKind::DimensionMismatch)
            );
            assert_eq!(inplace, sentinel);
            workspace.try_reserve(&requirements)?;
            plan.execute_into(
                NdView::new(&input, layout)?,
                NdViewMut::new(&mut output, layout)?,
                &mut workspace,
                context,
            )?;
            let mut inplace = input.clone();
            plan.execute_in_place(
                NdViewMut::new(&mut inplace, layout)?,
                &mut workspace,
                context,
            )?;
            assert_eq!(inplace, output);
        }
        Ok(())
    }
}
