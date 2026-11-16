//! Trigonometric plan preparation.
use super::*;

impl<S: scioxide_core::RealScalar> RealTransformPlan<S> {
    /// Build a tensor-axis child; a rejected extent
    /// is attributed to its `length` argument.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn with_backend_context<'a>(
        n: usize,
        kind: RealTransformKind,
        direction: FftDirection,
        normalization: Normalization,
        policy: ExecutionPolicy,
        backend: crate::FftBackendStrategy,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
        length: Argument,
    ) -> Result<Self, SciOxideError> {
        let context = context.into();
        validate_length(n, kind.kernel(), length)?;
        let ty = kind.kernel();
        let spec = RealTransformPlanSpec::new(n, kind, direction, normalization)
            .with_execution_policy(policy)
            .with_backend_strategy(backend);
        let backend = context.resolve(backend)?;
        let kernel = if ty == KernelType::Four && n.is_multiple_of(2) {
            Kernel::EvenIv(crate::complex::ComplexCore::with_backend_and_context(
                n / 2,
                policy,
                backend,
                context,
                length,
                scioxide_core::argument!("context"),
            )?)
        } else {
            let real = RealFftPlan::with_backend_and_context(
                real_len(n, ty)?,
                policy,
                backend,
                context,
                length,
                scioxide_core::argument!("context"),
            )?;
            if n >= 4 && n.is_multiple_of(2) && matches!(ty, KernelType::Two | KernelType::Three) {
                Kernel::EvenHalf(real)
            } else {
                Kernel::Real(real)
            }
        };
        let phases = phases::<S>(n, ty, length)?;
        // The inverse transform is executed as the raw transform of the dual
        // type (II <-> III; I and IV are self-dual), so each direction gets
        // its own scaling: the round-trip factor belongs to the configured
        // pair, while orthonormal endpoint weights follow the raw type run.
        let inverse = direction == FftDirection::Inverse;
        let raw_type = if inverse { dual_type(ty) } else { ty };
        let scaling = scaling::<S>(n, ty, raw_type, normalization, inverse, length)?;
        let short = ShortTransform::<S>::new(n, raw_type, kind, &phases, scaling);
        // The n outputs are candidates when overflow cannot be excluded in
        // advance. Real children stage
        // their own packed input; IV's complex child needs n/2 packed values.
        let work = if short.is_some() {
            0
        } else {
            match &kernel {
                Kernel::Real(plan) | Kernel::EvenHalf(plan) => plan.scratch_len(),
                Kernel::EvenIv(plan) => checked_add(
                    n / 2,
                    plan.bounded_scratch_len(),
                    scioxide_core::quantity!("real_transform_packed_scratch"),
                )?,
            }
        };
        let scratch_len = checked_add(n, work, scioxide_core::quantity!("real_transform_scratch"))?;
        let _ = crate::FftWorkspaceRequirements::new::<S>(scratch_len)?;
        // Range scaling bounds staged samples by 2^-2, or sqrt(2) * 2^-2 with
        // endpoint weights, so every raw output is below 3n; 4n leaves margin.
        let provider = match &kernel {
            Kernel::Real(p) | Kernel::EvenHalf(p) => p
                .resolved_backends()
                .any(|b| b == crate::FftBackend::Provider),
            Kernel::EvenIv(p) => p.resolved_backend() == crate::FftBackend::Provider,
        };
        let output_bound = if provider {
            S::INFINITY
        } else {
            scalar_from_usize_rounded::<S>(checked_mul(
                4,
                n,
                scioxide_core::quantity!("real_transform_output_bound"),
            )?)? * [
                scaling.output_scale,
                scaling.output_first,
                scaling.output_last,
            ]
            .into_iter()
            .fold(S::ZERO, |a, b| a.max(b.abs()))
        };
        // Native even II/III/IV kernels whose half-length rows fit the stack
        // block run N-D lanes in blocks.
        let child = match (&kernel, short, provider) {
            (Kernel::EvenHalf(real), None, false) => real.even_parts().map(|(child, _)| child),
            (Kernel::EvenIv(child), None, false) => Some(child),
            _ => None,
        }
        .filter(|_| n / 2 <= ROWS);
        let blocks = child
            .map(|child| {
                Ok::<_, SciOxideError>(LaneBlocks {
                    scratch_len: checked_add(
                        n / 2,
                        child.bounded_scratch_len(),
                        scioxide_core::quantity!("real_transform_lane_scratch"),
                    )?
                    .max(
                        if child
                            .smooth_stages()
                            .is_some_and(|stages| !stages.native_rows_supported())
                        {
                            WIDTH * n
                        } else {
                            0
                        },
                    ),
                    dispatch: crate::backend::dispatch(backend, context.execution().processor())?,
                })
            })
            .transpose()?;
        let digit_rows = child
            .and_then(crate::complex::ComplexCore::smooth_stages)
            .map_or(Ok(Vec::new()), crate::kernel::StagePlan::digit_rows)?;
        let mut plan = Self {
            plan_bytes: 0,
            spec,
            kernel,
            phases,
            scaling,
            short,
            scratch_len,
            blocks,
            digit_rows,
            output_bound,
        };
        plan.plan_bytes = plan.retained_bytes()?;
        Ok(plan)
    }
}
