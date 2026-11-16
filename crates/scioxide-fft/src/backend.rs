//! Planning contexts and the interface for injected FFT providers.
use crate::{FftBackendStrategy, FftDirection, FftPlanDiagnostics};
use alloc::sync::Arc;
use scioxide_core::backend::{KernelBackend, KernelDispatch};
use scioxide_core::{
    Argument, Complex, ExecutionContext, ExecutionPolicy, ProcessorCapabilities, RealScalar,
    SciOxideError,
};

/// Selects what happens when an injected [`FftProvider`] declines a length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftProviderPolicy {
    /// Uses SciOxide's CPU kernels for lengths the provider declines.
    Preferred,
    /// Fails preparation with `InvalidInput` (`UnsupportedConfiguration`) for
    /// lengths the provider declines.
    Required,
}

/// Identifies the implementation that runs a one-dimensional complex FFT.
///
/// Plans report one value per complex FFT they contain through their
/// `resolved_backends` method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftBackend {
    /// SciOxide's CPU kernels for the given instruction set.
    Cpu(KernelBackend),
    /// An injected [`FftProvider`].
    Provider,
}

/// Sets byte limits for an injected provider.
///
/// Both limits default to `usize::MAX`. A provider must keep its peak memory
/// during preparation within `preparation_bytes` before it allocates. After
/// preparation, SciOxide checks the sizes the provider reports and returns
/// `ResourceLimit` if either limit is exceeded.
#[derive(Debug, Clone, Copy)]
pub struct FftProviderLimits {
    /// Maximum bytes live at once while the provider prepares a plan.
    pub preparation_bytes: usize,
    /// Maximum bytes of complex scratch that the prepared plan may require.
    pub workspace_bytes: usize,
}
impl Default for FftProviderLimits {
    fn default() -> Self {
        Self {
            preparation_bytes: usize::MAX,
            workspace_bytes: usize::MAX,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ProviderKernel<S: RealScalar> {
    pub plan: Arc<dyn FftProviderPlan<S>>,
    pub scratch_len: usize,
    pub bytes: usize,
    pub(crate) workspace_limit: usize,
    pub diagnostics: FftPlanDiagnostics,
}

/// Prepares one-dimensional complex FFTs from an external implementation.
///
/// Pass a provider to plan preparation through
/// [`FftPlanningContext::with_provider`]. The complex FFTs that a plan
/// contains, including those inside real, N-D, chirp-z and nonuniform
/// transforms, are then offered to the provider first. [`Self::prepare`] returns `Ok(None)` to
/// decline a length; [`FftProviderPolicy`] decides whether SciOxide's CPU kernels
/// take over. An error from `prepare` is returned to the caller and never
/// falls back to the CPU kernels.
///
/// The prepared [`FftProviderPlan`] must own all of its state and must not keep
/// the context it was prepared with. SciOxide applies normalization and
/// rescaling around each provider call, so the provider computes the unscaled
/// transform. The accuracy of a provider-backed plan is the provider's; the
/// bounds in [Accuracy](crate#accuracy) apply to SciOxide's own kernels.
///
/// # Examples
///
/// An adapter that serves one length from state prepared in advance. Here a
/// pair of unscaled SciOxide plans stands in for an external library. SciOxide
/// applies the plan's normalization, so the orthonormal DC bin of eight ones
/// is `√8`. A declined length falls back to the CPU kernels under
/// [`FftProviderPolicy::Preferred`] and is rejected under
/// [`FftProviderPolicy::Required`].
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext, ExecutionPolicy, SciOxideError, SciOxideErrorKind};
/// use scioxide_fft::{
///     FftBackend, FftDirection, FftLayout, FftPlan, FftPlanDiagnostics, FftPlanSpec,
///     FftPlanningContext, FftProvider, FftProviderLimits, FftProviderPlan, FftProviderPolicy,
///     Normalization,
/// };
/// use std::sync::Arc;
///
/// #[derive(Debug)]
/// struct Pair {
///     forward: FftPlan<f64>,
///     inverse: FftPlan<f64>,
///     scratch: usize,
/// }
///
/// impl FftProviderPlan<f64> for Pair {
///     fn workspace_len(&self) -> usize {
///         self.scratch
///     }
///     fn plan_bytes(&self) -> usize {
///         self.forward.plan_bytes() + self.inverse.plan_bytes()
///     }
///     fn preparation_bytes(&self) -> usize {
///         self.plan_bytes()
///     }
///     fn diagnostics(&self) -> FftPlanDiagnostics {
///         FftPlanDiagnostics::for_provider(self.forward.diagnostics().stage_count())
///     }
///     fn execute(
///         &self,
///         direction: FftDirection,
///         data: &mut [Complex<f64>],
///         scratch: &mut [Complex<f64>],
///         context: ExecutionContext<'_>,
///     ) -> Result<(), SciOxideError> {
///         let plan = match direction {
///             FftDirection::Forward => &self.forward,
///             FftDirection::Inverse => &self.inverse,
///         };
///         plan.execute_in_place(data, scratch, context)
///     }
/// }
///
/// struct Adapter(Arc<Pair>);
///
/// impl FftProvider<f64> for Adapter {
///     fn prepare(
///         &self,
///         length: usize,
///         _: ExecutionPolicy,
///         _: FftProviderLimits,
///         _: ExecutionContext<'_>,
///     ) -> Result<Option<Arc<dyn FftProviderPlan<f64>>>, SciOxideError> {
///         // Sharing prepared state needs no new storage. SciOxide checks the
///         // reported sizes against its limits.
///         Ok((length == self.0.forward.spec().length())
///             .then(|| Arc::clone(&self.0) as Arc<dyn FftProviderPlan<f64>>))
///     }
/// }
///
/// let unscaled = |direction| {
///     FftPlan::new(FftPlanSpec::new(
///         8,
///         direction,
///         Normalization::Unscaled,
///         FftLayout::Complex,
///     ))
/// };
/// let forward = unscaled(FftDirection::Forward)?;
/// let scratch = forward.workspace_requirements(())?.complex_len();
/// let inverse = unscaled(FftDirection::Inverse)?;
/// let adapter = Adapter(Arc::new(Pair {
///     forward,
///     inverse,
///     scratch,
/// }));
///
/// let cx = ExecutionContext::sequential();
/// let spec = |n| {
///     FftPlanSpec::new(
///         n,
///         FftDirection::Forward,
///         Normalization::Orthonormal,
///         FftLayout::Complex,
///     )
/// };
/// let planning = |policy| FftPlanningContext::with_provider(cx, &adapter, policy);
///
/// let plan = FftPlan::build(spec(8), planning(FftProviderPolicy::Required))?;
/// assert_eq!(plan.resolved_backends().next(), Some(FftBackend::Provider));
/// let mut data = [Complex::new(1.0, 0.0); 8];
/// plan.execute_in_place(&mut data, &mut plan.workspace(())?, cx)?;
/// assert!((data[0] - Complex::new(8.0_f64.sqrt(), 0.0)).norm() < 1e-12);
///
/// let fallback = FftPlan::build(spec(6), planning(FftProviderPolicy::Preferred))?;
/// assert!(matches!(
///     fallback.resolved_backends().next(),
///     Some(FftBackend::Cpu(_))
/// ));
/// let error = FftPlan::build(spec(6), planning(FftProviderPolicy::Required)).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
/// # Ok::<(), SciOxideError>(())
/// ```
pub trait FftProvider<S: RealScalar> {
    /// Prepares unscaled transforms of `length` in both directions, or returns
    /// `Ok(None)` to decline the length.
    ///
    /// `policy` is the execution policy of the plan being prepared. The
    /// implementation must keep its peak memory within `limits` and use only
    /// the capabilities in `context`.
    ///
    /// # Errors
    ///
    /// Return an error for any failure, such as `AllocationFailure`,
    /// `CapabilityUnavailable` or `ExecutionStopped`. SciOxide passes it on
    /// unchanged.
    fn prepare(
        &self,
        length: usize,
        policy: ExecutionPolicy,
        limits: FftProviderLimits,
        context: ExecutionContext<'_>,
    ) -> Result<Option<Arc<dyn FftProviderPlan<S>>>, SciOxideError>;
}

/// Runs a prepared one-dimensional complex FFT for an [`FftProvider`].
///
/// [`Self::execute`] must compute the unscaled DFT of
/// [Conventions](crate#conventions) in place: the exponent sign is negative for
/// [`FftDirection::Forward`] and positive for [`FftDirection::Inverse`].
/// SciOxide scales the input by a power of two to keep the result in range,
/// applies the normalization afterwards, and reports NaN or infinity in the
/// result as `NonFinite` with origin `Callback`.
///
/// The plan is shared between threads and must not allocate or change state
/// during execution. Processor requirements belong in preparation: `execute`
/// must accept any execution context, including
/// [`ExecutionContext::sequential`], and must finish all work before it
/// returns. Scratch contents are arbitrary on entry.
///
/// See the [`FftProvider`] example for an implementation.
pub trait FftProviderPlan<S: RealScalar>: core::fmt::Debug + Send + Sync {
    /// Returns the number of complex scratch elements [`Self::execute`] needs.
    fn workspace_len(&self) -> usize;
    /// Returns the bytes the prepared plan holds, including its metadata and
    /// buffers.
    fn plan_bytes(&self) -> usize;
    /// Returns the peak bytes live during preparation, including the prepared
    /// state.
    ///
    /// The value must be at least [`Self::plan_bytes`]; otherwise preparation
    /// fails with `InvalidInput`. It excludes the caller's inputs and
    /// allocator overhead.
    fn preparation_bytes(&self) -> usize;
    /// Returns the algorithm and stage count to report in
    /// [`FftPlan::diagnostics`](crate::FftPlan::diagnostics); see
    /// [`FftPlanDiagnostics::for_provider`].
    fn diagnostics(&self) -> FftPlanDiagnostics;
    /// Transforms `data` in place in the given direction, using `scratch` of
    /// [`Self::workspace_len`] elements.
    ///
    /// # Errors
    ///
    /// Return an error for any failure, such as `ExecutionStopped` or
    /// `NumericalInstability`. SciOxide passes it on and discards `data`.
    fn execute(
        &self,
        direction: FftDirection,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>;
}

/// Supplies the capabilities used to prepare a plan: an execution context and
/// an optional [`FftProvider`].
///
/// Every `build` constructor in this crate takes an `FftPlanningContext` or
/// anything that converts into one. An [`ExecutionContext`] converts into a
/// context without a provider, whose processor capabilities choose the CPU
/// kernels (see [Backends and providers](crate#backends-and-providers)).
/// [`Default`] uses [`ExecutionContext::sequential_native`]. Plans do not keep
/// the context after preparation.
///
/// With a provider, the plan's [`FftBackendStrategy`] selects the CPU kernels
/// used for lengths the provider declines.
///
/// # Examples
///
/// [`ExecutionContext::sequential`] has portable processor capabilities, so
/// the plan uses the scalar kernels.
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext, backend::KernelBackend};
/// use scioxide_fft::{
///     FftBackend, FftDirection, FftLayout, FftPlan, FftPlanSpec, FftPlanningContext, Normalization,
/// };
/// let planning: FftPlanningContext<'_, f64> = ExecutionContext::sequential().into();
/// let (dir, norm) = (FftDirection::Forward, Normalization::Inverse);
/// let spec = FftPlanSpec::new(16, dir, norm, FftLayout::Complex);
/// let plan = FftPlan::build(spec, planning)?;
/// let backend = plan.resolved_backends().next();
/// assert_eq!(backend, Some(FftBackend::Cpu(KernelBackend::Scalar)));
/// let mut data = [Complex::new(1.0, 0.0); 16];
/// plan.execute_in_place(&mut data, &mut plan.workspace(())?, planning.execution())?;
/// assert!((data[0].re - 16.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct FftPlanningContext<'a, S: RealScalar> {
    execution: ExecutionContext<'a>,
    limits: FftProviderLimits,
    provider: Option<(&'a dyn FftProvider<S>, FftProviderPolicy)>,
}
impl<'a, S: RealScalar> From<ExecutionContext<'a>> for FftPlanningContext<'a, S> {
    fn from(execution: ExecutionContext<'a>) -> Self {
        Self {
            execution,
            provider: None,
            limits: FftProviderLimits::default(),
        }
    }
}
impl<S: RealScalar> Default for FftPlanningContext<'_, S> {
    fn default() -> Self {
        scioxide_core::ExecutionContext::sequential_native().into()
    }
}
impl<'a, S: RealScalar> FftPlanningContext<'a, S> {
    /// Creates a planning context that offers each complex FFT length to
    /// `provider` under `policy`.
    ///
    /// The provider limits start at `usize::MAX`. See the [`FftProvider`]
    /// example. The context borrows the provider, so it cannot outlive it:
    ///
    /// ```compile_fail
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_fft::{FftPlanningContext, FftProvider, FftProviderPolicy};
    /// fn escape(provider: &dyn FftProvider<f64>) -> FftPlanningContext<'static, f64> {
    ///     let policy = FftProviderPolicy::Required;
    ///     FftPlanningContext::with_provider(ExecutionContext::sequential(), provider, policy)
    /// }
    /// ```
    pub fn with_provider(
        execution: ExecutionContext<'a>,
        provider: &'a dyn FftProvider<S>,
        policy: FftProviderPolicy,
    ) -> Self {
        Self {
            execution,
            provider: Some((provider, policy)),
            limits: FftProviderLimits::default(),
        }
    }
    /// Tightens the provider byte limits.
    ///
    /// Each limit becomes the smaller of its current value and the new one, so
    /// repeated calls can only lower them.
    pub fn with_provider_limits(mut self, limits: FftProviderLimits) -> Self {
        self.limits.preparation_bytes = self.limits.preparation_bytes.min(limits.preparation_bytes);
        self.limits.workspace_bytes = self.limits.workspace_bytes.min(limits.workspace_bytes);
        self
    }
    /// Returns the execution context.
    pub const fn execution(self) -> ExecutionContext<'a> {
        self.execution
    }

    pub(crate) const fn has_provider(self) -> bool {
        self.provider.is_some()
    }
    pub(crate) fn resolve(
        self,
        strategy: FftBackendStrategy,
    ) -> Result<KernelBackend, SciOxideError> {
        let processor = self.execution.processor();
        match strategy {
            // FFT's AVX2 leaves have fused gather/stage pairs and specialised
            // final stages. ISA width alone is not an FFT cost model.
            FftBackendStrategy::Automatic => Ok(LEAF_BACKENDS
                .into_iter()
                .find(|backend| backend.is_available(processor))
                .unwrap_or(KernelBackend::Scalar)),
            FftBackendStrategy::Required(backend) => {
                let backend = leaves(backend)?;
                // With a provider, the CPU backend is only the fallback for
                // lengths the provider declines, so it is recorded without
                // probing the processor here; an unavailable CPU backend is
                // reported only if the fallback is actually built. A backend
                // without FFT leaves is rejected either way.
                if self.has_provider() {
                    Ok(backend)
                } else {
                    Ok(dispatch(backend, processor)?.backend())
                }
            }
        }
    }

    /// Ask the borrowed provider, if any, for a prepared kernel of `length`.
    ///
    /// Provider refusals and budget violations are attributed to `provider`, the caller parameter
    /// that carried this planning context.
    pub(crate) fn prepare(
        self,
        length: usize,
        policy: ExecutionPolicy,
        provider: Argument,
    ) -> Result<Option<ProviderKernel<S>>, SciOxideError> {
        self.execution.check()?;
        let Some((implementation, preference)) = self.provider else {
            return Ok(None);
        };
        let plan = implementation.prepare(length, policy, self.limits, self.execution)?;
        // Provider preparation is opaque and may be long-running, so honour a
        // cancellation that arrived during it before accepting its result.
        self.execution.check()?;
        if plan.is_none() && preference == FftProviderPolicy::Required {
            return Err(SciOxideError::invalid_input(
                provider,
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "required provider does not support this FFT length",
            ));
        }
        plan.map(|plan| {
            let scratch_len = plan.workspace_len();
            let bytes = plan.plan_bytes();
            let diagnostics = plan.diagnostics();
            // Providers promise to respect the limits, but the reported sizes
            // are re-checked here so an untrusted provider cannot make a plan
            // exceed the caller's budget unnoticed.
            let scratch_bytes = scioxide_core::backend::checked_mul(
                scratch_len,
                core::mem::size_of::<Complex<S>>(),
                scioxide_core::quantity!("provider_scratch_bytes"),
            )?;
            let peak = plan.preparation_bytes();
            if peak < bytes {
                return Err(SciOxideError::invalid_input(
                    provider,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "provider peak is smaller than retained storage",
                ));
            }
            if peak > self.limits.preparation_bytes {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("provider_preparation_bytes"),
                    peak,
                    self.limits.preparation_bytes,
                ));
            }
            if scratch_bytes > self.limits.workspace_bytes {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("provider_workspace_bytes"),
                    scratch_bytes,
                    self.limits.workspace_bytes,
                ));
            }
            Ok(ProviderKernel {
                plan,
                scratch_len,
                bytes,
                workspace_limit: self.limits.workspace_bytes,
                diagnostics,
            })
        })
        .transpose()
    }
}

/// CPU backends with FFT leaves, in automatic preference order. AVX-512 has
/// none: AVX2's fused and specialised leaves beat a wider generic stage.
const LEAF_BACKENDS: [KernelBackend; 3] = [
    KernelBackend::Avx2Fma,
    KernelBackend::Neon,
    KernelBackend::Scalar,
];

/// Accept only CPU backends that have FFT leaves, independently of the host.
fn leaves(backend: KernelBackend) -> Result<KernelBackend, SciOxideError> {
    if LEAF_BACKENDS.contains(&backend) {
        Ok(backend)
    } else {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("backend"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "FFT has no AVX-512 leaves",
        ))
    }
}

/// Host-verify a CPU backend that has FFT leaves; the only way FFT
/// preparation mints a dispatch token.
pub(crate) fn dispatch(
    backend: KernelBackend,
    processor: &dyn ProcessorCapabilities,
) -> Result<KernelDispatch, SciOxideError> {
    KernelDispatch::for_backend(leaves(backend)?, processor)
}
