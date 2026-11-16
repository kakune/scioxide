//! Reference adapter over explicitly prepared CPU plans.
use scioxide_core::backend::{KernelBackend, checked_add};
use scioxide_core::{Complex, ExecutionContext, ExecutionPolicy, RealScalar, SciOxideError};
use scioxide_fft::{
    FftBackendStrategy, FftDirection, FftLayout, FftPlan, FftPlanDiagnostics, FftPlanSpec,
    FftProvider, FftProviderLimits, FftProviderPlan, Normalization,
};
use std::sync::Arc;

pub struct CpuProvider<S: RealScalar> {
    plan: Arc<CpuPair<S>>,
}
#[derive(Debug)]
struct CpuPair<S: RealScalar> {
    forward: FftPlan<S>,
    inverse: FftPlan<S>,
    bytes: usize,
    scratch: usize,
    diagnostics: FftPlanDiagnostics,
}
impl<S: RealScalar> CpuProvider<S> {
    pub fn new(length: usize, policy: ExecutionPolicy) -> Result<Self, SciOxideError> {
        let request = |direction| {
            FftPlanSpec::new(
                length,
                direction,
                Normalization::Unscaled,
                FftLayout::Complex,
            )
            .with_execution_policy(policy)
            .with_backend_strategy(FftBackendStrategy::Required(KernelBackend::Scalar))
        };
        let forward = FftPlan::build(
            request(FftDirection::Forward),
            scioxide_fft::FftPlanningContext::default(),
        )?;
        let inverse = FftPlan::build(
            request(FftDirection::Inverse),
            scioxide_fft::FftPlanningContext::default(),
        )?;
        let bytes = checked_add(
            checked_add(
                forward.plan_bytes(),
                inverse.plan_bytes(),
                scioxide_core::quantity!("provider_plans"),
            )?,
            core::mem::size_of::<CpuPair<S>>() + 2 * core::mem::size_of::<usize>(),
            scioxide_core::quantity!("provider_state"),
        )?;
        let scratch = forward
            .workspace_requirements(())?
            .complex_len()
            .max(inverse.workspace_requirements(())?.complex_len());
        let diagnostics = FftPlanDiagnostics::for_provider(forward.diagnostics().stage_count());
        Ok(Self {
            plan: Arc::new(CpuPair {
                forward,
                inverse,
                bytes,
                scratch,
                diagnostics,
            }),
        })
    }
}
impl<S: RealScalar> FftProvider<S> for CpuProvider<S> {
    fn prepare(
        &self,
        length: usize,
        policy: ExecutionPolicy,
        limits: FftProviderLimits,
        context: ExecutionContext<'_>,
    ) -> Result<Option<Arc<dyn FftProviderPlan<S>>>, SciOxideError> {
        context.check()?;
        if length != self.plan.forward.spec().length()
            || policy != self.plan.forward.spec().execution_policy()
        {
            return Ok(None);
        }
        if self.plan.bytes > limits.preparation_bytes
            || self.plan.scratch > limits.workspace_bytes / core::mem::size_of::<Complex<S>>()
        {
            return Err(SciOxideError::external(
                "prepared adapter exceeds the provider budget",
            ));
        }
        {
            let plan = Arc::clone(&self.plan);
            Ok(Some(plan))
        }
    }
}
impl<S: RealScalar> FftProviderPlan<S> for CpuPair<S> {
    fn workspace_len(&self) -> usize {
        self.scratch
    }
    fn plan_bytes(&self) -> usize {
        self.bytes
    }
    fn preparation_bytes(&self) -> usize {
        self.bytes
    }
    fn diagnostics(&self) -> FftPlanDiagnostics {
        self.diagnostics
    }
    fn execute(
        &self,
        direction: FftDirection,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let plan = match direction {
            FftDirection::Forward => &self.forward,
            FftDirection::Inverse => &self.inverse,
        };
        plan.execute_in_place(data, scratch, context)
    }
}
