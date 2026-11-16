#![allow(
    dead_code,
    reason = "each test target uses part of the shared provider"
)]
use super::numerics;
use scioxide_core::{Complex, ExecutionContext, ExecutionPolicy, RealScalar, SciOxideError};
use scioxide_fft::{
    FftDirection, FftPlanDiagnostics, FftProvider, FftProviderLimits, FftProviderPlan,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Direct,
    Decline,
    PreparationFailure,
    ExecutionFailure,
    NonFinite,
    LargeFinite,
    ImaginaryResidual,
    ExcessScratch,
    UnderreportedPeak,
    ExcessPreparation,
}
#[derive(Debug)]
pub struct DirectProvider {
    pub calls: Arc<AtomicUsize>,
    pub preparations: AtomicUsize,
    pub preparation_limit: AtomicUsize,
    pub workspace_limit: AtomicUsize,
    pub mode: Mode,
}
impl DirectProvider {
    pub fn new(mode: Mode) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            preparations: AtomicUsize::new(0),
            preparation_limit: AtomicUsize::new(0),
            workspace_limit: AtomicUsize::new(0),
            mode,
        }
    }
}
#[derive(Debug)]
struct DirectPlan {
    n: usize,
    mode: Mode,
    calls: Arc<AtomicUsize>,
}
impl<S: RealScalar> FftProvider<S> for DirectProvider {
    fn prepare(
        &self,
        n: usize,
        _: ExecutionPolicy,
        limits: FftProviderLimits,
        _: ExecutionContext<'_>,
    ) -> Result<Option<Arc<dyn FftProviderPlan<S>>>, SciOxideError> {
        self.preparations.fetch_add(1, Ordering::Relaxed);
        self.preparation_limit
            .store(limits.preparation_bytes, Ordering::Relaxed);
        self.workspace_limit
            .store(limits.workspace_bytes, Ordering::Relaxed);
        match self.mode {
            Mode::Decline => return Ok(None),
            Mode::PreparationFailure => return Err(failure()),
            _ => {}
        }
        if std::mem::size_of::<DirectPlan>() > limits.preparation_bytes
            || n.saturating_mul(std::mem::size_of::<Complex<S>>()) > limits.workspace_bytes
        {
            return Err(failure());
        }
        Ok(Some(Arc::new(DirectPlan {
            n,
            mode: self.mode,
            calls: Arc::clone(&self.calls),
        })))
    }
}
fn failure() -> SciOxideError {
    SciOxideError::execution_failure(
        None,
        scioxide_core::ExecutionFailureReason::ProviderFailure,
        "injected failure",
    )
}
impl<S: RealScalar> FftProviderPlan<S> for DirectPlan {
    fn workspace_len(&self) -> usize {
        if matches!(self.mode, Mode::ExcessScratch) {
            usize::MAX
        } else {
            self.n
        }
    }
    fn plan_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
    fn preparation_bytes(&self) -> usize {
        match self.mode {
            Mode::UnderreportedPeak => 0,
            Mode::ExcessPreparation => usize::MAX,
            _ => core::mem::size_of::<Self>(),
        }
    }
    fn diagnostics(&self) -> FftPlanDiagnostics {
        FftPlanDiagnostics::for_provider(1)
    }
    fn execute(
        &self,
        direction: FftDirection,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        self.calls.fetch_add(1, Ordering::Relaxed);
        if matches!(self.mode, Mode::ExecutionFailure) {
            data.fill(Complex::new(S::ZERO, S::ZERO));
            return Err(failure());
        }
        if matches!(self.mode, Mode::NonFinite) {
            data[0] = Complex::new(S::NAN, S::ZERO);
            return Ok(());
        }
        if matches!(self.mode, Mode::LargeFinite) {
            data.fill(Complex::new(S::ZERO, S::ZERO));
            data[0].re = S::MAX.scalbn(-10);
            return Ok(());
        }
        scratch.copy_from_slice(data);
        let sign = if direction == FftDirection::Forward {
            -1.0
        } else {
            1.0
        };
        for (k, out) in data.iter_mut().enumerate() {
            let value = numerics::direct_sum(scratch, |j| {
                (
                    sign * std::f64::consts::TAU * (j * k % self.n) as f64 / self.n as f64,
                    1.0,
                )
            });
            *out = Complex::new(
                S::try_from_f64_lossy(value.re)?,
                S::try_from_f64_lossy(value.im)?,
            );
        }
        if matches!(self.mode, Mode::ImaginaryResidual) && direction == FftDirection::Inverse {
            data[1].im += S::ONE;
        }
        Ok(())
    }
}
