//! Owned evaluation of configured CZT and NUFFT plans for tests.
//!
//! Prepared plans publish only caller-storage forms; the allocating
//! conveniences own their output. Tests that configure a plan beyond what a
//! convenience exposes allocate the output and workspace here and call
//! `execute_into` with a sequential native context.
#![allow(
    dead_code,
    reason = "each test target uses only part of this shared support"
)]

use scioxide_core::{Complex, ExecutionContext, NumericScalar, RealScalar, SciOxideError};
use scioxide_fft::{CztPlan, NufftPlan};

/// Execute a prepared plan into freshly allocated output.
pub(crate) trait ExecuteOwned<S: RealScalar> {
    /// Allocate the output and workspace, then run `execute_into`.
    fn execute_owned(&self, input: &[Complex<S>]) -> Result<Vec<Complex<S>>, SciOxideError>;
}

impl<S: RealScalar> ExecuteOwned<S> for CztPlan<S>
where
    Complex<S>: NumericScalar,
{
    fn execute_owned(&self, input: &[Complex<S>]) -> Result<Vec<Complex<S>>, SciOxideError> {
        let mut output = vec![Complex::new(S::ZERO, S::ZERO); self.spec().output_len()];
        self.execute_into(
            input,
            &mut output,
            &mut self.workspace(())?,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<S: RealScalar> ExecuteOwned<S> for NufftPlan<S>
where
    Complex<S>: NumericScalar,
{
    fn execute_owned(&self, input: &[Complex<S>]) -> Result<Vec<Complex<S>>, SciOxideError> {
        let mut output = vec![Complex::new(S::ZERO, S::ZERO); self.output_len()];
        self.execute_into(
            input,
            &mut output,
            &mut self.workspace(())?,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}
