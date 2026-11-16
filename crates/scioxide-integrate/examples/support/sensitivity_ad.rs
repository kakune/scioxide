//! The integrand exp(p·x) with passive x and an active real parameter, on a tape.
use alloc::vec::Vec;
use scioxide_ad::{Linearization, Real, Tape, TapeOptions};
use scioxide_core::{ExecutionContext, LinearOperator, NumericScalar, RealScalar, SciOxideError};
use scioxide_integrate::{SensitivityIntegrand, SensitivityMode, SensitivitySample};
extern crate alloc;

/// The integrand `exp(p·x)`.
pub struct Exponential;

/// One worker's tape and operator scratch.
pub struct Workspace<S: RealScalar> {
    tape: Tape<S>,
    operator: Vec<S>,
}

/// Records `exp(p·x)` at one abscissa; `x` stays passive.
fn record<'a, S: RealScalar>(
    tape: &'a mut Tape<S>,
    x: S,
    p: S,
    execution: ExecutionContext<'_>,
) -> Result<Linearization<'a, S>, SciOxideError> {
    tape.linearize(
        |v, y| {
            y.push((v[0] * x).exp());
            Ok(())
        },
        &[p],
        execution,
    )
}

impl Exponential {
    /// Records once, so every evaluation of a query allocates nothing.
    pub fn workspace<S: RealScalar>(&self) -> Result<Workspace<S>, SciOxideError> {
        let mut tape = Tape::new(TapeOptions::new());
        let len = record(&mut tape, S::ZERO, S::ZERO, ExecutionContext::sequential())?
            .jacobian_operator()?
            .apply_workspace_len()?;
        Ok(Workspace {
            tape,
            operator: scioxide_core::backend::try_filled(
                len,
                S::ZERO,
                scioxide_core::quantity!("ad_quadrature_scratch"),
            )?,
        })
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> SensitivityIntegrand<S> for Exponential {
    type Workspace = Workspace<S>;
    fn parameter_dimension(&self) -> usize {
        1
    }
    fn value_dimension(&self) -> usize {
        1
    }
    fn supports(&self, _: SensitivityMode) -> bool {
        true
    }
    fn evaluate(
        &self,
        x: S,
        p: &[S],
        sample: &mut SensitivitySample<'_, S>,
        ws: &mut Workspace<S>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let point = record(&mut ws.tape, x, p[0], execution)?;
        sample.value_mut()[0] = point.values()[0];
        if sample.mode().is_some() {
            let operator = point.jacobian_operator()?;
            sample.apply(&operator, &mut ws.operator)?;
        }
        Ok(())
    }
}
