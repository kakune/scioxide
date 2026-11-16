//! Instrumented diagonal operators for solver resource contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    AdjointOperator, CancellationToken, LinearOperator, ProblemToken, SciOxideError, SolveIdentity,
};
use std::cell::Cell;

pub fn parent() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("tests", "solver_parent", 1),
        0,
        0,
    )
}

pub struct Provider<'a> {
    pub rows: usize,
    pub diagonal: &'a [f64],
    pub calls: Cell<usize>,
    pub fail_at: usize,
    pub cancel: Option<&'a CancellationToken>,
}
impl Provider<'_> {
    fn invoke(
        &self,
        input: &[f64],
        output: &mut [f64],
        scratch: &mut [f64],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), if adjoint { 3 } else { 2 });
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == self.fail_at {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "injected failure",
            ));
        }
        output.fill(0.);
        for (i, &d) in self.diagonal.iter().enumerate() {
            output[i] = d * input[i];
        }
        if call == 5
            && let Some(cancel) = self.cancel
        {
            cancel.cancel();
        }
        Ok(())
    }
}
impl LinearOperator<f64> for Provider<'_> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.diagonal.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(2)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], w: &mut [f64]) -> Result<(), SciOxideError> {
        self.invoke(x, y, w, false)
    }
}
impl AdjointOperator<f64> for Provider<'_> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(3)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.invoke(x, y, w, true)
    }
}

pub struct FlexibleProvider<'a> {
    pub calls: Cell<usize>,
    pub fail: usize,
    pub cancel: Option<&'a CancellationToken>,
    pub scale: f64,
}
impl LinearOperator<f64> for FlexibleProvider<'_> {
    fn rows(&self) -> usize {
        3
    }
    fn cols(&self) -> usize {
        3
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(2)
    }
    fn apply_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), 2);
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == self.fail {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "injected failure",
            ));
        }
        for (i, (out, &input)) in y.iter_mut().zip(x).enumerate() {
            *out = self.scale * (i + 1) as f64 * input;
        }
        if call == 4
            && let Some(cancel) = self.cancel
        {
            cancel.cancel();
        }
        Ok(())
    }
}
