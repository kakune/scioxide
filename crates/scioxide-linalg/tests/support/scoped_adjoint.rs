//! A provider that requires scoped dispatch and performs a nested numerical probe.
use scioxide_core::{AdjointOperator, EvaluationRole, LinearOperator, SciOxideError, SolveContext};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct ScopedDiagonal {
    pub forward: AtomicUsize,
    pub adjoint: AtomicUsize,
}
impl ScopedDiagonal {
    pub fn new() -> Self {
        Self {
            forward: AtomicUsize::new(0),
            adjoint: AtomicUsize::new(0),
        }
    }
    fn unscoped() -> Result<(), SciOxideError> {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("context"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "scoped dispatch required",
        ))
    }
    fn run(
        &self,
        x: &[f64],
        y: &mut [f64],
        c: &mut SolveContext<'_, '_>,
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        c.execution().check()?;
        let mut state = scioxide_core::SolveState::new();
        let id = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(2),
            scioxide_core::algorithm_id!("test", "nested-probe", 1),
            0,
            0,
        );
        scioxide_core::SolveScope::from(c).run(
            &mut state,
            id,
            scioxide_core::SolveBudgets::unbounded(),
            |child| {
                child.record_provider(EvaluationRole::Function)?.require()?;
                child.record_iterations(2)?.require()?;
                for ((out, &input), d) in y.iter_mut().zip(x).zip([2.0, 3.0, 4.0]) {
                    *out = input * d;
                }
                Ok(())
            },
        )?;
        if adjoint {
            self.adjoint.fetch_add(1, Ordering::Relaxed);
        } else {
            self.forward.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}
impl LinearOperator<f64> for ScopedDiagonal {
    fn rows(&self) -> usize {
        3
    }
    fn cols(&self) -> usize {
        3
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, _: &[f64], _: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        Self::unscoped()
    }
    fn apply_into_scoped(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut [f64],
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.run(x, y, c, false)
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl AdjointOperator<f64> for ScopedDiagonal {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        _: &[f64],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        Self::unscoped()
    }
    fn apply_adjoint_into_scoped(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut [f64],
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.run(x, y, c, true)
            .map(scioxide_core::Invocation::Evaluated)
    }
}
