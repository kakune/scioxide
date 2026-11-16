//! Operators and assertions for public action reservation contracts.
use scioxide_core::{AdjointOperator, LinearOperator, NumericScalar, SciOxideError};
use std::cell::Cell;

pub struct Operator {
    n: usize,
    scratch: usize,
    pub calls: Cell<usize>,
}
impl Operator {
    pub fn new(n: usize, scratch: usize) -> Self {
        Self {
            n,
            scratch,
            calls: Cell::new(0),
        }
    }
}
impl<S: NumericScalar> LinearOperator<S> for Operator {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn apply_into(&self, x: &[S], y: &mut [S], scratch: &mut [S]) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), self.scratch);
        self.calls.set(self.calls.get() + 1);
        y.copy_from_slice(x);
        Ok(())
    }
}
impl<S: NumericScalar> AdjointOperator<S> for Operator {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn apply_adjoint_into(
        &self,
        x: &[S],
        y: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, y, scratch)
    }
}

impl<S: NumericScalar> scioxide_core::Preconditioner<S> for Operator {
    fn dimension(&self) -> usize {
        self.n
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn precondition_into(
        &self,
        x: &[S],
        y: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, y, scratch)
    }
}
