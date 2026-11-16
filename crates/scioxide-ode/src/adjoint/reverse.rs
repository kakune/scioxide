//! Error control and step acceptance of the reverse sweep, which runs on the IVP driver.
use super::*;
use crate::termination::{Fault, Work};
use core::cell::RefCell;
use scioxide_linalg::Vector;

pub(super) struct Reverse<'a, 'r, S: RealScalar> {
    pub options: &'a Policies<S>,
    pub dimension: usize,
    pub attempts: &'a mut usize,
    pub limit: usize,
    pub restart: bool,
    pub report: &'a RefCell<&'r mut AdjointReport<S>>,
    pub adjoint: S,
    pub quadrature: S,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> crate::ivp::Recorder<S, T>
    for Reverse<'_, '_, S>
{
    fn restart(&self) -> bool {
        self.restart
    }
    fn self_metered(&self) -> bool {
        true
    }
    fn admit(&mut self, cx: &mut SolveContext<'_, '_>) -> Result<(), Fault> {
        if *self.attempts >= self.limit {
            self.report.borrow_mut().termination =
                Some(AdjointTermination::BudgetLimit(BudgetKind::Iteration));
            return Err(convergence(
                scioxide_core::ConvergenceReason::IterationLimit,
                "continuous reverse-step limit exhausted",
            )
            .into());
        }
        crate::termination::record(cx, Work::Proposal)?;
        *self.attempts += 1;
        Ok(())
    }
    fn error_norm(
        &mut self,
        initial: &Vector<T>,
        candidate: &Vector<T>,
        step: &crate::result::StepResult<'_, S, T>,
    ) -> Result<Option<S>, SciOxideError> {
        let errors = step.raw_error.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("method"),
                "continuous reverse trials require local errors",
            )
        })?;
        let n = self.dimension;
        self.adjoint = self.options.adjoint_error.normalized_norm_by(n, |i| {
            (
                errors[i].abs(),
                initial.as_slice()[i]
                    .abs()
                    .max(candidate.as_slice()[i].abs()),
            )
        })?;
        let p = initial.len() - n;
        self.quadrature = if p == 0 {
            S::ZERO
        } else {
            self.options.quadrature_error.normalized_norm_by(p, |i| {
                let j = n + i;
                (
                    errors[j].abs(),
                    initial.as_slice()[j]
                        .abs()
                        .max(candidate.as_slice()[j].abs()),
                )
            })?
        };
        Ok(Some(self.adjoint.max(self.quadrature)))
    }
    fn accepted(
        &mut self,
        _: S,
        _: S,
        _: S,
        _: &Vector<T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let mut report = self.report.borrow_mut();
        report.errors.adjoint = Some(report.errors.adjoint.unwrap_or(S::ZERO).max(self.adjoint));
        report.errors.quadrature = Some(
            report
                .errors
                .quadrature
                .unwrap_or(S::ZERO)
                .max(self.quadrature),
        );
        Ok(())
    }
}
