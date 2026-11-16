// Deterministic finite-grid reference solver, exclusively for tests and examples.
// Convergence means exhaustive search of the declared grid, not continuous optimality.
use scioxide_core::{EvaluationRole, RealScalar, SciOxideError, SolveContext};
use scioxide_stats::{FitRequest, FitSolver, FitSolverCapabilities, StatsTermination};
pub struct GridSolver {
    pub intervals: usize,
}
impl<S: RealScalar, W: Send> FitSolver<S, W> for GridSolver {
    fn capabilities(&self) -> FitSolverCapabilities {
        FitSolverCapabilities {
            bounds: true,
            integrality: true,
            ..Default::default()
        }
    }
    fn solve(
        &mut self,
        r: FitRequest<'_, S, W>,
        out: &mut [S],
        objective_workspace: &mut W,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<StatsTermination, SciOxideError> {
        let dimension = out.len();
        if dimension == 0 || dimension > 8 || self.intervals == 0 {
            return Err(SciOxideError::not_implemented("reference grid dimension"));
        }
        let mut candidate = [S::ZERO; 8];
        let mut best = None;
        let width = self.intervals + 1;
        let total = r
            .bounds
            .as_slice()
            .iter()
            .try_fold(1usize, |n, &(low, high)| {
                n.checked_mul(if low == high { 1 } else { width })
                    .ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("reference_grid"))
                    })
            })?;
        for index in 0..total {
            if let Some(stop) = context.stop_reason()? {
                return Ok(StatsTermination::Stopped(stop));
            }
            if !context.record_iteration()?.is_granted() {
                return Ok(StatsTermination::BudgetExhausted);
            }
            let mut index = index;
            for (j, x) in candidate[..dimension].iter_mut().enumerate() {
                let (low, high) = r.bounds.as_slice()[j];
                let low = low.ok_or_else(|| {
                    SciOxideError::not_implemented("reference grid needs finite bounds")
                })?;
                let high = high.ok_or_else(|| {
                    SciOxideError::not_implemented("reference grid needs finite bounds")
                })?;
                let radix = if low == high { 1 } else { width };
                let fraction =
                    S::try_from_f64_lossy((index % radix) as f64 / self.intervals as f64)?;
                index /= radix;
                *x = low + (high - low) * fraction;
                if r.integrality[j] {
                    *x = x.round();
                }
            }
            if !context
                .record_provider(EvaluationRole::Objective)?
                .is_granted()
            {
                return Ok(StatsTermination::BudgetExhausted);
            }
            match r
                .objective
                .value(&candidate[..dimension], objective_workspace)
            {
                Ok(value) => {
                    if best.is_none_or(|best| value < best) {
                        best = Some(value);
                        out.copy_from_slice(&candidate[..dimension]);
                    }
                }
                Err(__scioxide_error)
                    if matches!(
                        __scioxide_error.view(),
                        scioxide_core::SciOxideErrorView::DomainError { .. }
                            | scioxide_core::SciOxideErrorView::NonFinite { .. }
                            | scioxide_core::SciOxideErrorView::InvalidInput { .. }
                    ) =>
                {
                    let (scioxide_core::SciOxideErrorView::DomainError { .. }
                    | scioxide_core::SciOxideErrorView::NonFinite { .. }
                    | scioxide_core::SciOxideErrorView::InvalidInput { .. }) =
                        __scioxide_error.view()
                    else {
                        unreachable!("guard validated the error view")
                    };
                }
                Err(e) => return Err(e),
            }
        }
        Ok(if best.is_some() {
            StatsTermination::Converged
        } else {
            StatsTermination::NotConverged
        })
    }
}
