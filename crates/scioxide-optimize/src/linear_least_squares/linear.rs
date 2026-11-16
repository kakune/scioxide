//! Direct and matrix-free subproblem solves. No normal equations are formed.
use super::workspace::Storage;
use super::*;
use crate::context::SolveSession;

use scioxide_core::{EvaluationRole, LinearOperator};
use scioxide_linalg::{LinearSolveTermination, LsmrSolver};
pub(super) struct Model<'a, 'p, S: RealScalar + NumericScalar<Real = S>> {
    pub p: &'a LinearLeastSquaresProblem<'p, S>,
    pub passive: &'a [bool],
}
impl<S: RealScalar + NumericScalar<Real = S>> Model<'_, '_, S> {
    fn apply_forward(
        &self,
        x: &[S],
        out: &mut [S],
        scratch: &mut [S],
        context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    ) -> Result<(), SciOxideError> {
        shape(self.cols(), x.len())?;
        shape(self.rows(), out.len())?;
        if scratch.len() < self.apply_workspace_len()? {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "insufficient model scratch",
            ));
        }
        let (temp, provider) = scratch.split_at_mut(self.cols());
        for ((v, &x), &passive) in temp.iter_mut().zip(x).zip(self.passive) {
            *v = if passive { x } else { S::ZERO };
        }
        let m = self.p.rows();
        match context {
            Some(context) => self
                .p
                .operator()
                .apply_into_scoped(
                    temp,
                    &mut out[..m],
                    &mut provider[..self.p.operator().apply_workspace_len()?],
                    &mut scioxide_core::OperatorContext::new(context),
                )
                .and_then(scioxide_core::Invocation::require),
            None => self.p.operator().apply_into(
                temp,
                &mut out[..m],
                &mut provider[..self.p.operator().apply_workspace_len()?],
            ),
        }?;
        finite_product(out)
    }
    fn apply_adjoint(
        &self,
        x: &[S],
        out: &mut [S],
        scratch: &mut [S],
        context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    ) -> Result<(), SciOxideError> {
        shape(self.rows(), x.len())?;
        shape(self.cols(), out.len())?;
        if scratch.len() < self.apply_workspace_len()? {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "insufficient model scratch",
            ));
        }
        let (_, provider) = scratch.split_at_mut(self.cols());
        let m = self.p.rows();
        match context {
            Some(context) => self
                .p
                .operator()
                .apply_adjoint_into_scoped(
                    &x[..m],
                    out,
                    &mut provider[..self.p.operator().apply_adjoint_workspace_len()?],
                    &mut scioxide_core::OperatorContext::new(context),
                )
                .and_then(scioxide_core::Invocation::require),
            None => self.p.operator().apply_adjoint_into(
                &x[..m],
                out,
                &mut provider[..self.p.operator().apply_adjoint_workspace_len()?],
            ),
        }?;
        finite_product(out)?;
        for (v, &passive) in out.iter_mut().zip(self.passive) {
            if !passive {
                *v = S::ZERO;
            }
        }
        finite_product(out)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> LinearOperator<S> for Model<'_, '_, S> {
    fn rows(&self) -> usize {
        self.p.rows()
    }
    fn cols(&self) -> usize {
        self.p.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.p
            .cols()
            .checked_add(
                self.p
                    .operator()
                    .apply_workspace_len()?
                    .max(self.p.operator().apply_adjoint_workspace_len()?),
            )
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("model_scratch")))
    }
    fn apply_into(&self, x: &[S], out: &mut [S], scratch: &mut [S]) -> Result<(), SciOxideError> {
        self.apply_forward(x, out, scratch, None)
    }
    fn apply_into_scoped(
        &self,
        x: &[S],
        out: &mut [S],
        scratch: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_forward(x, out, scratch, Some(context))
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> AdjointOperator<S> for Model<'_, '_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.apply_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        x: &[S],
        out: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_adjoint(x, out, scratch, None)
    }
    fn apply_adjoint_into_scoped(
        &self,
        x: &[S],
        out: &mut [S],
        scratch: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply_adjoint(x, out, scratch, Some(context))
            .map(scioxide_core::Invocation::Evaluated)
    }
}
pub(super) fn solver<S: RealScalar + NumericScalar<Real = S>>(
    o: &LinearLeastSquaresOptions<S>,
) -> Result<LsmrSolver<S>, SciOxideError> {
    // An absolute epsilon floor can accept zero on a uniformly tiny operator,
    // despite a caller requesting a much smaller original-coordinate gradient.
    let tolerance =
        scioxide_core::Tolerance::new(<_ as scioxide_core::AlgebraicScalar>::ZERO, S::EPSILON)?;
    let normal_absolute =
        super::engine::add(S::ZERO, o.gradient.threshold(S::ONE)?, S::HALF * S::HALF)?;
    let normal = scioxide_core::GradientTolerance::new(normal_absolute, S::EPSILON)?;
    Ok(LsmrSolver::new(
        scioxide_core::ResidualTolerance::from_tolerance(tolerance),
        o.inner,
    )
    .with_normal_tolerance(normal))
}
pub(super) fn step<S: RealScalar + NumericScalar<Real = S>>(
    p: &LinearLeastSquaresProblem<'_, S>,
    o: &LinearLeastSquaresOptions<S>,
    w: &mut Storage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    w.rank = None;
    w.step.fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
    if let Some(d) = w.dense.as_mut() {
        let mut k = 0;
        for (j, &passive) in w.passive.iter().enumerate() {
            if passive {
                w.map[k] = j;
                k += 1;
            }
        }
        if k == 0 {
            return Ok(());
        }
        let m = p.rows();
        let updated = session.invoke(EvaluationRole::Factorization, || {
            d.active.synchronize(&w.passive, &mut d.matrix)
        })?;
        if updated {
            let rank = d.rank.estimate(&d.matrix, o.rank)?;
            if let Some(rank) = rank.filter(|rank| rank.is_full_rank()) {
                w.rank = Some(rank);
                session.invoke(EvaluationRole::LinearSolve, || {
                    d.active.step(&w.accepted.x, &mut w.step)
                })?;
                return finite(&w.step);
            }
        }
        // Wide, dependent or inconclusive systems retain the original
        // rank-revealing minimum-norm solve, never normal equations.
        d.matrix.try_resize_flat(m, k)?;
        d.rhs.try_resize(m, S::ZERO)?;
        d.solution.try_resize(k, S::ZERO)?;
        let a = p.dense().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "missing dense coefficients",
            )
        })?;
        for i in 0..m {
            for (j, &original) in w.map[..k].iter().enumerate() {
                d.matrix.as_mut_slice()[i * k + j] = a.get(i, original)?;
            }
            d.rhs.as_mut_slice()[i] = -w.accepted.residual[i];
        }
        let mut use_svd = m < k;
        if !use_svd {
            session.invoke(EvaluationRole::Factorization, || {
                scioxide_linalg::HouseholderDecomposition::factor_into(
                    &d.matrix,
                    scioxide_linalg::HouseholderOrientation::Qr,
                    scioxide_linalg::HouseholderPivoting::Column,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
                    &mut d.qr,
                    &mut d.qw,
                    session.execution()?,
                )
            })?;
            let rank = d.qr.analyze_rank(o.rank)?;
            use_svd = !rank.is_full_rank();
            w.rank = Some(rank);
        }
        if use_svd {
            session.invoke(EvaluationRole::Factorization, || {
                scioxide_linalg::ThinSvdDecomposition::compute_into(
                    &d.matrix,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    &mut d.svd,
                    &mut d.sw,
                    session.execution()?,
                )
            })?;
            w.rank = Some(d.svd.analyze_rank(o.rank)?);
            session.invoke(EvaluationRole::LinearSolve, || {
                d.svd.solve_into(
                    &d.rhs,
                    o.rank,
                    S::ZERO,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    d.solution.view_mut(),
                    &mut d.sw,
                    session.execution()?,
                )
            })?;
        } else {
            let _ = session.invoke(EvaluationRole::LinearSolve, || {
                d.qr.solve_into(
                    &d.rhs,
                    &scioxide_linalg::DirectSolveOptions::new(o.rank),
                    d.solution.view_mut(),
                    &mut d.qw,
                    session.execution()?,
                )
            })?;
        }
        for (j, &original) in w.map[..k].iter().enumerate() {
            w.step[original] = d.solution.as_slice()[j];
        }
    } else {
        let model = Model {
            p,
            passive: &w.passive,
        };
        w.rhs
            .as_mut_slice()
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        for (v, &r) in w.rhs.as_mut_slice().iter_mut().zip(&w.accepted.residual) {
            *v = -r;
        }
        w.solution
            .as_mut_slice()
            .fill(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        let solver = solver(o)?;
        let scratch = w.lsmr.as_mut().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "unprepared LSMR storage",
            )
        })?;
        // LSMR runs in a child epoch whose iteration allowance is the solver limit.
        let result = session.invoke(EvaluationRole::LinearSolve, || {
            session
                .with_context(|context| {
                    solver.solve_into(&model, &w.rhs, w.solution.view_mut(), scratch, context)
                })
                .and_then(|report| Ok(report?.termination()))
        });
        let termination = match result {
            Ok(t) => t,
            Err(e) => {
                session.record_execution_stop(&e);
                session.inherit_child_stop(Some(OptimizeTermination::EvaluationLimit))?;
                session.inherit_child_stop(Some(OptimizeTermination::IterationLimit))?;
                return Err(e);
            }
        };
        let stop = match termination {
            LinearSolveTermination::IterationLimit => Some(OptimizeTermination::IterationLimit),
            LinearSolveTermination::EvaluationLimit => Some(OptimizeTermination::EvaluationLimit),
            LinearSolveTermination::ExecutionStopped(
                scioxide_core::ExecutionStopReason::DeadlineReached,
            ) => Some(OptimizeTermination::TimeLimit),
            LinearSolveTermination::ExecutionStopped(_) => Some(OptimizeTermination::Cancelled),
            _ => None,
        };
        session.inherit_child_stop(stop)?;
        for ((v, &x), &passive) in w.step.iter_mut().zip(w.solution.as_slice()).zip(&w.passive) {
            *v = if passive { x } else { S::ZERO };
        }
    }
    finite(&w.step)
}
