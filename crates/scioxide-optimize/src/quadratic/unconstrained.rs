//! Rank-aware unconstrained convex QP.
use alloc::borrow::ToOwned;

use scioxide_core::backend::{try_copy_slice, try_filled, try_with_capacity};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, Vector};

use crate::KktProvider;
use crate::context::SolveSession;
use crate::problem::QpProblem;
use crate::programming::ProgrammingResult;
use crate::result::OptimizeTermination;
use scioxide_core::PartialResultTrust;

use super::{QpSolver, common::*};
impl<S: RealScalar + NumericScalar<Real = S>> QpSolver<S> {
    /// A zero Hessian is an LP; reuse the bounded revised simplex and its
    /// anti-cycling rather than repeatedly solve singular quadratic models.
    pub(super) fn solve_affine(
        &self,
        problem: &QpProblem<S>,
        h: &[S],
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        use scioxide_core::backend::try_filled;
        let n = problem.dimension();
        let mut lp = crate::LpProblem::new(try_copy_slice(
            problem.linear_objective(),
            scioxide_core::quantity!("qp_affine_objective"),
        )?)?;
        if let Some(c) = problem.constraints() {
            let mut a = Matrix::zeros(c.len(), n)?;
            for (i, row) in a.as_mut_slice().chunks_exact_mut(n).enumerate() {
                c.coefficients().write_row(i, row)?;
            }
            lp = lp.with_constraints(crate::LinearConstraints::new(
                a,
                try_copy_slice(c.rhs(), scioxide_core::quantity!("qp_affine_rhs"))?,
                try_copy_slice(c.senses(), scioxide_core::quantity!("qp_affine_senses"))?,
            )?)?;
        }
        let bounds = match problem.bounds() {
            Some(b) => try_copy_slice(b.as_slice(), scioxide_core::quantity!("qp_affine_bounds"))?,
            None => try_filled(
                n,
                (None, None),
                scioxide_core::quantity!("qp_affine_bounds"),
            )?,
        };
        lp = lp.with_bounds(scioxide_core::Bounds::new(bounds)?)?;
        let solver = crate::SimplexSolver::default()
            .with_tolerances(self.tolerances)
            .with_max_iterations(self.max_iterations);
        let mut state = scioxide_core::SolveState::new();
        let mut result = session.with_child(
            &mut state,
            solver.identity(&lp),
            scioxide_core::SolveBudgets::unbounded()
                .with_limit(scioxide_core::BudgetKind::Iteration, self.max_iterations),
            |context| {
                let child = SolveSession::new(context, scioxide_core::argument!("context"));
                let result = solver.solve_observed(&lp, &child);
                if let Some(stop) = child.stopped() {
                    session.set_stopped(stop);
                }
                result
            },
        )?;
        result.fun = eval_objective(h, problem.linear_objective(), &result.x, n)?;
        if let Some(evidence) = &result.evidence {
            let view = crate::QpVerificationProblem::from_problem(problem);
            let mut workspace = crate::VerificationWorkspace::for_qp(&view)?;
            let report = crate::verification::kernel::qp_session(
                &view,
                evidence.borrow(&result.x),
                self.tolerances,
                &mut workspace,
                Some(session),
            )?;
            if result.is_success() && !report.is_accepted() {
                result.termination = OptimizeTermination::Stagnation;
            }
            result.diagnostics = report.diagnostics();
            result.verification = Some(report);
        }
        Ok(result)
    }
    /// Solve an unconstrained QP by solving `Q x = -c`.
    pub(super) fn solve_unconstrained(
        &self,
        problem: &QpProblem<S>,
        hessian: &[S],
        session: &SolveSession<'_, '_, '_>,
        provider: Option<&mut (dyn KktProvider<S> + Send)>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        session.proposal()?;
        let n = problem.dimension();
        let q_mat = Matrix::from_vec(
            n,
            n,
            try_copy_slice(hessian, scioxide_core::quantity!("qp_hessian_copy"))?,
        )?;
        let mut neg_c = try_with_capacity(n, scioxide_core::quantity!("qp_right_hand_side"))?;
        neg_c.extend(
            problem
                .linear_objective()
                .iter()
                .map(|&coefficient| -coefficient),
        );
        let rhs = Vector::from_vec(neg_c);

        let mut factor = crate::kkt::Factor::new(
            provider,
            n,
            Some(scioxide_linalg::RankPolicy::DimensionScaledEpsilon),
        )?;
        let mut x = try_filled(
            n,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("qp_solution"),
        )?;
        factor.solve(&q_mat, rhs.as_slice(), &mut x, session)?;

        let fun = eval_objective(hessian, problem.linear_objective(), &x, n)?;
        let assessment = observed_certificate(problem, hessian, &x, self.tolerances, session)?;
        let certified = assessment.accepted;
        session.accept(if certified {
            PartialResultTrust::ResidualChecked
        } else {
            PartialResultTrust::DiagnosticOnly
        })?;

        Ok(ProgrammingResult {
            evidence: assessment.evidence,
            verification: assessment.report,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            diagnostics: assessment.diagnostics,
            x,
            fun,
            iterations: 1,
            termination: if certified {
                OptimizeTermination::Converged
            } else {
                OptimizeTermination::SuspectedUnbounded
            },
            message: if certified {
                "unconstrained convex QP certified by a rank-revealing solve".to_owned()
            } else {
                "PSD Hessian has an objective component in its null space".to_owned()
            },
        })
    }
}
