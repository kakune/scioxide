//! Projected Barzilai–Borwein bound-only QP.
use alloc::borrow::ToOwned;

use scioxide_core::backend::{try_copy_slice, try_filled};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

use crate::context::SolveSession;
use crate::problem::QpProblem;
use crate::programming::ProgrammingResult;
use crate::result::OptimizeTermination;
use crate::utils::{dot, norm2};
use scioxide_core::PartialResultTrust;

use super::{QpSolver, common::*, model::validate_warm};
impl<S: RealScalar + NumericScalar<Real = S>> QpSolver<S> {
    /// Solve a bound-only constrained QP via projected gradient descent.
    ///
    /// Uses a simple projected gradient method with Barzilai-Borwein step size.
    pub(super) fn solve_bound_constrained(
        &self,
        problem: &QpProblem<S>,
        hessian: &[S],
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<ProgrammingResult<S>, SciOxideError> {
        let n = problem.dimension();
        let bounds = problem.bounds().ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("problem"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "expected bounds for bound-constrained solve",
            )
        })?;

        // Start at a feasible point: midpoint of bounds or zero projected.
        let mut x = try_filled(
            n,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("qp_projected_iterate"),
        )?;
        if let Some(warm) = &self.warm_start {
            validate_warm(problem, &warm.point, self.tolerances)?;
            x.copy_from_slice(&warm.point);
        } else {
            project_to_bounds(&mut x, bounds.as_slice());
        }

        let mut grad = compute_gradient(hessian, problem.linear_objective(), &x, n)?;
        let mut x_prev = try_copy_slice(&x, scioxide_core::quantity!("qp_previous_iterate"))?;
        let mut grad_prev =
            try_copy_slice(&grad, scioxide_core::quantity!("qp_previous_gradient"))?;
        let mut s = try_filled(
            n,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("qp_iterate_difference"),
        )?;
        let mut y = try_filled(
            n,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("qp_gradient_difference"),
        )?;

        // Initial step size from scalar approximation: 1 / max(diag(Q))
        let mut alpha = initial_step_size(hessian, n)?;

        for iter in 0..self.max_iterations.get() {
            if let Some(stop) = session.trial_iteration()? {
                return stopped_result(problem, hessian, x, iter, stop);
            }
            // Projected gradient step
            x_prev.copy_from_slice(&x);
            grad_prev.copy_from_slice(&grad);

            for i in 0..n {
                x[i] = crate::utils::trial_component(x[i], -alpha, grad[i])?;
            }
            project_to_bounds(&mut x, bounds.as_slice());

            let previous_value = eval_objective(hessian, problem.linear_objective(), &x_prev, n)?;
            for _ in 0..64 {
                let trial_value = eval_objective(hessian, problem.linear_objective(), &x, n)?;
                if trial_value <= previous_value {
                    break;
                }
                alpha *= S::HALF;
                for i in 0..n {
                    x[i] = crate::utils::trial_component(x_prev[i], -alpha, grad_prev[i])?;
                }
                project_to_bounds(&mut x, bounds.as_slice());
            }
            if eval_objective(hessian, problem.linear_objective(), &x, n)? > previous_value {
                return stopped_result(
                    problem,
                    hessian,
                    x_prev,
                    iter,
                    OptimizeTermination::Stagnation,
                );
            }
            compute_gradient_into(hessian, problem.linear_objective(), &x, &mut grad)?;

            session.accept(PartialResultTrust::Admissible)?;

            // Check convergence: projected gradient norm
            let pg_norm = projected_gradient_norm(
                &x,
                &grad,
                bounds.as_slice(),
                self.tolerances.feasibility().tolerance(),
            )?;

            if self
                .tolerances
                .gradient()
                .is_satisfied(pg_norm, <S as scioxide_core::AlgebraicScalar>::ONE)?
            {
                let fun = eval_objective(hessian, problem.linear_objective(), &x, n)?;
                let assessment =
                    observed_certificate(problem, hessian, &x, self.tolerances, session)?;
                let diagnostics = assessment.diagnostics;
                let certified = assessment.accepted;
                // This norm is a screening test. A rejected witness requires
                // further iterations while the original point can still improve.
                if certified || x == x_prev || session.stopped().is_some() {
                    return Ok(ProgrammingResult {
                        evidence: assessment.evidence,
                        verification: assessment.report,
                        counts: scioxide_core::SolveCounts::new(),
                        algorithm: None,
                        trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
                        diagnostics,
                        x,
                        fun,
                        iterations: iter + 1,
                        termination: if certified {
                            OptimizeTermination::Converged
                        } else {
                            OptimizeTermination::Stagnation
                        },
                        message: if certified {
                            "bound-constrained QP KKT certificate satisfied".to_owned()
                        } else {
                            "projected gradient stalled without a KKT certificate".to_owned()
                        },
                    });
                }
            }

            // Barzilai-Borwein step size update
            for i in 0..n {
                s[i] = scioxide_core::checked_difference(x[i], x_prev[i])?;
                y[i] = scioxide_core::checked_difference(grad[i], grad_prev[i])?;
            }

            let sn = norm2(&s)?;
            let yn = norm2(&y)?;
            if sn > <S as scioxide_core::AlgebraicScalar>::ZERO
                && yn > <S as scioxide_core::AlgebraicScalar>::ZERO
            {
                for v in &mut s {
                    *v = scioxide_core::checked_float_quotient(*v, sn)?;
                }
                for v in &mut y {
                    *v = scioxide_core::checked_float_quotient(*v, yn)?;
                }
                let cosine = dot(&s, &y)?;
                if cosine > S::SQRT_EPSILON {
                    use scioxide_core::Scaled;
                    alpha = Scaled::from_value(sn)?
                        .div(Scaled::from_value(yn)?)?
                        .div(Scaled::from_value(cosine)?)?
                        .round()?;
                }
            }
        }

        let fun = eval_objective(hessian, problem.linear_objective(), &x, n)?;
        let assessment = observed_certificate(problem, hessian, &x, self.tolerances, session)?;
        let diagnostics = assessment.diagnostics;
        Ok(ProgrammingResult {
            evidence: assessment.evidence,
            verification: assessment.report,
            counts: scioxide_core::SolveCounts::new(),
            algorithm: None,
            trust: scioxide_core::PartialResultTrust::DiagnosticOnly,
            diagnostics,
            x,
            fun,
            iterations: self.max_iterations.get(),
            termination: OptimizeTermination::IterationLimit,
            message: "projected gradient did not converge within max iterations".to_owned(),
        })
    }
}
