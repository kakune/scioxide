//! Quadratic penalty method.

super::outer::outer_method!(
    /// Minimizes a nonlinear program by the quadratic penalty method.
    ///
    /// Each outer iteration minimizes the penalty function
    ///
    /// ```text
    /// Q(x; μ) = f(x) + (μ/2) (Σᵢ hᵢ(x)² + Σⱼ max(gⱼ(x), 0)²)
    /// ```
    ///
    /// with [`Bfgs`](crate::Bfgs), starting from the previous iterate, where
    /// `hᵢ` are the equality rows and `gⱼ` the inequality rows of the
    /// constraints, scaled as described in [Termination](crate#termination).
    /// The penalty parameter starts at `μ = 1` and is multiplied by 10 when the
    /// constraint violation does not decrease enough. If a subproblem does not
    /// reduce the violation, a restoration subproblem that minimizes the
    /// violation alone is solved instead. Each finite bound side is an inequality row, so iterates may lie
    /// outside the bounds until the solve converges.
    ///
    /// The subproblems become ill conditioned as `μ` grows.
    /// [`AugmentedLagrangian`](crate::AugmentedLagrangian) avoids this with
    /// multiplier estimates.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_optimize::{
    ///     ConstrainedProblem, ConstraintFn, Constraints, ObjectiveFn, PenaltyMethod,
    /// };
    ///
    /// // minimize x² + y² subject to x + y = 1; the minimum is 1/2
    /// let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
    /// let constraints = Constraints::new(2)?
    ///     .equality(ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0)))?;
    /// let problem = ConstrainedProblem::new(&objective, &constraints)?;
    /// let result = PenaltyMethod::default().minimize(&problem, &[2.0, 2.0])?;
    /// assert!((result.fun() - 0.5).abs() < 0.05);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
    /// 2006, §17.1.
    PenaltyMethod,
    PenaltyOptions,
    PenaltyMethodWorkspace,
    PenaltyMethodWorkspaceRequirements,
    "PenaltyMethod",
    false
);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{
        Bfgs, ConstrainedProblem, ConstraintFn, Constraints, ObjectiveFunction, OptimizeResult,
        OptimizeTermination,
    };
    use scioxide_core::{Budget, ExecutionContext, SciOxideError};

    struct Quadratic;
    impl ObjectiveFunction<f64> for Quadratic {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }
        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output[0] = 2.0 * x[0];
            output[1] = 2.0 * x[1];
            Ok(true)
        }
    }

    #[test]
    fn prepared_penalty_converges_with_typed_constraints() {
        let inequality = ConstraintFn::new(2, |x: &[f64]| Ok(1.0 - x[0] - x[1]));
        let constraints = Constraints::new(2).unwrap().inequality(inequality).unwrap();
        let problem = ConstrainedProblem::new(&Quadratic, &constraints).unwrap();
        let solver = PenaltyMethod::<f64>::default();
        let mut workspace = solver.workspace(&problem).unwrap();
        let mut output = OptimizeResult::empty();
        solver
            .minimize_into(
                &problem,
                &[2.0, 2.0],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(output.is_success(), "{}", output.message);
        assert!((output.fun - 0.5).abs() < 0.05);
        let counts = workspace.last_attempt().unwrap().counts().total().unwrap();
        assert!(counts.provider(scioxide_core::EvaluationRole::Constraint) > 0);
    }

    #[test]
    fn local_inner_limit_allows_outer_progress() {
        let equality = ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0));
        let constraints = Constraints::new(2).unwrap().equality(equality).unwrap();
        let problem = ConstrainedProblem::new(&Quadratic, &constraints).unwrap();
        let inner = Bfgs::new(crate::BfgsOptions::default().with_max_iterations(Budget::ONE));
        let solver = PenaltyMethod::new(PenaltyOptions::default()).with_inner_optimizer(inner);
        let mut workspace = solver.workspace(&problem).unwrap();
        let mut output = OptimizeResult::empty();

        solver
            .minimize_into(
                &problem,
                &[2.0, 2.0],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();

        assert_eq!(
            output.termination(),
            OptimizeTermination::Converged,
            "{output:?}"
        );
        assert!(output.is_success());
        assert!(output.counts().nested().iterations() > 1);
        assert!(output.primal_feasibility().unwrap() < 3.0);
    }

    #[test]
    fn configured_and_no_constraint_paths_converge() {
        let constraint = ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0));
        let constraints = Constraints::new(2).unwrap().equality(constraint).unwrap();
        let problem = ConstrainedProblem::new(&Quadratic, &constraints).unwrap();
        let result = PenaltyMethod::default()
            .minimize(&problem, &[2.0, 2.0])
            .unwrap();
        assert!(result.is_success(), "{}", result.message);
        assert!((result.fun - 0.5).abs() < 0.05);

        let empty = Constraints::new(2).unwrap();
        let problem = ConstrainedProblem::new(&Quadratic, &empty).unwrap();
        let result = PenaltyMethod::<f64>::default()
            .minimize(&problem, &[3.0, 4.0])
            .unwrap();
        assert!(result.is_success());
        assert!(result.fun < 0.01);
    }

    #[test]
    fn supports_f32_and_rejects_bad_dimension() {
        struct QuadraticF32;
        impl ObjectiveFunction<f32> for QuadraticF32 {
            fn dim(&self) -> usize {
                2
            }
            fn eval(&self, x: &[f32]) -> Result<f32, SciOxideError> {
                Ok(x[0] * x[0] + x[1] * x[1])
            }
            fn gradient_into(&self, x: &[f32], output: &mut [f32]) -> Result<bool, SciOxideError> {
                output[0] = 2.0 * x[0];
                output[1] = 2.0 * x[1];
                Ok(true)
            }
        }
        let empty = Constraints::new(2).unwrap();
        let problem = ConstrainedProblem::new(&QuadraticF32, &empty).unwrap();
        assert!(
            PenaltyMethod::<f32>::default()
                .minimize(&problem, &[3.0, 4.0])
                .unwrap()
                .is_success()
        );
        let empty = Constraints::new(2).unwrap();
        let problem = ConstrainedProblem::new(&Quadratic, &empty).unwrap();
        assert!(
            PenaltyMethod::<f64>::default()
                .minimize(&problem, &[1.0])
                .is_err()
        );
    }
}
