//! Augmented Lagrangian method.

super::outer::outer_method!(
    /// Minimizes a nonlinear program by the augmented Lagrangian method
    /// (method of multipliers).
    ///
    /// Each outer iteration minimizes the Powell–Hestenes–Rockafellar augmented
    /// Lagrangian
    ///
    /// ```text
    /// L(x; λ, μ) = f(x) + Σᵢ (λᵢ hᵢ(x) + (μ/2) hᵢ(x)²)
    ///                   + (1/(2μ)) Σⱼ (max(λⱼ + μ gⱼ(x), 0)² − λⱼ²)
    /// ```
    ///
    /// with [`Bfgs`](crate::Bfgs), starting from the previous iterate, and then
    /// updates the multipliers to `λᵢ + μ hᵢ(x)` and `max(λⱼ + μ gⱼ(x), 0)`.
    /// Here `hᵢ` are the equality rows and `gⱼ` the inequality rows, scaled as
    /// described in [Termination](crate#termination). The penalty parameter
    /// starts at `μ = 1` and is multiplied by 10 when the constraint violation
    /// does not decrease enough. Bounds are handled as in
    /// [`PenaltyMethod`](crate::PenaltyMethod): each finite bound side is an
    /// inequality row.
    ///
    /// With the multiplier estimates, `μ` need not grow without bound, so the
    /// subproblems stay better conditioned than those of the quadratic penalty
    /// method.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_optimize::{
    ///     AugmentedLagrangian, ConstrainedProblem, ConstraintFn, Constraints, ObjectiveFn,
    /// };
    ///
    /// // minimize x² + y² subject to x + y ≥ 1; the minimum is 1/2
    /// let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
    /// let constraints = Constraints::new(2)?
    ///     .inequality(ConstraintFn::new(2, |x: &[f64]| Ok(1.0 - x[0] - x[1])))?;
    /// let problem = ConstrainedProblem::new(&objective, &constraints)?;
    /// let result = AugmentedLagrangian::default().minimize(&problem, &[2.0, 2.0])?;
    /// assert!((result.fun() - 0.5).abs() < 0.05);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.,
    ///   Springer, 2006, §17.3–17.4.
    /// - R. T. Rockafellar, "The multiplier method of Hestenes and Powell
    ///   applied to convex programming", *J. Optim. Theory Appl.* 12(6),
    ///   555–562, 1973.
    AugmentedLagrangian,
    AugmentedLagrangianOptions,
    AugmentedLagrangianWorkspace,
    AugmentedLagrangianWorkspaceRequirements,
    "AugmentedLagrangian",
    true
);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{ConstrainedProblem, ConstraintFn, Constraints, ObjectiveFunction, OptimizeResult};
    use scioxide_core::{ExecutionContext, SciOxideError};

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
    fn prepared_equality_and_inequality_paths_converge() {
        let equality = ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0));
        let constraints = Constraints::new(2).unwrap().equality(equality).unwrap();
        let problem = ConstrainedProblem::new(&Quadratic, &constraints).unwrap();
        let solver = AugmentedLagrangian::<f64>::default();
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

        let inequality = ConstraintFn::new(2, |x: &[f64]| Ok(1.0 - x[0] - x[1]));
        let constraints = Constraints::new(2).unwrap().inequality(inequality).unwrap();
        let problem = ConstrainedProblem::new(&Quadratic, &constraints).unwrap();
        let output = solver.minimize(&problem, &[2.0, 2.0]).unwrap();
        assert!(output.is_success(), "{}", output.message);
        assert!((output.fun - 0.5).abs() < 0.05);
    }

    #[test]
    fn no_constraints_f32_and_dimension_validation() {
        let empty = Constraints::new(2).unwrap();
        let problem = ConstrainedProblem::new(&Quadratic, &empty).unwrap();
        assert!(
            AugmentedLagrangian::<f64>::default()
                .minimize(&problem, &[3.0, 4.0])
                .unwrap()
                .is_success()
        );
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
        let empty32 = Constraints::new(2).unwrap();
        let problem32 = ConstrainedProblem::new(&QuadraticF32, &empty32).unwrap();
        assert!(
            AugmentedLagrangian::<f32>::default()
                .minimize(&problem32, &[3.0, 4.0])
                .unwrap()
                .is_success()
        );
        assert!(
            AugmentedLagrangian::<f64>::default()
                .minimize(&problem, &[1.0])
                .is_err()
        );
    }
}
