//! Newton's method for nonlinear systems.

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;

use super::dense::jacobian_product_into;
use super::driver::{Done, Settled, Start, drive, full_step, point_scale, settle, step_into};
use super::options::SystemRootOptions;
use crate::RootLineSearch;
use crate::context::Session;
use crate::residual::DenseJacobian;
use crate::result::{RootTermination, SystemRootResult};
use crate::system::{
    StepModel, SystemWorkspace, accept_trial, evaluate_jacobian_into, line_search_direction_into,
    line_search_into, perturbed_step_into, scaled_dot, solve_linear_into,
};

/// Numerical kernel of [`crate::NewtonSystem`]; storage lives in the workspace.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NewtonSystemKernel {
    derivative_strategy: DerivativeStrategy,
}

impl NewtonSystemKernel {
    /// Creates the kernel with the given Jacobian strategy.
    #[must_use]
    pub(crate) const fn new(derivative_strategy: DerivativeStrategy) -> Self {
        Self {
            derivative_strategy,
        }
    }
}

impl NewtonSystemKernel {
    pub(crate) fn solve_with_context<S, F>(
        &self,
        f: &F,
        initial: &[S],
        opts: &SystemRootOptions<'_, S>,
        result: &mut SystemRootResult<S>,
        workspace: &mut SystemWorkspace<S>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        F: DenseJacobian<S> + ?Sized,
    {
        drive(
            f,
            initial,
            opts,
            result,
            workspace,
            context,
            |workspace, session, start| self.iterate(f, opts, workspace, session, start),
        )
    }

    /// Newton iterations after an unconverged initial residual.
    fn iterate<S, F>(
        &self,
        f: &F,
        opts: &SystemRootOptions<'_, S>,
        workspace: &mut SystemWorkspace<S>,
        session: &Session<'_, '_, '_>,
        start: Start<S>,
    ) -> Result<Done<S>, SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        F: DenseJacobian<S> + ?Sized,
    {
        let mut norm = start.norm;
        for _ in 0..opts.max_iterations.get() {
            session.iteration()?;
            let model = self.direction(f, opts, workspace, session)?;
            let point_scale = point_scale(&start, workspace)?;
            if let Some(search) = opts.line_search {
                if !globalize(search, f, opts, workspace, session, model)? {
                    return Ok(Done::new(norm, RootTermination::GlobalizationFailure));
                }
            } else if !full_step(f, workspace, session)? {
                return Ok(Done::new(norm, RootTermination::Stagnation));
            }
            let step_norm = if start.step_test {
                step_into(workspace)?
            } else {
                S::ZERO
            };
            match settle(
                opts,
                &start,
                workspace,
                session,
                norm,
                point_scale,
                step_norm,
            )? {
                Settled::Done(done) => return Ok(done),
                Settled::Continue(trial_norm) => {
                    accept_trial(&mut workspace.state, session)?;
                    norm = trial_norm;
                }
            }
        }
        Ok(Done::new(norm, RootTermination::IterationLimit))
    }

    /// Evaluate the Jacobian at the accepted point and solve for the step direction.
    fn direction<S, F>(
        &self,
        f: &F,
        opts: &SystemRootOptions<'_, S>,
        workspace: &mut SystemWorkspace<S>,
        session: &Session<'_, '_, '_>,
    ) -> Result<StepModel, SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        F: DenseJacobian<S> + ?Sized,
    {
        evaluate_jacobian_into(
            f,
            &workspace.state.current_point,
            &workspace.state.current_residual,
            self.derivative_strategy,
            &mut workspace.jacobian,
            &mut workspace.candidate_jacobian,
            &mut workspace.state.scratch,
            workspace.finite_difference.as_mut(),
            session,
        )?;
        if opts.line_search.is_some() {
            line_search_direction_into(
                opts,
                &workspace.jacobian,
                &workspace.state.current_residual,
                &mut workspace.work,
                &mut workspace.direction,
                &mut workspace.step,
                &mut workspace.candidate_jacobian,
                &mut workspace.lu,
                &mut workspace.factor_workspace,
                &mut workspace.linear,
                session,
            )
        } else {
            solve_linear_into(
                opts,
                &workspace.jacobian,
                &workspace.state.current_residual,
                &mut workspace.work,
                &mut workspace.direction,
                &mut workspace.step,
                &mut workspace.lu,
                &mut workspace.factor_workspace,
                &mut workspace.linear,
                session,
            )?;
            Ok(StepModel::Newton)
        }
    }
}

/// Line search along the model direction with the exact merit slope `FᵀJd`.
///
/// A direction without strict merit descent counts as an exhausted search. An
/// exhausted Newton direction falls back once to the perturbed model at the
/// same point; `false` reports unsuccessful globalization.
fn globalize<S, F>(
    search: &dyn RootLineSearch<S>,
    f: &F,
    opts: &SystemRootOptions<'_, S>,
    workspace: &mut SystemWorkspace<S>,
    session: &Session<'_, '_, '_>,
    mut model: StepModel,
) -> Result<bool, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
{
    loop {
        session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
        jacobian_product_into(
            &workspace.jacobian,
            &workspace.direction,
            &mut workspace.work,
        )?;
        let slope = scaled_dot(
            workspace.state.current_residual.as_slice(),
            workspace.work.as_slice(),
        )?;
        // A model step without merit descent (for example at a stationary
        // non-root, where `JᵀF = 0`) is an exhausted search, as for Broyden.
        let alpha = if slope.cmp_value(scioxide_core::Scaled::ZERO) == core::cmp::Ordering::Less {
            line_search_into(
                search,
                f,
                &workspace.state.current_point,
                &workspace.direction,
                &workspace.state.current_residual,
                slope,
                &mut workspace.state.trial_point,
                &mut workspace.state.trial_residual,
                &mut workspace.line_search_point,
                &mut workspace.line_search_residual,
                &mut workspace.state.scratch,
                session,
            )?
        } else {
            None
        };
        match (alpha, model) {
            (Some(_), _) => return Ok(true),
            (None, StepModel::Newton) if opts.linear_solver.is_none() => {
                perturbed_step_into(
                    &workspace.jacobian,
                    &workspace.state.current_residual,
                    &mut workspace.work,
                    &mut workspace.direction,
                    &mut workspace.step,
                    &mut workspace.candidate_jacobian,
                    &mut workspace.lu,
                    &mut workspace.factor_workspace,
                    session,
                )?;
                model = StepModel::Perturbed;
            }
            (None, _) => return Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NewtonSystem, NewtonSystemOptions, SystemRootSolver};

    // F(x, y) = (x^2 + y - 1, x + y^2 - 1)
    // Roots: (0, 1) and (1, 0)
    struct TwoDSystem;
    impl scioxide_core::ResidualOperator<f64> for TwoDSystem {
        fn input_dimension(&self) -> usize {
            2
        }
        fn output_dimension(&self) -> usize {
            2
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<(), SciOxideError> {
            let x0 = x[0];
            let x1 = x[1];
            out[0] = x0.mul_add(x0, x1 - 1.0);
            out[1] = x1.mul_add(x1, x0 - 1.0);
            Ok(())
        }
    }
    impl crate::DenseJacobian<f64> for TwoDSystem {
        fn jacobian_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<bool, SciOxideError> {
            let x0 = x[0];
            let x1 = x[1];
            out.copy_from_slice(&[2.0 * x0, 1.0, 1.0, 2.0 * x1]);
            Ok(true)
        }
    }

    // Same system without Jacobian (to test finite-difference)
    struct TwoDSystemNoJac;
    impl scioxide_core::ResidualOperator<f64> for TwoDSystemNoJac {
        fn input_dimension(&self) -> usize {
            2
        }
        fn output_dimension(&self) -> usize {
            2
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<(), SciOxideError> {
            let x0 = x[0];
            let x1 = x[1];
            out[0] = x0.mul_add(x0, x1 - 1.0);
            out[1] = x1.mul_add(x1, x0 - 1.0);
            Ok(())
        }
    }
    impl crate::DenseJacobian<f64> for TwoDSystemNoJac {}

    struct FailingJacobian;

    impl scioxide_core::ResidualOperator<f64> for FailingJacobian {
        fn input_dimension(&self) -> usize {
            1
        }
        fn output_dimension(&self) -> usize {
            1
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = x[0] - 2.0;
            Ok(())
        }
    }
    impl crate::DenseJacobian<f64> for FailingJacobian {
        fn jacobian_into(
            &self,
            _x: &[f64],
            _out: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<bool, SciOxideError> {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("jacobian_callback_sentinel"),
                scioxide_core::InvalidInputReason::Malformed,
                "intentional analytic failure",
            ))
        }
    }

    #[test]
    fn test_newton_system_2d_analytic_jac() {
        let solver = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap();

        let result = solver.solve(&TwoDSystem, &[0.1, 0.9]).unwrap();
        assert!(result.is_success());
        assert!(result.residual_norm().unwrap() < 1e-6);
    }

    #[test]
    fn test_newton_system_2d_finite_diff() {
        let solver = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Forward),
        )
        .unwrap();

        let result = solver.solve(&TwoDSystemNoJac, &[0.5, 0.5]).unwrap();
        assert!(result.is_success());
    }

    #[test]
    fn test_newton_system_name() {
        let solver = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap();
        assert_eq!(SystemRootSolver::<f64>::name(&solver), "NewtonSystem");
    }

    #[test]
    fn test_newton_system_max_iter() {
        let solver = NewtonSystem::new(
            (NewtonSystemOptions::default()
                .with_residual_tolerance(
                    scioxide_core::ResidualTolerance::new(1e-15, 1e-15).unwrap(),
                )
                .with_step_tolerance(scioxide_core::StepTolerance::new(1e-15, 1e-15).unwrap())
                .with_max_iterations(scioxide_core::Budget::ONE))
            .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap();

        let result = solver.solve(&TwoDSystem, &[0.1, 0.9]).unwrap();
        // May or may not converge in 1 iteration
        assert!(result.iterations() <= 1);
    }

    #[test]
    fn test_newton_system_rejects_dimension_mismatch() {
        let solver = NewtonSystem::new(NewtonSystemOptions::default()).unwrap();
        let result = solver.solve(&TwoDSystem, &[0.5]);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }

    #[test]
    fn test_newton_system_rejects_non_square_jacobian() {
        struct BadJacSystem;
        impl scioxide_core::ResidualOperator<f64> for BadJacSystem {
            fn input_dimension(&self) -> usize {
                2
            }
            fn output_dimension(&self) -> usize {
                2
            }
            fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(0)
            }
            fn residual_into(
                &self,
                _x: &[f64],
                out: &mut [f64],
                _scratch: &mut [f64],
            ) -> Result<(), SciOxideError> {
                out.copy_from_slice(&[1.0, -1.0]);
                Ok(())
            }
        }
        impl crate::DenseJacobian<f64> for BadJacSystem {
            fn jacobian_into(
                &self,
                _x: &[f64],
                _out: &mut [f64],
                _scratch: &mut [f64],
            ) -> Result<bool, SciOxideError> {
                Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::two(2, 2),
                    scioxide_core::Dimensions::two(2, 1),
                ))
            }
        }

        let solver = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap();

        let result = solver.solve(&BadJacSystem, &[0.0, 0.0]);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }

    #[test]
    fn test_newton_system_propagates_singular_jacobian() {
        struct SingularSystem;
        impl scioxide_core::ResidualOperator<f64> for SingularSystem {
            fn input_dimension(&self) -> usize {
                2
            }
            fn output_dimension(&self) -> usize {
                2
            }
            fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(0)
            }
            fn residual_into(
                &self,
                _x: &[f64],
                out: &mut [f64],
                _scratch: &mut [f64],
            ) -> Result<(), SciOxideError> {
                out.fill(1.0);
                Ok(())
            }
        }
        impl crate::DenseJacobian<f64> for SingularSystem {
            fn jacobian_into(
                &self,
                _x: &[f64],
                out: &mut [f64],
                _scratch: &mut [f64],
            ) -> Result<bool, SciOxideError> {
                out.copy_from_slice(&[1.0, 2.0, 2.0, 4.0]);
                Ok(true)
            }
        }

        let solver = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap();

        let result = solver.solve(&SingularSystem, &[0.0, 0.0]);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::Singular
        );
    }

    #[test]
    fn test_newton_system_rejects_non_finite_initial_guess() {
        let solver = NewtonSystem::new(NewtonSystemOptions::default()).unwrap();
        let result = solver.solve(&TwoDSystemNoJac, &[f64::NAN, 0.0]);
        assert!(matches!(
            result.unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_newton_system_rejects_zero_dimension_system() {
        struct ZeroDimSystem;
        impl scioxide_core::ResidualOperator<f64> for ZeroDimSystem {
            fn input_dimension(&self) -> usize {
                0
            }
            fn output_dimension(&self) -> usize {
                0
            }
            fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(0)
            }
            fn residual_into(
                &self,
                _x: &[f64],
                _out: &mut [f64],
                _scratch: &mut [f64],
            ) -> Result<(), SciOxideError> {
                Ok(())
            }
        }
        impl crate::DenseJacobian<f64> for ZeroDimSystem {}

        let solver = NewtonSystem::new(NewtonSystemOptions::default()).unwrap();
        let result = solver.solve(&ZeroDimSystem, &[]);
        assert!(matches!(
            result.unwrap_err().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("function")
        ));
    }

    #[test]
    fn test_newton_system_analytic_requires_jacobian() {
        let solver = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap();
        let result = solver.solve(&TwoDSystemNoJac, &[0.5, 0.5]);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }

    #[test]
    fn test_newton_system_default_strategy_uses_finite_difference() {
        let solver = NewtonSystem::new(NewtonSystemOptions::default()).unwrap();
        let result = solver.solve(&TwoDSystemNoJac, &[0.5, 0.5]).unwrap();
        assert!(result.is_success());
    }

    #[test]
    fn test_newton_system_finite_difference_skips_analytic_callback() {
        let result = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Forward),
        )
        .unwrap()
        .solve(&FailingJacobian, &[0.0])
        .unwrap();
        assert!(result.is_success());
    }

    #[test]
    fn test_newton_system_fallback_propagates_analytic_error() {
        let result = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::AnalyticOrForward),
        )
        .unwrap()
        .solve(&FailingJacobian, &[0.0]);
        let error = result.unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("jacobian_callback_sentinel")
        ));
    }

    #[test]
    fn test_newton_system_finite_difference_f32() {
        struct LinearF32;

        impl scioxide_core::ResidualOperator<f32> for LinearF32 {
            fn input_dimension(&self) -> usize {
                1
            }
            fn output_dimension(&self) -> usize {
                1
            }
            fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(0)
            }
            fn residual_into(
                &self,
                x: &[f32],
                out: &mut [f32],
                _scratch: &mut [f32],
            ) -> Result<(), SciOxideError> {
                out[0] = x[0] - 2.0;
                Ok(())
            }
        }
        impl crate::DenseJacobian<f32> for LinearF32 {}

        let result = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Forward),
        )
        .unwrap()
        .solve(&LinearF32, &[0.0])
        .unwrap();
        assert!(result.is_success());
        assert!((result.root.view().get(0).unwrap() - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_newton_system_rejects_malformed_callback_output() {
        struct MalformedOutput;
        impl scioxide_core::ResidualOperator<f64> for MalformedOutput {
            fn input_dimension(&self) -> usize {
                2
            }
            fn output_dimension(&self) -> usize {
                2
            }
            fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(0)
            }
            fn residual_into(
                &self,
                _x: &[f64],
                _out: &mut [f64],
                _scratch: &mut [f64],
            ) -> Result<(), SciOxideError> {
                Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::one(2),
                    scioxide_core::Dimensions::one(1),
                ))
            }
        }
        impl crate::DenseJacobian<f64> for MalformedOutput {}

        let result = NewtonSystem::new(
            (crate::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Forward),
        )
        .unwrap()
        .solve(&MalformedOutput, &[0.0, 0.0]);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }
}
