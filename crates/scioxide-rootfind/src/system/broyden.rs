//! Broyden's quasi-Newton method for nonlinear systems.

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;

use super::driver::{Done, Settled, Start, drive, full_step, point_scale, settle, step_into};
use super::options::SystemRootOptions;
use crate::RootLineSearch;
use crate::context::Session;
use crate::nonlinear_math::norm;
use crate::residual::DenseJacobian;
use crate::result::{RootTermination, SystemRootResult};
use crate::system::dense::{
    gauss_newton_into, jacobian_product_into, orthogonal_product_into, triangular_product_into,
};
use crate::system::secant::{factor_secant_into, is_full_rank, secant_update_into};
use crate::system::{
    StepModel, SystemWorkspace, accept_trial, evaluate_jacobian_into, line_search_into,
    perturbed_step_into,
};

/// Selects the secant update of [`crate::Broyden`].
///
/// With step `s` and residual change `y = F(x + s) − F(x)`, both updates make
/// the Jacobian model `B` satisfy the secant equation `B s = y`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroydenUpdate {
    /// First ("good") update `B ← B + (y − B s) sᵀ / (sᵀ s)`, the least change
    /// to `B` in the Frobenius norm.
    First,
    /// Second ("bad") update `B⁻¹ ← B⁻¹ + (s − B⁻¹ y) yᵀ / (yᵀ y)`, the least
    /// change to `B⁻¹` in the Frobenius norm.
    Second,
}

/// Numerical kernel of [`crate::Broyden`]; storage lives in the workspace.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BroydenKernel {
    derivative_strategy: DerivativeStrategy,
    update: BroydenUpdate,
}

impl BroydenKernel {
    /// Creates the kernel with the given initial-Jacobian strategy.
    #[must_use]
    pub(crate) const fn with_initial_jacobian(
        update: BroydenUpdate,
        derivative_strategy: DerivativeStrategy,
    ) -> Self {
        Self {
            derivative_strategy,
            update,
        }
    }
}

impl BroydenKernel {
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
        workspace.good_broyden.reset();
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

    /// Secant iterations after an unconverged initial residual.
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
        refresh(f, workspace, self.derivative_strategy, self.update, session)?;
        let mut stale = false;
        let mut norm = start.norm;
        let iterations = opts.max_iterations.get();
        for iteration in 0..iterations {
            session.iteration()?;
            let model = secant_direction(workspace, opts, norm, session)?;
            let point_scale = point_scale(&start, workspace)?;
            if let Some(search) = opts.line_search {
                if !self.globalize(search, f, opts, workspace, session, norm, model, &mut stale)? {
                    return Ok(Done::new(norm, RootTermination::GlobalizationFailure));
                }
            } else if !full_step(f, workspace, session)? {
                return Ok(Done::new(norm, RootTermination::Stagnation));
            }
            let step_norm = step_into(workspace)?;
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
                    // The last iteration's correction could never be used.
                    if iteration + 1 < iterations
                        && secant_correction(workspace, self.update, step_norm, session)?
                    {
                        stale = true;
                    }
                    accept_trial(&mut workspace.state, session)?;
                    norm = trial_norm;
                }
            }
        }
        Ok(Done::new(norm, RootTermination::IterationLimit))
    }

    /// Line search along the model direction with the exact model slope `FᵀBd`.
    ///
    /// An exhausted search with a secant-corrected (`stale`) model re-evaluates
    /// the Jacobian and retries; an exhausted model step falls back once to the
    /// perturbed model. `false` reports unsuccessful globalization.
    #[expect(
        clippy::too_many_arguments,
        reason = "Numerical execution borrows separate caller-owned buffers and solve state."
    )]
    fn globalize<S, F>(
        &self,
        search: &dyn RootLineSearch<S>,
        f: &F,
        opts: &SystemRootOptions<'_, S>,
        workspace: &mut SystemWorkspace<S>,
        session: &Session<'_, '_, '_>,
        norm: S,
        mut model: StepModel,
        stale: &mut bool,
    ) -> Result<bool, SciOxideError>
    where
        S: RealScalar + NumericScalar<Real = S>,
        F: DenseJacobian<S> + ?Sized,
    {
        loop {
            let slope = model_slope(workspace, session)?;
            let trial = if slope.cmp_value(scioxide_core::Scaled::ZERO) != core::cmp::Ordering::Less
            {
                None
            } else {
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
            };
            match trial {
                Some(_) => return Ok(true),
                None if *stale => {
                    refresh(f, workspace, self.derivative_strategy, self.update, session)?;
                    *stale = false;
                    model = secant_direction(workspace, opts, norm, session)?;
                }
                None if model == StepModel::Newton => {
                    perturbed_direction(workspace, session)?;
                    model = StepModel::Perturbed;
                }
                None => return Ok(false),
            }
        }
    }
}

/// Evaluate a Jacobian at the accepted point and refactor the secant model.
fn refresh<S, F>(
    f: &F,
    workspace: &mut SystemWorkspace<S>,
    strategy: DerivativeStrategy,
    update: BroydenUpdate,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
{
    workspace.good_broyden.reset();
    evaluate_jacobian_into(
        f,
        &workspace.state.current_point,
        &workspace.state.current_residual,
        strategy,
        &mut workspace.jacobian,
        &mut workspace.candidate_jacobian,
        &mut workspace.state.scratch,
        workspace.finite_difference.as_mut(),
        session,
    )?;
    if update == BroydenUpdate::First {
        return workspace.good_broyden.initialize(
            &workspace.jacobian,
            &mut workspace.lu,
            &mut workspace.factor_workspace,
            session,
        );
    }
    session.admit(scioxide_core::EvaluationRole::Factorization)?;
    factor_secant_into(
        &workspace.jacobian,
        &mut workspace.secant,
        &mut workspace.secant_workspace,
        session.policy(),
        session.execution()?,
    )
}

/// Model step `Bd = −F`, or the perturbed step for a rank-deficient model under
/// a line search.
fn secant_direction<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut SystemWorkspace<S>,
    opts: &SystemRootOptions<'_, S>,
    residual_norm: S,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<StepModel, SciOxideError> {
    if workspace.good_broyden.is_active() {
        if workspace.good_broyden.direction(
            &workspace.state.current_residual,
            residual_norm,
            &mut workspace.direction,
            &mut workspace.lu,
            &mut workspace.factor_workspace,
            session,
        )? {
            return Ok(StepModel::Newton);
        }
        if opts.line_search.is_none() {
            return Err(SciOxideError::singular(None));
        }
        perturbed_direction(workspace, session)?;
        return Ok(StepModel::Perturbed);
    }
    session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
    orthogonal_product_into(
        workspace.secant.q_matrix(),
        true,
        &workspace.state.current_residual,
        &mut workspace.qtf,
    );
    if is_full_rank(workspace.secant.r_matrix()) {
        session.admit(scioxide_core::EvaluationRole::LinearSolve)?;
        gauss_newton_into(
            workspace.secant.r_matrix(),
            &workspace.qtf,
            &mut workspace.direction,
        );
        workspace
            .direction
            .as_mut_slice()
            .iter_mut()
            .for_each(|value| *value = -*value);
        return crate::residual::finite(
            workspace.direction.as_slice(),
            scioxide_core::NonFiniteOrigin::Intermediate,
        )
        .map(|()| StepModel::Newton);
    }
    if opts.line_search.is_none() {
        return Err(SciOxideError::singular(None));
    }
    perturbed_direction(workspace, session)?;
    Ok(StepModel::Perturbed)
}

/// Perturbed model step `(RᵀR + μI)d = −RᵀQᵀF`, equal to `(BᵀB + μI)d = −BᵀF`,
/// from the current `QᵀF`.
fn perturbed_direction<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut SystemWorkspace<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if workspace.good_broyden.is_active() {
        workspace.good_broyden.forget_direction();
        perturbed_step_into(
            &workspace.good_broyden.forward,
            &workspace.state.current_residual,
            &mut workspace.work,
            &mut workspace.direction,
            &mut workspace.step,
            &mut workspace.candidate_jacobian,
            &mut workspace.model_factor,
            &mut workspace.model_workspace,
            session,
        )?;
        jacobian_product_into(
            &workspace.good_broyden.forward,
            &workspace.direction,
            &mut workspace.good_broyden.product,
        )?;
        return Ok(());
    }
    perturbed_step_into(
        workspace.secant.r_matrix(),
        &workspace.qtf,
        &mut workspace.work,
        &mut workspace.direction,
        &mut workspace.step,
        &mut workspace.candidate_jacobian,
        &mut workspace.model_factor,
        &mut workspace.model_workspace,
        session,
    )
}

/// Exact model merit slope `FᵀBd = (QᵀF)ᵀ(Rd)` in extended range.
fn model_slope<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut SystemWorkspace<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<scioxide_core::Scaled<S>, SciOxideError> {
    session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
    if workspace.good_broyden.is_active() {
        return super::scaled_dot(
            workspace.state.current_residual.as_slice(),
            workspace.good_broyden.product.as_slice(),
        );
    }
    triangular_product_into(
        workspace.secant.r_matrix(),
        &workspace.direction,
        &mut workspace.product,
    );
    super::scaled_dot(workspace.qtf.as_slice(), workspace.product.as_slice())
}

/// Apply the configured secant correction for the step `s` (in `step`) and
/// `y = F(x + s) − F(x)`. Returns whether the model changed.
fn secant_correction<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut SystemWorkspace<S>,
    update: BroydenUpdate,
    step_norm: S,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<bool, SciOxideError> {
    if update == BroydenUpdate::First {
        return workspace.good_broyden.update(
            &workspace.step,
            step_norm,
            &workspace.direction,
            &workspace.state.current_residual,
            &workspace.state.trial_residual,
            session,
        );
    }
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    // `Bs = Q(Rs)` into `product`, then `u = y − Bs` into `difference`.
    session.admit(scioxide_core::EvaluationRole::LinearOperator)?;
    triangular_product_into(
        workspace.secant.r_matrix(),
        &workspace.step,
        &mut workspace.work,
    );
    orthogonal_product_into(
        workspace.secant.q_matrix(),
        false,
        &workspace.work,
        &mut workspace.product,
    );
    for (((u, &trial), &current), &model) in workspace
        .difference
        .as_mut_slice()
        .iter_mut()
        .zip(workspace.state.trial_residual.as_slice())
        .zip(workspace.state.current_residual.as_slice())
        .zip(workspace.product.as_slice())
    {
        *u = (trial - current) - model;
    }
    let (denominator, threshold, row_norm) = {
        // `Bᵀy = Rᵀ(Qᵀy)`, with `y` formed in `work`.
        for ((y, &trial), &current) in workspace
            .work
            .as_mut_slice()
            .iter_mut()
            .zip(workspace.state.trial_residual.as_slice())
            .zip(workspace.state.current_residual.as_slice())
        {
            *y = trial - current;
        }
        orthogonal_product_into(
            workspace.secant.q_matrix(),
            true,
            &workspace.work,
            &mut workspace.qtf,
        );
        super::dense::upper_transpose_into(
            workspace.secant.r_matrix().as_slice(),
            workspace.qtf.as_slice(),
            None,
            workspace.gradient.as_mut_slice(),
        );
        let denominator =
            super::scaled_dot(workspace.gradient.as_slice(), workspace.step.as_slice())?.round()?;
        let row_norm = norm(workspace.gradient.as_slice())?;
        (
            denominator,
            S::SQRT_EPSILON * row_norm * step_norm,
            row_norm,
        )
    };

    if denominator.abs() <= threshold || denominator == zero || !denominator.is_finite() {
        return Ok(false);
    }
    // Balance `u vᵀ` with `‖u‖ = ‖v‖` before the rotation update.
    let left = norm(workspace.difference.as_slice())?;
    let right = row_norm / denominator.abs();
    if left == zero || right == zero {
        return Ok(false);
    }
    let balance = (right / left).sqrt();
    workspace
        .difference
        .as_mut_slice()
        .iter_mut()
        .for_each(|value| *value *= balance);
    workspace
        .gradient
        .as_mut_slice()
        .iter_mut()
        .for_each(|value| *value = *value / denominator / balance);
    secant_update_into(
        &mut workspace.secant,
        &mut workspace.secant_workspace,
        &workspace.difference,
        &workspace.gradient,
        session.policy(),
        session.execution()?,
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Broyden, SystemRootSolver};

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
    impl crate::DenseJacobian<f64> for TwoDSystem {}

    #[test]
    fn test_broyden_2d() {
        let solver = Broyden::new(
            crate::BroydenUpdate::First,
            (crate::BroydenOptions::default())
                .with_derivative_strategy(DerivativeStrategy::AnalyticOrForward),
        )
        .unwrap();

        let result = solver.solve(&TwoDSystem, &[0.5, 0.5]).unwrap();
        assert!(result.is_success());
        assert!(result.residual_norm().unwrap() < 1e-6);
    }

    #[test]
    fn test_broyden_name() {
        let solver = Broyden::new(
            crate::BroydenUpdate::First,
            (crate::BroydenOptions::default())
                .with_derivative_strategy(DerivativeStrategy::AnalyticOrForward),
        )
        .unwrap();
        assert_eq!(SystemRootSolver::<f64>::name(&solver), "Broyden");
    }

    #[test]
    fn test_broyden_rejects_dimension_mismatch() {
        let solver = Broyden::new(
            crate::BroydenUpdate::First,
            crate::BroydenOptions::default(),
        )
        .unwrap();
        let result = solver.solve(&TwoDSystem, &[0.5]);
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }
}
