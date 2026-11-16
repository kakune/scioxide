//! Dense solvers for square nonlinear systems `F(x) = 0`.
//!
//! - [`NewtonSystem`]: Newton's method with an LU or iterative linear solve.
//! - [`Broyden`]: Broyden's first or second quasi-Newton update.
//! - [`TrustRegionSystem`]: Powell's hybrid dogleg trust-region method.
//!
//! [`BacktrackingLineSearch`] is the Armijo backtracking line search used to
//! globalize Newton and Broyden steps.

mod api;
mod broyden;
mod broyden_model;
mod dense;
mod driver;
mod line_search;
mod newton_system;
mod options;
pub(crate) mod secant;
mod trust_region;
mod workspace;

pub use api::{Broyden, NewtonSystem, TrustRegionSystem, solve_system};
pub use broyden::BroydenUpdate;
pub use line_search::BacktrackingLineSearch;
pub use options::{BroydenOptions, NewtonSystemOptions, TrustRegionSystemOptions};
use scioxide_linalg::Preconditioning;

use workspace::SystemState;
pub use workspace::{SystemWorkspace, SystemWorkspaceRequirements};

#[cfg(test)]
use alloc::vec;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{Matrix, Vector};

use crate::residual::DenseJacobian;
use crate::result::{RootTermination, SystemRootResult};
use crate::traits::RootLineSearch;
use options::SystemRootOptions;

/// Initializes numerical scratch after the input checks have passed.
pub(crate) fn prepare_solve<S: RealScalar + NumericScalar<Real = S>>(
    initial: &[S],
    workspace: &mut SystemState<S>,
) {
    workspace.accepted = None;
    workspace
        .current_point
        .as_mut_slice()
        .copy_from_slice(initial);
}

/// Evaluates and checks the residual, writing to `output` only on success.
pub(crate) fn evaluate_system_into<
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
>(
    function: &F,
    point: &Vector<S>,
    output: &mut Vector<S>,
    candidate: &mut Vector<S>,
    scratch: &mut [S],
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    validate_vector(output, function.output_dimension(), false)?;
    crate::residual::evaluate(
        function,
        scioxide_core::EvaluationRole::Residual,
        point.as_slice(),
        candidate.as_mut_slice(),
        scratch,
        session,
    )?;
    core::mem::swap(output, candidate);
    Ok(())
}

/// Evaluate a globalized trial, staging output until it is finite.
pub(crate) fn evaluate_trial_into<
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
>(
    function: &F,
    point: &Vector<S>,
    output: &mut Vector<S>,
    candidate: &mut Vector<S>,
    scratch: &mut [S],
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<crate::residual::TrialStatus, SciOxideError> {
    let evaluation = crate::residual::evaluate_raw(
        function,
        scioxide_core::EvaluationRole::Residual,
        point.as_slice(),
        candidate.as_mut_slice(),
        scratch,
        session,
    );
    let status = crate::residual::trial_status(evaluation, candidate.as_slice())?;
    if status == crate::residual::TrialStatus::Finite {
        core::mem::swap(output, candidate);
    }
    Ok(status)
}

/// Evaluates and checks the Jacobian, writing to `jacobian` only on success.
#[expect(
    clippy::too_many_arguments,
    reason = "Numerical execution borrows separate caller-owned buffers and solve state."
)]
pub(crate) fn evaluate_jacobian_into<
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
>(
    function: &F,
    point: &Vector<S>,
    base: &Vector<S>,
    strategy: DerivativeStrategy,
    jacobian: &mut Matrix<S>,
    candidate: &mut Matrix<S>,
    scratch: &mut [S],
    finite_difference: Option<&mut scioxide_differentiate::FiniteDifferenceWorkspace<S>>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    crate::residual::jacobian_into(
        function,
        point.as_slice(),
        base.as_slice(),
        strategy,
        candidate.as_mut_slice(),
        scratch,
        finite_difference,
        session,
    )?;
    core::mem::swap(jacobian, candidate);
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Numerical execution borrows separate caller-owned buffers and solve state."
)]
pub(crate) fn solve_linear_into<S: RealScalar + NumericScalar<Real = S>>(
    options: &SystemRootOptions<'_, S>,
    jacobian: &Matrix<S>,
    residual: &Vector<S>,
    right_hand_side: &mut Vector<S>,
    solution: &mut Vector<S>,
    solve_scratch: &mut Vector<S>,
    factor: &mut scioxide_linalg::LuDecomposition<S>,
    factor_workspace: &mut scioxide_linalg::LinalgWorkspace<'static, S>,
    linear_workspace: &mut scioxide_linalg::IterativeWorkspace<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    right_hand_side
        .as_mut_slice()
        .iter_mut()
        .zip(residual.as_slice())
        .for_each(|(target, &value)| *target = -value);
    if let Some(iterative) = options.linear_solver.as_ref() {
        solve_iterative_into(
            *iterative,
            jacobian,
            right_hand_side,
            solution,
            linear_workspace,
            session,
        )?;
    } else {
        factor_into(jacobian, factor, factor_workspace, session)?;
        factored_solve_into(factor, right_hand_side, solution, solve_scratch, session)?;
    }

    Ok(())
}

/// Counted sequential LU factorization.
fn factor_into<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    factor: &mut scioxide_linalg::LuDecomposition<S>,
    factor_workspace: &mut scioxide_linalg::LinalgWorkspace<'static, S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    session.admit(scioxide_core::EvaluationRole::Factorization)?;
    scioxide_linalg::LuDecomposition::factor_into(
        matrix,
        &scioxide_linalg::LinalgExecutionOptions::new().with_execution_policy(session.policy()),
        factor,
        factor_workspace,
        session.execution()?,
    )
}

/// Counted solve with an exactly nonsingular LU factor.
fn factored_solve_into<S: RealScalar + NumericScalar<Real = S>>(
    factor: &scioxide_linalg::LuDecomposition<S>,
    right_hand_side: &Vector<S>,
    solution: &mut Vector<S>,
    scratch: &mut Vector<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    session.invoke_context(scioxide_core::EvaluationRole::LinearSolve, |context| {
        scioxide_core::LinearOperator::apply_into_scoped(
            &factor.inverse_operator(),
            right_hand_side.as_slice(),
            solution.as_mut_slice(),
            scratch.as_mut_slice(),
            &mut scioxide_core::OperatorContext::new(context),
        )
        .and_then(scioxide_core::Invocation::require)
    })
}

/// Line-search Newton direction.
///
/// A Jacobian whose LU factor is full rank under the dimension-scaled `ε`
/// policy gives the Newton step `J d = −F`. A numerically rank-deficient Jacobian
/// gives the perturbed model step `(JᵀJ + μI) d = −JᵀF` with `μ = √(nε)‖JᵀJ‖₁`
/// (Dennis–Schnabel, Algorithm A6.5.1), a descent direction for `½‖F‖²` that stays
/// defined when `J` is singular. Ill-conditioned but full-rank Jacobians keep the
/// Newton step, preserving fast local convergence at ill-conditioned roots. The
/// model is formed from `J` scaled by its largest magnitude, so `JᵀJ` cannot
/// overflow.
/// An injected iterative linear solver always solves the Newton system.
#[expect(
    clippy::too_many_arguments,
    reason = "Numerical execution borrows separate caller-owned buffers and solve state."
)]
pub(crate) fn line_search_direction_into<S: RealScalar + NumericScalar<Real = S>>(
    options: &SystemRootOptions<'_, S>,
    jacobian: &Matrix<S>,
    residual: &Vector<S>,
    right_hand_side: &mut Vector<S>,
    direction: &mut Vector<S>,
    solve_scratch: &mut Vector<S>,
    model: &mut Matrix<S>,
    factor: &mut scioxide_linalg::LuDecomposition<S>,
    factor_workspace: &mut scioxide_linalg::LinalgWorkspace<'static, S>,
    linear_workspace: &mut scioxide_linalg::IterativeWorkspace<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<StepModel, SciOxideError> {
    if options.linear_solver.is_some() {
        solve_linear_into(
            options,
            jacobian,
            residual,
            right_hand_side,
            direction,
            solve_scratch,
            factor,
            factor_workspace,
            linear_workspace,
            session,
        )?;
        return Ok(StepModel::Newton);
    }
    let full_rank = match factor_into(jacobian, factor, factor_workspace, session) {
        Ok(()) => factor
            .analyze_rank(scioxide_linalg::RankPolicy::DimensionScaledEpsilon)?
            .is_full_rank(),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::Singular { .. }
            ) =>
        {
            false
        }
        Err(error) => return Err(error),
    };
    if full_rank {
        right_hand_side
            .as_mut_slice()
            .iter_mut()
            .zip(residual.as_slice())
            .for_each(|(target, &value)| *target = -value);
        factored_solve_into(factor, right_hand_side, direction, solve_scratch, session)?;
        return Ok(StepModel::Newton);
    }
    perturbed_step_into(
        jacobian,
        residual,
        right_hand_side,
        direction,
        solve_scratch,
        model,
        factor,
        factor_workspace,
        session,
    )?;
    Ok(StepModel::Perturbed)
}

/// Linear model behind a line-search direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepModel {
    /// The model's Newton step.
    Newton,
    /// The Dennis–Schnabel perturbed model step.
    Perturbed,
}

/// Dennis–Schnabel perturbed model step (Algorithm A6.5.1) minimizing
/// `‖A d + b‖² + μ‖d‖²`: `(AᵀA + μI) d = −Aᵀb` with `μ = √(nε)‖AᵀA‖₁`.
///
/// It is a descent direction for `½‖Ad + b‖²` at `d = 0` that stays defined when
/// `A` is singular. The normal equations are formed from `A` scaled by its largest
/// magnitude, so `AᵀA` cannot overflow.
#[expect(
    clippy::too_many_arguments,
    reason = "Numerical execution borrows separate caller-owned buffers and solve state."
)]
pub(crate) fn perturbed_step_into<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    residual: &Vector<S>,
    right_hand_side: &mut Vector<S>,
    direction: &mut Vector<S>,
    solve_scratch: &mut Vector<S>,
    model: &mut Matrix<S>,
    factor: &mut scioxide_linalg::LuDecomposition<S>,
    factor_workspace: &mut scioxide_linalg::LinalgWorkspace<'static, S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let n = residual.len();
    let scale = matrix
        .as_slice()
        .iter()
        .fold(zero, |largest, &value| largest.max(value.abs()));
    if scale == zero {
        return Err(SciOxideError::singular(None));
    }
    dense::normal_equations_into(matrix, residual, scale, right_hand_side, model);
    let norm = model.as_slice().chunks_exact(n).fold(zero, |largest, row| {
        largest.max(row.iter().fold(zero, |sum, &value| sum + value.abs()))
    });
    let shift = (S::try_from_usize_exact(n).unwrap_or(S::MAX) * S::EPSILON).sqrt() * norm;

    for index in 0..n {
        model.as_mut_slice()[index * n + index] = model.as_slice()[index * n + index] + shift;
    }
    crate::residual::finite(
        model.as_slice(),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )?;
    factor_into(model, factor, factor_workspace, session)?;
    factored_solve_into(factor, right_hand_side, direction, solve_scratch, session)?;
    direction
        .as_mut_slice()
        .iter_mut()
        .for_each(|value| *value /= scale);
    crate::residual::finite(
        direction.as_slice(),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )
}

/// Runs the line search on the counted view of `function`. The trial point
/// and residual change only when a step is accepted.
#[expect(
    clippy::too_many_arguments,
    reason = "Numerical execution borrows separate caller-owned buffers and solve state."
)]
pub(crate) fn line_search_into<
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized,
>(
    line_search: &dyn RootLineSearch<S>,
    function: &F,
    point: &Vector<S>,
    direction: &Vector<S>,
    residual: &Vector<S>,
    merit_slope: scioxide_core::Scaled<S>,
    trial_point: &mut Vector<S>,
    trial_residual: &mut Vector<S>,
    candidate_point: &mut Vector<S>,
    candidate_residual: &mut Vector<S>,
    scratch: &mut [S],
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<Option<S>, SciOxideError> {
    let admitted = crate::residual::Admitted::new(function, session);
    let result = line_search.search_into(
        &admitted,
        point.as_slice(),
        direction.as_slice(),
        residual.as_slice(),
        merit_slope,
        candidate_point.as_mut_slice(),
        candidate_residual.as_mut_slice(),
        scratch,
    );
    let result = result?;
    admitted.finish(result.is_some())?;
    let Some(alpha) = result else {
        return Ok(None);
    };
    if !alpha.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Callback,
            "line search returned a non-finite step length",
        ));
    }
    if alpha <= <S as scioxide_core::AlgebraicScalar>::ZERO {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("opts.line_search"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "line search must return a positive step length",
        ));
    }
    for candidate in [&*candidate_point, &*candidate_residual] {
        validate_vector(candidate, function.input_dimension(), false)?;
        crate::residual::finite(
            candidate.as_slice(),
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
    }
    core::mem::swap(trial_point, candidate_point);
    core::mem::swap(trial_residual, candidate_residual);
    Ok(Some(alpha))
}

pub(crate) fn accept_trial<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut SystemState<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    session.accept()?;
    core::mem::swap(&mut workspace.current_point, &mut workspace.trial_point);
    core::mem::swap(
        &mut workspace.current_residual,
        &mut workspace.trial_residual,
    );
    Ok(())
}

pub(crate) fn finish_solve<S: RealScalar + NumericScalar<Real = S>>(
    result: &mut SystemRootResult<S>,
    workspace: &mut SystemState<S>,
    residual_norm: S,
    termination: RootTermination,
    session: &crate::context::Session<'_, '_, '_>,
    scale: S,
) -> Result<(), SciOxideError> {
    if session.stopped.get().is_none() {
        let _ = session.check_stop()?;
    }
    let termination = session.stopped.get().unwrap_or(termination);
    let residual = crate::context::residual(residual_norm, scale)?;
    let counts = session.accounting()?;
    let algorithm = session.algorithm()?;
    session.assess()?;
    result
        .root
        .as_mut_slice()
        .copy_from_slice(workspace.current_point.as_slice());
    result
        .function_value
        .as_mut_slice()
        .copy_from_slice(workspace.current_residual.as_slice());
    workspace.accepted = Some(residual);
    result.residual = Some(residual);
    result.termination = termination;
    result.counts = counts;
    result.algorithm = Some(algorithm);
    Ok(())
}

/// Require `vector` to hold `dimension` components, all finite when
/// `require_finite`.
fn validate_vector<S: RealScalar + NumericScalar<Real = S>>(
    vector: &Vector<S>,
    dimension: usize,
    require_finite: bool,
) -> Result<(), SciOxideError> {
    if vector.len() != dimension {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(dimension),
            scioxide_core::Dimensions::one(vector.len()),
        ));
    }
    if require_finite {
        crate::residual::finite(vector.as_slice(), scioxide_core::NonFiniteOrigin::Callback)?;
    }
    Ok(())
}

/// Scaled residual test.
pub(crate) use crate::termination::residual_ok;

/// Dot product whose intermediate exponent range is independent of scalar range.
pub(crate) fn scaled_dot<S: RealScalar + NumericScalar<Real = S>>(
    left: &[S],
    right: &[S],
) -> Result<scioxide_core::Scaled<S>, SciOxideError> {
    crate::nonlinear_math::dot(left, right).map(|value| value.real())
}

pub(crate) fn solve_iterative_into<S: RealScalar + NumericScalar<Real = S>>(
    iterative: &dyn scioxide_linalg::IterativeLinearSolver<S>,
    jacobian: &Matrix<S>,
    right_hand_side: &Vector<S>,
    solution: &mut Vector<S>,
    workspace: &mut scioxide_linalg::IterativeWorkspace<S>,
    session: &crate::context::Session<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    solution
        .as_mut_slice()
        .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
    session.invoke_context(scioxide_core::EvaluationRole::LinearSolve, |context| {
        let report = iterative.solve_into(
            &jacobian.view().try_finite()?,
            right_hand_side.view(),
            Preconditioning::None,
            solution.view_mut(),
            workspace,
            context.into(),
        )?;
        iterative.require_success(&report)
    })
}

#[cfg(test)]
mod transaction_tests {
    use super::*;

    use scioxide_core::ResidualOperator;

    /// A two-dimensional residual whose evaluation writes `written` and then
    /// returns `outcome`.
    struct Faulty {
        written: [f64; 2],
        fail: bool,
    }

    impl ResidualOperator<f64> for Faulty {
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
            _point: &[f64],
            output: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<(), SciOxideError> {
            output.copy_from_slice(&self.written);
            if self.fail {
                Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("partial_callback_sentinel"),
                    scioxide_core::InvalidInputReason::Malformed,
                    "intentional failure after writing the output",
                ))
            } else {
                Ok(())
            }
        }
    }
    impl DenseJacobian<f64> for Faulty {}

    fn with_session(test: impl FnOnce(&crate::context::Session<'_, '_, '_>)) {
        let mut state = scioxide_core::SolveState::new();
        let mut budgets = scioxide_core::SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential(),
                crate::context::identity(
                    scioxide_core::algorithm_id!("rootfind", "test", 1),
                    scioxide_core::ProblemToken::new(0),
                    2,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        test(&crate::context::Session::new(&mut context));
    }

    #[test]
    fn failed_or_non_finite_callbacks_are_counted_without_committing_output() {
        for (function, kind) in [
            (
                Faulty {
                    written: [99.0, 98.0],
                    fail: true,
                },
                scioxide_core::SciOxideErrorKind::InvalidInput,
            ),
            (
                Faulty {
                    written: [f64::NAN, 1.0],
                    fail: false,
                },
                scioxide_core::SciOxideErrorKind::NonFinite,
            ),
        ] {
            with_session(|session| {
                let point = Vector::from_vec(vec![0.0, 0.0]);
                let mut output = Vector::from_vec(vec![7.0, 8.0]);
                let mut candidate = Vector::from_vec(vec![1.0, 2.0]);
                let result = evaluate_system_into(
                    &function,
                    &point,
                    &mut output,
                    &mut candidate,
                    &mut [],
                    session,
                );
                assert_eq!(result.unwrap_err().kind(), kind);
                assert_eq!(output.as_slice(), &[7.0, 8.0]);
                assert_eq!(
                    session
                        .accounting()
                        .unwrap()
                        .own()
                        .provider(scioxide_core::EvaluationRole::Residual),
                    1
                );
            });
        }
    }

    #[test]
    fn wrong_output_shapes_are_rejected_before_callbacks() {
        with_session(|session| {
            let function = Faulty {
                written: [1.0, 2.0],
                fail: false,
            };
            let point = Vector::from_vec(vec![0.0, 0.0]);
            let mut output = Vector::from_vec(vec![7.0]);
            let mut candidate = Vector::from_vec(vec![1.0, 2.0]);
            let result = evaluate_system_into(
                &function,
                &point,
                &mut output,
                &mut candidate,
                &mut [],
                session,
            );
            assert_eq!(
                result.unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(
                session
                    .accounting()
                    .unwrap()
                    .own()
                    .provider(scioxide_core::EvaluationRole::Residual),
                0
            );
        });
    }

    struct MutatingLineSearch;

    impl RootLineSearch<f64> for MutatingLineSearch {
        fn search_into(
            &self,
            _function: &dyn scioxide_core::ResidualOperator<f64>,
            _point: &[f64],
            _direction: &[f64],
            _residual: &[f64],
            _merit_slope: scioxide_core::Scaled<f64>,
            trial_point: &mut [f64],
            trial_residual: &mut [f64],
            _scratch: &mut [f64],
        ) -> Result<Option<f64>, SciOxideError> {
            trial_point.fill(91.0);
            trial_residual.fill(92.0);
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("line_search_sentinel"),
                scioxide_core::InvalidInputReason::Malformed,
                "intentional line-search failure",
            ))
        }
    }

    #[test]
    fn dynamic_line_search_failure_preserves_commit_buffers() {
        with_session(|session| {
            let function = Faulty {
                written: [1.0, 2.0],
                fail: true,
            };
            let point = Vector::from_vec(vec![0.0, 0.0]);
            let direction = Vector::from_vec(vec![1.0, 1.0]);
            let mut trial_point = Vector::from_vec(vec![7.0, 8.0]);
            let mut trial_residual = Vector::from_vec(vec![9.0, 10.0]);
            let mut candidate_point = Vector::from_vec(vec![0.0, 0.0]);
            let mut candidate_residual = Vector::from_vec(vec![0.0, 0.0]);
            let line_search: &dyn RootLineSearch<f64> = &MutatingLineSearch;
            let result = line_search_into(
                line_search,
                &function,
                &point,
                &direction,
                &point,
                scioxide_core::Scaled::from_value(-1.0).unwrap(),
                &mut trial_point,
                &mut trial_residual,
                &mut candidate_point,
                &mut candidate_residual,
                &mut [],
                session,
            );
            assert_eq!(
                result.unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::InvalidInput
            );
            assert_eq!(trial_point.as_slice(), &[7.0, 8.0]);
            assert_eq!(trial_residual.as_slice(), &[9.0, 10.0]);
        });
    }
}

#[cfg(test)]
mod residual_policy_tests {
    use super::*;

    fn check<S: RealScalar>() {
        let tolerance = scioxide_core::Tolerance::new(
            S::try_from_f64_lossy(1e-6).unwrap(),
            S::try_from_f64_lossy(1e-6).unwrap(),
        )
        .unwrap();
        assert!(!residual_ok(&tolerance, S::ONE, S::ONE).unwrap());
        assert!(residual_ok(&tolerance, S::ONE, S::try_from_f64_lossy(1e8).unwrap()).unwrap());
        assert!(residual_ok(&tolerance, S::ZERO, S::ONE).unwrap());
    }

    #[test]
    fn residual_threshold_uses_explicit_scale_in_both_precisions() {
        check::<f32>();
        check::<f64>();
        for scale in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(
                crate::NewtonSystem::new(
                    crate::NewtonSystemOptions::<f64>::default().with_residual_scale(scale)
                )
                .is_err()
            );
        }
    }
}
