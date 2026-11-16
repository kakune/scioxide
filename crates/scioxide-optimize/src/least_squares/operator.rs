//! Borrowed weighted Jacobian operators and context-controlled LSMR steps.
use super::{LeastSquaresLinearSolver, LeastSquaresOptions};
use crate::context::SolveSession;
use crate::least_squares::LeastSquaresStorage;
use crate::{LeastSquaresProblem, OptimizeTermination};
use core::cmp::Ordering;

use scioxide_core::{
    AdjointOperator, EvaluationRole, LinearOperator, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_core::{Scaled, ScaledAccumulator};
use scioxide_linalg::{LinearSolveTermination, LsmrSolver};

pub(crate) fn selected<S, P>(
    problem: &P,
    options: &LeastSquaresOptions<S>,
) -> Result<bool, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let available = problem.jacobian_operator_workspace_len()?.is_some();
    match options.linear_solver {
        LeastSquaresLinearSolver::Auto => Ok(available),
        LeastSquaresLinearSolver::DenseSvd => Ok(false),
        LeastSquaresLinearSolver::Lsmr if available => Ok(true),
        LeastSquaresLinearSolver::Lsmr => Err(SciOxideError::not_implemented(
            "LSMR requires Jacobian and adjoint products",
        )),
    }
}

struct Jacobian<'a, S: RealScalar + NumericScalar<Real = S>, P: ?Sized> {
    problem: &'a P,
    x: &'a [S],
    weights: &'a [Scaled<S>],
    radial: &'a [Scaled<S>],
    residual: &'a [S],
    scales: &'a [S],
    normalizer: Scaled<S>,
    damping: S,
    augmented: bool,
    scratch: usize,
}
impl<S, P> LinearOperator<S> for Jacobian<'_, S, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    fn rows(&self) -> usize {
        self.problem.residual_dim() + if self.augmented { self.scales.len() } else { 0 }
    }
    fn cols(&self) -> usize {
        self.scales.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        check(
            input.len(),
            output.len(),
            scratch.len(),
            self.cols(),
            self.rows(),
            self.scratch,
        )?;
        let (temporary, provider) =
            scratch.split_at_mut(self.cols().max(self.problem.residual_dim()));
        for ((x, &value), &scale) in temporary.iter_mut().zip(input).zip(self.scales) {
            *x = Scaled::from_value(value)?
                .mul(Scaled::from_value(scale)?)?
                .round()?;
        }
        let m = self.problem.residual_dim();
        self.problem.jacobian_product_into(
            self.x,
            &temporary[..self.cols()],
            &mut output[..m],
            provider,
        )?;
        crate::utils::validate_finite_origin(
            &output[..m],
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        for start in (0..m).step_by(self.problem.residual_group_size()) {
            transform(
                self,
                start,
                &mut output[start..start + self.problem.residual_group_size()],
            )?;
        }
        if self.augmented {
            for (value, &x) in output[m..].iter_mut().zip(input) {
                *value = Scaled::from_value(x)?
                    .mul(Scaled::from_value(self.damping.sqrt())?)?
                    .round()?;
            }
        }
        Ok(())
    }
}
impl<S, P> AdjointOperator<S> for Jacobian<'_, S, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        check(
            input.len(),
            output.len(),
            scratch.len(),
            self.rows(),
            self.cols(),
            self.scratch,
        )?;
        let m = self.problem.residual_dim();
        let (temporary, provider) = scratch.split_at_mut(self.cols().max(m));
        temporary[..m].copy_from_slice(&input[..m]);
        for start in (0..m).step_by(self.problem.residual_group_size()) {
            transform(
                self,
                start,
                &mut temporary[start..start + self.problem.residual_group_size()],
            )?;
        }
        self.problem
            .jacobian_adjoint_product_into(self.x, &temporary[..m], output, provider)?;
        crate::utils::validate_finite_origin(output, scioxide_core::NonFiniteOrigin::Callback)?;
        for (index, (value, &scale)) in output.iter_mut().zip(self.scales).enumerate() {
            let mut sum = scioxide_core::RangeAccumulator::new();
            sum.add_product(*value, scale)?;
            if self.augmented {
                sum.add_product(self.damping.sqrt(), input[m + index])?;
            }
            *value = sum.finish()?;
        }
        Ok(())
    }
}
/// Symmetric radial/tangential square root of the robust observation curvature.
fn transform<S, P>(
    op: &Jacobian<'_, S, P>,
    start: usize,
    values: &mut [S],
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let weight = op.weights[start];
    let radial = op.radial[start];
    if weight.is_zero() {
        values.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
        return Ok(());
    }
    if values.len() == 1 {
        values[0] = Scaled::from_value(values[0])?
            .mul(radial)?
            .div(op.normalizer)?
            .round()?;
        return Ok(());
    }
    let residual = &op.residual[start..start + 2];
    let norm = super::engine::extended_norm(residual)?;
    if norm.is_zero() {
        for value in values {
            *value = Scaled::from_value(*value)?
                .mul(weight)?
                .div(op.normalizer)?
                .round()?;
        }
        return Ok(());
    }
    let u = Scaled::from_value(residual[0])?.div(norm)?;
    let v = Scaled::from_value(residual[1])?.div(norm)?;
    let a = Scaled::from_value(values[0])?;
    let b = Scaled::from_value(values[1])?;
    let correction = radial.sub(weight)?.mul(u.mul(a)?.add(v.mul(b)?)?)?;
    values[0] = weight
        .mul(a)?
        .add(u.mul(correction)?)?
        .div(op.normalizer)?
        .round()?;
    values[1] = weight
        .mul(b)?
        .add(v.mul(correction)?)?
        .div(op.normalizer)?
        .round()?;
    Ok(())
}

fn check(
    input: usize,
    output: usize,
    scratch: usize,
    expected_input: usize,
    expected_output: usize,
    expected_scratch: usize,
) -> Result<(), SciOxideError> {
    if input != expected_input || output != expected_output || scratch < expected_scratch {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(expected_input, expected_output),
            scioxide_core::Dimensions::two(input, output),
        ));
    }
    Ok(())
}
fn solver<S: RealScalar + NumericScalar<Real = S>>(
    options: &LeastSquaresOptions<S>,
) -> LsmrSolver<S> {
    LsmrSolver::new(
        scioxide_core::ResidualTolerance::default(),
        options.max_inner_iterations,
    )
    .with_normal_tolerance(scioxide_core::GradientTolerance::default())
}

pub(crate) fn prepare<S, P>(
    problem: &P,
    options: &LeastSquaresOptions<S>,
    w: &mut LeastSquaresStorage<S>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let provider = problem.jacobian_operator_workspace_len()?.ok_or_else(|| {
        SciOxideError::not_implemented("least-squares operator workspace unavailable")
    })?;
    let scratch = provider
        .checked_add(w.parameter_dimension.max(w.residual_dimension))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("operator_scratch"))
        })?;
    if w.operator_scratch.len() != scratch {
        let additional = scratch.saturating_sub(w.operator_scratch.len());
        scioxide_core::backend::try_reserve(
            &mut w.operator_scratch,
            additional,
            scioxide_core::quantity!("workspace.operator_scratch"),
        )?;
        w.operator_scratch
            .resize(scratch, <S as scioxide_core::AlgebraicScalar>::ZERO);
    }
    let rows = w
        .parameter_dimension
        .checked_add(w.residual_dimension)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("augmented_rows")))?;
    w.residual_rhs
        .try_resize(rows, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
    let op = Jacobian {
        problem,
        x: &w.x,
        weights: &w.row_weights,
        radial: &w.curvature,
        residual: &w.residual,
        scales: &w.parameter_scales,
        normalizer: w.model_scale,
        damping: w.damping,
        augmented: true,
        scratch,
    };
    if w.lsmr_workspace.is_none() || w.operator_shape != (rows, w.parameter_dimension, scratch) {
        w.lsmr_workspace = Some(
            solver(options)
                .workspace(&scioxide_linalg::IterativeWorkspaceRequest::adjoint(&op)?)?,
        );
        w.operator_shape = (rows, w.parameter_dimension, scratch);
    }
    Ok(())
}

pub(crate) fn linearize<S, P>(
    problem: &P,
    options: &LeastSquaresOptions<S>,
    w: &mut LeastSquaresStorage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    super::engine::model_weights(
        problem,
        &w.residual,
        options,
        &mut w.row_weights,
        &mut w.curvature,
    )?;
    for (i, scale) in w.parameter_scales.iter_mut().enumerate() {
        *scale = problem
            .parameter_scale(i)
            .unwrap_or(<S as scioxide_core::AlgebraicScalar>::ONE);
    }
    // Keep the right-hand side finite; no probing or dense materialization is needed.
    let mut normalizer = Scaled::from_value(<S as scioxide_core::AlgebraicScalar>::ONE)?;
    for (i, (&r, &weight)) in w.residual.iter().zip(&w.row_weights).enumerate() {
        let value = if weight.is_zero() {
            Scaled::ZERO
        } else {
            Scaled::from_value(r)?
                .mul(weight)?
                .mul(weight)?
                .div(w.curvature[i])?
                .abs()
        };
        if value.cmp_value(normalizer) == Ordering::Greater {
            normalizer = value;
        }
    }
    w.model_scale = normalizer;
    for (i, ((value, &r), &weight)) in w
        .weighted_residual
        .iter_mut()
        .zip(&w.residual)
        .zip(&w.row_weights)
        .enumerate()
    {
        *value = if weight.is_zero() {
            <S as scioxide_core::AlgebraicScalar>::ZERO
        } else {
            Scaled::from_value(r)?
                .mul(weight)?
                .mul(weight)?
                .div(w.curvature[i])?
                .div(normalizer)?
                .round()?
        };
    }
    let op = Jacobian {
        problem,
        x: &w.x,
        weights: &w.row_weights,
        radial: &w.curvature,
        residual: &w.residual,
        scales: &w.parameter_scales,
        normalizer,
        damping: w.damping,
        augmented: false,
        scratch: w.operator_scratch.len(),
    };
    session.invoke(EvaluationRole::LinearOperator, || {
        session.with_context(|context| {
            op.apply_adjoint_into_scoped(
                &w.weighted_residual,
                &mut w.gradient,
                &mut w.operator_scratch,
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)
        })?
    })?;
    let mut norm = ScaledAccumulator::ZERO;
    for (&gradient, &scale) in w.gradient.iter().zip(&w.parameter_scales) {
        let value = Scaled::from_value(gradient)?
            .mul(normalizer)?
            .mul(normalizer)?
            .div(Scaled::from_value(scale)?)?;
        norm.add_product(value, value)?;
    }
    w.gradient_norm = Some(norm.total()?.sqrt()?);
    w.rank = None;
    w.accepted_gradient = true;
    Ok(())
}

/// Safeguard the iterative model solve in the same scaled trust-region metric.
pub(crate) fn step<S, P>(
    problem: &P,
    options: &LeastSquaresOptions<S>,
    w: &mut LeastSquaresStorage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let Some(radius) = w.radius else {
        return solve_step(problem, options, w, session);
    };
    let gradient = super::engine::extended_norm(&w.gradient)?;
    let upper = gradient.div(radius)?;
    let maximum = Scaled::from_value(S::MAX)?;
    let mut upper = if upper.cmp_value(maximum) == Ordering::Greater {
        S::MAX
    } else {
        upper.round()?.max(S::MIN_POSITIVE)
    };
    let mut lower = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let hint = w.damping.min(upper);
    w.damping = <S as scioxide_core::AlgebraicScalar>::ZERO;
    for iteration in 0..10 {
        solve_step(problem, options, w, session)?;
        if session.stopped().is_some() {
            return Ok(());
        }
        let norm = super::engine::extended_norm(w.factor_solution.as_slice())?;
        let relative = norm.div(radius)?.round()?;
        let tolerance = <S as scioxide_core::AlgebraicScalar>::ONE / (S::TEN * S::TWO);
        if relative <= <S as scioxide_core::AlgebraicScalar>::ONE + tolerance
            && (w.damping == <S as scioxide_core::AlgebraicScalar>::ZERO
                || relative >= <S as scioxide_core::AlgebraicScalar>::ONE - tolerance)
        {
            return Ok(());
        }
        if iteration == 9 {
            break;
        }
        if relative > <S as scioxide_core::AlgebraicScalar>::ONE {
            lower = lower.max(w.damping);
        } else {
            upper = upper.min(w.damping);
        }
        w.damping = if iteration == 0 && hint > lower && hint < upper {
            hint
        } else {
            lower * S::HALF + upper * S::HALF
        };
    }
    // An inexact Krylov solve may miss the radius; radial projection preserves
    // feasibility and reuses the linear model product without another callback.
    let norm = super::engine::extended_norm(w.factor_solution.as_slice())?;
    if norm.cmp_value(radius) == Ordering::Greater {
        let alpha = radius.div(norm)?;
        for values in [
            w.factor_solution.as_mut_slice(),
            &mut w.step,
            &mut w.model_product,
        ] {
            for value in values {
                *value = Scaled::from_value(*value)?.mul(alpha)?.round()?;
            }
        }
    }
    Ok(())
}

fn solve_step<S, P>(
    problem: &P,
    options: &LeastSquaresOptions<S>,
    w: &mut LeastSquaresStorage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let mut op = Jacobian {
        problem,
        x: &w.x,
        weights: &w.row_weights,
        radial: &w.curvature,
        residual: &w.residual,
        scales: &w.parameter_scales,
        normalizer: w.model_scale,
        damping: w.damping,
        augmented: true,
        scratch: w.operator_scratch.len(),
    };
    w.residual_rhs
        .as_mut_slice()
        .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
    for (b, &r) in w
        .residual_rhs
        .as_mut_slice()
        .iter_mut()
        .zip(&w.weighted_residual)
    {
        *b = -r;
    }
    w.factor_solution
        .as_mut_slice()
        .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
    let solver = solver(options);
    let workspace = w
        .lsmr_workspace
        .as_mut()
        .ok_or_else(|| SciOxideError::not_implemented("unprepared LSMR workspace"))?;
    // LSMR runs in a child epoch whose iteration allowance is the solver limit.
    let outcome = session.invoke(EvaluationRole::LinearSolve, || {
        session
            .with_context(|context| {
                solver.solve_into(
                    &op,
                    &w.residual_rhs,
                    w.factor_solution.view_mut(),
                    workspace,
                    context,
                )
            })
            .and_then(|report| {
                let report = report?;
                w.condition = report.krylov().and_then(|k| k.condition_estimate());
                Ok(report.termination())
            })
    });
    let termination = match outcome {
        Ok(termination) => termination,
        Err(error) => {
            let stop = match error.view() {
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: scioxide_core::ConvergenceReason::EvaluationLimit,
                    ..
                } => Some(OptimizeTermination::EvaluationLimit),
                scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: scioxide_core::ConvergenceReason::IterationLimit,
                    ..
                } => Some(OptimizeTermination::IterationLimit),
                scioxide_core::SciOxideErrorView::ExecutionStopped {
                    reason: scioxide_core::ExecutionStopReason::DeadlineReached,
                    ..
                } => Some(OptimizeTermination::TimeLimit),
                scioxide_core::SciOxideErrorView::ExecutionStopped { .. } => {
                    Some(OptimizeTermination::Cancelled)
                }
                _ => None,
            };
            session.inherit_child_stop(stop)?;
            return Err(error);
        }
    };
    w.inner_termination = Some(termination);
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
    for ((step, &value), &scale) in w
        .step
        .iter_mut()
        .zip(w.factor_solution.as_slice())
        .zip(&w.parameter_scales)
    {
        *step = Scaled::from_value(value)?
            .mul(Scaled::from_value(scale)?)?
            .round()?;
    }
    op.augmented = false;
    session.invoke(EvaluationRole::LinearOperator, || {
        op.apply_into(
            w.factor_solution.as_slice(),
            &mut w.model_product,
            &mut w.operator_scratch,
        )
    })
}

/// Verify a proposed convergence result through the original adjoint, without
/// the step model's normalization or parameter scaling. Failure to represent a
/// provider input/output cannot be promoted to successful optimality.
pub(crate) fn verify_gradient<S, P>(
    problem: &P,
    w: &mut LeastSquaresStorage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    w.gradient_norm = None;
    for ((value, &r), &weight) in w
        .model_product
        .iter_mut()
        .zip(&w.residual)
        .zip(&w.row_weights)
    {
        *value = Scaled::from_value(r)?.mul(weight)?.mul(weight)?.round()?;
    }
    let provider_start = w.parameter_dimension.max(w.residual_dimension);
    session.invoke(EvaluationRole::LinearOperator, || {
        problem.jacobian_adjoint_product_into(
            &w.x,
            &w.model_product,
            &mut w.step,
            &mut w.operator_scratch[provider_start..],
        )
    })?;
    crate::utils::validate_finite_origin(&w.step, scioxide_core::NonFiniteOrigin::Callback)?;
    w.gradient_norm = Some(super::engine::extended_norm(&w.step)?);
    Ok(())
}
