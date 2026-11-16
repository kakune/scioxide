//! Internal utility functions for optimization algorithms.
//!
//! These are private helpers shared across modules within the crate.

use alloc::vec::Vec;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{checked_scaled_norm, checked_sum_of_products};
#[cfg(test)]
use scioxide_differentiate::DerivativeStrategy;
#[cfg(test)]
use scioxide_linalg::Matrix;

use crate::least_squares::LeastSquaresStorage;
use crate::result::{OptimalityDiagnostics, OptimizeResult, OptimizeTermination};
use crate::traits::ObjectiveFunction;
use crate::unconstrained::{FirstOrderStorage, SecondOrderStorage};
use crate::{LeastSquaresResult, LineSearchResult};

/// Objective magnitude per coordinate unit. The unit reference preserves the
/// absolute-gradient policy for objectives below one coordinate unit.
pub(crate) fn gradient_scale<S: RealScalar>(value: S, x: &[S]) -> S {
    let coordinate = x.iter().fold(S::ONE, |a, &v| a.max(v.abs()));
    (value.abs() / coordinate).max(S::ONE)
}

/// Cancellation-free reference magnitudes for each original-model KKT role.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimality_scales<S: RealScalar + NumericScalar<Real = S>>(
    x: &[S],
    value: S,
    gradient: &[S],
    jacobian: scioxide_linalg::MatrixView<'_, S>,
    multipliers: &[S],
    lower: &[S],
    upper: &[S],
) -> Result<crate::options::OptimalityScales<S>, SciOxideError> {
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let mut stationarity = scioxide_core::ProductSum::new();
    stationarity.add_product(norm2(gradient)?, one)?;
    for (row, &lambda) in multipliers.iter().enumerate() {
        // Observation matrices are contiguous by construction.
        let mut norm = scioxide_core::ScaledNorm::new();
        for j in 0..jacobian.cols() {
            norm.add(jacobian.get(row, j)?)?;
        }
        stationarity.add_product(NumericScalar::abs(lambda), norm.finish()?)?;
    }
    stationarity.add_product(norm2(lower)?, one)?;
    stationarity.add_product(norm2(upper)?, one)?;
    Ok(crate::options::OptimalityScales {
        primal: x.iter().fold(one, |a, &v| a.max(NumericScalar::abs(v))),
        gradient: stationarity.finish()?,
        complementarity: NumericScalar::abs(value),
        dual: multipliers
            .iter()
            .chain(lower)
            .chain(upper)
            .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |a, &v| {
                a.max(NumericScalar::abs(v))
            }),
    })
}

fn attach_metadata<S: RealScalar>(
    output: &mut OptimizeResult<S>,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let trust = if output.is_success() {
        scioxide_core::PartialResultTrust::ResidualChecked
    } else {
        scioxide_core::PartialResultTrust::Admissible
    };
    session.assess(trust)?;
    synchronize_metadata(output, session, trust)
}

pub(crate) fn synchronize_metadata<S: RealScalar>(
    output: &mut OptimizeResult<S>,
    session: &crate::context::SolveSession<'_, '_, '_>,
    trust: scioxide_core::PartialResultTrust,
) -> Result<(), SciOxideError> {
    output.counts = session.accounting()?;
    output.fun_evals = output
        .counts
        .total()?
        .provider(scioxide_core::EvaluationRole::Objective);
    output.grad_evals = output
        .counts
        .total()?
        .provider(scioxide_core::EvaluationRole::Gradient);
    output.algorithm = Some(session.algorithm()?);
    output.trust = trust;
    Ok(())
}

pub(crate) fn accepted_constrained_value<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &crate::constrained::ConstrainedStorage<S>,
) -> Result<S, SciOxideError> {
    workspace.accepted_value.ok_or_else(|| {
        SciOxideError::invalid_input(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "no objective observation has been accepted",
        )
    })
}

pub(crate) fn complete_constrained<S: RealScalar + NumericScalar<Real = S>>(
    outcome: Result<(), SciOxideError>,
    workspace: &mut crate::constrained::ConstrainedStorage<S>,
    output: &mut OptimizeResult<S>,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if let Err(error) = &outcome {
        session.record_execution_stop(error);
    }
    if let Err(error) = outcome {
        let (Some(termination), Some(value)) = (session.stopped(), workspace.accepted_value) else {
            return Err(error);
        };
        commit_optimize_result(
            output,
            &workspace.x,
            value,
            None,
            workspace.accepted_iterations,
            termination,
            "shared solve budget or execution limit reached",
        )?;
        output.diagnostics = workspace.accepted_diagnostics;
    }
    output.method = crate::OptimizeDiagnostics::FirstOrder(workspace.first_order.globalization);
    let trust = if output.is_success() {
        scioxide_core::PartialResultTrust::ResidualChecked
    } else {
        scioxide_core::PartialResultTrust::DiagnosticOnly
    };
    session.assess(trust)?;
    synchronize_metadata(output, session, trust)
}

pub(crate) fn complete_first_order<S: RealScalar + NumericScalar<Real = S>>(
    outcome: Result<(), SciOxideError>,
    workspace: &mut FirstOrderStorage<S>,
    output: &mut OptimizeResult<S>,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if let Err(error) = &outcome {
        session.record_execution_stop(error);
    }
    if let Err(error) = outcome {
        let (Some(termination), Some(value)) = (session.stopped(), workspace.accepted_value) else {
            return Err(error);
        };
        commit_optimize_result(
            output,
            &workspace.x,
            value,
            workspace
                .accepted_gradient
                .then_some(workspace.gradient.as_slice()),
            workspace.accepted_iterations,
            termination,
            "shared solve budget or execution limit reached",
        )?;
    }
    output.method = crate::OptimizeDiagnostics::FirstOrder(workspace.globalization);
    attach_metadata(output, session)
}

pub(crate) fn complete_second_order<S: RealScalar + NumericScalar<Real = S>>(
    outcome: Result<(), SciOxideError>,
    workspace: &mut SecondOrderStorage<S>,
    output: &mut OptimizeResult<S>,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if let Err(error) = &outcome {
        session.record_execution_stop(error);
    }
    if let Err(error) = outcome {
        let (Some(termination), Some(value)) = (session.stopped(), workspace.accepted_value) else {
            return Err(error);
        };
        commit_optimize_result(
            output,
            &workspace.x,
            value,
            workspace
                .accepted_gradient
                .then_some(workspace.gradient.as_slice()),
            workspace.accepted_iterations,
            termination,
            "shared solve budget or execution limit reached",
        )?;
    }
    output.method = crate::OptimizeDiagnostics::Newton(workspace.newton);
    attach_metadata(output, session)
}

pub(crate) fn complete_least_squares<S: RealScalar + NumericScalar<Real = S>>(
    outcome: Result<(), SciOxideError>,
    workspace: &mut LeastSquaresStorage<S>,
    output: &mut LeastSquaresResult<S>,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if let Err(error) = &outcome {
        session.record_execution_stop(error);
    }
    if let Err(error) = outcome {
        let (Some(termination), Some(cost)) = (session.stopped(), workspace.accepted_value) else {
            return Err(error);
        };
        finish_least_squares(
            workspace,
            output,
            cost,
            workspace.accepted_iterations,
            termination,
            "shared solve budget or execution limit reached",
        )?;
    }
    let trust = if output.termination() == OptimizeTermination::GradientSmall {
        scioxide_core::PartialResultTrust::ResidualChecked
    } else {
        scioxide_core::PartialResultTrust::Admissible
    };
    session.assess(trust)?;
    output.counts = session.accounting()?;
    output.fun_evals = output
        .counts
        .total()?
        .provider(scioxide_core::EvaluationRole::Residual);
    output.jac_evals = output
        .counts
        .total()?
        .provider(scioxide_core::EvaluationRole::Jacobian);
    output.algorithm = Some(session.algorithm()?);
    output.trust = trust;
    Ok(())
}

/// Derive stopping behavior solely from the typed line-search result.
/// Provider counts come from the solve context, never from callback metadata.
pub(crate) fn line_search_termination<S: RealScalar>(
    result: LineSearchResult<S>,
) -> Option<OptimizeTermination> {
    match result.termination() {
        crate::LineSearchTermination::Accepted => None,
        crate::LineSearchTermination::Unsuccessful(reason) => {
            Some(OptimizeTermination::LineSearchFailed(reason))
        }
    }
}

/// Copy a completed derivative-based solve into reusable owned result storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_optimize_result<S: RealScalar + NumericScalar<Real = S>>(
    output: &mut OptimizeResult<S>,
    x: &[S],
    fun: S,
    gradient: Option<&[S]>,
    iterations: usize,
    termination: OptimizeTermination,
    message: &'static str,
) -> Result<(), SciOxideError> {
    let gradient_capacity = output
        .grad
        .as_ref()
        .map_or(output.gradient_storage.capacity(), Vec::capacity);
    if output.x.capacity() < x.len()
        || gradient.is_some_and(|gradient| gradient_capacity < gradient.len())
    {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(x.len()),
            scioxide_core::Dimensions::one(output.x.capacity()),
        ));
    }
    validate_finite_origin(x, scioxide_core::NonFiniteOrigin::Output)?;
    if !fun.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Output,
            "result objective must be finite",
        ));
    }
    let stationarity = gradient.map(|gradient| norm2(gradient)).transpose()?;
    if let Some(gradient) = gradient {
        validate_finite_origin(gradient, scioxide_core::NonFiniteOrigin::Output)?;
    }
    output.method = crate::OptimizeDiagnostics::Unavailable;
    output
        .x
        .resize(x.len(), <S as scioxide_core::AlgebraicScalar>::ZERO);
    output.x.copy_from_slice(x);
    if let Some(gradient) = gradient {
        let mut values = output
            .grad
            .take()
            .unwrap_or_else(|| core::mem::take(&mut output.gradient_storage));
        values.resize(gradient.len(), <S as scioxide_core::AlgebraicScalar>::ZERO);
        values.copy_from_slice(gradient);
        output.grad = Some(values);
    } else {
        if let Some(values) = output.grad.take() {
            output.gradient_storage = values;
        }
    }
    output.fun = fun;
    output.iterations = iterations;
    output.termination = termination;
    output.diagnostics = OptimalityDiagnostics {
        stationarity,
        ..OptimalityDiagnostics::unavailable()
    };
    output.message.clear();
    output.message.push_str(message);
    Ok(())
}

/// Finalize a first-order solve into caller output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_first_order<S>(
    workspace: &mut FirstOrderStorage<S>,
    output: &mut OptimizeResult<S>,
    fun: S,
    has_gradient: bool,
    iterations: usize,
    termination: OptimizeTermination,
    message: &'static str,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    commit_optimize_result(
        output,
        &workspace.x,
        fun,
        has_gradient.then_some(workspace.gradient.as_slice()),
        iterations,
        termination,
        message,
    )
}

/// Finalize a second-order solve into caller output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_second_order<S>(
    workspace: &mut SecondOrderStorage<S>,
    output: &mut OptimizeResult<S>,
    fun: S,
    iterations: usize,
    termination: OptimizeTermination,
    message: &'static str,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    commit_optimize_result(
        output,
        &workspace.x,
        fun,
        Some(&workspace.gradient),
        iterations,
        termination,
        message,
    )
}

/// Copy a completed least-squares solve into reusable result storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_least_squares<S>(
    workspace: &mut LeastSquaresStorage<S>,
    output: &mut LeastSquaresResult<S>,
    cost: scioxide_core::Scaled<S>,
    iterations: usize,
    termination: OptimizeTermination,
    message: &'static str,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    if output.x.capacity() < workspace.x.len()
        || output.residual.capacity() < workspace.residual.len()
    {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(workspace.x.len(), workspace.residual.len()),
            scioxide_core::Dimensions::two(output.x.capacity(), output.residual.capacity()),
        ));
    }
    validate_finite_origin(&workspace.x, scioxide_core::NonFiniteOrigin::Output)?;
    validate_finite_origin(&workspace.residual, scioxide_core::NonFiniteOrigin::Output)?;
    let stationarity = workspace.gradient_norm.and_then(|norm| norm.round().ok());
    output.x.resize(
        workspace.x.len(),
        <S as scioxide_core::AlgebraicScalar>::ZERO,
    );
    output.residual.resize(
        workspace.residual.len(),
        <S as scioxide_core::AlgebraicScalar>::ZERO,
    );
    output.x.copy_from_slice(&workspace.x);
    output.residual.copy_from_slice(&workspace.residual);
    output.cost = cost;
    output.condition = if workspace.accepted_gradient || workspace.inner_termination.is_some() {
        workspace.condition
    } else {
        None
    };
    output.damping = workspace.damping;
    output.last_gain = workspace.last_gain;
    output.inner_termination = workspace.inner_termination;
    output.rank = if workspace.accepted_gradient {
        workspace.rank
    } else {
        None
    };
    output.gradient_norm = if workspace.accepted_gradient {
        workspace.gradient_norm
    } else {
        None
    };
    output.iterations = iterations;
    output.termination = termination;
    output.diagnostics = OptimalityDiagnostics {
        stationarity,
        ..OptimalityDiagnostics::unavailable()
    };
    output.message.clear();
    output.message.push_str(message);
    Ok(())
}

/// Cast a `usize` into scalar type `S` exactly.
///
/// # Errors
///
/// - `scioxide_core::SciOxideErrorKind::NumericalInstability` — if the cast fails.
pub(crate) fn cast_usize<S: RealScalar>(value: usize) -> Result<S, SciOxideError> {
    S::try_from_usize_exact(value)
}

pub(crate) use crate::derivatives::{
    gradient_into as get_gradient_into, hessian_into as get_hessian_into,
};

/// Compute an exact, scale-safe dot product for model/certificate arithmetic.
pub(crate) fn dot<S: RealScalar + NumericScalar<Real = S>>(
    a: &[S],
    b: &[S],
) -> Result<S, SciOxideError> {
    checked_sum_of_products(a, b)
}

/// Compensated numerical contraction, replayed exactly on range hazards.
/// A factory permits sparse traversals without collecting their entries.
pub(crate) fn compensated_products<S, I>(entries: impl Fn() -> I) -> Result<S, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    I: Iterator<Item = (S, S)>,
{
    let mut sum = scioxide_core::CompensatedSum::ZERO;
    let mut ordinary = true;
    for (a, b) in entries() {
        let product = a * b;
        if scioxide_core::backend::standard::product_needs_retry(a, b, product)
            || sum.add(product).is_err()
        {
            ordinary = false;
            break;
        }
    }
    if ordinary && let Ok(value) = sum.total() {
        return Ok(value);
    }
    let mut sum = scioxide_core::ProductSum::new();
    for (a, b) in entries() {
        sum.add_product(a, b)?;
    }
    sum.finish()
}

/// Compensated BLAS-1 reduction for line-search directional derivatives.
/// Short vectors keep exact products; range hazards replay the exact reduction.
#[inline]
pub(crate) fn search_dot<S: RealScalar>(a: &[S], b: &[S]) -> Result<S, SciOxideError> {
    if a.len() >= 16
        && let Some(value) = crate::quasi_newton::ordinary_dot(a, b)
    {
        Ok(value)
    } else {
        dot(a, b)
    }
}

/// Compute a checked, scale-safe Euclidean norm.
pub(crate) fn norm2<S: RealScalar + NumericScalar<Real = S>>(v: &[S]) -> Result<S, SciOxideError> {
    checked_scaled_norm(v)
}

/// Form the complete fused trial expression, rescaling to avoid overflow and cancellation.
#[inline]
pub(crate) fn trial_component<S: RealScalar>(
    current: S,
    alpha: S,
    direction: S,
) -> Result<S, SciOxideError> {
    scioxide_core::checked_mul_add(alpha, direction, current)
}

/// Form a trial coordinate with ordinary BLAS-1 arithmetic in a resolved frame.
/// Range hazards and cancellation retry the complete fused expression.
#[inline]
pub(crate) fn ordinary_trial_component<S: RealScalar>(
    current: S,
    alpha: S,
    direction: S,
) -> Result<S, SciOxideError> {
    let product = alpha * direction;
    let trial = scioxide_core::fast_mul_add(alpha, direction, current);
    let scale = current.abs().max(product.abs());
    if !scioxide_core::backend::standard::product_needs_retry(alpha, direction, product)
        && current.is_finite()
        && trial.is_finite()
        && (scale == S::ZERO || trial.abs() > S::TWO * S::TWO * S::TWO * S::EPSILON * scale)
    {
        Ok(trial)
    } else {
        scioxide_core::checked_mul_add(alpha, direction, current)
    }
}

/// Write the trial point `x + alpha * direction`, returning `false` when a
/// coordinate overflows. Such a trial lies outside every representable domain,
/// so searches reject it like an out-of-domain or non-finite trial.
/// Line searches use ordinary BLAS-1 arithmetic for vectors of at least 16 entries.
pub(crate) fn fill_trial<S: RealScalar>(
    trial: &mut [S],
    x: &[S],
    alpha: S,
    direction: &[S],
) -> Result<bool, SciOxideError> {
    fill_trial_with(trial, x, alpha, direction, x.len() >= 16)
}

/// [`fill_trial`] with the complete fused expression for every coordinate.
pub(crate) fn fill_exact_trial<S: RealScalar>(
    trial: &mut [S],
    x: &[S],
    alpha: S,
    direction: &[S],
) -> Result<bool, SciOxideError> {
    fill_trial_with(trial, x, alpha, direction, false)
}

fn fill_trial_with<S: RealScalar>(
    trial: &mut [S],
    x: &[S],
    alpha: S,
    direction: &[S],
    ordinary: bool,
) -> Result<bool, SciOxideError> {
    for (candidate, (&current, &component)) in trial.iter_mut().zip(x.iter().zip(direction)) {
        let value = if ordinary {
            ordinary_trial_component(current, alpha, component)
        } else {
            trial_component(current, alpha, component)
        };
        *candidate = match value {
            Ok(value) => value,
            Err(error)
                if error.kind() == scioxide_core::SciOxideErrorKind::NumericalInstability =>
            {
                return Ok(false);
            }
            Err(error) => return Err(error),
        };
    }
    Ok(true)
}

/// Compare the complete Armijo expression in extended range.
#[inline]
pub(crate) fn armijo_satisfied<S: RealScalar>(
    value: S,
    initial: S,
    c1: S,
    alpha: S,
    slope: S,
    roundoff_units: S,
) -> Result<bool, SciOxideError> {
    // Ordinary comparisons far from the rounding boundary need no exponent arithmetic.
    let weight = c1 * alpha;
    let decrease = weight * slope;
    let bound = initial + decrease;
    let scale = value
        .abs()
        .max(initial.abs())
        .max(decrease.abs())
        .max(bound.abs());
    if !scioxide_core::backend::standard::product_needs_retry(c1, alpha, weight)
        && !scioxide_core::backend::standard::product_needs_retry(weight, slope, decrease)
        && bound.is_finite()
        && scale.is_finite()
        && (value - bound).abs() > S::try_from_usize_exact(16)? * S::EPSILON * scale
    {
        return Ok(value <= bound + roundoff_units * S::EPSILON * scale);
    }
    use scioxide_core::Scaled;
    let decrease = Scaled::from_value(c1)?
        .mul(Scaled::from_value(alpha)?)?
        .mul(Scaled::from_value(slope)?)?;
    let bound = Scaled::from_value(initial)?.add(decrease)?;
    let scale = [
        Scaled::from_value(value)?,
        Scaled::from_value(initial)?,
        decrease,
        bound,
    ]
    .into_iter()
    .map(Scaled::abs)
    .fold(Scaled::ZERO, |scale, value| {
        if scale.abs_le(value) { value } else { scale }
    });
    let slack = scale.mul(Scaled::from_value(roundoff_units * S::EPSILON)?)?;
    Ok(Scaled::from_value(value)?.cmp_value(bound.add(slack)?) != core::cmp::Ordering::Greater)
}

/// Quadratic decrease, kept scaled so intermediate products need not fit `S`.
pub(crate) fn model_prediction<S: RealScalar>(
    gradient: &[S],
    step: &[S],
    product: &[S],
) -> Result<scioxide_core::Scaled<S>, SciOxideError> {
    use scioxide_core::{Scaled, ScaledAccumulator};
    let mut predicted = ScaledAccumulator::ZERO;
    for ((&gradient, &step), &product) in gradient.iter().zip(step).zip(product) {
        predicted.add_product(Scaled::from_value(-gradient)?, Scaled::from_value(step)?)?;
        predicted.add_scaled(
            Scaled::from_value(-S::HALF)?
                .mul(Scaled::from_value(step)?)?
                .mul(Scaled::from_value(product)?)?,
        )?;
    }
    predicted.total()
}

/// Validate that the initial guess has the correct dimension.
///
/// # Errors
///
/// - [`scioxide_core::SciOxideErrorKind::DimensionMismatch`] — if `x0.len() != f.dim()`
///
pub(crate) fn validate_x0<S, F>(f: &F, x0: &[S]) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    if x0.len() != f.dim() {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(f.dim()),
            scioxide_core::Dimensions::one(x0.len()),
        ));
    }
    validate_finite_origin(x0, scioxide_core::NonFiniteOrigin::Input)
}

/// Validate that all components of a slice are finite.
///
/// # Errors
///
/// - [`scioxide_core::SciOxideErrorKind::NonFinite`] — if any component is NaN or infinite
pub(crate) use crate::validation::finite_callback as validate_finite;

/// Validate that all components of a slice are finite and keep their origin.
///
/// # Errors
///
/// - [`scioxide_core::SciOxideErrorKind::NonFinite`] — if any component is NaN or infinite.
pub(crate) use crate::validation::finite_real as validate_finite_origin;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    #[test]
    fn scaled_trials_and_costs_round_only_the_complete_expression() {
        assert_eq!(trial_component(-f64::MAX, 2.0, f64::MAX).unwrap(), f64::MAX);
        assert_eq!(trial_component(-f32::MAX, 2.0, f32::MAX).unwrap(), f32::MAX);
        assert!(armijo_satisfied(-f64::MAX, f64::MAX, 0.5, 4.0, -f64::MAX, 0.0).unwrap());
        assert!(armijo_satisfied(-f32::MAX, f32::MAX, 0.5, 4.0, -f32::MAX, 0.0).unwrap());
        assert!(matches!(
            (trial_component(f64::MAX, 2.0, f64::MAX))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NumericalInstability { .. })
        ));
    }

    struct Rosenbrock;

    impl ObjectiveFunction<f64> for Rosenbrock {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            Ok(a * a + 100.0 * b * b)
        }
        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            let dx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            let dx1 = 200.0 * (x[1] - x[0] * x[0]);
            output.copy_from_slice(&[dx0, dx1]);
            Ok(true)
        }
    }

    struct NoGradObj;

    impl ObjectiveFunction<f64> for NoGradObj {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }
    }

    // ---- get_gradient_into ----

    #[test]
    fn test_get_gradient_analytic() {
        let f = Rosenbrock;
        let x = [1.0, 1.0];
        let f_x = f.eval(&x).unwrap();
        let mut g = [0.0; 2];
        let mut workspace = crate::DerivativeWorkspace::new(
            scioxide_differentiate::DerivativeOperation::Gradient,
            2,
            1,
        )
        .unwrap();
        get_gradient_into(
            &f,
            &x,
            f_x,
            DerivativeStrategy::default(),
            &mut g,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert!(g[0].abs() < 1e-12);
        assert!(g[1].abs() < 1e-12);
    }

    #[test]
    fn test_get_gradient_fd() {
        let f = NoGradObj;
        let x = [3.0, 4.0];
        let f_x = f.eval(&x).unwrap();
        let mut g = [0.0; 2];
        let mut workspace = crate::DerivativeWorkspace::new(
            scioxide_differentiate::DerivativeOperation::Gradient,
            2,
            1,
        )
        .unwrap();
        get_gradient_into(
            &f,
            &x,
            f_x,
            DerivativeStrategy::default(),
            &mut g,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert!((g[0] - 6.0).abs() < 1e-5);
        assert!((g[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_get_gradient_automatic_source_has_no_function_evaluations() {
        struct AutomaticObjective;

        impl ObjectiveFunction<f64> for AutomaticObjective {
            fn dim(&self) -> usize {
                2
            }

            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(x[0] * x[0] + x[1] * x[1])
            }

            fn automatic_gradient_into(
                &self,
                x: &[f64],
                output: &mut [f64],
            ) -> Result<bool, SciOxideError> {
                output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
                Ok(true)
            }
        }

        let mut gradient = [0.0; 2];
        let mut workspace = crate::DerivativeWorkspace::new(
            scioxide_differentiate::DerivativeOperation::Gradient,
            2,
            1,
        )
        .unwrap();
        get_gradient_into(
            &AutomaticObjective,
            &[2.0, -3.0],
            13.0,
            DerivativeStrategy::Automatic,
            &mut gradient,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(gradient, [4.0, -6.0]);
    }

    // ---- get_hessian_into ----

    #[test]
    fn test_get_hessian_fd() {
        let f = NoGradObj;
        let x = [1.0, 1.0];
        let f_x = f.eval(&x).unwrap();
        let mut h = Matrix::zeros(2, 2).unwrap();
        let mut workspace = crate::DerivativeWorkspace::new(
            scioxide_differentiate::DerivativeOperation::Hessian,
            2,
            1,
        )
        .unwrap();
        get_hessian_into(
            &f,
            &x,
            f_x,
            DerivativeStrategy::default(),
            &mut h,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert!((h.view().get(0, 0).unwrap() - 2.0).abs() < 1e-4);
    }

    // ---- dot ----

    #[test]
    fn test_dot_product() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((dot(&a, &b).unwrap() - 32.0).abs() < 1e-12);
    }

    #[test]
    fn checked_dot_preserves_large_finite_and_cancellation_heavy_results() {
        let large = dot(&[f64::MAX / 4.0, f64::MAX / 4.0], &[1.0, 1.0]).unwrap();
        assert!(large.is_finite());
        assert_eq!(large, f64::MAX / 2.0);

        let cancelled = dot(&[1.0e200, 1.0e200, 3.0], &[1.0, -1.0, 1.0]).unwrap();
        assert!((cancelled - 3.0_f64).abs() <= 1e-12);

        assert!(matches!(
            (dot(&[f64::MAX, f64::MAX], &[1.0, 1.0]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NumericalInstability { .. })
        ));
    }

    // ---- norm2 ----

    #[test]
    fn test_norm2_value() {
        let v = [3.0_f64, 4.0];
        assert!((norm2(&v).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn checked_norm_handles_large_and_subnormal_finite_vectors() {
        let large = norm2(&[f64::MAX / 2.0, f64::MAX / 2.0]).unwrap();
        assert!(large.is_finite());
        assert!(large > f64::MAX / 2.0);

        let tiny = norm2(&[f64::MIN_POSITIVE, f64::MIN_POSITIVE]).unwrap();
        assert!(tiny.is_finite());
        assert!(tiny > 0.0);
    }

    // ---- validate_x0 ----

    #[test]
    fn test_validate_x0_valid() {
        let f = NoGradObj;
        assert!(validate_x0(&f, &[1.0, 2.0]).is_ok());
    }

    #[test]
    fn test_validate_x0_wrong_dim() {
        let f = NoGradObj;
        assert!(matches!(
            (validate_x0(&f, &[1.0]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
    }

    // ---- validate_finite ----

    #[test]
    fn test_validate_finite_ok() {
        assert!(validate_finite(&[1.0_f64, 2.0]).is_ok());
    }

    #[test]
    fn test_validate_finite_nan() {
        assert!(matches!(
            (validate_finite(&[1.0_f64, f64::NAN]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }

    #[test]
    fn test_validate_finite_inf() {
        assert!(matches!(
            (validate_finite(&[f64::INFINITY]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }
    #[test]
    fn ordinary_trials_retry_overflow_underflow_and_cancellation() {
        assert_eq!(
            ordinary_trial_component(f64::MAX, 2.0, -f64::MAX).unwrap(),
            -f64::MAX
        );
        let tiny = f64::from_bits(1);
        assert_eq!(
            ordinary_trial_component(tiny, 0.5, tiny).unwrap(),
            2.0 * tiny
        );
        let epsilon = f64::EPSILON;
        assert_eq!(
            ordinary_trial_component(-1.0, 1.0 + epsilon, 1.0 - epsilon).unwrap(),
            -epsilon * epsilon
        );
        assert!(ordinary_trial_component(f64::NAN, 1.0, 0.0).is_err());
        assert!(ordinary_trial_component(f64::MAX, 2.0, f64::MAX).is_err());
        let mut a = [0.0; 16];
        let mut b = a;
        a[..3].copy_from_slice(&[f64::MAX, f64::MAX, tiny]);
        b[..3].copy_from_slice(&[2.0, -2.0, 1.0]);
        assert_eq!(search_dot(&a, &b).unwrap(), tiny);
    }
}
