//! First derivatives for scalar root solvers.

use scioxide_core::{ExtendedInterval, RealScalar, SciOxideError};
use scioxide_differentiate::{DerivativeCandidate, DerivativeStrategy};

use crate::scalar::Run;
use crate::traits::{ScalarFunction, finite_output};

/// Evaluates `f′(x)`, where `f(x) = function_value`, from the first source in
/// `strategy` that provides one.
///
/// Each source tried counts as one Jacobian evaluation. Analytic and automatic
/// derivatives are checked for NaN and infinity. Finite differences sample
/// only inside `probes`, reuse `function_value`, count each sample as a
/// function evaluation and use stack storage.
#[inline]
pub(crate) fn scalar_derivative<S: RealScalar, F: ScalarFunction<S> + ?Sized>(
    function: &F,
    x: S,
    function_value: S,
    strategy: DerivativeStrategy,
    run: &mut Run<'_, '_, '_, S>,
    probes: ExtendedInterval<S>,
) -> Result<S, SciOxideError> {
    strategy
        .select(|source| {
            run.admit(scioxide_core::EvaluationRole::Jacobian)?;
            match source {
                DerivativeCandidate::Analytic => function.derivative(x)?.map(finite_output),
                DerivativeCandidate::AutomaticDifferentiation => {
                    function.automatic_derivative(x)?.map(finite_output)
                }
                DerivativeCandidate::FiniteDifference(formula) => {
                    return finite_difference(function, x, function_value, formula, run, probes);
                }
            }
            .transpose()
        })
        .map(|(_, derivative)| derivative)
}

/// Keep finite-difference preparation out of the analytic selection path.
#[inline(never)]
fn finite_difference<S: RealScalar, F: ScalarFunction<S> + ?Sized>(
    function: &F,
    x: S,
    function_value: S,
    formula: scioxide_differentiate::DifferenceScheme,
    run: &mut Run<'_, '_, '_, S>,
    probes: ExtendedInterval<S>,
) -> Result<Option<S>, SciOxideError> {
    let mut slope = S::ZERO;
    let report = scioxide_differentiate::FiniteDifference::default()
        .with_scheme(formula)
        .derivative_into(
            |t: S| Ok(run.evaluate(function, t)?.value),
            x,
            probes,
            Some(function_value),
            &mut slope,
            scioxide_core::ExecutionContext::sequential(),
        )?;
    if !report.output_available() {
        return Ok(None);
    }
    if slope.is_finite() {
        Ok(Some(slope))
    } else {
        Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "finite-difference slope must be finite",
        ))
    }
}
