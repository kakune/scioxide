//! Real evaluation adapters and family accuracy envelopes.

use crate::result::{Bound, Evaluation, Ulps};
use crate::{EllipticOptions, SpecialFunctionTermination};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, RealScalar, SciOxideError, algorithm_component,
};

const COMPLETE_ELLIPTIC_ULPS: Ulps = Ulps::new(2.0, 1.0);

fn elliptic_result<S: RealScalar>(
    evaluation: Result<crate::elliptic_family::elliptic::EllipticEvaluation<S>, SciOxideError>,
    family: AlgorithmComponent,
    work: &crate::support::Work,
) -> Result<Evaluation<S>, SciOxideError> {
    match evaluation {
        Ok(e) if e.value.is_infinite() => Ok(Evaluation::limit(
            e.value,
            AlgorithmId::from_components(family, e.branch_id, 1),
        )),
        Ok(e) => {
            // Rounding model of native endpoint arithmetic versus widened
            // binary64 Carlson duplication, proportional to the steps taken.
            let eps = <S as RealScalar>::EPSILON;
            let work_eps = if e.widened() {
                S::try_from_f64_lossy(f64::EPSILON)?
            } else {
                eps
            };
            let relative = S::try_from_f64_lossy(12.0)? * eps
                + S::try_from_f64_lossy((e.iterations + 1) as f64)?
                    * S::try_from_f64_lossy(8.0)?
                    * work_eps;
            Ok(Evaluation::finite(
                e.value,
                Bound::derived((relative * e.value.abs()).min(S::MAX)),
                None,
                AlgorithmId::from_components(family, e.branch_id, 1),
                e.iterations,
                SpecialFunctionTermination::Converged,
            ))
        }
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            ) =>
        {
            Err(crate::result::exhausted_work(work.used))
        }
        Err(error) => Err(error),
    }
}

/// Complete K/E have a calibrated fixed-cost branch; other elliptic kernels
/// keep their step-dependent rounding model.
#[inline]
fn complete_elliptic_result<S: RealScalar>(
    evaluation: Result<crate::elliptic_family::elliptic::EllipticEvaluation<S>, SciOxideError>,
    family: AlgorithmComponent,
    work: &crate::support::Work,
) -> Result<Evaluation<S>, SciOxideError> {
    match evaluation {
        Ok(e) if e.branch_id == algorithm_component!("elliptic_e.exact_m_one") => {
            Ok(Evaluation::finite(
                e.value,
                Bound::derived(S::ZERO),
                None,
                AlgorithmId::from_components(family, e.branch_id, 1),
                e.iterations,
                SpecialFunctionTermination::Exact,
            ))
        }
        Ok(e) if e.branch_id.as_str() == "complete_minimax" => Ok(Evaluation::calibrated(
            e.value,
            COMPLETE_ELLIPTIC_ULPS,
            AlgorithmId::from_components(family, e.branch_id, 1),
            0,
            SpecialFunctionTermination::Converged,
        )),
        other => elliptic_result(other, family, work),
    }
}
#[inline]
pub(crate) fn elliptic_k<S: RealScalar>(parameter: S) -> Result<S, scioxide_core::SciOxideError> {
    Ok(crate::elliptic_family::elliptic::complete_k(parameter)?.value)
}

#[inline]
pub(crate) fn elliptic_e<S: RealScalar>(parameter: S) -> Result<S, scioxide_core::SciOxideError> {
    Ok(crate::elliptic_family::elliptic::complete_e(parameter)?.value)
}

pub(crate) fn elliptic_f<S: RealScalar>(
    amplitude: S,
    parameter: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <EllipticOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    elliptic_f_evaluate(amplitude, parameter, options, None)?.value()
}

pub(crate) fn elliptic_e_inc<S: RealScalar>(
    amplitude: S,
    parameter: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <EllipticOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    elliptic_e_inc_evaluate(amplitude, parameter, options, None)?.value()
}

pub(crate) fn elliptic_pi<S: RealScalar>(
    characteristic: S,
    parameter: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <EllipticOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    elliptic_pi_evaluate(characteristic, parameter, options, None)?.value()
}

/// Evaluate before projecting the value or forming the detailed result.
#[inline]
pub(crate) fn elliptic_k_evaluate<S: RealScalar>(
    parameter: S,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work =
        crate::support::Work::elliptic(options.iteration_budget().get(), f64::EPSILON, context);
    work.check()?;
    let evaluation =
        crate::elliptic_family::elliptic::elliptic_k_evaluate_with_work(parameter, &mut work);
    work.check()?;
    complete_elliptic_result(evaluation, algorithm_component!("elliptic_k"), &work)
}

/// Evaluate before projecting the value or forming the detailed result.
#[inline]
pub(crate) fn elliptic_e_evaluate<S: RealScalar>(
    parameter: S,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work =
        crate::support::Work::elliptic(options.iteration_budget().get(), f64::EPSILON, context);
    work.check()?;
    let evaluation =
        crate::elliptic_family::elliptic::elliptic_e_evaluate_with_work(parameter, &mut work);
    work.check()?;
    complete_elliptic_result(evaluation, algorithm_component!("elliptic_e"), &work)
}

/// Evaluate before projecting the value or forming the detailed result.
pub(crate) fn elliptic_f_evaluate<S: RealScalar>(
    amplitude: S,
    parameter: S,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work =
        crate::support::Work::elliptic(options.iteration_budget().get(), f64::EPSILON, context);
    work.check()?;
    let evaluation = crate::elliptic_family::elliptic::elliptic_f_evaluate_with_work(
        amplitude, parameter, &mut work,
    );
    work.check()?;
    elliptic_result(evaluation, algorithm_component!("elliptic_f"), &work)
}

/// Evaluate before projecting the value or forming the detailed result.
pub(crate) fn elliptic_e_inc_evaluate<S: RealScalar>(
    amplitude: S,
    parameter: S,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work =
        crate::support::Work::elliptic(options.iteration_budget().get(), f64::EPSILON, context);
    work.check()?;
    let evaluation = crate::elliptic_family::elliptic::elliptic_e_inc_evaluate_with_work(
        amplitude, parameter, &mut work,
    );
    work.check()?;
    elliptic_result(evaluation, algorithm_component!("elliptic_e_inc"), &work)
}

/// Evaluate before projecting the value or forming the detailed result.
pub(crate) fn elliptic_pi_evaluate<S: RealScalar>(
    characteristic: S,
    parameter: S,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work =
        crate::support::Work::elliptic(options.iteration_budget().get(), f64::EPSILON, context);
    work.check()?;
    let evaluation = crate::elliptic_family::elliptic::elliptic_pi_evaluate_with_work(
        characteristic,
        parameter,
        &mut work,
    );
    work.check()?;
    elliptic_result(evaluation, algorithm_component!("elliptic_pi"), &work)
}
