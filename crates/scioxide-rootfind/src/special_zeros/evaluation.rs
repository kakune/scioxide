use super::KelvinZeroFunction;
use super::SpecialZeroFamily as F;
use scioxide_core::{Budget, BudgetKind, Complex, RealScalar, SciOxideError, SolveContext};
use scioxide_special::*;
pub(super) type C = Complex<f64>;
#[derive(Clone, Copy)]
pub(super) struct Evaluation {
    pub value: C,
    pub derivative: C,
    pub error: f64,
    pub derivative_error: f64,
}
/// Special-function callback of one zero request.
#[derive(Clone, Copy)]
pub(super) struct Provider {
    pub family: F<f64>,
    pub budget: Budget,
}
/// Unavailable value or derivative.
fn missing() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::LossOfSignificance,
        "value or analytic derivative is unavailable",
    )
}
/// Charge `used` special-function iterations to the zero's allowance in `context`.
///
/// Scoped special functions run in child epochs whose iteration allowances are
/// local, so the zero's own allowance is consumed here.
fn charge(context: &mut SolveContext<'_, '_>, used: usize) -> Result<(), SciOxideError> {
    if context
        .try_consume(BudgetKind::Iteration, used)?
        .is_granted()
    {
        Ok(())
    } else {
        Err(exhausted())
    }
}
/// Charges the iterations of a special-function call, including a failed one,
/// before returning its result.
fn account<V: scioxide_core::NumericScalar>(
    result: Result<SpecialFunctionResult<V>, SciOxideError>,
    context: &mut SolveContext<'_, '_>,
) -> Result<SpecialFunctionResult<V>, SciOxideError> {
    let used = match &result {
        Ok(report) => report.counts().iterations(),
        Err(error) => match error.view() {
            scioxide_core::SciOxideErrorView::ConvergenceFailure { diagnostics, .. } => {
                diagnostics.counts().total()?.iterations()
            }
            _ => 0,
        },
    };
    charge(context, used)?;
    result
}

/// Real special-function result.
fn real_result(
    result: Result<SpecialFunctionResult<f64>, SciOxideError>,
    context: &mut SolveContext<'_, '_>,
) -> Result<Evaluation, SciOxideError> {
    let result = account(result, context)?;
    if result.termination() == SpecialFunctionTermination::IterationLimit {
        return Err(exhausted());
    }
    let unavailable = || missing();
    Ok(Evaluation {
        value: C::new(*result.value().ok_or_else(unavailable)?.value(), 0.0),
        derivative: C::new(*result.derivative().ok_or_else(unavailable)?.value(), 0.0),
        error: result.value().ok_or_else(unavailable)?.absolute_error(),
        derivative_error: result
            .derivative()
            .ok_or_else(unavailable)?
            .absolute_error(),
    })
}
/// Complex special-function result.
fn complex_result(
    result: Result<SpecialFunctionResult<C>, SciOxideError>,
    context: &mut SolveContext<'_, '_>,
) -> Result<Evaluation, SciOxideError> {
    let result = account(result, context)?;
    if result.termination() == SpecialFunctionTermination::IterationLimit {
        return Err(exhausted());
    }
    let unavailable = || missing();
    Ok(Evaluation {
        value: *result.value().ok_or_else(unavailable)?.value(),
        derivative: *result.derivative().ok_or_else(unavailable)?.value(),
        error: result.value().ok_or_else(unavailable)?.absolute_error(),
        derivative_error: result
            .derivative()
            .ok_or_else(unavailable)?
            .absolute_error(),
    })
}
/// Exhausted special-function work.
pub(super) fn exhausted() -> SciOxideError {
    SciOxideError::convergence_failure(
        scioxide_core::ConvergenceReason::IterationLimit,
        scioxide_core::ConvergenceDiagnostics::new(),
        "shared special-function work exhausted",
    )
}
impl Provider {
    /// One counted function evaluation: the value and derivative at `z` come
    /// from a single provider call, charged as one `Function` evaluation.
    pub(super) fn evaluate(
        self,
        z: C,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Evaluation, SciOxideError> {
        crate::context::Session::new(context)
            .invoke_context(scioxide_core::EvaluationRole::Function, |context| {
                self.at(z, context)
            })
    }
    pub(super) fn at(
        self,
        z: C,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Evaluation, SciOxideError> {
        let budget = self.allowance(context)?;
        let mut e = match self.family {
            F::BesselJ(n) | F::BesselJPrime(n) if z.im == 0.0 => real_result(
                bessel_j_scoped(n, z.re, BesselOptions::new(budget), &mut *context),
                context,
            )?,
            F::BesselY(n) | F::BesselYPrime(n) if z.im == 0.0 && z.re > 0.0 => real_result(
                bessel_y_scoped(n, z.re, BesselOptions::new(budget), &mut *context),
                context,
            )?,
            F::BesselJ(n) | F::BesselJPrime(n) => complex_result(
                bessel_j_scoped(n, z, BesselOptions::new(budget), &mut *context),
                context,
            )?,
            F::BesselY(n) | F::BesselYPrime(n) => complex_result(
                bessel_y_scoped(n, z, BesselOptions::new(budget), &mut *context),
                context,
            )?,
            F::AiryAi | F::AiryAiPrime if z.im == 0.0 => real_result(
                airy_ai_scoped(z.re, AiryOptions::new(budget), &mut *context),
                context,
            )?,
            F::AiryBi | F::AiryBiPrime if z.im == 0.0 => real_result(
                airy_bi_scoped(z.re, AiryOptions::new(budget), &mut *context),
                context,
            )?,
            F::AiryAi | F::AiryAiPrime => complex_result(
                airy_ai_scoped(z, AiryOptions::new(budget), &mut *context),
                context,
            )?,
            F::AiryBi | F::AiryBiPrime => complex_result(
                airy_bi_scoped(z, AiryOptions::new(budget), &mut *context),
                context,
            )?,
            F::Kelvin(function) => {
                let mut value = real_result(
                    match function {
                        KelvinZeroFunction::Ber => {
                            ber_scaled_scoped(z.re, KelvinOptions::new(budget), &mut *context)
                        }
                        KelvinZeroFunction::Bei => {
                            bei_scaled_scoped(z.re, KelvinOptions::new(budget), &mut *context)
                        }
                        KelvinZeroFunction::Ker => {
                            ker_scaled_scoped(z.re, KelvinOptions::new(budget), &mut *context)
                        }
                        KelvinZeroFunction::Kei => {
                            kei_scaled_scoped(z.re, KelvinOptions::new(budget), &mut *context)
                        }
                        KelvinZeroFunction::BerPrime => {
                            berp_scaled_scoped(z.re, KelvinOptions::new(budget), &mut *context)
                        }
                        KelvinZeroFunction::BeiPrime => {
                            beip_scaled_scoped(z.re, KelvinOptions::new(budget), &mut *context)
                        }
                        KelvinZeroFunction::KerPrime => {
                            kerp_scaled_scoped(z.re, KelvinOptions::new(budget), &mut *context)
                        }
                        KelvinZeroFunction::KeiPrime => {
                            keip_scaled_scoped(z.re, KelvinOptions::new(budget), &mut *context)
                        }
                    },
                    context,
                )?;
                let log_slope = match function {
                    KelvinZeroFunction::Ber
                    | KelvinZeroFunction::Bei
                    | KelvinZeroFunction::BerPrime
                    | KelvinZeroFunction::BeiPrime => -core::f64::consts::FRAC_1_SQRT_2,
                    _ => core::f64::consts::FRAC_1_SQRT_2,
                };
                // Special reports the scaled original derivative. Newton's callback
                // differentiates the normalized equation, including its multiplier.
                value.derivative += value.value * log_slope;
                value
            }
            F::Erf => complex_result(
                erf_scoped(z, ErrorFunctionOptions::new(budget), &mut *context),
                context,
            )?,
            F::FresnelC => complex_result(
                fresnel_c_scoped(z, IntegralOptions::new(budget), &mut *context),
                context,
            )?,
            F::FresnelS => complex_result(
                fresnel_s_scoped(z, IntegralOptions::new(budget), &mut *context),
                context,
            )?,
            F::FresnelCPrime | F::FresnelSPrime => {
                crate::context::Session::new(context).iteration_only()?;
                let phase = z * z * core::f64::consts::FRAC_PI_2;
                let (value, derivative) = if matches!(self.family, F::FresnelCPrime) {
                    (phase.cos(), -phase.sin() * z * core::f64::consts::PI)
                } else {
                    (phase.sin(), phase.cos() * z * core::f64::consts::PI)
                };
                Evaluation {
                    value,
                    derivative,
                    error: 8.0 * f64::EPSILON * (1.0 + phase.norm()) * value.norm().max(1.0),
                    derivative_error: 8.0
                        * f64::EPSILON
                        * (1.0 + phase.norm())
                        * derivative.norm().max(1.0),
                }
            }
        };
        let second = match self.family {
            F::BesselJPrime(n) | F::BesselYPrime(n) => Some(
                -e.derivative / z - (C::new(1.0, 0.0) - C::new(n * n, 0.0) / (z * z)) * e.value,
            ),
            F::AiryAiPrime | F::AiryBiPrime => Some(z * e.value),
            _ => None,
        };
        if let Some(second) = second {
            e = Evaluation {
                value: e.derivative,
                derivative: second,
                error: e.derivative_error,
                derivative_error: e.error * (1.0 + z.norm()) + e.derivative_error / z.norm(),
            };
        }
        if !e.value.is_finite() || !e.derivative.is_finite() || !e.error.is_finite() {
            return Err(missing());
        }
        Ok(e)
    }
    /// The request budget capped by the iteration allowance `context` has left.
    fn allowance(self, context: &SolveContext<'_, '_>) -> Result<Budget, SciOxideError> {
        match context.budget_report(BudgetKind::Iteration).remaining() {
            Some(0) => Err(exhausted()),
            Some(left) => Ok(Budget::try_from(left.min(self.budget.get()))?),
            None => Ok(self.budget),
        }
    }
}
pub(super) fn widen<S: RealScalar>(family: F<S>) -> Result<F<f64>, SciOxideError> {
    Ok(match family {
        F::BesselJ(n) => F::BesselJ(n.try_to_f64_exact()?),
        F::BesselJPrime(n) => F::BesselJPrime(n.try_to_f64_exact()?),
        F::BesselY(n) => F::BesselY(n.try_to_f64_exact()?),
        F::BesselYPrime(n) => F::BesselYPrime(n.try_to_f64_exact()?),
        F::AiryAi => F::AiryAi,
        F::AiryAiPrime => F::AiryAiPrime,
        F::AiryBi => F::AiryBi,
        F::AiryBiPrime => F::AiryBiPrime,
        F::Kelvin(k) => F::Kelvin(k),
        F::Erf => F::Erf,
        F::FresnelC => F::FresnelC,
        F::FresnelCPrime => F::FresnelCPrime,
        F::FresnelS => F::FresnelS,
        F::FresnelSPrime => F::FresnelSPrime,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_special_evaluation_debits_the_local_allowance() {
        let mut state = scioxide_core::SolveState::new();
        let mut budgets = scioxide_core::SolveBudgets::unbounded().with_limit(
            BudgetKind::Iteration,
            scioxide_core::Budget::try_from(4).unwrap(),
        );
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(1508),
            scioxide_core::algorithm_id!("rootfind", "special_work_test", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential(),
                identity,
                &mut budgets,
            )
            .unwrap();
        let error = SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new()
                .with_counts(scioxide_core::EvaluationCounts::from_iterations(3).into()),
            "special-function work exhausted",
        );
        let returned = account::<f64>(Err(error), &mut context).unwrap_err();
        assert!(
            matches!(returned.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::IterationLimit, diagnostics, ..
        } if diagnostics.counts().total().unwrap().iterations() == 3)
        );
        assert!(
            !context
                .try_consume(BudgetKind::Iteration, 2)
                .unwrap()
                .is_granted()
        );
        assert!(
            context
                .try_consume(BudgetKind::Iteration, 1)
                .unwrap()
                .is_granted()
        );
    }
}
