//! Real evaluation adapters and family accuracy envelopes.

use crate::result::{Bound, Evaluation, Ulps};
use crate::{BesselOptions, SpecialFunctionTermination};
use scioxide_core::{
    AlgorithmComponent, AlgorithmId, Complex, NumericScalar, RealScalar, SciOxideError,
    algorithm_component, algorithm_id,
};

/// Error envelopes of the fixed-accuracy kernels, in ulps of the output type, checked by
/// `tests/accuracy_oracle.rs` and `reported_errors_cover_oracle_references`.
pub(crate) const BESSEL_ULPS: Ulps = Ulps::new(64.0, 64.0);
pub(crate) fn bessel_j<S: RealScalar>(order: S, x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    cylindrical_evaluate(order, x, options, None, false, false)?.value()
}

pub(crate) fn bessel_y<S: RealScalar>(order: S, x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    cylindrical_evaluate(order, x, options, None, true, false)?.value()
}

pub(crate) fn bessel_i<S: RealScalar>(order: S, x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_i_evaluate(order, x, options, None)?.value()
}

pub(crate) fn bessel_i_scaled<S: RealScalar>(
    order: S,
    x: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_i_scaled_evaluate(order, x, options, None)?.value()
}

pub(crate) fn bessel_k<S: RealScalar>(order: S, x: S) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_k_evaluate(order, x, options, None)?.value()
}

pub(crate) fn bessel_k_scaled<S: RealScalar>(
    order: S,
    x: S,
) -> Result<S, scioxide_core::SciOxideError> {
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    bessel_k_scaled_evaluate(order, x, options, None)?.value()
}

pub(crate) fn hankel_1<S>(order: S, x: S) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    S: RealScalar,
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    hankel_evaluate(order, x, options, Hankel::First, None)?.value()
}

#[inline]
pub(crate) fn hankel_1_evaluation<S>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    S: RealScalar,
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    hankel_evaluate(order, x, options, Hankel::First, context)
}

pub(crate) fn hankel_2<S>(order: S, x: S) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    S: RealScalar,
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BesselOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    hankel_evaluate(order, x, options, Hankel::Second, None)?.value()
}

#[inline]
pub(crate) fn hankel_2_evaluation<S>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    S: RealScalar,
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    hankel_evaluate(order, x, options, Hankel::Second, context)
}

fn bessel_result<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    family: AlgorithmComponent,
    second: bool,
    derivative: bool,
) -> Result<Evaluation<S>, SciOxideError> {
    const DISPATCH: AlgorithmComponent = algorithm_component!("budgeted_regime_dispatch");
    crate::support::require_finite_param(order, "Bessel parameter order must be finite")?;
    crate::support::require_finite_param(x, "Bessel parameter x must be finite")?;
    let mut work =
        crate::support::Work::with_batched_context(options.iteration_budget().get(), context)?;
    let low = x > S::ZERO && (order == S::ZERO || order == S::ONE);
    let mut adjacent = None;
    let result = if low {
        super::bessel_low::pair(
            x.into_f64(),
            second,
            usize::from(order == S::ONE),
            derivative,
            &mut work,
        )
        .and_then(|pair| {
            adjacent = Some(pair);
            crate::support::check_finite(pair[usize::from(order == S::ONE)])
        })
    } else if second {
        super::bessel::bessel_y(order.into_f64(), x.into_f64(), &mut work)
    } else {
        super::bessel::bessel_j(order.into_f64(), x.into_f64(), &mut work)
    };
    work.check()?;
    let wide_value = match result {
        Ok(value) => value,
        Err(_) if !work.complete => {
            return Err(crate::result::exhausted_work(work.used));
        }
        Err(error) => return Err(error),
    };
    if wide_value.is_infinite() {
        return Ok(Evaluation::limit(
            if wide_value.is_sign_negative() {
                S::NEG_INFINITY
            } else {
                S::INFINITY
            },
            AlgorithmId::from_components(family, algorithm_component!("endpoint"), 1),
        )
        .with_counts(scioxide_core::EvaluationCounts::from_iterations(work.used)));
    }
    let value = crate::support::project_real::<S>(wide_value)?;
    let derivative = if derivative && x > S::ZERO && order >= -S::HALF {
        let upper = if let Some(pair) = adjacent {
            Ok(if order == S::ZERO {
                pair[1]
            } else {
                pair[1] / x.into_f64() - pair[0]
            })
        } else if second {
            super::bessel::bessel_y(order.into_f64() + 1.0, x.into_f64(), &mut work)
        } else {
            super::bessel::bessel_j(order.into_f64() + 1.0, x.into_f64(), &mut work)
        };
        work.check()?;
        match upper {
            Ok(upper) => {
                let term = if low {
                    0.0
                } else {
                    (order.into_f64() / x.into_f64()) * wide_value
                };
                let slope = term - upper;
                let magnitude = if let Some(pair) = adjacent {
                    if order == S::ZERO {
                        pair[1].abs()
                    } else {
                        pair[0].abs() + (pair[1] / x.into_f64()).abs()
                    }
                } else {
                    term.abs() + upper.abs()
                };
                let error = Bound::envelope(magnitude, BESSEL_ULPS)?.absolute
                    + S::EPSILON.into_f64() * (term.abs() + slope.abs());
                if slope.is_finite()
                    && S::try_from_f64_lossy(slope).is_ok()
                    && error.is_finite()
                    && error <= S::MAX.into_f64()
                {
                    Some((
                        S::try_from_f64_lossy(slope)?,
                        Bound::calibrated_absolute(S::try_from_f64_lossy(error)?)
                            .with_subnormal_rounding(
                                S::try_from_f64_lossy(slope)?.abs(),
                                S::try_from_f64_exact(4.0)?,
                            ),
                    ))
                } else {
                    None
                }
            }
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    }
                ) =>
            {
                None
            }
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    Ok(crate::expansions::bounded(
        value,
        Bound::envelope(value, BESSEL_ULPS)?,
        AlgorithmId::from_components(family, DISPATCH, 1),
        work.used,
        true,
    )?
    .with_derivative(derivative))
}

/// Evaluates the detailed result before scalar projection.
pub(crate) fn bessel_j_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    cylindrical_evaluate(order, x, options, context, false, true)
}

/// Paired reports share one work allowance; plain values omit adjacent-order work.
fn cylindrical_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    second: bool,
    derivative: bool,
) -> Result<Evaluation<S>, SciOxideError> {
    let family = if second {
        algorithm_component!("bessel_y")
    } else {
        algorithm_component!("bessel_j")
    };
    if order >= S::try_from_f64_exact(1000.0)?
        && x > S::ZERO
        && order.into_f64() > 2.0 * x.into_f64().sqrt()
    {
        return large_result(order, x, options, context, family, (second, derivative));
    }
    bessel_result(order, x, options, context, family, second, derivative)
}

/// Evaluates the detailed result before scalar projection.
pub(crate) fn bessel_y_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    cylindrical_evaluate(order, x, options, context, true, true)
}

/// Evaluates the detailed result before scalar projection.
pub(crate) fn bessel_i_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    modified_result(
        order,
        x,
        options,
        context,
        algorithm_component!("bessel_i"),
        (false, false),
    )
}

/// Evaluates the detailed result before scalar projection.
pub(crate) fn bessel_i_scaled_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    modified_result(
        order,
        x,
        options,
        context,
        algorithm_component!("bessel_i_scaled"),
        (false, true),
    )
}

/// Evaluates the detailed result before scalar projection.
pub(crate) fn bessel_k_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    modified_result(
        order,
        x,
        options,
        context,
        algorithm_component!("bessel_k"),
        (true, false),
    )
}

/// Evaluates the detailed result before scalar projection.
pub(crate) fn bessel_k_scaled_evaluate<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<S>, SciOxideError> {
    modified_result(
        order,
        x,
        options,
        context,
        algorithm_component!("bessel_k_scaled"),
        (true, true),
    )
}

fn modified_result<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    family: AlgorithmComponent,
    (second, scaled): (bool, bool),
) -> Result<Evaluation<S>, SciOxideError> {
    let mut work = crate::support::Work::with_context(options.iteration_budget().get(), context);
    let computed =
        super::bessel::modified(order.into_f64(), x.into_f64(), second, scaled, &mut work);
    work.check()?;
    let e = computed?;
    let algorithm =
        AlgorithmId::from_components(family, algorithm_component!("modified_regime_dispatch"), 1);
    if x == S::ZERO && second {
        return Evaluation::limit(f64::INFINITY, algorithm)
            .with_counts(scioxide_core::EvaluationCounts::from_iterations(work.used))
            .narrow();
    }
    estimated_result(e, algorithm, work.used)
}

fn large_result<S: RealScalar>(
    order: S,
    x: S,
    options: BesselOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
    family: AlgorithmComponent,
    (second, derivative): (bool, bool),
) -> Result<Evaluation<S>, SciOxideError> {
    crate::support::require_finite_param(order, "Bessel parameter order must be finite")?;
    crate::support::require_finite_param(x, "Bessel parameter x must be finite")?;
    let mut work = crate::support::Work::with_context(options.iteration_budget().get(), context);
    let e = super::bessel_uniform::evaluate(order.into_f64(), x.into_f64(), second, &mut work)?;
    let derivative = if derivative {
        let next = super::bessel_uniform::evaluate(
            order.into_f64() + 1.0,
            x.into_f64(),
            second,
            &mut work,
        )?;
        let ratio = order.into_f64() / x.into_f64();
        let term = ratio * e.value;
        let value = term - next.value;
        let error = ratio.abs() * e.error
            + next.error
            + 4.0 * f64::EPSILON * (term.abs() + next.value.abs());
        if value.is_finite() && error.is_finite() {
            Some((
                value,
                Bound::calibrated_absolute(error).with_subnormal_rounding(value.abs(), 4.0),
            ))
        } else {
            None
        }
    } else {
        None
    };
    work.check()?;
    // Attach before narrowing so logarithmic range information and derivative
    // projection use the same binary64 evaluation.
    let algorithm =
        AlgorithmId::from_components(family, algorithm_component!("uniform_large_order"), 1);
    estimated_native(e, algorithm, work.used)?
        .with_derivative(derivative)
        .narrow()
}

fn estimated_result<S: RealScalar>(
    e: super::bessel::RealEstimate,
    algorithm: AlgorithmId,
    used: usize,
) -> Result<Evaluation<S>, SciOxideError> {
    estimated_native(e, algorithm, used)?.narrow()
}

fn estimated_native(
    e: super::bessel::RealEstimate,
    algorithm: AlgorithmId,
    used: usize,
) -> Result<Evaluation<f64>, SciOxideError> {
    let result = if let Some(logarithm) = e.logarithm.filter(|_| e.relative < 0.5) {
        Evaluation::logarithmic(
            e.sign,
            logarithm.value(),
            Bound::calibrated_absolute(e.relative + 2.0 * f64::EPSILON * logarithm.value().abs()),
            crate::result::Projection::Direct(
                e.value.abs(),
                Bound::calibrated_absolute(e.error).with_subnormal_rounding(e.value.abs(), 4.0),
            ),
            algorithm,
            used,
        )
    } else {
        crate::expansions::bounded(
            e.value,
            Bound::calibrated_absolute(e.error),
            algorithm,
            used,
            true,
        )?
    };
    let result = if e.value == 0.0 && e.logarithm.is_some() {
        result.with_projection(Some(e.value), crate::ScalarRepresentation::Finite)
    } else {
        result
    };
    Ok(result)
}

/// Hankel function kind.
#[derive(Clone, Copy)]
pub(crate) enum Hankel {
    First,
    Second,
}

/// Evaluates the detailed result before scalar projection.
pub(crate) fn hankel_evaluate<S>(
    order: S,
    x: S,
    options: BesselOptions,
    kind: Hankel,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<Evaluation<Complex<S>>, SciOxideError>
where
    S: RealScalar,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let algorithm = match kind {
        Hankel::First => algorithm_id!("hankel_1", "budgeted_jy_pair", 1),
        Hankel::Second => algorithm_id!("hankel_2", "budgeted_jy_pair", 1),
    };
    crate::support::require_finite_param(order, "Bessel parameter order must be finite")?;
    crate::support::require_finite_param(x, "Bessel parameter x must be finite")?;
    let mut work = crate::support::Work::with_context(options.iteration_budget().get(), context);
    let evaluated = match kind {
        Hankel::First => crate::bessel_airy::bessel::hankel_1(order, x, &mut work),
        Hankel::Second => crate::bessel_airy::bessel::hankel_2(order, x, &mut work),
    };
    work.check()?;
    let value = match evaluated {
        Ok(value) => value,
        Err(_) if !work.complete => Complex::new(S::ZERO, S::ZERO),
        Err(error) => return Err(error),
    };
    if !work.complete {
        return Err(crate::result::exhausted_work(work.used));
    }
    // The modulus envelope bounds both the J and Y components.
    Ok(Evaluation::finite(
        value,
        Bound::envelope(value.abs(), BESSEL_ULPS)?,
        None,
        algorithm,
        work.used,
        SpecialFunctionTermination::Converged,
    ))
}
