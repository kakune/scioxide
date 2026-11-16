//! Inverse one-norm and separation estimators that rescale to avoid overflow and underflow.

use scioxide_core::{
    AlgebraicScalar, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError,
};

/// Apply an inverse in a common binary frame. Overflow retries with a
/// right-hand side scaled down by a power of two; an unrepresentable frame asks
/// the caller to use its extended-range action. The estimator vector is
/// written only on success.
pub(crate) fn ordinary_action<S: NumericScalar>(
    rhs: &mut [ScaledValue<S>],
    scratch: &mut [S],
    mut apply: impl FnMut(&mut [S], &mut [S]) -> Result<(), SciOxideError>,
) -> Result<bool, SciOxideError> {
    let (values, work) = crate::workspace::split_mut(scratch, rhs.len())?;
    let mut exponent = rhs
        .iter()
        .filter(|x| !x.is_zero())
        .map(|x| x.scale_exponent())
        .max()
        .unwrap_or(0)
        .max(0);
    let step = i64::from(crate::kernel::standard::exponent(S::Real::MAX)? / 2);
    let mut scaled = false;
    loop {
        for (value, &source) in values.iter_mut().zip(rhs.iter()) {
            let Ok(projected) = source.shifted(-exponent)?.try_value() else {
                return Ok(false);
            };
            *value = projected;
        }
        match apply(values, work) {
            Ok(()) if values.iter().all(|x| x.has_finite_components()) => {
                // A second frame must not silently erase a small inverse
                // component. Reevaluate such actions in independent exponents.
                if scaled
                    && values
                        .iter()
                        .zip(rhs.iter())
                        .any(|(&value, source)| value == S::ZERO && !source.is_zero())
                {
                    return Ok(false);
                }
                for (destination, &value) in rhs.iter_mut().zip(values.iter()) {
                    *destination = ScaledValue::from_parts(value, exponent)?;
                }
                return Ok(true);
            }
            Err(error)
                if !matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow
                            | scioxide_core::NumericalIssue::Underflow,
                        ..
                    } | scioxide_core::SciOxideErrorView::NonFinite {
                        origin: scioxide_core::NonFiniteOrigin::Intermediate,
                        ..
                    }
                ) =>
            {
                return Err(error);
            }
            _ => {
                exponent += step;
                scaled = true;
            }
        }
    }
}

/// Extended-range and ordinary scratch shared by prepared direct-factor condition
/// actions: an inverse-action vector and a work vector of order `n` in each arena.
pub(crate) fn condition_requirements(
    n: usize,
    byte_limit: Option<usize>,
) -> Result<crate::LinalgRequirements, SciOxideError> {
    let count = scioxide_core::backend::checked_product(
        &[2, n],
        scioxide_core::quantity!("condition_scratch"),
    )?;
    Ok(crate::LinalgRequirements::new(byte_limit)
        .with_scaled(count)
        .with_scalar(crate::workspace::ScalarRegion::Direct, count))
}

/// Facts of one condition query, established before its solve scope begins.
pub(crate) struct ConditionQuery<R: RealScalar> {
    /// Identity of the query's solve scope.
    pub(crate) algorithm: scioxide_core::AlgorithmId,
    /// Order of the inverse actions; an empty factor has no condition estimate.
    pub(crate) order: usize,
    /// Original-coordinate one-norm.
    pub(crate) norm: Scaled<R>,
    /// Structurally singular factors report zero without inverse actions.
    pub(crate) singular: bool,
}

/// One inverse action requested by the estimator: `rhs` in extended range, an
/// extended-range work vector and ordinary scratch, each of the query order.
pub(crate) struct InverseAction<'a, S: NumericScalar> {
    pub(crate) rhs: &'a mut [ScaledValue<S>],
    pub(crate) work: &'a mut [ScaledValue<S>],
    pub(crate) ordinary: &'a mut [S],
    pub(crate) adjoint: bool,
}
impl<S: NumericScalar> InverseAction<'_, S> {
    /// Apply `ordinary` in a common binary frame when that preserves the action,
    /// otherwise the extended-range `scaled` inverse.
    pub(crate) fn solve(
        self,
        ordinary: impl FnMut(&mut [S], &mut [S]) -> Result<(), SciOxideError>,
        scaled: impl FnOnce(&mut [ScaledValue<S>], &mut [ScaledValue<S>]) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        if ordinary_action(self.rhs, self.ordinary, ordinary)? {
            return Ok(());
        }
        scaled(self.rhs, self.work)
    }
}

/// Estimate the original-coordinate rcond₁ of a prepared factor in its own solve scope.
///
/// The call records a fresh workspace attempt and applies `options` to `context`.
/// `query`, an empty factor and the scratch reservation are rejected before the
/// epoch begins; each inverse action then counts one iteration of the local
/// allowance, so budgets, deadlines and nested counts follow the attempt.
pub(crate) fn scoped_condition<'capability, S: NumericScalar>(
    options: &crate::LinalgExecutionOptions,
    workspace: &mut crate::LinalgWorkspace<'_, S>,
    context: crate::LinalgExecutionContext<'_, 'capability>,
    query: impl FnOnce() -> Result<ConditionQuery<S::Real>, SciOxideError>,
    mut inverse: impl FnMut(
        InverseAction<'_, S>,
        crate::KernelContext<'capability>,
    ) -> Result<(), SciOxideError>,
) -> Result<crate::ConditionEstimate<S::Real>, SciOxideError> {
    workspace.begin_attempt();
    let context = context.with_options(options)?;
    context.execution().check()?;
    let ConditionQuery {
        algorithm,
        order,
        norm,
        singular,
    } = query()?;
    if order == 0 {
        return Err(crate::analysis::empty_condition());
    }
    workspace.try_reserve(&condition_requirements(order, options.byte_limit())?)?;
    workspace.run_scope(
        context,
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            algorithm,
            order as u128,
            0,
        ),
        options.solve_budgets(),
        |workspace, kernels, solve| {
            workspace.with_scaled(|storage, _, child| {
                let (rhs, work) = crate::workspace::split_mut(storage, order)?;
                let work = &mut work[..order];
                let ordinary = child.scalar_mut(crate::workspace::ScalarRegion::Direct);
                reciprocal_condition(norm, singular, rhs, |rhs, adjoint| {
                    crate::solve_support::admit_iteration(
                        Some(kernels.execution()),
                        Some(&mut *solve),
                        "condition action budget exhausted",
                    )?;
                    inverse(
                        InverseAction {
                            rhs,
                            work: &mut *work,
                            ordinary: &mut *ordinary,
                            adjoint,
                        },
                        kernels,
                    )?;
                    Ok(true)
                })
            })
        },
    )
}

/// Sum of scalar magnitudes without requiring the norm to fit the scalar range.
pub(crate) fn vector_one_norm<S: NumericScalar>(
    values: &[S],
) -> Result<Scaled<S::Real>, SciOxideError> {
    let ordinary = values
        .iter()
        .fold(S::Real::ZERO, |sum, value| sum + value.abs());
    if ordinary.is_finite() {
        return Scaled::from_value(ordinary);
    }
    values.iter().try_fold(Scaled::ZERO, |sum, &value| {
        sum.add(ScaledValue::from_value(value)?.magnitude()?)
    })
}

/// Inverse actions remain extended-range until the final reciprocal is rounded.
pub(crate) fn reciprocal_condition<S: NumericScalar>(
    norm: Scaled<S::Real>,
    singular: bool,
    rhs: &mut [ScaledValue<S>],
    apply: impl FnMut(&mut [ScaledValue<S>], bool) -> Result<bool, SciOxideError>,
) -> Result<crate::ConditionEstimate<S::Real>, SciOxideError> {
    let reciprocal = if singular || norm.is_zero() {
        S::Real::ZERO
    } else {
        let inverse = operator_one_norm(rhs, apply)?;
        if !inverse.complete || inverse.value.is_zero() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::IllConditioned,
                "inverse one-norm estimation did not complete",
            ));
        }
        Scaled::ONE
            .div(norm.mul(inverse.value)?)?
            .round()?
            .max(S::Real::MIN_POSITIVE * S::Real::EPSILON)
            .min(S::Real::ONE)
    };
    crate::ConditionEstimate::new(reciprocal, scioxide_core::EstimateReliability::Heuristic)
}

/// Bounded Hager–Higham reciprocal inverse one-norm estimate for Schur-based solvers.
///
pub(crate) fn separation_estimate<V: NumericScalar>(
    rhs: &mut [ScaledValue<V>],
    apply: impl FnMut(&mut [ScaledValue<V>], bool) -> Result<bool, SciOxideError>,
) -> Result<Option<Scaled<V::Real>>, SciOxideError> {
    let estimate = operator_one_norm(rhs, apply)?;
    if estimate.complete && !estimate.value.is_zero() {
        Ok(Some(Scaled::from_value(V::Real::ONE)?.div(estimate.value)?))
    } else {
        Ok(None)
    }
}

/// Best completed lower estimate and whether the estimator finished its checks.
pub(crate) struct NormEstimate<R: RealScalar> {
    pub(crate) value: Scaled<R>,
    pub(crate) complete: bool,
}

/// Bounded forward/adjoint one-norm estimate; zero operators have zero norm.
///
pub(crate) fn operator_one_norm<V: NumericScalar>(
    rhs: &mut [ScaledValue<V>],
    mut apply: impl FnMut(&mut [ScaledValue<V>], bool) -> Result<bool, SciOxideError>,
) -> Result<NormEstimate<V::Real>, SciOxideError> {
    let count = rhs.len();
    // Small operators need fewer direct column solves than an iterative estimate.
    // Evaluate their inverse one-norm without storing an inverse or Kronecker matrix.
    if count <= 4 {
        let mut maximum = Scaled::ZERO;
        for column in 0..count {
            rhs.fill(ScaledValue::ZERO);
            rhs[column] = ScaledValue::from_value(V::from_components(
                V::Real::ONE,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
            ))?;
            if !apply(rhs, false)? {
                return Ok(NormEstimate {
                    value: maximum,
                    complete: false,
                });
            }
            let mut sum = Scaled::ZERO;
            for &value in rhs.iter() {
                sum = sum.add(value.magnitude()?)?;
            }
            if !sum.abs_le(maximum) {
                maximum = sum;
            }
        }
        return Ok(NormEstimate {
            value: maximum,
            complete: true,
        });
    }
    let dimension = crate::checked::rounded_count::<V::Real>(count)?;
    rhs.fill(ScaledValue::from_value(V::from_components(
        V::Real::ONE / dimension,
        <_ as scioxide_core::AlgebraicScalar>::ZERO,
    ))?);
    let mut best = Scaled::ZERO;
    let mut previous_index = None;
    // Hager/Higham inverse one-norm estimation, using Sylvester solves and their
    // Frobenius adjoints. Eight iterations bound estimator work independently of n.
    for _ in 0..8 {
        if !apply(rhs, false)? {
            return Ok(NormEstimate {
                value: best,
                complete: false,
            });
        }
        let mut estimate = Scaled::ZERO;
        for value in rhs.iter_mut() {
            let magnitude = value.magnitude()?;
            estimate = estimate.add(magnitude)?;
            *value = if magnitude.is_zero() {
                ScaledValue::from_value(V::from_components(
                    V::Real::ONE,
                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                ))?
            } else {
                value.div(ScaledValue::from_components(magnitude, Scaled::ZERO))?
            };
        }
        if estimate.abs_le(best) {
            break;
        }
        best = estimate;
        if !apply(rhs, true)? {
            return Ok(NormEstimate {
                value: best,
                complete: false,
            });
        }
        let mut index = 0;
        let mut maximum = Scaled::ZERO;
        for (i, value) in rhs.iter().enumerate() {
            let magnitude = value.magnitude()?;
            if !magnitude.abs_le(maximum) {
                maximum = magnitude;
                index = i;
            }
        }
        if previous_index == Some(index) {
            break;
        }
        previous_index = Some(index);
        rhs.fill(ScaledValue::ZERO);
        rhs[index] = ScaledValue::from_value(V::from_components(
            V::Real::ONE,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        ))?;
    }
    // Higham's alternating ramp guards against an uninformative initial vector.
    for (i, value) in rhs.iter_mut().enumerate() {
        let magnitude = ramp_weight::<V::Real>(i, count)?;
        *value = ScaledValue::from_value(V::from_components(
            if i % 2 == 0 { magnitude } else { -magnitude },
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        ))?;
    }
    if !apply(rhs, false)? {
        return Ok(NormEstimate {
            value: best,
            complete: false,
        });
    }
    let mut estimate = Scaled::ZERO;
    for &value in rhs.iter() {
        estimate = estimate.add(value.magnitude()?)?;
    }
    // The ramp's 1-norm is roughly `3n/2`; dividing by it turns the result
    // into a lower bound on the operator norm comparable with `best`.
    let divisor = if count == 1 {
        V::Real::ONE
    } else {
        dimension * (V::Real::TWO + V::Real::ONE) / (V::Real::TWO)
    };
    estimate = estimate.div(Scaled::from_value(divisor)?)?;
    if !estimate.abs_le(best) {
        best = estimate;
    }
    Ok(NormEstimate {
        value: best,
        complete: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangular_extreme_range_retries_the_original_estimator_vector() -> Result<(), SciOxideError>
    {
        let a = crate::Matrix::from_vec(
            3,
            3,
            alloc::vec![1e-300_f64, 1e300, 0., 0., 1e-300, 1e300, 0., 0., 1e-300,],
        )?;
        let mut rhs = [ScaledValue::ZERO; 3];
        let mut ordinary = [0.; 6];
        let mut fallbacks = 0;
        let inverse = operator_one_norm(&mut rhs, |values, adjoint| {
            if !ordinary_action(values, &mut ordinary, |values, _| {
                crate::kernel::triangular::solve(
                    crate::kernel::TriangularOperand {
                        view: crate::kernel::triangular::Storage::Dense(a.view()),
                        triangle: scioxide_core::Triangle::Upper,
                        op: if adjoint {
                            crate::kernel::GemmOp::Adjoint
                        } else {
                            crate::kernel::GemmOp::None
                        },
                        diagonal: scioxide_core::DiagonalKind::Explicit,
                    },
                    1e300,
                    values,
                    2,
                    crate::KernelContext::from(scioxide_core::ExecutionContext::sequential())
                        .dispatch(),
                )
            })? {
                fallbacks += 1;
                crate::diagnostics::matrix_substitute(
                    a.view(),
                    scioxide_core::Triangle::Upper,
                    scioxide_core::DiagonalKind::Explicit,
                    adjoint,
                    values,
                )?;
            }
            Ok(true)
        })?;
        // Last column of A^-1: [t²/s³, -t/s², 1/s].
        let s = Scaled::from_value(1e-300)?;
        let t = Scaled::from_value(1e300)?;
        let exact = t
            .square()?
            .div(s.square()?.mul(s)?)?
            .add(t.div(s.square()?)?)?
            .add(Scaled::ONE.div(s)?)?;
        assert!(fallbacks > 0);
        assert!(inverse.complete);
        assert!((inverse.value.div(exact)?.round()? - 1.).abs() < 32. * f64::EPSILON);
        let norm = crate::diagnostics::matrix_one_norm(a.view())?;
        let exact_rcond = Scaled::ONE
            .div(norm.mul(exact)?)?
            .round()?
            .max(f64::from_bits(1));
        let estimate = crate::TriangularPlan::from_dense(
            a.view(),
            scioxide_core::Triangle::Upper,
            scioxide_core::DiagonalKind::Explicit,
        )?
        .condition_estimate()?;
        assert_eq!(estimate.reciprocal().value(), exact_rcond);
        Ok(())
    }

    #[test]
    fn scaled_ordinary_action_preserves_an_inverse_beyond_scalar_range() -> Result<(), SciOxideError>
    {
        let mut rhs = [ScaledValue::from_value(1_f64)?];
        let mut scratch = [0.; 2];
        let mut attempts = 0;
        assert!(ordinary_action(&mut rhs, &mut scratch, |values, _| {
            attempts += 1;
            values[0] = crate::checked::finite_candidate(values[0] / 1e-310)?;
            Ok(())
        })?);
        assert!(attempts > 1);
        let exact = Scaled::ONE.div(Scaled::from_value(1e-310)?)?;
        assert!((rhs[0].real().div(exact)?.round()? - 1.).abs() < 8. * f64::EPSILON);
        Ok(())
    }
    #[test]
    fn estimator_action_count_is_independent_of_order() -> Result<(), scioxide_core::SciOxideError>
    {
        for n in [5, 17, 60, 400] {
            let mut work = alloc::vec![scioxide_core::ScaledValue::<f64>::ZERO;n];
            let mut calls = 0;
            let result = super::operator_one_norm(&mut work, |values, _| {
                calls += 1;
                for (i, value) in values.iter_mut().enumerate() {
                    *value = value.mul(scioxide_core::ScaledValue::from_value((i + 1) as f64)?)?;
                }
                Ok(true)
            })?;
            assert!(result.complete);
            assert!(calls <= 17);
            assert!((result.value.round()? - n as f64).abs() < 1e-12);
        }
        Ok(())
    }
}

/// Higham's alternating-ramp magnitude; counts are numerical weights.
fn ramp_weight<R: RealScalar>(index: usize, count: usize) -> Result<R, SciOxideError> {
    Ok(R::ONE
        + crate::checked::rounded_count::<R>(index)?
            / crate::checked::rounded_count::<R>(count.saturating_sub(1).max(1))?)
}

#[cfg(test)]
mod large_order_tests {
    use super::*;
    #[test]
    fn binary32_estimator_weights_accept_inexact_orders() {
        for count in [1 << 23, (1 << 24) + 1, usize::MAX] {
            let initial = 1_f32 / crate::checked::rounded_count::<f32>(count).unwrap();
            assert!(initial.is_finite() && initial > 0.);
            assert_eq!(ramp_weight::<f32>(0, count).unwrap(), 1.);
            assert_eq!(ramp_weight::<f32>(count - 1, count).unwrap(), 2.);
            assert!(
                (ramp_weight::<f32>(count / 2, count).unwrap() - 1.5).abs() <= 2. * f32::EPSILON
            );
        }
    }
}
