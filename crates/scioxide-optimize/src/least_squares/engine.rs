//! Linearization and all-or-nothing trial acceptance for Gauss–Newton and
//! Levenberg–Marquardt.
use super::{LeastSquaresGlobalization, LeastSquaresOptions, options};
use crate::context::SolveSession;
use crate::least_squares::LeastSquaresStorage;
use crate::{LeastSquaresProblem, LeastSquaresResult, OptimizeTermination};
use core::cmp::Ordering;

use scioxide_core::{Budget, EvaluationRole, NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledAccumulator};
use scioxide_differentiate::DerivativeStrategy;

pub(crate) fn validate<S, P>(problem: &P) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let group = problem.residual_group_size();
    if !matches!(group, 1 | 2) || !problem.residual_dim().is_multiple_of(group) {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("problem"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "residual groups must have one or two components",
        ));
    }
    for i in 0..problem.param_dim() {
        if let Some(scale) = problem.parameter_scale(i) {
            options::positive(scale, scioxide_core::argument!("problem"))?;
        }
    }
    for i in 0..problem.residual_dim() / group {
        options::positive(
            problem.residual_scale(i),
            scioxide_core::argument!("problem"),
        )?;
        let weight = problem.residual_weight(i);
        if !weight.is_finite() || weight < <S as scioxide_core::AlgebraicScalar>::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("problem"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "weights must be finite and nonnegative",
            ));
        }
    }
    Ok(())
}

pub(crate) fn cost<S, P>(
    problem: &P,
    residual: &[S],
    options: &LeastSquaresOptions<S>,
) -> Result<Scaled<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    crate::utils::validate_finite_origin(residual, scioxide_core::NonFiniteOrigin::Callback)?;
    if let Some(value) = super::kernels::cost(problem, residual, options) {
        return Scaled::from_value(value);
    }
    let group = problem.residual_group_size();
    let mut sum = ScaledAccumulator::ZERO;
    for (index, components) in residual.chunks_exact(group).enumerate() {
        let scale = Scaled::from_value(problem.residual_scale(index))?;
        let weight = Scaled::from_value(problem.residual_weight(index))?;
        let mut square = ScaledAccumulator::ZERO;
        for &value in components {
            let value = Scaled::from_value(value)?.div(scale)?;
            square.add_product(value, value)?;
        }
        let magnitude = square.total()?.sqrt()?;
        let (loss, _) = options::loss_value(options.loss, magnitude, options.loss_scale)?;
        sum.add_scaled(weight.mul(loss)?)?;
    }
    sum.total()
}

/// Original IRLS weights and independent radial curvature of each observation.
pub(super) fn model_weights<S, P>(
    problem: &P,
    residual: &[S],
    options: &LeastSquaresOptions<S>,
    weights: &mut [Scaled<S>],
    radial: &mut [Scaled<S>],
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let one = Scaled::ONE;
    let epsilon = Scaled::from_value(S::EPSILON)?;
    let group = problem.residual_group_size();
    for (i, values) in residual.chunks_exact(group).enumerate() {
        let weight = problem.residual_weight(i);
        if weight == <S as scioxide_core::AlgebraicScalar>::ZERO {
            weights[i * group..(i + 1) * group].fill(Scaled::ZERO);
            radial[i * group..(i + 1) * group].fill(Scaled::ZERO);
            continue;
        }
        let scale = problem.residual_scale(i);
        let a = super::kernels::quotient(values[0], scale);
        let native = a
            .and_then(|a| {
                if group == 2 {
                    super::kernels::quotient(values[1], scale).map(|b| a.hypot(b))
                } else {
                    Some(NumericScalar::abs(a))
                }
            })
            .and_then(|m| super::kernels::loss(m, options));
        let (original, curvature) = if let Some((_, first, curvature)) = native {
            let original = weight.sqrt() * first.sqrt() / scale;
            let curvature = weight.sqrt() * curvature.sqrt() / scale;
            if original.is_finite()
                && curvature.is_finite()
                && original >= S::MIN_POSITIVE
                && curvature >= S::MIN_POSITIVE
            {
                (
                    Scaled::from_value(original)?,
                    Scaled::from_value(curvature)?,
                )
            } else {
                extended_weights(values, weight, scale, options, epsilon, one)?
            }
        } else {
            extended_weights(values, weight, scale, options, epsilon, one)?
        };
        weights[i * group..(i + 1) * group].fill(original);
        radial[i * group..(i + 1) * group].fill(curvature);
    }
    Ok(())
}

fn extended_weights<S: RealScalar>(
    values: &[S],
    weight: S,
    scale: S,
    options: &LeastSquaresOptions<S>,
    epsilon: Scaled<S>,
    one: Scaled<S>,
) -> Result<(Scaled<S>, Scaled<S>), SciOxideError> {
    let scale = Scaled::from_value(scale)?;
    let mut square = ScaledAccumulator::ZERO;
    for &r in values {
        let r = Scaled::from_value(r)?.div(scale)?;
        square.add_product(r, r)?;
    }
    let magnitude = square.total()?.sqrt()?;
    let (_, derivative) = options::loss_value(options.loss, magnitude, options.loss_scale)?;
    let z = magnitude.div(Scaled::from_value(options.loss_scale)?)?;
    let z2 = z.mul(z)?;
    let curvature = match options.loss {
        super::LeastSquaresLoss::Quadratic => one,
        super::LeastSquaresLoss::Huber if z.cmp_value(one) != Ordering::Greater => one,
        super::LeastSquaresLoss::Huber => epsilon,
        super::LeastSquaresLoss::SoftL1 => {
            let first = derivative.mul(derivative)?;
            first.mul(first)?.mul(first)?
        }
        super::LeastSquaresLoss::Cauchy => one.sub(z2)?.div(one.add(z2)?.mul(one.add(z2)?)?)?,
    };
    let curvature = if curvature.cmp_value(epsilon) == Ordering::Less {
        epsilon
    } else {
        curvature
    };
    let root_weight = Scaled::from_value(weight)?.sqrt()?.div(scale)?;
    Ok((
        root_weight.mul(derivative)?,
        root_weight.mul(curvature.sqrt()?)?,
    ))
}

#[cold]
fn extended_linearization<S, P>(
    problem: &P,
    w: &mut LeastSquaresStorage<S>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let n = w.parameter_dimension;
    let mut norm = ScaledAccumulator::ZERO;
    let mut residual_norm = ScaledAccumulator::ZERO;
    for (&r, &weight) in w.residual.iter().zip(&w.row_weights) {
        let value = Scaled::from_value(r)?.mul(weight)?;
        residual_norm.add_product(value, value)?;
    }
    let residual_norm = residual_norm.total()?.sqrt()?;
    w.orthogonality = <S as scioxide_core::AlgebraicScalar>::ZERO;
    for column in 0..n {
        let mut gradient = ScaledAccumulator::ZERO;
        let mut column_norm = ScaledAccumulator::ZERO;
        for (row, &r) in w.residual.iter().enumerate() {
            let j = Scaled::from_value(w.jacobian.as_slice()[row * n + column])?
                .mul(w.row_weights[row])?;
            gradient.add_scaled(j.mul(Scaled::from_value(r)?)?.mul(w.row_weights[row])?)?;
            column_norm.add_product(j, j)?;
        }
        let gradient = gradient.total()?;
        norm.add_product(gradient, gradient)?;
        let column_norm = column_norm.total()?.sqrt()?;
        if !column_norm.is_zero() && !residual_norm.is_zero() {
            w.orthogonality = w.orthogonality.max(
                gradient
                    .abs()
                    .div(column_norm)?
                    .div(residual_norm)?
                    .round()?,
            );
        }
    }
    w.gradient_norm = Some(norm.total()?.sqrt()?);
    let group = problem.residual_group_size();
    // Normalize exceptional models before projecting them into scalar storage.
    let mut normalizer = Scaled::ONE;
    for (i, &j) in w.jacobian.as_slice().iter().enumerate() {
        let value = Scaled::from_value(j)?.mul(w.curvature[i / n])?.abs();
        if value.cmp_value(normalizer) == Ordering::Greater {
            normalizer = value;
        }
    }
    for start in (0..w.residual_dimension).step_by(group) {
        let original = w.row_weights[start];
        let radial = w.curvature[start];
        if original.is_zero() {
            w.weighted_residual[start..start + group]
                .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
            w.model_jacobian.as_mut_slice()[start * n..(start + group) * n]
                .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
            continue;
        }
        let factor = original.mul(original)?.div(radial)?;
        for i in start..start + group {
            w.weighted_residual[i] = Scaled::from_value(w.residual[i])?
                .mul(factor)?
                .div(normalizer)?
                .round()?;
        }
        let mut magnitude = ScaledAccumulator::ZERO;
        for &r in &w.residual[start..start + group] {
            let r = Scaled::from_value(r)?;
            magnitude.add_product(r, r)?;
        }
        let magnitude = magnitude.total()?.sqrt()?;
        for column in 0..n {
            let mut projection = ScaledAccumulator::ZERO;
            if !magnitude.is_zero() {
                for i in start..start + group {
                    projection.add_product(
                        Scaled::from_value(w.residual[i])?.div(magnitude)?,
                        Scaled::from_value(w.jacobian.as_slice()[i * n + column])?,
                    )?;
                }
            }
            let correction = radial.sub(original)?.mul(projection.total()?)?;
            for i in start..start + group {
                let value = if group == 1 {
                    Scaled::from_value(w.jacobian.as_slice()[i * n + column])?.mul(radial)?
                } else {
                    let u = if magnitude.is_zero() {
                        Scaled::ZERO
                    } else {
                        Scaled::from_value(w.residual[i])?.div(magnitude)?
                    };
                    Scaled::from_value(w.jacobian.as_slice()[i * n + column])?
                        .mul(original)?
                        .add(u.mul(correction)?)?
                };
                w.model_jacobian.as_mut_slice()[i * n + column] = value.div(normalizer)?.round()?;
            }
        }
    }
    w.model_scale = normalizer;
    Ok(())
}

fn dense_linearization<S, P>(
    problem: &P,
    strategy: DerivativeStrategy,
    options: &LeastSquaresOptions<S>,
    w: &mut LeastSquaresStorage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let fd = crate::derivatives::prepared(&mut w.finite_difference)?;
    super::jacobian_into(
        problem,
        &w.x,
        &w.residual,
        strategy,
        &mut w.jacobian,
        fd,
        session.execution()?,
    )?;
    let previous_ratio = w.model_scale.div(w.metric_scale)?;
    w.model_scale = Scaled::ONE;
    let ordinary = super::kernels::weights(
        problem,
        &w.residual,
        options,
        &mut w.native_weights,
        &mut w.native_curvature,
    )
    .is_some()
        && super::kernels::linearize(problem, options, w).is_some();
    if !ordinary {
        model_weights(
            problem,
            &w.residual,
            options,
            &mut w.row_weights,
            &mut w.curvature,
        )?;
        extended_linearization(problem, w)?;
    }
    let gradient_threshold = Scaled::from_value(options.gradient.threshold(S::ONE)?)?;
    let stopped = match options.stopping {
        super::LeastSquaresStopping::OriginalGradient => w
            .gradient_norm
            .is_some_and(|norm| norm.cmp_value(gradient_threshold) != Ordering::Greater),
        super::LeastSquaresStopping::Minpack { orthogonality, .. } => {
            w.orthogonality <= orthogonality.relative()
        }
    };
    if stopped
        && options.condition == scioxide_linalg::Estimation::Skip
        && options.linear_solver != super::LeastSquaresLinearSolver::DenseSvd
    {
        w.accepted_gradient = true;
        w.rank = None;
        w.condition = None;
        return Ok(());
    }
    let base_scale = w.model_scale;
    let n = w.parameter_dimension;
    let one = Scaled::from_value(<S as scioxide_core::AlgebraicScalar>::ONE)?;
    for column in 0..n {
        let norm = if ordinary && options.loss == super::LeastSquaresLoss::Quadratic {
            w.current_column_norms[column]
        } else {
            let mut norm = scioxide_core::ScaledNorm::new();
            for row in w.model_jacobian.as_slice().chunks_exact(n) {
                norm.add(row[column])?;
            }
            norm
        };
        w.current_column_norms[column] = norm;
        let norm = norm_value(norm)?.mul(base_scale)?;
        w.column_norms[column] = if norm.cmp_value(w.column_norms[column]) == Ordering::Greater {
            norm
        } else {
            w.column_norms[column]
        };
        w.parameter_scales[column] = match problem.parameter_scale(column) {
            Some(scale) => scale,
            None if w.column_norms[column].is_zero() => <S as scioxide_core::AlgebraicScalar>::ONE,
            None => {
                let inverse = one.div(w.column_norms[column])?;
                if inverse.cmp_value(Scaled::from_value(S::MAX)?) == Ordering::Greater {
                    S::MAX
                } else {
                    inverse.round()?
                }
            }
        };
    }
    let maximum = w
        .column_norms
        .iter()
        .copied()
        .max_by(|a, b| a.cmp_value(*b))
        .unwrap_or(Scaled::ONE);
    let minimum = w
        .column_norms
        .iter()
        .copied()
        .filter(|v| !v.is_zero())
        .min_by(|a, b| a.cmp_value(*b))
        .unwrap_or(Scaled::ONE);
    let scaled_factor = options.linear_solver == super::LeastSquaresLinearSolver::DenseSvd
        || w.residual_dimension < n
        || (ordinary
            && maximum.div(minimum)?.cmp_value(Scaled::from_value(S::MAX)?) != Ordering::Greater);
    for (factor, &scale) in w.factor_scales.iter_mut().zip(&w.parameter_scales) {
        *factor = if scaled_factor {
            scale
        } else {
            <S as scioxide_core::AlgebraicScalar>::ONE
        };
    }
    // Column norms bound every entry. In a resolved frame this certificate
    // avoids another full matrix scan just to choose the unit normalizer.
    let unit_frame = ordinary
        && (|| {
            let mut bound = <S as scioxide_core::AlgebraicScalar>::ZERO;
            for (norm, &scale) in w.current_column_norms.iter().zip(&w.factor_scales) {
                bound = bound.max(super::kernels::product(norm.finish().ok()?, scale)?);
            }
            let extent = S::try_from_usize_exact(w.residual_dimension.max(n).max(1)).ok()?;
            Some(
                bound < S::MAX.sqrt() / (S::TWO * extent)
                    && bound > S::TWO * S::MIN_POSITIVE.sqrt() * extent.sqrt(),
            )
        })()
        .unwrap_or(false);
    let mut normalizer = if unit_frame {
        Scaled::ONE
    } else {
        // Keep physical units separate from the finite normalized factorization.
        let normal = w.model_jacobian.as_slice().iter().enumerate().try_fold(
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            |maximum, (i, &j)| {
                super::kernels::product(j, w.factor_scales[i % n])
                    .map(|v| maximum.max(NumericScalar::abs(v)))
            },
        );
        let mut normalizer = if let Some(value) = normal {
            Scaled::from_value(value)?
        } else {
            let mut maximum = Scaled::ZERO;
            for (i, &j) in w.model_jacobian.as_slice().iter().enumerate() {
                let value = Scaled::from_value(j)?
                    .mul(Scaled::from_value(w.factor_scales[i % n])?)?
                    .abs();
                if value.cmp_value(maximum) == Ordering::Greater {
                    maximum = value;
                }
            }
            maximum
        };
        if ordinary
            && normalizer.cmp_value(Scaled::from_value(
                S::MAX.sqrt() / S::try_from_usize_exact(w.residual_dimension.max(n).max(1))?,
            )?) == Ordering::Less
            && normalizer.cmp_value(Scaled::from_value(S::MIN_POSITIVE.sqrt())?)
                == Ordering::Greater
        {
            normalizer = Scaled::ONE;
        }
        normalizer
    };
    let residual_max = w
        .weighted_residual
        .iter()
        .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |a, &b| {
            a.max(NumericScalar::abs(b))
        });
    let residual_floor =
        Scaled::from_value(residual_max)?.div(Scaled::from_value(S::MAX * S::HALF)?)?;
    if residual_floor.cmp_value(normalizer) == Ordering::Greater {
        normalizer = residual_floor;
    }
    if normalizer.is_zero() {
        normalizer = one;
    }
    w.model_scale = normalizer.mul(base_scale)?;
    if scaled_factor {
        w.metric_scale = Scaled::ONE;
        w.damping_diagonal
            .fill(<S as scioxide_core::AlgebraicScalar>::ONE);
    } else {
        w.metric_scale = Scaled::ZERO;
        for (&scale, &factor) in w.parameter_scales.iter().zip(&w.factor_scales) {
            let value = Scaled::from_value(factor)?.div(Scaled::from_value(scale)?)?;
            if value.cmp_value(w.metric_scale) == Ordering::Greater {
                w.metric_scale = value;
            }
        }
        if w.metric_scale.is_zero() {
            w.metric_scale = Scaled::ONE;
        }
        for ((diagonal, &scale), &factor) in w
            .damping_diagonal
            .iter_mut()
            .zip(&w.parameter_scales)
            .zip(&w.factor_scales)
        {
            *diagonal = Scaled::from_value(factor)?
                .div(Scaled::from_value(scale)?)?
                .div(w.metric_scale)?
                .round()?;
        }
    }
    let normalizer_native = normalizer.round().ok();
    for (i, value) in w.model_jacobian.as_mut_slice().iter_mut().enumerate() {
        let fast = normalizer_native.and_then(|normalizer| {
            super::kernels::product(*value, w.factor_scales[i % n]).and_then(|v| {
                if normalizer == <S as scioxide_core::AlgebraicScalar>::ONE {
                    Some(v)
                } else {
                    super::kernels::quotient(v, normalizer)
                }
            })
        });
        *value = match fast {
            Some(value) => value,
            None => Scaled::from_value(*value)?
                .mul(Scaled::from_value(w.factor_scales[i % n])?)?
                .div(normalizer)?
                .round()?,
        };
    }
    if normalizer_native != Some(<S as scioxide_core::AlgebraicScalar>::ONE) {
        for value in &mut w.weighted_residual {
            *value = match normalizer_native
                .and_then(|normalizer| super::kernels::quotient(*value, normalizer))
            {
                Some(value) => value,
                None => Scaled::from_value(*value)?.div(normalizer)?.round()?,
            };
        }
    }
    w.use_svd = options.linear_solver == super::LeastSquaresLinearSolver::DenseSvd
        || w.residual_dimension < n;
    if !w.use_svd {
        session.invoke(EvaluationRole::Factorization, || {
            scioxide_linalg::HouseholderDecomposition::factor_into(
                &w.model_jacobian,
                scioxide_linalg::HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
                &mut w.qr_factor,
                &mut w.qr_workspace,
                session.execution()?,
            )
        })?;
        let rank = w.qr_factor.analyze_rank(options.rank)?;
        w.use_svd = !rank.is_full_rank();
        w.rank = Some(rank);
        w.condition = None;
    }
    if w.use_svd {
        if !scaled_factor {
            let mut maximum = Scaled::ZERO;
            for (i, &value) in w.model_jacobian.as_slice().iter().enumerate() {
                let value = Scaled::from_value(value)?
                    .mul(Scaled::from_value(w.parameter_scales[i % n])?)?
                    .abs();
                if value.cmp_value(maximum) == Ordering::Greater {
                    maximum = value;
                }
            }
            if maximum.is_zero() {
                maximum = Scaled::ONE;
            }
            for (i, value) in w.model_jacobian.as_mut_slice().iter_mut().enumerate() {
                *value = Scaled::from_value(*value)?
                    .mul(Scaled::from_value(w.parameter_scales[i % n])?)?
                    .div(maximum)?
                    .round()?;
            }
            for value in &mut w.weighted_residual {
                *value = Scaled::from_value(*value)?.div(maximum)?.round()?;
            }
            w.model_scale = w.model_scale.mul(maximum)?;
            w.metric_scale = Scaled::ONE;
            w.factor_scales.copy_from_slice(&w.parameter_scales);
        }
        session.invoke(EvaluationRole::Factorization, || {
            scioxide_linalg::ThinSvdDecomposition::compute_into(
                &w.model_jacobian,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                &mut w.svd_factor,
                &mut w.svd_workspace,
                session.execution()?,
            )
        })?;
        w.rank = Some(w.svd_factor.analyze_rank(options.rank)?);
        w.condition = if options.condition == scioxide_linalg::Estimation::Skip
            || w.svd_factor.singular_values().is_empty()
        {
            None
        } else {
            Some(w.svd_factor.condition_estimate()?)
        };
    }
    let next_ratio = w.model_scale.div(w.metric_scale)?;
    if w.radius.is_some() && previous_ratio.cmp_value(next_ratio) != Ordering::Equal {
        let damping = Scaled::from_value(w.damping)?
            .mul(previous_ratio)?
            .mul(previous_ratio)?
            .div(next_ratio)?
            .div(next_ratio)?;
        w.damping = if damping.cmp_value(Scaled::from_value(S::MAX)?) == Ordering::Greater {
            S::MAX
        } else {
            damping.round()?
        };
    }
    w.accepted_gradient = true;
    Ok(())
}

fn dense_step<S: RealScalar + NumericScalar<Real = S>>(
    w: &mut LeastSquaresStorage<S>,
    options: &LeastSquaresOptions<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    // Normalize the QR right-hand side before applying Q^T: its norm can
    // overflow even when every component of the final step is representable.
    let rhs_scale = if extended_norm(&w.weighted_residual)?
        .cmp_value(Scaled::from_value(S::MAX * S::HALF)?)
        != Ordering::Greater
    {
        <S as scioxide_core::AlgebraicScalar>::ONE
    } else {
        w.weighted_residual.iter().fold(
            <S as scioxide_core::AlgebraicScalar>::ONE,
            |scale, &value| scale.max(NumericScalar::abs(value)),
        )
    };
    w.residual_rhs
        .as_mut_slice()
        .iter_mut()
        .zip(&w.weighted_residual)
        .for_each(|(b, &r)| *b = -r / rhs_scale);
    let radius = w
        .radius
        .map(|radius| {
            radius
                .div(w.metric_scale)?
                .div(Scaled::from_value(rhs_scale)?)
                .and_then(|value| {
                    let value = value.round()?;
                    if value <= <S as scioxide_core::AlgebraicScalar>::ZERO {
                        return Err(SciOxideError::numerical_instability(
                            scioxide_core::NumericalIssue::Underflow,
                            "trust radius is unrepresentable in model coordinates",
                        ));
                    }
                    Ok(value)
                })
        })
        .transpose()?;
    if let Some(radius) = radius {
        let report = session.invoke(EvaluationRole::LinearSolve, || {
            if w.use_svd {
                w.svd_factor.solve_trust_region_into(
                    &w.residual_rhs,
                    radius,
                    w.damping,
                    options.rank,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    w.factor_solution.view_mut(),
                    &mut w.svd_workspace,
                    session.execution()?,
                )
            } else {
                w.qr_factor.solve_trust_region_into(
                    &w.residual_rhs,
                    &w.damping_diagonal,
                    radius,
                    w.damping,
                    &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
                    w.factor_solution.view_mut(),
                    &mut w.qr_workspace,
                    session.execution()?,
                )
            }
        })?;
        report.solve_report().termination().require_accepted()?;
        w.damping = report.damping().ok_or_else(|| {
            SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "accepted trust-region report has no step",
            )
        })?;
    } else if w.use_svd {
        session.invoke(EvaluationRole::LinearSolve, || {
            w.svd_factor.solve_into(
                &w.residual_rhs,
                options.rank,
                w.damping,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                w.factor_solution.view_mut(),
                &mut w.svd_workspace,
                session.execution()?,
            )
        })?;
    } else {
        let report = session.invoke(EvaluationRole::LinearSolve, || {
            w.qr_factor.solve_damped_into(
                &w.residual_rhs,
                w.damping,
                &scioxide_linalg::DirectSolveOptions::new(options.rank),
                w.factor_solution.view_mut(),
                &mut w.qr_workspace,
                session.execution()?,
            )
        })?;
        report.termination().require_accepted()?;
    }
    for value in w.factor_solution.as_mut_slice() {
        *value = match super::kernels::product(*value, rhs_scale) {
            Some(v) => v,
            None => Scaled::from_value(*value)?
                .mul(Scaled::from_value(rhs_scale)?)?
                .round()?,
        };
    }
    for ((step, &value), &scale) in w
        .step
        .iter_mut()
        .zip(w.factor_solution.as_slice())
        .zip(&w.factor_scales)
    {
        *step = match super::kernels::product(value, scale) {
            Some(v) => v,
            None => Scaled::from_value(value)?
                .mul(Scaled::from_value(scale)?)?
                .round()?,
        };
    }
    session.invoke(EvaluationRole::LinearOperator, || {
        let execution = session.execution()?;
        for (out, row) in w.model_product.iter_mut().zip(
            w.model_jacobian
                .as_slice()
                .chunks_exact(w.parameter_dimension),
        ) {
            execution.check()?;
            *out = scioxide_core::backend::standard::sum_products(
                row.iter()
                    .copied()
                    .zip(w.factor_solution.as_slice().iter().copied()),
                scioxide_core::Accuracy::Standard,
            )?;
        }
        Ok(())
    })
}

/// Recover from an unrepresentable or non-descent undamped model without accepting it.
fn contract_radius<S: RealScalar + NumericScalar<Real = S>>(
    w: &mut LeastSquaresStorage<S>,
) -> Result<(), SciOxideError> {
    if let Some(radius) = w.radius {
        w.radius = Some(radius.mul(Scaled::from_value(S::HALF)?)?);
    }
    w.damping = (w.damping * S::TWO).max(S::MIN_POSITIVE).min(S::MAX);
    Ok(())
}

fn scaled_parameter_norm<S: RealScalar + NumericScalar<Real = S>>(
    w: &LeastSquaresStorage<S>,
) -> Result<Scaled<S>, SciOxideError> {
    coordinate_norm(&w.x, &w.parameter_scales)
}

fn coordinate_norm<S: RealScalar + NumericScalar<Real = S>>(
    values: &[S],
    scales: &[S],
) -> Result<Scaled<S>, SciOxideError> {
    let native = (|| {
        let mut norm = scioxide_core::ScaledNorm::new();
        for (&x, &scale) in values.iter().zip(scales) {
            norm.add(super::kernels::quotient(x, scale)?).ok()?;
        }
        Some(norm)
    })();
    if let Some(norm) = native {
        return norm_value(norm);
    }
    let mut norm = ScaledAccumulator::ZERO;
    for (&x, &scale) in values.iter().zip(scales) {
        let value = Scaled::from_value(x)?.div(Scaled::from_value(scale)?)?;
        norm.add_product(value, value)?;
    }
    norm.total()?.sqrt()
}

fn norm_value<S: RealScalar>(
    norm: scioxide_core::ScaledNorm<S>,
) -> Result<Scaled<S>, SciOxideError> {
    if let Ok(value) = norm.finish() {
        return Scaled::from_value(value);
    }
    Scaled::from_value(norm.scale())?.mul(Scaled::from_value(norm.sum_squares().sqrt())?)
}

/// The undamped model predicts actual objective reduction; damping is a step policy.
pub(crate) fn prediction<S: RealScalar + NumericScalar<Real = S>>(
    residual: &[S],
    product: &[S],
    alpha: S,
    scale: Scaled<S>,
) -> Result<Scaled<S>, SciOxideError> {
    let fast = (|| {
        let mut total = scioxide_core::CompensatedSum::ZERO;
        for (&r, &jp) in residual.iter().zip(product) {
            let jp = super::kernels::product(jp, alpha)?;
            total
                .add(-super::kernels::product(r, jp)? - super::kernels::product(jp, jp)? * S::HALF)
                .ok()?;
        }
        total.total().ok()
    })();
    if let Some(value) = fast {
        return Scaled::from_value(value)?.mul(scale)?.mul(scale);
    }
    let mut predicted = ScaledAccumulator::ZERO;
    for (&r, &jp) in residual.iter().zip(product) {
        let jp = Scaled::from_value(jp)?.mul(Scaled::from_value(alpha)?)?;
        predicted.add_product(Scaled::from_value(-r)?, jp)?;
        predicted.add_scaled(jp.mul(jp)?.mul(Scaled::from_value(-S::HALF)?)?)?;
    }
    predicted.total()?.mul(scale)?.mul(scale)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve<S, P>(
    problem: &P,
    x0: &[S],
    options: &LeastSquaresOptions<S>,
    output: &mut LeastSquaresResult<S>,
    w: &mut LeastSquaresStorage<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    validate(problem)?;
    if x0.len() != problem.param_dim() {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(problem.param_dim()),
            scioxide_core::Dimensions::one(x0.len()),
        ));
    }
    crate::utils::validate_finite_origin(x0, scioxide_core::NonFiniteOrigin::Input)?;
    let operator = super::operator::selected(problem, options)?;
    if operator
        && matches!(
            options.stopping,
            super::LeastSquaresStopping::Minpack { .. }
        )
    {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("stopping"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "MINPACK orthogonality stopping requires a dense Jacobian",
        ));
    }
    // Public execution and the stats adapter both prepare storage before the
    // epoch. Rebuilding QR/SVD requirements here duplicates preparation work.
    if w.parameter_dimension != x0.len() || w.residual_dimension != problem.residual_dim() {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(w.residual_dimension, w.parameter_dimension),
            scioxide_core::Dimensions::two(problem.residual_dim(), x0.len()),
        ));
    }
    w.reset();
    w.x.copy_from_slice(x0);
    let (backtracking, initial_damping) = match options.globalization {
        LeastSquaresGlobalization::GaussNewton {
            max_backtracking_steps,
        } => (
            Some(max_backtracking_steps),
            <S as scioxide_core::AlgebraicScalar>::ZERO,
        ),
        LeastSquaresGlobalization::LevenbergMarquardt { initial_damping } => {
            (None, initial_damping)
        }
    };
    let strategy = options.derivative_strategy;
    w.damping = initial_damping;
    problem.residual_into(&w.x, &mut w.residual)?;
    let mut current = cost(problem, &w.residual, options)?;
    w.accepted_value = Some(current);
    let threshold = Scaled::from_value(options.gradient.threshold(S::ONE)?)?;
    let step_tolerance = options.step.tolerance();
    let zero_step_tolerance = step_tolerance.absolute()
        == <S as scioxide_core::AlgebraicScalar>::ZERO
        && step_tolerance.relative() == <S as scioxide_core::AlgebraicScalar>::ZERO;
    let mut termination = OptimizeTermination::IterationLimit;
    let mut detail = "least-squares iteration limit; inspect accepted state and diagnostics";
    let mut completed = 0;
    for iteration in 0..options.max_iterations.get() {
        if let Some(stop) = session.iteration()? {
            termination = stop;
            break;
        }
        if !w.accepted_gradient {
            if operator {
                super::operator::linearize(problem, options, w, session)?;
            } else {
                dense_linearization(problem, strategy, options, w, session)?;
            }
        }
        if operator
            && w.gradient_norm
                .is_some_and(|norm| norm.cmp_value(threshold) != Ordering::Greater)
        {
            super::operator::verify_gradient(problem, w, session)?;
        }
        let original_stopping = matches!(
            options.stopping,
            super::LeastSquaresStopping::OriginalGradient
        );
        if (original_stopping || operator)
            && w.gradient_norm
                .is_some_and(|norm| norm.cmp_value(threshold) != Ordering::Greater)
        {
            termination = OptimizeTermination::GradientSmall;
            detail = "original objective gradient satisfies the gradient tolerance";
            break;
        }
        if let super::LeastSquaresStopping::Minpack { orthogonality, .. } = options.stopping
            && !operator
            && w.orthogonality <= orthogonality.relative()
        {
            termination = OptimizeTermination::Converged;
            detail = "Jacobian-residual orthogonality satisfies the MINPACK tolerance";
            break;
        }
        if backtracking.is_none() && w.radius.is_none() {
            let norm = scaled_parameter_norm(w)?;
            let factor = Scaled::from_value(S::TEN * S::TEN)?;
            w.radius = Some(if norm.is_zero() {
                let residual_norm = extended_norm(&w.weighted_residual)?.mul(w.model_scale)?;
                if residual_norm.cmp_value(Scaled::from_value(S::MAX.sqrt())?) == Ordering::Greater
                {
                    residual_norm.mul(factor)?
                } else {
                    factor
                }
            } else {
                norm.mul(factor)?
            });
        }
        if operator {
            super::operator::step(problem, options, w, session)?;
        } else {
            match dense_step(w, options, session) {
                Ok(()) => {}
                Err(error)
                    if backtracking.is_none()
                        && matches!(error.view(), scioxide_core::SciOxideErrorView::NumericalInstability { issue, .. } if issue != scioxide_core::NumericalIssue::Underflow) =>
                {
                    contract_radius(w)?;
                    completed = iteration + 1;
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        if let Some(stop) = session.stopped() {
            termination = stop;
            break;
        }
        let step_norm = coordinate_norm(&w.step, &w.parameter_scales)?;
        if backtracking.is_none() && w.accepted_iterations == 0 && !step_norm.is_zero() {
            w.radius = w.radius.map(|radius| {
                if step_norm.cmp_value(radius) == Ordering::Less {
                    step_norm
                } else {
                    radius
                }
            });
        }
        let small = if zero_step_tolerance {
            w.step
                .iter()
                .all(|&step| step == <S as scioxide_core::AlgebraicScalar>::ZERO)
        } else {
            scioxide_core::backend::scaled::scaled_magnitude_le_tolerance(
                extended_norm(&w.step)?,
                &step_tolerance,
                extended_norm(&w.x)?,
            )?
        };
        if small && backtracking.is_some() {
            termination = OptimizeTermination::StepSmall;
            break;
        }
        let previous_cost = current;
        let mut rejected_nonfinite = false;
        let mut last_reduction = Scaled::ZERO;
        let mut last_prediction = Scaled::ZERO;
        let accepted = crate::globalization::backtrack(
            <S as scioxide_core::AlgebraicScalar>::ONE,
            S::EPSILON,
            S::HALF,
            backtracking.unwrap_or(Budget::ONE),
            |_, alpha| {
                let predicted =
                    prediction(&w.weighted_residual, &w.model_product, alpha, w.model_scale)?;
                if predicted.is_zero() || predicted.is_sign_negative() {
                    return Ok(Some(false));
                }
                session.proposal()?;
                for (trial, (&x, &step)) in w.trial_x.iter_mut().zip(w.x.iter().zip(&w.step)) {
                    *trial = match crate::utils::trial_component(x, alpha, step) {
                        Ok(value) => value,
                        Err(error)
                            if error.kind()
                                == scioxide_core::SciOxideErrorKind::NumericalInstability =>
                        {
                            session.reject()?;
                            rejected_nonfinite = true;
                            if backtracking.is_none() {
                                contract_radius(w)?;
                            }
                            return Ok(None);
                        }
                        Err(error) => return Err(error),
                    };
                }
                problem.residual_into(&w.trial_x, &mut w.trial_residual)?;
                if w.trial_residual.iter().any(|value| !value.is_finite()) {
                    session.reject()?;
                    rejected_nonfinite = true;
                    if backtracking.is_none() {
                        contract_radius(w)?;
                    }
                    return Ok(None);
                }
                let trial = cost(problem, &w.trial_residual, options)?;
                let actual = actual_reduction(problem, &w.residual, &w.trial_residual, options)?;
                let gain = actual.div(predicted)?;
                w.last_gain = Some(gain);
                last_reduction = actual;
                last_prediction = predicted;
                if backtracking.is_none() {
                    // Moré/MINPACK: acceptance and radius adjustment are independent.
                    let quarter = Scaled::from_value(S::HALF * S::HALF)?;
                    if gain.cmp_value(quarter) != Ordering::Greater {
                        let mut shrink = S::HALF;
                        if actual.is_sign_negative() {
                            let mut norm = scioxide_core::ScaledNorm::new();
                            for &v in &w.model_product {
                                norm.add(v)?;
                            }
                            let jp2 = Scaled::from_value(norm.scale())?
                                .mul(Scaled::from_value(norm.scale())?)?
                                .mul(Scaled::from_value(norm.sum_squares())?)?
                                .mul(w.model_scale)?
                                .mul(w.model_scale)?;
                            let directional =
                                predicted.add(jp2.mul(Scaled::from_value(S::HALF)?)?)?;
                            shrink = directional
                                .mul(Scaled::from_value(S::HALF)?)?
                                .div(directional.sub(actual)?)?
                                .round()?
                                .max(<S as scioxide_core::AlgebraicScalar>::ONE / S::TEN);
                        }
                        if let Some(radius) = w.radius {
                            let cap = step_norm.mul(Scaled::from_value(S::TEN)?)?;
                            let radius = if radius.cmp_value(cap) == Ordering::Less {
                                radius
                            } else {
                                cap
                            };
                            w.radius = Some(radius.mul(Scaled::from_value(shrink)?)?);
                        }
                        w.damping = (w.damping / shrink).min(S::MAX);
                    } else if w.damping == <S as scioxide_core::AlgebraicScalar>::ZERO
                        || gain.cmp_value(Scaled::from_value(S::try_from_f64_lossy(0.75)?)?)
                            != Ordering::Less
                    {
                        w.radius = Some(step_norm.mul(Scaled::from_value(S::TWO)?)?);
                        w.damping *= S::HALF;
                    }
                }
                if gain.cmp_value(Scaled::from_value(
                    crate::globalization::sufficient_decrease::<S>(),
                )?) != Ordering::Less
                {
                    session.accept(scioxide_core::PartialResultTrust::Admissible)?;
                    w.x.swap_with_slice(&mut w.trial_x);
                    w.residual.swap_with_slice(&mut w.trial_residual);
                    current = trial;
                    w.accepted_value = Some(current);
                    w.accepted_gradient = false;
                    w.gradient_norm = None;
                    w.rank = None;
                    w.condition = None;
                    w.accepted_iterations = iteration + 1;
                    return Ok(Some(true));
                }
                session.reject()?;
                Ok(None)
            },
        )?
        .is_ok_and(|accepted| accepted);
        completed = iteration + 1;
        if let super::LeastSquaresStopping::Minpack { objective, .. } = options.stopping {
            let cost_scale = previous_cost;
            let two = Scaled::from_value(S::TWO)?;
            let objective_small = !last_prediction.is_zero()
                && scioxide_core::backend::scaled::scaled_magnitude_le_tolerance(
                    last_reduction.abs(),
                    &objective,
                    cost_scale,
                )?
                && scioxide_core::backend::scaled::scaled_magnitude_le_tolerance(
                    last_prediction,
                    &objective,
                    cost_scale,
                )?
                && w.last_gain
                    .is_some_and(|gain| gain.cmp_value(two) != Ordering::Greater);
            let radius_small = if let Some(radius) = w.radius {
                scioxide_core::backend::scaled::scaled_magnitude_le_tolerance(
                    radius,
                    &options.step.tolerance(),
                    scaled_parameter_norm(w)?,
                )?
            } else {
                false
            };
            if w.accepted_iterations > 0 && !rejected_nonfinite && (objective_small || radius_small)
            {
                detail = if objective_small {
                    "actual and predicted objective reductions satisfy the MINPACK tolerance"
                } else {
                    "scaled trust radius satisfies the MINPACK step tolerance"
                };
                termination = OptimizeTermination::Converged;
                break;
            }
        }
        if !accepted
            && backtracking.is_none()
            && !small
            && last_prediction.is_zero()
            && !rejected_nonfinite
        {
            contract_radius(w)?;
            continue;
        }
        if !accepted
            && (backtracking.is_some()
                || small
                || (last_prediction.is_zero() && !rejected_nonfinite))
        {
            termination = if small {
                OptimizeTermination::StepSmall
            } else {
                OptimizeTermination::Stagnation
            };
            break;
        }
    }
    if options.condition == scioxide_linalg::Estimation::Estimate
        && !operator
        && w.accepted_gradient
        && !w.use_svd
        && w.parameter_dimension > 0
        && session.stopped().is_none()
    {
        w.condition = Some(session.invoke(EvaluationRole::LinearSolve, || {
            w.qr_factor.condition_estimate_scoped(
                &scioxide_linalg::LinalgExecutionOptions::new(),
                &mut w.qr_workspace,
                session.execution()?,
            )
        })?);
    }
    crate::utils::finish_least_squares(w, output, current, completed, termination, detail)
}

/// Subtract observation losses before summing, keeping small improvements
/// against a large irreducible residual floor.
fn actual_reduction<S, P>(
    problem: &P,
    current: &[S],
    trial: &[S],
    options: &LeastSquaresOptions<S>,
) -> Result<Scaled<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    use super::LeastSquaresLoss;
    if let Some(value) = super::kernels::reduction(problem, current, trial, options) {
        return Scaled::from_value(value);
    }
    let mut total = ScaledAccumulator::ZERO;
    let c = Scaled::from_value(options.loss_scale)?;
    let c2 = c.mul(c)?;
    let half = Scaled::from_value(S::HALF)?;
    for (i, (old, new)) in current
        .chunks_exact(problem.residual_group_size())
        .zip(trial.chunks_exact(problem.residual_group_size()))
        .enumerate()
    {
        let scale = Scaled::from_value(problem.residual_scale(i))?;
        let mut old2 = ScaledAccumulator::ZERO;
        let mut new2 = ScaledAccumulator::ZERO;
        let mut difference = ScaledAccumulator::ZERO;
        for (&old, &new) in old.iter().zip(new) {
            let a = Scaled::from_value(old)?.div(scale)?;
            let b = Scaled::from_value(new)?.div(scale)?;
            old2.add_product(a, a)?;
            new2.add_product(b, b)?;
            difference.add_product(a.sub(b)?, a.add(b)?)?;
        }
        let a2 = old2.total()?;
        let b2 = new2.total()?;
        let difference = difference.total()?;
        let decrease = match options.loss {
            LeastSquaresLoss::Quadratic => difference.mul(half)?,
            LeastSquaresLoss::Huber
                if a2.cmp_value(c2) != Ordering::Greater
                    && b2.cmp_value(c2) != Ordering::Greater =>
            {
                difference.mul(half)?
            }
            LeastSquaresLoss::Huber
                if a2.cmp_value(c2) != Ordering::Less && b2.cmp_value(c2) != Ordering::Less =>
            {
                let denominator = a2.sqrt()?.add(b2.sqrt()?)?;
                difference.div(denominator)?.mul(c)?
            }
            LeastSquaresLoss::Huber => {
                options::loss_value(options.loss, a2.sqrt()?, options.loss_scale)?
                    .0
                    .sub(options::loss_value(options.loss, b2.sqrt()?, options.loss_scale)?.0)?
            }
            LeastSquaresLoss::SoftL1 => difference
                .mul(c)?
                .div(c2.add(a2)?.sqrt()?.add(c2.add(b2)?.sqrt()?)?)?,
            LeastSquaresLoss::Cauchy => {
                let ratio = difference.div(c2.add(b2)?)?;
                let log = if ratio.cmp_value(Scaled::from_value(
                    -<S as scioxide_core::AlgebraicScalar>::ONE,
                )?) == Ordering::Greater
                    && ratio.cmp_value(Scaled::from_value(S::MAX)?) != Ordering::Greater
                {
                    let rounded = ratio.round()?;
                    if rounded > -<S as scioxide_core::AlgebraicScalar>::ONE {
                        rounded.ln_1p()
                    } else {
                        options::log_magnitude(c2.add(a2)?)? - options::log_magnitude(c2.add(b2)?)?
                    }
                } else {
                    options::log_magnitude(c2.add(a2)?)? - options::log_magnitude(c2.add(b2)?)?
                };
                c2.mul(Scaled::from_value(log)?)?.mul(half)?
            }
        };
        total.add_scaled(decrease.mul(Scaled::from_value(problem.residual_weight(i))?)?)?;
    }
    total.total()
}

/// Keep the shared norm's scale separate from its normalized sum of squares.
pub(crate) fn extended_norm<S: RealScalar>(values: &[S]) -> Result<Scaled<S>, SciOxideError> {
    let mut norm = scioxide_core::ScaledNorm::new();
    for &value in values {
        norm.add(value)?;
    }
    norm_value(norm)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn complex_soft_l1_model_preserves_gradient_and_radial_curvature() {
        struct Pair;
        impl LeastSquaresProblem<f64> for Pair {
            fn param_dim(&self) -> usize {
                2
            }
            fn residual_dim(&self) -> usize {
                2
            }
            fn residual_group_size(&self) -> usize {
                2
            }
            fn residual_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
                out.copy_from_slice(x);
                Ok(())
            }
        }
        let options =
            LeastSquaresOptions::default().with_loss(super::super::LeastSquaresLoss::SoftL1, 1.0);
        let mut w = LeastSquaresStorage::new();
        w.try_reserve(2, 2, true).unwrap();
        w.residual.copy_from_slice(&[3.0, 4.0]);
        w.jacobian
            .as_mut_slice()
            .copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        super::super::kernels::weights(
            &Pair,
            &w.residual,
            &options,
            &mut w.native_weights,
            &mut w.native_curvature,
        )
        .unwrap();
        super::super::kernels::linearize(&Pair, &options, &mut w).unwrap();
        let first = 1.0 / 26.0_f64.sqrt();
        for column in 0..2 {
            let gradient: f64 = (0..2)
                .map(|row| w.model_jacobian.as_slice()[row * 2 + column] * w.weighted_residual[row])
                .sum();
            assert!((gradient - first * w.residual[column]).abs() < 1e-14);
            for other in 0..2 {
                let curvature: f64 = (0..2)
                    .map(|row| {
                        w.model_jacobian.as_slice()[row * 2 + column]
                            * w.model_jacobian.as_slice()[row * 2 + other]
                    })
                    .sum();
                let expected = if column == other { first } else { 0.0 }
                    - first.powi(3) * w.residual[column] * w.residual[other];
                assert!((curvature - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn rejected_trial_contracts_with_the_objective_directional_derivative() {
        struct Curved;
        impl LeastSquaresProblem<f64> for Curved {
            fn param_dim(&self) -> usize {
                1
            }
            fn residual_dim(&self) -> usize {
                1
            }
            fn residual_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
                out[0] = 1.0 + x[0] + 3.0_f64.sqrt() * x[0] * x[0];
                Ok(())
            }
            fn jacobian_into(
                &self,
                x: &[f64],
                out: &mut scioxide_linalg::Matrix<f64>,
            ) -> Result<bool, SciOxideError> {
                out.as_mut_slice()[0] = 1.0 + 2.0 * 3.0_f64.sqrt() * x[0];
                Ok(true)
            }
        }
        let solver = crate::LeastSquares::new(
            LeastSquaresOptions::default().with_max_iterations(Budget::ONE),
        )
        .unwrap();
        let mut workspace = solver
            .workspace(crate::LeastSquaresWorkspaceRequest::of(&Curved).unwrap())
            .unwrap();
        let mut result = LeastSquaresResult::empty();
        solver
            .solve_into(
                &Curved,
                &[0.0],
                &mut result,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        // p=-1, -gᵀp=1 and actual reduction=-1. Moré's contraction is
        // 0.5*(-gᵀp)/(-gᵀp-actual) = 1/4; the rejected state stays unchanged.
        assert!((workspace.storage.radius.unwrap().round().unwrap() - 0.25).abs() < 1e-14);
        assert_eq!(result.x(), &[0.0]);
        assert!((result.last_model_gain().unwrap().round().unwrap() + 2.0).abs() < 1e-14);
    }

    #[test]
    fn cauchy_reduction_handles_full_range_ratio_and_tiny_scaled_argument() {
        struct Identity;
        impl LeastSquaresProblem<f64> for Identity {
            fn param_dim(&self) -> usize {
                1
            }
            fn residual_dim(&self) -> usize {
                1
            }
            fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
                output.copy_from_slice(x);
                Ok(())
            }
        }
        let options =
            LeastSquaresOptions::default().with_loss(super::super::LeastSquaresLoss::Cauchy, 1.0);
        let reduction = actual_reduction(&Identity, &[1e-300], &[1e300], &options)
            .unwrap()
            .round()
            .unwrap();
        assert!((reduction + 1e300_f64.ln()).abs() < 1e-12);
        let options = options.with_loss(super::super::LeastSquaresLoss::Cauchy, 1e300);
        assert_eq!(
            cost(&Identity, &[1.0], &options).unwrap().round().unwrap(),
            0.5
        );
    }

    #[test]
    fn residual_model_matches_independent_polynomial() {
        // r=(3,4), Jp=(-1,-2): decrease = 12.5 - 4 = 8.5.
        let gain = prediction(
            &[3.0_f64, 4.0],
            &[-1.0, -2.0],
            1.0,
            Scaled::from_value(1.0).unwrap(),
        )
        .unwrap();
        assert_eq!(gain.round().unwrap(), 8.5);
        assert!(
            prediction(&[1.0_f32], &[1.0], 1.0, Scaled::from_value(1.0).unwrap())
                .unwrap()
                .is_sign_negative()
        );
        assert!(
            prediction(&[1.0_f64], &[0.0], 1.0, Scaled::from_value(1.0).unwrap())
                .unwrap()
                .is_zero()
        );
        // Squaring the model scale would overflow, but its comparison remains usable.
        let huge = prediction(
            &[3.0_f64, 4.0],
            &[-1.0, -2.0],
            1.0,
            Scaled::from_value(1e200).unwrap(),
        )
        .unwrap();
        assert!(huge.round().is_err());
        assert_eq!(
            huge.div(Scaled::from_value(1e200).unwrap())
                .unwrap()
                .div(Scaled::from_value(1e200).unwrap())
                .unwrap()
                .round()
                .unwrap(),
            8.5
        );
    }
}
