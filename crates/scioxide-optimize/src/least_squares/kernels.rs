//! Guarded arithmetic in the range of `S`; callers repeat a computation in scaled
//! arithmetic when it returns `None`.
use super::{LeastSquaresLoss, LeastSquaresOptions};
use crate::LeastSquaresProblem;
use scioxide_core::{CompensatedSum, NumericScalar, RealScalar, Scaled};

pub(super) fn product<S: RealScalar + NumericScalar<Real = S>>(a: S, b: S) -> Option<S> {
    let value = a * b;
    (!scioxide_core::backend::standard::product_needs_retry(a, b, value)).then_some(value)
}
pub(super) fn quotient<S: RealScalar>(a: S, b: S) -> Option<S> {
    let value = a / b;
    (value.is_finite() && (a == S::ZERO || value.abs() >= S::MIN_POSITIVE)).then_some(value)
}

/// Cost, first derivative and safeguarded radial curvature in residual units.
pub(super) fn loss<S: RealScalar + NumericScalar<Real = S>>(
    magnitude: S,
    options: &LeastSquaresOptions<S>,
) -> Option<(S, S, S)> {
    if options.loss == LeastSquaresLoss::Quadratic {
        return Some((product(magnitude, magnitude)? * S::HALF, S::ONE, S::ONE));
    }
    let z = quotient(magnitude, options.loss_scale)?;
    let square = product(z, z)?;
    let c2 = product(options.loss_scale, options.loss_scale)?;
    let (value, first, second) = match options.loss {
        LeastSquaresLoss::Quadratic => return None,
        LeastSquaresLoss::Huber if z <= S::ONE => {
            (product(magnitude, magnitude)? * S::HALF, S::ONE, S::ONE)
        }
        LeastSquaresLoss::Huber => (
            product(options.loss_scale, magnitude - options.loss_scale * S::HALF)?,
            S::ONE / z,
            S::ZERO,
        ),
        LeastSquaresLoss::SoftL1 => {
            let root = (S::ONE + square).sqrt();
            let first = S::ONE / root;
            (
                product(c2, square / (root + S::ONE))?,
                first,
                first * first * first,
            )
        }
        LeastSquaresLoss::Cauchy => {
            let denominator = S::ONE + square;
            (
                product(c2, square.ln_1p())? * S::HALF,
                S::ONE / denominator,
                (S::ONE - square) / denominator / denominator,
            )
        }
    };
    (value.is_finite() && first >= S::MIN_POSITIVE).then_some((
        value,
        first,
        second.max(S::EPSILON),
    ))
}

pub(super) fn cost<S, P>(problem: &P, residual: &[S], options: &LeastSquaresOptions<S>) -> Option<S>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    if options.loss == LeastSquaresLoss::Quadratic
        && (0..problem.residual_dim() / problem.residual_group_size())
            .all(|i| problem.residual_scale(i) == S::ONE && problem.residual_weight(i) == S::ONE)
        && let Some(square) = crate::quasi_newton::ordinary_dot(residual, residual)
    {
        return product(square, S::HALF);
    }
    let mut total = CompensatedSum::ZERO;
    for (i, values) in residual
        .chunks_exact(problem.residual_group_size())
        .enumerate()
    {
        let weight = problem.residual_weight(i);
        if weight == S::ZERO {
            continue;
        }
        let scale = problem.residual_scale(i);
        let a = quotient(values[0], scale)?;
        let magnitude = if values.len() == 2 {
            a.hypot(quotient(values[1], scale)?)
        } else {
            a.abs()
        };
        let (loss, _, _) = loss(magnitude, options)?;
        total.add(product(weight, loss)?).ok()?;
    }
    total.total().ok()
}

pub(super) fn reduction<S, P>(
    problem: &P,
    old: &[S],
    new: &[S],
    options: &LeastSquaresOptions<S>,
) -> Option<S>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    if options.loss == LeastSquaresLoss::Quadratic {
        if (0..problem.residual_dim() / problem.residual_group_size())
            .all(|i| problem.residual_scale(i) == S::ONE && problem.residual_weight(i) == S::ONE)
        {
            let mut total = CompensatedSum::ZERO;
            for (&a, &b) in old.iter().zip(new) {
                total.add(product(a - b, a + b)?).ok()?;
            }
            return product(total.total().ok()?, S::HALF);
        }
        let mut total = CompensatedSum::ZERO;
        let group = problem.residual_group_size();
        for (i, (old, new)) in old
            .chunks_exact(group)
            .zip(new.chunks_exact(group))
            .enumerate()
        {
            let weight = problem.residual_weight(i);
            if weight == S::ZERO {
                continue;
            }
            let scale = problem.residual_scale(i);
            let mut difference = CompensatedSum::ZERO;
            for (&a, &b) in old.iter().zip(new) {
                let a = quotient(a, scale)?;
                let b = quotient(b, scale)?;
                difference.add(product(a - b, a + b)?).ok()?;
            }
            total
                .add(product(
                    weight,
                    product(difference.total().ok()?, S::HALF)?,
                )?)
                .ok()?;
        }
        return total.total().ok();
    }
    let mut total = CompensatedSum::ZERO;
    let c2 = product(options.loss_scale, options.loss_scale)?;
    for (i, (old, new)) in old
        .chunks_exact(problem.residual_group_size())
        .zip(new.chunks_exact(problem.residual_group_size()))
        .enumerate()
    {
        let weight = problem.residual_weight(i);
        if weight == S::ZERO {
            continue;
        }
        let scale = problem.residual_scale(i);
        let mut difference = CompensatedSum::ZERO;
        let mut a2 = CompensatedSum::ZERO;
        let mut b2 = CompensatedSum::ZERO;
        for (&a, &b) in old.iter().zip(new) {
            let a = quotient(a, scale)?;
            let b = quotient(b, scale)?;
            difference.add(product(a - b, a + b)?).ok()?;
            a2.add(product(a, a)?).ok()?;
            b2.add(product(b, b)?).ok()?;
        }
        let difference = difference.total().ok()?;
        let a2 = a2.total().ok()?;
        let b2 = b2.total().ok()?;
        let reduction = match options.loss {
            LeastSquaresLoss::Quadratic => difference * S::HALF,
            LeastSquaresLoss::Huber if a2 <= c2 && b2 <= c2 => difference * S::HALF,
            LeastSquaresLoss::Huber if a2 >= c2 && b2 >= c2 => quotient(
                product(difference, options.loss_scale)?,
                a2.sqrt() + b2.sqrt(),
            )?,
            LeastSquaresLoss::Huber => loss(a2.sqrt(), options)?.0 - loss(b2.sqrt(), options)?.0,
            LeastSquaresLoss::SoftL1 => quotient(
                product(difference, options.loss_scale)?,
                (c2 + a2).sqrt() + (c2 + b2).sqrt(),
            )?,
            LeastSquaresLoss::Cauchy => {
                product(c2, quotient(difference, c2 + b2)?.ln_1p())? * S::HALF
            }
        };
        total.add(product(weight, reduction)?).ok()?;
    }
    total.total().ok()
}

/// Observation transforms without extended-range conversions in the ordinary case.
pub(super) fn weights<S, P>(
    problem: &P,
    residual: &[S],
    options: &LeastSquaresOptions<S>,
    weights: &mut [S],
    radial: &mut [S],
) -> Option<()>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let group = problem.residual_group_size();
    for (i, values) in residual.chunks_exact(group).enumerate() {
        let weight = problem.residual_weight(i);
        if weight == S::ZERO {
            weights[i * group..(i + 1) * group].fill(S::ZERO);
            radial[i * group..(i + 1) * group].fill(S::ZERO);
            continue;
        }
        let scale = problem.residual_scale(i);
        let (first, second) = if options.loss == LeastSquaresLoss::Quadratic {
            (S::ONE, S::ONE)
        } else {
            let a = quotient(values[0], scale)?;
            let magnitude = if group == 2 {
                a.hypot(quotient(values[1], scale)?)
            } else {
                a.abs()
            };
            let (_, first, second) = loss(magnitude, options)?;
            (first, second)
        };
        let root = quotient(weight.sqrt(), scale)?;
        let a = product(root, first.sqrt())?;
        let b = product(root, second.sqrt())?;
        weights[i * group..(i + 1) * group].fill(a);
        radial[i * group..(i + 1) * group].fill(b);
    }
    Some(())
}

/// Form the original gradient and the curvature-corrected model before normalization.
pub(super) fn linearize<S, P>(
    problem: &P,
    options: &LeastSquaresOptions<S>,
    w: &mut crate::least_squares::LeastSquaresStorage<S>,
) -> Option<()>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let n = w.parameter_dimension;
    let quadratic = options.loss == LeastSquaresLoss::Quadratic;
    let angular = matches!(
        options.stopping,
        super::LeastSquaresStopping::Minpack { .. }
    );
    let unit_rows = w.native_weights.iter().all(|&weight| weight == S::ONE);
    let mut gradient_norm = scioxide_core::ScaledNorm::new();
    let mut orthogonality = S::ZERO;
    let mut residual_norm = scioxide_core::ScaledNorm::new();
    for ((value, &r), &weight) in w
        .weighted_residual
        .iter_mut()
        .zip(&w.residual)
        .zip(&w.native_weights)
    {
        *value = if unit_rows { r } else { product(r, weight)? };
        if angular {
            residual_norm.add(*value).ok()?;
        }
    }
    let residual_norm = if angular {
        residual_norm.finish().ok()?
    } else {
        S::ONE
    };
    for column in 0..n {
        let mut gradient = CompensatedSum::ZERO;
        let mut column_norm = scioxide_core::ScaledNorm::new();
        for (row, (&r, &weight)) in w
            .weighted_residual
            .iter()
            .zip(&w.native_weights)
            .enumerate()
        {
            let j = if unit_rows {
                w.jacobian.as_slice()[row * n + column]
            } else {
                product(w.jacobian.as_slice()[row * n + column], weight)?
            };
            if quadratic || angular {
                column_norm.add(j).ok()?;
            }
            gradient.add(product(j, r)?).ok()?;
            if quadratic {
                w.model_jacobian.as_mut_slice()[row * n + column] = j;
            }
        }
        let value = gradient.total().ok()?;
        gradient_norm.add(value).ok()?;
        if quadratic {
            w.current_column_norms[column] = column_norm;
        }
        let norm = if angular {
            column_norm.finish().ok()?
        } else {
            S::ONE
        };
        if angular && norm > S::ZERO && residual_norm > S::ZERO {
            let cosine = (value / norm) / residual_norm;
            if !cosine.is_finite() {
                return None;
            }
            orthogonality = orthogonality.max(cosine.abs());
        }
    }
    w.gradient_norm = Some(Scaled::from_value(gradient_norm.finish().ok()?).ok()?);
    w.orthogonality = orthogonality;
    if quadratic {
        return Some(());
    }
    let group = problem.residual_group_size();
    for start in (0..w.residual_dimension).step_by(group) {
        let weight = w.native_weights[start];
        let radial = w.native_curvature[start];
        if weight == S::ZERO {
            w.weighted_residual[start..start + group].fill(S::ZERO);
            w.model_jacobian.as_mut_slice()[start * n..(start + group) * n].fill(S::ZERO);
            continue;
        }
        let residual_factor = quotient(product(weight, weight)?, radial)?;
        for i in start..start + group {
            w.weighted_residual[i] = product(w.residual[i], residual_factor)?;
        }
        if group == 1 {
            for (out, &j) in w.model_jacobian.as_mut_slice()[start * n..(start + 1) * n]
                .iter_mut()
                .zip(&w.jacobian.as_slice()[start * n..(start + 1) * n])
            {
                *out = product(j, radial)?;
            }
        } else {
            let norm = w.residual[start].hypot(w.residual[start + 1]);
            if !norm.is_finite() {
                return None;
            }
            let (u, v) = if norm == S::ZERO {
                (S::ONE, S::ZERO)
            } else {
                (w.residual[start] / norm, w.residual[start + 1] / norm)
            };
            for column in 0..n {
                let a = w.jacobian.as_slice()[start * n + column];
                let b = w.jacobian.as_slice()[(start + 1) * n + column];
                let projection = product(u, a)? + product(v, b)?;
                let correction = product(radial - weight, projection)?;
                let x = product(weight, a)? + product(u, correction)?;
                let y = product(weight, b)? + product(v, correction)?;
                if !x.is_finite() || !y.is_finite() {
                    return None;
                }
                w.model_jacobian.as_mut_slice()[start * n + column] = x;
                w.model_jacobian.as_mut_slice()[(start + 1) * n + column] = y;
            }
        }
    }
    Some(())
}
