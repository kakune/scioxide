//! Finite-difference stencils on rounded nodes, and steps along a direction.
//!
//! A direction is zero only when every component is zero. Direction scaling and
//! distances to bounds of opposite sign are computed on exponent and significand
//! separately, so they cannot overflow or underflow. When no distinct finite
//! displacement exists, the stencil functions return `None`: fixed-step plans
//! turn that into an error, and adaptive plans stop at the step-size limit.

use crate::DifferenceScheme;
use crate::validation::{breakdown, invalid};
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{
    Bounds, ConvergenceDiagnostics, ConvergenceReason, DifferentiationLocation, Dimensions,
    EvaluationCounts, EvaluationRole, NumericalIssue, RangeAccumulator, RealScalar, SciOxideError,
    checked_float_product, checked_float_quotient,
};

// The largest stencil is the one-sided fallback for a central second
// derivative at a bound (offsets 0, h, 2h, 3h). Fixed-size arrays of this
// length keep stencil construction and application allocation-free.
pub(super) const MAX_STENCIL_POINTS: usize = 4;

#[derive(Debug, Clone, Copy)]
pub(super) struct Stencil<S: RealScalar> {
    pub(super) offsets: [S; MAX_STENCIL_POINTS],
    pub(super) weights: [S; MAX_STENCIL_POINTS],
    pub(super) scale: S,
    pub(super) derivative_order: usize,
    pub(super) len: usize,
}

/// Two-point first-derivative stencil, used for every output component.
#[derive(Clone, Copy)]
pub(super) struct FirstDifference<S: RealScalar> {
    scale: S,
    sign: S,
}
impl<S: RealScalar> Stencil<S> {
    pub(super) fn first_difference(self) -> Option<FirstDifference<S>> {
        (self.derivative_order == 1
            && self.len == 2
            && self.offsets[0] == S::ZERO
            && self.weights[1].abs() == S::ONE)
            .then_some(FirstDifference {
                scale: self.scale,
                sign: self.weights[1],
            })
    }
}
impl<S: RealScalar> FirstDifference<S> {
    /// Returns the two-point difference in ordinary arithmetic, or `None` if it
    /// would overflow or underflow; the caller then recomputes it with rescaling.
    #[inline]
    pub(super) fn native(self, base: S, sample: S) -> Option<S> {
        let normalized = (sample - base) * self.sign;
        if normalized == S::ZERO {
            return Some(S::ZERO);
        }
        let value = normalized / self.scale;
        (normalized.is_finite()
            && normalized.abs() >= S::MIN_POSITIVE
            && value.is_finite()
            && value != S::ZERO)
            .then_some(value)
    }
}

pub(super) fn vacant_stencil<S: RealScalar>() -> Stencil<S> {
    Stencil {
        offsets: [S::ZERO; MAX_STENCIL_POINTS],
        weights: [S::ZERO; MAX_STENCIL_POINTS],
        scale: S::ONE,
        derivative_order: 1,
        len: 0,
    }
}

pub(super) fn stencil_weights<S: RealScalar>(
    nodes: &[S],
    derivative_order: usize,
) -> Result<[S; MAX_STENCIL_POINTS], SciOxideError> {
    // These normalized geometries have exact dyadic Fornberg weights. Keep
    // asymmetric rounded nodes and one-sided higher orders in the recurrence.
    if derivative_order == 1 {
        if nodes.len() == 2 && nodes[0] == S::ZERO && nodes[1].abs() == S::ONE {
            return Ok([-nodes[1], nodes[1], S::ZERO, S::ZERO]);
        }
        if nodes == [-S::ONE, S::ZERO, S::ONE] {
            return Ok([-S::HALF, S::ZERO, S::HALF, S::ZERO]);
        }
        if let &[negative, zero, positive] = nodes
            && negative < S::ZERO
            && zero == S::ZERO
            && positive > S::ZERO
        {
            // Expand the order-one, three-node recurrence without its table.
            // This is the same sequence of rounded operations as Fornberg,
            // including the center weight for asymmetric rounded offsets.
            let inverse = S::ONE / -negative;
            let span = positive - negative;
            let product = span * positive;
            let weights = [
                positive * -inverse / span,
                (positive * inverse - S::ONE) / positive,
                -negative / product,
                S::ZERO,
            ];
            if inverse.is_finite()
                && span.is_finite()
                && product != S::ZERO
                && product.is_finite()
                && weights.iter().all(|w| w.is_finite())
            {
                return Ok(weights);
            }
        }
    }
    // Fornberg fills a row-major `nodes x (order + 1)` table holding weights for
    // every derivative order up to `derivative_order`; three columns cover the
    // first and second derivatives used here, and only the last column is kept.
    let mut table = [S::ZERO; MAX_STENCIL_POINTS * 3];
    fornberg(nodes, derivative_order, &mut table)?;
    let mut weights = [S::ZERO; MAX_STENCIL_POINTS];
    for (i, weight) in weights[..nodes.len()].iter_mut().enumerate() {
        *weight = table[i * (derivative_order + 1) + derivative_order];
    }
    Ok(weights)
}

pub(super) fn nominal_step<S: RealScalar>(value: S, formula: DifferenceScheme, order: usize) -> S {
    // Each exponent balances truncation error against rounding error amplified
    // by h^-order: O(h) vs eps/h gives sqrt(eps), O(h^2) vs eps/h and O(h) vs
    // eps/h^2 give cbrt(eps), and O(h^2) vs eps/h^2 gives eps^(1/4). Scaling by
    // max(1, |x|) makes the step relative for large coordinates while keeping it
    // absolute near zero, where a relative step would vanish. Only orders 1 and 2
    // reach this function; the fallback arm just keeps it total.
    let relative = match (formula, order) {
        (DifferenceScheme::Forward, 1) => <S as RealScalar>::SQRT_EPSILON,
        (DifferenceScheme::Central, 1) | (DifferenceScheme::Forward, 2) => S::CBRT_EPSILON,
        (DifferenceScheme::Central, 2) => <S as RealScalar>::SQRT_EPSILON.sqrt(),
        _ => <S as RealScalar>::SQRT_EPSILON,
    };
    relative * value.abs().max(S::ONE)
}

pub(super) fn directional_nodes<S: RealScalar>(
    value: S,
    nominal: S,
    direction: S,
    multiples: usize,
    bound: Option<(Option<S>, Option<S>)>,
) -> Option<[S; MAX_STENCIL_POINTS]> {
    // Shrink the step so that all `multiples` displacements stay inside the
    // bound on the chosen side; slot 0 of `nodes` is the base point itself.
    let room = match (direction > S::ZERO, bound) {
        (true, Some((_, Some(upper)))) => Some(upper - value),
        (false, Some((Some(lower), _))) => Some(value - lower),
        _ => None,
    };
    let magnitude = if multiples == 1 {
        room.map_or(nominal, |available| nominal.min(available))
    } else {
        let divisor = S::try_from_usize_exact(multiples).ok()?;
        room.map_or(nominal, |available| nominal.min(available / divisor))
    };
    if !magnitude.is_finite() || magnitude <= S::ZERO {
        return None;
    }
    let mut nodes = [S::ZERO; MAX_STENCIL_POINTS];
    for multiple in 1..=multiples {
        let multiplier = S::try_from_usize_exact(multiple).ok()?;
        let mut perturbed = value + direction * magnitude * multiplier;
        if !perturbed.is_finite() {
            return None;
        }
        if let Some((lower, upper)) = bound {
            if let Some(lower) = lower {
                perturbed = perturbed.max(lower);
            }
            if let Some(upper) = upper {
                perturbed = perturbed.min(upper);
            }
        }
        // Callers reconstruct the point from its offset. Even a clamped point
        // can round outside the bound after subtraction and reconstruction.
        let inside = |point: S| {
            point.is_finite()
                && bound.is_none_or(|(lower, upper)| {
                    lower.is_none_or(|lower| point >= lower)
                        && upper.is_none_or(|upper| point <= upper)
                })
        };
        let mut offset = perturbed - value;
        if !offset.is_finite() {
            return None;
        }
        if !inside(value + offset) {
            offset = offset.next_after(S::ZERO);
        }
        if !inside(value + offset)
            || value + offset == value
            || (multiple > 1 && value + offset == value + nodes[multiple - 1])
        {
            return None;
        }
        nodes[multiple] = offset;
    }
    Some(nodes)
}

/// Returns `None` if no stencil with distinct finite nodes fits in the domain.
/// Writes `output` only on `Some(())`; numerical failures are errors.
pub(super) fn axis_stencil_with_nominal<S: RealScalar>(
    value: S,
    nominal: S,
    formula: DifferenceScheme,
    derivative_order: usize,
    bound: Option<(Option<S>, Option<S>)>,
    output: &mut Stencil<S>,
) -> Result<Option<()>, SciOxideError> {
    // Choose the geometry before limiting its step. A short outward room
    // must not turn a central stencil into an ill-conditioned asymmetric one.
    let (lower, upper) = bound.unwrap_or((None, None));
    let backward = value - lower.unwrap_or(S::MIN);
    let forward = upper.unwrap_or(S::MAX) - value;
    let multiples = match (formula, derivative_order) {
        (DifferenceScheme::Forward, 1 | 2) => derivative_order,
        (DifferenceScheme::Central, 1 | 2) => derivative_order + 1,
        _ => return Ok(None),
    };
    let divisor = S::try_from_usize_exact(multiples)?;
    let prefer_forward = match formula {
        DifferenceScheme::Forward if nominal <= forward / divisor => true,
        DifferenceScheme::Forward if nominal <= backward / divisor => false,
        _ => forward >= backward,
    };
    let direction = if prefer_forward { S::ONE } else { -S::ONE };
    let central = |step| {
        let negative = directional_nodes(value, step, -S::ONE, 1, bound)?;
        let positive = directional_nodes(value, step, S::ONE, 1, bound)?;
        Some(([negative[1], S::ZERO, positive[1], S::ZERO], 3))
    };
    let geometry =
        if formula == DifferenceScheme::Central && nominal <= backward && nominal <= forward {
            central(nominal)
        } else {
            directional_nodes(value, nominal, direction, multiples, bound)
                .or_else(|| directional_nodes(value, nominal, -direction, multiples, bound))
                .map(|nodes| (nodes, multiples + 1))
                .or_else(|| {
                    // A very narrow interval may have three representable
                    // points straddling x, but too few on either side alone.
                    (formula == DifferenceScheme::Central)
                        .then(|| central(nominal.min(backward).min(forward)))
                        .flatten()
                })
        };
    let Some((nodes, len)) = geometry else {
        return Ok(None);
    };
    let scale = nodes[..len]
        .iter()
        .fold(S::ZERO, |scale, offset| scale.max(offset.abs()));
    if !scale.is_finite() || scale == S::ZERO {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "stencil scale is not finite and positive",
        ));
    }
    if len == 2 && derivative_order == 1 {
        let sign = if nodes[1] > S::ZERO { S::ONE } else { -S::ONE };
        *output = Stencil {
            offsets: nodes,
            weights: [-sign, sign, S::ZERO, S::ZERO],
            scale,
            derivative_order,
            len,
        };
        return Ok(Some(()));
    }
    // Weights for raw offsets scale like h^-order and can overflow for tiny
    // steps, so they are computed on O(1) nodes and the derivative is divided by
    // `scale` once per derivative order when the stencil is applied.
    let mut normalized_nodes = [S::ZERO; MAX_STENCIL_POINTS];
    for index in 0..len {
        normalized_nodes[index] = nodes[index] / scale;
    }
    let weights = stencil_weights(&normalized_nodes[..len], derivative_order)?;
    *output = Stencil {
        offsets: nodes,
        weights,
        scale,
        derivative_order,
        len,
    };
    Ok(Some(()))
}

pub(super) fn stencil_derivative<S: RealScalar>(
    stencil: Stencil<S>,
    samples: &[S],
    base: S,
    reduction: &mut scioxide_core::RangeAccumulator<S>,
) -> Result<S, SciOxideError> {
    if samples.len() != stencil.len {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(stencil.len),
            Dimensions::one(samples.len()),
        ));
    }

    if let Some(first) = stencil.first_difference()
        && samples[0] == base
        && let Some(value) = first.native(base, samples[1])
    {
        return Ok(value);
    }

    // Derivative weights sum to zero, so subtracting f(x) from every sample
    // leaves the result unchanged mathematically. Numerically it removes the
    // f(x) * sum(weights) leak caused by rounded weights and turns a sum of
    // large, nearly equal terms into a sum of small differences. If a
    // difference overflows, fall back to the raw samples, which are
    // algebraically equivalent.
    let mut differences = [S::ZERO; MAX_STENCIL_POINTS];
    let differences_are_finite = samples.iter().enumerate().all(|(index, &sample)| {
        let difference = sample - base;
        differences[index] = difference;
        difference.is_finite()
    });
    let values = if differences_are_finite {
        &differences[..stencil.len]
    } else {
        samples
    };
    if differences_are_finite && stencil.derivative_order == 1 {
        let normalized = if stencil.len == 3
            && stencil.weights[..3] == [-S::HALF, S::ZERO, S::HALF]
            && [values[0], values[2]]
                .iter()
                .all(|v| *v == S::ZERO || v.abs() >= S::TWO * S::MIN_POSITIVE)
        {
            // Each halving is exact. One addition therefore rounds the exact
            // two-product sum once, just like checked_sum_of_products.
            Some(-S::HALF * values[0] + S::HALF * values[2])
        } else {
            None
        };
        if let Some(normalized) = normalized {
            if normalized == S::ZERO {
                return Ok(S::ZERO);
            }
            let value = normalized / stencil.scale;
            if normalized.is_finite()
                && normalized.abs() >= S::MIN_POSITIVE
                && value.is_finite()
                && value != S::ZERO
            {
                return Ok(value);
            }
        }
    }
    // Dividing by `scale` once per order avoids forming scale^order, which could
    // underflow to zero for second derivatives with tiny steps.
    scaled_stencil_sum(
        &stencil.weights[..stencil.len],
        values,
        &[(stencil.scale, stencil.derivative_order)],
        reduction,
    )
}

// Preserve the existing rounding order in ordinary geometry, but never discard
// a range-boundary weighted sum before the physical derivative is formed.
pub(super) fn scaled_stencil_sum<S: RealScalar>(
    weights: &[S],
    values: &[S],
    steps: &[(S, usize)],
    reduction: &mut scioxide_core::RangeAccumulator<S>,
) -> Result<S, SciOxideError> {
    match scioxide_core::checked_sum_of_products(weights, values) {
        Ok(normalized) => {
            let mut value = normalized;
            let mut ordinary = value != S::ZERO;
            for &(scale, order) in steps {
                for _ in 0..order {
                    if value != S::ZERO && value.abs() < S::MIN_POSITIVE {
                        ordinary = false;
                        break;
                    }
                    match checked_float_quotient(value, scale) {
                        Ok(next) if next != S::ZERO || value == S::ZERO => value = next,
                        _ => {
                            ordinary = false;
                            break;
                        }
                    }
                }
                if !ordinary {
                    break;
                }
            }
            if ordinary {
                return Ok(value);
            }
        }
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::Overflow,
                    ..
                }
            ) => {}
        Err(error) => return Err(error),
    }
    reduction.clear();
    weights
        .iter()
        .zip(values)
        .try_for_each(|(&weight, &value)| reduction.add_product(weight, value))?;
    if reduction.sign() == core::cmp::Ordering::Equal {
        return Ok(S::ZERO);
    }
    finish_scaled_steps(reduction, steps)
}

pub(super) fn mixed_stencil_derivative<S: RealScalar>(
    first: Stencil<S>,
    second: Stencil<S>,
    samples: &[S],
    base: S,
    reduction: &mut scioxide_core::RangeAccumulator<S>,
) -> Result<S, SciOxideError> {
    let expected = checked_mul(
        first.len,
        second.len,
        scioxide_core::quantity!("mixed_stencil_sample_count"),
    )?;
    if samples.len() != expected {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(expected),
            Dimensions::one(samples.len()),
        ));
    }

    // The mixed partial uses the tensor product of two first-derivative
    // stencils, with samples laid out row-major by (first node, second node).
    // Tensor weights also sum to zero, so the same base subtraction applies.
    let mut weights = [S::ZERO; MAX_STENCIL_POINTS * MAX_STENCIL_POINTS];
    let mut differences = [S::ZERO; MAX_STENCIL_POINTS * MAX_STENCIL_POINTS];
    let mut differences_are_finite = true;
    for first_index in 0..first.len {
        for second_index in 0..second.len {
            let index = first_index * second.len + second_index;
            weights[index] =
                checked_float_product(first.weights[first_index], second.weights[second_index])?;
            let difference = samples[index] - base;
            differences[index] = difference;
            differences_are_finite &= difference.is_finite();
        }
    }
    let values = if differences_are_finite {
        &differences[..expected]
    } else {
        samples
    };
    scaled_stencil_sum(
        &weights[..expected],
        values,
        &[(first.scale, 1), (second.scale, 1)],
        reduction,
    )
}

pub(super) fn bound_for<S: RealScalar>(
    bounds: Option<&Bounds<S>>,
    variable: usize,
) -> Option<(Option<S>, Option<S>)> {
    bounds.map(|bounds| bounds.as_slice()[variable])
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PositiveMagnitude<S: RealScalar>(pub(super) scioxide_core::Scaled<S>);

impl<S: RealScalar> PositiveMagnitude<S> {
    pub(super) fn from_positive(value: S) -> Result<Self, SciOxideError> {
        let value = scioxide_core::Scaled::from_parts(value, 0)?;
        if value.is_zero() || value.is_sign_negative() {
            return Err(step_size_limit_error(
                0,
                "a step magnitude must be positive",
            ));
        }
        Ok(Self(value))
    }

    pub(super) fn ratio(numerator: S, denominator: S) -> Result<Self, SciOxideError> {
        Self::from_positive(numerator)?.divide(Self::from_positive(denominator)?)
    }

    pub(super) fn divide(self, denominator: Self) -> Result<Self, SciOxideError> {
        self.0.div(denominator.0).map(Self)
    }

    pub(super) fn multiply(self, factor: Self) -> Result<Self, SciOxideError> {
        self.0.mul(factor.0).map(Self)
    }

    pub(super) fn divide_usize(self, denominator: usize) -> Result<Self, SciOxideError> {
        self.divide(Self::from_positive(S::try_from_usize_exact(denominator)?)?)
    }

    pub(super) fn multiply_usize(self, multiplier: usize) -> Result<Self, SciOxideError> {
        self.multiply(Self::from_positive(S::try_from_usize_exact(multiplier)?)?)
    }

    pub(super) fn add(self, other: Self) -> Result<Self, SciOxideError> {
        self.0.add(other.0).map(Self)
    }

    pub(super) fn representable_step(self) -> Result<Option<S>, SciOxideError> {
        match self.0.round() {
            Ok(value) => Ok((value > S::ZERO).then_some(value)),
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: NumericalIssue::Overflow,
                        ..
                    }
                ) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    pub(super) fn minimum(self, other: Self) -> Self {
        if self.0.abs_cmp(other.0).is_gt() {
            other
        } else {
            self
        }
    }

    pub(super) fn maximum(self, other: Self) -> Self {
        if self.0.abs_cmp(other.0).is_gt() {
            self
        } else {
            other
        }
    }
}

pub(super) fn positive_distance<S: RealScalar>(
    left: S,
    right: S,
) -> Result<Option<PositiveMagnitude<S>>, SciOxideError> {
    if left == right {
        return Ok(None);
    }
    // With opposite signs, `right - left` can overflow (for example between
    // -MAX and MAX), so the distance is formed as |left| + |right| in scaled
    // form. Equal endpoints return `None`: the point sits on that bound.
    let same_sign =
        left == S::ZERO || right == S::ZERO || left.is_sign_positive() == right.is_sign_positive();
    if same_sign {
        PositiveMagnitude::from_positive((right - left).abs()).map(Some)
    } else {
        let left = PositiveMagnitude::from_positive(left.abs())?;
        let right = PositiveMagnitude::from_positive(right.abs())?;
        Ok(Some(left.add(right)?))
    }
}

pub(super) fn distance_ratio<S: RealScalar>(
    distance: Option<PositiveMagnitude<S>>,
    divisor: S,
) -> Result<Option<PositiveMagnitude<S>>, SciOxideError> {
    distance
        .map(|distance| distance.divide(PositiveMagnitude::from_positive(divisor.abs())?))
        .transpose()
}

pub(super) fn update_minimum<S: RealScalar>(
    current: &mut Option<PositiveMagnitude<S>>,
    candidate: Option<PositiveMagnitude<S>>,
) {
    if let Some(candidate) = candidate {
        *current = Some(current.map_or(candidate, |value| value.minimum(candidate)));
    }
}

pub(super) fn update_directional_room<S: RealScalar>(
    current: &mut Option<PositiveMagnitude<S>>,
    blocked: &mut bool,
    distance: Option<PositiveMagnitude<S>>,
    component: S,
) -> Result<(), SciOxideError> {
    let candidate = distance_ratio(distance, component)?;
    if candidate.is_none() {
        *blocked = true;
    } else {
        update_minimum(current, candidate);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DirectionalBounds<S: RealScalar> {
    pub(super) backward: Option<PositiveMagnitude<S>>,
    pub(super) forward: Option<PositiveMagnitude<S>>,
}

pub(super) fn directional_bounds<S: RealScalar>(
    x: &[S],
    direction: &[S],
    bounds: Option<&Bounds<S>>,
) -> Result<DirectionalBounds<S>, SciOxideError> {
    // For the ray x + alpha * d, each coordinate allows |alpha| up to
    // distance_i / |d_i| on each side, and the admissible room is the minimum
    // over coordinates. Unbounded sides use +/-MAX so generated points remain
    // finite. A coordinate already on a bound on the side it moves toward blocks
    // that whole direction, because any nonzero alpha would leave the box.
    let mut backward = None;
    let mut forward = None;
    let mut backward_blocked = false;
    let mut forward_blocked = false;
    for (variable, (&value, &component)) in x.iter().zip(direction.iter()).enumerate() {
        let (variable_lower, variable_upper) = bounds
            .map(|bounds| bounds.as_slice()[variable])
            .unwrap_or((None, None));
        let lower = variable_lower.unwrap_or(S::MIN);
        let upper = variable_upper.unwrap_or(S::MAX);
        if component > S::ZERO {
            update_directional_room(
                &mut backward,
                &mut backward_blocked,
                positive_distance(lower, value)?,
                component,
            )?;
            update_directional_room(
                &mut forward,
                &mut forward_blocked,
                positive_distance(value, upper)?,
                component,
            )?;
        } else if component < S::ZERO {
            update_directional_room(
                &mut backward,
                &mut backward_blocked,
                positive_distance(value, upper)?,
                component,
            )?;
            update_directional_room(
                &mut forward,
                &mut forward_blocked,
                positive_distance(lower, value)?,
                component,
            )?;
        }
    }
    Ok(DirectionalBounds {
        backward: if backward_blocked { None } else { backward },
        forward: if forward_blocked { None } else { forward },
    })
}

pub(super) fn scaled_directional_nodes<S: RealScalar>(
    nominal: PositiveMagnitude<S>,
    sign: S,
    multiples: usize,
    bound: Option<PositiveMagnitude<S>>,
) -> Result<Option<[S; MAX_STENCIL_POINTS]>, SciOxideError> {
    let Some(bound) = bound else {
        return Ok(None);
    };
    // Nodes here are values of alpha along the direction, not coordinate
    // offsets. Capping at MAX / multiples keeps every multiple of the step
    // finite, and room / multiples keeps every node inside the bounds.
    let maximum = PositiveMagnitude::from_positive(S::MAX)?.divide_usize(multiples)?;
    let room = bound.divide_usize(multiples)?;
    let step = nominal.minimum(maximum).minimum(room);
    let Some(step) = step.representable_step()? else {
        return Ok(None);
    };
    let mut nodes = [S::ZERO; MAX_STENCIL_POINTS];
    for multiple in 1..=multiples {
        let magnitude = PositiveMagnitude::from_positive(step)?
            .multiply_usize(multiple)?
            .representable_step()?;
        let Some(magnitude) = magnitude else {
            return Ok(None);
        };
        nodes[multiple] = sign * magnitude;
        if nodes[multiple] == S::ZERO || (multiple > 1 && nodes[multiple] == nodes[multiple - 1]) {
            return Ok(None);
        }
    }
    Ok(Some(nodes))
}

fn directional_stencil<S: RealScalar>(
    nominal: PositiveMagnitude<S>,
    scheme: DifferenceScheme,
    bounds: DirectionalBounds<S>,
) -> Result<Option<Stencil<S>>, SciOxideError> {
    let fits = |room: Option<PositiveMagnitude<S>>, step: PositiveMagnitude<S>| {
        room.is_some_and(|room| !room.0.abs_cmp(step.0).is_lt())
    };
    let prefer_forward = match scheme {
        DifferenceScheme::Forward if fits(bounds.forward, nominal) => true,
        DifferenceScheme::Forward if fits(bounds.backward, nominal) => false,
        _ => match (bounds.forward, bounds.backward) {
            (Some(forward), Some(backward)) => !forward.0.abs_cmp(backward.0).is_lt(),
            (Some(_), None) => true,
            _ => false,
        },
    };
    let nodes = if scheme == DifferenceScheme::Central
        && fits(bounds.backward, nominal)
        && fits(bounds.forward, nominal)
    {
        scaled_directional_nodes(nominal, -S::ONE, 1, bounds.backward)?
            .zip(scaled_directional_nodes(
                nominal,
                S::ONE,
                1,
                bounds.forward,
            )?)
            .map(|(backward, forward)| ([backward[1], S::ZERO, forward[1], S::ZERO], 3))
    } else {
        let multiples = if scheme == DifferenceScheme::Central {
            2
        } else {
            1
        };
        let (sign, first, second) = if prefer_forward {
            (S::ONE, bounds.forward, bounds.backward)
        } else {
            (-S::ONE, bounds.backward, bounds.forward)
        };
        scaled_directional_nodes(nominal, sign, multiples, first)?
            .or(scaled_directional_nodes(nominal, -sign, multiples, second)?)
            .map(|nodes| (nodes, multiples + 1))
    };
    nodes
        .map(|(nodes, len)| {
            let scale = nodes[..len]
                .iter()
                .fold(S::ZERO, |maximum, value| maximum.max(value.abs()));
            let normalized = nodes.map(|value| value / scale);
            Ok(Stencil {
                offsets: nodes,
                weights: stencil_weights(&normalized[..len], 1)?,
                scale,
                derivative_order: 1,
                len,
            })
        })
        .transpose()
}

/// Chooses representable physical nodes, falling back to a smaller symmetric
/// stencil when a narrow interval cannot hold the inward one-sided nodes.
pub(super) fn directional_stencil_for_points<S: RealScalar>(
    nominal: PositiveMagnitude<S>,
    scheme: DifferenceScheme,
    room: DirectionalBounds<S>,
    x: &[S],
    direction: &[S],
    bounds: Option<&Bounds<S>>,
    scratch: (&mut [S], &mut [S]),
) -> Result<Option<Stencil<S>>, SciOxideError> {
    let (point, previous) = scratch;
    if let Some(stencil) = directional_stencil(nominal, scheme, room)?
        && distinct_directional_nodes(stencil, x, direction, bounds, point, previous)?
    {
        return Ok(Some(stencil));
    }
    if scheme == DifferenceScheme::Central
        && let (Some(backward), Some(forward)) = (room.backward, room.forward)
        && let Some(stencil) =
            directional_stencil(nominal.minimum(backward).minimum(forward), scheme, room)?
        && distinct_directional_nodes(stencil, x, direction, bounds, point, previous)?
    {
        return Ok(Some(stencil));
    }
    Ok(None)
}

// A nonzero parameter is not sufficient: two parameter nodes can round to the
// same physical vector. Check actual vectors without assuming a dominant lane.
fn distinct_directional_nodes<S: RealScalar>(
    stencil: Stencil<S>,
    x: &[S],
    direction: &[S],
    bounds: Option<&Bounds<S>>,
    point: &mut [S],
    previous: &mut [S],
) -> Result<bool, SciOxideError> {
    for (i, &alpha) in stencil.offsets[..stencil.len].iter().enumerate() {
        if alpha == S::ZERO {
            continue;
        }
        if !generated_directional_point(x, direction, alpha, bounds, point)? {
            return Ok(false);
        }
        for &earlier in &stencil.offsets[..i] {
            let _ = generated_directional_point(x, direction, earlier, bounds, previous)?;
            if point == previous {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

pub(super) fn step_size_limit_error(evaluations: usize, detail: &'static str) -> SciOxideError {
    SciOxideError::differentiation_failure(
        DifferentiationLocation::direction(),
        evaluations,
        SciOxideError::convergence_failure(
            ConvergenceReason::StepSizeLimit,
            ConvergenceDiagnostics::new().with_counts(
                (EvaluationCounts::new()
                    .with_provider_count(EvaluationRole::Function, evaluations))
                .into(),
            ),
            detail,
        ),
    )
}

pub(super) fn normalized_direction<S: RealScalar>(
    x: &[S],
    direction: &[S],
) -> Result<Option<PositiveMagnitude<S>>, SciOxideError> {
    // The largest relative component max_i |d_i| / max(1, |x_i|) measures how
    // fast the ray moves in the per-coordinate step scaling. `None` means that
    // the direction is exactly zero componentwise, which is the only zero test.
    x.iter()
        .copied()
        .zip(direction.iter().copied())
        .filter(|(_, component)| *component != S::ZERO)
        .try_fold(None, |maximum, (value, component)| {
            let ratio = PositiveMagnitude::ratio(component.abs(), value.abs().max(S::ONE))?;
            Ok::<Option<PositiveMagnitude<S>>, SciOxideError>(Some(
                maximum.map_or(ratio, |current: PositiveMagnitude<S>| {
                    current.maximum(ratio)
                }),
            ))
        })
}

pub(super) fn generated_directional_point<S: RealScalar>(
    x: &[S],
    direction: &[S],
    alpha: S,
    bounds: Option<&Bounds<S>>,
    output: &mut [S],
) -> Result<bool, SciOxideError> {
    let mut changed = false;
    for (variable, ((&value, &component), generated)) in
        x.iter().zip(direction).zip(output.iter_mut()).enumerate()
    {
        // The bound check is repeated after rounding because the rounded point can
        // step outside a bound that the exact alpha respected. If no coordinate
        // changes at all, rounding absorbed the whole perturbation and the stencil
        // carries no information.
        *generated = scioxide_core::checked_mul_add(alpha, component, value)?;
        changed |= *generated != value;
        if let Some((lower, upper)) = bounds.map(|bounds| bounds.as_slice()[variable])
            && (lower.is_some_and(|bound| *generated < bound)
                || upper.is_some_and(|bound| *generated > bound))
        {
            return Ok(false);
        }
    }
    Ok(changed)
}

// Fornberg (1988), first recursion. One implementation for small stack stencils
// and arbitrary prepared grids. Rows retain derivatives 0..=order.
pub(super) fn fornberg<S: RealScalar>(
    nodes: &[S],
    order: usize,
    table: &mut [S],
) -> Result<(), SciOxideError> {
    let width = checked_add(order, 1, scioxide_core::quantity!("fornberg_table"))?;
    if nodes.is_empty()
        || order >= nodes.len()
        || table.len()
            < checked_mul(
                nodes.len(),
                width,
                scioxide_core::quantity!("fornberg_table"),
            )?
    {
        return Err(invalid(
            scioxide_core::argument!("nodes"),
            "require nonempty nodes, order below node count, and sufficient scratch",
        ));
    }
    // Nodes are displacements from the evaluation point, so Fornberg's z is 0
    // and the (z - x) factors reduce to -x. Correspondence with the paper:
    // c1 = prod_{j<i-1}(x_{i-1} - x_j) from the previous row, c2 = the same
    // product for row i, c3 = x_i - x_j, c4 = x_i, and c5 = x_{i-1}. The products
    // grow or shrink geometrically with the node count, which is why callers
    // normalize nodes to O(1) and why every step checks for overflow and
    // underflow.
    table.fill(S::ZERO);
    table[0] = S::ONE;
    let mut c1 = S::ONE;
    let mut c4 = nodes[0];
    for i in 1..nodes.len() {
        let mut c2 = S::ONE;
        let c5 = c4;
        c4 = nodes[i];
        for j in 0..i {
            let c3 = nodes[i] - nodes[j];
            if !c3.is_finite() || c3 == S::ZERO {
                return Err(breakdown("nodes must remain distinct after normalization"));
            }
            c2 *= c3;
            if !c2.is_finite() || c2 == S::ZERO {
                return Err(breakdown(
                    "node product exceeds the recurrence's representable range",
                ));
            }
            if j == i - 1 {
                for k in (1..=order.min(i)).rev() {
                    let k_scalar = S::try_from_usize_exact(k)?;
                    table[i * width + k] = c1
                        * (k_scalar * table[(i - 1) * width + k - 1]
                            - c5 * table[(i - 1) * width + k])
                        / c2;
                }
                table[i * width] = -c1 * c5 * table[(i - 1) * width] / c2;
            }
            for k in (1..=order.min(i)).rev() {
                let k_scalar = S::try_from_usize_exact(k)?;
                table[j * width + k] =
                    (c4 * table[j * width + k] - k_scalar * table[j * width + k - 1]) / c3;
            }
            table[j * width] = c4 * table[j * width] / c3;
        }
        c1 = c2;
        if table[..(i + 1) * width].iter().any(|v| !v.is_finite()) {
            return Err(breakdown("finite-difference recurrence overflowed"));
        }
    }
    Ok(())
}

// Normalize the exact sum before scaling, retaining subnormal-range values
// until the physical derivative is formed.
pub(super) fn finish_scaled_steps<S: RealScalar>(
    sum: &RangeAccumulator<S>,
    steps: &[(S, usize)],
) -> Result<S, SciOxideError> {
    let mut value = sum.finish_scaled::<S>()?;
    for &(scale, order) in steps {
        let divisor = scioxide_core::Scaled::from_parts(scale, 0)?;
        for _ in 0..order {
            value = value.div(divisor)?;
        }
    }
    value.try_value()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn exact_contractions<S: RealScalar>() {
        for scale in [
            S::MIN_SUBNORMAL,
            S::MIN_POSITIVE,
            S::HALF,
            S::ONE,
            S::MAX / S::TWO,
        ] {
            for nodes in [
                [S::ZERO, S::ONE, S::ZERO],
                [S::ZERO, -S::ONE, S::ZERO],
                [-S::ONE, S::ZERO, S::ONE],
                [-S::ONE, S::ZERO, S::HALF],
            ] {
                let len = if nodes[0] == S::ZERO { 2 } else { 3 };
                let weights = stencil_weights(&nodes[..len], 1).unwrap();
                let mut table = [S::ZERO; 6];
                fornberg(&nodes[..len], 1, &mut table).unwrap();
                for i in 0..len {
                    assert!(
                        (weights[i] - table[2 * i + 1]).abs()
                            <= S::EPSILON * table[2 * i + 1].abs()
                    );
                }
                let stencil = Stencil {
                    offsets: [
                        nodes[0] * scale,
                        nodes[1] * scale,
                        nodes[2] * scale,
                        S::ZERO,
                    ],
                    weights,
                    scale,
                    derivative_order: 1,
                    len,
                };
                for (base, lower, upper) in [
                    (S::ONE, S::ONE, S::ONE),
                    (S::ZERO, S::MIN_SUBNORMAL, S::TWO * S::MIN_SUBNORMAL),
                    (S::ZERO, -S::MAX, S::MAX),
                    (S::MAX / S::TWO, -S::MAX, S::MAX),
                    (S::ONE, S::ONE - S::TWO * S::EPSILON, S::ONE + S::EPSILON),
                    (S::ZERO, S::MIN_SUBNORMAL, S::MIN_POSITIVE),
                ] {
                    let samples = if len == 2 {
                        [base, upper, S::ZERO]
                    } else {
                        [lower, base, upper]
                    };
                    let differences = samples.map(|value| value - base);
                    let values = if differences[..len].iter().all(|x| x.is_finite()) {
                        &differences[..len]
                    } else {
                        &samples[..len]
                    };
                    let mut exact = RangeAccumulator::new();
                    for (&weight, &value) in weights[..len].iter().zip(values) {
                        exact.add_product(weight, value).unwrap();
                    }
                    let expected = finish_scaled_steps(&exact, &[(scale, 1)]);
                    let actual = stencil_derivative(
                        stencil,
                        &samples[..len],
                        base,
                        &mut RangeAccumulator::new(),
                    );
                    match (actual, expected) {
                        (Ok(actual), Ok(expected)) => assert!(
                            (actual - expected).abs()
                                <= (S::TWO * S::EPSILON * expected.abs()).max(S::MIN_SUBNORMAL)
                        ),
                        (Err(actual), Err(expected)) => assert_eq!(actual.kind(), expected.kind()),
                        pair => panic!("stencil and exact contraction disagree: {pair:?}"),
                    }
                }
            }
        }
    }
    #[test]
    fn short_stencils_match_exact_f32() {
        exact_contractions::<f32>();
    }
    #[test]
    fn short_stencils_match_exact_f64() {
        exact_contractions::<f64>();
    }
    fn rounded_bounds<S: RealScalar>() {
        let tiny = S::EPSILON / S::try_from_usize_exact(16).unwrap();
        for sign in [S::ONE, -S::ONE] {
            let x = -sign;
            let bound = if sign > S::ZERO {
                (Some(-S::TWO), Some(-tiny))
            } else {
                (Some(tiny), Some(S::TWO))
            };
            for multiples in 1..=3 {
                let nodes = directional_nodes(x, S::ONE, sign, multiples, Some(bound)).unwrap();
                for &offset in &nodes[1..=multiples] {
                    let point = x + offset;
                    assert!(point >= bound.0.unwrap() && point <= bound.1.unwrap());
                    assert!(point != x);
                }
            }
        }
    }
    #[test]
    fn rounded_bounds_f32() {
        rounded_bounds::<f32>();
    }
    #[test]
    fn rounded_bounds_f64() {
        rounded_bounds::<f64>();
    }
}
