//! Second-form barycentric evaluation and Chebyshev points.
//!
//! Polynomial interpolants (sample values) and rational functions (independent
//! numerator coefficients) use the same evaluation: terms relative to a
//! reference node, compensated sums and a restart with a wider exponent range.
//! Mapping a physical coordinate to the nodes and choosing the cancellation
//! threshold are left to the caller.

use crate::admission::{
    Arithmetic, ArithmeticError, ArithmeticResult, Extended, Native, candidate, extended,
};
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{AlgebraicScalar, NumericScalar, NumericalIssue, RealScalar, SciOxideError};

/// Holds the value, optional derivative and condition numbers of a barycentric
/// quotient.
#[derive(Clone, Copy, Debug)]
pub struct BarycentricEvaluation<C: NumericScalar> {
    /// The value `N(z)/D(z)`.
    pub value: C,
    /// The derivative with respect to `z`, or zero when it was not requested.
    pub derivative: C,
    /// `Σⱼ |bⱼ/(z − zⱼ)| / |Σⱼ bⱼ/(z − zⱼ)|`, the condition number of the
    /// denominator sum.
    pub denominator_condition: C::Real,
    /// `Σⱼ |aⱼ/(z − zⱼ)| / |Σⱼ bⱼ/(z − zⱼ)|`, or zero when not requested.
    pub numerator_bound: C::Real,
}

/// Evaluates a quotient in second (true) barycentric form at a finite point.
///
/// ```text
/// r(z) = N(z) / D(z),   N(z) = Σⱼ aⱼ/(z − zⱼ),   D(z) = Σⱼ bⱼ/(z − zⱼ),
/// ```
///
/// where `zⱼ` are the `nodes` and `bⱼ` the `weights`. With `SAMPLES`,
/// `aⱼ = bⱼ fⱼ` for the sample values `fⱼ = coefficient(j)`, which is the
/// barycentric interpolation polynomial. Without it, `aⱼ = coefficient(j)` are
/// independent numerator coefficients of a rational function. `DERIVATIVE`
/// also computes `r′(z)`, and `ENVELOPE` computes the numerator condition
/// number. Terms are taken relative to the term of node `reference`, which must
/// have a nonzero weight; the node nearest `z` is the usual choice. For sample
/// values the value at `reference` is subtracted first, which keeps the
/// accuracy near a node. At a node the result is its limiting value, and
/// rational terms with `aⱼ = bⱼ = 0` are skipped. A positive `cancellation`
/// rejects points where `|D(z)| ≤ cancellation · Σⱼ |bⱼ/(z − zⱼ)|`; zero rejects
/// only `D(z) = 0`.
///
/// The nodes must be finite, distinct and nonempty, the coefficients finite,
/// and `coefficient` defined for every node index. These are not checked here.
/// The lengths, `reference`, `cancellation` and `point` are checked. The sums
/// are compensated. If a value overflows or underflows, the whole sum is
/// recomputed with a wider exponent range. The cost is O(n) and the function
/// does not allocate.
///
/// # Errors
///
/// - `DimensionMismatch` if `nodes` and `weights` differ in length.
/// - `InvalidInput` if `reference` is out of range or has a zero weight, or
///   `cancellation` is negative or not finite.
/// - `NonFinite` if `point` is NaN or infinite.
/// - `NumericalInstability` with `Cancellation` if the cancellation test
///   rejects the point, with `Breakdown` at a pole on a node, or with
///   `Overflow` if a result is not representable.
///
/// ```
/// use scioxide_poly::backend::barycentric::evaluate;
///
/// // The line through (−1, 1) and (1, 3) is 2 + x, so r(0.5) = 2.5.
/// let nodes = [-1.0_f64, 1.0];
/// let values = [1.0, 3.0];
/// let result =
///     evaluate::<_, _, false, true, false>(&nodes, &[1.0, -1.0], |j| values[j], 0.5, 1, 0.0)?;
/// assert!((result.value - 2.5).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation",
/// *SIAM Rev.* 46(3), 2004, 501–517.
#[inline]
pub fn evaluate<
    C: NumericScalar,
    F: Fn(usize) -> C,
    const DERIVATIVE: bool,
    const SAMPLES: bool,
    const ENVELOPE: bool,
>(
    nodes: &[C],
    weights: &[C],
    coefficient: F,
    point: C,
    reference: usize,
    cancellation: C::Real,
) -> Result<BarycentricEvaluation<C>, SciOxideError> {
    if nodes.len() != weights.len() {
        return Err(SciOxideError::dimension_mismatch(
            [nodes.len()],
            [weights.len()],
        ));
    }
    if reference >= nodes.len()
        || weights[reference] == C::ZERO
        || cancellation < C::Real::ZERO
        || !cancellation.is_finite()
    {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("reference"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "nonzero reference weight and nonnegative finite cancellation threshold required",
        ));
    }
    if !point.has_finite_components() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "finite barycentric query required",
        ));
    }
    if let Some(result) = candidate(fold::<C, Native, _, DERIVATIVE, SAMPLES, ENVELOPE>(
        nodes,
        weights,
        &coefficient,
        point,
        reference,
        cancellation,
        |a, b| a <= b,
    ))? {
        return Ok(BarycentricEvaluation {
            value: result.0,
            derivative: result.1,
            denominator_condition: result.2,
            numerator_bound: result.3,
        });
    }
    let result = extended(fold::<C, Extended, _, DERIVATIVE, SAMPLES, ENVELOPE>(
        nodes,
        weights,
        &coefficient,
        point,
        reference,
        cancellation,
        |a, b| a.abs_le(b),
    ))?;
    Ok(BarycentricEvaluation {
        value: result.0.round()?,
        derivative: result.1.round()?,
        denominator_condition: result.2.round()?,
        numerator_bound: result.3.round()?,
    })
}

/// One compensated sum in either arithmetic representation.
struct Sum<V> {
    value: V,
    correction: V,
}
impl<V: Copy> Sum<V> {
    fn zero(zero: V) -> Self {
        Self {
            value: zero,
            correction: zero,
        }
    }
    #[inline]
    fn add<C: NumericScalar, A: Arithmetic<C, Value = V>>(
        &mut self,
        term: V,
    ) -> ArithmeticResult<()> {
        let next = A::checked(A::add(self.value, term)?)?;
        // Error-free TwoSum, componentwise for complex fields. Unlike Kahan,
        // retaining the correction also recovers a small term bracketed by
        // large cancelling terms, as the extended accumulator does.
        let recovered = A::sub(next, self.value)?;
        let error = A::add(
            A::sub(self.value, A::sub(next, recovered)?)?,
            A::sub(term, recovered)?,
        )?;
        self.correction = A::add(self.correction, error)?;
        self.value = next;
        Ok(())
    }

    #[inline]
    fn total<C: NumericScalar, A: Arithmetic<C, Value = V>>(&self) -> ArithmeticResult<V> {
        A::checked(A::add(self.value, self.correction)?)
    }
}

type State<C, A> = (
    <A as Arithmetic<C>>::Value,
    <A as Arithmetic<C>>::Value,
    <A as Arithmetic<C>>::Bound,
    <A as Arithmetic<C>>::Bound,
);

#[inline]
fn divide<C: NumericScalar, A: Arithmetic<C>>(
    a: A::Value,
    b: A::Value,
) -> ArithmeticResult<A::Value> {
    let result = A::checked(A::div(a, b)?)?;
    if A::NATIVE && result == A::scalar(C::ZERO)? && a != A::scalar(C::ZERO)? {
        return Err(ArithmeticError::Retry);
    }
    Ok(result)
}

#[inline]
fn fold<
    C: NumericScalar,
    A: Arithmetic<C>,
    F: Fn(usize) -> C,
    const DERIVATIVE: bool,
    const SAMPLES: bool,
    const ENVELOPE: bool,
>(
    nodes: &[C],
    weights: &[C],
    coefficient: &F,
    point: C,
    reference: usize,
    cancellation: C::Real,
    less_equal: impl Fn(A::Bound, A::Bound) -> bool,
) -> ArithmeticResult<State<C, A>> {
    let zero = A::scalar(C::ZERO)?;
    let point = A::checked_scalar(point)?;
    let anchor_node = A::checked_scalar(nodes[reference])?;
    let gap = A::checked(A::sub(point, anchor_node)?)?;
    let anchor_weight = A::checked_scalar(weights[reference])?;
    let anchor_value = if SAMPLES {
        A::checked_scalar(coefficient(reference))?
    } else {
        zero
    };
    let normalized_gap = divide::<C, A>(gap, anchor_weight)?;
    let mut numerator = Sum::zero(zero);
    let mut denominator = Sum::zero(zero);
    let mut numerator_d = Sum::zero(zero);
    let mut denominator_d = Sum::zero(zero);
    let mut mass = A::bound_scalar(C::Real::ZERO)?;
    let mut bound = A::bound_scalar(C::Real::ZERO)?;
    for (index, (&node, &weight)) in nodes.iter().zip(weights).enumerate() {
        // Prepared nodes are finite and enter arithmetic only through checked
        // differences. Their absolute coordinates need no product-range admission.
        let node = A::scalar(node)?;
        let weight = A::checked_scalar(weight)?;
        // Value-only samples likewise enter products only after their checked
        // anchor difference. Envelopes and independent numerators use raw values.
        let value = if SAMPLES && !ENVELOPE {
            A::scalar(coefficient(index))?
        } else {
            A::checked_scalar(coefficient(index))?
        };
        let distance = A::checked(A::sub(point, node)?)?;
        if !SAMPLES && distance == zero {
            if weight != zero {
                let quotient = divide::<C, A>(value, weight)?;
                let mut derivative = Sum::zero(zero);
                if DERIVATIVE {
                    for (other_index, (&other_node, &other_weight)) in
                        nodes.iter().zip(weights).enumerate()
                    {
                        if other_index == index {
                            continue;
                        }
                        let other_node = A::checked_scalar(other_node)?;
                        let other_weight = A::checked_scalar(other_weight)?;
                        let other_value = A::checked_scalar(coefficient(other_index))?;
                        let delta = A::checked(A::sub(
                            other_value,
                            A::checked(A::mul(quotient, other_weight)?)?,
                        )?)?;
                        let distance = A::checked(A::sub(node, other_node)?)?;
                        derivative.add::<C, A>(divide::<C, A>(
                            divide::<C, A>(delta, distance)?,
                            weight,
                        )?)?;
                    }
                }
                return Ok((
                    quotient,
                    derivative.total::<C, A>()?,
                    A::bound_scalar(C::Real::ONE)?,
                    if ENVELOPE {
                        A::magnitude(quotient)?
                    } else {
                        A::bound_scalar(C::Real::ZERO)?
                    },
                ));
            }
            if value != zero {
                return Err(breakdown("pole at support point").into());
            }
            continue;
        }
        // Derivative queries reuse one checked reciprocal of each non-anchor
        // distance. A tiny distance that cannot admit its reciprocal restarts in
        // extended range; the tiny anchor gap is never inverted.
        let reciprocal = if DERIVATIVE && index != reference {
            divide::<C, A>(A::scalar(C::ONE)?, distance)?
        } else {
            zero
        };
        let (term, factor) = if index == reference {
            (
                A::scalar(C::ONE)?,
                if SAMPLES {
                    zero
                } else {
                    divide::<C, A>(A::scalar(C::ONE)?, anchor_weight)?
                },
            )
        } else {
            // Do not form a reciprocal of the tiny anchor gap.
            let factor = if DERIVATIVE {
                A::checked(A::mul(normalized_gap, reciprocal)?)?
            } else {
                divide::<C, A>(normalized_gap, distance)?
            };
            (A::checked(A::mul(weight, factor)?)?, factor)
        };
        let difference = if SAMPLES {
            A::checked(A::sub(value, anchor_value)?)?
        } else {
            zero
        };
        let contribution = if SAMPLES {
            A::checked(A::mul(term, difference)?)?
        } else {
            A::checked(A::mul(value, factor)?)?
        };
        numerator.add::<C, A>(contribution)?;
        denominator.add::<C, A>(term)?;
        mass = A::checked_bound(A::bound_add(mass, A::magnitude(term)?)?)?;
        if ENVELOPE {
            let raw = if SAMPLES {
                A::checked(A::mul(term, value)?)?
            } else {
                contribution
            };
            bound = A::checked_bound(A::bound_add(bound, A::magnitude(raw)?)?)?;
        }
        if DERIVATIVE && index != reference {
            let node_gap = A::checked(A::sub(anchor_node, node)?)?;
            let first = A::checked(A::mul(
                divide::<C, A>(node_gap, anchor_weight)?,
                reciprocal,
            )?)?;
            let factor_d = A::checked(A::mul(first, reciprocal)?)?;
            let term_d = A::checked(A::mul(weight, factor_d)?)?;
            let contribution_d = if SAMPLES {
                A::checked(A::mul(term_d, difference)?)?
            } else {
                A::checked(A::mul(value, factor_d)?)?
            };
            numerator_d.add::<C, A>(contribution_d)?;
            denominator_d.add::<C, A>(term_d)?;
        }
    }
    let d = denominator.total::<C, A>()?;
    if d == zero {
        return Err(if A::NATIVE {
            ArithmeticError::Retry
        } else {
            if cancellation > C::Real::ZERO {
                SciOxideError::numerical_instability(
                    NumericalIssue::Cancellation,
                    "barycentric denominator cancelled to an unusable value",
                )
                .into()
            } else {
                breakdown("zero denominator: pole or indeterminate cancellation").into()
            }
        });
    }
    let magnitude = A::magnitude(d)?;
    let cutoff = A::bound_mul(mass, A::bound_scalar(cancellation)?)?;
    if cancellation > C::Real::ZERO && less_equal(magnitude, cutoff) {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Cancellation,
            "barycentric denominator cancelled to an unusable value",
        )
        .into());
    }
    let correction = divide::<C, A>(numerator.total::<C, A>()?, d)?;
    let value = if SAMPLES {
        A::checked(A::add(anchor_value, correction)?)?
    } else {
        correction
    };
    let derivative = if DERIVATIVE {
        divide::<C, A>(
            A::checked(A::sub(
                numerator_d.total::<C, A>()?,
                A::checked(A::mul(correction, denominator_d.total::<C, A>()?)?)?,
            )?)?,
            d,
        )?
    } else {
        zero
    };
    Ok((
        value,
        derivative,
        A::checked_bound(A::bound_div(mass, magnitude)?)?,
        if ENVELOPE {
            A::checked_bound(A::bound_div(bound, magnitude)?)?
        } else {
            A::bound_scalar(C::Real::ZERO)?
        },
    ))
}

fn breakdown(detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(NumericalIssue::Breakdown, detail)
}

/// Selects the Chebyshev points of [`chebyshev_points`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChebyshevPoints {
    /// Zeros of `Tₙ`, `cos((2j + 1)π/(2n))`, all inside `(−1, 1)`.
    Roots,
    /// Extrema of `T₍ₙ₋₁₎` (Chebyshev–Lobatto points), `cos(jπ/(n − 1))`,
    /// including both endpoints.
    Extrema,
}

/// Returns `n` Chebyshev points on `[−1, 1]` in ascending order, with their
/// barycentric weights.
///
/// The points are computed as sines of the angles that
/// [`OrthogonalFamily::zeros`](crate::OrthogonalFamily::zeros) uses, so they are
/// exactly symmetric, with an exact zero in the middle for odd `n`. The
/// endpoints of the extrema are exactly `±1`. The weights are those of Berrut
/// and Trefethen (2004), alternating in sign and positive at the first point:
/// `sin θⱼ` for the zeros, where `θⱼ = (2j + 1)π/(2n)`, scaled to a maximum of
/// one, and `1` with `1/2` at both ends for the extrema. A single zero is `0`
/// with weight `1`. This takes O(n) work and allocates two vectors of length
/// `n`.
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `n` is zero, or less than two for the
///   extrema.
/// - `NumericalInstability` (`Underflow`) if a weight is not representable.
/// - `ScalarConversion` if an angle index is not exactly representable.
/// - `AllocationFailure` if the vectors cannot be allocated.
///
/// ```
/// use scioxide_poly::backend::barycentric::{ChebyshevPoints, chebyshev_points};
///
/// let (nodes, weights) = chebyshev_points::<f64>(3, ChebyshevPoints::Extrema)?;
/// assert_eq!(nodes, [-1.0, 0.0, 1.0]);
/// assert_eq!(weights, [0.5, -1.0, 0.5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn chebyshev_points<R: RealScalar>(
    n: usize,
    points: ChebyshevPoints,
) -> Result<(Vec<R>, Vec<R>), SciOxideError> {
    let extrema = points == ChebyshevPoints::Extrema;
    if n == 0 || (extrema && n < 2) {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("n"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Chebyshev roots need one point and extrema two",
        ));
    }
    let order = if extrema { n - 1 } else { n };
    let mut nodes = try_filled(n, R::ZERO, scioxide_core::quantity!("chebyshev_points"))?;
    let mut weights = try_filled(n, R::ONE, scioxide_core::quantity!("chebyshev_weights"))?;
    for index in 0..n / 2 {
        let angle = crate::orthogonal::zeros::chebyshev_angle::<R>(n, order, index)?;
        let (node, weight) = if extrema && index == 0 {
            (R::ONE, R::HALF)
        } else if extrema {
            (angle.sin(), R::ONE)
        } else {
            (angle.sin(), angle.cos())
        };
        nodes[index] = -node;
        nodes[n - 1 - index] = node;
        weights[index] = weight;
        weights[n - 1 - index] = weight;
    }
    let maximum = weights.iter().fold(R::ZERO, |maximum, w| maximum.max(*w));
    for (index, weight) in weights.iter_mut().enumerate() {
        let magnitude = if extrema { *weight } else { *weight / maximum };
        if magnitude == R::ZERO || !magnitude.is_finite() {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Underflow,
                "Chebyshev barycentric weight is not representable",
            ));
        }
        *weight = if index % 2 == 0 {
            magnitude
        } else {
            -magnitude
        };
    }
    Ok((nodes, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    fn close<C: NumericScalar>(actual: C, expected: C) {
        let tolerance = C::Real::try_from_usize_exact(128).unwrap() * C::Real::EPSILON;
        assert!((actual - expected).abs() <= tolerance * expected.abs().max(C::Real::ONE));
    }

    fn agreement<C: NumericScalar>() {
        let real = |value| C::from_real(C::Real::try_from_f64_lossy(value).unwrap());
        let nodes = [real(-1.0), real(0.0), real(1.0)];
        let weights = [real(0.5), real(-1.0), real(0.5)];
        let values = [real(2.0), real(1.0), real(4.0)];
        let point = C::from_components(
            C::Real::HALF,
            if C::IS_REAL {
                C::Real::ZERO
            } else {
                C::Real::HALF
            },
        );
        let coefficients = |j| values[j];
        let native = candidate(fold::<C, Native, _, true, true, true>(
            &nodes,
            &weights,
            &coefficients,
            point,
            1,
            C::Real::ZERO,
            |a, b| a <= b,
        ))
        .unwrap()
        .unwrap();
        let wide = extended(fold::<C, Extended, _, true, true, true>(
            &nodes,
            &weights,
            &coefficients,
            point,
            1,
            C::Real::ZERO,
            |a, b| a.abs_le(b),
        ))
        .unwrap();
        let expected = real(1.0) + point + real(2.0) * point * point;
        close(native.0, expected);
        close(native.1, real(1.0) + real(4.0) * point);
        close(native.0, wide.0.round().unwrap());
        close(native.1, wide.1.round().unwrap());
        close(
            C::from_real(native.2),
            C::from_real(wide.2.round().unwrap()),
        );
        close(
            C::from_real(native.3),
            C::from_real(wide.3.round().unwrap()),
        );
        let numerator = |j| weights[j] * values[j];
        let rational = evaluate::<C, _, true, false, true>(
            &nodes,
            &weights,
            numerator,
            point,
            1,
            C::Real::ZERO,
        )
        .unwrap();
        close(rational.value, expected);
        close(rational.derivative, native.1);
        close(
            C::from_real(rational.numerator_bound),
            C::from_real(native.3),
        );
        for (index, &node) in nodes.iter().enumerate() {
            let result = evaluate::<C, _, true, false, true>(
                &nodes,
                &weights,
                numerator,
                node,
                index,
                C::Real::ZERO,
            )
            .unwrap();
            close(result.value, values[index]);
            close(result.derivative, real(1.0) + real(4.0) * node);
        }
    }

    #[test]
    fn value_only_admits_checked_differences_of_large_prepared_inputs() {
        for (nodes, samples, point, expected) in [
            ([0.0_f64, 4e153], [1.0, 3.0], 2e153, 2.0),
            ([-1.0, 1.0], [2e153, 4e153], 0.0, 3e153),
        ] {
            let weights = [1.0, -1.0];
            let values = |j| samples[j];
            let native = candidate(fold::<_, Native, _, false, true, false>(
                &nodes,
                &weights,
                &values,
                point,
                0,
                0.0,
                |a, b| a <= b,
            ))
            .unwrap()
            .unwrap();
            let wide = extended(fold::<_, Extended, _, false, true, false>(
                &nodes,
                &weights,
                &values,
                point,
                0,
                0.0,
                |a, b| a.abs_le(b),
            ))
            .unwrap();
            close(native.0, expected);
            close(native.0, wide.0.round().unwrap());
        }
    }

    #[test]
    fn native_and_extended_values_derivatives_and_envelopes_agree() {
        agreement::<f32>();
        agreement::<f64>();
        agreement::<Complex<f32>>();
        agreement::<Complex<f64>>();
    }

    #[test]
    fn extreme_coordinates_and_one_ulp_queries_force_extended_restart() {
        for scale in [1e-300_f64, 1e300] {
            let nodes = [-scale, scale];
            let weights = [1.0, -1.0];
            let samples = [1.0, 3.0];
            for point in [scale * 0.25, scale.next_after(0.0)] {
                let coefficients = |j| samples[j];
                assert!(
                    candidate(fold::<f64, Native, _, false, true, false>(
                        &nodes,
                        &weights,
                        &coefficients,
                        point,
                        1,
                        0.0,
                        |a, b| a <= b,
                    ))
                    .unwrap()
                    .is_none()
                );
                let result = evaluate::<_, _, false, true, false>(
                    &nodes,
                    &weights,
                    coefficients,
                    point,
                    1,
                    0.0,
                )
                .unwrap();
                close(result.value, 2.0 + point / scale);
                let numerator = |j| weights[j] * samples[j];
                let rational = evaluate::<_, _, false, false, false>(
                    &nodes, &weights, numerator, point, 1, 0.0,
                )
                .unwrap();
                close(rational.value, result.value);
            }
        }
    }

    #[test]
    fn overflowing_differences_and_underflowed_terms_restart() {
        let nodes = [-f64::MAX, f64::MAX];
        let result = evaluate::<_, _, false, true, false>(
            &nodes,
            &[1.0, -1.0],
            |j| [1.0, 3.0][j],
            f64::MAX / 2.0,
            1,
            0.0,
        )
        .unwrap();
        close(result.value, 2.5);
        let nodes = [-1.0, 1.0];
        let weights = [1e-300, -1e300];
        let coefficients = |j| [1e-300, -3e300][j];
        assert!(
            candidate(fold::<f64, Native, _, false, false, false>(
                &nodes,
                &weights,
                &coefficients,
                0.5,
                1,
                0.0,
                |a, b| a <= b,
            ))
            .unwrap()
            .is_none()
        );
        close(
            evaluate::<_, _, false, false, false>(&nodes, &weights, coefficients, 0.5, 1, 0.0)
                .unwrap()
                .value,
            3.0,
        );
    }

    #[test]
    fn compensated_denominator_recovers_terms_between_cancelling_large_terms() {
        let nodes = [-1.0_f64, -0.5, 1.0];
        let weights = [1e16, 0.5, 1e16];
        let result =
            evaluate::<_, _, false, false, false>(&nodes, &weights, |j| weights[j], 0.0, 1, 0.0)
                .unwrap();
        close(result.value, 1.0);
        close(result.denominator_condition, 2e16);
    }

    #[test]
    fn independent_coefficients_preserve_support_limits_and_cancellation_policy() {
        let nodes = [-1.0_f64, 0.0, 1.0];
        let weights = [1.0, 0.0, -1.0];
        let numerator = |j| [1.0, 0.0, -3.0][j];
        close(
            evaluate::<_, _, false, false, false>(&nodes, &weights, numerator, 0.0, 0, 0.0)
                .unwrap()
                .value,
            2.0,
        );
        close(
            evaluate::<_, _, false, false, false>(&nodes, &weights, numerator, 1.0, 2, 0.0)
                .unwrap()
                .value,
            3.0,
        );
        assert!(
            evaluate::<_, _, false, false, false>(
                &nodes,
                &weights,
                |j| [1.0, 1.0, -3.0][j],
                0.0,
                0,
                0.0
            )
            .is_err()
        );
        assert!(
            evaluate::<_, _, false, false, false>(&[-1.0, 1.0], &[1.0, 1.0], |_| 1.0, 0.0, 0, 0.0)
                .is_err()
        );
        assert!(
            evaluate::<_, _, false, false, false>(
                &[-1.0, 1.0],
                &[1.0, 1.0],
                |_| 1.0,
                1e-10,
                0,
                1e-8
            )
            .is_err()
        );
        assert!(
            evaluate::<_, _, false, false, false>(
                &[-1.0, 1.0],
                &[1.0, 1.0],
                |_| 1.0,
                1e-10,
                0,
                0.0
            )
            .is_ok()
        );
    }

    #[test]
    fn chebyshev_points_are_symmetric_with_alternating_weights() {
        fn check<R: RealScalar>() {
            let tolerance = R::try_from_usize_exact(8).unwrap() * R::EPSILON;
            for n in [1, 2, 3, 4, 7, 64] {
                let (nodes, weights) = chebyshev_points::<R>(n, ChebyshevPoints::Roots).unwrap();
                let mut maximum = R::ZERO;
                for j in 0..n {
                    assert_eq!(nodes[j], -nodes[n - 1 - j]);
                    let theta = R::PI * R::try_from_usize_exact(2 * j + 1).unwrap()
                        / R::try_from_usize_exact(2 * n).unwrap();
                    assert!((nodes[j] + theta.cos()).abs() <= tolerance);
                    assert_eq!(weights[j] < R::ZERO, j % 2 == 1);
                    maximum = maximum.max(weights[j].abs());
                }
                assert_eq!(maximum, R::ONE);
                if n == 1 {
                    assert_eq!((nodes[0], weights[0]), (R::ZERO, R::ONE));
                }
            }
            for n in [2, 3, 5, 64] {
                let (nodes, weights) = chebyshev_points::<R>(n, ChebyshevPoints::Extrema).unwrap();
                assert_eq!((nodes[0], nodes[n - 1]), (-R::ONE, R::ONE));
                for j in 0..n {
                    assert_eq!(nodes[j], -nodes[n - 1 - j]);
                    let magnitude = if j == 0 || j == n - 1 {
                        R::HALF
                    } else {
                        R::ONE
                    };
                    let sign = if j % 2 == 0 { R::ONE } else { -R::ONE };
                    assert_eq!(weights[j], sign * magnitude);
                }
            }
        }
        check::<f32>();
        check::<f64>();
        for (n, points) in [(0, ChebyshevPoints::Roots), (1, ChebyshevPoints::Extrema)] {
            let error = chebyshev_points::<f64>(n, points).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        }
    }
}
