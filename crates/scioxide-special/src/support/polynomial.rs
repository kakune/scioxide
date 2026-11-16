//! Ascending-power evaluation with explicit arithmetic policies.
//!
//! Fused Horner and fused Clenshaw steps use `scioxide_core::fast_mul_add`: one
//! rounding where the target has hardware FMA, two otherwise. No policy calls a
//! software or library FMA. Compensated Horner uses the target-aware product
//! policy of `Wide`; its low word is kept.
use super::wide::{Wide, two_product};
use scioxide_core::RealScalar;

#[derive(Clone, Copy)]
pub(crate) enum PolynomialPolicy {
    Fused,
    FusedDescending,
    UnfusedDescending,
    MonicDescending,
    Unfused,
    Grouped,
}

/// Evaluate with the selected coefficient order and arithmetic policy.
/// Grouped chains use ascending powers and avoid portable out-of-line FMA calls.
#[inline(always)]
pub(crate) fn polynomial<S: RealScalar>(x: S, coefficients: &[S], policy: PolynomialPolicy) -> S {
    let n = coefficients.len();
    if matches!(
        policy,
        PolynomialPolicy::FusedDescending
            | PolynomialPolicy::UnfusedDescending
            | PolynomialPolicy::MonicDescending
    ) {
        let (leading, remaining) = if matches!(policy, PolynomialPolicy::MonicDescending) {
            (S::ONE, coefficients)
        } else if let Some((&first, rest)) = coefficients.split_first() {
            (first, rest)
        } else {
            return S::ZERO;
        };
        return remaining.iter().fold(leading, |value, &c| {
            if matches!(policy, PolynomialPolicy::FusedDescending) {
                scioxide_core::fast_mul_add(value, x, c)
            } else {
                value * x + c
            }
        });
    }
    if !matches!(policy, PolynomialPolicy::Grouped) {
        return coefficients.iter().rev().fold(S::ZERO, |value, &c| {
            if matches!(policy, PolynomialPolicy::Fused) {
                scioxide_core::fast_mul_add(value, x, c)
            } else {
                value * x + c
            }
        });
    }
    let square = x * x;
    if n <= 9 {
        // Short polynomials need only two independent chains.
        let complete = n - n % 2;
        let leading = coefficients.get(complete).copied().unwrap_or(S::ZERO);
        let [even, odd] = coefficients[..complete]
            .chunks_exact(2)
            .rev()
            .fold([leading, S::ZERO], |[even, odd], pair| {
                [even * square + pair[0], odd * square + pair[1]]
            });
        return even + x * odd;
    }
    let fourth = square * square;
    let [a, b, c, d] =
        coefficients
            .chunks(4)
            .rev()
            .fold([S::ZERO; 4], |[a, b, c, d], coefficients| {
                [
                    a * fourth + coefficients[0],
                    b * fourth + coefficients.get(1).copied().unwrap_or(S::ZERO),
                    c * fourth + coefficients.get(2).copied().unwrap_or(S::ZERO),
                    d * fourth + coefficients.get(3).copied().unwrap_or(S::ZERO),
                ]
            });
    (a + x * b) + square * (c + x * d)
}

#[inline(always)]
pub(crate) fn compensated(offset: Wide, coefficients: &[(f64, f64)], leading: f64) -> Wide {
    // Compensated Horner keeps the polynomial's rounding residual in
    // one low component, without renormalizing at every coefficient.
    coefficients
        .iter()
        .rev()
        .fold(Wide::from(leading), |sum, &(hi, lo)| {
            compensated_step(sum, offset, hi, lo)
        })
}

#[inline(always)]
pub(crate) fn compensated_step(sum: Wide, offset: Wide, ch: f64, cl: f64) -> Wide {
    let product = sum.hi * offset.hi;
    let hi = product + ch;
    let residual = if product.abs() >= ch.abs() {
        (product - hi) + ch
    } else {
        (ch - hi) + product
    };
    Wide {
        hi,
        lo: two_product(sum.hi, offset.hi).lo
            + residual
            + sum.lo * offset.hi
            + sum.hi * offset.lo
            + cl,
    }
}

/// One backwards Clenshaw recurrence over ascending coefficients.
///
/// The arithmetic callback selects fused or unfused arithmetic explicitly,
/// performs fallible/scaled operations and work accounting, and can carry a
/// derivative in the state. The caller finalizes T/U and full/half constant
/// conventions from the last two states without modifying the coefficient table.
#[inline(always)]
pub(crate) fn clenshaw<C, T: Copy, E>(
    coefficients: impl DoubleEndedIterator<Item = C>,
    zero: T,
    mut step: impl FnMut(T, T, C) -> Result<T, E>,
) -> Result<(T, T), E> {
    let mut first = zero;
    let mut second = zero;
    for coefficient in coefficients.rev() {
        let next = step(first, second, coefficient)?;
        second = first;
        first = next;
    }
    Ok((first, second))
}

/// Clenshaw evaluation with explicit constant weighting and arithmetic.
/// The running magnitudes bound rounding even when the final value cancels.
/// Interrupted tables keep their counted leading coefficients and bound
/// the omitted coefficients on [-1,1]. One coefficient is one work unit.
#[inline(always)]
pub(crate) fn chebyshev<S: RealScalar, C: Copy, E>(
    x: S,
    coefficients: &[C],
    half_constant: bool,
    fused: bool,
    convert: impl Fn(C) -> Result<S, E>,
    work: &mut super::Work,
) -> Result<(S, S), E> {
    let before = work.used;
    work.admit(coefficients.len());
    let count = work.used - before;
    let mut magnitude = S::ZERO;
    let (first, second) = clenshaw(coefficients[..count].iter().skip(1), S::ZERO, |a, b, &c| {
        let c = convert(c)?;
        let product = S::TWO * x * a;
        magnitude += product.abs() + b.abs() + c.abs();
        Ok::<_, E>(if fused {
            scioxide_core::fast_mul_add(S::TWO * x, a, c - b)
        } else {
            product - b + c
        })
    })?;
    let weight = if half_constant { S::HALF } else { S::ONE };
    let constant = if count == 0 {
        S::ZERO
    } else {
        convert(coefficients[0])? * weight
    };
    magnitude += (x * first).abs() + second.abs() + constant.abs();
    let value = if fused {
        scioxide_core::fast_mul_add(x, first, constant - second)
    } else {
        x * first - second + constant
    };
    let omitted =
        coefficients[count..]
            .iter()
            .enumerate()
            .try_fold(S::ZERO, |sum, (index, &c)| {
                Ok::<_, E>(
                    sum + convert(c)?.abs() * if count + index == 0 { weight } else { S::ONE },
                )
            })?;
    let truncation = coefficients
        .last()
        .copied()
        .map(&convert)
        .transpose()?
        .map_or(S::ZERO, |c| c.abs());
    Ok((
        value,
        omitted + truncation + S::TWO * S::TWO * S::EPSILON * magnitude,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn chebyshev_bounds_survive_cancellation_and_interruption() {
        for fused in [false, true] {
            let mut work = crate::support::Work::new(4);
            let (value, error) = chebyshev(
                1.0,
                &[-1.0, 0.0, 1.0, 0.0],
                false,
                fused,
                Ok::<_, core::convert::Infallible>,
                &mut work,
            )
            .unwrap();
            assert_eq!(value, 0.0);
            assert!(error > f64::EPSILON);
            assert!(work.complete);
            assert_eq!(work.used, 4);
            for half in [false, true] {
                let mut work = crate::support::Work::new(1);
                let (value, error) = chebyshev(
                    0.25,
                    &[2.0, 3.0, 4.0],
                    half,
                    fused,
                    Ok::<_, core::convert::Infallible>,
                    &mut work,
                )
                .unwrap();
                assert_eq!(value, if half { 1.0 } else { 2.0 });
                assert!(error >= 7.0);
                assert!(!work.complete);
                assert_eq!(work.used, 1);
            }
        }
    }

    #[test]
    fn policies_preserve_coefficient_order_and_constant_cases() {
        for policy in [
            PolynomialPolicy::Fused,
            PolynomialPolicy::Unfused,
            PolynomialPolicy::Grouped,
        ] {
            assert_eq!(polynomial(0.5_f64, &[], policy), 0.0);
            assert_eq!(polynomial(0.5_f64, &[3.0], policy), 3.0);
            // 1 + 2x + 3x² + 4x³ = 3.25 at x=1/2.
            assert!((polynomial(0.5_f64, &[1.0, 2.0, 3.0, 4.0], policy) - 3.25).abs() < 1e-15);
            assert!((polynomial(0.5_f32, &[1.0, 2.0, 3.0, 4.0], policy) - 3.25).abs() < 1e-6);
        }
        assert!(
            (polynomial(
                0.5_f64,
                &[4.0, 3.0, 2.0, 1.0],
                PolynomialPolicy::FusedDescending
            ) - 3.25)
                .abs()
                < 1e-15
        );
        assert!(
            (polynomial(
                0.5_f64,
                &[4.0, 3.0, 2.0, 1.0],
                PolynomialPolicy::UnfusedDescending
            ) - 3.25)
                .abs()
                < 1e-15
        );
        assert!(
            (polynomial(0.5_f64, &[3.0, 2.0, 1.0], PolynomialPolicy::MonicDescending) - 2.875)
                .abs()
                < 1e-15
        );
        // Exercise both grouped widths, including an incomplete final group.
        for n in 0..=15 {
            let coefficients = [1.0_f64; 15];
            let expected = 2.0 * (1.0 - 0.5_f64.powi(n as i32));
            assert!(
                (polynomial(0.5, &coefficients[..n], PolynomialPolicy::Grouped) - expected).abs()
                    < 1e-15
            );
        }
    }

    #[test]
    fn fused_policy_retains_a_product_residual_with_hardware_fma() {
        let epsilon = f64::EPSILON;
        let x = 1.0 - epsilon;
        let coefficients = [-1.0, 1.0 + epsilon];
        let fused = polynomial(x, &coefficients, PolynomialPolicy::Fused);
        if scioxide_core::HARDWARE_FMA {
            assert!((fused + epsilon * epsilon).abs() < epsilon.powi(3));
        } else {
            assert_eq!(fused, 0.0);
        }
        assert_eq!(polynomial(x, &coefficients, PolynomialPolicy::Unfused), 0.0);
    }

    #[test]
    fn clenshaw_states_support_t_u_weights_and_derivatives() {
        let x = 0.25_f64;
        let coefficients = [2.0, 3.0, 4.0];
        let ((b1, d1), (b2, d2)) =
            clenshaw(coefficients.iter(), (0.0, 0.0), |(b1, d1), (b2, d2), &c| {
                Ok::<_, core::convert::Infallible>((
                    2.0 * x * b1 - b2 + c,
                    2.0 * b1 + 2.0 * x * d1 - d2,
                ))
            })
            .unwrap();
        let t = b1 - x * b2;
        let derivative = d1 - b2 - x * d2;
        assert!((t - (2.0 + 3.0 * x + 4.0 * (2.0 * x * x - 1.0))).abs() < 1e-15_f64);
        assert!((derivative - (3.0 + 16.0 * x)).abs() < 1e-15);
        assert!((b1 - (2.0 + 6.0 * x + 4.0 * (4.0 * x * x - 1.0))).abs() < 1e-15);
        let (first, second) = clenshaw(coefficients[1..].iter(), 0.0, |a, b, &c| {
            Ok::<_, core::convert::Infallible>(2.0 * x * a - b + c)
        })
        .unwrap();
        assert!((x * first - second + coefficients[0] * 0.5 - (t - 1.0)).abs() < 1e-15);
    }

    #[test]
    fn clenshaw_propagates_failure_before_processing_more_coefficients() {
        let mut admitted = 0;
        let result = clenshaw([1.0, 2.0, 3.0].iter(), 0.0, |a, b, &c| {
            if admitted == 1 {
                return Err("budget");
            }
            admitted += 1;
            Ok(a - b + c)
        });
        assert_eq!(result, Err("budget"));
        assert_eq!(admitted, 1);
    }
}
