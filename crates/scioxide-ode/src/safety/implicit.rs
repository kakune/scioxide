//! Native implicit equations with replay from original finite operands.
use super::*;
use scioxide_core::RangeAccumulator;

/// Accumulate one weighted component without forming the step-weight product.
#[inline(always)]
fn weighted<T: NumericScalar>(
    amplified: bool,
    mut terms: impl Iterator<Item = (T::Real, T)> + Clone,
    imaginary: bool,
) -> (T::Real, bool) {
    let mut sum = T::Real::ZERO;
    let mut ordinary = true;
    for (weight, value) in terms.clone() {
        if weight == T::Real::ZERO {
            continue;
        }
        let value = component(value, imaginary);
        sum = if weight == T::Real::ONE {
            sum + value
        } else {
            scioxide_core::fast_mul_add(weight, value, sum)
        };
        if amplified {
            ordinary &= !native_product_loses_range(weight, value);
        }
    }
    // Several products below the subnormal grid may sum to a representable
    // value even without amplification. Only tiny sums need this second pass.
    if !amplified && !sum.is_normal() {
        ordinary &= !terms.any(|(w, v)| native_product_loses_range(w, component(v, imaginary)));
    }
    (sum, ordinary && sum.is_finite())
}

#[inline(always)]
fn component<T: NumericScalar>(value: T, imaginary: bool) -> T::Real {
    if imaginary {
        value.imaginary_component()
    } else {
        value.real_component()
    }
}

/// Identity-mass equation `alpha*state - history - forcing - step*Σ(weight*rhs)`.
/// Ordinary rounding is permitted; range loss replays the original operands.
#[inline(always)]
pub(crate) fn scalar_stage_residual<T: NumericScalar, I>(
    alpha: T::Real,
    state: T,
    history: T,
    step: T::Real,
    terms: I,
    forcing: T,
) -> Result<T, SciOxideError>
where
    I: IntoIterator<Item = (T::Real, T)>,
    I::IntoIter: Clone,
{
    let terms = terms.into_iter();
    let evaluate = |imaginary| {
        let state = component(state, imaginary);
        let history = component(history, imaginary);
        let forcing = component(forcing, imaginary);
        let difference = if alpha == T::Real::ONE {
            state - history
        } else {
            alpha.mul_add(state, -history)
        };
        let base = difference - forcing;
        let (sum, ordinary) = weighted(step.abs() > T::Real::ONE, terms.clone(), imaginary);
        let result = scioxide_core::fast_mul_add(-step, sum, base);
        if ordinary
            && base.is_finite()
            && result.is_finite()
            && !native_product_loses_range(step, sum)
        {
            Ok(result)
        } else {
            scalar_replay(
                alpha,
                state,
                history,
                step,
                terms.clone().map(|(w, v)| (w, component(v, imaginary))),
                forcing,
            )
        }
    };
    let real = evaluate(false)?;
    let imaginary = if T::IS_REAL {
        T::Real::ZERO
    } else {
        evaluate(true)?
    };
    Ok(T::from_components(real, imaginary))
}

/// Evaluate a BDF equation divided by its leading coefficient. Preparation
/// accepts the normalized history and step; exceptional rows replay the original
/// unnormalized operands before division.
#[inline(always)]
pub(crate) fn normalized_stage_residual<T: NumericScalar>(
    alpha: T::Real,
    state: T,
    history: T,
    step: T::Real,
    rhs: T,
    normalized_history: T,
    normalized_step: T::Real,
) -> Result<T, SciOxideError> {
    let evaluate = |imaginary| {
        let rhs = component(rhs, imaginary);
        let difference = component(state, imaginary) - component(normalized_history, imaginary);
        let value = scioxide_core::fast_mul_add(-normalized_step, rhs, difference);
        (difference.is_finite()
            && value.is_finite()
            && !native_product_loses_range(normalized_step, rhs))
        .then_some(value)
    };
    if let Some(real) = evaluate(false) {
        if T::IS_REAL {
            return Ok(T::from_real(real));
        }
        if let Some(imaginary) = evaluate(true) {
            return Ok(T::from_components(real, imaginary));
        }
    }
    scalar_stage_residual(alpha, state, history, step, [(T::Real::ONE, rhs)], T::ZERO)
        .map(|value| value / T::from_real(alpha))
}

#[cold]
#[inline(never)]
fn scalar_replay<S: RealScalar>(
    alpha: S,
    state: S,
    history: S,
    step: S,
    terms: impl Iterator<Item = (S, S)>,
    forcing: S,
) -> Result<S, SciOxideError> {
    let mut sum = RangeAccumulator::new();
    sum.add_product(alpha, state)?;
    sum.add(-history)?;
    sum.add(-forcing)?;
    sum.add_triple_products(-step, terms)?;
    sum.finish()
}

/// Stream a coupled identity-mass stage across four coordinates at a time.
/// Prepared slices have the output shape. Each lane accepts its native value
/// under the classification of [`scalar_stage_residual`] at every step size;
/// exceptional lanes and the short tail use the scalar equation, keeping the
/// original operands for complete replay.
#[inline]
pub(crate) fn implicit_stage_rows<T: NumericScalar>(
    state: &[T],
    history: &[T],
    step: T::Real,
    weights: &[T::Real],
    rhs: &[Vector<T>],
    output: &mut [T],
) -> Result<(), SciOxideError> {
    if step.abs() > T::Real::ONE {
        blocked_stage_rows::<T, true>(state, history, step, weights, rhs, output)
    } else {
        blocked_stage_rows::<T, false>(state, history, step, weights, rhs, output)
    }
}

/// Four-lane kernel of [`implicit_stage_rows`]. An amplifying step tracks
/// range loss of every lane product; otherwise only lanes with a zero or
/// subnormal sum scan their products, as the scalar equation does.
#[inline(always)]
fn blocked_stage_rows<T: NumericScalar, const AMPLIFIED: bool>(
    state: &[T],
    history: &[T],
    step: T::Real,
    weights: &[T::Real],
    rhs: &[Vector<T>],
    output: &mut [T],
) -> Result<(), SciOxideError> {
    let terms = |i| {
        weights
            .iter()
            .zip(rhs)
            .map(move |(&w, row)| (w, row.as_slice()[i]))
    };
    let scalar =
        |i| scalar_stage_residual(T::Real::ONE, state[i], history[i], step, terms(i), T::ZERO);
    let length = output.len();
    let mut chunks = output.chunks_exact_mut(4);
    for (block, out) in chunks.by_ref().enumerate() {
        let offset = block * 4;
        let mut sums = [T::ZERO; 4];
        let mut lost = [false; 4];
        for (&weight, row) in weights.iter().zip(rhs) {
            if weight == T::Real::ZERO {
                continue;
            }
            let row = &row.as_slice()[offset..offset + 4];
            for k in 0..4 {
                sums[k] = native_add(weight, row[k], sums[k]);
                if AMPLIFIED {
                    lost[k] |= native_product_loses_range(weight, row[k].real_component())
                        | (!T::IS_REAL
                            && native_product_loses_range(weight, row[k].imaginary_component()));
                }
            }
        }
        for (j, out) in out.iter_mut().enumerate() {
            let i = offset + j;
            let sum = sums[j];
            let real = scioxide_core::fast_mul_add(
                -step,
                sum.real_component(),
                state[i].real_component() - history[i].real_component(),
            );
            let imaginary = if T::IS_REAL {
                T::Real::ZERO
            } else {
                scioxide_core::fast_mul_add(
                    -step,
                    sum.imaginary_component(),
                    state[i].imaginary_component() - history[i].imaginary_component(),
                )
            };
            let value = T::from_components(real, imaginary);
            let ordinary = value.has_finite_components()
                && !lost[j]
                && !native_product_loses_range(step, sum.real_component())
                && (T::IS_REAL || !native_product_loses_range(step, sum.imaginary_component()))
                && (AMPLIFIED
                    || ((sum.real_component().is_normal()
                        || !tiny_sum_loses_range(terms(i), false))
                        && (T::IS_REAL
                            || sum.imaginary_component().is_normal()
                            || !tiny_sum_loses_range(terms(i), true))));
            *out = if ordinary { value } else { scalar(i)? };
        }
    }
    let offset = length / 4 * 4;
    for (j, out) in chunks.into_remainder().iter_mut().enumerate() {
        *out = scalar(offset + j)?;
    }
    Ok(())
}

/// Whether a product of a zero or subnormal sum lost range (the second pass
/// of the unamplified classification in `weighted`).
#[cold]
#[inline(never)]
fn tiny_sum_loses_range<T: NumericScalar>(
    mut terms: impl Iterator<Item = (T::Real, T)>,
    imaginary: bool,
) -> bool {
    terms.any(|(w, v)| native_product_loses_range(w, component(v, imaginary)))
}

/// One row of `M(alpha*state - history) - forcing - step*Σ(weight*rhs)`.
/// The native dot keeps a forward-error bound for cancellation screening.
/// Only exceptional rows replay; real rows never construct complex carriers.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn stage_residual<T: NumericScalar, I>(
    mass: &[T],
    alpha: T::Real,
    state: &[T],
    history: &[T],
    step: T::Real,
    terms: I,
    forcing: T,
) -> Result<T, SciOxideError>
where
    I: IntoIterator<Item = (T::Real, T)>,
    I::IntoIter: Clone,
{
    let terms = terms.into_iter();
    let mut real = T::Real::ZERO;
    let mut imaginary = T::Real::ZERO;
    let mut real_bound = T::Real::ZERO;
    let mut imaginary_bound = T::Real::ZERO;
    let mut ordinary = true;
    let mut operations = T::Real::ZERO;
    for ((&m, &state), &history) in mass.iter().zip(state).zip(history) {
        if m == T::ZERO {
            continue;
        }
        let delta = |imaginary| {
            let state = component(state, imaginary);
            let history = component(history, imaginary);
            if alpha == T::Real::ONE {
                state - history
            } else {
                alpha.mul_add(state, -history)
            }
        };
        let dr = delta(false);
        let mr = m.real_component();
        ordinary &= dr.is_finite();
        if alpha != T::Real::ONE {
            ordinary &= !native_product_loses_range(alpha, state.real_component());
        }
        let mut accumulate = |a: T::Real, b: T::Real, sum: &mut T::Real, bound: &mut T::Real| {
            let product = a * b;
            ordinary &= !native_product_loses_range(a, b);
            *sum = scioxide_core::fast_mul_add(a, b, *sum);
            *bound += product.abs();
        };
        accumulate(mr, dr, &mut real, &mut real_bound);
        if !T::IS_REAL {
            let di = delta(true);
            let mi = m.imaginary_component();
            accumulate(-mi, di, &mut real, &mut real_bound);
            accumulate(mr, di, &mut imaginary, &mut imaginary_bound);
            accumulate(mi, dr, &mut imaginary, &mut imaginary_bound);
            ordinary &= di.is_finite();
            if alpha != T::Real::ONE {
                ordinary &= !native_product_loses_range(alpha, state.imaginary_component());
            }
        }
        operations += if T::IS_REAL {
            T::Real::TWO + T::Real::TWO
        } else {
            T::Real::TWO * (T::Real::TWO + T::Real::TWO)
        };
    }
    let error = operations * T::Real::EPSILON;
    let gamma = error / (T::Real::ONE - error);
    let reliable = |value: T::Real, bound: T::Real| {
        value.is_finite()
            && bound.is_finite()
            && (bound == T::Real::ZERO || value.abs() > gamma * bound)
    };
    ordinary &= error < T::Real::ONE && reliable(real, real_bound);
    if !T::IS_REAL {
        ordinary &= reliable(imaginary, imaginary_bound);
    }
    let evaluate = |dot, imaginary| {
        let base = dot - component(forcing, imaginary);
        let (sum, valid) = weighted(step.abs() > T::Real::ONE, terms.clone(), imaginary);
        let result = (-step).mul_add(sum, base);
        if ordinary && valid && base.is_finite() && result.is_finite() {
            Ok(result)
        } else {
            mass_replay(
                mass,
                alpha,
                state,
                history,
                step,
                terms.clone(),
                forcing,
                imaginary,
            )
        }
    };
    Ok(T::from_components(
        evaluate(real, false)?,
        if T::IS_REAL {
            T::Real::ZERO
        } else {
            evaluate(imaginary, true)?
        },
    ))
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn mass_replay<T: NumericScalar>(
    mass: &[T],
    alpha: T::Real,
    state: &[T],
    history: &[T],
    step: T::Real,
    terms: impl Iterator<Item = (T::Real, T)>,
    forcing: T,
    imaginary: bool,
) -> Result<T::Real, SciOxideError> {
    let mut sum = RangeAccumulator::new();
    for ((&mass, &state), &history) in mass.iter().zip(state).zip(history) {
        let mr = mass.real_component();
        let sr = component(state, imaginary);
        let hr = component(history, imaginary);
        sum.add_triple_product(mr, alpha, sr)?;
        sum.subtract_product(mr, hr)?;
        if !T::IS_REAL {
            let mi = if imaginary {
                mass.imaginary_component()
            } else {
                -mass.imaginary_component()
            };
            sum.add_triple_product(mi, alpha, component(state, !imaginary))?;
            sum.subtract_product(mi, component(history, !imaginary))?;
        }
    }
    sum.add(-component(forcing, imaginary))?;
    sum.add_triple_products(-step, terms.map(|(w, v)| (w, component(v, imaginary))))?;
    sum.finish()
}

/// Native BDF derivative; replay keeps the numerator range until after division.
#[inline]
pub(crate) fn stage_derivative<T: NumericScalar, I>(
    step: T::Real,
    terms: I,
) -> Result<T, SciOxideError>
where
    I: IntoIterator<Item = (T::Real, T)>,
    I::IntoIter: Clone,
{
    let terms = terms.into_iter();
    let evaluate = |imaginary| {
        let (sum, ordinary) = weighted(step.abs() < T::Real::ONE, terms.clone(), imaginary);
        let derivative = sum / step;
        if ordinary && derivative.is_finite() && (sum == T::Real::ZERO || derivative.is_normal()) {
            Ok(derivative)
        } else {
            derivative_replay(
                step,
                terms.clone().map(|(w, v)| (w, component(v, imaginary))),
            )
        }
    };
    Ok(T::from_components(
        evaluate(false)?,
        if T::IS_REAL {
            T::Real::ZERO
        } else {
            evaluate(true)?
        },
    ))
}

#[cold]
#[inline(never)]
fn derivative_replay<S: RealScalar>(
    step: S,
    terms: impl Iterator<Item = (S, S)>,
) -> Result<S, SciOxideError> {
    let mut sum = RangeAccumulator::new();
    for (weight, value) in terms {
        sum.add_product(weight, value)?;
    }
    sum.finish_quotient(step)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use scioxide_core::Complex;

    #[test]
    fn normalized_bdf_rows_replay_original_operands_on_overflow_and_underflow() {
        let large = f64::MAX;
        let value = normalized_stage_residual(1., large, -large, 1., large, -large, 1.).unwrap();
        assert_eq!(value, large);
        let tiny = f64::from_bits(1);
        let original = scalar_stage_residual(2., tiny, 0., 1., [(1., tiny)], 0.).unwrap() / 2.;
        let value = normalized_stage_residual(2., tiny, 0., 1., tiny, 0., 0.5).unwrap();
        assert_eq!(value, original);
        let value = normalized_stage_residual(1.5, 3., 2., 0.1, 4., 2. / 1.5, 0.1 / 1.5).unwrap();
        assert!((value - (4.5 - 2. - 0.4) / 1.5).abs() < 1e-14);
    }

    #[test]
    fn native_residuals_agree_with_the_equations() {
        let terms = [(0.25, 2.0), (0.75, 4.0)];
        let actual = scalar_stage_residual(1.5, 3.0_f64, 2.0, 0.1, terms, 0.2).unwrap();
        assert!((actual - 1.95).abs() < 8.0 * f64::EPSILON);
        let actual =
            stage_residual(&[2.0, -0.5], 1.5, &[3.0, 4.0], &[2.0, 1.0], 0.1, terms, 0.2).unwrap();
        assert!((actual - 1.95).abs() < 8.0 * f64::EPSILON);
        let terms = [(0.5, Complex::new(2.0, -4.0))];
        let actual = stage_residual(
            &[Complex::new(2.0, 1.0)],
            1.0,
            &[Complex::new(3.0, 2.0)],
            &[Complex::new(1.0, 1.0)],
            0.1,
            terms,
            Complex::new(0.2, 0.3),
        )
        .unwrap();
        assert!((actual.real_component() - 2.7).abs() < 8.0 * f64::EPSILON);
        assert!((actual.imaginary_component() - 3.9).abs() < 8.0 * f64::EPSILON);
    }

    #[test]
    fn overflowing_residual_rows_replay_original_operands() {
        fn check<S: RealScalar + NumericScalar<Real = S>>() {
            let large = S::MAX;
            // The native state difference overflows; subtracting the forcing restores range.
            let actual =
                scalar_stage_residual(S::ONE, large, -large, S::ONE, [(S::ONE, large)], S::ZERO)
                    .unwrap();
            assert!((actual / large - S::ONE).abs() <= S::EPSILON);
            // A native mass dot overflows before its cancelling term arrives.
            let actual = stage_residual(
                &[S::TWO, -S::TWO, S::ONE],
                S::ONE,
                &[large, large, S::ONE],
                &[S::ZERO; 3],
                S::ONE,
                [],
                S::ZERO,
            )
            .unwrap();
            let expected = stage_sum([(S::TWO, large), (-S::TWO, large), (S::ONE, S::ONE)])
                .unwrap()
                .round()
                .unwrap();
            assert!((actual - expected).abs() <= S::EPSILON);
            let actual = scalar_stage_residual(
                S::ONE,
                S::ONE,
                S::ZERO,
                S::ONE,
                [(S::TWO, large), (-S::TWO, large)],
                S::ZERO,
            )
            .unwrap();
            assert!((actual - S::ONE).abs() <= S::EPSILON);
            assert!(scalar_stage_residual(S::ONE, large, -large, S::ONE, [], S::ZERO).is_err());
            assert!(
                stage_residual(&[S::TWO], S::ONE, &[large], &[S::ZERO], S::ONE, [], S::ZERO)
                    .is_err()
            );
        }
        check::<f32>();
        check::<f64>();
        let large = Complex::new(f64::MAX, f64::MAX);
        let actual = stage_residual(
            &[Complex::new(0.5, 0.5)],
            2.0,
            &[large],
            &[large],
            1.0,
            [],
            Complex::ZERO,
        )
        .unwrap();
        assert_eq!(actual, Complex::new(0.0, f64::MAX));
    }

    #[test]
    fn residual_replay_recovers_amplified_tiny_products() {
        let tiny = f64::from_bits(1);
        let expected = (tiny * f64::MAX) * 0.5;
        let actual =
            scalar_stage_residual(1.0, 0.0_f64, 0.0, f64::MAX, [(0.5, tiny)], 0.0).unwrap();
        assert!((actual + expected).abs() <= 4.0 * f64::EPSILON * expected);
        let actual = stage_residual(&[f64::MAX], 0.5, &[tiny], &[0.0], 1.0, [], 0.0).unwrap();
        assert!((actual - expected).abs() <= 4.0 * f64::EPSILON * expected);
    }

    #[test]
    fn residual_replay_preserves_subnormal_step_products_before_cancellation() {
        fn check<S: RealScalar + NumericScalar<Real = S>>() {
            let tiny = S::ZERO.next_after(S::ONE);
            let result =
                scalar_stage_residual(S::ONE, tiny, S::ZERO, S::HALF, [(S::ONE, tiny)], S::ZERO)
                    .unwrap();
            assert_eq!(result, S::ZERO);
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn derivatives_retain_numerator_range_until_division() {
        fn check<S: RealScalar + NumericScalar<Real = S>>(tiny: S) {
            let large = S::MAX;
            let expected = large / S::TWO;
            let actual = stage_derivative(S::TWO, [(S::TWO, large), (-S::ONE, large)]).unwrap();
            assert!((actual / expected - S::ONE).abs() <= S::EPSILON);
            let actual = stage_derivative(tiny, [(S::ONE / S::TWO, tiny)]).unwrap();
            assert!((actual - S::ONE / S::TWO).abs() <= S::EPSILON);
            let actual = stage_derivative(S::ONE, [(S::ONE / S::TWO, tiny); 2]).unwrap();
            assert_eq!(actual, tiny);
            assert!(stage_derivative(tiny, [(S::ONE, large)]).is_err());
        }
        check(f32::from_bits(1));
        check(f64::from_bits(1));
        let value = Complex::new(f64::MAX, -f64::MAX);
        let actual = stage_derivative(2.0, [(2.0, value), (-1.0, value)]).unwrap();
        assert!((actual.real_component() / (f64::MAX / 2.0) - 1.0).abs() <= f64::EPSILON);
        assert!((actual.imaginary_component() / (-f64::MAX / 2.0) - 1.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn coupled_rows_match_the_scalar_equation_at_every_step_size() {
        let tiny = f64::from_bits(1);
        // Two blocks and a tail: structural zeros, ordinary lanes, a lane whose
        // only product underflows, and cancelling normal products.
        let rhs = [
            Vector::from_vec(alloc::vec![0., 1., 0., tiny, 3., -2., 0., 1e-300, 5.]),
            Vector::from_vec(alloc::vec![0., 2., -0., 0., 1., 2., 0., -1e-300, 0.]),
        ];
        let state = [1., 2., -3., 0., 4., 5., -0., 6., 7.];
        let history = [0.5, 1., 1., 0., -1., 2., 0., 1., 2.];
        for step in [0.25, -1., 10., -1e3, 1e300] {
            for weights in [[0.5, 0.5], [1., -0.25], [0., 2.]] {
                let mut output = [0.0; 9];
                implicit_stage_rows(&state, &history, step, &weights, &rhs, &mut output).unwrap();
                for (i, actual) in output.iter().enumerate() {
                    let expected = scalar_stage_residual(
                        1.,
                        state[i],
                        history[i],
                        step,
                        weights
                            .iter()
                            .zip(&rhs)
                            .map(|(&w, row)| (w, row.as_slice()[i])),
                        0.,
                    )
                    .unwrap();
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{step} {weights:?} {i}"
                    );
                }
            }
        }
        let mut output = [0.0; 9];
        implicit_stage_rows(&history, &history, 1e300, &[0.5, 0.], &rhs, &mut output).unwrap();
        let expected = -(tiny * 1e300) * 0.5;
        assert!((output[3] - expected).abs() <= f64::EPSILON * expected.abs());
    }

    #[test]
    fn coupled_rows_replay_overflow_and_tiny_products_in_blocks_and_tails() {
        for n in [0, 1, 4, 7] {
            let state = alloc::vec![1.0; n];
            let history = alloc::vec![0.0; n];
            let rhs = [
                Vector::from_vec(alloc::vec![f64::MAX; n]),
                Vector::from_vec(alloc::vec![f64::MAX; n]),
            ];
            let mut output = alloc::vec![0.0; n];
            implicit_stage_rows(&state, &history, 1.0, &[2.0, -2.0], &rhs, &mut output).unwrap();
            assert!(output.iter().all(|&v| (v - 1.0).abs() <= f64::EPSILON));
            let tiny = f64::from_bits(1);
            let rhs = [Vector::from_vec(alloc::vec![tiny; n])];
            implicit_stage_rows(&history, &history, f64::MAX, &[0.5], &rhs, &mut output).unwrap();
            let expected = -(tiny * f64::MAX) * 0.5;
            assert!(
                output
                    .iter()
                    .all(|&v| (v - expected).abs() <= f64::EPSILON * expected.abs())
            );
            let rhs = [
                Vector::from_vec(alloc::vec![tiny; n]),
                Vector::from_vec(alloc::vec![tiny; n]),
            ];
            implicit_stage_rows(&history, &history, 1.0, &[0.5, 0.5], &rhs, &mut output).unwrap();
            assert!(output.iter().all(|&v| v == -tiny));
            let large = Complex::new(f64::MAX, -f64::MAX);
            let rhs = [
                Vector::from_vec(alloc::vec![large; n]),
                Vector::from_vec(alloc::vec![large; n]),
            ];
            let state = alloc::vec![Complex::new(1.0, 2.0); n];
            let history = alloc::vec![Complex::ZERO; n];
            let mut output = alloc::vec![Complex::ZERO; n];
            implicit_stage_rows(&state, &history, 1.0, &[2.0, -2.0], &rhs, &mut output).unwrap();
            assert!(output.iter().all(|&v| v == Complex::new(1.0, 2.0)));
            let rhs = [Vector::from_vec(alloc::vec![Complex::new(2.0, 4.0); n])];
            implicit_stage_rows(&state, &history, 0.1, &[0.5], &rhs, &mut output).unwrap();
            assert!(
                output
                    .iter()
                    .all(|v| (v.real_component() - 0.9).abs() <= f64::EPSILON
                        && (v.imaginary_component() - 1.8).abs() <= f64::EPSILON)
            );
        }
    }
}
