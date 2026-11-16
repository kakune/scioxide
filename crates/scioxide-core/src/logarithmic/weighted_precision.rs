//! Compensated weighted reductions, certified before exact accumulation.
//!
//! Let u=2^-53 and g=n*u/(1-n*u). TwoSum compensation loses at most
//! 4*g² times the absolute component mass. Normal Cartesian terms have
//! exponential/product error <=6*u times that mass; 8*u also covers final
//! compensated rounding. For unit phases, sin/cos have <=8*u absolute
//! component error. Normalizing their vector amplifies its norm error by
//! at most two: 16*sqrt(2)*u. Squaring/adding/sqrt/division adds <4*u,
//! exp/product <6*u and final compensation <sqrt(2)*u, hence 34*u suffices.
//! Subnormal products, tails below -700 and overflowing intermediates are
//! deferred to the exact owner, so these are relative normal-arithmetic bounds.
//!
//! The ordinary L1 mass sum has relative error <=gamma_(2n). Inflate/deflate
//! it by this and twice the term bound to enclose the exact component mass.
//! L1/sqrt(2) <= vector mass <= L1 gives bounds on the true conditioning,
//! rather than granting a larger envelope from an overestimated kappa.
//! Hypot contributes <=10*u of its computed norm (four native ULP, with
//! computed/reference conversion). If E bounds the norm error, rho=E/(h-E)
//! gives log error <=rho and phase error <=2*rho. Elementary logarithm,
//! reconstruction and destination rounding are added separately. All bound
//! arithmetic rounds outward; allowance arithmetic rounds downward.

use super::working_precision::{Compensated, exponential};
use super::{LogWeight, WeightedSum, add_up, destination_ulp, mul_up};
use crate::{Complex, RealScalar, SciOxideError};

const U: f64 = 1.0 / (1_u64 << 53) as f64;
const CONDITION_LIMIT: f64 = (1_u64 << 20) as f64;

fn down(value: f64) -> f64 {
    RealScalar::next_after(value, 0.0)
}

fn up(value: f64) -> f64 {
    RealScalar::next_after(value, f64::INFINITY)
}

pub(super) fn weighted_sum<S: RealScalar>(
    len: usize,
    value: impl Fn(usize) -> S,
    weight: impl Fn(usize) -> LogWeight,
    anchor: f64,
) -> Result<Option<WeightedSum<S>>, SciOxideError> {
    if len > 1 << 24 {
        return Ok(None);
    }
    let mut sums = [Compensated::default(); 2];
    let mut mass = 0.0;
    let mut nonnegative = true;
    let mut unit_phases = false;
    let mut contributed = [false; 2];
    let mut active = 0;
    for i in 0..len {
        let x = value(i).into_f64();
        let w = weight(i);
        if x == f64::NEG_INFINITY || !w.nonzero() {
            continue;
        }
        // Never discard a weighted tail: it may survive a cancelling window.
        if x - anchor < -700.0 {
            return Ok(None);
        }
        active += 1;
        let parts = match w {
            LogWeight::Cartesian(w) => {
                nonnegative &= w.re >= 0.0 && w.im == 0.0;
                [w.re, w.im]
            }
            LogWeight::UnitPhase(angle) => {
                nonnegative = false;
                unit_phases = true;
                let (s, c) = RealScalar::sin_cos(angle);
                let norm = RealScalar::sqrt(c * c + s * s);
                [c / norm, s / norm]
            }
        };
        let exponential = exponential(x, anchor);
        let terms = parts.map(|part| part * exponential);
        for ((sum, contributed), (part, term)) in sums
            .iter_mut()
            .zip(&mut contributed)
            .zip(parts.into_iter().zip(terms))
        {
            if part != 0.0 {
                *contributed = true;
                if !term.is_finite() || term.abs() < f64::MIN_POSITIVE {
                    return Ok(None);
                }
            }
            sum.add(term);
        }
        mass += terms[0].abs() + terms[1].abs();
    }
    let z = sums.map(Compensated::value);
    let magnitude = if z[1] == 0.0 {
        z[0].abs()
    } else if z[0] == 0.0 {
        z[1].abs()
    } else {
        RealScalar::hypot(z[0], z[1])
    };
    if !mass.is_finite() || !magnitude.is_finite() || magnitude < f64::MIN_POSITIVE {
        return Ok(None);
    }
    let n = up(active as f64);
    let nu = mul_up(n, U);
    let gamma = up(nu / down(1.0 - nu));
    let twice = mul_up(2.0, nu);
    let mass_gamma = up(twice / down(1.0 - twice));
    let eta = if unit_phases { 34.0 * U } else { 8.0 * U };
    let mass_relative = add_up(mass_gamma, mul_up(add_up(1.0, mass_gamma), 2.0 * eta));
    let upper_mass = up(mass / down(1.0 - mass_relative));
    let lower_mass = down(mass / add_up(1.0, mass_relative));
    let error = mul_up(add_up(eta, mul_up(4.0, mul_up(gamma, gamma))), upper_mass);
    // A component interval containing zero leaves its sign/cancellation
    // decision to the exact kernel, even if the vector norm is well resolved.
    if z.into_iter()
        .zip(contributed)
        .any(|(z, c)| c && z.abs() <= error)
    {
        return Ok(None);
    }
    let norm_error = if z[0] == 0.0 || z[1] == 0.0 {
        error
    } else {
        add_up(error, mul_up(10.0 * U, magnitude))
    };
    let lower = down(magnitude - norm_error);
    let upper = add_up(magnitude, norm_error);
    let condition_upper = up(upper_mass / lower);
    // Heavy cancellation and near-zero results retain exact reconstruction.
    if lower <= 0.0 || condition_upper > CONDITION_LIMIT {
        return Ok(None);
    }
    let rho = up(norm_error / lower);
    let shifted = RealScalar::ln(magnitude);
    let logarithm = anchor + shifted;
    let log_bound = add_up(
        add_up(rho, mul_up(4.0, destination_ulp::<f64>(shifted))),
        destination_ulp::<f64>(logarithm),
    );
    if !logarithm.is_finite()
        || !log_bound.is_finite()
        || logarithm.abs() <= (1.0 + anchor.abs() + shifted.abs()) / CONDITION_LIMIT
    {
        return Ok(None);
    }
    let low_log = down(logarithm.abs() - log_bound);
    if low_log < S::MIN_POSITIVE.into_f64() {
        return Ok(None);
    }
    if nonnegative {
        // Two destination ULP remain reserved for projection and endpoints.
        if log_bound > 14.0 * destination_ulp::<S>(low_log) {
            return Ok(None);
        }
    } else {
        let epsilon = S::EPSILON.into_f64();
        if mul_up(32.0 * epsilon, condition_upper) >= 0.5 {
            return Ok(None);
        }
        let condition_lower =
            down(down(lower_mass / up(core::f64::consts::SQRT_2)) / upper).max(1.0);
        let phase_allowance = down(64.0 * epsilon * condition_lower);
        let log_allowance = down(phase_allowance * down(1.0 + low_log));
        let projection = destination_ulp::<S>(add_up(logarithm.abs(), log_bound));
        let phase_bound = add_up(mul_up(2.0, rho), 2.0 * epsilon);
        if add_up(log_bound, projection) > log_allowance || phase_bound > phase_allowance {
            return Ok(None);
        }
    }
    Ok(Some(WeightedSum::Nonzero {
        phase: Complex::new(z[0] / magnitude, z[1] / magnitude),
        log: super::output(logarithm)?,
    }))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;
    use crate::logarithmic::{WEIGHTED_TIERS, windowed_components};
    use crate::{NumericScalar, logsumexp, logsumexp_value, weighted_logsumexp};

    fn scalar<S: RealScalar>(x: f64) -> S {
        S::try_from_f64_lossy(x).unwrap()
    }

    #[track_caller]
    fn selection(expected: [usize; 2], call: impl FnOnce()) {
        WEIGHTED_TIERS.with(|counts| counts.set([0; 2]));
        call();
        assert_eq!(WEIGHTED_TIERS.with(core::cell::Cell::get), expected);
    }

    fn short<S: RealScalar>()
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        for len in [1, 2, 3, 8, 16, 17, 32, 64, 256, 4096] {
            let values: alloc::vec::Vec<S> = (0..len)
                .map(|i| scalar(5.0 + (i % 16) as f64 * 0.125))
                .collect();
            for signed in [false, true] {
                let weights: alloc::vec::Vec<S> = (0..len)
                    .map(|i| scalar(if signed && i % 4 == 3 { -0.125 } else { 1.0 }))
                    .collect();
                selection([1, 0], || {
                    let _ = weighted_logsumexp(&values, &weights).unwrap();
                });
                let complex = weights
                    .iter()
                    .map(|&re| Complex::new(re, scalar(0.25)))
                    .collect::<alloc::vec::Vec<_>>();
                selection([1, 0], || {
                    let _ = weighted_logsumexp(&values, &complex).unwrap();
                });
                let masked = values
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| if i % 7 == 3 { S::NEG_INFINITY } else { x })
                    .collect::<alloc::vec::Vec<_>>();
                selection([1, 0], || {
                    let _ = weighted_logsumexp(&masked, &weights).unwrap();
                });
                if len >= 2 {
                    selection([1, 0], || {
                        let _ = weighted_logsumexp(&alloc::vec![S::ZERO; len], &weights).unwrap();
                    });
                }
            }
            let logits: alloc::vec::Vec<_> = values
                .iter()
                .enumerate()
                .map(|(i, &re)| Complex::new(re, scalar(0.125 + (i % 7) as f64 * 0.03125)))
                .collect();
            // A single/common phase delegates to the positive reduction owner.
            let expected = if len == 1 { [0, 0] } else { [1, 0] };
            selection(expected, || {
                let _ = logsumexp_value(&logits).unwrap();
            });
            selection(expected, || {
                logsumexp(&logits).unwrap();
            });
            if len >= 2 {
                let centered = logits
                    .iter()
                    .map(|z| Complex::new(S::ZERO, z.im))
                    .collect::<alloc::vec::Vec<_>>();
                selection([1, 0], || {
                    let _ = logsumexp_value(&centered).unwrap();
                });
            }
        }
    }

    #[test]
    fn short_weighted_and_general_phase_inputs_are_certified() {
        short::<f32>();
        short::<f64>();
    }

    fn cancellation<S: RealScalar>()
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        let a = Complex::new(S::ONE, scalar(0.25));
        let minus = Complex::new(-S::ONE, scalar(-0.25));
        selection([0, 1], || {
            let result = weighted_logsumexp(&[S::ZERO; 2], &[a, minus]).unwrap();
            assert!(result.log_magnitude().is_none());
        });
        selection([0, 1], || {
            let result =
                weighted_logsumexp(&[S::ZERO, S::ZERO, scalar(-1300.0)], &[a, minus, a]).unwrap();
            let expected = -1300.0 + 0.5 * RealScalar::ln(1.0625_f64);
            assert!(
                (result.log_magnitude().unwrap().into_f64() - expected).abs()
                    <= 16.0 * destination_ulp::<S>(expected)
            );
        });
        let tiny = scalar::<S>(1.0 / (1_u64 << 24) as f64);
        selection([0, 1], || {
            let result = weighted_logsumexp(
                &[S::ZERO, S::ZERO, S::ZERO],
                &[a, minus, Complex::new(tiny, tiny)],
            )
            .unwrap();
            assert!(result.log_magnitude().unwrap() < scalar(-10.0));
        });
        // A nonzero norm can still have an unresolved component cancellation.
        selection([0, 1], || {
            let _ = weighted_logsumexp(
                &[scalar::<S>(2.0); 2],
                &[a, Complex::new(S::ONE, scalar(-0.25))],
            )
            .unwrap();
        });
        // A near-zero logarithm needs the exact unit-residual reconstruction.
        selection([0, 1], || {
            let _ = weighted_logsumexp(&[-S::LN_2; 2], &[S::ONE; 2]).unwrap();
        });
        let logits = [Complex::new(S::ZERO, S::ZERO), Complex::new(S::ZERO, S::PI)];
        selection([0, 1], || {
            let _ = logsumexp_value(&logits).unwrap();
        });
        selection([0, 1], || {
            logsumexp(&logits).unwrap();
        });
    }

    #[test]
    fn cancellation_and_near_zero_results_keep_the_exact_owner() {
        cancellation::<f32>();
        cancellation::<f64>();
    }

    fn bounds<S: RealScalar>() {
        // Use the existing four-component kernel as an independent arithmetic
        // comparison; the committed MPFR/MPC fixtures qualify public results.
        for len in [2, 3, 8, 16, 64, 256] {
            for phase in [false, true] {
                let value = |i| scalar::<S>(3.0 - (i % 17) as f64 * 0.125);
                let weight = |i| {
                    if phase {
                        LogWeight::UnitPhase(-1.5 + (i % 31) as f64 * 0.1)
                    } else {
                        LogWeight::Cartesian(Complex::new(
                            if i % 3 == 0 { -0.25 } else { 1.0 },
                            0.125,
                        ))
                    }
                };
                let candidate = weighted_sum(len, value, weight, 3.0).unwrap();
                let exact = windowed_components(len, value, weight).unwrap();
                match (candidate, exact) {
                    (
                        Some(WeightedSum::Nonzero { phase, log }),
                        WeightedSum::Nonzero { phase: p, log: l },
                    ) => {
                        let mass = (0..len)
                            .map(|i| {
                                let norm = match weight(i) {
                                    LogWeight::UnitPhase(_) => 1.0,
                                    LogWeight::Cartesian(w) => RealScalar::hypot(w.re, w.im),
                                };
                                norm * RealScalar::exp(value(i).into_f64() - 3.0)
                            })
                            .sum::<f64>();
                        let condition = mass / RealScalar::exp(l.into_f64() - 3.0);
                        let bound = 64.0 * S::EPSILON.into_f64() * condition;
                        assert!(
                            (log.into_f64() - l.into_f64()).abs()
                                <= bound * (1.0 + l.into_f64().abs())
                        );
                        assert!((phase.re - p.re).abs().max((phase.im - p.im).abs()) <= bound);
                    }
                    _ => panic!("ordinary input did not certify: len={len}, phase={phase}"),
                }
            }
        }
        // Admission limits and normal-product/tail boundaries are conservative
        // fallbacks, not input errors or silent weighted-tail omissions.
        let weight = |_| LogWeight::Cartesian(Complex::new(1.0, 0.0));
        for (bits, expected) in [
            (8, true),
            (18, S::MANTISSA_DIGITS == 53),
            (20, false),
            (24, false),
        ] {
            let gap = RealScalar::scalbn(1.0_f64, -bits);
            let weights = [1.0, -1.0 + gap];
            let result = weighted_sum(
                2,
                |_| scalar::<S>(5.0),
                |i| LogWeight::Cartesian(Complex::new(weights[i], 0.0)),
                5.0,
            )
            .unwrap();
            assert_eq!(
                result.is_some(),
                expected,
                "conditioning boundary 2^-{bits}"
            );
        }
        assert!(
            weighted_sum((1 << 24) + 1, |_| scalar::<S>(3.0), weight, 3.0)
                .unwrap()
                .is_none()
        );
        for delta in [
            -700.0,
            RealScalar::next_after(-700.0_f64, f64::NEG_INFINITY),
        ] {
            let values = [scalar::<S>(3.0), scalar(3.0 + delta)];
            let result = weighted_sum(2, |i| values[i], weight, 3.0).unwrap();
            let actual_delta = values[1].into_f64() - 3.0;
            assert_eq!(result.is_some(), actual_delta >= -700.0);
        }
        assert!(
            weighted_sum(
                1,
                |_| scalar::<S>(3.0),
                |_| LogWeight::Cartesian(Complex::new(f64::from_bits(1), 0.0)),
                3.0
            )
            .unwrap()
            .is_none()
        );
        assert!(
            weighted_sum(
                2,
                |_| scalar::<S>(3.0),
                |_| LogWeight::Cartesian(Complex::new(f64::MAX, 0.0)),
                3.0
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn certified_results_and_arithmetic_boundaries_both_widths() {
        bounds::<f32>();
        bounds::<f64>();
    }

    fn axes<S: RealScalar>()
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        use crate::{
            ExecutionContext, ExecutionPolicy, LogValue, NdView, NdViewMut, StridedLayout,
            logsumexp_axis_into, weighted_logsumexp_axis_into,
        };
        let values = [scalar::<S>(5.0); 8];
        let weights = [
            S::ONE,
            scalar(-0.125),
            S::ONE,
            S::ONE,
            S::ONE,
            S::ONE,
            scalar(-0.125),
            S::ONE,
        ];
        let complex_weights = weights.map(|re| Complex::new(re, scalar(0.25)));
        let logits = core::array::from_fn::<_, 8, _>(|i| {
            Complex::new(values[i], scalar(0.25 + i as f64 * 0.03125))
        });
        let shape = [2, 4];
        let output_layout = StridedLayout::new(&[2], &[1]).unwrap();
        let policy = ExecutionPolicy::Sequential;
        for strides in [[4, 1], [1, 2]] {
            let layout = StridedLayout::new(&shape, &strides).unwrap();
            let mut real_output = [LogValue::<S, S>::ZERO; 2];
            let mut real_scratch = real_output;
            selection([2, 0], || {
                weighted_logsumexp_axis_into(
                    NdView::new(&values, layout).unwrap(),
                    NdView::new(&weights, layout).unwrap(),
                    1,
                    policy,
                    NdViewMut::new(&mut real_output, output_layout).unwrap(),
                    &mut real_scratch,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            });
            let mut complex_output = [LogValue::<Complex<S>, S>::ZERO; 2];
            let mut complex_scratch = complex_output;
            selection([2, 0], || {
                weighted_logsumexp_axis_into(
                    NdView::new(&values, layout).unwrap(),
                    NdView::new(&complex_weights, layout).unwrap(),
                    1,
                    policy,
                    NdViewMut::new(&mut complex_output, output_layout).unwrap(),
                    &mut complex_scratch,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            });
            let sentinel = [Complex::new(S::NEG_ZERO, S::MAX); 2];
            let mut output = sentinel;
            let mut scratch = output;
            selection([2, 0], || {
                logsumexp_axis_into(
                    NdView::new(&logits, layout).unwrap(),
                    1,
                    policy,
                    NdViewMut::new(&mut output, output_layout).unwrap(),
                    &mut scratch,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            });
            for (i, actual) in output.into_iter().enumerate() {
                let lane = layout.axis(1).unwrap().lane(i).unwrap();
                let indices = core::array::from_fn::<_, 4, _>(|j| lane.start() + j * lane.stride());
                let expected = logsumexp(&indices.map(|j| logits[j])).unwrap();
                let bound = 64.0 * S::EPSILON.into_f64();
                assert!(
                    (actual.re.into_f64() - expected.re.into_f64()).abs()
                        <= bound * (1.0 + expected.re.into_f64().abs())
                );
                assert!((actual.im.into_f64() - expected.im.into_f64()).abs() <= bound);
                let expected =
                    weighted_logsumexp(&indices.map(|j| values[j]), &indices.map(|j| weights[j]))
                        .unwrap();
                let expected_log = expected.log_magnitude().unwrap().into_f64();
                assert!(
                    (real_output[i].log_magnitude().unwrap().into_f64() - expected_log).abs()
                        <= 16.0 * destination_ulp::<S>(expected_log)
                );
                let expected = weighted_logsumexp(
                    &indices.map(|j| values[j]),
                    &indices.map(|j| complex_weights[j]),
                )
                .unwrap();
                assert!(
                    (complex_output[i].log_magnitude().unwrap().into_f64()
                        - expected.log_magnitude().unwrap().into_f64())
                    .abs()
                        <= bound * (1.0 + expected_log.abs())
                );
            }
            // A later invalid lane must leave the whole output untouched even
            // after an earlier lane has produced a certified result.
            let mut invalid = logits;
            invalid[7].im = S::NAN;
            output = sentinel;
            selection([1, 0], || {
                assert!(
                    logsumexp_axis_into(
                        NdView::new(&invalid, layout).unwrap(),
                        1,
                        policy,
                        NdViewMut::new(&mut output, output_layout).unwrap(),
                        &mut scratch,
                        ExecutionContext::sequential(),
                    )
                    .is_err()
                );
            });
            assert_eq!(
                output.map(|z| (z.re.into_f64().to_bits(), z.im.into_f64().to_bits())),
                sentinel.map(|z| (z.re.into_f64().to_bits(), z.im.into_f64().to_bits()))
            );
        }
    }

    #[test]
    fn weighted_and_complex_axis_reductions_inherit_certification() {
        axes::<f32>();
        axes::<f64>();
    }
}
