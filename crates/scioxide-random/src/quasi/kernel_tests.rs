//! Independent scalar references for exact kernel transformations.
#![allow(clippy::unwrap_used)]

use super::{QmcPlan, QmcRandomization, QmcScramble, halton, scramble};
use crate::{OutputPolicy, QuasiRng, SamplingPolicy, SamplingWorkspace, SeedableRng, SplitMix64};
use alloc::{vec, vec::Vec};
use scioxide_core::{RangeAccumulator, RealScalar};

fn radical_reference<S: RealScalar>(
    mut index: u64,
    base: u32,
    reverse: bool,
    p: Option<&[u32]>,
) -> S {
    let (mut numerator, mut denominator) = (0_u128, 1_u128);
    while index != 0 {
        let digit = (index % u64::from(base)) as u32;
        let digit = if let Some(p) = p {
            p[digit as usize]
        } else if reverse && digit != 0 {
            base - digit
        } else {
            digit
        };
        numerator = numerator * u128::from(base) + u128::from(digit);
        denominator *= u128::from(base);
        index /= u64::from(base);
    }
    let value = if denominator <= u128::from(u64::MAX) {
        (numerator as u64) as f64 / (denominator as u64) as f64
    } else {
        numerator as f64 / denominator as f64
    };
    S::try_from_f64_lossy(value)
        .unwrap()
        .min(S::from_u64_half_open01(u64::MAX))
}

#[test]
fn radical_integer_fast_paths_preserve_both_scalar_widths() {
    let bases = QmcPlan::halton(100).unwrap();
    for base in bases
        .bases()
        .unwrap()
        .iter()
        .copied()
        .chain([1009, 4_294_967_291])
    {
        let prepared = halton::Base::new(base);
        let mut indices: Vec<_> = (0..128).collect();
        for bit in 0..64 {
            indices.extend([(1_u64 << bit) - 1, 1_u64 << bit, (1_u64 << bit) + 1]);
        }
        let mut power = u64::from(base);
        loop {
            indices.extend([power - 1, power, power.saturating_add(1)]);
            let Some(next) = power.checked_mul(u64::from(base)) else {
                break;
            };
            power = next;
        }
        indices.extend([u64::MAX - 1, u64::MAX]);
        let permutation = (base <= 541).then(|| {
            (0..base)
                .map(|x| if x == 0 { 0 } else { base - x })
                .collect::<Vec<_>>()
        });
        for p in [None, permutation.as_deref()] {
            for reverse in [false, true] {
                for &index in &indices {
                    let wide =
                        halton::radical_inverse::<f64>(index, &prepared, p, reverse).unwrap();
                    let narrow =
                        halton::radical_inverse::<f32>(index, &prepared, p, reverse).unwrap();
                    assert_eq!(
                        wide.to_bits(),
                        radical_reference::<f64>(index, base, reverse, p).to_bits(),
                        "base={base} index={index}"
                    );
                    assert_eq!(
                        narrow.to_bits(),
                        radical_reference::<f32>(index, base, reverse, p).to_bits(),
                        "base={base} index={index}"
                    );
                }
            }
        }
    }
}

fn digital_reference<S: RealScalar>(plan: &QmcPlan, index: u64) -> Vec<S> {
    let gray = index ^ (index >> 1);
    plan.direction_words()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(dimension, columns)| {
            let word = columns
                .iter()
                .enumerate()
                .filter(|(bit, _)| gray & (1 << bit) != 0)
                .fold(0, |word, (_, column)| word ^ column);
            S::from_u64_half_open01(scramble::transform(
                word,
                dimension,
                plan.scramble,
                plan.randomization_words(),
            ))
        })
        .collect()
}

fn digital_check<S: RealScalar>() {
    for dim in [1, 3, 4, 7, 12] {
        for root in [
            QmcPlan::sobol(dim).unwrap(),
            QmcPlan::niederreiter_base2(dim).unwrap(),
        ] {
            let linear = root
                .clone()
                .with_randomization(
                    QmcRandomization::LinearMatrixShift,
                    &mut SplitMix64::from_seed(42),
                )
                .unwrap();
            let shifted = linear
                .clone()
                .with_scramble(QmcScramble::DigitalShift(17))
                .unwrap();
            for plan in [root.clone(), linear, shifted] {
                assert_eq!(plan.direction_words(), root.direction_words());
                let mut cursor = plan.sequence::<S>().unwrap();
                for start in [0, 1, 63, 255, (1 << 32) - 1, 1 << 63, u64::MAX - 7] {
                    for policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
                        cursor.skip_to(start).unwrap();
                        let mut values = vec![S::ZERO; 8 * dim];
                        cursor
                            .points_into(
                                SamplingPolicy::rollback(policy),
                                &mut values,
                                &mut SamplingWorkspace::new(),
                            )
                            .unwrap();
                        for (offset, row) in values.chunks_exact(dim).enumerate() {
                            assert_eq!(row, digital_reference::<S>(&plan, start + offset as u64));
                        }
                        let bytes = cursor.export_state().unwrap();
                        let restored = QmcPlan::import_state(&bytes).unwrap();
                        assert_eq!(
                            restored.sequence::<S>().unwrap().export_state().unwrap(),
                            bytes
                        );
                        let mut replay = restored.sequence::<S>().unwrap();
                        replay.skip_to(start).unwrap();
                        assert_eq!(
                            replay.next_point().unwrap(),
                            digital_reference::<S>(&plan, start)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cached_scrambles_and_replacement_preserve_direct_matrix_evaluation() {
    digital_check::<f32>();
    digital_check::<f64>();
}

#[test]
fn digital_exhaustion_preserves_each_transaction_policy() {
    let plan = QmcPlan::sobol(4)
        .unwrap()
        .with_scramble(QmcScramble::DigitalShift(42))
        .unwrap();
    let mut cursor = plan.sequence::<f64>().unwrap();
    for atomic in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        for rollback in [false, true] {
            for count in [1, 2, 3] {
                cursor.skip_to(u64::MAX - 1).unwrap();
                let saved = cursor.position();
                let mut values = vec![99.; count * 4];
                let mut workspace = SamplingWorkspace::new();
                let result = if rollback {
                    cursor.points_into(
                        SamplingPolicy::rollback(atomic),
                        &mut values,
                        &mut workspace,
                    )
                } else {
                    cursor.points_into(
                        SamplingPolicy::advance_on_error(atomic),
                        &mut values,
                        &mut workspace,
                    )
                };
                assert_eq!(result.is_ok(), count <= 2);
                if count == 3 && rollback {
                    assert_eq!(cursor.position(), saved);
                } else {
                    assert_eq!(
                        cursor.position().next_index(),
                        if count == 1 { Some(u64::MAX) } else { None }
                    );
                }
                if count == 3 && atomic == OutputPolicy::Atomic {
                    assert_eq!(values, vec![99.; 12]);
                } else {
                    assert_eq!(&values[..4], digital_reference::<f64>(&plan, u64::MAX - 1));
                    if count >= 2 {
                        assert_eq!(&values[4..8], digital_reference::<f64>(&plan, u64::MAX));
                    }
                    if count == 3 {
                        assert_eq!(&values[8..], &[99.; 4]);
                    }
                }
            }
        }
    }
}

fn centered_reference<S: RealScalar>(points: &[S], dim: usize) -> f64 {
    let mut a = RangeAccumulator::new();
    let mut b = RangeAccumulator::new();
    for row in points.chunks_exact(dim) {
        let single = row
            .iter()
            .map(|x| {
                let a = (x.into_f64() - 0.5).abs();
                1.0 + 0.5 * a - 0.5 * a * a
            })
            .try_fold(1.0, scioxide_core::checked_float_product)
            .unwrap();
        a.add(single).unwrap();
        for other in points.chunks_exact(dim) {
            let pair = row
                .iter()
                .zip(other)
                .map(|(a, b)| {
                    let (a, b) = (a.into_f64(), b.into_f64());
                    1.0 + 0.5 * (a - 0.5).abs() + 0.5 * (b - 0.5).abs() - 0.5 * (a - b).abs()
                })
                .try_fold(1.0, scioxide_core::checked_float_product)
                .unwrap();
            b.add(pair).unwrap();
        }
    }
    let constant = (0..dim)
        .try_fold(1.0, |p, _| {
            scioxide_core::checked_float_product(p, 13.0 / 12.0)
        })
        .unwrap();
    let n = (points.len() / dim) as f64;
    let mut result = RangeAccumulator::new();
    for term in [
        constant,
        -2.0 * (a.finish().unwrap() / n),
        (b.finish().unwrap() / n) / n,
    ] {
        result.add(term).unwrap();
    }
    result.finish().unwrap().max(0.0)
}

fn diagnostics_check<S: RealScalar>() {
    use super::{
        DiscrepancyMethod as D, GeometricDiscrepancy as G, discrepancy, geometric_discrepancy,
    };
    let mut rng = SplitMix64::from_seed(42);
    for dim in [1, 2, 3, 4, 7, 16, 1024, 1025] {
        let n = if dim > 16 { 3 } else { 23 };
        let points: Vec<_> = (0..dim * n)
            .map(|_| S::from_u64_half_open01(crate::BitGenerator::next_u64(&mut rng).unwrap()))
            .collect();
        assert_eq!(
            discrepancy(&points, dim, D::Centered).unwrap().to_bits(),
            centered_reference(&points, dim).to_bits()
        );
        let mut best = f64::INFINITY;
        for (i, row) in points.chunks_exact(dim).enumerate() {
            for other in points[..i * dim].chunks_exact(dim) {
                let norm = row.iter().zip(other).fold(0.0_f64, |norm, (a, b)| {
                    RealScalar::hypot(norm, a.into_f64() - b.into_f64())
                });
                best = best.min(norm);
            }
        }
        assert_eq!(
            geometric_discrepancy(&points, dim, G::MinimumDistance)
                .unwrap()
                .to_bits(),
            best.to_bits()
        );
    }
}

#[test]
fn diagnostics_keep_exact_reductions_and_original_norm_rounding() {
    diagnostics_check::<f32>();
    diagnostics_check::<f64>();
    use super::{
        DiscrepancyMethod as D, GeometricDiscrepancy as G, discrepancy, geometric_discrepancy,
    };
    for tiny in [f64::from_bits(1), f64::MIN_POSITIVE, 1e-150, f64::EPSILON] {
        let points = [0., 0., tiny, tiny, 1., 1.];
        let expected = RealScalar::hypot(tiny, tiny);
        assert_eq!(
            geometric_discrepancy(&points, 2, G::MinimumDistance)
                .unwrap()
                .to_bits(),
            expected.to_bits()
        );
        assert_eq!(
            discrepancy(&points, 2, D::Centered).unwrap().to_bits(),
            centered_reference(&points, 2).to_bits()
        );
    }
    assert!(discrepancy(&vec![0.; 20000], 20000, D::Centered).is_err());
    // Input validation still visits coordinates after a duplicate pair.
    assert!(geometric_discrepancy(&[0., 0., f64::NAN], 1, G::MinimumDistance).is_err());
    assert_eq!(
        geometric_discrepancy(&[0., 0., 1.], 1, G::MinimumDistance).unwrap(),
        0.0
    );
}

fn halton_batch_check<S: RealScalar>() {
    let root = QmcPlan::halton_with_bases(vec![2, 3, 5, 7, 11, 541, 4_294_967_291]).unwrap();
    let permuted = QmcPlan::halton(6)
        .unwrap()
        .with_scramble(QmcScramble::DigitPermutation(42))
        .unwrap();
    let shifted = root
        .clone()
        .with_randomization(
            QmcRandomization::RandomShift,
            &mut SplitMix64::from_seed(27),
        )
        .unwrap();
    let reversed = QmcPlan::reverse_halton(8).unwrap();
    for plan in [root, permuted, shifted, reversed] {
        let mut starts = vec![0, 1, 7, 1000, (1 << 32) - 3, (1 << 53) - 2, u64::MAX - 31];
        for &base in plan.bases().unwrap() {
            let mut power = u64::from(base);
            loop {
                starts.push(power - 2);
                let Some(next) = power.checked_mul(u64::from(base)) else {
                    break;
                };
                power = next;
            }
        }
        for start in starts {
            let count = 32.min((u64::MAX - start).saturating_add(1)) as usize;
            let mut direct = plan.sequence::<S>().unwrap();
            direct.skip_to(start).unwrap();
            let mut expected = Vec::new();
            for _ in 0..count {
                expected.extend(direct.next_point().unwrap());
            }
            for policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
                let mut sequence = plan.sequence::<S>().unwrap();
                sequence.skip_to(start).unwrap();
                let mut actual = vec![S::ZERO; count * plan.dim()];
                sequence
                    .points_into(
                        SamplingPolicy::rollback(policy),
                        &mut actual,
                        &mut SamplingWorkspace::new(),
                    )
                    .unwrap();
                assert_eq!(actual, expected, "start={start}");
                assert_eq!(
                    sequence.export_state().unwrap(),
                    direct.export_state().unwrap()
                );
            }
        }
    }
}

#[test]
fn halton_digit_carries_match_independent_points_across_every_base_power() {
    halton_batch_check::<f64>();
    halton_batch_check::<f32>();
}

#[test]
fn centered_integer_reduction_retains_endpoint_and_cancellation_bits() {
    use super::{CenteredDiscrepancy, DiscrepancyMethod as D, discrepancy};
    // Each pair is 1.5^16; summing 64 such fixed-point integers carries
    // beyond the low u64 limb before publication to RangeAccumulator.
    let corners = vec![0.0; 64 * 16];
    assert_eq!(
        discrepancy(&corners, 16, D::Centered).unwrap().to_bits(),
        centered_reference(&corners, 16).to_bits()
    );
    let coordinates = [
        0.,
        f64::from_bits(1),
        0.25,
        0.5_f64.next_down(),
        0.5,
        0.5_f64.next_up(),
        0.75,
        1.0_f64.next_down(),
        1.0,
    ];
    for dim in [1, 2, 3, 4, 5, 16, 17] {
        let points: Vec<_> = (0..31 * dim)
            .map(|i| coordinates[(i * 7 + i / dim) % coordinates.len()])
            .collect();
        let value = discrepancy(&points, dim, D::Centered).unwrap();
        assert_eq!(value.to_bits(), centered_reference(&points, dim).to_bits());
        let mut state = CenteredDiscrepancy::new(&points, dim, 32).unwrap();
        assert_eq!(state.value().unwrap().to_bits(), value.to_bits());
        let candidate = state.evaluate_append(&points[..dim]).unwrap();
        assert_eq!(
            state.append(&points[..dim]).unwrap().to_bits(),
            candidate.to_bits()
        );
    }
}

fn point_into_check<S: RealScalar>() {
    let mut rng = SplitMix64::from_seed(17);
    let plans = [
        QmcPlan::sobol(5).unwrap(),
        QmcPlan::sobol(5)
            .unwrap()
            .with_scramble(QmcScramble::DigitalShift(3))
            .unwrap(),
        QmcPlan::sobol(5)
            .unwrap()
            .with_randomization(QmcRandomization::LinearMatrixShift, &mut rng)
            .unwrap(),
        QmcPlan::niederreiter_base2(3).unwrap(),
        QmcPlan::halton(4).unwrap(),
        QmcPlan::halton(4)
            .unwrap()
            .with_scramble(QmcScramble::DigitPermutation(9))
            .unwrap(),
        QmcPlan::halton(4)
            .unwrap()
            .with_randomization(QmcRandomization::RandomShift, &mut rng)
            .unwrap(),
        QmcPlan::reverse_halton(3).unwrap(),
    ];
    for plan in &plans {
        let mut sequence = plan.sequence::<S>().unwrap();
        let mut point = vec![S::ZERO; plan.dim()];
        for index in 0..300_u64 {
            plan.point_into(index, &mut point).unwrap();
            let expected = sequence.next_point().unwrap();
            assert!(
                point
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.into_f64().to_bits() == b.into_f64().to_bits())
            );
        }
        for index in [1_u64 << 40, (1 << 53) + 7, u64::MAX - 1, u64::MAX] {
            sequence.skip_to(index).unwrap();
            plan.point_into(index, &mut point).unwrap();
            let expected = sequence.next_point().unwrap();
            assert!(
                point
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.into_f64().to_bits() == b.into_f64().to_bits())
            );
        }
        let mut wrong = vec![S::ONE; plan.dim() + 1];
        assert!(plan.point_into(0, &mut wrong).is_err());
        assert!(wrong.iter().all(|&value| value == S::ONE));
    }
}

#[test]
fn cursor_free_points_match_sequential_cursors_bit_for_bit() {
    point_into_check::<f32>();
    point_into_check::<f64>();
}
