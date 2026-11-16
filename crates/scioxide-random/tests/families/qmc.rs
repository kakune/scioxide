//! Independent full-range QMC tests, lifecycle and quality contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::backend::RealBackend;
#[path = "../support/qmc_vectors.rs"]
mod reference;
use scioxide_random::{
    BitGenerator, OutputPolicy, QmcPlan, QmcRandomization, QmcScramble, QuasiRng, SamplingPolicy,
    SamplingWorkspace, SeedableRng, SplitMix64,
};

fn check_words(plan: &QmcPlan, cases: &[(usize, u64, u64)]) {
    let columns = plan.direction_words().unwrap();
    let mut wide = plan.sequence::<f64>().unwrap();
    let mut narrow = plan.sequence::<f32>().unwrap();
    let mut output64 = vec![0.0; plan.dim()];
    let mut output32 = vec![0.0; plan.dim()];
    for &(dimension, index, expected) in cases {
        let gray = index ^ (index >> 1);
        let word = columns[dimension - 1]
            .iter()
            .enumerate()
            .filter(|(bit, _)| gray & (1 << bit) != 0)
            .fold(0, |result, (_, column)| result ^ column);
        assert_eq!(word, expected, "dimension={dimension} index={index}");
        wide.skip_to(index).unwrap();
        narrow.skip_to(index).unwrap();
        wide.next_point_into(&mut output64).unwrap();
        narrow.next_point_into(&mut output32).unwrap();
        assert_eq!(output64[dimension - 1], f64::from_u64_half_open01(expected));
        assert_eq!(output32[dimension - 1], f32::from_u64_half_open01(expected));
    }
}
#[test]
fn independent_64_bit_digital_vectors() {
    check_words(&QmcPlan::sobol(21_201).unwrap(), &reference::SOBOL_WORDS);
    check_words(
        &QmcPlan::niederreiter_base2(12).unwrap(),
        &reference::NIEDERREITER_WORDS,
    );
}
#[test]
fn independent_scramble_vectors() {
    for linear in [false, true] {
        let plan = if linear {
            QmcPlan::sobol(21)
                .unwrap()
                .with_randomization(
                    QmcRandomization::LinearMatrixShift,
                    &mut <scioxide_random::SplitMix64 as scioxide_random::SeedableRng>::from_seed(
                        42,
                    ),
                )
                .unwrap()
        } else {
            QmcPlan::sobol(21)
                .unwrap()
                .with_scramble(QmcScramble::DigitalShift(42))
                .unwrap()
        };
        let mut wide = plan.sequence::<f64>().unwrap();
        let mut narrow = plan.sequence::<f32>().unwrap();
        for &(dimension, index, shifted, matrix_shifted) in &reference::SCRAMBLED_WORDS {
            let expected = if linear { matrix_shifted } else { shifted };
            wide.skip_to(index).unwrap();
            narrow.skip_to(index).unwrap();
            assert_eq!(
                wide.next_point().unwrap()[dimension - 1],
                f64::from_u64_half_open01(expected)
            );
            assert_eq!(
                narrow.next_point().unwrap()[dimension - 1],
                f32::from_u64_half_open01(expected)
            );
        }
    }
}
#[test]
fn rational_halton_and_reverse_halton_vectors() {
    for &(base, index, halton, reverse) in &reference::RADICAL_VALUES {
        for (plan, expected) in [
            (QmcPlan::halton_with_bases(vec![base]).unwrap(), halton),
            (
                QmcPlan::reverse_halton_with_bases(vec![base]).unwrap(),
                reverse,
            ),
        ] {
            let mut wide = plan.sequence::<f64>().unwrap();
            let mut narrow = plan.sequence::<f32>().unwrap();
            wide.skip_to(index).unwrap();
            narrow.skip_to(index).unwrap();
            let a = wide.next_point().unwrap()[0];
            let b = narrow.next_point().unwrap()[0];
            assert!((0.0..1.0).contains(&a) && (0.0..1.0).contains(&b));
            assert!(
                (a - expected).abs() <= 4.0 * f64::EPSILON * expected.max(f64::MIN_POSITIVE),
                "base={base} index={index} actual={a} expected={expected}"
            );
            assert!(
                (f64::from(b) - expected).abs()
                    <= 2.0 * f64::from(f32::EPSILON) * expected.max(f64::MIN_POSITIVE)
            );
        }
    }
}

#[test]
fn replay_checkpoints_chunks_and_terminal_state() {
    for plan in plans() {
        for start in [0, 1, (1 << 32) - 7, 1 << 53, u64::MAX - 6] {
            let mut cursor = plan.sequence::<f64>().unwrap();
            cursor.skip_to(start).unwrap();
            let bytes = cursor.export_state().unwrap();
            let checkpoint = QmcPlan::import_state(&bytes).unwrap();
            let restored_plan = checkpoint.plan();
            assert_eq!(restored_plan.family(), plan.family());
            assert_eq!(
                restored_plan.randomization_words(),
                plan.randomization_words()
            );
            assert_eq!(restored_plan.permutations(), plan.permutations());
            let mut restored = checkpoint.sequence::<f64>().unwrap();
            assert_eq!(restored.export_state().unwrap(), bytes);
            let mut narrow = checkpoint.sequence::<f32>().unwrap();
            let mut direct_narrow = plan.sequence::<f32>().unwrap();
            direct_narrow.skip_to(start).unwrap();
            for _ in 0..7 {
                assert_eq!(cursor.next_point().unwrap(), restored.next_point().unwrap());
                assert_eq!(
                    narrow.next_point().unwrap(),
                    direct_narrow.next_point().unwrap()
                );
            }
            if start == u64::MAX - 6 {
                assert_eq!(cursor.position().next_index(), None);
                let terminal = QmcPlan::import_state(&cursor.export_state().unwrap()).unwrap();
                let mut exhausted = terminal.sequence::<f32>().unwrap();
                let mut output = [0.25; 3];
                assert!(exhausted.next_point_into(&mut output).is_err());
                assert_eq!(output, [0.25; 3]);
                exhausted.skip(0).unwrap();
                exhausted.skip_to(0).unwrap();
                assert_eq!(exhausted.position().next_index(), Some(0));
            }
        }
        let mut sequential = plan.sequence::<f64>().unwrap();
        let expected = sequential.points(64).unwrap();
        let mut chunks = Vec::new();
        for start in [0, 16, 32, 48] {
            let mut cursor = plan.sequence::<f64>().unwrap();
            cursor.skip_to(start).unwrap();
            chunks.extend(cursor.points(16).unwrap());
        }
        assert_eq!(expected, chunks);
    }
}

#[test]
fn invalid_dimensions_bases_permutations_and_scrambles() {
    for dim in [0, 21202] {
        assert!(QmcPlan::sobol(dim).is_err());
    }
    for dim in [0, 13] {
        assert!(QmcPlan::niederreiter_base2(dim).is_err());
    }
    for dim in [0, 101] {
        assert!(QmcPlan::halton(dim).is_err());
    }
    for bases in [vec![], vec![1], vec![4], vec![2, 2], vec![4294967295]] {
        assert!(QmcPlan::halton_with_bases(bases).is_err());
    }
    for permutation in [
        vec![],
        vec![vec![1, 0]],
        vec![vec![0, 0]],
        vec![vec![0, 2]],
        vec![vec![0, 1, 2]],
    ] {
        assert!(
            QmcPlan::halton(1)
                .unwrap()
                .with_permutations(permutation)
                .is_err()
        );
    }
    assert!(
        QmcPlan::sobol(2)
            .unwrap()
            .with_scramble(QmcScramble::DigitPermutation(1))
            .is_err()
    );
    assert!(
        QmcPlan::reverse_halton(2)
            .unwrap()
            .with_randomization(
                scioxide_random::QmcRandomization::LinearMatrixShift,
                &mut <scioxide_random::SplitMix64 as scioxide_random::SeedableRng>::from_seed(1)
            )
            .is_err()
    );
    let base_plan = QmcPlan::halton(100).unwrap();
    let mut bases = base_plan.bases().unwrap().to_vec();
    bases.push(547);
    assert_eq!(QmcPlan::halton_with_bases(bases).unwrap().dim(), 101);
    let plan = QmcPlan::sobol(2).unwrap();
    let mut cursor = plan.sequence::<f64>().unwrap();
    let position = cursor.position();
    let error = cursor.next_point_into(&mut [0.0; 3]).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(cursor.position(), position);
    cursor.skip_to(u64::MAX).unwrap();
    let position = cursor.position();
    assert!(cursor.skip(2).is_err());
    assert_eq!(cursor.position(), position);
    cursor.skip(1).unwrap();
    assert_eq!(cursor.position().next_index(), None);
}
#[test]
fn checkpoint_corruption_and_export_shape_are_transactional() {
    for plan in plans() {
        let cursor = plan.sequence::<f64>().unwrap();
        let bytes = cursor.export_state().unwrap();
        for offset in 0..bytes.len() {
            let mut corrupt = bytes.clone();
            corrupt[offset] ^= 0x80;
            assert!(QmcPlan::import_state(&corrupt).is_err(), "offset={offset}");
        }
        for length in 0..bytes.len() {
            assert!(QmcPlan::import_state(&bytes[..length]).is_err());
        }
        let mut wrong = vec![0xa5; bytes.len() + 1];
        assert!(cursor.export_state_into(&mut wrong).is_err());
        assert!(wrong.iter().all(|&byte| byte == 0xa5));
    }
}
#[test]
fn origin_and_reference_small_points() {
    for plan in [
        QmcPlan::sobol(3).unwrap(),
        QmcPlan::halton(3).unwrap(),
        QmcPlan::reverse_halton(3).unwrap(),
        QmcPlan::niederreiter_base2(3).unwrap(),
    ] {
        assert_eq!(
            plan.sequence::<f64>().unwrap().next_point().unwrap(),
            [0.0; 3]
        );
    }
    let sobol = QmcPlan::sobol(3).unwrap();
    let points = sobol.sequence::<f64>().unwrap().points(4).unwrap();
    assert_eq!(
        points,
        [
            0., 0., 0., 0.5, 0.5, 0.5, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75
        ]
    );
}
#[test]
fn digital_nets_and_anchored_box_discrepancy() {
    for plan in [
        QmcPlan::sobol(2).unwrap(),
        QmcPlan::sobol(2)
            .unwrap()
            .with_randomization(
                scioxide_random::QmcRandomization::LinearMatrixShift,
                &mut <scioxide_random::SplitMix64 as scioxide_random::SeedableRng>::from_seed(71),
            )
            .unwrap(),
        QmcPlan::niederreiter_base2(2).unwrap(),
    ] {
        let points = plan.sequence::<f64>().unwrap().points(256).unwrap();
        // Every dyadic elementary rectangle of volume 1/256 contains one point.
        for xbits in 0..=8 {
            let ybits = 8 - xbits;
            let mut occupancy = [0; 256];
            for p in points.chunks_exact(2) {
                let x = (p[0] * f64::from(1 << xbits)) as usize;
                let y = (p[1] * f64::from(1 << ybits)) as usize;
                occupancy[x * (1 << ybits) + y] += 1;
            }
            assert!(occupancy.iter().all(|&count| count == 1));
        }
        let mut discrepancy = 0.0_f64;
        for x in 0..=32 {
            for y in 0..=32 {
                let a = f64::from(x) / 32.;
                let b = f64::from(y) / 32.;
                let count = points
                    .chunks_exact(2)
                    .filter(|p| p[0] < a && p[1] < b)
                    .count();
                discrepancy = discrepancy.max((count as f64 / 256. - a * b).abs());
            }
        }
        assert!(
            discrepancy <= 0.02,
            "anchored-box discrepancy {discrepancy}"
        );
    }
}
#[test]
fn randomized_replicate_calibration_against_exact_integral() {
    // 128 fixed independent seed labels; no retry-until-pass. The XOR-shift
    // ensemble is tested on integral x*y = 1/4. Standard error is empirical.
    let mut estimates = Vec::new();
    for seed in 1..=128 {
        let plan = QmcPlan::sobol(2)
            .unwrap()
            .with_scramble(QmcScramble::DigitalShift(seed))
            .unwrap();
        let points = plan.sequence::<f64>().unwrap().points(256).unwrap();
        estimates.push(points.chunks_exact(2).map(|p| p[0] * p[1]).sum::<f64>() / 256.);
    }
    let mean = estimates.iter().sum::<f64>() / 128.;
    let variance = estimates.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 127.;
    assert!(variance > 0. && variance < 2e-6);
    assert!((mean - 0.25).abs() <= 5. * (variance / 128.).sqrt());
    // Across groups, check empirical interval coverage with a broad, fixed
    // binomial envelope rather than interpreting any single interval as proof.
    let covered = estimates
        .chunks_exact(8)
        .filter(|group| {
            let mean = group.iter().sum::<f64>() / 8.;
            let variance = group.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 7.;
            (mean - 0.25).abs() <= 2.365 * (variance / 8.).sqrt()
        })
        .count();
    assert!((11..=16).contains(&covered), "covered {covered}/16");
}

fn recheck(bytes: &mut [u8]) {
    // Independent bit-serial Castagnoli checksum bypasses integrity failure so
    // these tests exercise semantic validation of well-checksummed payloads.
    let end = bytes.len() - 4;
    let crc = bytes[..end].iter().fold(u32::MAX, |mut crc, &byte| {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0x82f6_3b78 } else { 0 };
        }
        crc
    });
    bytes[end..].copy_from_slice(&(!crc).to_le_bytes());
}
#[test]
fn semantic_checkpoint_rejection_with_valid_checksums() {
    let plan = QmcPlan::sobol(3).unwrap();
    let bytes = plan.sequence::<f64>().unwrap().export_state().unwrap();
    for (offset, value) in [
        (4, 2),
        (5, 1),
        (6, 255),
        (7, 2),
        (8, 2),
        (9, 2),
        (10, 1),
        (11, 1),
        (16, 0),
        (28, 99),
        (29, 1),
        (37, 2),
    ] {
        let mut bad = bytes.clone();
        bad[offset] = value;
        recheck(&mut bad);
        assert!(QmcPlan::import_state(&bad).is_err(), "field {offset}");
    }
    let mut bad = bytes.clone();
    bad[9] = 1;
    recheck(&mut bad);
    assert!(QmcPlan::import_state(&bad).is_err()); // nonterminal index marked exhausted
    let mut bad = bytes;
    bad.insert(bad.len() - 4, 0);
    bad[12] += 1;
    recheck(&mut bad);
    assert!(QmcPlan::import_state(&bad).is_err()); // exact envelope, trailing payload
    let plan = QmcPlan::halton(2)
        .unwrap()
        .with_permutations(vec![vec![0, 1], vec![0, 2, 1]])
        .unwrap();
    let bytes = plan.sequence::<f64>().unwrap().export_state().unwrap();
    for (offset, value) in [(38, 4), (42, 2), (46, 1), (50, 0), (54, 3), (58, 1)] {
        let mut bad = bytes.clone();
        bad[offset] = value;
        recheck(&mut bad);
        assert!(
            QmcPlan::import_state(&bad).is_err(),
            "Halton field {offset}"
        );
    }
}
#[test]
fn permutation_nets_and_seed_identity() {
    let a = QmcPlan::halton(8)
        .unwrap()
        .with_scramble(QmcScramble::DigitPermutation(42))
        .unwrap();
    let b = QmcPlan::halton(8)
        .unwrap()
        .with_scramble(QmcScramble::DigitPermutation(42))
        .unwrap();
    let c = QmcPlan::halton(8)
        .unwrap()
        .with_scramble(QmcScramble::DigitPermutation(43))
        .unwrap();
    assert_eq!(a.permutations(), b.permutations());
    assert_ne!(a.permutations(), c.permutations());
    for (&base, permutation) in a.bases().unwrap().iter().zip(a.permutations().unwrap()) {
        let plan = QmcPlan::halton_with_bases(vec![base])
            .unwrap()
            .with_permutations(vec![permutation.clone()])
            .unwrap();
        let points = plan
            .sequence::<f64>()
            .unwrap()
            .points(base as usize)
            .unwrap();
        let mut sorted = points;
        sorted.sort_by(f64::total_cmp);
        for (digit, value) in sorted.into_iter().enumerate() {
            assert!((value - digit as f64 / f64::from(base)).abs() < 2. * f64::EPSILON);
        }
    }
}
#[test]
fn parallel_chunk_assignment_replays_exactly() {
    let plan = QmcPlan::sobol(8)
        .unwrap()
        .with_randomization(
            scioxide_random::QmcRandomization::LinearMatrixShift,
            &mut <scioxide_random::SplitMix64 as scioxide_random::SeedableRng>::from_seed(42),
        )
        .unwrap();
    let expected = plan.sequence::<f64>().unwrap().points(256).unwrap();
    let actual = std::thread::scope(|scope| {
        // Collect before joining so every worker is launched concurrently.
        #[allow(clippy::needless_collect)]
        let handles: Vec<_> = (0..4)
            .map(|chunk| {
                let plan = &plan;
                scope.spawn(move || {
                    let mut cursor = plan.sequence::<f64>().unwrap();
                    cursor.skip_to(chunk * 64).unwrap();
                    cursor.points(64).unwrap()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(actual, expected);
}

#[path = "../support/qmc_plans.rs"]
mod fixtures;
use fixtures::plans;

fn movement_boundaries<S: scioxide_core::RealScalar>() {
    for plan in plans() {
        let mut cursor = plan.sequence::<S>().unwrap();
        let mut reference = plan.sequence::<S>().unwrap();
        cursor.skip(15_u64).unwrap();
        cursor.skip_to(7).unwrap();
        reference.skip_to(7).unwrap();
        assert_eq!(
            cursor.next_point().unwrap(),
            reference.next_point().unwrap()
        );
        let saved = cursor.export_state().unwrap();
        cursor.skip_to(8).unwrap();
        assert_eq!(cursor.export_state().unwrap(), saved);

        cursor.skip_to(u64::MAX).unwrap();
        let saved = cursor.export_state().unwrap();
        assert!(cursor.skip(2).is_err());
        assert_eq!(cursor.export_state().unwrap(), saved);
        let last = cursor.next_point().unwrap();
        let exhausted = cursor.export_state().unwrap();
        cursor.skip(0).unwrap();
        assert_eq!(cursor.export_state().unwrap(), exhausted);
        assert!(cursor.skip(1).is_err());
        assert_eq!(cursor.export_state().unwrap(), exhausted);
        cursor.skip_to(u64::MAX).unwrap();
        assert_eq!(cursor.next_point().unwrap(), last);
        cursor.skip_to(1).unwrap();
        cursor.skip(u64::MAX).unwrap();
        assert_eq!(cursor.position().next_index(), None);
        cursor.skip_to(0).unwrap();
        assert_eq!(cursor.position().next_index(), Some(0));
    }
}

#[test]
fn movement_replays_and_checks_terminal_boundaries_at_both_widths() {
    movement_boundaries::<f32>();
    movement_boundaries::<f64>();
}

/// The words drawn for a randomization are the generator's uniform words, the
/// matrix rows made unit lower triangular; seed 42 pins them.
#[test]
fn randomization_words_are_canonical_generator_words() {
    let mut rng = SplitMix64::from_seed(42);
    let plan = QmcPlan::sobol(3)
        .unwrap()
        .with_randomization(QmcRandomization::LinearMatrixShift, &mut rng)
        .unwrap();
    let mut reference = SplitMix64::from_seed(42);
    let expected: Vec<u64> = (0..3 * 65)
        .map(|i| {
            let (row, word) = (i % 65, reference.next_uniform_u64().unwrap());
            if row < 64 {
                let diagonal = 1_u64 << (63 - row);
                (word & !(diagonal | (diagonal - 1))) | diagonal
            } else {
                word
            }
        })
        .collect();
    assert_eq!(plan.randomization_words(), expected);
    assert_eq!(rng.position(), reference.position());
    let mut rng = SplitMix64::from_seed(42);
    let plan = QmcPlan::halton(3)
        .unwrap()
        .with_randomization(QmcRandomization::RandomShift, &mut rng)
        .unwrap();
    let mut reference = SplitMix64::from_seed(42);
    let expected: Vec<u64> = (0..3)
        .map(|_| reference.next_uniform_u64().unwrap())
        .collect();
    assert_eq!(plan.randomization_words(), expected);
    assert_eq!(rng.position(), Some(3));
}

#[test]
fn owned_points_equal_caller_buffer_points() {
    let plans = [
        QmcPlan::sobol(4).unwrap(),
        QmcPlan::niederreiter_base2(4).unwrap(),
        QmcPlan::halton(4).unwrap(),
        QmcPlan::reverse_halton(4).unwrap(),
        QmcPlan::sobol(4)
            .unwrap()
            .with_randomization(
                QmcRandomization::LinearMatrixShift,
                &mut SplitMix64::from_seed(42),
            )
            .unwrap(),
    ];
    for plan in &plans {
        let mut owned = plan.sequence::<f64>().unwrap();
        owned.skip_to(5).unwrap();
        let expected = owned.points(37).unwrap();
        assert_eq!(owned.position().next_index(), Some(42));
        for output in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
            let mut sequence = plan.sequence::<f64>().unwrap();
            sequence.skip_to(5).unwrap();
            let mut points = vec![0.0; 37 * 4];
            let policy = SamplingPolicy::rollback(output);
            sequence
                .points_into(policy, &mut points, &mut SamplingWorkspace::new())
                .unwrap();
            assert_eq!(points, expected);
            assert_eq!(sequence.position().next_index(), Some(42));
        }
        assert!(
            plan.sequence::<f64>()
                .unwrap()
                .points(0)
                .unwrap()
                .is_empty()
        );
    }
}
