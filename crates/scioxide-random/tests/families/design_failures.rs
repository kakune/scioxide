//! Provider failures and resource admission preserve published designs.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Cancellation, ExecutionContext, InvalidInputReason, SciOxideError};
use scioxide_random::*;
use std::sync::atomic::{AtomicUsize, Ordering};
#[derive(Clone)]
struct Failing {
    inner: SplitMix64,
    remaining: usize,
}
impl BitGenerator for Failing {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        if self.remaining == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("provider"),
                InvalidInputReason::OutOfRange,
                "injected failure",
            ));
        }
        self.remaining -= 1;
        self.inner.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok(self.next_u64()? as u32)
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        for x in out {
            *x = self.next_u64()? as u8;
        }
        Ok(())
    }
}
impl SnapshotGenerator for Failing {
    type Snapshot = Self;
    fn snapshot(&self) -> Result<Self, SciOxideError> {
        Ok(self.clone())
    }
    fn restore_snapshot(&mut self, state: Self) {
        *self = state;
    }
}
#[test]
fn randomization_and_design_errors_are_atomic() {
    for limit in [0, 1, 20, 64, 100] {
        let mut rng = Failing {
            inner: SplitMix64::from_seed(42),
            remaining: limit,
        };
        assert!(
            QmcPlan::sobol(2)
                .unwrap()
                .with_randomization(QmcRandomization::LinearMatrixShift, &mut rng)
                .is_err()
        );
        // The generator stays advanced by every word it served.
        assert_eq!(rng.remaining, 0);
    }
    for limit in 0..15 {
        let mut rng = Failing {
            inner: SplitMix64::from_seed(42),
            remaining: limit,
        };
        let mut out = [123.; 16];
        assert!(
            latin_hypercube_into(
                2,
                &mut rng,
                Rollback,
                LatinHypercubeOptions::default(),
                &mut out,
                &mut SamplingWorkspace::new(),
                ExecutionContext::sequential_native(),
            )
            .is_err()
        );
        assert_eq!(out, [123.; 16]);
        assert_eq!(rng.remaining, limit);
    }
    let plan = PoissonDisk::with_options(
        &[0.; 2],
        &[1.; 2],
        0.1,
        16,
        PoissonDiskOptions::default()
            .with_candidates_per_point(30)
            .with_max_proposals(1000)
            .with_max_grid_cells(1000)
            .with_max_distance_checks(1000),
    )
    .unwrap();
    let mut work = plan.workspace().unwrap();
    for limit in 0..12 {
        let mut out = [123.; 32];
        let mut rng = Failing {
            inner: SplitMix64::from_seed(42),
            remaining: limit,
        };
        assert!(
            plan.generate_into(
                &mut rng,
                Rollback,
                &mut out,
                &mut work,
                ExecutionContext::sequential_native(),
            )
            .is_err()
        );
        assert_eq!(out, [123.; 32]);
        assert_eq!(rng.remaining, limit);
    }
}
/// Requests cancellation from its `limit + 1`-th query on.
struct CancelAfter {
    queries: AtomicUsize,
    limit: usize,
}
impl Cancellation for CancelAfter {
    fn is_cancelled(&self) -> bool {
        self.queries.fetch_add(1, Ordering::Relaxed) >= self.limit
    }
}
#[test]
fn cancellation_stops_designs_with_valid_points() {
    let cancelled = scioxide_core::CancellationToken::new();
    cancelled.cancel();
    let options = LatinHypercubeOptions::default().with_max_proposals(100);
    // Cancellation before the first draw is an error and draws nothing.
    let mut rng = SplitMix64::from_seed(42);
    let error = latin_hypercube_into(
        2,
        &mut rng,
        AdvanceOnError,
        options,
        &mut [0.; 16],
        &mut SamplingWorkspace::new(),
        ExecutionContext::sequential().with_cancellation(&cancelled),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(rng.position(), Some(0));
    // A cancellation seen before a proposal ends optimization with a complete design.
    for limit in [1, 4] {
        let stop = CancelAfter {
            queries: AtomicUsize::new(0),
            limit,
        };
        let mut design = [0.; 16];
        let report = latin_hypercube_into(
            2,
            &mut SplitMix64::from_seed(42),
            AdvanceOnError,
            options,
            &mut design,
            &mut SamplingWorkspace::new(),
            ExecutionContext::sequential().with_cancellation(&stop),
        )
        .unwrap();
        assert_eq!(report.termination(), LatinHypercubeTermination::Stopped);
        assert_eq!(report.proposals(), limit - 1);
        for col in 0..2 {
            let mut bins: Vec<_> = design
                .chunks_exact(2)
                .map(|r| (r[col] * 8.) as usize)
                .collect();
            bins.sort();
            assert_eq!(bins, (0..8).collect::<Vec<_>>());
        }
    }
    let plan = PoissonDisk::new(&[0.; 2], &[1.; 2], 0.1, 32).unwrap();
    let mut work = plan.workspace().unwrap();
    let mut out = vec![0.; plan.output_elements()];
    let mut rng = SplitMix64::from_seed(42);
    let error = plan
        .generate_into(
            &mut rng,
            AdvanceOnError,
            &mut out,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&cancelled),
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(rng.position(), Some(0));
    let stop = CancelAfter {
        queries: AtomicUsize::new(0),
        limit: 20,
    };
    let report = plan
        .generate_into(
            &mut SplitMix64::from_seed(42),
            AdvanceOnError,
            &mut out,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&stop),
        )
        .unwrap();
    assert_eq!(report.termination(), PoissonDiskTermination::Stopped);
    assert_eq!(report.proposals(), 20);
    assert!(report.points() >= 1);
    let points = &out[..2 * report.points()];
    if report.points() > 1 {
        let separation =
            geometric_discrepancy(points, 2, GeometricDiscrepancy::MinimumDistance).unwrap();
        assert!(separation >= 0.1, "{separation}");
    }
}
#[test]
fn halton_shift_checkpoint_and_removed_modes() {
    let plan = QmcPlan::halton(3)
        .unwrap()
        .with_randomization(
            QmcRandomization::RandomShift,
            &mut SplitMix64::from_seed(42),
        )
        .unwrap();
    let mut seq = plan.sequence::<f64>().unwrap();
    seq.skip_to(1000).unwrap();
    let bytes = seq.export_state().unwrap();
    let restored = QmcPlan::import_state(&bytes).unwrap();
    assert_eq!(
        seq.next_point().unwrap(),
        restored.sequence::<f64>().unwrap().next_point().unwrap()
    );
    for tag in [2, 3] {
        let mut bytes = QmcPlan::sobol(1)
            .unwrap()
            .sequence::<f64>()
            .unwrap()
            .export_state()
            .unwrap();
        bytes[28] = tag;
        let len = bytes.len();
        let mut crc = !0_u32;
        for &byte in &bytes[..len - 4] {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0x82f6_3b78 & 0_u32.wrapping_sub(crc & 1));
            }
        }
        bytes[len - 4..].copy_from_slice(&(!crc).to_le_bytes());
        assert!(QmcPlan::import_state(&bytes).is_err());
    }
}

#[test]
fn incompatible_scrambles_fail_at_the_one_rule_before_drawing() {
    let invalid = |error: scioxide_core::SciOxideError| {
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    };
    for plan in [
        QmcPlan::halton(2).unwrap(),
        QmcPlan::reverse_halton(2).unwrap(),
    ] {
        invalid(
            plan.clone()
                .with_scramble(QmcScramble::DigitalShift(1))
                .unwrap_err(),
        );
        let mut rng = SplitMix64::from_seed(5);
        invalid(
            plan.with_randomization(QmcRandomization::LinearMatrixShift, &mut rng)
                .unwrap_err(),
        );
        assert_eq!(
            rng.next_u64().unwrap(),
            SplitMix64::from_seed(5).next_u64().unwrap()
        );
    }
    for plan in [
        QmcPlan::sobol(2).unwrap(),
        QmcPlan::niederreiter_base2(2).unwrap(),
        QmcPlan::reverse_halton(2).unwrap(),
    ] {
        invalid(
            plan.clone()
                .with_scramble(QmcScramble::DigitPermutation(1))
                .unwrap_err(),
        );
        invalid(
            plan.clone()
                .with_randomization(QmcRandomization::RandomShift, &mut SplitMix64::from_seed(5))
                .unwrap_err(),
        );
        invalid(plan.with_permutations(vec![vec![0, 1]; 2]).unwrap_err());
    }
}

#[test]
fn poisson_disk_six_dimensional_work_is_bounded() {
    for limit in [0, 1, 10_000] {
        let options = PoissonDiskOptions::default()
            .with_max_proposals(5_000)
            .with_max_distance_checks(limit);
        let plan = PoissonDisk::with_options(&[0.0_f64; 6], &[1.0; 6], 0.4, 32, options).unwrap();
        let mut work = plan.workspace().unwrap();
        let mut output = vec![99.0; plan.output_elements()];
        let report = plan
            .generate_into(
                &mut SplitMix64::from_seed(1593),
                Rollback,
                &mut output,
                &mut work,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!(report.distance_checks() <= limit);
        assert!(report.proposals() <= options.max_proposals());
        if limit <= 1 {
            assert_eq!(
                report.termination(),
                PoissonDiskTermination::BudgetExhausted
            );
        } else {
            assert!(report.points() > 1);
        }
        let points = &output[..report.points() * 6];
        assert!(points.iter().all(|x| (0.0..=1.0).contains(x)));
        for (i, a) in points.chunks_exact(6).enumerate() {
            for b in points.chunks_exact(6).take(i) {
                let square = a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
                assert!(square >= 0.16 * (1.0 - 32.0 * f64::EPSILON));
            }
        }
        assert!(output[report.points() * 6..].iter().all(|&x| x == 99.0));
    }
}
