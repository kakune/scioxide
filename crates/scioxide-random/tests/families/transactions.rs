//! Independent publication, sequence rollback, and exact allocation contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Capacity, InvalidInputReason, RealScalar, SciOxideError, SciOxideErrorKind};
use scioxide_random::{
    AdvanceOnError, BitGenerator, Distribution, Normal, OutputPolicy, QmcPlan, QuasiRng, Rng,
    SamplingPolicy, SamplingRequest, SamplingWorkspace, SeedableRng, SnapshotGenerator, SplitMix64,
    StatePolicy, Uniform, WeightedSampler, shuffle, shuffle_in_place,
};

#[derive(Clone)]
struct Script {
    position: u64,
    fail_at: u64,
    snapshot_fails: bool,
}
impl Script {
    fn new(fail_at: u64) -> Self {
        Self {
            position: 0,
            fail_at,
            snapshot_fails: false,
        }
    }
}
impl BitGenerator for Script {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        Ok(u32::try_from(self.next_u64().unwrap()).unwrap())
    }
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        let word = self.position;
        self.position += 1;
        Ok(word)
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
        output
            .iter_mut()
            .for_each(|byte| *byte = self.next_u64().unwrap().to_le_bytes()[0]);
        Ok(())
    }
}
impl<S: RealScalar> Rng<S> for Script {
    fn next_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
        Ok(if self.next_u64().unwrap() == self.fail_at {
            S::NAN
        } else {
            S::HALF
        })
    }
}
impl SnapshotGenerator for Script {
    type Snapshot = u64;
    fn snapshot(&self) -> Result<u64, SciOxideError> {
        if self.snapshot_fails {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("snapshot"),
                InvalidInputReason::Inconsistent,
                "injected snapshot failure",
            ))
        } else {
            Ok(self.position)
        }
    }
    fn restore_snapshot(&mut self, snapshot: u64) {
        self.position = snapshot;
    }
}

fn failure_matrix<S: RealScalar, P: StatePolicy<Script> + Copy>(
    policy: SamplingPolicy<P>,
    rollback: bool,
) {
    let distribution = Uniform::new(S::ZERO, S::ONE).unwrap();
    for fail_at in 0..=8 {
        let mut rng = Script::new(fail_at);
        let mut output = [S::ONE; 8];
        let mut workspace = SamplingWorkspace::new();
        let result = distribution.sample_n_into(&mut rng, policy, &mut output, &mut workspace);
        assert_eq!(result.is_ok(), fail_at == 8);
        let committed = if fail_at == 8 {
            8
        } else if policy.output_policy() == OutputPolicy::Atomic {
            0
        } else {
            fail_at as usize
        };
        assert!(output[..committed].iter().all(|&value| value == S::HALF));
        assert!(output[committed..].iter().all(|&value| value == S::ONE));
        assert_eq!(
            rng.position,
            if rollback && fail_at < 8 {
                0
            } else {
                (fail_at + 1).min(8)
            }
        );
    }
}
#[test]
fn failure_at_every_scalar_boundary_both_widths_and_all_policies() {
    for output in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        failure_matrix::<f32, _>(SamplingPolicy::advance_on_error(output), false);
        failure_matrix::<f64, _>(SamplingPolicy::advance_on_error(output), false);
        failure_matrix::<f32, _>(SamplingPolicy::rollback(output), true);
        failure_matrix::<f64, _>(SamplingPolicy::rollback(output), true);
    }
}

fn quasi_rows<S: RealScalar, Q: QuasiRng<S> + SnapshotGenerator, P: StatePolicy<Q> + Copy>(
    mut rng: Q,
    policy: SamplingPolicy<P>,
    maximum: u64,
    rollback: bool,
) {
    for completed in 0_u64..4 {
        // Leave exactly `completed` valid points before the range limit.
        rng.skip_to(maximum - completed.saturating_sub(1)).unwrap();
        if completed == 0 {
            rng.next_point().unwrap();
        }
        let before = rng.snapshot().unwrap();
        let mut expected = vec![S::ONE; 12];
        for row in expected[..completed as usize * 3].chunks_exact_mut(3) {
            rng.next_point_into(row).unwrap();
        }
        rng.restore_snapshot(before);
        let mut output = vec![S::ONE; 12];
        assert!(
            rng.points_into(policy, &mut output, &mut SamplingWorkspace::new())
                .is_err()
        );
        if policy.output_policy() == OutputPolicy::Atomic {
            assert_eq!(output, vec![S::ONE; 12]);
        } else {
            assert_eq!(output, expected);
        }
        let mut next = [S::ZERO; 3];
        assert_eq!(
            rng.next_point_into(&mut next).is_ok(),
            rollback && completed > 0
        );
    }
}
#[test]
fn quasi_point_boundaries_and_rollback() {
    for output in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        quasi_rows(
            QmcPlan::sobol(3).unwrap().sequence::<f32>().unwrap(),
            SamplingPolicy::advance_on_error(output),
            u64::MAX,
            false,
        );
        quasi_rows(
            QmcPlan::sobol(3).unwrap().sequence::<f32>().unwrap(),
            SamplingPolicy::rollback(output),
            u64::MAX,
            true,
        );
        quasi_rows(
            QmcPlan::halton(3).unwrap().sequence::<f32>().unwrap(),
            SamplingPolicy::advance_on_error(output),
            u64::MAX,
            false,
        );
        quasi_rows(
            QmcPlan::halton(3).unwrap().sequence::<f32>().unwrap(),
            SamplingPolicy::rollback(output),
            u64::MAX,
            true,
        );
        quasi_rows(
            QmcPlan::sobol(3).unwrap().sequence::<f64>().unwrap(),
            SamplingPolicy::rollback(output),
            u64::MAX,
            true,
        );
        quasi_rows(
            QmcPlan::halton(3).unwrap().sequence::<f64>().unwrap(),
            SamplingPolicy::rollback(output),
            u64::MAX,
            true,
        );
    }
}

fn owned_quasi_rows<S: RealScalar>(plan: &QmcPlan) {
    for completed in 0_u64..4 {
        // Leave exactly `completed` valid points before the range limit.
        let mut sequence = plan.sequence::<S>().unwrap();
        sequence
            .skip_to(u64::MAX - completed.saturating_sub(1))
            .unwrap();
        if completed == 0 {
            sequence.next_point().unwrap();
        }
        assert!(sequence.points(4).is_err());
        // The owned form leaves the sequence at the limit, past every point it produced.
        assert_eq!(sequence.position().next_index(), None);
        assert!(sequence.next_point().is_err());
    }
}
#[test]
fn owned_quasi_points_leave_the_sequence_at_the_range_limit() {
    for plan in [QmcPlan::sobol(3).unwrap(), QmcPlan::halton(3).unwrap()] {
        owned_quasi_rows::<f32>(&plan);
        owned_quasi_rows::<f64>(&plan);
    }
}

#[test]
fn borrowed_engines_and_scrambled_sobol_restore_the_same_sequence() {
    let mut engine = SplitMix64::from_seed(525);
    let expected = engine.next_u64().unwrap();
    let mut engine = SplitMix64::from_seed(525);
    let borrowed = &mut engine;
    let snapshot = borrowed.snapshot().unwrap();
    borrowed.next_u64().unwrap();
    borrowed.restore_snapshot(snapshot);
    assert_eq!(borrowed.next_u64().unwrap(), expected);
    let sobol_plan = QmcPlan::sobol(7)
        .unwrap()
        .with_randomization(
            scioxide_random::QmcRandomization::LinearMatrixShift,
            &mut <scioxide_random::SplitMix64 as scioxide_random::SeedableRng>::from_seed(12),
        )
        .unwrap();
    let mut sobol = sobol_plan.sequence::<f64>().unwrap();
    sobol.skip_to(17).unwrap();
    let snapshot = sobol.snapshot().unwrap();
    let expected = sobol.next_point().unwrap();
    sobol.skip(23).unwrap();
    sobol.restore_snapshot(snapshot);
    assert_eq!(sobol.next_point().unwrap(), expected);
}

#[test]
fn weighted_failures_obey_the_same_publication_policy() {
    let sampler = WeightedSampler::new(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    for output_policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        for fail_at in 0..4 {
            let mut rng = Script::new(2 * fail_at + 1);
            let mut output = [usize::MAX; 4];
            let result = sampler.sample_n_into(
                &mut rng,
                SamplingPolicy::rollback(output_policy),
                &mut output,
                &mut SamplingWorkspace::new(),
            );
            assert!(result.is_err());
            assert_eq!(rng.position, 0);
            if output_policy == OutputPolicy::Atomic {
                assert_eq!(output, [usize::MAX; 4]);
            }
        }
    }
}

struct RejectWords {
    calls: u64,
}
impl BitGenerator for RejectWords {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.calls += 1;
        Ok(0)
    }
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.calls += 1;
        Ok(0)
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
        output.fill(0);
        Ok(())
    }
}
impl SnapshotGenerator for RejectWords {
    type Snapshot = u64;
    fn snapshot(&self) -> Result<u64, SciOxideError> {
        Ok(self.calls)
    }
    fn restore_snapshot(&mut self, snapshot: u64) {
        self.calls = snapshot;
    }
}
#[test]
fn shuffle_rejection_counts_words_independently_of_commits() {
    for output in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        let mut rng = RejectWords { calls: 0 };
        let mut values = [1, 2, 3, 4];
        assert!(
            shuffle_in_place(
                &mut values,
                &mut rng,
                SamplingPolicy::advance_on_error(output),
                &mut SamplingWorkspace::new()
            )
            .is_err()
        );
        assert_eq!(rng.calls, 1 + 1_024);
        assert_eq!(
            values,
            if output == OutputPolicy::Atomic {
                [1, 2, 3, 4]
            } else {
                [4, 2, 3, 1]
            }
        );
        let mut rng = RejectWords { calls: 0 };
        let mut values = [1, 2, 3, 4];
        assert!(
            shuffle_in_place(
                &mut values,
                &mut rng,
                SamplingPolicy::rollback(output),
                &mut SamplingWorkspace::new()
            )
            .is_err()
        );
        assert_eq!(rng.calls, 0);
    }
    // The owned shuffle runs the same kernel: the first swap stands, the
    // rejected words of the second stay consumed.
    let mut rng = RejectWords { calls: 0 };
    let mut values = [1, 2, 3, 4];
    assert!(shuffle(&mut values, &mut rng).is_err());
    assert_eq!(rng.calls, 1 + 1_024);
    assert_eq!(values, [4, 2, 3, 1]);
    let mut rng = RejectWords { calls: 0 };
    let error = scioxide_random::sample_indices(4, 4, &mut rng).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert_eq!(rng.calls, 1 + 1_024);
}

// A provider without snapshot capability is accepted by advance-on-error.
fn accepts_unsnapshotable<G: Rng<f64>>(rng: &mut G) {
    Uniform::new(0.0, 1.0)
        .unwrap()
        .sample_n_into(
            rng,
            SamplingPolicy::<AdvanceOnError>::advance_on_error(OutputPolicy::Incremental),
            &mut [0.0],
            &mut SamplingWorkspace::new(),
        )
        .unwrap();
}
#[test]
fn advance_mode_has_no_snapshot_bound() {
    accepts_unsnapshotable(&mut Script::new(u64::MAX));
}

#[derive(Clone)]
struct NormalFailure<S> {
    calls: usize,
    fail_at: usize,
    word: u64,
    uniform: S,
}
impl<S: RealScalar> BitGenerator for NormalFailure<S> {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        Ok(u32::from_le_bytes(
            self.next_u64().unwrap().to_le_bytes()[..4]
                .try_into()
                .unwrap(),
        ))
    }
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        let failed = self.calls >= self.fail_at;
        self.calls += 1;
        Ok(if failed { self.word } else { 0 })
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
        out.fill(0);
        Ok(())
    }
}
impl<S: RealScalar> Rng<S> for NormalFailure<S> {
    fn next_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
        Ok(self.uniform)
    }
}
impl<S: RealScalar> SnapshotGenerator for NormalFailure<S> {
    type Snapshot = usize;
    fn snapshot(&self) -> Result<usize, SciOxideError> {
        Ok(self.calls)
    }
    fn restore_snapshot(&mut self, snapshot: usize) {
        self.calls = snapshot;
    }
}
fn multivariate_boundaries<
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
    P: StatePolicy<NormalFailure<S>> + Copy,
>(
    word: u64,
    uniform: S,
    policy: SamplingPolicy<P>,
    rollback: bool,
) {
    multivariate_rank_boundaries(word, uniform, policy, rollback, false);
    multivariate_rank_boundaries(word, uniform, policy, rollback, true);
}
fn multivariate_rank_boundaries<
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
    P: StatePolicy<NormalFailure<S>> + Copy,
>(
    word: u64,
    uniform: S,
    policy: SamplingPolicy<P>,
    rollback: bool,
    singular: bool,
) {
    use scioxide_linalg::Matrix;
    use scioxide_random::MultivariateNormal;
    let sampler = MultivariateNormal::new(
        &[<S as scioxide_core::AlgebraicScalar>::ZERO; 2],
        &Matrix::from_vec(
            2,
            2,
            vec![
                <S as scioxide_core::AlgebraicScalar>::ONE,
                if singular {
                    <S as scioxide_core::AlgebraicScalar>::ONE
                } else {
                    <S as scioxide_core::AlgebraicScalar>::ZERO
                },
                if singular {
                    <S as scioxide_core::AlgebraicScalar>::ONE
                } else {
                    <S as scioxide_core::AlgebraicScalar>::ZERO
                },
                <S as scioxide_core::AlgebraicScalar>::ONE,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    for fail_at in 0..4 * sampler.rank() {
        let mut rng = NormalFailure {
            calls: 0,
            fail_at,
            word,
            uniform,
        };
        let mut output = vec![<S as scioxide_core::AlgebraicScalar>::ONE; 8];
        let mut workspace = SamplingWorkspace::new();
        assert!(
            sampler
                .sample_n_into(&mut rng, policy, &mut output, &mut workspace)
                .is_err()
        );
        let published = if policy.output_policy() == OutputPolicy::Atomic {
            0
        } else {
            fail_at / sampler.rank() * 2
        };
        assert!(
            output.as_slice()[..published]
                .iter()
                .all(|&value| value == <S as scioxide_core::AlgebraicScalar>::ZERO)
        );
        assert!(
            output.as_slice()[published..]
                .iter()
                .all(|&value| value == <S as scioxide_core::AlgebraicScalar>::ONE)
        );
        // Each rejected ziggurat proposal draws its layer word and a decoded
        // wedge uniform before the 1,024-candidate guard stops the draw.
        assert_eq!(rng.calls, if rollback { 0 } else { fail_at + 2 * 1_024 });
        assert_eq!(
            workspace
                .check(&sampler.workspace_requirements(policy, 4).unwrap())
                .unwrap(),
            Capacity::Sufficient
        );
    }
}
#[test]
fn multivariate_failure_at_every_coordinate_keeps_failed_rows_unpublished() {
    let word32 = u64::from(((1_u32 << 23) - 1) << 9 | 1);
    let word64 = (((1_u64 << 52) - 1) << 9) | 1;
    for output in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
        multivariate_boundaries(
            word32,
            0.999_999_94_f32,
            SamplingPolicy::advance_on_error(output),
            false,
        );
        multivariate_boundaries(
            word32,
            0.999_999_94_f32,
            SamplingPolicy::rollback(output),
            true,
        );
        multivariate_boundaries(
            word64,
            f64::from_bits(1.0_f64.to_bits() - 1),
            SamplingPolicy::advance_on_error(output),
            false,
        );
        multivariate_boundaries(
            word64,
            f64::from_bits(1.0_f64.to_bits() - 1),
            SamplingPolicy::rollback(output),
            true,
        );
    }
}

#[test]
fn direct_law_failures_follow_publication_and_rollback_policies() {
    fn check<D: Distribution<f64>, P: StatePolicy<Script>>(
        law: &D,
        policy: SamplingPolicy<P>,
        rollback: bool,
        fail_at: u64,
    ) {
        let mut rng = Script::new(fail_at);
        let mut output = [123.0; 3];
        let mut workspace = policy.workspace(SamplingRequest::scalars(3)).unwrap();
        let error = law
            .sample_n_into(&mut rng, policy, &mut output, &mut workspace)
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
        if policy.output_policy() == OutputPolicy::Atomic {
            assert_eq!(output, [123.0; 3]);
        } else {
            assert_ne!(output[0], 123.0);
            assert_eq!(output[1..], [123.0; 2]);
        }
        assert_eq!(rng.position, if rollback { 0 } else { fail_at + 1 });
    }
    fn policies<D: Distribution<f64>>(law: D, fail_at: u64) {
        for output in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
            check(
                &law,
                SamplingPolicy::advance_on_error(output),
                false,
                fail_at,
            );
            check(&law, SamplingPolicy::rollback(output), true, fail_at);
        }
    }
    policies(
        scioxide_random::Truncated::new(Normal::new(0.0_f64, 1.0).unwrap(), 8.0, 9.0).unwrap(),
        1,
    );
    policies(
        scioxide_random::Histogram::from_masses(&[0.0, 1.0, 2.0], &[1.0, 3.0]).unwrap(),
        1,
    );
    policies(scioxide_random::PowerNormal::new(2.0).unwrap(), 1);
    policies(
        scioxide_random::Mixture::new(
            vec![
                Uniform::new(0.0, 1.0).unwrap(),
                Uniform::new(2.0, 3.0).unwrap(),
            ],
            &[1.0, 3.0],
        )
        .unwrap(),
        4,
    );
}

#[test]
fn owned_sampling_leaves_the_generator_advanced() {
    let law = Uniform::new(0.0_f64, 1.0).unwrap();
    let mut rng = Script::new(2);
    assert!(law.sample_n(&mut rng, 4).is_err());
    assert_eq!(rng.position, 3); // The rejected provider word is consumed too.
    // Owned forms take no snapshot, so a generator that cannot save its state still draws.
    let mut rng = Script::new(u64::MAX);
    rng.snapshot_fails = true;
    assert_eq!(law.sample_n(&mut rng, 2).unwrap(), [0.5, 0.5]);
    assert_eq!(rng.position, 2);
    assert!(law.sample_n(&mut rng, 0).unwrap().is_empty());
    assert_eq!(rng.position, 2);
}

#[test]
fn sample_indices_leaves_the_generator_advanced() {
    use crate::vector_laws::Failing;
    // Dense (partial Fisher-Yates) and sparse (Floyd) paths both fail on
    // their third index draw; storage is reserved before any draw.
    for (n, k) in [(8, 6), (1_000, 6)] {
        let reference = scioxide_random::sample_indices(n, k, &mut Failing::new(None)).unwrap();
        assert_eq!(reference.len(), k);
        let mut rng = Failing::new(Some(2));
        let error = scioxide_random::sample_indices(n, k, &mut rng).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NotImplemented);
        assert_eq!(rng.requests, 3);
        let mut rng = Failing::new(None);
        let error = scioxide_random::sample_indices(n, n + 1, &mut rng).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    }
}
