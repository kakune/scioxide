//! Prepared multivariate queries and batches preserve numerical and transaction contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Capacity, RealScalar, SciOxideError, SciOxideErrorKind, Tolerance};
use scioxide_linalg::Matrix;
use scioxide_random::{
    BitGenerator, Dirichlet, LawOptions, LogDensity, Multinomial, MultivariateNormal, OutputPolicy,
    SamplingPolicy, SamplingWorkspace, SeedableRng, SnapshotGenerator, SplitMix64,
};

#[test]
fn empty_owned_rows_need_no_scratch_or_rng_calls() {
    let law = Dirichlet::new(&[2.0_f64, 3.0]).unwrap();
    let matrix = scioxide_random::Wishart::new(3.0_f64, &Matrix::identity(2).unwrap()).unwrap();
    let mut rng = SplitMix64::from_seed(7);
    {
        assert!(law.sample_n(&mut rng, 0).unwrap().is_empty());
        assert!(matrix.sample_n(&mut rng, 0).unwrap().is_empty());
    };
    assert_eq!(rng.position(), Some(0));
}

fn close<S: RealScalar>(actual: S, expected: f64) {
    assert!(
        (actual.into_f64() - expected).abs()
            <= 128.0 * S::EPSILON.into_f64() * (1.0 + expected.abs()),
        "{actual:?} != {expected}"
    );
}

fn gaussian<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let law = MultivariateNormal::new(
        &[S::ZERO; 2],
        &Matrix::from_vec(2, 2, vec![s(2.0), s(0.25), s(0.25), S::ONE]).unwrap(),
    )
    .unwrap();
    for x in [[0.0, 0.0], [0.25, -0.5], [32.0, 64.0], [1e-10, -1e-10]] {
        let point = x.map(s);
        let q = (x[0] * x[0] - 0.5 * x[0] * x[1] + 2.0 * x[1] * x[1]) / 1.9375;
        let expected = -core::f64::consts::TAU.ln() - 0.5 * 1.9375_f64.ln() - 0.5 * q;
        close(law.log_pdf(&point).unwrap(), expected);
    }
    // The displacement exceeds MAX, but its log-density is representable.
    let extreme = MultivariateNormal::new(
        &[S::MAX / s(4.0)],
        &Matrix::from_vec(1, 1, vec![S::MAX]).unwrap(),
    )
    .unwrap();
    let density = extreme.log_pdf(&[-S::MAX]).unwrap();
    close(density / S::MAX, -0.78125);
    {
        assert!(
            extreme
                .evaluate_support(
                    &[-S::MAX],
                    Tolerance::from_machine_epsilon(),
                    &mut SamplingWorkspace::new()
                )
                .unwrap()
        );
        close(
            extreme
                .evaluate_log_pseudo_density(
                    &[-S::MAX],
                    Tolerance::from_machine_epsilon(),
                    &mut SamplingWorkspace::new(),
                )
                .unwrap()
                / S::MAX,
            -0.78125,
        );
    };
    {
        assert_eq!(
            law.log_pdf(&[S::NAN, S::ZERO]).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(
            law.log_pdf(&[S::ZERO]).unwrap_err().kind(),
            SciOxideErrorKind::DimensionMismatch
        );
    }

    let tolerance = Tolerance::new(s(0.01), S::ZERO).unwrap();
    for covariance in [vec![S::ONE, S::ZERO, S::ZERO, S::ZERO], vec![S::ZERO; 4]] {
        let singular =
            MultivariateNormal::new(&[S::ZERO; 2], &Matrix::from_vec(2, 2, covariance).unwrap())
                .unwrap();
        let policy = SamplingPolicy::rollback(OutputPolicy::Incremental);
        let mut work = singular.workspace(policy, 1).unwrap();
        for point in [[S::ZERO, S::ZERO], [S::ZERO, s(0.01)], [S::ZERO, s(0.02)]] {
            let supported = point[1] <= s(0.01);
            {
                assert_eq!(
                    singular
                        .evaluate_support(&point, tolerance, &mut work)
                        .unwrap(),
                    supported
                );
                let density = singular
                    .evaluate_log_pseudo_density(&point, tolerance, &mut work)
                    .unwrap();
                if supported {
                    close(
                        density,
                        -(singular.rank() as f64) * 0.5 * core::f64::consts::TAU.ln(),
                    );
                } else {
                    assert_eq!(density, S::NEG_INFINITY);
                }
            };
        }
        assert_eq!(
            singular
                .evaluate_support(&[S::ZERO; 2], tolerance, &mut SamplingWorkspace::new())
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::DimensionMismatch
        );
    }

    for covariance in [
        vec![s(2.0), s(0.25), s(0.25), S::ONE],
        vec![S::ONE; 4],
        vec![S::ZERO; 4],
    ] {
        let law =
            MultivariateNormal::new(&[S::ZERO; 2], &Matrix::from_vec(2, 2, covariance).unwrap())
                .unwrap();
        for output_policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
            let policy = SamplingPolicy::rollback(output_policy);
            let mut work = law.workspace(policy, 4).unwrap();
            let mut rng = SplitMix64::from_seed(1347);
            let mut out = [s(99.0); 8];
            {
                law.sample_n_into(&mut rng, policy, &mut out, &mut work)
                    .unwrap();
            };
            assert!(out.iter().all(|x| x.is_finite()));
        }
    }
}

fn simplex_and_counts<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    for alpha in [
        vec![s(1.0), s(2.0), s(3.0)],
        vec![s(0.1), s(0.2), s(0.3)],
        vec![s(1e-30); 3],
        vec![s(0.1)],
    ] {
        let law = Dirichlet::new(&alpha).unwrap();
        for output_policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
            let policy = SamplingPolicy::rollback(output_policy);
            let mut work = law.workspace(policy, 8).unwrap();
            let mut out = vec![S::ZERO; 8 * law.dimension()];
            let mut rng = SplitMix64::from_seed(1347);
            for _ in 0..8 {
                law.sample_n_into(&mut rng, policy, &mut out, &mut work)
                    .unwrap();
                assert!(
                    out.chunks_exact(law.dimension())
                        .all(|row| law.contains_support(row).unwrap())
                );
            }
        }
    }
    for probabilities in [[s(0.0), s(0.25), s(0.75)], [S::ONE, s(1e-30), S::ZERO]] {
        let law = Multinomial::new(100, &probabilities).unwrap();
        for output_policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
            let policy = SamplingPolicy::rollback(output_policy);
            let mut work = policy
                .workspace(scioxide_random::SamplingRequest::rows(8, 3))
                .unwrap();
            let mut out = [0; 24];
            law.sample_n_into(
                &mut SplitMix64::from_seed(1347),
                policy,
                &mut out,
                &mut work,
            )
            .unwrap();
            assert!(
                out.chunks_exact(3)
                    .all(|row| law.contains_support(row).unwrap())
            );
        }
    }
}

#[test]
fn prepared_queries_and_batches_f32() {
    gaussian::<f32>();
    simplex_and_counts::<f32>();
}
#[test]
fn prepared_queries_and_batches_f64() {
    gaussian::<f64>();
    simplex_and_counts::<f64>();
}

#[derive(Clone)]
struct Fails {
    engine: SplitMix64,
    reads: usize,
    limit: usize,
}
impl Fails {
    fn new(limit: usize) -> Self {
        Self {
            engine: SplitMix64::from_seed(1347),
            reads: 0,
            limit,
        }
    }
    fn charge(&mut self) -> Result<(), SciOxideError> {
        if self.reads == self.limit {
            return Err(SciOxideError::not_implemented("injected RNG failure"));
        }
        self.reads += 1;
        Ok(())
    }
}
impl BitGenerator for Fails {
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        self.charge()?;
        self.engine.next_u32()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.charge()?;
        self.engine.next_u64()
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.next_u32()
    }
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.next_u64()
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        for value in out {
            *value = self.next_u32()? as u8;
        }
        Ok(())
    }
}
impl SnapshotGenerator for Fails {
    type Snapshot = Self;
    fn snapshot(&self) -> Result<Self, SciOxideError> {
        Ok(self.clone())
    }
    fn restore_snapshot(&mut self, snapshot: Self) {
        *self = snapshot;
    }
}

#[test]
fn dirichlet_failures_preserve_rows_and_independent_rng_policy() {
    for alpha in [[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]] {
        let law = Dirichlet::new(&alpha).unwrap();
        let mut probe = Fails::new(usize::MAX);
        let first = law.sample(&mut probe).unwrap();
        let limit = probe.reads + 1;
        for output_policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
            for rollback in [false, true] {
                let mut work = law
                    .workspace(SamplingPolicy::rollback(output_policy), 3)
                    .unwrap();
                let mut out = [99.0; 9];
                let mut rng = Fails::new(limit);
                let result = if rollback {
                    law.sample_n_into(
                        &mut rng,
                        SamplingPolicy::rollback(output_policy),
                        &mut out,
                        &mut work,
                    )
                } else {
                    law.sample_n_into(
                        &mut rng,
                        SamplingPolicy::advance_on_error(output_policy),
                        &mut out,
                        &mut work,
                    )
                };
                assert!(result.is_err());
                assert_eq!(rng.reads, if rollback { 0 } else { limit });
                assert_eq!(&out[3..], &[99.0; 6]);
                if output_policy == OutputPolicy::Atomic {
                    assert_eq!(out, [99.0; 9]);
                } else {
                    for (&a, &b) in out[..3].iter().zip(&first) {
                        close(a, b);
                    }
                }
                rng.limit = usize::MAX;
                law.sample_n_into(
                    &mut rng,
                    SamplingPolicy::rollback(output_policy),
                    &mut out,
                    &mut work,
                )
                .unwrap();
                assert!(
                    out.chunks_exact(3)
                        .all(|row| law.contains_support(row).unwrap())
                );
            }
        }
    }
}

#[test]
fn dirichlet_cached_kernels_share_the_vector_word_budget() {
    for alpha in [[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]] {
        let law = Dirichlet::with_options(
            &alpha,
            LawOptions::default().with_sample_word_budget(scioxide_core::Budget::ONE),
        )
        .unwrap();
        for rollback in [false, true] {
            let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
            let mut workspace = law.workspace(policy, 1).unwrap();
            let mut output = [99.0; 3];
            let mut rng = Fails::new(usize::MAX);
            let error = if rollback {
                law.sample_n_into(&mut rng, policy, &mut output, &mut workspace)
            } else {
                law.sample_n_into(
                    &mut rng,
                    SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
                    &mut output,
                    &mut workspace,
                )
            }
            .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
            assert_eq!(rng.reads, usize::from(!rollback));
            assert_eq!(output, [99.0; 3]);
        }
    }
}

#[test]
fn dirichlet_storage_limits_count_parameters_staging_and_widened_scratch() {
    let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
    let alpha = [1.0_f32, 2.0];
    let large_limit = 1024;
    let probe =
        Dirichlet::with_options(&alpha, LawOptions::default().with_max_bytes(large_limit)).unwrap();
    let requirements = probe.workspace_requirements(policy, 3).unwrap();
    assert_eq!(
        (requirements.elements(), requirements.f64_scratch_elements()),
        (6, 2)
    );
    let parameter_bytes = large_limit - requirements.max_bytes();
    let scratch_bytes = 6 * size_of::<f32>() + 2 * size_of::<f64>();
    let exact = Dirichlet::with_options(
        &alpha,
        LawOptions::default().with_max_bytes(parameter_bytes + scratch_bytes),
    )
    .unwrap();
    let requirements = exact.workspace_requirements(policy, 3).unwrap();
    let mut workspace = exact.workspace(policy, 3).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let mut out = [99.0; 6];
    let mut rng = Fails::new(usize::MAX);
    exact
        .sample_n_into(&mut rng, policy, &mut out, &mut workspace)
        .unwrap();
    let short = Dirichlet::with_options(
        &alpha,
        LawOptions::default().with_max_bytes(parameter_bytes + scratch_bytes - 1),
    )
    .unwrap();
    let mut rng = Fails::new(usize::MAX);
    let mut out = [99.0; 6];
    assert_eq!(
        short
            .sample_n_into(&mut rng, policy, &mut out, &mut workspace)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(out, [99.0; 6]);
    assert_eq!(rng.reads, 0);

    let larger = Dirichlet::new(&[1.0_f32; 32]).unwrap();
    let mut retained = larger.workspace(policy, 1).unwrap();
    let limited = Dirichlet::with_options(
        &alpha,
        LawOptions::default().with_max_bytes(parameter_bytes + 256),
    )
    .unwrap();
    // The primary f32 arena is only 128 bytes; retained f64 scratch adds 256.
    assert_eq!(
        limited
            .sample_n_into(&mut rng, policy, &mut out, &mut retained)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(rng.reads, 0);
    assert!(
        limited
            .sample_n_into(&mut rng, policy, &mut [], &mut retained)
            .is_ok()
    );
    assert_eq!(rng.reads, 0);
    assert_eq!(
        probe
            .sample_n_into(
                &mut rng,
                policy,
                &mut out[..3],
                &mut SamplingWorkspace::new()
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(rng.reads, 0);
    let one = Dirichlet::new(&[0.1_f32]).unwrap();
    let mut no_words = Fails::new(0);
    let mut out = [0.0; 3];
    one.sample_n_into(
        &mut no_words,
        policy,
        &mut out,
        &mut SamplingWorkspace::new(),
    )
    .unwrap();
    assert_eq!(out, [1.0; 3]);
    assert_eq!(no_words.reads, 0);
}

/// A repeated interior word makes the tiny-concentration Gamma logarithms
/// identical. Their normalization must retain ln(d) even beyond its ULP range.
struct Repeated;
impl BitGenerator for Repeated {
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok(0x8000_0000)
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        Ok(0x8000_0000_0000_0001)
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.next_u32()
    }
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.next_u64()
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        out.fill(128);
        Ok(())
    }
}
#[test]
fn tiny_equal_gamma_logarithms_normalize_before_output_conversion() {
    let law = Dirichlet::new(&[1e-30_f32; 3]).unwrap();
    let policy = SamplingPolicy::advance_on_error(OutputPolicy::Atomic);
    let mut workspace = law.workspace(policy, 1).unwrap();
    let mut output = [0.0; 3];
    law.sample_n_into(&mut Repeated, policy, &mut output, &mut workspace)
        .unwrap();
    assert!(law.contains_support(&output).unwrap());
    for value in output {
        close(value, 1.0 / 3.0);
    }
}
