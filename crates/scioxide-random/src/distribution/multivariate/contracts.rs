#![allow(clippy::unwrap_used)]
use super::*;
use crate::distribution::LawOptions;
use crate::{
    Multinomial, OutputPolicy, SamplingPolicy, SamplingWorkspace, SeedableRng, SerializableState,
    SnapshotGenerator, SplitMix64,
};

#[derive(Clone)]
struct Fails {
    inner: SplitMix64,
    left: usize,
    reads: usize,
}
impl BitGenerator for Fails {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        if self.left == 0 {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "injected provider failure",
            ));
        }
        self.left -= 1;
        self.reads += 1;
        self.inner.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok((self.next_u64()? >> 32) as u32)
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        for b in out {
            *b = self.next_u64()? as u8;
        }
        Ok(())
    }
}
impl SnapshotGenerator for Fails {
    type Snapshot = Self;
    fn snapshot(&self) -> Result<Self, SciOxideError> {
        Ok(self.clone())
    }
    fn restore_snapshot(&mut self, s: Self) {
        *self = s;
    }
}
#[test]
fn publication_and_rng_policies_are_independent() {
    let law = Multinomial::with_options(
        1,
        &[0.5, 0.5],
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    for atomic in [false, true] {
        for rollback in [false, true] {
            let mut rng = Fails {
                inner: SplitMix64::from_seed(5),
                left: 1,
                reads: 0,
            };
            let mut out = [99; 4];
            let mut work = SamplingWorkspace::new();
            let policy = if atomic {
                OutputPolicy::Atomic
            } else {
                OutputPolicy::Incremental
            };
            let result = if rollback {
                law.sample_n_into(
                    &mut rng,
                    SamplingPolicy::rollback(policy),
                    &mut out,
                    &mut work,
                )
            } else {
                law.sample_n_into(
                    &mut rng,
                    SamplingPolicy::advance_on_error(policy),
                    &mut out,
                    &mut work,
                )
            };
            assert!(result.is_err());
            assert_eq!(rng.reads, usize::from(!rollback));
            assert_eq!(&out[2..], &[99, 99]);
            if atomic {
                assert_eq!(out, [99; 4]);
            } else {
                assert_eq!(out[0] + out[1], 1);
            }
        }
    }
}
#[test]
fn matrix_rows_and_word_budget_preserve_state() {
    let law = Wishart::with_options(
        5.0,
        &Matrix::identity(2).unwrap(),
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let mut rng = SplitMix64::from_seed(12);
    let mut storage = [77.0; 4];
    let mut work = law
        .workspace(SamplingPolicy::rollback(OutputPolicy::Atomic), 1)
        .unwrap();
    law.sample_n_into(
        &mut rng,
        SamplingPolicy::rollback(OutputPolicy::Atomic),
        &mut storage,
        &mut work,
    )
    .unwrap();
    // A row-major matrix sample: symmetric with a positive diagonal.
    assert_eq!(law.shape(), (2, 2));
    assert!(storage[0] > 0.0 && storage[3] > 0.0 && storage[1] == storage[2]);
    let options = LawOptions::default().with_sample_word_budget(Budget::ONE);
    let identity = Matrix::identity(2).unwrap();
    let law = Wishart::with_options(5.0, identity.view(), options).unwrap();
    let before = rng.export_state().unwrap();
    let old = storage;
    assert!(
        law.sample_n_into(
            &mut rng,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut storage,
            &mut work
        )
        .is_err()
    );
    assert_eq!(storage, old);
    assert_eq!(rng.export_state().unwrap(), before);
    let law = RandomTable::<f32>::with_options(
        &[2, 2],
        &[2, 2],
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let mut out = [99; 4];
    law.sample_n_into(
        &mut rng,
        SamplingPolicy::rollback(OutputPolicy::Atomic),
        &mut out,
        &mut SamplingWorkspace::new(),
    )
    .unwrap();
    assert!(law.contains_support(&out).unwrap());
    let layout = scioxide_core::StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
    assert!(
        law.pmf_view(scioxide_core::NdView::new(&out, layout).unwrap())
            .unwrap()
            > 0.0
    );
}
#[test]
fn deterministic_limits_and_invalid_domains() {
    let mut rng = Fails {
        inner: SplitMix64::from_seed(1),
        left: 0,
        reads: 0,
    };
    assert_eq!(
        Dirichlet::with_options(
            &[0.5_f32],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .sample(&mut rng)
        .unwrap(),
        [1.0]
    );
    assert_eq!(
        DirichletMultinomial::with_options(
            7,
            &[0.5_f32],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .sample(&mut rng)
        .unwrap(),
        [7]
    );
    assert_eq!(
        MultivariateHypergeometric::<f32>::with_options(
            &[0, 0],
            0,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .sample(&mut rng)
        .unwrap(),
        [0, 0]
    );
    assert_eq!(
        RandomTable::<f32>::with_options(
            &[0, 0],
            &[0, 0],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .sample(&mut rng)
        .unwrap(),
        [0; 4]
    );
    assert_eq!(
        SpecialOrthogonalGroup::<f32>::with_options(
            1,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .sample(&mut rng)
        .unwrap(),
        [1.0]
    );
    assert!(
        Wishart::with_options(
            1.0,
            &Matrix::identity(2).unwrap(),
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        MatrixNormal::with_options(
            &Matrix::<f64>::zeros(1, 2).unwrap(),
            &Matrix::identity(2).unwrap(),
            &Matrix::identity(2).unwrap(),
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        RandomCorrelation::with_options(
            &[1.0, 2.0],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        UnitaryGroup::<f64>::with_options(
            usize::MAX,
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    let law = Dirichlet::with_options(
        &[0.5, 0.5],
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(law.pdf(&[0.0, 1.0]).unwrap(), f64::INFINITY);
}
#[test]
fn haar_bounded_moment_and_sign_checks() {
    let mut rng = SplitMix64::from_seed(542);
    let o = OrthogonalGroup::<f64>::with_options(
        3,
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let u = UnitaryGroup::<f64>::with_options(
        3,
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let s = SpecialOrthogonalGroup::<f64>::with_options(
        3,
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let mut sums = [0.0; 6];
    let n = 4096;
    for _ in 0..n {
        let a = Matrix::from_vec(3, 3, o.sample(&mut rng).unwrap()).unwrap();
        let b = u.sample(&mut rng).unwrap();
        let c = s.sample(&mut rng).unwrap();
        let x = a.view().get(0, 0).unwrap();
        let y = b[0].re * b[0].re + b[0].im * b[0].im;
        sums[0] += x * x;
        sums[1] += x.powi(4);
        sums[2] += y;
        sums[3] += y * y;
        sums[4] += f64::from(
            scioxide_linalg::LuDecomposition::factor(a.view())
                .and_then(|factor| factor.determinant())
                .unwrap()
                > 0.0,
        );
        sums[5] += c[0] * c[0];
    }
    // Six bounded [0,1] statistics: union-bound false-positive probability <=1e-10.
    let bound = ((12.0_f64 / 1e-10).ln() / (2.0 * n as f64)).sqrt();
    for (sum, mean) in
        sums.into_iter()
            .zip([1.0 / 3.0, 1.0 / 5.0, 1.0 / 3.0, 1.0 / 6.0, 0.5, 1.0 / 3.0])
    {
        assert!((sum / n as f64 - mean).abs() < bound);
    }
}
#[test]
fn singular_scalar_marginals_are_point_masses() {
    let covariance = Matrix::zeros(2, 2).unwrap();
    let law = MultivariateStudentT::with_options(
        &[1.0, 2.0],
        &covariance,
        0.25,
        Default::default(),
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(law.mean().unwrap(), [1.0, 2.0]);
    assert_eq!(law.variance().unwrap(), [0.0, 0.0]);
    let gaussian = crate::MultivariateNormal::new(&[1.0, 2.0], &covariance).unwrap();
    assert_eq!(gaussian.covariance().unwrap().as_slice(), &[0.0; 4]);
}

#[test]
fn covariance_buffers_respect_payload_preflight() {
    let options = LawOptions::default().with_max_bytes(16);
    let prepared = LawOptions::default().with_max_bytes(16 * 16 * 8 - 1);
    assert!(
        DirichletMultinomial::with_options(2, &[1.0; 16], prepared)
            .unwrap()
            .covariance()
            .is_err()
    );
    assert!(
        Multinomial::with_options(2, &[1.0 / 16.0; 16], prepared)
            .unwrap()
            .covariance()
            .is_err()
    );
    assert!(
        MultivariateHypergeometric::<f64>::with_options(&[1, 1], 1, options)
            .unwrap()
            .covariance()
            .is_err()
    );
}

#[test]
fn gaussian_strided_input_rows_reuse_kernel() {
    let mean = [0.0, 1.0];
    let covariance_storage = [1.0, 0.25, 99.0, 0.25, 2.0, 99.0];
    let covariance =
        scioxide_linalg::MatrixView::from_slice_with_strides(2, 2, &covariance_storage, 3, 1)
            .unwrap();
    let law = crate::MultivariateNormal::new(&mean, covariance).unwrap();
    let mut rng = SplitMix64::from_seed(3);
    let mut expected_rng = rng.clone();
    let mut expected = law.sample(&mut expected_rng).unwrap();
    expected.extend(law.sample(&mut expected_rng).unwrap());
    let mut rows = [99.0; 4];
    law.sample_n_into(
        &mut rng,
        SamplingPolicy::rollback(OutputPolicy::Atomic),
        &mut rows,
        &mut SamplingWorkspace::new(),
    )
    .unwrap();
    assert_eq!(&rows[..], expected.as_slice());
}

#[test]
fn matrix_t_tiny_gram_retains_gaussian_limit() {
    let mean = Matrix::from_vec(1, 2, vec![0.0, 0.0]).unwrap();
    let row = Matrix::from_vec(1, 1, vec![1e20]).unwrap();
    let col = Matrix::identity(2).unwrap();
    let x = Matrix::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
    let law = MatrixT::with_options(
        &mean,
        &row,
        &col,
        1e20,
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let log = law.log_pdf(x.view()).unwrap();
    assert!((log - (-2.3378770664093453)).abs() < 5e-12, "{log}");
    assert!(
        Dirichlet::with_options(
            &[1e100, 1e100],
            crate::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .log_pdf(&[0.5, 0.5])
        .is_err()
    );
}

#[test]
fn vector_moments_and_covariances_match_defining_identities() {
    let d = Dirichlet::with_options(
        &[2.0, 3.0, 5.0],
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(d.mean().unwrap(), [0.2, 0.3, 0.5]);
    assert!((d.covariance().unwrap().view().get(0, 1).unwrap() + 0.06 / 11.0).abs() < 1e-14);
    let dm = DirichletMultinomial::with_options(
        20,
        &[2.0, 3.0, 5.0],
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(dm.mean().unwrap(), [4.0, 6.0, 10.0]);
    assert!((dm.covariance().unwrap().view().get(0, 1).unwrap() + 36.0 / 11.0).abs() < 1e-13);
    let m = Multinomial::with_options(
        20,
        &[0.2, 0.3, 0.5],
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(m.mean().unwrap(), [4.0, 6.0, 10.0]);
    assert!((m.variance().unwrap()[0] - 3.2).abs() < 1e-14);
    let h = MultivariateHypergeometric::<f64>::with_options(
        &[20, 30, 50],
        20,
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(h.mean().unwrap(), [4.0, 6.0, 10.0]);
    assert!((h.covariance().unwrap().view().get(0, 0).unwrap() - 3.2 * 80.0 / 99.0).abs() < 1e-13);
    let n = NormalInverseGamma::with_options(
        0.2,
        2.5,
        3.5,
        4.0,
        crate::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert_eq!(n.mean().unwrap(), [0.2, 1.6]);
    assert!((n.variance().unwrap()[0] - 0.64).abs() < 1e-14);
}
