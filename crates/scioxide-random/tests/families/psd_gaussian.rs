//! Analytic support, covariance and density tests, independent of factorization.
#![allow(clippy::unwrap_used)]

use scioxide_core::{RankThreshold, Tolerance};
use scioxide_linalg::{Matrix, RankPolicy};
use scioxide_random::{
    BitGenerator, MultivariateNormal, MultivariateNormalOptions, OutputPolicy, SamplingPolicy,
    SamplingWorkspace, SeedableRng, SplitMix64,
};

macro_rules! gaussian_cases {
    ($module:ident, $scalar:ty, $accuracy:expr, $small:expr, $large:expr) => {
        mod $module {
            use super::*;
            type S = $scalar;
            fn tolerance() -> Tolerance<S> {
                Tolerance::new($accuracy, $accuracy).unwrap()
            }

            #[test]
            fn point_mass_preserves_mean_and_rng() {
                let mean = vec![2.0 as S, -0.0, -3.0];
                let law = MultivariateNormal::new(&mean, &Matrix::zeros(3, 3).unwrap()).unwrap();
                assert_eq!(law.rank(), 0);
                assert_eq!(
                    (law.support_basis().rows(), law.support_basis().cols()),
                    (3, 0)
                );
                let mut rng = SplitMix64::from_seed(17);
                let before = rng.position();
                let mut reference = rng.clone();
                let mut samples = vec![0.0 as S; 24];
                law.sample_n_into(
                    &mut rng,
                    SamplingPolicy::rollback(OutputPolicy::Atomic),
                    &mut samples,
                    &mut SamplingWorkspace::new(),
                )
                .unwrap();
                for row in samples.as_slice().chunks_exact(3) {
                    assert_eq!(row, mean);
                    assert_eq!(row[1].to_bits(), (-0.0 as S).to_bits());
                }
                assert_eq!(rng.position(), before);
                assert_eq!(rng.next_u64().unwrap(), reference.next_u64().unwrap());
                assert_eq!(law.log_pseudo_density(&mean, tolerance()).unwrap(), 0.0);
                assert_eq!(
                    law.log_pseudo_density(&[2.0, 1.0, -3.0], tolerance())
                        .unwrap(),
                    S::NEG_INFINITY
                );
            }

            #[test]
            fn rotated_rank_one_has_analytic_density_and_moments() {
                // X = (1, -2, 3) + (Z, 2Z, 0). Nonzero eigenvalue = 5.
                let mean = [1.0 as S, -2.0, 3.0];
                let covariance =
                    Matrix::from_vec(3, 3, vec![1.0, 2.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0])
                        .unwrap();
                let law = MultivariateNormal::new(&mean, &covariance).unwrap();
                assert_eq!(law.rank(), 1);
                let expected = -0.5 * ((2.0 * core::f64::consts::PI * 5.0).ln() + 1.0);
                assert!(
                    (law.log_pseudo_density(&[2.0, 0.0, 3.0], tolerance())
                        .unwrap() as f64
                        - expected)
                        .abs()
                        < $accuracy as f64 * 8.0
                );
                assert!(
                    !law.contains_support(&[1.0, -2.0, 4.0], tolerance())
                        .unwrap()
                );
                let basis = law.support_basis().as_slice();
                for row in 0..3 {
                    for col in 0..3 {
                        assert!(
                            (5.0 * basis[row] * basis[col] - covariance.as_slice()[row * 3 + col])
                                .abs()
                                < $accuracy * 8.0
                        );
                    }
                }
                let mut rng = SplitMix64::from_seed(0x4752_4144_4941);
                let mut samples = vec![0.0 as S; 60_000];
                law.sample_n_into(
                    &mut rng,
                    SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
                    &mut samples,
                    &mut SamplingWorkspace::new(),
                )
                .unwrap();
                let mut sums = [0.0_f64; 3];
                let mut covariance_sums = [[0.0_f64; 3]; 3];
                for row in samples.as_slice().chunks_exact(3) {
                    assert!(law.contains_support(row, tolerance()).unwrap());
                    assert!(((row[1] + 2.0) - 2.0 * (row[0] - 1.0)).abs() < $accuracy * 16.0);
                    assert_eq!(row[2], 3.0);
                    for i in 0..3 {
                        let delta = (row[i] - mean[i]) as f64;
                        sums[i] += delta;
                        for j in 0..3 {
                            covariance_sums[i][j] += delta * (row[j] - mean[j]) as f64;
                        }
                    }
                }
                for i in 0..3 {
                    let variance = covariance.as_slice()[i * 3 + i] as f64;
                    assert!(sums[i].abs() / 20_000.0 <= 6.0 * (variance / 20_000.0).sqrt());
                    for j in 0..3 {
                        let expected = covariance.as_slice()[i * 3 + j] as f64;
                        let other_variance = covariance.as_slice()[j * 3 + j] as f64;
                        // Gaussian fourth-moment identity: Var(X_i X_j) = C_ii C_jj + C_ij².
                        let six_standard_errors = 6.0
                            * ((variance * other_variance + expected * expected) / 20_000.0).sqrt();
                        assert!(
                            (covariance_sums[i][j] / 20_000.0 - expected).abs()
                                <= six_standard_errors
                        );
                    }
                }
            }

            #[test]
            fn covariance_scaling_including_subnormals_and_overflowing_eigenvalue() {
                for scale in [S::from_bits(1), $small, 1.0, $large, S::MAX] {
                    let covariance = Matrix::from_vec(2, 2, vec![scale; 4]).unwrap();
                    let law = MultivariateNormal::new(&[0.0; 2], &covariance).unwrap();
                    assert_eq!(law.rank(), 1);
                    let expected = -0.5
                        * ((2.0 * core::f64::consts::PI).ln() + (scale as f64).ln() + 2.0_f64.ln());
                    assert!(
                        (law.log_pseudo_density(&[0.0; 2], tolerance()).unwrap() as f64 - expected)
                            .abs()
                            < $accuracy as f64 * 512.0
                    );
                    let mut rng = SplitMix64::from_seed(31);
                    for _ in 0..16 {
                        let sample = law.sample(&mut rng).unwrap();
                        let x = sample.as_slice();
                        assert!(x.iter().all(|x| x.is_finite()));
                        assert!(
                            (x[0] / scale.sqrt() - x[1] / scale.sqrt()).abs() < $accuracy * 8.0
                        );
                    }
                }
            }

            #[test]
            fn repeated_positive_eigenvalues_have_a_two_dimensional_support() {
                let covariance = Matrix::from_vec(
                    4,
                    4,
                    vec![
                        1.0 as S, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                        0.0, 1.0,
                    ],
                )
                .unwrap();
                let law = MultivariateNormal::new(&[0.0; 4], &covariance).unwrap();
                assert_eq!(law.rank(), 2);
                let expected = -(4.0 * core::f64::consts::PI).ln() - 1.0;
                assert!(
                    (law.log_pseudo_density(&[1.0; 4], tolerance()).unwrap() as f64 - expected)
                        .abs()
                        < $accuracy as f64 * 8.0
                );
                let q = law.support_basis().as_slice();
                for row in 0..4 {
                    for col in 0..4 {
                        let projection = q[row * 2] * q[col * 2] + q[row * 2 + 1] * q[col * 2 + 1];
                        assert!(
                            (2.0 * projection - covariance.as_slice()[row * 4 + col]).abs()
                                < $accuracy * 8.0
                        );
                    }
                }
                let mut rng = SplitMix64::from_seed(81);
                for _ in 0..32 {
                    let sample = law.sample(&mut rng).unwrap();
                    let x = sample.as_slice();
                    assert!((x[0] - x[2]).abs() < $accuracy * 16.0);
                    assert!((x[1] - x[3]).abs() < $accuracy * 16.0);
                }
            }

            #[test]
            fn rank_policy_does_not_admit_indefinite_covariance() {
                let covariance =
                    Matrix::from_vec(2, 2, vec![1.0 as S, 0.0, 0.0, S::EPSILON / 2.0]).unwrap();
                assert_eq!(
                    MultivariateNormal::new(&[0.0; 2], &covariance)
                        .unwrap()
                        .rank(),
                    1
                );
                let exact =
                    MultivariateNormalOptions::default().with_rank_policy(RankPolicy::Exact);
                assert_eq!(
                    MultivariateNormal::with_options(
                        &[0.0; 2],
                        &covariance,
                        exact,
                        Default::default()
                    )
                    .unwrap()
                    .rank(),
                    2
                );
                let truncate = MultivariateNormalOptions::default().with_rank_policy(
                    RankPolicy::Threshold(RankThreshold::new(10.0, 0.0).unwrap()),
                );
                assert_eq!(
                    MultivariateNormal::with_options(
                        &[0.0; 2],
                        &covariance,
                        truncate,
                        Default::default()
                    )
                    .unwrap()
                    .rank(),
                    0
                );
                for scale in [$small, 1.0, $large] {
                    for entries in [
                        [1.0, 2.0, 2.0, 1.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0, 1.0],
                    ] {
                        let matrix =
                            Matrix::from_vec(2, 2, entries.map(|x| x * scale).to_vec()).unwrap();
                        let error = MultivariateNormal::with_options(
                            &[0.0; 2],
                            &matrix,
                            truncate,
                            Default::default(),
                        )
                        .unwrap_err();
                        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
                    }
                }
            }

            #[test]
            fn full_rank_density_and_invalid_inputs() {
                let covariance = Matrix::from_vec(2, 2, vec![4.0 as S, 0.0, 0.0, 9.0]).unwrap();
                let law = MultivariateNormal::new(&[1.0, -1.0], &covariance).unwrap();
                let expected = -(2.0 * core::f64::consts::PI).ln() - 6.0_f64.ln() - 1.0;
                assert!(
                    (law.log_pseudo_density(&[3.0, 2.0], tolerance()).unwrap() as f64 - expected)
                        .abs()
                        < $accuracy as f64 * 8.0
                );
                assert!(
                    law.contains_support(&[S::MAX, -S::MAX], tolerance())
                        .unwrap()
                );
                let error = law.contains_support(&[0.0], tolerance()).unwrap_err();
                assert_eq!(
                    error.kind(),
                    scioxide_core::SciOxideErrorKind::DimensionMismatch
                );
                let error = law
                    .log_pseudo_density(&[S::NAN, 0.0], tolerance())
                    .unwrap_err();
                assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
                for entries in [[1.0, 0.0, 1.0, 1.0], [1.0, S::INFINITY, S::INFINITY, 1.0]] {
                    assert!(
                        MultivariateNormal::new(
                            &[0.0; 2],
                            &Matrix::from_vec(2, 2, entries.to_vec()).unwrap()
                        )
                        .is_err()
                    );
                }
            }

            #[test]
            fn prepared_singular_batches_reuse_exact_workspace() {
                let law = MultivariateNormal::new(
                    &[0.0 as S; 2],
                    &Matrix::from_vec(2, 2, vec![1.0; 4]).unwrap(),
                )
                .unwrap();
                for output_policy in [OutputPolicy::Atomic, OutputPolicy::Incremental] {
                    let count = 7;
                    let policy = SamplingPolicy::rollback(output_policy);
                    let requirements = law.workspace_requirements(policy, count).unwrap();
                    assert_eq!(
                        requirements.elements(),
                        if output_policy == OutputPolicy::Atomic {
                            15
                        } else {
                            3
                        }
                    );
                    let mut workspace = law.workspace(policy, count).unwrap();
                    let mut output = vec![0.0 as S; count * 2];
                    let mut rng = SplitMix64::from_seed(37);
                    let mut reference = rng.clone();
                    law.sample_n_into(
                        &mut rng,
                        SamplingPolicy::rollback(output_policy),
                        &mut output,
                        &mut workspace,
                    )
                    .unwrap();
                    for row in output.as_slice().chunks_exact(2) {
                        assert_eq!(row, law.sample(&mut reference).unwrap().as_slice());
                    }
                    assert_eq!(rng.position(), reference.position());
                    assert_eq!(
                        workspace.check(&requirements).unwrap(),
                        scioxide_core::Capacity::Sufficient
                    );
                    let before = rng.position();
                    law.sample_n_into(
                        &mut rng,
                        SamplingPolicy::rollback(output_policy),
                        &mut [],
                        &mut workspace,
                    )
                    .unwrap();
                    assert_eq!(rng.position(), before);
                    assert_eq!(law.workspace_requirements(policy, 0).unwrap().elements(), 0);
                }
            }
        }
    };
}
gaussian_cases!(
    f32_cases,
    f32,
    2.0e-6_f32,
    2.0_f32.powi(-100),
    2.0_f32.powi(100)
);
gaussian_cases!(
    f64_cases,
    f64,
    2.0e-14_f64,
    2.0_f64.powi(-900),
    2.0_f64.powi(900)
);
