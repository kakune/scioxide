//! Multiple marginals and mixed-coordinate projections, rather than a single
//! moment. Seed 1335 and 32768 joint draws per case; each joint sample is reused
//! for every projection. References are fixture-validated binary64 scalar laws.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Matrix;
use scioxide_random::*;

#[allow(dead_code)]
#[path = "../support/laws.rs"]
mod laws;
#[path = "../support/statistics.rs"]
mod statistics;
use laws::CheckedLaw;

fn samples(mut draw: impl FnMut() -> Vec<f64>) -> Vec<Vec<f64>> {
    (0..statistics::DRAWS)
        .map(|_| {
            let row = draw();
            assert!(row.iter().all(|x| x.is_finite()));
            row
        })
        .collect()
}
fn marginals(label: &str, samples: &[Vec<f64>], references: &[Box<dyn CheckedLaw<f64>>]) {
    for (i, reference) in references.iter().enumerate() {
        let mut rows = samples.iter();
        statistics::continuous(&format!("{label}/{i}"), &**reference, || {
            rows.next().unwrap()[i]
        });
    }
}
fn real_row<S: RealScalar>(row: Vec<S>) -> Vec<f64> {
    row.into_iter().map(|x| x.into_f64()).collect()
}

fn continuous_marginals<S: RealScalar + NumericScalar<Real = S>>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let mut rng = SplitMix64::from_seed(1335);
    for alpha in [[0.3, 2.0, 5.0], [1.0, 2.0, 3.0]] {
        let law = Dirichlet::with_options(
            &alpha.map(s),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let values = samples(|| real_row(law.sample(&mut rng).unwrap()));
        let total: f64 = alpha.iter().sum();
        let references: Vec<Box<dyn CheckedLaw<f64>>> = alpha
            .iter()
            .map(|&a| Box::new(Beta::new(a, total - a).unwrap()) as Box<dyn CheckedLaw<f64>>)
            .collect();
        marginals("Dirichlet", &values, &references);
    }
    let shape = Matrix::from_vec(2, 2, vec![s(4.0), s(1.0), s(1.0), s(2.0)]).unwrap();
    let normal = MultivariateNormal::new(&[s(1.0), s(-2.0)], &shape).unwrap();
    let values = samples(|| {
        let mut x = real_row(normal.sample(&mut rng).unwrap().as_slice().to_vec());
        x.push(x[0] + x[1]);
        x
    });
    marginals(
        "MVN",
        &values,
        &[
            Box::new(Normal::new(1.0, 2.0).unwrap()),
            Box::new(Normal::new(-2.0, 2.0_f64.sqrt()).unwrap()),
            Box::new(Normal::new(-1.0, 8.0_f64.sqrt()).unwrap()),
        ],
    );
    for df in [1.0, 5.0, f64::INFINITY] {
        let law = if df.is_infinite() {
            MultivariateStudentT::gaussian_limit(&[s(1.0), s(-2.0)], &shape)
        } else {
            MultivariateStudentT::with_options(
                &[s(1.0), s(-2.0)],
                &shape,
                s(df),
                Default::default(),
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
        }
        .unwrap();
        let values = samples(|| {
            let mut x = real_row(law.sample(&mut rng).unwrap());
            x.push(x[0] + x[1]);
            x
        });
        let references: Vec<Box<dyn CheckedLaw<f64>>> =
            [(1.0, 2.0), (-2.0, 2.0_f64.sqrt()), (-1.0, 8.0_f64.sqrt())]
                .into_iter()
                .map(|(location, scale)| {
                    if df.is_infinite() {
                        Box::new(Normal::new(location, scale).unwrap()) as Box<dyn CheckedLaw<f64>>
                    } else {
                        Box::new(Affine::new(StudentT::new(df).unwrap(), location, scale).unwrap())
                            as Box<dyn CheckedLaw<f64>>
                    }
                })
                .collect();
        marginals("MultivariateStudentT", &values, &references);
    }
    let nig = NormalInverseGamma::with_options(
        s(1.0),
        s(2.0),
        s(3.0),
        s(4.0),
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let values = samples(|| real_row(nig.sample(&mut rng).unwrap()));
    marginals(
        "NormalInverseGamma",
        &values,
        &[
            Box::new(
                Affine::new(StudentT::new(6.0).unwrap(), 1.0, (4.0_f64 / 6.0).sqrt()).unwrap(),
            ),
            Box::new(Affine::new(GeneralizedGamma::new(3.0, -1.0).unwrap(), 0.0, 4.0).unwrap()),
        ],
    );

    let mean = Matrix::zeros(2, 2).unwrap();
    let column = Matrix::from_vec(2, 2, vec![s(2.0), s(0.5), s(0.5), s(1.0)]).unwrap();
    let law = MatrixNormal::with_options(
        &mean,
        &shape,
        &column,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let values = samples(|| {
        let mut x = real_row(law.sample(&mut rng).unwrap());
        x.push(x[0] + x[3]);
        x
    });
    marginals(
        "MatrixNormal",
        &values,
        &[
            Box::new(Normal::new(0.0, 8.0_f64.sqrt()).unwrap()),
            Box::new(Normal::new(0.0, 2.0).unwrap()),
            Box::new(Normal::new(0.0, 2.0).unwrap()),
            Box::new(Normal::new(0.0, 2.0_f64.sqrt()).unwrap()),
            Box::new(Normal::new(0.0, 11.0_f64.sqrt()).unwrap()),
        ],
    );
    for df in [1.0, 5.0] {
        let law = MatrixT::with_options(
            &mean,
            &shape,
            &column,
            s(df),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        // A single entry has scale sqrt(U_ii V_jj / df). A same-column
        // row sum is another one-dimensional projection with that convention.
        let values = samples(|| {
            let mut x = real_row(law.sample(&mut rng).unwrap());
            x.push(x[0] + x[2]);
            x
        });
        let references: Vec<Box<dyn CheckedLaw<f64>>> = [8.0, 4.0, 4.0, 2.0, 16.0]
            .into_iter()
            .map(|v| {
                Box::new(Affine::new(StudentT::new(df).unwrap(), 0.0, (v / df).sqrt()).unwrap())
                    as Box<dyn CheckedLaw<f64>>
            })
            .collect();
        marginals("MatrixT", &values, &references);
    }
    for df in [2.5, 6.0] {
        let law = Wishart::with_options(
            s(df),
            &shape,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let values = samples(|| {
            let x = real_row(law.sample(&mut rng).unwrap());
            vec![x[0] / 4.0, x[3] / 2.0, (x[0] + 2.0 * x[1] + x[3]) / 8.0]
        });
        let references: Vec<Box<dyn CheckedLaw<f64>>> = (0..3)
            .map(|_| Box::new(ChiSquared::new(df).unwrap()) as Box<dyn CheckedLaw<f64>>)
            .collect();
        marginals("Wishart", &values, &references);
        let law = InverseWishart::with_options(
            s(df),
            &shape,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let values = samples(|| {
            let x = real_row(law.sample(&mut rng).unwrap());
            vec![x[0] / 4.0, x[3] / 2.0, (x[0] + 2.0 * x[1] + x[3]) / 8.0]
        });
        let references: Vec<Box<dyn CheckedLaw<f64>>> = (0..3)
            .map(|_| {
                Box::new(
                    Affine::new(
                        GeneralizedGamma::new((df - 1.0) / 2.0, -1.0).unwrap(),
                        0.0,
                        0.5,
                    )
                    .unwrap(),
                ) as Box<dyn CheckedLaw<f64>>
            })
            .collect();
        marginals("InverseWishart", &values, &references);
    }
}

fn count_marginals<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let mut rng = SplitMix64::from_seed(1335);
    macro_rules! check {
        ($label:literal,$law:expr,$references:expr,$hi:expr) => {{
            let law = $law;
            let values: Vec<_> = (0..statistics::DRAWS)
                .map(|_| law.sample(&mut rng).unwrap())
                .collect();
            for (i, reference) in $references.into_iter().enumerate() {
                let mut rows = values.iter();
                statistics::discrete(
                    &format!("{}/{i}", $label),
                    0,
                    $hi,
                    |k| {
                        if k < 0 {
                            0.0
                        } else {
                            reference.pmf(k as usize).unwrap()
                        }
                    },
                    |k| {
                        if k < 0 {
                            0.0
                        } else {
                            reference.cdf(k as usize).unwrap()
                        }
                    },
                    0.0,
                    || rows.next().unwrap()[i] as i64,
                );
            }
        }};
    }
    check!(
        "Multinomial",
        Multinomial::with_options(
            12,
            &[s(0.25), s(0.25), s(0.5)],
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap(),
        [
            Binomial::new(12, 0.25).unwrap(),
            Binomial::new(12, 0.25).unwrap(),
            Binomial::new(12, 0.5).unwrap()
        ],
        12
    );
    check!(
        "DirichletMultinomial",
        DirichletMultinomial::with_options(
            12,
            &[s(1.0), s(2.0), s(3.0)],
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap(),
        [
            BetaBinomial::new(12, 1.0, 5.0).unwrap(),
            BetaBinomial::new(12, 2.0, 4.0).unwrap(),
            BetaBinomial::new(12, 3.0, 3.0).unwrap()
        ],
        12
    );
    check!(
        "MultivariateHypergeometric",
        MultivariateHypergeometric::<S>::with_options(
            &[10, 20, 30],
            12,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap(),
        [
            Hypergeometric::<f64>::new(60, 10, 12).unwrap(),
            Hypergeometric::new(60, 20, 12).unwrap(),
            Hypergeometric::new(60, 30, 12).unwrap()
        ],
        12
    );
    check!(
        "RandomTable",
        RandomTable::<S>::with_options(
            &[4, 6],
            &[3, 7],
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap(),
        [
            Hypergeometric::<f64>::new(10, 3, 4).unwrap(),
            Hypergeometric::new(10, 7, 4).unwrap(),
            Hypergeometric::new(10, 3, 6).unwrap(),
            Hypergeometric::new(10, 7, 6).unwrap()
        ],
        7
    );
}
#[test]
fn continuous_marginals_f32() {
    continuous_marginals::<f32>();
}
#[test]
fn continuous_marginals_f64() {
    continuous_marginals::<f64>();
}
#[test]
fn count_marginals_f32() {
    count_marginals::<f32>();
}
#[test]
fn count_marginals_f64() {
    count_marginals::<f64>();
}

/// Haar squared entries have Beta(1/2,(d-1)/2) for O/SO and Beta(1,d-1)
/// for U. Signs and the two possible 2x2 fixed-spectrum correlations are fair.
fn group_marginals<S: RealScalar + NumericScalar<Real = S>>() {
    let mut rng = SplitMix64::from_seed(1335);
    macro_rules! real_group {
        ($law:expr) => {{
            let law = $law;
            let mut signs = [0; 2];
            let values = samples(|| {
                let row = real_row(law.sample(&mut rng).unwrap());
                signs[usize::from(row[0] > 0.0)] += 1;
                row.into_iter().take(3).map(|x| x * x).collect()
            });
            marginals(
                stringify!($law),
                &values,
                &[
                    Box::new(Beta::new(0.5, 1.0).unwrap()),
                    Box::new(Beta::new(0.5, 1.0).unwrap()),
                    Box::new(Beta::new(0.5, 1.0).unwrap()),
                ],
            );
            statistics::assert_fit("Haar real entry/sign", &signs, &[0.5, 0.5], 0.0);
        }};
    }
    real_group!(
        OrthogonalGroup::<S>::with_options(
            3,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
    );
    real_group!(
        SpecialOrthogonalGroup::<S>::with_options(
            3,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
    );
    let unitary = UnitaryGroup::<S>::with_options(
        3,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let values = samples(|| {
        let row = unitary.sample(&mut rng).unwrap();
        let mut values: Vec<_> = row
            .iter()
            .take(3)
            .map(|x| x.re.into_f64().powi(2) + x.im.into_f64().powi(2))
            .collect();
        values.push(row[0].im.into_f64().atan2(row[0].re.into_f64()));
        values
    });
    marginals(
        "UnitaryGroup",
        &values,
        &[
            Box::new(Beta::new(1.0, 2.0).unwrap()),
            Box::new(Beta::new(1.0, 2.0).unwrap()),
            Box::new(Beta::new(1.0, 2.0).unwrap()),
            Box::new(Uniform::new(-core::f64::consts::PI, core::f64::consts::PI).unwrap()),
        ],
    );
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let correlation = RandomCorrelation::with_options(
        &[s(0.5), s(1.5)],
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    let mut counts = [0; 2];
    for _ in 0..statistics::DRAWS {
        let x = correlation.sample(&mut rng).unwrap()[1].into_f64();
        assert!((x.abs() - 0.5).abs() < 64.0 * S::EPSILON.into_f64());
        counts[usize::from(x > 0.0)] += 1;
    }
    statistics::assert_fit("RandomCorrelation/sign", &counts, &[0.5, 0.5], 0.0);
}
#[test]
fn group_marginals_f32() {
    group_marginals::<f32>();
}
#[test]
fn group_marginals_f64() {
    group_marginals::<f64>();
}

/// At the smallest byte limit, in steps of eight bytes, that admits the law
/// built by `law` and one owned sample of it, a batch whose rows alone exceed
/// the limit is refused before any word is drawn.
fn byte_limit_refuses_large_batches<L>(
    law: impl Fn(LawOptions<f64>) -> Result<L, scioxide_core::SciOxideError>,
    sample_n: impl Fn(&L, &mut SplitMix64, usize) -> Result<Vec<f64>, scioxide_core::SciOxideError>,
) {
    let limited = |bytes| law(LawOptions::default().with_max_bytes(bytes));
    let limit = (0..4096)
        .map(|words| 8 * words)
        .find(|&bytes| {
            limited(bytes)
                .is_ok_and(|law| sample_n(&law, &mut SplitMix64::from_seed(2469), 1).is_ok())
        })
        .unwrap();
    let law = limited(limit).unwrap();
    let mut rng = SplitMix64::from_seed(2469);
    let width = sample_n(&law, &mut rng, 1).unwrap().len();
    let position = rng.position();
    let error = sample_n(&law, &mut rng, limit / (8 * width) + 1).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(rng.position(), position);
    assert!(sample_n(&law, &mut rng, 0).unwrap().is_empty());
}

#[test]
fn owned_vector_batches_respect_the_byte_limit() {
    let spread = Matrix::from_vec(2, 2, vec![2.0, 0.5, 0.5, 1.0]).unwrap();
    let rank = MultivariateNormalOptions::default();
    byte_limit_refuses_large_batches(
        |options| MultivariateNormal::with_options(&[1.0, -2.0], &spread, rank, options),
        |law, rng, n| law.sample_n(rng, n),
    );
    byte_limit_refuses_large_batches(
        |options| UniformDirection::with_options(3, options),
        |law, rng, n| law.sample_n(rng, n),
    );
    byte_limit_refuses_large_batches(
        |options| Wishart::with_options(4.5, &spread, options),
        |law, rng, n| law.sample_n(rng, n),
    );
    byte_limit_refuses_large_batches(
        |options| Dirichlet::with_options(&[0.5, 1.5, 2.0], options),
        |law, rng, n| law.sample_n(rng, n),
    );
}

#[test]
fn owned_vector_batches_equal_single_samples_and_overflow_is_checked() {
    let law = MatrixNormal::new(
        &Matrix::from_vec(2, 3, vec![0.0, 1.0, -1.0, 2.0, 0.5, 0.0]).unwrap(),
        &Matrix::identity(2).unwrap(),
        &Matrix::identity(3).unwrap(),
    )
    .unwrap();
    let mut rng = SplitMix64::from_seed(2469);
    let singles: Vec<f64> = (0..7).flat_map(|_| law.sample(&mut rng).unwrap()).collect();
    let mut owned_rng = SplitMix64::from_seed(2469);
    let owned = law.sample_n(&mut owned_rng, 7).unwrap();
    assert_eq!(owned.len(), 7 * law.sample_width());
    assert_eq!(owned, singles);
    assert_eq!(owned_rng.position(), rng.position());
    let error = law.sample_n(&mut owned_rng, usize::MAX).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::SizeOverflow);
    assert_eq!(owned_rng.position(), rng.position());
}

/// Runs `body` inside a parent solve whose evaluation limit is `limit`.
fn in_parent<T>(
    limit: usize,
    body: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    use scioxide_core::{
        Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity,
        SolveScope, SolveState,
    };
    SolveScope::from(ExecutionContext::sequential()).run(
        &mut SolveState::new(),
        SolveIdentity::new(
            ProblemToken::new(1),
            scioxide_core::algorithm_id!("test", "parent", 1),
            0,
            0,
        ),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(limit)?),
        body,
    )
}

#[test]
fn law_rectangles_equal_their_plans_and_joint_cdfs() {
    use scioxide_core::{Budget, SciOxideErrorKind, Tolerance};
    let covariance = Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]).unwrap();
    let tolerance = Tolerance::new(1e-4, 1e-4).unwrap();
    let options = LawOptions::default().with_tolerance(tolerance);
    let normal =
        MultivariateNormal::with_options(&[0.5, -0.25], &covariance, Default::default(), options)
            .unwrap();
    let t = MultivariateStudentT::with_options(
        &[0.5, -0.25],
        &covariance,
        5.0,
        Default::default(),
        options,
    )
    .unwrap();
    // The plan a law prepares for its default work limit of 65,536 evaluations.
    let plan_options = MultivariateProbabilityOptions::default()
        .with_points_per_replicate(8192)
        .with_tolerance(tolerance);
    let root = Philox4x64::from_seed(17);
    let (lower, upper) = ([-1.0, f64::NEG_INFINITY], [1.5, 0.75]);
    for (law, plan) in [
        (
            normal.rectangle(&lower, &upper, &root).unwrap(),
            normal.prepare_probability(plan_options, &root).unwrap(),
        ),
        (
            t.rectangle(&lower, &upper, &root).unwrap(),
            t.prepare_probability(plan_options, &root).unwrap(),
        ),
    ] {
        let report = plan.rectangle(&lower, &upper).unwrap();
        assert_eq!(
            report.termination,
            MultivariateProbabilityTermination::Converged
        );
        assert_eq!(law.to_bits(), report.probability.to_bits());
    }
    let origin = Philox4x64::from_seed(0);
    let corner = [f64::NEG_INFINITY; 2];
    assert_eq!(
        normal
            .rectangle(&corner, &upper, &origin)
            .unwrap()
            .to_bits(),
        normal.cdf(&upper[..]).unwrap().to_bits()
    );
    assert_eq!(
        t.rectangle(&corner, &upper, &origin).unwrap().to_bits(),
        t.cdf(&upper[..]).unwrap().to_bits()
    );
    // Rank one has a closed form; a correlated law with few points misses the tolerance.
    let line = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let degenerate = MultivariateNormal::new(&[0.0; 2], &line).unwrap();
    let half = degenerate.rectangle(&corner, &[0.0; 2], &origin).unwrap();
    assert!((half - 0.5).abs() < 1e-15);
    let coarse = MultivariateNormal::with_options(
        &[0.0; 2],
        &covariance,
        Default::default(),
        LawOptions::default().with_work_budget(Budget::try_from(64).unwrap()),
    )
    .unwrap();
    let error = coarse.rectangle(&corner, &[0.0; 2], &origin).unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::EvaluationLimit,
                ..
            }
        ),
        "{error}"
    );
    assert_eq!(
        normal.rectangle(&[0.0], &upper, &root).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        normal
            .rectangle(&[1.0, 0.0], &[0.0, 1.0], &root)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    let f32_law = MultivariateNormal::<f32>::new(&[0.0; 2], &Matrix::identity(2).unwrap()).unwrap();
    let quadrant = f32_law
        .rectangle(&[0.0; 2], &[f32::INFINITY; 2], &origin)
        .unwrap();
    assert!((quadrant - 0.25).abs() < 1e-6);
}

#[test]
fn scoped_plan_rectangles_equal_owned_and_observe_cancellation() {
    use scioxide_core::{
        BudgetKind, CancellationToken, ExecutionContext, SciOxideErrorKind, Tolerance,
    };
    let covariance = Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]).unwrap();
    let normal = MultivariateNormal::new(&[0.5, -0.25], &covariance).unwrap();
    let plan = normal
        .prepare_probability(
            MultivariateProbabilityOptions::default()
                .with_points_per_replicate(1024)
                .with_tolerance(Tolerance::new(1e-3, 1e-3).unwrap()),
            &Philox4x64::from_seed(17),
        )
        .unwrap();
    let (lower, upper) = ([-1.0, f64::NEG_INFINITY], [1.5, 0.75]);
    let owned = plan.rectangle(&lower, &upper).unwrap();
    let mut work = plan.workspace().unwrap();
    let scoped = plan
        .rectangle_scoped(
            &lower,
            &upper,
            &mut work,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let same = |a: MultivariateProbabilityReport, b: MultivariateProbabilityReport| {
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
        assert_eq!(a.log_probability.to_bits(), b.log_probability.to_bits());
        assert_eq!(
            a.estimated_absolute_error.to_bits(),
            b.estimated_absolute_error.to_bits()
        );
        assert_eq!(
            (a.evaluations, a.termination),
            (b.evaluations, b.termination)
        );
    };
    same(owned, scoped);
    // The plan calls no caller function, so an enclosing solve counts no work.
    let nested = in_parent(1, |parent| {
        let report = plan.rectangle_scoped(&lower, &upper, &mut work, &mut *parent)?;
        assert_eq!(parent.budget_report(BudgetKind::Evaluation).consumed(), 0);
        Ok(report)
    })
    .unwrap();
    same(owned, nested);
    let token = CancellationToken::new();
    token.cancel();
    let error = plan
        .rectangle_scoped(
            &lower,
            &upper,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&token),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
}
