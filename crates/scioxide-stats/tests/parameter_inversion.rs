#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{
    Budget, BudgetKind, ExecutionContext, ExecutionPolicy, ExtendedInterval, Interval, RealScalar,
    SolveBudgets, SolveIdentity, SolveState, StepTolerance,
};
use scioxide_random::{InversionOptions, LawOptions};
use scioxide_stats::*;
#[path = "support/executor.rs"]
mod executor;
fn options<S: RealScalar>() -> InversionOptions<S> {
    let tolerance = if S::EPSILON.into_f64() > 1e-10 {
        2e-6
    } else {
        2e-12
    };
    let root = InversionOptions::<S>::default()
        .root_options()
        .clone()
        .with_step_tolerance(
            StepTolerance::new(
                S::try_from_f64_lossy(tolerance).unwrap(),
                S::try_from_f64_lossy(tolerance).unwrap(),
            )
            .unwrap(),
        );
    InversionOptions::default()
        .with_root_options(root)
        .with_work_budget(Budget::try_from(4096).unwrap())
}
fn value<S: RealScalar>(report: ParameterInversionReport<S>) -> f64 {
    assert!(report.root().is_none_or(|r| r.is_success()));
    match report.solution() {
        ParameterSolution::Unique { value, .. } => value.into_f64(),
        other => panic!("{other:?}"),
    }
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} != {b}");
}
fn request<S: RealScalar>(
    model: ProbabilityParameter<S>,
    lo: f64,
    hi: f64,
    log: f64,
    upper: bool,
) -> ParameterInversionRequest<S> {
    ParameterInversionRequest {
        model,
        bracket: Interval::new(
            S::try_from_f64_lossy(lo).unwrap(),
            S::try_from_f64_lossy(hi).unwrap(),
        )
        .unwrap(),
        target: ParameterProbability::from_log(
            S::try_from_f64_lossy(log).unwrap(),
            if upper {
                scioxide_stats::TailSide::Upper
            } else {
                scioxide_stats::TailSide::Lower
            },
        )
        .unwrap(),
        options: options(),
        tail_options: LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    }
}
fn analytic<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let tol = if S::EPSILON.into_f64() > 1e-10 {
        3e-5
    } else {
        5e-11
    };
    // Exponential (gamma shape=1): sf(x)=exp(-rate*x).
    close(
        value(
            request::<S>(
                ProbabilityParameter::GammaRate {
                    x: s(2.),
                    shape: s(1.),
                },
                0.1,
                5.,
                -6.,
                true,
            )
            .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
            .unwrap(),
        ),
        3.,
        tol,
    );
    close(
        value(
            request::<S>(
                ProbabilityParameter::GammaScale {
                    x: s(2.),
                    shape: s(1.),
                },
                0.1,
                5.,
                -4.,
                true,
            )
            .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
            .unwrap(),
        ),
        0.5,
        tol,
    );
    // Beta(a,1): cdf(x)=x^a; beta(1,b): sf(x)=(1-x)^b.
    close(
        value(
            request::<S>(
                ProbabilityParameter::BetaAlpha {
                    x: s(0.5),
                    beta: s(1.),
                },
                0.1,
                10.,
                3. * 0.5f64.ln(),
                false,
            )
            .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
            .unwrap(),
        ),
        3.,
        tol,
    );
    close(
        value(
            request::<S>(
                ProbabilityParameter::BetaBeta {
                    x: s(0.5),
                    alpha: s(1.),
                },
                0.1,
                10.,
                4. * 0.5f64.ln(),
                true,
            )
            .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
            .unwrap(),
        ),
        4.,
        tol,
    );
    // Zero-count Poisson tail and no-success binomial tail are exact identities.
    close(
        value(
            request::<S>(
                ProbabilityParameter::PoissonMean { count: 0 },
                0.,
                10.,
                -3.,
                false,
            )
            .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
            .unwrap(),
        ),
        3.,
        tol,
    );
    close(
        value(
            request::<S>(
                ProbabilityParameter::BinomialProbability {
                    count: 0,
                    trials: 4,
                },
                0.,
                1.,
                4. * 0.75f64.ln(),
                false,
            )
            .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
            .unwrap(),
        ),
        0.25,
        tol,
    );
    // Extreme logarithm is far below either scalar's smallest positive probability.
    close(
        value(
            request::<S>(
                ProbabilityParameter::GammaRate {
                    x: s(1.),
                    shape: s(1.),
                },
                1.,
                2000.,
                -1000.,
                true,
            )
            .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
            .unwrap(),
        ),
        1000.,
        tol,
    );
    let constant = request::<S>(
        ProbabilityParameter::StudentTDegrees { x: s(0.) },
        1.,
        10.,
        -core::f64::consts::LN_2,
        false,
    )
    .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
    .unwrap();
    assert!(matches!(
        constant.solution(),
        ParameterSolution::Multiple(_)
    ));
    let unknown = request::<S>(
        ProbabilityParameter::FNumeratorDegrees {
            x: s(1.),
            denominator: s(4.),
        },
        1.,
        10.,
        -1.,
        false,
    )
    .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
    .unwrap();
    assert!(matches!(
        unknown.solution(),
        ParameterSolution::Inconclusive
    ));
    assert_eq!(unknown.evaluations(), 0);
}
#[test]
fn analytic_f32() {
    analytic::<f32>();
}
#[test]
fn analytic_f64() {
    analytic::<f64>();
}
#[test]
fn ambiguity_domains_and_partial_termination() {
    let opts = options();
    let bracket = Interval::new(0., 1.).unwrap();
    let target = ParameterProbability::from_log(-1., scioxide_stats::TailSide::Lower).unwrap();
    let r = invert_parameter_scoped(
        |_| panic!("must not probe"),
        bracket,
        target,
        ParameterMonotonicity::Unknown,
        &opts,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert!(matches!(r.solution(), ParameterSolution::Inconclusive));
    assert!(matches!(
        invert_parameter_scoped(
            |_| Ok(-1.),
            bracket,
            target,
            ParameterMonotonicity::Constant,
            &opts,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .unwrap()
        .solution(),
        ParameterSolution::Multiple(_)
    ));
    assert!(matches!(
        invert_parameter_scoped(
            |x| Ok(-4. + x),
            bracket,
            target,
            ParameterMonotonicity::Increasing,
            &opts,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .unwrap()
        .solution(),
        ParameterSolution::NoSolution
    ));
    assert!(
        invert_parameter_scoped(
            |_| Ok(f64::NAN),
            bracket,
            target,
            ParameterMonotonicity::Increasing,
            &opts,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    assert!(
        invert_parameter_scoped(
            |x| Ok(-0.5 - x),
            bracket,
            target,
            ParameterMonotonicity::Increasing,
            &opts,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    let r = invert_parameter_scoped(
        |x| Ok(-1. + x),
        bracket,
        ParameterProbability::from_log(-0.321, scioxide_stats::TailSide::Lower).unwrap(),
        ParameterMonotonicity::Increasing,
        &InversionOptions::default()
            .with_root_options(opts.root_options().clone().with_max_iterations(Budget::ONE))
            .with_work_budget(Budget::try_from(20).unwrap()),
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert!(!r.root().unwrap().is_success());
    assert!(matches!(
        r.solution(),
        ParameterSolution::Unique {
            log_residual: None,
            ..
        }
    ));
    assert!(ParameterProbability::<f64>::new(1., TailSide::Lower).is_err());
    assert!(
        request(
            ProbabilityParameter::GammaRate { x: 1., shape: 1. },
            0.,
            10.,
            -1.,
            true
        )
        .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
        .is_err()
    );
}
#[test]
fn exact_budget_and_provider_failure_accounting() {
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        scioxide_core::ProblemToken::new(7),
        scioxide_core::algorithm_id!("test", "parameter", 1),
        1,
        0,
    );
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(3).unwrap());
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let calls = std::cell::Cell::new(0);
    let error = invert_parameter_scoped(
        |x| {
            calls.set(calls.get() + 1);
            Ok(-1. + x)
        },
        Interval::new(0., 1.).unwrap(),
        ParameterProbability::from_log(-0.321, scioxide_stats::TailSide::Lower).unwrap(),
        ParameterMonotonicity::Increasing,
        &options(),
        &mut context,
    );
    let stopped = error.unwrap();
    assert!(!stopped.root().unwrap().is_success());
    assert_eq!(calls.get(), 3);
    assert_eq!(
        context.counts().total().unwrap().total_providers().unwrap(),
        3
    );
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();

    let failure = scioxide_core::SciOxideError::invalid_input(
        scioxide_core::argument!("sentinel"),
        scioxide_core::InvalidInputReason::Inconsistent,
        "provider failed",
    );
    let expected = failure.to_string();
    let result = invert_parameter_scoped(
        |_| {
            Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("sentinel"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "provider failed",
            ))
        },
        Interval::new(0., 1.).unwrap(),
        ParameterProbability::from_log(-1., scioxide_stats::TailSide::Lower).unwrap(),
        ParameterMonotonicity::Increasing,
        &options(),
        &mut context,
    )
    .unwrap_err();
    assert_eq!(result.to_string(), expected);
    assert_eq!(
        context.counts().total().unwrap().total_providers().unwrap(),
        1
    );
}
#[test]
fn integer_thresholds_full_width_and_law_adapters() {
    let r = invert_integer_parameter_scoped(
        |n| Ok(if n < 17 { -2. } else { -0.5 }),
        (0, usize::MAX),
        ParameterProbability::from_log(-1., scioxide_stats::TailSide::Lower).unwrap(),
        ParameterMonotonicity::Increasing,
        &options(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(
        r.solution,
        IntegerParameterSolution::Threshold {
            first: 17,
            previous: Some(16)
        }
    );
    let r = IntegerProbabilityParameter::BinomialTrials {
        count: 0,
        probability: 0.5,
    }
    .solve_scoped(
        (0, 20),
        ParameterProbability::new(0.2, scioxide_stats::TailSide::Lower).unwrap(),
        &options(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(
        r.solution,
        IntegerParameterSolution::Threshold {
            first: 3,
            previous: Some(2)
        }
    );
    let r = IntegerProbabilityParameter::NegativeBinomialShape {
        count: 0,
        probability: 0.5,
    }
    .solve_scoped(
        (1, 20),
        ParameterProbability::new(0.2, scioxide_stats::TailSide::Lower).unwrap(),
        &options(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(
        r.solution,
        IntegerParameterSolution::Threshold {
            first: 3,
            previous: Some(2)
        }
    );
    assert!(
        IntegerProbabilityParameter::NegativeBinomialShape {
            count: 0,
            probability: 0.5f32
        }
        .solve_scoped(
            (1, usize::MAX),
            ParameterProbability::new(0.2, scioxide_stats::TailSide::Lower).unwrap(),
            &options(),
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
}
#[test]
fn batches_order_executor_parity_and_rollback() {
    let requests: [ParameterInversionRequest<f64>; 2] = [
        request(
            ProbabilityParameter::PoissonMean { count: 0 },
            0.,
            10.,
            -1.,
            false,
        ),
        request(
            ProbabilityParameter::PoissonMean { count: 0 },
            0.,
            10.,
            -3.,
            false,
        ),
    ];
    let mut staging = [None; 2];
    let mut output = [None; 2];
    invert_parameters_into(
        &requests,
        ExecutionPolicy::Sequential,
        &mut output,
        &mut staging,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let baseline = output.map(|r| value(r.unwrap()));
    let executor = executor::PartitionExecutor(std::num::NonZeroUsize::new(2).unwrap());
    invert_parameters_into(
        &requests,
        ExecutionPolicy::Auto,
        &mut output,
        &mut staging,
        ExecutionContext::new(&executor),
    )
    .unwrap();
    assert_eq!(baseline, output.map(|r| value(r.unwrap())));
    let mut invalid = requests.clone();
    invalid[1].bracket = Interval::new(-1., 3.).unwrap();
    assert!(
        invert_parameters_into(
            &invalid,
            ExecutionPolicy::Auto,
            &mut output,
            &mut staging,
            ExecutionContext::new(&executor)
        )
        .is_err()
    );
    assert_eq!(baseline, output.map(|r| value(r.unwrap())));
}
#[test]
fn exact_limits_one_sided_boundaries_and_missing_brackets() {
    let confidence = ConfidenceProbability::new(0.95).unwrap();
    let opts = options();
    let zero = binomial_confidence_limits_scoped::<f64>(
        0,
        10,
        confidence,
        TestAlternative::TwoSided,
        &opts,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    let (a, b) = zero.set().bounds().unwrap();
    assert_eq!(a, Some(0.));
    close(b.unwrap(), 1. - 0.025f64.powf(0.1), 5e-11);
    let all = binomial_confidence_limits_scoped::<f64>(
        10,
        10,
        confidence,
        TestAlternative::TwoSided,
        &opts,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    close(
        all.set().bounds().unwrap().0.unwrap(),
        0.025f64.powf(0.1),
        5e-11,
    );
    assert_eq!(all.set().bounds().unwrap().1, Some(1.));
    let poisson = poisson_rate_confidence_limits_scoped(
        0,
        2.,
        Interval::new(0., 10.).unwrap(),
        confidence,
        TestAlternative::Less,
        &opts,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    close(
        poisson.set().bounds().unwrap().1.unwrap(),
        -0.05f64.ln() / 2.,
        5e-11,
    );
    assert!(
        poisson_rate_confidence_limits_scoped(
            0,
            2.,
            Interval::new(0., 0.1).unwrap(),
            confidence,
            TestAlternative::Less,
            &opts,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    let lower = poisson_rate_confidence_limits_scoped(
        3,
        2.,
        Interval::new(0., 10.).unwrap(),
        confidence,
        TestAlternative::Greater,
        &opts,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert!(lower.set().bounds().unwrap().1.is_none());
    assert!(lower.evaluations() <= opts.work_budget().get());
    let empty = noncentrality_confidence_limits_scoped(
        ProbabilityParameter::ChiSquaredNoncentrality {
            x: 0.001,
            degrees: 5.,
        },
        Interval::new(0., 30.).unwrap(),
        confidence,
        TestAlternative::TwoSided,
        &opts,
        LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert!(empty.set().is_empty());
    assert!(
        normal_variance_confidence_limits_scoped(
            0.,
            5.,
            Interval::new(0.01, 10.).unwrap(),
            confidence,
            TestAlternative::TwoSided,
            &opts,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                scioxide_core::ProblemToken::new(1),
                scioxide_core::algorithm_id!("test", "confidence", 1),
                1,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let generic = invert_confidence_limits_scoped(
        |p| {
            Ok(if p == 1. {
                f64::NEG_INFINITY
            } else {
                (-p).ln_1p()
            })
        },
        ExtendedInterval::new(Some(0.), Some(1.)).unwrap(),
        Interval::new(0., 1.).unwrap(),
        confidence,
        TestAlternative::TwoSided,
        ParameterMonotonicity::Decreasing,
        &opts,
        &mut context,
    )
    .unwrap();
    close(generic.set().bounds().unwrap().0.unwrap(), 0.025, 5e-11);
    close(generic.set().bounds().unwrap().1.unwrap(), 0.975, 5e-11);
}

fn independent_references<S: RealScalar>() {
    let s = |v: f64| S::try_from_f64_lossy(v).unwrap();
    let width32 = S::EPSILON.into_f64() > 1e-10;
    assert_eq!(
        include_str!("data/parameter_inversion.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .count(),
        20,
        "all independent reference cases must be active"
    );
    for line in include_str!("data/parameter_inversion.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let c: Vec<_> = line.split('\t').collect();
        let x = c[1].parse::<f64>().unwrap();
        let fixed = c[2].parse::<f64>().unwrap();
        let fixed2 = c[3].parse::<f64>().unwrap();
        let lo = c[4].parse::<f64>().unwrap();
        let hi = c[5].parse::<f64>().unwrap();
        let log = c[6].parse::<f64>().unwrap();
        let upper = c[7] == "1";
        let expected = c[8].parse::<f64>().unwrap();
        assert_eq!(
            (expected as f32).to_bits(),
            u32::from_str_radix(c[9], 16).unwrap()
        );
        assert_eq!(expected.to_bits(), u64::from_str_radix(c[10], 16).unwrap());
        let opts = options::<S>();
        let resources = LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215));
        let confidence = ConfidenceProbability::new(0.95).unwrap();
        let bounds = Interval::new(s(lo), s(hi)).unwrap();
        let actual = match c[0] {
            "binomial_lower" | "binomial_upper" => {
                let result = binomial_confidence_limits_scoped::<S>(
                    x as usize,
                    fixed as usize,
                    confidence,
                    TestAlternative::TwoSided,
                    &opts,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
                let (a, b) = result.set().bounds().unwrap();
                if upper {
                    b.unwrap().into_f64()
                } else {
                    a.unwrap().into_f64()
                }
            }
            "poisson_lower" | "poisson_upper" => {
                let result = poisson_rate_confidence_limits_scoped(
                    x as usize,
                    s(fixed),
                    bounds,
                    confidence,
                    TestAlternative::TwoSided,
                    &opts,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
                let (a, b) = result.set().bounds().unwrap();
                if upper {
                    b.unwrap().into_f64()
                } else {
                    a.unwrap().into_f64()
                }
            }
            "variance_lower" | "variance_upper" => {
                let result = normal_variance_confidence_limits_scoped(
                    s(x),
                    s(fixed),
                    bounds,
                    confidence,
                    TestAlternative::TwoSided,
                    &opts,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
                let (a, b) = result.set().bounds().unwrap();
                if upper {
                    b.unwrap().into_f64()
                } else {
                    a.unwrap().into_f64()
                }
            }
            "nc_t_lower" | "nc_t_upper" | "nc_chi_lower" | "nc_chi_upper" | "nc_f_lower"
            | "nc_f_upper" => {
                let result = noncentrality_confidence_limits_scoped(
                    if c[0].starts_with("nc_chi") {
                        ProbabilityParameter::ChiSquaredNoncentrality {
                            x: s(x),
                            degrees: s(fixed),
                        }
                    } else if c[0].starts_with("nc_f") {
                        ProbabilityParameter::FNoncentrality {
                            x: s(x),
                            numerator: s(fixed),
                            denominator: s(fixed2),
                        }
                    } else {
                        ProbabilityParameter::StudentTNoncentrality {
                            x: s(x),
                            degrees: s(fixed),
                        }
                    },
                    bounds,
                    confidence,
                    TestAlternative::TwoSided,
                    &opts,
                    resources,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
                let (a, b) = result.set().bounds().unwrap();
                if c[0].ends_with("upper") {
                    b.unwrap().into_f64()
                } else {
                    a.unwrap().into_f64()
                }
            }
            name => {
                let model = match name {
                    "gamma_shape" => ProbabilityParameter::GammaShape {
                        x: s(x),
                        scale: s(fixed),
                    },
                    "chi_degrees" => ProbabilityParameter::ChiSquaredDegrees { x: s(x) },
                    "t_degrees" => ProbabilityParameter::StudentTDegrees { x: s(x) },
                    "f_numerator" => ProbabilityParameter::FNumeratorDegrees {
                        x: s(x),
                        denominator: s(fixed),
                    },
                    "f_denominator" => ProbabilityParameter::FDenominatorDegrees {
                        x: s(x),
                        numerator: s(fixed),
                    },
                    "nc_chi" => ProbabilityParameter::ChiSquaredNoncentrality {
                        x: s(x),
                        degrees: s(fixed),
                    },
                    "nc_t" => ProbabilityParameter::StudentTNoncentrality {
                        x: s(x),
                        degrees: s(fixed),
                    },
                    "nc_f" => ProbabilityParameter::FNoncentrality {
                        x: s(x),
                        numerator: s(fixed),
                        denominator: s(fixed2),
                    },
                    _ => panic!("unknown reference {name}"),
                };
                let result = ParameterInversionRequest {
                    model,
                    bracket: bounds,
                    target: ParameterProbability::from_log(
                        s(log),
                        if upper {
                            scioxide_stats::TailSide::Upper
                        } else {
                            scioxide_stats::TailSide::Lower
                        },
                    )
                    .unwrap(),
                    options: opts,
                    tail_options: resources,
                }
                .solve_scoped(scioxide_core::ExecutionContext::sequential_native())
                .unwrap();
                assert!(result.evaluations() <= 4096);
                if let ParameterSolution::Unique {
                    log_residual: Some(residual),
                    ..
                } = result.solution()
                {
                    let tol = if width32 {
                        2e-4
                    } else if name.starts_with("nc_") {
                        2e-8
                    } else {
                        5e-10
                    };
                    assert!(
                        residual.into_f64().abs() < tol,
                        "{name}: residual {residual:?}"
                    );
                } else {
                    panic!("missing verified residual");
                }
                value(result)
            }
        };
        let tolerance = if width32 {
            5e-5
        } else if c[0].starts_with("nc_") {
            5e-8
        } else {
            5e-11
        };
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.),
            "{}: {actual} != {expected} (bound {tolerance})",
            c[0]
        );
    }
}
#[test]
fn multiprecision_reference_f32() {
    independent_references::<f32>();
}
#[test]
fn multiprecision_reference_f64() {
    independent_references::<f64>();
}

#[test]
fn cancellation_and_integer_batches() {
    let cancellation = scioxide_core::CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            SolveIdentity::new(
                scioxide_core::ProblemToken::new(1),
                scioxide_core::algorithm_id!("test", "cancel", 1),
                1,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let calls = std::cell::Cell::new(0);
    let result = invert_parameter_scoped(
        |_| {
            calls.set(calls.get() + 1);
            Ok(-1.)
        },
        Interval::new(0., 1.).unwrap(),
        ParameterProbability::from_log(-1., scioxide_stats::TailSide::Lower).unwrap(),
        ParameterMonotonicity::Increasing,
        &options(),
        &mut context,
    );
    assert!(result.is_err());
    assert_eq!(calls.get(), 0);
    let request = IntegerParameterInversionRequest {
        model: IntegerProbabilityParameter::BinomialTrials {
            count: 0,
            probability: 0.5,
        },
        domain: (0, 20),
        target: ParameterProbability::new(0.2, scioxide_stats::TailSide::Lower).unwrap(),
        options: options(),
    };
    let mut staging = [None; 2];
    let mut output = [None; 2];
    let requests = [request.clone(), request];
    let executor = executor::PartitionExecutor(std::num::NonZeroUsize::new(2).unwrap());
    invert_integer_parameters_into(
        &requests,
        ExecutionPolicy::Auto,
        &mut output,
        &mut staging,
        ExecutionContext::new(&executor),
    )
    .unwrap();
    assert!(output.iter().all(|r| r.unwrap().solution
        == IntegerParameterSolution::Threshold {
            first: 3,
            previous: Some(2)
        }));
    let baseline = output.map(|r| r.unwrap().solution);
    let mut bad = requests;
    bad[1].domain = (3, 1);
    assert!(
        invert_integer_parameters_into(
            &bad,
            ExecutionPolicy::Auto,
            &mut output,
            &mut staging,
            ExecutionContext::new(&executor)
        )
        .is_err()
    );
    assert_eq!(baseline, output.map(|r| r.unwrap().solution));
}
#[test]
fn exact_binomial_coverage_and_joint_endpoint_budget() {
    // Enumerate exact binomial probabilities independently with integer combinations.
    let n = 12;
    let confidence = ConfidenceProbability::new(0.95).unwrap();
    let opts = options();
    for p in [0.01_f64, 0.2, 0.5, 0.8, 0.99] {
        let mut coverage = 0.;
        let mut combination = 1_u64;
        for k in 0..=n {
            let limits = binomial_confidence_limits_scoped::<f64>(
                k,
                n,
                confidence,
                TestAlternative::TwoSided,
                &opts,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
            let mass = combination as f64 * p.powi(k as i32) * (1. - p).powi((n - k) as i32);
            if limits.set().contains(p) {
                coverage += mass;
            }
            if k < n {
                combination = combination * (n - k) as u64 / (k + 1) as u64;
            }
        }
        assert!(coverage >= 0.95 - 1e-12, "coverage {coverage}");
    }
    let result = binomial_confidence_limits_scoped::<f64>(
        3,
        10,
        confidence,
        TestAlternative::TwoSided,
        &opts,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    let small = InversionOptions::default()
        .with_root_options(opts.root_options().clone())
        .with_work_budget(Budget::try_from(result.evaluations() - 1).unwrap());
    assert!(
        binomial_confidence_limits_scoped::<f64>(
            3,
            10,
            confidence,
            TestAlternative::TwoSided,
            &small,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .is_err()
    );
    let exact = InversionOptions::default()
        .with_root_options(opts.root_options().clone())
        .with_work_budget(Budget::try_from(result.evaluations()).unwrap());
    assert_eq!(
        binomial_confidence_limits_scoped::<f64>(
            3,
            10,
            confidence,
            TestAlternative::TwoSided,
            &exact,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .unwrap()
        .evaluations(),
        result.evaluations()
    );
}

#[test]
fn secant_conditioning_and_exact_integer_shape_boundary() {
    let report = invert_parameter_scoped(
        |x| Ok(-x),
        Interval::new(0., 1.).unwrap(),
        ParameterProbability::from_log(-0.314159, scioxide_stats::TailSide::Lower).unwrap(),
        ParameterMonotonicity::Decreasing,
        &options(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    if let ParameterSolution::Unique {
        log_tail_slope: Some(slope),
        log_residual: Some(residual),
        ..
    } = report.solution()
    {
        close(slope, -1., 1e-14);
        assert!(residual.abs() < 1e-11);
    } else {
        panic!("expected conditioning evidence");
    }
    #[cfg(target_pointer_width = "64")]
    {
        let model = IntegerProbabilityParameter::NegativeBinomialShape {
            count: 0,
            probability: 0.5_f64,
        };
        assert!(
            model
                .solve_scoped(
                    (1, (1usize << 53) + 1),
                    ParameterProbability::new(0.2, scioxide_stats::TailSide::Lower).unwrap(),
                    &options(),
                    scioxide_core::ExecutionContext::sequential_native()
                )
                .is_err()
        );
    }
}

#[cfg(feature = "std")]
#[test]
fn actual_thread_pool_preserves_parameter_order() {
    let requests: [ParameterInversionRequest<f64>; 2] = [
        request(
            ProbabilityParameter::PoissonMean { count: 0 },
            0.,
            10.,
            -1.,
            false,
        ),
        request(
            ProbabilityParameter::PoissonMean { count: 0 },
            0.,
            10.,
            -3.,
            false,
        ),
    ];
    let executor = scioxide_core::ThreadExecutor::new(std::num::NonZeroUsize::new(2).unwrap());
    let mut staging = [None; 2];
    let mut output = [None; 2];
    invert_parameters_into(
        &requests,
        ExecutionPolicy::Auto,
        &mut output,
        &mut staging,
        ExecutionContext::new(&executor),
    )
    .unwrap();
    close(value(output[0].unwrap()), 1., 5e-11);
    close(value(output[1].unwrap()), 3., 5e-11);
}

#[test]
fn degenerate_integer_tail_inequality_orientation() {
    let binomial = IntegerProbabilityParameter::BinomialTrials {
        count: 0,
        probability: 0.0_f64,
    };
    let negative = IntegerProbabilityParameter::NegativeBinomialShape {
        count: 0,
        probability: 1.0_f64,
    };
    for model in [binomial, negative] {
        for upper in [false, true] {
            let result = model
                .solve_scoped(
                    (1, 20),
                    ParameterProbability::new(
                        0.2,
                        if upper {
                            scioxide_stats::TailSide::Upper
                        } else {
                            scioxide_stats::TailSide::Lower
                        },
                    )
                    .unwrap(),
                    &options(),
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
            // Constant CDF=1 cannot be <=0.2; constant survival=0 cannot be >=0.2.
            assert_eq!(result.solution, IntegerParameterSolution::None);
        }
    }
}
