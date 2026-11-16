#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
use scioxide_core::{
    Budget, BudgetKind, Complex, ExecutionContext, ExecutionPolicy, ProblemToken, SolveBudgets,
    SolveIdentity, SolveState,
};
use scioxide_rootfind::*;

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(682),
        scioxide_core::algorithm_id!("zero-contract", "test", 1),
        0,
        0,
    )
}
/// `J_{1/2}`, whose positive zeros are `kπ`.
const HALF: SpecialZeroFamily<f64> = SpecialZeroFamily::BesselJ(0.5);
fn workspace(count: usize) -> SpecialZeroWorkspace<f64> {
    SpecialZeroWorkspace::workspace(count, &SpecialZeroOptions::default()).unwrap()
}
#[test]
fn reserved_execution_and_ancestor_budget() {
    let options = SpecialZeroOptions::default();
    let mut workspace = workspace(5);
    let mut output = [None; 5];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let report = special_zeros_into(
        HALF,
        SpecialZeroBranch::Real,
        1,
        &options,
        &mut output,
        &mut workspace,
        &mut context,
    )
    .unwrap();
    assert_eq!(report.termination(), RootTermination::Complete);
    assert_eq!(
        report.accounting().total().unwrap(),
        context.counts().total().unwrap()
    );
    for (index, zero) in output.iter().enumerate() {
        assert!(
            (zero.unwrap().point().re - (index + 1) as f64 * core::f64::consts::PI).abs() < 1e-12
        );
    }
    let saved = output.map(|z| z.unwrap().point());
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    let report = special_zeros_into(
        HALF,
        SpecialZeroBranch::Real,
        1,
        &options,
        &mut output,
        &mut workspace,
        &mut context,
    )
    .unwrap();
    assert_eq!(report.termination(), RootTermination::Complete);
    assert_eq!(report.completed(), 5);
    assert_eq!(context.counts().own().iterations(), 0);
    assert!(context.counts().nested().iterations() > 1);
    assert_eq!(output.map(|z| z.unwrap().point()), saved);

    // A shared evaluation allowance still stops nested refinement before output
    // publication; this replaces the former aggregate iteration control.
    let limited = SpecialZeroOptions::default().with_evaluation_budget(Budget::ONE);
    let report = special_zeros_into(
        HALF,
        SpecialZeroBranch::Real,
        1,
        &limited,
        &mut output,
        &mut workspace,
        &mut context,
    )
    .unwrap();
    assert_eq!(
        report.termination(),
        RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
    );
    assert_eq!(report.completed(), 0);
    assert!(
        report
            .accounting()
            .total()
            .unwrap()
            .total_providers()
            .unwrap()
            <= 1
    );
    assert_eq!(output.map(|z| z.unwrap().point()), saved);
}
#[test]
fn a_workspace_reserved_from_requirements_serves_without_allocation() {
    let options = SpecialZeroOptions::<f64>::default();
    let requirements = SpecialZeroWorkspace::workspace_requirements(8, &options).unwrap();
    assert_eq!(requirements.capacity(), 8);
    let mut workspace = SpecialZeroWorkspace::new();
    workspace.try_reserve(&requirements).unwrap();
    for count in [8, 3] {
        let mut output = vec![None; count];
        let report = special_zeros_into(
            HALF,
            SpecialZeroBranch::Real,
            1,
            &options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(report.termination(), RootTermination::Complete);
        assert_eq!(report.completed(), count);
        for (index, zero) in output.iter().enumerate() {
            let expected = (index + 1) as f64 * core::f64::consts::PI;
            assert!((zero.unwrap().point().re - expected).abs() < 1e-12);
        }
    }
    // The byte limit bounds the size of the request, not the family or index.
    let bytes = requirements.required_bytes();
    let exact = options.clone().with_workspace_byte_limit(bytes);
    assert!(SpecialZeroWorkspace::workspace_requirements(8, &exact).is_ok());
    let short = options.with_workspace_byte_limit(bytes - 1);
    assert_eq!(
        SpecialZeroWorkspace::workspace_requirements(8, &short)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
}
#[test]
fn validation_and_empty_sequences() {
    use SpecialZeroBranch as B;
    use SpecialZeroFamily as F;
    for (family, branch, start) in [
        (F::BesselJ(-0.5), B::Real, 1),
        (F::BesselY(f64::NAN), B::Real, 1),
        (F::BesselJPrime(-0.1), B::Real, 1),
        (F::AiryAi, B::UPPER, 1),
        (F::Erf, B::Real, 0),
        (F::BesselY(2.0), B::UPPER, 1),
        (F::AiryAi, B::Real, usize::MAX),
    ] {
        assert!(
            special_zeros_with_options(family, branch, start, 2, &SpecialZeroOptions::default())
                .is_err()
        );
    }
    assert!(
        special_zeros_with_options(
            SpecialZeroFamily::BesselJ(0.5),
            SpecialZeroBranch::Real,
            1,
            usize::MAX,
            &SpecialZeroOptions::default()
        )
        .is_err()
    );
    assert!(
        SpecialZeroWorkspace::<f64>::workspace(
            2,
            &SpecialZeroOptions::default().with_workspace_byte_limit(1)
        )
        .is_err()
    );
    let options = SpecialZeroOptions::default();
    let empty = special_zeros_with_options(HALF, SpecialZeroBranch::Real, 1, 0, &options).unwrap();
    assert_eq!(empty.report().completed(), 0);
    assert_eq!(
        empty.report().counts(),
        scioxide_core::EvaluationCounts::new()
    );
    // The output length is the count: an output longer than the workspace is rejected.
    let mut out = [None; 3];
    let mut w = workspace(2);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut c = state
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    assert!(
        special_zeros_into(
            HALF,
            SpecialZeroBranch::Real,
            1,
            &options,
            &mut out,
            &mut w,
            &mut c
        )
        .is_err()
    );
    assert!(out.iter().all(Option::is_none));
    assert_eq!(c.counts().own(), scioxide_core::EvaluationCounts::new());
}
#[test]
fn explicit_origins_and_proved_empty_sequences() {
    use SpecialZeroFamily as F;
    for (f, multiplicity) in [
        (F::Erf, 1),
        (F::FresnelC, 1),
        (F::FresnelS, 3),
        (F::FresnelSPrime, 2),
        (F::BesselJPrime(0.0), 1),
    ] {
        let r = special_zeros_with_options(
            f,
            SpecialZeroBranch::Origin,
            0,
            1,
            &SpecialZeroOptions::default(),
        )
        .unwrap();
        assert_eq!(r.roots()[0].multiplicity(), multiplicity);
        assert_eq!(r.roots()[0].point(), Complex::new(0.0, 0.0));
    }
    for f in [F::Erf, F::FresnelC, F::FresnelS] {
        let r = special_zeros_with_options(
            f,
            SpecialZeroBranch::Real,
            1,
            3,
            &SpecialZeroOptions::<f64>::default(),
        )
        .unwrap();
        assert_eq!(r.report().termination(), RootTermination::NoZeros);
        assert!(r.roots().is_empty());
    }
}
#[test]
fn interlacing_and_projected_symmetries() {
    use SpecialZeroFamily as F;
    let roots = |f| {
        special_zeros_with_options(
            f,
            SpecialZeroBranch::Real,
            1,
            8,
            &SpecialZeroOptions::default(),
        )
        .unwrap()
    };
    let j = roots(F::BesselJ(2.5));
    let d = roots(F::BesselJPrime(2.5));
    assert_eq!(j.roots().len(), 8, "{:?}", j.report());
    assert_eq!(d.roots().len(), 8, "{:?}", d.report());
    for i in 0..7 {
        assert!(d.roots()[i].point().re < j.roots()[i].point().re);
        assert!(j.roots()[i].point().re < d.roots()[i + 1].point().re);
    }
    for f in [F::Erf, F::FresnelC, F::FresnelS] {
        let base = special_zeros_with_options(
            f,
            SpecialZeroBranch::UPPER,
            1,
            3,
            &SpecialZeroOptions::<f32>::default(),
        )
        .unwrap();
        assert_eq!(base.report().completed(), 3);
        for conjugate in [false, true] {
            for negate in [false, true] {
                for rotate in [false, true] {
                    if rotate && matches!(f, F::Erf) {
                        continue;
                    }
                    let r = special_zeros_with_options(
                        f,
                        SpecialZeroBranch::Complex {
                            conjugate,
                            negate,
                            rotate,
                        },
                        1,
                        3,
                        &SpecialZeroOptions::default(),
                    )
                    .unwrap();
                    assert_eq!(r.report().completed(), 3, "{f:?}: {:?}", r.report());
                    for (a, b) in base.roots().iter().zip(r.roots()) {
                        let z = if conjugate {
                            a.point().conj()
                        } else {
                            a.point()
                        };
                        let z = if negate { -z } else { z };
                        let z = if rotate { Complex::new(-z.im, z.re) } else { z };
                        assert_eq!(z, b.point());
                        assert!(b.estimated_error().is_finite());
                    }
                }
            }
        }
    }
}
#[cfg(feature = "std")]
#[test]
fn executor_agrees_with_sequential() {
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(3).unwrap());
    for (family, branch) in [
        (SpecialZeroFamily::Erf, SpecialZeroBranch::UPPER),
        (SpecialZeroFamily::BesselJ(0.0), SpecialZeroBranch::Real),
        (SpecialZeroFamily::AiryBiPrime, SpecialZeroBranch::Real),
    ] {
        let parallel = SpecialZeroOptions::<f64>::default().with_execution_policy(
            ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(3).unwrap()),
        );
        let reference =
            special_zeros_with_options(family, branch, 1, 8, &SpecialZeroOptions::<f64>::default())
                .unwrap();
        let mut output = [None; 8];
        let mut workspace = workspace(8);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::new(&pool), identity(), &mut budgets)
            .unwrap();
        let report = special_zeros_into(
            family,
            branch,
            1,
            &parallel,
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap();
        assert_eq!(report.termination(), RootTermination::Complete);
        assert_eq!(report.counts(), reference.report().counts());
        for (a, b) in output.iter().zip(reference.roots()) {
            assert_eq!(a.unwrap().point(), b.point());
        }
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::new(&pool), identity(), &mut budgets)
            .unwrap();
        let mut repeated = [None; 8];
        let repeated_report = special_zeros_into(
            family,
            branch,
            1,
            &parallel,
            &mut repeated,
            &mut workspace,
            &mut context,
        )
        .unwrap();
        assert_eq!(repeated_report.counts(), report.counts());
        let owned = special_zeros_with_options(family, branch, 1, 8, &parallel).unwrap();
        assert_eq!(owned.report().counts(), report.counts());
        assert_eq!(
            owned
                .roots()
                .iter()
                .map(SpecialZero::point)
                .collect::<Vec<_>>(),
            reference
                .roots()
                .iter()
                .map(SpecialZero::point)
                .collect::<Vec<_>>()
        );
    }
}
struct BrokenExecutor(bool);
impl scioxide_core::Executor for BrokenExecutor {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::new(3).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<(), scioxide_core::SciOxideError> {
        if self.0 {
            scioxide_core::SequentialExecutor.execute(context, plan, operation)?;
            scioxide_core::SequentialExecutor.execute(context, plan, operation)
        } else {
            scioxide_core::SequentialExecutor.execute(context, plan, &|_, (), _| Ok(()))
        }
    }
}
#[test]
fn cancellation_and_malformed_executor_preserve_output() {
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    let options = SpecialZeroOptions::default();
    let mut output = [None; 3];
    let mut w = workspace(3);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut c = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&token),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let r = special_zeros_into(
        HALF,
        SpecialZeroBranch::Real,
        1,
        &options,
        &mut output,
        &mut w,
        &mut c,
    )
    .unwrap();
    assert_eq!(r.termination(), RootTermination::Cancelled);
    assert!(output.iter().all(Option::is_none));
    assert_eq!(c.counts().own(), scioxide_core::EvaluationCounts::new());
    for duplicate in [false, true] {
        let executor = BrokenExecutor(duplicate);
        let parallel = SpecialZeroOptions::default().with_execution_policy(
            ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(3).unwrap()),
        );
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut c = state
            .begin_solve(ExecutionContext::new(&executor), identity(), &mut budgets)
            .unwrap();
        assert!(
            special_zeros_into(
                HALF,
                SpecialZeroBranch::Real,
                1,
                &parallel,
                &mut output,
                &mut w,
                &mut c
            )
            .is_err()
        );
        assert!(output.iter().all(Option::is_none));
    }
    let r = special_zeros_with_options(HALF, SpecialZeroBranch::Real, 1, 3, &options).unwrap();
    assert_eq!(r.report().completed(), 3);
}
struct TestClock(bool);
impl scioxide_core::Clock for TestClock {
    fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
        if self.0 {
            Err(scioxide_core::CapabilityError::new(
                "injected zero clock failure",
            ))
        } else {
            Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_secs(1),
            ))
        }
    }
}
#[test]
fn deadline_and_clock_failure_are_distinct() {
    let options = SpecialZeroOptions::default();
    let mut w = workspace(2);
    for failure in [false, true] {
        let clock = TestClock(failure);
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut output = [None; 2];
        let deadline = scioxide_core::Deadline::at(
            &clock_binding,
            scioxide_core::MonotonicInstant::from_duration_since_epoch(core::time::Duration::ZERO),
        );
        let mut c = state
            .begin_solve(
                ExecutionContext::sequential()
                    .with_deadline(deadline)
                    .unwrap(),
                identity(),
                &mut budgets,
            )
            .unwrap();
        let result = special_zeros_into(
            HALF,
            SpecialZeroBranch::Real,
            1,
            &options,
            &mut output,
            &mut w,
            &mut c,
        );
        if failure {
            assert_eq!(
                result.unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::CapabilityFailure
            );
        } else {
            assert_eq!(result.unwrap().termination(), RootTermination::TimeLimit);
        }
        assert!(output.iter().all(Option::is_none));
        assert_eq!(c.counts().own(), scioxide_core::EvaluationCounts::new());
    }
}
#[test]
fn ordinary_stop_publishes_only_the_verified_prefix() {
    let complete = special_zeros_with_options(
        SpecialZeroFamily::BesselJ(32.0_f64),
        SpecialZeroBranch::Real,
        1,
        5,
        &SpecialZeroOptions::default(),
    )
    .unwrap();
    let sentinel = Some(complete.roots()[0]);
    let mut witnessed = false;
    for work in [5000, 10000, 20000, 40000, 80000, 160000] {
        let options =
            SpecialZeroOptions::default().with_work_per_zero(Budget::try_from(work).unwrap());
        let mut w = workspace(5);
        let mut output = [sentinel; 5];
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut c = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let r = special_zeros_into(
            SpecialZeroFamily::BesselJ(32.0),
            SpecialZeroBranch::Real,
            1,
            &options,
            &mut output,
            &mut w,
            &mut c,
        )
        .unwrap();
        if r.completed() > 0 && r.completed() < 5 {
            assert_eq!(
                r.termination(),
                RootTermination::WorkLimit(scioxide_core::BudgetKind::Iteration)
            );
            for (a, b) in output[..r.completed()].iter().zip(complete.roots()) {
                assert_eq!(a.unwrap().point(), b.point());
            }
            assert!(
                output[r.completed()..]
                    .iter()
                    .all(|z| z.unwrap().index() == 1)
            );
            witnessed = true;
            break;
        }
    }
    assert!(
        witnessed,
        "expected a bounded request to expose a verified proper prefix"
    );
}

#[test]
fn projection_cannot_silently_change_the_root_index() {
    let r = special_zeros_with_options(
        SpecialZeroFamily::FresnelCPrime,
        SpecialZeroBranch::Real,
        1_usize << 30,
        1,
        &SpecialZeroOptions::<f32>::default(),
    )
    .unwrap();
    assert_eq!(r.report().termination(), RootTermination::Unverified);
    assert!(r.roots().is_empty());
    assert!(r.report().candidate().is_some());
    let strict = SpecialZeroOptions::<f64>::default()
        .with_tolerance(scioxide_core::StepTolerance::new(1e-30, 0.0).unwrap());
    assert_eq!(
        special_zeros_with_options(
            SpecialZeroFamily::Erf,
            SpecialZeroBranch::UPPER,
            1,
            1,
            &strict
        )
        .unwrap()
        .report()
        .termination(),
        RootTermination::Unverified
    );
}

#[test]
fn indexed_bessel_and_airy_need_at_most_five_paired_evaluations() {
    use SpecialZeroFamily as F;
    for family in [
        F::BesselJ(0.0),
        F::BesselJ(2.5),
        F::BesselJPrime(1.0),
        F::BesselY(0.0),
        F::BesselYPrime(0.0),
        F::AiryAi,
        F::AiryAiPrime,
        F::AiryBi,
        F::AiryBiPrime,
    ] {
        for index in 1..=256 {
            let result = special_zeros_with_options(
                family,
                SpecialZeroBranch::Real,
                index,
                1,
                &SpecialZeroOptions::<f64>::default(),
            )
            .unwrap();
            assert_eq!(
                result.report().termination(),
                RootTermination::Complete,
                "{family:?} {index}"
            );
            let evaluations = result
                .report()
                .accounting()
                .total()
                .unwrap()
                .total_providers()
                .unwrap();
            assert!(evaluations <= 5, "{family:?} {index}: {evaluations}");
        }
    }
}

#[test]
fn refinement_limit_is_distinct_from_work_exhaustion() {
    let result = special_zeros_with_options(
        SpecialZeroFamily::Kelvin(KelvinZeroFunction::Ber),
        SpecialZeroBranch::Real,
        9,
        1,
        &SpecialZeroOptions::<f64>::default().with_refinement_steps(Budget::ONE),
    )
    .unwrap();
    assert_eq!(
        result.report().termination(),
        RootTermination::IterationLimit
    );
    assert!(result.roots().is_empty());
    assert!(result.report().candidate().is_some());
}

fn accelerated_families() -> impl Iterator<Item = (SpecialZeroFamily<f64>, SpecialZeroBranch)> {
    use KelvinZeroFunction as K;
    use SpecialZeroFamily as F;
    [
        K::Ber,
        K::Bei,
        K::Ker,
        K::Kei,
        K::BerPrime,
        K::BeiPrime,
        K::KerPrime,
        K::KeiPrime,
    ]
    .into_iter()
    .map(|k| (F::Kelvin(k), SpecialZeroBranch::Real))
    .chain([F::Erf, F::FresnelC, F::FresnelS].map(|f| (f, SpecialZeroBranch::UPPER)))
}

#[test]
fn kelvin_erf_and_fresnel_need_at_most_four_paired_evaluations() {
    for (family, branch) in accelerated_families() {
        for index in 1..=256 {
            let result = special_zeros_with_options(
                family,
                branch,
                index,
                1,
                &SpecialZeroOptions::default(),
            )
            .unwrap();
            assert_eq!(
                result.report().termination(),
                RootTermination::Complete,
                "{family:?} {index}"
            );
            let evaluations = result
                .report()
                .accounting()
                .total()
                .unwrap()
                .total_providers()
                .unwrap();
            assert!(evaluations <= 4, "{family:?} {index}: {evaluations}");
        }
    }
}

#[test]
fn accelerated_pairs_reuse_storage_and_publish_only_verified_prefixes() {
    for (family, branch) in accelerated_families() {
        let options = SpecialZeroOptions::default();
        let mut workspace = SpecialZeroWorkspace::workspace(16, &options).unwrap();
        let mut output = [None; 16];
        let first = special_zeros_into(
            family,
            branch,
            1,
            &options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(first.termination(), RootTermination::Complete);
        let saved = output;
        let repeated = special_zeros_into(
            family,
            branch,
            1,
            &options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(first.accounting(), repeated.accounting());
        assert_eq!(
            saved.map(|z| z.unwrap().point()),
            output.map(|z| z.unwrap().point())
        );
        let first_count =
            special_zeros_with_options(family, branch, 1, 1, &SpecialZeroOptions::default())
                .unwrap()
                .report()
                .accounting()
                .total()
                .unwrap()
                .total_providers()
                .unwrap();
        // Partitioned solves divide shared allowances among zeros. Give the
        // first partition enough to finish while later ones stop mid-refinement.
        let mut witnessed_prefix = false;
        for budget in [1, 16 * first_count - 15, 16 * first_count - 1] {
            let limited = SpecialZeroOptions::default()
                .with_evaluation_budget(Budget::try_from(budget).unwrap());
            output.fill(saved[15]);
            let report = special_zeros_into(
                family,
                branch,
                1,
                &limited,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(
                report.termination(),
                RootTermination::WorkLimit(BudgetKind::Evaluation)
            );
            assert!(report.completed() < 16, "{family:?} {budget}");
            witnessed_prefix |= report.completed() > 0;
            for (actual, expected) in output[..report.completed()].iter().zip(saved) {
                assert_eq!(actual.unwrap().point(), expected.unwrap().point());
            }
            assert!(
                output[report.completed()..]
                    .iter()
                    .all(|z| z.unwrap().index() == 16)
            );
            assert!(
                report
                    .accounting()
                    .total()
                    .unwrap()
                    .total_providers()
                    .unwrap()
                    <= budget
            );
        }
        assert!(witnessed_prefix, "{family:?}");
    }
}

#[test]
fn kelvin_brackets_retain_independent_endpoint_signs() {
    use KelvinZeroFunction as K;
    for (family, _) in accelerated_families() {
        let SpecialZeroFamily::Kelvin(k) = family else {
            continue;
        };
        let evaluate = |x| {
            let value = match k {
                K::Ber => scioxide_special::ber_scaled(x),
                K::Bei => scioxide_special::bei_scaled(x),
                K::Ker => scioxide_special::ker_scaled(x),
                K::Kei => scioxide_special::kei_scaled(x),
                K::BerPrime => scioxide_special::berp_scaled(x),
                K::BeiPrime => scioxide_special::beip_scaled(x),
                K::KerPrime => scioxide_special::kerp_scaled(x),
                K::KeiPrime => scioxide_special::keip_scaled(x),
            };
            value.unwrap()
        };
        for index in [1, 8, 9, 10, 64, 256, 10000] {
            let result = special_zeros_with_options(
                family,
                SpecialZeroBranch::Real,
                index,
                1,
                &SpecialZeroOptions::default(),
            )
            .unwrap();
            let root = result.roots()[0];
            let bracket = root.bracket().unwrap();
            assert!(bracket.lower() <= root.point().re && root.point().re <= bracket.upper());
            assert_ne!(
                evaluate(bracket.lower()).is_sign_negative(),
                evaluate(bracket.upper()).is_sign_negative(),
                "{k:?} {index}"
            );
            let projected = special_zeros_with_options(
                SpecialZeroFamily::Kelvin(k),
                SpecialZeroBranch::Real,
                index,
                1,
                &SpecialZeroOptions::<f32>::default(),
            )
            .unwrap();
            let bracket32 = projected.roots()[0].bracket().unwrap();
            assert!(
                (bracket32.lower() as f64) <= root.point().re
                    && root.point().re <= bracket32.upper() as f64
            );
            assert_ne!(
                evaluate(bracket32.lower() as f64).is_sign_negative(),
                evaluate(bracket32.upper() as f64).is_sign_negative()
            );
        }
    }
}

#[test]
fn complex_projection_and_symmetry_require_fresh_verification() {
    use SpecialZeroFamily as F;
    for family in [F::Erf, F::FresnelC, F::FresnelS] {
        for index in [1, 8, 9, 64, 10000] {
            let base = special_zeros_with_options(
                family,
                SpecialZeroBranch::UPPER,
                index,
                1,
                &SpecialZeroOptions::<f64>::default(),
            )
            .unwrap();
            let count = base
                .report()
                .accounting()
                .total()
                .unwrap()
                .total_providers()
                .unwrap();
            for branch in [
                SpecialZeroBranch::Complex {
                    conjugate: true,
                    negate: false,
                    rotate: false,
                },
                SpecialZeroBranch::Complex {
                    conjugate: false,
                    negate: true,
                    rotate: false,
                },
                SpecialZeroBranch::Complex {
                    conjugate: false,
                    negate: false,
                    rotate: true,
                },
            ] {
                if matches!(family, F::Erf)
                    && matches!(branch, SpecialZeroBranch::Complex { rotate: true, .. })
                {
                    continue;
                }
                let transformed = special_zeros_with_options(
                    family,
                    branch,
                    index,
                    1,
                    &SpecialZeroOptions::default(),
                )
                .unwrap();
                assert_eq!(
                    transformed.report().termination(),
                    RootTermination::Complete
                );
                assert_eq!(
                    transformed
                        .report()
                        .accounting()
                        .total()
                        .unwrap()
                        .total_providers()
                        .unwrap(),
                    count + 1
                );
            }
            let family32 = match family {
                F::Erf => F::Erf,
                F::FresnelC => F::FresnelC,
                _ => F::FresnelS,
            };
            let projected = special_zeros_with_options(
                family32,
                SpecialZeroBranch::UPPER,
                index,
                1,
                &SpecialZeroOptions::<f32>::default(),
            )
            .unwrap();
            assert_eq!(projected.report().termination(), RootTermination::Complete);
            assert_eq!(
                projected
                    .report()
                    .accounting()
                    .total()
                    .unwrap()
                    .total_providers()
                    .unwrap(),
                count + 1
            );
        }
        let strict = special_zeros_with_options(
            family,
            SpecialZeroBranch::UPPER,
            9,
            1,
            &SpecialZeroOptions::default()
                .with_tolerance(scioxide_core::StepTolerance::new(1e-30, 0.0).unwrap()),
        )
        .unwrap();
        assert_eq!(strict.report().termination(), RootTermination::Unverified);
        assert!(strict.roots().is_empty());
        let limited = special_zeros_with_options(
            family,
            SpecialZeroBranch::UPPER,
            9,
            1,
            &SpecialZeroOptions::default().with_refinement_steps(Budget::ONE),
        )
        .unwrap();
        assert_eq!(
            limited.report().termination(),
            RootTermination::IterationLimit
        );
        assert!(limited.report().candidate().is_some());
    }
}

#[test]
fn real_pairs_reuse_reserved_storage_and_preserve_output_on_exhaustion() {
    for family in [
        SpecialZeroFamily::BesselJ(0.0),
        SpecialZeroFamily::AiryAi,
        SpecialZeroFamily::AiryBiPrime,
    ] {
        let options = SpecialZeroOptions::<f64>::default();
        let mut workspace = SpecialZeroWorkspace::workspace(16, &options).unwrap();
        let mut output = [None; 16];
        let first = special_zeros_into(
            family,
            SpecialZeroBranch::Real,
            1,
            &options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(first.termination(), RootTermination::Complete);
        let saved = output.map(|z| z.unwrap().point());
        let repeated = special_zeros_into(
            family,
            SpecialZeroBranch::Real,
            1,
            &options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(first.accounting(), repeated.accounting());
        assert_eq!(saved, output.map(|z| z.unwrap().point()));

        let sentinel = output[0];
        let limited = SpecialZeroOptions::<f64>::default()
            .with_evaluation_budget(Budget::try_from(5).unwrap());
        output.fill(sentinel);
        let report = special_zeros_into(
            family,
            SpecialZeroBranch::Real,
            1,
            &limited,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(
            report.termination(),
            RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
        );
        assert_eq!(report.completed(), 0);
        assert!(output.iter().all(|zero| zero.unwrap().index() == 1));
        assert_eq!(
            report
                .accounting()
                .total()
                .unwrap()
                .total_providers()
                .unwrap(),
            5
        );
    }
}

#[test]
fn airy_ode_brackets_have_independently_checked_endpoint_signs() {
    use SpecialZeroFamily as F;
    for family in [F::AiryAi, F::AiryAiPrime, F::AiryBi, F::AiryBiPrime] {
        for index in [1, 8, 9, 64, 256, 10000] {
            let result = special_zeros_with_options(
                family,
                SpecialZeroBranch::Real,
                index,
                1,
                &SpecialZeroOptions::<f64>::default(),
            )
            .unwrap();
            let root = result.roots()[0];
            let bracket = root.bracket().unwrap();
            let evaluate = |x| {
                let result = if matches!(family, F::AiryAi | F::AiryAiPrime) {
                    scioxide_special::airy_ai_with_options(
                        x,
                        scioxide_special::AiryOptions::default(),
                    )
                    .unwrap()
                } else {
                    scioxide_special::airy_bi_with_options(
                        x,
                        scioxide_special::AiryOptions::default(),
                    )
                    .unwrap()
                };
                *if matches!(family, F::AiryAiPrime | F::AiryBiPrime) {
                    result.derivative().unwrap()
                } else {
                    result.value().unwrap()
                }
                .value()
            };
            assert!(bracket.lower() <= root.point().re && root.point().re <= bracket.upper());
            assert_ne!(
                evaluate(bracket.lower()).is_sign_negative(),
                evaluate(bracket.upper()).is_sign_negative(),
                "{family:?} {index}"
            );
            let family32 = match family {
                F::AiryAi => F::AiryAi,
                F::AiryAiPrime => F::AiryAiPrime,
                F::AiryBi => F::AiryBi,
                _ => F::AiryBiPrime,
            };
            let result32 = special_zeros_with_options(
                family32,
                SpecialZeroBranch::Real,
                index,
                1,
                &SpecialZeroOptions::<f32>::default(),
            )
            .unwrap();
            let bracket32 = result32.roots()[0].bracket().unwrap();
            assert!(
                (bracket32.lower() as f64) <= root.point().re
                    && root.point().re <= bracket32.upper() as f64
            );
            assert_ne!(
                evaluate(bracket32.lower() as f64).is_sign_negative(),
                evaluate(bracket32.upper() as f64).is_sign_negative()
            );
        }
    }
}

#[test]
fn complex_indexed_zeros_charge_one_paired_evaluation_per_refinement() {
    use scioxide_core::EvaluationRole;
    for family in [
        SpecialZeroFamily::BesselY(0.0),
        SpecialZeroFamily::BesselY(1.0),
        SpecialZeroFamily::BesselYPrime(1.0),
        SpecialZeroFamily::AiryBi,
        SpecialZeroFamily::Erf,
    ] {
        let zeros = special_zeros_with_options(
            family,
            SpecialZeroBranch::UPPER,
            1,
            3,
            &SpecialZeroOptions::default(),
        )
        .unwrap();
        let report = zeros.report();
        assert!(report.is_success(), "{family:?}: {report:?}");
        let total = report.accounting().total().unwrap();
        // The derivative comes with the value: no separate Jacobian request.
        // Every refinement step makes one paired call and proposes the next
        // point unless it converged, and verification reuses that last call.
        assert_eq!(total.provider(EvaluationRole::Jacobian), 0, "{family:?}");
        assert_eq!(
            total.provider(EvaluationRole::Function),
            total.proposals() + 3,
            "{family:?}"
        );
    }
}
