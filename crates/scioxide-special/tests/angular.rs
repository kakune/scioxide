#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{
    Argument, Budget, ExecutionContext, SciOxideError, SciOxideErrorView, argument,
};
use scioxide_special::{
    AngularMomentum, AngularOptions, AngularPlan, AngularState, SpecialFunctionTermination,
    clebsch_gordan, gaunt, racah_w, wigner_3j, wigner_6j, wigner_9j,
};

fn momenta<const N: usize>(j: [i32; N]) -> [AngularMomentum; N] {
    j.map(|j| AngularMomentum::new(j).unwrap())
}
fn invalid_argument(error: SciOxideError) -> Argument {
    match error.view() {
        SciOxideErrorView::InvalidInput { argument, .. } => argument,
        other => panic!("unexpected error: {other:?}"),
    }
}
#[test]
fn states_take_doubled_quantum_numbers() {
    let half = AngularState::new(1, 1).unwrap();
    assert_eq!((half.momentum().twice_j(), half.twice_m()), (1, 1));
    for (twice_j, twice_m, rejected) in [
        (0, 1, argument!("twice_m")),
        (2, 4, argument!("twice_m")),
        (1, 0, argument!("twice_m")),
        (-1, 0, argument!("twice_j")),
        (-1, 1, argument!("twice_j")),
    ] {
        assert_eq!(
            invalid_argument(AngularState::new(twice_j, twice_m).unwrap_err()),
            rejected,
            "({twice_j}, {twice_m})"
        );
    }
}
#[test]
fn one_shots_take_doubled_integers() {
    // ⟨½ ½; ½ −½ | 1 0⟩ = 1/√2, (1 1 0; 1 −1 0) = 1/√3 and {0 ½ ½; 0 ½ ½} = −½
    let cg = clebsch_gordan::<f64>([1, 1, 2], [1, -1, 0]).unwrap();
    assert!((cg - 0.5_f64.sqrt()).abs() < 1e-14);
    let three = wigner_3j::<f64>([2, 2, 0], [2, -2, 0]).unwrap();
    assert!((three - 1.0 / 3.0_f64.sqrt()).abs() < 1e-14);
    assert!((wigner_6j::<f64>([0, 1, 1, 0, 1, 1]).unwrap() + 0.5).abs() < 1e-14);
    let narrow = clebsch_gordan::<f32>([1, 1, 2], [1, -1, 0]).unwrap();
    assert!((narrow - core::f32::consts::FRAC_1_SQRT_2).abs() < 2e-7);
    // {1 1 0; 1 1 0; 0 0 0} = {1 1 0; 1 1 0} = 1/3 and W(½ ½ ½ ½; 0 0) = −½.
    assert!((wigner_9j::<f64>([2, 2, 0, 2, 2, 0, 0, 0, 0]).unwrap() - 1.0 / 3.0).abs() < 1e-14);
    assert!((racah_w::<f64>([1, 1, 1, 1, 0, 0]).unwrap() + 0.5).abs() < 1e-14);
    // Forbidden couplings are exact zeros.
    for value in [
        wigner_3j::<f64>([2, 2, 2], [2, 0, 0]),
        wigner_3j::<f64>([2, 2, 2], [0, 0, 0]),
        clebsch_gordan::<f64>([1, 1, 2], [1, 1, 0]),
        clebsch_gordan::<f64>([1, 1, 6], [1, 1, 2]),
        wigner_6j::<f64>([2, 2, 6, 2, 2, 2]),
        wigner_9j::<f64>([2; 9]),
        racah_w::<f64>([1, 1, 1, 1, 6, 0]),
        gaunt::<f64>([1, 1, 1], [0, 0, 0]),
        gaunt::<f64>([1, 1, 0], [1, 1, 0]),
    ] {
        assert_eq!(value.unwrap().to_bits(), 0.0_f64.to_bits());
    }
}
#[test]
fn one_shots_reject_invalid_quantum_numbers() {
    for (error, rejected) in [
        (
            wigner_3j::<f64>([-1, 1, 0], [1, 1, 0]),
            argument!("twice_j"),
        ),
        (
            wigner_3j::<f64>([2, 2, 0], [4, -2, 0]),
            argument!("twice_m"),
        ),
        (
            wigner_3j::<f64>([2, 2, 0], [1, -1, 0]),
            argument!("twice_m"),
        ),
        (
            clebsch_gordan::<f64>([1, -2, 2], [1, 0, 0]),
            argument!("twice_j"),
        ),
        (
            clebsch_gordan::<f64>([1, 1, 2], [0, 1, 1]),
            argument!("twice_m"),
        ),
        (wigner_6j::<f64>([0, 1, 1, 0, 1, -1]), argument!("twice_j")),
        (wigner_9j::<f64>([-2; 9]), argument!("twice_j")),
        (racah_w::<f64>([1, 1, 1, 1, -1, 0]), argument!("twice_j")),
        (gaunt::<f64>([1, 1, 2], [2, 0, 0]), argument!("order")),
        (gaunt::<f64>([1, 1, 2], [0, -2, 0]), argument!("order")),
        (
            gaunt::<f64>([0, 0, 0], [i32::MAX, 0, 0]),
            argument!("order"),
        ),
        (
            gaunt::<f64>([0, 0, 0], [i32::MIN, 0, 0]),
            argument!("order"),
        ),
        (
            gaunt::<f64>([1 << 30, 0, 0], [0, 0, 0]),
            argument!("degree"),
        ),
        (
            gaunt::<f64>([0, usize::MAX, 0], [0, 0, 0]),
            argument!("degree"),
        ),
    ] {
        assert_eq!(invalid_argument(error.unwrap_err()), rejected);
    }
}
#[test]
fn gaunt_takes_undoubled_degrees_and_orders() {
    let plan = AngularPlan::new(AngularMomentum::new(4).unwrap(), Default::default()).unwrap();
    for order in [[0, 0, 0], [1, -1, 0], [1, 1, -2], [-1, 0, 1]] {
        let doubled = [0, 1, 2].map(|i| AngularState::new([2, 2, 4][i], 2 * order[i]).unwrap());
        let prepared = plan.gaunt::<f64>(doubled).unwrap().try_value().unwrap();
        let value = gaunt::<f64>([1, 1, 2], order).unwrap();
        assert_eq!(value.to_bits(), prepared.to_bits(), "{order:?}");
    }
    // ∫ (Y₁⁰)² Y₂⁰ dΩ = √(3·3·5/(4π)) (1 1 2; 0 0 0)² with (1 1 2; 0 0 0)² = 2/15
    let expected = (45.0 / (4.0 * core::f64::consts::PI)).sqrt() * 2.0 / 15.0;
    assert!((gaunt::<f64>([1, 1, 2], [0, 0, 0]).unwrap() - expected).abs() < 1e-15);
}
#[test]
fn selection_rules_and_low_orders() {
    let p = AngularPlan::new(
        scioxide_special::AngularMomentum::new(20).unwrap(),
        AngularOptions::default(),
    )
    .unwrap();
    let q = |j, m| AngularState::new(j, m).unwrap();
    assert_eq!(
        p.wigner_3j::<f64>([q(2, 2), q(2, 0), q(2, 0)])
            .unwrap()
            .try_value()
            .unwrap(),
        0.0
    );
    assert_eq!(
        p.wigner_3j::<f64>([q(2, 0), q(2, 0), q(2, 0)])
            .unwrap()
            .try_value()
            .unwrap(),
        0.0
    );
    assert_eq!(
        p.wigner_6j::<f64>(momenta([0, 0, 0, 0, 0, 0]))
            .unwrap()
            .try_value()
            .unwrap(),
        1.0
    );
    assert_eq!(
        p.wigner_9j::<f64>(momenta([0; 9]))
            .unwrap()
            .try_value()
            .unwrap(),
        1.0
    );
    assert!(
        (p.wigner_3j::<f64>([q(1, 1), q(1, -1), q(0, 0)])
            .unwrap()
            .try_value()
            .unwrap()
            - 1.0 / 2.0_f64.sqrt())
        .abs()
            < 1e-14
    );
    assert!(
        (p.wigner_6j::<f64>(momenta([2; 6]))
            .unwrap()
            .try_value()
            .unwrap()
            - 1.0 / 6.0)
            .abs()
            < 1e-14
    );
    assert!(
        (p.clebsch_gordan::<f64>(q(1, 1), q(1, 1), q(2, 2))
            .unwrap()
            .try_value()
            .unwrap()
            - 1.0)
            .abs()
            < 1e-14
    );
    assert_eq!(
        p.wigner_9j::<f64>(momenta([2; 9]))
            .unwrap()
            .try_value()
            .unwrap(),
        0.0
    );
}
#[test]
fn orthogonality() {
    let p = AngularPlan::new(
        scioxide_special::AngularMomentum::new(20).unwrap(),
        AngularOptions::default(),
    )
    .unwrap();
    let mut scratch = p.workspace().unwrap();
    for j in [0, 2, 4, 6] {
        for m in (-j..=j).step_by(2) {
            let mut sum = 0.0;
            for a in (-4..=4).step_by(2) {
                let b = m - a;
                if !(-2..=2).contains(&b) {
                    continue;
                }
                let c = p
                    .clebsch_gordan_scoped::<f64>(
                        AngularState::new(4, a).unwrap(),
                        AngularState::new(2, b).unwrap(),
                        AngularState::new(j, m).unwrap(),
                        &mut scratch,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap()
                    .try_value()
                    .unwrap();
                sum += c * c;
            }
            if j == 0 {
                assert_eq!(sum, 0.0);
            } else {
                assert!((sum - 1.0).abs() < 1e-13, "j={j} m={m} sum={sum}");
            }
        }
    }
}
#[test]
fn resources_and_f32() {
    let p = AngularPlan::new(
        scioxide_special::AngularMomentum::new(20).unwrap(),
        AngularOptions::default(),
    )
    .unwrap();
    let v = p
        .wigner_6j::<f32>(momenta([2; 6]))
        .unwrap()
        .try_value()
        .unwrap();
    assert!((v - 1.0 / 6.0).abs() < 2e-7);
    let options = AngularOptions::default().with_iteration_budget(Budget::ONE);
    {
        let outcome =
            &(AngularPlan::new(scioxide_special::AngularMomentum::new(20).unwrap(), options));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
        ));
    }
    let p = AngularPlan::new(
        scioxide_special::AngularMomentum::new(100).unwrap(),
        AngularOptions::default(),
    )
    .unwrap();
    assert_eq!(
        invalid_argument(p.wigner_6j::<f64>(momenta([101; 6])).unwrap_err()),
        argument!("j")
    );
    assert!(
        p.wigner_3j::<f64>([AngularState::new(2, 0).unwrap(); 3])
            .unwrap()
            .termination()
            .is_success()
    );
    let _ = SpecialFunctionTermination::IterationLimit;
}

#[test]
fn owned_plan_methods_equal_scoped_methods_on_a_reserved_workspace() {
    let plan = AngularPlan::new(AngularMomentum::new(8).unwrap(), Default::default()).unwrap();
    let mut scratch = plan.workspace().unwrap();
    let scope = ExecutionContext::sequential_native;
    let q = [(2, 2), (4, -2), (6, 0)].map(|(j, m)| AngularState::new(j, m).unwrap());
    let j = momenta([2, 4, 6, 4, 2, 4]);
    let nine = momenta([2, 4, 6, 4, 2, 4, 6, 4, 8]);
    let even = [(2, 0), (4, 0), (6, 0)].map(|(j, m)| AngularState::new(j, m).unwrap());
    let pairs = [
        (
            plan.wigner_3j::<f64>(q),
            plan.wigner_3j_scoped(q, &mut scratch, scope()),
        ),
        (
            plan.clebsch_gordan(q[0], q[1], q[2]),
            plan.clebsch_gordan_scoped(q[0], q[1], q[2], &mut scratch, scope()),
        ),
        (
            plan.wigner_6j(j),
            plan.wigner_6j_scoped(j, &mut scratch, scope()),
        ),
        (
            plan.wigner_9j(nine),
            plan.wigner_9j_scoped(nine, &mut scratch, scope()),
        ),
        (
            plan.racah_w(j),
            plan.racah_w_scoped(j, &mut scratch, scope()),
        ),
        (
            plan.gaunt(even),
            plan.gaunt_scoped(even, &mut scratch, scope()),
        ),
    ];
    for (owned, scoped) in pairs {
        assert_eq!(owned.unwrap(), scoped.unwrap());
    }
}

#[test]
fn prepared_wigner_reuses_scratch_with_reused_storage() {
    let plan = AngularPlan::new(AngularMomentum::new(200).unwrap(), Default::default()).unwrap();
    let mut scratch = plan.workspace().unwrap();
    let scope = ExecutionContext::sequential_native;
    for _ in 0..2 {
        {
            let large = plan
                .wigner_6j_scoped::<f64>(momenta([200; 6]), &mut scratch, scope())
                .unwrap();
            assert!((large.try_value().unwrap() + 0.0004698416232987442).abs() < 1e-15);
            let zero = plan
                .wigner_9j_scoped::<f64>(momenta([2; 9]), &mut scratch, scope())
                .unwrap();
            assert_eq!(zero.try_value().unwrap(), 0.0);
            let general = plan
                .wigner_9j_scoped::<f64>(
                    momenta([4, 6, 8, 6, 4, 8, 8, 8, 8]),
                    &mut scratch,
                    scope(),
                )
                .unwrap();
            assert!((general.try_value().unwrap() - 0.0007369614512471656).abs() < 1e-16);
            let reduced = plan
                .wigner_9j_scoped::<f64>(
                    momenta([8, 8, 8, 8, 8, 8, 8, 8, 0]),
                    &mut scratch,
                    scope(),
                )
                .unwrap();
            assert!((reduced.try_value().unwrap() + 0.0028798362131695465).abs() < 1e-15);
            let half = plan
                .wigner_3j_scoped::<f32>(
                    [
                        AngularState::new(1, 1).unwrap(),
                        AngularState::new(1, -1).unwrap(),
                        AngularState::new(0, 0).unwrap(),
                    ],
                    &mut scratch,
                    scope(),
                )
                .unwrap();
            assert!((half.try_value().unwrap() - core::f32::consts::FRAC_1_SQRT_2).abs() < 2e-7);
            assert!(
                plan.wigner_6j_scoped::<f64>(momenta([201; 6]), &mut scratch, scope())
                    .is_err()
            );
            let identity = plan
                .wigner_9j_scoped::<f64>(momenta([0; 9]), &mut scratch, scope())
                .unwrap();
            assert_eq!(identity.try_value().unwrap(), 1.0);
        };
    }
}

#[test]
fn angular_storage_limits_and_workspace_mismatch() {
    use scioxide_core::SciOxideErrorView;
    let maximum = AngularMomentum::new(20).unwrap();
    // The retained factor indices and prime capacity each reserve bound+1 u32s.
    let preparation_bytes = (3 * 20 + 3) * 2 * size_of::<u32>();
    let options = AngularOptions::default()
        .with_preparation_bytes(Budget::try_from(preparation_bytes).unwrap());
    let plan = AngularPlan::new(maximum, options).unwrap();
    let workspace_bytes = plan.workspace_bytes();
    AngularPlan::new(
        maximum,
        options.with_workspace_bytes(Budget::try_from(workspace_bytes).unwrap()),
    )
    .unwrap();
    for options in [
        options.with_preparation_bytes(Budget::try_from(preparation_bytes - 1).unwrap()),
        options.with_workspace_bytes(Budget::try_from(workspace_bytes - 1).unwrap()),
    ] {
        assert!(matches!(
            AngularPlan::new(maximum, options).unwrap_err().view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
    }
    let small = AngularPlan::new(AngularMomentum::new(0).unwrap(), Default::default()).unwrap();
    let mut scratch = small.workspace().unwrap();
    let scope = ExecutionContext::sequential_native;
    assert_eq!(
        invalid_argument(
            plan.wigner_9j_scoped::<f64>(momenta([0; 9]), &mut scratch, scope())
                .unwrap_err()
        ),
        argument!("scratch")
    );
    assert_eq!(
        small
            .wigner_9j_scoped::<f64>(momenta([0; 9]), &mut scratch, scope())
            .unwrap()
            .try_value()
            .unwrap(),
        1.0
    );
}

#[test]
fn exhausted_angular_evaluations_publish_no_partial_sum() {
    let maximum = AngularMomentum::new(40).unwrap();
    let reference = AngularPlan::new(maximum, Default::default()).unwrap();
    // Enough to prepare, but not enough for the large general 9j sum.
    let options = AngularOptions::default()
        .with_iteration_budget(Budget::try_from(reference.preparation_work()).unwrap());
    let plan = AngularPlan::new(maximum, options).unwrap();
    let mut scratch = plan.workspace().unwrap();
    let scope = ExecutionContext::sequential_native;
    for _ in 0..2 {
        let report = plan
            .wigner_9j_scoped::<f64>(momenta([40; 9]), &mut scratch, scope())
            .unwrap_err();
        let report_counts = publication::exhausted_counts(&report);
        assert_eq!(report_counts.iterations(), options.iteration_budget().get());
        assert_eq!(
            publication::exhausted_counts(&plan.wigner_9j::<f64>(momenta([40; 9])).unwrap_err())
                .iterations(),
            options.iteration_budget().get()
        );
        assert_eq!(
            plan.wigner_9j_scoped::<f64>(momenta([0; 9]), &mut scratch, scope())
                .unwrap()
                .try_value()
                .unwrap(),
            1.0
        );
    }
}

#[test]
fn scoped_preparation_and_evaluation_report_nested_work() {
    use scioxide_core::{
        BudgetKind, CancellationToken, ProblemToken, SciOxideErrorView, SolveBudgets,
        SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(500),
        scioxide_core::algorithm_id!("test", "angular", 1),
        0,
        0,
    );
    let query = momenta([2; 6]);
    let options = AngularOptions::default();
    let reference =
        AngularPlan::new(scioxide_special::AngularMomentum::new(20).unwrap(), options).unwrap();
    let expected = reference.wigner_6j::<f64>(query).unwrap();
    let mut state = SolveState::new();
    // Iteration allowances are scope-local; the parent limit does not bound children.
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let plan = AngularPlan::build(
        scioxide_special::AngularMomentum::new(20).unwrap(),
        options,
        &mut context,
    )
    .unwrap();
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(
        context.counts().nested().iterations(),
        plan.preparation_work()
    );
    let mut scratch = plan.workspace().unwrap();
    let before = context.counts().nested().iterations();
    let scoped = plan
        .wigner_6j_scoped::<f64>(query, &mut scratch, &mut context)
        .unwrap();
    assert_eq!(scoped, expected);
    assert_eq!(
        context.counts().nested().iterations() - before,
        scoped.counts().iterations()
    );
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let stopped = ExecutionContext::sequential().with_cancellation(&cancellation);
    for error in [
        AngularPlan::build(
            scioxide_special::AngularMomentum::new(20).unwrap(),
            options,
            stopped,
        )
        .unwrap_err(),
        plan.wigner_6j_scoped::<f64>(query, &mut scratch, stopped)
            .unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ExecutionStopped { .. }
        ));
    }
}

fn oracle<S: scioxide_core::RealScalar>(width: &str)
where
    scioxide_core::Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    use scioxide_special::{AngularWorkspace, spherical_harmonic_with_options};
    let options = AngularOptions::default();
    let plan = AngularPlan::new(
        scioxide_special::AngularMomentum::new(400).unwrap(),
        options,
    )
    .unwrap();
    let mut scratch: AngularWorkspace = plan.workspace().unwrap();
    let scope = ExecutionContext::sequential_native;
    let state = |j, m| AngularState::new(j, m).unwrap();
    let mut count = 0;
    for line in include_str!("data/angular.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        count += 1;
        let args: Vec<f64> = f[2].split(',').map(|v| v.parse().unwrap()).collect();
        let number = |i: usize| S::try_from_f64_lossy(f[i].parse().unwrap()).unwrap();
        let expected = scioxide_core::Complex::new(number(3), number(4));
        let actual = if f[1] == "harmonic" {
            let r = spherical_harmonic_with_options(
                args[0] as usize,
                args[1] as i32,
                S::try_from_f64_lossy(args[2]).unwrap(),
                S::try_from_f64_lossy(args[3]).unwrap(),
                scioxide_special::LegendreOptions::new((options).iteration_budget()),
            )
            .unwrap();
            assert_eq!(
                r.scalar_representation(),
                scioxide_special::ScalarRepresentation::Finite
            );
            let value = r.try_value().unwrap();
            // A rounded zero keeps its magnitude in the scaled and log forms.
            if let Some(log) = r
                .logarithmic()
                .and_then(|log| log.value().log_magnitude())
                .filter(|_| value == scioxide_core::Complex::new(S::ZERO, S::ZERO))
            {
                assert_eq!(expected, value);
                assert!(r.scaled().is_some());
                assert!((log - number(5)).abs().try_to_f64_exact().unwrap() < 2e-4);
                continue;
            }
            value
        } else if f[1] == "associated" {
            scioxide_core::Complex::new(
                scioxide_special::legendre_p_assoc(
                    args[0] as usize,
                    args[1] as i32,
                    S::try_from_f64_lossy(args[2]).unwrap(),
                )
                .unwrap(),
                S::ZERO,
            )
        } else {
            let a: Vec<i32> = args.iter().map(|&v| v as i32).collect();
            let r = match f[1] {
                "3j" => plan.wigner_3j_scoped::<S>(
                    [state(a[0], a[3]), state(a[1], a[4]), state(a[2], a[5])],
                    &mut scratch,
                    scope(),
                ),
                "cg" => plan.clebsch_gordan_scoped::<S>(
                    state(a[0], a[3]),
                    state(a[1], a[4]),
                    state(a[2], a[5]),
                    &mut scratch,
                    scope(),
                ),
                "6j" => plan.wigner_6j_scoped::<S>(
                    momenta(a.as_slice().try_into().unwrap()),
                    &mut scratch,
                    scope(),
                ),
                "9j" => plan.wigner_9j_scoped::<S>(
                    momenta(a.as_slice().try_into().unwrap()),
                    &mut scratch,
                    scope(),
                ),
                "racah" => plan.racah_w_scoped::<S>(
                    momenta(a.as_slice().try_into().unwrap()),
                    &mut scratch,
                    scope(),
                ),
                "gaunt" => plan.gaunt_scoped::<S>(
                    core::array::from_fn(|i| state(2 * a[i], 2 * a[i + 3])),
                    &mut scratch,
                    scope(),
                ),
                _ => unreachable!(),
            }
            .unwrap();
            assert!(r.termination().is_success(), "{line}: {r:?}");
            let value = r.value().unwrap();
            // Fixtures are rounded in the destination precision. Include that
            // final rounding when checking the published absolute estimate.
            let rounding = S::EPSILON * expected.re.abs();
            assert!(
                (*value.value() - expected.re).abs() <= value.absolute_error() + rounding,
                "{line}: reported error does not cover reference: {r:?}"
            );
            assert!(r.derivative().is_none(), "discrete angular momenta: {line}");
            if expected.re == S::ZERO {
                assert_eq!(r.termination(), SpecialFunctionTermination::Exact);
                assert_eq!(value.absolute_error(), S::ZERO);
            }
            scioxide_core::Complex::new(r.try_value().unwrap(), S::ZERO)
        };
        let (relative, absolute) = if width == "64" {
            (5e-11_f64, 5e-13_f64)
        } else {
            (2e-4, 5e-6)
        };
        for (x, y) in [(actual.re, expected.re), (actual.im, expected.im)] {
            let error = (x - y).abs().try_to_f64_exact().unwrap();
            let bound = if f[1] == "harmonic" {
                absolute.max(relative * y.abs().try_to_f64_exact().unwrap())
            } else {
                relative * y.abs().try_to_f64_exact().unwrap()
            };
            assert!(
                error <= bound,
                "{line}: {actual:?}, error {error}, bound {bound}"
            );
        }
    }
    assert!(count > 30);
}
#[test]
fn independent_f64() {
    oracle::<f64>("64");
}
#[test]
fn independent_f32() {
    oracle::<f32>("32");
}

#[test]
fn harmonic_addition_theorem_and_real_basis() {
    use scioxide_special::{legendre_p_assoc, spherical_harmonic, spherical_harmonic_real};
    for l in [0, 1, 10, 50, 100] {
        for theta in [0.0, 0.1, 1.25, core::f64::consts::PI] {
            let mut complex = 0.0;
            let mut real = 0.0;
            for m in -(l as i32)..=l as i32 {
                let y = spherical_harmonic(l, m, theta, 0.75_f64).unwrap();
                let r = spherical_harmonic_real(l, m, theta, 0.75_f64).unwrap();
                complex += y.norm_sqr();
                real += r * r;
                if m > 0 && theta == 0.0 {
                    assert_eq!(y, scioxide_core::Complex::new(0.0, 0.0));
                }
            }
            let expected = (2 * l + 1) as f64 / (4.0 * core::f64::consts::PI);
            assert!(
                (complex - expected).abs() < 5e-11 * expected,
                "l={l} theta={theta} {complex}"
            );
            assert!((real - expected).abs() < 5e-11 * expected);
        }
    }
    // The old double-factorial path overflowed before applying this small factor.
    assert!(
        legendre_p_assoc(200, 200, 0.999999_f64)
            .unwrap()
            .is_finite()
    );
}

#[test]
fn associated_legendre_options_admit_actual_work() {
    use scioxide_special::{LegendreOptions, legendre_p_assoc_with_options};
    let options = LegendreOptions::default().with_iteration_budget(Budget::ONE);
    let r = legendre_p_assoc_with_options(10, 2, 0.5_f64, options).unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 1);
    let r = legendre_p_assoc_with_options(100, 0, -1.0_f64, options).unwrap();
    assert_eq!(r.termination(), SpecialFunctionTermination::Exact);
    assert_eq!(r.try_value().unwrap(), 1.0);
    let r = legendre_p_assoc_with_options(10, -2, 0.5_f64, LegendreOptions::default()).unwrap();
    assert!(r.termination().is_success());
    assert!(r.counts().iterations() >= 10);
}

#[test]
fn associated_range_reports_retain_scale() {
    use scioxide_special::{LegendreOptions, ScalarRepresentation, legendre_p_assoc_with_options};
    for (degree, order, x, expected) in [
        (200, 200, 0.25_f64, ScalarRepresentation::Overflow),
        (200, -200, 0.25, ScalarRepresentation::Finite),
        (
            200,
            200,
            f64::from_bits(1.0_f64.to_bits() - 1),
            ScalarRepresentation::Finite,
        ),
    ] {
        let report =
            legendre_p_assoc_with_options(degree, order, x, LegendreOptions::default()).unwrap();
        assert!(report.termination().is_success());
        assert_eq!(report.scalar_representation(), expected);
        // Below the range the ordinary value is a rounded zero.
        assert_eq!(
            report.value().map(|value| *value.value()),
            (expected == ScalarRepresentation::Finite).then_some(0.0)
        );
        assert!(report.scaled().is_some());
        assert!(report.logarithmic().is_some());
    }
}

#[test]
fn harmonic_error_estimate_retains_cancellation_scale() {
    use scioxide_special::{
        LegendreOptions, legendre_p_assoc_with_options, spherical_harmonic_with_options,
    };
    // P_2(1/√3) = 0: the cancellation keeps an absolute estimate and the
    // value is never downgraded.
    let x = 1.0_f64 / 3.0_f64.sqrt();
    let harmonic =
        spherical_harmonic_with_options(2, 0, x.acos(), 0.0, LegendreOptions::default()).unwrap();
    assert_eq!(
        harmonic.termination(),
        SpecialFunctionTermination::Converged
    );
    let associated = legendre_p_assoc_with_options(2, 0, x, LegendreOptions::default()).unwrap();
    assert_eq!(
        associated.termination(),
        SpecialFunctionTermination::Converged
    );
    assert!(harmonic.value().unwrap().absolute_error() > 1e-17);
    assert!(associated.value().unwrap().absolute_error() > 1e-17);
}

#[test]
fn harmonics_share_legendre_defaults_and_scoped_evaluation() {
    use scioxide_special::{
        LegendreOptions, legendre_p_assoc, spherical_harmonic, spherical_harmonic_real_scoped,
        spherical_harmonic_real_with_options,
    };
    let degree = 100_000;
    let theta = core::f64::consts::FRAC_PI_3;
    assert_eq!(
        legendre_p_assoc(degree, 0, 0.5_f64).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(
        spherical_harmonic(degree, 0, theta, 0.0)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let options = LegendreOptions::new(scioxide_core::Budget::try_from(200_000).unwrap());
    let detailed = spherical_harmonic_real_with_options(degree, 0, theta, 0.0, options).unwrap();
    let scoped = spherical_harmonic_real_scoped(
        degree,
        0,
        theta,
        0.0,
        options,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(detailed, scoped);
    assert!(detailed.termination().is_success());
}
