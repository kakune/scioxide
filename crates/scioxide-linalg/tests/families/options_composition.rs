//! Resource controls compose independently of mathematical and diagnostic settings.
#![allow(clippy::unwrap_used)]

use core::time::Duration;
use scioxide_core::Accuracy;
use scioxide_core::{
    Budget, ClockBinding, ExecutionContext, ExecutionPolicy, RankThreshold, ResidualTolerance,
    SciOxideError, SciOxideErrorKind, Tolerance,
};
use scioxide_linalg::{
    CsdDecomposition, DirectSolveTermination, Estimation, GsvdDecomposition, GsvdOptions,
    LinalgExecutionOptions, LinalgWorkspace, Matrix, PolarDecomposition, PolarOptions, PolarSide,
    QrRankSource, QrUpdateDecomposition, QrUpdateOptions, RankPolicy, RiccatiBalancing,
    RiccatiKind, RiccatiOptions, RiccatiPlan, RiccatiProblem, SvdValues,
};

fn permutations(order: &mut [usize], start: usize, visit: &mut impl FnMut(&[usize])) {
    if start == order.len() {
        visit(order);
    } else {
        for i in start..order.len() {
            order.swap(start, i);
            permutations(order, start + 1, visit);
            order.swap(start, i);
        }
    }
}

fn reordered(order: &[usize], expected: LinalgExecutionOptions) -> LinalgExecutionOptions {
    order.iter().fold(
        LinalgExecutionOptions::new(),
        |options, field| match field {
            0 => options.with_execution_policy(expected.execution_policy()),
            1 => options.with_byte_limit(expected.byte_limit().unwrap()),
            2 => options.with_time_limit(expected.time_limit().unwrap()),
            _ => options.with_iteration_budget(expected.iteration_budget().unwrap()),
        },
    )
}

fn controls() -> LinalgExecutionOptions {
    LinalgExecutionOptions::new()
        .with_execution_policy(ExecutionPolicy::Sequential)
        .with_byte_limit(1 << 24)
        .with_time_limit(Duration::from_secs(30))
        .with_iteration_budget(Budget::ONE.saturating_add(63))
}

#[test]
fn every_control_order_composes_identically_in_each_family() {
    for policy in [
        ExecutionPolicy::Auto,
        ExecutionPolicy::Sequential,
        ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(2).unwrap()),
    ] {
        let expected = controls().with_execution_policy(policy);
        permutations(&mut [0, 1, 2, 3], 0, &mut |order| {
            let execution = reordered(order, expected);
            assert_eq!(execution, expected); // SVD and CSD take this value directly.
            let rank = RankPolicy::<f64>::Threshold(RankThreshold::new(0.0, 1e-6).unwrap());
            let gsvd = GsvdOptions::new(rank).with_execution_options(execution);
            let polar = PolarOptions::new(rank).with_execution_options(execution);
            let qr = QrUpdateOptions::new(rank).with_execution_options(execution);
            assert_eq!(gsvd.execution_options(), expected);
            assert_eq!(polar.execution_options(), expected);
            assert_eq!(qr.execution_options(), expected);
            assert_eq!(gsvd.rank_policy(), rank);
            assert_eq!(polar.rank_policy(), rank);
            assert_eq!(qr.rank_policy(), rank);
            assert_eq!(
                RiccatiOptions::<f64>::new()
                    .with_execution_options(execution)
                    .execution_options(),
                expected
            );
        });
    }
}

#[test]
fn every_riccati_setter_order_preserves_controls_and_verification() {
    let tolerance = ResidualTolerance::new(1e-8, 1e-7).unwrap();
    let stability = Tolerance::new(1e-12, 1e-10).unwrap();
    for verification in [Estimation::Skip, Estimation::Estimate] {
        let expected = RiccatiOptions::new()
            .with_execution_options(controls())
            .with_tolerance(tolerance)
            .with_stability_tolerance(stability)
            .with_balancing(RiccatiBalancing::None)
            .with_closed_loop_verification(verification);
        permutations(&mut [0, 1, 2, 3, 4], 0, &mut |order| {
            let actual = order
                .iter()
                .fold(RiccatiOptions::new(), |options, field| match field {
                    0 => options.with_execution_options(controls()),
                    1 => options.with_tolerance(tolerance),
                    2 => options.with_stability_tolerance(stability),
                    3 => options.with_balancing(RiccatiBalancing::None),
                    _ => options.with_closed_loop_verification(verification),
                });
            assert_eq!(actual, expected);
        });
    }
}

use crate::support::clocks::Frozen;

#[derive(Clone, Copy, Debug)]
enum Family {
    Svd,
    Csd,
    Gsvd,
    Polar,
    Qr,
    Riccati,
}

fn run(
    family: Family,
    controls: LinalgExecutionOptions,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let a = Matrix::<f64>::identity(2)?;
    let mut workspace = LinalgWorkspace::new();
    match family {
        Family::Svd => {
            let mut out = SvdValues::empty();
            SvdValues::compute_into(&a, &controls, &mut out, &mut workspace, context)?;
            assert!(
                out.singular_values()
                    .iter()
                    .all(|v| (v - 1.0).abs() < 1e-14)
            );
        }
        Family::Csd => {
            let a = Matrix::<f64>::identity(4)?;
            let mut out = CsdDecomposition::empty();
            CsdDecomposition::factor_into(&a, 2, 2, &controls, &mut out, &mut workspace, context)?;
            assert!(out.report().unwrap().residual() < 1e-14);
        }
        Family::Gsvd => {
            let mut out = GsvdDecomposition::empty();
            let options = GsvdOptions::new(RankPolicy::Exact).with_execution_options(controls);
            GsvdDecomposition::factor_into(&a, &a, &options, &mut out, &mut workspace, context)?;
            assert!(out.report().unwrap().residual_a() < 1e-14);
        }
        Family::Polar => {
            let mut out = PolarDecomposition::empty();
            let options = PolarOptions::new(RankPolicy::Exact).with_execution_options(controls);
            PolarDecomposition::factor_into(
                &a,
                PolarSide::Right,
                &options,
                &mut out,
                &mut workspace,
                context,
            )?;
            assert!(out.report().unwrap().residual() < 1e-14);
        }
        Family::Qr => {
            let mut out = QrUpdateDecomposition::empty();
            let options = QrUpdateOptions::new(RankPolicy::Exact).with_execution_options(controls);
            QrUpdateDecomposition::factor_into(&a, &options, &mut out, &mut workspace, context)?;
            assert_eq!(out.rank_decision().report().rank(), 2);
        }
        Family::Riccati => {
            let options = RiccatiOptions::new()
                .with_execution_options(controls)
                .with_closed_loop_verification(Estimation::Estimate);
            let problem = RiccatiProblem {
                a: a.view(),
                b: a.view(),
                q: a.view(),
                r: a.view(),
                e: None,
                s: None,
            };
            let plan = RiccatiPlan::build(problem, RiccatiKind::Continuous, options, context)?;
            let mut x = Matrix::zeros(2, 2)?;
            let mut gain = Matrix::zeros(2, 2)?;
            let report = plan.solve_into(x.view_mut(), gain.view_mut(), &mut workspace, context)?;
            assert_eq!(report.termination(), DirectSolveTermination::Accepted);
            assert!(report.stability_margin().is_some());
            assert!((x.view().get(0, 0)? - (1.0 + 2.0_f64.sqrt())).abs() < 1e-12);
        }
    }
    Ok(())
}

#[test]
fn every_control_order_enforces_limits_at_execution() {
    let binding = ClockBinding::new(&Frozen);
    let timed = ExecutionContext::sequential().with_clock(&binding).unwrap();
    for family in [
        Family::Svd,
        Family::Csd,
        Family::Gsvd,
        Family::Polar,
        Family::Qr,
        Family::Riccati,
    ] {
        permutations(&mut [0, 1, 2, 3], 0, &mut |order| {
            run(family, reordered(order, controls()), timed).unwrap();
            for (controls, context, expected) in [
                (
                    controls(),
                    ExecutionContext::sequential(),
                    SciOxideErrorKind::CapabilityUnavailable,
                ),
                (
                    controls().with_byte_limit(0),
                    timed,
                    SciOxideErrorKind::ResourceLimit,
                ),
                (
                    controls().with_time_limit(Duration::ZERO),
                    timed,
                    SciOxideErrorKind::ExecutionStopped,
                ),
            ] {
                assert_eq!(
                    run(family, reordered(order, controls), context)
                        .unwrap_err()
                        .kind(),
                    expected,
                    "{family:?} {order:?}"
                );
            }
        });
    }
}

#[test]
fn jacobi_allowance_counts_preparation_and_the_final_sweep() {
    let binding = ClockBinding::new(&Frozen);
    let context = ExecutionContext::sequential().with_clock(&binding).unwrap();
    for family in [Family::Gsvd, Family::Csd] {
        assert_eq!(
            run(
                family,
                controls().with_iteration_budget(Budget::ONE),
                context
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
        run(
            family,
            controls().with_iteration_budget(Budget::ONE.saturating_add(1)),
            context,
        )
        .unwrap();
    }
}

#[test]
fn nested_svd_limits_preserve_outputs_and_allow_retry() {
    let a = Matrix::from_vec(
        4,
        4,
        vec![
            4.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.5, 1.0, 2.0, 0.5, 5.0, -1.0, 0.0, 1.0, -1.0, 2.0,
        ],
    )
    .unwrap();
    let limited = LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE);
    let options = PolarOptions::new(RankPolicy::Exact);
    let mut out = PolarDecomposition::factor(&a, PolarSide::Right).unwrap();
    let before = out.u_matrix().clone();
    let mut workspace = LinalgWorkspace::new();
    assert_eq!(
        PolarDecomposition::factor_into(
            &a,
            PolarSide::Right,
            &options.with_execution_options(limited),
            &mut out,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(out.u_matrix(), &before);
    assert_eq!(
        workspace
            .last_attempt()
            .unwrap()
            .counts()
            .nested()
            .iterations(),
        1
    );
    PolarDecomposition::factor_into(
        &a,
        PolarSide::Right,
        &options,
        &mut out,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(out.report().unwrap().residual() < 1e-12);

    let options = GsvdOptions::new(RankPolicy::Exact);
    let mut out = GsvdDecomposition::factor(&a, &a).unwrap();
    let before = out.r_matrix().clone();
    assert_eq!(
        GsvdDecomposition::factor_into(
            &a,
            &a,
            &options.with_execution_options(limited),
            &mut out,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(out.r_matrix(), &before);
    assert_eq!(
        workspace
            .last_attempt()
            .unwrap()
            .counts()
            .nested()
            .iterations(),
        1
    );
    GsvdDecomposition::factor_into(
        &a,
        &a,
        &options,
        &mut out,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(out.report().unwrap().residual_a() < 1e-12);

    let options =
        QrUpdateOptions::new(RankPolicy::Threshold(RankThreshold::new(0.0, 0.5).unwrap()));
    let mut out = QrUpdateDecomposition::factor_with_options(&a, &options).unwrap();
    assert_eq!(out.rank_decision().source(), QrRankSource::SingularValues);
    let before = out.r_matrix().clone();
    assert_eq!(
        QrUpdateDecomposition::factor_into(
            &a,
            &options.with_execution_options(limited),
            &mut out,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(out.r_matrix(), &before);
    assert_eq!(
        workspace
            .last_attempt()
            .unwrap()
            .counts()
            .nested()
            .iterations(),
        1
    );
    QrUpdateDecomposition::factor_into(
        &a,
        &options,
        &mut out,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(out.rank_decision().source(), QrRankSource::SingularValues);
}

#[test]
fn sparse_and_action_settings_preserve_the_single_controls_value() {
    use scioxide_linalg::*;
    let expected = controls().with_accuracy(Accuracy::Exact);
    let settings = MatrixFunctionSettings::<f64>::new()
        .with_tolerance(Tolerance::new(0.0, 1e-10).unwrap())
        .with_series_limit(Budget::try_from(32).unwrap());
    for before in [true, false] {
        let function = if before {
            FunctionAction::default()
                .with_execution_options(expected)
                .with_function_settings(settings)
        } else {
            FunctionAction::default()
                .with_function_settings(settings)
                .with_execution_options(expected)
        };
        let sparse = if before {
            SparseFunctionOptions::new()
                .with_execution_options(expected)
                .with_function_settings(settings)
                .with_max_entries(32)
        } else {
            SparseFunctionOptions::new()
                .with_max_entries(32)
                .with_function_settings(settings)
                .with_execution_options(expected)
        };
        assert_eq!(function.execution_options(), expected);
        assert_eq!(sparse.execution_options(), expected);
        assert_eq!(function.function_settings(), settings);
        assert_eq!(sparse.function_settings(), settings);
    }
    assert_eq!(
        SparsePowerOptions::new()
            .with_execution_options(expected)
            .with_max_entries(32)
            .execution_options(),
        expected
    );
    assert_eq!(
        ExponentialAction::<f64>::new(1.0)
            .with_execution_options(expected)
            .with_max_phi(2)
            .execution_options(),
        expected
    );
    assert_eq!(
        PartialSpectrumSolver::<f64>::new(1)
            .with_execution_options(expected)
            .with_basis(Budget::try_from(4).unwrap())
            .execution_options(),
        expected
    );
    assert_eq!(
        PartialSvdSolver::<f64>::new(1)
            .with_execution_options(expected)
            .with_basis(Budget::try_from(4).unwrap())
            .execution_options(),
        expected
    );
    assert_eq!(
        LobpcgSolver::<f64>::new(1)
            .with_execution_options(expected)
            .with_selection(LobpcgSelection::Largest)
            .execution_options(),
        expected
    );
    assert_eq!(
        RandomizedLowRank::<f64>::new(1)
            .with_execution_options(expected)
            .execution_options(),
        expected
    );
    assert_eq!(
        OneNormEstimator::new()
            .with_execution_options(expected)
            .with_applications(Budget::ONE)
            .execution_options(),
        expected
    );
}

#[test]
fn sparse_controls_enforce_outer_limits_and_allow_retry() {
    use scioxide_linalg::{
        MatrixFunction, SparseFunctionOptions, SparseFunctionPlan, SparsePowerOptions,
        SparsePowerPlan,
    };
    let matrix =
        scioxide_sparse::CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0])
            .unwrap();
    for function in [false, true] {
        let run = |execution: LinalgExecutionOptions,
                   context: ExecutionContext<'_>|
         -> Result<(), SciOxideError> {
            if function {
                let options = SparseFunctionOptions::new().with_execution_options(execution);
                let plan = SparseFunctionPlan::new(matrix.view(), MatrixFunction::Exp, &options)?;
                let mut output = plan.create_output::<f64>()?;
                let mut workspace = plan.workspace(())?;
                plan.evaluate_into(matrix.view(), &mut output, &mut workspace, context)?;
                assert!((output.values()[0] - 1.0_f64.exp()).abs() < 1e-12);
            } else {
                let options = SparsePowerOptions::new().with_execution_options(execution);
                let plan = SparsePowerPlan::new(matrix.view(), 3, &options)?;
                let mut output = plan.create_output()?;
                let mut workspace = plan.workspace(())?;
                plan.evaluate_into(matrix.view(), &mut output, &mut workspace, context)?;
                assert!((output.values()[1] - 8.0).abs() < 1e-12);
            }
            Ok(())
        };
        let context = ExecutionContext::sequential();
        assert_eq!(
            run(LinalgExecutionOptions::new().with_byte_limit(16), context)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            run(
                LinalgExecutionOptions::new().with_time_limit(Duration::from_secs(1)),
                context
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::CapabilityUnavailable
        );
        let binding = ClockBinding::new(&Frozen);
        let context = context.with_clock(&binding).unwrap();
        assert_eq!(
            run(
                LinalgExecutionOptions::new().with_time_limit(Duration::ZERO),
                context
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::ExecutionStopped
        );
        run(
            LinalgExecutionOptions::new().with_time_limit(Duration::from_secs(1)),
            context,
        )
        .unwrap();
    }
}

#[test]
fn newly_composed_families_round_trip_controls_without_touching_settings() {
    use scioxide_linalg::{
        BicgSolver, BicgstabSolver, BlockGcrotSolver, BlockGmresSolver, BlockLgmresSolver,
        BorderedSystemOptions, CgSolver, CgsSolver, CompositeSolveOptions, GaussSeidelSolver,
        GcrotSolver, GmresSolver, JacobiSolver, LgmresSolver, LsmrSolver, LsqrSolver, MinresSolver,
        QmrSolver, RefinementCriterion, RefinementOptions, RelaxationWeight, RichardsonSolver,
        SorSolver, SorWeight, TfqmrSolver,
    };
    let expected = controls();
    let (t, b) = (
        ResidualTolerance::<f64>::new(0., 1e-10).unwrap(),
        Budget::ONE.saturating_add(9),
    );
    macro_rules! solvers {
        ($($solver:expr),+ $(,)?) => {
            $(assert_eq!($solver.with_execution_options(expected).execution_options(), expected);)+
        };
    }
    solvers!(
        CgSolver::<f64>::new(t, b),
        BicgstabSolver::<f64>::new(t, b),
        GmresSolver::<f64>::new(t, b),
        LgmresSolver::<f64>::new(t, b),
        GcrotSolver::<f64>::new(t, b),
        BlockGmresSolver::<f64>::new(t, b),
        BlockLgmresSolver::<f64>::new(t, b),
        BlockGcrotSolver::<f64>::new(t, b),
        MinresSolver::<f64>::new(t, b),
        LsqrSolver::<f64>::new(t, b),
        LsmrSolver::<f64>::new(t, b),
        BicgSolver::<f64>::new(t, b),
        CgsSolver::<f64>::new(t, b),
        QmrSolver::<f64>::new(t, b),
        TfqmrSolver::<f64>::new(t, b),
        RichardsonSolver::<f64>::new(t, b, RelaxationWeight::new(0.5).unwrap()),
        JacobiSolver::<f64>::new(t, b),
        GaussSeidelSolver::<f64>::new(t, b),
        SorSolver::<f64>::new(t, b, SorWeight::new(1.5).unwrap()),
    );
    let refinement = RefinementOptions::new(t)
        .with_execution_options(expected)
        .with_criterion(RefinementCriterion::Componentwise);
    assert_eq!(refinement.execution_options(), expected);
    assert_eq!(refinement.criterion(), RefinementCriterion::Componentwise);
    assert_eq!(refinement.tolerance(), t);
    let tolerance = Tolerance::new(0., 1e-12).unwrap();
    let composite = CompositeSolveOptions::new()
        .with_execution_options(expected)
        .with_tolerance(tolerance);
    assert_eq!(composite.execution_options(), expected);
    assert_eq!(composite.tolerance(), tolerance);
    let bordered = BorderedSystemOptions::new()
        .with_validation_tolerance(tolerance)
        .with_execution_options(expected);
    assert_eq!(bordered.execution_options(), expected);
    assert_eq!(bordered.validation_tolerance(), tolerance);
}
