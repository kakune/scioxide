//! Public reliability, work-accounting, and allocation contracts for cubature.

#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]

use std::sync::atomic::AtomicUsize as Cell;
use std::sync::atomic::Ordering;

use scioxide_core::{Budget, ErrorTolerance, Interval, NormSelection, SciOxideError};
use scioxide_integrate::{CubaturePlan, IntegrationTermination, SubdivisionOptions};

fn unit_bounds(dimension: usize) -> Vec<Interval<f64>> {
    (0..dimension)
        .map(|_| Interval::new(0.0, 1.0).unwrap())
        .collect()
}

fn zero_tolerance(subdivisions: usize) -> SubdivisionOptions<f64> {
    SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap())
        .with_max_subdivisions(Budget::try_from(subdivisions).unwrap())
}

fn assert_usable(result: &(f64, scioxide_integrate::IntegrationReport<f64>)) {
    assert!(result.0.is_finite());
    assert!(
        result
            .1
            .error_estimate()
            .is_some_and(|error| error.is_finite() && error >= 0.0)
    );
}

#[test]
fn dimension_one_matches_independent_exponential_and_lorentzian_integrals() {
    let interval = Interval::new(0.0_f64, 1.0).unwrap();
    let nd_calls = Cell::new(0_usize);
    let nd = |point: &[f64]| {
        nd_calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(point[0].exp())
    };
    let cubature = scioxide_integrate::CubaturePlan::new(
        &[interval],
        scioxide_integrate::SubdivisionOptions::default(),
    )
    .and_then(|plan| plan.integrate(&nd))
    .unwrap();
    let expected = 1.0_f64.exp() - 1.0;
    let error = (cubature.0 - expected).abs();
    assert!(error <= 4.0 * f64::EPSILON * expected);
    assert!(error <= cubature.1.error_estimate().unwrap());
    assert_eq!(cubature.1.evaluations(), 15);
    assert_eq!(nd_calls.load(core::sync::atomic::Ordering::Relaxed), 15);

    let options = zero_tolerance(2);
    nd_calls.store(0, Ordering::Relaxed);
    let peaked_nd = |point: &[f64]| {
        nd_calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.0 / 100.0_f64.mul_add((point[0] - 0.3).powi(2), 1.0))
    };
    let cubature_adaptive = scioxide_integrate::CubaturePlan::new(&[interval], options)
        .and_then(|plan| plan.integrate(&peaked_nd))
        .unwrap();
    // Antiderivative atan(10(x-.3))/10; no SciOxide integration is the oracle.
    let expected = (7.0_f64.atan() + 3.0_f64.atan()) / 10.0;
    assert!(
        (cubature_adaptive.0 - expected).abs() <= cubature_adaptive.1.error_estimate().unwrap()
    );
    assert_eq!(
        cubature_adaptive.1.evaluations(),
        nd_calls.load(Ordering::Relaxed)
    );
    assert_eq!(
        cubature_adaptive.1.termination(),
        IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Subdivision)
    );
}

#[test]
fn multidimensional_base_and_split_callback_costs_are_exact() {
    for (dimension, base_cost) in [(2_usize, 17_usize), (3, 33), (4, 57)] {
        let calls = Cell::new(0_usize);
        let constant = |_point: &[f64]| {
            calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Ok(1.0)
        };
        let initial = scioxide_integrate::CubaturePlan::new(
            &unit_bounds(dimension),
            scioxide_integrate::SubdivisionOptions::default(),
        )
        .and_then(|plan| plan.integrate(&constant))
        .unwrap();
        assert_eq!(initial.1.evaluations(), base_cost);
        assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), base_cost);

        calls.store(0, core::sync::atomic::Ordering::Relaxed);
        let polynomial = |point: &[f64]| {
            calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Ok(point[0].powi(8))
        };
        let split =
            scioxide_integrate::CubaturePlan::new(&unit_bounds(dimension), zero_tolerance(1))
                .and_then(|plan| plan.integrate(&polynomial))
                .unwrap();
        assert_eq!(split.1.evaluations(), base_cost * 3);
        assert_eq!(
            calls.load(core::sync::atomic::Ordering::Relaxed),
            base_cost * 3
        );
        assert_eq!(
            split.1.termination(),
            IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Subdivision)
        );
        assert_usable(&split);
    }
}

#[test]
fn smooth_anisotropic_localized_and_cancelling_integrals_converge() {
    let bounds = unit_bounds(2);
    let smooth = |point: &[f64]| Ok((point[0] + point[1]).exp());
    let smooth_result = scioxide_integrate::CubaturePlan::new(
        &bounds,
        scioxide_integrate::SubdivisionOptions::default(),
    )
    .and_then(|plan| plan.integrate(&smooth))
    .unwrap();
    let smooth_expected = (std::f64::consts::E - 1.0).powi(2);
    assert!((smooth_result.0 - smooth_expected).abs() < 1.0e-8);
    assert_usable(&smooth_result);

    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(1.0e-7, 1.0e-7, NormSelection::Infinity).unwrap())
        .with_max_subdivisions(Budget::try_from(500).unwrap());
    let anisotropic = |point: &[f64]| {
        let offset = point[0] - 0.2;
        Ok((1.0 + point[1]) / 400.0_f64.mul_add(offset * offset, 1.0))
    };
    let anisotropic_expected = 1.5 * (16.0_f64.atan() + 4.0_f64.atan()) / 20.0;
    let anisotropic_result = scioxide_integrate::CubaturePlan::new(&bounds, options.clone())
        .and_then(|plan| plan.integrate(&anisotropic))
        .unwrap();
    assert!(
        (anisotropic_result.0 - anisotropic_expected).abs() < 3.0e-7,
        "value={} expected={} error={:?} termination={:?} evals={}",
        anisotropic_result.0,
        anisotropic_expected,
        anisotropic_result.1.error_estimate(),
        anisotropic_result.1.termination(),
        anisotropic_result.1.evaluations()
    );
    assert_usable(&anisotropic_result);

    let localized_value = |point: &[f64]| {
        let x = 1.0 / 400.0_f64.mul_add((point[0] - 0.2).powi(2), 1.0);
        let y = 1.0 / 400.0_f64.mul_add((point[1] - 0.7).powi(2), 1.0);
        x * y
    };
    let localized = |point: &[f64]| Ok(localized_value(point));
    let localized_expected =
        (16.0_f64.atan() + 4.0_f64.atan()) / 20.0 * ((6.0_f64.atan() + 14.0_f64.atan()) / 20.0);
    let localized_result = scioxide_integrate::CubaturePlan::new(&bounds, options.clone())
        .and_then(|plan| plan.integrate(&localized))
        .unwrap();
    assert!((localized_result.0 - localized_expected).abs() < 1.0e-7);
    assert_usable(&localized_result);

    let cancelling = |point: &[f64]| Ok(1.0e8 * (point[0] - 0.5) + localized_value(point));
    let cancelling_result = scioxide_integrate::CubaturePlan::new(&bounds, options)
        .and_then(|plan| plan.integrate(&cancelling))
        .unwrap();
    assert!((cancelling_result.0 - localized_expected).abs() < 2.0e-6);
    assert_usable(&cancelling_result);
}

#[test]
fn f32_and_erased_callbacks_follow_the_same_schedule() {
    let bounds = [
        Interval::new(0.0_f32, 1.0).unwrap(),
        Interval::new(0.0_f32, 1.0).unwrap(),
    ];
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(0.0_f32, 0.0, NormSelection::Infinity).unwrap())
        .with_max_subdivisions(Budget::ONE);
    let function = |point: &[f32]| Ok(point[0].powi(8));
    let dynamic: &(dyn Fn(&[f32]) -> Result<f32, SciOxideError> + Sync) = &function;
    let cubature = scioxide_integrate::CubaturePlan::new(&bounds, options).unwrap();
    let static_result = cubature.integrate(&function).unwrap();
    let dynamic_result = cubature.integrate(dynamic).unwrap();
    assert_eq!(static_result.0, dynamic_result.0);
    assert_eq!(
        static_result.1.error_estimate(),
        dynamic_result.1.error_estimate()
    );
    assert_eq!(static_result.1.evaluations(), 51);
    assert_eq!(
        static_result.1.evaluations(),
        dynamic_result.1.evaluations()
    );
    assert_eq!(
        static_result.1.termination(),
        dynamic_result.1.termination()
    );
}

#[test]
fn narrow_and_unrepresentable_volumes_are_distinguished_before_callbacks() {
    let narrow_calls = Cell::new(0_usize);
    let narrow = |_point: &[f64]| {
        narrow_calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(1.0)
    };
    let narrow_bounds = [
        Interval::new(0.0, 1.0e-150).unwrap(),
        Interval::new(0.0, 1.0e-150).unwrap(),
    ];
    let narrow_result = scioxide_integrate::CubaturePlan::new(
        &narrow_bounds,
        scioxide_integrate::SubdivisionOptions::default(),
    )
    .and_then(|plan| plan.integrate(&narrow))
    .unwrap();
    assert!((narrow_result.0 - 1.0e-300).abs() <= 1.0e-313);
    assert_eq!(narrow_calls.load(core::sync::atomic::Ordering::Relaxed), 17);
    assert_usable(&narrow_result);

    let underflow_calls = Cell::new(0_usize);
    let underflow = |_point: &[f64]| {
        underflow_calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(1.0)
    };
    let underflow_bounds = [
        Interval::new(0.0, 1.0e-200).unwrap(),
        Interval::new(0.0, 1.0e-200).unwrap(),
    ];
    let underflow_result = scioxide_integrate::CubaturePlan::new(
        &underflow_bounds,
        scioxide_integrate::SubdivisionOptions::default(),
    )
    .and_then(|plan| plan.integrate(&underflow))
    .unwrap();
    assert_eq!(underflow_result.0, 0.0);
    assert_eq!(
        underflow_calls.load(core::sync::atomic::Ordering::Relaxed),
        17
    );

    let overflow_calls = Cell::new(0_usize);
    let overflow = |_point: &[f64]| {
        overflow_calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(1.0)
    };
    let overflow_bounds = [
        Interval::new(0.0, 1.0e200).unwrap(),
        Interval::new(0.0, 1.0e200).unwrap(),
    ];
    assert!(matches!(
        (scioxide_integrate::CubaturePlan::new(
            &overflow_bounds,
            scioxide_integrate::SubdivisionOptions::default()
        )
        .and_then(|plan| plan.integrate(&overflow)))
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    assert_eq!(
        overflow_calls.load(core::sync::atomic::Ordering::Relaxed),
        17
    );

    let mixed_bounds =
        [1.0e200, 1.0e200, 1.0e-200, 1.0e-200].map(|width| Interval::new(0.0, width).unwrap());
    let mixed = |_point: &[f64]| Ok(1.0);
    let mixed_result = scioxide_integrate::CubaturePlan::new(
        &mixed_bounds,
        scioxide_integrate::SubdivisionOptions::default(),
    )
    .and_then(|plan| plan.integrate(&mixed))
    .unwrap();
    assert!((mixed_result.0 - 1.0).abs() < 5.0e-14);

    let calls = Cell::new(0_usize);
    let one_dimensional = |_point: &[f64]| {
        calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(1.0)
    };
    let smallest = f64::from_bits(1);
    let collapsed = scioxide_integrate::CubaturePlan::new(
        &[Interval::new(0.0, smallest).unwrap()],
        scioxide_integrate::SubdivisionOptions::default(),
    )
    .and_then(|plan| plan.integrate(&one_dimensional))
    .unwrap_err();
    assert!(matches!(
        collapsed.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Breakdown,
            ..
        }
    ));
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn invalid_dimensions_and_checked_cost_overflow_precede_callbacks() {
    let calls = Cell::new(0);
    let function = |_: &[f64]| {
        calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok::<f64, SciOxideError>(1.0)
    };
    let dimension = usize::BITS as usize - 1;
    let error = CubaturePlan::new(&unit_bounds(dimension), SubdivisionOptions::default())
        .and_then(|plan| plan.integrate(&function))
        .unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::SizeOverflow { .. }
        ),
        "{error:?}"
    );
    let error = CubaturePlan::new(
        &unit_bounds(usize::BITS as usize),
        SubdivisionOptions::default(),
    )
    .and_then(|plan| plan.integrate(&function))
    .unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ),
        "{error:?}"
    );
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn callback_failure_propagates_at_the_exact_sample() {
    let calls = Cell::new(0_usize);
    let function = |point: &[f64]| {
        let call = calls.load(core::sync::atomic::Ordering::Relaxed) + 1;
        calls.store(call, core::sync::atomic::Ordering::Relaxed);
        if call == 18 {
            Err(SciOxideError::domain_error(
                scioxide_core::argument!("point"),
                scioxide_core::DomainReason::Undefined,
                "sentinel split failure",
            ))
        } else {
            Ok(point[0].powi(8))
        }
    };
    assert!(matches!(
        (scioxide_integrate::CubaturePlan::new(&unit_bounds(2), zero_tolerance(2))
            .and_then(|plan| plan.integrate(&function)))
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
    assert_eq!(calls.load(core::sync::atomic::Ordering::Relaxed), 18);
}

#[test]
fn evaluation_and_subdivision_limits_keep_complete_measured_results() {
    let bounds = unit_bounds(2);
    for budget in 17..=84 {
        let calls = Cell::new(0_usize);
        let function = |point: &[f64]| {
            calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Ok(point[0].powi(8))
        };
        let options =
            zero_tolerance(20).with_max_evaluations(Some(Budget::try_from(budget).unwrap()));
        let result = scioxide_integrate::CubaturePlan::new(&bounds, options)
            .and_then(|plan| plan.integrate(&function))
            .unwrap();
        let expected = 17 + 34 * ((budget - 17) / 34);
        assert_eq!(result.1.evaluations(), expected, "budget {budget}");
        assert_eq!(
            result.1.evaluations(),
            calls.load(core::sync::atomic::Ordering::Relaxed)
        );
        assert_eq!(
            result.1.termination(),
            IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Evaluation)
        );
        assert!((result.0 - 1.0 / 9.0).abs() < 2.0e-4);
        assert_usable(&result);
    }

    for subdivisions in 1..=4 {
        let function = |point: &[f64]| Ok(point[0].powi(8));
        let result = scioxide_integrate::CubaturePlan::new(&bounds, zero_tolerance(subdivisions))
            .and_then(|plan| plan.integrate(&function))
            .unwrap();
        assert_eq!(result.1.evaluations(), 17 + 34 * subdivisions);
        assert_eq!(
            result.1.termination(),
            IntegrationTermination::BudgetLimit(scioxide_core::BudgetKind::Subdivision)
        );
        assert!((result.0 - 1.0 / 9.0).abs() < 2.0e-4);
        assert_usable(&result);
    }

    let too_small = zero_tolerance(1).with_max_evaluations(Some(Budget::try_from(16).unwrap()));
    let function = |point: &[f64]| Ok(point[0].powi(8));
    let result = scioxide_integrate::CubaturePlan::new(&bounds, too_small)
        .and_then(|plan| plan.integrate(&function))
        .unwrap_err();
    assert!(matches!(
        result.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::EvaluationLimit,
            ..
        }
    ));
}

#[test]
fn multidimensional_roundoff_stops_once_values_settle_at_the_floor() {
    // Below `50·ε·Σ|w7|·|I|` a tolerance is unreachable. The guarded degree-7/5
    // difference would need about 2,000 (d = 2) and 32,000 (d = 3) splits to
    // reach the floor, although the degree-7 values settle long before.
    for (dimension, limit) in [(2, 2_500), (3, 40_000)] {
        let exact = (1f64.exp() - 1.).powi(i32::try_from(dimension).unwrap());
        for relative in [1e-15, 1e-17] {
            let options = SubdivisionOptions::default()
                .with_tolerance(
                    ErrorTolerance::scalar(0.0, relative, NormSelection::Infinity).unwrap(),
                )
                .with_max_subdivisions(Budget::try_from(100_000).unwrap());
            let (value, report) = CubaturePlan::new(&unit_bounds(dimension), options)
                .unwrap()
                .integrate(|x: &[f64]| Ok(x.iter().sum::<f64>().exp()))
                .unwrap();
            assert_eq!(
                report.termination(),
                IntegrationTermination::Stagnation,
                "{report:?}"
            );
            assert!(report.evaluations() < limit, "{report:?}");
            let evidence = report.error_components().unwrap();
            let error = (value - exact).abs();
            assert!(
                error <= 4.0 * evidence.roundoff().unwrap(),
                "{error:e} {report:?}"
            );
            assert!(error <= report.error_estimate().unwrap());
            assert!(report.error_estimate().unwrap() > relative * exact);
        }
    }
    // A tolerance at the floor bound itself is unreachable as well.
    let options = SubdivisionOptions::<f32>::default()
        .with_tolerance(ErrorTolerance::scalar(0.0, 1e-6, NormSelection::Infinity).unwrap());
    let (value, report) = CubaturePlan::new(&[Interval::new(0.0_f32, 1.0).unwrap(); 2], options)
        .unwrap()
        .integrate(|x: &[f32]| Ok((x[0] + x[1]).exp()))
        .unwrap();
    assert_eq!(report.termination(), IntegrationTermination::Stagnation);
    assert!(report.evaluations() < 2_500, "{report:?}");
    let exact = (1f64.exp() - 1.).powi(2);
    assert!((f64::from(value) - exact).abs() <= f64::from(report.error_estimate().unwrap()));
    // Reachable tolerances keep the guarded estimate: no early stop.
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(0.0, 1e-10, NormSelection::Infinity).unwrap())
        .with_max_subdivisions(Budget::try_from(100_000).unwrap());
    let (_, report) = CubaturePlan::new(&unit_bounds(2), options)
        .unwrap()
        .integrate(|x: &[f64]| Ok((x[0] + x[1]).exp()))
        .unwrap();
    assert_eq!(
        report.termination(),
        IntegrationTermination::HeuristicToleranceReached
    );
}

#[test]
fn scalar_cancellation_priority_is_independent_of_local_relative_scales() {
    let bounds = [
        Interval::new(-1.0, 1.0).unwrap(),
        Interval::new(0.0, 1.0).unwrap(),
    ];
    let f =
        |x: &[f64]| Ok((20.0 * x[0]).sin() * (1.0 + x[1]) + (-50.0 * (x[0] - 0.5).powi(2)).exp());
    let solve = |absolute| {
        CubaturePlan::new(
            &bounds,
            SubdivisionOptions::default()
                .with_tolerance(
                    ErrorTolerance::scalar(absolute, 1e-8, NormSelection::Infinity).unwrap(),
                )
                .with_max_subdivisions(Budget::try_from(2000).unwrap()),
        )
        .unwrap()
        .integrate(&f)
        .unwrap()
    };
    let relative = solve(0.0);
    let mixed = solve(1e-8);
    assert!(relative.1.is_success() && mixed.1.is_success());
    // These targets differ: the mixed target is looser. Compare absolute-only
    // control at the relative solve's global target to isolate heap priority.
    let equivalent = CubaturePlan::new(
        &bounds,
        SubdivisionOptions::default()
            .with_tolerance(
                ErrorTolerance::scalar(1e-8 * relative.0.abs(), 0.0, NormSelection::Infinity)
                    .unwrap(),
            )
            .with_max_subdivisions(Budget::try_from(2000).unwrap()),
    )
    .unwrap()
    .integrate(&f)
    .unwrap();
    assert_eq!(relative.1.evaluations(), equivalent.1.evaluations());
    assert_eq!(relative.0.to_bits(), equivalent.0.to_bits());
    assert!((relative.0 - mixed.0).abs() <= 2e-8);
    assert_eq!(
        relative.1.termination(),
        IntegrationTermination::HeuristicToleranceReached
    );
    assert_eq!(
        relative
            .1
            .error_components()
            .unwrap()
            .discretization()
            .unwrap()
            .reliability(),
        scioxide_integrate::ErrorReliability::Heuristic
    );
}

#[test]
fn vector_cubature_uses_the_selected_norm_with_multiple_initial_regions() {
    for norm in [
        NormSelection::L1,
        NormSelection::L2,
        NormSelection::Infinity,
        NormSelection::RootMeanSquare,
    ] {
        let tolerance = ErrorTolerance::try_vector(&[1e-8, 1e-5], 0.0, norm).unwrap();
        let plan = CubaturePlan::new(
            &unit_bounds(2),
            SubdivisionOptions::default()
                .with_tolerance(tolerance.clone())
                .with_max_subdivisions(Budget::try_from(500).unwrap()),
        )
        .unwrap()
        .with_breakpoints(0, &[0.5])
        .unwrap();
        let mut ws = plan.workspace(2).unwrap();
        let mut out = [0.0; 2];
        let report = plan
            .integrate_into(
                |x: &[f64]| Ok([x[0].powi(8), 1000.0 * x[1].powi(8)]),
                &mut out,
                &mut ws,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            IntegrationTermination::HeuristicToleranceReached
        );
        let evidence = ws.component_errors().unwrap();
        let errors = [evidence[0].total().unwrap(), evidence[1].total().unwrap()];
        assert!(tolerance.is_satisfied(&errors, &out.map(f64::abs)).unwrap());
        assert!((out[0] - 1.0 / 9.0).abs() < 1e-8);
        assert!((out[1] - 1000.0 / 9.0).abs() < 1e-5);
        assert!(
            evidence
                .iter()
                .all(|e| e.discretization().unwrap().reliability()
                    == scioxide_integrate::ErrorReliability::Heuristic)
        );
    }
}

#[test]
fn directional_error_plateaus_do_not_trigger_one_dimensional_roundoff_counters() {
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(1e-6, 1e-6, NormSelection::Infinity).unwrap())
        .with_max_subdivisions(Budget::try_from(1000).unwrap());
    let plan = CubaturePlan::new(&unit_bounds(3), options).unwrap();
    for sharpness in [1.0, 2.0, 3.0, 4.0] {
        let (value, report) = plan
            .integrate(|x: &[f64]| {
                Ok((-sharpness * x.iter().map(|x| (x - 0.5).powi(2)).sum::<f64>()).exp())
            })
            .unwrap();
        assert_eq!(
            report.termination(),
            IntegrationTermination::HeuristicToleranceReached,
            "{sharpness}: {report:?}"
        );
        assert!(value.is_finite() && value > 0.0 && value < 1.0);
    }
}
