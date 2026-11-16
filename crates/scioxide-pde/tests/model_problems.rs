//! Default equations, coefficient adapters and progressive state contracts.
#![allow(clippy::unwrap_used)]
use core::f64::consts::PI;
use scioxide_core::{AttemptOutcome, ExecutionContext, LinearOperator, SciOxideErrorView};
use scioxide_pde::*;

fn mode<const D: usize>() -> FieldData<f64, f64, D> {
    FieldData::time_invariant(|p: FieldPoint<f64, D>| {
        Ok(p.point.iter().map(|x| (PI * x).sin()).product())
    })
}

#[test]
fn default_heat_is_accurate_without_a_mesh_dependent_step_count() {
    for nodes in [81, 321, 1281] {
        heat([nodes], 40);
    }
    for nodes in [51, 101] {
        heat([nodes; 2], 60);
    }
}
fn heat<const D: usize>(nodes: [usize; D], limit: usize) {
    let fdm = StructuredFdmPlan::dirichlet_laplacian([0.; D], [1.; D], nodes).unwrap();
    let source = FieldData::zero();
    let initial = mode();
    let problem = ParabolicProblem::new(&fdm, &source, 0.1).unwrap();
    let solution = problem
        .solve_with_options(&initial, &ParabolicOptions::default())
        .unwrap();
    assert!(solution.report().is_success());
    assert!(
        solution.report().diagnostics().num_time_steps() <= limit,
        "{:?}",
        solution.report()
    );
    let decay = (-(D as f64) * PI * PI * 0.1).exp();
    for (i, &value) in solution.values().iter().enumerate() {
        let exact = decay
            * fdm
                .node_point(i)
                .unwrap()
                .iter()
                .map(|x| (PI * x).sin())
                .product::<f64>();
        assert!(
            (value - exact).abs() < 1e-4,
            "{nodes:?}: {value} vs {exact}"
        );
    }
    if nodes[0] == 81 {
        assert_eq!(problem.solve(&initial).unwrap(), solution.values());
    }
}

#[test]
fn default_wave_and_owned_forms() {
    for nodes in [81, 321] {
        let fdm = StructuredFdmPlan::dirichlet_laplacian([0.], [1.], [nodes]).unwrap();
        let source = FieldData::zero();
        let initial = mode();
        let problem = HyperbolicProblem::new(&fdm, &source, 1.).unwrap();
        let solution = problem
            .solve_with_options(&initial, &FieldData::zero(), &HyperbolicOptions::default())
            .unwrap();
        assert!(solution.report().is_success());
        for (i, &u) in solution.values().iter().enumerate() {
            assert!((u + (PI * fdm.node_point(i).unwrap()[0]).sin()).abs() < 1e-3);
        }
        assert_eq!(
            problem.solve(&initial, &FieldData::zero()).unwrap(),
            solution.values()
        );
    }
}

#[test]
fn implicit_factors_admit_two_and_three_dimensional_fill() {
    fn run<const D: usize>(nodes: usize) {
        let fdm = StructuredFdmPlan::dirichlet_laplacian([0.; D], [1.; D], [nodes; D]).unwrap();
        let source = FieldData::zero();
        let options = ParabolicOptions::new()
            .with_scheme(ParabolicScheme::BackwardEuler)
            .with_step(0.01);
        let solution = ParabolicProblem::new(&fdm, &source, 0.01)
            .unwrap()
            .solve_with_options(&mode(), &options)
            .unwrap();
        assert!(solution.report().is_success());
    }
    run::<2>(51);
    run::<3>(21);
}

#[test]
fn box_shortcuts_and_elliptic_defaults() {
    fn check<const D: usize>() {
        let short = StructuredFdmPlan::dirichlet_laplacian([0.; D], [1.; D], [5; D]).unwrap();
        let mesh = short.mesh().clone();
        let boundaries =
            StructuredBoundaries::uniform(&mesh, 1, BoundaryKind::Dirichlet, BoundaryData::zero())
                .unwrap();
        let long = StructuredFdmPlan::new(mesh, &ConstantDiffusion::new(1.), &boundaries).unwrap();
        let variable = StructuredFdmPlan::new(
            short.mesh().clone(),
            &IsotropicDiffusion::new(|_: [f64; D]| Ok(1.)),
            &boundaries,
        )
        .unwrap();
        assert_eq!(
            short.assemble_csr().unwrap().values(),
            long.assemble_csr().unwrap().values()
        );
        assert_eq!(
            short.assemble_csr().unwrap().values(),
            variable.assemble_csr().unwrap().values()
        );
        assert_eq!(short.lumped_mass(), long.lumped_mass());
        let source = FieldData::constant(1.);
        assert_eq!(
            short.load(&source, 0.).unwrap(),
            long.load(&source, 0.).unwrap()
        );
        let problem = EllipticProblem::new(&short, &source);
        assert_eq!(
            problem.solve().unwrap(),
            problem
                .solve_with_options(&EllipticOptions::default())
                .unwrap()
                .values()
        );
        let tiny = StructuredFdmPlan::dirichlet_laplacian([0.; D], [1.; D], [2; D]).unwrap();
        assert!(
            EllipticProblem::new(&tiny, &source)
                .solve()
                .unwrap()
                .iter()
                .all(|v| v.abs() < 1e-14)
        );
    }
    check::<1>();
    check::<2>();
    check::<3>();
    for (lower, upper, nodes) in [(0., 1., 1), (1., 1., 2), (1., 0., 2), (f64::NAN, 1., 2)] {
        assert!(StructuredFdmPlan::dirichlet_laplacian([lower], [upper], [nodes]).is_err());
    }
    let fdm = StructuredFdmPlan::dirichlet_laplacian([0.; 2], [1.; 2], [101; 2]).unwrap();
    let source = FieldData::time_invariant(|p: FieldPoint<f64, 2>| {
        Ok(2. * PI * PI * (PI * p.point[0]).sin() * (PI * p.point[1]).sin())
    });
    for (i, u) in EllipticProblem::new(&fdm, &source)
        .solve()
        .unwrap()
        .iter()
        .enumerate()
    {
        let [x, y] = fdm.node_point(i).unwrap();
        assert!((u - (PI * x).sin() * (PI * y).sin()).abs() < 1e-4);
    }
}

#[test]
fn sampled_states_reconstruction_and_preflight_are_transactional() {
    let fdm = StructuredFdmPlan::dirichlet_laplacian([0.], [1.], [9]).unwrap();
    let source = FieldData::zero();
    let problem = ParabolicProblem::new(&fdm, &source, 0.02).unwrap();
    let options = ParabolicOptions::new();
    let mut samples: Vec<_> = (0..fdm.rows())
        .map(|i| (PI * fdm.node_point(i).unwrap()[0]).sin())
        .collect();
    samples[0] = f64::NAN;
    let mut state = problem
        .initial_state_from_values(&samples, &options)
        .unwrap();
    assert_eq!(
        state.coordinates(),
        problem
            .initial_state(&mode(), &options)
            .unwrap()
            .coordinates()
    );
    assert!(
        problem
            .initial_state_from_values(&samples[..8], &options)
            .is_err()
    );
    samples[4] = f64::NAN;
    assert!(matches!(
        problem
            .initial_state_from_values(&samples, &options)
            .unwrap_err()
            .view(),
        SciOxideErrorView::NonFinite { .. }
    ));
    let original = state.coordinates().to_vec();
    let mut workspace = problem.workspace().unwrap();
    for invalid in [
        ParabolicOptions::new().with_scheme(ParabolicScheme::BackwardEuler),
        options.clone().with_step(0.),
        options.clone().with_step(-1.),
        options.clone().with_step(f64::NAN),
        options
            .clone()
            .with_rannacher_startup(scioxide_core::Budget::ONE),
    ] {
        assert!(
            problem
                .advance(
                    &invalid,
                    &mut state,
                    &mut (),
                    &mut workspace,
                    ExecutionContext::sequential_native()
                )
                .is_err()
        );
        assert_eq!(state.coordinates(), original);
        assert!(matches!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        ));
    }
    let mut emitted = Vec::new();
    let report = problem
        .advance(
            &options,
            &mut state,
            &mut |p: PdeObservation<'_, f64, f64>| {
                emitted = p.values().to_vec();
                Ok(())
            },
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(report.is_success());
    let mut out = vec![0.; fdm.rows()];
    problem.values_into(&state, &mut out).unwrap();
    assert_eq!(out, emitted);
    let other = StructuredFdmPlan::dirichlet_laplacian([0.], [1.], [9]).unwrap();
    assert!(
        ParabolicProblem::new(&other, &source, 0.02)
            .unwrap()
            .values_into(&state, &mut out)
            .is_err()
    );
    assert_eq!(out, emitted);
}

#[test]
fn gauge_errors_name_the_public_argument_and_reject_before_output() {
    let fdm = StructuredFdmPlan::dirichlet_laplacian([0.], [1.], [9]).unwrap();
    let boundary =
        StructuredBoundaries::uniform(fdm.mesh(), 1, BoundaryKind::Natural, BoundaryData::zero())
            .unwrap();
    let neumann =
        StructuredFdmPlan::new(fdm.mesh().clone(), &ConstantDiffusion::new(1.), &boundary).unwrap();
    let source = FieldData::zero();
    let problem = EllipticProblem::new(&neumann, &source);
    for (error, expected) in [
        (
            problem.solve().unwrap_err(),
            scioxide_core::argument!("self"),
        ),
        (
            problem
                .solve_with_options(&EllipticOptions::default())
                .unwrap_err(),
            scioxide_core::argument!("options.gauge"),
        ),
    ] {
        let SciOxideErrorView::InvalidInput { argument, .. } = error.view() else {
            unreachable!()
        };
        assert_eq!(argument, expected);
    }
    let mut out = vec![7.; 9];
    let mut workspace = EllipticWorkspace::new();
    let options = EllipticOptions::new().with_gauge(EllipticGauge::WeightedMean(f64::NAN));
    assert!(
        problem
            .solve_into(
                &options,
                &mut out,
                &mut workspace,
                ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(out, vec![7.; 9]);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
}

#[test]
fn default_limits_and_radau_requested_times() {
    assert_eq!(
        SpatialLimits::default(),
        SpatialLimits::new(usize::MAX, usize::MAX, usize::MAX)
    );
    assert_eq!(
        SimplexFemLimits::default(),
        SimplexFemLimits::new(usize::MAX, usize::MAX, 31, usize::MAX)
    );
    assert_eq!(
        AmrLimits::default(),
        AmrLimits::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX)
    );
    assert_eq!(
        ConservationLimits::default(),
        ConservationLimits::new(usize::MAX, usize::MAX, usize::MAX)
    );
    assert_eq!(ConservationOptions::<f64>::default().cfl(), 0.45);
    assert_eq!(
        ParabolicOptions::<f64>::default().scheme(),
        ParabolicScheme::RadauIia
    );
    assert_eq!(ParabolicOptions::<f64>::default().step(), None);
    assert_eq!(HyperbolicOptions::<f64>::default().initial_step(), None);
    let fdm = StructuredFdmPlan::dirichlet_laplacian([0.], [1.], [9]).unwrap();
    let source = FieldData::zero();
    let options = ParabolicOptions::new()
        .with_step(0.001)
        .with_output_schedule(PdeOutputSchedule::Times(vec![0., 0.01, 0.02]));
    let result = ParabolicProblem::new(&fdm, &source, 0.02)
        .unwrap()
        .solve_with_options(&mode(), &options)
        .unwrap();
    assert!(result.report().is_success());
    assert_eq!(result.trajectory().times(), [0., 0.01, 0.02]);
}
