#![allow(clippy::unwrap_used)]
#[path = "support/mesh.rs"]
mod mesh_fixture;
mod support;

use mesh_fixture::explicit;
use scioxide_core::SciOxideError;
use scioxide_ode::OdeSystem;
use scioxide_pde::{
    BoundaryData, EllipticOptions, EllipticProblem, EllipticSolution, FieldData, FieldPoint,
    ParabolicOptions, ParabolicProblem, ParabolicScheme, SemidiscreteSystem, StructuredFvmPlan,
};
use scioxide_spatial::{Cell, FieldView, SegmentMesh};
use support::{
    Discretization, EllipticPde, End, Ends, Heat, ParabolicPde, Plan1d, Problem, Steady,
    TestScalar, as_f64, elliptic_options, initial, scalar, source, value,
};

/// Prepared scalar finite volumes of a problem.
fn fvm<S: TestScalar, P: Problem<S>>(
    mesh: &SegmentMesh<S>,
    problem: &P,
    ends: &Ends<S>,
) -> Result<StructuredFvmPlan<S, S, 1>, SciOxideError> {
    match Discretization::Fvm.plan(mesh, problem, ends)? {
        Plan1d::Fvm(plan) => Ok(plan),
        _ => unreachable!("an FVM plan was requested"),
    }
}

/// Prepared scalar finite volumes of a steady problem.
fn fvm_plan<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    mesh: &SegmentMesh<S>,
    problem: &P,
    ends: &Ends<S>,
) -> Result<StructuredFvmPlan<S, S, 1>, SciOxideError> {
    fvm(mesh, &Steady(problem.clone()), ends)
}

/// Steady FVM solve; the solution values are cell averages in cell order.
fn solve_steady<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    mesh: &SegmentMesh<S>,
    problem: &P,
    ends: &Ends<S>,
    options: EllipticOptions<S>,
) -> Result<EllipticSolution<S>, SciOxideError> {
    let plan = fvm_plan(mesh, problem, ends)?;
    let source = source(&Steady(problem.clone()));
    EllipticProblem::new(&plan, &source).solve_with_options(&(options))
}

fn constant<S: TestScalar>(g: f64) -> BoundaryData<S, S, 1> {
    value(move |_| Ok(scalar(g)))
}

/// Integrated face fluxes `-k du/dx` at time zero, oriented toward increasing x.
fn fluxes<S: TestScalar>(
    plan: &StructuredFvmPlan<S, S, 1>,
    cells: FieldView<'_, S, Cell, 1>,
) -> Vec<S> {
    plan.face_fluxes::<0>(cells, S::zero_value())
        .unwrap()
        .values()
        .to_vec()
}

fn jump_material<S: TestScalar>() {
    #[derive(Clone)]
    struct Jump(bool);
    impl<S: TestScalar> EllipticPde<S> for Jump {
        fn diffusion(&self, x: &[S]) -> Result<S, SciOxideError> {
            if x[0] == scalar::<S>(0.5) {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("x"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "interface has no owned material",
                ));
            }
            Ok(scalar(if x[0] < scalar(0.5) { 1. } else { 100. }))
        }
        fn reaction(&self, _: &[S]) -> Result<S, SciOxideError> {
            Ok(S::zero_value())
        }
        fn source(&self, _: &[S]) -> Result<S, SciOxideError> {
            Ok(scalar(if self.0 { -2. } else { 0. }))
        }
    }
    let exact = |x: f64, quadratic: bool| {
        let p = if quadratic { x * x } else { x };
        if x < 0.5 {
            p
        } else {
            (if quadratic { 0.25 } else { 0.5 }) * 0.99 + p / 100.
        }
    };
    let mut errors = Vec::new();
    for n in [10, 20, 40] {
        let mesh = explicit((0..=n).map(|i| scalar::<S>(i as f64 / n as f64)).collect());
        for quadratic in [false, true] {
            let ends = Ends::new(
                End::dirichlet(constant(0.0)),
                End::dirichlet(constant(exact(1., quadratic))),
            );
            let solution =
                solve_steady(&mesh, &Jump(quadratic), &ends, elliptic_options::<S>()).unwrap();
            assert!(solution.report().is_success(), "{}", solution.report());
            let values = solution.values();
            let error = values
                .iter()
                .enumerate()
                .map(|(i, &u)| (as_f64(u) - exact((i as f64 + 0.5) / n as f64, quadratic)).abs())
                .fold(0., f64::max);
            if quadratic {
                errors.push(error);
            } else {
                assert!(
                    error < 512. * as_f64(S::EPSILON),
                    "piecewise linear error {error}"
                );
                let plan = fvm_plan(&mesh, &Jump(false), &ends).unwrap();
                let field = FieldView::<_, Cell, 1>::new(&mesh, values).unwrap();
                let flux_error = fluxes(&plan, field)
                    .iter()
                    .map(|&q| (as_f64(q) + 1.).abs())
                    .fold(0., f64::max);
                // Flux differences amplify solution rounding by the largest
                // conductance, 100*n, in this deliberately contrasted material.
                assert!(
                    flux_error < 128. * n as f64 * 100. * as_f64(S::EPSILON),
                    "flux error {flux_error}"
                );
            }
        }
    }
    for pair in errors.windows(2) {
        assert!((pair[0] / pair[1]).log2() > 1.7, "{errors:?}");
    }
}

#[test]
fn cell_owned_material_jump_is_exact_or_second_order() {
    jump_material::<f64>();
    jump_material::<f32>();
}

// Constant material makes the affine patch exact on arbitrary cell widths.
// Smooth materials use harmonic midpoint approximation (refinement above),
// while the discontinuous stationary patch below checks cell-owned sampling.
#[derive(Clone)]
struct Stationary;
impl<S: TestScalar> ParabolicPde<S> for Stationary {
    fn diffusion(&self, _: &[S], _: S) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }
    fn source(&self, _: &[S], _: S) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
    fn initial_condition(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(x[0])
    }
}
fn transient_balance<S: TestScalar>() {
    for coordinates in [vec![0.0, 1.0], vec![0.0, 0.05, 0.7, 1.0]] {
        let mesh = explicit(coordinates.iter().copied().map(scalar::<S>).collect());
        let ends = Ends::new(
            End::dirichlet(value(|x| Ok(x))),
            End::dirichlet(value(|x| Ok(x))),
        );
        let heat = Heat(Stationary);
        let plan = fvm(&mesh, &heat, &ends).unwrap();
        let data = source(&heat);
        let problem = ParabolicProblem::new(&plan, &data, scalar::<S>(0.02)).unwrap();
        for scheme in [
            ParabolicScheme::ExplicitDopri5,
            ParabolicScheme::BackwardEuler,
            ParabolicScheme::CrankNicolson,
        ] {
            let step = scalar::<S>(0.001);
            let options = ParabolicOptions::new().with_scheme(scheme).with_step(step);
            let solution = problem
                .solve_with_options(&initial(&Stationary), &(options))
                .unwrap();
            assert!(solution.report().is_success());
            for (&value, w) in solution.values().iter().zip(coordinates.windows(2)) {
                assert!(
                    (as_f64(value) - (w[0] + w[1]) * 0.5).abs() < 100.0 * as_f64(S::EPSILON),
                    "{scheme:?} {value:?}"
                );
            }
        }
    }
}
#[test]
fn transient_balance_f32() {
    transient_balance::<f32>();
}
#[test]
fn transient_balance_f64() {
    transient_balance::<f64>();
}

fn transient_material_jump<S: TestScalar>() {
    #[derive(Clone)]
    struct JumpHeat;
    impl<S: TestScalar> EllipticPde<S> for JumpHeat {
        fn diffusion(&self, x: &[S]) -> Result<S, SciOxideError> {
            if x[0] == scalar::<S>(0.5) {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("x"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "interface has no owned material",
                ));
            }
            Ok(scalar(if x[0] < scalar(0.5) { 1. } else { 100. }))
        }
        fn reaction(&self, _: &[S]) -> Result<S, SciOxideError> {
            Ok(S::zero_value())
        }
        fn source(&self, _: &[S]) -> Result<S, SciOxideError> {
            Ok(S::zero_value())
        }
    }
    let mesh = explicit((0..=20).map(|i| scalar::<S>(i as f64 / 20.)).collect());
    let ends = Ends::new(
        End::dirichlet(constant(0.0)),
        End::dirichlet(constant(0.505)),
    );
    let plan = fvm_plan(&mesh, &JumpHeat, &ends).unwrap();
    let source = FieldData::zero();
    let system = SemidiscreteSystem::new(&plan, &source).unwrap();
    let initial = system
        .initial_state(&FieldData::time_invariant(|p: FieldPoint<S, 1>| {
            let x = p.point[0];
            Ok(if x < scalar(0.5) {
                x
            } else {
                scalar::<S>(0.5) + (x - scalar(0.5)) / scalar(100.)
            })
        }))
        .unwrap();
    let mut rhs = vec![S::zero_value(); 20];
    let mut epoch = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded();
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "fvm-material-jump", 1),
        0,
        0,
    );
    let mut cx = epoch
        .begin_solve(
            scioxide_core::ExecutionContext::sequential(),
            identity,
            &mut budgets,
        )
        .unwrap();
    for time in [S::zero_value(), S::one_value()] {
        system.rhs(time, &initial, &mut rhs, &mut cx).unwrap();
        assert!(
            rhs.iter()
                .all(|v| as_f64(*v).abs() < 32. * 100. * 400. * as_f64(S::EPSILON))
        );
    }
}

#[test]
fn transient_cell_owned_jump_preserves_stationary_flux() {
    transient_material_jump::<f64>();
    transient_material_jump::<f32>();
}

fn positivity<S: TestScalar>() {
    #[derive(Clone)]
    struct Load;
    impl<S: TestScalar> EllipticPde<S> for Load {
        fn diffusion(&self, x: &[S]) -> Result<S, SciOxideError> {
            Ok(S::one_value() + x[0])
        }
        fn reaction(&self, _: &[S]) -> Result<S, SciOxideError> {
            Ok(S::one_value())
        }
        fn source(&self, _: &[S]) -> Result<S, SciOxideError> {
            Ok(S::one_value())
        }
    }
    let mesh = explicit(
        [0.0, 0.01, 0.2, 0.9, 1.0]
            .into_iter()
            .map(scalar::<S>)
            .collect(),
    );
    let solution = solve_steady(
        &mesh,
        &Load,
        &support::homogeneous_dirichlet(),
        elliptic_options::<S>(),
    )
    .unwrap();
    assert!(solution.report().is_success());
    assert!(solution.values().iter().all(|v| *v > S::zero_value()));
}
#[test]
fn positivity_f32() {
    positivity::<f32>();
}
#[test]
fn positivity_f64() {
    positivity::<f64>();
}
