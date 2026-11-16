#![allow(clippy::unwrap_used)]
#[path = "support/mesh.rs"]
mod mesh_fixture;
#[path = "support/mod.rs"]
mod support;
use mesh_fixture::{explicit, nodes, periodic};
use scioxide_core::LinearOperator;
use scioxide_core::{
    Budget, ErrorTolerance, ExecutionContext, NormSelection, NumericScalar, ResidualTolerance,
    SciOxideError, Tolerance,
};
use scioxide_linalg::{CompatibilityPolicy, Preconditioning};
use scioxide_linalg::{GmresSolver, IterativeLinearSolver};
use scioxide_pde::{
    EllipticGauge, EllipticOptions, EllipticProblem, ParabolicOptions, ParabolicProblem,
    ParabolicScheme, PdeOutputSchedule, PdeSolveReport, SpatialDiscretization,
    StructuredFdmCoefficients, StructuredFdmPlan,
};
use scioxide_spatial::{Cell, CoordinateAxis, Field, FieldView, SegmentMesh, Vertex};
use std::sync::{Arc, Mutex};
use support::{
    DISCRETIZATIONS, Discretization, EllipticPde, End, Ends, Heat, ParabolicPde, Steady,
    TestScalar, homogeneous_dirichlet, initial, moving, source, value,
};

/// Linear fields: `u = x` is harmonic and `u = x + t` solves `u_t = u_xx + 1`.
#[derive(Clone)]
struct Linear;
impl<S: TestScalar> EllipticPde<S> for Linear {
    fn diffusion(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }
    fn reaction(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
    fn source(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
}
impl<S: TestScalar> ParabolicPde<S> for Linear {
    fn diffusion(&self, _: &[S], _: S) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }
    fn source(&self, _: &[S], _: S) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }
    fn initial_condition(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(x[0])
    }
}
/// Unit diffusion for plans built directly against foreign boundaries.
struct Unit;
impl StructuredFdmCoefficients<f64, f64, 1> for Unit {
    fn diffusion(&self, _: usize, _: usize, _: [f64; 1]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
}
fn mesh<S: TestScalar>() -> SegmentMesh<S> {
    explicit(
        [0.0, 0.08, 0.27, 0.53, 0.81, 1.0]
            .into_iter()
            .map(S::scalar_from_f64)
            .collect(),
    )
}
/// Steady `u = x` solved with `options` and an optional injected solver; returns
/// the coordinate-order field and the report.
fn steady<S: TestScalar>(
    discretization: Discretization,
    mesh: &SegmentMesh<S>,
    ends: &Ends<S>,
    options: EllipticOptions<S>,
    solver: Option<Box<dyn IterativeLinearSolver<S> + Send>>,
) -> Result<(Vec<S>, PdeSolveReport<S>), SciOxideError> {
    let problem = Steady(Linear);
    let plan = discretization.plan(mesh, &problem, ends)?;
    let data = source(&problem);
    let elliptic = options;
    let mut workspace = scioxide_pde::EllipticWorkspace::new();
    if let Some(solver) = solver {
        workspace = workspace.with_linear_solver(solver);
    }
    let mut values = vec![S::zero_value(); with_plan!(&plan, |p| p.unknowns())];
    let report = with_plan!(&plan, |p| EllipticProblem::new(p, &data).solve_into(
        &elliptic,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential()
    ))?;
    Ok((plan.full(&values)?, report))
}
fn moving_ends<S: TestScalar>() -> Ends<S> {
    let end = || End::dirichlet(moving(|x: S, t| Ok(x + t)));
    Ends::new(end(), end())
}
fn transient<S: TestScalar>(mesh: SegmentMesh<S>) {
    let ends = moving_ends();
    for discretization in DISCRETIZATIONS {
        for scheme in [
            ParabolicScheme::ExplicitDopri5,
            ParabolicScheme::BackwardEuler,
            ParabolicScheme::CrankNicolson,
        ] {
            let step = S::scalar_from_f64(0.001);
            // The fixture checks a small absolute error, while f32 cannot
            // resolve the default 1e-8 inner correction at unit-scale states.
            let threshold = S::scalar_from_f64(1e-8).max(S::EPSILON * S::scalar_from_f64(16.0));
            let tolerance = ErrorTolerance::from_tolerance(
                Tolerance::new(threshold, threshold).unwrap(),
                NormSelection::RootMeanSquare,
            );
            let options = ParabolicOptions::new()
                .with_scheme(scheme)
                .with_step(step)
                .with_error_tolerance(tolerance)
                .with_output_schedule(PdeOutputSchedule::TimeSteps);
            let heat = Heat(Linear);
            let plan = discretization.plan(&mesh, &heat, &ends).unwrap();
            let data = source(&heat);
            let start = initial::<S, _>(&Linear);
            let parabolic = options;
            let solution = with_plan!(&plan, |p| ParabolicProblem::new(
                p,
                &data,
                S::scalar_from_f64(0.02)
            )
            .unwrap()
            .solve_with_options(&start, &parabolic))
            .unwrap();
            assert!(solution.report().is_success());
            // FEM plans live on the interval simplices of the same nodes.
            if discretization != Discretization::Fem {
                assert!(
                    solution
                        .state()
                        .mesh_token()
                        .is_compatible_with(mesh.mesh_token())
                );
            }
            let coordinates = discretization.coordinates(&mesh);
            let trajectory = solution.trajectory();
            assert!(!trajectory.times().is_empty());
            for (&time, state) in trajectory.times().iter().zip(trajectory.states()) {
                for (&value, &x) in plan.full(state).unwrap().iter().zip(&coordinates) {
                    assert!(
                        NumericScalar::abs(value - x - time) < S::scalar_from_f64(3e-5),
                        "{} {scheme:?}: {value} at {x}",
                        discretization.name()
                    );
                }
            }
        }
    }
}
#[test]
fn moving_boundaries_f32() {
    transient(mesh::<f32>());
}
#[test]
fn moving_boundaries_f64() {
    transient(mesh::<f64>());
}

/// Dirichlet, natural and Robin left ends (`k du/dn + αu = g`) all reproduce `u = x`.
fn robin<S: TestScalar>() {
    let mesh = mesh::<S>();
    let flux = || value(|_| Ok(-S::one_value()));
    for left in [
        End::dirichlet(value(|_| Ok(S::zero_value()))),
        End::natural(flux()),
        End::robin(S::zero_value(), flux()),
        End::robin(S::scalar_from_f64(2.0 / 3.0), flux()),
    ] {
        let ends = Ends::new(left, End::dirichlet(value(|_| Ok(S::one_value()))));
        for discretization in DISCRETIZATIONS {
            let tolerance =
                ResidualTolerance::new(S::scalar_from_f64(1e-5), S::scalar_from_f64(1e-5)).unwrap();
            let options = EllipticOptions::new()
                .with_compatibility(CompatibilityPolicy::RequireCompatible, tolerance)
                .with_residual_tolerance(tolerance);
            let (values, report) = steady(discretization, &mesh, &ends, options, None).unwrap();
            assert!(report.is_success());
            for (&value, &x) in values.iter().zip(&discretization.coordinates(&mesh)) {
                assert!(
                    NumericScalar::abs(value - x) < S::scalar_from_f64(2e-4),
                    "{}: {value} != {x}",
                    discretization.name()
                );
            }
        }
    }
}
#[test]
fn robin_equivalence_f32() {
    robin::<f32>();
}
#[test]
fn robin_equivalence_f64() {
    robin::<f64>();
}

#[test]
fn field_locations_identity_and_revisions_are_checked() {
    let mesh = mesh::<f64>();
    let vertices = Field::<_, Vertex, 1>::new(&mesh, vec![0.0; 6]).unwrap();
    let cells = Field::<_, Cell, 1>::new(&mesh, vec![0.0; 5]).unwrap();
    let faces = Field::<_, scioxide_spatial::Face<0>, 1>::new(&mesh, vec![0.0; 6]).unwrap();
    assert!(vertices.is_compatible_with(&mesh));
    assert!(cells.is_compatible_with(&mesh));
    assert!(faces.is_compatible_with(&mesh));
    assert!(FieldView::<_, Cell, 1>::new(&mesh, vertices.values()).is_err());
    let independent = explicit(nodes(&mesh));
    let revised = SegmentMesh::new(
        mesh.grid()
            .with_axis(0, CoordinateAxis::uniform(0.0, 1.0, 6).unwrap())
            .unwrap(),
    )
    .unwrap();
    let boundaries = moving_ends().structured(&mesh).unwrap();
    for other in [&independent, &revised] {
        assert!(!vertices.is_compatible_with(other));
        let plan =
            StructuredFdmPlan::with_options(other.clone(), &Unit, &boundaries, support::limits());
        assert!(matches!(
            plan.map(|_| ()).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }
}

struct Inspect(Arc<Mutex<Vec<f64>>>);
impl IterativeLinearSolver<f64> for Inspect {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "Inspect", 1)
    }

    fn solve_into(
        &self,
        a: &dyn LinearOperator<f64>,
        b: scioxide_linalg::VectorView<'_, f64>,
        preconditioning: Preconditioning<'_, f64>,
        x: scioxide_linalg::VectorViewMut<'_, f64>,
        solve_workspace: &mut scioxide_linalg::IterativeWorkspace<f64>,
        context: scioxide_linalg::LinalgExecutionContext<'_, '_>,
    ) -> Result<scioxide_linalg::IterativeSolveReport<f64>, SciOxideError> {
        let n = a.rows();
        let mut dense = vec![0.0; n * n];
        let mut input = vec![0.0; n];
        let mut output = vec![0.0; n];
        let mut workspace = vec![0.0; a.apply_workspace_len()?];
        for j in 0..n {
            input[j] = 1.0;
            a.apply_into(&input, &mut output, &mut workspace)?;
            for i in 0..n {
                dense[i * n + j] = output[i];
            }
            input[j] = 0.0;
        }
        for i in 0..n {
            for j in 0..n {
                assert!((dense[i * n + j] - dense[j * n + i]).abs() < 1e-12);
            }
        }
        *self.0.lock().unwrap() = dense;
        GmresSolver::new(
            ResidualTolerance::new(1e-10, 1e-10)?,
            Budget::try_from(100)?,
        )
        .with_restart(Budget::try_from(n)?)
        .solve_into(a, b, preconditioning, x, solve_workspace, context)
    }
}
#[test]
fn nonuniform_operators_preserve_symmetry_and_dirichlet_energy() {
    let mesh = mesh::<f64>();
    for discretization in DISCRETIZATIONS {
        let dense = Arc::new(Mutex::new(Vec::new()));
        let (_, report) = steady(
            discretization,
            &mesh,
            &homogeneous_dirichlet(),
            EllipticOptions::new(),
            Some(Box::new(Inspect(Arc::clone(&dense)))),
        )
        .unwrap();
        // Symmetric elimination keeps unit rows for the two Dirichlet nodes;
        // cell-centred FVM imposes them through the boundary face law.
        let n = if discretization == Discretization::Fvm {
            5
        } else {
            6
        };
        let a = dense.lock().unwrap();
        assert_eq!(a.len(), n * n);
        // Independent dense Cholesky establishes positive definiteness.
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let value = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
                l[i * n + j] = if i == j {
                    assert!(value > 0.0);
                    value.sqrt()
                } else {
                    value / l[j * n + j]
                };
            }
        }
        assert!(report.is_success());
    }
}
#[test]
fn periodic_constant_nullspace_and_symmetric_gauge() {
    let mesh = periodic(&mesh::<f64>());
    let dense = Arc::new(Mutex::new(Vec::new()));
    let (values, report) = steady(
        Discretization::Fdm,
        &mesh,
        &Ends::periodic(),
        EllipticOptions::new().with_gauge(EllipticGauge::WeightedMean(0.0)),
        Some(Box::new(Inspect(Arc::clone(&dense)))),
    )
    .unwrap();
    assert!(report.is_success());
    let a = dense.lock().unwrap();
    let n = mesh.topology_sizes().vertices(); // five canonical DOFs plus one gauge
    assert_eq!(a.len(), n * n);
    for row in 0..n - 1 {
        assert!(a[row * n..row * n + n - 1].iter().sum::<f64>().abs() < 1e-12);
    }
    assert_eq!(values.len(), n);
    assert_eq!(values[0], values[n - 1]);
}

#[test]
fn minimum_mesh_supports_all_transient_schemes() {
    transient(explicit(vec![0.0_f64, 1.0]));
    transient(explicit(vec![0.0_f32, 1.0]));
}
