//! Independent polynomial and assembled-equation tests for conservative FDM.
//!
//! The accuracy statements (second-order interior on uniform or smoothly graded
//! axes, first order on arbitrary axes, second-order boundary relations) live in
//! the `StructuredFdmPlan` rustdoc; these tests check them by refinement.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
#[path = "support/mesh.rs"]
mod mesh_fixture;
mod support;

use mesh_fixture::{explicit, uniform};
use scioxide_core::LinearOperator;
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance, SciOxideError};
use scioxide_linalg::{CompatibilityPolicy, Preconditioning};
use scioxide_linalg::{GmresSolver, IterativeLinearSolver, LuDecomposition, Matrix};
use scioxide_pde::{
    EllipticGauge, EllipticOptions, EllipticProblem, PdeSolveReport, SpatialDiscretization,
    StructuredFdmPlan,
};
use scioxide_spatial::SegmentMesh;
use support::{
    Discretization, EllipticPde, End, Ends, Plan1d, Steady, TestScalar, as_f64, source, value,
};

#[derive(Clone)]
struct Quartic {
    variable: bool,
}
impl<S: TestScalar> EllipticPde<S> for Quartic {
    fn diffusion(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value() + if self.variable { x[0] } else { S::zero_value() })
    }
    fn reaction(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }
    fn source(&self, x: &[S]) -> Result<S, SciOxideError> {
        let x = x[0];
        let twelve = S::scalar_from_f64(12.0);
        let derivative = if self.variable {
            S::one_value() + twelve * x * x + S::scalar_from_f64(16.0) * x * x * x
        } else {
            twelve * x * x
        };
        Ok(S::one_value() + x + x * x * x * x - derivative)
    }
}
fn exact(x: f64) -> f64 {
    1.0 + x + x * x * x * x
}
fn quartic_source(x: f64, variable: bool) -> f64 {
    exact(x)
        - if variable {
            1.0 + 12.0 * x * x + 16.0 * x * x * x
        } else {
            12.0 * x * x
        }
}

/// Materialize the injected sparse operator by basis application, solve by
/// independent dense LU, then let GMRES verify the true initial residual.
struct Dense;
impl<S: TestScalar> IterativeLinearSolver<S> for Dense {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("linear", "Dense", 1)
    }

    fn solve_into(
        &self,
        a: &dyn LinearOperator<S>,
        b: scioxide_linalg::VectorView<'_, S>,
        _preconditioning: Preconditioning<'_, S>,
        mut x: scioxide_linalg::VectorViewMut<'_, S>,
        _workspace: &mut scioxide_linalg::IterativeWorkspace<S>,
        _context: scioxide_linalg::LinalgExecutionContext<'_, '_>,
    ) -> Result<scioxide_linalg::IterativeSolveReport<S>, SciOxideError> {
        let n = a.rows();
        let mut entries = vec![S::zero_value(); n * n];
        let mut basis = vec![S::zero_value(); n];
        let mut output = vec![S::zero_value(); n];
        let mut scratch = vec![S::zero_value(); a.apply_workspace_len()?];
        for j in 0..n {
            basis[j] = S::one_value();
            a.apply_into(&basis, &mut output, &mut scratch)?;
            for (i, &value) in output.iter().enumerate() {
                entries[i * n + j] = value;
            }
            basis[j] = S::zero_value();
        }
        let matrix = Matrix::from_vec(n, n, entries)?;
        let direct = LuDecomposition::factor(&matrix)?.solve(b)?;
        let tolerance = S::scalar_from_f64(if S::NAME == "f32" { 2e-5 } else { 1e-11 });
        let result = GmresSolver::new(
            ResidualTolerance::new(tolerance, tolerance)?,
            Budget::try_from(n + 1)?,
        )
        .with_restart(Budget::try_from(n)?)
        .solve_with_options(a, b, Preconditioning::None, Some(direct.view()))?;
        scioxide_linalg::VectorView::from(result.solution()).copy_into(
            scioxide_linalg::CopyPolicy::Preserve,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            (x).reborrow(),
            &mut scioxide_linalg::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok(result.report())
    }
}

fn mesh<S: TestScalar>(n: usize, family: usize) -> SegmentMesh<S> {
    explicit(
        (0..=n)
            .map(|i| {
                let t = i as f64 / n as f64;
                S::scalar_from_f64(match family {
                    0 => t,
                    1 => (t + t * t) / 2.0,
                    _ => t + if i % 2 == 1 { 0.25 / n as f64 } else { 0.0 },
                })
            })
            .collect(),
    )
}

/// Prepared conservative FDM plan of a steady problem.
fn fdm<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    mesh: &SegmentMesh<S>,
    problem: &P,
    ends: &Ends<S>,
) -> Result<StructuredFdmPlan<S, S, 1>, SciOxideError> {
    match Discretization::Fdm.plan(mesh, &Steady(problem.clone()), ends)? {
        Plan1d::Fdm(plan) => Ok(plan),
        _ => unreachable!("an FDM plan was requested"),
    }
}

/// Steady FDM solve with the default direct solve, or with `solver` injected;
/// the values are the nodal field.
fn solve<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    mesh: &SegmentMesh<S>,
    problem: &P,
    ends: &Ends<S>,
    options: EllipticOptions<S>,
    solver: Option<Box<dyn IterativeLinearSolver<S> + Send>>,
) -> Result<(Vec<S>, PdeSolveReport<S>), SciOxideError> {
    let plan = fdm(mesh, problem, ends)?;
    let source = source(&Steady(problem.clone()));
    let elliptic = options;
    let mut workspace = scioxide_pde::EllipticWorkspace::new();
    if let Some(solver) = solver {
        workspace = workspace.with_linear_solver(solver);
    }
    let mut values = vec![S::zero_value(); plan.unknowns()];
    let report = EllipticProblem::new(&plan, &source).solve_into(
        &elliptic,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok((values, report))
}

/// Residual and compatibility tolerance `tolerance` with a weighted-mean gauge.
fn gauged<S: TestScalar>(tolerance: scioxide_core::Tolerance<S>) -> EllipticOptions<S> {
    let tolerance = ResidualTolerance::from_tolerance(tolerance);
    EllipticOptions::new()
        .with_gauge(EllipticGauge::WeightedMean(S::scalar_from_f64(0.0)))
        .with_residual_tolerance(tolerance)
        .with_compatibility(CompatibilityPolicy::RequireCompatible, tolerance)
}

fn constant<S: TestScalar>(g: f64) -> scioxide_pde::BoundaryData<S, S, 1> {
    value(move |_| Ok(S::scalar_from_f64(g)))
}

/// Kind 0: Dirichlet 1/3. Kind 1: outward derivative −1 at the left, Dirichlet 3
/// at the right. Kind 2: `2u + 3 du/dn = g` with g = −1 and 21. Natural data is
/// the conormal flux, so derivative data scale by the endpoint diffusion `k`
/// (the Robin law becomes `k du/dn + (2k/3) u = k g/3`).
fn ends<S: TestScalar>(kind: usize, variable: bool) -> Ends<S> {
    let k = |x: f64| 1.0 + if variable { x } else { 0.0 };
    match kind {
        0 => Ends::new(End::dirichlet(constant(1.0)), End::dirichlet(constant(3.0))),
        1 => Ends::new(
            End::natural(constant(-k(0.0))),
            End::dirichlet(constant(3.0)),
        ),
        _ => {
            let robin = |x: f64, g: f64| {
                End::robin(
                    S::scalar_from_f64(2.0 * k(x) / 3.0),
                    constant(k(x) * g / 3.0),
                )
            };
            Ends::new(robin(0.0, -1.0), robin(1.0, 21.0))
        }
    }
}

fn refinement<S: TestScalar>() {
    for variable in [false, true] {
        for family in 0..3 {
            for kind in 0..3 {
                let mut previous = None;
                for n in [8, 16, 32] {
                    let mesh = mesh::<S>(n, family);
                    let nodes: Vec<_> =
                        mesh_fixture::nodes(&mesh).into_iter().map(as_f64).collect();
                    let (values, report) = solve(
                        &mesh,
                        &Quartic { variable },
                        &ends(kind, variable),
                        EllipticOptions::new(),
                        Some(Box::new(Dense)),
                    )
                    .unwrap();
                    assert!(report.is_success());
                    let values: Vec<_> = values.into_iter().map(as_f64).collect();
                    let error = nodes
                        .iter()
                        .zip(&values)
                        .map(|(&x, &u)| (u - exact(x)).abs())
                        .fold(0.0_f64, f64::max);
                    if let Some(coarse) = previous {
                        assert!(
                            coarse / error > 3.0,
                            "{} variable={variable} family={family} kind={kind} n={n}: {coarse:e}/{error:e}",
                            S::NAME
                        );
                    }
                    previous = Some(error);
                    // Independently assembled equations in original integrated scaling.
                    let conductance = |i: usize| {
                        let midpoint = (nodes[i] + nodes[i + 1]) / 2.0;
                        (1.0 + if variable { midpoint } else { 0.0 }) / (nodes[i + 1] - nodes[i])
                    };
                    let tolerance = if S::NAME == "f32" { 3e-4 } else { 2e-10 };
                    for i in 1..n {
                        let w = (nodes[i + 1] - nodes[i - 1]) / 2.0;
                        let residual = conductance(i - 1) * (values[i] - values[i - 1])
                            + conductance(i) * (values[i] - values[i + 1])
                            + w * (values[i] - quartic_source(nodes[i], variable));
                        assert!(residual.abs() < tolerance, "interior residual {residual:e}");
                    }
                    if kind != 0 {
                        for (i, j) in [(0, 1), (n, n - 1)] {
                            if kind == 1 && i == n {
                                continue;
                            }
                            let w = (nodes[i] - nodes[j]).abs() / 2.0;
                            let k = 1.0 + if variable { nodes[i] } else { 0.0 };
                            let outward = (conductance(i.min(j)) * (values[i] - values[j])
                                + w * (values[i] - quartic_source(nodes[i], variable)))
                                / k;
                            let residual = if kind == 1 {
                                outward + 1.0
                            } else {
                                2.0 * values[i] + 3.0 * outward - if i == 0 { -1.0 } else { 21.0 }
                            };
                            assert!(
                                residual.abs() < 4.0 * tolerance,
                                "boundary residual {residual:e}"
                            );
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn manufactured_refinement_f32() {
    refinement::<f32>();
}
#[test]
fn manufactured_refinement_f64() {
    refinement::<f64>();
}

#[test]
fn exact_boundary_relations_have_second_order_defects() {
    for variable in [false, true] {
        for family in 0..3 {
            let mut previous = None;
            for n in [16, 32, 64] {
                let nodes = mesh_fixture::nodes(&mesh::<f64>(n, family));
                let error = [(0, 1), (n, n - 1)]
                    .into_iter()
                    .map(|(i, j)| {
                        let x = nodes[i];
                        let y = nodes[j];
                        let h = (x - y).abs();
                        let face_k = 1.0 + if variable { (x + y) / 2.0 } else { 0.0 };
                        let boundary_k = 1.0 + if variable { x } else { 0.0 };
                        let outward = (face_k * (exact(x) - exact(y)) / h
                            + h / 2.0 * (exact(x) - quartic_source(x, variable)))
                            / boundary_k;
                        (outward - if i == 0 { -1.0 } else { 5.0 }).abs()
                    })
                    .fold(0.0_f64, f64::max);
                if let Some(coarse) = previous {
                    assert!(coarse / error > 3.7);
                }
                previous = Some(error);
            }
        }
    }
}

#[derive(Clone)]
struct TinyDiffusion<S>(S);
impl<S: TestScalar> EllipticPde<S> for TinyDiffusion<S> {
    fn diffusion(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(self.0)
    }
    fn reaction(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
    fn source(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
}

fn coordinates<S: TestScalar>() {
    let large = <S as scioxide_core::RealScalar>::MAX / S::scalar_from_f64(2.0);
    let mesh = uniform(-large, large, 5);
    let plan = Discretization::Fdm
        .plan(
            &mesh,
            &Steady(TinyDiffusion(S::one_value())),
            &support::homogeneous_dirichlet(),
        )
        .unwrap();
    let Plan1d::Fdm(plan) = plan else {
        unreachable!("an FDM plan was requested")
    };
    let nodes: Vec<S> = (0..5)
        .map(|node| plan.node_point(node).unwrap()[0])
        .collect();
    assert_eq!(nodes, mesh_fixture::nodes(&mesh));
    assert_eq!(nodes[0], -large);
    assert_eq!(nodes[4], large);
    assert_eq!(nodes[2], S::zero_value());
    assert!(nodes.windows(2).all(|pair| pair[0] < pair[1]));
}
#[test]
fn coordinate_consumers_f32() {
    coordinates::<f32>();
}
#[test]
fn coordinate_consumers_f64() {
    coordinates::<f64>();
}

#[derive(Clone)]
struct Quadratic;
impl<S: TestScalar> EllipticPde<S> for Quadratic {
    fn diffusion(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::one_value())
    }
    fn reaction(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::zero_value())
    }
    fn source(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::scalar_from_f64(-2.0))
    }
}
fn neumann<S: TestScalar>() {
    for family in 0..3 {
        let mesh = mesh::<S>(16, family);
        let nodes: Vec<_> = mesh_fixture::nodes(&mesh).into_iter().map(as_f64).collect();
        // u = x² with unit diffusion: outward fluxes 0 (left) and 2 (right).
        let ends = Ends::new(End::natural(constant(0.0)), End::natural(constant(2.0)));
        let options = gauged(scioxide_core::Tolerance::from_decimal_places(
            if S::NAME == "f32" { 5 } else { 11 },
            if S::NAME == "f32" { 5 } else { 11 },
        ));
        let (values, report) =
            solve(&mesh, &Quadratic, &ends, options, Some(Box::new(Dense))).unwrap();
        assert!(report.is_success());
        let mean: f64 = nodes
            .windows(2)
            .map(|p| (p[1] - p[0]) * (p[0] * p[0] + p[1] * p[1]) / 2.0)
            .sum();
        let tolerance = if S::NAME == "f32" { 2e-5 } else { 1e-11 };
        for (&value, &x) in values.iter().zip(&nodes) {
            assert!((as_f64(value) - (x * x - mean)).abs() < tolerance);
        }
    }
}
#[test]
fn quadratic_neumann_gauge_f32() {
    neumann::<f32>();
}
#[test]
fn quadratic_neumann_gauge_f64() {
    neumann::<f64>();
}

#[test]
fn dense_and_default_sparse_solve_agree() {
    let mesh = mesh::<f64>(16, 1);
    let ends = ends(2, true);
    let problem = Quartic { variable: true };
    let options = EllipticOptions::new();
    let (sparse, sparse_report) = solve(&mesh, &problem, &ends, options, None).unwrap();
    let (dense, dense_report) =
        solve(&mesh, &problem, &ends, options, Some(Box::new(Dense))).unwrap();
    assert!(sparse_report.is_success() && dense_report.is_success());
    for (&a, &b) in sparse.iter().zip(&dense) {
        assert!((a - b).abs() < 1e-7);
    }
}

fn subnormal_solve<S: TestScalar>(tiny: S) {
    let mesh = explicit(vec![S::zero_value(), tiny, tiny + tiny]);
    let one = Ends::new(
        End::dirichlet(value(|_| Ok(S::one_value()))),
        End::dirichlet(value(|_| Ok(S::one_value()))),
    );
    let (values, report) = solve(
        &mesh,
        &TinyDiffusion(tiny),
        &one,
        EllipticOptions::new(),
        None,
    )
    .unwrap();
    assert!(report.is_success());
    assert!(values.iter().all(|&v| v == S::one_value()));
}
/// Lumped mass must be positive, so the smallest subnormal spacing (whose
/// boundary half-width rounds to zero) is rejected before any solve.
fn unrepresentable_half_width<S: TestScalar>(tiny: S) {
    let mesh = explicit(vec![S::zero_value(), tiny, tiny + tiny]);
    let error = Discretization::Fdm
        .plan(
            &mesh,
            &Steady(TinyDiffusion(tiny)),
            &support::homogeneous_dirichlet(),
        )
        .err()
        .unwrap();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}
#[test]
fn subnormal_mesh_solve_f32() {
    subnormal_solve(f32::from_bits(2));
    unrepresentable_half_width(f32::from_bits(1));
}
#[test]
fn subnormal_mesh_solve_f64() {
    subnormal_solve(f64::from_bits(2));
    unrepresentable_half_width(f64::from_bits(1));
}

fn injected_execution_and_views<S: TestScalar>() {
    let mesh = mesh::<S>(8, 1);
    let problem = Quartic { variable: true };
    let plan = fdm(&mesh, &problem, &ends(2, true)).unwrap();
    let source = source(&Steady(problem));
    let elliptic = EllipticOptions::new();
    let problem = EllipticProblem::new(&plan, &source);
    let mut workspace = scioxide_pde::EllipticWorkspace::new().with_linear_solver(Box::new(Dense));
    let mut sequential = vec![S::zero_value(); plan.unknowns()];
    let sequential_report = problem
        .solve_into(
            &elliptic,
            &mut sequential,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    // Assembly is sequential; the injected reference executor shares the
    // admission/capability protocol without promising parallel assembly.
    let executor = scioxide_core::SequentialExecutor;
    let mut injected = vec![S::zero_value(); plan.unknowns()];
    let injected_report = problem
        .solve_into(
            &elliptic,
            &mut injected,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert!(sequential_report.is_success() && injected_report.is_success());
    // Reports carry no mesh identity; the solved plan holds the mesh revision.
    assert!(SpatialDiscretization::mesh_token(&plan).is_compatible_with(mesh.mesh_token()));
    assert_eq!(sequential, injected);
}
#[test]
fn execution_and_views_f32() {
    injected_execution_and_views::<f32>();
}
#[test]
fn execution_and_views_f64() {
    injected_execution_and_views::<f64>();
}
