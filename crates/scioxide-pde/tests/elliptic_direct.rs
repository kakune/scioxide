//! Default elliptic complexity, precision, residual and gauge regressions.
#![allow(clippy::unwrap_used)]
#[path = "support/mesh.rs"]
mod mesh_fixture;
#[path = "support/mod.rs"]
mod support;

use scioxide_core::{ExecutionContext, PartialResultTrust, SciOxideError};
use scioxide_pde::{
    EllipticGauge, EllipticOptions, EllipticProblem, EllipticWorkspace, PdeSolveReport,
    PdeTermination, SpatialDiscretization,
};
use scioxide_spatial::SegmentMesh;
use support::{Discretization, EllipticPde, End, Ends, Plan1d, Steady, TestScalar, source, value};

/// `-u'' = 12 x²` with exact solutions `x - x⁴ + c`.
#[derive(Clone)]
struct Polynomial;
impl<S: TestScalar> EllipticPde<S> for Polynomial {
    fn diffusion(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::ONE)
    }
    fn reaction(&self, _: &[S]) -> Result<S, SciOxideError> {
        Ok(S::ZERO)
    }
    fn source(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(S::try_from_usize_exact(12)? * x[0] * x[0])
    }
}
fn mesh<S: TestScalar>(n: usize, periodic: bool) -> SegmentMesh<S> {
    let mesh = mesh_fixture::uniform(S::ZERO, S::ONE, n);
    if periodic {
        mesh_fixture::periodic(&mesh)
    } else {
        mesh
    }
}
/// 0: Dirichlet; 1: natural/Dirichlet; 2: Robin/Dirichlet; 3, 4: pure natural.
/// Natural data is the outward flux of `x - x⁴`: `-1` at `x = 0`, `-3` at `x = 1`.
fn boundaries<S: TestScalar>(kind: usize) -> Ends<S> {
    let constant = |g: f64| value(move |_| Ok(support::scalar::<S>(g)));
    let left = match kind {
        0 => End::dirichlet(constant(0.0)),
        2 => End::robin(S::ONE, constant(-1.0)),
        _ => End::natural(constant(-1.0)),
    };
    let right = if kind < 3 {
        End::dirichlet(constant(0.0))
    } else {
        End::natural(constant(-3.0))
    };
    Ends::new(left, right)
}
fn plan<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    discretization: Discretization,
    mesh: &SegmentMesh<S>,
    problem: &P,
    ends: &Ends<S>,
) -> Plan1d<S> {
    discretization
        .plan(mesh, &Steady(problem.clone()), ends)
        .unwrap()
}
/// Solve into fresh output storage with a caller-retained workspace; returns the
/// unknown-order field and the report.
fn solve<S: TestScalar, P: EllipticPde<S> + Send + Sync + Clone + 'static>(
    elliptic: &EllipticOptions<S>,
    workspace: &mut EllipticWorkspace<S>,
    plan: &Plan1d<S>,
    problem: &P,
) -> Result<(Vec<S>, PdeSolveReport<S>), SciOxideError> {
    let source = source(&Steady(problem.clone()));
    let mut values = vec![S::ZERO; with_plan!(plan, |p| p.unknowns())];
    let report = with_plan!(plan, |p| EllipticProblem::new(p, &source).solve_into(
        elliptic,
        &mut values,
        workspace,
        ExecutionContext::sequential()
    ))?;
    Ok((values, report))
}

fn large(discretization: Discretization) {
    for kind in 0..5 {
        let mesh = mesh::<f64>(100_001, false);
        let plan = plan(discretization, &mesh, &Polynomial, &boundaries(kind));
        let options = match kind {
            3 => EllipticOptions::new().with_gauge(EllipticGauge::Orthogonal),
            4 => EllipticOptions::new().with_gauge(EllipticGauge::WeightedMean(2.0)),
            _ => EllipticOptions::new(),
        };
        let elliptic = options;
        let (values, report) = solve(
            &elliptic,
            &mut scioxide_pde::EllipticWorkspace::new(),
            &plan,
            &Polynomial,
        )
        .unwrap();
        assert_eq!(report.termination(), PdeTermination::Completed);
        assert_eq!(report.trust(), PartialResultTrust::ResidualChecked);
        assert_eq!(report.diagnostics().num_linear_iterations(), 0);
        assert_eq!(report.diagnostics().num_factorizations(), 1);
        assert!(
            report
                .diagnostics()
                .linear_residual_report()
                .unwrap()
                .scaled_norm()
                < 1e-8
        );
        if kind >= 3 {
            assert!(report.diagnostics().nullspace_gauge_defect().unwrap() < 1e-8);
        }
        let values = plan.full(&values).unwrap();
        let cell = values.len() == 100_000;
        for (i, &u) in values.iter().enumerate().step_by(997) {
            let x = (i as f64 + if cell { 0.5 } else { 0.0 }) / 100_000.0;
            let shift = match kind {
                3 => -0.3,
                4 => 1.7,
                _ => 0.0,
            };
            assert!(
                (u - (x - x.powi(4) + shift)).abs() < 3e-5,
                "kind={kind}, x={x}, u={u}"
            );
        }
    }
}
#[test]
fn large_fdm() {
    large(Discretization::Fdm);
}
#[test]
fn large_fem() {
    large(Discretization::Fem);
}
#[test]
fn large_fvm() {
    large(Discretization::Fvm);
}

fn single_precision(discretization: Discretization) {
    for n in [33, 1001, 10_000] {
        let mesh = mesh::<f32>(n, false);
        let plan = plan(discretization, &mesh, &Polynomial, &boundaries(0));
        let elliptic = EllipticOptions::new();
        let (_, report) = solve(
            &elliptic,
            &mut scioxide_pde::EllipticWorkspace::new(),
            &plan,
            &Polynomial,
        )
        .unwrap();
        assert!(report.is_success());
        assert_eq!(report.trust(), PartialResultTrust::ResidualChecked);
        assert!(
            report
                .diagnostics()
                .linear_residual_report()
                .unwrap()
                .scaled_norm()
                < f32::EPSILON.sqrt()
        );
    }
}
#[test]
fn default_fdm_f32() {
    single_precision(Discretization::Fdm);
}
#[test]
fn default_fem_f32() {
    single_precision(Discretization::Fem);
}
#[test]
fn default_fvm_f32() {
    single_precision(Discretization::Fvm);
}

/// `-u'' + c u = f` with periodic solution `sin 2πx + 0.2 cos 6πx`.
#[derive(Clone)]
struct Periodic {
    reaction: f64,
}
impl EllipticPde<f64> for Periodic {
    fn diffusion(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }
    fn reaction(&self, _: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.reaction)
    }
    fn source(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        let t = 2.0 * core::f64::consts::PI * x[0];
        Ok(
            (4.0 * core::f64::consts::PI.powi(2) + self.reaction) * t.sin()
                + 0.2 * (36.0 * core::f64::consts::PI.powi(2) + self.reaction) * (3.0 * t).cos(),
        )
    }
}
#[test]
fn large_periodic_fdm() {
    for reaction in [0.0, 1.0] {
        let mesh = mesh::<f64>(100_001, true);
        let problem = Periodic { reaction };
        let plan = plan(Discretization::Fdm, &mesh, &problem, &Ends::periodic());
        let options = if reaction == 0.0 {
            EllipticOptions::new().with_gauge(EllipticGauge::WeightedMean(0.0))
        } else {
            EllipticOptions::new()
        };
        let elliptic = options;
        let (values, report) = solve(
            &elliptic,
            &mut scioxide_pde::EllipticWorkspace::new(),
            &plan,
            &problem,
        )
        .unwrap();
        assert!(report.is_success());
        assert_eq!(report.trust(), PartialResultTrust::ResidualChecked);
        let values = plan.full(&values).unwrap();
        assert_eq!(values.len(), 100_001);
        for (i, &u) in values.iter().enumerate().step_by(997) {
            let t = 2.0 * core::f64::consts::PI * i as f64 / 100_000.0;
            assert!((u - t.sin() - 0.2 * (3.0 * t).cos()).abs() < 1e-5);
        }
    }
}

/// Plan preparation and solve storage grow linearly with the mesh.

#[test]
fn tiny_periodic_systems_fold_corner_entries() {
    for n in [2, 3, 4] {
        let mesh = mesh::<f64>(n, true);
        let problem = Periodic { reaction: 0.0 };
        let plan = plan(Discretization::Fdm, &mesh, &problem, &Ends::periodic());
        let elliptic = EllipticOptions::new()
            .with_gauge(EllipticGauge::WeightedMean(2.0))
            .with_compatibility(
                scioxide_linalg::CompatibilityPolicy::Project,
                scioxide_core::ResidualTolerance::from_tolerance(
                    scioxide_core::Tolerance::from_sqrt_epsilon(),
                ),
            );
        let (_, report) = solve(
            &elliptic,
            &mut scioxide_pde::EllipticWorkspace::new(),
            &plan,
            &problem,
        )
        .unwrap();
        assert!(report.is_success());
        assert!(report.diagnostics().nullspace_gauge_defect().unwrap() < 1e-12);
    }
}
