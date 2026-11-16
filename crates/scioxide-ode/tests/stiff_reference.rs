//! Published IVPtestset endpoints and pinned offline SciPy comparison bounds.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection, SciOxideError, SolveContext,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::*;

#[derive(Clone, Copy, Debug)]
struct Problem(&'static str);
impl Problem {
    fn initial(self) -> Vec<f64> {
        match self.0 {
            "rober" => vec![1., 0., 0.],
            "vdpol" => vec![2., 0.],
            "hires" => vec![1., 0., 0., 0., 0., 0., 0., 0.0057],
            _ => vec![1., 2., 3.],
        }
    }
    fn end(self) -> f64 {
        match self.0 {
            "rober" => 1e11,
            "vdpol" => 2.,
            "hires" => 321.8122,
            _ => 360.,
        }
    }
}
impl OdeSystem<f64> for Problem {
    fn dimension(&self) -> usize {
        match self.0 {
            "vdpol" => 2,
            "hires" => 8,
            _ => 3,
        }
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[f64],
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(0.);
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        f: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        match self.0 {
            "rober" => {
                let a = 0.04 * y[0] - 1e4 * y[1] * y[2];
                let b = 3e7 * y[1] * y[1];
                f.copy_from_slice(&[-a, a - b, b]);
            }
            "vdpol" => f.copy_from_slice(&[y[1], ((1. - y[0] * y[0]) * y[1] - y[0]) / 1e-6]),
            "orego" => f.copy_from_slice(&[
                77.27 * (y[1] + y[0] * (1. - 8.375e-6 * y[0] - y[1])),
                (y[2] - (1. + y[0]) * y[1]) / 77.27,
                0.161 * (y[0] - y[2]),
            ]),
            _ => {
                let r = 280. * y[5] * y[7];
                f.copy_from_slice(&[
                    -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007,
                    1.71 * y[0] - 8.75 * y[1],
                    -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4],
                    8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
                    -1.745 * y[4] + 0.43 * (y[5] + y[6]),
                    -r + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6],
                    r - 1.81 * y[6],
                    -r + 1.81 * y[6],
                ]);
            }
        }
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..self.dimension() {
            for k in 0..self.dimension() {
                j.set(i, k, 0.)?;
            }
        }
        let entries: &[(usize, usize, f64)] = match self.0 {
            "rober" => &[
                (0, 0, -0.04),
                (0, 1, 1e4 * y[2]),
                (0, 2, 1e4 * y[1]),
                (1, 0, 0.04),
                (1, 1, -1e4 * y[2] - 6e7 * y[1]),
                (1, 2, -1e4 * y[1]),
                (2, 1, 6e7 * y[1]),
            ],
            "vdpol" => &[
                (0, 1, 1.),
                (1, 0, (-2. * y[0] * y[1] - 1.) / 1e-6),
                (1, 1, (1. - y[0] * y[0]) / 1e-6),
            ],
            "orego" => &[
                (0, 0, 77.27 * (1. - 2. * 8.375e-6 * y[0] - y[1])),
                (0, 1, 77.27 * (1. - y[0])),
                (1, 0, -y[1] / 77.27),
                (1, 1, -(1. + y[0]) / 77.27),
                (1, 2, 1. / 77.27),
                (2, 0, 0.161),
                (2, 2, -0.161),
            ],
            _ => &[
                (0, 0, -1.71),
                (0, 1, 0.43),
                (0, 2, 8.32),
                (1, 0, 1.71),
                (1, 1, -8.75),
                (2, 2, -10.03),
                (2, 3, 0.43),
                (2, 4, 0.035),
                (3, 1, 8.32),
                (3, 2, 1.71),
                (3, 3, -1.12),
                (4, 4, -1.745),
                (4, 5, 0.43),
                (4, 6, 0.43),
                (5, 3, 0.69),
                (5, 4, 1.71),
                (5, 5, -280. * y[7] - 0.43),
                (5, 6, 0.69),
                (5, 7, -280. * y[5]),
                (6, 5, 280. * y[7]),
                (6, 6, -1.81),
                (6, 7, 280. * y[5]),
                (7, 5, -280. * y[7]),
                (7, 6, 1.81),
                (7, 7, -280. * y[5]),
            ],
        };
        for &(i, k, v) in entries {
            j.set(i, k, v)?;
        }
        Ok(Some(()))
    }
}

/// The same problems without Jacobians: stages use finite-difference Jacobians.
struct RhsOnly(Problem);
impl OdeSystem<f64> for RhsOnly {
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        f: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.0.rhs(t, y, f, cx)
    }
}

fn run<M: OdeMethod<f64> + Clone>(
    method: M,
    name: &str,
    comparator: &str,
    step_ratio: f64,
    error_ratio: f64,
) {
    let mut failures = Vec::new();
    let mut cases = 0;
    // Reuse one workspace across epochs and dimensions to also exercise cache reset.
    let mut workspace = OdeWorkspace::new();
    for line in include_str!("data/stiff.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let row: Vec<_> = line.split_whitespace().collect();
        if row[2] != comparator {
            continue;
        }
        cases += 1;
        let problem = Problem(match row[0] {
            "rober" => "rober",
            "vdpol" => "vdpol",
            "hires" => "hires",
            _ => "orego",
        });
        let rtol: f64 = row[1].parse().unwrap();
        let steps: f64 = row[3].parse().unwrap();
        let baseline: f64 = row[4].parse().unwrap();
        let error_limit = if comparator == "Radau" {
            1.0
        } else if name.starts_with("rosenbrock") {
            10.0
        } else {
            error_ratio * baseline
        };
        let reference: Vec<f64> = row[5].split(',').map(|v| v.parse().unwrap()).collect();
        let options = OdeOptions::new().with_error_tolerance(
            ErrorTolerance::scalar(1e-4 * rtol, rtol, NormSelection::RootMeanSquare).unwrap(),
        );
        let plan = (method.clone(), options);
        let rhs_only = RhsOnly(problem);
        let system: &dyn OdeSystem<f64> = if name.ends_with("_fd") {
            &rhs_only
        } else {
            &problem
        };
        let mut state = OdeState::new(0., Vector::from_vec(problem.initial()));
        let outcome = plan.0.advance(
            &OdeProblem::new(system, problem.end()),
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        );
        match outcome {
            Ok(report) => {
                let etu = state
                    .values()
                    .iter()
                    .zip(&reference)
                    .map(|(y, r)| (y - r).abs() / (rtol * (1e-4 + r.abs())))
                    .fold(0_f64, f64::max);
                let accepted = report.counts().accepted();
                // Cold-start baselines on 991da1dc5 at rtol=1e-6. Counting all
                // nested iterations is at least as strict as the old successful
                // proposal-only Newton totals (failed attempts were omitted).
                let work_baseline = match (name, problem.0, rtol) {
                    ("bdf", "hires", 1e-6) => Some((317, 1178)),
                    ("bdf", "vdpol", 1e-6) => Some((1236, 3903)),
                    _ => None,
                };
                let newton = report.accounting().nested().iterations();
                if let Some((old_steps, old_newton)) = work_baseline
                    && (accepted > old_steps || newton * old_steps > old_newton * accepted)
                {
                    failures.push(format!(
                        "{name} {} work regression: accepted={accepted}, newton={newton}",
                        problem.0,
                    ));
                }
                if name == "bdf" && rtol == 1e-6 && problem.0 != "rober" {
                    let jacobians: usize = row[6].parse().unwrap();
                    let factors: usize = row[7].parse().unwrap();
                    let (old_steps, old_etu) = match problem.0 {
                        // Accepted-step baselines from #2295; ETU measured on
                        // 56b684ccc. Permit 0.1% roundoff/controller jitter.
                        "vdpol" => (1258, 8.9210),
                        "hires" => (320, 10.6985),
                        _ => (1396, 50.1722),
                    };
                    let counts = report.counts();
                    if counts.provider(scioxide_core::EvaluationRole::Jacobian) > 2 * jacobians
                        || counts.provider(scioxide_core::EvaluationRole::Factorization)
                            > 2 * factors
                        || accepted > old_steps
                        || etu > old_etu * 1.001
                    {
                        failures.push(format!("{name} {} reuse regression: steps={accepted}, ETU={etu}, counts={counts}", problem.0));
                    }
                }
                eprintln!(
                    "{name} {} {rtol}: {:?}, steps={accepted}, newton={newton}, J={}, LU={}, ETU={etu:.4}, limits={:.0}/{:.4}",
                    problem.0,
                    report.termination(),
                    report
                        .counts()
                        .provider(scioxide_core::EvaluationRole::Jacobian),
                    report
                        .counts()
                        .provider(scioxide_core::EvaluationRole::Factorization),
                    step_ratio * steps,
                    error_limit
                );
                if report.termination() != Some(OdeTermination::ReachedEnd)
                    || (!name.starts_with("rosenbrock") && accepted as f64 > step_ratio * steps)
                    || etu > error_limit
                {
                    failures.push(format!(
                        "{name} {} {rtol}: {:?}, steps={accepted}, ETU={etu}",
                        problem.0,
                        report.termination()
                    ));
                }
            }
            Err(error) => failures.push(format!("{name} {} {rtol}: {error:?}", problem.0)),
        }
    }
    assert_eq!(
        cases, 8,
        "all four problems at both tolerances must execute"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn radau_reference() {
    run(RadauIia::new(), "radau", "Radau", 1.5, 0.0);
}

#[test]
fn radau_finite_difference_reference() {
    run(RadauIia::new(), "radau_fd", "Radau", 1.5, 0.0);
}

/// Robertson to `t = 10¹¹` at the default tolerances (absolute `10⁻⁶`, relative
/// `10⁻³`), with analytic and finite-difference Jacobians. Late steps exceed `10¹⁰`,
/// where a non-contracting chord iteration can look converged; SciPy's Radau takes
/// 78 steps.
#[test]
fn radau_default_tolerance_robertson() {
    let reference = [
        2.083_340_149_701_255e-8,
        8.333_360_770_334_713e-14,
        0.999_999_979_166_505,
    ];
    let problem = Problem("rober");
    let rhs_only = RhsOnly(problem);
    for system in [&problem as &dyn OdeSystem<f64>, &rhs_only] {
        let plan = (RadauIia::new(), OdeOptions::new());
        let mut state = OdeState::new(0., Vector::from_vec(problem.initial()));
        let report = plan
            .0
            .advance(
                &OdeProblem::new(system, problem.end()),
                &plan.1,
                &mut state,
                &mut (),
                &mut OdeWorkspace::new(),
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        let etu = state
            .values()
            .iter()
            .zip(reference)
            .map(|(y, r)| (y - r).abs() / (1e-6 + 1e-3 * r.abs()))
            .fold(0_f64, f64::max);
        let steps = report.accepted_steps();
        eprintln!("radau rober default: steps={steps}, ETU={etu:.4}");
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert!(etu <= 1.0 && steps <= 117, "steps={steps}, ETU={etu}");
    }
}

#[test]
fn bdf_reference() {
    run(
        Bdf::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
        "bdf",
        "BDF",
        1.5,
        3.,
    );
}
#[test]
fn ndf_reference() {
    run(
        Bdf::with_order_policy(MultistepOrder::Automatic(5))
            .unwrap()
            .with_formula(BdfFormula::Ndf),
        "ndf",
        "BDF",
        1.5,
        3.,
    );
}
#[test]
fn rosenbrock_reference() {
    run(
        RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap(),
        "rosenbrock",
        "BDF",
        f64::INFINITY,
        0.,
    );
}
/// Right-hand-side-only systems reach the same bounds with finite-difference Jacobians.
#[test]
fn bdf_finite_difference_reference() {
    run(
        Bdf::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
        "bdf_fd",
        "BDF",
        1.5,
        3.,
    );
}
#[test]
fn rosenbrock_finite_difference_reference() {
    run(
        RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap(),
        "rosenbrock_fd",
        "BDF",
        f64::INFINITY,
        0.,
    );
}
#[test]
fn adams_bdf_reference() {
    run(AdamsBdf::adams_bdf().unwrap(), "adams_bdf", "LSODA", 2., 3.);
}
/// Without Jacobians, difference probes detect stiffness at the same bounds.
#[test]
fn adams_bdf_finite_difference_reference() {
    run(
        AdamsBdf::adams_bdf().unwrap(),
        "adams_bdf_fd",
        "LSODA",
        2.,
        3.,
    );
}

struct RobertsonDae;
impl DaeSystem<f64> for RobertsonDae {
    fn dimension(&self) -> usize {
        3
    }
    fn residual(
        &self,
        _: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let a = 0.04 * y[0] - 1e4 * y[1] * y[2];
        out.copy_from_slice(&[
            d[0] + a,
            d[1] - a + 3e7 * y[1] * y[1],
            y[0] + y[1] + y[2] - 1.,
        ]);
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..3 {
            for j in 0..3 {
                a.set(i, j, 0.)?;
                b.set(i, j, 0.)?;
            }
        }
        for (i, j, v) in [
            (0, 0, 0.04),
            (0, 1, -1e4 * y[2]),
            (0, 2, -1e4 * y[1]),
            (1, 0, -0.04),
            (1, 1, 1e4 * y[2] + 6e7 * y[1]),
            (1, 2, 1e4 * y[1]),
            (2, 0, 1.),
            (2, 1, 1.),
            (2, 2, 1.),
        ] {
            a.set(i, j, v)?;
        }
        b.set(0, 0, 1.)?;
        b.set(1, 1, 1.)?;
        Ok(Some(()))
    }

    fn sparse_derivatives(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        pattern: scioxide_core::StructuralPatternView<'_>,
        a: &mut [f64],
        b: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for (i, bounds) in pattern.row_ptr().windows(2).enumerate() {
            for k in bounds[0]..bounds[1] {
                let j = pattern.col_indices()[k];
                a[k] = match (i, j) {
                    (0, 0) => 0.04,
                    (0, 1) => -1e4 * y[2],
                    (0, 2) => -1e4 * y[1],
                    (1, 0) => -0.04,
                    (1, 1) => 1e4 * y[2] + 6e7 * y[1],
                    (1, 2) => 1e4 * y[1],
                    _ => 1.,
                };
                b[k] = if i == j && i < 2 { 1. } else { 0. };
            }
        }
        Ok(Some(()))
    }
}
#[test]
fn residual_bdf_robertson_reference() {
    for line in include_str!("data/stiff.tsv")
        .lines()
        .filter(|s| s.starts_with("rober40"))
    {
        let row: Vec<_> = line.split_whitespace().collect();
        let rtol: f64 = row[1].parse().unwrap();
        let reference: Vec<f64> = row[5].split(',').map(|s| s.parse().unwrap()).collect();
        let tolerance =
            ErrorTolerance::scalar(1e-4 * rtol, rtol, NormSelection::RootMeanSquare).unwrap();
        let options = DaeOptions::new()
            .with_numerics(
                scioxide_ode::DaeNumericalOptions::new()
                    .with_differential_error(tolerance.clone())
                    .with_algebraic_error(tolerance),
            )
            .with_initial_step(1e-6);
        let plan = (
            DaeBdf::new(
                DaePartition::new(&[true, true, false], &[true, true, false]).unwrap(),
                OdeStageBackend::Dense,
            )
            .unwrap(),
            options,
        );
        let solution = plan
            .0
            .solve(
                &DaeProblem::new(&RobertsonDae, 40.),
                &plan.1,
                DaeState::from_slices(0., &[1., 0., 0.], &[-0.04, 0.04, 0.]).unwrap(),
            )
            .unwrap();
        let report = *solution.report();
        let y = solution.state().values();
        let etu = y
            .iter()
            .zip(reference)
            .map(|(y, r)| (y - r).abs() / (rtol * (1e-4 + r.abs())))
            .fold(0_f64, f64::max);
        eprintln!(
            "DAE Robertson {rtol}: {:?}, steps={}, Jacobians={}, LU={}, ETU={etu}",
            report.termination(),
            report.accepted_steps(),
            report.counts().provider(EvaluationRole::Jacobian),
            report.counts().provider(EvaluationRole::Factorization),
        );
        let reference_jacobians: usize = row[6].parse().unwrap();
        let reference_factors: usize = row[7].parse().unwrap();
        assert!(report.counts().provider(EvaluationRole::Jacobian) <= 2 * reference_jacobians);
        assert!(report.counts().provider(EvaluationRole::Factorization) <= 2 * reference_factors);
        let baseline_etu = if rtol == 1e-6 {
            2.4558262840487277
        } else {
            6.510558289458388
        };
        assert!(etu <= 1.5 * baseline_etu, "{etu}");
        assert!(report.is_success(), "{:?}", report.termination());
        assert!(etu <= 10., "{etu}");
        assert!(
            report.accepted_steps() <= if rtol == 1e-6 { 300 } else { 600 },
            "{}",
            report.accepted_steps()
        );
    }
}

struct RobertsonResidual(core::cell::Cell<Option<(f64, f64, f64)>>);
impl scioxide_core::Preconditioner<f64> for RobertsonResidual {
    fn dimension(&self) -> usize {
        3
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn precondition_into(
        &self,
        rhs: &[f64],
        out: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        if let Some((inverse, b, c)) = self.0.get() {
            // Eliminate the constraint, then solve the differential 2x2 block.
            let (a00, a01, a10, a11) = (
                inverse + 0.04 + b,
                -c + b,
                -0.04 - b,
                inverse + c + 6e3 * b - b,
            );
            let (r0, r1) = (rhs[0] + b * rhs[2], rhs[1] - b * rhs[2]);
            let determinant = a00 * a11 - a01 * a10;
            out[0] = (r0 * a11 - a01 * r1) / determinant;
            out[1] = (a00 * r1 - r0 * a10) / determinant;
            out[2] = rhs[2] - out[0] - out[1];
        } else {
            out.copy_from_slice(rhs);
        }
        Ok(())
    }
}
impl DaeSystem<f64> for RobertsonResidual {
    fn dimension(&self) -> usize {
        3
    }
    fn residual(
        &self,
        t: f64,
        y: &[f64],
        v: &[f64],
        out: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        RobertsonDae.residual(t, y, v, out, cx)
    }
    fn preconditioner(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        initialization: bool,
        inverse: f64,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<&dyn scioxide_core::Preconditioner<f64>>, SciOxideError> {
        self.0.set(if initialization {
            None
        } else {
            Some((inverse, 1e4 * y[1], 1e4 * y[2]))
        });
        Ok(Some(self))
    }
}
#[test]
fn residual_only_robertson_matches_analytic_blocks() {
    let residual = RobertsonResidual(core::cell::Cell::new(None));
    for row in include_str!("data/stiff.tsv")
        .lines()
        .filter(|s| s.starts_with("rober40"))
    {
        let row: Vec<_> = row.split_whitespace().collect();
        let rtol: f64 = row[1].parse().unwrap();
        let reference: Vec<f64> = row[5].split(',').map(|s| s.parse().unwrap()).collect();
        let tolerance =
            ErrorTolerance::scalar(1e-4 * rtol, rtol, NormSelection::RootMeanSquare).unwrap();
        let options = DaeOptions::new()
            .with_numerics(
                scioxide_ode::DaeNumericalOptions::new()
                    .with_differential_error(tolerance.clone())
                    .with_algebraic_error(tolerance),
            )
            .with_initial_step(1e-6);
        let partition = DaePartition::new(&[true, true, false], &[true, true, false]).unwrap();
        let backends = [
            OdeStageBackend::Dense,
            OdeStageBackend::Sparse {
                pattern: scioxide_core::StructuralPattern::new(
                    3,
                    3,
                    vec![0, 3, 6, 9],
                    vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
                )
                .unwrap(),
                limits: scioxide_sparse::SparseLuLimits::new(32, 1024 * 1024),
            },
            OdeStageBackend::Krylov {
                iterations: scioxide_core::Budget::try_from(100).unwrap(),
                restart: scioxide_core::Budget::try_from(3).unwrap(),
                operator_scratch: 0,
                preconditioner_scratch: 0,
            },
        ];
        let initial = || DaeState::from_slices(0., &[1., 0., 0.], &[-0.04, 0.04, 0.]).unwrap();
        let etu = |y: &[f64]| {
            y.iter()
                .zip(&reference)
                .map(|(y, r)| (y - r).abs() / (rtol * (1e-4 + r.abs())))
                .fold(0_f64, f64::max)
        };
        let analytic = DaeBdf::new(partition.clone(), OdeStageBackend::Dense)
            .unwrap()
            .solve(&DaeProblem::new(&RobertsonDae, 40.), &options, initial())
            .unwrap();
        let baseline = etu(analytic.state().values());
        for backend in backends {
            let krylov = matches!(backend, OdeStageBackend::Krylov { .. });
            let plan = (
                DaeBdf::new(partition.clone(), backend).unwrap(),
                options.clone(),
            );
            let baseline = if !krylov {
                let analytic = plan
                    .0
                    .solve(&DaeProblem::new(&RobertsonDae, 40.), &plan.1, initial())
                    .unwrap();
                assert!(analytic.report().is_success());
                etu(analytic.state().values())
            } else {
                baseline
            };
            let solution = plan
                .0
                .solve(&DaeProblem::new(&residual, 40.), &plan.1, initial())
                .unwrap();
            let error = etu(solution.state().values());
            assert!(
                solution.report().is_success(),
                "rtol={rtol}, krylov={krylov}: {:?}",
                solution.report().termination()
            );
            assert!(
                krylov || error <= 2. * baseline,
                "rtol={rtol}: ETU={error}, analytic={baseline}"
            );
        }
    }
}
