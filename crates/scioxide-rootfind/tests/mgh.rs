//! Independent original MINPACK problem definitions and hybrj/hybrd outcomes.
//!
//! Portable scoped execution fixes the reference trajectories; native conveniences
//! are independently residual-checked by the solver/API suites.
//!
//! Every unsuccessful or over-allowance outcome must be listed in
//! `data/mgh_expected.tsv` with its exact std and no_std status and a reason that
//! [`verified`] checks against the reference data and the table itself. no_std
//! uses SciOxide's owned elementary kernels, so its column is host independent.
//! std routes `exp`, `atan`, `sin` and `cos` to the platform library, which may
//! differ by host within the elementary envelope; for problems using them a std
//! cell may list `|`-separated unsuccessful alternatives ([`host_alternatives`]).
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
#[path = "support/mgh.rs"]
mod mgh;
use mgh::{Mgh, r};
use scioxide_core::{
    Budget, NumericScalar, RealScalar, ResidualTolerance, SciOxideError, StepTolerance,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    BacktrackingLineSearch, Broyden, BroydenUpdate, NewtonKrylovOptions, NewtonKrylovProblem,
    NewtonKrylovSolver, NewtonSystem, SystemRootSolver, TrustRegionSystem,
};

fn rows() -> impl Iterator<Item = Vec<&'static str>> {
    include_str!("data/mgh.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split('\t').collect())
}
fn rounding_rows() -> impl Iterator<Item = Vec<&'static str>> {
    include_str!("data/mgh_rounding.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split('\t').collect())
}

/// Validate the three probes and their aggregate flag; return (all, any) qualified.
fn rounding_qualification(runs: &[Vec<&str>]) -> Option<(bool, bool)> {
    if runs.len() != 3 || runs.iter().any(|r| r.len() != 13) {
        return None;
    }
    let labels: std::collections::BTreeSet<_> = runs.iter().map(|r| r[4]).collect();
    if labels != ["original", "up", "down"].into_iter().collect()
        || runs
            .iter()
            .any(|r| r[..4] != runs[0][..4] || !["0", "1"].contains(&r[6]))
    {
        return None;
    }
    let all = runs.iter().all(|r| r[6] == "1");
    let any = runs.iter().any(|r| r[6] == "1");
    runs.iter()
        .all(|r| r[12] == if all { "1" } else { "0" })
        .then_some((all, any))
}

fn numbers<S: RealScalar>(s: &str) -> Vec<S> {
    s.split(',').map(|v| r(v.parse().unwrap())).collect()
}
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 2e-10 * (1.0 + b.abs()),
            "entry {i}: {a} vs {b}"
        );
    }
}

#[test]
fn rounding_reference_coverage_and_qualification() {
    let mut groups = std::collections::BTreeMap::<_, Vec<_>>::new();
    for row in rounding_rows() {
        assert_eq!(row.len(), 13);
        assert!(rows().any(|r| r[..4] == row[..4]), "unknown rounding case");
        let p = Mgh::new(row[0].parse().unwrap(), row[1].parse().unwrap());
        let x = numbers::<f64>(row[10]);
        let residual = numbers::<f64>(row[11]);
        assert_eq!(x.len(), p.n);
        assert_eq!(residual.len(), p.n);
        assert!(x.iter().chain(&residual).all(|v| v.is_finite()));
        let mut actual = vec![0.0; p.n];
        p.values(&x, &mut actual);
        close(&actual, &residual);
        let actual_norm = actual.iter().fold(0.0f64, |a, &v| a.hypot(v));
        let norm = residual.iter().fold(0.0f64, |a, &v| a.hypot(v));
        let recorded = row[9].parse::<f64>().unwrap();
        assert!((norm - recorded).abs() <= 1e-14 * norm.max(f64::MIN_POSITIVE));
        assert_eq!(row[6], if norm <= 1e-8 { "1" } else { "0" });
        assert_eq!(actual_norm <= 1e-8, norm <= 1e-8);
        assert!((1..=5).contains(&row[5].parse::<usize>().unwrap()));
        assert!(row[7].parse::<usize>().unwrap() > 0);
        let jacobians = row[8].parse::<usize>().unwrap();
        assert_eq!(jacobians > 0, row[3] == "hybrj");
        groups.entry(row[..4].to_vec()).or_default().push(row);
    }
    assert_eq!(groups.len(), 132);
    for runs in groups.values() {
        assert!(
            rounding_qualification(runs).is_some(),
            "invalid probes: {runs:?}"
        );
    }
}
#[test]
fn original_minpack_residual_and_jacobian_checkpoints() {
    let mut count = 0;
    for row in rows() {
        let p = Mgh::new(row[0].parse().unwrap(), row[1].parse().unwrap());
        for (point, values, jacobian) in [(row[9], row[12], row[13]), (row[10], row[11], row[14])] {
            let x = numbers(point);
            let mut f = vec![0.0; p.n];
            let mut j = vec![0.0; p.n * p.n];
            p.values(&x, &mut f);
            p.jacobian(&x, &mut j);
            close(&f, &numbers(values));
            close(&j, &numbers(jacobian));
        }
        count += 1;
    }
    assert_eq!(count, 132);
    let expected: Vec<Vec<&str>> = include_str!("data/mgh_expected.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split('\t').collect())
        .collect();
    let mut keys = std::collections::BTreeSet::new();
    for fields in &expected {
        let line = fields.join("\t");
        assert_eq!(fields.len(), 8);
        assert!(
            keys.insert(fields[..5].join("\t")),
            "duplicate exception {line}"
        );
        assert!(
            rows().any(|r| r[..3] == fields[1..4]),
            "uncovered case {line}"
        );
        let (method, mode) = fields[4].rsplit_once('-').unwrap();
        assert!(
            ["newton", "newton-armijo", "broyden", "trust", "nk"].contains(&method)
                && ["analytic", "numerical"].contains(&mode),
            "uncovered solver {line}"
        );
        let success = |status: &str| ["ResidualTolerance", "Converged"].contains(&status);
        assert!(
            !success(fields[5]) || !success(fields[6]),
            "obsolete exception {line}"
        );
        assert!(
            !fields[5].contains('|') || host_alternatives(fields),
            "unsupported std alternatives {line}"
        );
        assert!(!fields[6].contains('|'), "no_std alternatives {line}");
        assert!(verified(fields, &expected), "unverified reason {line}");
    }
}

/// MINPACK problems whose residuals call natively routed transcendentals:
/// Powell badly scaled (`exp`), helical valley (`atan`) and trigonometric
/// (`sin`, `cos`). Square roots are correctly rounded and `powi` is owned.
fn native_transcendentals(problem: &str) -> bool {
    ["3", "5", "11"].contains(&problem)
}

/// Whether a std cell's `|`-separated alternatives are admissible: the problem
/// calls native transcendentals, the alternatives are distinct, and none
/// certifies success, so no host can turn an exception into a pass.
fn host_alternatives(fields: &[&str]) -> bool {
    let alternatives: Vec<_> = fields[5].split('|').collect();
    let distinct: std::collections::BTreeSet<_> = alternatives.iter().collect();
    native_transcendentals(fields[1])
        && alternatives.len() > 1
        && distinct.len() == alternatives.len()
        && alternatives
            .iter()
            .all(|status| !["ResidualTolerance", "Converged", ""].contains(status))
}

/// Whether an expected row's reason holds against the reference data and table.
///
/// - `reference-unqualified`: MINPACK's own result from this start fails the
///   checked residual (hybrd for numerical dense solvers, otherwise hybrj).
/// - `reference-rounding-sensitive`: the matching reference has mixed residual
///   qualification across its original and one-ULP up/down runs. This is a
///   bounded probe of rounding sensitivity, not a universal stability claim.
/// - `undamped`: Newton without a line search, whose iterates are fixed by the
///   undamped recurrence.
/// - `rounding-fragile`: exactly one of the std and no_std transcendental
///   implementations succeeds.
/// - `line-search`: a line-search method (Armijo Newton, Broyden or Newton–Krylov)
///   at a start where Powell's hybrid trust region with the same derivative mode
///   succeeds within the work allowance, so it has no row.
/// - A follow-up issue link records a remaining reference-qualified limitation.
fn verified(fields: &[&str], expected: &[Vec<&str>]) -> bool {
    let (method, mode) = fields[4].rsplit_once('-').unwrap();
    let success = |status: &str| ["ResidualTolerance", "Converged"].contains(&status);
    let reference = if mode == "numerical" && method != "nk" {
        "hybrd"
    } else {
        "hybrj"
    };
    match fields[7] {
        "reference-unqualified" => {
            rows().any(|r| r[..3] == fields[1..4] && r[3] == reference && r[5] == "0")
        }
        "reference-rounding-sensitive" => {
            let runs: Vec<_> = rounding_rows()
                .filter(|r| r[..3] == fields[1..4] && r[3] == reference)
                .collect();
            rounding_qualification(&runs) == Some((false, true))
        }
        "undamped" => method == "newton",
        "rounding-fragile" => success(fields[5]) != success(fields[6]),
        "line-search" => {
            let trust = format!("trust-{mode}");
            ["newton-armijo", "broyden", "nk"].contains(&method)
                && !expected
                    .iter()
                    .any(|other| other[..4] == fields[..4] && other[4] == trust)
        }
        reason => reason.starts_with("https://github.com/kakune/scioxide/issues/"),
    }
}

struct Observation<S> {
    point: Vec<S>,
    success: bool,
    status: String,
    work: usize,
}
fn assess<S: RealScalar>(
    p: &Mgh,
    row: &[&str],
    name: &str,
    observation: Result<Observation<S>, SciOxideError>,
    tolerance: S,
    failures: &mut Vec<String>,
) {
    let precision = if S::EPSILON.into_f64() > 1e-10 {
        "f32"
    } else {
        "f64"
    };
    let key = format!("{precision}\t{}\t{}\t{}\t{name}", row[0], row[1], row[2]);
    let status = match observation {
        Ok(o) => {
            let mut f = vec![S::ZERO; p.n];
            p.values(&o.point, &mut f);
            let norm = f.iter().fold(0.0f64, |a, v| a.hypot(v.into_f64()));
            if o.success {
                assert!(
                    norm <= tolerance.into_f64() * 1.001,
                    "false success {key}: {norm}"
                );
            }
            let reference_work: usize =
                row[6].parse::<usize>().unwrap() + p.n * row[7].parse::<usize>().unwrap();
            let envelope = 100.max(10 * reference_work);
            if o.success && o.work > envelope {
                format!("WorkEnvelope:{}", o.status)
            } else {
                o.status
            }
        }
        Err(e) => format!("Error:{:?}", e.kind()),
    };
    let expected = include_str!("data/mgh_expected.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 8);
            (fields[..5].join("\t") == key)
                .then(|| fields[if cfg!(feature = "std") { 5 } else { 6 }])
        });
    if let Some(expected) = expected {
        if !expected.split('|').any(|alternative| alternative == status) {
            failures.push(format!("{key}\t{status}\texpected={expected}"));
        }
    } else if status != "ResidualTolerance" {
        failures.push(format!(
            "{key}\t{status}\treference={} qualified={}",
            row[4], row[5]
        ));
    }
}
fn systems<S: RealScalar + NumericScalar<Real = S>>(tolerance: S, single: bool) {
    let mut failures = vec![];
    let mut count = 0;
    for row in rows().filter(|r| r[3] == "hybrj") {
        let p = Mgh::new(row[0].parse().unwrap(), row[1].parse().unwrap());
        if single && ![1, 5, 9, 10, 13, 14].contains(&p.problem) {
            continue;
        }
        let x = numbers::<S>(row[9]);
        for strategy in [DerivativeStrategy::Analytic, DerivativeStrategy::Forward] {
            let numerical = if strategy == DerivativeStrategy::Analytic {
                "analytic"
            } else {
                "numerical"
            };
            let reference = rows()
                .find(|r| {
                    r[..3] == row[..3]
                        && r[3]
                            == if numerical == "analytic" {
                                "hybrj"
                            } else {
                                "hybrd"
                            }
                })
                .unwrap();
            macro_rules! options {
                ($type:ident) => {
                    scioxide_rootfind::$type::default()
                        .with_derivative_strategy(strategy)
                        .with_residual_tolerance(ResidualTolerance::new(tolerance, r(0.0)).unwrap())
                        .with_step_tolerance(StepTolerance::new(r(0.0), r(0.0)).unwrap())
                        .with_max_iterations(Budget::try_from(1000).unwrap())
                };
            }
            macro_rules! run {
                ($name:expr, $solver:expr) => {{
                    p.reset_counts();
                    let result = portable_dense(&$solver, &p, &x).map(|o| Observation {
                        point: o.point().to_vec(),
                        success: o.is_success(),
                        status: format!("{:?}", o.termination()),
                        work: p.residual_calls.get() + p.n * p.jacobian_calls.get(),
                    });
                    assess(
                        &p,
                        &reference,
                        &format!("{}-{numerical}", $name),
                        result,
                        tolerance,
                        &mut failures,
                    );
                }};
            }
            run!(
                "newton",
                NewtonSystem::new(options!(NewtonSystemOptions)).unwrap()
            );
            run!(
                "newton-armijo",
                NewtonSystem::new(
                    options!(NewtonSystemOptions)
                        .with_line_search(BacktrackingLineSearch::default())
                )
                .unwrap()
            );
            run!(
                "broyden",
                Broyden::new(
                    BroydenUpdate::First,
                    options!(BroydenOptions).with_line_search(BacktrackingLineSearch::default())
                )
                .unwrap()
            );
            run!(
                "trust",
                TrustRegionSystem::new(options!(TrustRegionSystemOptions)).unwrap()
            );
        }
        for analytic in [true, false] {
            let problem = if analytic {
                NewtonKrylovProblem::with_jacobian(&p)
            } else {
                NewtonKrylovProblem::numerical(&p)
            };
            let options = NewtonKrylovOptions::default()
                .with_residual_tolerance(ResidualTolerance::new(tolerance, r(0.0)).unwrap())
                .with_step_tolerance(StepTolerance::new(r(0.0), r(0.0)).unwrap())
                .with_max_iterations(Budget::try_from(1000).unwrap())
                .with_krylov(
                    Budget::try_from(4 * p.n + 20).unwrap(),
                    Budget::try_from(p.n).unwrap(),
                );
            p.reset_counts();
            let result = portable_newton_krylov(&problem, &x, options).map(|o| Observation {
                point: o.point().to_vec(),
                success: o.is_success(),
                status: format!("{:?}", o.termination()),
                work: p.residual_calls.get() + p.jvp_calls.get(),
            });
            let name = if analytic {
                "nk-analytic"
            } else {
                "nk-numerical"
            };
            assess(&p, &row, name, result, tolerance, &mut failures);
        }
        count += 1;
    }
    assert_eq!(count, if single { 21 } else { 66 });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[test]
fn reasons_reject_unsupported_claims() {
    let row = |solver: &'static str, std: &'static str, no_std: &'static str, reason| {
        vec!["f64", "1", "2", "1", solver, std, no_std, reason]
    };
    let table = [row(
        "trust-numerical",
        "Stagnation",
        "Stagnation",
        "https://github.com/kakune/scioxide/issues/1",
    )];
    // Rosenbrock's unit start qualifies in both references.
    assert!(!verified(
        &row(
            "broyden-analytic",
            "IterationLimit",
            "IterationLimit",
            "reference-unqualified"
        ),
        &table
    ));
    assert!(verified(
        &row(
            "newton-numerical",
            "IterationLimit",
            "IterationLimit",
            "undamped"
        ),
        &table
    ));
    assert!(!verified(
        &row(
            "newton-armijo-numerical",
            "IterationLimit",
            "IterationLimit",
            "undamped"
        ),
        &table
    ));
    assert!(verified(
        &row(
            "broyden-analytic",
            "IterationLimit",
            "ResidualTolerance",
            "rounding-fragile"
        ),
        &table
    ));
    assert!(!verified(
        &row(
            "broyden-analytic",
            "IterationLimit",
            "Stagnation",
            "rounding-fragile"
        ),
        &table
    ));
    // The analytic trust region succeeded; the numerical one is listed.
    assert!(verified(
        &row(
            "nk-analytic",
            "IterationLimit",
            "IterationLimit",
            "line-search"
        ),
        &table
    ));
    assert!(!verified(
        &row(
            "nk-numerical",
            "IterationLimit",
            "IterationLimit",
            "line-search"
        ),
        &table
    ));
    assert!(!verified(
        &row("trust-analytic", "Stagnation", "Stagnation", "line-search"),
        &table
    ));
    assert!(!verified(
        &row("trust-analytic", "Stagnation", "Stagnation", "unknown"),
        &table
    ));
    assert!(!verified(
        &row(
            "trust-numerical",
            "Stagnation",
            "Stagnation",
            "reference-rounding-sensitive"
        ),
        &table
    ));
    for solver in [
        "trust-numerical",
        "broyden-numerical",
        "trust-analytic",
        "nk-numerical",
    ] {
        let fields = [
            "f64",
            "6",
            "9",
            "10",
            solver,
            "Stagnation",
            "Stagnation",
            "reference-rounding-sensitive",
        ];
        assert_eq!(
            verified(&fields, &table),
            solver == "trust-numerical" || solver == "broyden-numerical"
        );
    }
    // Hybrj's 100x Powell badly scaled start does not qualify.
    assert!(verified(
        &[
            "f64",
            "3",
            "2",
            "100",
            "nk-analytic",
            "IterationLimit",
            "IterationLimit",
            "reference-unqualified"
        ],
        &table
    ));
    // Host alternatives need native transcendentals and never certify success.
    let alternatives = |problem: &'static str, std: &'static str| {
        [
            "f64",
            problem,
            "10",
            "10",
            "newton-analytic",
            std,
            "IterationLimit",
            "undamped",
        ]
    };
    assert!(host_alternatives(&alternatives(
        "11",
        "IterationLimit|WorkEnvelope:ResidualTolerance"
    )));
    for (problem, std) in [
        ("9", "IterationLimit|WorkEnvelope:ResidualTolerance"),
        ("11", "IterationLimit|ResidualTolerance"),
        ("11", "IterationLimit|IterationLimit"),
        ("11", "IterationLimit|"),
        ("11", "IterationLimit"),
    ] {
        assert!(!host_alternatives(&alternatives(problem, std)), "{std}");
    }
}

#[test]
fn rounding_metadata_rejects_incomplete_or_inconsistent_probes() {
    let runs: Vec<_> = rounding_rows()
        .filter(|r| r[..4] == ["6", "9", "10", "hybrd"])
        .collect();
    assert_eq!(rounding_qualification(&runs), Some((false, true)));
    assert_eq!(rounding_qualification(&runs[..2]), None);
    let mut malformed = runs.clone();
    malformed[0].pop();
    assert_eq!(rounding_qualification(&malformed), None);
    let mut duplicate = runs.clone();
    duplicate[1][4] = duplicate[0][4];
    assert_eq!(rounding_qualification(&duplicate), None);
    let mut wrong_case = runs.clone();
    wrong_case[1][0] = "1";
    assert_eq!(rounding_qualification(&wrong_case), None);
    let mut wrong_flag = runs.clone();
    wrong_flag[0][12] = "1";
    assert_eq!(rounding_qualification(&wrong_flag), None);
    for (flag, expected) in [("0", (false, false)), ("1", (true, true))] {
        let mut uniform = runs.clone();
        for run in &mut uniform {
            run[6] = flag;
            run[12] = flag;
        }
        assert_eq!(rounding_qualification(&uniform), Some(expected));
    }
    let mut invalid_flag = runs;
    invalid_flag[0][6] = "true";
    assert_eq!(rounding_qualification(&invalid_flag), None);
}

#[test]
fn mgh_f64() {
    systems::<f64>(1e-8, false);
}
#[test]
fn mgh_f32() {
    systems::<f32>(1e-4, true);
}

fn portable_dense<S, M>(
    solver: &M,
    problem: &Mgh,
    initial: &[S],
) -> Result<scioxide_rootfind::SystemRootResult<S>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    M: SystemRootSolver<S, Mgh>,
{
    let mut workspace = solver.workspace(problem)?;
    let mut output = scioxide_rootfind::SystemRootResult::empty();
    output.try_reserve(problem.n)?;
    solver.solve_into(
        problem,
        initial,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )?;
    Ok(output)
}
fn portable_newton_krylov<S: RealScalar + NumericScalar<Real = S>>(
    problem: &NewtonKrylovProblem<'_, S>,
    initial: &[S],
    options: NewtonKrylovOptions<S>,
) -> Result<scioxide_rootfind::NewtonKrylovResult<S>, SciOxideError> {
    let solver = NewtonKrylovSolver::new(options)?;
    let mut workspace = solver.workspace(problem)?;
    let mut output =
        scioxide_rootfind::NewtonKrylovResult::new(problem.dimension(), solver.options())?;
    solver.solve_into(
        problem,
        initial,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )?;
    Ok(output)
}

/// The existing exception set is the budget for a representation change. Exact
/// failure trajectories remain checked above; this check also rejects a new
/// unsuccessful or over-allowance case when other existing trajectories change.
fn broyden_exception_set<S: RealScalar + NumericScalar<Real = S>>(tolerance: S, single: bool) {
    let precision = if single { "f32" } else { "f64" };
    let mut failures = Vec::new();
    for row in rows().filter(|row| row[3] == "hybrj") {
        let problem = Mgh::new(row[0].parse().unwrap(), row[1].parse().unwrap());
        if single && ![1, 5, 9, 10, 13, 14].contains(&problem.problem) {
            continue;
        }
        let point = numbers::<S>(row[9]);
        for (strategy, mode) in [
            (DerivativeStrategy::Analytic, "analytic"),
            (DerivativeStrategy::Forward, "numerical"),
        ] {
            let key = format!(
                "{precision}\t{}\t{}\t{}\tbroyden-{mode}",
                row[0], row[1], row[2]
            );
            let exception = include_str!("data/mgh_expected.tsv")
                .lines()
                .filter(|line| !line.starts_with('#'))
                .any(|line| {
                    let fields: Vec<_> = line.split('\t').collect();
                    fields[..5].join("\t") == key
                        && fields[if cfg!(feature = "std") { 5 } else { 6 }] != "ResidualTolerance"
                });
            let reference = rows()
                .find(|reference| {
                    reference[..3] == row[..3]
                        && reference[3] == if mode == "analytic" { "hybrj" } else { "hybrd" }
                })
                .unwrap();
            let reference_work = reference[6].parse::<usize>().unwrap()
                + problem.n * reference[7].parse::<usize>().unwrap();
            let solver = Broyden::new(
                BroydenUpdate::First,
                scioxide_rootfind::BroydenOptions::default()
                    .with_derivative_strategy(strategy)
                    .with_line_search(BacktrackingLineSearch::default())
                    .with_residual_tolerance(ResidualTolerance::new(tolerance, S::ZERO).unwrap())
                    .with_step_tolerance(StepTolerance::new(S::ZERO, S::ZERO).unwrap())
                    .with_max_iterations(Budget::try_from(1000).unwrap()),
            )
            .unwrap();
            problem.reset_counts();
            match portable_dense(&solver, &problem, &point) {
                Ok(result) => {
                    if result.is_success() {
                        let mut residual = vec![S::ZERO; problem.n];
                        problem.values(result.point(), &mut residual);
                        let norm = residual
                            .iter()
                            .fold(0.0_f64, |norm, x| norm.hypot(x.into_f64()));
                        assert!(
                            norm <= tolerance.into_f64() * 1.001,
                            "false success {key}: {norm}"
                        );
                    }
                    let work =
                        problem.residual_calls.get() + problem.n * problem.jacobian_calls.get();
                    if !exception && (!result.is_success() || work > 100.max(10 * reference_work)) {
                        failures.push(format!("{key}: {:?}, work={work}", result.termination()));
                    }
                }
                Err(error) if !exception => failures.push(format!("{key}: {error:?}")),
                Err(_) => {}
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[test]
fn broyden_has_no_new_mgh_exceptions_f64() {
    broyden_exception_set::<f64>(1e-8, false);
}
#[test]
fn broyden_has_no_new_mgh_exceptions_f32() {
    broyden_exception_set::<f32>(1e-4, true);
}
