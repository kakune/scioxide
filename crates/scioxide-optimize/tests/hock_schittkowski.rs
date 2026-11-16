//! Pinned HS definitions, independent KKT checks, and solver work bounds.
#![allow(clippy::unwrap_used, clippy::panic, clippy::suspicious_arithmetic_impl)]
#[path = "support/reference_program.rs"]
mod reference_program;
use reference_program::{Program, Token};
use scioxide_core::{
    Bounds, Budget, FeasibilityTolerance, NumericScalar, RealScalar, SciOxideError,
};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    Cobyla, CobylaOptions, Cobyqa, CobyqaOptions, ConstrainedProblem, ConstraintFn, Constraints,
    ObjectiveFunction,
};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
struct Deck {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    problem: usize,
    n: usize,
    initial_radius: f64,
    minimum: f64,
    published_minimum: Option<f64>,
    minimizer: Option<Vec<f64>>,
    start: Vec<f64>,
    bounds: Vec<(Option<f64>, Option<f64>)>,
    objective: Vec<Token>,
    equalities: Vec<Vec<Token>>,
    inequalities: Vec<Vec<Token>>,
    references: Vec<Reference>,
}
#[derive(Deserialize)]
struct Reference {
    method: String,
    success: bool,
    status: i32,
    nfev: usize,
    x: Vec<f64>,
    f: f64,
    gradient: Vec<f64>,
    constraints: Vec<f64>,
    jacobian: Vec<f64>,
}
struct Function<S> {
    n: usize,
    code: Program<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> ObjectiveFunction<S> for Function<S> {
    fn dim(&self) -> usize {
        self.n
    }
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(self.code.evaluate(x, None, None).value)
    }
    fn gradient_into(&self, x: &[S], out: &mut [S]) -> Result<bool, SciOxideError> {
        for (j, v) in out.iter_mut().enumerate() {
            *v = self.code.evaluate(x, Some(j), None).first;
        }
        Ok(true)
    }
    fn hessian_into(&self, x: &[S], out: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        for i in 0..self.n {
            for j in 0..self.n {
                out.as_mut_slice()[i * self.n + j] = self.code.evaluate(x, Some(i), Some(j)).mixed;
            }
        }
        Ok(true)
    }
}
fn deck() -> Deck {
    serde_json::from_str(include_str!("data/hs.json")).unwrap()
}
fn function<S: RealScalar>(case: &Case) -> Function<S> {
    Function {
        n: case.n,
        code: Program::new(&case.objective, case.n),
    }
}
fn constraints<S: RealScalar + NumericScalar<Real = S>>(case: &Case, scale: S) -> Constraints<S> {
    let mut constraints = Constraints::new(case.n).unwrap();
    for (equality, programs) in [(true, &case.equalities), (false, &case.inequalities)] {
        for tokens in programs {
            let values = Arc::new(Program::new(tokens, case.n));
            let derivatives = Arc::clone(&values);
            let n = case.n;
            let row = ConstraintFn::new(n, move |x: &[S]| {
                Ok(scale * values.evaluate(x, None, None).value)
            })
            .with_jacobian(move |x: &[S], output| {
                for (j, value) in output.iter_mut().enumerate() {
                    *value = scale * derivatives.evaluate(x, Some(j), None).first;
                }
                Ok(())
            });
            constraints = if equality {
                constraints.equality(row).unwrap()
            } else {
                constraints.inequality(row).unwrap()
            };
        }
    }
    let bounds = case
        .bounds
        .iter()
        .map(|&(lo, hi)| {
            (
                lo.map(|v| S::try_from_f64_lossy(v).unwrap()),
                hi.map(|v| S::try_from_f64_lossy(v).unwrap()),
            )
        })
        .collect();
    constraints.bounds(Bounds::new(bounds).unwrap()).unwrap()
}

/// Solve with portable sequential dispatch, the execution the pinned matrix and
/// references were recorded with.
macro_rules! portable {
    ($solver:expr, $problem:expr, $x0:expr) => {{
        let solver = $solver;
        let problem = $problem;
        solver.workspace(problem).and_then(|mut workspace| {
            let mut output = scioxide_optimize::OptimizeResult::empty();
            solver
                .minimize_into(
                    problem,
                    $x0,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .map(|()| output)
        })
    }};
}

#[derive(Debug)]
struct Kkt {
    raw_violation: f64,
    normalized_violation: f64,
    stationarity: f64,
    dual: f64,
    complementarity: f64,
}
fn kkt(case: &Case, x: &[f64]) -> Kkt {
    kkt_at(case, x, 0.02)
}
fn kkt_at(case: &Case, x: &[f64], active_window: f64) -> Kkt {
    let f = function::<f64>(case);
    let mut gradient = vec![0.; case.n];
    f.gradient_into(x, &mut gradient).unwrap();
    let scale = gradient.iter().fold(1_f64, |a, &b| a.max(b.abs()));
    let mut rows = Vec::<(Vec<f64>, f64, bool)>::new();
    let mut violation = 0_f64;
    let mut normalized_violation = 0_f64;
    for (equality, programs) in [(true, &case.equalities), (false, &case.inequalities)] {
        for tokens in programs {
            let p = Program::<f64>::new(tokens, case.n);
            let value = p.evaluate(x, None, None).value;
            violation = violation.max(if equality { value.abs() } else { value.max(0.) });
            let g: Vec<_> = (0..case.n)
                .map(|j| p.evaluate(x, Some(j), None).first)
                .collect();
            let norm = g
                .iter()
                .fold(0_f64, |a, &v| a.max(v.abs()))
                .max(f64::MIN_POSITIVE);
            normalized_violation = normalized_violation.max(if equality {
                value.abs() / norm
            } else {
                value.max(0.) / norm
            });
            if equality || value / norm >= -active_window {
                rows.push((
                    g.iter().map(|v| v / norm).collect(),
                    value / norm,
                    !equality,
                ));
            }
        }
    }
    for (j, &(lo, hi)) in case.bounds.iter().enumerate() {
        for (endpoint, sign) in [(lo, -1.), (hi, 1.)] {
            if let Some(b) = endpoint {
                let value = sign * (x[j] - b);
                violation = violation.max(value.max(0.));
                normalized_violation = normalized_violation.max(value.max(0.));
                if value >= -active_window {
                    let mut row = vec![0.; case.n];
                    row[j] = sign;
                    rows.push((row, value, true));
                }
            }
        }
    }
    // Independent, reorthogonalized QR of active normals. No solver-supplied
    // multipliers, gradients, active sets, or optimality diagnostics are used.
    let mut q = Vec::<Vec<f64>>::new();
    let mut selected = Vec::new();
    let mut r = vec![0.; rows.len() * rows.len()];
    let width = rows.len();
    for (index, (row, _, _)) in rows.iter().enumerate() {
        let k = q.len();
        let mut v = row.clone();
        for _ in 0..2 {
            for (i, basis) in q.iter().enumerate() {
                let projection = v.iter().zip(basis).map(|(&a, &b)| a * b).sum::<f64>();
                r[i * width + k] += projection;
                for (a, &b) in v.iter_mut().zip(basis) {
                    *a -= projection * b;
                }
            }
        }
        let norm = v.iter().fold(0_f64, |a, &b| a.hypot(b));
        if norm > 1e-10 {
            r[k * width + k] = norm;
            q.push(v.iter().map(|v| v / norm).collect());
            selected.push(index);
        }
    }
    let mut lambda: Vec<f64> = q
        .iter()
        .map(|v| {
            -v.iter()
                .zip(&gradient)
                .map(|(&a, &b)| a * b / scale)
                .sum::<f64>()
        })
        .collect();
    for i in (0..lambda.len()).rev() {
        lambda[i] = (lambda[i]
            - (i + 1..lambda.len())
                .map(|j| r[i * width + j] * lambda[j])
                .sum::<f64>())
            / r[i * width + i];
    }
    let mut residual: Vec<_> = gradient.iter().map(|v| v / scale).collect();
    let mut dual = 0_f64;
    let mut comp = 0_f64;
    for (&index, &l) in selected.iter().zip(&lambda) {
        let (row, value, inequality) = &rows[index];
        for (v, &a) in residual.iter_mut().zip(row) {
            *v += l * a;
        }
        if *inequality {
            dual = dual.max(-l);
            comp = comp.max((l * value).abs());
        }
    }
    Kkt {
        raw_violation: violation,
        normalized_violation,
        stationarity: residual.iter().fold(0_f64, |a, &b| a.hypot(b)),
        dual,
        complementarity: comp,
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-8 * (1. + b.abs()), "{a} vs {b}");
}
#[test]
fn source_values_derivatives_and_published_minima() {
    let deck = deck();
    assert_eq!(deck.cases.len(), 10);
    for case in &deck.cases {
        let f = function::<f64>(case);
        assert_eq!(case.bounds.len(), case.n);
        if let Some(x) = &case.minimizer {
            close(f.eval(x).unwrap(), case.minimum);
        }
        if let Some(published) = case.published_minimum {
            if [14, 106].contains(&case.problem) {
                assert!((published - case.minimum).abs() > 1e-3);
            } else {
                close(published, case.minimum);
            }
        }
        for reference in &case.references {
            assert!(reference.nfev > 0);
            assert!(reference.status >= -1);
            if !reference.success {
                assert!(!reference.method.is_empty());
            }
            close(f.eval(&reference.x).unwrap(), reference.f);
            let mut gradient = vec![0.; case.n];
            f.gradient_into(&reference.x, &mut gradient).unwrap();
            for (&a, &b) in gradient.iter().zip(&reference.gradient) {
                close(a, b);
            }
            for (i, tokens) in case.equalities.iter().chain(&case.inequalities).enumerate() {
                let row = Program::<f64>::new(tokens, case.n);
                close(
                    row.evaluate(&reference.x, None, None).value,
                    reference.constraints[i],
                );
                for j in 0..case.n {
                    close(
                        row.evaluate(&reference.x, Some(j), None).first,
                        reference.jacobian[i * case.n + j],
                    );
                }
            }
        }
    }
}

fn derivative_free_references(method: &str) {
    let deck = deck();
    let mut failures = Vec::new();
    for case in deck
        .cases
        .iter()
        .filter(|c| [0, 21, 35, 71].contains(&c.problem))
    {
        let reference = case.references.iter().find(|r| r.method == method).unwrap();
        let cap = 2 * reference.nfev;
        let allowance = cap;
        macro_rules! options {
            ($options:ident) => {
                $options::default()
                    .with_radii(case.initial_radius, 1e-8)
                    .with_feasibility_tolerance(FeasibilityTolerance::new(1e-10, 0.).unwrap())
                    .with_max_evaluations(Budget::try_from(allowance).unwrap())
                    .with_max_iterations(Budget::try_from(allowance).unwrap())
            };
        }
        let objective = function::<f64>(case);
        let constraints = constraints(case, 1_f64);
        let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
        let outcome = if method == "COBYLA" {
            Cobyla::new(options!(CobylaOptions))
                .and_then(|solver| portable!(&solver, &problem, &case.start))
        } else {
            Cobyqa::new(options!(CobyqaOptions))
                .and_then(|solver| portable!(&solver, &problem, &case.start))
        };
        let result = match outcome {
            Ok(result) => result,
            Err(error) => {
                eprintln!("HS{} {method}: {error:?}", case.problem);
                failures.push(case.problem);
                continue;
            }
        };
        let kkt = kkt(case, result.x());
        if method == "COBYQA" {
            eprintln!("model: {:?}", result.method_diagnostics());
        }
        eprintln!(
            "HS{} {method} {:?} nfev={} cap={cap} gap={} {kkt:?} x={:?}",
            case.problem,
            result.termination(),
            result.fun_evals(),
            result.fun() - case.minimum,
            result.x()
        );
        if !(result.is_success()
            && (result.fun() - case.minimum).abs() <= 1e-6
            && kkt.raw_violation <= 1e-8
            && kkt.stationarity <= 1e-5
            && kkt.dual <= 1e-7
            && kkt.complementarity <= 1e-7
            && result.fun_evals() <= cap)
        {
            failures.push(case.problem);
        }
    }
    assert!(
        failures.is_empty(),
        "failed {method} references: {failures:?}"
    );
}
#[test]
fn cobyla_reference_objectives_feasibility_kkt_and_evaluation_caps() {
    derivative_free_references("COBYLA");
}
#[test]
fn cobyqa_reference_objectives_feasibility_kkt_and_evaluation_caps() {
    derivative_free_references("COBYQA");
}

#[derive(Deserialize)]
struct Expected {
    outcome: String,
    no_std: Option<String>,
}
fn constrained_matrix<S: RealScalar + NumericScalar<Real = S> + Default>(small: bool) {
    use scioxide_optimize::*;
    fn pack<S: RealScalar>(r: OptimizeResult<S>) -> (Vec<S>, S, OptimizeTermination, usize) {
        (r.x().to_vec(), r.fun(), r.termination(), r.fun_evals())
    }
    let expected: std::collections::BTreeMap<String, Expected> =
        serde_json::from_str(include_str!("data/hs_matrix.json")).unwrap();
    let mut missing = Vec::new();
    let mut invalid_successes = Vec::new();
    for case in deck()
        .cases
        .iter()
        .filter(|c| c.problem != 0 && (!small || [6, 21, 35, 51].contains(&c.problem)))
    {
        let f = function::<S>(case);
        let x: Vec<S> = case
            .start
            .iter()
            .map(|&v| S::try_from_f64_lossy(v).unwrap())
            .collect();
        for method in 0..6 {
            let name = [
                "sqp",
                "interior",
                "penalty",
                "augmented",
                "cobyla",
                "cobyqa",
            ][method];
            let reference = &case.references[if method == 4 {
                2
            } else if method == 5 {
                3
            } else {
                0
            }];
            let cap = (if method >= 4 {
                4 * reference.nfev + 64
            } else {
                64 * reference.nfev + 512
            })
            .clamp(512, 30000);
            let budget = Budget::try_from(cap).unwrap();
            let c = constraints(case, S::ONE);
            let problem = ConstrainedProblem::new(&f, &c).unwrap();
            let result = match method {
                0 => portable!(
                    &Sqp::new(SqpOptions::default().with_max_iterations(budget)),
                    &problem,
                    &x
                )
                .map(pack),
                1 => portable!(
                    &InteriorPoint::new(
                        InteriorPointOptions::default().with_max_iterations(budget)
                    ),
                    &problem,
                    &x
                )
                .map(pack),
                2 => portable!(
                    &PenaltyMethod::new(
                        PenaltyOptions::default()
                            .with_max_outer_iterations(Budget::try_from(100).unwrap()),
                    ),
                    &problem,
                    &x
                )
                .map(pack),
                3 => portable!(
                    &AugmentedLagrangian::new(
                        AugmentedLagrangianOptions::default()
                            .with_max_outer_iterations(Budget::try_from(100).unwrap()),
                    ),
                    &problem,
                    &x
                )
                .map(pack),
                4 => Cobyla::new(
                    CobylaOptions::default()
                        .with_radii(
                            S::try_from_f64_lossy(case.initial_radius).unwrap(),
                            S::try_from_f64_lossy(if small { 1e-4 } else { 1e-8 }).unwrap(),
                        )
                        .with_max_iterations(budget)
                        .with_max_evaluations(budget),
                )
                .and_then(|solver| portable!(&solver, &problem, &x))
                .map(pack),
                _ => Cobyqa::new(
                    CobyqaOptions::default()
                        .with_radii(
                            S::try_from_f64_lossy(case.initial_radius).unwrap(),
                            S::try_from_f64_lossy(if small { 1e-4 } else { 1e-8 }).unwrap(),
                        )
                        .with_max_iterations(budget)
                        .with_max_evaluations(budget),
                )
                .and_then(|solver| portable!(&solver, &problem, &x))
                .map(pack),
            };
            let outcome = match result {
                Ok((x, value, status, count)) => {
                    let point: Vec<f64> =
                        x.iter().map(|&v| v.try_to_f64_exact().unwrap()).collect();
                    let k = kkt_at(case, &point, if small { 0.25 } else { 0.02 });
                    let tolerance = if small { 0.03 } else { 2e-4 };
                    let coordinate = point.iter().fold(1_f64, |a, &v| a.max(v.abs()));
                    let valid = k.normalized_violation <= tolerance * coordinate
                        && k.stationarity <= tolerance
                        && k.dual <= tolerance
                        && k.complementarity <= tolerance;
                    if status.is_success() && !valid {
                        invalid_successes
                            .push(format!("HS{} {name}: {status:?} {k:?}", case.problem));
                    }
                    let gap = (value.try_to_f64_exact().unwrap() - case.minimum).abs();
                    let class = if count > cap {
                        "work-limit"
                    } else if gap <= tolerance * (1. + case.minimum.abs()) && valid {
                        "minimum"
                    } else {
                        "nonminimum"
                    };
                    eprintln!(
                        "HS{} {name} f={value:?} count={count}/{cap} {k:?}",
                        case.problem
                    );
                    format!("{class}:{status:?}")
                }
                Err(e) => {
                    eprintln!("HS{} {name}: {e:?}", case.problem);
                    format!("error:{:?}", e.view())
                        .split(['{', '('])
                        .next()
                        .unwrap()
                        .trim()
                        .to_owned()
                }
            };
            let key = format!(
                "{}:{}:{name}",
                if small { "f32" } else { "f64" },
                case.problem
            );
            println!("MATRIX {key} {outcome}");
            if let Some(row) = expected.get(&key) {
                let wanted = if cfg!(feature = "std") {
                    &row.outcome
                } else {
                    row.no_std.as_ref().unwrap_or(&row.outcome)
                };
                if &outcome != wanted {
                    missing.push(format!("{key}: {wanted} -> {outcome}"));
                }
            } else {
                missing.push(key);
            }
        }
    }
    assert!(
        invalid_successes.is_empty(),
        "unverified successes: {invalid_successes:?}"
    );
    assert!(missing.is_empty(), "HS matrix changes: {missing:?}");
}
#[test]
fn constrained_solver_matrix_f64() {
    constrained_matrix::<f64>(false);
}
#[test]
fn constrained_solver_matrix_f32() {
    constrained_matrix::<f32>(true);
}
