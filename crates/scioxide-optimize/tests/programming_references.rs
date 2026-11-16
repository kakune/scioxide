//! Original manufactured LP/QP/MIP/QAP problems with analytic or exhaustive optima.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{Bounds, Budget, NumericScalar, RealScalar};
use scioxide_linalg::{Matrix, MatrixView};
use scioxide_optimize::{
    BranchAndBound, InteriorPointLp, LinearConstraintSense, LinearConstraints, LpProblem,
    LpVerificationProblem, MipProblem, OptimizeTermination, ProgrammingVerifier, QapDirection,
    QapMethod, QapOptions, QapProblemView, QpProblem, QpSolver, QpVerificationProblem,
    SimplexSolver, VerificationStatus,
};
use scioxide_sparse::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
use serde::Deserialize;

#[derive(Deserialize)]
struct Deck {
    lp: Vec<Case>,
    qp: Vec<Case>,
    mip: Vec<Case>,
    qap: Vec<QapCase>,
}
#[derive(Clone, Deserialize)]
struct Case {
    name: String,
    n: usize,
    m: usize,
    ptr: Vec<usize>,
    index: Vec<usize>,
    value: Vec<f64>,
    objective: Vec<f64>,
    offset: f64,
    lower: Vec<Option<f64>>,
    upper: Vec<Option<f64>>,
    row_lower: Vec<Option<f64>>,
    row_upper: Vec<Option<f64>>,
    integer: Vec<usize>,
    quadratic_diagonal: f64,
    minimum: f64,
    reference: Reference,
}
#[derive(Clone, Deserialize)]
struct Reference {
    status: String,
    objective: f64,
    primal: Option<Vec<f64>>,
}
#[derive(Deserialize)]
struct QapCase {
    name: String,
    n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    minimum: f64,
    minimizer: Vec<usize>,
    references: Vec<QapReference>,
}
#[derive(Deserialize)]
struct QapReference {
    objective: f64,
    permutation: Vec<usize>,
}
fn deck() -> Deck {
    serde_json::from_str(include_str!("data/programming.json")).unwrap()
}
fn cv<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance * (1. + b.abs()), "{a} vs {b}");
}
fn bounds<S: RealScalar>(case: &Case) -> Bounds<S> {
    Bounds::new(
        case.lower
            .iter()
            .zip(&case.upper)
            .map(|(&a, &b)| (a.map(cv), b.map(cv)))
            .collect(),
    )
    .unwrap()
}
fn rows<S: RealScalar + NumericScalar<Real = S>>(case: &Case) -> LinearConstraints<S> {
    let count = (0..case.m)
        .map(|i| {
            if case.row_lower[i] == case.row_upper[i] && case.row_lower[i].is_some() {
                1
            } else {
                usize::from(case.row_lower[i].is_some()) + usize::from(case.row_upper[i].is_some())
            }
        })
        .sum();
    let mut builder = SparseBuilder::new(count, case.n);
    let mut rhs = Vec::new();
    let mut senses = Vec::new();
    for i in 0..case.m {
        let mut emit = |value: f64, sense| {
            let row = rhs.len();
            for k in case.ptr[i]..case.ptr[i + 1] {
                builder
                    .push(row, case.index[k], cv::<S>(case.value[k]))
                    .unwrap();
            }
            rhs.push(cv::<S>(value));
            senses.push(sense);
        };
        if let (Some(lo), Some(hi)) = (case.row_lower[i], case.row_upper[i])
            && lo == hi
        {
            emit(lo, LinearConstraintSense::Equal);
        } else {
            if let Some(lo) = case.row_lower[i] {
                emit(lo, LinearConstraintSense::GreaterEqual);
            }
            if let Some(hi) = case.row_upper[i] {
                emit(hi, LinearConstraintSense::LessEqual);
            }
        }
    }
    LinearConstraints::new(
        builder
            .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
            .unwrap(),
        rhs,
        senses,
    )
    .unwrap()
}
fn lp<S: RealScalar + NumericScalar<Real = S>>(case: &Case) -> LpProblem<S> {
    LpProblem::new(case.objective.iter().map(|&v| cv(v)).collect())
        .unwrap()
        .with_constraints(rows(case))
        .unwrap()
        .with_bounds(bounds(case))
        .unwrap()
}
fn qp<S: RealScalar + NumericScalar<Real = S>>(case: &Case) -> QpProblem<S> {
    let mut h = Matrix::zeros(case.n, case.n).unwrap();
    for i in 0..case.n {
        h.as_mut_slice()[i * case.n + i] = cv(case.quadratic_diagonal);
    }
    QpProblem::new(h, case.objective.iter().map(|&v| cv(v)).collect())
        .unwrap()
        .with_constraints(rows(case))
        .unwrap()
        .with_bounds(bounds(case))
        .unwrap()
}
fn original(case: &Case, x: &[f64], tolerance: f64) -> f64 {
    assert_eq!(x.len(), case.n);
    assert!(x.iter().all(|v| v.is_finite()));
    for (j, &v) in x.iter().enumerate() {
        if let Some(lo) = case.lower[j] {
            assert!(lo - v <= tolerance * (1. + lo.abs()));
        }
        if let Some(hi) = case.upper[j] {
            assert!(v - hi <= tolerance * (1. + hi.abs()));
        }
    }
    for i in 0..case.m {
        let value = (case.ptr[i]..case.ptr[i + 1])
            .map(|k| case.value[k] * x[case.index[k]])
            .sum::<f64>();
        let scale = (case.ptr[i]..case.ptr[i + 1])
            .map(|k| (case.value[k] * x[case.index[k]]).abs())
            .sum::<f64>()
            .max(1.);
        if let Some(lo) = case.row_lower[i] {
            assert!(
                lo - value <= tolerance * scale.max(lo.abs()),
                "{} row {i}: {value} < {lo}",
                case.name
            );
        }
        if let Some(hi) = case.row_upper[i] {
            assert!(
                value - hi <= tolerance * scale.max(hi.abs()),
                "{} row {i}: {value} > {hi}",
                case.name
            );
        }
    }
    case.offset
        + case
            .objective
            .iter()
            .zip(x)
            .map(|(&c, &v)| c * v)
            .sum::<f64>()
        + 0.5 * case.quadratic_diagonal * x.iter().map(|v| v * v).sum::<f64>()
}

#[test]
fn independent_reference_data_are_feasible_and_match_exact_objectives() {
    let deck = deck();
    assert_eq!(deck.lp.len(), 6);
    assert_eq!(deck.qp.len(), 2);
    assert_eq!(deck.mip.len(), 2);
    assert_eq!(deck.qap.len(), 3);
    for case in deck.lp.iter().chain(&deck.qp).chain(&deck.mip) {
        assert_eq!(case.ptr.len(), case.m + 1);
        assert_eq!(case.ptr[case.m], case.value.len());
        assert_eq!(case.index.len(), case.value.len());
        assert!(case.reference.status.ends_with("kOptimal"));
        let x = case.reference.primal.as_ref().unwrap();
        close(original(case, x, 1e-7), case.minimum, 1e-7);
        close(case.reference.objective, case.minimum, 1e-7);
        for &j in &case.integer {
            assert!((x[j] - x[j].round()).abs() <= 1e-7);
        }
    }
    for case in &deck.qap {
        close(qap_objective(case, &case.minimizer), case.minimum, 1e-12);
        for reference in &case.references {
            close(
                qap_objective(case, &reference.permutation),
                reference.objective,
                1e-12,
            );
            assert!(reference.objective >= case.minimum);
        }
    }
}

fn lp_cases<S: RealScalar + NumericScalar<Real = S>>(cases: &[Case], tolerance: f64) {
    for case in cases {
        let problem = lp::<S>(case);
        // Cancellation against zero RHS rows needs precision-appropriate
        // absolute tolerances for f32; original-coordinate checks remain independent.
        let simplex = SimplexSolver::default();
        let simplex = if S::MANTISSA_DIGITS == 24 {
            simplex.with_tolerances(scioxide_optimize::OptimalityTolerances::default())
        } else {
            simplex
        };
        let interior = InteriorPointLp::default();
        let interior = if S::MANTISSA_DIGITS == 24 {
            interior.with_tolerances(scioxide_optimize::OptimalityTolerances::default())
        } else {
            interior
        };
        for (method, result) in [
            ("simplex", simplex.solve(&problem)),
            ("interior", interior.solve(&problem)),
        ] {
            let result = result.unwrap_or_else(|error| panic!("{} {method}: {error:?}", case.name));
            assert!(result.is_success(), "{} {method}: {result:?}", case.name);
            let verification = ProgrammingVerifier::default()
                .verify(&LpVerificationProblem::from_problem(&problem), &result)
                .unwrap();
            assert_eq!(
                verification.status(),
                VerificationStatus::Accepted,
                "{} {method}: {verification:?}",
                case.name
            );
            let x: Vec<f64> = result
                .x()
                .iter()
                .map(|&v| v.try_to_f64_exact().unwrap())
                .collect();
            close(original(case, &x, tolerance), case.minimum, tolerance);
            close(
                result.fun().try_to_f64_exact().unwrap() + case.offset,
                case.minimum,
                tolerance,
            );
        }
    }
}
#[test]
fn manufactured_both_lp_solvers_and_original_certificates_f64() {
    lp_cases::<f64>(&deck().lp, 1e-6);
}
#[test]
fn small_manufactured_both_lp_solvers_f32() {
    lp_cases::<f32>(&deck().lp[..3], 2e-4);
}

#[test]
fn convex_qps_and_equivalent_linear_programs_have_verified_certificates() {
    let deck = deck();
    for case in deck.lp.iter().chain(&deck.qp) {
        let problem = qp::<f64>(case);
        let result = QpSolver::default().solve(&problem);
        match result {
            Ok(result) => {
                eprintln!(
                    "QP {}: {:?} f={}",
                    case.name,
                    result.termination(),
                    result.fun()
                );
                assert_eq!(
                    result.termination(),
                    OptimizeTermination::Converged,
                    "{}: iterations={}, report={:?}, diagnostics={:?}",
                    case.name,
                    result.iterations(),
                    result.verification_report(),
                    result.diagnostics()
                );
                if result.is_success() {
                    let report = ProgrammingVerifier::default()
                        .verify(&QpVerificationProblem::from_problem(&problem), &result)
                        .unwrap();
                    assert_eq!(
                        report.status(),
                        VerificationStatus::Accepted,
                        "{}: {report:?}",
                        case.name
                    );
                    close(original(case, result.x(), 1e-6), case.minimum, 1e-6);
                }
            }
            Err(error) => panic!("unexpected #1982 error {}: {error:?}", case.name),
        }
    }
}

#[test]
fn manufactured_degeneracy_scaling_infeasibility_and_unboundedness() {
    let base = deck().lp.remove(0);
    let mut duplicate = base.clone();
    duplicate.name.push_str("-duplicate");
    let end = duplicate.ptr[1];
    duplicate.index.extend_from_slice(&base.index[..end]);
    duplicate.value.extend_from_slice(&base.value[..end]);
    duplicate.ptr.push(duplicate.value.len());
    duplicate.row_lower.push(base.row_lower[0]);
    duplicate.row_upper.push(base.row_upper[0]);
    duplicate.m += 1;
    let mut scaled = base.clone();
    scaled.name.push_str("-scaled");
    for i in 0..scaled.m {
        let scale = if i % 2 == 0 { 1e-6 } else { 1e6 };
        for a in &mut scaled.value[scaled.ptr[i]..scaled.ptr[i + 1]] {
            *a *= scale;
        }
        scaled.row_lower[i] = scaled.row_lower[i].map(|v| scale * v);
        scaled.row_upper[i] = scaled.row_upper[i].map(|v| scale * v);
    }
    lp_cases::<f64>(&[duplicate, scaled], 1e-6);
    let mut infeasible = base.clone();
    infeasible.index.push(0);
    infeasible.value.push(1.);
    infeasible.ptr.push(infeasible.value.len());
    infeasible.row_lower.push(None);
    infeasible.row_upper.push(Some(base.lower[0].unwrap() - 1.));
    infeasible.m += 1;
    let mut unbounded = base.clone();
    unbounded.n += 1;
    unbounded.objective.push(-1.);
    unbounded.lower.push(Some(0.));
    unbounded.upper.push(None);
    for (case, expected) in [
        (infeasible, OptimizeTermination::Infeasible),
        (unbounded, OptimizeTermination::Unbounded),
    ] {
        let problem = lp::<f64>(&case);
        for result in [
            SimplexSolver::default().solve(&problem),
            InteriorPointLp::default().solve(&problem),
        ] {
            let result = result.unwrap();
            assert_eq!(result.termination(), expected);
            let report = ProgrammingVerifier::default()
                .verify(&LpVerificationProblem::from_problem(&problem), &result)
                .unwrap();
            assert_eq!(report.status(), VerificationStatus::Accepted, "{report:?}");
        }
    }
}

#[test]
fn manufactured_integer_instances_reach_exhaustive_optima() {
    for case in &deck().mip {
        let problem = MipProblem::new(lp::<f64>(case), case.integer.clone()).unwrap();
        let result = BranchAndBound::default()
            .with_max_nodes(Budget::try_from(10000).unwrap())
            .with_max_lp_iterations(Budget::try_from(10000).unwrap())
            .solve(&problem)
            .unwrap();
        eprintln!(
            "MIP {}: {:?} f={} nodes={}",
            case.name,
            result.termination(),
            result.fun(),
            result.nodes()
        );
        assert!(result.is_success(), "{}: {result:?}", case.name);
        close(original(case, result.x(), 1e-7), case.minimum, 1e-8);
        for &j in &case.integer {
            assert!((result.x()[j] - result.x()[j].round()).abs() <= 1e-8);
        }
    }
}

fn qap_objective(case: &QapCase, p: &[usize]) -> f64 {
    assert_eq!(p.len(), case.n);
    let mut sorted = p.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..case.n).collect::<Vec<_>>());
    (0..case.n)
        .flat_map(|i| {
            (0..case.n).map(move |j| case.a[i * case.n + j] * case.b[p[i] * case.n + p[j]])
        })
        .sum()
}
fn qap_cases<S: RealScalar + NumericScalar<Real = S>>() {
    for case in &deck().qap {
        let a: Vec<S> = case.a.iter().map(|&v| cv(v)).collect();
        let b: Vec<S> = case.b.iter().map(|&v| cv(v)).collect();
        let plan = QapProblemView::new(
            MatrixView::from_slice(case.n, case.n, &a).unwrap(),
            MatrixView::from_slice(case.n, case.n, &b).unwrap(),
            &[],
        )
        .unwrap()
        .prepare_with_limit(4000000)
        .unwrap();
        let identity: Vec<_> = (0..case.n).collect();
        let initial = qap_objective(case, &identity);
        let mut validation = vec![0; case.n];
        let initial_permutation =
            scioxide_core::PermutationView::new(&identity, &mut validation).unwrap();
        for method in [QapMethod::Faq, QapMethod::TwoOpt, QapMethod::FaqThenTwoOpt] {
            let options = QapOptions::default()
                .with_method(method)
                .with_max_workspace_bytes(4000000)
                .with_seed(1729)
                .with_restarts(8);
            let result = plan
                .solve_with_options(QapDirection::Minimize, Some(initial_permutation), &options)
                .unwrap();
            let score = qap_objective(case, result.permutation().unwrap());
            close(
                result.objective().unwrap().try_to_f64_exact().unwrap(),
                score,
                1e-6,
            );
            assert!(
                score >= case.minimum - 1e-6 && score <= initial + 1e-6,
                "{} {method:?}: {score}",
                case.name
            );
            let gap = (score - case.minimum) / case.minimum.abs().max(1.);
            assert!(gap.is_finite() && gap >= -1e-6);
            eprintln!("{} {method:?}: non-certifying gap={gap}", case.name);
        }
    }
}
#[test]
fn manufactured_assignment_heuristic_gaps_f64() {
    qap_cases::<f64>();
}
#[test]
fn manufactured_assignment_heuristic_gaps_f32() {
    qap_cases::<f32>();
}
