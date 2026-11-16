//! Original MGH/MINPACK values, derivatives, starts and independent solver references.
#![allow(clippy::unwrap_used, clippy::panic, clippy::suspicious_arithmetic_impl)]
#[path = "support/reference_program.rs"]
mod reference_program;
#[path = "support/reference_squares.rs"]
mod reference_squares;
use reference_program::Token;
use reference_squares::SumOfSquares;
use scioxide_core::{NumericScalar, RealScalar};
use scioxide_linalg::Matrix;
use scioxide_optimize::{LeastSquaresProblem, ObjectiveFunction};
use serde::Deserialize;

#[derive(Deserialize)]
struct Deck {
    models: Vec<Model>,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Model {
    family: String,
    problem: usize,
    n: usize,
    m: Option<usize>,
    residuals: Vec<Vec<Token>>,
    published_minima: Option<Vec<f64>>,
}
#[derive(Deserialize)]
struct Case {
    model: usize,
    factor: usize,
    initial: Point,
    references: Vec<Reference>,
    bounds: Vec<(f64, f64)>,
    bounded: Reference,
}
#[derive(Deserialize)]
struct Point {
    x: Vec<Option<f64>>,
    f: Option<f64>,
    gradient: Vec<Option<f64>>,
    residual: Option<Vec<Option<f64>>>,
    jacobian: Option<Vec<Option<f64>>>,
    hessian: Option<Vec<Option<f64>>>,
}
#[derive(Deserialize)]
struct Reference {
    method: String,
    success: bool,
    status: i32,
    nfev: usize,
    njev: usize,
    outcome: Point,
}
fn deck() -> Deck {
    serde_json::from_str(include_str!("data/mgh.json")).unwrap()
}
fn cast<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn close<S: RealScalar>(actual: &[S], expected: &[Option<f64>], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, b)) in actual.iter().zip(expected).enumerate() {
        match b {
            Some(b) => {
                let b = cast::<S>(*b);
                assert!(
                    a.is_finite() && (a - b).abs() <= cast::<S>(tol) * (S::ONE + b.abs()),
                    "entry {index}: {a:?} vs {b:?}"
                );
            }
            None => assert!(
                !a.is_finite(),
                "expected nonfinite reference entry {index}: {a:?}"
            ),
        }
    }
}

fn check_point<S: RealScalar + NumericScalar<Real = S>>(
    model: &SumOfSquares<S>,
    point: &Point,
    tol: f64,
    hessian: bool,
) {
    let x: Vec<S> = point.x.iter().map(|v| cast(v.unwrap())).collect();
    close(&[model.eval(&x).unwrap()], &[point.f], tol);
    if point.f.is_none() {
        return;
    }
    let mut gradient = vec![<S as scioxide_core::AlgebraicScalar>::ZERO; x.len()];
    model.gradient_into(&x, &mut gradient).unwrap();
    if let Some(expected) = &point.residual {
        let mut residual = vec![<S as scioxide_core::AlgebraicScalar>::ZERO; model.residual_dim()];
        model.residual_into(&x, &mut residual).unwrap();
        close(&residual, expected, tol);
        let mut jacobian = Matrix::zeros(model.residual_dim(), x.len()).unwrap();
        model.jacobian_into(&x, &mut jacobian).unwrap();
        close(jacobian.as_slice(), point.jacobian.as_ref().unwrap(), tol);
        // Algebraically equivalent residual reductions may round differently
        // at a large null-space component. Bound the resulting gradient change
        // by 2*(|J|*|delta r| + |r_ref|*|delta J|), without relaxing value/J checks.
        for (j, &g) in gradient.iter().enumerate() {
            if let Some(reference) = point.gradient[j] {
                let mut propagation = <S as scioxide_core::AlgebraicScalar>::ZERO;
                for i in 0..residual.len() {
                    if let (Some(r), Some(a)) = (
                        expected[i],
                        point.jacobian.as_ref().unwrap()[i * x.len() + j],
                    ) {
                        let actual = jacobian.as_slice()[i * x.len() + j];
                        propagation += cast::<S>(2.)
                            * (NumericScalar::abs(actual)
                                * NumericScalar::abs(residual[i] - cast(r))
                                + NumericScalar::abs(cast::<S>(r))
                                    * NumericScalar::abs(actual - cast(a)));
                    }
                }
                let reference = cast::<S>(reference);
                assert!(
                    NumericScalar::abs(g - reference)
                        <= propagation
                            + cast::<S>(tol)
                                * (<S as scioxide_core::AlgebraicScalar>::ONE
                                    + NumericScalar::abs(reference)),
                    "gradient {j}: {g:?} vs {reference:?}, propagation {propagation:?}"
                );
            } else {
                assert!(!g.is_finite());
            }
        }
    } else {
        close(&gradient, &point.gradient, tol);
    }
    if hessian && let Some(expected) = &point.hessian {
        let mut matrix = Matrix::zeros(x.len(), x.len()).unwrap();
        model.hessian_into(&x, &mut matrix).unwrap();
        close(matrix.as_slice(), expected, tol * 10.);
    }
}

#[test]
fn original_fortran_checkpoints_and_reference_coverage_f64() {
    let deck = deck();
    assert_eq!(deck.models.len(), 52);
    assert_eq!(deck.cases.len(), 156);
    let models: Vec<_> = deck
        .models
        .iter()
        .map(|m| SumOfSquares::<f64>::new(m.n, &m.residuals))
        .collect();
    for case in &deck.cases {
        let data = &deck.models[case.model];
        let model = &models[case.model];
        assert!([1, 10, 100].contains(&case.factor));
        assert_eq!(case.bounds.len(), data.n);
        assert_eq!(data.m.is_some(), data.family == "least-squares");
        assert!((1..=18).contains(&data.problem));
        assert!(
            std::panic::catch_unwind(|| check_point(model, &case.initial, 1e-9, true)).is_ok(),
            "{} {} n={} factor={} initial",
            data.family,
            data.problem,
            data.n,
            case.factor
        );
        for reference in case.references.iter().chain([&case.bounded]) {
            assert!(reference.nfev > 0);
            assert!(reference.njev > 0 || !reference.success);
            assert!(!reference.method.is_empty());
            assert!(reference.status >= 0);
            if reference.success {
                assert!(
                    std::panic::catch_unwind(|| check_point(
                        model,
                        &reference.outcome,
                        2e-8,
                        false
                    ))
                    .is_ok(),
                    "{} {} n={} factor={} {} status={}",
                    data.family,
                    data.problem,
                    data.n,
                    case.factor,
                    reference.method,
                    reference.status
                );
            } else {
                assert!(!reference.outcome.x.is_empty());
                assert!(matches!(reference.status, 0..=4));
            }
        }
    }
}

#[test]
fn well_conditioned_fortran_checkpoints_f32() {
    let deck = deck();
    let mut checked = 0;
    for case in deck.cases.iter().filter(|c| c.factor == 1) {
        let data = &deck.models[case.model];
        let meaningful = if data.family == "unconstrained" {
            [1, 5, 14, 15, 16, 17].contains(&data.problem)
        } else {
            [1, 4, 5, 6, 8, 9, 12].contains(&data.problem)
        };
        if meaningful {
            let model = SumOfSquares::<f32>::new(data.n, &data.residuals);
            check_point(&model, &case.initial, 2e-4, true);
            checked += 1;
        }
    }
    assert_eq!(checked, 14);
}

// The matrix is deliberately explicit: a change in either direction requires
// reviewing the returned objective and termination, rather than silently skipping
// a difficult member. Cost limits are relative to the independently recorded work.
#[derive(Deserialize)]
struct Expected {
    outcome: String,
    no_std: Option<String>,
    max_iterations: Option<usize>,
    no_std_max_iterations: Option<usize>,
    baseline_iterations: Option<usize>,
    no_std_baseline_iterations: Option<usize>,
    // #1977 pins own solver work from main b3d3c649e, separately per feature mode.
    baseline_evaluations: Option<usize>,
    no_std_baseline_evaluations: Option<usize>,
}
fn error_class(e: &scioxide_core::SciOxideError) -> String {
    format!("error:{:?}", e.view())
        .split(['{', '('])
        .next()
        .unwrap()
        .trim()
        .to_owned()
}
#[derive(Clone, Copy)]
enum Methods {
    Local,
    QuasiNewton,
    LeastSquares,
}

fn local_matrix<S: RealScalar + NumericScalar<Real = S>>(
    small: bool,
    bounded_only: bool,
    methods: Methods,
) {
    use scioxide_core::{Bounds, Budget};
    use scioxide_optimize::*;
    type LocalOutcome<S> = (Vec<S>, S, OptimizeTermination, usize, usize);
    fn pack<S: RealScalar>(r: OptimizeResult<S>) -> LocalOutcome<S> {
        (
            r.x().to_vec(),
            r.fun(),
            r.termination(),
            r.fun_evals(),
            r.iterations(),
        )
    }
    // Reference outcomes pin portable kernels; owned forms use native dispatch.
    macro_rules! portable {
        ($solver:expr, $request:expr, $problem:expr, $x:expr) => {{
            let solver = $solver;
            solver.workspace($request).and_then(|mut workspace| {
                let mut output = OptimizeResult::empty();
                solver
                    .minimize_into(
                        $problem,
                        $x,
                        &mut output,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .map(|()| output)
            })
        }};
    }
    let data = deck();
    let expected: std::collections::BTreeMap<String, Expected> =
        serde_json::from_str(include_str!("data/mgh_matrix.json")).unwrap();
    let mut failures = Vec::new();
    let mut rows = 0;
    for (index, case) in data.cases.iter().enumerate() {
        let spec = &data.models[case.model];
        if small
            && (case.factor != 1
                || spec.family != "least-squares"
                || ![1, 4, 5, 6, 8, 9, 12].contains(&spec.problem))
        {
            continue;
        }
        let model = SumOfSquares::<S>::new(spec.n, &spec.residuals);
        let x: Vec<S> = case.initial.x.iter().map(|v| cast(v.unwrap())).collect();
        for method in 0..11 {
            if (match methods {
                Methods::Local => method >= 9,
                Methods::QuasiNewton => method > 1,
                Methods::LeastSquares => method < 9,
            }) || (bounded_only != (method == 6))
                || (method >= 9 && spec.m.is_none())
            {
                continue;
            }
            let name = [
                "bfgs",
                "lbfgs",
                "ncg",
                "newton",
                "dogleg",
                "newton-cg",
                "lbfgsb",
                "nelder-mead",
                "powell",
                "gauss-newton",
                "lm",
            ][method];
            let reference = if method == 6 {
                &case.bounded
            } else {
                &case.references[if method == 9 || method == 10 {
                    2
                } else if method == 1 {
                    1
                } else {
                    0
                }]
            };
            // The direct projected method is allowed different iteration counts;
            // keep its reference-relative work bound explicit and finite.
            let cap = if method == 6 {
                (32 * reference.nfev + 512).clamp(512, 4096)
            } else {
                (8 * reference.nfev + 128).clamp(128, 1024)
            };
            let budget = Budget::try_from(cap).unwrap();
            let result: Result<LocalOutcome<S>, scioxide_core::SciOxideError> = if method >= 9 {
                let solver = LeastSquares::new(
                    LeastSquaresOptions::default()
                        .with_max_iterations(budget)
                        .with_globalization(if method == 9 {
                            LeastSquaresGlobalization::gauss_newton()
                        } else {
                            LeastSquaresGlobalization::levenberg_marquardt()
                        }),
                )
                .unwrap();
                let r = LeastSquaresWorkspaceRequest::of(&model)
                    .and_then(|request| solver.workspace(request))
                    .and_then(|mut workspace| {
                        let mut output = LeastSquaresResult::empty();
                        solver
                            .solve_into(
                                &model,
                                &x,
                                &mut output,
                                &mut workspace,
                                scioxide_core::ExecutionContext::sequential(),
                            )
                            .map(|()| output)
                    });
                r.and_then(|r| {
                    Ok((
                        r.x().to_vec(),
                        r.cost()? * cast(2.),
                        r.termination(),
                        r.fun_evals(),
                        r.iterations(),
                    ))
                })
            } else {
                match method {
                    0 => portable!(
                        Bfgs::new(BfgsOptions::default().with_max_iterations(budget)),
                        x.len(),
                        &model,
                        &x
                    )
                    .map(pack),
                    1 => portable!(
                        Lbfgs::new(LbfgsOptions::default().with_max_iterations(budget)),
                        x.len(),
                        &model,
                        &x
                    )
                    .map(pack),
                    2 => {
                        portable!(
                            NonlinearCg::new(
                                NonlinearCgOptions::default().with_max_iterations(budget),
                            ),
                            x.len(),
                            &model,
                            &x
                        )
                        .map(pack)
                    }
                    3 => portable!(
                        NewtonOptimizer::new(NewtonOptions::default().with_max_iterations(budget),),
                        x.len(),
                        &model,
                        &x
                    )
                    .map(pack),
                    4 => portable!(
                        TrustRegionOptimizer::new(
                            TrustRegionOptions::default().with_max_iterations(budget),
                        ),
                        TrustRegionWorkspaceRequest::of(&model),
                        &model,
                        &x
                    )
                    .map(pack),
                    5 => portable!(
                        TrustRegionOptimizer::new(
                            TrustRegionOptions::default()
                                .with_max_iterations(budget)
                                .with_subproblem(TrustRegionSubproblem::SteihaugCg {
                                    max_inner_iterations: Budget::try_from(100).unwrap(),
                                    second_order_stopping: false,
                                }),
                        ),
                        TrustRegionWorkspaceRequest::of(&model),
                        &model,
                        &x
                    )
                    .map(pack),
                    6 => {
                        let bounds = Bounds::new(
                            case.bounds
                                .iter()
                                .map(|&(a, b)| (Some(cast(a)), Some(cast(b))))
                                .collect(),
                        )
                        .unwrap();
                        portable!(
                            LbfgsB::new(LbfgsBOptions::default().with_max_iterations(budget)),
                            x.len(),
                            &BoundedObjective::new(&model, &bounds).unwrap(),
                            &x
                        )
                        .map(|r| {
                            let key =
                                format!("{}:{index}:lbfgsb", if small { "f32" } else { "f64" });
                            let row = &expected[&key];
                            let maximum = if cfg!(feature = "std") {
                                row.max_iterations
                            } else {
                                row.no_std_max_iterations.or(row.max_iterations)
                            };
                            if let Some(maximum) = maximum
                                && r.iterations() > maximum
                            {
                                failures.push(format!(
                                    "{key}: {} iterations exceed {maximum}",
                                    r.iterations()
                                ));
                            }
                            println!(
                                "LBFGSB {}:{index} {}",
                                if small { "f32" } else { "f64" },
                                r.iterations()
                            );
                            pack(r)
                        })
                    }
                    7 => portable!(
                        NelderMead::new(NelderMeadOptions::default().with_max_iterations(budget),)
                            .unwrap(),
                        x.len(),
                        &model,
                        &x
                    )
                    .map(pack),
                    _ => portable!(
                        Powell::new(PowellOptions::default().with_max_iterations(budget)).unwrap(),
                        x.len(),
                        &model,
                        &x
                    )
                    .map(pack),
                }
            };
            let outcome = match result {
                Ok((point, value, status, count, iterations)) => {
                    println!(
                        "ITER {}:{index}:{name} {iterations}",
                        if small { "f32" } else { "f64" }
                    );
                    println!(
                        "EVAL {}:{index}:{name} {count}",
                        if small { "f32" } else { "f64" }
                    );
                    let key = format!("{}:{index}:{name}", if small { "f32" } else { "f64" });
                    if let Some(row) = expected.get(&key) {
                        let baseline = if cfg!(feature = "std") {
                            row.baseline_iterations
                        } else {
                            row.no_std_baseline_iterations
                        };
                        let scoped = matches!(name, "ncg" | "dogleg" | "newton-cg" | "nelder-mead");
                        if let Some(baseline) = baseline {
                            let cap = if scoped { baseline } else { baseline * 11 / 10 };
                            if iterations > cap {
                                failures.push(format!(
                                    "{key}: {iterations} iterations exceed baseline bound {cap}"
                                ));
                            }
                        }
                        let evaluations = if cfg!(feature = "std") {
                            row.baseline_evaluations
                        } else {
                            row.no_std_baseline_evaluations
                        };
                        if let Some(baseline) = evaluations
                            && count > baseline
                        {
                            failures.push(format!(
                                "{key}: {count} evaluations exceed baseline {baseline}"
                            ));
                        }
                    }
                    // Independent function evaluation; local stationary solutions
                    // are distinguished from the published global objective.
                    let actual = model.eval(&point).unwrap();
                    let a = actual.try_to_f64_exact().unwrap();
                    let v = value.try_to_f64_exact().unwrap();
                    assert!((a - v).abs() <= (if small { 1e-3 } else { 1e-9 }) * (1. + a.abs()));
                    let target = if method == 6 {
                        reference.outcome.f.unwrap_or(f64::NAN)
                    } else {
                        spec.published_minima
                            .as_ref()
                            .and_then(|values| {
                                values
                                    .iter()
                                    .min_by(|a, b| (*a - v).abs().total_cmp(&(*b - v).abs()))
                            })
                            .copied()
                            .unwrap_or_else(|| {
                                case.references
                                    .iter()
                                    .filter_map(|r| r.outcome.f)
                                    .fold(f64::INFINITY, f64::min)
                            })
                    };
                    let evaluation_cap = if method == 6 { 8 * cap } else { cap };
                    let qualified = a.is_finite()
                        && target.is_finite()
                        && a - target <= (if small { 5e-3 } else { 2e-5 }) * (1. + target.abs());
                    let label = if qualified && count <= evaluation_cap {
                        "minimum"
                    } else if count > evaluation_cap {
                        "work-limit"
                    } else {
                        "nonminimum"
                    };
                    format!("{label}:{status:?}")
                }
                Err(e) => error_class(&e),
            };
            let key = format!("{}:{index}:{name}", if small { "f32" } else { "f64" });
            println!("MATRIX {key} {outcome}");
            match expected.get(&key) {
                Some(row) => {
                    let wanted = if cfg!(feature = "std") {
                        &row.outcome
                    } else {
                        row.no_std.as_ref().unwrap_or(&row.outcome)
                    };
                    if &outcome != wanted {
                        failures.push(format!("{key}: {wanted} -> {outcome}"));
                    }
                }
                None => failures.push(key),
            }
            rows += 1;
        }
    }
    assert!(failures.is_empty(), "MGH matrix changes: {failures:?}");
    assert!(rows > 0);
}
#[test]
fn all_applicable_mgh_solvers_f64() {
    local_matrix::<f64>(false, false, Methods::Local);
}
#[test]
fn meaningful_mgh_solvers_f32() {
    local_matrix::<f32>(true, false, Methods::Local);
}

#[test]
fn bounded_mgh_solvers_f64() {
    local_matrix::<f64>(false, true, Methods::Local);
}
#[test]
fn bounded_mgh_solvers_f32() {
    local_matrix::<f32>(true, true, Methods::Local);
}

#[test]
fn quasi_newton_mgh_solvers_f64() {
    local_matrix::<f64>(false, false, Methods::QuasiNewton);
}
#[test]
fn quasi_newton_mgh_solvers_f32() {
    local_matrix::<f32>(true, false, Methods::QuasiNewton);
}

#[test]
fn least_squares_mgh_solvers_f64() {
    local_matrix::<f64>(false, false, Methods::LeastSquares);
}
#[test]
fn least_squares_mgh_solvers_f32() {
    local_matrix::<f32>(true, false, Methods::LeastSquares);
}

// Brown almost-linear has a numerically flat region where the linear
// residuals vanish but prod(x) is near zero. At large n, rounding can send LM
// here instead of to a zero-residual root. Check the equations independently
// rather than treating any objective near one as an acceptable result.
fn brown_plateau(x: &[f64]) -> bool {
    if x.len() < 30 || x.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let sum: f64 = x.iter().sum();
    let linear: Vec<_> = x[..x.len() - 1]
        .iter()
        .map(|v| v + sum - (x.len() + 1) as f64)
        .collect();
    let product: f64 = x.iter().product();
    let linear_sum: f64 = linear.iter().sum();
    linear.iter().all(|v| v.abs() <= 1e-7)
        && product.abs() <= 1e-7
        && x.iter().enumerate().all(|(j, _)| {
            // Do not divide by x[j]: plateau coordinates may be zero.
            let derivative: f64 = x
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != j)
                .map(|(_, v)| v)
                .product();
            let half_gradient =
                linear_sum + linear.get(j).copied().unwrap_or(0.0) + (product - 1.0) * derivative;
            half_gradient.abs() <= 1e-6
        })
}

#[test]
fn brown_plateau_requires_small_residuals_and_gradient() {
    for n in [30, 40] {
        let mut x = vec![0.2; n];
        x[n - 1] = (n + 1) as f64 - n as f64 * 0.2;
        assert!(brown_plateau(&x));
        x[0] += 0.01;
        assert!(!brown_plateau(&x));
        // A zero product alone does not imply stationarity.
        x.fill((n + 1) as f64 / n as f64);
        x[n - 1] = 0.0;
        assert!(!brown_plateau(&x));
        x[0] = f64::NAN;
        assert!(!brown_plateau(&x));
    }
}

#[test]
fn minpack_lm_work_and_objectives() {
    use scioxide_core::{Budget, ExecutionContext, RelativeTolerance, StepTolerance, Tolerance};
    use scioxide_optimize::*;
    let data = deck();
    let mut failures = Vec::new();
    for (index, case) in data.cases.iter().enumerate() {
        let spec = &data.models[case.model];
        if spec.m.is_none() {
            continue;
        }
        let reference = &case.references[2];
        let model = SumOfSquares::<f64>::new(spec.n, &spec.residuals);
        let x: Vec<_> = case.initial.x.iter().map(|v| v.unwrap()).collect();
        let fcap = (3 * reference.nfev).div_ceil(2);
        let jcap = (3 * reference.njev).div_ceil(2);
        let unchanged_reference =
            reference.success && reference.status == 3 && reference.outcome.x == case.initial.x;
        let brown = spec.problem == 16 && spec.n >= 30;
        let step_stopped_reference = brown
            && reference.success
            && reference.status == 3
            && reference.outcome.f.is_some_and(|f| f > 1.0);
        let iteration_cap = if unchanged_reference {
            fcap.saturating_sub(1).max(1)
        } else if reference.success {
            fcap.max(jcap).max(1)
        } else {
            1024
        };
        let solver = LeastSquares::new(
            LeastSquaresOptions::default()
                .with_rank_policy(scioxide_linalg::RankPolicy::Exact)
                .with_max_iterations(Budget::try_from(iteration_cap).unwrap())
                .with_step_tolerance(StepTolerance::new(0.0, 1e-8).unwrap())
                .with_stopping(LeastSquaresStopping::Minpack {
                    // At a zero-residual singular root, portable arithmetic can
                    // retain a different rounding floor than MINPACK. Resolve cost
                    // to a squared scalar epsilon instead of demanding exact zero.
                    objective: Tolerance::new(f64::EPSILON * f64::EPSILON, 1e-8).unwrap(),
                    orthogonality: RelativeTolerance::new(1e-8).unwrap(),
                }),
        )
        .unwrap();
        let mut workspace = solver
            .workspace(LeastSquaresWorkspaceRequest::of(&model).unwrap())
            .unwrap();
        let mut result = LeastSquaresResult::empty();
        let outcome = solver.solve_into(
            &model,
            &x,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        );
        match outcome {
            Ok(()) => {
                let objective = 2.0 * result.cost().unwrap();
                let actual = model.eval(result.x()).unwrap();
                assert!((actual - objective).abs() <= 1e-9 * (1.0 + actual.abs()));
                println!(
                    "LM {index} {} {} {:?} {objective:.16e} reference {} {} {:?}",
                    result.fun_evals(),
                    result.jac_evals(),
                    result.termination(),
                    reference.nfev,
                    reference.njev,
                    reference.outcome.f
                );
                // A reference which declares a step stop without leaving its
                // start supplies no convergence work baseline. These pinned
                // Chebyquad starts must instead preserve finite, non-increasing
                // accepted costs within the same residual-evaluation allowance.
                let contextual = if unchanged_reference {
                    assert_eq!(spec.problem, 15);
                    assert!(case.factor > 1);
                    assert!(matches!(index, 133 | 136 | 137 | 139 | 140));
                    let expected = if matches!(index, 137 | 139) {
                        OptimizeTermination::IterationLimit
                    } else {
                        OptimizeTermination::Converged
                    };
                    assert_eq!(result.termination(), expected);
                    assert_eq!(
                        result.trust(),
                        scioxide_core::PartialResultTrust::Admissible
                    );
                    assert!(result.fun_evals() <= fcap && result.jac_evals() <= fcap);
                    assert!(objective <= case.initial.f.unwrap() * (1.0 + 1e-12));
                    println!(
                        "LM_CONTEXT {index} {:?} {objective:.16e}",
                        result.termination()
                    );
                    true
                } else if step_stopped_reference {
                    // MINPACK stops on a small step far from a minimum (f near
                    // 2.7e40 or 2.1e50). The local method keeps reducing the cost
                    // and may end at its iteration limit; require a finite cost
                    // no worse than the reference, within the work allowance.
                    assert!(matches!(
                        result.termination(),
                        OptimizeTermination::Converged | OptimizeTermination::IterationLimit
                    ));
                    assert!(objective.is_finite());
                    assert!(reference.outcome.f.is_some_and(|f| objective <= f));
                    // One residual at the start, then at most one trial and
                    // one Jacobian per permitted iteration.
                    assert!(result.fun_evals() <= iteration_cap + 1);
                    assert!(result.jac_evals() <= iteration_cap);
                    println!(
                        "LM_CONTEXT {index} step-stopped-reference {:?} {objective:.16e}",
                        result.termination()
                    );
                    true
                } else {
                    false
                };
                let target = reference.outcome.f.map(|f| {
                    if brown && brown_plateau(result.x()) {
                        f.max(1.0)
                    } else {
                        f
                    }
                });
                if !reference.success {
                    assert_eq!(index, 134);
                    assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
                    assert!(objective <= case.initial.f.unwrap());
                }
                if !contextual
                    && (result.fun_evals() > fcap
                        || result.jac_evals() > jcap
                        || (reference.success && !result.is_success())
                        || (reference.success
                            && target.is_some_and(|f| objective > f + 2e-5 * (1.0 + f.abs()))))
                {
                    failures.push(format!("case {index}: f={} j={} {:?} objective={objective}; caps {fcap}/{jcap}, reference {:?}", result.fun_evals(), result.jac_evals(), result.termination(), reference.outcome.f));
                }
            }
            Err(error) => failures.push(format!("case {index}: {error:?}")),
        }
    }
    assert!(
        failures.is_empty(),
        "MINPACK work/objectives: {failures:#?}"
    );
}
