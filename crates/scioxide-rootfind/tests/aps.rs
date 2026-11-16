//! Independent SciPy APS roots and endpoint-inclusive callback counts.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
#![expect(
    clippy::panic,
    reason = "Tests report unexpected fixture cases or solver outcomes."
)]
use scioxide_core::{
    Budget, EvaluationRole, Interval, RealScalar, ResidualTolerance, StepTolerance,
};
use scioxide_rootfind::{
    Ridder, RootTermination, ScalarFn, ScalarFunction, ScalarRootInput, ScalarRootMethod,
    ScalarRootOptions, ScalarRootSolver,
};
use std::cell::Cell;

fn r<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn equation<S: RealScalar>(problem: usize, x: S, p: &[f64]) -> S {
    let n: S = r(p.first().copied().unwrap_or(1.0));
    match problem {
        1 => x.sin() - x / S::TWO,
        2 => (1..=20)
            .map(|i| {
                let t = x - r::<S>((i * i) as f64);
                -S::TWO * r::<S>(((2 * i - 5_i32).pow(2)) as f64) / (t * t * t)
            })
            .fold(S::ZERO, |a, b| a + b),
        3 => n * x * (r::<S>(p[1]) * x).exp(),
        4 => x.powi(p[0] as i32) - r::<S>(p[1]),
        5 => x.sin() - S::HALF,
        6 => S::TWO * x * (-n).exp() - S::TWO * (-n * x).exp() + S::ONE,
        7 => (S::ONE + (S::ONE - n).powi(2)) * x - (S::ONE - n * x).powi(2),
        8 => x * x - (S::ONE - x).powi(p[0] as i32),
        9 => (S::ONE + (S::ONE - n).powi(4)) * x - (S::ONE - n * x).powi(4),
        10 => (-n * x).exp() * (x - S::ONE) + x.powi(p[0] as i32),
        11 => (n * x - S::ONE) / ((n - S::ONE) * x),
        12 => x.powf(S::ONE / n) - n.powf(S::ONE / n),
        // Match the reference's finite callback, including its underflow plateau.
        13 => {
            if x == S::ZERO || S::ONE / (x * x) > S::MAX.ln() {
                S::ZERO
            } else {
                x / (S::ONE / (x * x)).exp()
            }
        }
        14 => {
            if x <= S::ZERO {
                -n / r(20.0)
            } else {
                n / r(20.0) * (x / r(1.5) + x.sin() - S::ONE)
            }
        }
        15 => {
            if x < S::ZERO {
                r(-0.859)
            } else if x > r::<S>(0.002) / (S::ONE + n) {
                S::ONE.exp() - r(1.859)
            } else {
                ((n + S::ONE) * x / S::TWO * r(1000.0)).exp() - r(1.859)
            }
        }
        _ => panic!("unknown APS problem"),
    }
}

struct ApsFunction {
    problem: usize,
    parameters: Vec<f64>,
    calls: Cell<usize>,
}
impl<S: RealScalar> ScalarFunction<S> for ApsFunction {
    fn eval(&self, x: S) -> Result<S, scioxide_core::SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        Ok(equation(self.problem, x, &self.parameters))
    }
}

fn aps<S: RealScalar>(xtol: S, counts: bool) {
    let methods = [
        ("bisect", ScalarRootMethod::Bisection),
        ("ridder", ScalarRootMethod::Ridder),
        ("brentq", ScalarRootMethod::Brent),
        ("brenth", ScalarRootMethod::Brenth),
        ("toms748", ScalarRootMethod::Toms748),
        ("chandrupatla", ScalarRootMethod::Chandrupatla),
        ("illinois", ScalarRootMethod::Illinois),
    ];
    let options = ScalarRootOptions::default()
        .with_residual_tolerance(ResidualTolerance::new(S::ZERO, S::ZERO).unwrap())
        .with_step_tolerance(StepTolerance::new(xtol, r::<S>(4.0) * S::EPSILON).unwrap())
        .with_max_iterations(Budget::try_from(1000).unwrap());
    let mut failures = Vec::new();
    for (name, solver) in methods {
        let mut total = 0;
        let mut reference_total = 0;
        let mut cases = 0;
        for line in include_str!("data/aps.tsv")
            .lines()
            .filter(|l| !l.starts_with('#'))
        {
            let row: Vec<_> = line.split('\t').collect();
            if row[5] != if name == "illinois" { "ridder" } else { name } {
                continue;
            }
            cases += 1;
            let problem: usize = row[0][4..6].parse().unwrap();
            let parameters: Vec<f64> = if row[1] == "-" {
                vec![]
            } else {
                row[1].split(',').map(|p| p.parse().unwrap()).collect()
            };
            let mut a: S = r(row[2].parse().unwrap());
            let mut b: S = r(row[3].parse().unwrap());
            if problem == 2 {
                // The 1e-9 pole offsets disappear on conversion to f32.
                if a == a.sqrt().floor().powi(2) {
                    a = a.next_after(b);
                }
                if b == b.sqrt().floor().powi(2) {
                    b = b.next_after(a);
                }
            }
            let function = ApsFunction {
                problem,
                parameters,
                calls: Cell::new(0),
            };
            let calls = &function.calls;
            let result = solver.solve(
                &function as &dyn ScalarFunction<S>,
                ScalarRootInput::Bracket(Interval::new(a, b).unwrap()),
                &options,
            );
            let result = match result {
                Ok(v) => v,
                Err(e) => {
                    failures.push(format!("{name} {}: {e}", row[0]));
                    continue;
                }
            };
            if !result.is_success() {
                failures.push(format!("{name} {}: {:?}", row[0], result.termination()));
                continue;
            }
            assert_eq!(
                result.counts().provider(EvaluationRole::Function),
                calls.get()
            );
            let bracket = result.bracket().unwrap();
            assert!(bracket.lower() >= a && bracket.upper() <= b);
            match result.termination() {
                RootTermination::ResidualTolerance => assert_eq!(result.residual(), S::ZERO),
                RootTermination::BracketTolerance => assert!(
                    options
                        .step_tolerance()
                        .tolerance()
                        .is_converged(bracket.lower(), bracket.upper())
                        .unwrap()
                ),
                reason => panic!("unexpected success {reason:?}"),
            }
            if problem != 13 {
                for reference in [row[4], row[6]] {
                    let reference: S = r(reference.parse().unwrap());
                    // Subtraction of nth roots in p12 has relative condition
                    // proportional to n; a rounded zero alone cannot remove it.
                    let conditioning = if problem == 12 {
                        r(function.parameters[0])
                    } else {
                        S::ONE
                    };
                    let allowance = r::<S>(8.0) * xtol
                        + r::<S>(64.0) * S::EPSILON * reference.abs().max(S::ONE) * conditioning;
                    assert!(
                        (result.point() - reference).abs() <= allowance,
                        "{name} {} {:?} vs {reference:?}",
                        row[0],
                        result.point()
                    );
                }
            }
            let reference: usize = row[7].parse().unwrap();
            total += calls.get();
            reference_total += reference;
            if counts
                && ((name == "chandrupatla" && ["aps.07.00", "aps.08.00"].contains(&row[0]))
                    || (name == "brenth" && row[0] == "aps.02.04")
                    || (name == "toms748" && ["aps.03.00", "aps.04.00"].contains(&row[0])))
                && calls.get() > reference + 2
            {
                failures.push(format!(
                    "{name} {} outlier {} > {}+2",
                    row[0],
                    calls.get(),
                    reference
                ));
            }
        }
        assert_eq!(cases, 154);
        if counts
            && [
                "bisect",
                "ridder",
                "brentq",
                "brenth",
                "toms748",
                "chandrupatla",
            ]
            .contains(&name)
            && total * 5 > reference_total * 6
        {
            failures.push(format!("{name} total {total} > 1.2 * {reference_total}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[test]
fn aps_f64() {
    aps::<f64>(2e-12, true);
}
#[test]
fn aps_f32() {
    aps::<f32>(4.0 * f32::EPSILON, false);
}

#[test]
fn width_only_kepler() {
    let options = ScalarRootOptions::default()
        .with_step_tolerance(StepTolerance::new(1e-12, 4.0 * f64::EPSILON).unwrap());
    for solver in [ScalarRootMethod::Ridder, ScalarRootMethod::Illinois] {
        for i in 0..512 {
            let m = core::f64::consts::PI * i as f64 / 511.0;
            let f = ScalarFn::new(move |x: f64| Ok(x - 0.5 * x.sin() - m));
            let result = solver
                .solve(
                    &f,
                    ScalarRootInput::Bracket(Interval::new(0.0, core::f64::consts::PI).unwrap()),
                    &options,
                )
                .unwrap();
            assert!(
                result.is_success(),
                "{} {m}: {result:?}",
                ScalarRootSolver::<f64>::name(&solver)
            );
            assert!((result.point() - 0.5 * result.point().sin() - m).abs() <= 3e-12);
        }
    }
}

#[test]
fn ridder_resource_stop_retains_the_evaluated_midpoint() {
    use scioxide_core::{BudgetKind, SolveBudgets};
    for (kind, limit, reason) in [
        (
            BudgetKind::Evaluation,
            3,
            RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation),
        ),
        (
            BudgetKind::Proposal,
            1,
            RootTermination::WorkLimit(scioxide_core::BudgetKind::Proposal),
        ),
    ] {
        let calls = Cell::new(0);
        let f = ScalarFn::new(|x: f64| {
            calls.set(calls.get() + 1);
            Ok(x.cos() - x)
        });
        let options = ScalarRootOptions::default().with_budgets(
            SolveBudgets::unbounded().with_limit(kind, Budget::try_from(limit).unwrap()),
        );
        let result = Ridder
            .solve(&f, Interval::new(0.0, 1.0).unwrap(), &options)
            .unwrap();
        assert_eq!(result.termination(), reason);
        assert_eq!(calls.get(), 3);
        assert_eq!(result.counts().provider(EvaluationRole::Function), 3);
        let bracket = result.bracket().unwrap();
        assert_eq!(bracket.lower(), 0.5);
        assert_eq!(bracket.upper(), 1.0);
        assert_eq!(result.point(), 0.5);
        assert_eq!(result.residual(), 0.5f64.cos() - 0.5);
    }
}
