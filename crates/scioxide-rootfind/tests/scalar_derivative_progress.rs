#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

use core::cell::RefCell;
use scioxide_core::{EvaluationRole, Interval, RealScalar, ResidualTolerance, StepTolerance};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    Brent, Halley, Newton, RootResult, RootTermination, SafeguardedHalley, SafeguardedInitialGuess,
    SafeguardedNewton, ScalarFn, ScalarFunction, ScalarRootOptions, ScalarRootSolver, Secant,
};

fn r<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

fn evaluations<S: RealScalar>(result: &RootResult<S>) -> usize {
    result.counts().provider(EvaluationRole::Function)
}

fn fast_certified<S: RealScalar>(f: &impl ScalarFunction<S>, lower: S, upper: S, root: S) {
    let interval = Interval::new(lower, upper).unwrap();
    let initial = SafeguardedInitialGuess::new(interval, upper).unwrap();
    let options = ScalarRootOptions::default();
    let open = Newton::new(DerivativeStrategy::Analytic)
        .solve(f, upper, &options)
        .unwrap();
    let brent = Brent.solve(f, interval, &options).unwrap();
    assert!(open.is_success(), "{open:?}");
    assert!(brent.is_success(), "{brent:?}");
    for result in [
        SafeguardedNewton::new(DerivativeStrategy::Analytic).solve(f, initial, &options),
        SafeguardedHalley.solve(f, initial, &options),
    ] {
        let result = result.unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!((result.point() - root).abs() <= r::<S>(16.) * S::EPSILON * root.abs().max(S::ONE));
        assert!(
            evaluations(&result) <= evaluations(&open) + 3,
            "guarded={result:?}, open={open:?}"
        );
        assert!(
            evaluations(&result) <= evaluations(&brent),
            "guarded={result:?}, brent={brent:?}"
        );
        if result.residual() != S::ZERO {
            assert_eq!(result.termination(), RootTermination::BracketTolerance);
            let bracket = result.bracket().unwrap();
            assert!(
                options
                    .step_tolerance()
                    .tolerance()
                    .is_converged(bracket.lower(), bracket.upper())
                    .unwrap()
            );
        }
    }
}

fn ordinary<S: RealScalar>() {
    for c in [1.2, 1.5, 2., 3.] {
        let c = r::<S>(c);
        let f = ScalarFn::new(move |x: S| Ok(x * x - c))
            .with_derivative(|x| Ok(S::TWO * x))
            .with_second_derivative(|_| Ok(S::TWO));
        fast_certified(&f, S::ZERO, S::TWO, c.sqrt());
    }
    let f = ScalarFn::new(|x: S| Ok(x.exp() - S::TWO))
        .with_derivative(|x: S| Ok(x.exp()))
        .with_second_derivative(|x: S| Ok(x.exp()));
    fast_certified(&f, r(-5.), r(5.), S::TWO.ln());
}

#[test]
fn safeguarded_derivatives_keep_fast_default_convergence() {
    ordinary::<f32>();
    ordinary::<f64>();
    for c in [1.2, 1.5, 2., 3.] {
        let f = ScalarFn::new(move |x: f64| Ok(x * x - c))
            .with_derivative(|x| Ok(2. * x))
            .with_second_derivative(|_| Ok(2.));
        for upper in [1e6, 1e13] {
            fast_certified(&f, 0., upper, c.sqrt());
        }
    }
}

#[test]
fn stagnating_open_methods_publish_the_best_evaluated_point() {
    for method in 0..4 {
        let samples = RefCell::new(Vec::new());
        let f = ScalarFn::new(|x: f64| {
            let value = x * x + 1.;
            samples.borrow_mut().push((x, value));
            Ok(value)
        })
        .with_derivative(|x| Ok(2. * x))
        .with_second_derivative(|_| Ok(0.));
        let options = ScalarRootOptions::default()
            .with_step_tolerance(StepTolerance::new(100., 0.).unwrap())
            .with_residual_tolerance(ResidualTolerance::new(0., 0.).unwrap());
        let result = match method {
            0 => Secant.solve(&f, (0., 0.1), &options),
            1 => Newton::new(DerivativeStrategy::Analytic).solve(&f, 0.1, &options),
            2 => Halley.solve(&f, 0.1, &options),
            _ => Newton::new(DerivativeStrategy::Central).solve(&f, 0.1, &options),
        }
        .unwrap();
        assert_eq!(result.termination(), RootTermination::Stagnation);
        let samples = samples.borrow();
        assert!(samples.last().unwrap().1 > result.residual());
        let best = samples
            .iter()
            .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap();
        assert_eq!((result.point(), result.residual()), *best);
        assert_eq!(evaluations(&result), samples.len());
    }
}

#[test]
fn stagnation_ties_keep_the_first_observation() {
    let f = ScalarFn::new(|_: f64| Ok(1.));
    let result = Secant
        .solve(&f, (0., 1.), &ScalarRootOptions::default())
        .unwrap();
    assert_eq!(result.termination(), RootTermination::Stagnation);
    assert_eq!(result.point(), 0.);
    assert_eq!(evaluations(&result), 2);
}

#[test]
fn iteration_limits_publish_the_best_evaluated_point() {
    use scioxide_core::{Budget, Complex};
    use scioxide_rootfind::{
        ComplexMuller, ComplexNewton, ComplexRootSolver, ComplexScalarFn, ComplexSecant, Muller,
    };
    let options = ScalarRootOptions::default().with_max_iterations(Budget::try_from(3).unwrap());
    for method in 0..3 {
        let samples = RefCell::new(Vec::new());
        let f = ScalarFn::new(|x: f64| {
            let value = if method == 2 { x.atan() } else { x.cos() - x };
            samples.borrow_mut().push((x, value));
            Ok(value)
        })
        .with_derivative(|x: f64| Ok(-x.sin() - 1.));
        let result = match method {
            0 => Secant.solve(&f, (3., 3.25), &options),
            1 => Newton::new(DerivativeStrategy::Analytic).solve(&f, -3., &options),
            _ => Muller.solve(&f, (3., 3.25, 3.75), &options),
        }
        .unwrap();
        assert_eq!(result.termination(), RootTermination::IterationLimit);
        let samples = samples.borrow();
        let best = samples
            .iter()
            .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap();
        assert_ne!(samples.last().unwrap(), best, "method {method}");
        assert_eq!((result.point(), result.residual()), *best);
        assert_eq!(evaluations(&result), samples.len());
    }
    for method in 0..3 {
        let samples = RefCell::new(Vec::new());
        // Newton stays on the real axis of z² + 1; the cubic needs more than
        // one Muller step.
        let f = ComplexScalarFn::new(|z: Complex<f64>| {
            let value = if method == 2 {
                z * z * z - z * 2. + Complex::new(2., 0.)
            } else {
                z * z + Complex::new(1., 0.)
            };
            samples.borrow_mut().push((z, value));
            Ok(value)
        })
        .with_derivative(|z| Ok(z + z));
        let start = Complex::new(-2., 0.);
        let result = match method {
            0 => ComplexNewton.solve(&f, start, &options),
            1 => ComplexSecant.solve(&f, (start, start + Complex::new(0.25, 0.25)), &options),
            _ => ComplexMuller.solve(
                &f,
                (
                    Complex::new(3., 0.),
                    Complex::new(3.5, 0.),
                    Complex::new(4., 0.),
                ),
                &options.clone().with_max_iterations(Budget::ONE),
            ),
        }
        .unwrap();
        assert_eq!(result.termination(), RootTermination::IterationLimit);
        let samples = samples.borrow();
        let best = samples
            .iter()
            .min_by(|a, b| a.1.norm().total_cmp(&b.1.norm()))
            .unwrap();
        assert_eq!((result.point(), result.residual()), *best);
        assert_eq!(
            result.counts().provider(EvaluationRole::Function),
            samples.len()
        );
    }
}

#[test]
fn exact_initial_zeros_end_before_later_seeds_are_evaluated() {
    use scioxide_core::{Complex, DomainReason, SciOxideError};
    use scioxide_rootfind::{
        ComplexMuller, ComplexRootSolver, ComplexScalarFn, ComplexSecant, Muller,
    };
    let outside = || {
        SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            DomainReason::OutsideDomain,
            "only the first seed is in the domain",
        )
    };
    // Later seeds lie outside the domain: evaluating one would fail the solve.
    let f = ScalarFn::new(|x: f64| if x == 1.0 { Ok(0.0) } else { Err(outside()) });
    let options = ScalarRootOptions::default();
    for result in [
        Secant.solve(&f, (1.0, 2.0), &options),
        Muller.solve(&f, (1.0, 2.0, 3.0), &options),
    ] {
        let result = result.unwrap();
        assert_eq!(result.termination(), RootTermination::ResidualTolerance);
        assert_eq!(result.point(), 1.0);
        assert_eq!(evaluations(&result), 1);
    }
    let one = Complex::new(1.0, 0.0);
    let g = ComplexScalarFn::new(|z: Complex<f64>| {
        if z == Complex::new(1.0, 0.0) {
            Ok(Complex::new(0.0, 0.0))
        } else {
            Err(outside())
        }
    });
    for result in [
        ComplexSecant.solve(&g, (one, Complex::new(2.0, 0.0)), &options),
        ComplexMuller.solve(
            &g,
            (one, Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)),
            &options,
        ),
    ] {
        let result = result.unwrap();
        assert_eq!(result.termination(), RootTermination::ResidualTolerance);
        assert_eq!(result.point(), one);
        assert_eq!(result.counts().provider(EvaluationRole::Function), 1);
    }
    // A zero at the second seed still needs only the two evaluations.
    let h = ScalarFn::new(|x: f64| Ok(x - 2.0));
    let result = Secant.solve(&h, (1.0, 2.0), &options).unwrap();
    assert_eq!(result.point(), 2.0);
    assert_eq!(evaluations(&result), 2);
}

#[test]
fn an_exactly_zero_derivative_probe_is_published_as_a_root() {
    use scioxide_core::Budget;
    let options = ScalarRootOptions::default().with_max_iterations(Budget::ONE);
    // Record where a central-difference Newton step from 1 probes.
    let points = RefCell::new(Vec::new());
    let record = ScalarFn::new(|x: f64| {
        points.borrow_mut().push(x);
        Ok(x * x - 2.0)
    });
    Newton::new(DerivativeStrategy::Central)
        .solve(&record, 1.0, &options)
        .unwrap();
    let probe = points
        .borrow()
        .iter()
        .copied()
        .find(|&x| x < 1.0 && x > 0.999)
        .unwrap();
    // Roots at the lower probe and just below it: the Newton candidate lands
    // between them, so only the probe observation is an exact zero.
    let other = 1.0 - 100.0 * (1.0 - probe);
    let f = ScalarFn::new(move |x: f64| Ok((x - probe) * (x - other)));
    let result = Newton::new(DerivativeStrategy::Central)
        .solve(&f, 1.0, &options)
        .unwrap();
    assert_eq!(result.termination(), RootTermination::ResidualTolerance);
    assert_eq!(result.point(), probe);
    assert_eq!(result.residual(), 0.0);
}

#[test]
fn a_stop_between_seeds_reports_against_the_evaluated_seed() {
    use scioxide_core::{Budget, BudgetKind, SolveBudgets};
    let options = ScalarRootOptions::default()
        .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE));
    let f = ScalarFn::new(|x: f64| Ok(x - 10.0));
    let result = Secant.solve(&f, (0.0, 1.0), &options).unwrap();
    assert_eq!(
        result.termination(),
        RootTermination::WorkLimit(BudgetKind::Evaluation)
    );
    assert_eq!(result.point(), 0.0);
    // The reference is |f(0)| = 10, the only sample, not one function unit.
    assert_eq!(result.residual_report().scaled_norm(), 1.0);
}
