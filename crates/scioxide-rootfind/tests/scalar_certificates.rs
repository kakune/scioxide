use std::cell::Cell;

use scioxide_core::{EvaluationRole, Interval, NonFiniteOrigin, SciOxideError, Tolerance};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    Bisection, BracketSearchOptions, Brent, Muller, Newton, RootResult, RootTermination,
    ScalarFunction, ScalarRootInput, ScalarRootMethod, ScalarRootOptions, ScalarRootSolver, Secant,
    find_bracket_with_options,
};

fn evaluations(result: &RootResult<f64>, role: EvaluationRole) -> usize {
    result.counts().provider(role)
}

struct Identity;

impl ScalarFunction<f64> for Identity {
    fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
        Ok(x)
    }
}

#[test]
fn bracket_solvers_return_certificates_and_exact_counts() {
    let interval = Interval::new(-f64::MAX, f64::MAX).unwrap();
    for method in [
        ScalarRootMethod::Bisection,
        ScalarRootMethod::Brent,
        ScalarRootMethod::RegulaFalsi,
        ScalarRootMethod::Illinois,
        ScalarRootMethod::Ridder,
    ] {
        let result = method
            .solve(
                &Identity,
                ScalarRootInput::Bracket(interval),
                &ScalarRootOptions::default(),
            )
            .unwrap();
        let certificate = result.bracket().unwrap();
        assert_eq!(result.termination(), RootTermination::ResidualTolerance);
        assert_eq!(result.point(), 0.0);
        assert_eq!(evaluations(&result, EvaluationRole::Jacobian), 0);
        assert_eq!(evaluations(&result, EvaluationRole::Function), 3);
        assert_eq!(certificate.lower_value(), 0.0);
        assert_eq!(certificate.upper_value(), 0.0);
    }
}

#[test]
fn sign_classification_preserves_subnormal_values() {
    let smallest = f64::from_bits(1);
    let result = Bisection
        .solve(
            &Identity,
            Interval::new(-smallest, smallest).unwrap(),
            &ScalarRootOptions::default()
                .with_residual_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
                    Tolerance::new(0.0, 0.0).unwrap(),
                ))
                .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
                    Tolerance::new(0.0, 0.0).unwrap(),
                )),
        )
        .unwrap();
    assert_eq!(result.point(), 0.0);
    assert_eq!(evaluations(&result, EvaluationRole::Function), 3);
}

struct SubnormalStep;

impl ScalarFunction<f64> for SubnormalStep {
    fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
        assert!(x.is_finite());
        let magnitude = f64::from_bits(1);
        Ok(if x < 0.0 { -magnitude } else { magnitude })
    }
}

#[test]
fn every_false_position_family_solver_preserves_subnormal_signs() {
    let options = ScalarRootOptions::default()
        .with_residual_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
            Tolerance::new(0.0, 0.0).unwrap(),
        ))
        .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
            Tolerance::new(0.0, 0.0).unwrap(),
        ));
    for method in [
        ScalarRootMethod::Bisection,
        ScalarRootMethod::Brent,
        ScalarRootMethod::RegulaFalsi,
        ScalarRootMethod::Illinois,
    ] {
        let result = method
            .solve(
                &SubnormalStep,
                ScalarRootInput::Bracket(Interval::new(-1.0, 1.0).unwrap()),
                &options,
            )
            .unwrap();
        let certificate = result.bracket().unwrap();
        assert!(certificate.lower_value() < 0.0);
        assert!(certificate.upper_value() > 0.0);
        assert!(certificate.lower() <= certificate.upper());
    }
}

struct AlmostConstant;

impl ScalarFunction<f64> for AlmostConstant {
    fn eval(&self, _x: f64) -> Result<f64, SciOxideError> {
        Ok(1.0)
    }

    fn derivative(&self, _x: f64) -> Result<Option<f64>, SciOxideError> {
        Ok(Some(f64::MAX))
    }
}

#[test]
fn collapsed_open_steps_are_stagnation_without_redundant_callbacks() {
    let options = ScalarRootOptions::default()
        .with_residual_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
            Tolerance::new(0.0, 0.0).unwrap(),
        ))
        .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
            Tolerance::new(0.0, 0.0).unwrap(),
        ));
    let newton = Newton::new(DerivativeStrategy::Analytic)
        .solve(&AlmostConstant, 1.0, &options)
        .unwrap();
    assert_eq!(newton.termination(), RootTermination::Stagnation);
    assert_eq!(evaluations(&newton, EvaluationRole::Function), 1);
    assert_eq!(evaluations(&newton, EvaluationRole::Jacobian), 1);

    let secant = Secant.solve(&AlmostConstant, (0.0, 1.0), &options).unwrap();
    assert_eq!(secant.termination(), RootTermination::Stagnation);
    assert_eq!(evaluations(&secant, EvaluationRole::Function), 2);
    assert!(!secant.is_success());

    let muller = Muller
        .solve(&AlmostConstant, (0.0, 1.0, 2.0), &options)
        .unwrap();
    assert_eq!(muller.termination(), RootTermination::Stagnation);
    assert_eq!(evaluations(&muller, EvaluationRole::Function), 3);
    assert!(!muller.is_success());
}

struct ScaledLinear(f64);

impl ScalarFunction<f64> for ScaledLinear {
    fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
        Ok(self.0 * (x - 0.75))
    }
}

#[test]
fn scaled_equivalent_functions_have_equivalent_outcomes() {
    let interval = Interval::new(0.0, 2.0).unwrap();
    let options = ScalarRootOptions::default().with_residual_tolerance(
        scioxide_core::ResidualTolerance::from_tolerance(Tolerance::new(0.0, 1e-12).unwrap()),
    );
    let reference = Brent.solve(&ScaledLinear(1.0), interval, &options).unwrap();
    for scale in [f64::MIN_POSITIVE, 1.0e200] {
        let scaled = Brent
            .solve(
                &ScaledLinear(scale),
                interval,
                &options.clone().with_residual_scale(scale),
            )
            .unwrap();
        assert_eq!(scaled.termination(), reference.termination());
        assert_eq!(
            scaled.counts().iterations(),
            reference.counts().iterations()
        );
        assert!((scaled.point() - reference.point()).abs() <= 4.0 * f64::EPSILON);
    }
}

struct DifferentiableScaledLinear(f64);

impl ScalarFunction<f64> for DifferentiableScaledLinear {
    fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
        Ok(self.0 * (x - 1.0))
    }

    fn derivative(&self, _x: f64) -> Result<Option<f64>, SciOxideError> {
        Ok(Some(self.0))
    }
}

#[test]
fn tiny_derivative_and_differences_use_operand_scales() {
    for scale in [1.0e-300, 1.0, 1.0e300] {
        let function = DifferentiableScaledLinear(scale);
        let options = ScalarRootOptions::default()
            .with_residual_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
                Tolerance::new(0.0, 1e-12).unwrap(),
            ))
            .with_residual_scale(scale);
        let newton = Newton::new(DerivativeStrategy::Analytic)
            .solve(&function, 0.0, &options)
            .unwrap();
        let secant = Secant.solve(&function, (0.0, 2.0), &options).unwrap();
        assert_eq!(newton.termination(), RootTermination::ResidualTolerance);
        assert_eq!(secant.termination(), RootTermination::ResidualTolerance);
        assert_eq!(newton.point(), 1.0);
        assert_eq!(secant.point(), 1.0);
    }
}

#[test]
fn automatic_bracket_search_never_calls_back_with_infinity() {
    struct FiniteOnly(Cell<usize>);

    impl ScalarFunction<f64> for FiniteOnly {
        fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
            assert!(x.is_finite());
            self.0.set(self.0.get() + 1);
            Ok(1.0)
        }
    }

    let function = FiniteOnly(Cell::new(0));
    let result = find_bracket_with_options(
        &function,
        scioxide_rootfind::BracketSearchInput::Guess(f64::MAX),
        scioxide_rootfind::BracketSearchDirection::Automatic,
        &BracketSearchOptions::default(),
    );
    assert_eq!(
        result.unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(function.0.get(), 3);
}

struct NonFiniteCallback;

impl ScalarFunction<f64> for NonFiniteCallback {
    fn eval(&self, _x: f64) -> Result<f64, SciOxideError> {
        Ok(f64::NAN)
    }
}

#[test]
fn non_finite_callback_values_have_callback_provenance() {
    let error = Bisection
        .solve(
            &NonFiniteCallback,
            Interval::new(-1.0, 1.0).unwrap(),
            &ScalarRootOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Callback,
            ..
        }
    ));
}

struct CountedLinear<'a>(&'a Cell<usize>);

impl ScalarFunction<f64> for CountedLinear<'_> {
    fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
        self.0.set(self.0.get() + 1);
        Ok(x - 1.0)
    }
}

#[test]
fn public_function_counter_matches_completed_callbacks() {
    let count = Cell::new(0);
    let result = Secant
        .solve(
            &CountedLinear(&count),
            (0.0, 2.0),
            &ScalarRootOptions::default(),
        )
        .unwrap();
    assert_eq!(evaluations(&result, EvaluationRole::Function), count.get());
}
