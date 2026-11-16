use super::*;
use scioxide_core::{
    Budget, BudgetKind, Complex, ErrorTolerance, ExecutionContext, ExecutionPolicy, NormSelection,
    RealScalar, SciOxideError, SolveBudgets, SolveIdentity, SolveState,
};

fn budget(n: usize) -> Budget {
    Budget::try_from(n).unwrap()
}

fn tolerance<R: RealScalar>(x: R) -> ErrorTolerance<R> {
    ErrorTolerance::scalar(x, x, NormSelection::Infinity).unwrap()
}

#[test]
fn levin_u_accelerates_zeta_two_without_claiming_a_bound() {
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::LevinU,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(24))
            .with_order(budget(10))
            .with_tolerance(tolerance(1e-9)),
    )
    .unwrap();
    let result = plan
        .sum(SeriesProblem::Terms(&|n: usize, row: &mut [_]| {
            let value: Result<_, SciOxideError> = {
                let x = (n + 1) as f64;
                Ok(1.0 / (x * x))
            };
            row[0] = value?;
            Ok::<(), SciOxideError>(())
        }))
        .unwrap();
    let expected = core::f64::consts::PI.powi(2) / 6.0;
    assert!((result.value - expected).abs() < 1e-7, "{result:?}");
    assert!(result.error.tail_bound.is_none());
    assert_eq!(
        result.report.termination,
        SeriesTermination::HeuristicToleranceReached
    );
    assert!(result.report.termination.is_success());
    assert!(result.report.terms <= 24);
}

#[test]
fn wynn_alternating_zeta_two_and_complex_geometric() {
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(24))
            .with_order(budget(12))
            .with_tolerance(tolerance(1e-11)),
    )
    .unwrap();
    let result = plan
        .sum(SeriesProblem::Terms(&|n: usize, row: &mut [_]| {
            let value: Result<_, SciOxideError> = {
                let x = (n + 1) as f64;
                Ok(if n.is_multiple_of(2) { 1.0 } else { -1.0 } / (x * x))
            };
            row[0] = value?;
            Ok::<(), SciOxideError>(())
        }))
        .unwrap();
    let expected = core::f64::consts::PI.powi(2) / 12.0;
    assert!((result.value - expected).abs() < 1e-8, "{result:?}");
    assert_eq!(
        result.report.termination,
        SeriesTermination::HeuristicToleranceReached
    );
    assert!(result.report.termination.is_success());

    let z = Complex::new(0.5_f64, 0.2);
    let complex = SeriesSummation::<Complex<f64>>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(16))
            .with_order(budget(8))
            .with_tolerance(tolerance(1e-9)),
    )
    .unwrap();
    let result = complex
        .sum(SeriesProblem::Terms(&|n: usize, row: &mut [_]| {
            let value: Result<_, SciOxideError> = {
                let mut term = Complex::new(1.0, 0.0);
                for _ in 0..n {
                    term *= z;
                }
                Ok(term)
            };
            row[0] = value?;
            Ok::<(), SciOxideError>(())
        }))
        .unwrap();
    let exact = Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) - z);
    assert!((result.value - exact).norm() < 1e-8, "{result:?}");
}

struct InverseSquare;
impl SeriesTerms<f64> for InverseSquare {
    fn term(&self, index: usize, output: &mut [f64]) -> Result<(), SciOxideError> {
        let x = index as f64;
        output[0] = 1.0 / (x * x);
        Ok::<(), SciOxideError>(())
    }
}
impl EulerMaclaurinProvider<f64> for InverseSquare {
    fn tail_integral(
        &self,
        at: usize,
        value: &mut [f64],
        error: &mut [f64],
    ) -> Result<(), SciOxideError> {
        value[0] = 1.0 / at as f64;
        error[0] = 64.0 * f64::EPSILON * value[0].abs();
        Ok::<(), SciOxideError>(())
    }
    fn odd_derivative(
        &self,
        at: usize,
        order: usize,
        value: &mut [f64],
        error: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let factorial = (2..=order + 1).fold(1.0, |a, b| a * b as f64);
        value[0] = -factorial / (at as f64).powi((order + 2) as i32);
        error[0] = 64.0 * f64::EPSILON * value[0].abs();
        Ok::<(), SciOxideError>(())
    }
    fn even_derivative_l1_bound(
        &self,
        at: usize,
        order: usize,
        output: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let factorial = (2..=order).fold(1.0, |a, b| a * b as f64);
        output[0] = factorial / (at as f64).powi((order + 1) as i32);
        output[0] *= 1.0 + 64.0 * f64::EPSILON;
        Ok::<(), SciOxideError>(())
    }
}

impl SeriesTerms<f32> for InverseSquare {
    fn term(&self, index: usize, output: &mut [f32]) -> Result<(), SciOxideError> {
        let x = index as f32;
        output[0] = 1.0 / (x * x);
        Ok::<(), SciOxideError>(())
    }
}
impl EulerMaclaurinProvider<f32> for InverseSquare {
    fn tail_integral(
        &self,
        at: usize,
        value: &mut [f32],
        error: &mut [f32],
    ) -> Result<(), SciOxideError> {
        value[0] = 1.0 / at as f32;
        error[0] = 64.0 * f32::EPSILON * value[0].abs();
        Ok::<(), SciOxideError>(())
    }
    fn odd_derivative(
        &self,
        at: usize,
        order: usize,
        value: &mut [f32],
        error: &mut [f32],
    ) -> Result<(), SciOxideError> {
        let factorial = (2..=order + 1).fold(1.0_f32, |a, b| a * b as f32);
        value[0] = -factorial / (at as f32).powi((order + 2) as i32);
        error[0] = 64.0 * f32::EPSILON * value[0].abs();
        Ok::<(), SciOxideError>(())
    }
    fn even_derivative_l1_bound(
        &self,
        at: usize,
        order: usize,
        output: &mut [f32],
    ) -> Result<(), SciOxideError> {
        let factorial = (2..=order).fold(1.0_f32, |a, b| a * b as f32);
        output[0] = factorial / (at as f32).powi((order + 1) as i32);
        output[0] *= 1.0 + 64.0 * f32::EPSILON;
        Ok::<(), SciOxideError>(())
    }
}

#[test]
fn euler_maclaurin_reports_conditional_tail_bound() {
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::EulerMaclaurin,
        1,
        SeriesOptions::default()
            .with_max_terms(budget(32))
            .with_order(budget(2))
            .with_tolerance(tolerance(1e-8)),
    )
    .unwrap();
    let result = plan
        .sum(SeriesProblem::EulerMaclaurin(&InverseSquare))
        .unwrap();
    let expected = core::f64::consts::PI.powi(2) / 6.0;
    let actual = (result.value - expected).abs();
    let tail = result.error.tail_bound.unwrap();
    assert!(actual < tail + 1e-14, "{result:?}");
    assert_eq!(
        result.report.termination,
        SeriesTermination::BoundedTailReached
    );
    assert!(result.report.termination.is_success());
    assert_eq!(result.report.provider_calls, result.report.terms * 5);
}

#[test]
fn binary32_euler_maclaurin_has_bounded_tail_evidence() {
    let plan = SeriesSummation::<f32>::new(
        SeriesMethod::EulerMaclaurin,
        1,
        SeriesOptions::default()
            .with_max_terms(budget(32))
            .with_order(budget(2))
            .with_tolerance(tolerance(2e-5)),
    )
    .unwrap();
    let result = plan
        .sum(SeriesProblem::EulerMaclaurin(&InverseSquare))
        .unwrap();
    let expected = core::f32::consts::PI.powi(2) / 6.0;
    assert!((result.value - expected).abs() < 1e-5, "{result:?}");
    assert!(result.error.tail_bound.is_some());
    assert!(result.report.termination.is_success());
}

#[test]
fn vector_resume_budget_and_callback_failure() {
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(12))
            .with_order(budget(6))
            .with_tolerance(tolerance(1e-12)),
    )
    .unwrap();
    let mut ws = plan.state(2).unwrap();
    let mut ws_workspace = plan.workspace(2).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = SolveIdentity::new(
        scioxide_core::ProblemToken::new(596),
        scioxide_core::algorithm_id!("integrate", "series-test", 1),
        0,
        0,
    );
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut output = [99.0, 99.0];
    let mut errors = [SeriesError::default(); 2];
    let first = SeriesSummation::new(
        plan.method,
        plan.first,
        plan.options.clone().with_terms_per_advance(budget(4)),
    )
    .unwrap()
    .advance(
        SeriesProblem::Terms(&|n: usize, row: &mut [f64]| {
            let x = (n + 1) as f64;
            row.copy_from_slice(&[
                0.5_f64.powi(n as i32),
                if n.is_multiple_of(2) { 1.0 } else { -1.0 } / x,
            ]);
            Ok::<(), SciOxideError>(())
        }),
        &mut ws,
        |accepted| {
            if let Some(values) = accepted.value() {
                output.copy_from_slice(values);
            }
            if let Some(accepted_errors) = accepted.errors() {
                errors.copy_from_slice(accepted_errors);
            }
            Ok::<(), SciOxideError>(())
        },
        &mut ws_workspace,
        &mut ctx,
    )
    .unwrap();
    assert_eq!(first.terms, 4);
    assert_eq!(first.termination, SeriesTermination::CallLimit);
    let prior = output;
    let failure = SeriesSummation::new(
        plan.method,
        plan.first,
        plan.options.clone().with_terms_per_advance(budget(1)),
    )
    .unwrap()
    .advance(
        SeriesProblem::Terms(&|_, _: &mut [f64]| {
            Err(crate::kernel::invalid(
                scioxide_core::argument!("provider"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "provider failure",
            ))
        }),
        &mut ws,
        |accepted| {
            if let Some(values) = accepted.value() {
                output.copy_from_slice(values);
            }
            if let Some(accepted_errors) = accepted.errors() {
                errors.copy_from_slice(accepted_errors);
            }
            Ok::<(), SciOxideError>(())
        },
        &mut ws_workspace,
        &mut ctx,
    );
    assert!(failure.is_err());
    assert_eq!(output, prior);
    assert_eq!(ws.terms(), 4);
    let second = SeriesSummation::new(
        plan.method,
        plan.first,
        plan.options
            .with_terms_per_advance(budget(8))
            .with_execution_policy(ExecutionPolicy::Sequential),
    )
    .unwrap()
    .advance(
        SeriesProblem::Terms(&|n: usize, row: &mut [f64]| {
            let x = (n + 1) as f64;
            row.copy_from_slice(&[
                0.5_f64.powi(n as i32),
                if n.is_multiple_of(2) { 1.0 } else { -1.0 } / x,
            ]);
            Ok::<(), SciOxideError>(())
        }),
        &mut ws,
        |accepted| {
            if let Some(values) = accepted.value() {
                output.copy_from_slice(values);
            }
            if let Some(accepted_errors) = accepted.errors() {
                errors.copy_from_slice(accepted_errors);
            }
            Ok::<(), SciOxideError>(())
        },
        &mut ws_workspace,
        &mut ctx,
    )
    .unwrap();
    assert!(second.terms >= 5);
    assert!((output[0] - 2.0).abs() < 1e-5);
    assert!(!second.termination.is_success());
}

#[test]
fn zero_terms_and_false_convergence_do_not_certify() {
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(12))
            .with_order(budget(6))
            .with_tolerance(tolerance(1e-8)),
    )
    .unwrap();
    let result = plan
        .sum(SeriesProblem::Terms(&|n: usize, row: &mut [_]| {
            let value: Result<_, SciOxideError> = {
                Ok(if n == 0 {
                    1.0
                } else if n == 11 {
                    1e10
                } else {
                    0.0
                })
            };
            row[0] = value?;
            Ok::<(), SciOxideError>(())
        }))
        .unwrap();
    assert_eq!(
        result.report.termination,
        SeriesTermination::HeuristicToleranceReached
    );
    assert!(result.report.termination.is_success());
    assert!(result.error.tail_bound.is_none());
    let hidden_spike = SeriesSummation::<f64>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(16))
            .with_order(budget(8))
            .with_tolerance(tolerance(1e-10)),
    )
    .unwrap()
    .sum(SeriesProblem::Terms(&|n: usize, row: &mut [_]| {
        let value: Result<_, SciOxideError> =
            { Ok(0.5_f64.powi(n as i32) + if n == 1000 { 1000.0 } else { 0.0 }) };
        row[0] = value?;
        Ok::<(), SciOxideError>(())
    }))
    .unwrap();
    assert_eq!(
        hidden_spike.report.termination,
        SeriesTermination::HeuristicToleranceReached
    );
    assert!(hidden_spike.report.termination.is_success());
    assert!(hidden_spike.value < 3.0);
    let harmonic = SeriesSummation::<f64>::new(
        SeriesMethod::LevinU,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(16))
            .with_order(budget(8))
            .with_tolerance(tolerance(1e-8)),
    )
    .unwrap()
    .sum(SeriesProblem::Terms(&|n: usize, row: &mut [_]| {
        let value: Result<_, SciOxideError> = { Ok(1.0 / (n + 1) as f64) };
        row[0] = value?;
        Ok::<(), SciOxideError>(())
    }))
    .unwrap();
    assert!(!harmonic.report.termination.is_success());
}

#[test]
fn both_binary32_real_and_complex_are_supported() {
    let plan = SeriesSummation::<f32>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(20))
            .with_order(budget(8))
            .with_tolerance(tolerance(1e-4)),
    )
    .unwrap();
    let real = plan
        .sum(SeriesProblem::Terms(&|n: usize, row: &mut [_]| {
            let value: Result<_, SciOxideError> = {
                let x = (n + 1) as f32;
                Ok(if n.is_multiple_of(2) { 1.0 } else { -1.0 } / (x * x))
            };
            row[0] = value?;
            Ok::<(), SciOxideError>(())
        }))
        .unwrap();
    assert!((real.value - core::f32::consts::PI.powi(2) / 12.0).abs() < 2e-5);

    let z = Complex::new(0.4_f32, 0.2);
    let plan = SeriesSummation::<Complex<f32>>::new(
        SeriesMethod::LevinU,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(20))
            .with_order(budget(8))
            .with_tolerance(tolerance(1e-4)),
    )
    .unwrap();
    let complex = plan
        .sum(SeriesProblem::Terms(&|n: usize, row: &mut [_]| {
            let value: Result<_, SciOxideError> = {
                let mut term = Complex::new(1.0, 0.0);
                for _ in 0..n {
                    term *= z;
                }
                Ok(term)
            };
            row[0] = value?;
            Ok::<(), SciOxideError>(())
        }))
        .unwrap();
    let exact = Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) - z);
    assert!((complex.value - exact).norm() < 5e-4, "{complex:?}");
}

#[cfg(feature = "std")]
#[test]
fn injected_thread_pool_matches_sequential_index_reduction() {
    use std::num::NonZeroUsize;
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(16))
            .with_order(budget(8))
            .with_tolerance(tolerance(1e-12)),
    )
    .unwrap();
    let provider = |n: usize, row: &mut [f64]| {
        row[0] = 0.6_f64.powi(n as i32);
        Ok::<(), SciOxideError>(())
    };
    let mut sequential = plan.state(1).unwrap();
    let mut sequential_workspace = plan.workspace(1).unwrap();
    let mut parallel = plan.state(1).unwrap();
    let mut parallel_workspace = plan.workspace(1).unwrap();
    let mut y0 = [0.0];
    let mut y1 = [0.0];
    let mut e0 = [SeriesError::default()];
    let mut e1 = [SeriesError::default()];
    let mut state0 = SolveState::new();
    let mut budgets0 = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        scioxide_core::ProblemToken::new(596),
        scioxide_core::algorithm_id!("integrate", "series-executor-test", 1),
        0,
        0,
    );
    let mut ctx0 = state0
        .begin_solve(ExecutionContext::sequential(), id, &mut budgets0)
        .unwrap();
    let r0 = SeriesSummation::new(
        plan.method,
        plan.first,
        plan.options.clone().with_terms_per_advance(budget(16)),
    )
    .unwrap()
    .advance(
        SeriesProblem::Terms(&provider),
        &mut sequential,
        |accepted| {
            if let Some(values) = accepted.value() {
                y0.copy_from_slice(values);
            }
            if let Some(accepted_errors) = accepted.errors() {
                e0.copy_from_slice(accepted_errors);
            }
            Ok::<(), SciOxideError>(())
        },
        &mut sequential_workspace,
        &mut ctx0,
    )
    .unwrap();
    let pool = scioxide_core::ThreadExecutor::new(NonZeroUsize::new(3).unwrap());
    let mut state1 = SolveState::new();
    let mut budgets1 = SolveBudgets::unbounded();
    let mut ctx1 = state1
        .begin_solve(ExecutionContext::new(&pool), id, &mut budgets1)
        .unwrap();
    let r1 = SeriesSummation::new(
        plan.method,
        plan.first,
        plan.options
            .with_terms_per_advance(budget(16))
            .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(3).unwrap())),
    )
    .unwrap()
    .advance(
        SeriesProblem::Terms(&provider),
        &mut parallel,
        |accepted| {
            if let Some(values) = accepted.value() {
                y1.copy_from_slice(values);
            }
            if let Some(accepted_errors) = accepted.errors() {
                e1.copy_from_slice(accepted_errors);
            }
            Ok::<(), SciOxideError>(())
        },
        &mut parallel_workspace,
        &mut ctx1,
    )
    .unwrap();
    assert_eq!(y0, y1);
    assert_eq!(r0.terms, r1.terms);
    assert!(r1.provider_calls >= r1.terms);
    assert!((y1[0] - 2.5).abs() < 1e-10);
}

#[test]
fn context_evaluation_budget_returns_measured_partial() {
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(10))
            .with_order(budget(4))
            .with_tolerance(tolerance(1e-10)),
    )
    .unwrap();
    let mut ws = plan.state(1).unwrap();
    let mut ws_workspace = plan.workspace(1).unwrap();
    let mut output = [99.0];
    let mut errors = [SeriesError::default()];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, budget(3));
    let id = SolveIdentity::new(
        scioxide_core::ProblemToken::new(596),
        scioxide_core::algorithm_id!("integrate", "series-budget-test", 1),
        0,
        0,
    );
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
        .unwrap();
    let report = SeriesSummation::new(
        plan.method,
        plan.first,
        plan.options.with_terms_per_advance(budget(10)),
    )
    .unwrap()
    .advance(
        SeriesProblem::Terms(&|n: usize, row: &mut [f64]| {
            row[0] = 0.5_f64.powi(n as i32);
            Ok::<(), SciOxideError>(())
        }),
        &mut ws,
        |accepted| {
            if let Some(values) = accepted.value() {
                output.copy_from_slice(values);
            }
            if let Some(accepted_errors) = accepted.errors() {
                errors.copy_from_slice(accepted_errors);
            }
            Ok::<(), SciOxideError>(())
        },
        &mut ws_workspace,
        &mut ctx,
    )
    .unwrap();
    assert_eq!(
        report.termination,
        SeriesTermination::ContextBudgetLimit(BudgetKind::Evaluation)
    );
    assert_eq!(report.terms, 3);
    assert_eq!(report.provider_calls, 3);
    assert!(report.terms() > 0);
    assert_ne!(output[0], 99.0);
    assert!(errors[0].tail_bound.is_none());
}

#[test]
fn complex_dense_vector_uses_one_indexed_provider_call_per_term() {
    let plan = SeriesSummation::<Complex<f64>>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(18))
            .with_order(budget(8))
            .with_tolerance(tolerance(1e-9)),
    )
    .unwrap();
    let mut ws = plan.state(2).unwrap();
    let mut ws_workspace = plan.workspace(2).unwrap();
    let z = [Complex::new(0.4, 0.2), Complex::new(-0.3, 0.1)];
    let mut output = [Complex::new(0.0, 0.0); 2];
    let mut errors = [SeriesError::default(); 2];
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        scioxide_core::ProblemToken::new(596),
        scioxide_core::algorithm_id!("integrate", "series-complex-vector", 1),
        0,
        0,
    );
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
        .unwrap();
    let report = SeriesSummation::new(
        plan.method,
        plan.first,
        plan.options.with_terms_per_advance(budget(18)),
    )
    .unwrap()
    .advance(
        SeriesProblem::Terms(&|n: usize, row: &mut [Complex<f64>]| {
            for (slot, &base) in row.iter_mut().zip(z.iter()) {
                *slot = (0..n).fold(Complex::new(1.0, 0.0), |term, _| term * base);
            }
            Ok::<(), SciOxideError>(())
        }),
        &mut ws,
        |accepted| {
            if let Some(values) = accepted.value() {
                output.copy_from_slice(values);
            }
            if let Some(accepted_errors) = accepted.errors() {
                errors.copy_from_slice(accepted_errors);
            }
            Ok::<(), SciOxideError>(())
        },
        &mut ws_workspace,
        &mut ctx,
    )
    .unwrap();
    for (value, base) in output.iter().zip(z) {
        let expected = Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) - base);
        assert!((*value - expected).norm() < 1e-8);
    }
    assert!(report.provider_calls <= 18);
    assert!(errors.iter().all(|error| error.tail_bound.is_none()));
}

struct ComplexInverseSquare;
impl SeriesTerms<Complex<f64>> for ComplexInverseSquare {
    fn term(&self, n: usize, output: &mut [Complex<f64>]) -> Result<(), SciOxideError> {
        let x = n as f64;
        output[0] = Complex::new(1.0 / (x * x), 1.0 / (x * x));
        Ok::<(), SciOxideError>(())
    }
}
impl EulerMaclaurinProvider<Complex<f64>> for ComplexInverseSquare {
    fn tail_integral(
        &self,
        at: usize,
        value: &mut [Complex<f64>],
        error: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let x = 1.0 / at as f64;
        value[0] = Complex::new(x, x);
        error[0] = 64.0 * f64::EPSILON * value[0].norm();
        Ok::<(), SciOxideError>(())
    }
    fn odd_derivative(
        &self,
        at: usize,
        order: usize,
        value: &mut [Complex<f64>],
        error: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let factorial = (2..=order + 1).fold(1.0, |a, b| a * b as f64);
        let x = -factorial / (at as f64).powi((order + 2) as i32);
        value[0] = Complex::new(x, x);
        error[0] = 64.0 * f64::EPSILON * value[0].norm();
        Ok::<(), SciOxideError>(())
    }
    fn even_derivative_l1_bound(
        &self,
        at: usize,
        order: usize,
        output: &mut [f64],
    ) -> Result<(), SciOxideError> {
        let factorial = (2..=order).fold(1.0, |a, b| a * b as f64);
        output[0] = core::f64::consts::SQRT_2 * factorial / (at as f64).powi((order + 1) as i32);
        output[0] *= 1.0 + 64.0 * f64::EPSILON;
        Ok::<(), SciOxideError>(())
    }
}

#[test]
fn euler_maclaurin_complex_tail_bound() {
    let plan = SeriesSummation::<Complex<f64>>::new(
        SeriesMethod::EulerMaclaurin,
        1,
        SeriesOptions::default()
            .with_max_terms(budget(32))
            .with_order(budget(2))
            .with_tolerance(tolerance(1e-8)),
    )
    .unwrap();
    let result = plan
        .sum(SeriesProblem::EulerMaclaurin(&ComplexInverseSquare))
        .unwrap();
    let exact = core::f64::consts::PI.powi(2) / 6.0;
    assert!((result.value - Complex::new(exact, exact)).norm() < 1e-7);
    assert!(result.error.tail_bound.is_some());
    assert!(result.report.termination.is_success());
}

#[test]
fn empty_vectors_skip_callbacks_and_nonfinite_terms_do_not_publish() {
    use core::sync::atomic::AtomicBool as Cell;
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::WynnEpsilon,
        0,
        SeriesOptions::default()
            .with_max_terms(budget(8))
            .with_order(budget(4))
            .with_tolerance(tolerance(1e-8)),
    )
    .unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let id = SolveIdentity::new(
        scioxide_core::ProblemToken::new(596),
        scioxide_core::algorithm_id!("integrate", "series-empty-test", 1),
        0,
        0,
    );
    let mut ctx = state
        .begin_solve(ExecutionContext::sequential(), id, &mut budgets)
        .unwrap();
    let mut empty = plan.state(0).unwrap();
    let mut empty_workspace = plan.workspace(0).unwrap();
    let called = Cell::new(false);
    let report = SeriesSummation::new(
        plan.method,
        plan.first,
        plan.options.clone().with_terms_per_advance(budget(8)),
    )
    .unwrap()
    .advance(
        SeriesProblem::Terms(&|_, _: &mut [f64]| {
            called.store(true, core::sync::atomic::Ordering::Relaxed);
            Ok::<(), SciOxideError>(())
        }),
        &mut empty,
        |accepted| {
            if let Some(values) = accepted.value() {
                [].copy_from_slice(values);
            }
            if let Some(accepted_errors) = accepted.errors() {
                [].copy_from_slice(accepted_errors);
            }
            Ok::<(), SciOxideError>(())
        },
        &mut empty_workspace,
        &mut ctx,
    )
    .unwrap();
    assert_eq!(report.termination, SeriesTermination::ExactEmpty);
    assert!(!called.load(core::sync::atomic::Ordering::Relaxed));
    let mut ws = plan.state(1).unwrap();
    let mut ws_workspace = plan.workspace(1).unwrap();
    let mut output = [77.0];
    let mut errors = [SeriesError::default()];
    assert!(
        SeriesSummation::new(
            plan.method,
            plan.first,
            plan.options.with_terms_per_advance(budget(1))
        )
        .unwrap()
        .advance(
            SeriesProblem::Terms(&|_, row: &mut [f64]| {
                row[0] = f64::NAN;
                Ok::<(), SciOxideError>(())
            }),
            &mut ws,
            |accepted| {
                if let Some(values) = accepted.value() {
                    output.copy_from_slice(values);
                }
                if let Some(accepted_errors) = accepted.errors() {
                    errors.copy_from_slice(accepted_errors);
                }
                Ok::<(), SciOxideError>(())
            },
            &mut ws_workspace,
            &mut ctx
        )
        .is_err()
    );
    assert_eq!(output, [77.0]);
    assert_eq!(ws.terms(), 0);
}

#[test]
fn levin_divided_differences_reproduce_its_model_on_any_points() {
    // A geometric remainder r^(n+1) / (1 - r) is omega_n * c / x_n with x_n = n + 1, so
    // Levin u of order two or more is exact on any distinct points.
    fn check<T: scioxide_core::NumericScalar>(points: &[usize], ratio: T) {
        use scioxide_core::AlgebraicScalar;
        let mut terms = [T::ZERO; 8];
        let mut partials = [T::ZERO; 8];
        let mut gaps = [double::Wide::ZERO; 8];
        for (j, &point) in points.iter().enumerate() {
            let mut power = T::ONE;
            let mut sum = T::ZERO;
            for coordinate in 1..=point {
                sum = sum + power;
                terms[j] = power;
                // Terms of the coordinates strictly between the rows are omitted.
                if j > 0 && coordinate > points[j - 1] && coordinate < point {
                    gaps[j] = gaps[j].add_value(power);
                }
                power = power * ratio;
            }
            partials[j] = sum;
        }
        let n = points.len();
        let mut magnitudes = [T::Real::ZERO; 8];
        let estimate = kernel::levin_u(
            &terms[..n],
            &gaps[..n],
            &partials[..n],
            points,
            &mut magnitudes,
        )
        .unwrap()
        .unwrap();
        let exact = T::ONE / (T::ONE - ratio);
        let tolerance = T::Real::try_from_f64_lossy(64.0).unwrap() * T::Real::EPSILON;
        assert!((estimate.value - exact).abs() <= tolerance * exact.abs());
        assert!(estimate.roundoff < tolerance * exact.abs());
    }
    for points in [&[1, 2, 3][..], &[2, 4, 7, 8], &[3, 5, 6, 9, 12, 13]] {
        check::<f64>(points, -0.6);
        check::<f64>(points, 0.5);
        check::<f32>(points, 0.25);
        check::<Complex<f64>>(points, Complex::new(0.3, 0.4));
    }
    let mut magnitudes = [0.0; 2];
    assert!(
        kernel::levin_u(&[1.0], &[double::Wide::ZERO], &[1.0], &[1], &mut magnitudes)
            .unwrap()
            .is_none()
    );
}

/// `InverseSquare` whose tail integral fails at one index, after that row's
/// term entered the exact sums.
struct FailingTail(usize);
impl SeriesTerms<f64> for FailingTail {
    fn term(&self, index: usize, output: &mut [f64]) -> Result<(), SciOxideError> {
        InverseSquare.term(index, output)
    }
}
impl EulerMaclaurinProvider<f64> for FailingTail {
    fn tail_integral(
        &self,
        at: usize,
        value: &mut [f64],
        error: &mut [f64],
    ) -> Result<(), SciOxideError> {
        if at == self.0 {
            return Err(SciOxideError::external("tail failed"));
        }
        InverseSquare.tail_integral(at, value, error)
    }
    fn odd_derivative(
        &self,
        at: usize,
        order: usize,
        value: &mut [f64],
        error: &mut [f64],
    ) -> Result<(), SciOxideError> {
        InverseSquare.odd_derivative(at, order, value, error)
    }
    fn even_derivative_l1_bound(
        &self,
        at: usize,
        order: usize,
        output: &mut [f64],
    ) -> Result<(), SciOxideError> {
        EulerMaclaurinProvider::<f64>::even_derivative_l1_bound(&InverseSquare, at, order, output)
    }
}

#[test]
fn failed_steps_withdraw_their_term_from_the_exact_sums() {
    let options = SeriesOptions::default()
        .with_max_terms(budget(16))
        .with_terms_per_advance(budget(16));
    // The second partial sum overflows after its term was added; retrying that row with
    // the opposite term must see the first prefix exactly, so the sum is zero. A large
    // absolute tolerance keeps the huge early changes comparable with it.
    let large = 0.75 * f64::MAX;
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::WynnEpsilon,
        0,
        options
            .clone()
            .with_order(budget(2))
            .with_tolerance(ErrorTolerance::scalar(1e300, 0.0, NormSelection::Infinity).unwrap()),
    )
    .unwrap();
    let mut state = plan.state(1).unwrap();
    let mut ws = plan.workspace(1).unwrap();
    let huge = |_: usize, row: &mut [f64]| {
        row[0] = large;
        Ok::<(), SciOxideError>(())
    };
    let context = || ExecutionContext::sequential_native();
    let overflow = plan.advance(
        SeriesProblem::Terms(&huge),
        &mut state,
        |_| Ok(()),
        &mut ws,
        context(),
    );
    assert!(overflow.is_err());
    assert_eq!(state.terms(), 1);
    let cancelling = |n: usize, row: &mut [f64]| {
        row[0] = match n {
            0 => large,
            1 => -large,
            _ => 0.0,
        };
        Ok::<(), SciOxideError>(())
    };
    plan.advance(
        SeriesProblem::Terms(&cancelling),
        &mut state,
        |_| Ok(()),
        &mut ws,
        context(),
    )
    .unwrap();
    assert_eq!(state.value().unwrap(), &[0.0]);
    // A failing Euler–Maclaurin provider after the term was added leaves the previous
    // prefix, and resuming matches an uninterrupted summation bit for bit.
    let plan = SeriesSummation::<f64>::new(
        SeriesMethod::EulerMaclaurin,
        1,
        options
            .with_order(budget(2))
            .with_tolerance(tolerance(1e-12)),
    )
    .unwrap();
    let mut state = plan.state(1).unwrap();
    let mut ws = plan.workspace(1).unwrap();
    let failed = plan.advance(
        SeriesProblem::EulerMaclaurin(&FailingTail(4)),
        &mut state,
        |_| Ok(()),
        &mut ws,
        context(),
    );
    assert!(failed.is_err());
    assert_eq!(state.terms(), 3);
    let resumed = plan
        .advance(
            SeriesProblem::EulerMaclaurin(&InverseSquare),
            &mut state,
            |_| Ok(()),
            &mut ws,
            context(),
        )
        .unwrap();
    let mut fresh = plan.state(1).unwrap();
    let direct = plan
        .advance(
            SeriesProblem::EulerMaclaurin(&InverseSquare),
            &mut fresh,
            |_| Ok(()),
            &mut ws,
            context(),
        )
        .unwrap();
    assert_eq!(resumed.termination, direct.termination);
    assert_eq!(state.terms(), fresh.terms());
    assert_eq!(
        state.value().unwrap()[0].to_bits(),
        fresh.value().unwrap()[0].to_bits()
    );
}
