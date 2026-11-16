//! Domain recovery must preserve accepted state, accounting and reserved storage.

use core::cell::Cell;
use scioxide_core::{
    DomainReason, EvaluationRole, JacobianOperator, NumericScalar, RealScalar, ResidualOperator,
    SciOxideError,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    BacktrackingLineSearch, Broyden, BroydenOptions, BroydenUpdate, DenseJacobian,
    NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver, NewtonSystem,
    NewtonSystemOptions, RootTermination, SystemRootSolver, TrustRegionSystem,
    TrustRegionSystemOptions,
};

#[derive(Clone, Copy)]
enum Failure {
    Nan,
}
struct DomainSystem<S> {
    square_root: bool,
    blocked: bool,
    failure: Failure,
    calls: Cell<usize>,
    invalid: Cell<usize>,
    scalar: core::marker::PhantomData<S>,
}
impl<S> DomainSystem<S> {
    fn new(square_root: bool, blocked: bool, failure: Failure) -> Self {
        Self {
            square_root,
            blocked,
            failure,
            calls: Cell::new(0),
            invalid: Cell::new(0),
            scalar: core::marker::PhantomData,
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> ResidualOperator<S> for DomainSystem<S> {
    fn input_dimension(&self) -> usize {
        if self.square_root { 2 } else { 1 }
    }
    fn output_dimension(&self) -> usize {
        self.input_dimension()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[S], out: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        let four = S::TWO * S::TWO;
        if x.iter()
            .any(|&v| v <= S::ZERO || (self.blocked && v != four))
        {
            self.invalid.set(self.invalid.get() + 1);
            match self.failure {
                Failure::Nan => out.fill(S::NAN),
            }
        } else {
            for (f, &v) in out.iter_mut().zip(x) {
                *f = if self.square_root {
                    v.sqrt() - S::HALF
                } else {
                    v.ln() - S::HALF.ln()
                };
            }
        }
        Ok(())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> DenseJacobian<S> for DomainSystem<S> {
    fn jacobian_into(&self, x: &[S], out: &mut [S], _: &mut [S]) -> Result<bool, SciOxideError> {
        out.fill(S::ZERO);
        for (i, &v) in x.iter().enumerate() {
            out[i * x.len() + i] = if self.square_root {
                S::HALF / v.sqrt()
            } else {
                S::ONE / v
            };
        }
        Ok(true)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> JacobianOperator<S> for DomainSystem<S> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(&self, x: &[S], v: &[S], out: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        for ((f, &x), &v) in out.iter_mut().zip(x).zip(v) {
            *f = v * if self.square_root {
                S::HALF / x.sqrt()
            } else {
                S::ONE / x
            };
        }
        Ok(())
    }
}

#[test]
fn initial_and_undamped_residuals_remain_strict() {
    let f = DomainSystem::<f64>::new(false, false, Failure::Nan);
    let solver = NewtonSystem::new(
        NewtonSystemOptions::default().with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap();
    for x in [-1.0, 4.0] {
        let point = [x];
        assert_eq!(
            solver.solve(&f, &point).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }
    let solver = NewtonSystem::new(
        NewtonSystemOptions::default()
            .with_derivative_strategy(DerivativeStrategy::Analytic)
            .with_line_search(BacktrackingLineSearch::default()),
    )
    .unwrap();
    assert_eq!(
        solver.solve(&f, &[-1.0]).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
}

#[test]
fn derivative_failures_and_other_domain_reasons_are_not_recovered() {
    use scioxide_rootfind::ResidualFn;
    let f = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
        out[0] = x[0] - 1.0;
        Ok(())
    })
    .with_jacobian(|_: &[f64], out: &mut [f64]| {
        out[0] = f64::NAN;
        Ok(())
    });
    let solver = NewtonSystem::new(
        NewtonSystemOptions::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap();
    assert_eq!(
        solver.solve(&f, &[4.0]).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    for reason in [DomainReason::Pole, DomainReason::Undefined] {
        let f = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
            if x[0] != 4.0 {
                return Err(SciOxideError::domain_error(
                    scioxide_core::argument!("x"),
                    reason,
                    "unrecoverable domain reason",
                ));
            }
            out[0] = x[0] - 1.0;
            Ok(())
        })
        .with_jacobian(|_: &[f64], out: &mut [f64]| {
            out[0] = 1.0;
            Ok(())
        });
        assert_eq!(
            solver.solve(&f, &[4.0]).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DomainError
        );
    }
}

#[test]
fn evaluation_limits_during_domain_recovery_preserve_the_start() {
    use scioxide_core::{Budget, BudgetKind, SolveBudgets};
    let f = DomainSystem::<f64>::new(false, true, Failure::Nan);
    let budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(20).unwrap());
    macro_rules! check {
        ($solver:expr) => {{
            f.calls.set(0);
            f.invalid.set(0);
            let out = $solver.solve(&f, &[4.0]).unwrap();
            assert_eq!(
                out.termination(),
                RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
            );
            assert_eq!(out.point(), &[4.0]);
            assert_eq!(out.counts().total_providers().unwrap(), 20);
            assert_eq!(
                out.counts().provider(EvaluationRole::Residual),
                f.calls.get()
            );
            assert!(f.invalid.get() > 0);
        }};
    }
    check!(
        NewtonSystem::new(
            NewtonSystemOptions::default()
                .with_line_search(BacktrackingLineSearch::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic)
                .with_budgets(budgets)
        )
        .unwrap()
    );
    check!(
        Broyden::new(
            BroydenUpdate::First,
            BroydenOptions::default()
                .with_line_search(BacktrackingLineSearch::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic)
                .with_budgets(budgets)
        )
        .unwrap()
    );
    check!(
        TrustRegionSystem::new(
            TrustRegionSystemOptions::default()
                .with_derivative_strategy(DerivativeStrategy::Analytic)
                .with_budgets(budgets)
        )
        .unwrap()
    );
    f.calls.set(0);
    f.invalid.set(0);
    let problem = NewtonKrylovProblem::with_jacobian(&f);
    let out = NewtonKrylovSolver::new(NewtonKrylovOptions::default().with_budgets(budgets))
        .unwrap()
        .solve(&problem, &[4.0])
        .unwrap();
    assert_eq!(
        out.termination(),
        RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
    );
    assert_eq!(out.point(), &[4.0]);
    assert_eq!(
        out.accounting().total().unwrap().total_providers().unwrap(),
        20
    );
    assert_eq!(
        out.accounting().own().provider(EvaluationRole::Residual),
        f.calls.get()
    );
    assert!(f.invalid.get() > 0);
}
