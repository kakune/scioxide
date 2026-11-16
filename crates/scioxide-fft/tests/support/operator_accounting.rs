//! Shared evaluation admission contract for the three operator families.
#![expect(clippy::unwrap_used)]
use scioxide_core::{
    AdjointOperator, Budget, BudgetKind, Complex, ConvergenceReason, EvaluationRole,
    ExecutionContext, NumericScalar, OperatorContext, RealScalar, SciOxideError, SciOxideErrorView,
    SolveBudgets, SolveContext, SolveIdentity, SolveState,
};

pub fn admission<S: RealScalar>(
    output_len: usize,
    mut apply: impl FnMut(&mut [Complex<S>], &mut SolveContext<'_, '_>) -> Result<(), SciOxideError>,
) {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential_native(),
            identity(),
            &mut budgets,
        )
        .unwrap();
    let mut output = vec![
        Complex::new(
            S::try_from_f64_lossy(123.0).unwrap(),
            S::try_from_f64_lossy(456.0).unwrap()
        );
        output_len
    ];
    apply(&mut output, &mut context).unwrap();
    assert_eq!(
        context
            .counts()
            .nested()
            .provider(EvaluationRole::LinearOperator),
        1
    );
    assert_eq!(
        context
            .counts()
            .own()
            .provider(EvaluationRole::LinearOperator),
        0
    );
    let before = output.clone();
    let iterations = context.counts().nested().iterations();
    let error = apply(&mut output, &mut context).unwrap_err();
    assert!(
        matches!(error.view(), SciOxideErrorView::ConvergenceFailure {
        reason: ConvergenceReason::EvaluationLimit, diagnostics, ..
    } if diagnostics.budget().unwrap().consumed() == 1
        && diagnostics.counts().own().provider(EvaluationRole::LinearOperator) == 0)
    );
    assert_eq!(
        context
            .counts()
            .nested()
            .provider(EvaluationRole::LinearOperator),
        1
    );
    assert_eq!(context.counts().nested().iterations(), iterations);
    assert_eq!(output, before);
}

pub fn operators<S: RealScalar>(plan: &impl AdjointOperator<Complex<S>>, input: &[Complex<S>])
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut work = vec![Complex::new(S::ZERO, S::ZERO); plan.apply_workspace_len().unwrap()];
    admission(plan.rows(), |output, context| {
        plan.apply_into_scoped(input, output, &mut work, &mut OperatorContext::new(context))
            .and_then(scioxide_core::Invocation::require)
    });
    let input = vec![Complex::new(input.first().map_or(S::ZERO, |z| z.re), S::ZERO); plan.rows()];
    let mut work =
        vec![Complex::new(S::ZERO, S::ZERO); plan.apply_adjoint_workspace_len().unwrap()];
    admission(plan.cols(), |output, context| {
        plan.apply_adjoint_into_scoped(
            &input,
            output,
            &mut work,
            &mut OperatorContext::new(context),
        )
        .and_then(scioxide_core::Invocation::require)
    });
}

pub fn empty_axis(mut apply: impl FnMut(&mut SolveContext<'_, '_>) -> Result<(), SciOxideError>) {
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential_native(),
            identity(),
            &mut budgets,
        )
        .unwrap();
    for _ in 0..2 {
        apply(&mut context).unwrap();
    }
    assert_eq!(
        context
            .counts()
            .nested()
            .provider(EvaluationRole::LinearOperator),
        0
    );
    assert_eq!(context.counts().nested().iterations(), 0);
}

fn identity() -> SolveIdentity {
    SolveIdentity::new(
        scioxide_core::ProblemToken::new(1709),
        scioxide_core::algorithm_id!("fft", "operator-accounting", 1),
        0,
        0,
    )
}
