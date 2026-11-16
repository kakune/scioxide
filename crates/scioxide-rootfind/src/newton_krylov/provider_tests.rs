//! Scoped injected-correction regression tests.

use super::*;
struct Quadratic;
impl ResidualOperator<f64> for Quadratic {
    fn input_dimension(&self) -> usize {
        1
    }
    fn output_dimension(&self) -> usize {
        1
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        y[0] = x[0] * x[0] - 2.;
        Ok(())
    }
}
impl JacobianOperator<f64> for Quadratic {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(
        &self,
        x: &[f64],
        v: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        y[0] = 2. * x[0] * v[0];
        Ok(())
    }
}
struct Direct(bool);
impl NonlinearLinearSolver<f64> for Direct {
    fn solve_into(
        &self,
        a: &crate::NonlinearLinearization<'_, f64>,
        b: &Vector<f64>,
        x: &mut Vector<f64>,
        _: ResidualTolerance<f64>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        let scope = scioxide_core::SolveScope::from(context);
        let a = a.operator();
        let mut state = scioxide_core::SolveState::new();
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "correction", 1),
            a.cols() as u128,
            0,
        );
        scope.run(&mut state, identity, SolveBudgets::unbounded(), |c| {
            if !c
                .record_provider(EvaluationRole::LinearOperator)?
                .is_granted()
            {
                return Err(invalid(
                    scioxide_core::argument!("c"),
                    InvalidInputReason::Inconsistent,
                    "test budget",
                ));
            }
            let mut j = [0.];
            a.apply_into_scoped(
                &[1.],
                &mut j,
                &mut [],
                &mut scioxide_core::OperatorContext::new(c),
            )
            .and_then(scioxide_core::Invocation::require)?;
            x.as_mut_slice()[0] = if self.0 { b.as_slice()[0] / j[0] } else { 0. };
            Ok(IterativeSolveReport::new(
                LinearSolveTermination::Converged,
                0,
                None,
                PartialResultTrust::DiagnosticOnly,
            ))
        })
    }
}
#[test]
fn injected_corrections_are_verified_and_match_native_solver() {
    let opts = NewtonKrylovOptions::default()
        .with_step_tolerance(StepTolerance::new(1e-15, 0.).unwrap())
        .with_residual_tolerance(ResidualTolerance::new(1e-12, 0.).unwrap());
    let problem = NewtonKrylovProblem::with_jacobian(&Quadratic);
    for provider in [None, Some(&Direct(true) as &dyn NonlinearLinearSolver<f64>)] {
        let p = provider.map_or(problem, |s| problem.with_linear_solver(s));
        let result = NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.solve(&p, &[1.]))
            .unwrap();
        assert!(result.is_success());
        assert!((result.point()[0] - 2_f64.sqrt()).abs() < 1e-12);
    }
    let result = NewtonKrylovSolver::new(opts)
        .and_then(|root_plan| root_plan.solve(&problem.with_linear_solver(&Direct(false)), &[1.]))
        .unwrap();
    assert!(matches!(
        result.termination(),
        RootTermination::LinearFailure(_)
    ));
    assert_eq!(result.point(), &[1.]);
}

struct ScaledQuadratic;
impl NonlinearScaling<f64> for ScaledQuadratic {
    fn scales_into(
        &self,
        _: &[f64],
        rows: &mut [f64],
        variables: &mut [f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        rows.fill(10.);
        variables.fill(100.);
        Ok(())
    }
}
struct Assembled;
impl NonlinearLinearSolver<f64> for Assembled {
    fn solve_into(
        &self,
        view: &NonlinearLinearization<'_, f64>,
        rhs: &Vector<f64>,
        out: &mut Vector<f64>,
        _: ResidualTolerance<f64>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        let scope = scioxide_core::SolveScope::from(context);
        let point = view.point()[0];
        assert_eq!(view.residual()[0], point * point - 2.);
        assert_eq!(view.row_divisors(), &[10.]);
        assert_eq!(view.variable_divisors(), &[100.]);
        let entry = 2. * point * view.variable_divisors()[0] / view.row_divisors()[0];
        scope.run(
            &mut scioxide_core::SolveState::new(),
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("test", "assembled-correction", 1),
                1,
                0,
            ),
            SolveBudgets::unbounded(),
            |cx| {
                let mut product = [0.];
                view.operator()
                    .apply_into_scoped(
                        &[1.],
                        &mut product,
                        &mut [0.],
                        &mut scioxide_core::OperatorContext::new(cx),
                    )?
                    .require()?;
                assert!((product[0] - entry).abs() <= f64::EPSILON * entry.abs());
                out.as_mut_slice()[0] = rhs.as_slice()[0] / entry;
                Ok(IterativeSolveReport::new(
                    LinearSolveTermination::Converged,
                    0,
                    None,
                    PartialResultTrust::DiagnosticOnly,
                ))
            },
        )
    }
}
#[test]
fn explicit_linearization_exposes_physical_values_and_frozen_units() {
    let problem = NewtonKrylovProblem::with_jacobian(&Quadratic)
        .with_scaling(&ScaledQuadratic)
        .with_linear_solver(&Assembled);
    let result = NewtonKrylovSolver::new(
        NewtonKrylovOptions::default()
            .with_residual_tolerance(ResidualTolerance::new(1e-12, 0.).unwrap()),
    )
    .unwrap()
    .solve(&problem, &[1.])
    .unwrap();
    assert!(result.is_success());
    assert!((result.point()[0] - 2_f64.sqrt()).abs() < 1e-10);
}
