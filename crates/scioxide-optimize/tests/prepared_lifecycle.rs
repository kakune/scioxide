//! Prepared-solver lifecycle contracts shared by every migrated local solver:
//! success, preflight rejection with unchanged output, budget stops with an
//! accepted state, and allocation-free prepared calls.

#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{
    AttemptOutcome, Bounds, Budget, ExecutionContext, Interval, PartialResultTrust, SciOxideError,
    SciOxideErrorKind,
};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    Bfgs, BfgsOptions, BoundedLinearLeastSquares, BoundedObjective, GradientDescent,
    GradientDescentOptions, Lbfgs, LbfgsB, LbfgsBOptions, LbfgsOptions, LeastSquares,
    LeastSquaresGlobalization, LeastSquaresOptions, LeastSquaresProblem, LeastSquaresResult,
    LeastSquaresWorkspaceRequest, LinearLeastSquaresOptions, LinearLeastSquaresProblem,
    LinearLeastSquaresResult, NelderMead, NelderMeadOptions, NewtonOptimizer, NewtonOptions, Nnls,
    NonlinearCg, NonlinearCgOptions, ObjectiveFunction, OptimizeDiagnostics, OptimizeResult,
    OptimizeTermination, Optimizer, Powell, PowellOptions, ScalarMinimizer, ScalarMinimumMethod,
    ScalarMinimumOptions, ScalarMinimumStart, TrustRegionOptimizer, TrustRegionOptions,
    TrustRegionSubproblem, TrustRegionWorkspaceRequest,
};

/// `(x - 1)² + 4 (y + 2)²` with analytic derivatives, minimized at `(1, -2)`.
struct Bowl {
    hessian_products: bool,
}

impl ObjectiveFunction<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((x[0] - 1.0).powi(2) + 4.0 * (x[1] + 2.0).powi(2))
    }
    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output[0] = 2.0 * (x[0] - 1.0);
        output[1] = 8.0 * (x[1] + 2.0);
        Ok(true)
    }
    fn hessian_into(&self, _: &[f64], output: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        output.as_mut_slice().copy_from_slice(&[2.0, 0.0, 0.0, 8.0]);
        Ok(true)
    }
    fn has_hessian_vector_product(&self) -> bool {
        self.hessian_products
    }
    fn hessian_vector_product_into(
        &self,
        _: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), SciOxideError> {
        output[0] = 2.0 * direction[0];
        output[1] = 8.0 * direction[1];
        Ok(())
    }
}

const BOWL: Bowl = Bowl {
    hessian_products: false,
};
const START: [f64; 2] = [3.0, 1.0];

fn assert_unchanged(before: &OptimizeResult<f64>, after: &OptimizeResult<f64>) {
    assert_eq!(before.x(), after.x());
    assert_eq!(before.fun(), after.fun());
    assert_eq!(before.termination(), after.termination());
    assert_eq!(before.counts(), after.counts());
}

/// Exercise one local minimizer's prepared lifecycle.
macro_rules! local_lifecycle {
    (
        $test:ident,
        $solver:expr,
        $limited:expr,
        $stop:expr,
        $request:expr,
        $tolerance:expr
    ) => {
        #[test]
        fn $test() {
            let solver = $solver;
            let mut workspace = solver.workspace($request).unwrap();
            let mut output = OptimizeResult::<f64>::empty();
            output.try_reserve(2).unwrap();
            let sequential = ExecutionContext::sequential;

            // Success with prepared storage does not allocate, including the first call.
            for _ in 0..2 {
                solver
                    .minimize_into(&BOWL, &START, &mut output, &mut workspace, sequential())
                    .unwrap();
                assert!(output.is_success(), "{output:?}");
                assert!((output.x()[0] - 1.0).abs() < $tolerance, "{output:?}");
                assert!((output.x()[1] + 2.0).abs() < $tolerance, "{output:?}");
                let attempt = workspace.last_attempt().unwrap();
                assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
                assert!(attempt.epoch().is_some());
                let accepted = workspace.last_accepted().unwrap();
                assert_eq!(accepted.point(), output.x());
                assert_eq!(accepted.progress().value(), output.fun());
                assert_eq!(
                    accepted.progress().trust(),
                    PartialResultTrust::ResidualChecked
                );
            }

            // Preflight rejection changes neither the output nor any accepted state.
            let before = output.clone();
            let error = solver
                .minimize_into(&BOWL, &[1.0], &mut output, &mut workspace, sequential())
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
            assert!(attempt.epoch().is_none());
            assert!(workspace.last_accepted().is_none());
            assert_unchanged(&before, &output);

            // A budget stop publishes the accepted state as an unsuccessful result.
            let limited = $limited;
            let mut workspace = limited.workspace($request).unwrap();
            limited
                .minimize_into(&BOWL, &START, &mut output, &mut workspace, sequential())
                .unwrap();
            assert!(!output.is_success());
            assert_eq!(output.termination(), $stop);
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::Stopped
            );
            let accepted = workspace.last_accepted().unwrap();
            assert_eq!(accepted.point(), output.x());
            assert_eq!(accepted.progress().trust(), PartialResultTrust::Admissible);
            assert!(output.fun() < BOWL.eval(&START).unwrap());
        }
    };
}

local_lifecycle!(
    bfgs,
    Bfgs::<f64>::default(),
    Bfgs::new(BfgsOptions::default().with_max_iterations(Budget::ONE)),
    OptimizeTermination::IterationLimit,
    2,
    1e-6
);
local_lifecycle!(
    lbfgs,
    Lbfgs::<f64>::default(),
    Lbfgs::new(LbfgsOptions::default().with_max_iterations(Budget::ONE)),
    OptimizeTermination::IterationLimit,
    2,
    1e-6
);
local_lifecycle!(
    gradient_descent,
    GradientDescent::<f64>::new(
        GradientDescentOptions::default().with_max_iterations(Budget::try_from(10_000).unwrap())
    ),
    GradientDescent::new(GradientDescentOptions::default().with_max_iterations(Budget::ONE)),
    OptimizeTermination::IterationLimit,
    2,
    1e-4
);
local_lifecycle!(
    nonlinear_cg,
    NonlinearCg::<f64>::default(),
    NonlinearCg::new(NonlinearCgOptions::default().with_max_iterations(Budget::ONE)),
    OptimizeTermination::IterationLimit,
    2,
    1e-6
);
local_lifecycle!(
    newton,
    NewtonOptimizer::<f64>::default(),
    NewtonOptimizer::new(NewtonOptions::default().with_max_iterations(Budget::ONE)),
    OptimizeTermination::IterationLimit,
    2,
    1e-8
);
local_lifecycle!(
    trust_region_dogleg,
    TrustRegionOptimizer::<f64>::default(),
    TrustRegionOptimizer::new(TrustRegionOptions::default().with_max_iterations(Budget::ONE)),
    OptimizeTermination::IterationLimit,
    TrustRegionWorkspaceRequest::of(&BOWL),
    1e-8
);
local_lifecycle!(
    trust_region_steihaug,
    TrustRegionOptimizer::<f64>::new(TrustRegionOptions::default().with_subproblem(steihaug())),
    TrustRegionOptimizer::new(
        TrustRegionOptions::default()
            .with_subproblem(steihaug())
            .with_max_iterations(Budget::ONE)
    ),
    OptimizeTermination::IterationLimit,
    TrustRegionWorkspaceRequest::of(&BOWL),
    1e-8
);
local_lifecycle!(
    nelder_mead,
    NelderMead::<f64>::default(),
    NelderMead::new(NelderMeadOptions::default().with_max_iterations(Budget::ONE)).unwrap(),
    OptimizeTermination::IterationLimit,
    2,
    1e-3
);
local_lifecycle!(
    powell,
    Powell::<f64>::default(),
    Powell::new(PowellOptions::default().with_max_evaluations(Budget::try_from(4).unwrap()))
        .unwrap(),
    OptimizeTermination::EvaluationLimit,
    2,
    1e-6
);

fn steihaug() -> TrustRegionSubproblem {
    TrustRegionSubproblem::SteihaugCg {
        max_inner_iterations: Budget::try_from(10).unwrap(),
        second_order_stopping: false,
    }
}

#[test]
fn steihaug_matrix_free_storage_holds_no_dense_hessian() {
    let solver =
        TrustRegionOptimizer::<f64>::new(TrustRegionOptions::default().with_subproblem(steihaug()));
    let objective = Bowl {
        hessian_products: true,
    };
    let matrix_free = solver
        .workspace_requirements(TrustRegionWorkspaceRequest::of(&objective))
        .unwrap();
    assert!(!matrix_free.dense());
    assert!(
        solver
            .workspace_requirements(TrustRegionWorkspaceRequest::dense(2))
            .unwrap()
            .dense()
    );
    let mut workspace = solver
        .workspace(TrustRegionWorkspaceRequest::of(&objective))
        .unwrap();
    let mut output = OptimizeResult::empty();
    output.try_reserve(2).unwrap();
    solver
        .minimize_into(
            &objective,
            &START,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(output.is_success());
    assert!((output.x()[1] + 2.0).abs() < 1e-8);
    assert!(matches!(
        output.method_diagnostics(),
        OptimizeDiagnostics::Newton(diagnostics) if diagnostics.cg_iterations() > 0
    ));
    let dogleg = TrustRegionOptimizer::<f64>::default();
    assert!(
        dogleg
            .workspace_requirements(TrustRegionWorkspaceRequest::of(&objective))
            .unwrap()
            .dense()
    );
}

#[test]
fn bounded_problems_require_a_bound_aware_method() {
    let bounds = Bounds::new(vec![(Some(-1.0), Some(2.0)), (Some(-1.0), Some(2.0))]).unwrap();
    let problem = BoundedObjective::new(&BOWL, &bounds).unwrap();
    for error in [
        Bfgs::default().minimize(&problem, &[0.0, 0.0]),
        NewtonOptimizer::default().minimize(&problem, &[0.0, 0.0]),
        NelderMead::default().minimize(&problem, &[0.0, 0.0]),
    ] {
        assert_eq!(error.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
    }
    let powell = Powell::default().minimize(&problem, &[0.0, 0.0]).unwrap();
    let lbfgsb = LbfgsB::default().minimize(&problem, &[0.0, 0.0]).unwrap();
    for result in [powell, lbfgsb] {
        assert!(result.is_success(), "{result:?}");
        assert!((result.x()[0] - 1.0).abs() < 1e-6);
        assert!((result.x()[1] + 1.0).abs() < 1e-6);
    }
}

#[test]
fn lbfgsb_prepared_lifecycle() {
    let bounds = Bounds::new(vec![(Some(-5.0), Some(5.0)), (Some(-1.0), Some(5.0))]).unwrap();
    let problem = BoundedObjective::new(&BOWL, &bounds).unwrap();
    let solver = LbfgsB::<f64>::default();
    let mut workspace = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    output.try_reserve(2).unwrap();
    let sequential = ExecutionContext::sequential;
    solver
        .minimize_into(&problem, &START, &mut output, &mut workspace, sequential())
        .unwrap();
    assert!(output.is_success());
    assert!((output.x()[1] + 1.0).abs() < 1e-8);
    assert_eq!(output.bound_violation(), Some(0.0));
    let before = output.clone();
    assert!(
        solver
            .minimize_into(
                &problem,
                &[9.0, 0.0],
                &mut output,
                &mut workspace,
                sequential()
            )
            .is_err()
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_unchanged(&before, &output);
    let limited = LbfgsB::new(LbfgsBOptions::default().with_max_iterations(Budget::ONE));
    limited
        .minimize_into(&problem, &START, &mut output, &mut workspace, sequential())
        .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(workspace.last_accepted().unwrap().point(), output.x());
}

#[test]
fn every_local_minimizer_shares_one_runtime_interface() {
    let solvers: Vec<(Box<dyn Optimizer<f64>>, &str)> = vec![
        (Box::new(Bfgs::default()), "first-order"),
        (Box::new(Lbfgs::default()), "first-order"),
        (
            Box::new(GradientDescent::new(
                GradientDescentOptions::default()
                    .with_max_iterations(Budget::try_from(10_000).unwrap()),
            )),
            "first-order",
        ),
        (Box::new(NonlinearCg::default()), "first-order"),
        (Box::new(NewtonOptimizer::default()), "newton"),
        (Box::new(TrustRegionOptimizer::default()), "newton"),
        (
            Box::new(TrustRegionOptimizer::new(
                TrustRegionOptions::default().with_subproblem(steihaug()),
            )),
            "newton",
        ),
        (Box::new(NelderMead::default()), "nelder-mead"),
        (Box::new(Powell::default()), "powell"),
    ];
    for (solver, family) in &solvers {
        let result = solver.minimize(&BOWL, &START).unwrap();
        assert!(result.is_success(), "{family}: {result:?}");
        assert!(result.algorithm().is_some());
        let observed = match result.method_diagnostics() {
            OptimizeDiagnostics::FirstOrder(_) => "first-order",
            OptimizeDiagnostics::Newton(_) => "newton",
            OptimizeDiagnostics::NelderMead(_) => "nelder-mead",
            OptimizeDiagnostics::Powell(_) => "powell",
            other => panic!("unexpected diagnostics {other:?}"),
        };
        assert_eq!(&observed, family);
    }
}

#[test]
fn owned_forms_require_a_clock_for_time_limits() {
    let limit = core::time::Duration::from_secs(1);
    let solvers: [Box<dyn Optimizer<f64>>; 3] = [
        Box::new(Bfgs::new(BfgsOptions::default().with_time_limit(limit))),
        Box::new(TrustRegionOptimizer::new(
            TrustRegionOptions::default().with_time_limit(limit),
        )),
        Box::new(NelderMead::new(NelderMeadOptions::default().with_time_limit(limit)).unwrap()),
    ];
    for solver in &solvers {
        assert_eq!(
            solver.minimize(&BOWL, &START).unwrap_err().kind(),
            SciOxideErrorKind::CapabilityUnavailable
        );
    }
}

#[test]
fn scalar_scoped_lifecycle() {
    let solver = ScalarMinimizer::<f64>::default();
    let objective = |x: f64| Ok((x - 0.25).powi(2));
    let start = ScalarMinimumStart::Bounded(Interval::new(-1.0, 2.0).unwrap());
    let result = solver
        .minimize_scoped(&objective, start, ExecutionContext::sequential())
        .unwrap();
    assert!(result.is_success());
    assert!((result.x() - 0.25).abs() < 1e-6);
    assert_eq!(
        solver
            .minimize(&objective, ScalarMinimumStart::Seeds(1.0, 1.0))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    let limited = ScalarMinimizer::new(
        ScalarMinimumMethod::Brent,
        ScalarMinimumOptions::default().with_max_evaluations(Budget::try_from(3).unwrap()),
    );
    let stopped = limited.minimize(&objective, start).unwrap();
    assert_eq!(stopped.termination(), OptimizeTermination::EvaluationLimit);
    assert_eq!(stopped.fun_evals(), 3);
    assert!(!stopped.is_success());
}

/// Residual `(x - 1, 2 (y + 2), x y)` with an analytic Jacobian.
struct Curve;

impl LeastSquaresProblem<f64> for Curve {
    fn param_dim(&self) -> usize {
        2
    }
    fn residual_dim(&self) -> usize {
        3
    }
    fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
        output.copy_from_slice(&[x[0] - 1.0, 2.0 * (x[1] + 2.0), 0.1 * (x[0] - 1.0) * x[1]]);
        Ok(())
    }
    fn jacobian_into(&self, x: &[f64], output: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        output.as_mut_slice().copy_from_slice(&[
            1.0,
            0.0,
            0.0,
            2.0,
            0.1 * x[1],
            0.1 * (x[0] - 1.0),
        ]);
        Ok(true)
    }
}

#[test]
fn least_squares_prepared_lifecycle() {
    for globalization in [
        LeastSquaresGlobalization::gauss_newton(),
        LeastSquaresGlobalization::levenberg_marquardt(),
    ] {
        let options = LeastSquaresOptions::default().with_globalization(globalization);
        let solver = LeastSquares::new(options).unwrap();
        let mut workspace = solver
            .workspace(LeastSquaresWorkspaceRequest::of(&Curve).unwrap())
            .unwrap();
        let mut output = LeastSquaresResult::empty();
        output.try_reserve(2, 3).unwrap();
        let sequential = ExecutionContext::sequential;
        solver
            .solve_into(&Curve, &START, &mut output, &mut workspace, sequential())
            .unwrap();
        assert!(output.is_success(), "{output:?}");
        assert!((output.x()[0] - 1.0).abs() < 1e-8 && (output.x()[1] + 2.0).abs() < 1e-8);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Succeeded
        );
        let accepted = workspace.last_accepted().unwrap();
        assert_eq!(accepted.parameters(), output.x());
        assert_eq!(accepted.residual(), output.residual());

        let before = output.clone();
        assert_eq!(
            solver
                .solve_into(&Curve, &[1.0], &mut output, &mut workspace, sequential())
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
        assert!(workspace.last_accepted().is_none());
        assert_eq!(before.x(), output.x());
        assert_eq!(before.termination(), output.termination());

        let limited = LeastSquares::new(options.with_max_iterations(Budget::ONE)).unwrap();
        limited
            .solve_into(
                &Curve,
                &[30.0, 40.0],
                &mut output,
                &mut workspace,
                sequential(),
            )
            .unwrap();
        assert_eq!(output.termination(), OptimizeTermination::IterationLimit);
        assert!(!output.is_success());
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Stopped
        );
        assert_eq!(workspace.last_accepted().unwrap().parameters(), output.x());
    }
}

#[test]
fn linear_least_squares_prepared_lifecycle() {
    let a = Matrix::from_vec(3, 2, vec![1., 0., 0., 1., 1., 1.]).unwrap();
    let b = [1., 2., 3.];
    let problem = LinearLeastSquaresProblem::from_dense(&a, &b).unwrap();
    let solver = Nnls::<f64>::default();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut output = LinearLeastSquaresResult::empty();
    output.try_reserve(3, 2).unwrap();
    let sequential = ExecutionContext::sequential;
    solver
        .solve_into(&problem, &mut output, &mut workspace, sequential())
        .unwrap();
    assert!(output.is_success());
    assert!((output.x()[0] - 1.0).abs() < 1e-10 && (output.x()[1] - 2.0).abs() < 1e-10);
    assert_eq!(output.algorithm().unwrap().method(), "Nnls");
    assert_eq!(output.trust(), PartialResultTrust::ResidualChecked);
    let accepted = workspace.last_accepted().unwrap();
    assert_eq!(accepted.point(), output.x());
    assert_eq!(accepted.progress().value(), output.objective());

    let other = Matrix::from_vec(2, 2, vec![1., 0., 0., 1.]).unwrap();
    let mismatched = LinearLeastSquaresProblem::from_dense(&other, &b[..2]).unwrap();
    let before = output.x().to_vec();
    assert!(
        solver
            .solve_into(&mismatched, &mut output, &mut workspace, sequential())
            .is_err()
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    assert_eq!(output.x(), before);

    let bounds = Bounds::new(vec![(Some(0.0), Some(10.0)); 2]).unwrap();
    // Clipping the unconstrained (-1, 2) leaves a free-coordinate resolve,
    // so one BVLS iteration genuinely exhausts the configured allowance.
    let bounded_rhs = [-1., 2., 1.];
    let problem = LinearLeastSquaresProblem::from_dense(&a, &bounded_rhs).unwrap();
    let bounded_problem = problem.with_bounds(&bounds).unwrap();
    let limited = BoundedLinearLeastSquares::<f64>::new(
        LinearLeastSquaresOptions::default().with_max_iterations(Budget::ONE),
    );
    let mut workspace = limited.workspace(&bounded_problem).unwrap();
    limited
        .solve_into(&bounded_problem, &mut output, &mut workspace, sequential())
        .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    assert_eq!(workspace.last_accepted().unwrap().point(), output.x());
}

#[test]
fn closure_adapters_execute_in_prepared_solves() {
    use scioxide_optimize::{
        ConstrainedProblem, ConstraintFn, Constraints, LeastSquaresFn, ObjectiveFn, Sqp,
    };
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2))
    })
    .with_gradient(|x, g| {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] + 2.0);
        Ok(())
    });
    let mut output = OptimizeResult::empty();
    output.try_reserve(2).unwrap();
    let bfgs = Bfgs::default();
    let mut workspace = bfgs.workspace(2).unwrap();
    bfgs.minimize_into(
        &objective,
        &[0.0, 0.0],
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    output.require_success().unwrap();
    let bounds = Bounds::new(vec![(Some(-3.0), Some(3.0)); 2]).unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let lbfgsb = LbfgsB::default();
    let mut workspace = lbfgsb.workspace(2).unwrap();
    lbfgsb
        .minimize_into(
            &problem,
            &[0.0, 0.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    output.require_success().unwrap();
    let constraints = Constraints::new(2)
        .unwrap()
        .equality(
            ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] + 1.0)).with_jacobian(|_, g| {
                g.fill(1.0);
                Ok(())
            }),
        )
        .unwrap();
    let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
    let sqp = Sqp::default();
    let mut workspace = sqp.workspace(&problem).unwrap();
    sqp.minimize_into(
        &problem,
        &[0.0, 0.0],
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    output.require_success().unwrap();
    let problem = LeastSquaresFn::new(2, 2, |x: &[f64], r: &mut [f64]| {
        r[0] = x[0] - 1.0;
        r[1] = x[1] + 2.0;
        Ok(())
    })
    .unwrap()
    .with_jacobian(|_, j| {
        j.as_mut_slice().copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    });
    let solver = LeastSquares::default();
    let mut workspace = solver
        .workspace(LeastSquaresWorkspaceRequest::of(&problem).unwrap())
        .unwrap();
    let mut output = LeastSquaresResult::empty();
    output.try_reserve(2, 2).unwrap();
    solver
        .solve_into(
            &problem,
            &[0.0, 0.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    output.require_success().unwrap();
}
