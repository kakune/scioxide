//! Independent rectangular and deficient linear-fit regression cases.
#![allow(clippy::unwrap_used)]
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    LeastSquares, LeastSquaresGlobalization, LeastSquaresOptions, LeastSquaresProblem,
    LeastSquaresWorkspaceRequest,
};

struct Fit<S: RealScalar + NumericScalar<Real = S>> {
    a: Matrix<S>,
    b: Vec<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> LeastSquaresProblem<S> for Fit<S> {
    fn param_dim(&self) -> usize {
        self.a.cols()
    }
    fn residual_dim(&self) -> usize {
        self.a.rows()
    }
    fn residual_into(&self, x: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        for (row, value) in output.iter_mut().enumerate() {
            let mut sum = scioxide_core::RangeAccumulator::new();
            sum.add_product(-self.b[row], <S as scioxide_core::AlgebraicScalar>::ONE)?;
            for (column, &x) in x.iter().enumerate() {
                sum.add_product(self.a.as_slice()[row * self.a.cols() + column], x)?;
            }
            *value = sum.finish()?;
        }
        Ok(())
    }
    fn jacobian_into(&self, _: &[S], output: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        output.as_mut_slice().copy_from_slice(self.a.as_slice());
        Ok(true)
    }
}
fn deficient<S: RealScalar + NumericScalar<Real = S>>() {
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    for rows in [1, 3] {
        let mut a = Matrix::zeros(rows, 2).unwrap();
        a.as_mut_slice().fill(one);
        let fit = Fit {
            a,
            b: vec![one + one; rows],
        };
        let gn = gn().solve(&fit, &[zero, zero]).unwrap();
        let lm = lm().solve(&fit, &[zero, zero]).unwrap();
        for result in [gn, lm] {
            assert!(result.is_success(), "{:?}", result);
            for &x in result.x() {
                assert!(NumericScalar::abs(x - one) < S::SQRT_EPSILON);
            }
        }
    }
}
#[test]
fn rank_deficient_and_wide_f64() {
    deficient::<f64>();
}
#[test]
fn rank_deficient_and_wide_f32() {
    deficient::<f32>();
}

struct Products<'a, S: RealScalar + NumericScalar<Real = S>>(&'a Fit<S>);
impl<S: RealScalar + NumericScalar<Real = S>> LeastSquaresProblem<S> for Products<'_, S> {
    fn param_dim(&self) -> usize {
        self.0.param_dim()
    }
    fn residual_dim(&self) -> usize {
        self.0.residual_dim()
    }
    fn residual_into(&self, x: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        self.0.residual_into(x, output)
    }
    fn jacobian_operator_workspace_len(&self) -> Result<Option<usize>, SciOxideError> {
        Ok(Some(0))
    }
    fn jacobian_product_into(
        &self,
        _: &[S],
        input: &[S],
        output: &mut [S],
        _: &mut [S],
    ) -> Result<(), SciOxideError> {
        for (row, value) in output.iter_mut().enumerate() {
            let mut sum = scioxide_core::RangeAccumulator::new();
            for (column, &x) in input.iter().enumerate() {
                sum.add_product(self.0.a.as_slice()[row * input.len() + column], x)?;
            }
            *value = sum.finish()?;
        }
        Ok(())
    }
    fn jacobian_adjoint_product_into(
        &self,
        _: &[S],
        input: &[S],
        output: &mut [S],
        _: &mut [S],
    ) -> Result<(), SciOxideError> {
        let n = output.len();
        for (column, value) in output.iter_mut().enumerate() {
            let mut sum = scioxide_core::RangeAccumulator::new();
            for (row, &x) in input.iter().enumerate() {
                sum.add_product(self.0.a.as_slice()[row * n + column], x)?;
            }
            *value = sum.finish()?;
        }
        Ok(())
    }
}
fn operator_equivalence<S: RealScalar + NumericScalar<Real = S>>() {
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    for rows in [1, 3] {
        let mut a = Matrix::zeros(rows, 2).unwrap();
        a.as_mut_slice().fill(one);
        let fit = Fit {
            a,
            b: vec![one + one; rows],
        };
        for result in [
            gn().solve(&Products(&fit), &[zero, zero]).unwrap(),
            lm().solve(&Products(&fit), &[zero, zero]).unwrap(),
        ] {
            assert!(result.is_success(), "{result:?}");
            assert!(result.rank().is_none());
            for &x in result.x() {
                assert!(NumericScalar::abs(x - one) < S::SQRT_EPSILON * (one + one));
            }
        }
    }
}
#[test]
fn operator_equivalence_f64() {
    operator_equivalence::<f64>();
}
#[test]
fn operator_equivalence_f32() {
    operator_equivalence::<f32>();
}

#[test]
fn overflowing_cost_and_gradient_do_not_prevent_a_finite_solution() {
    let fit = Fit {
        a: Matrix::from_vec(2, 1, vec![1e200, 1e200]).unwrap(),
        b: vec![1e200, 1e200],
    };
    let result = gn().solve(&fit, &[0.0]).unwrap();
    assert!((result.x()[0] - 1.0_f64).abs() < 1e-14);
    let relative = result
        .scaled_cost()
        .div(scioxide_core::Scaled::from_value(1e200).unwrap())
        .unwrap()
        .div(scioxide_core::Scaled::from_value(1e200).unwrap())
        .unwrap()
        .round()
        .unwrap();
    assert!(relative < 1e-26);
    let stopped = LeastSquares::new(
        LeastSquaresOptions::default()
            .with_globalization(LeastSquaresGlobalization::gauss_newton())
            .with_max_iterations(scioxide_core::Budget::ONE),
    )
    .unwrap()
    .solve(
        &Fit {
            a: Matrix::from_vec(2, 1, vec![0.0, 0.0]).unwrap(),
            b: vec![1e200, 1e200],
        },
        &[0.0],
    )
    .unwrap();
    assert!(stopped.is_success());
    assert!(stopped.cost().is_err());
    assert!(!stopped.scaled_cost().is_zero());
}

#[test]
fn robust_losses_reduce_outlier_bias() {
    use scioxide_optimize::LeastSquaresLoss;
    let fit = Fit {
        a: Matrix::from_vec(5, 1, vec![1.0; 5]).unwrap(),
        b: vec![1.0, 1.0, 1.0, 1.0, 100.0],
    };
    for loss in [
        LeastSquaresLoss::Huber,
        LeastSquaresLoss::SoftL1,
        LeastSquaresLoss::Cauchy,
    ] {
        let options = LeastSquaresOptions::default().with_loss(loss, 1.0);
        let result = LeastSquares::new(options)
            .unwrap()
            .solve(&fit, &[0.0])
            .unwrap();
        assert!(result.is_success(), "{loss:?}: {result:?}");
        assert!((result.x()[0] - 1.0_f64).abs() < 0.3);
    }
}

#[test]
fn complex_non_holomorphic_fit_uses_real_derivatives() {
    use scioxide_core::Complex;
    use scioxide_optimize::ComplexLeastSquares;
    let mut x = [Complex::new(0.0, 0.0)];
    let mut residual = [Complex::new(0.0, 0.0); 2];
    let callback = |x: &[Complex<f64>], output: &mut [Complex<f64>]| {
        output.fill(Complex::new(x[0].re - 2.0, -x[0].im - 3.0));
        Ok(())
    };
    let coefficients = [1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0];
    let jacobian = |_: &[Complex<f64>], output: &mut Matrix<f64>| {
        output.as_mut_slice().copy_from_slice(&coefficients);
        Ok(())
    };
    let fit = ComplexLeastSquares::new(callback, &mut x, &mut residual)
        .unwrap()
        .with_jacobian(&jacobian);
    let result = gn().solve(&fit, &[0.0, 0.0]).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.x()[0] - 2.0).abs() < 1e-7);
    assert!((result.x()[1] + 3.0).abs() < 1e-7);
}

#[test]
fn poor_initial_damping_recovers_without_false_success() {
    let fit = Fit {
        a: Matrix::from_vec(2, 1, vec![1.0, 1.0]).unwrap(),
        b: vec![2.0, 2.0],
    };
    for damping in [1e-300, 1e8, 1e300] {
        let options = LeastSquaresOptions::default().with_globalization(
            LeastSquaresGlobalization::LevenbergMarquardt {
                initial_damping: damping,
            },
        );
        let result = LeastSquares::new(options)
            .unwrap()
            .solve(&fit, &[0.0])
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!((result.x()[0] - 2.0_f64).abs() < 1e-7);
    }
}

#[test]
fn operator_child_obeys_exact_ancestor_evaluation_budgets() {
    use scioxide_core::{
        Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveState,
    };
    let fit = Fit {
        a: Matrix::from_vec(2, 1, vec![1.0, 2.0]).unwrap(),
        b: vec![1.0, 2.0],
    };
    let problem = Products(&fit);
    let solver = gn::<f64>();
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(91),
        scioxide_core::algorithm_id!("test", "Parent", 1),
        1,
        0,
    );
    let mut workspace = solver
        .workspace(LeastSquaresWorkspaceRequest::of(&problem).unwrap())
        .unwrap();
    for limit in 1..20 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(
            BudgetKind::Evaluation,
            Budget::ONE.saturating_add(limit - 1),
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let mut result = scioxide_optimize::LeastSquaresResult::empty();
        solver
            .solve_into(&problem, &[0.0], &mut result, &mut workspace, &mut context)
            .unwrap();
        assert_eq!(result.counts(), workspace.last_attempt().unwrap().counts());
        assert_eq!(context.counts().nested(), result.counts().total().unwrap());
        assert!(result.x()[0].is_finite());
        if !result.is_success() {
            assert_eq!(
                result.termination(),
                scioxide_optimize::OptimizeTermination::EvaluationLimit
            );
        }
        let mut residual = [0.0; 2];
        problem.residual_into(result.x(), &mut residual).unwrap();
        assert_eq!(result.residual(), residual);
    }
}

#[test]
fn weights_and_scales_validate_and_zero_weight_excludes_an_observation() {
    struct Weighted {
        weight: f64,
    }
    impl LeastSquaresProblem<f64> for Weighted {
        fn param_dim(&self) -> usize {
            1
        }
        fn residual_dim(&self) -> usize {
            2
        }
        fn parameter_scale(&self, _: usize) -> Option<f64> {
            Some(1000.0)
        }
        fn residual_scale(&self, _: usize) -> f64 {
            0.01
        }
        fn residual_weight(&self, i: usize) -> f64 {
            if i == 1 { self.weight } else { 1.0 }
        }
        fn residual_into(&self, x: &[f64], r: &mut [f64]) -> Result<(), SciOxideError> {
            r.copy_from_slice(&[x[0] - 3.0, x[0] - 1000.0]);
            Ok(())
        }
        fn jacobian_into(&self, _: &[f64], j: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
            j.as_mut_slice().fill(1.0);
            Ok(true)
        }
    }
    let result = gn().solve(&Weighted { weight: 0.0 }, &[0.0]).unwrap();
    assert!(result.is_success());
    assert!((result.x()[0] - 3.0).abs() < 1e-10);
    for weight in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            (gn().solve(&Weighted { weight }, &[0.0]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));
    }
}

#[test]
fn dense_owned_borrowed_and_sparse_operators_agree() {
    use core::cell::RefCell;
    use scioxide_core::AdjointOperator;
    struct Borrowed<'a> {
        operator: &'a dyn AdjointOperator<f64>,
        scratch: RefCell<Vec<f64>>,
    }
    impl LeastSquaresProblem<f64> for Borrowed<'_> {
        fn param_dim(&self) -> usize {
            self.operator.cols()
        }
        fn residual_dim(&self) -> usize {
            self.operator.rows()
        }
        fn residual_into(&self, x: &[f64], r: &mut [f64]) -> Result<(), SciOxideError> {
            self.operator
                .apply_into(x, r, &mut self.scratch.borrow_mut())?;
            for (r, b) in r.iter_mut().zip([1.0, 2.0, 3.0]) {
                *r -= b;
            }
            Ok(())
        }
        fn jacobian_operator_workspace_len(&self) -> Result<Option<usize>, SciOxideError> {
            Ok(Some(
                self.operator
                    .apply_workspace_len()?
                    .max(self.operator.apply_adjoint_workspace_len()?),
            ))
        }
        fn jacobian_product_into(
            &self,
            _: &[f64],
            x: &[f64],
            r: &mut [f64],
            scratch: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.operator
                .apply_into(x, r, &mut scratch[..self.operator.apply_workspace_len()?])
        }
        fn jacobian_adjoint_product_into(
            &self,
            _: &[f64],
            r: &[f64],
            x: &mut [f64],
            scratch: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.operator.apply_adjoint_into(
                r,
                x,
                &mut scratch[..self.operator.apply_adjoint_workspace_len()?],
            )
        }
    }
    let matrix = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let view = matrix.view();
    let sparse = {
        let mut builder = scioxide_sparse::SparseBuilder::new(3, 2);
        builder
            .extend(&[0, 1, 2, 2], &[0, 1, 0, 1], &[1.0; 4])
            .and_then(|()| {
                builder.finish_coo(
                    scioxide_sparse::DuplicatePolicy::Reject,
                    scioxide_sparse::ExplicitZeroPolicy::Drop,
                )
            })
    }
    .unwrap()
    .to_csr()
    .unwrap();
    for operator in [
        &matrix.view().try_finite().unwrap() as &dyn AdjointOperator<f64>,
        &view.try_finite().unwrap(),
        &sparse,
    ] {
        let problem = Borrowed {
            operator,
            scratch: RefCell::new(vec![0.0; operator.apply_workspace_len().unwrap()]),
        };
        let solver = gn();
        let result = solver.solve(&problem, &[0.0, 0.0]).unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!((result.x()[0] - 1.0).abs() < 1e-7 && (result.x()[1] - 2.0).abs() < 1e-7);
    }
}

#[test]
fn f32_extreme_cost_and_complex_coordinates() {
    let fit = Fit {
        a: Matrix::from_vec(2, 1, vec![1e25_f32; 2]).unwrap(),
        b: vec![1e25_f32; 2],
    };
    let result = gn().solve(&fit, &[0.0]).unwrap();
    assert!((result.x()[0] - 1.0).abs() < 1e-5);
    let mut x = [scioxide_core::Complex::new(0.0_f32, 0.0)];
    let mut r = [scioxide_core::Complex::new(0.0_f32, 0.0)];
    let fit = scioxide_optimize::ComplexLeastSquares::new(
        |x: &[scioxide_core::Complex<f32>], r: &mut [scioxide_core::Complex<f32>]| {
            r[0] = scioxide_core::Complex::new(x[0].re - 1.0, -x[0].im - 2.0);
            Ok(())
        },
        &mut x,
        &mut r,
    )
    .unwrap();
    let result = gn().solve(&fit, &[0.0, 0.0]).unwrap();
    assert!(result.is_success());
    assert!((result.x()[0] - 1.0).abs() < 1e-3 && (result.x()[1] + 2.0).abs() < 1e-3);
}

#[test]
fn ill_conditioned_fit_resolves_the_weak_direction_with_explicit_tolerances() {
    let fit = Fit {
        a: Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1e-6, 0.0, 1e-6]).unwrap(),
        b: vec![1.0, 2e-6, 2e-6],
    };
    let options = LeastSquaresOptions::default()
        .with_condition_estimation(scioxide_linalg::Estimation::Estimate)
        .with_gradient_tolerance(scioxide_core::GradientTolerance::new(1e-20, 0.0).unwrap())
        .with_step_tolerance(scioxide_core::StepTolerance::new(0.0, 0.0).unwrap());
    for globalization in [
        LeastSquaresGlobalization::gauss_newton(),
        LeastSquaresGlobalization::levenberg_marquardt(),
    ] {
        let result = LeastSquares::new(options.with_globalization(globalization))
            .unwrap()
            .solve(&fit, &[0.0, 0.0])
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!((result.x()[0] - 1.0_f64).abs() < 1e-12 && (result.x()[1] - 2.0_f64).abs() < 1e-7);
        assert_eq!(result.rank().unwrap().rank(), 2);
    }
}

#[test]
fn injected_executor_preserves_dense_and_operator_results() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionContext, ExecutionPlan, Executor, PartitionFn};
    struct CallerExecutor;
    impl Executor for CallerExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(1).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Err(SciOxideError::not_implemented(
                "single-worker fixture must not spawn a team",
            ))
        }
    }
    let fit = Fit {
        a: Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap(),
        b: vec![1.0, 2.0, 3.0],
    };
    let products = Products(&fit);
    let solver = gn();
    for problem in [&fit as &dyn LeastSquaresProblem<f64>, &products] {
        let mut workspace = solver
            .workspace(LeastSquaresWorkspaceRequest::of(problem).unwrap())
            .unwrap();
        let mut result = scioxide_optimize::LeastSquaresResult::empty();
        solver
            .solve_into(
                problem,
                &[0.0, 0.0],
                &mut result,
                &mut workspace,
                ExecutionContext::new(&CallerExecutor),
            )
            .unwrap();
        assert!(result.is_success());
        assert!((result.x()[0] - 1.0).abs() < 1e-7 && (result.x()[1] - 2.0).abs() < 1e-7);
    }
}

#[test]
fn rank_truncation_cannot_hide_a_nonzero_original_gradient() {
    let fit = Fit {
        a: Matrix::from_vec(2, 2, vec![1e300, 0.0, 0.0, 1e-300]).unwrap(),
        b: vec![0.0, 1e300],
    };
    let products = Products(&fit);
    for problem in [&fit as &dyn LeastSquaresProblem<f64>, &products] {
        let result = gn().solve(problem, &[0.0, 0.0]).unwrap();
        assert_eq!(
            result.termination(),
            scioxide_optimize::OptimizeTermination::StepSmall
        );
        assert!(!result.is_success());
        assert!((result.stationarity().unwrap() - 1.0).abs() < 1e-14);
    }
}

#[test]
fn qr_rhs_norm_overflow_does_not_destroy_a_representable_step() {
    let fit = Fit {
        a: Matrix::from_vec(2, 1, vec![1.0, 1.0]).unwrap(),
        b: vec![1.5e308_f64, 1.5e308],
    };
    for result in [
        gn().solve(&fit, &[0.0]).unwrap(),
        lm().solve(&fit, &[0.0]).unwrap(),
    ] {
        assert!(result.x()[0].is_finite());
        assert!((result.x()[0] / 1.5e308 - 1.0).abs() < 1e-14);
    }
}

fn gn<S: RealScalar + NumericScalar<Real = S>>() -> LeastSquares<S> {
    LeastSquares::new(
        LeastSquaresOptions::default()
            .with_globalization(LeastSquaresGlobalization::gauss_newton()),
    )
    .unwrap()
}

fn lm<S: RealScalar + NumericScalar<Real = S>>() -> LeastSquares<S> {
    LeastSquares::default()
}

#[test]
fn lm_accepts_a_useful_low_gain_step() {
    struct Curved;
    impl LeastSquaresProblem<f64> for Curved {
        fn param_dim(&self) -> usize {
            1
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn residual_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = 1.0 + x[0] + 0.8_f64.sqrt() * x[0] * x[0];
            Ok(())
        }
        fn jacobian_into(&self, x: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
            out.as_mut_slice()[0] = 1.0 + 2.0 * 0.8_f64.sqrt() * x[0];
            Ok(true)
        }
    }
    let result = LeastSquares::new(
        LeastSquaresOptions::default().with_max_iterations(scioxide_core::Budget::ONE),
    )
    .unwrap()
    .solve(&Curved, &[0.0])
    .unwrap();
    assert!((result.x()[0] + 1.0).abs() < 1e-12);
    assert!((result.last_model_gain().unwrap().round().unwrap() - 0.2).abs() < 1e-12);
    assert_eq!(result.fun_evals(), 2);
}

#[test]
fn rejected_lm_trial_reuses_the_jacobian() {
    struct Curved(core::cell::Cell<usize>);
    impl LeastSquaresProblem<f64> for Curved {
        fn param_dim(&self) -> usize {
            1
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn residual_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = x[0] * x[0] - 1.0;
            Ok(())
        }
        fn jacobian_into(&self, x: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
            self.0.set(self.0.get() + 1);
            out.as_mut_slice()[0] = 2.0 * x[0];
            Ok(true)
        }
    }
    let problem = Curved(core::cell::Cell::new(0));
    let result = LeastSquares::new(
        LeastSquaresOptions::default()
            .with_max_iterations(scioxide_core::Budget::try_from(2).unwrap()),
    )
    .unwrap()
    .solve(&problem, &[0.1])
    .unwrap();
    assert_eq!(problem.0.get(), 1);
    assert_eq!(result.jac_evals(), 1);
    assert_eq!(result.fun_evals(), 3);
}

#[test]
fn minpack_cost_stopping_is_explicit_and_does_not_claim_original_stationarity() {
    struct Floor;
    impl LeastSquaresProblem<f64> for Floor {
        fn param_dim(&self) -> usize {
            1
        }
        fn residual_dim(&self) -> usize {
            2
        }
        fn residual_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            out.copy_from_slice(&[1e8, x[0].exp() - 2.0]);
            Ok(())
        }
        fn jacobian_into(&self, x: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
            out.as_mut_slice().copy_from_slice(&[0.0, x[0].exp()]);
            Ok(true)
        }
    }
    let options = LeastSquaresOptions::default()
        .with_step_tolerance(scioxide_core::StepTolerance::new(0.0, 0.0).unwrap());
    let original = LeastSquares::new(options)
        .unwrap()
        .solve(&Floor, &[0.0])
        .unwrap();
    assert_eq!(
        original.termination(),
        scioxide_optimize::OptimizeTermination::GradientSmall
    );
    assert!((original.x()[0] - 2.0_f64.ln()).abs() < 1e-8);
    let solver = LeastSquares::new(options.with_stopping(
        scioxide_optimize::LeastSquaresStopping::Minpack {
            objective: scioxide_core::Tolerance::new(0.0, 1e-8).unwrap(),
            orthogonality: scioxide_core::RelativeTolerance::new(0.0).unwrap(),
        },
    ))
    .unwrap();
    let mut workspace = solver
        .workspace(LeastSquaresWorkspaceRequest::of(&Floor).unwrap())
        .unwrap();
    let mut model = scioxide_optimize::LeastSquaresResult::empty();
    solver
        .solve_into(
            &Floor,
            &[0.0],
            &mut model,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(
        workspace.last_accepted().unwrap().progress().trust(),
        scioxide_core::PartialResultTrust::Admissible
    );
    assert_eq!(
        model.termination(),
        scioxide_optimize::OptimizeTermination::Converged
    );
    assert!((model.x()[0] - 1.0).abs() < 1e-12);
    assert!(model.scaled_gradient_norm().is_none());
    assert_eq!(model.trust(), scioxide_core::PartialResultTrust::Admissible);
    assert!(model.message().contains("objective reductions"));
}

#[test]
fn dense_condition_estimation_is_requested_explicitly() {
    let fit = Fit {
        a: Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
        b: vec![1.0, -2.0],
    };
    assert!(
        lm().solve(&fit, &[0.0, 0.0])
            .unwrap()
            .condition_estimate()
            .is_none()
    );
    let solver = LeastSquares::new(
        LeastSquaresOptions::default()
            .with_condition_estimation(scioxide_linalg::Estimation::Estimate),
    )
    .unwrap();
    let result = solver.solve(&fit, &[0.0, 0.0]).unwrap();
    assert!((result.condition_estimate().unwrap().reciprocal().value() - 1.0).abs() < 1e-14);
}

#[test]
fn invalid_trials_never_certify_minpack_convergence() {
    struct Isolated;
    impl LeastSquaresProblem<f64> for Isolated {
        fn param_dim(&self) -> usize {
            1
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn residual_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = if x[0] == 0.0 { 1.0 } else { f64::NAN };
            Ok(())
        }
        fn jacobian_into(&self, _: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
            out.as_mut_slice()[0] = 1.0;
            Ok(true)
        }
    }
    let result = LeastSquares::new(LeastSquaresOptions::default().with_stopping(
        scioxide_optimize::LeastSquaresStopping::Minpack {
            objective: scioxide_core::Tolerance::new(0.0, 1e-8).unwrap(),
            orthogonality: scioxide_core::RelativeTolerance::new(1e-8).unwrap(),
        },
    ))
    .unwrap()
    .solve(&Isolated, &[0.0])
    .unwrap();
    assert!(!result.is_success());
    assert_eq!(result.x(), &[0.0]);
    assert_eq!(result.residual(), &[1.0]);
    assert_eq!(
        result.trust(),
        scioxide_core::PartialResultTrust::Admissible
    );
}

#[test]
fn subnormal_jacobian_can_satisfy_the_original_gradient_tolerance() {
    let fit = Fit {
        a: Matrix::from_vec(1, 1, vec![f64::MIN_POSITIVE / 16.0]).unwrap(),
        b: vec![1.0],
    };
    let result = lm().solve(&fit, &[0.0]).unwrap();
    assert_eq!(
        result.termination(),
        scioxide_optimize::OptimizeTermination::GradientSmall
    );
    assert!(result.stationarity().unwrap() < 1e-300);
}

#[test]
fn matrix_free_lm_safeguards_a_curved_model() {
    struct Curved;
    impl LeastSquaresProblem<f64> for Curved {
        fn param_dim(&self) -> usize {
            1
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn residual_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = x[0] * x[0] - 1.0;
            Ok(())
        }
        fn jacobian_operator_workspace_len(&self) -> Result<Option<usize>, SciOxideError> {
            Ok(Some(0))
        }
        fn jacobian_product_into(
            &self,
            x: &[f64],
            direction: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = 2.0 * x[0] * direction[0];
            Ok(())
        }
        fn jacobian_adjoint_product_into(
            &self,
            x: &[f64],
            input: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = 2.0 * x[0] * input[0];
            Ok(())
        }
    }
    let result = lm().solve(&Curved, &[0.1]).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.x()[0] - 1.0).abs() < 1e-8);
    assert!(
        result
            .counts()
            .total()
            .unwrap()
            .provider(scioxide_core::EvaluationRole::LinearSolve)
            > result.iterations()
    );
    let solver = LeastSquares::new(LeastSquaresOptions::default().with_stopping(
        scioxide_optimize::LeastSquaresStopping::Minpack {
            objective: scioxide_core::Tolerance::new(0.0, 1e-8).unwrap(),
            orthogonality: scioxide_core::RelativeTolerance::new(1e-8).unwrap(),
        },
    ))
    .unwrap();
    assert!(
        solver
            .workspace_requirements(LeastSquaresWorkspaceRequest::of(&Curved).unwrap())
            .is_err()
    );
}

#[test]
fn unrepresentable_trust_radius_is_a_numerical_failure() {
    struct FixedScale;
    impl LeastSquaresProblem<f64> for FixedScale {
        fn param_dim(&self) -> usize {
            1
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn parameter_scale(&self, _: usize) -> Option<f64> {
            Some(1e-300)
        }
        fn residual_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = x[0] - 1.0;
            Ok(())
        }
        fn jacobian_into(&self, _: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
            out.as_mut_slice()[0] = 1.0;
            Ok(true)
        }
    }
    let solver = LeastSquares::new(
        LeastSquaresOptions::default()
            .with_gradient_tolerance(scioxide_core::GradientTolerance::new(0.0, 0.0).unwrap())
            .with_step_tolerance(scioxide_core::StepTolerance::new(0.0, 0.0).unwrap())
            .with_max_iterations(scioxide_core::Budget::try_from(2048).unwrap()),
    )
    .unwrap();
    let error = solver.solve(&FixedScale, &[0.0]).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
}
