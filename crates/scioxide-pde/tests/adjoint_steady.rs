//! Manufactured discrete Poisson gradients, transpose identities and transactions.
#![allow(clippy::unwrap_used, clippy::too_many_arguments)]
use scioxide_core::*;
use scioxide_linalg::*;
use scioxide_pde::*;
#[path = "support/adjoint_poisson.rs"]
mod fixture;
use fixture::*;

fn tolerance<T: NumericScalar>() -> ResidualTolerance<T::Real> {
    ResidualTolerance::new(T::Real::EPSILON * r(1024.), T::Real::EPSILON * r(1024.)).unwrap()
}
/// Test tolerances for the primal and linear checks, unit residual scale and a byte ceiling.
fn options<T: NumericScalar>(max_bytes: usize) -> PdeSteadyAdjointOptions<T::Real> {
    PdeSteadyAdjointOptions::new()
        .with_primal_tolerance(tolerance::<T>())
        .with_linear_tolerance(tolerance::<T>())
        .with_primal_scale(T::Real::ONE)
        .with_max_workspace_bytes(max_bytes)
}
fn manufactured<T: NumericScalar>() {
    for policy in [false, true] {
        let model = Poisson::<T>::new(7, policy);
        let solution = model.solution();
        let parameters = model.parameter_matrix().unwrap();
        let dense = model.matrix().unwrap();
        let sparse = model.sparse().unwrap();
        let mut padded = vec![T::from_real(r(99.)); solution.len() * 2];
        for (i, &v) in solution.iter().enumerate() {
            padded[2 * i] = v;
        }
        let view = VectorView::from_slice_with_stride(&padded, solution.len(), 2).unwrap();
        for state in [
            &model as &dyn AdjointOperator<T>,
            &dense.view().try_finite().unwrap(),
            &sparse,
        ] {
            let parameter_view = parameters.view().try_finite().unwrap();
            // Keep the borrowed parameter view alive for every call.
            let derivatives =
                PdeDiscretizationDerivatives::new(view, &model.p, &model, state, &parameter_view)
                    .unwrap();
            let plan = options::<T>(1024 * 1024);
            let mut work = derivatives.workspace(&model, &plan).unwrap();
            let mut solver = PreparedCg::new(state).unwrap();
            let mut lambda = vec![T::from_real(r(77.)); 2 * solution.len()];
            let mut gradient = vec![r::<T::Real>(88.); 8];
            context(|ctx| {
                derivatives.solve_adjoint_into(
                    &model,
                    T::Real::ZERO,
                    &mut |a, b, x, c| solver.solve(a, b, x, c),
                    &plan,
                    &mut VectorViewMut::from_slice_with_stride(&mut lambda, solution.len(), 2)
                        .unwrap(),
                    &mut VectorViewMut::from_slice_with_stride(&mut gradient, 4, 2).unwrap(),
                    &mut work,
                    ctx,
                )
            })
            .unwrap();
            for (i, &expected) in model.expected().iter().enumerate() {
                assert!(
                    (gradient[2 * i] - expected).abs()
                        <= r::<T::Real>(2048.) * T::Real::EPSILON * (T::Real::ONE + expected.abs()),
                    "component {i}: {:?} vs {:?}",
                    gradient[2 * i],
                    expected
                );
                assert_eq!(gradient[2 * i + 1], r(88.));
            }
            assert!(
                lambda
                    .iter()
                    .skip(1)
                    .step_by(2)
                    .all(|&v| v == T::from_real(r(77.)))
            );
            let direction = [r(0.3), r(-0.2), r(0.4), r(0.5)];
            let mut tangent = vec![T::ZERO; solution.len()];
            context(|ctx| {
                derivatives.sensitivity_into(
                    &direction,
                    &mut |a, b, x, c| solver.solve(a, b, x, c),
                    &plan,
                    &mut VectorViewMut::from_slice(&mut tangent),
                    &mut work,
                    ctx,
                )
            })
            .unwrap();
            let mut forward = solution
                .iter()
                .zip(&tangent)
                .map(|(&y, &v)| model.h() * (y.conj() * v).real_component())
                .fold(T::Real::ZERO, |a, b| a + b);
            if policy {
                forward += direction[3]
                    * solution
                        .iter()
                        .map(|y| model.h() * y.norm_sqr() * T::Real::HALF / model.p[3])
                        .fold(T::Real::ZERO, |a, b| a + b);
            }
            let reverse = direction
                .iter()
                .enumerate()
                .map(|(i, &v)| v * gradient[2 * i])
                .fold(T::Real::ZERO, |a, b| a + b);
            assert!(
                (forward - reverse).abs()
                    <= r::<T::Real>(2048.) * T::Real::EPSILON * (T::Real::ONE + reverse.abs())
            );
            let before = lambda.clone();
            let before_g = gradient.clone();
            assert!(
                context(|ctx| derivatives.solve_adjoint_into(
                    &model,
                    T::Real::ZERO,
                    &mut |_, _, x, _| {
                        x.fill(T::ZERO);
                        Ok(())
                    },
                    &plan,
                    &mut VectorViewMut::from_slice_with_stride(&mut lambda, solution.len(), 2)
                        .unwrap(),
                    &mut VectorViewMut::from_slice_with_stride(&mut gradient, 4, 2).unwrap(),
                    &mut work,
                    ctx
                ))
                .is_err()
            );
            assert_eq!(lambda, before);
            assert_eq!(gradient, before_g);
        }
    }
}
#[test]
fn all_domains_and_representations() {
    manufactured::<f32>();
    manufactured::<f64>();
    manufactured::<Complex32>();
    manufactured::<Complex64>();
}
#[test]
fn invalid_bytes_primal_and_budget() {
    let model = Poisson::<f64>::new(7, false);
    let p = model.parameter_matrix().unwrap();
    let solution = model.solution();
    let p_finite = p.view().try_finite().unwrap();
    let bind = |point| {
        PdeDiscretizationDerivatives::new(
            VectorView::from_slice(point),
            &model.p,
            &model,
            &model,
            &p_finite,
        )
        .unwrap()
    };
    assert!(
        bind(&solution)
            .workspace(&model, &options::<f64>(0))
            .is_err()
    );
    let wrong = vec![0.; 7];
    let unbalanced = bind(&wrong);
    let plan = options::<f64>(1 << 20);
    assert!(
        unbalanced
            .solve_adjoint_with_options(&model, 0., &mut |_, _, _, _| Ok(()), &plan)
            .is_err()
    );
    let equilibrium = bind(&solution);
    let plan = options::<f64>(1 << 20);
    let mut work = equilibrium.workspace(&model, &plan).unwrap();
    let mut l = vec![17.; 7];
    let mut g = vec![19.; 4];
    let mut state = SolveState::new();
    let mut limits = SolveBudgets::unbounded();
    limits = limits.with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut ctx = state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(1),
                scioxide_core::algorithm_id!("pde", "adjoint-test", 1),
                0,
                0,
            ),
            &mut limits,
        )
        .unwrap();
    assert!(
        equilibrium
            .solve_adjoint_into(
                &model,
                0.,
                &mut |_, _, _, _| Ok(()),
                &plan,
                &mut VectorViewMut::from_slice(&mut l),
                &mut VectorViewMut::from_slice(&mut g),
                &mut work,
                &mut ctx
            )
            .is_err()
    );
    assert_eq!(l, vec![17.; 7]);
    assert_eq!(g, vec![19.; 4]);
}

#[test]
fn supplied_ad_parameter_transpose_reuses_core_operator() {
    use scioxide_ad::{Tape, TapeOptions};
    let policy = true;
    let model = Poisson::<f64>::new(1, policy);
    let solution = model.solution();
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let linearization = tape
        .linearize(
            |x, y| {
                let (k, b, source, length) = (x[0], x[1], x[2], x[3]);
                let numerator = (-b + solution[0]) * 2.;
                let coefficient = k * 4. / (length * length);
                y.push(coefficient * numerator - source);
                Ok(())
            },
            &model.p,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let parameter = linearization.jacobian_operator().unwrap();
    let derivatives = PdeDiscretizationDerivatives::new(
        VectorView::from_slice(&solution),
        &model.p,
        &model,
        &model,
        &parameter,
    )
    .unwrap();
    let plan = options::<f64>(1 << 20);
    let mut solver = PreparedCg::new(&model).unwrap();
    let result = derivatives
        .solve_adjoint_with_options(
            &model,
            0.,
            &mut |a, b, x, c| solver.solve(a, b, x, c),
            &plan,
        )
        .unwrap();
    for (&a, &b) in result.parameter_gradient().iter().zip(&model.expected()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn nonlinear_pde_residual_and_tangent_compose_without_a_second_solver() {
    struct Physical<'a>(&'a Poisson<f64>);
    impl scioxide_ode::OdeSystem<f64> for Physical<'_> {
        fn dimension(&self) -> usize {
            self.0.n
        }
        fn rhs(
            &self,
            _: f64,
            y: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            self.0.residual_into(y, out, &mut [])
        }
        fn jacobian_product(
            &self,
            _: f64,
            _: &[f64],
            x: &[f64],
            out: &mut [f64],
            scratch: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<Option<()>, SciOxideError> {
            self.0.apply_into(x, out, scratch).map(Some)
        }
    }
    let policy = false;
    let model = Poisson::<f64>::new(7, policy);
    let solution = model.solution();
    let parameters = model.parameter_matrix().unwrap();
    let fields = PdeFieldLayout::new(&[7]).unwrap();
    let physical = Physical(&model);
    let transpose = |_: f64, _: &[f64], seed: &[f64], out: &mut [f64], scratch: &mut [f64]| {
        model.apply_adjoint_into(seed, out, scratch)
    };
    let nonlinear_fields = fields;
    let nonlinear_mode = PdeJacobianMode::Operator {
        operator_scratch: 0,
    };
    let mut storage = NonlinearPdeProblem::new(&physical, 0.0, &nonlinear_fields, &nonlinear_mode)
        .workspace()
        .unwrap();
    let residual = NonlinearPdeProblem::new(&physical, 0., &nonlinear_fields, &nonlinear_mode)
        .with_adjoint(&transpose, 0)
        .bind(&mut storage)
        .unwrap();
    let tangent = PdeTangent::new(&residual, &solution).unwrap();
    let parameters_finite = parameters.view().try_finite().unwrap();
    let derivatives = PdeDiscretizationDerivatives::new(
        VectorView::from_slice(&solution),
        &model.p,
        &residual,
        &tangent,
        &parameters_finite,
    )
    .unwrap();
    let plan = options::<f64>(1 << 20);
    let mut solver = PreparedCg::new(&tangent).unwrap();
    let result = derivatives
        .solve_adjoint_with_options(
            &model,
            0.,
            &mut |a, b, x, c| solver.solve(a, b, x, c),
            &plan,
        )
        .unwrap();
    for (&a, &b) in result.parameter_gradient().iter().zip(&model.expected()) {
        assert!((a - b).abs() < 1e-11);
    }
}

// The scoped entry points consume a shared resource to expose lost caller ledgers.
struct ScopeRequiredOperator<'a> {
    inner: &'a dyn scioxide_core::AdjointOperator<f64>,
    calls: std::cell::Cell<usize>,
}
impl ScopeRequiredOperator<'_> {
    fn unscoped() -> Result<(), scioxide_core::SciOxideError> {
        Err(scioxide_core::SciOxideError::invalid_input(
            scioxide_core::argument!("context"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "caller scope required",
        ))
    }
    fn scoped(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
        c: &mut scioxide_core::SolveContext<'_, '_>,
        adjoint: bool,
    ) -> Result<(), scioxide_core::SciOxideError> {
        c.execution().check()?;
        c.try_consume(scioxide_core::BudgetKind::Node, 1)?
            .require()?;
        self.calls.set(self.calls.get() + 1);
        if adjoint {
            self.inner.apply_adjoint_into(x, y, w)
        } else {
            self.inner.apply_into(x, y, w)
        }
    }
}
impl scioxide_core::LinearOperator<f64> for ScopeRequiredOperator<'_> {
    fn rows(&self) -> usize {
        self.inner.rows()
    }
    fn cols(&self) -> usize {
        self.inner.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        self.inner.apply_workspace_len()
    }
    fn apply_into(
        &self,
        _: &[f64],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        Self::unscoped()
    }
    fn apply_into_scoped(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
    ) -> Result<scioxide_core::Invocation<()>, scioxide_core::SciOxideError> {
        self.scoped(x, y, w, c, false)
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl scioxide_core::AdjointOperator<f64> for ScopeRequiredOperator<'_> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
        self.inner.apply_adjoint_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        _: &[f64],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), scioxide_core::SciOxideError> {
        Self::unscoped()
    }
    fn apply_adjoint_into_scoped(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
        c: &mut scioxide_core::OperatorContext<'_, '_, '_, f64>,
    ) -> Result<scioxide_core::Invocation<()>, scioxide_core::SciOxideError> {
        self.scoped(x, y, w, c, true)
            .map(scioxide_core::Invocation::Evaluated)
    }
}

#[test]
fn steady_adjoint_retains_scope_through_transpose_solve_and_residual_check() {
    let policy = false;
    let model = Poisson::<f64>::new(7, policy);
    let solution = model.solution();
    let parameters = model.parameter_matrix().unwrap();
    let state = ScopeRequiredOperator {
        inner: &model,
        calls: std::cell::Cell::new(0),
    };
    let parameters_finite = parameters.view().try_finite().unwrap();
    let derivatives = PdeDiscretizationDerivatives::new(
        VectorView::from_slice(&solution),
        &model.p,
        &model,
        &state,
        &parameters_finite,
    )
    .unwrap();
    let plan = options::<f64>(1 << 20);
    let mut work = derivatives.workspace(&model, &plan).unwrap();
    let mut solver = PreparedCg::new(&state).unwrap();
    let mut adjoint = vec![0.; solution.len()];
    let mut gradient = vec![0.; model.p.len()];
    context(|ctx| {
        derivatives.solve_adjoint_into(
            &model,
            0.0,
            &mut |a, b, x, c| solver.solve(a, b, x, c),
            &plan,
            &mut VectorViewMut::from_slice(&mut adjoint),
            &mut VectorViewMut::from_slice(&mut gradient),
            &mut work,
            &mut *ctx,
        )?;
        assert!(
            gradient
                .iter()
                .zip(model.expected())
                .all(|(a, b)| (*a - b).abs() < 1e-12)
        );
        assert!(state.calls.get() > 0);
        assert_eq!(
            ctx.budget_report(BudgetKind::Node).consumed(),
            state.calls.get()
        );
        Ok(())
    })
    .unwrap();
}
