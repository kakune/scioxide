//! Independent order, backend, and cache checks of the transformed Radau solves.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AlgebraicScalar, Budget, BudgetKind, Complex, ErrorTolerance, EvaluationRole, ExecutionContext,
    NormSelection, NumericScalar, RealScalar, SciOxideError, SolveContext, StructuralPattern,
    StructuralPatternView,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{
    OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeSolveReport, OdeStageBackend,
    OdeState, OdeStepControl, OdeSystem, OdeTermination, OdeWorkspace,
    ProportionalIntegralController, RadauIia,
};
use scioxide_sparse::SparseLuLimits;

struct Linear<T>([T; 2]);
impl<T: NumericScalar> Linear<T> {
    fn eval(&self, y: &[T], out: &mut [T]) {
        out[0] = self.0[0] * y[0] + y[1];
        out[1] = self.0[1] * y[1];
    }
}
impl<S, T> OdeSystem<S, T> for Linear<T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    fn dimension(&self) -> usize {
        2
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
    fn rhs(
        &self,
        _: S,
        y: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.eval(y, out);
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[T],
        out: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, self.0[0])?;
        out.set(0, 1, T::ONE)?;
        out.set(1, 0, T::ZERO)?;
        out.set(1, 1, self.0[1])?;
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: S,
        _: &[T],
        _: StructuralPatternView<'_>,
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.copy_from_slice(&[self.0[0], T::ONE, T::ZERO, self.0[1]]);
        Ok(Some(()))
    }
    fn jacobian_product(
        &self,
        _: S,
        _: &[T],
        v: &[T],
        out: &mut [T],
        _: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.eval(v, out);
        Ok(Some(()))
    }
}
fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn system<T: NumericScalar>() -> Linear<T> {
    Linear([
        T::from_components(
            scalar(-4.0),
            if T::IS_REAL {
                T::Real::ZERO
            } else {
                scalar(1.0)
            },
        ),
        T::from_components(
            scalar(-7.0),
            if T::IS_REAL {
                T::Real::ZERO
            } else {
                scalar(-2.0)
            },
        ),
    ])
}

/// Advance `system` from `(t0, y)` to `end` with a prepared workspace.
fn advance<S, T, Sys>(
    plan: &(RadauIia<S, T>, OdeOptions<S>),
    system: &Sys,
    t0: S,
    end: S,
    y: &[T],
    workspace: &mut OdeWorkspace<S, RadauIia<S, T>, T>,
) -> Result<(OdeSolveReport<S>, OdeState<S, T>), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    Sys: OdeSystem<S, T>,
{
    let problem = OdeProblem::new(system, end);
    let mut state = OdeState::from_slice(t0, y)?;
    let report = plan.0.advance(
        &problem,
        &plan.1,
        &mut state,
        &mut (),
        workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok((report, state))
}

/// Exactly `h`-sized accepted steps: a unit-factor controller never changes the step.
fn fixed<S: RealScalar + NumericScalar<Real = S>>(options: OdeOptions<S>, h: S) -> OdeOptions<S> {
    options
        .with_initial_step(h)
        .with_step_control(OdeStepControl::ProportionalIntegral(
            ProportionalIntegralController::new(S::ONE, S::ONE, S::ONE, S::ONE, S::ZERO).unwrap(),
        ))
}

fn order<S, T>()
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    let problem = system::<T>();
    let exp = |x: T| Complex::new(x.real_component(), x.imaginary_component()).exp();
    let a = exp(problem.0[0]);
    let b = exp(problem.0[1]);
    let exact = [
        a + (b - a)
            / Complex::new(
                (problem.0[1] - problem.0[0]).real_component(),
                (problem.0[1] - problem.0[0]).imaginary_component(),
            ),
        b,
    ];
    // Every fixed step must be accepted, so the error control is loose. The exact
    // constant Jacobian makes the first simplified-Newton correction solve these linear
    // stage equations to roundoff, as the tight tolerance of direct stepping did.
    let tolerance = ErrorTolerance::scalar(S::ONE, S::ONE, NormSelection::Infinity).unwrap();
    let mut errors = Vec::new();
    for steps in [1_u8, 2, 4] {
        let h = scalar::<S>(1.0 / f64::from(steps));
        let plan = (
            RadauIia::<S, T>::new(),
            fixed(OdeOptions::new().with_error_tolerance(tolerance.clone()), h),
        );
        let mut workspace = OdeWorkspace::new();
        let (report, state) = advance(
            &plan,
            &problem,
            S::ZERO,
            S::ONE,
            &[T::ONE; 2],
            &mut workspace,
        )
        .unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert_eq!(report.accepted_steps(), usize::from(steps));
        assert_eq!(report.rejected_steps(), 0);
        let error = state
            .values()
            .iter()
            .zip(exact)
            .map(|(&y, exact)| {
                (Complex::new(y.real_component(), y.imaginary_component()) - exact).norm()
            })
            .fold(<S as AlgebraicScalar>::ZERO, |a, b| a.max(b));
        errors.push(error);
    }
    // Coarse refinements keep the order-five error above binary32 roundoff.
    assert!(errors[1] < errors[0] / scalar(24.0), "{errors:?}");
    assert!(errors[2] < errors[1] / scalar(24.0), "{errors:?}");
}

#[test]
fn order_five_in_all_scalar_domains() {
    order::<f32, f32>();
    order::<f64, f64>();
    order::<f32, Complex<f32>>();
    order::<f64, Complex<f64>>();
}

/// Lagrange basis value and derivative weights on `nodes` at `x`.
fn lagrange<S: RealScalar>(nodes: &[S; 4], x: S) -> ([S; 4], [S; 4]) {
    let mut value = [S::ZERO; 4];
    let mut slope = [S::ZERO; 4];
    for j in 0..4 {
        let denominator = (0..4)
            .filter(|&k| k != j)
            .fold(S::ONE, |p, k| p * (nodes[j] - nodes[k]));
        value[j] = (0..4)
            .filter(|&k| k != j)
            .fold(S::ONE, |p, k| p * (x - nodes[k]))
            / denominator;
        slope[j] = (0..4)
            .filter(|&m| m != j)
            .map(|m| {
                (0..4)
                    .filter(|&k| k != j && k != m)
                    .fold(S::ONE, |p, k| p * (x - nodes[k]))
            })
            .fold(S::ZERO, |sum, term| sum + term)
            / denominator;
    }
    (value, slope)
}

fn collocation<S, T>()
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    let problem = system::<T>();
    let root = scalar::<S>(6.0).sqrt();
    let nodes = [
        (scalar::<S>(4.0) - root) / scalar(10.0),
        (scalar::<S>(4.0) + root) / scalar(10.0),
        S::ONE,
    ];
    // The collocation output is cubic in time, so four requested samples determine it
    // and its derivative exactly (up to roundoff).
    let fractions = [
        S::ZERO,
        S::ONE / scalar(3.0),
        scalar::<S>(2.0) / scalar(3.0),
        S::ONE,
    ];
    let mut workspace = OdeWorkspace::new();
    for direction in [-1.0, 1.0] {
        let dt = scalar::<S>(direction * 0.1);
        let plan = (
            RadauIia::<S, T>::new(),
            OdeOptions::new()
                .with_initial_step(dt.abs())
                .with_output_schedule(OdeOutputSchedule::times(
                    fractions.iter().map(|&s| dt * s).collect(),
                )),
        );
        let problem_ref = OdeProblem::new(&problem, dt);
        let mut state = OdeState::new(S::ZERO, Vector::from_vec(vec![T::ONE; 2]));
        let mut samples = Vec::new();
        let mut sink = |_: S, y: &[T]| {
            samples.push([y[0], y[1]]);
            Ok(())
        };
        let report = plan
            .0
            .advance(
                &problem_ref,
                &plan.1,
                &mut state,
                &mut sink,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(report.counts().proposals(), 1);
        assert_eq!(samples.len(), 4);
        for node in nodes {
            let (weights, slopes) = lagrange(&fractions, node);
            let mut value = [T::ZERO; 2];
            let mut slope = [T::ZERO; 2];
            for (sample, (&w, &dw)) in samples.iter().zip(weights.iter().zip(&slopes)) {
                for i in 0..2 {
                    value[i] = value[i] + sample[i] * T::from_components(w, S::ZERO);
                    slope[i] = slope[i] + sample[i] * T::from_components(dw / dt, S::ZERO);
                }
            }
            let mut rhs = [T::ZERO; 2];
            problem.eval(&value, &mut rhs);
            for (&slope, &rhs) in slope.iter().zip(&rhs) {
                let tolerance = if S::EPSILON > scalar(1e-10) {
                    scalar(2e-4)
                } else {
                    scalar(2e-11)
                };
                assert!((slope - rhs).abs() < tolerance);
            }
        }
    }
}

#[test]
fn dense_derivatives_satisfy_collocation_in_real_and_complex_domains() {
    collocation::<f32, f32>();
    collocation::<f64, f64>();
    collocation::<f32, Complex<f32>>();
    collocation::<f64, Complex<f64>>();
}

fn backends<S, T>()
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    let problem = system::<T>();
    let sparse = OdeStageBackend::Sparse {
        pattern: StructuralPattern::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1]).unwrap(),
        limits: SparseLuLimits::new(8, 1 << 20),
    };
    let krylov = OdeStageBackend::Krylov {
        iterations: Budget::try_from(20).unwrap(),
        restart: Budget::try_from(2).unwrap(),
        operator_scratch: 0,
        preconditioner_scratch: 0,
    };
    let tolerance =
        ErrorTolerance::scalar(scalar(1e-4), scalar(1e-4), NormSelection::Infinity).unwrap();
    let mut reference = None;
    let mut reference_error = None;
    let mut reference_iterations = None;
    for backend in [OdeStageBackend::Dense, sparse, krylov] {
        let iterative = matches!(backend, OdeStageBackend::Krylov { .. });
        let plan = (
            RadauIia::<S, T>::new().with_backend(backend),
            OdeOptions::new()
                .with_error_tolerance(tolerance.clone())
                .with_initial_step(scalar(0.01)),
        );
        let mut workspace = OdeWorkspace::new();
        let (report, state) = advance(
            &plan,
            &problem,
            S::ZERO,
            scalar(0.01),
            &[T::ONE; 2],
            &mut workspace,
        )
        .unwrap();
        assert_eq!(report.counts().proposals(), 1);
        let counts = report.counts();
        let systems = if T::IS_REAL { 2 } else { 3 };
        assert_eq!(
            counts.provider(EvaluationRole::Factorization),
            if iterative { 0 } else { systems }
        );
        // Direct stages nest only Newton iterations; Krylov stages also nest their
        // GMRES iterations, so their Newton count is the direct backends' one.
        let nested = report.accounting().nested().iterations();
        let iterations = if iterative {
            assert!(nested > reference_iterations.unwrap());
            reference_iterations.unwrap()
        } else {
            reference_iterations.get_or_insert(nested);
            nested
        };
        assert_eq!(
            counts.provider(EvaluationRole::LinearSolve),
            systems * iterations + 1
        );
        let error = report.last_step_error_estimate().unwrap();
        if let Some(reference) = reference_error {
            // Infinity norm of the raw error over atol + rtol * max(|y0|, |y1|) = 2e-4.
            let difference: S = error - reference;
            assert!(difference.abs() * scalar::<S>(2e-4) < scalar::<S>(2e-6));
        } else {
            reference_error = Some(error);
        }
        if let Some(reference) = &reference {
            for (&actual, &expected) in state.values().iter().zip(reference) {
                let difference: T = actual - expected;
                assert!(difference.abs() < scalar::<S>(2e-5));
            }
        } else {
            reference = Some(state.values().to_vec());
        }
    }
}
#[test]
fn dense_sparse_and_operator_modes_agree() {
    backends::<f32, f32>();
    backends::<f64, f64>();
    backends::<f32, Complex<f32>>();
    backends::<f64, Complex<f64>>();
}

struct Preconditioned {
    inverse: core::cell::Cell<[f64; 3]>,
    queries: core::cell::Cell<usize>,
    calls: core::cell::Cell<usize>,
    incomplete: core::cell::Cell<bool>,
}
impl scioxide_core::Preconditioner<f64> for Preconditioned {
    fn dimension(&self) -> usize {
        2
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(3)
    }
    fn precondition_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), 3);
        self.calls.set(self.calls.get() + 1);
        let [a, b, c] = self.inverse.get();
        y[0] = a * x[0] + b * x[1];
        if !self.incomplete.get() {
            y[1] = c * x[1];
        }
        Ok(())
    }
}
impl OdeSystem<f64> for Preconditioned {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        system::<f64>().eval(y, out);
        Ok(())
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
        assert_eq!(scratch.len(), 5);
        system::<f64>().eval(x, out);
        Ok(Some(()))
    }
    fn stage_preconditioner(
        &self,
        _: f64,
        _: &[f64],
        shift: f64,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<&dyn scioxide_core::Preconditioner<f64>>, SciOxideError> {
        self.queries.set(self.queries.get() + 1);
        let (a, c) = (1.0 / (1.0 + 4.0 * shift), 1.0 / (1.0 + 7.0 * shift));
        self.inverse.set([a, shift * a * c, c]);
        Ok(Some(self))
    }
    fn coupled_preconditioner(
        &self,
        _: f64,
        _: &[f64],
        _: f64,
        _: &[f64],
        _: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<&dyn scioxide_core::Preconditioner<f64>>, SciOxideError> {
        unreachable!("native Radau uses shifted n-dimensional preconditioners")
    }
}
#[test]
fn krylov_lifts_native_products_and_preconditioners_with_reserved_scratch() {
    let problem = Preconditioned {
        inverse: core::cell::Cell::new([0.0; 3]),
        queries: core::cell::Cell::new(0),
        calls: core::cell::Cell::new(0),
        incomplete: core::cell::Cell::new(false),
    };
    let plan = (
        RadauIia::<f64>::new().with_backend(OdeStageBackend::Krylov {
            iterations: Budget::try_from(20).unwrap(),
            restart: Budget::try_from(2).unwrap(),
            operator_scratch: 5,
            preconditioner_scratch: 3,
        }),
        OdeOptions::new().with_initial_step(0.01),
    );
    let mut workspace = OdeWorkspace::new();
    let (report, state) = advance(&plan, &problem, 0.0, 0.01, &[1.0; 2], &mut workspace).unwrap();
    assert_eq!(report.counts().proposals(), 1);
    assert_eq!(
        problem.queries.get(),
        report.counts().provider(EvaluationRole::LinearSolve)
    );
    assert!(problem.calls.get() > problem.queries.get());
    assert_eq!(report.counts().provider(EvaluationRole::Factorization), 0);
    // A non-finite preconditioned stage solve is a rejected trial: with no smaller step
    // permitted, integration stops without committing it.
    problem.incomplete.set(true);
    let strict = (
        plan.0.clone(),
        OdeOptions::new()
            .with_initial_step(0.01)
            .with_minimum_step(0.01),
    );
    let mut after = state.clone();
    let report = strict
        .0
        .advance(
            &OdeProblem::new(&problem, 0.02),
            &strict.1,
            &mut after,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::StepSizeLimit));
    assert_eq!(
        (report.counts().proposals(), report.rejected_steps()),
        (1, 1)
    );
    assert_eq!(after, state);
}

#[test]
fn exact_work_budgets_and_partial_factor_bundles_preserve_state() {
    let problem = system::<f64>();
    let tolerance = ErrorTolerance::scalar(1e-4, 1e-4, NormSelection::RootMeanSquare).unwrap();
    let options = OdeOptions::new()
        .with_error_tolerance(tolerance)
        .with_initial_step(0.01);
    let roles = [
        EvaluationRole::RightHandSide,
        EvaluationRole::Jacobian,
        EvaluationRole::Factorization,
        EvaluationRole::LinearSolve,
    ];
    let plan = (RadauIia::<f64>::new(), options.clone());
    let mut workspace = OdeWorkspace::new();
    let (reference, expected_state) =
        advance(&plan, &problem, 0.0, 0.01, &[1.0; 2], &mut workspace).unwrap();
    assert_eq!(reference.termination(), Some(OdeTermination::ReachedEnd));
    assert_eq!(reference.counts().proposals(), 1);
    let reference_total = reference.accounting().total().unwrap();
    let expected = roles.map(|role| reference_total.provider(role));
    let total = expected.iter().sum::<usize>();
    assert_eq!(reference_total.total_providers().unwrap(), total);
    for limit in 1..=total {
        let plan = (
            RadauIia::<f64>::new(),
            options
                .clone()
                .with_max_evaluations(Budget::try_from(limit).unwrap()),
        );
        let (report, state) =
            advance(&plan, &problem, 0.0, 0.01, &[1.0; 2], &mut workspace).unwrap();
        if limit == total {
            assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
            assert_eq!(state.values(), expected_state.values());
            for (role, count) in roles.into_iter().zip(expected) {
                assert_eq!(report.accounting().total().unwrap().provider(role), count);
            }
        } else {
            assert_eq!(
                report.termination(),
                Some(OdeTermination::BudgetLimit(BudgetKind::Evaluation)),
                "limit={limit}, expected={expected:?}, actual={:?}",
                report.accounting().total()
            );
            assert_eq!(report.final_time(), 0.0);
            assert_eq!(state.time(), 0.0);
            assert_eq!(state.values(), [1.0; 2]);
        }
        assert_eq!(
            report
                .accounting()
                .total()
                .unwrap()
                .total_providers()
                .unwrap(),
            limit
        );
    }
}
