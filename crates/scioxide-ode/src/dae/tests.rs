//! Internal residual BDF invariants; behavior through the public API is covered by
//! `tests/dae.rs`.
use super::*;
use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState};

fn partition() -> DaePartition {
    DaePartition::new(&[true, false], &[true, false]).unwrap()
}
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(623),
        scioxide_core::algorithm_id!("test", "dae", 1),
        2,
        0,
    )
}

/// x' = -x, 0 = z - x.
struct LinearDecay;
impl<T: NumericScalar> DaeSystem<T> for LinearDecay {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        _: T::Real,
        y: &[T],
        d: &[T],
        out: &mut [T],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] + y[0];
        out[1] = y[1] - y[0];
        Ok(())
    }
    fn derivatives(
        &self,
        _: T::Real,
        _: &[T],
        _: &[T],
        a: &mut MatrixViewMut<'_, T>,
        b: &mut MatrixViewMut<'_, T>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, T::ZERO)?;
                b.set(i, j, T::ZERO)?;
            }
        }
        a.set(0, 0, T::ONE)?;
        a.set(1, 0, -T::ONE)?;
        a.set(1, 1, T::ONE)?;
        b.set(0, 0, T::ONE)?;
        Ok(Some(()))
    }
}
fn default_decay_raises_order<S: RealScalar + NumericScalar<Real = S>>() {
    let plan = (
        DaeBdf::new(partition(), OdeStageBackend::Dense).unwrap(),
        DaeOptions::new().with_initial_step(real(0.1).unwrap()),
    );
    let mut workspace = plan.0.workspace().unwrap();
    let mut state = DaeState::from_slices(S::ZERO, &[S::ONE, S::ONE], &[-S::ONE, S::ZERO]).unwrap();
    let report = plan
        .0
        .advance(
            &DaeProblem::new(&LinearDecay, S::ONE),
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(DaeTermination::ReachedEnd));
    assert!(workspace.storage.as_ref().unwrap().order > 1);
}
#[test]
fn dae_default_tolerance_raises_the_bdf_order() {
    default_decay_raises_order::<f64>();
    default_decay_raises_order::<f32>();
}

/// x' = q t^(q-1), 0 = z - x² - t.
struct PolynomialDae(usize);
impl<S: RealScalar + NumericScalar<Real = S>> DaeSystem<S> for PolynomialDae {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        t: S,
        y: &[S],
        d: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] - S::try_from_usize_exact(self.0)? * t.powi(self.0 as i32 - 1);
        out[1] = y[1] - y[0] * y[0] - t;
        Ok(())
    }
    fn derivatives(
        &self,
        _: S,
        y: &[S],
        _: &[S],
        a: &mut MatrixViewMut<'_, S>,
        b: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, S::ZERO)?;
                b.set(i, j, S::ZERO)?;
            }
        }
        a.set(1, 0, -S::TWO * y[0])?;
        a.set(1, 1, S::ONE)?;
        b.set(0, 0, S::ONE)?;
        Ok(Some(()))
    }
}
fn polynomial_stages<S: RealScalar + NumericScalar<Real = S>>() {
    let step = real::<S>(0.125).unwrap();
    let plan = (
        DaeBdf::new(partition(), OdeStageBackend::Dense).unwrap(),
        DaeOptions::new(),
    );
    for q in 1..=5 {
        let mut solve = plan.0.storage().unwrap();
        solve.accepted.copy_from_slice(&[S::ONE, S::TWO]);
        solve.accepted_derivative[0] = S::try_from_usize_exact(q).unwrap();
        solve.history_count = 5;
        solve.order = q;
        for j in 0..5 {
            let t = S::ONE - S::try_from_usize_exact(j + 1).unwrap() * step;
            let x = t.powi(q as i32);
            solve.history_times[j] = t;
            solve.history[2 * j] = x;
            solve.history[2 * j + 1] = x * x + t;
        }
        let before = solve.history.clone();
        let mut epoch = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = epoch
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let trial = solve
            .propose(
                (&plan.0, &plan.1),
                &PolynomialDae(q),
                S::ONE,
                step,
                5,
                &mut context,
            )
            .unwrap();
        assert!(trial.converged);
        assert_eq!(trial.geometry.order, q);
        let x = (S::ONE + step).powi(q as i32);
        let bound = real::<S>(128.).unwrap() * S::EPSILON * (S::ONE + x * x);
        assert!(NumericScalar::abs(solve.candidate[0] - x) < bound);
        assert!(NumericScalar::abs(solve.candidate[1] - x * x - S::ONE - step) < bound);
        assert_eq!(solve.history, before);
        assert_eq!(solve.accepted, [S::ONE, S::TWO]);
    }
}
#[test]
fn residual_bdf_orders_one_through_five_reproduce_polynomials_without_committing() {
    polynomial_stages::<f64>();
    polynomial_stages::<f32>();
}

/// Prepare and exercise the actual correction solve at a chosen cache point.
pub(super) fn prepared_correction(
    solve: &mut DaeStorage<f64>,
    plan: &(DaeBdf, DaeOptions<f64>),
    time: f64,
    inverse: f64,
    context: &mut SolveContext<'_, '_>,
) {
    evaluate(
        &LinearDecay,
        time,
        &solve.candidate,
        &solve.candidate_derivative,
        &mut solve.residual,
        context,
    )
    .unwrap();
    let mut effective = BoundDerivative {
        source: core::cell::Cell::new(None),
        problem: &LinearDecay,
        options: plan.1.numerics(),
        base: &solve.residual,
        partition: plan.0.partition(),
        time,
        state: &solve.candidate,
        derivative: &solve.candidate_derivative,
        initialization: false,
        inverse_step: inverse,
        scratch: plan.0.operator_scratch(),
        preconditioner: None,
    };
    solve
        .linear
        .prepare(
            &mut effective,
            Some(linear::NewtonPoint {
                time,
                state: &solve.candidate,
                derivative: &solve.candidate_derivative,
                strategy: plan.1.numerics().derivative_strategy(),
                initialization: false,
                inverse_step: inverse,
            }),
            context,
        )
        .unwrap();
    solve
        .linear
        .solve(
            &mut effective,
            &[1., 0.],
            &mut solve.correction,
            1e-6,
            context,
        )
        .unwrap();
}

fn direct_backends() -> [OdeStageBackend; 2] {
    [
        OdeStageBackend::Dense,
        OdeStageBackend::Sparse {
            pattern: scioxide_core::StructuralPattern::new(
                2,
                2,
                alloc::vec![0, 1, 3],
                alloc::vec![0, 0, 1],
            )
            .unwrap(),
            limits: scioxide_sparse::SparseLuLimits::new(64, 128 * 1024),
        },
    ]
}

#[test]
fn dae_raw_blocks_and_nearby_factors_are_reused_and_movement_refreshes() {
    use scioxide_differentiate::DerivativeStrategy;
    for backend in direct_backends() {
        for strategy in [
            DerivativeStrategy::AnalyticOrForward,
            DerivativeStrategy::Forward,
        ] {
            let plan = (
                DaeBdf::new(partition(), backend.clone()).unwrap(),
                DaeOptions::new().with_numerics(
                    crate::DaeNumericalOptions::new().with_derivative_strategy(strategy),
                ),
            );
            let mut solve = plan.0.storage().unwrap();
            let mut epoch = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut context = epoch
                .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                .unwrap();
            prepared_correction(&mut solve, &plan, 0., 10., &mut context);
            let counts = context.counts().own();
            prepared_correction(&mut solve, &plan, 0.1, 11., &mut context);
            assert!(solve.linear.requires_rate());
            assert!((solve.correction[0] - 1. / 11.).abs() < 1e-8);
            assert_eq!(
                context
                    .counts()
                    .own()
                    .provider(EvaluationRole::Factorization),
                1
            );
            assert_eq!(
                context.counts().own().provider(EvaluationRole::Jacobian),
                counts.provider(EvaluationRole::Jacobian)
            );
            // Both direct backends recombine the cached raw blocks for an exact factor.
            solve.linear.invalidate_factor();
            prepared_correction(&mut solve, &plan, 0.1, 11., &mut context);
            assert!((solve.correction[0] - 1. / 12.).abs() < 1e-8);
            assert_eq!(
                context
                    .counts()
                    .own()
                    .provider(EvaluationRole::Factorization),
                2
            );
            assert_eq!(
                context.counts().own().provider(EvaluationRole::Jacobian),
                counts.provider(EvaluationRole::Jacobian)
            );
            prepared_correction(&mut solve, &plan, 0.1, 20., &mut context);
            assert!((solve.correction[0] - 1. / 21.).abs() < 1e-8);
            assert_eq!(
                context
                    .counts()
                    .own()
                    .provider(EvaluationRole::Factorization),
                3
            );
            assert_eq!(
                context.counts().own().provider(EvaluationRole::Jacobian),
                counts.provider(EvaluationRole::Jacobian)
            );
            // Time, state and rate movement independently invalidate raw blocks.
            for axis in 0..3 {
                let time = 1.;
                solve.candidate.fill(if axis >= 1 { 2. } else { 0. });
                solve
                    .candidate_derivative
                    .fill(if axis >= 2 { 2. } else { 0. });
                let before = context.counts().own();
                prepared_correction(&mut solve, &plan, time, 11., &mut context);
                assert_eq!(
                    context
                        .counts()
                        .own()
                        .provider(EvaluationRole::Factorization),
                    before.provider(EvaluationRole::Factorization) + 1
                );
                if strategy == DerivativeStrategy::AnalyticOrForward {
                    assert_eq!(
                        context.counts().own().provider(EvaluationRole::Jacobian),
                        before.provider(EvaluationRole::Jacobian) + 1
                    );
                } else {
                    assert!(
                        context.counts().own().provider(EvaluationRole::Residual)
                            > before.provider(EvaluationRole::Residual)
                    );
                }
            }
        }
    }
}

#[test]
fn dae_slow_contraction_refreshes_the_factor_then_the_jacobian() {
    let plan = (
        DaeBdf::new(partition(), OdeStageBackend::Dense).unwrap(),
        DaeOptions::new(),
    );
    let mut solve = plan.0.storage().unwrap();
    let mut epoch = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = epoch
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    prepared_correction(&mut solve, &plan, 0., 10., &mut context);
    prepared_correction(&mut solve, &plan, 0.1, 11., &mut context);
    let slow = crate::stiff::newton::NewtonReport {
        iterations: 3,
        corrections: 3,
        rate: Some(0.4),
    };
    solve.linear.observe_newton(slow).unwrap();
    prepared_correction(&mut solve, &plan, 0.1, 11., &mut context);
    assert_eq!(context.counts().own().provider(EvaluationRole::Jacobian), 1);
    assert_eq!(
        context
            .counts()
            .own()
            .provider(EvaluationRole::Factorization),
        2
    );
    solve.linear.observe_newton(slow).unwrap();
    prepared_correction(&mut solve, &plan, 0.1, 11., &mut context);
    assert_eq!(context.counts().own().provider(EvaluationRole::Jacobian), 2);
    assert_eq!(
        context
            .counts()
            .own()
            .provider(EvaluationRole::Factorization),
        3
    );
}

/// A time-dependent slope changes abruptly while the state/time movement is small.
struct SwitchedDecay;
impl DaeSystem<f64> for SwitchedDecay {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        time: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] + if time == 0. { y[0] } else { 100. * y[0] };
        out[1] = y[1] - y[0];
        Ok(())
    }
    fn derivatives(
        &self,
        time: f64,
        _: &[f64],
        _: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, 0.)?;
                b.set(i, j, 0.)?;
            }
        }
        a.set(0, 0, if time == 0. { 1. } else { 100. })?;
        a.set(1, 0, -1.)?;
        a.set(1, 1, 1.)?;
        b.set(0, 0, 1.)?;
        Ok(Some(()))
    }
}

#[test]
fn dae_failed_chord_retries_an_exact_factor_then_fresh_blocks() {
    let plan = (
        DaeBdf::new(partition(), OdeStageBackend::Dense).unwrap(),
        DaeOptions::new(),
    );
    let mut solve = plan.0.storage().unwrap();
    solve.candidate.fill(1.);
    let mut epoch = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = epoch
        .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
        .unwrap();
    prepared_correction(&mut solve, &plan, 0., 10., &mut context);
    // Convergence on a recovery retry is reported to the step controller.
    assert!(matches!(
        solve
            .newton(
                (&plan.0, &plan.1),
                &SwitchedDecay,
                0.1,
                10.,
                false,
                &mut context
            )
            .unwrap(),
        (NewtonOutcome::Converged(_), true)
    ));
    assert!((solve.candidate[0] - 1. / 11.).abs() < 1e-10);
    assert!((solve.candidate[1] - solve.candidate[0]).abs() < 1e-10);
    assert!(
        residual_ok(
            residual_norms(plan.0.partition(), &solve.residual),
            plan.1.numerics()
        )
        .unwrap()
    );
    assert_eq!(context.counts().own().provider(EvaluationRole::Jacobian), 2);
    assert_eq!(
        context
            .counts()
            .own()
            .provider(EvaluationRole::Factorization),
        3
    );
}
