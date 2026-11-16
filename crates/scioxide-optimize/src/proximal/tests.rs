use super::*;
use alloc::vec;
use core::cell::Cell;
use core::time::Duration;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AttemptKind, AttemptOutcome, Budget, BudgetKind, Capacity, Complex, ExecutionContext,
    GradientTolerance, LinearOperator, NumericScalar, ProblemToken, RealScalar, ResidualTolerance,
    SciOxideError, SciOxideErrorKind, SciOxideErrorView, SolveBudgets, SolveContext, SolveIdentity,
    SolveState,
};
use scioxide_linalg::Matrix;
use scioxide_sparse::CsrMatrix;

struct IdentityProx;
impl<S: NumericScalar> ProximalMap<S> for IdentityProx {
    fn apply_into(&self, point: &[S], _: S::Real, output: &mut [S]) -> Result<(), SciOxideError> {
        output.copy_from_slice(point);
        Ok(())
    }
}

#[test]
fn rounded_away_smooth_step_does_not_certify_stationarity() -> Result<(), SciOxideError> {
    struct Linear;
    impl<S: NumericScalar> SmoothGradient<S> for Linear {
        fn dimension(&self) -> usize {
            1
        }
        fn value(&self, x: &[S]) -> Result<S::Real, SciOxideError> {
            Ok(x[0].real_component())
        }
        fn gradient_into(&self, _: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
            output.fill(S::ONE);
            Ok(())
        }
    }
    let problem = CompositeProblem::new(&Linear, &IdentityProx);
    for accelerated in [false, true] {
        let options = ProximalOptions::new(1.0_f64);
        let result = if accelerated {
            Fista::new(options)?.minimize(&problem, &[1e20_f64])?
        } else {
            ProximalGradient::new(options)?.minimize(&problem, &[1e20_f64])?
        };
        assert_eq!(result.termination(), crate::OptimizeTermination::Stagnation);
        assert_eq!(result.iterations(), 0);
        assert!(result.mapping_norm().is_none());
    }
    let result =
        ProximalGradient::new(ProximalOptions::new(1.0_f32))?.minimize(&problem, &[1e10_f32])?;
    assert_eq!(result.termination(), crate::OptimizeTermination::Stagnation);
    Ok(())
}

#[test]
fn proximal_callback_cancellation_preempts_convergence() -> Result<(), SciOxideError> {
    struct Cancel<'a> {
        token: &'a scioxide_core::CancellationToken,
        calls: Cell<usize>,
        at: usize,
    }
    impl Cancel<'_> {
        fn call(&self) {
            self.calls.set(self.calls.get() + 1);
            if self.calls.get() == self.at {
                self.token.cancel();
            }
        }
    }
    impl SmoothGradient<f64> for Cancel<'_> {
        fn dimension(&self) -> usize {
            1
        }
        fn value(&self, _: &[f64]) -> Result<f64, SciOxideError> {
            self.call();
            Ok(0.0)
        }
        fn gradient_into(&self, _: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
            self.call();
            output.fill(0.0);
            Ok(())
        }
    }
    impl ProximalMap<f64> for Cancel<'_> {
        fn apply_into(
            &self,
            point: &[f64],
            _: f64,
            output: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.call();
            output.copy_from_slice(point);
            Ok(())
        }
    }
    for at in 1..=4 {
        let token = scioxide_core::CancellationToken::new();
        let callbacks = Cancel {
            token: &token,
            calls: Cell::new(0),
            at,
        };
        let problem = CompositeProblem::new(&callbacks, &callbacks);
        let solver =
            ProximalGradient::new(ProximalOptions::new(1.0).with_backtracking(2.0, Budget::ONE))?;
        let mut workspace = solver.workspace::<f64>(1)?;
        let mut output = ProximalResult::empty();
        solver.minimize_into(
            &problem,
            &[0.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token),
        )?;
        assert_eq!(output.termination(), crate::OptimizeTermination::Cancelled);
        assert_eq!(callbacks.calls.get(), at);
        assert_eq!(output.iterations(), 0);
        assert!(output.mapping_norm().is_none());
    }
    Ok(())
}

fn scaled_proximal<S: NumericScalar>(
    scale: S::Real,
    tolerance: S::Real,
) -> Result<(), SciOxideError> {
    struct ScaledQuadratic<R>(R);
    impl<S: NumericScalar> SmoothGradient<S> for ScaledQuadratic<S::Real> {
        fn dimension(&self) -> usize {
            1
        }
        fn value(&self, x: &[S]) -> Result<S::Real, SciOxideError> {
            let re = x[0].real_component();
            let im = x[0].imaginary_component();
            Ok(((self.0 * re) * re + (self.0 * im) * im) * S::Real::HALF)
        }
        fn gradient_into(&self, x: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
            output[0] = S::from_real(self.0) * x[0];
            Ok(())
        }
    }
    let bound = scale.recip();
    let smooth = ScaledQuadratic(bound);
    let problem = CompositeProblem::new(&smooth, &IdentityProx);
    let options = ProximalOptions::new(bound)
        .with_tolerance(GradientTolerance::new(tolerance, S::Real::ZERO)?)
        .with_backtracking(S::Real::TWO, Budget::try_from(4)?);
    for accelerated in [false, true] {
        let start = [S::from_components(scale, scale)];
        let result = if accelerated {
            Fista::new(options)?.minimize(&problem, &start)?
        } else {
            ProximalGradient::new(options)?.minimize(&problem, &start)?
        };
        assert!(result.is_success(), "{result:?}");
        assert!((result.x()[0] / S::from_real(scale)).abs() <= tolerance);
    }
    Ok(())
}

#[test]
fn proximal_majorization_keeps_extended_range_for_real_and_complex() -> Result<(), SciOxideError> {
    for scale in [1e200_f64, 1e-200] {
        scaled_proximal::<f64>(scale, 1e-12)?;
        scaled_proximal::<Complex<f64>>(scale, 1e-12)?;
    }
    for scale in [1e20_f32, 1e-30] {
        scaled_proximal::<f32>(scale, 1e-5)?;
        scaled_proximal::<Complex<f32>>(scale, 1e-5)?;
    }
    Ok(())
}

#[test]
fn backtracking_recovers_from_unrepresentable_proposals() -> Result<(), SciOxideError> {
    let smooth = Quadratic {
        b: 0.0_f64,
        curvature: 1.0,
    };
    let problem = CompositeProblem::new(&smooth, &IdentityProx);
    let options = ProximalOptions::new(1e-308).with_backtracking(1e154, Budget::try_from(4)?);
    for accelerated in [false, true] {
        let result = if accelerated {
            Fista::new(options)?.minimize(&problem, &[2.0])?
        } else {
            ProximalGradient::new(options)?.minimize(&problem, &[2.0])?
        };
        assert!(result.is_success(), "{result:?}");
        assert!(result.x()[0].abs() < 1e-8);
    }
    Ok(())
}

#[test]
fn exhausted_backtracking_keeps_last_finite_lipschitz_estimate() -> Result<(), SciOxideError> {
    struct Isolated;
    impl SmoothGradient<f64> for Isolated {
        fn dimension(&self) -> usize {
            1
        }
        fn value(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(if x[0] == 0.0 { 0.0 } else { f64::INFINITY })
        }
        fn gradient_into(&self, _: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
            output[0] = 1.0;
            Ok(())
        }
    }
    let problem = CompositeProblem::new(&Isolated, &IdentityProx);
    let options = ProximalOptions::new(2.0).with_backtracking(f64::MAX, Budget::ONE);
    let result = ProximalGradient::new(options)?.minimize(&problem, &[0.0])?;
    assert_eq!(result.termination(), crate::OptimizeTermination::Stagnation);
    assert_eq!(result.x(), &[0.0]);
    Ok(())
}

/// Begin a parent epoch whose budgets bound the child solves run inside it.
fn with_parent<T>(
    budgets: SolveBudgets,
    body: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let mut budgets = budgets;
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("test", "parent", 1),
        0,
        0,
    );
    let mut parent = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    body(&mut parent)
}

fn evaluation_budget(limit: Budget) -> SolveBudgets {
    SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, limit)
}

fn invalid_argument(error: &SciOxideError, name: &'static str) -> bool {
    matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { argument, .. }
            if scioxide_core::Argument::new(name).is_ok_and(|expected| expected == argument)
    ) && error.kind() == SciOxideErrorKind::InvalidInput
}

fn outcome<P>(record: Option<&scioxide_core::AttemptRecord<P>>) -> Option<AttemptOutcome> {
    let record = record?;
    assert_eq!(record.kind(), AttemptKind::Call);
    Some(record.outcome())
}

struct Quadratic<S: NumericScalar> {
    b: S,
    curvature: S::Real,
}
impl<S: NumericScalar> SmoothGradient<S> for Quadratic<S> {
    fn dimension(&self) -> usize {
        1
    }
    fn value(&self, x: &[S]) -> Result<S::Real, SciOxideError> {
        Ok(self.curvature * (x[0] - self.b).norm_sqr() * S::Real::HALF)
    }
    fn gradient_into(&self, x: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        output[0] = S::from_real(self.curvature) * (x[0] - self.b);
        Ok(())
    }
}

/// Real quadratic `0.5 c (x-b)^2` that counts calls and fails its `fail_at`-th gradient.
struct Counted {
    inner: Quadratic<f64>,
    calls: Cell<usize>,
    fail_at: Option<usize>,
}
impl Counted {
    fn new(fail_at: Option<usize>) -> Self {
        Self {
            inner: Quadratic {
                b: 3.0,
                curvature: 2.0,
            },
            calls: Cell::new(0),
            fail_at,
        }
    }
}
impl SmoothGradient<f64> for Counted {
    fn dimension(&self) -> usize {
        1
    }
    fn value(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        self.inner.value(x)
    }
    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail_at == Some(call) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("gradient"),
                scioxide_core::InvalidInputReason::Malformed,
                "injected gradient failure",
            ));
        }
        self.inner.gradient_into(x, output)
    }
}

struct Shrink<R: RealScalar>(R);
impl<S: NumericScalar> ProximalMap<S> for Shrink<S::Real> {
    fn apply_into(
        &self,
        point: &[S],
        step: S::Real,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        let threshold = self.0 * step;
        for (&value, dest) in point.iter().zip(output.iter_mut()) {
            let magnitude = value.abs();
            *dest = if magnitude <= threshold {
                S::ZERO
            } else {
                S::from_real(S::Real::ONE - threshold / magnitude) * value
            };
        }
        Ok(())
    }
}

fn proximal_fixture<S: NumericScalar>(b: S, precision: S::Real) -> Result<(), SciOxideError> {
    let one = S::Real::ONE;
    let two = one + one;
    let quadratic = Quadratic { b, curvature: two };
    let prox = Shrink(one);
    let problem = CompositeProblem::new(&quadratic, &prox);
    let options =
        ProximalOptions::new(two).with_tolerance(GradientTolerance::new(precision, S::Real::ZERO)?);
    let expected = S::from_real(one - one / (two * b.abs())) * b;
    for accelerated in [false, true] {
        let result = if accelerated {
            Fista::new(options)?.minimize(&problem, &[S::ZERO])?
        } else {
            ProximalGradient::new(options)?.minimize(&problem, &[S::ZERO])?
        };
        assert!(result.is_success(), "{result:?}");
        assert!((result.x()[0] - expected).abs() <= precision);
        // Independent proximal optimality: 0 ∈ 2(x-b) + ∂|x|.
        let stationarity = S::from_real(two) * (result.x()[0] - b)
            + S::from_real(one / result.x()[0].abs()) * result.x()[0];
        assert!(stationarity.abs() <= precision);
        assert!(result.mapping_norm().is_some_and(|norm| norm <= precision));
    }
    Ok(())
}

#[test]
fn proximal_real_and_complex_oracles() -> Result<(), SciOxideError> {
    proximal_fixture::<f64>(3.0, 1e-9)?;
    proximal_fixture::<f32>(3.0, 1e-4)?;
    proximal_fixture::<Complex<f64>>(Complex { re: 3.0, im: 4.0 }, 1e-9)?;
    proximal_fixture::<Complex<f32>>(Complex { re: 3.0, im: 4.0 }, 1e-4)
}

#[test]
fn elastic_net_has_an_independent_soft_threshold_solution() -> Result<(), SciOxideError> {
    struct Elastic;
    impl SmoothGradient<f64> for Elastic {
        fn dimension(&self) -> usize {
            1
        }
        fn value(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(0.5 * (x[0] - 3.0).powi(2) + 0.5 * x[0] * x[0])
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = 2.0 * x[0] - 3.0;
            Ok(())
        }
    }
    let smooth: &dyn SmoothGradient<f64> = &Elastic;
    let nonsmooth: &dyn ProximalMap<f64> = &Shrink(1.0);
    let result = Fista::new(ProximalOptions::new(2.0))?
        .minimize(&CompositeProblem::new(smooth, nonsmooth), &[0.0])?;
    assert!(result.is_success());
    assert!((result.x()[0] - 1.0).abs() < 1e-9);
    assert!((2.0 * result.x()[0] - 3.0 + 1.0).abs() < 1e-9);
    Ok(())
}

#[test]
fn backtracking_recovers_an_underestimated_lipschitz_bound() -> Result<(), SciOxideError> {
    let quadratic = Quadratic {
        b: 3.0_f64,
        curvature: 8.0,
    };
    let options = ProximalOptions::new(1.0).with_backtracking(2.0, Budget::try_from(8)?);
    let result = ProximalGradient::new(options)?
        .minimize(&CompositeProblem::new(&quadratic, &Shrink(1.0)), &[0.0])?;
    assert!(result.is_success(), "{result:?}");
    assert!(result.lipschitz() >= 8.0);
    assert!((result.x()[0] - 2.875).abs() < 1e-7);
    Ok(())
}

#[test]
fn fista_gradient_restart_is_exercised() -> Result<(), SciOxideError> {
    struct IllConditioned;
    impl SmoothGradient<f64> for IllConditioned {
        fn dimension(&self) -> usize {
            2
        }
        fn value(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(0.5 * (x[0] - 2.0).powi(2) + 20.0 * (x[1] + 1.0).powi(2))
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
            out[0] = x[0] - 2.0;
            out[1] = 40.0 * (x[1] + 1.0);
            Ok(())
        }
    }
    let options = ProximalOptions::new(40.0)
        .with_max_iterations(Budget::try_from(300)?)
        .with_tolerance(GradientTolerance::new(1e-8, 0.0)?);
    let result = Fista::new(options)?.minimize(
        &CompositeProblem::new(&IllConditioned, &Shrink(0.0)),
        &[0.0, 0.0],
    )?;
    assert!(result.is_success(), "{result:?}");
    assert!(result.restarts() > 0, "{result:?}");
    assert!((result.x()[0] - 2.0).abs() < 1e-7);
    assert!((result.x()[1] + 1.0).abs() < 1e-7);
    Ok(())
}

#[test]
fn proximal_plan_constructors_validate_the_policy() -> Result<(), SciOxideError> {
    for lipschitz in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let options = ProximalOptions::new(lipschitz);
        let error = ProximalGradient::new(options).unwrap_err();
        assert!(invalid_argument(&error, "lipschitz"), "{error:?}");
        assert!(invalid_argument(
            &Fista::new(options).unwrap_err(),
            "lipschitz"
        ));
    }
    for growth in [1.0, 0.5, -2.0, f64::NAN, f64::INFINITY] {
        let options = ProximalOptions::new(1.0).with_backtracking(growth, Budget::ONE);
        assert!(invalid_argument(
            &ProximalGradient::new(options).unwrap_err(),
            "growth"
        ));
        assert!(invalid_argument(
            &Fista::new(options).unwrap_err(),
            "growth"
        ));
    }
    // The boundary just above one and the default policy are accepted.
    let accepted =
        ProximalOptions::new(f64::MIN_POSITIVE).with_backtracking(1.0 + f64::EPSILON, Budget::ONE);
    let solver = ProximalGradient::new(accepted)?;
    assert_eq!(
        solver.options().backtracking(),
        Some((1.0 + f64::EPSILON, Budget::ONE))
    );
    assert_eq!(
        Fista::new(ProximalOptions::new(1.0))?
            .options()
            .backtracking(),
        None
    );
    Ok(())
}

#[test]
fn proximal_workspace_capacity_tracks_its_dimension() -> Result<(), SciOxideError> {
    let solver = ProximalGradient::new(ProximalOptions::new(1.0))?;
    let workspace = solver.workspace::<f64>(2)?;
    let requirements = solver.workspace_requirements::<f64>(2)?;
    assert_eq!(requirements.dimension(), 2);
    assert_eq!(requirements.bytes(), 14 * core::mem::size_of::<f64>());
    assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    assert_eq!(
        workspace.check(&solver.workspace_requirements::<f64>(3)?)?,
        Capacity::NeedsGrowth
    );
    assert_eq!(
        workspace.check(&solver.workspace_requirements::<f64>(1)?)?,
        Capacity::NeedsGrowth
    );
    assert!(workspace.last_attempt().is_none());
    assert!(workspace.last_accepted().is_none());
    assert_eq!(
        ProximalWorkspace::<f64>::new().check(&requirements)?,
        Capacity::NeedsGrowth
    );
    Ok(())
}

#[test]
fn proximal_attempts_record_success_and_failure_and_preserve_output() -> Result<(), SciOxideError> {
    let solver = ProximalGradient::new(ProximalOptions::new(2.0))?;
    let mut workspace = solver.workspace::<f64>(1)?;
    let mut output = ProximalResult::empty();
    let prox = Shrink(1.0);
    let smooth = Counted::new(None);
    solver.minimize_into(
        &CompositeProblem::new(&smooth, &prox),
        &[0.0],
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert!(output.is_success(), "{output:?}");
    assert!((output.x()[0] - 2.5).abs() < 1e-12);
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Succeeded)
    );
    let progress = *workspace.last_attempt().and_then(|r| r.progress()).unwrap();
    assert_eq!(progress.iterations(), output.iterations());
    assert_eq!(
        progress.trust(),
        scioxide_core::PartialResultTrust::ResidualChecked
    );
    assert!(
        workspace
            .last_accepted()
            .is_some_and(|x| (x[0] - 2.5).abs() < 1e-12)
    );

    let snapshot = (
        output.x()[0].to_bits(),
        output.mapping_norm().map(f64::to_bits),
        output.lipschitz().to_bits(),
        output.iterations(),
        output.restarts(),
        output.termination(),
        output.counts(),
    );
    let same = |output: &ProximalResult<f64>| {
        snapshot
            == (
                output.x()[0].to_bits(),
                output.mapping_norm().map(f64::to_bits),
                output.lipschitz().to_bits(),
                output.iterations(),
                output.restarts(),
                output.termination(),
                output.counts(),
            )
            && output.x().len() == 1
    };

    // The second gradient fails after one complete step.
    let failing = Counted::new(Some(4));
    let error = solver
        .minimize_into(
            &CompositeProblem::new(&failing, &prox),
            &[0.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert_eq!(failing.calls.get(), 4);
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Failed)
    );
    assert!(
        workspace
            .last_accepted()
            .is_some_and(|x| (x[0] - 2.5).abs() < 1e-12)
    );
    assert!(same(&output));

    // A failure in the very first gradient accepts nothing.
    let first = Counted::new(Some(2));
    assert!(
        solver
            .minimize_into(
                &CompositeProblem::new(&first, &prox),
                &[0.0],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .is_err()
    );
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Failed)
    );
    assert!(workspace.last_accepted().is_none());
    assert!(same(&output));

    // Shape and non-finite input are rejected before any callback.
    let untouched = Counted::new(None);
    for x0 in [&[0.0, 0.0][..], &[f64::NAN][..]] {
        assert!(
            solver
                .minimize_into(
                    &CompositeProblem::new(&untouched, &prox),
                    x0,
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .is_err()
        );
        assert_eq!(
            outcome(workspace.last_attempt()),
            Some(AttemptOutcome::PreflightRejected)
        );
        assert!(workspace.last_accepted().is_none());
    }
    assert_eq!(untouched.calls.get(), 0);
    assert!(same(&output));
    Ok(())
}

#[test]
fn time_limit_without_clock_is_rejected_before_any_callback() -> Result<(), SciOxideError> {
    let smooth = Counted::new(None);
    let prox = Shrink(1.0);
    let problem = CompositeProblem::new(&smooth, &prox);
    let options = ProximalOptions::new(2.0).with_time_limit(Duration::from_secs(1));
    let mut output = ProximalResult::empty();
    for accelerated in [false, true] {
        let error = if accelerated {
            let solver = Fista::new(options)?;
            let mut workspace = solver.workspace(1)?;
            let error = solver
                .minimize_into(
                    &problem,
                    &[0.0],
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            assert_eq!(
                outcome(workspace.last_attempt()),
                Some(AttemptOutcome::PreflightRejected)
            );
            error
        } else {
            let solver = ProximalGradient::new(options)?;
            let mut workspace = solver.workspace(1)?;
            let error = solver
                .minimize_into(
                    &problem,
                    &[0.0],
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap_err();
            assert_eq!(
                outcome(workspace.last_attempt()),
                Some(AttemptOutcome::PreflightRejected)
            );
            error
        };
        assert_eq!(error.kind(), SciOxideErrorKind::CapabilityUnavailable);
    }
    assert_eq!(smooth.calls.get(), 0);
    assert_eq!(output.termination(), crate::OptimizeTermination::NotStarted);
    assert!(output.x().is_empty());

    let x_update = XUpdate::new(3.0);
    let z_update = ZUpdate(1.0);
    let (a, b, c) = unit_split();
    let problem = AdmmProblem::new(&a, &b, &c, &x_update, &z_update)?;
    let admm = Admm::new(AdmmOptions::new(1.0).with_time_limit(Duration::from_secs(1)))?;
    let mut workspace = admm.workspace(&problem)?;
    let mut output = AdmmResult::empty();
    let error = admm
        .solve_into(
            &problem,
            AdmmStart::new(&[0.0], &[0.0], &[0.0]),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::CapabilityUnavailable);
    assert_eq!(x_update.calls.get(), 0);
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::PreflightRejected)
    );
    assert_eq!(output.termination(), crate::OptimizeTermination::NotStarted);
    Ok(())
}

struct Scale<S: NumericScalar> {
    coefficient: S,
}
impl<S: NumericScalar> LinearOperator<S> for Scale<S> {
    fn rows(&self) -> usize {
        1
    }
    fn cols(&self) -> usize {
        1
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[S], y: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        y[0] = self.coefficient * x[0];
        Ok(())
    }
}
impl<S: NumericScalar> scioxide_core::AdjointOperator<S> for Scale<S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(&self, x: &[S], y: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        y[0] = self.coefficient.conj() * x[0];
        Ok(())
    }
}

/// Split `x - z = 0` of a scalar model.
fn unit_split() -> (Scale<f64>, Scale<f64>, [f64; 1]) {
    (
        Scale { coefficient: 1.0 },
        Scale { coefficient: -1.0 },
        [0.0],
    )
}

#[test]
fn admm_cancellation_stops_at_every_provider_boundary() -> Result<(), SciOxideError> {
    struct Provider<'a> {
        token: &'a scioxide_core::CancellationToken,
        calls: Cell<usize>,
        at: usize,
    }
    impl Provider<'_> {
        fn call(&self, output: &mut [f64]) {
            output.fill(0.0);
            self.calls.set(self.calls.get() + 1);
            if self.calls.get() == self.at {
                self.token.cancel();
            }
        }
    }
    impl LinearOperator<f64> for Provider<'_> {
        fn rows(&self) -> usize {
            1
        }
        fn cols(&self) -> usize {
            1
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(
            &self,
            _: &[f64],
            output: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.call(output);
            Ok(())
        }
    }
    impl scioxide_core::AdjointOperator<f64> for Provider<'_> {
        fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_adjoint_into(
            &self,
            _: &[f64],
            output: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.call(output);
            Ok(())
        }
    }
    impl AdmmBlockUpdate<f64> for Provider<'_> {
        fn update_into(
            &self,
            _: AdmmUpdateInput<'_, f64>,
            output: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<AdmmUpdateReport<f64>, SciOxideError> {
            self.call(output);
            Ok(AdmmUpdateReport::new(true, 0.0))
        }
    }
    for at in 1..=7 {
        let token = scioxide_core::CancellationToken::new();
        let provider = Provider {
            token: &token,
            calls: Cell::new(0),
            at,
        };
        let problem = AdmmProblem::new(&provider, &provider, &[0.0], &provider, &provider)?;
        let solver = Admm::new(AdmmOptions::new(1.0))?;
        let mut workspace = solver.workspace(&problem)?;
        let mut output = AdmmResult::empty();
        solver.solve_into(
            &problem,
            AdmmStart::new(&[0.0], &[0.0], &[0.0]),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token),
        )?;
        assert_eq!(output.termination(), crate::OptimizeTermination::Cancelled);
        assert_eq!(provider.calls.get(), at);
        assert_eq!(output.iterations(), 0);
        assert!(output.primal_norm().is_none());
    }
    Ok(())
}

#[test]
fn admm_residual_cancels_before_rounding() -> Result<(), SciOxideError> {
    struct Large;
    impl AdmmBlockUpdate<f64> for Large {
        fn update_into(
            &self,
            _: AdmmUpdateInput<'_, f64>,
            output: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<AdmmUpdateReport<f64>, SciOxideError> {
            output.fill(1e308);
            Ok(AdmmUpdateReport::new(true, 0.0))
        }
    }
    let identity = Scale { coefficient: 1.0 };
    let problem = AdmmProblem::new(&identity, &identity, &[1.7e308], &Large, &Large)?;
    let solver = Admm::new(AdmmOptions::new(0.1).with_max_iterations(Budget::ONE))?;
    let result = solver.solve(&problem, AdmmStart::new(&[0.0], &[0.0], &[0.0]))?;
    assert_eq!(
        result.termination(),
        crate::OptimizeTermination::IterationLimit
    );
    assert!((result.primal_norm().unwrap_or(f64::NAN) / 3e307 - 1.0).abs() < 1e-14);
    Ok(())
}

/// Exact minimizer of `0.5 |x-b|^2 + rho/2 |x - z + u|^2` for the split `x - z = 0`.
struct XUpdate<S: NumericScalar> {
    b: S,
    calls: Cell<usize>,
}
impl<S: NumericScalar> XUpdate<S> {
    fn new(b: S) -> Self {
        Self {
            b,
            calls: Cell::new(0),
        }
    }
}
impl<S: NumericScalar> AdmmBlockUpdate<S> for XUpdate<S> {
    fn update_into(
        &self,
        input: AdmmUpdateInput<'_, S>,
        output: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdmmUpdateReport<S::Real>, SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(input.block(), AdmmBlock::X);
        assert_eq!(input.current().len(), 1);
        assert_eq!(input.rhs().len(), input.operator().rows());
        assert_eq!(input.other_operator().cols(), input.other().len());
        let one = S::Real::ONE;
        let rho = input.rho();
        output[0] = (self.b + S::from_real(rho) * (input.other()[0] - input.scaled_dual()[0]))
            / S::from_real(one + rho);
        Ok(AdmmUpdateReport::new(true, S::Real::ZERO))
    }
}
struct ZUpdate<R: RealScalar>(R);
impl<S: NumericScalar> AdmmBlockUpdate<S> for ZUpdate<S::Real> {
    fn update_into(
        &self,
        input: AdmmUpdateInput<'_, S>,
        output: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdmmUpdateReport<S::Real>, SciOxideError> {
        assert_eq!(input.block(), AdmmBlock::Z);
        Shrink(self.0).apply_into(
            &[input.other()[0] + input.scaled_dual()[0]],
            S::Real::ONE / input.rho(),
            output,
        )?;
        Ok(AdmmUpdateReport::new(true, S::Real::ZERO))
    }
}

fn admm_fixture<S: NumericScalar>(b: S, precision: S::Real) -> Result<(), SciOxideError> {
    let one = S::Real::ONE;
    let a = Scale {
        coefficient: S::ONE,
    };
    let minus = Scale {
        coefficient: -S::ONE,
    };
    let c = [S::ZERO];
    let x_update = XUpdate::new(b);
    let z_update = ZUpdate(one);
    let problem = AdmmProblem::new(&a, &minus, &c, &x_update, &z_update)?;
    assert_eq!(
        (
            problem.x_dimension(),
            problem.z_dimension(),
            problem.constraints()
        ),
        (1, 1, 1)
    );
    let options = AdmmOptions::new(one)
        .with_max_iterations(Budget::try_from(200)?)
        .with_residual_tolerance(ResidualTolerance::new(precision, S::Real::ZERO)?);
    let result =
        Admm::new(options)?.solve(&problem, AdmmStart::new(&[S::ZERO], &[S::ZERO], &[S::ZERO]))?;
    assert!(result.is_success(), "{result:?}");
    let expected = S::from_real(one - one / b.abs()) * b;
    assert!((result.x()[0] - expected).abs() <= precision + precision);
    assert!((result.z()[0] - expected).abs() <= precision + precision);
    assert!(
        result
            .primal_residual()
            .is_some_and(|r| (r[0] - (result.x()[0] - result.z()[0])).abs() <= precision)
    );
    assert!(
        result
            .primal_norm()
            .is_some_and(|v| v <= precision + precision)
    );
    assert!(
        result
            .dual_norm()
            .is_some_and(|v| v <= precision + precision)
    );
    Ok(())
}

#[test]
fn admm_lasso_oracles_and_independent_residual() -> Result<(), SciOxideError> {
    admm_fixture::<f64>(3.0, 1e-8)?;
    admm_fixture::<f32>(3.0, 1e-4)?;
    admm_fixture::<Complex<f64>>(Complex { re: 3.0, im: 4.0 }, 1e-8)?;
    admm_fixture::<Complex<f32>>(Complex { re: 3.0, im: 4.0 }, 1e-4)
}

#[test]
fn inconsistent_split_never_claims_convergence() -> Result<(), SciOxideError> {
    struct Zero;
    impl AdmmBlockUpdate<f64> for Zero {
        fn update_into(
            &self,
            _: AdmmUpdateInput<'_, f64>,
            output: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<AdmmUpdateReport<f64>, SciOxideError> {
            output[0] = 0.0;
            Ok(AdmmUpdateReport::new(true, 0.0))
        }
    }
    let a = Scale {
        coefficient: 0.0_f64,
    };
    let b = Scale {
        coefficient: 0.0_f64,
    };
    let c = [1.0];
    let p = AdmmProblem::new(&a, &b, &c, &Zero, &Zero)?;
    let admm = Admm::new(AdmmOptions::new(1.0).with_max_iterations(Budget::try_from(5)?))?;
    let mut workspace = admm.workspace(&p)?;
    let mut result = AdmmResult::empty();
    admm.solve_into(
        &p,
        AdmmStart::new(&[0.0], &[0.0], &[0.0]),
        &mut result,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert_eq!(
        result.termination(),
        crate::OptimizeTermination::IterationLimit
    );
    assert_eq!(result.iterations(), 5);
    assert_eq!(result.primal_residual().map(<[f64]>::len), Some(1));
    assert!(
        result
            .primal_residual()
            .is_some_and(|r| (r[0] + 1.0).abs() <= 1e-12)
    );
    assert!(
        result
            .primal_norm()
            .is_some_and(|v| (v - 1.0).abs() <= 1e-12)
    );
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Stopped)
    );
    Ok(())
}

#[test]
fn dense_view_sparse_and_injected_context_agree() -> Result<(), SciOxideError> {
    let a = Matrix::from_vec(1, 1, vec![1.0_f64])?;
    let b = Matrix::from_vec(1, 1, vec![-1.0_f64])?;
    let a_view = a.view();
    let b_view = b.view();
    let sparse_a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![1.0])?;
    let sparse_b = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![-1.0])?;
    let c = [0.0];
    let options = AdmmOptions::new(1.0).with_residual_tolerance(ResidualTolerance::new(1e-8, 0.0)?);
    let solver = Admm::new(options)?;
    let x_update = XUpdate::new(3.0);
    let z_update = ZUpdate(1.0);
    let x_update: &dyn AdmmBlockUpdate<f64> = &x_update;
    let z_update: &dyn AdmmBlockUpdate<f64> = &z_update;
    let mut solutions = vec![];
    for (left, right) in [
        (
            &a.view().try_finite()? as &dyn scioxide_core::AdjointOperator<f64>,
            &b.view().try_finite()? as &dyn LinearOperator<f64>,
        ),
        (&a_view.try_finite()?, &b_view.try_finite()?),
        (&sparse_a, &sparse_b),
    ] {
        let problem = AdmmProblem::new(left, right, &c, x_update, z_update)?;
        let mut output = AdmmResult::empty();
        let mut workspace = solver.workspace(&problem)?;
        let executor = scioxide_core::SequentialExecutor;
        solver.solve_into(
            &problem,
            AdmmStart::new(&[0.0], &[0.0], &[0.0]),
            &mut output,
            &mut workspace,
            ExecutionContext::new(&executor),
        )?;
        assert!(output.is_success(), "{output:?}");
        assert!((output.x()[0] - 2.0).abs() < 1e-7);
        let owned = solver.solve(&problem, AdmmStart::new(&[0.0], &[0.0], &[0.0]))?;
        assert_eq!(owned.iterations(), output.iterations());
        assert!((owned.x()[0] - output.x()[0]).abs() <= 1e-15);
        solutions.push((output.x()[0], output.z()[0], output.iterations()));
    }
    for &(x, z, iterations) in &solutions[1..] {
        assert!((x - solutions[0].0).abs() <= 1e-15);
        assert!((z - solutions[0].1).abs() <= 1e-15);
        assert_eq!(iterations, solutions[0].2);
    }
    Ok(())
}

#[test]
fn admm_plan_constructor_validates_the_policy() -> Result<(), SciOxideError> {
    for rho in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error = Admm::new(AdmmOptions::new(rho)).unwrap_err();
        assert!(invalid_argument(&error, "rho"), "{error:?}");
    }
    for (ratio, factor) in [
        (1.0, 2.0),
        (2.0, 1.0),
        (0.5, 2.0),
        (2.0, f64::NAN),
        (f64::INFINITY, 2.0),
    ] {
        let error =
            Admm::new(AdmmOptions::new(1.0).with_residual_balancing(ratio, factor)).unwrap_err();
        assert_eq!(
            error.kind(),
            SciOxideErrorKind::InvalidInput,
            "{ratio} {factor}"
        );
    }
    let accepted =
        Admm::new(AdmmOptions::new(1.0).with_residual_balancing(1.0 + f64::EPSILON, 2.0))?;
    assert_eq!(
        accepted.options().residual_balancing(),
        Some((1.0 + f64::EPSILON, 2.0))
    );
    Ok(())
}

#[test]
fn residual_balancing_reaches_the_same_split_solution() -> Result<(), SciOxideError> {
    let (a, b, c) = unit_split();
    let x_update = XUpdate::new(3.0);
    let z_update = ZUpdate(1.0);
    let problem = AdmmProblem::new(&a, &b, &c, &x_update, &z_update)?;
    let admm = Admm::new(
        AdmmOptions::new(100.0)
            .with_residual_balancing(2.0, 2.0)
            .with_residual_tolerance(ResidualTolerance::new(1e-9, 0.0)?),
    )?;
    let result = admm.solve(&problem, AdmmStart::new(&[0.0], &[0.0], &[0.0]))?;
    assert!(result.is_success(), "{result:?}");
    assert!(result.rho() < 100.0);
    assert!((result.x()[0] - 2.0).abs() < 1e-7);
    // y = rho*u is the multiplier 1 of the soft-threshold optimality condition.
    assert!((result.rho() * result.scaled_dual()[0] - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn admm_workspace_capacity_tracks_the_problem_shape() -> Result<(), SciOxideError> {
    let (a, b, c) = unit_split();
    let x_update = XUpdate::new(3.0);
    let z_update = ZUpdate(1.0);
    let problem = AdmmProblem::new(&a, &b, &c, &x_update, &z_update)?;
    let admm = Admm::new(AdmmOptions::new(1.0))?;
    let workspace = admm.workspace(&problem)?;
    let requirements = admm.workspace_requirements(&problem)?;
    assert_eq!(
        (
            requirements.x_dimension(),
            requirements.z_dimension(),
            requirements.constraints()
        ),
        (1, 1, 1)
    );
    assert!(requirements.bytes() > 0);
    assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    assert!(workspace.last_attempt().is_none());

    let wide = Matrix::from_vec(1, 2, vec![1.0_f64, 1.0])?;
    let wide = wide.view().try_finite()?;
    let wide_problem = AdmmProblem::new(&wide, &b, &c, &x_update, &z_update)?;
    assert_eq!(
        workspace.check(&admm.workspace_requirements(&wide_problem)?)?,
        Capacity::NeedsGrowth
    );
    assert_eq!(
        AdmmWorkspace::<f64>::new().check(&requirements)?,
        Capacity::NeedsGrowth
    );
    // Mismatched starting blocks are rejected before any block update.
    let mut workspace = workspace;
    let mut output = AdmmResult::empty();
    assert_eq!(
        admm.solve_into(
            &wide_problem,
            AdmmStart::new(&[0.0], &[0.0], &[0.0]),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::PreflightRejected)
    );
    assert_eq!(x_update.calls.get(), 0);
    Ok(())
}

struct Bad;
impl AdmmBlockUpdate<f64> for Bad {
    fn update_into(
        &self,
        _: AdmmUpdateInput<'_, f64>,
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdmmUpdateReport<f64>, SciOxideError> {
        output[0] = 123.0;
        Ok(AdmmUpdateReport::new(false, 1.0))
    }
}

#[test]
fn bad_block_report_preserves_caller_result() -> Result<(), SciOxideError> {
    let (a, b, c) = unit_split();
    let z_update = ZUpdate(1.0);
    let p = AdmmProblem::new(&a, &b, &c, &Bad, &z_update)?;
    let solver = Admm::new(AdmmOptions::new(1.0))?;
    let mut result = AdmmResult::empty();
    let mut workspace = solver.workspace(&p)?;
    let error = solver
        .solve_into(
            &p,
            AdmmStart::new(&[0.0], &[0.0], &[0.0]),
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(invalid_argument(&error, "x_update"), "{error:?}");
    assert_eq!(result.termination(), crate::OptimizeTermination::NotStarted);
    assert!(result.x().is_empty());
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Failed)
    );
    assert!(workspace.last_accepted().is_none());
    Ok(())
}

#[test]
fn admm_error_leaves_a_published_result_bit_identical() -> Result<(), SciOxideError> {
    let (a, b, c) = unit_split();
    let x_update = XUpdate::new(3.0);
    let z_update = ZUpdate(1.0);
    let good = AdmmProblem::new(&a, &b, &c, &x_update, &z_update)?;
    let bad = AdmmProblem::new(&a, &b, &c, &x_update, &Bad)?;
    let solver = Admm::new(
        AdmmOptions::new(1.0).with_residual_tolerance(ResidualTolerance::new(1e-8, 0.0)?),
    )?;
    let mut workspace = solver.workspace(&good)?;
    let mut output = AdmmResult::empty();
    let start = AdmmStart::new(&[0.0], &[0.0], &[0.0]);
    solver.solve_into(
        &good,
        start,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert!(output.is_success(), "{output:?}");
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Succeeded)
    );
    let progress = *workspace.last_attempt().and_then(|r| r.progress()).unwrap();
    assert_eq!(progress.iterations(), output.iterations());
    assert_eq!(
        output.primal_norm().map(f64::to_bits),
        Some(progress.primal_norm().to_bits())
    );
    assert!(
        workspace
            .last_accepted()
            .is_some_and(|(x, z)| x == output.x() && z == output.z())
    );

    let bits = |values: &[f64]| {
        values
            .iter()
            .map(|v| v.to_bits())
            .collect::<alloc::vec::Vec<_>>()
    };
    let snapshot = |output: &AdmmResult<f64>| {
        (
            bits(output.x()),
            bits(output.z()),
            bits(output.scaled_dual()),
            output.primal_residual().map(bits),
            output.dual_residual().map(bits),
            output.primal_norm().map(f64::to_bits),
            output.dual_norm().map(f64::to_bits),
            output.rho().to_bits(),
            output.iterations(),
            output.termination(),
            output.counts(),
        )
    };
    let before = snapshot(&output);
    let calls = x_update.calls.get();
    let error = solver
        .solve_into(
            &bad,
            start,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(invalid_argument(&error, "z_update"), "{error:?}");
    assert_eq!(x_update.calls.get(), calls + 1);
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Failed)
    );
    assert!(workspace.last_accepted().is_none());
    assert_eq!(snapshot(&output), before);
    Ok(())
}

#[test]
fn constrained_quadratic_reaches_the_fixed_constraint() -> Result<(), SciOxideError> {
    struct Fixed;
    impl AdmmBlockUpdate<f64> for Fixed {
        fn update_into(
            &self,
            input: AdmmUpdateInput<'_, f64>,
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<AdmmUpdateReport<f64>, SciOxideError> {
            assert_eq!(input.block(), AdmmBlock::Z);
            out[0] = 1.0;
            Ok(AdmmUpdateReport::new(true, 0.0))
        }
    }
    let (a, b, c) = unit_split();
    let x_update = XUpdate::new(2.0);
    let p = AdmmProblem::new(&a, &b, &c, &x_update, &Fixed)?;
    let solver = Admm::new(
        AdmmOptions::new(1.0).with_residual_tolerance(ResidualTolerance::new(1e-8, 0.0)?),
    )?;
    let result = solver.solve(&p, AdmmStart::new(&[0.0], &[0.0], &[0.0]))?;
    assert!(result.is_success(), "{result:?}");
    assert!((result.x()[0] - 1.0).abs() < 1e-8);
    assert!((result.z()[0] - 1.0).abs() < 1e-8);
    // y = rho*u = 1 is the original equality multiplier for this model.
    assert!((result.rho() * result.scaled_dual()[0] - 1.0).abs() < 1e-7);
    Ok(())
}

#[test]
fn evaluation_budget_preserves_only_complete_steps() -> Result<(), SciOxideError> {
    let smooth = Quadratic {
        b: 3.0_f64,
        curvature: 2.0,
    };
    let prox = Shrink(1.0);
    let pg = ProximalGradient::new(ProximalOptions::new(2.0))?;
    let mut pg_result = ProximalResult::empty();
    let mut pg_workspace = pg.workspace::<f64>(1)?;
    with_parent(evaluation_budget(Budget::try_from(3)?), |parent| {
        pg.minimize_into(
            &CompositeProblem::new(&smooth, &prox),
            &[0.0],
            &mut pg_result,
            &mut pg_workspace,
            &mut *parent,
        )?;
        assert_eq!(parent.counts().own().total_providers()?, 0);
        assert_eq!(parent.counts().nested().total_providers()?, 3);
        Ok(())
    })?;
    assert_eq!(
        pg_result.termination(),
        crate::OptimizeTermination::EvaluationLimit
    );
    assert_eq!(pg_result.iterations(), 1);
    assert!((pg_result.x()[0] - 2.5).abs() < 1e-12);
    assert!(
        pg_result
            .mapping_norm()
            .is_some_and(|norm| (norm - 5.0).abs() < 1e-12)
    );
    assert_eq!(
        outcome(pg_workspace.last_attempt()),
        Some(AttemptOutcome::Stopped)
    );
    let progress = *pg_workspace
        .last_attempt()
        .and_then(|r| r.progress())
        .unwrap();
    assert_eq!(progress.iterations(), 1);
    assert_eq!(
        progress.trust(),
        scioxide_core::PartialResultTrust::Admissible
    );
    assert!(
        pg_workspace
            .last_accepted()
            .is_some_and(|x| (x[0] - 2.5).abs() < 1e-12)
    );

    let (a, b, c) = unit_split();
    let x_update = XUpdate::new(3.0);
    let z_update = ZUpdate(1.0);
    let p = AdmmProblem::new(&a, &b, &c, &x_update, &z_update)?;
    let admm = Admm::new(AdmmOptions::new(1.0))?;
    let mut result = AdmmResult::empty();
    let mut workspace = admm.workspace(&p)?;
    with_parent(evaluation_budget(Budget::try_from(7)?), |parent| {
        admm.solve_into(
            &p,
            AdmmStart::new(&[0.0], &[0.0], &[0.0]),
            &mut result,
            &mut workspace,
            parent,
        )
    })?;
    assert_eq!(
        result.termination(),
        crate::OptimizeTermination::EvaluationLimit
    );
    assert_eq!(result.iterations(), 1);
    assert!((result.x()[0] - 1.5).abs() < 1e-12);
    assert!((result.z()[0] - 0.5).abs() < 1e-12);
    assert!(
        result
            .primal_residual()
            .is_some_and(|r| (r[0] - 1.0).abs() < 1e-12)
    );
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Stopped)
    );
    assert!(
        workspace
            .last_accepted()
            .is_some_and(|(x, z)| (x[0] - 1.5).abs() < 1e-12 && (z[0] - 0.5).abs() < 1e-12)
    );
    Ok(())
}

#[test]
fn parent_iteration_budget_does_not_stop_a_child_solve() -> Result<(), SciOxideError> {
    let parent_budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let smooth = Quadratic {
        b: 3.0_f64,
        curvature: 2.0,
    };
    let prox = Shrink(1.0);
    let pg = ProximalGradient::new(ProximalOptions::new(2.0))?;
    let mut output = ProximalResult::empty();
    let mut workspace = pg.workspace::<f64>(1)?;
    with_parent(parent_budgets, |parent| {
        pg.minimize_into(
            &CompositeProblem::new(&smooth, &prox),
            &[0.0],
            &mut output,
            &mut workspace,
            parent,
        )
    })?;
    assert!(output.is_success(), "{output:?}");
    assert_eq!(output.iterations(), 2);

    // The solver's own allowance does stop it.
    let limited =
        ProximalGradient::new(ProximalOptions::new(2.0).with_max_iterations(Budget::ONE))?;
    limited.minimize_into(
        &CompositeProblem::new(&smooth, &prox),
        &[0.0],
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert_eq!(
        output.termination(),
        crate::OptimizeTermination::IterationLimit
    );
    assert_eq!(output.iterations(), 1);
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Stopped)
    );

    let (a, b, c) = unit_split();
    let x_update = XUpdate::new(3.0);
    let z_update = ZUpdate(1.0);
    let p = AdmmProblem::new(&a, &b, &c, &x_update, &z_update)?;
    let admm = Admm::new(
        AdmmOptions::new(1.0).with_residual_tolerance(ResidualTolerance::new(1e-8, 0.0)?),
    )?;
    let mut result = AdmmResult::empty();
    let mut admm_workspace = admm.workspace(&p)?;
    with_parent(parent_budgets, |parent| {
        admm.solve_into(
            &p,
            AdmmStart::new(&[0.0], &[0.0], &[0.0]),
            &mut result,
            &mut admm_workspace,
            parent,
        )
    })?;
    assert!(result.is_success(), "{result:?}");
    assert!(result.iterations() > 1);
    Ok(())
}

#[test]
fn unfinished_first_step_has_no_residual_observation() -> Result<(), SciOxideError> {
    let solver = ProximalGradient::new(ProximalOptions::new(1.0))?;
    let mut output = ProximalResult::empty();
    let mut workspace = solver.workspace::<f64>(1)?;
    let smooth = Quadratic {
        b: 3.0,
        curvature: 1.0,
    };
    let prox = Shrink(1.0);
    with_parent(evaluation_budget(Budget::ONE), |parent| {
        solver.minimize_into(
            &CompositeProblem::new(&smooth, &prox),
            &[0.0],
            &mut output,
            &mut workspace,
            parent,
        )
    })?;
    assert_eq!(output.iterations(), 0);
    assert_eq!(output.mapping_norm(), None);
    assert_eq!(
        output.termination(),
        crate::OptimizeTermination::EvaluationLimit
    );
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Stopped)
    );
    assert!(workspace.last_accepted().is_none());

    let (a, b, c) = unit_split();
    let x_update = XUpdate::new(3.0);
    let z_update = ZUpdate(1.0);
    let p = AdmmProblem::new(&a, &b, &c, &x_update, &z_update)?;
    let solver = Admm::new(AdmmOptions::new(1.0))?;
    let mut output = AdmmResult::empty();
    let mut workspace = solver.workspace(&p)?;
    with_parent(evaluation_budget(Budget::ONE), |parent| {
        solver.solve_into(
            &p,
            AdmmStart::new(&[0.0], &[0.0], &[0.0]),
            &mut output,
            &mut workspace,
            parent,
        )
    })?;
    assert_eq!(output.iterations(), 0);
    assert_eq!(output.primal_residual(), None);
    assert_eq!(output.dual_norm(), None);
    assert_eq!(
        output.termination(),
        crate::OptimizeTermination::EvaluationLimit
    );
    assert_eq!(
        outcome(workspace.last_attempt()),
        Some(AttemptOutcome::Stopped)
    );
    assert!(workspace.last_accepted().is_none());
    Ok(())
}
