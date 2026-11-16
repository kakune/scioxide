//! Method-level step contracts: estimator powers, step arithmetic, inner limits, FSAL
//! reuse, multistep restarts and derivative-policy cache keys.

use core::cell::Cell;

use scioxide_core::{
    Budget, ConvergenceReason, EvaluationRole, NumericScalar, RealScalar, SciOxideError,
    SolveContext,
};
use scioxide_linalg::{MatrixViewMut, Vector};

use super::{Decay, prepared};
use crate::stepper::testing::with_context;
use crate::stepper::{StepRequest, StepTrace, Stepper};
use crate::{
    AdamsBashforth, AdamsMoulton, BackwardEuler, Bdf, DormandPrince5, DormandPrince853,
    ForwardEuler, NewtonMode, OdeSystem, ProportionalIntegralController, RadauIia, Rkf45, Sdirk,
    Verner65,
};

fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}

fn estimator<M: Stepper<f64, f64>>(solver: M, expected: usize) {
    let mut workspace = prepared(&solver, 2);
    let mut y = Vector::from_vec(vec![0., 1.]);
    let trace = StepTrace::default();
    let order = with_context(|cx| {
        let result = solver.step(
            &Decay(2),
            0.,
            &mut y,
            0.1,
            StepRequest::new(&trace),
            &mut workspace,
            cx,
        )?;
        Ok(result.error_order)
    })
    .unwrap();
    assert_eq!(order, Some(expected));
    let mut controller = crate::step_control::PiController::new(
        ProportionalIntegralController::new(0.9, 5., 0.2, 1.0, 0.0).unwrap(),
    );
    let (next, accept) = controller.compute_step(0.5, 0.1, expected).unwrap();
    assert!(accept);
    #[allow(clippy::cast_precision_loss)]
    let exponent = -1. / expected as f64;
    assert!((next - 0.09 * 0.5_f64.powf(exponent)).abs() < 1e-15);
}

#[test]
fn embedded_pairs_supply_the_estimator_power() {
    estimator(Rkf45::new(), 5);
    estimator(DormandPrince5::new(), 5);
    estimator(DormandPrince853::new(), 8);
    estimator(Verner65::new(), 6);
    estimator(RadauIia::new(), 4);
}

struct Counting<'a>(&'a Cell<usize>);

impl OdeSystem<f64> for Counting<'_> {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        derivative: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.0.set(self.0.get() + 1);
        derivative.fill(0.0);
        Ok(())
    }
}

#[test]
fn direct_steps_reject_invalid_time_arithmetic_before_rhs() {
    let calls = Cell::new(0);
    let system = Counting(&calls);
    let trace = StepTrace::default();

    let explicit = ForwardEuler::new();
    let mut explicit_workspace = prepared(&explicit, 1);
    for (time, step, state) in [(f64::NAN, 0.1, vec![0.0]), (0.0, 0.0, vec![0.0])] {
        let mut state = Vector::from_vec(state);
        let result = with_context(|cx| {
            explicit
                .step(
                    &system,
                    time,
                    &mut state,
                    step,
                    StepRequest::new(&trace),
                    &mut explicit_workspace,
                    cx,
                )
                .map(drop)
        });
        assert!(result.is_err());
    }

    let multistep = Bdf::new(2).unwrap();
    let mut multistep_workspace = prepared(&multistep, 1);
    let mut state = Vector::from_vec(vec![0.0]);
    let result = with_context(|cx| {
        multistep
            .step(
                &system,
                0.0,
                &mut state,
                f64::INFINITY,
                StepRequest::new(&trace),
                &mut multistep_workspace,
                cx,
            )
            .map(drop)
    });
    assert!(result.is_err());
    assert_eq!(calls.get(), 0);
}

/// `y' = -10 y^3` with its analytic Jacobian.
struct Cubic;

impl OdeSystem<f64> for Cubic {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        o: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        o[0] = -10. * y[0].powi(3);
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, -30. * y[0] * y[0])?;
        Ok(Some(()))
    }
}

#[test]
fn full_sdirk_honors_the_inner_iteration_limit() {
    for mode in [NewtonMode::Full, NewtonMode::Simplified] {
        let method = Sdirk::new().with_newton_mode(mode);
        let mut workspace = prepared(&method, 1);
        let mut y = Vector::from_vec(vec![1.]);
        let trace = StepTrace::default();
        let request = StepRequest {
            inner_limit: Budget::ONE,
            ..StepRequest::new(&trace)
        };
        let error = with_context(|cx| {
            method
                .step(&Cubic, 0., &mut y, 0.5, request, &mut workspace, cx)
                .map(drop)
        })
        .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: ConvergenceReason::IterationLimit,
                ..
            }
        ));
        assert_eq!(y.as_slice(), &[1.]);
    }
}

/// `y' = rate * y` in every component, with its constant analytic Jacobian.
struct Linear<S> {
    rate: S,
    dimension: usize,
}

impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for Linear<S> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn rhs(
        &self,
        _: S,
        y: &[S],
        out: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.iter_mut()
            .zip(y)
            .for_each(|(out, y)| *out = self.rate * *y);
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[S],
        out: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                out.set(row, column, if row == column { self.rate } else { S::ZERO })?;
            }
        }
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

/// Run one step and return the RHS evaluations it counted.
#[allow(clippy::too_many_arguments)]
fn metered_step<S, M>(
    solver: &M,
    accepted_step: Option<usize>,
    system: &Linear<S>,
    t: S,
    y: &mut Vector<S>,
    h: S,
    workspace: &mut M::Workspace,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(usize, Option<S>), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    M: Stepper<S, S>,
{
    let trace = StepTrace::default();
    let before = cx.counts().own().provider(EvaluationRole::RightHandSide);
    let result = solver.step(
        system,
        t,
        y,
        h,
        StepRequest {
            accepted_step,
            ..StepRequest::new(&trace)
        },
        workspace,
        cx,
    )?;
    let error = result.raw_error.map(|error| error[0]);
    let after = cx.counts().own().provider(EvaluationRole::RightHandSide);
    Ok((after - before, error))
}

fn raw_error_and_fsal<S: RealScalar + NumericScalar<Real = S>>() {
    let system = Linear {
        rate: scalar::<S>(1.0),
        dimension: 1,
    };
    let other = Linear {
        rate: scalar::<S>(-1.0),
        dimension: 1,
    };
    let solver = DormandPrince5::<S>::new();
    let mut workspace = prepared(&solver, 1);
    let h = scalar::<S>(0.5);
    let mut y = Vector::from_vec(vec![scalar::<S>(1.0)]);
    let evaluations = with_context(|cx| {
        let (first, error) = metered_step(
            &solver,
            Some(0),
            &system,
            scalar(0.0),
            &mut y,
            h,
            &mut workspace,
            cx,
        )?;
        // Exact rational expansion of the two stability polynomials on y'=y.
        let expected = h
            * h
            * h
            * h
            * h
            * (scalar::<S>(-97.0 / 120000.0)
                + h * (scalar::<S>(13.0 / 40000.0) - h * scalar::<S>(1.0 / 24000.0)));
        assert!(NumericScalar::abs(error.unwrap() - expected) < scalar::<S>(4.0) * S::EPSILON);
        solver.accept(&mut workspace)?;
        let (second, _) =
            metered_step(&solver, Some(1), &system, h, &mut y, h, &mut workspace, cx)?;
        solver.accept(&mut workspace)?;
        // A new epoch (here for another system) restarts the FSAL derivative.
        solver.begin(1, false, &mut workspace)?;
        let (third, _) = metered_step(
            &solver,
            Some(0),
            &other,
            h + h,
            &mut y,
            h,
            &mut workspace,
            cx,
        )?;
        solver.discard(&mut workspace)?;
        let (fourth, _) = metered_step(
            &solver,
            None,
            &other,
            h + h + h,
            &mut y,
            h,
            &mut workspace,
            cx,
        )?;
        Ok([first, second, third, fourth])
    })
    .unwrap();
    assert_eq!(evaluations, [7, 6, 7, 7]);
}

fn restart_matches_fresh<S, M>(solver: M)
where
    S: RealScalar + NumericScalar<Real = S>,
    M: Stepper<S, S>,
{
    let system = Linear {
        rate: scalar::<S>(-1.0),
        dimension: 1,
    };
    let trace = StepTrace::default();
    let request = StepRequest::new(&trace);
    let mut workspace = prepared(&solver, 1);
    let mut y = Vector::from_vec(vec![scalar::<S>(1.0)]);
    let h = scalar::<S>(0.125);
    let mut t = scalar::<S>(0.0);
    with_context(|cx| {
        for _ in 0..8 {
            solver.step(&system, t, &mut y, h, request, &mut workspace, cx)?;
            solver.accept(&mut workspace)?;
            t += h;
        }
        // Each direction reversal invalidates history. Same-direction changes are
        // independently covered by the variable-step multistep tests.
        for changed in [scalar::<S>(-0.0625), scalar(0.0625)] {
            let mut fresh_workspace = prepared(&solver, 1);
            let mut expected = y.clone();
            solver.step(
                &system,
                t,
                &mut expected,
                changed,
                request,
                &mut fresh_workspace,
                cx,
            )?;
            solver.step(&system, t, &mut y, changed, request, &mut workspace, cx)?;
            assert_eq!(y.as_slice(), expected.as_slice());
            solver.accept(&mut workspace)?;
            t += changed;
        }
        // Beginning an epoch resets history even when the dimension matches.
        solver.begin(1, false, &mut workspace)?;
        let mut fresh_workspace = prepared(&solver, 1);
        let mut expected = y.clone();
        solver.step(
            &system,
            t,
            &mut expected,
            h,
            request,
            &mut fresh_workspace,
            cx,
        )?;
        solver.step(&system, t, &mut y, h, request, &mut workspace, cx)?;
        assert_eq!(y.as_slice(), expected.as_slice());
        Ok(())
    })
    .unwrap();
}

fn evidence<S: RealScalar + NumericScalar<Real = S>>() {
    raw_error_and_fsal::<S>();
    restart_matches_fresh::<S, _>(AdamsBashforth::new(4).unwrap());
    restart_matches_fresh::<S, _>(AdamsMoulton::new(4).unwrap());
    restart_matches_fresh::<S, _>(Bdf::new(3).unwrap());
}

#[test]
fn f32_epoch_and_error_evidence() {
    evidence::<f32>();
}

#[test]
fn f64_epoch_and_error_evidence() {
    evidence::<f64>();
}

#[test]
fn changed_derivative_policy_cannot_reuse_a_constant_analytic_jacobian() {
    let system = Linear {
        rate: -1.0,
        dimension: 1,
    };
    let analytic = BackwardEuler::new()
        .with_derivative_strategy(scioxide_differentiate::DerivativeStrategy::Analytic);
    let mut workspace = prepared(&analytic, 1);
    let mut state = Vector::from_vec(vec![1.0]);
    let trace = StepTrace::default();
    let request = StepRequest::new(&trace);
    let automatic = analytic
        .clone()
        .with_derivative_strategy(scioxide_differentiate::DerivativeStrategy::Automatic);
    let mut accepted = Vector::from_vec(vec![0.0]);
    let failed = with_context(|cx| {
        analytic.step(&system, 0.0, &mut state, 0.125, request, &mut workspace, cx)?;
        analytic.accept(&mut workspace)?;
        accepted.as_mut_slice().copy_from_slice(state.as_slice());
        // This provider has no automatic Jacobian: a cached analytic Jacobian must
        // never turn the unsupported request into a successful step.
        Ok(automatic
            .step(
                &system,
                0.125,
                &mut state,
                0.125,
                request,
                &mut workspace,
                cx,
            )
            .is_err())
    })
    .unwrap();
    assert!(failed);
    assert_eq!(state, accepted);
}
