//! Jacobian acquisition by derivative strategy on dense, sparse and Krylov stages.
#![allow(clippy::unwrap_used, clippy::needless_range_loop)]
use scioxide_core::{
    Budget, ErrorTolerance, EvaluationRole, ExecutionContext, InvalidInputReason, NormSelection,
    SciOxideError, SciOxideErrorView, SolveContext, StructuralPattern, StructuralPatternView,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::*;
use scioxide_sparse::SparseLuLimits;
use std::cell::Cell;

/// A stiff nonlinear pair with exact solution `(cos t, e^-t)`:
/// `y₀' = -10⁴(1 + y₁²)(y₀ - cos t) - sin t`, `y₁' = -y₁`.
struct Stiff {
    analytic: bool,
}
impl OdeSystem<f64> for Stiff {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        f: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        f[0] = -1e4 * (1. + y[1] * y[1]) * (y[0] - t.cos()) - t.sin();
        f[1] = -y[1];
        Ok(())
    }
    fn jacobian(
        &self,
        t: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        if !self.analytic {
            return Ok(None);
        }
        j.set(0, 0, -1e4 * (1. + y[1] * y[1]))?;
        j.set(0, 1, -2e4 * y[1] * (y[0] - t.cos()))?;
        j.set(1, 0, 0.)?;
        j.set(1, 1, -1.)?;
        Ok(Some(()))
    }
}

/// Step policy of one integration from `(0, y0)` with steps of at most `h`.
#[derive(Clone, Copy)]
enum Steps {
    /// Exactly this many accepted steps of `h`, at stage accuracy `√ε`.
    Pinned(u32),
    /// Exactly this many accepted steps of `h` under a looser error control that
    /// accepts every pinned step of an adaptive method.
    PinnedAt(u32, f64),
    /// Controller-selected steps over this many multiples of `h`, at this tolerance.
    Adaptive(u32, f64),
}

/// Integrate from `(0, y0)` under `steps`. Pinned bounds fix every proposal to `h`;
/// the maximum leaves a few ulps of the end time so endpoint rounding never splits off
/// an extra step. Returns the endpoint and all RHS evaluations, including nested
/// difference products.
fn integrate<M, Sys>(
    method: M,
    system: &Sys,
    y0: Vec<f64>,
    h: f64,
    steps: Steps,
) -> Result<(Vec<f64>, usize), SciOxideError>
where
    M: OdeMethod<f64> + Clone,
    Sys: OdeSystem<f64> + ?Sized,
{
    let scalar = |t: f64| ErrorTolerance::scalar(t, t, NormSelection::RootMeanSquare).unwrap();
    let (count, tolerance, pinned) = match steps {
        Steps::Pinned(count) => (count, ErrorTolerance::default(), true),
        Steps::PinnedAt(count, tolerance) => (count, scalar(tolerance), true),
        Steps::Adaptive(count, tolerance) => (count, scalar(tolerance), false),
    };
    let end = f64::from(count) * h;
    let options = OdeOptions::new()
        .with_error_tolerance(tolerance)
        .with_maximum_step(h + 4. * f64::EPSILON * end.max(1.));
    let options = if pinned {
        options.with_initial_step(h).with_minimum_step(h)
    } else {
        options
    };
    let plan = (method, options);
    let problem = OdeProblem::new(system, end);
    let mut workspace = plan.0.workspace(&problem, &plan.1)?;
    let mut state = OdeState::new(0., Vector::from_vec(y0));
    let report = plan.0.advance(
        &problem,
        &plan.1,
        &mut state,
        &mut (),
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    if pinned {
        assert_eq!(report.accepted_steps(), count as usize);
        assert_eq!(report.rejected_steps(), 0);
    }
    let evaluations = report
        .accounting()
        .total()?
        .provider(EvaluationRole::RightHandSide);
    Ok((state.values().to_vec(), evaluations))
}

/// Fixed steps of `h = 1/100` on [0, 1]; returns the endpoint and RHS evaluations.
fn fixed<M: OdeMethod<f64> + Clone>(
    method: M,
    system: &dyn OdeSystem<f64>,
    steps: Steps,
) -> (Vec<f64>, usize) {
    integrate(method, system, vec![1., 1.], 0.01, steps).unwrap()
}

/// Rosenbrock-W has no Newton iteration; its embedded estimate at `h = 1/100` needs
/// `10⁻⁵` error control to accept every pinned step.
const ROSENBROCK_STIFF: Steps = Steps::PinnedAt(100, 1e-5);

#[test]
fn rhs_only_stiff_stages_use_finite_difference_jacobians() {
    let exact = [1f64.cos(), (-1f64).exp()];
    let analytic = Stiff { analytic: true };
    let rhs_only = Stiff { analytic: false };
    let check = |name: &str,
                 (y, calls): (Vec<f64>, usize),
                 (reference, analytic_calls): (Vec<f64>, usize),
                 bound: f64| {
        // Difference Jacobians cost right-hand-side evaluations but not accuracy.
        assert!(
            calls > analytic_calls,
            "{name}: {calls} <= {analytic_calls}"
        );
        for ((&value, &expected), &exact) in y.iter().zip(&reference).zip(&exact) {
            assert!(
                (value - expected).abs() < 1e-8,
                "{name}: {value} vs {expected}"
            );
            assert!((value - exact).abs() < bound, "{name}: {value} vs {exact}");
        }
    };
    for strategy in [
        DerivativeStrategy::AnalyticOrAutomaticOrForward,
        DerivativeStrategy::Forward,
        DerivativeStrategy::Central,
    ] {
        check(
            "RadauIia",
            fixed(
                RadauIia::new().with_derivative_strategy(strategy),
                &rhs_only,
                Steps::Pinned(100),
            ),
            fixed(RadauIia::new(), &analytic, Steps::Pinned(100)),
            1e-7,
        );
        check(
            "Sdirk",
            fixed(
                Sdirk::new().with_derivative_strategy(strategy),
                &rhs_only,
                Steps::Pinned(100),
            ),
            fixed(Sdirk::new(), &analytic, Steps::Pinned(100)),
            1e-5,
        );
        check(
            "Bdf",
            fixed(
                Bdf::new(2).unwrap().with_derivative_strategy(strategy),
                &rhs_only,
                Steps::Pinned(100),
            ),
            fixed(Bdf::new(2).unwrap(), &analytic, Steps::Pinned(100)),
            1e-3,
        );
        check(
            "RosenbrockW",
            fixed(
                RosenbrockW::new(RosenbrockMethod::Ros34Pw2)
                    .unwrap()
                    .with_derivative_strategy(strategy),
                &TimeDerivative(&rhs_only),
                ROSENBROCK_STIFF,
            ),
            fixed(
                RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap(),
                &TimeDerivative(&analytic),
                ROSENBROCK_STIFF,
            ),
            1e-5,
        );
    }
}

/// Adds the exact partial time derivative required by Rosenbrock-W.
struct TimeDerivative<'a>(&'a Stiff);
impl OdeSystem<f64> for TimeDerivative<'_> {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        f: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.0.rhs(t, y, f, cx)
    }
    fn jacobian(
        &self,
        t: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.0.jacobian(t, y, j, cx)
    }
    fn time_derivative(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out[0] = -1e4 * (1. + y[1] * y[1]) * t.sin() - t.cos();
        out[1] = 0.;
        Ok(Some(()))
    }
}

/// Reaction–diffusion `y' = a(y_{i-1} - 2y_i + y_{i+1}) - y_i³` on a tridiagonal pattern,
/// with an optional counted analytic sparse Jacobian.
struct Reaction {
    n: usize,
    sparse: bool,
    calls: Cell<usize>,
}
impl Reaction {
    fn new(n: usize, sparse: bool) -> Self {
        Self {
            n,
            sparse,
            calls: Cell::new(0),
        }
    }
    fn pattern(&self) -> StructuralPattern {
        let mut rows = vec![0];
        let mut cols = vec![];
        for row in 0..self.n {
            cols.extend(row.saturating_sub(1)..(row + 2).min(self.n));
            rows.push(cols.len());
        }
        StructuralPattern::new(self.n, self.n, rows, cols).unwrap()
    }
    fn coupling(&self) -> f64 {
        0.1 * ((self.n + 1) * (self.n + 1)) as f64
    }
}
impl OdeSystem<f64> for Reaction {
    fn dimension(&self) -> usize {
        self.n
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[f64],
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(0.);
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let a = self.coupling();
        for (i, value) in out.iter_mut().enumerate() {
            let left = if i > 0 { y[i - 1] } else { 0. };
            let right = y.get(i + 1).copied().unwrap_or(0.);
            *value = a * (left - 2. * y[i] + right) - y[i].powi(3);
        }
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let a = self.coupling();
        for i in 0..self.n {
            for k in 0..self.n {
                let value = if i == k {
                    -2. * a - 3. * y[i] * y[i]
                } else if i.abs_diff(k) == 1 {
                    a
                } else {
                    0.
                };
                j.set(i, k, value)?;
            }
        }
        Ok(Some(()))
    }
    fn sparse_jacobian(
        &self,
        _: f64,
        y: &[f64],
        pattern: StructuralPatternView<'_>,
        values: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        if !self.sparse {
            return Ok(None);
        }
        self.calls.set(self.calls.get() + 1);
        let a = self.coupling();
        for row in 0..self.n {
            for k in pattern.row_ptr()[row]..pattern.row_ptr()[row + 1] {
                values[k] = if pattern.col_indices()[k] == row {
                    -2. * a - 3. * y[row] * y[row]
                } else {
                    a
                };
            }
        }
        Ok(Some(()))
    }
}

/// Twenty steps of `h = 5·10⁻³` from the first sine mode.
fn sweep<M: OdeMethod<f64> + Clone>(
    method: M,
    system: &Reaction,
    steps: Steps,
) -> Result<Vec<f64>, SciOxideError> {
    let n = system.n;
    let y = (0..n)
        .map(|i| (core::f64::consts::PI * (i + 1) as f64 / (n + 1) as f64).sin())
        .collect();
    integrate(method, system, y, 5e-3, steps).map(|(y, _)| y)
}

/// Every implicit family on the sparse backend, including coupled Gauss and Radau stages.
fn sparse_methods(
    backend: &OdeStageBackend,
    strategy: DerivativeStrategy,
    system: &Reaction,
) -> Vec<(&'static str, Result<Vec<f64>, SciOxideError>)> {
    let b = || backend.clone();
    vec![
        (
            "BackwardEuler",
            sweep(
                BackwardEuler::new()
                    .with_backend(b())
                    .with_derivative_strategy(strategy),
                system,
                Steps::Pinned(20),
            ),
        ),
        (
            "GaussRk",
            sweep(
                GaussRk::new()
                    .with_backend(b())
                    .with_derivative_strategy(strategy),
                system,
                Steps::Pinned(20),
            ),
        ),
        (
            "RadauIia",
            sweep(
                RadauIia::new()
                    .with_backend(b())
                    .with_derivative_strategy(strategy),
                system,
                Steps::Pinned(20),
            ),
        ),
        (
            "Sdirk",
            sweep(
                Sdirk::new()
                    .with_backend(b())
                    .with_derivative_strategy(strategy),
                system,
                Steps::Pinned(20),
            ),
        ),
        // The startup estimate of automatic BDF at h = 5·10⁻³ exceeds any error control
        // that keeps its Krylov stages within the parity tolerance, so its steps adapt.
        (
            "Bdf",
            sweep(
                Bdf::with_order_policy(MultistepOrder::Automatic(3))
                    .unwrap()
                    .with_backend(b())
                    .with_derivative_strategy(strategy),
                system,
                Steps::Adaptive(20, 1e-7),
            ),
        ),
        // Rosenbrock-W has no Newton iteration; 10⁻⁶ accepts every pinned step.
        (
            "RosenbrockW",
            sweep(
                RosenbrockW::new(RosenbrockMethod::Ros34Pw2)
                    .unwrap()
                    .with_backend(b())
                    .with_derivative_strategy(strategy),
                system,
                Steps::PinnedAt(20, 1e-6),
            ),
        ),
    ]
}

#[test]
fn sparse_stages_honour_the_derivative_strategy() {
    let n = 12;
    let probe = Reaction::new(n, true);
    let sparse = OdeStageBackend::Sparse {
        pattern: probe.pattern(),
        limits: SparseLuLimits::new(64 * n, 1 << 22),
    };
    let reference = sparse_methods(
        &OdeStageBackend::Dense,
        DerivativeStrategy::Analytic,
        &probe,
    );
    for (strategy, analytic) in [
        (DerivativeStrategy::AnalyticOrAutomaticOrForward, true),
        (DerivativeStrategy::Analytic, true),
        (DerivativeStrategy::Forward, false),
        (DerivativeStrategy::Central, false),
    ] {
        for provided in [true, false] {
            let system = Reaction::new(n, provided);
            let results = sparse_methods(&sparse, strategy, &system);
            for ((name, result), (_, expected)) in results.iter().zip(&reference) {
                let expected = expected.as_ref().unwrap();
                // An analytic-only strategy cannot use differences of an RHS-only system.
                let available = provided || strategy != DerivativeStrategy::Analytic;
                assert_eq!(result.is_ok(), available, "{name} {strategy:?} {provided}");
                match result {
                    Ok(actual) => {
                        for (a, b) in actual.iter().zip(expected) {
                            assert!((a - b).abs() < 1e-7, "{name} {strategy:?}: {a} vs {b}");
                        }
                    }
                    Err(error) => assert!(matches!(
                        error.view(),
                        SciOxideErrorView::InvalidInput {
                            reason: InvalidInputReason::UnsupportedConfiguration,
                            ..
                        }
                    )),
                }
            }
            // Difference strategies never call an available analytic sparse Jacobian.
            assert_eq!(
                system.calls.get() > 0,
                provided && analytic,
                "{strategy:?} provided={provided}"
            );
        }
    }
    // OdeSystem has no sparse automatic derivative, so that source is unavailable.
    for (name, result) in sparse_methods(&sparse, DerivativeStrategy::Automatic, &probe) {
        assert!(result.is_err(), "{name}");
    }
}

#[test]
fn krylov_stages_use_directional_differences() {
    // Unpreconditioned GMRES resolves every coupled `3n` system within the request's
    // default of 20 inner iterations.
    let n = 6;
    let reference = sparse_methods(
        &OdeStageBackend::Dense,
        DerivativeStrategy::Analytic,
        &Reaction::new(n, false),
    );
    let krylov = OdeStageBackend::Krylov {
        iterations: Budget::try_from(60).unwrap(),
        restart: Budget::try_from(30).unwrap(),
        operator_scratch: 0,
        preconditioner_scratch: 0,
    };
    for strategy in [
        DerivativeStrategy::AnalyticOrAutomaticOrForward,
        DerivativeStrategy::Central,
    ] {
        let results = sparse_methods(&krylov, strategy, &Reaction::new(n, false));
        for ((name, result), (_, expected)) in results.iter().zip(&reference) {
            for (a, b) in result
                .as_ref()
                .unwrap()
                .iter()
                .zip(expected.as_ref().unwrap())
            {
                assert!((a - b).abs() < 1e-6, "{name} {strategy:?}: {a} vs {b}");
            }
        }
    }
}
