//! Nonlinear order conditions on the eccentric Kepler problem and on a non-autonomous
//! problem that exercises every stage node, independent of any ODE reference solver.
//! Fixed stepping isolates formula order from adaptive control; tight stage tolerances
//! prevent nonlinear iteration error from masking it.

use alloc::vec::Vec;
use scioxide_core::{ErrorTolerance, NormSelection, SciOxideError, SolveContext};
use scioxide_linalg::{MatrixViewMut, Vector};

use super::prepared;
use crate::stepper::{StepRequest, StepTrace, Stepper, testing::with_context};
use crate::*;

struct Kepler;
impl OdeSystem<f64> for Kepler {
    fn dimension(&self) -> usize {
        4
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let r = y[0].hypot(y[1]);
        out.copy_from_slice(&[y[2], y[3], -y[0] / r.powi(3), -y[1] / r.powi(3)]);
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        y: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let r = y[0].hypot(y[1]);
        for row in 0..4 {
            for col in 0..4 {
                out.set(row, col, 0.)?;
            }
        }
        out.set(0, 2, 1.)?;
        out.set(1, 3, 1.)?;
        for row in 0..2 {
            for col in 0..2 {
                out.set(
                    row + 2,
                    col,
                    3. * y[row] * y[col] / r.powi(5) - if row == col { r.powi(-3) } else { 0. },
                )?;
            }
        }
        Ok(Some(()))
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.fill(0.);
        Ok(Some(()))
    }
}

/// `y' = −(t + cos t) y²` and `z' = z cos t` with `y(0) = z(0) = 1`, solved by
/// `y = 1/(1 + t²/2 + sin t)` and `z = exp(sin t)`.
struct Forced;
impl OdeSystem<f64> for Forced {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(&[-(t + t.cos()) * y[0] * y[0], t.cos() * y[1]]);
        Ok(())
    }
    fn jacobian(
        &self,
        t: f64,
        y: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, -2. * (t + t.cos()) * y[0])?;
        out.set(0, 1, 0.)?;
        out.set(1, 0, 0.)?;
        out.set(1, 1, t.cos())?;
        Ok(Some(()))
    }
    fn time_derivative(
        &self,
        t: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.copy_from_slice(&[-(1. - t.sin()) * y[0] * y[0], -t.sin() * y[1]]);
        Ok(Some(()))
    }
}

fn forced(time: f64) -> Vec<f64> {
    alloc::vec![1. / (1. + time * time / 2. + time.sin()), time.sin().exp()]
}

fn kepler(time: f64) -> Vec<f64> {
    exact(time).to_vec()
}

fn exact(time: f64) -> [f64; 4] {
    // a = mu = 1, eccentricity e = 1/2. Solve E - e sin(E) = t;
    // the monotone Kepler equation has derivative >= 1/2. Twelve Newton
    // iterations give scalar precision, independently of every tested tableau.
    let e = 0.5;
    let mut anomaly = time;
    for _ in 0..12 {
        anomaly -= (anomaly - e * anomaly.sin() - time) / (1. - e * anomaly.cos());
    }
    let denominator = 1. - e * anomaly.cos();
    let minor = (1. - e * e).sqrt();
    [
        anomaly.cos() - e,
        minor * anomaly.sin(),
        -anomaly.sin() / denominator,
        minor * anomaly.cos() / denominator,
    ]
}

fn error<M: Stepper<f64, f64>>(
    method: &M,
    steps: usize,
    end: f64,
    tolerance: f64,
) -> (f64, usize, usize) {
    error_on(method, &Kepler, kepler, steps, end, tolerance)
}

fn error_on<M: Stepper<f64, f64>>(
    method: &M,
    system: &dyn OdeSystem<f64>,
    exact: fn(f64) -> Vec<f64>,
    steps: usize,
    end: f64,
    tolerance: f64,
) -> (f64, usize, usize) {
    let mut workspace = prepared(method, system.dimension());
    let trace = StepTrace::default();
    let tolerance =
        ErrorTolerance::scalar(tolerance, tolerance, NormSelection::RootMeanSquare).unwrap();
    let request = StepRequest {
        error_tolerance: Some(&tolerance),
        ..StepRequest::new(&trace)
    };
    let mut state = Vector::from_vec(exact(0.));
    let h = end / steps as f64;
    let mut time = 0.;
    let mut maximum_error_order = 0;
    let mut switches = 0;
    with_context(|cx| {
        for _ in 0..steps {
            trace.reset();
            let result = method.step(system, time, &mut state, h, request, &mut workspace, cx)?;
            maximum_error_order = maximum_error_order.max(result.error_order.unwrap_or(0));
            switches += usize::from(result.method_switch.is_some());
            method.accept(&mut workspace)?;
            time += h;
        }
        Ok(())
    })
    .unwrap();
    let reference = exact(time);
    let error = state
        .as_slice()
        .iter()
        .zip(reference)
        .fold(0_f64, |error, (&actual, expected)| {
            error.max((actual - expected).abs())
        });
    (error, maximum_error_order, switches)
}

fn order<M: Stepper<f64, f64>>(method: M, expected: usize, coarse: usize, end: f64) {
    let errors = [coarse, 2 * coarse, 4 * coarse].map(|n| error(&method, n, end, 2e-14).0);
    assert_orders(method.info().name, expected, &errors);
}

fn forced_order<M: Stepper<f64, f64>>(method: M, expected: usize, coarse: usize) {
    let errors = [coarse, 2 * coarse, 4 * coarse]
        .map(|n| error_on(&method, &Forced, forced, n, 2., 2e-14).0);
    assert_orders(method.info().name, expected, &errors);
}

fn assert_orders(name: &str, expected: usize, errors: &[f64; 3]) {
    // The infinity-norm reference is O(1). 64 eps accounts for the four state
    // components and reference evaluation. Every pair must stay above this floor;
    // reaching it requires a coarser window, never silently dropping a failed pair.
    let floor = 64. * f64::EPSILON;
    for pair in errors.windows(2) {
        let observed = (pair[0] / pair[1]).log2();
        assert!(
            pair.iter().all(|e| e.is_finite() && *e > floor) && observed >= expected as f64 - 0.35,
            "{name}: expected {expected}, observed {observed}, floor={floor:e}, errors={errors:?}"
        );
    }
}

macro_rules! order_case {
    ($name:ident, $method:expr, $order:expr, $coarse:expr, $end:expr) => {
        #[test]
        fn $name() {
            order($method, $order, $coarse, $end);
        }
    };
}
macro_rules! forced_case {
    ($name:ident, $method:expr, $order:expr, $coarse:expr) => {
        #[test]
        fn $name() {
            forced_order($method, $order, $coarse);
        }
    };
}
forced_case!(forced_forward_euler, ForwardEuler::new(), 1, 160);
forced_case!(forced_midpoint, ExplicitMidpoint::new(), 2, 40);
forced_case!(forced_heun, Heun::new(), 2, 40);
forced_case!(forced_rk4, ClassicalRk4::new(), 4, 20);
forced_case!(forced_rkf45, Rkf45::new(), 5, 10);
forced_case!(forced_dp5, DormandPrince5::new(), 5, 10);
forced_case!(forced_verner65, Verner65::new(), 6, 8);
forced_case!(forced_dop853, DormandPrince853::new(), 8, 4);
forced_case!(forced_radau, RadauIia::new(), 5, 8);
forced_case!(
    forced_ros34pw2,
    RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap(),
    3,
    20
);

order_case!(forward_euler, ForwardEuler::new(), 1, 160, 2.);
order_case!(midpoint, ExplicitMidpoint::new(), 2, 40, 2.);
order_case!(heun, Heun::new(), 2, 40, 2.);
order_case!(rk4, ClassicalRk4::new(), 4, 40, 2.);
order_case!(rkf45, Rkf45::new(), 5, 80, 2.);
order_case!(dp5, DormandPrince5::new(), 5, 80, 2.);
order_case!(verner65, Verner65::new(), 6, 16, 2.);
order_case!(dop853, DormandPrince853::new(), 8, 8, 2.);
order_case!(backward_euler, BackwardEuler::new(), 1, 160, 2.);
order_case!(trapezoid, ImplicitTrapezoid::new(), 2, 40, 2.);
order_case!(gauss, GaussRk::new(), 4, 20, 2.);
order_case!(sdirk, Sdirk::new(), 3, 40, 2.);
order_case!(radau, RadauIia::new(), 5, 20, 2.);
order_case!(
    ros2,
    RosenbrockW::new(RosenbrockMethod::Ros2).unwrap(),
    2,
    80,
    2.
);
order_case!(
    ros34pw2,
    RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap(),
    3,
    80,
    2.
);

// Fifth-order Adams errors cancel on short Kepler arcs; [0,20] accumulates a
// measurable principal error. Orders three and four need finer startup windows.
#[test]
fn adams_bashforth() {
    for q in 1..=5 {
        order(
            AdamsBashforth::new(q).unwrap(),
            q,
            if q == 5 {
                3200
            } else if q >= 3 {
                320
            } else {
                80
            },
            if q == 5 { 20. } else { 2. },
        );
    }
}
#[test]
fn adams_moulton() {
    for q in 1..=5 {
        order(
            AdamsMoulton::new(q).unwrap(),
            q,
            if q == 5 {
                3200
            } else if q >= 3 {
                320
            } else {
                80
            },
            if q == 5 { 20. } else { 2. },
        );
    }
}
#[test]
fn bdf() {
    for q in 1..=5 {
        order(Bdf::new(q).unwrap(), q, if q == 3 { 160 } else { 80 }, 2.);
    }
}
#[test]
fn ndf() {
    for q in 1..=5 {
        order(
            Bdf::new(q).unwrap().with_formula(BdfFormula::Ndf),
            q,
            80,
            2.,
        );
    }
}

#[test]
fn adams_bdf_cold_start() {
    let method = AdamsBdf::adams_bdf().unwrap();
    // Automatic order begins at one. A finite number of startup steps contributes
    // O(h^2), even when the selected formula subsequently has higher order.
    let samples = [80, 160, 320].map(|n| error(&method, n, 2., 0.01));
    assert!(
        samples.iter().all(|sample| sample.1 > 2 && sample.2 == 0),
        "{samples:?}"
    );
    assert_orders(
        "AdamsBdf automatic startup",
        2,
        &samples.map(|sample| sample.0),
    );
}
#[test]
fn auto_stiff_nonstiff_regime() {
    let method = AutoStiff::new(
        DormandPrince5::new(),
        RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap(),
        StiffnessSwitchOptions::default(),
    )
    .unwrap();
    order(method, 5, 20, 2.);
}
#[test]
fn auto_stiff_stiff_regime() {
    let method = AutoStiff::new(
        DormandPrince5::new(),
        RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap(),
        StiffnessSwitchOptions::new()
            .with_thresholds(1e-9, 1e-10)
            .with_evidence_steps(1, 1)
            .with_residence(1)
            .with_rejections(1),
    )
    .unwrap();
    let samples = [80, 160, 320].map(|n| error(&method, n, 2., 2e-14));
    assert!(samples.iter().all(|sample| sample.2 == 1), "{samples:?}");
    assert_orders("AutoStiff stiff regime", 3, &samples.map(|sample| sample.0));
}

#[test]
fn default_adams_bdf_adaptive_nonlinear_accuracy() {
    let mut previous = f64::INFINITY;
    for tolerance in [1e-4, 1e-6, 1e-8] {
        let options = OdeOptions::new().with_error_tolerance(
            ErrorTolerance::scalar(tolerance, tolerance, NormSelection::RootMeanSquare).unwrap(),
        );
        let plan = (AdamsBdf::adams_bdf().unwrap(), options);
        let solution = plan
            .0
            .solve(
                &OdeProblem::new(&Kepler, 2.),
                &plan.1,
                OdeState::new(0., Vector::from_vec(exact(0.).to_vec())),
            )
            .unwrap();
        assert_eq!(
            solution.report().termination(),
            Some(OdeTermination::ReachedEnd)
        );
        let error = solution
            .state()
            .values()
            .iter()
            .zip(exact(2.))
            .fold(0_f64, |error, (&actual, expected)| {
                error.max((actual - expected).abs())
            });
        assert!(
            error.is_finite() && error < previous / 2. && error < 100. * tolerance,
            "tol={tolerance:e}, error={error:e}, previous={previous:e}"
        );
        previous = error;
    }
}
