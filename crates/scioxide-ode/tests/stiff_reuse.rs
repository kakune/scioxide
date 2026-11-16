//! Reuse boundaries, processor propagation and admitted stiff work.
#![allow(clippy::unwrap_used)]
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection,
    ProcessorCapabilities, ProcessorFeature, SciOxideError, SolveContext,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::*;

struct ObservedProcessor {
    queries: AtomicUsize,
    native: bool,
}
impl ProcessorCapabilities for ObservedProcessor {
    fn supports(&self, _: ProcessorFeature) -> bool {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.native
    }
}
struct Decay {
    n: usize,
    jacobians: Cell<usize>,
    masses: Cell<usize>,
    mass: bool,
}
impl Decay {
    fn new(n: usize, mass: bool) -> Self {
        Self {
            n,
            jacobians: Cell::new(0),
            masses: Cell::new(0),
            mass,
        }
    }
}
impl OdeSystem<f64> for Decay {
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
        f: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        f.iter_mut().zip(y).for_each(|(f, y)| *f = -2. * y);
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.jacobians.set(self.jacobians.get() + 1);
        for i in 0..self.n {
            for k in 0..self.n {
                j.set(i, k, if i == k { -2. } else { 0. })?;
            }
        }
        Ok(Some(()))
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
    fn mass_matrix_revision(&self) -> Option<u64> {
        self.mass.then_some(1)
    }
    fn mass_matrix(
        &self,
        m: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.masses.set(self.masses.get() + 1);
        for i in 0..self.n {
            for k in 0..self.n {
                m.set(i, k, if i == k { 2. } else { 0. })?;
            }
        }
        Ok(())
    }
}

/// One step of `h = 0.01` on `[0, 0.01]` per processor configuration, reusing one
/// workspace across both epochs.
fn processor_reaches_stages<M: OdeMethod<f64> + Clone>(method: M) {
    let system = Decay::new(64, false);
    let plan = (method, OdeOptions::new().with_initial_step(0.01));
    let problem = OdeProblem::new(&system, 0.01);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    for native in [false, true] {
        let processor = ObservedProcessor {
            queries: AtomicUsize::new(0),
            native,
        };
        let execution = ExecutionContext::sequential().with_processor(&processor);
        let mut state = OdeState::new(0., Vector::from_vec(vec![1.; 64]));
        let report = plan
            .0
            .advance(
                &problem,
                &plan.1,
                &mut state,
                &mut (),
                &mut workspace,
                execution,
            )
            .unwrap();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert!(report.counts().provider(EvaluationRole::Factorization) > 0);
        #[cfg(feature = "std")]
        assert!(processor.queries.load(Ordering::Relaxed) > 0);
        assert!(state.values().iter().all(|v| v.is_finite() && *v < 1.));
    }
}
#[test]
fn caller_processor_reaches_each_stage_factorization() {
    processor_reaches_stages(BackwardEuler::new());
    processor_reaches_stages(Bdf::new(1).unwrap());
    processor_reaches_stages(RadauIia::new());
    processor_reaches_stages(RosenbrockW::new(RosenbrockMethod::Ros34Pw2).unwrap());
}
#[test]
fn managed_solve_propagates_processor_and_reports_direct_stage_work() {
    let system = Decay::new(64, false);
    let processor = ObservedProcessor {
        queries: AtomicUsize::new(0),
        native: true,
    };
    let plan = (
        Bdf::new(1).unwrap(),
        OdeOptions::new().with_initial_step(0.01),
    );
    let problem = OdeProblem::new(&system, 0.02);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0., Vector::from_vec(vec![1.; 64]));
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential().with_processor(&processor),
        )
        .unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    #[cfg(feature = "std")]
    assert!(processor.queries.load(Ordering::Relaxed) > 0);
    let counts = report.counts();
    assert!(counts.provider(EvaluationRole::Factorization) > 0);
    assert!(counts.provider(EvaluationRole::LinearSolve) > 0);
    assert_eq!(
        counts.provider(EvaluationRole::Jacobian),
        system.jacobians.get()
    );
    assert_eq!(
        counts.provider(EvaluationRole::Factorization),
        report
            .accounting()
            .total()
            .unwrap()
            .provider(EvaluationRole::Factorization)
    );
}

/// Adaptive BDF/NDF on `M y' = -2y` (so `y' = -y`) over `[0, 0.81]`; the oversized
/// first step is rejected.
fn constant_mass_solve(formula: BdfFormula, maximum_order: usize) -> (Decay, OdeSolution<f64>) {
    let system = Decay::new(2, true);
    let method = Bdf::with_order_policy(MultistepOrder::Automatic(maximum_order))
        .unwrap()
        .with_formula(formula);
    let options = OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(1e-6, 1e-4, NormSelection::RootMeanSquare).unwrap(),
        )
        .with_initial_step(0.3);
    let plan = (method, options);
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&system, 0.81),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![1.; 2])),
        )
        .unwrap();
    (system, solution)
}
#[test]
fn constant_jacobian_and_mass_survive_order_changes_and_rejection() {
    for formula in [BdfFormula::Bdf, BdfFormula::Ndf] {
        let (system, solution) = constant_mass_solve(formula, 5);
        let report = solution.report();
        assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        assert!(report.rejected_steps() > 0, "{:?}", report.accepted_steps());
        // Raising the order beyond one is what lets the fifth-order policy take far
        // fewer steps than the first-order policy at the same tolerance.
        let (_, first_order) = constant_mass_solve(formula, 1);
        assert!(
            2 * report.accepted_steps() < first_order.report().accepted_steps(),
            "{:?} vs {:?}",
            report.accepted_steps(),
            first_order.report().accepted_steps()
        );
        assert_eq!(system.jacobians.get(), 1);
        assert_eq!(system.masses.get(), 1);
        assert!((solution.state().values()[0] - (-0.81_f64).exp()).abs() < 1e-4);
    }
}

/// `inner` with a caller-scheduled sequence of step limits, one per proposal.
struct Scheduled<'a, Sys> {
    inner: &'a Sys,
    limits: &'a [f64],
    proposals: Cell<usize>,
}
impl<Sys: OdeSystem<f64>> OdeSystem<f64> for Scheduled<'_, Sys> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn step_limit(
        &self,
        _: f64,
        _: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<f64>, SciOxideError> {
        let index = self.proposals.get();
        self.proposals.set(index + 1);
        Ok(self.limits.get(index).copied())
    }
    fn time_derivative(
        &self,
        time: f64,
        state: &[f64],
        output: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.inner.time_derivative(time, state, output, cx)
    }
    fn rhs(
        &self,
        t: f64,
        y: &[f64],
        f: &mut [f64],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.inner.rhs(t, y, f, cx)
    }
    fn jacobian(
        &self,
        t: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.inner.jacobian(t, y, j, cx)
    }
    // Leave constancy undeclared so this fixture exercises chord factor reuse.
}

#[test]
fn nearby_effective_steps_reuse_lu_and_large_changes_refactor() {
    // Fixed-step methods keep a reduced step, so the schedule only shrinks: the
    // first three steps stay within the [0.8, 1.2] factor-reuse window of the first
    // and the last one leaves it. Dyadic steps keep every endpoint exact.
    // Stay inside the state-movement threshold while testing only factor changes.
    let steps = [0.125, 0.117_187_5, 0.109_375, 0.0625];
    let factors = [1, 0, 0, 1];
    let end = steps.iter().sum::<f64>();
    let tolerance = ErrorTolerance::scalar(1e-12, 1e-10, NormSelection::Infinity).unwrap();
    let mut expected = Vec::new();
    let mut value = 1.;
    for h in steps {
        value /= 1. + 2. * h;
        expected.push(value);
    }
    let mut previous = 0;
    for attempts in 1..=steps.len() {
        let decay = Decay::new(1, false);
        let system = Scheduled {
            inner: &decay,
            limits: &steps,
            proposals: Cell::new(0),
        };
        let options = OdeOptions::new()
            .with_error_tolerance(tolerance.clone())
            .with_initial_step(1.)
            .with_output_schedule(OdeOutputSchedule::AcceptedSteps)
            .with_max_steps(Budget::try_from(attempts).unwrap());
        let plan = (BackwardEuler::new(), options);
        let solution = plan
            .0
            .solve(
                &OdeProblem::new(&system, end),
                &plan.1,
                OdeState::new(0., Vector::from_vec(vec![1.])),
            )
            .unwrap();
        let report = solution.report();
        let factorizations = report.counts().provider(EvaluationRole::Factorization);
        assert_eq!(
            factorizations - previous,
            factors[attempts - 1],
            "{attempts}"
        );
        previous = factorizations;
        let samples = solution.trajectory().states();
        assert_eq!(samples.len(), attempts + 1);
        for (sample, &expected) in samples[1..].iter().zip(&expected) {
            assert!((sample.as_slice()[0] - expected).abs() < 1e-10);
        }
        if attempts < steps.len() {
            assert_eq!(
                report.termination(),
                Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
            );
        } else {
            assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
        }
        assert_eq!(decay.jacobians.get(), 1);
    }
}

struct NonlinearDecay(Cell<usize>);
impl OdeSystem<f64> for NonlinearDecay {
    fn dimension(&self) -> usize {
        1
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
        f: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        f[0] = -y[0] * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.0.set(self.0.get() + 1);
        j.set(0, 0, -2. * y[0])?;
        Ok(Some(()))
    }
}
/// `steps` accepted steps of `1/steps` on `[0, 1]`: the dyadic step is exact and the
/// step bounds pin every adaptive proposal to it.
fn nonlinear_w(method: RosenbrockMethod, steps: u32) -> f64 {
    let system = NonlinearDecay(Cell::new(0));
    let h = 1. / f64::from(steps);
    let options = OdeOptions::new()
        .with_initial_step(h)
        .with_minimum_step(h)
        .with_maximum_step(h);
    let plan = (RosenbrockW::new(method).unwrap(), options);
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&system, 1.),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![1.])),
        )
        .unwrap();
    let report = solution.report();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert_eq!(report.accepted_steps(), steps as usize);
    assert_eq!(report.rejected_steps(), 0);
    assert!(system.0.get() < usize::try_from(steps / 2).unwrap());
    (solution.state().values()[0] - 0.5).abs()
}
#[test]
fn lagged_nonlinear_w_jacobians_preserve_classical_order() {
    for (method, ratio) in [
        (RosenbrockMethod::Ros2, 3.5),
        (RosenbrockMethod::Ros34Pw2, 7.),
    ] {
        let coarse = nonlinear_w(method, 128);
        let fine = nonlinear_w(method, 256);
        assert!(coarse > ratio * fine, "{method:?}: {coarse}/{fine}");
    }
}

/// `y' = drive - y²/100`: a weakly nonlinear decay pushed by a constant drive, so
/// one step of `h = 0.01` moves the state by about `drive/100`.
struct Driven {
    drive: f64,
    jacobians: Cell<usize>,
}
impl OdeSystem<f64> for Driven {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        f: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        f[0] = self.drive - 0.01 * y[0] * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        y: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.jacobians.set(self.jacobians.get() + 1);
        j.set(0, 0, -0.02 * y[0])?;
        Ok(Some(()))
    }
}
/// Jacobian and factorization counts after one and after two backward Euler steps.
fn driven_work(drive: f64) -> [(usize, usize); 2] {
    [1, 2].map(|attempts| {
        let system = Driven {
            drive,
            jacobians: Cell::new(0),
        };
        let options = OdeOptions::new()
            .with_initial_step(0.01)
            .with_max_steps(Budget::try_from(attempts).unwrap());
        let plan = (BackwardEuler::new(), options);
        let solution = plan
            .0
            .solve(
                &OdeProblem::new(&system, 0.02),
                &plan.1,
                OdeState::new(0., Vector::from_vec(vec![1.])),
            )
            .unwrap();
        let counts = solution.report().counts();
        assert_eq!(
            counts.provider(EvaluationRole::Jacobian),
            system.jacobians.get()
        );
        (
            counts.provider(EvaluationRole::Jacobian),
            counts.provider(EvaluationRole::Factorization),
        )
    })
}
#[test]
fn substantial_movement_refreshes_a_nonconstant_jacobian() {
    // The first step moves the state from 1 to about 3, beyond the relative
    // movement limit of one half: the second step evaluates a fresh Jacobian and
    // factors once.
    let [(jacobians, factors), (total_jacobians, total_factors)] = driven_work(200.);
    assert_eq!((jacobians, factors), (1, 1));
    assert_eq!(total_jacobians - jacobians, 1);
    assert_eq!(total_factors - factors, 1);
    assert_eq!(total_jacobians, 2);
    // A small movement keeps the lagged Jacobian and its factor.
    let [first, both] = driven_work(1.);
    assert_eq!(first, (1, 1));
    assert_eq!(both, (1, 1));
}
