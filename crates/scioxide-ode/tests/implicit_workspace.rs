//! Reusable implicit workspace, cache-validity, and Newton acceptance tests.
//!
//! Every check runs through [`OdeMethod::advance`] with pinned steps. Per-step work is
//! the difference between an `n`-step and an `(n - 1)`-step integration of the same
//! deterministic problem; each `advance` is a fresh epoch that invalidates stage caches.

#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]

use std::cell::Cell;

use scioxide_core::{
    AttemptOutcome, Budget, Capacity, ErrorTolerance, EvaluationRole, ExecutionContext,
    InvalidInputReason, NonFiniteOrigin, NormSelection, SciOxideError, SciOxideErrorView,
    SolveContext,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::{
    BackwardEuler, Bdf, BuiltinOdeMethod, GaussRk, ImplicitTrapezoid, NewtonMode, OdeMethod,
    OdeOptions, OdeOutputSchedule, OdeProblem, OdeSolveReport, OdeState, OdeSystem, OdeWorkspace,
    RadauIia, Sdirk,
};

/// Fixed-step methods: the first (and every) step has size `h`; Newton uses the
/// core default component policy, as the former unconfigured step requests did.
fn fixed(h: f64) -> OdeOptions<f64> {
    OdeOptions::new()
        .with_error_tolerance(ErrorTolerance::default())
        .with_initial_step(h)
}

/// Adaptive Radau IIA pinned to steps of `h`. The loose tolerance stands in for the
/// former uncontrolled steps; stages of the linear test systems are exact after one
/// Newton correction regardless of the tolerance.
fn pinned(h: f64) -> OdeOptions<f64> {
    OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(1.0, 1.0, NormSelection::RootMeanSquare).unwrap(),
        )
        .with_initial_step(h)
        .with_minimum_step(h)
        .with_maximum_step(h)
}

fn start(values: &[f64]) -> OdeState<f64> {
    OdeState::from_slice(0.0, values).unwrap()
}

fn sequential() -> ExecutionContext<'static> {
    ExecutionContext::sequential()
}

/// Integrate `system` from `state` to `end` with a nonallocating sink.
fn run<M, Sys>(
    plan: &(M, OdeOptions<f64>),
    system: &Sys,
    end: f64,
    state: &mut OdeState<f64>,
    workspace: &mut OdeWorkspace<f64, M>,
) -> Result<OdeSolveReport<f64>, SciOxideError>
where
    M: OdeMethod<f64> + Clone,
    Sys: OdeSystem<f64> + ?Sized,
{
    plan.0.advance(
        &OdeProblem::new(system, end),
        &plan.1,
        state,
        &mut (),
        workspace,
        sequential(),
    )
}

/// [`run`] that must reach the end.
fn complete<M, Sys>(
    plan: &(M, OdeOptions<f64>),
    system: &Sys,
    end: f64,
    state: &mut OdeState<f64>,
    workspace: &mut OdeWorkspace<f64, M>,
) where
    M: OdeMethod<f64> + Clone,
    Sys: OdeSystem<f64> + ?Sized,
{
    let report = run(plan, system, end, state, workspace).unwrap();
    assert!(report.is_success(), "{:?}", report.termination());
    assert_eq!(state.time(), end);
}

/// Integrate with a fresh workspace and return the report and final values.
fn solve<M, Sys>(
    plan: &(M, OdeOptions<f64>),
    system: &Sys,
    end: f64,
    initial: &[f64],
) -> (OdeSolveReport<f64>, Vec<f64>)
where
    M: OdeMethod<f64> + Clone,
    Sys: OdeSystem<f64> + ?Sized,
{
    let mut workspace = OdeWorkspace::new();
    let mut state = start(initial);
    let report = run(plan, system, end, &mut state, &mut workspace).unwrap();
    assert!(report.is_success(), "{:?}", report.termination());
    (report, state.values().to_vec())
}

/// Stage work of one integration, read from its report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Work {
    rhs: usize,
    jacobians: usize,
    factorizations: usize,
    linear_solves: usize,
    newton_iterations: usize,
    cache_hits: usize,
}

impl Work {
    fn of(report: &OdeSolveReport<f64>) -> Self {
        let own = report.counts();
        Self {
            rhs: own.provider(EvaluationRole::RightHandSide),
            jacobians: own.provider(EvaluationRole::Jacobian),
            factorizations: own.provider(EvaluationRole::Factorization),
            linear_solves: own.provider(EvaluationRole::LinearSolve),
            newton_iterations: report.accounting().nested().iterations(),
            cache_hits: own.cache_hits(),
        }
    }

    /// Work of the steps that follow an earlier, shorter run of the same problem.
    fn since(self, earlier: Self) -> Self {
        Self {
            rhs: self.rhs - earlier.rhs,
            jacobians: self.jacobians - earlier.jacobians,
            factorizations: self.factorizations - earlier.factorizations,
            linear_solves: self.linear_solves - earlier.linear_solves,
            newton_iterations: self.newton_iterations - earlier.newton_iterations,
            cache_hits: self.cache_hits - earlier.cache_hits,
        }
    }
}

fn row_major<'a>(
    matrix: &'a mut MatrixViewMut<'_, f64>,
    name: &'static str,
) -> Result<&'a mut [f64], SciOxideError> {
    matrix.as_contiguous_slice_mut().ok_or_else(|| {
        SciOxideError::invalid_input(
            scioxide_core::argument!("matrix"),
            InvalidInputReason::UnsupportedConfiguration,
            name,
        )
    })
}

struct ConstantLinear {
    dimension: usize,
    rate: f64,
}

impl ConstantLinear {
    const fn new(dimension: usize, rate: f64) -> Self {
        Self { dimension, rate }
    }
}

impl OdeSystem<f64> for ConstantLinear {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative
            .iter_mut()
            .zip(state)
            .for_each(|(output, &value)| *output = self.rate * value);
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let jacobian_data = row_major(jacobian, "row-major output required")?;
        jacobian_data.fill(0.0);
        (0..self.dimension).for_each(|index| {
            jacobian_data[index * self.dimension + index] = self.rate;
        });
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

#[test]
fn backward_euler_reports_exact_linear_work() {
    let plan = (BackwardEuler::<f64>::new(), fixed(0.1));
    let (report, state) = solve(&plan, &ConstantLinear::new(1, -2.0), 0.1, &[1.0]);
    assert!((state[0] - 1.0 / 1.2).abs() < 1e-14);
    assert_eq!(
        Work::of(&report),
        Work {
            rhs: 3,
            jacobians: 1,
            factorizations: 1,
            linear_solves: 1,
            newton_iterations: 1,
            cache_hits: 0,
        }
    );
}

struct StateDependent;

impl OdeSystem<f64> for StateDependent {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let value = state[0];
        derivative[0] = -value - 0.1 * value * value;
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        row_major(jacobian, "row-major output required")?[0] = -1.0 - 0.2 * state[0];
        Ok(Some(()))
    }
}

#[test]
fn nearby_states_reuse_the_jacobian_and_factor() {
    let plan = (BackwardEuler::<f64>::new(), fixed(0.05));
    let (first, _) = solve(&plan, &StateDependent, 0.05, &[1.0]);
    let (both, _) = solve(&plan, &StateDependent, 0.1, &[1.0]);
    assert_eq!(both.accepted_steps(), 2);
    let second = Work::of(&both).since(Work::of(&first));
    assert_eq!(second.jacobians, 0);
    assert_eq!(second.factorizations, 0);
    assert_eq!(
        second.cache_hits, 2,
        "the Jacobian and its factor are reused"
    );
}

#[test]
fn step_size_change_reuses_only_the_constant_raw_jacobian() {
    let system = ConstantLinear::new(1, -2.0);
    let plan = (BackwardEuler::<f64>::new(), fixed(0.1));
    let (first, _) = solve(&plan, &system, 0.1, &[1.0]);
    // The final step is shortened to the remaining 0.05.
    let (both, _) = solve(&plan, &system, 0.15, &[1.0]);
    assert_eq!(both.accepted_steps(), 2);
    let second = Work::of(&both).since(Work::of(&first));
    assert_eq!(second.jacobians, 0);
    assert_eq!(second.cache_hits, 1, "only the raw Jacobian is reused");
    assert_eq!(second.factorizations, 1);
}

struct FailOnceLinear {
    calls: Cell<usize>,
    fail_at: Cell<Option<usize>>,
}

impl FailOnceLinear {
    const fn new(fail_at: usize) -> Self {
        Self {
            calls: Cell::new(0),
            fail_at: Cell::new(Some(fail_at)),
        }
    }
}

impl OdeSystem<f64> for FailOnceLinear {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail_at.get() == Some(call) {
            self.fail_at.set(None);
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("derivative"),
                InvalidInputReason::Inconsistent,
                "deliberate residual failure",
            ));
        }
        derivative[0] = -state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        row_major(jacobian, "row-major output required")?[0] = -1.0;
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

/// A failed epoch leaves the state and attempt record transactional; the next
/// `advance` is a fresh epoch that rebuilds its Jacobian and factor.
#[test]
fn failed_factor_is_reused_only_by_an_identical_retry() {
    for h in [0.1, 0.2] {
        let system = FailOnceLinear::new(2);
        let plan = (BackwardEuler::<f64>::new(), fixed(h));
        let mut workspace = OdeWorkspace::new();
        let mut state = start(&[1.0]);
        assert!(run(&plan, &system, h, &mut state, &mut workspace).is_err());
        assert_eq!(state.values(), &[1.0]);
        assert_eq!(state.time(), 0.0);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Failed
        );

        let retry = run(&plan, &system, h, &mut state, &mut workspace).unwrap();
        assert!((state.values()[0] - 1.0 / (1.0 + h)).abs() < 1e-14);
        let retry = Work::of(&retry);
        assert_eq!(retry.jacobians, 1);
        assert_eq!(retry.factorizations, 1);
        assert_eq!(retry.cache_hits, 0);
    }
}

#[test]
fn full_newton_rebuilds_after_every_jacobian_change() {
    let simplified = (BackwardEuler::<f64>::new(), fixed(0.2));
    let full = (
        BackwardEuler::<f64>::new().with_newton_mode(NewtonMode::Full),
        fixed(0.2),
    );
    let (simplified_report, simplified_state) = solve(&simplified, &StateDependent, 0.2, &[1.0]);
    let (full_report, full_state) = solve(&full, &StateDependent, 0.2, &[1.0]);
    assert!((simplified_state[0] - full_state[0]).abs() < 2e-8);
    let simplified = Work::of(&simplified_report);
    let full = Work::of(&full_report);
    assert_eq!(simplified.jacobians, 1);
    assert_eq!(simplified.factorizations, 1);
    assert!(full.jacobians > 1);
    assert_eq!(full.jacobians, full.linear_solves);
    assert_eq!(full.factorizations, full.linear_solves);
}

struct RevisionedMass {
    revision: Cell<u64>,
    mass: Cell<f64>,
}

impl RevisionedMass {
    const fn new() -> Self {
        Self {
            revision: Cell::new(1),
            mass: Cell::new(2.0),
        }
    }
}

struct ResizableMassSystem {
    dimension: Cell<usize>,
}

impl OdeSystem<f64> for ResizableMassSystem {
    fn dimension(&self) -> usize {
        self.dimension.get()
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative
            .iter_mut()
            .zip(state)
            .for_each(|(output, &value)| *output = -2.0 * value);
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let jacobian_data = row_major(jacobian, "row-major output required")?;
        let dimension = self.dimension.get();
        jacobian_data.fill(0.0);
        (0..dimension).for_each(|index| {
            jacobian_data[index * dimension + index] = -2.0;
        });
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }

    fn mass_matrix_revision(&self) -> Option<u64> {
        Some(1)
    }

    fn mass_matrix(
        &self,
        mass: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let mass_data = row_major(mass, "row-major output required")?;
        let dimension = self.dimension.get();
        mass_data.fill(0.0);
        (0..dimension).for_each(|index| {
            mass_data[index * dimension + index] = 2.0;
        });
        Ok(())
    }
}

impl OdeSystem<f64> for RevisionedMass {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = -2.0 * state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        row_major(jacobian, "row-major output required")?[0] = -2.0;
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }

    fn mass_matrix_revision(&self) -> Option<u64> {
        Some(self.revision.get())
    }

    fn mass_matrix(
        &self,
        mass: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        row_major(mass, "row-major output required")?[0] = self.mass.get();
        Ok(())
    }
}

/// Two steps of `h`; after the first accepted step the sink records the state and
/// switches the mass matrix to revision 2 (`M = 4`). Returns the work of the second
/// step, the state after the first step and the final state.
fn revise_mass_between_steps<M>(method: M, options: OdeOptions<f64>, h: f64) -> (Work, f64, f64)
where
    M: OdeMethod<f64> + Clone,
{
    let plan = (
        method,
        options.with_output_schedule(OdeOutputSchedule::AcceptedSteps),
    );
    let (first, _) = solve(&plan, &RevisionedMass::new(), h, &[1.0]);

    let system = RevisionedMass::new();
    let after_first = Cell::new(f64::NAN);
    let mut sink = |time: f64, state: &[f64]| {
        if time == h {
            after_first.set(state[0]);
            system.revision.set(2);
            system.mass.set(4.0);
        }
        Ok(())
    };
    let mut workspace = OdeWorkspace::new();
    let mut state = start(&[1.0]);
    let both = plan
        .0
        .advance(
            &OdeProblem::new(&system, h + h),
            &plan.1,
            &mut state,
            &mut sink,
            &mut workspace,
            sequential(),
        )
        .unwrap();
    assert_eq!(both.accepted_steps(), 2);
    (
        Work::of(&both).since(Work::of(&first)),
        after_first.get(),
        state.values()[0],
    )
}

#[test]
fn mass_revision_changes_invalidate_the_effective_factor() {
    let (second, before, after) =
        revise_mass_between_steps(BackwardEuler::<f64>::new(), fixed(0.1), 0.1);
    assert!((before - 2.0 / 2.2).abs() < 1e-14);
    assert!((after - before * 4.0 / 4.2).abs() < 1e-14);
    assert_eq!(second.cache_hits, 1, "only the raw Jacobian is reused");
    assert_eq!(second.factorizations, 1);
}

#[test]
fn radau_mass_revision_invalidates_both_factors_but_keeps_the_jacobian() {
    let (second, before, after) =
        revise_mass_between_steps(RadauIia::<f64>::new(), pinned(0.1), 0.1);
    assert!((before - (-0.1_f64).exp()).abs() < 2e-10);
    assert!((after - before * (-0.05_f64).exp()).abs() < 1e-11);
    assert_eq!(second.cache_hits, 1, "only the raw Jacobian is reused");
    assert_eq!(second.factorizations, 2);
}

/// Continue on `shared` (already used by another stage shape) and on a fresh
/// workspace from the same state; both must agree.
fn assert_shape_change_keeps_mass<M>(first: (M, OdeOptions<f64>), second: (M, OdeOptions<f64>))
where
    M: OdeMethod<f64> + Clone,
{
    let system = RevisionedMass::new();
    let mut shared = OdeWorkspace::new();
    let mut shared_state = start(&[1.0]);
    complete(&first, &system, 0.001, &mut shared_state, &mut shared);

    let mut expected = shared_state.clone();
    let mut fresh = OdeWorkspace::new();
    complete(&second, &system, 0.002, &mut expected, &mut fresh);
    complete(&second, &system, 0.002, &mut shared_state, &mut shared);
    assert!((shared_state.values()[0] - expected.values()[0]).abs() <= 1e-14);
}

#[test]
fn stage_shape_change_preserves_a_valid_mass_matrix() {
    // Runtime method selection shares one workspace across method families.
    assert_shape_change_keeps_mass(
        (
            BuiltinOdeMethod::from(BackwardEuler::<f64>::new()),
            fixed(0.001),
        ),
        (BuiltinOdeMethod::from(GaussRk::<f64>::new()), fixed(0.001)),
    );
    // Full Newton prepares one block per Gauss stage; simplified Newton shares one.
    assert_shape_change_keeps_mass(
        (
            GaussRk::<f64>::new().with_newton_mode(NewtonMode::Full),
            fixed(0.001),
        ),
        (GaussRk::<f64>::new(), fixed(0.001)),
    );
}

#[test]
fn dimension_change_reloads_an_unchanged_mass_revision() {
    let system = ResizableMassSystem {
        dimension: Cell::new(1),
    };
    let mut workspace = OdeWorkspace::new();
    let narrow = (BackwardEuler::<f64>::new(), fixed(0.1));
    let mut state = start(&[1.0]);
    complete(&narrow, &system, 0.1, &mut state, &mut workspace);

    system.dimension.set(2);
    let wide = (BackwardEuler::<f64>::new(), fixed(0.1));
    let mut resized_state = start(&[1.0, 1.0]);
    complete(&wide, &system, 0.1, &mut resized_state, &mut workspace);
    resized_state
        .values()
        .iter()
        .for_each(|&value| assert!((value - 1.0 / 1.1).abs() <= 1e-12));
}

fn assert_full_newton_step<M>(method: M)
where
    M: OdeMethod<f64> + Clone,
{
    let name = core::any::type_name::<M>();
    let plan = (method, fixed(0.05));
    let (report, state) = solve(&plan, &StateDependent, 0.05, &[1.0]);
    assert!(state[0].is_finite(), "{name}");
    let work = Work::of(&report);
    assert!(work.jacobians > 0, "{name}");
    assert!(work.linear_solves > 0, "{name}");
    assert_eq!(work.factorizations, work.linear_solves, "{name}");
}

#[test]
fn full_newton_is_available_on_endpoint_gauss_sdirk_and_bdf_methods() {
    assert_full_newton_step(BackwardEuler::<f64>::new().with_newton_mode(NewtonMode::Full));
    assert_full_newton_step(ImplicitTrapezoid::<f64>::new().with_newton_mode(NewtonMode::Full));
    assert_full_newton_step(Sdirk::<f64>::new().with_newton_mode(NewtonMode::Full));
    assert_full_newton_step(GaussRk::<f64>::new().with_newton_mode(NewtonMode::Full));
    assert_full_newton_step(
        Bdf::<f64>::new(1)
            .unwrap()
            .with_newton_mode(NewtonMode::Full),
    );
}

struct SingularMass;

impl OdeSystem<f64> for SingularMass {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = 0.0;
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        row_major(jacobian, "row-major output required")?[0] = 0.0;
        Ok(Some(()))
    }

    fn mass_matrix_revision(&self) -> Option<u64> {
        Some(0)
    }

    fn mass_matrix(
        &self,
        mass: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        row_major(mass, "row-major output required")?[0] = 0.0;
        Ok(())
    }
}

#[test]
fn singular_effective_matrix_is_structured_and_transactional() {
    let plan = (BackwardEuler::<f64>::new(), fixed(0.1));
    let mut workspace = OdeWorkspace::new();
    let mut state = start(&[1.0]);
    let error = run(&plan, &SingularMass, 0.1, &mut state, &mut workspace).unwrap_err();
    assert!(matches!(error.view(), SciOxideErrorView::Singular { .. }));
    assert_eq!(state.values(), &[1.0]);
    assert_eq!(state.time(), 0.0);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Failed
    );
}

struct AutomaticLinear;

impl OdeSystem<f64> for AutomaticLinear {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = -state[0];
        Ok(())
    }

    fn automatic_jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        row_major(jacobian, "row-major output required")?[0] = -1.0;
        Ok(Some(()))
    }
}

#[test]
fn automatic_and_finite_difference_jacobians_use_workspace_buffers() {
    let automatic = (
        BackwardEuler::<f64>::new().with_derivative_strategy(DerivativeStrategy::Automatic),
        fixed(0.1),
    );
    let (report, _) = solve(&automatic, &AutomaticLinear, 0.1, &[1.0]);
    let work = Work::of(&report);
    assert_eq!(work.jacobians, 1);
    assert_eq!(work.rhs, 4);

    let finite_difference = (
        BackwardEuler::<f64>::new().with_derivative_strategy(DerivativeStrategy::Forward),
        fixed(0.1),
    );
    let (report, _) = solve(
        &finite_difference,
        &ConstantLinear::new(1, -1.0),
        0.1,
        &[1.0],
    );
    let work = Work::of(&report);
    // Difference Jacobians are formed from counted RHS probes, not Jacobian providers.
    assert_eq!(work.jacobians, 0);
    assert_eq!(
        work.rhs, 5,
        "four stage evaluations and one difference probe"
    );
}

struct FiniteDifferenceFailure {
    fail: Cell<bool>,
    calls: Cell<usize>,
}

impl OdeSystem<f64> for FiniteDifferenceFailure {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail.get() && call == 2 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("derivative"),
                InvalidInputReason::Inconsistent,
                "deliberate perturbation failure",
            ));
        }
        derivative[0] = -state[0];
        Ok(())
    }
}

#[test]
fn finite_difference_error_restores_inputs_and_allows_retry() {
    let system = FiniteDifferenceFailure {
        fail: Cell::new(true),
        calls: Cell::new(0),
    };
    let plan = (
        BackwardEuler::<f64>::new().with_derivative_strategy(DerivativeStrategy::Forward),
        fixed(0.1),
    );
    let mut workspace = OdeWorkspace::new();
    let mut state = start(&[1.0]);
    assert!(run(&plan, &system, 0.1, &mut state, &mut workspace).is_err());
    assert_eq!(state.values(), &[1.0]);
    system.fail.set(false);
    complete(&plan, &system, 0.1, &mut state, &mut workspace);
    assert!((state.values()[0] - 1.0 / 1.1).abs() < 1e-8);
}

struct MalformedRhs;

impl OdeSystem<f64> for MalformedRhs {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        _state: &[f64],
        _derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(1),
            scioxide_core::Dimensions::one(0),
        ))
    }
}

struct MalformedJacobian;

impl OdeSystem<f64> for MalformedJacobian {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = -state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        _jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(1),
            scioxide_core::Dimensions::one(0),
        ))
    }
}

#[derive(Clone, Copy)]
enum NonFiniteJacobianSource {
    Analytic,
    Automatic,
}

struct NonFiniteJacobian {
    source: NonFiniteJacobianSource,
}

impl OdeSystem<f64> for NonFiniteJacobian {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = -state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let jacobian_data = row_major(jacobian, "row-major output required")?;
        if matches!(self.source, NonFiniteJacobianSource::Analytic) {
            jacobian_data[0] = f64::NAN;
            return Ok(Some(()));
        }
        Ok(None)
    }

    fn automatic_jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let jacobian_data = row_major(jacobian, "row-major output required")?;
        if matches!(self.source, NonFiniteJacobianSource::Automatic) {
            jacobian_data[0] = f64::INFINITY;
            return Ok(Some(()));
        }
        Ok(None)
    }
}

struct MalformedRejectedResidual {
    fail: Cell<bool>,
}

impl OdeSystem<f64> for MalformedRejectedResidual {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if time > 0.0 && self.fail.replace(false) {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(1),
                scioxide_core::Dimensions::one(0),
            ));
        }
        derivative[0] = -state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        row_major(jacobian, "row-major output required")?[0] = -1.0;
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

#[test]
fn malformed_callback_shapes_return_dimension_errors_without_mutating_state() {
    for system in [
        &MalformedRhs as &dyn OdeSystem<f64>,
        &MalformedJacobian as &dyn OdeSystem<f64>,
    ] {
        let plan = (BackwardEuler::<f64>::new(), fixed(0.1));
        let mut workspace = OdeWorkspace::new();
        let mut state = start(&[1.0]);
        let error = run(&plan, system, 0.1, &mut state, &mut workspace).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(state.values(), &[1.0]);

        complete(
            &plan,
            &ConstantLinear::new(1, -1.0),
            0.1,
            &mut state,
            &mut workspace,
        );
        assert!((state.values()[0] - 1.0 / 1.1).abs() <= 1e-12);
    }
}

#[test]
fn non_finite_jacobians_have_callback_provenance() {
    for source in [
        NonFiniteJacobianSource::Analytic,
        NonFiniteJacobianSource::Automatic,
    ] {
        let system = NonFiniteJacobian { source };
        let strategy = match source {
            NonFiniteJacobianSource::Analytic => DerivativeStrategy::Analytic,
            NonFiniteJacobianSource::Automatic => DerivativeStrategy::Automatic,
        };
        let plan = (
            BackwardEuler::<f64>::new().with_derivative_strategy(strategy),
            fixed(0.1),
        );
        let mut workspace = OdeWorkspace::new();
        let mut state = start(&[1.0]);
        let error = run(&plan, &system, 0.1, &mut state, &mut workspace).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        ));
        assert_eq!(state.values(), &[1.0]);
    }
}

/// A trial residual failure leaves the accepted state; the retry epoch rebuilds its
/// factor and converges in the same two Newton iterations as an undisturbed step.
#[test]
fn identical_retry_repairs_residual_scratch_without_overwriting_its_factor() {
    let system = MalformedRejectedResidual {
        fail: Cell::new(true),
    };
    let plan = (BackwardEuler::<f64>::new(), fixed(0.1));
    let mut workspace = OdeWorkspace::new();
    let mut state = start(&[1.0]);
    assert!(run(&plan, &system, 0.1, &mut state, &mut workspace).is_err());
    assert_eq!(state.values(), &[1.0]);

    let report = run(&plan, &system, 0.1, &mut state, &mut workspace).unwrap();
    assert!((state.values()[0] - 1.0 / 1.1).abs() <= 1e-12);
    let work = Work::of(&report);
    assert_eq!(work.factorizations, 1);
    assert_eq!(work.newton_iterations, 1);
}

#[test]
fn convergence_exhaustion_does_not_commit_bdf_state_or_history() {
    let exhausted = (
        Bdf::<f64>::new(2).unwrap().with_max_iterations(Budget::ONE),
        fixed(0.5),
    );
    let mut workspace = OdeWorkspace::new();
    let mut state = start(&[1.0]);
    let error = run(&exhausted, &StateDependent, 0.5, &mut state, &mut workspace).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert_eq!(state.values(), &[1.0]);
    assert_eq!(state.time(), 0.0);

    let recovery = (Bdf::<f64>::new(2).unwrap(), fixed(0.01));
    let mut control_workspace = OdeWorkspace::new();
    let mut control_state = start(&[1.0]);
    let system = ConstantLinear::new(1, -1.0);
    complete(&recovery, &system, 0.01, &mut state, &mut workspace);
    complete(
        &recovery,
        &system,
        0.01,
        &mut control_state,
        &mut control_workspace,
    );
    assert_eq!(state, control_state);
}

#[test]
fn gauss_accepts_an_empty_system() {
    let system = ConstantLinear::new(0, -1.0);
    let plan = (GaussRk::<f64>::new(), fixed(0.01));
    let mut workspace = plan
        .0
        .workspace(&OdeProblem::new(&system, 0.01), &plan.1)
        .unwrap();
    let mut state = start(&[]);
    complete(&plan, &system, 0.01, &mut state, &mut workspace);
}

#[test]
fn implicit_workspace_supports_dimension_changes_and_checked_failures() {
    let mut workspace = OdeWorkspace::new();
    for dimension in [1, 4, 2, 0] {
        let system = ConstantLinear::new(dimension, -1.0);
        let plan = (BackwardEuler::<f64>::new(), fixed(0.01));
        let mut state = start(&vec![1.0; dimension]);
        complete(&plan, &system, 0.01, &mut state, &mut workspace);
        let requirements = plan
            .0
            .workspace_requirements(&OdeProblem::new(&system, 0.01), &plan.1)
            .unwrap();
        assert_eq!(requirements.dimension(), dimension);
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::Sufficient
        );
    }

    let huge = ConstantLinear::new(usize::MAX, -1.0);
    let plan = (BackwardEuler::<f64>::new(), fixed(0.01));
    let requirements = plan
        .0
        .workspace_requirements(&OdeProblem::new(&huge, 0.01), &plan.1)
        .unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    // Driver vectors of `usize::MAX` elements fail before any stage storage is sized.
    let error = workspace.try_reserve(&requirements).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::AllocationFailure { .. } | SciOxideErrorView::SizeOverflow { .. }
    ));

    let system = ConstantLinear::new(1, -1.0);
    let plan = (BackwardEuler::<f64>::new(), fixed(0.01));
    let mut state = start(&[1.0]);
    complete(&plan, &system, 0.01, &mut state, &mut workspace);
    assert!((state.values()[0] - 1.0 / 1.01).abs() <= 1e-14);
}

struct ConstantLinearF32;

impl OdeSystem<f32> for ConstantLinearF32 {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f32,
        state: &[f32],
        derivative: &mut [f32],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = -state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f32,
        _state: &[f32],
        jacobian: &mut MatrixViewMut<'_, f32>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let jacobian_data = jacobian.as_contiguous_slice_mut().ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("jacobian"),
                InvalidInputReason::UnsupportedConfiguration,
                "row-major output required",
            )
        })?;
        jacobian_data[0] = -1.0;
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

#[test]
fn prepared_implicit_path_supports_f32_and_f64() {
    let options = OdeOptions::<f32>::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(1.0, 1.0, NormSelection::RootMeanSquare).unwrap(),
        )
        .with_initial_step(0.1)
        .with_minimum_step(0.1)
        .with_maximum_step(0.1);
    let plan = (RadauIia::<f32>::new(), options);
    let problem = OdeProblem::new(&ConstantLinearF32, 0.1);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::from_slice(0.0_f32, &[1.0]).unwrap();
    let report = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            sequential(),
        )
        .unwrap();
    assert_eq!(report.accepted_steps(), 1);
    assert!((state.values()[0] - (-0.1_f32).exp()).abs() < 1e-5);

    let plan = (RadauIia::<f64>::new(), pinned(0.1));
    let (report, state) = solve(&plan, &ConstantLinear::new(1, -1.0), 0.1, &[1.0]);
    assert_eq!(report.accepted_steps(), 1);
    assert!((state[0] - (-0.1_f64).exp()).abs() < 1e-8);
}

struct VanDerPol {
    stiffness: f64,
}

impl OdeSystem<f64> for VanDerPol {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let position = state[0];
        let velocity = state[1];
        derivative[0] = velocity;
        derivative[1] = self.stiffness * (1.0 - position * position) * velocity - position;
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let position = state[0];
        let velocity = state[1];
        row_major(jacobian, "row-major output required")?.copy_from_slice(&[
            0.0,
            1.0,
            -2.0 * self.stiffness * position * velocity - 1.0,
            self.stiffness * (1.0 - position * position),
        ]);
        Ok(Some(()))
    }
}

#[test]
fn radau_workspace_handles_van_der_pol_at_multiple_stiffnesses() {
    for stiffness in [10.0, 1_000.0] {
        // Five steps of 1e-4 (endpoint rounding may add one final ulp-sized step).
        let plan = (RadauIia::<f64>::new(), pinned(1e-4));
        let (report, state) = solve(&plan, &VanDerPol { stiffness }, 5e-4, &[2.0, 0.0]);
        assert_eq!(report.final_time(), 5e-4);
        assert!(state.iter().all(|value| value.is_finite()));
        assert!(state[0] < 2.001);
    }
}
