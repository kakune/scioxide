//! Initial-value problems and the integration state.

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Vector;

use crate::stiff::stage::Provider;
use crate::{EventFunction, OdeSystem};

/// Describes an initial-value problem: the system, the end time, the event
/// functions and an optional stage solver.
///
/// The start time and initial values are the [`OdeState`] passed to
/// [`OdeMethod::advance`](crate::OdeMethod::advance). Integration runs from
/// `state.time()` to [`Self::end`], forward or backward in time.
#[must_use]
pub struct OdeProblem<'a, S, Sys: ?Sized, T = S>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    system: &'a Sys,
    end: S,
    events: &'a [&'a dyn EventFunction<S, T>],
    stage: Provider<'a, S, T>,
}

impl<S, Sys: ?Sized, T> Clone for OdeProblem<'_, S, Sys, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<S, Sys: ?Sized, T> Copy for OdeProblem<'_, S, Sys, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
}

impl<'a, S, Sys, T> OdeProblem<'a, S, Sys, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    Sys: OdeSystem<S, T> + ?Sized,
{
    /// Creates a problem that integrates `system` to `end`, without events.
    pub const fn new(system: &'a Sys, end: S) -> Self {
        Self {
            system,
            end,
            events: &[],
            stage: Provider::Native,
        }
    }

    /// Sets the event functions to locate during integration.
    ///
    /// Event indices in reports and sinks are positions in this slice; see
    /// [Events and output](crate#events-and-output).
    pub const fn with_events(mut self, events: &'a [&'a dyn EventFunction<S, T>]) -> Self {
        self.events = events;
        self
    }

    /// Sets the solver of the implicit stage equations: the built-in Newton
    /// iteration, or a linear or nonlinear solver you provide; see
    /// [Implicit stages](crate#implicit-stages).
    ///
    /// Only methods with implicit stages accept a solver you provide.
    pub const fn with_stage_solver(mut self, solver: crate::OdeStageSolver<'a, S, T>) -> Self {
        self.stage = solver;
        self
    }

    /// Returns whether a stage solver you provide replaces the built-in one.
    pub const fn has_stage_solver(&self) -> bool {
        self.stage.is_provided()
    }

    pub(crate) const fn stage_provider(&self) -> Provider<'a, S, T> {
        self.stage
    }

    /// Returns the system.
    pub const fn system(&self) -> &'a Sys {
        self.system
    }

    /// Returns the end time.
    pub const fn end(&self) -> S {
        self.end
    }

    /// Returns the event functions.
    pub const fn events(&self) -> &'a [&'a dyn EventFunction<S, T>] {
        self.events
    }
}
/// Holds the time and state values that [`OdeMethod::advance`](crate::OdeMethod::advance)
/// integrates in place.
///
/// After `advance` returns, it holds the last accepted point. If `advance` rejects
/// its arguments before integrating, the state is unchanged.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct OdeState<S, T: NumericScalar = S> {
    time: S,
    values: Vector<T>,
}

impl<S: RealScalar, T: NumericScalar<Real = S>> OdeState<S, T> {
    /// Creates a state at `time` with `values`.
    pub const fn new(time: S, values: Vector<T>) -> Self {
        Self { time, values }
    }

    /// Creates a state at `time` with a copy of `values`.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the values cannot be allocated.
    pub fn from_slice(time: S, values: &[T]) -> Result<Self, SciOxideError> {
        let mut vector = Vector::zeros(values.len())?;
        vector.as_mut_slice().copy_from_slice(values);
        Ok(Self::new(time, vector))
    }

    /// Returns the time.
    pub const fn time(&self) -> S {
        self.time
    }

    /// Returns the state values.
    pub fn values(&self) -> &[T] {
        self.values.as_slice()
    }

    /// Sets the time to `time` and copies `values` into the state, without
    /// allocating.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `values` has a different length than the state.
    pub fn reset(&mut self, time: S, values: &[T]) -> Result<(), SciOxideError> {
        if values.len() != self.values.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.values.len()),
                scioxide_core::Dimensions::one(values.len()),
            ));
        }
        self.values.as_mut_slice().copy_from_slice(values);
        self.time = time;
        Ok(())
    }

    /// Returns the time and the values.
    pub fn into_parts(self) -> (S, Vector<T>) {
        (self.time, self.values)
    }

    pub(crate) fn parts_mut(&mut self) -> (&mut S, &mut Vector<T>) {
        (&mut self.time, &mut self.values)
    }
}
