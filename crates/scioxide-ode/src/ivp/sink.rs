//! Output sinks of initial-value integration.

use alloc::vec::Vec;

use scioxide_core::backend::try_reserve;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Vector;

use crate::{EventEvidence, EventOccurrence};

/// Describes one located event, as passed to [`OdeOutputSink::event`].
///
/// The state borrows driver storage for the duration of the call; use
/// [`EventInfo::from_event`] to keep a copy.
#[derive(Debug, Clone, Copy)]
pub struct OdeEvent<'a, S: RealScalar, T = S> {
    pub(crate) index: usize,
    pub(crate) time: S,
    pub(crate) state: &'a [T],
    pub(crate) occurrence: EventOccurrence,
    pub(crate) terminal: bool,
    pub(crate) evidence: EventEvidence<S>,
}

impl<'a, S: RealScalar, T> OdeEvent<'a, S, T> {
    /// Returns the index of the event function in the problem's event slice.
    pub const fn index(&self) -> usize {
        self.index
    }
    /// Returns the located event time.
    pub const fn time(&self) -> S {
        self.time
    }
    /// Returns the state at the event time, from the dense output of the step.
    pub const fn state(&self) -> &'a [T] {
        self.state
    }
    /// Returns whether the event function rose through zero, fell through zero
    /// or touched it.
    pub const fn occurrence(&self) -> EventOccurrence {
        self.occurrence
    }
    /// Returns whether this event stops the integration.
    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }
    /// Returns how the event was located: a sampled zero, or a final bracket and
    /// residual.
    ///
    /// It describes the root search only, not the accuracy of the ODE solution.
    pub const fn evidence(&self) -> EventEvidence<S> {
        self.evidence
    }
}

/// Receives the states and events of an initial-value integration.
///
/// The [`OdeOutputSchedule`](crate::OdeOutputSchedule) chooses the states. The
/// sink receives the output of each step when the step is accepted, so a later
/// failure does not withdraw it. The events of a step come after its states, in
/// integration order, and by event index at equal times. An error returned by
/// the sink stops the integration and is returned unchanged.
///
/// `()` discards all output, a closure `FnMut(S, &[T]) -> Result<(), SciOxideError>`
/// receives the states, and [`OdeTrajectory`] stores states and events.
pub trait OdeOutputSink<S: RealScalar, T = S> {
    /// Receives the state `state` at `time`.
    ///
    /// # Errors
    ///
    /// Any error stops the integration and is returned unchanged.
    fn emit(&mut self, time: S, state: &[T]) -> Result<(), SciOxideError>;

    /// Receives the state at the output time with position `index` in the output
    /// schedule. The default calls [`Self::emit`].
    ///
    /// # Errors
    ///
    /// Any error stops the integration and is returned unchanged.
    fn emit_requested(&mut self, index: usize, time: S, state: &[T]) -> Result<(), SciOxideError> {
        let _ = index;
        self.emit(time, state)
    }

    /// Receives one located event. The default ignores events.
    ///
    /// # Errors
    ///
    /// Any error stops the integration and is returned unchanged.
    fn event(&mut self, event: OdeEvent<'_, S, T>) -> Result<(), SciOxideError> {
        let _ = event;
        Ok(())
    }
}

/// Discards all output.
impl<S: RealScalar, T> OdeOutputSink<S, T> for () {
    fn emit(&mut self, _: S, _: &[T]) -> Result<(), SciOxideError> {
        Ok(())
    }
}

impl<S: RealScalar, T, F> OdeOutputSink<S, T> for F
where
    F: FnMut(S, &[T]) -> Result<(), SciOxideError>,
{
    fn emit(&mut self, time: S, state: &[T]) -> Result<(), SciOxideError> {
        self(time, state)
    }
}

/// Holds a copy of one located event.
#[derive(Debug, Clone, PartialEq)]
pub struct EventInfo<S: RealScalar, T: NumericScalar = S> {
    index: usize,
    time: S,
    state: Vector<T>,
    occurrence: EventOccurrence,
    terminal: bool,
    evidence: EventEvidence<S>,
}

impl<S: RealScalar, T: NumericScalar<Real = S>> EventInfo<S, T> {
    /// Copies `event`, including its state.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the state cannot be allocated.
    pub fn from_event(event: OdeEvent<'_, S, T>) -> Result<Self, SciOxideError> {
        let mut state = Vector::zeros(event.state.len())?;
        state.as_mut_slice().copy_from_slice(event.state);
        Ok(Self {
            index: event.index,
            time: event.time,
            state,
            occurrence: event.occurrence,
            terminal: event.terminal,
            evidence: event.evidence,
        })
    }
    /// Returns the index of the event function.
    pub const fn index(&self) -> usize {
        self.index
    }
    /// Returns the located event time.
    pub const fn time(&self) -> S {
        self.time
    }
    /// Returns the state at the event time.
    pub fn state(&self) -> &[T] {
        self.state.as_slice()
    }
    /// Returns whether the event function rose through zero, fell through zero
    /// or touched it.
    pub const fn occurrence(&self) -> EventOccurrence {
        self.occurrence
    }
    /// Returns whether this event stopped the integration.
    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }
    /// Returns how the event was located.
    pub const fn evidence(&self) -> EventEvidence<S> {
        self.evidence
    }
}

/// Stores every state and event it receives, as an output sink.
///
/// It allocates for each state and event.
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct OdeTrajectory<S: RealScalar, T: NumericScalar = S> {
    times: Vec<S>,
    states: Vec<Vector<T>>,
    events: Vec<EventInfo<S, T>>,
}

impl<S: RealScalar, T: NumericScalar<Real = S>> OdeTrajectory<S, T> {
    /// Creates an empty trajectory.
    pub const fn new() -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
            events: Vec::new(),
        }
    }
    /// Returns the times of the stored states, in the order received.
    pub fn times(&self) -> &[S] {
        &self.times
    }
    /// Returns the stored states, in the order received.
    pub fn states(&self) -> &[Vector<T>] {
        &self.states
    }
    /// Returns the stored events, in the order received.
    pub fn events(&self) -> &[EventInfo<S, T>] {
        &self.events
    }
    /// Returns the times, states and events.
    #[allow(
        clippy::type_complexity,
        reason = "the three owned columns of the trajectory"
    )]
    pub fn into_parts(self) -> (Vec<S>, Vec<Vector<T>>, Vec<EventInfo<S, T>>) {
        (self.times, self.states, self.events)
    }
}

impl<S: RealScalar, T: NumericScalar<Real = S>> OdeOutputSink<S, T> for OdeTrajectory<S, T> {
    fn emit(&mut self, time: S, state: &[T]) -> Result<(), SciOxideError> {
        let mut copied = Vector::zeros(state.len())?;
        copied.as_mut_slice().copy_from_slice(state);
        try_reserve(
            &mut self.times,
            1,
            scioxide_core::quantity!("ode_trajectory_time_storage"),
        )?;
        try_reserve(
            &mut self.states,
            1,
            scioxide_core::quantity!("ode_trajectory_state_storage"),
        )?;
        self.times.push(time);
        self.states.push(copied);
        Ok(())
    }

    fn event(&mut self, event: OdeEvent<'_, S, T>) -> Result<(), SciOxideError> {
        let event = EventInfo::from_event(event)?;
        try_reserve(
            &mut self.events,
            1,
            scioxide_core::quantity!("ode_trajectory_event_storage"),
        )?;
        self.events.push(event);
        Ok(())
    }
}
