//! Caller-controlled output of transient PDE advances.

use alloc::vec::Vec;

use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_ode::OdeOutputSchedule;

use crate::mol::MethodOfLinesSystem;

/// Caller-selected output policy for time-dependent PDE advances.
#[derive(Debug, Clone, PartialEq, Default)]
#[must_use]
pub enum PdeOutputSchedule<S: RealScalar> {
    /// Emit only the final accepted spatial state of each advance.
    #[default]
    FinalOnly,
    /// Emit the initial state and every accepted time step.
    TimeSteps,
    /// Emit requested times in `[0, end]`, including repeated times, in caller order.
    /// Interior samples use the selected ODE method's dense output; a continued
    /// advance skips the times before its state. Methods without dense output (the
    /// built-in implicit schemes and Rannacher startup) reject requested times at
    /// preflight whenever the problem has free dynamics.
    Times(Vec<S>),
}

impl<S: RealScalar> PdeOutputSchedule<S> {
    /// Requested times must be finite, nonnegative and nondecreasing.
    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        if let Self::Times(times) = self
            && (times.iter().any(|t| !t.is_finite() || *t < S::ZERO)
                || times.windows(2).any(|pair| pair[0] > pair[1]))
        {
            return Err(crate::validation::invalid(
                scioxide_core::argument!("options"),
                "requested PDE times must be finite, nonnegative and ordered",
            ));
        }
        Ok(())
    }
}

/// Borrowed full physical field (discretization unknown order) at one observation.
///
/// Convert values to typed fields with [`crate::SpatialDiscretization::field`].
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct PdeObservation<'a, S: RealScalar, T> {
    time: S,
    values: &'a [T],
    request_index: Option<usize>,
}

impl<'a, S: RealScalar, T: Copy> PdeObservation<'a, S, T> {
    /// Construct an observation after checking its time.
    /// # Errors
    /// Rejects a non-finite time.
    pub fn new(
        time: S,
        values: &'a [T],
        request_index: Option<usize>,
    ) -> Result<Self, SciOxideError> {
        if !time.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "time must be finite",
            ));
        }
        Ok(Self {
            time,
            values,
            request_index,
        })
    }
    /// Observation time.
    pub const fn time(&self) -> S {
        self.time
    }
    /// Full physical field values, including reconstructed Dirichlet unknowns.
    pub const fn values(&self) -> &'a [T] {
        self.values
    }
    /// Index into [`PdeOutputSchedule::Times`], or `None` for final or step output.
    pub const fn request_index(&self) -> Option<usize> {
        self.request_index
    }
}

/// Fallible receiver for streamed PDE output; `()` ignores every observation.
pub trait PdeOutputSink<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    /// Consume one finite spatial state emitted in integration order.
    ///
    /// # Errors
    ///
    /// Receiver errors stop the advance and are returned unchanged; the caller's
    /// state keeps the last accepted point and emitted samples stay emitted.
    fn observe(&mut self, observation: PdeObservation<'_, S, T>) -> Result<(), SciOxideError>;
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> PdeOutputSink<S, T>
    for ()
{
    fn observe(&mut self, _: PdeObservation<'_, S, T>) -> Result<(), SciOxideError> {
        Ok(())
    }
}

impl<S, T: NumericScalar<Real = S>, F> PdeOutputSink<S, T> for F
where
    S: RealScalar + NumericScalar<Real = S>,
    F: for<'a> FnMut(PdeObservation<'a, S, T>) -> Result<(), SciOxideError>,
{
    fn observe(&mut self, observation: PdeObservation<'_, S, T>) -> Result<(), SciOxideError> {
        self(observation)
    }
}

/// Opt-in owned trajectory receiver for time-dependent PDE output.
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct PdeTrajectory<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    times: Vec<S>,
    states: Vec<Vec<T>>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> PdeTrajectory<S, T> {
    /// Construct empty trajectory storage.
    pub const fn new() -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
        }
    }

    /// Recorded times in emission order.
    pub fn times(&self) -> &[S] {
        &self.times
    }

    /// Recorded full fields corresponding to [`Self::times`].
    pub fn states(&self) -> &[Vec<T>] {
        &self.states
    }

    /// Consume the receiver into its time and state storage.
    pub fn into_parts(self) -> (Vec<S>, Vec<Vec<T>>) {
        (self.times, self.states)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> PdeOutputSink<S, T>
    for PdeTrajectory<S, T>
{
    fn observe(&mut self, observation: PdeObservation<'_, S, T>) -> Result<(), SciOxideError> {
        let copied = scioxide_core::backend::try_copy_slice(
            observation.values,
            scioxide_core::quantity!("pde_trajectory_state"),
        )?;
        self.times.try_reserve(1).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("pde_trajectory_time_storage"),
                self.times.len() + 1,
            )
        })?;
        self.states.try_reserve(1).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("pde_trajectory_state_storage"),
                self.states.len() + 1,
            )
        })?;
        self.times.push(observation.time);
        self.states.push(copied);
        Ok(())
    }
}

/// Reconstruct the full field at `time` from ODE state and emit it.
pub(super) fn emit<S, K>(
    system: &MethodOfLinesSystem<'_, S>,
    field: &mut [S],
    sink: &mut K,
    time: S,
    state: &[S],
    request_index: Option<usize>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    K: PdeOutputSink<S> + ?Sized,
{
    system.reconstruct_into(time, state, field)?;
    sink.observe(PdeObservation::new(time, field, request_index)?)
}

/// Position in the requested-time list of one advance.
///
/// Each integration phase receives the requests up to its end in one slice; the
/// cursor then moves past them, so request indices stay global.
#[derive(Debug, Clone, Copy)]
pub(super) struct Requests<'a, S> {
    times: &'a [S],
    next: usize,
}

impl<'a, S: RealScalar> Requests<'a, S> {
    /// Requests at or after `from`; empty unless the schedule lists times.
    pub(super) fn new(schedule: &'a PdeOutputSchedule<S>, from: S) -> Self {
        let times = match schedule {
            PdeOutputSchedule::Times(times) => times.as_slice(),
            _ => &[],
        };
        Self {
            times,
            next: times.partition_point(|&t| t < from),
        }
    }

    /// Index of the next pending request.
    pub(super) const fn next(&self) -> usize {
        self.next
    }

    /// Pending requests with index, in order.
    pub(super) fn pending(&self) -> impl Iterator<Item = (usize, S)> + 'a {
        let start = self.next;
        self.times[start..]
            .iter()
            .enumerate()
            .map(move |(offset, &t)| (start + offset, t))
    }

    /// Pending requests not after `end`.
    pub(super) fn through(&self, end: S) -> &'a [S] {
        let pending = &self.times[self.next..];
        &pending[..pending.partition_point(|&t| t <= end)]
    }

    /// Mark every request not after `end` as served.
    pub(super) fn serve_through(&mut self, end: S) {
        self.next += self.through(end).len();
    }

    /// Mark the request at `index` as served.
    pub(super) fn serve(&mut self, index: usize) {
        self.next = index + 1;
    }

    /// Whether every request lies within `[0, end]`.
    pub(super) fn within(&self, end: S) -> bool {
        self.times.last().is_none_or(|&last| last <= end)
    }
}

/// The ODE schedule of one phase ending at `end`.
pub(super) fn phase_schedule<S: RealScalar>(
    schedule: &PdeOutputSchedule<S>,
    requests: &Requests<'_, S>,
    end: S,
) -> Result<OdeOutputSchedule<S>, SciOxideError> {
    Ok(match schedule {
        PdeOutputSchedule::FinalOnly => OdeOutputSchedule::FinalOnly,
        PdeOutputSchedule::TimeSteps => OdeOutputSchedule::AcceptedSteps,
        PdeOutputSchedule::Times(_) => {
            OdeOutputSchedule::times(scioxide_core::backend::try_copy_slice(
                requests.through(end),
                scioxide_core::quantity!("pde_requested_phase_times"),
            )?)
        }
    })
}

/// Reconstructs the ODE output of one phase into streamed PDE observations.
pub(super) struct PhaseSink<'s, 'b, 'f, 'k, 'e, S: RealScalar, K: ?Sized> {
    pub(super) system: &'s MethodOfLinesSystem<'b, S>,
    pub(super) field: &'f mut [S],
    pub(super) sink: &'k mut K,
    /// Last streamed step time, shared across phases so a phase start is not repeated.
    pub(super) last_emitted: &'e mut Option<S>,
    pub(super) base_index: usize,
    pub(super) steps: bool,
}

impl<S, K> scioxide_ode::OdeOutputSink<S> for PhaseSink<'_, '_, '_, '_, '_, S, K>
where
    S: RealScalar + NumericScalar<Real = S>,
    K: PdeOutputSink<S> + ?Sized,
{
    fn emit(&mut self, time: S, state: &[S]) -> Result<(), SciOxideError> {
        if self.steps && *self.last_emitted != Some(time) {
            emit(self.system, self.field, self.sink, time, state, None)?;
            *self.last_emitted = Some(time);
        }
        Ok(())
    }

    fn emit_requested(&mut self, index: usize, time: S, state: &[S]) -> Result<(), SciOxideError> {
        emit(
            self.system,
            self.field,
            self.sink,
            time,
            state,
            Some(self.base_index + index),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn requests_split_phases_with_global_indices_and_duplicates() {
        let schedule = PdeOutputSchedule::Times(vec![0.0, 0.01, 0.01, 0.03, 0.05]);
        let mut requests = Requests::new(&schedule, 0.005);
        assert_eq!(requests.next(), 1);
        assert_eq!(requests.through(0.01), &[0.01, 0.01]);
        let OdeOutputSchedule::Times { times: first, .. } =
            phase_schedule(&schedule, &requests, 0.02).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(first, vec![0.01, 0.01]);
        requests.serve_through(0.02);
        assert_eq!(requests.next(), 3);
        assert_eq!(
            requests.pending().collect::<Vec<_>>(),
            vec![(3, 0.03), (4, 0.05)]
        );
        requests.serve(3);
        assert_eq!(requests.through(1.0), &[0.05]);
        assert!(requests.within(0.05) && !requests.within(0.04));
        let steps = PdeOutputSchedule::<f64>::TimeSteps;
        assert!(Requests::new(&steps, 0.0).through(1.0).is_empty());
        assert_eq!(
            phase_schedule(&steps, &Requests::new(&steps, 0.0), 1.0).unwrap(),
            OdeOutputSchedule::AcceptedSteps
        );
    }

    #[test]
    fn requested_times_must_be_finite_nonnegative_and_ordered() {
        for times in [vec![0.1, 0.0], vec![-0.1], vec![f64::NAN]] {
            assert!(PdeOutputSchedule::Times(times).validate().is_err());
        }
        assert!(
            PdeOutputSchedule::Times(vec![0.0, 0.0, 0.2])
                .validate()
                .is_ok()
        );
    }
}
