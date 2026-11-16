//! Event search on the continuous extension of each accepted step.
//!
//! The search finds a root only if the event function is continuous and the
//! subdivision samples a sign change. Even derivative-assisted sampling cannot
//! rule out roots or stationary points between samples. Crossing directions follow
//! integration order, also backward in time. A sample that is exactly zero starts no
//! crossing, so the next nonzero sample sets the reference sign. The workspace
//! requirements size every buffer, so detection does not allocate.

use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Ordering;

use scioxide_core::backend::try_reserve;
use scioxide_core::{
    EvaluationRole, Interval, NumericScalar, RealScalar, ResidualTolerance, SciOxideError,
    SolveContext, checked_difference, checked_float_product,
};
use scioxide_linalg::Vector;
use scioxide_rootfind::{
    RootTermination, ScalarFunction, ScalarRootOptions, ScalarRootSolver, Toms748,
};

use crate::stepper::StepTrace;
use crate::{
    EventDirection, EventEvidence, EventFunction, EventOccurrence, EventSearchPolicy,
    OdeEventOptions,
};

/// One detected event; its state is evaluated from dense output when emitted.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EventRecord<S: RealScalar> {
    pub(crate) index: usize,
    pub(crate) time: S,
    pub(crate) occurrence: EventOccurrence,
    pub(crate) terminal: bool,
    pub(crate) evidence: EventEvidence<S>,
}

/// Reusable event values, records and interpolation buffers.
#[derive(Debug)]
pub(crate) struct EventStorage<S: RealScalar, T: NumericScalar> {
    values: Vec<S>,
    pending: Vec<S>,
    pub(crate) records: Vec<EventRecord<S>>,
    pub(crate) state: Vector<T>,
    slope: Vector<T>,
    pub(crate) terminal: Vector<T>,
    /// Attempted event value and total-derivative callbacks in this epoch.
    pub(crate) evaluations: usize,
}

impl<S: RealScalar, T: NumericScalar> Default for EventStorage<S, T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            pending: Vec::new(),
            records: Vec::new(),
            state: Vector::empty(),
            slope: Vector::empty(),
            terminal: Vector::empty(),
            evaluations: 0,
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> EventStorage<S, T> {
    pub(crate) fn fits(&self, events: usize, records: usize, dimension: usize) -> bool {
        let n = if events == 0 { 0 } else { dimension };
        self.values.capacity() >= events
            && self.pending.capacity() >= events
            && self.records.capacity() >= records
            && self.state.len() == n
            && self.slope.len() == n
            && self.terminal.len() == n
    }

    pub(crate) fn reserve(
        &mut self,
        events: usize,
        records: usize,
        dimension: usize,
    ) -> Result<(), SciOxideError> {
        let n = if events == 0 { 0 } else { dimension };
        for buffer in [&mut self.values, &mut self.pending] {
            buffer.clear();
            try_reserve(buffer, events, scioxide_core::quantity!("ode_event_values"))?;
        }
        self.records.clear();
        try_reserve(
            &mut self.records,
            records,
            scioxide_core::quantity!("ode_event_records"),
        )?;
        for vector in [&mut self.state, &mut self.slope, &mut self.terminal] {
            if vector.len() != n {
                *vector = Vector::zeros(n)?;
            }
        }
        Ok(())
    }

    /// Begin an epoch: evaluate every event at the initial point. An exact zero there
    /// is not an event; the first step classifies from the next nonzero sample.
    pub(crate) fn init(
        &mut self,
        events: &[&dyn EventFunction<S, T>],
        time: S,
        state: &[T],
        trace: &StepTrace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.values.clear();
        self.pending.clear();
        self.records.clear();
        self.evaluations = 0;
        for event in events {
            let value = evaluate(
                *event,
                time,
                state,
                None,
                &mut self.evaluations,
                trace,
                cx,
                false,
            )?;
            push(&mut self.values, value)?;
        }
        Ok(())
    }

    /// Commit the endpoint values of the accepted step.
    pub(crate) fn commit(&mut self) {
        core::mem::swap(&mut self.values, &mut self.pending);
        self.pending.clear();
    }

    /// Evaluate every event at the endpoint `(end, end_state)` of a step and clear the
    /// step's records. Returns whether [`Self::detect`] must search the step on the
    /// continuous extension: always for subdivided or derivative-assisted search,
    /// otherwise only when some event changes sign in its direction across the step.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn endpoint(
        &mut self,
        events: &[&dyn EventFunction<S, T>],
        options: OdeEventOptions<S>,
        end: S,
        end_state: &[T],
        trace: &StepTrace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<bool, SciOxideError> {
        if self.values.len() != events.len() {
            return Err(crate::safety::invalid(
                scioxide_core::argument!("events"),
                "event collection changed after initialization",
            ));
        }
        self.pending.clear();
        self.records.clear();
        for event in events {
            let value = evaluate(
                *event,
                end,
                end_state,
                None,
                &mut self.evaluations,
                trace,
                cx,
                false,
            )?;
            push(&mut self.pending, value)?;
        }
        Ok(options.search().subdivisions() > 1
            || matches!(options.search(), EventSearchPolicy::DerivativeAssisted(_))
            || events
                .iter()
                .zip(self.values.iter().zip(&self.pending))
                .any(|(event, (&old, &new))| {
                    classify_occurrence(old, new, event.direction()).is_some()
                }))
    }

    /// Detect crossings on `[start, end]` from the endpoint values of [`Self::endpoint`],
    /// recording this step's events in integration order (ties by index). Cells that
    /// make no representable progress, as in a step only a few units in the last place
    /// wide, are merged into the next one; the last cell always ends at `end`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn detect(
        &mut self,
        events: &[&dyn EventFunction<S, T>],
        options: OdeEventOptions<S>,
        start: S,
        end: S,
        dense: &crate::dense_output::DenseOutputStorage<S, T>,
        trace: &StepTrace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if self.values.len() != events.len() || self.pending.len() != events.len() {
            return Err(crate::safety::invalid(
                scioxide_core::argument!("events"),
                "event collection changed after initialization",
            ));
        }
        self.records.clear();
        let count = options.search().subdivisions();
        let denominator = S::try_from_usize_exact(count)?;
        let width = checked_difference(end, start)?;
        let assisted = matches!(options.search(), EventSearchPolicy::DerivativeAssisted(_));
        let Self {
            values,
            pending,
            records,
            state,
            slope,
            evaluations,
            ..
        } = self;
        for (index, event) in events.iter().enumerate() {
            let mut search = Search {
                event: *event,
                index,
                dense,
                options,
                state,
                slope,
                evaluations,
                trace,
                records,
            };
            let mut left = start;
            let mut g_left = values[index];
            let mut d_left = if assisted {
                search.value(left, true, cx, false)?
            } else {
                S::ZERO
            };
            for cell in 1..=count {
                let right = if cell == count {
                    end
                } else {
                    let fraction = S::try_from_usize_exact(cell)? / denominator;
                    let right = start + checked_float_product(width, fraction)?;
                    let interior = if end > start {
                        left < right && right < end
                    } else {
                        end < right && right < left
                    };
                    if !interior {
                        continue;
                    }
                    right
                };
                let g_right = if cell == count {
                    pending[index]
                } else {
                    search.value(right, false, cx, false)?
                };
                let d_right = if assisted {
                    search.value(right, true, cx, false)?
                } else {
                    S::ZERO
                };
                let stationary = assisted
                    && d_left != S::ZERO
                    && (d_right == S::ZERO || opposite(d_left, d_right));
                if stationary {
                    let (time, bracket, _) = search.localize(left, right, true, cx)?;
                    let value = search.value(time, false, cx, false)?;
                    // A stationary point can expose two crossings hidden by equal endpoint signs.
                    if value != S::ZERO && (opposite(g_left, value) || opposite(value, g_right)) {
                        search.crossing(left, time, g_left, value, cx)?;
                        search.crossing(time, right, value, g_right, cx)?;
                    } else if options
                        .contact_tolerance()
                        .is_satisfied(NumericScalar::abs(value), options.contact_scale())?
                    {
                        if event.direction() == EventDirection::Either {
                            search.record(
                                time,
                                EventOccurrence::Tangency,
                                EventEvidence::Contact {
                                    bracket,
                                    residual: NumericScalar::abs(value),
                                },
                            )?;
                        }
                    } else {
                        search.crossing(left, right, g_left, g_right, cx)?;
                    }
                } else {
                    search.crossing(left, right, g_left, g_right, cx)?;
                }
                left = right;
                g_left = g_right;
                d_left = d_right;
            }
        }
        let backward = end < start;
        self.records.sort_unstable_by(|a, b| {
            let order = a.time.partial_cmp(&b.time).unwrap_or(Ordering::Equal);
            (if backward { order.reverse() } else { order }).then_with(|| a.index.cmp(&b.index))
        });
        Ok(())
    }

    /// Truncate the records after the first terminal coincidence group and order that
    /// group by event index. Returns the terminal record.
    pub(crate) fn retain_terminal_group(
        &mut self,
        tolerance: scioxide_core::StepTolerance<S>,
    ) -> Result<Option<EventRecord<S>>, SciOxideError> {
        let Some(position) = self.records.iter().position(|record| record.terminal) else {
            return Ok(None);
        };
        let stop = self.records[position];
        let mut end = position + 1;
        while end < self.records.len()
            && tolerance
                .tolerance()
                .is_converged(stop.time, self.records[end].time)?
        {
            end += 1;
        }
        self.records.truncate(end);
        // Compare every member to the anchor, not to the preceding member.
        let mut begin = position;
        while begin > 0
            && tolerance
                .tolerance()
                .is_converged(stop.time, self.records[begin - 1].time)?
        {
            begin -= 1;
        }
        let backward = self
            .records
            .first()
            .zip(self.records.last())
            .is_some_and(|(first, last)| first.time > last.time);
        self.records[begin..].sort_unstable_by(|a, b| {
            a.index.cmp(&b.index).then_with(|| {
                if a.time == b.time {
                    Ordering::Equal
                } else if (a.time < b.time) != backward {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
        });
        Ok(Some(stop))
    }
}

/// Pushes within reserved capacity, which the workspace requirements guarantee.
fn push<V>(buffer: &mut Vec<V>, value: V) -> Result<(), SciOxideError> {
    if buffer.len() == buffer.capacity() {
        return Err(crate::safety::invalid(
            scioxide_core::argument!("workspace"),
            "event storage is smaller than the workspace requirements",
        ));
    }
    buffer.push(value);
    Ok(())
}

/// Evaluates one event value or total derivative, counting it against the budget unless
/// the caller already did.
#[allow(clippy::too_many_arguments)]
fn evaluate<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>>(
    event: &dyn EventFunction<S, T>,
    time: S,
    state: &[T],
    slope: Option<&[T]>,
    evaluations: &mut usize,
    trace: &StepTrace,
    cx: &mut SolveContext<'_, '_>,
    admitted: bool,
) -> Result<S, SciOxideError> {
    if !admitted {
        crate::stepper::admit_provider(cx, EvaluationRole::Event, trace)?;
    }
    *evaluations = scioxide_core::backend::checked_add(
        *evaluations,
        1,
        scioxide_core::quantity!("event_evaluations"),
    )?;
    let value = match slope {
        Some(slope) => event.total_derivative(time, state, slope),
        None => event.eval(time, state),
    }
    .inspect_err(|_| trace.provider_failed.set(true))?;
    crate::safety::finite(&[value], scioxide_core::NonFiniteOrigin::Callback)?;
    Ok(value)
}

struct Search<'a, S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    event: &'a dyn EventFunction<S, T>,
    index: usize,
    dense: &'a crate::dense_output::DenseOutputStorage<S, T>,
    options: OdeEventOptions<S>,
    state: &'a mut Vector<T>,
    slope: &'a mut Vector<T>,
    evaluations: &'a mut usize,
    trace: &'a StepTrace,
    records: &'a mut Vec<EventRecord<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Search<'_, S, T> {
    fn value(
        &mut self,
        time: S,
        derivative: bool,
        cx: &mut SolveContext<'_, '_>,
        admitted: bool,
    ) -> Result<S, SciOxideError> {
        self.dense.eval_into(time, self.state.as_mut_slice())?;
        crate::safety::finite(
            self.state.as_slice(),
            scioxide_core::NonFiniteOrigin::Callback,
        )?;
        let slope = if derivative {
            self.dense
                .derivative_into(time, self.slope.as_mut_slice())?;
            crate::safety::finite(
                self.slope.as_slice(),
                scioxide_core::NonFiniteOrigin::Callback,
            )?;
            Some(self.slope.as_slice())
        } else {
            None
        };
        evaluate(
            self.event,
            time,
            self.state.as_slice(),
            slope,
            self.evaluations,
            self.trace,
            cx,
            admitted,
        )
    }

    fn localize(
        &mut self,
        left: S,
        right: S,
        derivative: bool,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(S, Interval<S>, S), SciOxideError> {
        let interval = if left < right {
            Interval::new(left, right)?
        } else {
            Interval::new(right, left)?
        };
        let options = ScalarRootOptions::default()
            .with_evaluation_role(EvaluationRole::Event)
            .with_step_tolerance(self.options.localization())
            .with_residual_tolerance(ResidualTolerance::new(S::ZERO, S::ZERO)?)
            .with_max_iterations(self.options.max_iterations());
        let result = Toms748.solve_scoped(
            &Localized {
                search: RefCell::new(self),
                derivative,
            },
            interval,
            &options,
            &mut *cx,
        )?;
        let bracket = result.bracket().map(|bracket| bracket.interval());
        // No representable time lies strictly inside a bracket of adjacent floating-point
        // times, so such a bracket is converged whatever the time tolerance.
        let resolved = matches!(
            result.termination(),
            RootTermination::Stagnation | RootTermination::IterationLimit
        ) && bracket
            .is_some_and(|bracket| bracket.lower().next_after(bracket.upper()) == bracket.upper());
        if !result.termination().is_success() && !resolved {
            use scioxide_core::{BudgetKind, ConvergenceReason, ExecutionStopReason};
            let reason = match result.termination() {
                RootTermination::Cancelled => {
                    return Err(SciOxideError::execution_stopped(
                        ExecutionStopReason::Cancelled,
                    ));
                }
                RootTermination::TimeLimit => {
                    return Err(SciOxideError::execution_stopped(
                        ExecutionStopReason::DeadlineReached,
                    ));
                }
                RootTermination::WorkLimit(BudgetKind::Evaluation) => {
                    ConvergenceReason::EvaluationLimit
                }
                RootTermination::IterationLimit
                | RootTermination::WorkLimit(BudgetKind::Iteration) => {
                    ConvergenceReason::IterationLimit
                }
                RootTermination::WorkLimit(kind) => ConvergenceReason::ResourceLimit(kind),
                _ => ConvergenceReason::NestedSolver,
            };
            let mut diagnostics =
                scioxide_core::ConvergenceDiagnostics::new().with_counts(result.accounting());
            if let RootTermination::WorkLimit(kind) = result.termination() {
                diagnostics = diagnostics.with_budget(cx.effective_budget_report(kind));
            }
            return Err(SciOxideError::convergence_failure(
                reason,
                diagnostics,
                "bracket localization did not satisfy the time tolerance or find an exact zero",
            ));
        }
        let bracket = bracket.ok_or_else(|| {
            crate::safety::invalid(
                scioxide_core::argument!("events"),
                "bracket solver returned no localization interval",
            )
        })?;
        Ok((
            result.point(),
            bracket,
            NumericScalar::abs(result.residual()),
        ))
    }

    fn crossing(
        &mut self,
        left: S,
        right: S,
        g_left: S,
        g_right: S,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let Some(occurrence) = classify_occurrence(g_left, g_right, self.event.direction()) else {
            return Ok(());
        };
        let (time, evidence) = if g_right == S::ZERO {
            (right, EventEvidence::SampledZero)
        } else {
            let (time, bracket, residual) = self.localize(left, right, false, cx)?;
            (time, EventEvidence::Crossing { bracket, residual })
        };
        self.record(time, occurrence, evidence)
    }

    fn record(
        &mut self,
        time: S,
        occurrence: EventOccurrence,
        evidence: EventEvidence<S>,
    ) -> Result<(), SciOxideError> {
        if self
            .records
            .last()
            .is_some_and(|last| last.index == self.index && last.time == time)
        {
            return Ok(());
        }
        push(
            self.records,
            EventRecord {
                index: self.index,
                time,
                occurrence,
                terminal: self.event.is_terminal(),
                evidence,
            },
        )
    }
}

fn opposite<S: RealScalar>(a: S, b: S) -> bool {
    (a < S::ZERO && b > S::ZERO) || (a > S::ZERO && b < S::ZERO)
}

fn classify_occurrence<S: RealScalar>(
    old: S,
    new: S,
    filter: EventDirection,
) -> Option<EventOccurrence> {
    let occurrence = if old < S::ZERO && new >= S::ZERO {
        EventOccurrence::Rising
    } else if old > S::ZERO && new <= S::ZERO {
        EventOccurrence::Falling
    } else {
        return None;
    };
    match (filter, occurrence) {
        (EventDirection::Either, _)
        | (EventDirection::Rising, EventOccurrence::Rising)
        | (EventDirection::Falling, EventOccurrence::Falling) => Some(occurrence),
        _ => None,
    }
}

/// Root function for TOMS 748; each evaluation counts as event work in its child scope.
struct Localized<'a, 'b, S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    search: RefCell<&'a mut Search<'b, S, T>>,
    derivative: bool,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> ScalarFunction<S>
    for Localized<'_, '_, S, T>
{
    fn eval(&self, _: S) -> Result<S, SciOxideError> {
        Err(crate::safety::invalid(
            scioxide_core::argument!("events"),
            "event localization requires its solve scope",
        ))
    }
    fn eval_scoped(&self, time: S, cx: &mut SolveContext<'_, '_>) -> Result<S, SciOxideError> {
        self.search
            .borrow_mut()
            .value(time, self.derivative, cx, true)
    }
}
