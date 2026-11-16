//! Survival curves, the log-rank test and Cox regression for right-censored data
//! with optional delayed entry.

use crate::error_helpers::{invalid, number};
// Observations have finite times; both signs of zero belong to the same tie.
use crate::ranking::finite_cmp;
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{Interval, RealScalar, SciOxideError};
use scioxide_random::{ChiSquared, Survival};

use crate::ConfidenceProbability;

mod cox;
pub use cox::{
    CoxBaselinePoint, CoxFit, CoxInferenceOptions, CoxReport, CoxTies, PreparedCox,
    SchoenfeldResidual, cox_fit, cox_fit_scoped, cox_fit_with_options,
};

/// One subject of a right-censored sample, with optional delayed entry (left
/// truncation).
///
/// The subject is observed on `(entry, time]` and either has the event at
/// `time` or is censored there. An entry of `None` means the subject is at risk
/// from before the first event time. The subject belongs to the risk set at
/// time `t` when `entry < t ≤ time`. So events and censorings at `t` both count
/// as at risk just before `t`. Subjects are assumed independent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurvivalObservation<S: RealScalar> {
    entry: Option<S>,
    time: S,
    event: bool,
}

impl<S: RealScalar> SurvivalObservation<S> {
    /// Creates an observation that ends with `outcome` at `time`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `time` or `entry` is NaN or infinite.
    /// - `InvalidInput` if `entry` is not less than `time`.
    pub fn new(
        entry: Option<S>,
        time: S,
        outcome: crate::SurvivalOutcome,
    ) -> Result<Self, SciOxideError> {
        if !time.is_finite() || entry.is_some_and(|x| !x.is_finite()) {
            return Err(invalid(
                scioxide_core::argument!("time"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "observation times must be finite",
            ));
        }
        if entry.is_some_and(|x| x >= time) {
            return Err(invalid(
                scioxide_core::argument!("entry"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "entry must precede exit time",
            ));
        }
        Ok(Self {
            entry,
            time,
            event: outcome == crate::SurvivalOutcome::Event,
        })
    }

    /// Returns the entry time, or `None` if the subject is at risk from the start.
    pub const fn entry(self) -> Option<S> {
        self.entry
    }

    /// Returns the event or censoring time.
    pub const fn time(self) -> S {
        self.time
    }

    /// Returns whether the event was observed (`false` means censored).
    pub const fn event(self) -> bool {
        self.event
    }

    fn at_risk(self, time: S) -> bool {
        self.entry.is_none_or(|entry| entry < time) && self.time >= time
    }
}

fn event_times<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
    max_bytes: usize,
) -> Result<Vec<S>, SciOxideError> {
    if observations.is_empty() {
        return Err(invalid(
            scioxide_core::argument!("observations"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "at least one observation is required",
        ));
    }
    let count = observations.iter().filter(|row| row.event).count();
    let bytes = count
        .checked_mul(core::mem::size_of::<S>())
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("survival_event_times"))
        })?;
    crate::error_helpers::limit(bytes, max_bytes, scioxide_core::quantity!("survival_bytes"))?;
    let mut times = try_with_capacity(count, scioxide_core::quantity!("survival_event_times"))?;
    times.extend(
        observations
            .iter()
            .filter(|row| row.event)
            .map(|row| row.time),
    );
    times.sort_unstable_by(finite_cmp);
    times.dedup();
    Ok(times)
}

#[derive(Clone, Copy)]
struct RiskRow<S: RealScalar> {
    time: S,
    at_risk: usize,
    events: usize,
    censored: usize,
}

fn payload_bytes<T>(count: usize) -> Result<usize, SciOxideError> {
    checked_mul(
        count,
        core::mem::size_of::<T>(),
        scioxide_core::quantity!("survival_payload"),
    )
}

fn admit_payload(bytes: usize, max_bytes: usize) -> Result<(), SciOxideError> {
    crate::error_helpers::limit(bytes, max_bytes, scioxide_core::quantity!("survival_bytes"))?;
    Ok(())
}

fn combined_bytes(first: usize, second: usize) -> Result<usize, SciOxideError> {
    checked_add(first, second, scioxide_core::quantity!("survival_payload"))
}

fn bounded_vec<T>(
    count: usize,
    live_bytes: usize,
    max_bytes: usize,
) -> Result<Vec<T>, SciOxideError> {
    admit_payload(
        combined_bytes(live_bytes, payload_bytes::<T>(count)?)?,
        max_bytes,
    )?;
    let values = try_with_capacity(count, scioxide_core::quantity!("survival_payload"))?;
    // Reservation may round capacity upward; charge retained capacity too.
    admit_payload(
        combined_bytes(live_bytes, payload_bytes::<T>(values.capacity())?)?,
        max_bytes,
    )?;
    Ok(values)
}

/// Risk-set counts of one group, swept forward through increasing times.
struct RiskSweep<'a, S: RealScalar> {
    observations: &'a [SurvivalObservation<S>],
    exits: Vec<usize>,
    entries: Vec<S>,
    entered: usize,
    exited: usize,
}

impl<'a, S: RealScalar> RiskSweep<'a, S> {
    fn required_bytes(observations: &[SurvivalObservation<S>]) -> Result<usize, SciOxideError> {
        combined_bytes(
            payload_bytes::<usize>(observations.len())?,
            payload_bytes::<S>(
                observations
                    .iter()
                    .filter(|row| row.entry.is_some())
                    .count(),
            )?,
        )
    }

    fn new(
        observations: &'a [SurvivalObservation<S>],
        max_bytes: usize,
    ) -> Result<Self, SciOxideError> {
        admit_payload(Self::required_bytes(observations)?, max_bytes)?;
        let mut exits = bounded_vec(observations.len(), 0, max_bytes)?;
        exits.extend(0..observations.len());
        exits.sort_unstable_by(|&a, &b| finite_cmp(&observations[a].time, &observations[b].time));
        let delayed = observations
            .iter()
            .filter(|row| row.entry.is_some())
            .count();
        let mut entries = bounded_vec(
            delayed,
            payload_bytes::<usize>(exits.capacity())?,
            max_bytes,
        )?;
        entries.extend(observations.iter().filter_map(|row| row.entry));
        entries.sort_unstable_by(finite_cmp);
        Ok(Self {
            observations,
            exits,
            entries,
            entered: 0,
            exited: 0,
        })
    }

    fn bytes(&self) -> Result<usize, SciOxideError> {
        combined_bytes(
            payload_bytes::<usize>(self.exits.capacity())?,
            payload_bytes::<S>(self.entries.capacity())?,
        )
    }

    fn next_time(&self) -> Option<S> {
        self.exits
            .get(self.exited)
            .map(|&index| self.observations[index].time)
    }

    /// Returns the risk-set row at `time`. Call it at increasing times from the
    /// union of both groups' exit times.
    fn take_at(&mut self, time: S) -> RiskRow<S> {
        while self
            .entries
            .get(self.entered)
            .is_some_and(|&entry| entry < time)
        {
            self.entered += 1;
        }
        let at_risk = self.observations.len() - self.entries.len() + self.entered - self.exited;
        let mut events = 0usize;
        let mut censored = 0usize;
        while let Some(&index) = self.exits.get(self.exited) {
            let observation = self.observations[index];
            if observation.time != time {
                break;
            }
            events += usize::from(observation.event);
            censored += usize::from(!observation.event);
            self.exited += 1;
        }
        RiskRow {
            time,
            at_risk,
            events,
            censored,
        }
    }
}

/// One step of a Kaplan–Meier curve, at a distinct event time.
///
/// With `nᵢ` subjects at risk and `dᵢ` events at the event times `tᵢ ≤ t`, the
/// product-limit estimate and the Greenwood sum are
///
/// ```text
/// Ŝ(t) = ∏ (1 − dᵢ/nᵢ)
/// σ²(t) = Σ dᵢ / (nᵢ (nᵢ − dᵢ))
/// ```
///
/// `σ²` estimates the variance of `ln Ŝ(t)`. Terms with `dᵢ = nᵢ` are left out
/// of the sum.
#[derive(Clone, Copy, Debug)]
pub struct KaplanMeierPoint<S: RealScalar> {
    /// Distinct event time.
    pub time: S,
    /// Number of subjects at risk just before this time.
    pub at_risk: usize,
    /// Number of events at this time.
    pub events: usize,
    /// Number of censorings at this time.
    pub censored: usize,
    /// Product-limit estimate `Ŝ(t)` just after the events at this time.
    pub survival: S,
    /// Pointwise log-log confidence interval `[Ŝ^exp(r), Ŝ^exp(−r)]`, with
    /// `r = z σ / |ln Ŝ|` and `z` the normal critical value. `None` when
    /// `Ŝ = 0`, where the log-log transform is undefined.
    pub interval: Option<Interval<S>>,
}

/// One step of a Nelson–Aalen cumulative-hazard curve, at a distinct event time.
///
/// With `nᵢ` subjects at risk and `dᵢ` events at the event times `tᵢ ≤ t`,
///
/// ```text
/// Ĥ(t) = Σ dᵢ/nᵢ
/// Var Ĥ(t) = Σ dᵢ/nᵢ²
/// ```
#[derive(Clone, Copy, Debug)]
pub struct NelsonAalenPoint<S: RealScalar> {
    /// Distinct event time.
    pub time: S,
    /// Number of subjects at risk just before this time.
    pub at_risk: usize,
    /// Number of events at this time.
    pub events: usize,
    /// Number of censorings at this time.
    pub censored: usize,
    /// Cumulative hazard `Ĥ(t)`.
    pub cumulative_hazard: S,
    /// Variance estimate `Σ dᵢ/nᵢ²`.
    pub variance: S,
    /// Pointwise log-transformed confidence interval `Ĥ exp(∓z √Var Ĥ / Ĥ)`,
    /// with `z` the normal critical value.
    pub interval: Interval<S>,
}

/// Memory limit for survival curves, the log-rank test and Cox preparation.
///
/// The limit counts the bytes of the sorted risk sets and curve points that
/// the computation stores. It does not count buffers you provide or allocator
/// overhead.
#[derive(Clone, Copy, Debug)]
pub struct SurvivalOptions {
    max_bytes: usize,
}
impl Default for SurvivalOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl SurvivalOptions {
    /// Returns options without a memory limit.
    pub const fn new() -> Self {
        Self {
            max_bytes: usize::MAX,
        }
    }
    /// Sets the memory limit in bytes. Exceeding it is a `ResourceLimit` error.
    pub const fn with_max_bytes(mut self, bytes: usize) -> Self {
        self.max_bytes = bytes;
        self
    }
    /// Returns the memory limit in bytes.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
}
/// Survival curve that a [`SurvivalWorkspace`] is sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurvivalEstimator {
    /// Kaplan–Meier product-limit survival with Greenwood intervals.
    KaplanMeier,
    /// Nelson–Aalen cumulative hazard and its variance.
    NelsonAalen,
}
/// Workspace size needed for one survival curve, from
/// [`SurvivalWorkspace::workspace_requirements`].
#[derive(Clone, Copy, Debug)]
pub struct SurvivalRequirements {
    exits: usize,
    entries: usize,
    points: usize,
    estimator: SurvivalEstimator,
    bytes: usize,
    max_bytes: usize,
}
impl SurvivalRequirements {
    /// Returns the number of bytes the workspace needs, without allocator
    /// overhead.
    pub const fn required_bytes(self) -> usize {
        self.bytes
    }
}
/// Reusable workspace for [`kaplan_meier_into`] and [`nelson_aalen_into`].
///
/// It holds the observations sorted by time and the curve points before they
/// are copied to the output. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct SurvivalWorkspace<S: RealScalar> {
    exits: Vec<usize>,
    entries: Vec<S>,
    km: Vec<KaplanMeierPoint<S>>,
    na: Vec<NelsonAalenPoint<S>>,
}
impl<S: RealScalar> Default for SurvivalWorkspace<S> {
    fn default() -> Self {
        Self {
            exits: Vec::new(),
            entries: Vec::new(),
            km: Vec::new(),
            na: Vec::new(),
        }
    }
}
impl<S: RealScalar> SurvivalWorkspace<S> {
    /// Returns the workspace size for one curve of `observations`.
    ///
    /// It reserves one curve point per event, which bounds the number of
    /// distinct event times. A sample without events needs no workspace.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `observations` is empty.
    /// - `ResourceLimit` if the size exceeds `options.max_bytes()`.
    pub fn workspace_requirements(
        observations: &[SurvivalObservation<S>],
        estimator: SurvivalEstimator,
        options: SurvivalOptions,
    ) -> Result<SurvivalRequirements, SciOxideError> {
        if observations.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("observations"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "at least one observation is required",
            ));
        }
        let points = observations.iter().filter(|row| row.event).count();
        let exits = if points == 0 { 0 } else { observations.len() };
        let entries = if points == 0 {
            0
        } else {
            observations
                .iter()
                .filter(|row| row.entry.is_some())
                .count()
        };
        let point_bytes = match estimator {
            SurvivalEstimator::KaplanMeier => payload_bytes::<KaplanMeierPoint<S>>(points)?,
            SurvivalEstimator::NelsonAalen => payload_bytes::<NelsonAalenPoint<S>>(points)?,
        };
        let bytes = combined_bytes(
            combined_bytes(payload_bytes::<usize>(exits)?, payload_bytes::<S>(entries)?)?,
            point_bytes,
        )?;
        crate::error_helpers::limit_bytes(bytes, options.max_bytes)?;
        Ok(SurvivalRequirements {
            exits,
            entries,
            points,
            estimator,
            bytes,
            max_bytes: options.max_bytes,
        })
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        combined_bytes(
            combined_bytes(
                payload_bytes::<usize>(self.exits.capacity())?,
                payload_bytes::<S>(self.entries.capacity())?,
            )?,
            combined_bytes(
                payload_bytes::<KaplanMeierPoint<S>>(self.km.capacity())?,
                payload_bytes::<NelsonAalenPoint<S>>(self.na.capacity())?,
            )?,
        )
    }
    /// Returns whether the workspace is large enough for `req`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage the workspace holds, including storage
    ///   for the other estimator, exceeds the memory limit of `req`.
    pub fn check(
        &self,
        req: &SurvivalRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        crate::error_helpers::limit_bytes(self.retained_bytes()?, req.max_bytes)?;
        let capacity = match req.estimator {
            SurvivalEstimator::KaplanMeier => self.km.capacity(),
            SurvivalEstimator::NelsonAalen => self.na.capacity(),
        };
        Ok(
            if self.exits.capacity() >= req.exits
                && self.entries.capacity() >= req.entries
                && capacity >= req.points
            {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to fit `req`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the memory limit of
    ///   `req`.
    /// - `AllocationFailure` if memory cannot be reserved. Storage already grown
    ///   is kept.
    pub fn try_reserve(&mut self, req: &SurvivalRequirements) -> Result<(), SciOxideError> {
        if self.check(req)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        let exit_capacity = self.exits.capacity().max(req.exits);
        let entry_capacity = self.entries.capacity().max(req.entries);
        let km_capacity =
            self.km
                .capacity()
                .max(if req.estimator == SurvivalEstimator::KaplanMeier {
                    req.points
                } else {
                    0
                });
        let na_capacity =
            self.na
                .capacity()
                .max(if req.estimator == SurvivalEstimator::NelsonAalen {
                    req.points
                } else {
                    0
                });
        let bytes = combined_bytes(
            combined_bytes(
                payload_bytes::<usize>(exit_capacity)?,
                payload_bytes::<S>(entry_capacity)?,
            )?,
            combined_bytes(
                payload_bytes::<KaplanMeierPoint<S>>(km_capacity)?,
                payload_bytes::<NelsonAalenPoint<S>>(na_capacity)?,
            )?,
        )?;
        crate::error_helpers::limit_bytes(bytes, req.max_bytes)?;
        reserve_curve(&mut self.exits, req.exits)?;
        reserve_curve(&mut self.entries, req.entries)?;
        match req.estimator {
            SurvivalEstimator::KaplanMeier => reserve_curve(&mut self.km, req.points)?,
            SurvivalEstimator::NelsonAalen => reserve_curve(&mut self.na, req.points)?,
        }
        if self.check(req)? != scioxide_core::Capacity::Sufficient {
            return Err(SciOxideError::allocation_failure(
                scioxide_core::quantity!("survival_workspace"),
                req.bytes,
            ));
        }
        Ok(())
    }
    /// Creates a workspace that fits `req`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::try_reserve`].
    pub fn workspace(req: &SurvivalRequirements) -> Result<Self, SciOxideError> {
        let mut ws = Self::default();
        ws.try_reserve(req)?;
        Ok(ws)
    }
    /// Returns a `ResourceLimit` error for the first buffer that is too small,
    /// counted in elements.
    fn shortfall(&self, req: &SurvivalRequirements) -> SciOxideError {
        let (quantity, required, available) = if self.exits.capacity() < req.exits {
            (
                scioxide_core::quantity!("survival_workspace_exits"),
                req.exits,
                self.exits.capacity(),
            )
        } else if self.entries.capacity() < req.entries {
            (
                scioxide_core::quantity!("survival_workspace_entries"),
                req.entries,
                self.entries.capacity(),
            )
        } else {
            let available = match req.estimator {
                SurvivalEstimator::KaplanMeier => self.km.capacity(),
                SurvivalEstimator::NelsonAalen => self.na.capacity(),
            };
            (
                scioxide_core::quantity!("survival_workspace_points"),
                req.points,
                available,
            )
        };
        SciOxideError::resource_limit(quantity, required, available)
    }
    fn prepare(
        &mut self,
        observations: &[SurvivalObservation<S>],
        req: &SurvivalRequirements,
    ) -> Result<(), SciOxideError> {
        if self.check(req)? != scioxide_core::Capacity::Sufficient {
            return Err(self.shortfall(req));
        }
        self.exits.clear();
        self.entries.clear();
        self.km.clear();
        self.na.clear();
        if req.points == 0 {
            return Ok(());
        }
        self.exits.extend(0..observations.len());
        self.exits
            .sort_unstable_by(|&a, &b| finite_cmp(&observations[a].time, &observations[b].time));
        self.entries
            .extend(observations.iter().filter_map(|row| row.entry));
        self.entries.sort_unstable_by(finite_cmp);
        Ok(())
    }
}
fn reserve_curve<T>(values: &mut Vec<T>, count: usize) -> Result<(), SciOxideError> {
    if values.capacity() < count {
        values
            .try_reserve_exact(count.saturating_sub(values.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("survival_workspace"),
                    count,
                )
            })?;
    }
    Ok(())
}
fn walk_curve<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
    exits: &[usize],
    entries: &[S],
    context: scioxide_core::ExecutionContext<'_>,
    mut project: impl FnMut(RiskRow<S>, f64, f64, f64, f64) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let mut entered = 0;
    let mut exited = 0;
    let mut survival = 1.0;
    let mut greenwood = 0.0;
    let mut hazard = 0.0;
    let mut variance = 0.0;
    while exited < exits.len() {
        context.check()?;
        let time = observations[exits[exited]].time;
        while entries.get(entered).is_some_and(|&entry| entry < time) {
            entered += 1;
        }
        let at_risk = observations.len() - entries.len() + entered - exited;
        let mut events = 0;
        let mut censored = 0;
        while let Some(&index) = exits.get(exited) {
            let observation = observations[index];
            if observation.time != time {
                break;
            }
            events += usize::from(observation.event);
            censored += usize::from(!observation.event);
            exited += 1;
        }
        if events == 0 {
            continue;
        }
        let n = at_risk as f64;
        let d = events as f64;
        survival *= 1.0 - d / n;
        if events < at_risk {
            greenwood += d / (n * (n - d));
        }
        hazard += d / n;
        variance += d / (n * n);
        project(
            RiskRow {
                time,
                at_risk,
                events,
                censored,
            },
            survival,
            greenwood,
            hazard,
            variance,
        )?;
    }
    Ok(())
}
fn fill_km<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
    confidence: ConfidenceProbability,
    options: SurvivalOptions,
    ws: &mut SurvivalWorkspace<S>,
    context: scioxide_core::ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    context.check()?;
    let req = SurvivalWorkspace::workspace_requirements(
        observations,
        SurvivalEstimator::KaplanMeier,
        options,
    )?;
    ws.prepare(observations, &req)?;
    let z = confidence.normal_critical_value::<f64>()?;
    walk_curve(
        observations,
        &ws.exits,
        &ws.entries,
        context,
        |row, survival, greenwood, _, _| {
            let interval = if survival == 0.0 {
                None
            } else if survival == 1.0 {
                Some(Interval::new(S::ONE, S::ONE)?)
            } else {
                let center = (-survival.ln()).ln();
                let radius = z * greenwood.sqrt() / survival.ln().abs();
                Some(Interval::new(
                    number((-((center + radius).exp())).exp())?,
                    number((-((center - radius).exp())).exp())?,
                )?)
            };
            ws.km.push(KaplanMeierPoint {
                time: row.time,
                at_risk: row.at_risk,
                events: row.events,
                censored: row.censored,
                survival: number(survival)?,
                interval,
            });
            Ok(())
        },
    )
}
fn fill_na<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
    confidence: ConfidenceProbability,
    options: SurvivalOptions,
    ws: &mut SurvivalWorkspace<S>,
    context: scioxide_core::ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    context.check()?;
    let req = SurvivalWorkspace::workspace_requirements(
        observations,
        SurvivalEstimator::NelsonAalen,
        options,
    )?;
    ws.prepare(observations, &req)?;
    let z = confidence.normal_critical_value::<f64>()?;
    walk_curve(
        observations,
        &ws.exits,
        &ws.entries,
        context,
        |row, _, _, hazard, variance| {
            let radius = z * variance.sqrt() / hazard;
            let interval = Interval::new(
                number(hazard * (-radius).exp())?,
                number(hazard * radius.exp())?,
            )?;
            ws.na.push(NelsonAalenPoint {
                time: row.time,
                at_risk: row.at_risk,
                events: row.events,
                censored: row.censored,
                cumulative_hazard: number(hazard)?,
                variance: number(variance)?,
                interval,
            });
            Ok(())
        },
    )
}
/// Computes the result of [`kaplan_meier_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`kaplan_meier_with_options`] for the fixed default settings.
pub fn kaplan_meier<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
) -> Result<Vec<KaplanMeierPoint<S>>, SciOxideError> {
    kaplan_meier_with_options(
        observations,
        ConfidenceProbability::default(),
        Default::default(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options", "confidence"],
            scioxide_core::argument!("observations"),
        )
    })
}
/// Computes the Kaplan–Meier product-limit estimate of the survival function.
///
/// Returns one [`KaplanMeierPoint`] per distinct event time, in increasing
/// order, with a pointwise log-log confidence interval of two-sided coverage
/// `confidence`. A subject is at risk at `t` when `entry < t ≤ time`. A sample
/// without events gives an empty curve. The estimates are computed in `f64`.
///
/// Sorting the observations costs O(n log n) time and the sweep O(n), for `n`
/// observations. [`kaplan_meier_into`] writes into a buffer you provide.
///
/// # Errors
///
/// - `InvalidInput` if `observations` is empty.
/// - `ResourceLimit` if the workspace exceeds `options.max_bytes()`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::SurvivalOutcome::Event;
/// use scioxide_stats::{ConfidenceProbability, SurvivalObservation, SurvivalOptions, kaplan_meier_with_options};
///
/// // One subject with an event at t = 1: the estimate drops from 1 to 0.
/// let rows = [SurvivalObservation::new(None, 1.0_f64, Event)?];
/// let confidence = ConfidenceProbability::default();
/// let curve = kaplan_meier_with_options(&rows, confidence, SurvivalOptions::new())?;
/// assert_eq!(curve[0].survival, 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// E. L. Kaplan and P. Meier, "Nonparametric estimation from incomplete
/// observations", *J. Amer. Statist. Assoc.* 53 (1958), 457–481.
///
/// J. P. Klein and M. L. Moeschberger, *Survival Analysis: Techniques for
/// Censored and Truncated Data*, 2nd ed., Springer, 2003, ch. 4.
pub fn kaplan_meier_with_options<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
    confidence: ConfidenceProbability,
    options: SurvivalOptions,
) -> Result<Vec<KaplanMeierPoint<S>>, SciOxideError> {
    let req = SurvivalWorkspace::workspace_requirements(
        observations,
        SurvivalEstimator::KaplanMeier,
        options,
    )?;
    let mut ws = SurvivalWorkspace::workspace(&req)?;
    fill_km(
        observations,
        confidence,
        options,
        &mut ws,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Ok(ws.km)
}
/// Computes the Kaplan–Meier curve of [`kaplan_meier_with_options`] into `output` and returns
/// the number of points written.
///
/// The points go to the start of `output`; the rest is not modified. `output`
/// needs one element per distinct event time; one per event always suffices.
///
/// # Errors
///
/// - `InvalidInput` if `observations` is empty.
/// - `ResourceLimit` if `ws` is smaller than
///   [`SurvivalWorkspace::workspace_requirements`] reports, or holds more than
///   `options.max_bytes()`.
/// - `DimensionMismatch` if `output` is shorter than the curve.
/// - `ExecutionStopped` if `context` is cancelled or its deadline passes.
pub fn kaplan_meier_into<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
    confidence: ConfidenceProbability,
    options: SurvivalOptions,
    output: &mut [KaplanMeierPoint<S>],
    ws: &mut SurvivalWorkspace<S>,
    context: scioxide_core::ExecutionContext<'_>,
) -> Result<usize, SciOxideError> {
    fill_km(observations, confidence, options, ws, context)?;
    let count = ws.km.len();
    if output.len() < count {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(count),
            scioxide_core::Dimensions::one(output.len()),
        ));
    }
    context.check()?;
    output[..count].copy_from_slice(&ws.km);
    Ok(count)
}
/// Computes the result of [`nelson_aalen_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`nelson_aalen_with_options`] for the fixed default settings.
pub fn nelson_aalen<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
) -> Result<Vec<NelsonAalenPoint<S>>, SciOxideError> {
    nelson_aalen_with_options(
        observations,
        ConfidenceProbability::default(),
        Default::default(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options", "confidence"],
            scioxide_core::argument!("observations"),
        )
    })
}
/// Computes the Nelson–Aalen estimate of the cumulative hazard function.
///
/// Returns one [`NelsonAalenPoint`] per distinct event time, in increasing
/// order, with a pointwise log-transformed confidence interval of two-sided
/// coverage `confidence`. Risk sets, the empty curve of a sample without events
/// and the cost are as in [`kaplan_meier_with_options`].
///
/// # Errors
///
/// - `InvalidInput` if `observations` is empty.
/// - `ResourceLimit` if the workspace exceeds `options.max_bytes()`.
///
/// # References
///
/// O. O. Aalen, "Nonparametric inference for a family of counting processes",
/// *Ann. Statist.* 6 (1978), 701–726.
///
/// J. P. Klein and M. L. Moeschberger, *Survival Analysis: Techniques for
/// Censored and Truncated Data*, 2nd ed., Springer, 2003, ch. 4.
pub fn nelson_aalen_with_options<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
    confidence: ConfidenceProbability,
    options: SurvivalOptions,
) -> Result<Vec<NelsonAalenPoint<S>>, SciOxideError> {
    let req = SurvivalWorkspace::workspace_requirements(
        observations,
        SurvivalEstimator::NelsonAalen,
        options,
    )?;
    let mut ws = SurvivalWorkspace::workspace(&req)?;
    fill_na(
        observations,
        confidence,
        options,
        &mut ws,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Ok(ws.na)
}
/// Computes the Nelson–Aalen curve of [`nelson_aalen_with_options`] into `output` and returns
/// the number of points written.
///
/// The points go to the start of `output`; the rest is not modified. `output`
/// needs one element per distinct event time; one per event always suffices.
///
/// # Errors
///
/// - `InvalidInput` if `observations` is empty.
/// - `ResourceLimit` if `ws` is smaller than
///   [`SurvivalWorkspace::workspace_requirements`] reports, or holds more than
///   `options.max_bytes()`.
/// - `DimensionMismatch` if `output` is shorter than the curve.
/// - `ExecutionStopped` if `context` is cancelled or its deadline passes.
pub fn nelson_aalen_into<S: RealScalar>(
    observations: &[SurvivalObservation<S>],
    confidence: ConfidenceProbability,
    options: SurvivalOptions,
    output: &mut [NelsonAalenPoint<S>],
    ws: &mut SurvivalWorkspace<S>,
    context: scioxide_core::ExecutionContext<'_>,
) -> Result<usize, SciOxideError> {
    fill_na(observations, confidence, options, ws, context)?;
    let count = ws.na.len();
    if output.len() < count {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(count),
            scioxide_core::Dimensions::one(output.len()),
        ));
    }
    context.check()?;
    output[..count].copy_from_slice(&ws.na);
    Ok(count)
}

/// Holds the result of the two-sample log-rank test from [`logrank_test_with_options`].
#[derive(Clone, Copy, Debug)]
pub struct LogrankResult<S: RealScalar> {
    /// Test statistic `(O₁ − E₁)² / V`, asymptotically χ² with one degree of
    /// freedom under the null hypothesis.
    statistic: S,
    /// Asymptotic p-value, the upper χ²₁ tail probability of `statistic`.
    p_value: f64,
    /// Observed numbers of events `O₁`, `O₂`, in the order of the input groups.
    observed: [usize; 2],
    /// Expected numbers of events `E₁`, `E₂` under the null hypothesis.
    expected: [S; 2],
    /// Hypergeometric variance `V` of `O₁ − E₁`.
    variance: S,
}
impl<S: RealScalar> LogrankResult<S> {
    /// Returns the statistic.
    pub const fn statistic(&self) -> S {
        self.statistic
    }
    /// Returns the p value.
    pub const fn p_value(&self) -> f64 {
        self.p_value
    }
    /// Returns the observed event counts in input-group order.
    pub const fn observed(&self) -> [usize; 2] {
        self.observed
    }
    /// Returns the expected event counts under the null hypothesis.
    pub const fn expected(&self) -> [S; 2] {
        self.expected
    }
    /// Returns the variance.
    pub const fn variance(&self) -> S {
        self.variance
    }
}

/// Computes the result of [`logrank_test_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`logrank_test_with_options`] for the fixed default settings.
pub fn logrank_test<S: RealScalar>(
    first: &[SurvivalObservation<S>],
    second: &[SurvivalObservation<S>],
) -> Result<LogrankResult<S>, SciOxideError> {
    logrank_test_with_options(first, second, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("first"),
        )
    })
}
/// Tests whether two groups have the same hazard function with the log-rank
/// (Mantel–Haenszel) test.
///
/// At each distinct event time `t`, let `rₖ` be the number at risk and `dₖ` the
/// number of events in group `k`, with `r = r₁ + r₂` and `d = d₁ + d₂`. Summing
/// over event times,
///
/// ```text
/// O₁ = Σ d₁,   E₁ = Σ d r₁ / r,   V = Σ d (r − d) r₁ r₂ / (r² (r − 1))
/// ```
///
/// and the statistic `(O₁ − E₁)² / V` is compared with the χ² distribution with
/// one degree of freedom. A subject is at risk at `t` when `entry < t ≤ time`.
/// The test assumes independent censoring.
///
/// Sorting costs O(N log N) time and the merged sweep O(N), for `N = n₁ + n₂`
/// observations. `options.max_bytes()` limits the sorted storage of both groups.
///
/// # Errors
///
/// - `InvalidInput` if either group is empty.
/// - `InvalidInput` if `V` is zero, for example when there are no events.
/// - `ResourceLimit` if the sorted storage exceeds `options.max_bytes()`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::SurvivalOutcome::{Censored, Event};
/// use scioxide_stats::{SurvivalObservation, SurvivalOptions, logrank_test_with_options};
///
/// // At t = 1 both subjects are at risk: O₁ − E₁ = 1/2 and V = 1/4, so the
/// // statistic is (1/2)² / (1/4) = 1.
/// let first = [SurvivalObservation::new(None, 1.0_f64, Event)?];
/// let second = [SurvivalObservation::new(Some(0.0), 2.0, Censored)?];
/// let result = logrank_test_with_options(&first, &second, SurvivalOptions::new().with_max_bytes(1024))?;
/// assert_eq!(result.observed(), [1, 0]);
/// assert_eq!(result.statistic(), 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// N. Mantel, "Evaluation of survival data and two new rank order statistics
/// arising in its consideration", *Cancer Chemother. Rep.* 50 (1966), 163–170.
pub fn logrank_test_with_options<S: RealScalar>(
    first: &[SurvivalObservation<S>],
    second: &[SurvivalObservation<S>],
    options: SurvivalOptions,
) -> Result<LogrankResult<S>, SciOxideError> {
    let max_bytes = options.max_bytes;
    if first.is_empty() || second.is_empty() {
        return Err(invalid(
            scioxide_core::argument!("first"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "log-rank test requires two nonempty groups",
        ));
    }
    admit_payload(
        combined_bytes(
            RiskSweep::required_bytes(first)?,
            RiskSweep::required_bytes(second)?,
        )?,
        max_bytes,
    )?;
    let mut first = RiskSweep::new(first, max_bytes)?;
    let mut second = RiskSweep::new(second, max_bytes - first.bytes()?)?;
    let mut observed = [0usize; 2];
    let mut expected_first = 0.0;
    let mut variance = 0.0;
    loop {
        let time = match (first.next_time(), second.next_time()) {
            (Some(a), Some(b)) => {
                if a < b {
                    a
                } else {
                    b
                }
            }
            (Some(time), None) | (None, Some(time)) => time,
            (None, None) => break,
        };
        let first_row = first.take_at(time);
        let second_row = second.take_at(time);
        let (r1, d1) = (first_row.at_risk, first_row.events);
        let (r2, d2) = (second_row.at_risk, second_row.events);
        if d1 == 0 && d2 == 0 {
            continue;
        }
        let r = (r1 + r2) as f64;
        let d = (d1 + d2) as f64;
        observed[0] += d1;
        observed[1] += d2;
        expected_first += d * r1 as f64 / r;
        if r > 1.0 {
            variance += d * (r - d) * r1 as f64 * r2 as f64 / (r * r * (r - 1.0));
        }
    }
    if variance <= 0.0 || !variance.is_finite() {
        return Err(invalid(
            scioxide_core::argument!("first"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "log-rank variance is zero or unrepresentable",
        ));
    }
    let difference = observed[0] as f64 - expected_first;
    let statistic = difference * difference / variance;
    Ok(LogrankResult {
        statistic: number(statistic)?,
        p_value: ChiSquared::<f64>::new(1.0)?.sf(statistic)?,
        observed,
        expected: [
            number(expected_first)?,
            number((observed[0] + observed[1]) as f64 - expected_first)?,
        ],
        variance: number(variance)?,
    })
}
