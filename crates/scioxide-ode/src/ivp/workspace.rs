//! Workspaces of initial-value plans and their requirements.

use core::marker::PhantomData;

use scioxide_core::{
    AttemptRecord, Capacity, EvaluationCounts, NumericScalar, PartialResultTrust, RealScalar,
    SciOxideError,
};
use scioxide_linalg::Vector;

use super::events::EventStorage;
use crate::OdeProgress;
use crate::shell::Shell;
use crate::stepper::{OdeMethod, Stepper};
/// Describes the workspace size one [`OdeMethod`](crate::OdeMethod) needs for one
/// problem.
///
/// [`OdeMethod::workspace_requirements`](crate::OdeMethod::workspace_requirements)
/// returns it; pass it to [`OdeWorkspace::check`] or [`OdeWorkspace::try_reserve`].
#[derive(Debug, Clone)]
pub struct OdeRequirements<S, M, T = S> {
    pub(crate) dimension: usize,
    pub(crate) events: usize,
    pub(crate) event_records: usize,
    pub(crate) provided: bool,
    pub(crate) method: M,
    pub(crate) scalar: PhantomData<fn() -> (S, T)>,
}

impl<S, M, T> OdeRequirements<S, M, T> {
    /// Returns the number of state components.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the number of event functions.
    pub const fn events(&self) -> usize {
        self.events
    }
    /// Returns the largest number of event records one accepted step can produce.
    pub const fn event_records(&self) -> usize {
        self.event_records
    }
    /// Returns whether the problem supplies its own stage solver.
    pub const fn stage_solver(&self) -> bool {
        self.provided
    }
}

/// Step counts of the current call; reports convert them to core counts.
#[derive(Default)]
pub(crate) struct StepCounts {
    pub(crate) attempts: usize,
    pub(crate) accepted: usize,
    pub(crate) rejected: usize,
}
impl StepCounts {
    pub(crate) const fn counts(&self) -> EvaluationCounts {
        EvaluationCounts::from_iterations(self.attempts).with_proposals(
            self.attempts,
            self.accepted,
            self.rejected,
        )
    }
}

/// Driver storage used by initial-value, delay and adjoint recording drivers.
pub(crate) struct IvpStorage<S: RealScalar, W, T: NumericScalar> {
    pub(crate) method: W,
    pub(crate) dimension: usize,
    pub(crate) candidate: Vector<T>,
    pub(crate) sample: Vector<T>,
    pub(crate) base_rhs: Vector<T>,
    pub(crate) scales: Vector<S>,
    pub(crate) magnitudes: Vector<S>,
    pub(crate) probe: [Vector<T>; 4],
    pub(crate) events: EventStorage<S, T>,
    /// Driver step attempts, acceptances and rejections of the current epoch.
    pub(crate) steps: StepCounts,
    pub(crate) last_step_error_estimate: Option<S>,
    pub(crate) trust: PartialResultTrust,
}

impl<S: RealScalar + NumericScalar<Real = S>, W: Default, T: NumericScalar<Real = S>> Default
    for IvpStorage<S, W, T>
{
    fn default() -> Self {
        Self {
            method: W::default(),
            dimension: 0,
            candidate: Vector::empty(),
            sample: Vector::empty(),
            base_rhs: Vector::empty(),
            scales: Vector::empty(),
            magnitudes: Vector::empty(),
            probe: core::array::from_fn(|_| Vector::empty()),
            events: EventStorage::default(),
            steps: StepCounts::default(),
            last_step_error_estimate: None,
            trust: PartialResultTrust::DiagnosticOnly,
        }
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, W: Default, T: NumericScalar<Real = S>>
    IvpStorage<S, W, T>
{
    fn vectors_fit(&self, dimension: usize) -> bool {
        self.dimension == dimension
            && self.candidate.len() == dimension
            && self.sample.len() == dimension
            && self.base_rhs.len() == dimension
            && self.scales.len() == dimension
            && self.magnitudes.len() == dimension
            && self.probe.iter().all(|vector| vector.len() == dimension)
    }

    pub(crate) fn fits<M: Stepper<S, T, Workspace = W>>(
        &self,
        method: &M,
        dimension: usize,
        events: usize,
        records: usize,
        provided: bool,
    ) -> bool {
        self.vectors_fit(dimension)
            && self.events.fits(events, records, dimension)
            && method.fits(dimension, provided, &self.method)
    }

    pub(crate) fn reserve<M: Stepper<S, T, Workspace = W>>(
        &mut self,
        method: &M,
        dimension: usize,
        events: usize,
        records: usize,
        provided: bool,
    ) -> Result<(), SciOxideError> {
        self.reserve_driver(dimension, events, records)?;
        if !method.fits(dimension, provided, &self.method) {
            method.reserve(dimension, provided, &mut self.method)?;
        }
        Ok(())
    }

    /// Reserve driver vectors independently of the numerical trial kernel.
    pub(crate) fn reserve_driver(
        &mut self,
        dimension: usize,
        events: usize,
        records: usize,
    ) -> Result<(), SciOxideError> {
        if !self.vectors_fit(dimension) {
            self.dimension = 0;
            self.candidate = Vector::zeros(dimension)?;
            self.sample = Vector::zeros(dimension)?;
            self.base_rhs = Vector::zeros(dimension)?;
            self.scales = Vector::zeros(dimension)?;
            self.magnitudes = Vector::zeros(dimension)?;
            for vector in &mut self.probe {
                *vector = Vector::zeros(dimension)?;
            }
            self.dimension = dimension;
        }
        if !self.events.fits(events, records, dimension) {
            self.events.reserve(events, records, dimension)?;
        }
        Ok(())
    }

    /// Reset the per-epoch driver ledger.
    pub(crate) fn begin(&mut self) {
        self.steps = StepCounts::default();
        self.last_step_error_estimate = None;
        self.trust = PartialResultTrust::DiagnosticOnly;
    }
}
/// Holds the storage that [`OdeMethod::advance`](crate::OdeMethod::advance) needs, and
/// the record of the last call.
///
/// Create one with [`OdeMethod::workspace`](crate::OdeMethod::workspace), or with
/// [`Self::new`] and [`Self::try_reserve`]. One workspace can serve several plans of
/// the same method type; it keeps the largest size it was given. See
/// [Reusing workspaces](crate#reusing-workspaces).
pub struct OdeWorkspace<S, M, T = S>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: OdeMethod<S, T>,
{
    pub(crate) storage: IvpStorage<S, <M as Stepper<S, T>>::Workspace, T>,
    pub(crate) shell: Shell<OdeProgress<S, T>>,
}

impl<S, M, T> Default for OdeWorkspace<S, M, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: OdeMethod<S, T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S, M, T> core::fmt::Debug for OdeWorkspace<S, M, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: OdeMethod<S, T>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OdeWorkspace")
            .field("dimension", &self.storage.dimension)
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}

impl<S, M, T> OdeWorkspace<S, M, T>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    M: OdeMethod<S, T>,
{
    /// Creates an empty workspace.
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage: IvpStorage::default(),
            shell: Shell::new(),
        }
    }

    /// Returns whether this workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// This method does not fail; it returns a `Result` like the `check` methods
    /// of other workspaces.
    pub fn check(
        &self,
        requirements: &OdeRequirements<S, M, T>,
    ) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.fits(
                &requirements.method,
                requirements.dimension,
                requirements.events,
                requirements.event_records,
                requirements.provided,
            ) {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Enlarges this workspace to fit `requirements`.
    ///
    /// On error, the workspace may keep part of the new storage and stays usable.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn try_reserve(
        &mut self,
        requirements: &OdeRequirements<S, M, T>,
    ) -> Result<(), SciOxideError> {
        self.reserve(
            &requirements.method,
            requirements.dimension,
            requirements.events,
            requirements.event_records,
            requirements.provided,
        )
    }

    fn fits(
        &self,
        method: &M,
        dimension: usize,
        events: usize,
        records: usize,
        provided: bool,
    ) -> bool {
        self.storage
            .fits(method, dimension, events, records, provided)
            && self
                .shell
                .has_spare(|progress| progress.state.len() == dimension)
    }

    fn reserve(
        &mut self,
        method: &M,
        dimension: usize,
        events: usize,
        records: usize,
        provided: bool,
    ) -> Result<(), SciOxideError> {
        self.storage
            .reserve(method, dimension, events, records, provided)?;
        self.shell.reserve_spare(
            |progress| progress.state.len() == dimension,
            || OdeProgress::new(dimension),
        )
    }

    /// Enlarges this workspace for `method` unless it already fits. It borrows the
    /// plan's method instead of cloning it into requirements, because a method may
    /// own allocated storage, such as a sparse stage pattern.
    pub(crate) fn prepare(
        &mut self,
        method: &M,
        dimension: usize,
        events: usize,
        records: usize,
        provided: bool,
    ) -> Result<(), SciOxideError> {
        if self.fits(method, dimension, events, records, provided) {
            Ok(())
        } else {
            self.reserve(method, dimension, events, records, provided)
        }
    }

    /// Returns the record of the last call of `advance`: its outcome, its counts
    /// and the last accepted [`OdeProgress`](crate::OdeProgress); `None` before the
    /// first call.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<OdeProgress<S, T>>> {
        self.shell.last_attempt()
    }
}
