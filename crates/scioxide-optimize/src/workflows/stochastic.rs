//! Stochastic gradient descent and a sampled first-order trust-region method.
use super::sampling::Batch;
use super::*;
use alloc::vec::Vec;

use core::time::Duration;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AttemptKind, AttemptRecord, Budget, EvaluationRole, ExecutionPolicy, GradientTolerance,
    PartialResultTrust, SolveBudgets, SolveCheckpointMetadata, SolveScope,
    backend::{checked_add, checked_mul, try_filled},
};

use crate::attempt::{Entry, Shape, Shell, Staged, assign, preflight, reserve_len};

/// Sets the step length `αₖ` of [`StochasticGradientDescent`] at iteration `k`
/// (the number of completed updates).
#[derive(Debug, Clone, Copy)]
pub enum SgdStep<R: RealScalar> {
    /// Constant step `αₖ = α > 0`. With noisy gradients the iterates do not
    /// converge to the minimizer but fluctuate around it.
    Constant(R),
    /// Diminishing step `αₖ = α₀ / (1 + d k)^p` with `1/2 < p ≤ 1`, which
    /// satisfies the Robbins–Monro conditions `Σ αₖ = ∞` and `Σ αₖ² < ∞`.
    Diminishing {
        /// Initial step `α₀ > 0`.
        initial: R,
        /// Decay rate `d > 0`.
        decay: R,
        /// Exponent `p` in `(1/2, 1]`.
        power: R,
    },
}
impl<R: RealScalar> SgdStep<R> {
    /// Validate the step supplied as the entry point's `step` argument.
    fn validate(self) -> Result<(), SciOxideError> {
        const STEP: scioxide_core::Argument = scioxide_core::argument!("step");
        match self {
            Self::Constant(s) => crate::proximal::positive(STEP, s),
            Self::Diminishing {
                initial,
                decay,
                power,
            } => {
                crate::proximal::positive(STEP, initial)?;
                crate::proximal::positive(STEP, decay)?;
                if !power.is_finite()
                    || power <= number(0.5)?
                    || power > <_ as scioxide_core::AlgebraicScalar>::ONE
                {
                    return Err(invalid(STEP, "SGD exponent must be in (1/2,1]"));
                }
                Ok(())
            }
        }
    }
    fn at(self, k: usize) -> Result<R, SciOxideError> {
        match self {
            Self::Constant(s) => Ok(s),
            Self::Diminishing {
                initial,
                decay,
                power,
            } => {
                finite(initial / finite(R::ONE + decay * R::try_from_usize_exact(k)?)?.powf(power))
            }
        }
    }
}
#[derive(Debug, Clone, Copy)]
struct Common<R: RealScalar> {
    replication: ReplicationOptions<R>,
    iterations: Budget,
    tolerance: GradientTolerance<R>,
    seed: u64,
    execution: ExecutionPolicy,
    time_limit: Option<Duration>,
}
impl<R: RealScalar> Common<R> {
    fn new(iterations: Budget, replication: ReplicationOptions<R>) -> Self {
        Self {
            replication,
            iterations,
            tolerance: GradientTolerance::default(),
            seed: 42,
            execution: ExecutionPolicy::Sequential,
            time_limit: None,
        }
    }
}
macro_rules! common_options {
    ($options:ident) => {
        impl<R: RealScalar> $options<R> {
            /// Sets the root seed of the sample keys. The default is 42.
            #[must_use]
            pub const fn with_seed(mut self, seed: u64) -> Self {
                self.common.seed = seed;
                self
            }
            /// Sets the gradient tolerance. The solve stops when
            /// `‖ḡ‖ + c·‖σ̂‖` meets it, where `ḡ` is the mean sampled gradient,
            /// `σ̂` its standard errors and `c` the error multiplier. For a
            /// deterministic objective this is
            /// [`GradientSmall`](crate::OptimizeTermination::GradientSmall);
            /// otherwise it is the unsuccessful
            /// [`EstimatedStationarity`](crate::OptimizeTermination::EstimatedStationarity).
            #[must_use]
            pub const fn with_tolerance(mut self, tolerance: GradientTolerance<R>) -> Self {
                self.common.tolerance = tolerance;
                self
            }
            /// Sets the execution policy for evaluating replicated samples. The
            /// default is sequential. Results do not depend on the number of
            /// workers.
            #[must_use]
            pub const fn with_execution(mut self, execution: ExecutionPolicy) -> Self {
                self.common.execution = execution;
                self
            }
            /// Sets a time limit. A solve then needs an execution context with a
            /// clock.
            #[must_use]
            pub const fn with_time_limit(mut self, limit: Duration) -> Self {
                self.common.time_limit = Some(limit);
                self
            }
            /// Returns the iteration limit, which counts the iterations of a
            /// resumed run together with those before the checkpoint.
            pub const fn max_iterations(&self) -> Budget {
                self.common.iterations
            }
            /// Returns the replication options.
            pub const fn replication(&self) -> ReplicationOptions<R> {
                self.common.replication
            }
            /// Returns the gradient tolerance.
            pub const fn tolerance(&self) -> GradientTolerance<R> {
                self.common.tolerance
            }
            /// Returns the root seed.
            pub const fn seed(&self) -> u64 {
                self.common.seed
            }
            /// Returns the execution policy.
            pub const fn execution(&self) -> ExecutionPolicy {
                self.common.execution
            }
            /// Returns the time limit, if set.
            pub const fn time_limit(&self) -> Option<Duration> {
                self.common.time_limit
            }
        }
    };
}
/// Configures [`StochasticGradientDescent`].
///
/// Each iteration averages a fixed number of gradient samples (the batch) and
/// evaluates the objective at the new point with separate samples. See also
/// [Termination](crate#termination).
#[derive(Debug, Clone, Copy)]
pub struct SgdOptions<R: RealScalar> {
    common: Common<R>,
    step: SgdStep<R>,
}
impl<R: RealScalar> SgdOptions<R> {
    /// Creates options with the step schedule `step`, at most `iterations`
    /// iterations and `batch` samples per estimate. The solver constructor
    /// checks the schedule.
    pub fn new(step: SgdStep<R>, iterations: Budget, batch: Budget) -> Self {
        Self {
            common: Common::new(iterations, ReplicationOptions::fixed(batch)),
            step,
        }
    }
    /// Sets the multiplier of the sample standard error used in the stopping
    /// test.
    #[must_use]
    pub const fn with_error_multiplier(mut self, multiplier: R) -> Self {
        self.common.replication = self.common.replication.with_error_multiplier(multiplier);
        self
    }
    /// Returns the step schedule.
    pub const fn step(&self) -> SgdStep<R> {
        self.step
    }
    fn validate(&self) -> Result<(), SciOxideError> {
        self.step.validate()?;
        self.common.replication.validate()
    }
}
common_options!(SgdOptions);
/// Configures [`StochasticTrustRegion`].
///
/// See [`StochasticTrustRegion`] for the meaning of the parameters and
/// [Termination](crate#termination) for the statuses.
#[derive(Debug, Clone, Copy)]
pub struct StochasticTrustOptions<R: RealScalar> {
    common: Common<R>,
    initial_radius: R,
    maximum_radius: R,
    minimum_radius: R,
    acceptance: R,
    gradient_radius: R,
    growth: R,
    gradient_accuracy: R,
}
impl<R: RealScalar> StochasticTrustOptions<R> {
    /// Creates options with at most `iterations` iterations and the given
    /// replication. The defaults are: initial radius `Δ₀ = 1`, maximum radius
    /// 10, minimum radius the machine epsilon, acceptance ratio `η = 0.1`,
    /// radius factor `γ = 2`, gradient–radius ratio `β = 0.001` and gradient
    /// accuracy `κ = 1/4`.
    pub fn new(iterations: Budget, replication: ReplicationOptions<R>) -> Self {
        let ten = R::TWO * R::TWO * R::TWO + R::TWO;
        Self {
            common: Common::new(iterations, replication),
            initial_radius: R::ONE,
            maximum_radius: ten,
            minimum_radius: R::EPSILON,
            acceptance: R::ONE / ten,
            gradient_radius: R::ONE / (ten * ten * ten),
            growth: R::TWO,
            gradient_accuracy: R::HALF * R::HALF,
        }
    }
    /// Sets the minimum, initial and maximum trust radii. The solver
    /// constructor requires them to be positive, finite and in that order.
    #[must_use]
    pub const fn with_radii(mut self, minimum: R, initial: R, maximum: R) -> Self {
        self.minimum_radius = minimum;
        self.initial_radius = initial;
        self.maximum_radius = maximum;
        self
    }
    /// Sets the acceptance ratio `η`, the gradient–radius ratio `β`, the radius
    /// factor `γ` and the gradient accuracy `κ`. The solver constructor requires
    /// `0 < η < 1`, `β > 0`, `γ > 1` and `κ > 0`.
    #[must_use]
    pub const fn with_model_policy(
        mut self,
        acceptance: R,
        gradient_radius: R,
        growth: R,
        gradient_accuracy: R,
    ) -> Self {
        self.acceptance = acceptance;
        self.gradient_radius = gradient_radius;
        self.growth = growth;
        self.gradient_accuracy = gradient_accuracy;
        self
    }
    /// Returns the minimum, initial and maximum trust radii.
    pub const fn radii(&self) -> (R, R, R) {
        (
            self.minimum_radius,
            self.initial_radius,
            self.maximum_radius,
        )
    }
    fn validate(&self) -> Result<(), SciOxideError> {
        self.common.replication.validate()?;
        for (argument, v) in [
            (scioxide_core::argument!("minimum"), self.minimum_radius),
            (scioxide_core::argument!("initial"), self.initial_radius),
            (scioxide_core::argument!("maximum"), self.maximum_radius),
            (scioxide_core::argument!("acceptance"), self.acceptance),
            (
                scioxide_core::argument!("gradient_radius"),
                self.gradient_radius,
            ),
            (scioxide_core::argument!("growth"), self.growth),
            (
                scioxide_core::argument!("gradient_accuracy"),
                self.gradient_accuracy,
            ),
        ] {
            crate::proximal::positive(argument, v)?;
        }
        if self.minimum_radius > self.initial_radius || self.initial_radius > self.maximum_radius {
            return Err(invalid(
                scioxide_core::argument!("initial"),
                "trust radii must be ordered",
            ));
        }
        if self.acceptance >= R::ONE || self.growth <= R::ONE {
            return Err(invalid(
                scioxide_core::argument!("acceptance"),
                "trust acceptance or radius factor",
            ));
        }
        Ok(())
    }
}
common_options!(StochasticTrustOptions);
#[derive(Debug, Clone, Copy)]
enum Method<R: RealScalar> {
    Sgd(SgdOptions<R>),
    Trust(StochasticTrustOptions<R>),
}
impl<R: RealScalar> Method<R> {
    fn common(self) -> Common<R> {
        match self {
            Self::Sgd(o) => o.common,
            Self::Trust(o) => o.common,
        }
    }
}

/// Borrows a sampled objective and the sampler that generates its sample
/// tokens.
pub struct StochasticProblem<'a, P: ?Sized, Sm: ?Sized = PhiloxSampler> {
    callback: &'a P,
    sampler: &'a Sm,
}
impl<P: ?Sized, Sm: ?Sized> Clone for StochasticProblem<'_, P, Sm> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P: ?Sized, Sm: ?Sized> Copy for StochasticProblem<'_, P, Sm> {}
impl<'a, P: ?Sized> StochasticProblem<'a, P> {
    /// Creates a problem that draws sample tokens with [`PhiloxSampler`].
    pub const fn new(callback: &'a P) -> Self {
        Self {
            callback,
            sampler: &PhiloxSampler,
        }
    }
}
impl<'a, P: ?Sized, Sm: ?Sized> StochasticProblem<'a, P, Sm> {
    /// Replaces the sampler. A checkpoint can be resumed only with a sampler of
    /// the same configuration.
    pub const fn with_sampler<T: ?Sized>(self, sampler: &'a T) -> StochasticProblem<'a, P, T> {
        StochasticProblem {
            callback: self.callback,
            sampler,
        }
    }
    /// Returns the sampled objective.
    pub const fn callback(&self) -> &'a P {
        self.callback
    }
    /// Returns the sampler.
    pub const fn sampler(&self) -> &'a Sm {
        self.sampler
    }
}

/// Holds the result of a [`StochasticGradientDescent`] or
/// [`StochasticTrustRegion`] solve: the last accepted point and sample
/// estimates of the objective and gradient there.
#[derive(Debug, Clone)]
pub struct StochasticResult<S: NumericScalar> {
    x: Vec<S>,
    gradient: Vec<S>,
    gradient_errors: Vec<S::Real>,
    value: Option<S::Real>,
    value_error: Option<S::Real>,
    value_variance: Option<S::Real>,
    value_samples: usize,
    gradient_current: bool,
    gradient_error_available: bool,
    gradient_samples: usize,
    iterations: usize,
    radius: S::Real,
    cursor: u64,
    termination: OptimizeTermination,
    counts: scioxide_core::SolveCounts,
}
impl<S: NumericScalar> Default for StochasticResult<S> {
    fn default() -> Self {
        Self::empty()
    }
}
impl<S: NumericScalar> StochasticResult<S> {
    /// Returns an empty result, without allocating, to pass to `minimize_into`.
    pub fn empty() -> Self {
        Self {
            x: Vec::new(),
            gradient: Vec::new(),
            gradient_errors: Vec::new(),
            value: None,
            value_error: None,
            value_variance: None,
            value_samples: 0,
            gradient_current: false,
            gradient_error_available: false,
            gradient_samples: 0,
            iterations: 0,
            radius: S::Real::ZERO,
            cursor: 0,
            termination: OptimizeTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
        }
    }
    fn filled(dimension: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            x: try_filled(dimension, S::ZERO, scioxide_core::quantity!("state.x"))?,
            gradient: try_filled(
                dimension,
                S::ZERO,
                scioxide_core::quantity!("state.gradient"),
            )?,
            gradient_errors: try_filled(
                dimension,
                S::Real::ZERO,
                scioxide_core::quantity!("state.gradient_errors"),
            )?,
            ..Self::empty()
        })
    }
    /// Reserves storage for a result with `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the storage cannot be reserved.
    pub fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        let q = scioxide_core::quantity!("stochastic_result");
        reserve_len(&mut self.x, dimension, q)?;
        reserve_len(&mut self.gradient, dimension, q)?;
        reserve_len(&mut self.gradient_errors, dimension, q)
    }
    fn fits(&self, dimension: usize) -> bool {
        self.x.capacity() >= dimension
            && self.gradient.capacity() >= dimension
            && self.gradient_errors.capacity() >= dimension
    }
    /// Returns the last accepted point.
    pub fn x(&self) -> &[S] {
        &self.x
    }
    /// Returns the sample mean of the objective at the accepted point.
    pub const fn value(&self) -> Option<S::Real> {
        self.value
    }
    /// Returns the standard error of [`Self::value`], if at least two samples
    /// were taken.
    pub const fn standard_error(&self) -> Option<S::Real> {
        self.value_error
    }
    /// Returns the sample variance of the objective at the accepted point.
    pub const fn sample_variance(&self) -> Option<S::Real> {
        self.value_variance
    }
    /// Returns the number of samples averaged in [`Self::value`].
    pub const fn value_replicates(&self) -> usize {
        self.value_samples
    }
    /// Returns the mean sampled gradient at the accepted point, or `None` if it
    /// has not been sampled there.
    pub fn gradient(&self) -> Option<&[S]> {
        self.gradient_current.then_some(&self.gradient)
    }
    /// Returns the standard error of each gradient component. For complex
    /// variables it combines the variances of the real and imaginary parts.
    pub fn gradient_standard_errors(&self) -> Option<&[S::Real]> {
        (self.gradient_current && self.gradient_error_available).then_some(&self.gradient_errors)
    }
    /// Returns the number of samples averaged in [`Self::gradient`].
    pub fn gradient_replicates(&self) -> Option<usize> {
        self.gradient_current.then_some(self.gradient_samples)
    }
    /// Returns the number of iterations, including rejected trust-region
    /// steps.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the current trust radius, or zero for SGD.
    pub const fn trust_radius(&self) -> S::Real {
        self.radius
    }
    /// Returns the sequence number of the next sample.
    pub const fn sample_cursor(&self) -> u64 {
        self.cursor
    }
    /// Returns why the solver stopped.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    /// Returns the iteration, sample and evaluation counts of the latest call.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns whether the solve stopped with a small gradient of a
    /// deterministic objective. For noisy objectives the best outcome is the
    /// unsuccessful
    /// [`EstimatedStationarity`](crate::OptimizeTermination::EstimatedStationarity).
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }

    /// Requires the method's stopping test to have passed.
    ///
    /// # Errors
    ///
    /// Returns `ExecutionStopped` for cancellation or a deadline, `DomainError`
    /// for verified infeasibility or unboundedness, `InvalidInput` before a solve,
    /// and `ConvergenceFailure` with counts for every other unsuccessful termination.
    pub fn require_success(&self) -> Result<(), SciOxideError> {
        crate::result::termination_error(
            scioxide_core::argument!("self"),
            self.termination(),
            self.counts(),
            None,
        )
        .map_or(Ok(()), Err)
    }

    /// Copy every field of `other`; reserved storage copies without growth.
    fn assign_from(&mut self, other: &Self) {
        assign(&mut self.x, &other.x);
        assign(&mut self.gradient, &other.gradient);
        assign(&mut self.gradient_errors, &other.gradient_errors);
        self.value = other.value;
        self.value_error = other.value_error;
        self.value_variance = other.value_variance;
        self.value_samples = other.value_samples;
        self.gradient_current = other.gradient_current;
        self.gradient_error_available = other.gradient_error_available;
        self.gradient_samples = other.gradient_samples;
        self.iterations = other.iterations;
        self.radius = other.radius;
        self.cursor = other.cursor;
        self.termination = other.termination;
        self.counts = other.counts;
    }
    // Checkpoint integrity covers every numerical lane; `Debug` prints each
    // float in its shortest exactly round-tripping form.
    fn state_token(&self) -> u128 {
        crate::context::options_token(&(
            &self.x,
            &self.gradient,
            &self.gradient_errors,
            (self.value, self.value_error, self.value_variance),
            (self.value_samples, self.gradient_samples, self.iterations),
            (self.gradient_current, self.gradient_error_available),
            self.radius,
            self.cursor,
        ))
    }
}
impl<S: NumericScalar> Staged for StochasticResult<S> {
    fn reset(&mut self) {
        let (x, gradient, errors) = (
            core::mem::take(&mut self.x),
            core::mem::take(&mut self.gradient),
            core::mem::take(&mut self.gradient_errors),
        );
        *self = Self {
            x,
            gradient,
            gradient_errors: errors,
            ..Self::empty()
        };
        self.x.clear();
        self.gradient.clear();
        self.gradient_errors.clear();
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        self.try_reserve(source.x.len())?;
        self.assign_from(source);
        Ok(())
    }
    fn succeeded(&self) -> bool {
        self.is_success()
    }
}

/// Stores the accepted state of a stochastic solve, to continue it later with
/// `resume_into`.
#[derive(Debug, Clone)]
pub struct StochasticCheckpoint<S: NumericScalar> {
    state: StochasticResult<S>,
    metadata: SolveCheckpointMetadata,
}
impl<S: NumericScalar> StochasticCheckpoint<S> {
    /// Returns the accepted point.
    pub fn x(&self) -> &[S] {
        &self.state.x
    }
    /// Returns the sample mean of the objective at the accepted point.
    pub const fn value(&self) -> Option<S::Real> {
        self.state.value
    }
    /// Returns the number of completed iterations.
    pub const fn iterations(&self) -> usize {
        self.state.iterations
    }
    /// Returns the sequence number of the next sample. A resumed run continues
    /// from it, so it never reuses a sample.
    pub const fn sample_cursor(&self) -> u64 {
        self.state.cursor
    }
    /// Returns the solve metadata stored with the checkpoint.
    pub const fn metadata(&self) -> SolveCheckpointMetadata {
        self.metadata
    }
}

/// Describes the workspace size of a stochastic solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StochasticWorkspaceRequirements {
    dimension: usize,
    replicates: usize,
    bytes: usize,
}
impl StochasticWorkspaceRequirements {
    fn new<S: NumericScalar>(dimension: usize, replicates: usize) -> Result<Self, SciOxideError> {
        if dimension == 0 {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                "stochastic dimension must be positive",
            ));
        }
        let q = scioxide_core::quantity!("stochastic_workspace");
        // Accepted state and staged result (three vectors each), the trial point,
        // and gradient plus two scalar replicate batches.
        let rows = checked_add(replicates, 3, q)?;
        let elements = checked_add(
            checked_mul(dimension, checked_add(7, rows, q)?, q)?,
            checked_mul(2, rows, q)?,
            q,
        )?;
        Ok(Self {
            dimension,
            replicates,
            bytes: checked_mul(elements, core::mem::size_of::<S>(), q)?,
        })
    }
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
    /// Returns the maximum number of samples per estimate.
    pub const fn replicates(self) -> usize {
        self.replicates
    }
    /// Returns the approximate workspace size in bytes.
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

/// Summarizes the last accepted point of a stochastic solve call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticProgress<R> {
    iterations: usize,
    value: R,
    trust: PartialResultTrust,
}
impl<R: Copy> StochasticProgress<R> {
    /// Returns the number of completed iterations.
    pub const fn iterations(self) -> usize {
        self.iterations
    }
    /// Returns the sample mean of the objective at the accepted point.
    pub const fn value(self) -> R {
        self.value
    }
    /// Returns how far the accepted point has been checked.
    pub const fn trust(self) -> PartialResultTrust {
        self.trust
    }
}

struct Storage<S: NumericScalar>
where
    S::Real: NumericScalar<Real = S::Real>,
{
    state: StochasticResult<S>,
    trial: Vec<S>,
    gradient: Batch<S>,
    base: Batch<S::Real>,
    candidate: Batch<S::Real>,
    maximum: usize,
}
impl<S: NumericScalar> Shape<Option<Storage<S>>> for StochasticWorkspaceRequirements
where
    S::Real: NumericScalar<Real = S::Real>,
{
    fn fits(&self, storage: &Option<Storage<S>>) -> bool {
        storage
            .as_ref()
            .is_some_and(|w| w.state.x.len() == self.dimension && w.maximum == self.replicates)
    }
    fn prepare(&self, storage: &mut Option<Storage<S>>) -> Result<(), SciOxideError> {
        let (dimension, maximum) = (self.dimension, self.replicates);
        *storage = Some(Storage {
            state: StochasticResult::filled(dimension)?,
            trial: try_filled(
                dimension,
                S::ZERO,
                scioxide_core::quantity!("workspace.trial"),
            )?,
            gradient: Batch::new(dimension, maximum)?,
            base: Batch::new(1, maximum)?,
            candidate: Batch::new(1, maximum)?,
            maximum,
        });
        Ok(())
    }
}

/// Holds the storage of a [`StochasticGradientDescent`] or
/// [`StochasticTrustRegion`] solve, a record of the last call, and the state
/// for [`Self::checkpoint`].
pub struct StochasticWorkspace<S: NumericScalar>
where
    S::Real: NumericScalar<Real = S::Real>,
{
    storage: Option<Storage<S>>,
    staged: StochasticResult<S>,
    shell: Shell<StochasticProgress<S::Real>>,
    checkpoint_metadata: Option<SolveCheckpointMetadata>,
}
impl<S: NumericScalar> Default for StochasticWorkspace<S>
where
    S::Real: NumericScalar<Real = S::Real>,
{
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> StochasticWorkspace<S>
where
    S::Real: NumericScalar<Real = S::Real>,
{
    /// Creates an empty workspace; reserve storage with [`Self::try_reserve`].
    pub fn new() -> Self {
        Self {
            storage: None,
            staged: StochasticResult::empty(),
            shell: Shell::new(),
            checkpoint_metadata: None,
        }
    }
    /// Returns whether the workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// Never fails; this workspace has no byte limit.
    pub fn check(
        &self,
        requirements: &StochasticWorkspaceRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        Ok(
            if requirements.fits(&self.storage) && self.staged.fits(requirements.dimension) {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    /// Reserves storage for `requirements`. On failure the existing storage is
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure`.
    pub fn try_reserve(
        &mut self,
        requirements: &StochasticWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        if self.check(requirements)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        if !requirements.fits(&self.storage) {
            requirements.prepare(&mut self.storage)?;
        }
        self.staged.try_reserve(requirements.dimension)
    }
    /// Returns the record of the most recent call or resume, or `None`.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<StochasticProgress<S::Real>>> {
        self.shell.last_attempt()
    }
    /// Returns the last accepted point of the most recent call, or `None`.
    pub fn last_accepted(&self) -> Option<&[S]> {
        self.shell.progress()?;
        Some(&self.storage.as_ref()?.state.x)
    }
    /// Copies the accepted state of the most recent call into a checkpoint.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Malformed`) if the most recent call accepted no point.
    /// - `AllocationFailure` if the checkpoint cannot be allocated.
    pub fn checkpoint(&self) -> Result<StochasticCheckpoint<S>, SciOxideError> {
        let missing = || {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Malformed,
                "no accepted state in the current attempt",
            )
        };
        let metadata = self.checkpoint_metadata.ok_or_else(missing)?;
        self.shell.progress().ok_or_else(missing)?;
        let source = &self.storage.as_ref().ok_or_else(missing)?.state;
        let mut state = StochasticResult::empty();
        state.publish(source)?;
        Ok(StochasticCheckpoint { state, metadata })
    }
}
/// Minimizes the expectation `f(x) = E[F(x, ξ)]` of a sampled objective by
/// stochastic gradient descent.
///
/// Each iteration averages a batch of sampled gradients `ḡₖ` and takes the
/// step `xₖ₊₁ = xₖ − αₖ ḡₖ`, with `αₖ` from [`SgdStep`]. Every step is
/// accepted; the objective is sampled at the new point only to report it. Each
/// iteration costs one gradient batch and one objective batch. Variables may be
/// real or complex. Samples come from the problem's [`OptimizationSampler`], so
/// a run is reproducible from its seed and does not depend on the number of
/// workers.
///
/// # Errors
///
/// - `InvalidInput` from [`Self::new`] if the step schedule or the error
///   multiplier is invalid.
/// - `InvalidInput` if `x0` is empty or does not match the problem dimension,
///   or the problem does not have exactly one objective component.
/// - `NonFinite` if `x0`, a sample or a step is not finite.
/// - `CapabilityUnavailable` if a time limit is set and the context has no
///   clock.
/// - `ExecutionStopped`, or the error of the exhausted limit, if the solve
///   stops before the first objective estimate.
/// - Any error returned by the objective or the sampler.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, SciOxideError};
/// use scioxide_optimize::{
///     NoiseModel, SampledGradient, SampledObjective, SgdOptions, SgdStep, StochasticGradientDescent,
///     StochasticProblem,
/// };
///
/// // f(x) = (x − 2)², without noise; the minimizer is x = 2.
/// struct Quadratic;
/// impl SampledObjective<f64> for Quadratic {
///     fn dimension(&self) -> usize {
///         1
///     }
///     fn objectives(&self) -> usize {
///         1
///     }
///     fn noise_model(&self) -> NoiseModel {
///         NoiseModel::Deterministic
///     }
///     fn sample_into(&self, x: &[f64], _: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
///         out[0] = (x[0] - 2.0).powi(2);
///         Ok(())
///     }
/// }
/// impl SampledGradient<f64> for Quadratic {
///     fn gradient_sample_into(&self, x: &[f64], _: u64, g: &mut [f64]) -> Result<(), SciOxideError> {
///         g[0] = 2.0 * (x[0] - 2.0);
///         Ok(())
///     }
/// }
///
/// let options = SgdOptions::new(SgdStep::Constant(0.25), Budget::try_from(100)?, Budget::ONE);
/// let result = StochasticGradientDescent::new(options)?
///     .minimize(&StochasticProblem::new(&Quadratic), &[0.0])?;
/// assert!(result.is_success() && (result.x()[0] - 2.0).abs() < 1e-6);
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// - H. Robbins and S. Monro, "A stochastic approximation method", *Ann. Math.
///   Statist.* 22(3), 400–407, 1951. <https://doi.org/10.1214/aoms/1177729586>
/// - L. Bottou, F. E. Curtis and J. Nocedal, "Optimization methods for
///   large-scale machine learning", *SIAM Rev.* 60(2), 223–311, 2018.
///   <https://doi.org/10.1137/16M1080173>
#[derive(Debug, Clone)]
pub struct StochasticGradientDescent<R: RealScalar> {
    options: SgdOptions<R>,
}
/// Minimizes the expectation `f(x) = E[F(x, ξ)]` of a sampled objective by a
/// first-order trust-region method with adaptive sampling.
///
/// Each iteration estimates the gradient `ḡ` at `x`, doubling the number of
/// samples (up to the replication maximum) until its error margin, the
/// [`ReplicationOptions`](crate::ReplicationOptions) error multiplier times the standard error, is at
/// most `κΔ`, where `κ` is the gradient accuracy and `Δ` the trust radius. The trial point is
/// `x − (Δ/‖ḡ‖) ḡ`, with predicted decrease `Δ‖ḡ‖`. It is rejected without
/// sampling if `‖ḡ‖ < βΔ`. Otherwise the objective is sampled at `x` and at the
/// trial, again doubling the samples until the observed decrease is clearly
/// above or below `η Δ‖ḡ‖` within its error margin. An accepted step
/// multiplies `Δ` by `γ` (up to the maximum radius); a rejected one divides it
/// by `γ`. The solve stops with
/// [`StepSmall`](crate::OptimizeTermination::StepSmall) when `Δ` falls below
/// the minimum radius, and with
/// [`NoiseLimit`](crate::OptimizeTermination::NoiseLimit) when the maximum
/// number of samples cannot resolve a decision. Other behavior and the methods
/// are those of [`StochasticGradientDescent`].
///
/// # Errors
///
/// As for [`StochasticGradientDescent`], and `InvalidInput` if the objective
/// is noisy and the replication minimum is less than two, or if the radii or
/// model parameters violate the conditions of [`StochasticTrustOptions`].
///
/// # References
///
/// R. Chen, M. Menickelly and K. Scheinberg, "Stochastic optimization using a
/// trust-region method and random models", *Math. Program.* 169(2), 447–487,
/// 2018. <https://doi.org/10.1007/s10107-017-1141-8>
#[derive(Debug, Clone)]
pub struct StochasticTrustRegion<R: RealScalar> {
    options: StochasticTrustOptions<R>,
}

macro_rules! solver {
    ($name:ident,$options:ident,$variant:ident) => {
        impl<R: RealScalar + NumericScalar<Real = R>> $name<R> {
            /// Creates a solver with checked options.
            ///
            /// # Errors
            ///
            /// Returns `InvalidInput` for an invalid step schedule, radius, model
            /// parameter or error multiplier.
            pub fn new(options: $options<R>) -> Result<Self, SciOxideError> {
                options.validate()?;
                Ok(Self { options })
            }
            /// Returns the options.
            pub const fn options(&self) -> &$options<R> {
                &self.options
            }
            /// Returns the workspace size for `dimension` variables.
            ///
            /// # Errors
            ///
            /// Returns `InvalidInput` if `dimension` is zero and `SizeOverflow` if
            /// the size does not fit in `usize`.
            pub fn workspace_requirements<S: NumericScalar<Real = R>>(
                &self,
                dimension: usize,
            ) -> Result<StochasticWorkspaceRequirements, SciOxideError> {
                StochasticWorkspaceRequirements::new::<S>(
                    dimension,
                    self.options.common.replication.maximum,
                )
            }
            /// Allocates a workspace for `dimension` variables.
            ///
            /// # Errors
            ///
            /// Returns `InvalidInput`, `SizeOverflow` or `AllocationFailure`.
            pub fn workspace<S: NumericScalar<Real = R>>(
                &self,
                dimension: usize,
            ) -> Result<StochasticWorkspace<S>, SciOxideError> {
                let mut workspace = StochasticWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements::<S>(dimension)?)?;
                Ok(workspace)
            }
            fn identity<S, P, Sm>(
                &self,
                problem: &StochasticProblem<'_, P, Sm>,
                dimension: usize,
            ) -> scioxide_core::SolveIdentity
            where
                S: NumericScalar<Real = R>,
                P: SampledGradient<S> + ?Sized,
                Sm: OptimizationSampler + ?Sized,
            {
                // Results are independent of the worker count and the time limit, so
                // a checkpoint resumes under any execution policy or deadline.
                let mut options = self.options;
                options.common.execution = ExecutionPolicy::Sequential;
                options.common.time_limit = None;
                crate::context::local_identity(
                    scioxide_core::algorithm_id!("optimize", stringify!($name), 1),
                    dimension,
                    &(
                        options,
                        problem.sampler.configuration_token(),
                        problem.callback.noise_model(),
                    ),
                )
            }
            /// Minimizes from `x0` with sequential execution and returns a new
            /// result.
            ///
            /// # Errors
            ///
            /// As [`Self::minimize_into`].
            pub fn minimize<S, P, Sm>(
                &self,
                problem: &StochasticProblem<'_, P, Sm>,
                x0: &[S],
            ) -> Result<StochasticResult<S>, SciOxideError>
            where
                S: NumericScalar<Real = R>,
                P: SampledGradient<S> + ?Sized,
                Sm: OptimizationSampler + ?Sized,
            {
                let mut workspace = self.workspace(x0.len())?;
                let mut output = StochasticResult::empty();
                self.minimize_into(
                    problem,
                    x0,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )?;
                Ok(output)
            }
            /// Starts a new run from `x0` and writes the result to `output`.
            ///
            /// A limit, cancellation or deadline after the first objective
            /// estimate ends the run with an unsuccessful result, and
            /// [`StochasticWorkspace::checkpoint`] can then save it for
            /// [`Self::resume_into`]. A stop before that estimate is an error.
            /// On error, `output` is unchanged.
            ///
            /// # Errors
            ///
            /// See the type-level `# Errors` section.
            pub fn minimize_into<'scope, 'capability, S, P, Sm>(
                &self,
                problem: &StochasticProblem<'_, P, Sm>,
                x0: &[S],
                output: &mut StochasticResult<S>,
                workspace: &mut StochasticWorkspace<S>,
                scope: impl Into<SolveScope<'scope, 'capability>>,
            ) -> Result<(), SciOxideError>
            where
                S: NumericScalar<Real = R>,
                P: SampledGradient<S> + ?Sized,
                Sm: OptimizationSampler + ?Sized,
            {
                workspace.shell.reject(AttemptKind::Call);
                workspace.checkpoint_metadata = None;
                let scope = scope.into();
                validate(problem.callback, x0)?;
                workspace.try_reserve(&self.workspace_requirements::<S>(x0.len())?)?;
                preflight(&scope, self.options.common.time_limit)?;
                let entry = Entry::Call {
                    identity: self.identity(problem, x0.len()),
                    budgets: self.budgets(),
                };
                let radius = match Method::$variant(self.options) {
                    Method::Trust(o) => o.initial_radius,
                    _ => R::ZERO,
                };
                let metadata = &mut workspace.checkpoint_metadata;
                crate::attempt::run(
                    &mut workspace.shell,
                    &mut workspace.storage,
                    &mut workspace.staged,
                    output,
                    scope,
                    entry,
                    |context, storage, staged| {
                        let w = storage.as_mut().ok_or_else(missing_storage)?;
                        let s = &mut w.state;
                        s.x.copy_from_slice(x0);
                        s.value = None;
                        s.value_error = None;
                        s.value_variance = None;
                        s.value_samples = 0;
                        s.gradient_current = false;
                        s.gradient_error_available = false;
                        s.gradient_samples = 0;
                        s.iterations = 0;
                        s.cursor = 0;
                        s.radius = radius;
                        epoch(
                            problem,
                            Method::$variant(self.options),
                            w,
                            staged,
                            metadata,
                            false,
                            context,
                        )
                    },
                    progress,
                )
            }
            /// Continues a run from `checkpoint` and writes the result to `output`.
            ///
            /// The iteration limit counts the iterations before the checkpoint.
            /// Samples of an interrupted batch are discarded, and sampling
            /// continues after the last sample used, so no sample is reused. The
            /// checkpoint is not changed and can be resumed again.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if the checkpoint comes from another method,
            ///   options, sampler, noise model or problem dimension. The
            ///   checkpoint, `output` and the parent context are then unchanged.
            /// - The errors of [`Self::minimize_into`].
            pub fn resume_into<'scope, 'capability, S, P, Sm>(
                &self,
                problem: &StochasticProblem<'_, P, Sm>,
                checkpoint: &StochasticCheckpoint<S>,
                output: &mut StochasticResult<S>,
                workspace: &mut StochasticWorkspace<S>,
                scope: impl Into<SolveScope<'scope, 'capability>>,
            ) -> Result<(), SciOxideError>
            where
                S: NumericScalar<Real = R>,
                P: SampledGradient<S> + ?Sized,
                Sm: OptimizationSampler + ?Sized,
            {
                workspace.shell.reject(AttemptKind::Resume);
                workspace.checkpoint_metadata = None;
                let scope = scope.into();
                let dimension = checkpoint.state.x.len();
                validate(problem.callback, &checkpoint.state.x)?;
                workspace.try_reserve(&self.workspace_requirements::<S>(dimension)?)?;
                preflight(&scope, self.options.common.time_limit)?;
                let entry = Entry::Resume {
                    identity: self.identity(problem, dimension),
                    state_token: checkpoint.state.state_token(),
                    metadata: checkpoint.metadata,
                    budgets: self.budgets(),
                };
                let metadata = &mut workspace.checkpoint_metadata;
                crate::attempt::run(
                    &mut workspace.shell,
                    &mut workspace.storage,
                    &mut workspace.staged,
                    output,
                    scope,
                    entry,
                    |context, storage, staged| {
                        let w = storage.as_mut().ok_or_else(missing_storage)?;
                        w.state.assign_from(&checkpoint.state);
                        epoch(
                            problem,
                            Method::$variant(self.options),
                            w,
                            staged,
                            metadata,
                            true,
                            context,
                        )
                    },
                    progress,
                )
            }
            fn budgets(&self) -> SolveBudgets {
                SolveBudgets::unbounded().with_limit(
                    scioxide_core::BudgetKind::Iteration,
                    self.options.common.iterations,
                )
            }
        }
    };
}
solver!(StochasticGradientDescent, SgdOptions, Sgd);
solver!(StochasticTrustRegion, StochasticTrustOptions, Trust);
fn missing_storage() -> SciOxideError {
    invalid(
        scioxide_core::argument!("workspace"),
        "stochastic workspace was not reserved",
    )
}
fn validate<S: NumericScalar>(
    problem: &(impl SampledGradient<S> + ?Sized),
    x: &[S],
) -> Result<(), SciOxideError>
where
    S::Real: NumericScalar<Real = S::Real>,
{
    if x.is_empty() || problem.dimension() != x.len() || problem.objectives() != 1 {
        return Err(invalid(
            scioxide_core::argument!("problem"),
            "stochastic problem shape",
        ));
    }
    finite_vector(x)
}
fn progress<S: NumericScalar>(
    storage: &Option<Storage<S>>,
    succeeded: bool,
) -> Option<StochasticProgress<S::Real>>
where
    S::Real: NumericScalar<Real = S::Real>,
{
    let state = &storage.as_ref()?.state;
    Some(StochasticProgress {
        iterations: state.iterations,
        value: state.value?,
        trust: crate::attempt::trust(succeeded),
    })
}
/// Run one epoch from the prepared state, record checkpoint metadata of the
/// accepted state and stage the result.
fn epoch<S, P, Sm>(
    problem: &StochasticProblem<'_, P, Sm>,
    method: Method<S::Real>,
    w: &mut Storage<S>,
    staged: &mut StochasticResult<S>,
    metadata: &mut Option<SolveCheckpointMetadata>,
    resumed: bool,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: NumericScalar,
    S::Real: NumericScalar<Real = S::Real>,
    P: SampledGradient<S> + ?Sized,
    Sm: OptimizationSampler + ?Sized,
{
    if let Some(limit) = method.common().time_limit {
        context.limit_time(limit)?;
    }
    let result = run(
        problem.callback,
        problem.sampler,
        method,
        w,
        method.common().execution,
        context,
    );
    if w.state.value.is_some() {
        let snapshot = context.checkpoint(w.state.state_token());
        *metadata = if resumed {
            snapshot.ok()
        } else {
            Some(snapshot?)
        };
    }
    let termination = match result {
        Ok(t) | Err(FlowError::Stop(t)) => t,
        Err(FlowError::Failure(e)) => return Err(e),
    };
    if w.state.value.is_none() {
        return Err(initial_stop(termination, context));
    }
    w.state.termination = termination;
    w.state.counts = context.counts();
    staged.publish(&w.state)
}
fn publish_value<S: NumericScalar>(state: &mut StochasticResult<S>, batch: &Batch<S::Real>)
where
    S::Real: NumericScalar<Real = S::Real>,
{
    state.value = Some(batch.mean[0]);
    state.value_samples = batch.count;
    state.value_error = batch.error_available.then_some(batch.error[0]);
    state.value_variance = batch.error_available.then_some(batch.variance[0]);
}
fn run<S: NumericScalar>(
    problem: &(impl SampledGradient<S> + ?Sized),
    sampler: &(impl OptimizationSampler + ?Sized),
    method: Method<S::Real>,
    w: &mut Storage<S>,
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
) -> Flow<OptimizeTermination>
where
    S::Real: NumericScalar<Real = S::Real>,
{
    let options = method.common();
    let replication = options.replication;
    let noise = problem.noise_model();
    if matches!(method, Method::Trust(_))
        && noise == NoiseModel::Independent
        && replication.minimum < 2
    {
        return Err(invalid(
            scioxide_core::argument!("self"),
            "noisy trust regions require at least two replicates",
        )
        .into());
    }
    let s = &mut w.state;
    if s.value.is_none() {
        w.base.reset();
        admitted(w.base.extend(
            replication.minimum,
            noise,
            options.seed,
            1,
            &mut s.cursor,
            sampler,
            EvaluationRole::Objective,
            policy,
            context,
            &|token, out| problem.sample_into(&s.x, token, out),
        )?)?;
        publish_value(s, &w.base);
        context.commit_accepted(PartialResultTrust::Admissible)?;
    }
    loop {
        admitted(stop(context)?)?;
        if s.iterations < options.iterations.get()
            && !context
                .can_consume(scioxide_core::BudgetKind::Iteration, 1)?
                .is_granted()
        {
            return Ok(OptimizeTermination::IterationLimit);
        }
        // Assess the final allowed iterate before reporting the iteration cap.
        w.gradient.reset();
        let mut count = replication.minimum;
        let gradient_norm;
        loop {
            admitted(w.gradient.extend(
                count,
                noise,
                options.seed,
                2,
                &mut s.cursor,
                sampler,
                EvaluationRole::Gradient,
                policy,
                context,
                &|token, out| problem.gradient_sample_into(&s.x, token, out),
            )?)?;
            let norm = crate::proximal::norm(&w.gradient.mean)?;
            let error = crate::proximal::norm(&w.gradient.error)?;
            let threshold = options
                .tolerance
                .threshold(<_ as scioxide_core::AlgebraicScalar>::ONE)?;
            s.gradient.copy_from_slice(&w.gradient.mean);
            s.gradient_errors.copy_from_slice(&w.gradient.error);
            s.gradient_current = true;
            s.gradient_error_available = w.gradient.error_available;
            s.gradient_samples = count;
            if w.gradient.error_available && finite(norm + replication.margin * error)? <= threshold
            {
                return Ok(if noise == NoiseModel::Deterministic {
                    OptimizeTermination::GradientSmall
                } else {
                    OptimizeTermination::EstimatedStationarity
                });
            }
            let sufficient = match method {
                Method::Sgd(_) => true,
                Method::Trust(o) => {
                    w.gradient.error_available
                        && finite(replication.margin * error)?
                            <= finite(o.gradient_accuracy * s.radius)?
                }
            };
            if sufficient {
                gradient_norm = norm;
                break;
            }
            if count == replication.maximum {
                return Ok(OptimizeTermination::NoiseLimit);
            }
            count = count.saturating_mul(2).min(replication.maximum);
        }
        if s.iterations >= options.iterations.get() {
            return Ok(OptimizeTermination::IterationLimit);
        }
        if !context.record_proposal()?.is_granted() {
            return Ok(OptimizeTermination::ProposalLimit);
        }
        let next = s.iterations.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("state.iterations"))
        })?;
        let scale = match method {
            Method::Sgd(o) => o.step.at(s.iterations)?,
            Method::Trust(_) => {
                if gradient_norm == <_ as scioxide_core::AlgebraicScalar>::ZERO {
                    return Ok(OptimizeTermination::NoiseLimit);
                }
                finite(s.radius / gradient_norm)?
            }
        };
        if scale <= <_ as scioxide_core::AlgebraicScalar>::ZERO {
            return Ok(OptimizeTermination::StepSmall);
        }
        for ((dest, &x), &g) in w.trial.iter_mut().zip(&s.x).zip(&w.gradient.mean) {
            *dest = S::from_components(
                scioxide_core::checked_mul_add(-scale, g.real_component(), x.real_component())?,
                scioxide_core::checked_mul_add(
                    -scale,
                    g.imaginary_component(),
                    x.imaginary_component(),
                )?,
            );
        }
        finite_vector(&w.trial)?;
        if w.trial == s.x {
            return Ok(OptimizeTermination::StepSmall);
        }
        w.candidate.reset();
        let accepted = match method {
            Method::Sgd(_) => {
                admitted(w.candidate.extend(
                    replication.minimum,
                    noise,
                    options.seed,
                    1,
                    &mut s.cursor,
                    sampler,
                    EvaluationRole::Objective,
                    policy,
                    context,
                    &|token, out| problem.sample_into(&w.trial, token, out),
                )?)?;
                true
            }
            Method::Trust(o) => {
                if gradient_norm < finite(o.gradient_radius * s.radius)? {
                    false
                } else {
                    let predicted = scioxide_core::checked_float_product(s.radius, gradient_norm)?;
                    if predicted <= <_ as scioxide_core::AlgebraicScalar>::ZERO {
                        return Ok(OptimizeTermination::StepSmall);
                    }
                    w.base.reset();
                    let mut count = replication.minimum;
                    loop {
                        admitted(w.base.extend(
                            count,
                            noise,
                            options.seed,
                            3,
                            &mut s.cursor,
                            sampler,
                            EvaluationRole::Objective,
                            policy,
                            context,
                            &|token, out| problem.sample_into(&s.x, token, out),
                        )?)?;
                        admitted(w.candidate.extend(
                            count,
                            noise,
                            options.seed,
                            4,
                            &mut s.cursor,
                            sampler,
                            EvaluationRole::Objective,
                            policy,
                            context,
                            &|token, out| problem.sample_into(&w.trial, token, out),
                        )?)?;
                        let reduction =
                            scioxide_core::checked_difference(w.base.mean[0], w.candidate.mean[0])?;
                        let error =
                            crate::proximal::norm(&[w.base.error[0], w.candidate.error[0]])?;
                        let margin = finite(replication.margin * error)?;
                        let target = finite(o.acceptance * predicted)?;
                        if finite(reduction - margin)? >= target {
                            break true;
                        }
                        if finite(reduction + margin)? < target {
                            break false;
                        }
                        if count == replication.maximum {
                            return Ok(OptimizeTermination::NoiseLimit);
                        }
                        count = count.saturating_mul(2).min(replication.maximum);
                    }
                }
            }
        };
        // The update is charged with its decision, so an update discarded by a
        // stop does not consume the total retained by checkpoints.
        if !context.record_iteration()?.is_granted() {
            return Ok(OptimizeTermination::IterationLimit);
        }
        if accepted {
            s.x.copy_from_slice(&w.trial);
            publish_value(s, &w.candidate);
            s.gradient_current = false;
            context.commit_accepted(PartialResultTrust::Admissible)?;
        } else {
            context.record_rejected()?;
        }
        if let Method::Trust(o) = method {
            s.radius = if accepted {
                if s.radius >= o.maximum_radius / o.growth {
                    o.maximum_radius
                } else {
                    finite(s.radius * o.growth)?
                }
            } else {
                finite(s.radius / o.growth)?
            };
            if s.radius < o.minimum_radius {
                s.iterations = next;
                return Ok(OptimizeTermination::StepSmall);
            }
        }
        s.iterations = next;
    }
}
