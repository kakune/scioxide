//! Surrogate-model search for expensive objectives over a finite box.
use alloc::vec::Vec;
use core::cell::RefCell;

use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    AttemptKind, AttemptRecord, Bounds, Budget, EvaluationRole, NumericScalar, RealScalar,
    SciOxideError, SolveBudgets, SolveContext, SolveScope, SolveState,
};
use scioxide_interp::{
    Covariance, ExtrapolationPolicy, GaussianProcessModel, Interpolant, KrigingMean,
    LocalModelOptions, LocalModelPlan, LocalPredictionWorkspace, LocalRbfModel, LocalRbfOptions,
    LocalSamples, RbfKernel, RbfRegularization,
};

use super::{
    GlobalSearchOptions, GlobalSearchResult, GlobalSearchWorkspace, LocalSolverFactory, MultiStart,
    StartDesign,
};
use crate::attempt::{Entry, Shell};
use crate::validation::invalid;
use crate::{BoundedObjective, FiniteBounds, ObjectiveFunction, OptimizeTermination};

/// Stores one evaluation of an expensive objective: the point, the value and
/// the variance of the observation noise.
#[derive(Debug, Clone, Copy)]
pub struct SurrogateObservation<S: RealScalar, const D: usize> {
    point: [S; D],
    value: S,
    noise_variance: S,
}
impl<S: RealScalar, const D: usize> SurrogateObservation<S, D> {
    /// Creates an observation of `value` at `point` with noise variance
    /// `noise_variance` (zero for exact values).
    ///
    /// [`SurrogateHistory::push`] checks the observation.
    pub const fn new(point: [S; D], value: S, noise_variance: S) -> Self {
        Self {
            point,
            value,
            noise_variance,
        }
    }
    /// Returns the point.
    pub const fn point(&self) -> [S; D] {
        self.point
    }
    /// Returns the observed value.
    pub const fn value(&self) -> S {
        self.value
    }
    /// Returns the noise variance.
    pub const fn noise_variance(&self) -> S {
        self.noise_variance
    }
}

/// Holds the problem of a [`SurrogateSearch`]: a finite box and the
/// observations of the objective in it, up to a fixed capacity.
///
/// Observations keep the order in which they were added; the search appends
/// new ones.
#[derive(Debug)]
pub struct SurrogateHistory<S: RealScalar, const D: usize> {
    bounds: FiniteBounds<S>,
    capacity: usize,
    observations: Vec<SurrogateObservation<S, D>>,
}
impl<S: RealScalar, const D: usize> SurrogateHistory<S, D> {
    /// Creates an empty history over `bounds` that holds at most `capacity`
    /// observations.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D` or `capacity` is zero, or `bounds` does not have
    ///   `D` variables.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn new(bounds: FiniteBounds<S>, capacity: usize) -> Result<Self, SciOxideError> {
        if D == 0 || bounds.len() != D || capacity == 0 {
            return Err(invalid(
                scioxide_core::argument!("capacity"),
                "positive matching dimension and history capacity required",
            ));
        }
        Ok(Self {
            bounds,
            capacity,
            observations: try_with_capacity(
                capacity,
                scioxide_core::quantity!("surrogate_history"),
            )?,
        })
    }
    /// Appends an observation. A point may be observed more than once, as for
    /// noisy data.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if the history is full, the value or noise variance is
    /// NaN or infinite, the noise variance is negative, or the point is not
    /// finite and inside the box.
    pub fn push(&mut self, observation: SurrogateObservation<S, D>) -> Result<(), SciOxideError> {
        if self.observations.len() == self.capacity
            || !observation.value.is_finite()
            || !observation.noise_variance.is_finite()
            || observation.noise_variance < S::ZERO
            || !self.bounds.contains(&observation.point)
        {
            return Err(invalid(
                scioxide_core::argument!("observation"),
                "invalid or over-capacity surrogate observation",
            ));
        }
        self.observations.push(observation);
        Ok(())
    }
    /// Returns the observations in the order they were added.
    pub fn observations(&self) -> &[SurrogateObservation<S, D>] {
        &self.observations
    }
    /// Returns the box.
    pub fn bounds(&self) -> &FiniteBounds<S> {
        &self.bounds
    }
    /// Returns the maximum number of observations.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Prediction of a surrogate model at one point: the mean and, for a Gaussian
/// process, the variance of the latent function.
#[derive(Debug, Clone, Copy)]
pub struct SurrogatePrediction<S: RealScalar> {
    mean: S,
    variance: Option<S>,
}
impl<S: RealScalar> SurrogatePrediction<S> {
    /// Creates a prediction from a mean and an optional variance.
    pub const fn new(mean: S, variance: Option<S>) -> Self {
        Self { mean, variance }
    }
    /// Returns the predicted mean.
    pub const fn mean(&self) -> S {
        self.mean
    }
    /// Returns the predicted variance of the latent function (without the
    /// observation noise); `None` for models without uncertainty, such as RBF
    /// interpolants.
    pub const fn variance(&self) -> Option<S> {
        self.variance
    }
}

fn checked_prediction<S: RealScalar>(
    p: SurrogatePrediction<S>,
) -> Result<SurrogatePrediction<S>, SciOxideError> {
    if !p.mean.is_finite() || p.variance.is_some_and(|v| !v.is_finite() || v < S::ZERO) {
        return Err(invalid(
            scioxide_core::argument!("self"),
            "surrogate provider returned an invalid prediction",
        ));
    }
    Ok(p)
}

fn checked_score<S: RealScalar>(score: S) -> Result<S, SciOxideError> {
    if score.is_finite() {
        Ok(score)
    } else {
        Err(invalid(
            scioxide_core::argument!("self"),
            "acquisition provider returned a non-finite score",
        ))
    }
}

/// Predicts objective values from a fitted surrogate model.
pub trait SurrogateModel<S: RealScalar, const D: usize> {
    /// Scratch storage for predictions.
    type Workspace;
    /// Allocates scratch storage for [`Self::predict`].
    ///
    /// # Errors
    ///
    /// `SizeOverflow` or `AllocationFailure`.
    fn workspace(&self) -> Result<Self::Workspace, SciOxideError>;
    /// Predicts the objective at `point`.
    ///
    /// # Errors
    ///
    /// Returns the model's domain or numerical errors.
    fn predict(
        &self,
        point: [S; D],
        work: &mut Self::Workspace,
    ) -> Result<SurrogatePrediction<S>, SciOxideError>;
}

/// Fits a surrogate model to the observations of a [`SurrogateHistory`].
///
/// The fitter's `Debug` output identifies its configuration.
pub trait SurrogateFitter<S: RealScalar, const D: usize>: core::fmt::Debug {
    /// Fitted model.
    type Model: SurrogateModel<S, D>;
    /// Fits a model to `history`. The work counts toward the limits of
    /// `context`.
    ///
    /// # Errors
    ///
    /// Returns the fit's errors unchanged, such as `Singular`, an
    /// ill-conditioned system or an exhausted limit.
    fn fit(
        &self,
        history: &SurrogateHistory<S, D>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self::Model, SciOxideError>;
}

impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> SurrogateModel<S, D>
    for GaussianProcessModel<S, D>
{
    type Workspace = LocalPredictionWorkspace<S>;
    fn workspace(&self) -> Result<Self::Workspace, SciOxideError> {
        GaussianProcessModel::workspace(self)
    }
    fn predict(
        &self,
        point: [S; D],
        work: &mut Self::Workspace,
    ) -> Result<SurrogatePrediction<S>, SciOxideError> {
        // Components: mean, latent variance, observation variance.
        let mut prediction = [S::ZERO; 3];
        self.prediction(S::ZERO)?.evaluate_into(
            point,
            ExtrapolationPolicy::MethodDefined,
            &mut prediction,
            work,
        )?;
        Ok(SurrogatePrediction {
            mean: prediction[0],
            variance: Some(prediction[1]),
        })
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> SurrogateModel<S, D>
    for LocalRbfModel<S, D>
{
    type Workspace = LocalPredictionWorkspace<S>;
    fn workspace(&self) -> Result<Self::Workspace, SciOxideError> {
        Interpolant::workspace(self)
    }
    fn predict(
        &self,
        point: [S; D],
        work: &mut Self::Workspace,
    ) -> Result<SurrogatePrediction<S>, SciOxideError> {
        let mut mean = [S::ZERO];
        Interpolant::evaluate_into(
            self,
            point,
            ExtrapolationPolicy::MethodDefined,
            &mut mean,
            work,
        )?;
        Ok(SurrogatePrediction {
            mean: mean[0],
            variance: None,
        })
    }
}

/// Fits an ordinary kriging model (a Gaussian process with constant unknown
/// mean) with fixed covariance hyperparameters and the noise variance of each
/// observation.
#[derive(Debug, Clone, Copy)]
pub struct KrigingSurrogateFitter<S: RealScalar, const D: usize> {
    covariance: Covariance<S, D>,
    numerical: LocalModelOptions<S>,
}
impl<S: RealScalar, const D: usize> KrigingSurrogateFitter<S, D> {
    /// Creates a fitter with covariance function `covariance` and the limits
    /// of the dense fit in `numerical` (order, memory, rank and condition
    /// number). The hyperparameters are not optimized.
    pub const fn new(covariance: Covariance<S, D>, numerical: LocalModelOptions<S>) -> Self {
        Self {
            covariance,
            numerical,
        }
    }
    /// Returns the covariance function.
    pub const fn covariance(&self) -> &Covariance<S, D> {
        &self.covariance
    }
    /// Returns the limits of the dense fit.
    pub const fn numerical(&self) -> &LocalModelOptions<S> {
        &self.numerical
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> SurrogateFitter<S, D>
    for KrigingSurrogateFitter<S, D>
{
    type Model = GaussianProcessModel<S, D>;
    fn fit(
        &self,
        history: &SurrogateHistory<S, D>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self::Model, SciOxideError> {
        let mut points = try_with_capacity(
            history.observations.len(),
            scioxide_core::quantity!("kriging_coordinates"),
        )?;
        let mut values = try_with_capacity(
            history.observations.len(),
            scioxide_core::quantity!("kriging_values"),
        )?;
        let mut noise = try_with_capacity(
            history.observations.len(),
            scioxide_core::quantity!("kriging_noise"),
        )?;
        for observation in &history.observations {
            points.push(observation.point);
            values.push(observation.value);
            noise.push(observation.noise_variance);
        }
        let samples = LocalSamples::new(&points, &values)?;
        GaussianProcessModel::build(
            &samples.points()?,
            samples.values()?,
            self.covariance,
            &noise,
            KrigingMean::Ordinary,
            self.numerical,
            context,
        )
    }
}

/// Fits a radial basis function (RBF) interpolant to all observations.
///
/// The interpolant has no predicted variance, so it works with acquisitions
/// that use the mean only.
#[derive(Debug, Clone, Copy)]
pub struct RbfSurrogateFitter<S: RealScalar> {
    kernel: RbfKernel<S>,
    regularization: RbfRegularization<S>,
    numerical: LocalModelOptions<S>,
}
impl<S: RealScalar> RbfSurrogateFitter<S> {
    /// Creates a fitter with kernel `kernel`, diagonal smoothing
    /// `regularization` and the limits of the dense fit in `numerical`. No
    /// smoothing is added automatically.
    pub const fn new(
        kernel: RbfKernel<S>,
        regularization: RbfRegularization<S>,
        numerical: LocalModelOptions<S>,
    ) -> Self {
        Self {
            kernel,
            regularization,
            numerical,
        }
    }
    /// Returns the kernel.
    pub const fn kernel(&self) -> &RbfKernel<S> {
        &self.kernel
    }
    /// Returns the diagonal smoothing.
    pub const fn regularization(&self) -> &RbfRegularization<S> {
        &self.regularization
    }
    /// Returns the limits of the dense fit.
    pub const fn numerical(&self) -> &LocalModelOptions<S> {
        &self.numerical
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, const D: usize> SurrogateFitter<S, D>
    for RbfSurrogateFitter<S>
{
    type Model = LocalRbfModel<S, D>;
    fn fit(
        &self,
        history: &SurrogateHistory<S, D>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self::Model, SciOxideError> {
        let mut points = try_with_capacity(
            history.observations.len(),
            scioxide_core::quantity!("rbf_coordinates"),
        )?;
        let mut values = try_with_capacity(
            history.observations.len(),
            scioxide_core::quantity!("rbf_values"),
        )?;
        for observation in &history.observations {
            points.push(observation.point);
            values.push(observation.value);
        }
        let samples = LocalSamples::new(&points, &values)?;
        let point_set = samples.points()?;
        let plan = LocalModelPlan::<S, D>::new(&point_set, samples.values()?)?;
        LocalRbfModel::build(
            &plan,
            points[0],
            LocalRbfOptions::new(points.len(), self.kernel)?
                .with_regularization(self.regularization)
                .with_numerical(self.numerical),
            context,
        )
    }
}

/// Scores a candidate point from its prediction; the search proposes the
/// point with the highest score.
///
/// The score ranks candidates; it is not an error bound. The acquisition's
/// `Debug` output identifies its configuration.
pub trait SurrogateAcquisition<S: RealScalar>: core::fmt::Debug {
    /// Scores `prediction` against `incumbent`, the lowest predicted mean at
    /// the observed points.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid prediction or a score that overflows.
    fn score(&self, prediction: SurrogatePrediction<S>, incumbent: S) -> Result<S, SciOxideError>;
}

/// Acquisitions for minimization based on a Gaussian prediction with mean
/// `μ` and standard deviation `σ`.
///
/// With incumbent `f*` (the lowest predicted mean at the observed points),
/// margin `ξ ≥ 0`, `I = f* − μ − ξ` and `z = I/σ`, and with `Φ` and `φ` the
/// standard normal distribution and density:
///
/// ```text
/// expected improvement       EI  = I Φ(z) + σ φ(z)
/// probability of improvement PI  = Φ(z)
/// lower confidence bound     LCB = κ σ − μ
/// ```
///
/// For `σ = 0`, `EI = max(I, 0)` and `PI` is 1 if `I > 0` and 0 otherwise.
/// All three need a predicted variance; scoring a prediction without one is
/// an `InvalidInput` error, as are a negative `ξ` or `κ`.
///
/// # References
///
/// - D. R. Jones, M. Schonlau and W. J. Welch, "Efficient global optimization
///   of expensive black-box functions", *J. Global Optim.* 13(4), 455–492,
///   1998.
#[derive(Debug, Clone, Copy)]
pub enum GaussianAcquisition<S: RealScalar> {
    /// Expected improvement `EI`.
    ExpectedImprovement {
        /// Margin `ξ ≥ 0` in objective units.
        margin: S,
    },
    /// Probability of improvement `PI`.
    ProbabilityOfImprovement {
        /// Margin `ξ ≥ 0` in objective units.
        margin: S,
    },
    /// Negated lower confidence bound `κ σ − μ`.
    LowerConfidenceBound {
        /// Weight `κ ≥ 0` of the standard deviation.
        weight: S,
    },
}
impl<S: RealScalar> SurrogateAcquisition<S> for GaussianAcquisition<S> {
    fn score(&self, p: SurrogatePrediction<S>, incumbent: S) -> Result<S, SciOxideError> {
        let variance = p.variance.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("p"),
                "Gaussian acquisition requires predictive variance",
            )
        })?;
        let parameter = match *self {
            Self::ExpectedImprovement { margin } | Self::ProbabilityOfImprovement { margin } => {
                margin
            }
            Self::LowerConfidenceBound { weight } => weight,
        };
        if !p.mean.is_finite()
            || !incumbent.is_finite()
            || !variance.is_finite()
            || variance < S::ZERO
            || !parameter.is_finite()
            || parameter < S::ZERO
        {
            return Err(invalid(
                scioxide_core::argument!("p"),
                "invalid acquisition mean, variance or parameter",
            ));
        }
        let sigma = variance.sqrt();
        let value = match *self {
            Self::LowerConfidenceBound { weight } => weight * sigma - p.mean,
            Self::ExpectedImprovement { margin } | Self::ProbabilityOfImprovement { margin } => {
                let improvement = incumbent - p.mean - margin;
                if !improvement.is_finite() {
                    return Err(invalid(
                        scioxide_core::argument!("incumbent"),
                        "improvement exceeds scalar range",
                    ));
                }
                if sigma == S::ZERO {
                    if matches!(self, Self::ExpectedImprovement { .. }) {
                        improvement.max(S::ZERO)
                    } else if improvement > S::ZERO {
                        S::ONE
                    } else {
                        S::ZERO
                    }
                } else {
                    let z = improvement / sigma;
                    if z >= S::TEN {
                        return Ok(if matches!(self, Self::ProbabilityOfImprovement { .. }) {
                            S::ONE
                        } else {
                            improvement
                        });
                    }
                    if z <= -S::TEN {
                        return Ok(S::ZERO);
                    }
                    let cdf = scioxide_special::normal_cdf(z)?;
                    if matches!(self, Self::ProbabilityOfImprovement { .. }) {
                        cdf
                    } else {
                        let two_pi = S::TAU;
                        let pdf = (-z * z / S::TWO).exp() / two_pi.sqrt();
                        (improvement * cdf + sigma * pdf).max(S::ZERO)
                    }
                }
            }
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(invalid(
                scioxide_core::argument!("p"),
                "acquisition score is non-finite",
            ))
        }
    }
}

/// Decides whether a point is feasible, for constraints known in advance.
///
/// The search proposes only feasible points. The answer must depend only on
/// the point, so that the search is reproducible. Closures
/// `Fn(&[S]) -> Result<bool, SciOxideError>` implement this trait.
pub trait SurrogateFeasibility<S: RealScalar> {
    /// Returns whether `point` is feasible.
    ///
    /// # Errors
    ///
    /// Returns the callback's errors.
    fn feasible(&self, point: &[S]) -> Result<bool, SciOxideError>;
}
impl<S: RealScalar, F: Fn(&[S]) -> Result<bool, SciOxideError>> SurrogateFeasibility<S> for F {
    fn feasible(&self, point: &[S]) -> Result<bool, SciOxideError> {
        self(point)
    }
}

/// Configures [`SurrogateSearch`]: the starts of the acquisition search,
/// the seed, the minimum separation of proposals, and the number and size of
/// batches.
///
/// The setters do not check their arguments; [`SurrogateSearch::new`]
/// checks the separation.
#[derive(Debug, Clone, Copy)]
pub struct SurrogateSearchOptions<S: RealScalar> {
    acquisition_starts: Budget,
    seed: u64,
    separation: S,
    batches: usize,
    batch_size: Budget,
}
impl<S: RealScalar> Default for SurrogateSearchOptions<S> {
    /// Returns ten acquisition starts, seed 42, separation zero and one batch
    /// of one point.
    fn default() -> Self {
        Self {
            acquisition_starts: Budget::ONE.saturating_add(9),
            seed: 42,
            separation: S::ZERO,
            batches: 1,
            batch_size: Budget::ONE,
        }
    }
}
impl<S: RealScalar> SurrogateSearchOptions<S> {
    /// Sets the number of multistart starts used to maximize the acquisition
    /// for each proposed point.
    pub const fn with_acquisition_starts(mut self, starts: Budget) -> Self {
        self.acquisition_starts = starts;
        self
    }
    /// Sets the seed of the acquisition starts.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Sets the minimum Euclidean distance, in box units, between a proposal
    /// and the observed and other proposed points. It must be finite and
    /// nonnegative; zero excludes only exact duplicates.
    pub const fn with_separation(mut self, separation: S) -> Self {
        self.separation = separation;
        self
    }
    /// Sets the number of batches. The search also stops when the history is
    /// full.
    pub const fn with_batches(mut self, batches: usize) -> Self {
        self.batches = batches;
        self
    }
    /// Sets the number of points proposed and evaluated per batch.
    pub const fn with_batch_size(mut self, size: Budget) -> Self {
        self.batch_size = size;
        self
    }
    /// Returns the number of acquisition starts per proposed point.
    pub const fn acquisition_starts(&self) -> Budget {
        self.acquisition_starts
    }
    /// Returns the seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    /// Returns the minimum separation in box units.
    pub const fn separation(&self) -> S {
        self.separation
    }
    /// Returns the number of batches.
    pub const fn batches(&self) -> usize {
        self.batches
    }
    /// Returns the number of points per batch.
    pub const fn batch_size(&self) -> Budget {
        self.batch_size
    }
}

/// Summarizes a [`SurrogateSearch`] search.
#[derive(Debug, Clone, Copy)]
pub struct SurrogateSearchReport {
    termination: OptimizeTermination,
    objective_evaluations: usize,
    completed_batches: usize,
    best_observation: usize,
}
impl SurrogateSearchReport {
    /// Returns [`OptimizeTermination::SearchBudgetCompleted`]: the search ran
    /// all batches or filled the history. It is not a success.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    /// Returns the number of calls to the expensive objective in this search.
    pub const fn objective_evaluations(&self) -> usize {
        self.objective_evaluations
    }
    /// Returns the number of batches whose points were all evaluated.
    pub const fn completed_batches(&self) -> usize {
        self.completed_batches
    }
    /// Returns the index in the history of the lowest observed value.
    pub const fn best_observation(&self) -> usize {
        self.best_observation
    }
}

struct AcquisitionObjective<
    'a,
    S: RealScalar,
    const D: usize,
    M: SurrogateModel<S, D>,
    A: SurrogateAcquisition<S>,
> {
    model: &'a M,
    work: &'a RefCell<M::Workspace>,
    acquisition: &'a A,
    incumbent: S,
}
impl<
    S: RealScalar + NumericScalar<Real = S>,
    const D: usize,
    M: SurrogateModel<S, D>,
    A: SurrogateAcquisition<S>,
> ObjectiveFunction<S> for AcquisitionObjective<'_, S, D, M, A>
{
    fn dim(&self) -> usize {
        D
    }
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        let point: [S; D] = x.try_into().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                "acquisition dimension mismatch",
            )
        })?;
        let mut work = self.work.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                "reentrant acquisition prediction",
            )
        })?;
        let p = checked_prediction(self.model.predict(point, &mut work)?)?;
        checked_score(-checked_score(self.acquisition.score(p, self.incumbent)?)?)
    }
}

/// Workspace of [`SurrogateSearch`]: the storage of the acquisition
/// search, the proposals and the record of the last call.
pub struct SurrogateSearchWorkspace<S: RealScalar, const D: usize, L> {
    bounds: Option<Bounds<S>>,
    search: GlobalSearchWorkspace<S, L>,
    result: GlobalSearchResult<S>,
    proposals: Vec<[S; D]>,
    shell: Shell<usize>,
}
impl<S: RealScalar, const D: usize, L> core::fmt::Debug for SurrogateSearchWorkspace<S, D, L> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SurrogateSearchWorkspace")
            .field("bounds", &self.bounds)
            .field("search", &self.search)
            .finish_non_exhaustive()
    }
}
impl<S: RealScalar, const D: usize, L> SurrogateSearchWorkspace<S, D, L> {
    /// Returns the record of the most recent call; its progress counts the
    /// points proposed or added to the history.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<usize>> {
        self.shell.last_attempt()
    }
}

/// Minimizes an expensive objective over a finite box with a surrogate model
/// (Bayesian optimization when the model is a Gaussian process).
///
/// The [`SurrogateHistory`] is the problem: it holds the box and the
/// observations so far. For each batch, the search fits a model with a
/// [`SurrogateFitter`], then proposes each point of the batch by maximizing a
/// [`SurrogateAcquisition`] with an inner [`MultiStart`] over the box. A
/// proposal must be feasible for the [`SurrogateFeasibility`] check and at
/// least the separation away from the observed and earlier proposed points.
/// The search then evaluates the objective at the proposals and adds them to
/// the history. Each batch costs one fit, one acquisition search per point
/// and one objective evaluation per point.
///
/// The search does not prove that it found the global minimum.
///
/// # References
///
/// - D. R. Jones, M. Schonlau and W. J. Welch, "Efficient global optimization
///   of expensive black-box functions", *J. Global Optim.* 13(4), 455–492,
///   1998.
/// - H.-M. Gutmann, "A radial basis function method for global
///   optimization", *J. Global Optim.* 19(3), 201–227, 2001.
#[derive(Debug, Clone)]
pub struct SurrogateSearch<S: RealScalar, const D: usize, F, A, C, Design, Factory> {
    options: SurrogateSearchOptions<S>,
    fitter: F,
    acquisition: A,
    feasibility: C,
    search: MultiStart<S, Design, Factory>,
}
impl<S, const D: usize, F, A, C, Design, Factory> SurrogateSearch<S, D, F, A, C, Design, Factory>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: SurrogateFitter<S, D>,
    A: SurrogateAcquisition<S>,
    C: SurrogateFeasibility<S>,
    Design: StartDesign<S>,
    Factory: LocalSolverFactory<S>,
{
    /// Creates a search from options, a fitter, an acquisition, a feasibility
    /// check, and the start design and local solver factory of the
    /// acquisition search.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if `D` is zero or the separation is NaN, infinite or
    /// negative.
    pub fn new(
        options: SurrogateSearchOptions<S>,
        fitter: F,
        acquisition: A,
        feasibility: C,
        design: Design,
        factory: Factory,
    ) -> Result<Self, SciOxideError> {
        if D == 0 || !options.separation.is_finite() || options.separation < S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "positive dimension and finite nonnegative separation required",
            ));
        }
        let global = GlobalSearchOptions::default()
            .with_starts(options.acquisition_starts)
            .with_seed(options.seed);
        Ok(Self {
            search: MultiStart::new(global, design, factory)?,
            options,
            fitter,
            acquisition,
            feasibility,
        })
    }
    /// Returns the options.
    pub const fn options(&self) -> &SurrogateSearchOptions<S> {
        &self.options
    }
    /// Allocates a workspace for `D` variables.
    ///
    /// # Errors
    ///
    /// Returns the design's and the local factory's errors, `SizeOverflow` or
    /// `AllocationFailure`.
    pub fn workspace(
        &self,
    ) -> Result<SurrogateSearchWorkspace<S, D, Factory::Solver>, SciOxideError> {
        Ok(SurrogateSearchWorkspace {
            bounds: None,
            search: self.search.workspace(D)?,
            result: GlobalSearchResult::empty(),
            proposals: try_with_capacity(
                self.options.batch_size.get(),
                scioxide_core::quantity!("surrogate_proposals"),
            )?,
            shell: Shell::new(),
        })
    }
    /// Proposes `count` points without changing the history, using a new
    /// workspace and sequential execution.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace`] and [`Self::propose_batch_into`].
    pub fn propose_batch(
        &self,
        history: &SurrogateHistory<S, D>,
        count: usize,
    ) -> Result<Vec<[S; D]>, SciOxideError> {
        let mut workspace = self.workspace()?;
        let mut proposals = try_filled(
            count,
            [S::ZERO; D],
            scioxide_core::quantity!("surrogate_proposals"),
        )?;
        self.propose_batch_into(
            history,
            &mut proposals,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(proposals)
    }
    /// Fits one model to `history` and writes `output.len()` proposals to
    /// `output`, without evaluating the objective.
    ///
    /// The proposals are feasible and separated from the history and from
    /// each other. The same inputs give the same proposals.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the history is empty, `output` is empty or larger
    ///   than the free capacity of the history, an observed point is
    ///   infeasible, the acquisition needs a variance that the model does not
    ///   predict, or no feasible and separated candidate is found.
    /// - `ConvergenceFailure` if the evaluation limit stops the feasibility
    ///   checks, and errors of the fit, the acquisition search and the
    ///   feasibility check.
    pub fn propose_batch_into<'scope, 'capability>(
        &self,
        history: &SurrogateHistory<S, D>,
        output: &mut [[S; D]],
        workspace: &mut SurrogateSearchWorkspace<S, D, Factory::Solver>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        let count = output.len();
        validate_batch(history, count)?;
        bind_box(&mut workspace.bounds, history)?;
        crate::attempt::preflight(&scope, None)?;
        let SurrogateSearchWorkspace {
            bounds,
            search,
            result,
            proposals,
            shell,
        } = workspace;
        let bounds = bounds
            .as_ref()
            .ok_or_else(|| invalid(scioxide_core::argument!("workspace"), "unbound box"))?;
        let (outcome, started) = shell.execute(scope, self.entry(), |context| {
            self.propose(history, count, bounds, search, result, proposals, context)
        });
        if outcome.is_ok() {
            output.copy_from_slice(&proposals[..count]);
        }
        let succeeded = outcome.is_ok();
        shell.finish(
            AttemptKind::Call,
            started,
            !succeeded,
            succeeded,
            started.then_some(if succeeded { count } else { 0 }),
        );
        outcome
    }
    /// Runs the configured batches, adding each new observation to `history`
    /// as soon as it is evaluated.
    ///
    /// `objective` returns the objective value and the variance of its noise.
    /// The report counts the calls to `objective`; the evaluations of the
    /// acquisition count only toward the limits of `scope`. The search stops
    /// after the last batch or when the history is full.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the history is empty, or as for
    ///   [`Self::propose_batch_into`] and [`SurrogateHistory::push`].
    /// - `ConvergenceFailure` if the evaluation limit stops the search, and
    ///   errors of `objective`.
    ///
    /// Observations added before an error stay in `history`.
    pub fn minimize_into<'scope, 'capability>(
        &self,
        history: &mut SurrogateHistory<S, D>,
        objective: impl Fn(&[S]) -> Result<(S, S), SciOxideError>,
        workspace: &mut SurrogateSearchWorkspace<S, D, Factory::Solver>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<SurrogateSearchReport, SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        if history.observations.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("history"),
                "nonempty history required",
            ));
        }
        bind_box(&mut workspace.bounds, history)?;
        crate::attempt::preflight(&scope, None)?;
        let SurrogateSearchWorkspace {
            bounds,
            search,
            result,
            proposals,
            shell,
        } = workspace;
        let bounds = bounds
            .as_ref()
            .ok_or_else(|| invalid(scioxide_core::argument!("workspace"), "unbound box"))?;
        let before = history.observations.len();
        let (outcome, started) = shell.execute(scope, self.entry(), |context| {
            let mut evaluations = 0;
            let mut completed_batches = 0;
            for _ in 0..self.options.batches {
                let count = self
                    .options
                    .batch_size
                    .get()
                    .min(history.capacity - history.observations.len());
                if count == 0 {
                    break;
                }
                self.propose(history, count, bounds, search, result, proposals, context)?;
                for &point in proposals.iter() {
                    if !context
                        .record_provider(EvaluationRole::Objective)?
                        .is_granted()
                    {
                        return Err(crate::context::budget_error(
                            context,
                            scioxide_core::BudgetKind::Evaluation,
                            "objective evaluation budget exhausted",
                        ));
                    }
                    evaluations += 1;
                    let (value, noise_variance) = objective(&point)?;
                    history.push(SurrogateObservation::new(point, value, noise_variance))?;
                    context.commit_accepted(scioxide_core::PartialResultTrust::Admissible)?;
                }
                completed_batches += 1;
            }
            let best_observation = history
                .observations
                .iter()
                .enumerate()
                .min_by(|(ia, a), (ib, b)| {
                    a.value
                        .partial_cmp(&b.value)
                        .unwrap_or(core::cmp::Ordering::Equal)
                        .then(ia.cmp(ib))
                })
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    invalid(scioxide_core::argument!("history"), "empty search history")
                })?;
            Ok(SurrogateSearchReport {
                termination: OptimizeTermination::SearchBudgetCompleted,
                objective_evaluations: evaluations,
                completed_batches,
                best_observation,
            })
        });
        shell.finish(
            AttemptKind::Call,
            started,
            outcome.is_err(),
            false,
            started.then(|| history.observations.len() - before),
        );
        outcome
    }
    /// Runs the search with a new workspace and sequential execution.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace`] and [`Self::minimize_into`].
    pub fn minimize(
        &self,
        history: &mut SurrogateHistory<S, D>,
        objective: impl Fn(&[S]) -> Result<(S, S), SciOxideError>,
    ) -> Result<SurrogateSearchReport, SciOxideError> {
        let mut workspace = self.workspace()?;
        self.minimize_into(
            history,
            objective,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    fn entry(&self) -> Entry {
        Entry::Call {
            identity: crate::context::local_identity(
                scioxide_core::algorithm_id!("optimize", "SurrogateSearch", 1),
                D,
                &(&self.options, &self.fitter, &self.acquisition),
            ),
            budgets: SolveBudgets::unbounded(),
        }
    }

    /// Fit once and stage `count` separated feasible proposals.
    #[allow(clippy::too_many_arguments)]
    fn propose(
        &self,
        history: &SurrogateHistory<S, D>,
        count: usize,
        bounds: &Bounds<S>,
        search: &mut GlobalSearchWorkspace<S, Factory::Solver>,
        result: &mut GlobalSearchResult<S>,
        proposals: &mut Vec<[S; D]>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for observation in &history.observations {
            if !context
                .record_provider(EvaluationRole::Constraint)?
                .is_granted()
            {
                return Err(crate::context::budget_error(
                    context,
                    scioxide_core::BudgetKind::Evaluation,
                    "feasibility evaluation budget exhausted",
                ));
            }
            if !self.feasibility.feasible(&observation.point)? {
                return Err(invalid(
                    scioxide_core::argument!("history"),
                    "training history contains infeasible point",
                ));
            }
        }
        let mut fit_state = SolveState::new();
        let fit_identity = crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "SurrogateFit", 1),
            D,
            &self.fitter,
        );
        let model = SolveScope::from(&mut *context).run(
            &mut fit_state,
            fit_identity,
            SolveBudgets::unbounded(),
            |child| self.fitter.fit(history, child),
        )?;
        let mut work = model.workspace()?;
        let mut incumbent = S::INFINITY;
        for observation in &history.observations {
            incumbent = incumbent
                .min(checked_prediction(model.predict(observation.point, &mut work)?)?.mean);
        }
        let acquisition_work = RefCell::new(model.workspace()?);
        proposals.clear();
        scioxide_core::backend::try_reserve(
            proposals,
            count,
            scioxide_core::quantity!("surrogate_proposals"),
        )?;
        for _ in 0..count {
            let objective = AcquisitionObjective {
                model: &model,
                work: &acquisition_work,
                acquisition: &self.acquisition,
                incumbent,
            };
            let problem = BoundedObjective::new(&objective, bounds)?;
            self.search
                .minimize_into(&problem, result, search, &mut *context)?;
            let mut best: Option<([S; D], S)> = None;
            for report in result.starts() {
                for coordinates in [
                    report.candidate().map(|candidate| candidate.x()),
                    Some(report.start()),
                ]
                .into_iter()
                .flatten()
                {
                    let point: [S; D] = coordinates.try_into().map_err(|_| {
                        invalid(
                            scioxide_core::argument!("self"),
                            "local candidate dimension mismatch",
                        )
                    })?;
                    if !context
                        .record_provider(EvaluationRole::Constraint)?
                        .is_granted()
                    {
                        return Err(crate::context::budget_error(
                            context,
                            scioxide_core::BudgetKind::Evaluation,
                            "feasibility evaluation budget exhausted",
                        ));
                    }
                    if !self.feasibility.feasible(&point)?
                        || !separated(point, history, proposals, self.options.separation)?
                    {
                        continue;
                    }
                    let prediction = checked_prediction(model.predict(point, &mut work)?)?;
                    let score = checked_score(self.acquisition.score(prediction, incumbent)?)?;
                    if best.is_none_or(|(_, value)| score > value) {
                        best = Some((point, score));
                    }
                }
            }
            let point = best
                .ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("history"),
                        "no distinct feasible acquisition candidate",
                    )
                })?
                .0;
            proposals.push(point);
        }
        Ok(())
    }
}

/// Reject an empty history and an empty or over-capacity batch.
fn validate_batch<S: RealScalar, const D: usize>(
    history: &SurrogateHistory<S, D>,
    count: usize,
) -> Result<(), SciOxideError> {
    if history.observations.is_empty()
        || count == 0
        || count > history.capacity - history.observations.len()
    {
        return Err(invalid(
            scioxide_core::argument!("output"),
            "history or batch capacity mismatch",
        ));
    }
    Ok(())
}

/// Keep a `Bounds` copy of the history box for the inner multistart problem.
fn bind_box<S: RealScalar, const D: usize>(
    bounds: &mut Option<Bounds<S>>,
    history: &SurrogateHistory<S, D>,
) -> Result<(), SciOxideError> {
    let matches = bounds.as_ref().is_some_and(|bounds| {
        bounds
            .as_slice()
            .iter()
            .zip(history.bounds.as_slice())
            .all(|(&(lower, upper), &(lo, hi))| lower == Some(lo) && upper == Some(hi))
    });
    if !matches {
        let mut entries =
            try_with_capacity(D, scioxide_core::quantity!("surrogate_search_bounds"))?;
        entries.extend(
            history
                .bounds
                .as_slice()
                .iter()
                .map(|&(lo, hi)| (Some(lo), Some(hi))),
        );
        *bounds = Some(Bounds::new(entries)?);
    }
    Ok(())
}

fn separated<S: RealScalar, const D: usize>(
    point: [S; D],
    history: &SurrogateHistory<S, D>,
    proposals: &[[S; D]],
    distance: S,
) -> Result<bool, SciOxideError> {
    for other in history
        .observations
        .iter()
        .map(|o| &o.point)
        .chain(proposals)
    {
        let mut norm = S::ZERO;
        for ((&x, &y), &(lo, hi)) in point.iter().zip(other).zip(history.bounds.as_slice()) {
            let width = hi - lo;
            let delta = if width.is_finite() {
                (x - y) / width
            } else {
                let two = S::TWO;
                (x / two - y / two) / (hi / two - lo / two)
            };
            norm = norm.hypot(delta);
        }
        if !norm.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("history"),
                "normalized point distance is non-finite",
            ));
        }
        if norm <= distance {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{PowellLocalFactory, UniformStartDesign};
    use alloc::{format, vec};
    use scioxide_interp::CovarianceKernel;

    fn test_acquisitions<S: RealScalar>() {
        let prediction = SurrogatePrediction::new(S::ZERO, Some(S::ONE));
        let ei = GaussianAcquisition::ExpectedImprovement { margin: S::ZERO };
        let pi = GaussianAcquisition::ProbabilityOfImprovement { margin: S::ZERO };
        let lcb = GaussianAcquisition::LowerConfidenceBound {
            weight: S::try_from_f64_lossy(2.0).unwrap(),
        };
        let expected = S::try_from_f64_lossy(0.3989422804014327).unwrap();
        assert!(
            (ei.score(prediction, S::ZERO).unwrap() - expected).abs()
                < S::try_from_f64_lossy(2e-6).unwrap()
        );
        assert_eq!(
            pi.score(prediction, S::ZERO).unwrap(),
            S::try_from_f64_lossy(0.5).unwrap()
        );
        assert_eq!(
            lcb.score(prediction, S::ZERO).unwrap(),
            S::try_from_f64_lossy(2.0).unwrap()
        );
        let certain = SurrogatePrediction::new(S::try_from_f64_lossy(-1.0).unwrap(), Some(S::ZERO));
        assert_eq!(ei.score(certain, S::ZERO).unwrap(), S::ONE);
        assert_eq!(pi.score(certain, S::ZERO).unwrap(), S::ONE);
        assert!(
            ei.score(SurrogatePrediction::new(prediction.mean(), None), S::ZERO)
                .is_err()
        );
        assert!(
            pi.score(
                SurrogatePrediction::new(prediction.mean(), Some(-S::ONE)),
                S::ZERO
            )
            .is_err()
        );
    }
    #[test]
    fn analytic_acquisitions_both_widths() {
        test_acquisitions::<f32>();
        test_acquisitions::<f64>();
    }

    fn history() -> SurrogateHistory<f64, 1> {
        let mut history =
            SurrogateHistory::new(FiniteBounds::new(vec![(-2.0, 2.0)]).unwrap(), 8).unwrap();
        for x in [-1.0, 0.0, 1.0] {
            history
                .push(SurrogateObservation::new([x], (x - 0.3) * (x - 0.3), 0.01))
                .unwrap();
        }
        history
    }
    fn kriging() -> KrigingSurrogateFitter<f64, 1> {
        KrigingSurrogateFitter::new(
            Covariance::new(CovarianceKernel::SquaredExponential, 1.0, [0.8]).unwrap(),
            LocalModelOptions::default(),
        )
    }
    fn options(starts: usize, seed: u64) -> SurrogateSearchOptions<f64> {
        SurrogateSearchOptions::default()
            .with_acquisition_starts(Budget::try_from(starts).unwrap())
            .with_seed(seed)
    }
    fn fitted<T>(run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>) -> T {
        SolveScope::from(scioxide_core::ExecutionContext::sequential())
            .run(
                &mut SolveState::new(),
                crate::context::local_identity(
                    scioxide_core::algorithm_id!("optimize", "test", 1),
                    1,
                    &(),
                ),
                SolveBudgets::unbounded(),
                run,
            )
            .unwrap()
    }
    #[test]
    fn history_rejects_invalid_and_preserves_repeated_noisy_points() {
        let mut history = history();
        assert!(
            history
                .push(SurrogateObservation::new([0.0], 0.1, 0.02))
                .is_ok()
        );
        let before = history.observations().len();
        assert!(
            history
                .push(SurrogateObservation::new([3.0], 0.0, 0.0))
                .is_err()
        );
        assert!(
            history
                .push(SurrogateObservation::new([0.0], 0.0, -1.0))
                .is_err()
        );
        assert_eq!(history.observations().len(), before);
    }
    #[test]
    fn kriging_prediction_and_rbf_variance_boundary() {
        let history = history();
        let gp = kriging();
        let rbf = RbfSurrogateFitter::new(
            RbfKernel::Gaussian(1.0),
            RbfRegularization::Diagonal(0.01),
            LocalModelOptions::default(),
        );
        fitted(|context| {
            let model = gp.fit(&history, context)?;
            let mut work = model.workspace()?;
            let p = model.predict([0.3], &mut work)?;
            assert!((p.mean() - 0.09).abs() < 0.15);
            assert!(p.variance().unwrap() > 0.0);
            let rbf_model = rbf.fit(&history, context)?;
            let mut rbf_work = SurrogateModel::workspace(&rbf_model)?;
            let p = rbf_model.predict([0.3], &mut rbf_work)?;
            assert!(p.mean().is_finite());
            assert!(p.variance().is_none());
            assert!(
                GaussianAcquisition::ExpectedImprovement { margin: 0.0 }
                    .score(p, 0.0)
                    .is_err()
            );
            Ok(())
        });
    }
    #[test]
    fn batch_replays_and_excludes_history() {
        let mut history = history();
        let search = SurrogateSearch::new(
            options(6, 27),
            kriging(),
            GaussianAcquisition::ExpectedImprovement { margin: 0.0 },
            |x: &[f64]| Ok(x[0] >= -1.5),
            UniformStartDesign,
            PowellLocalFactory::default(),
        )
        .unwrap();
        let first = search.propose_batch(&history, 2).unwrap();
        let second = search.propose_batch(&history, 2).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
        assert!(
            first
                .iter()
                .all(|p| p[0] >= -1.5 && ![-1.0, 0.0, 1.0].contains(&p[0]))
        );
        assert_ne!(first[0], first[1]);
        let mut workspace = search.workspace().unwrap();
        let mut prepared = [[0.0]; 2];
        search
            .propose_batch_into(
                &history,
                &mut prepared,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(prepared.to_vec(), first);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::Succeeded
        );
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential(),
                crate::context::local_identity(
                    scioxide_core::algorithm_id!("optimize", "parent", 1),
                    1,
                    &(),
                ),
                &mut budgets,
            )
            .unwrap();
        let report = search
            .minimize_into(
                &mut history,
                |x| Ok(((x[0] - 0.3).powi(2), 0.01)),
                &mut workspace,
                &mut context,
            )
            .unwrap();
        assert_eq!(report.objective_evaluations(), 1);
        assert_eq!(report.completed_batches(), 1);
        assert_eq!(history.observations().len(), 4);
        assert_eq!(workspace.last_attempt().unwrap().progress(), Some(&1));
        assert!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Objective)
                > 0
        );
    }

    #[test]
    fn singular_zero_noise_history_is_reported() {
        let mut history =
            SurrogateHistory::new(FiniteBounds::new(vec![(-1.0, 1.0)]).unwrap(), 3).unwrap();
        for value in [1.0, 1.1] {
            history
                .push(SurrogateObservation::new([0.0], value, 0.0))
                .unwrap();
        }
        let gp = KrigingSurrogateFitter::new(
            Covariance::new(CovarianceKernel::SquaredExponential, 1.0, [1.0]).unwrap(),
            LocalModelOptions::default(),
        );
        let rbf = RbfSurrogateFitter::new(
            RbfKernel::Gaussian(1.0),
            RbfRegularization::None,
            LocalModelOptions::default(),
        );
        fitted(|context| {
            assert!(gp.fit(&history, context).is_err());
            assert!(rbf.fit(&history, context).is_err());
            Ok(())
        });
    }

    #[test]
    fn denied_feasibility_budget_prevents_new_objective_calls() {
        let mut history = history();
        let search = SurrogateSearch::new(
            options(3, 1),
            kriging(),
            GaussianAcquisition::ExpectedImprovement { margin: 0.0 },
            |_x: &[f64]| Ok(true),
            UniformStartDesign,
            PowellLocalFactory::default(),
        )
        .unwrap();
        let mut workspace = search.workspace().unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(
            scioxide_core::BudgetKind::Evaluation,
            Budget::try_from(1).unwrap(),
        );
        let mut context = state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential(),
                crate::context::local_identity(
                    scioxide_core::algorithm_id!("optimize", "parent", 1),
                    1,
                    &(),
                ),
                &mut budgets,
            )
            .unwrap();
        let before = history.observations().len();
        assert!(
            search
                .minimize_into(
                    &mut history,
                    |_x| unreachable!("real callback must not run"),
                    &mut workspace,
                    &mut context,
                )
                .is_err()
        );
        assert_eq!(history.observations().len(), before);
        let counts = context.counts().total().unwrap();
        assert_eq!(counts.provider(EvaluationRole::Objective), 0);
        assert_eq!(counts.provider(EvaluationRole::Constraint), 1);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::Failed
        );
    }

    #[test]
    fn normalized_separation_handles_extreme_finite_bounds() {
        let mut history =
            SurrogateHistory::new(FiniteBounds::new(vec![(-f64::MAX, f64::MAX)]).unwrap(), 2)
                .unwrap();
        history
            .push(SurrogateObservation::new([-f64::MAX], 1.0, 0.0))
            .unwrap();
        assert!(separated([f64::MAX], &history, &[], 0.5).unwrap());
        assert!(!separated([-f64::MAX], &history, &[], 0.0).unwrap());
    }

    #[test]
    fn probabilistic_acquisition_rejects_variance_free_rbf() {
        let history = history();
        let fitter = RbfSurrogateFitter::new(
            RbfKernel::Gaussian(1.0),
            RbfRegularization::Diagonal(0.01),
            LocalModelOptions::default(),
        );
        let search = SurrogateSearch::new(
            options(3, 1),
            fitter,
            GaussianAcquisition::ExpectedImprovement { margin: 0.0 },
            |_x: &[f64]| Ok(true),
            UniformStartDesign,
            PowellLocalFactory::default(),
        )
        .unwrap();
        let error = search.propose_batch(&history, 1).unwrap_err();
        assert!(format!("{error}").contains("predictive variance"));
        let mut output = [[7.0]];
        let mut workspace = search.workspace().unwrap();
        assert!(
            search
                .propose_batch_into(
                    &history,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(output, [[7.0]]);
        assert!(
            search
                .propose_batch_into(
                    &history,
                    &mut [],
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::PreflightRejected
        );
    }
}
