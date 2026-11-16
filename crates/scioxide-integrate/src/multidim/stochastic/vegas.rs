//! VEGAS: training of separable importance maps (Lepage 1978).

use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_core::{
    AffineValue, AlgorithmId, Budget, ExecutionContext, ExecutionPolicy, Interval,
    PartialResultTrust, PromotingCompensatedSum, PromotingProduct, RealScalar, Scaled,
    SciOxideError, SolveScope,
};
use scioxide_random::{Rng, SplittableRng};

use super::kernel::{self, Draw, Job, Lane, Schedule, Summary, prepare, sample};
use super::training::{Trained, importance, train};
use super::{MonteCarloProposal, ProposalUniforms, RandomStreams, TrainingReport};
use crate::geometry::IntervalGeometry;
use crate::kernel::number;
use crate::report::IntegrationTermination;
use crate::workspace::Attempts;

/// Options of VEGAS training: bins, iterations, pilot samples, damping and
/// execution policy.
///
/// The setters do not validate; [`VegasTrainer::new`] checks the options.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_integrate::VegasOptions;
/// let options = VegasOptions::<f64>::default()
///     .with_bins(16)
///     .with_iterations(Budget::try_from(3)?)
///     .with_damping(1.0);
/// assert_eq!((options.bins(), options.iterations().get()), (16, 3));
/// // The default is 2,048 pilot samples per iteration.
/// assert_eq!(options.samples_per_iteration().get(), 2048);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct VegasOptions<S: RealScalar> {
    bins: usize,
    iterations: Budget,
    samples: Budget,
    damping: S,
    policy: ExecutionPolicy,
}

impl<S: RealScalar> Default for VegasOptions<S> {
    /// Returns 50 bins per axis, five iterations of 2,048 pilot samples, damping
    /// exponent `α = 1.5` and sequential execution.
    fn default() -> Self {
        Self {
            bins: 50,
            iterations: Budget::ONE.saturating_add(4),
            samples: Budget::ONE.saturating_add(2047),
            damping: S::ONE + S::HALF,
            policy: ExecutionPolicy::Sequential,
        }
    }
}

impl<S: RealScalar> VegasOptions<S> {
    /// Sets the number `K` of bins per axis; at least 2.
    pub const fn with_bins(mut self, bins: usize) -> Self {
        self.bins = bins;
        self
    }

    /// Sets the number of training iterations.
    pub const fn with_iterations(mut self, iterations: Budget) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the number of pilot observations per iteration.
    pub const fn with_samples_per_iteration(mut self, samples: Budget) -> Self {
        self.samples = samples;
        self
    }

    /// Sets the damping exponent `α > 0` of the rebinning.
    ///
    /// Smaller values change the map more slowly between iterations; see
    /// [`VegasTrainer`].
    pub const fn with_damping(mut self, damping: S) -> Self {
        self.damping = damping;
        self
    }

    /// Sets how pilot sample blocks are run in parallel.
    ///
    /// The trained map does not depend on the policy.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Returns the number of bins per axis.
    #[must_use]
    pub const fn bins(&self) -> usize {
        self.bins
    }

    /// Returns the number of training iterations.
    pub const fn iterations(&self) -> Budget {
        self.iterations
    }

    /// Returns the number of pilot observations per iteration.
    pub const fn samples_per_iteration(&self) -> Budget {
        self.samples
    }

    /// Returns the damping exponent.
    pub const fn damping(&self) -> S {
        self.damping
    }

    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
}

/// Separable importance map on the unit cube, trained by [`VegasTrainer`].
///
/// The map is defined on the unit cube; use it with the box it was trained for.
///
/// Each axis `k` is divided into `K` bins by boundaries
/// `0 = e_0 < e_1 < … < e_K = 1`, and each bin has probability `1/K`. A uniform
/// `u` on axis `k` falls into bin `j = ⌊K u⌋` and maps linearly onto
/// `[e_j, e_{j+1})`. The density of a point `y` of the unit cube is
///
/// ```text
/// q(y) = Π_k 1 / (K Δ_k(y)),
/// ```
///
/// where `Δ_k(y)` is the width of the bin that contains `y_k`. The map is a
/// [`MonteCarloProposal`]: evaluate an integral with it through
/// [`MonteCarloIntegrator::importance`](crate::MonteCarloIntegrator::importance).
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{MonteCarloProposal, ProposalUniforms, VegasOptions, VegasTrainer};
/// let options = VegasOptions::default().with_bins(4);
/// let plan = VegasTrainer::new(&[Interval::new(0.0, 1.0)?], options)?;
/// let map = plan.uniform_map()?;
/// assert_eq!(map.boundaries(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
/// // The untrained map is the identity, with density 1.
/// let mut y = [0.0_f64];
/// assert_eq!(map.sample(ProposalUniforms::new(&[0.3])?, &mut y)?, 1.0);
/// assert!((y[0] - 0.3).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct VegasMap<S: RealScalar> {
    dimension: usize,
    bins: usize,
    edges: Vec<S>,
    prepared: Vec<Bin<S>>,
}

impl<S: RealScalar> VegasMap<S> {
    /// Returns the dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of bins per axis.
    #[must_use]
    pub const fn bins(&self) -> usize {
        self.bins
    }

    /// Returns the bin boundaries: `K + 1` values from 0 to 1 for each axis in turn.
    pub fn boundaries(&self) -> &[S] {
        &self.edges
    }
}

impl<S: RealScalar> MonteCarloProposal<S> for VegasMap<S> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn uniform_count(&self) -> usize {
        self.dimension
    }
    fn sample(
        &self,
        uniforms: ProposalUniforms<'_, S>,
        point: &mut [S],
    ) -> Result<S, SciOxideError> {
        map_sample(self.bins, &self.prepared, uniforms, point)
    }
}

/// The map being trained, borrowed from a training workspace.
struct MapView<'a, S: RealScalar> {
    dimension: usize,
    bins: usize,
    prepared: &'a [Bin<S>],
}

impl<S: RealScalar> MonteCarloProposal<S> for MapView<'_, S> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn uniform_count(&self) -> usize {
        self.dimension
    }
    fn sample(
        &self,
        uniforms: ProposalUniforms<'_, S>,
        point: &mut [S],
    ) -> Result<S, SciOxideError> {
        map_sample(self.bins, self.prepared, uniforms, point)
    }
}

/// Maps uniforms through the bins of each axis and returns the density.
fn map_sample<S: RealScalar>(
    bins: usize,
    prepared: &[Bin<S>],
    uniforms: ProposalUniforms<'_, S>,
    point: &mut [S],
) -> Result<S, SciOxideError> {
    let uniforms = uniforms.as_slice();
    let dimension = prepared.len() / bins;
    crate::kernel::shape(dimension, uniforms.len())?;
    crate::kernel::shape(dimension, point.len())?;
    let scale = number::<S>(bins)?;
    let mut density = PromotingProduct::ONE;
    for ((&u, x), edges) in uniforms.iter().zip(point).zip(prepared.chunks_exact(bins)) {
        let scaled = u * scale;
        let bin = scaled.floor().try_to_usize_exact()?.min(bins - 1);
        let local =
            (scaled - number::<S>(bin)?).min(S::ONE.next_after(<S as RealScalar>::NEG_INFINITY));
        let prepared = edges[bin];
        *x = prepared.geometry.half_open_point_admitted(local)?;
        density = density.multiplied_by(prepared.density)?;
    }
    density.round()
}

/// Geometry and density of one bin.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bin<S: RealScalar> {
    geometry: IntervalGeometry<S>,
    density: PromotingProduct<S>,
}
fn empty_bin<S: RealScalar>() -> Result<Bin<S>, SciOxideError> {
    let geometry = IntervalGeometry::prepare_half_open(Interval::new(S::ZERO, S::ONE)?)?
        .ok_or_else(|| {
            super::invalid(
                scioxide_core::argument!("map"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "collapsed VEGAS bin",
            )
        })?;
    Ok(Bin {
        geometry,
        density: PromotingProduct::ONE,
    })
}
fn prepare_bins<S: RealScalar>(
    bins: usize,
    edges: &[S],
    prepared: &mut [Bin<S>],
) -> Result<(), SciOxideError> {
    let scale = number::<S>(bins)?;
    for (axis, prepared) in edges
        .chunks_exact(bins + 1)
        .zip(prepared.chunks_exact_mut(bins))
    {
        for (bounds, bin) in axis.windows(2).zip(prepared) {
            let geometry =
                IntervalGeometry::prepare_half_open(Interval::new(bounds[0], bounds[1])?)?
                    .ok_or_else(|| {
                        super::invalid(
                            scioxide_core::argument!("map"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "collapsed VEGAS bin",
                        )
                    })?;
            let density = geometry.native_span().and_then(|span| {
                let width = span * scale;
                let density = S::ONE / width;
                (width.is_finite()
                    && width >= S::MIN_POSITIVE
                    && density.is_finite()
                    && density >= S::MIN_POSITIVE)
                    .then_some(density)
            });
            *bin = Bin {
                geometry,
                density: match density {
                    Some(density) => PromotingProduct::from_value(density)?,
                    None => PromotingProduct::from_scaled(
                        Scaled::from_value(S::ONE)?
                            .div(geometry.span()?.mul(Scaled::from_value(scale)?)?)?,
                    ),
                },
            };
        }
    }
    Ok(())
}

/// Sum of `(‖f‖ / q)²` per axis and bin.
#[derive(Debug, Clone)]
struct Scores<S: RealScalar>(Vec<PromotingCompensatedSum<S>>);

impl<S: RealScalar> Summary for Scores<S> {
    fn clear(&mut self) {
        self.0.fill(PromotingCompensatedSum::ZERO);
    }
}

/// VEGAS training of a [`VegasMap`] for a finite box.
///
/// VEGAS adapts a separable importance density to the integrand, so that
/// [`MonteCarloIntegrator::importance`](crate::MonteCarloIntegrator::importance) samples
/// more where `|f|` is large. It works best when `f` is dominated by peaks that
/// are aligned with the coordinate axes; it cannot follow correlations between
/// coordinates.
///
/// [`train_into`](Self::train_into) starts from the uniform map and runs the
/// configured number of iterations. Each iteration draws `N` pilot points `y`
/// of the unit cube from the current map, evaluates `f` at the corresponding
/// points `x` of the box, adds `(‖f(x)‖ / q(y))²` to the bin that contains `y_k`
/// on every axis `k`, and then rebins each axis. Here `‖f‖` is the absolute value
/// of a real integrand and the Euclidean norm of the real components otherwise,
/// so one map serves all components. Rebinning one axis:
///
/// 1. Divides the bin sums by their maximum, averages each with its neighbors,
///    and divides the results by their total, giving `r_j` with `Σ_j r_j = 1`.
/// 2. Damps them to `w_j = ((1 − r_j) / ln(1/r_j))^α`, with `α` the damping
///    exponent and `r_j` bounded below by the machine epsilon.
/// 3. Places new boundaries so that each bin holds the same total of `w`,
///    interpolating linearly within the current bins.
/// 4. Averages the new boundaries with the current ones.
///
/// If an axis has no positive sums, or the new boundaries are not strictly
/// increasing, the axis keeps its boundaries. Each iteration costs
/// `O(N(d + C) + dK)` operations for `d` axes, `C` components and `K` bins,
/// plus `N` calls of `f`.
///
/// The training uniforms come from a [`RandomStreams`]; iteration `i` uses the
/// sample blocks after those of iterations `0..i`. The trained map does not
/// depend on the execution policy or the number of workers.
///
/// # Examples
///
/// Train a map for the peaked integrand `e^(20x)` on `[0, 1]`, then integrate with it:
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// use scioxide_integrate::{
///     MonteCarloOptions, MonteCarloIntegrator, RandomStreams, VegasOptions, VegasTrainer,
/// };
/// use scioxide_random::{Philox4x64, SeedableRng};
/// let bounds = [Interval::new(0.0, 1.0)?];
/// let f = |x: &[f64]| Ok((20.0 * x[0]).exp());
/// let root = Philox4x64::from_seed(9);
/// let vegas = VegasTrainer::new(&bounds, VegasOptions::default().with_bins(16))?;
/// let (map, report) = vegas.train(&f, RandomStreams::new(&root, 0))?;
/// assert!(report.is_success());
/// let plan = MonteCarloIntegrator::importance(&bounds, map, MonteCarloOptions::default())?;
/// let cx = ExecutionContext::sequential_native();
/// let (mut workspace, mut value) = (plan.workspace(1, cx)?, 0.0);
/// let streams = RandomStreams::new(&root, 1);
/// plan.integrate_into(&f, streams, &mut value, &mut workspace, cx)?;
/// // The exact integral is (e^20 − 1) / 20.
/// let exact = (20.0_f64.exp() - 1.0) / 20.0;
/// assert!((value - exact).abs() < 1e-2 * exact);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. P. Lepage, "A new algorithm for adaptive multidimensional integration",
/// *J. Comput. Phys.* 27(2), 192–203, 1978.
/// <https://doi.org/10.1016/0021-9991(78)90004-9>
#[derive(Debug, Clone)]
pub struct VegasTrainer<S: RealScalar> {
    bounds: Vec<Interval<S>>,
    options: VegasOptions<S>,
}

/// Workspace of [`VegasTrainer::train_into`].
///
/// Create one with [`VegasTrainer::workspace`]. It serves plans with the same
/// dimension and number of bins, and at most as many workers.
#[derive(Debug)]
pub struct VegasWorkspace<S: RealScalar> {
    attempts: Attempts<TrainingReport>,
    lanes: Vec<Lane<S, Scores<S>>>,
    scores: Vec<PromotingCompensatedSum<S>>,
    prepared: Vec<Bin<S>>,
    smooth: Vec<S>,
    edges: Vec<S>,
    candidate: Vec<S>,
}

impl<S: RealScalar> VegasWorkspace<S> {
    /// Returns the record of the last training call with this workspace: its
    /// outcome, solve counts and, if it returned one, its report. Returns `None`
    /// before the first call.
    #[must_use]
    pub fn last_attempt(&self) -> Option<scioxide_core::AttemptRecord<TrainingReport>> {
        self.attempts.last_attempt()
    }
}

impl<S: RealScalar> VegasTrainer<S> {
    /// Prepares VEGAS training for the box `bounds`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than 2 bins, if the uniform boundaries
    ///   `j/K` are not exactly representable and strictly increasing in `S`, or if
    ///   the damping exponent is not finite and positive.
    /// - `SizeOverflow` if the number of boundaries or of sample blocks overflows.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn new(bounds: &[Interval<S>], options: VegasOptions<S>) -> Result<Self, SciOxideError> {
        if options.bins < 2 || !distinct_edges::<S>(options.bins) {
            return Err(super::invalid(
                scioxide_core::argument!("bins"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "VEGAS needs at least two bins with distinct representable boundaries",
            ));
        }
        if !options.damping.is_finite() || options.damping <= S::ZERO {
            return Err(super::invalid(
                scioxide_core::argument!("damping"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "VEGAS damping must be finite and positive",
            ));
        }
        scioxide_core::backend::checked_mul(
            bounds.len(),
            options.bins + 1,
            scioxide_core::quantity!("vegas_edges"),
        )?;
        kernel::offset(
            options.iterations.get(),
            Schedule::blocks(options.samples.get()),
        )?;
        let mut owned = scioxide_core::backend::try_with_capacity(
            bounds.len(),
            scioxide_core::quantity!("bounds"),
        )?;
        owned.extend_from_slice(bounds);
        Ok(Self {
            bounds: owned,
            options,
        })
    }

    /// Returns the box.
    pub fn bounds(&self) -> &[Interval<S>] {
        &self.bounds
    }

    /// Returns the options.
    pub const fn options(&self) -> &VegasOptions<S> {
        &self.options
    }

    /// Returns the algorithm identifier `VegasTraining` recorded in solve diagnostics.
    pub const fn algorithm_id(&self) -> AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "VegasTraining", 1)
    }

    /// Returns the untrained map with equal bins, which is the identity map.
    ///
    /// Training starts from this map. Use it as the output of [`train_into`](Self::train_into).
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn uniform_map(&self) -> Result<VegasMap<S>, SciOxideError> {
        let mut edges = try_filled(
            self.bounds.len() * (self.options.bins + 1),
            S::ZERO,
            scioxide_core::quantity!("vegas_edges"),
        )?;
        uniform_edges(self.options.bins, &mut edges)?;
        let mut prepared = try_filled(
            self.bounds.len() * self.options.bins,
            empty_bin()?,
            scioxide_core::quantity!("vegas_bins"),
        )?;
        prepare_bins(self.options.bins, &edges, &mut prepared)?;
        Ok(VegasMap {
            dimension: self.bounds.len(),
            bins: self.options.bins,
            edges,
            prepared,
        })
    }

    /// Creates a training workspace for the workers that the execution policy and
    /// `execution` allow.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(
        &self,
        execution: ExecutionContext<'_>,
    ) -> Result<VegasWorkspace<S>, SciOxideError> {
        let dimension = self.bounds.len();
        let bins = self.options.bins;
        let scores = scioxide_core::backend::checked_mul(
            dimension,
            bins,
            scioxide_core::quantity!("vegas_scores"),
        )?;
        let edges = dimension * (bins + 1);
        Ok(VegasWorkspace {
            attempts: Attempts::default(),
            lanes: Lane::allocate_many(
                kernel::lanes(
                    self.options.policy,
                    self.options.samples.get().div_ceil(super::SAMPLE_BLOCK),
                    execution,
                ),
                dimension,
                || {
                    Ok(Scores(try_filled(
                        scores,
                        PromotingCompensatedSum::ZERO,
                        scioxide_core::quantity!("vegas_scores"),
                    )?))
                },
            )?,
            scores: try_filled(
                scores,
                PromotingCompensatedSum::ZERO,
                scioxide_core::quantity!("vegas_scores"),
            )?,
            prepared: try_filled(scores, empty_bin()?, scioxide_core::quantity!("vegas_bins"))?,
            smooth: try_filled(bins, S::ZERO, scioxide_core::quantity!("vegas_smooth"))?,
            edges: try_filled(edges, S::ZERO, scioxide_core::quantity!("vegas_edges"))?,
            candidate: try_filled(edges, S::ZERO, scioxide_core::quantity!("vegas_edges"))?,
        })
    }

    /// Trains a map for `f` from the uniform map and writes it to `map`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `map` must have this plan's dimension and number of bins, for example from
    /// [`uniform_map`](Self::uniform_map); its previous boundaries are not used.
    /// Pilot observations are counted like the observations of
    /// [`MonteCarloIntegrator::integrate_into`](crate::MonteCarloIntegrator::integrate_into).
    ///
    /// A budget, cancellation or deadline can stop training. A stop after at
    /// least one completed iteration returns `Ok`, writes the map of the last
    /// completed iteration and reports the stop in the [`TrainingReport`]. A box
    /// of dimension zero or of zero width writes the uniform map without calling
    /// `f`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `map` or `workspace` has another dimension or
    ///   number of bins, or the workspace has too few workers for `scope`.
    /// - `NonFinite` (origin `Callback`) if a component of `f` is NaN or infinite.
    /// - `ConvergenceFailure` if a budget stops training before the first
    ///   completed iteration.
    /// - `ExecutionStopped` if a cancellation or deadline stops training before
    ///   the first completed iteration.
    /// - Errors returned by `f`.
    pub fn train_into<'scope, 'capability, F, V, R>(
        &self,
        f: F,
        streams: RandomStreams<'_, R>,
        map: &mut VegasMap<S>,
        workspace: &mut VegasWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<TrainingReport, SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        R: Rng<S> + SplittableRng + Sync,
    {
        let f = &f;
        let scope = scope.into();
        let dimension = self.bounds.len();
        let bins = self.options.bins;
        let workers = kernel::lanes(
            self.options.policy,
            self.options.samples.get().div_ceil(super::SAMPLE_BLOCK),
            scope.execution(),
        );
        let preflight = crate::kernel::shape(dimension, map.dimension)
            .and_then(|()| crate::kernel::shape(bins, map.bins))
            .and_then(|()| crate::kernel::shape(workspace.edges.len(), map.edges.len()))
            .and_then(|()| crate::kernel::shape(dimension * bins, workspace.scores.len()))
            .and_then(|()| crate::kernel::shape(bins, workspace.smooth.len()))
            .and_then(|()| {
                if workspace.lanes.len() < workers {
                    crate::kernel::shape(workers, workspace.lanes.len())
                } else {
                    Ok(())
                }
            });
        if let Err(error) = preflight {
            return workspace.attempts.reject(error);
        }
        let VegasWorkspace {
            attempts,
            lanes,
            scores,
            prepared,
            smooth,
            edges,
            candidate,
        } = workspace;
        train(attempts, scope, self.algorithm_id(), dimension, |context| {
            uniform_edges(bins, edges)?;
            prepare_bins(bins, edges, prepared)?;
            if dimension == 0 || self.bounds.iter().any(|b| b.lower() == b.upper()) {
                map.edges.copy_from_slice(edges);
                map.prepared.copy_from_slice(prepared);
                return Ok(Trained {
                    termination: if dimension == 0 {
                        IntegrationTermination::FixedRuleCompleted
                    } else {
                        IntegrationTermination::DegenerateDomain
                    },
                    updates: 0,
                });
            }
            let lanes = &mut lanes[..workers];
            prepare(lanes, &self.bounds)?;
            let samples = self.options.samples.get();
            let mut termination = IntegrationTermination::SampleBudgetCompleted;
            let mut updates = 0;
            for iteration in 0..self.options.iterations.get() {
                scores.fill(PromotingCompensatedSum::ZERO);
                let progress = sample(
                    &Job {
                        source: &streams,
                        proposal: Some(&MapView {
                            dimension,
                            bins,
                            prepared: prepared.as_slice(),
                        }),
                        function: f,
                        dimension,
                        uniforms: dimension,
                    },
                    Schedule::uniform(iteration, samples)?,
                    lanes,
                    self.options.policy,
                    context,
                    &score::<S, V>(bins),
                    &mut |lane: &Scores<S>| {
                        scores
                            .iter_mut()
                            .zip(&lane.0)
                            .try_for_each(|(total, score)| total.merge(*score))
                    },
                )?;
                if let Some(stop) = progress.stop {
                    termination = stop;
                    break;
                }
                candidate.copy_from_slice(edges);
                for axis in 0..dimension {
                    rebin(
                        &scores[axis * bins..(axis + 1) * bins],
                        &edges[axis * (bins + 1)..(axis + 1) * (bins + 1)],
                        &mut candidate[axis * (bins + 1)..(axis + 1) * (bins + 1)],
                        smooth,
                        self.options.damping,
                    )?;
                }
                edges.copy_from_slice(candidate);
                prepare_bins(bins, edges, prepared)?;
                context.commit_accepted(PartialResultTrust::Admissible)?;
                updates += 1;
            }
            if termination.is_success() || updates > 0 {
                map.edges.copy_from_slice(edges);
                map.prepared.copy_from_slice(prepared);
            }
            Ok(Trained {
                termination,
                updates,
            })
        })
    }

    /// Trains a map for `f` on the calling thread and returns it.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// This allocates the map and a workspace and calls
    /// [`train_into`](Self::train_into) with
    /// [`ExecutionContext::sequential_native`], which sets no budgets.
    ///
    /// # Errors
    ///
    /// - The errors of [`train_into`](Self::train_into).
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn train<F, V, R>(
        &self,
        f: F,
        streams: RandomStreams<'_, R>,
    ) -> Result<(VegasMap<S>, TrainingReport), SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        R: Rng<S> + SplittableRng + Sync,
    {
        let f = &f;
        let mut map = self.uniform_map()?;
        let mut workspace = self.workspace(ExecutionContext::sequential_native())?;
        let report = self.train_into(
            f,
            streams,
            &mut map,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok((map, report))
    }
}

/// Returns whether the uniform boundaries `i / bins` are exact and strictly increasing.
fn distinct_edges<S: RealScalar>(bins: usize) -> bool {
    let Ok(scale) = number::<S>(bins) else {
        return false;
    };
    let mut previous = S::ZERO;
    (1..=bins).all(|index| {
        number::<S>(index).is_ok_and(|index| {
            let edge = index / scale;
            let increasing = edge > previous;
            previous = edge;
            increasing
        })
    })
}

/// Writes the uniform boundaries `i / bins` of every axis.
fn uniform_edges<S: RealScalar>(bins: usize, edges: &mut [S]) -> Result<(), SciOxideError> {
    let scale = number::<S>(bins)?;
    for axis in edges.chunks_exact_mut(bins + 1) {
        for (index, edge) in axis.iter_mut().enumerate() {
            *edge = number::<S>(index)? / scale;
        }
    }
    Ok(())
}

/// Returns the update that adds `(‖f‖ / q)²` to the bin of every axis.
fn score<S: RealScalar, V: AffineValue<Scale = S>>(
    bins: usize,
) -> impl Fn(&mut Scores<S>, Draw<'_, S>, &V) -> Result<(), SciOxideError> + Sync {
    #[inline(always)]
    move |scores, draw, value| {
        let native = if value.dimension()? == 1 {
            let value = crate::kernel::checked_callback_value(value.component(0)?)?.abs();
            let weighted = value / draw.density;
            let square = weighted * weighted;
            (value == S::ZERO
                || weighted.is_finite()
                    && weighted >= S::MIN_POSITIVE
                    && square.is_finite()
                    && square >= S::MIN_POSITIVE)
                .then_some(square)
        } else {
            None
        };
        let square = if native.is_none() {
            Some(
                importance(value)?
                    .div(Scaled::from_value(draw.density)?)?
                    .square()?,
            )
        } else {
            None
        };
        let scale = number::<S>(bins)?;
        for (axis, &u) in draw.uniforms.iter().enumerate() {
            let bin = (u * scale).floor().try_to_usize_exact()?.min(bins - 1);
            match native {
                Some(value) => scores.0[axis * bins + bin].add_value(value)?,
                None => {
                    if let Some(square) = square {
                        scores.0[axis * bins + bin].add_scaled(square)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Rebins one axis from its bin scores into `next`.
///
/// Scores are normalized by their maximum, smoothed over neighbors and damped by
/// `((1 − r) / −ln r)^damping`; boundaries are placed at equal damped mass and
/// averaged with `old`. Zero, underflowing or invalid results keep the axis
/// unchanged.
fn rebin<S: RealScalar>(
    scores: &[PromotingCompensatedSum<S>],
    old: &[S],
    next: &mut [S],
    smooth: &mut [S],
    damping: S,
) -> Result<(), SciOxideError> {
    let bins = scores.len();
    let native_maximum = scores.iter().try_fold(S::ZERO, |maximum, score| {
        score
            .native()
            .and_then(|sum| sum.total().ok())
            .map(|value| maximum.max(value))
    });
    if let Some(maximum) = native_maximum {
        if maximum == S::ZERO {
            return Ok(());
        }
        for (index, smoothed) in smooth.iter_mut().enumerate() {
            let lower = index.saturating_sub(1);
            let upper = (index + 2).min(bins);
            let mut sum = S::ZERO;
            for score in &scores[lower..upper] {
                sum += score.round()? / maximum;
            }
            *smoothed = sum / number::<S>(upper - lower)?;
        }
    } else {
        let maximum = scores.iter().try_fold(Scaled::ZERO, |maximum, score| {
            let value = score.total()?;
            Ok::<_, SciOxideError>(if value.abs_cmp(maximum).is_gt() {
                value
            } else {
                maximum
            })
        })?;
        if maximum.is_zero() {
            return Ok(());
        }
        for (index, smoothed) in smooth.iter_mut().enumerate() {
            let lower = index.saturating_sub(1);
            let upper = (index + 2).min(bins);
            let mut sum = S::ZERO;
            for score in &scores[lower..upper] {
                sum += score.total()?.div(maximum)?.round()?;
            }
            *smoothed = sum / number::<S>(upper - lower)?;
        }
    }
    let total = smooth.iter().fold(S::ZERO, |sum, &value| sum + value);
    for score in smooth.iter_mut() {
        let fraction = (*score / total).max(S::EPSILON);
        *score = if fraction >= S::ONE {
            S::ONE
        } else {
            ((S::ONE - fraction) / (-fraction.ln())).powf(damping)
        };
    }
    let total = smooth.iter().fold(S::ZERO, |sum, &value| sum + value);
    // An extreme but valid damping exponent can underflow every score; there is
    // then no representable redistribution and the axis keeps its boundaries.
    if !total.is_finite() || total <= S::ZERO {
        return Ok(());
    }
    let mut bin = 0;
    let mut cumulative = S::ZERO;
    for (index, edge) in next.iter_mut().enumerate().take(bins).skip(1) {
        let target = total * number::<S>(index)? / number::<S>(bins)?;
        while bin + 1 < bins && cumulative + smooth[bin] < target {
            cumulative += smooth[bin];
            bin += 1;
        }
        let fraction = ((target - cumulative) / smooth[bin])
            .max(S::ZERO)
            .min(S::ONE);
        let rebinned = old[bin] + fraction * (old[bin + 1] - old[bin]);
        *edge = old[index] * S::HALF + rebinned * S::HALF;
    }
    if next
        .windows(2)
        .any(|pair| !pair[1].is_finite() || pair[0] >= pair[1])
    {
        next.copy_from_slice(old);
    }
    Ok(())
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn training_storage_is_bounded_by_workers_and_blocks() {
        use core::num::NonZeroUsize;
        use scioxide_core::ThreadExecutor;
        let executor = ThreadExecutor::new(NonZeroUsize::new(3).unwrap());
        let execution = ExecutionContext::new(&executor);
        for (samples, expected) in [(64, 1), (5 * super::super::SAMPLE_BLOCK, 3)] {
            let plan = VegasTrainer::new(
                &[Interval::new(0.0_f64, 1.0).unwrap()],
                VegasOptions::default()
                    .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(8).unwrap()))
                    .with_samples_per_iteration(Budget::try_from(samples).unwrap()),
            )
            .unwrap();
            assert_eq!(plan.workspace(execution).unwrap().lanes.len(), expected);
            assert_eq!(
                plan.workspace(ExecutionContext::sequential_native())
                    .unwrap()
                    .lanes
                    .len(),
                1
            );
        }
    }
}
