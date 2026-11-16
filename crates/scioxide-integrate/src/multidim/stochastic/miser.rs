//! MISER: training of strata by recursive bisection (Press & Farrar 1990).

use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_core::{
    AffineValue, AlgorithmId, Budget, BudgetKind, ExecutionContext, Interval, PartialResultTrust,
    RealScalar, Scaled, ScaledAccumulator, SciOxideError, SolveScope,
};
use scioxide_random::{Rng, SplittableRng};
use scioxide_stats::{MomentOrder, OnlineMoments};

use super::kernel::{self, Draw, Job, Lane, Schedule, Summary, prepare, sample};
use super::training::{Trained, train};
use super::{MonteCarloOptions, RandomStreams, TrainingReport, UniformProposal};
use crate::report::IntegrationTermination;
use crate::workspace::Attempts;

/// Options of MISER training: evaluation options, pilot samples and limits on
/// the strata.
///
/// The evaluation options set the total number of samples that training
/// distributes over the strata, and the interval of the later stratified
/// estimate. They must request one replicate. Their execution policy also runs
/// the pilot sample blocks in parallel. The setters do not validate;
/// [`MiserTrainer::new`] checks the options.
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_integrate::{MiserOptions, MonteCarloOptions};
/// let options = MiserOptions::default()
///     .with_evaluation(MonteCarloOptions::default().with_samples(Budget::try_from(1 << 14)?))
///     .with_max_leaves(Budget::try_from(32)?);
/// assert_eq!(options.evaluation().samples().get(), 16_384);
/// // The defaults are 256 pilot samples and depth 8.
/// assert_eq!(options.pilot_samples().get(), 256);
/// assert_eq!(options.max_depth(), 8);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MiserOptions {
    evaluation: MonteCarloOptions,
    pilot_samples: Budget,
    max_leaves: Budget,
    max_depth: usize,
}

impl Default for MiserOptions {
    /// Returns the default [`MonteCarloOptions`], 256 pilot samples, at most 16
    /// strata and depth at most 8.
    fn default() -> Self {
        Self {
            evaluation: MonteCarloOptions::default(),
            pilot_samples: Budget::ONE.saturating_add(255),
            max_leaves: Budget::ONE.saturating_add(15),
            max_depth: 8,
        }
    }
}

impl MiserOptions {
    /// Sets the evaluation options of the trained strata.
    pub const fn with_evaluation(mut self, evaluation: MonteCarloOptions) -> Self {
        self.evaluation = evaluation;
        self
    }

    /// Sets the number of pilot observations drawn in each stratum before it is
    /// split; at least 4.
    pub const fn with_pilot_samples(mut self, samples: Budget) -> Self {
        self.pilot_samples = samples;
        self
    }

    /// Sets the maximum number of strata.
    pub const fn with_max_leaves(mut self, leaves: Budget) -> Self {
        self.max_leaves = leaves;
        self
    }

    /// Sets the maximum number of bisections from the box to a stratum.
    pub const fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Returns the evaluation options of the trained strata.
    pub const fn evaluation(&self) -> &MonteCarloOptions {
        &self.evaluation
    }

    /// Returns the number of pilot observations per stratum.
    pub const fn pilot_samples(&self) -> Budget {
        self.pilot_samples
    }

    /// Returns the maximum number of strata.
    pub const fn max_leaves(&self) -> Budget {
        self.max_leaves
    }

    /// Returns the maximum bisection depth.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }
}

/// Strata of a box with the number of evaluation samples of each, trained by
/// [`MiserTrainer`].
///
/// The strata are disjoint half-open boxes that cover the box. Each holds at
/// least 2 of the evaluation samples, and the counts add up to
/// [`samples`](MonteCarloOptions::samples) of the evaluation options. Evaluate
/// the strata with [`MonteCarloIntegrator::stratified`](crate::MonteCarloIntegrator::stratified).
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{MiserOptions, MiserTrainer};
/// let plan = MiserTrainer::new(&[Interval::new(0.0, 2.0)?], MiserOptions::default())?;
/// let strata = plan.initial_strata()?;
/// // Before training, the box is one stratum with all 4,096 default samples.
/// assert_eq!(strata.len(), 1);
/// assert_eq!(strata.samples(), &[4096]);
/// assert_eq!(strata.strata().next(), Some(strata.bounds()));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MiserStrata<S: RealScalar> {
    bounds: Vec<Interval<S>>,
    leaves: Vec<Interval<S>>,
    samples: Vec<usize>,
    count: usize,
    options: MonteCarloOptions,
}

impl<S: RealScalar> PartialEq for MiserStrata<S> {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds
            && self.options == other.options
            && self.samples() == other.samples()
            && self.strata().eq(other.strata())
    }
}

impl<S: RealScalar> MiserStrata<S> {
    /// Returns the box.
    pub fn bounds(&self) -> &[Interval<S>] {
        &self.bounds
    }

    /// Returns the boxes of the strata in evaluation order.
    pub fn strata(&self) -> impl ExactSizeIterator<Item = &[Interval<S>]> {
        (0..self.count).map(|index| self.stratum(index))
    }

    /// Returns the number of evaluation samples of each stratum, in evaluation
    /// order.
    pub fn samples(&self) -> &[usize] {
        &self.samples[..self.count]
    }

    /// Returns the number of strata.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns whether there are no strata; strata from a plan are never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the evaluation options.
    pub const fn options(&self) -> &MonteCarloOptions {
        &self.options
    }

    /// Returns the box of stratum `index`.
    pub(super) fn stratum(&self, index: usize) -> &[Interval<S>] {
        let dimension = self.bounds.len();
        &self.leaves[index * dimension..(index + 1) * dimension]
    }
}

/// Pilot values or moments of the two halves of every axis.
#[derive(Debug, Clone)]
struct Halves<S: RealScalar> {
    moments: Vec<OnlineMoments<S>>,
    values: Vec<S>,
    counts: Vec<usize>,
    single: bool,
    stride: usize,
    components: usize,
}

impl<S: RealScalar> Summary for Halves<S> {
    fn clear(&mut self) {
        if !self.single {
            self.moments.fill(OnlineMoments::new(MomentOrder::Second));
        }
        self.counts.fill(0);
    }
    fn finish(&mut self) -> Result<(), SciOxideError> {
        if self.single {
            return Ok(());
        }
        for (half, &count) in self.counts.iter().enumerate() {
            for component in 0..self.components {
                let index = half * self.components + component;
                let values = &self.values[index * self.stride..][..count];
                self.moments[index] = OnlineMoments::from_slice(values, MomentOrder::Second)?;
            }
        }
        Ok(())
    }
}

/// MISER training of [`MiserStrata`] for a finite box.
///
/// MISER divides the box into strata by recursive bisection and gives more
/// evaluation samples to strata where `f` varies more. The stratified estimate
/// of [`MonteCarloIntegrator::stratified`](crate::MonteCarloIntegrator::stratified) then
/// has a smaller variance than uniform sampling with the same number of samples.
/// It helps most when the variation of `f` is concentrated in part of the box.
///
/// [`train_into`](Self::train_into) starts with the box as one stratum that
/// holds all `N` evaluation samples, and repeats these steps:
///
/// 1. Selects the stratum with the largest estimated variance contribution
///    `(V σ)² / n` among those not yet visited, with depth below `max_depth`
///    and `n ≥ 4` samples. Here `V` is the stratum volume, `n` its samples, and
///    `σ` the pilot standard deviation of its half from the parent's pilot. The
///    box itself is selected first. Ties go to the stratum created first.
/// 2. Draws `p` uniform pilot points in it. For each axis, it computes the
///    sample standard deviations `σ_a` and `σ_b` of `f` in the two halves split
///    at the midpoint; for several components, the Euclidean norm of the
///    component standard deviations.
/// 3. Bisects the stratum along the axis with the smallest `σ_a + σ_b` (the
///    lowest such axis on ties), and gives the lower half
///    `n_a = ⌊n σ_a / (σ_a + σ_b)⌋` samples, clamped to `[2, n − 2]`, and the
///    upper half `n − n_a`. If no axis has at least 2 pilot points in each half,
///    the stratum is not split.
///
/// Training ends when there are `max_leaves` strata or no stratum can be
/// selected. Each split counts as one subdivision toward the scope's budget, and
/// each pilot point as one observation, as in
/// [`MonteCarloIntegrator::integrate_into`](crate::MonteCarloIntegrator::integrate_into).
/// Training makes at most `max_leaves − 1` splits and costs at most `p` calls of
/// `f` per visited stratum. Pilot points do not enter the later estimate.
///
/// The pilot uniforms come from a [`RandomStreams`]; pilot visit `v` uses the
/// sample blocks after those of visits `0..v`. The strata do not depend on the
/// execution policy or the number of workers.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{MiserOptions, MiserTrainer, RandomStreams};
/// use scioxide_random::{Pcg64, SeedableRng};
/// let root = Pcg64::from_seed(11);
/// let plan = MiserTrainer::new(&[Interval::new(-1.0, 1.0)?], MiserOptions::default())?;
/// let f = |x: &[f64]| Ok(x[0].max(0.0));
/// let (strata, report) = plan.train(&f, RandomStreams::new(&root, 0))?;
/// // Each split adds one stratum, and the strata share all 4,096 samples.
/// assert!(report.updates() > 0 && strata.len() == report.updates() + 1);
/// assert_eq!(strata.samples().iter().sum::<usize>(), 4096);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// W. H. Press and G. R. Farrar, "Recursive stratified sampling for
/// multidimensional Monte Carlo integration", *Computers in Physics* 4(2),
/// 190–195, 1990.
#[derive(Debug, Clone)]
pub struct MiserTrainer<S: RealScalar> {
    bounds: Vec<Interval<S>>,
    options: MiserOptions,
}

/// Workspace of [`MiserTrainer::train_into`].
///
/// Create one with [`MiserTrainer::workspace`] for a fixed number of real value
/// components. It serves plans with the same dimension, maximum number of
/// strata and components, at most as many workers, and no more pilot points per
/// sample block. Each worker stores `min(p, SAMPLE_BLOCK)` values per half-box
/// and component, for `p` pilot samples.
#[derive(Debug)]
pub struct MiserWorkspace<S: RealScalar> {
    attempts: Attempts<TrainingReport>,
    lanes: Vec<Lane<S, Halves<S>>>,
    moments: Vec<OnlineMoments<S>>,
    midpoints: Vec<S>,
    deviations: Vec<Option<S>>,
    leaves: Vec<Interval<S>>,
    samples: Vec<usize>,
    depths: Vec<usize>,
    priorities: Vec<Option<Scaled<S>>>,
    sequences: Vec<usize>,
    components: usize,
}

impl<S: RealScalar> MiserWorkspace<S> {
    /// Returns the record of the last training call with this workspace: its
    /// outcome, solve counts and, if it returned one, its report. Returns `None`
    /// before the first call.
    #[must_use]
    pub fn last_attempt(&self) -> Option<scioxide_core::AttemptRecord<TrainingReport>> {
        self.attempts.last_attempt()
    }
}

impl<S: RealScalar> MiserTrainer<S> {
    /// Prepares MISER training for the box `bounds`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the evaluation options are invalid (see
    ///   [`MonteCarloOptions`]) or request more than one replicate, or if there are
    ///   fewer than 4 pilot samples.
    /// - `SizeOverflow` if the size of the strata or the number of sample blocks
    ///   overflows.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn new(bounds: &[Interval<S>], options: MiserOptions) -> Result<Self, SciOxideError> {
        options.evaluation.validate()?;
        if options.evaluation.replicates() != 1 {
            return Err(super::invalid(
                scioxide_core::argument!("evaluation"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "stratified evaluation has exactly one replicate",
            ));
        }
        if options.pilot_samples.get() < 4 {
            return Err(super::invalid(
                scioxide_core::argument!("pilot_samples"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "MISER needs at least four pilot samples",
            ));
        }
        scioxide_core::backend::checked_mul(
            bounds.len().max(1),
            options.max_leaves.get(),
            scioxide_core::quantity!("miser_strata"),
        )?;
        kernel::offset(
            options.max_leaves.get(),
            Schedule::blocks(options.pilot_samples.get()),
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
    pub const fn options(&self) -> &MiserOptions {
        &self.options
    }

    /// Returns the algorithm identifier `MiserTraining` recorded in solve diagnostics.
    pub const fn algorithm_id(&self) -> AlgorithmId {
        scioxide_core::algorithm_id!("integrate", "MiserTraining", 1)
    }

    /// Returns the box as a single stratum with all evaluation samples, with room
    /// for `max_leaves` strata.
    ///
    /// Use it as the output of [`train_into`](Self::train_into).
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn initial_strata(&self) -> Result<MiserStrata<S>, SciOxideError> {
        let leaves = self.options.max_leaves.get();
        let mut bounds = scioxide_core::backend::try_with_capacity(
            self.bounds.len(),
            scioxide_core::quantity!("bounds"),
        )?;
        bounds.extend_from_slice(&self.bounds);
        let mut strata = MiserStrata {
            bounds,
            leaves: try_filled(
                self.bounds.len() * leaves,
                Interval::new(S::ZERO, S::ZERO)?,
                scioxide_core::quantity!("miser_strata"),
            )?,
            samples: try_filled(leaves, 0, scioxide_core::quantity!("miser_samples"))?,
            count: 1,
            options: self.options.evaluation,
        };
        strata.leaves[..self.bounds.len()].copy_from_slice(&self.bounds);
        strata.samples[0] = self.options.evaluation.samples().get();
        Ok(strata)
    }

    /// Creates a training workspace for `components` real value components and
    /// the workers that the execution policy and `execution` allow.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(
        &self,
        components: usize,
        execution: ExecutionContext<'_>,
    ) -> Result<MiserWorkspace<S>, SciOxideError> {
        let dimension = self.bounds.len();
        let leaves = self.options.max_leaves.get();
        let halves = scioxide_core::backend::checked_mul(
            2,
            dimension,
            scioxide_core::quantity!("miser_moments"),
        )?;
        let moment_count = scioxide_core::backend::checked_mul(
            halves,
            components,
            scioxide_core::quantity!("miser_components"),
        )?;
        let empty = OnlineMoments::new(MomentOrder::Second);
        let stride = self.options.pilot_samples.get().min(super::SAMPLE_BLOCK);
        Ok(MiserWorkspace {
            attempts: Attempts::default(),
            lanes: Lane::allocate_many(
                kernel::lanes(
                    self.options.evaluation.execution_policy(),
                    self.options
                        .pilot_samples
                        .get()
                        .div_ceil(super::SAMPLE_BLOCK),
                    execution,
                ),
                dimension,
                || {
                    Ok(Halves {
                        moments: try_filled(
                            moment_count,
                            empty,
                            scioxide_core::quantity!("miser_moments"),
                        )?,
                        values: try_filled(
                            scioxide_core::backend::checked_mul(
                                moment_count,
                                stride,
                                scioxide_core::quantity!("miser_values"),
                            )?,
                            S::ZERO,
                            scioxide_core::quantity!("miser_values"),
                        )?,
                        counts: try_filled(halves, 0, scioxide_core::quantity!("miser_counts"))?,
                        single: false,
                        stride,
                        components,
                    })
                },
            )?,
            moments: try_filled(
                moment_count,
                empty,
                scioxide_core::quantity!("miser_moments"),
            )?,
            deviations: try_filled(halves, None, scioxide_core::quantity!("miser_deviations"))?,
            midpoints: try_filled(
                dimension,
                S::ZERO,
                scioxide_core::quantity!("miser_midpoints"),
            )?,
            leaves: try_filled(
                dimension * leaves,
                Interval::new(S::ZERO, S::ZERO)?,
                scioxide_core::quantity!("miser_strata"),
            )?,
            samples: try_filled(leaves, 0, scioxide_core::quantity!("miser_samples"))?,
            depths: try_filled(leaves, 0, scioxide_core::quantity!("miser_depths"))?,
            priorities: try_filled(leaves, None, scioxide_core::quantity!("miser_priorities"))?,
            sequences: try_filled(leaves, 0, scioxide_core::quantity!("miser_sequences"))?,
            components,
        })
    }

    /// Trains strata for `f` and writes them to `strata`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `strata` must have this plan's dimension and room for `max_leaves` strata,
    /// for example from [`initial_strata`](Self::initial_strata); its previous
    /// content is not used. `f` must return the workspace's number of components.
    ///
    /// A budget, cancellation or deadline can stop training. A stop after at
    /// least one split returns `Ok`, writes the strata made so far and reports
    /// the stop in the [`TrainingReport`]. A box of dimension zero or zero width,
    /// or a workspace for zero components, writes the box as one stratum without
    /// calling `f`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `strata` or `workspace` does not match this plan,
    ///   if the workspace has too few workers for `scope`, or if a value of `f`
    ///   does not have the workspace's number of components.
    /// - `NonFinite` (origin `Callback`) if a component of `f` is NaN or infinite.
    /// - `ConvergenceFailure` if a budget stops training before the first split.
    /// - `ExecutionStopped` if a cancellation or deadline stops training before
    ///   the first split.
    /// - Errors returned by `f`.
    pub fn train_into<'scope, 'capability, F, V, R>(
        &self,
        f: F,
        streams: RandomStreams<'_, R>,
        strata: &mut MiserStrata<S>,
        workspace: &mut MiserWorkspace<S>,
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
        let capacity = self.options.max_leaves.get();
        let workers = kernel::lanes(
            self.options.evaluation.execution_policy(),
            self.options
                .pilot_samples
                .get()
                .div_ceil(super::SAMPLE_BLOCK),
            scope.execution(),
        );
        let components = workspace.components;
        let preflight = crate::kernel::shape(dimension, strata.bounds.len())
            .and_then(|()| crate::kernel::shape(capacity, strata.samples.len()))
            .and_then(|()| crate::kernel::shape(capacity, workspace.samples.len()))
            .and_then(|()| crate::kernel::shape(dimension * capacity, workspace.leaves.len()))
            .and_then(|()| {
                crate::kernel::shape(2 * dimension * components, workspace.moments.len())
            })
            .and_then(|()| {
                let required = self.options.pilot_samples.get().min(super::SAMPLE_BLOCK);
                for lane in &workspace.lanes {
                    if lane.summary.stride < required {
                        crate::kernel::shape(required, lane.summary.stride)?;
                    }
                }
                Ok(())
            })
            .and_then(|()| {
                if workspace.lanes.len() < workers {
                    crate::kernel::shape(workers, workspace.lanes.len())
                } else {
                    Ok(())
                }
            });
        let preflight = preflight.and_then(|()| {
            if let Some(expected) = V::FIXED_DIMENSION {
                crate::kernel::shape(components, expected)?;
            }
            Ok(())
        });
        if let Err(error) = preflight {
            return workspace.attempts.reject(error);
        }
        let components = V::FIXED_DIMENSION.unwrap_or(components);
        let MiserWorkspace {
            attempts,
            lanes,
            moments,
            midpoints,
            deviations,
            leaves,
            samples,
            depths,
            priorities,
            sequences,
            ..
        } = workspace;
        let publish = |strata: &mut MiserStrata<S>,
                       leaves: &[Interval<S>],
                       samples: &[usize],
                       count: usize| {
            strata.bounds.copy_from_slice(&self.bounds);
            strata.leaves[..count * dimension].copy_from_slice(&leaves[..count * dimension]);
            strata.samples[..count].copy_from_slice(&samples[..count]);
            strata.count = count;
            strata.options = self.options.evaluation;
        };
        train(attempts, scope, self.algorithm_id(), dimension, |context| {
            let lanes = &mut lanes[..workers];
            leaves[..dimension].copy_from_slice(&self.bounds);
            samples[0] = self.options.evaluation.samples().get();
            depths[0] = 0;
            priorities.fill(None);
            priorities[0] = Some(Scaled::ZERO);
            sequences[0] = 0;
            let mut count = 1;
            if dimension == 0
                || components == 0
                || self.bounds.iter().any(|b| b.lower() == b.upper())
            {
                publish(strata, leaves, samples, count);
                return Ok(Trained {
                    termination: if dimension == 0 || components == 0 {
                        IntegrationTermination::FixedRuleCompleted
                    } else {
                        IntegrationTermination::DegenerateDomain
                    },
                    updates: 0,
                });
            }
            let job = Job {
                source: &streams,
                proposal: None::<&UniformProposal>,
                function: f,
                dimension,
                uniforms: dimension,
            };
            let pilot = self.options.pilot_samples.get();
            let single = pilot <= super::SAMPLE_BLOCK;
            for lane in lanes.iter_mut() {
                lane.summary.single = single;
            }
            let mut termination = IntegrationTermination::SampleBudgetCompleted;
            let mut updates = 0;
            let mut visit = 0;
            while count < capacity {
                let index = (0..count)
                    .filter(|&i| {
                        priorities[i].is_some()
                            && depths[i] < self.options.max_depth
                            && samples[i] >= 4
                    })
                    .max_by(|&a, &b| {
                        priorities[a]
                            .unwrap_or(Scaled::ZERO)
                            .abs_cmp(priorities[b].unwrap_or(Scaled::ZERO))
                            .then_with(|| sequences[b].cmp(&sequences[a]))
                    });
                let Some(index) = index else {
                    break;
                };
                priorities[index] = None;
                let leaf = &leaves[index * dimension..(index + 1) * dimension];
                let volume = prepare(lanes, leaf)?;
                for (middle, bound) in midpoints.iter_mut().zip(leaf) {
                    *middle = bound.midpoint()?;
                }
                if !single {
                    moments.fill(OnlineMoments::new(MomentOrder::Second));
                }
                let progress = sample(
                    &job,
                    Schedule::uniform(visit, pilot)?,
                    lanes,
                    self.options.evaluation.execution_policy(),
                    context,
                    &halves::<S, V>(midpoints),
                    &mut |lane: &Halves<S>| {
                        if single {
                            if components == 1 {
                                for ((deviation, &count), values) in deviations
                                    .iter_mut()
                                    .zip(&lane.counts)
                                    .zip(lane.values.chunks_exact(lane.stride))
                                {
                                    *deviation = if count < 2 {
                                        None
                                    } else {
                                        Some(scioxide_stats::sample_standard_deviation(
                                            &values[..count],
                                        )?)
                                    };
                                }
                                return Ok(());
                            }
                            for (half, deviation) in deviations.iter_mut().enumerate() {
                                let count = lane.counts[half];
                                *deviation = dispersion((0..components).map(|component| {
                                    if count < 2 {
                                        return Ok(None);
                                    }
                                    let index = half * components + component;
                                    let values = &lane.values[index * lane.stride..][..count];
                                    scioxide_stats::sample_standard_deviation(values).map(Some)
                                }))?;
                            }
                            Ok(())
                        } else {
                            moments
                                .iter_mut()
                                .zip(&lane.moments)
                                .try_for_each(|(total, half)| total.merge(half))
                        }
                    },
                )?;
                visit += 1;
                if let Some(stop) = progress.stop {
                    termination = stop;
                    break;
                }
                if !single {
                    for (half, deviation) in deviations.iter_mut().enumerate() {
                        *deviation = if components == 1 {
                            moments[half].snapshot().standard_deviation(1)?
                        } else {
                            dispersion(
                                moments[half * components..][..components]
                                    .iter()
                                    .map(|moment| moment.snapshot().standard_deviation(1)),
                            )?
                        };
                    }
                }
                let Some((axis, fraction)) = best_split(deviations, leaf, midpoints)? else {
                    continue;
                };
                if !context
                    .can_consume(BudgetKind::Subdivision, 1)?
                    .is_granted()
                {
                    termination = IntegrationTermination::BudgetLimit(BudgetKind::Subdivision);
                    break;
                }
                let n = samples[index];
                let left_samples =
                    ((n as f64 * fraction.into_f64()).floor() as usize).clamp(2, n - 2);
                let bound = leaves[index * dimension + axis];
                let middle = midpoints[axis];
                let (left_bound, right_bound) = (
                    Interval::new(bound.lower(), middle)?,
                    Interval::new(middle, bound.upper())?,
                );
                context.try_consume(BudgetKind::Subdivision, 1)?.require()?;
                let (active, spare) = leaves.split_at_mut(count * dimension);
                let left = &mut active[index * dimension..(index + 1) * dimension];
                let right = &mut spare[..dimension];
                right.copy_from_slice(left);
                right[axis] = right_bound;
                left[axis] = left_bound;
                depths[index] += 1;
                depths[count] = depths[index];
                samples[index] = left_samples;
                samples[count] = n - left_samples;
                // Terminal leaves never enter another selection. Avoid
                // constructing variance keys that cannot be consumed.
                if count + 1 < capacity && depths[index] < self.options.max_depth {
                    let half_volume = volume.shifted(-1)?;
                    for (child, side) in [(index, 0), (count, 1)] {
                        if samples[child] >= 4 {
                            let deviation = deviations[2 * axis + side].unwrap_or(S::ZERO);
                            priorities[child] =
                                Some(variance_priority(half_volume, deviation, samples[child])?);
                            sequences[child] = 2 * updates + side + 1;
                        }
                    }
                }
                count += 1;
                updates += 1;
                context.commit_accepted(PartialResultTrust::Admissible)?;
            }
            if termination.is_success() || updates > 0 {
                publish(strata, leaves, samples, count);
            }
            Ok(Trained {
                termination,
                updates,
            })
        })
    }

    /// Trains strata for a scalar or fixed-size integrand on the calling thread.
    ///
    /// The component count comes from `V::default()`. Values whose length is
    /// known only at run time use [`Self::train_into`] with a caller-sized workspace.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// This allocates the strata and a workspace and calls
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
    ) -> Result<(MiserStrata<S>, TrainingReport), SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S> + Copy + Default,
        R: Rng<S> + SplittableRng + Sync,
    {
        let f = &f;
        let components = V::default().dimension()?;
        let mut strata = self.initial_strata()?;
        let mut workspace = self.workspace(components, ExecutionContext::sequential_native())?;
        let report = self.train_into(
            f,
            streams,
            &mut strata,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok((strata, report))
    }
}

/// Returns the variance contribution `(volume · deviation)² / samples` in extended
/// range. Normalized significands keep the products below 16, so even the
/// largest `usize` denominator leaves a normal quotient for `f32` and `f64`;
/// other scalar ranges use checked extended arithmetic.
fn variance_priority<S: RealScalar>(
    volume: Scaled<S>,
    deviation: S,
    samples: usize,
) -> Result<Scaled<S>, SciOxideError> {
    if deviation == S::ZERO {
        return Ok(Scaled::ZERO);
    }
    let deviation = Scaled::from_value(deviation)?;
    // Counts remain exact usize budgets; their heuristic priority scale uses
    // the same scalar projection as stochastic moment normalization.
    let count = S::try_from_f64_lossy(samples as f64)?;
    let exponent = volume
        .exponent()
        .checked_add(deviation.exponent())
        .and_then(|exponent| exponent.checked_mul(2));
    if let Some(exponent) = exponent {
        let product = volume.significand() * deviation.significand();
        let priority = (product * product) / count;
        if priority.is_normal() {
            return Scaled::from_parts(priority, exponent);
        }
    }
    volume
        .mul(deviation)?
        .square()?
        .div(Scaled::from_value(count)?)
}

/// Returns the Euclidean norm of the component standard deviations, or `None` if
/// any is missing.
fn dispersion<S: RealScalar>(
    components: impl Iterator<Item = Result<Option<S>, SciOxideError>>,
) -> Result<Option<S>, SciOxideError> {
    let mut squares = ScaledAccumulator::ZERO;
    for component in components {
        let Some(value) = component? else {
            return Ok(None);
        };
        squares.add_scaled(Scaled::from_value(value)?.square()?)?;
    }
    Ok(Some(squares.total()?.sqrt()?.round()?))
}

/// Returns the update that records each component value in the half-box of
/// every axis that contains the point.
fn halves<S: RealScalar, V: AffineValue<Scale = S>>(
    midpoints: &[S],
) -> impl Fn(&mut Halves<S>, Draw<'_, S>, &V) -> Result<(), SciOxideError> + Sync + '_ {
    #[inline(always)]
    move |halves, draw, value| {
        // Fixed shapes were admitted before sampling. Dynamic shapes still
        // need validation for each callback, before indexing prepared rows.
        let components = V::FIXED_DIMENSION.unwrap_or(halves.components);
        if V::FIXED_DIMENSION.is_none() {
            crate::kernel::shape(components, value.dimension()?)?;
        }
        let weighted_component = |component| {
            let value = crate::kernel::checked_callback_value(value.component(component)?)?;
            if draw.density == S::ONE {
                Ok(value)
            } else {
                Scaled::from_value(value)?
                    .div(Scaled::from_value(draw.density)?)?
                    .round()
            }
        };
        if components == 1 {
            let weighted = weighted_component(0)?;
            for (axis, (&x, &middle)) in draw.point.iter().zip(midpoints).enumerate() {
                let half = 2 * axis + usize::from(x >= middle);
                let count = halves.counts[half];
                halves.values[half * halves.stride + count] = weighted;
                halves.counts[half] = count + 1;
            }
            return Ok(());
        }
        for component in 0..components {
            let weighted = weighted_component(component)?;
            for (axis, (&x, bound)) in draw.point.iter().zip(midpoints).enumerate() {
                let half = 2 * axis + usize::from(x >= *bound);
                let index = half * components + component;
                halves.values[index * halves.stride + halves.counts[half]] = weighted;
                if component + 1 == components {
                    halves.counts[half] += 1;
                }
            }
        }
        Ok(())
    }
}

/// Returns the axis with the smallest sum of half standard deviations and the
/// lower half's share of that sum (one half when both are zero), or `None` when
/// no axis can be split.
fn best_split<S: RealScalar>(
    deviations: &[Option<S>],
    leaf: &[Interval<S>],
    midpoints: &[S],
) -> Result<Option<(usize, S)>, SciOxideError> {
    let mut ordinary = true;
    let mut native_best: Option<(usize, S, S)> = None;
    for (axis, ((halves, bound), &middle)) in deviations
        .chunks_exact(2)
        .zip(leaf)
        .zip(midpoints)
        .enumerate()
    {
        if middle <= bound.lower() || middle >= bound.upper() {
            continue;
        }
        let (Some(lower), Some(upper)) = (halves[0], halves[1]) else {
            continue;
        };
        let sum = lower + upper;
        if !sum.is_finite() || sum != S::ZERO && sum < S::MIN_POSITIVE {
            ordinary = false;
            break;
        }
        if native_best
            .as_ref()
            .is_none_or(|&(_, score, _)| sum < score)
        {
            native_best = Some((axis, sum, lower));
        }
    }
    if ordinary {
        return Ok(native_best
            .map(|(axis, sum, lower)| (axis, if sum == S::ZERO { S::HALF } else { lower / sum })));
    }
    let mut best: Option<(usize, Scaled<S>, S)> = None;
    for (axis, ((halves, bound), &middle)) in deviations
        .chunks_exact(2)
        .zip(leaf)
        .zip(midpoints)
        .enumerate()
    {
        if middle <= bound.lower() || middle >= bound.upper() {
            continue;
        }
        let (Some(lower), Some(upper)) = (halves[0], halves[1]) else {
            continue;
        };
        let lower = Scaled::from_value(lower)?;
        let sum = lower.add(Scaled::from_value(upper)?)?;
        let fraction = if sum.is_zero() {
            S::HALF
        } else {
            lower.div(sum)?.round()?
        };
        if best
            .as_ref()
            .is_none_or(|(_, score, _)| sum.abs_cmp(*score).is_lt())
        {
            best = Some((axis, sum, fraction));
        }
    }
    Ok(best.map(|(axis, _, fraction)| (axis, fraction)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn pilot_storage_is_bounded_by_workers_and_blocks() {
        use core::num::NonZeroUsize;
        use scioxide_core::{ExecutionPolicy, ThreadExecutor};
        let executor = ThreadExecutor::new(NonZeroUsize::new(3).unwrap());
        let execution = ExecutionContext::new(&executor);
        for (samples, expected) in [(64, 1), (5 * super::super::SAMPLE_BLOCK, 3)] {
            let plan = MiserTrainer::new(
                &[Interval::new(0.0_f64, 1.0).unwrap()],
                MiserOptions::default()
                    .with_evaluation(MonteCarloOptions::default().with_execution_policy(
                        ExecutionPolicy::Parallel(NonZeroUsize::new(8).unwrap()),
                    ))
                    .with_pilot_samples(Budget::try_from(samples).unwrap()),
            )
            .unwrap();
            assert_eq!(plan.workspace(2, execution).unwrap().lanes.len(), expected);
            assert_eq!(
                plan.workspace(2, ExecutionContext::sequential_native())
                    .unwrap()
                    .lanes
                    .len(),
                1
            );
        }
    }

    #[test]
    fn variance_priorities_keep_extreme_volumes_and_zero_dispersion() {
        macro_rules! check {
            ($s:ty) => {
                for exponent in [-1000, 0, 1000] {
                    let volume = Scaled::from_parts(1.5 as $s, exponent).unwrap();
                    // (1.5 * 2^e)^2 * 2^2 / 9 = 2^(2e), including
                    // volumes and variances outside the scalar's IEEE range.
                    let expected = Scaled::from_parts(1.0, 2 * exponent).unwrap();
                    assert_eq!(variance_priority(volume, 2.0, 9).unwrap(), expected);
                    assert_eq!(variance_priority(volume, 0.0, 9).unwrap(), Scaled::ZERO);
                    assert!(
                        variance_priority(volume, 2.0, 8)
                            .unwrap()
                            .abs_cmp(expected)
                            .is_gt()
                    );
                    assert!(
                        variance_priority(volume, 2.0, 10)
                            .unwrap()
                            .abs_cmp(expected)
                            .is_lt()
                    );
                }
            };
        }
        check!(f32);
        check!(f64);
    }
}
