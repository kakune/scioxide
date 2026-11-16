//! Monte Carlo plans: uniform, importance, replicated and stratified sampling.

use alloc::vec::Vec;

use scioxide_core::{
    AffineValue, AffineValueMut, AlgorithmId, Capacity, ExecutionContext, ExecutionPolicy,
    Interval, PartialResultTrust, RealScalar, Scaled, ScaledAccumulator, SciOxideError,
    SolveBudgets, SolveScope, check_value_shape, write_value,
};
use scioxide_stats::{MomentOrder, OnlineMoments};

use super::kernel::{self, Draw, Job, Schedule, each_child, prepare, sample};
use super::statistics::{Estimate, welch_satterthwaite};
use super::workspace::{Block, MonteCarloWorkspace};
use super::{
    MiserStrata, MonteCarloOptions, MonteCarloProposal, MonteCarloStatistics, UniformProposal,
    UniformSource, VarianceBasis,
};
use crate::report::{IntegrationReport, IntegrationTermination};
use crate::workspace::{Family, IntegrationRequirements, Outcome};

/// Monte Carlo integration of `∫_B f(x) dx` over a finite box `B`.
///
/// Each observation draws `m` uniforms `u` from a [`UniformSource`], maps them
/// with the proposal to a point `y` of the unit cube with density `q(y)`, maps
/// `y` affinely to `x ∈ B`, and records
///
/// ```text
/// Y = V f(x) / q(y),   V = Π_k (b_k − a_k),
/// ```
///
/// for every real component of `f`, where `B = [a_1, b_1) × … × [a_d, b_d)`. The
/// estimate is the mean of `Y`, with standard error and confidence interval in
/// [`MonteCarloStatistics`]. The standard error decreases as `N^(−1/2)` in the
/// number `N` of observations, independently of the dimension. See
/// [integrands](crate#integrands) for the value types `f` may return.
///
/// - [`new`](Self::new) samples uniformly (`q = 1`); [`importance`](Self::importance)
///   samples from any [`MonteCarloProposal`], such as a trained
///   [`VegasMap`](crate::VegasMap).
/// - With one replicate the estimate is the mean of `N` independent observations.
///   With `R ≥ 2` replicates, or with a QMC source, it is the mean of `R`
///   independent replicate estimates, and the standard error comes from their
///   spread.
/// - [`stratified`](Self::stratified) samples each of the [`MiserStrata`]
///   uniformly and adds the stratum estimates and their variances.
///
/// For a [`RandomStreams`](crate::RandomStreams) source, the result does not
/// depend on the execution policy or the number of workers. A box of dimension
/// zero evaluates `f` once at the empty point and returns
/// [`IntegrationTermination::FixedRuleCompleted`]. A box of zero width returns
/// zero with [`IntegrationTermination::DegenerateDomain`] without calling `f`.
///
/// Integrators are immutable and can be shared between threads. There is no
/// `integrate` method that allocates its own workspace, because the statistics
/// are stored in the [`MonteCarloWorkspace`]. The scalar one-shot is
/// [`integrate_monte_carlo`].
///
/// # Examples
///
/// Integrate the complex function `x₁x₂ + i(x₁ + x₂)` over the unit square:
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext, Interval};
/// use scioxide_integrate::{MonteCarloOptions, MonteCarloIntegrator, RandomStreams};
/// use scioxide_random::{Philox4x64, SeedableRng};
/// let plan = MonteCarloIntegrator::new(&[Interval::new(0.0, 1.0)?; 2], MonteCarloOptions::default())?;
/// let cx = ExecutionContext::sequential_native();
/// let mut workspace = plan.workspace(2, cx)?;
/// let root = Philox4x64::from_seed(42);
/// let streams = RandomStreams::new(&root, 0);
/// let f = |x: &[f64]| Ok(Complex::new(x[0] * x[1], x[0] + x[1]));
/// let mut value = Complex::new(0.0_f64, 0.0);
/// let report = plan.integrate_into(&f, streams, &mut value, &mut workspace, cx)?;
/// assert_eq!(report.evaluations(), 4096);
/// // The exact integral is 1/4 + i.
/// let real = workspace.statistics()[0].as_ref().expect("statistics");
/// assert!((value.re - 0.25).abs() < 5.0 * real.standard_error());
/// assert!((value.im - 1.0).abs() < 0.05);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. B. Owen, *Monte Carlo theory, methods and examples*, 2013 (plain and
/// importance sampling, stratification, randomized QMC replicates).
#[derive(Debug, Clone)]
pub struct MonteCarloIntegrator<S: RealScalar, P = UniformProposal> {
    bounds: Vec<Interval<S>>,
    proposal: P,
    identity_map: bool,
    options: MonteCarloOptions,
    strata: Option<MiserStrata<S>>,
    offsets: Vec<u64>,
}

impl<S: RealScalar> MonteCarloIntegrator<S> {
    /// Prepares uniform sampling of the box `bounds`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the options are invalid; see [`MonteCarloOptions`].
    /// - `SizeOverflow` if the number of sample blocks of all replicates overflows.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn new(bounds: &[Interval<S>], options: MonteCarloOptions) -> Result<Self, SciOxideError> {
        let mut plan = Self::importance(bounds, UniformProposal::new(bounds.len()), options)?;
        plan.identity_map = true;
        Ok(plan)
    }

    /// Prepares stratified sampling of trained MISER strata.
    ///
    /// Stratum `c` receives `n_c` = `strata.samples()[c]` uniform observations
    /// with the evaluation options stored in `strata`. The estimate is the sum of
    /// the stratum estimates and its variance the sum of their variances, with
    /// Welch–Satterthwaite degrees of freedom (see [`VarianceBasis::Strata`]).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the number of sample blocks overflows.
    /// - `AllocationFailure` if the plan cannot be stored.
    ///
    /// # Examples
    ///
    /// Integrate a step function, which MISER splits at its jump:
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, Interval};
    /// use scioxide_integrate::{MiserOptions, MiserTrainer, MonteCarloIntegrator, RandomStreams, VarianceBasis};
    /// use scioxide_random::{Pcg64, SeedableRng};
    /// let root = Pcg64::from_seed(3);
    /// let f = |x: &[f64]| Ok(if x[0] < 0.25 { 8.0 } else { 0.0 });
    /// let miser = MiserTrainer::new(&[Interval::new(0.0, 1.0)?], MiserOptions::default())?;
    /// let (strata, _) = miser.train(&f, RandomStreams::new(&root, 0))?;
    /// let plan = MonteCarloIntegrator::stratified(strata)?;
    /// let cx = ExecutionContext::sequential_native();
    /// let mut workspace = plan.workspace(1, cx)?;
    /// let mut value = 0.0;
    /// let streams = RandomStreams::new(&root, 1);
    /// plan.integrate_into(&f, streams, &mut value, &mut workspace, cx)?;
    /// let stats = workspace.statistics()[0].as_ref().expect("statistics");
    /// assert_eq!(stats.basis(), VarianceBasis::Strata);
    /// // The exact integral is 8 · 1/4 = 2.
    /// assert!((value - 2.0).abs() < 5.0 * stats.standard_error() + 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn stratified(strata: MiserStrata<S>) -> Result<Self, SciOxideError> {
        let options = *strata.options();
        options.validate()?;
        if options.replicates() != 1 {
            return Err(super::invalid(
                scioxide_core::argument!("strata"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "stratified evaluation has exactly one replicate",
            ));
        }
        let mut offsets = scioxide_core::backend::try_with_capacity(
            strata.len(),
            scioxide_core::quantity!("strata"),
        )?;
        let mut next = 0_u64;
        for &samples in strata.samples() {
            offsets.push(next);
            next = u64::try_from(Schedule::blocks(samples))
                .ok()
                .and_then(|blocks| next.checked_add(blocks))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("sample_blocks"))
                })?;
        }
        let mut bounds = scioxide_core::backend::try_with_capacity(
            strata.bounds().len(),
            scioxide_core::quantity!("bounds"),
        )?;
        bounds.extend_from_slice(strata.bounds());
        Ok(Self {
            proposal: UniformProposal::new(bounds.len()),
            identity_map: true,
            bounds,
            options,
            strata: Some(strata),
            offsets,
        })
    }
}

impl<S: RealScalar, P: MonteCarloProposal<S>> MonteCarloIntegrator<S, P> {
    /// Prepares importance sampling of the box `bounds` with `proposal`.
    ///
    /// Use it with a trained [`VegasMap`](crate::VegasMap) or your own
    /// [`MonteCarloProposal`]. The plan checks every sampled density and point;
    /// see [`MonteCarloProposal::sample`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the options are invalid (see [`MonteCarloOptions`]), or
    ///   if the proposal consumes no uniforms on a box of positive dimension.
    /// - `DimensionMismatch` if the proposal's dimension differs from `bounds.len()`.
    /// - `SizeOverflow` if the number of sample blocks of all replicates overflows.
    /// - `AllocationFailure` if the plan cannot be stored.
    pub fn importance(
        bounds: &[Interval<S>],
        proposal: P,
        options: MonteCarloOptions,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        crate::kernel::shape(bounds.len(), proposal.dimension())?;
        if proposal.uniform_count() == 0 && !bounds.is_empty() {
            return Err(super::invalid(
                scioxide_core::argument!("proposal"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "a proposal on a nonempty box must consume uniforms",
            ));
        }
        kernel::offset(
            options.replicates(),
            Schedule::blocks(options.samples().get()),
        )?;
        let mut owned = scioxide_core::backend::try_with_capacity(
            bounds.len(),
            scioxide_core::quantity!("bounds"),
        )?;
        owned.extend_from_slice(bounds);
        Ok(Self {
            bounds: owned,
            proposal,
            identity_map: false,
            options,
            strata: None,
            offsets: Vec::new(),
        })
    }

    /// Returns the integration box.
    pub fn bounds(&self) -> &[Interval<S>] {
        &self.bounds
    }

    /// Returns the proposal.
    pub const fn proposal(&self) -> &P {
        &self.proposal
    }

    /// Returns the options.
    pub const fn options(&self) -> &MonteCarloOptions {
        &self.options
    }

    /// Returns the strata of a stratified plan, or `None`.
    #[must_use]
    pub const fn strata(&self) -> Option<&MiserStrata<S>> {
        self.strata.as_ref()
    }

    /// Returns the algorithm identifier recorded in solve diagnostics:
    /// `StratifiedMonteCarlo`, `MonteCarloReplicates` or `MonteCarlo`.
    pub fn algorithm_id(&self) -> AlgorithmId {
        if self.strata.is_some() {
            scioxide_core::algorithm_id!("integrate", "StratifiedMonteCarlo", 1)
        } else if self.options.replicates() > 1 {
            scioxide_core::algorithm_id!("integrate", "MonteCarloReplicates", 1)
        } else {
            scioxide_core::algorithm_id!("integrate", "MonteCarlo", 1)
        }
    }

    /// Returns whether the plan is one run rather than replicates or strata.
    fn independent(&self) -> bool {
        self.strata.is_none() && self.options.replicates() == 1
    }

    /// Returns the number of replicates or strata, or one for a single run.
    fn children(&self) -> usize {
        self.strata
            .as_ref()
            .map_or(self.options.replicates(), MiserStrata::len)
    }

    fn schedule(&self, child: usize) -> Result<Schedule, SciOxideError> {
        match &self.strata {
            Some(strata) => Ok(Schedule {
                child,
                offset: self.offsets[child],
                samples: strata.samples()[child],
            }),
            None => Schedule::uniform(child, self.options.samples().get()),
        }
    }

    fn child_bounds(&self, child: usize) -> &[Interval<S>] {
        match &self.strata {
            Some(strata) => strata.stratum(child),
            None => &self.bounds,
        }
    }

    /// Returns the workspace size for `components` real value components.
    ///
    /// The size grows with the number of replicates or strata and with the
    /// number of workers that the execution policy and `execution` allow.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        components: usize,
        execution: ExecutionContext<'_>,
    ) -> Result<IntegrationRequirements, SciOxideError> {
        let work = if self.independent() {
            self.options.samples().get().div_ceil(super::SAMPLE_BLOCK)
        } else {
            self.children()
        };
        let workers = kernel::lanes(self.options.execution_policy(), work, execution);
        let slots = self.children();
        let coordinates = self.bounds.len().max(self.proposal.uniform_count());
        scioxide_core::backend::checked_mul(
            slots.max(workers),
            coordinates.max(components).saturating_add(1),
            scioxide_core::quantity!("mc_workspace"),
        )?;
        Ok(
            IntegrationRequirements::new(Family::MonteCarlo, components, [slots, workers])
                .with_dimension(coordinates),
        )
    }

    /// Creates a workspace for `components` real value components.
    ///
    /// `components` is 1 for a real integrand, 2 for a complex one, and the
    /// length for a vector; see [integrands](crate#integrands).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(
        &self,
        components: usize,
        execution: ExecutionContext<'_>,
    ) -> Result<MonteCarloWorkspace<S>, SciOxideError> {
        MonteCarloWorkspace::allocate(&self.workspace_requirements(components, execution)?)
    }

    /// Estimates the integral of `f`, drawing uniforms from `source`, and writes
    /// it to `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `out` must have the workspace's number of components. The statistics of
    /// the estimate are stored in the workspace; read them with
    /// [`MonteCarloWorkspace::statistics`]. The report counts the calls of `f`.
    ///
    /// Each observation counts as one sample, one iteration and one proposal, and
    /// as `m + 1` evaluations, where `m` is the proposal's `uniform_count()`: `m`
    /// uniforms and one call of `f` (see
    /// [solve accounting](scioxide_core#solve-accounting)). Replicates and strata
    /// run in child solves that share the scope's budgets in index order.
    ///
    /// A budget, cancellation or deadline can stop the call. A single run stopped
    /// after at least one observation returns `Ok` with the mean so far and an
    /// unsuccessful termination, [`IntegrationTermination::BudgetLimit`] or
    /// [`IntegrationTermination::ExecutionStopped`]. Replicates and strata return
    /// an estimate only when every child completes. The result of each child
    /// stays available through [`MonteCarloWorkspace::children`] after a stop.
    /// See [termination](crate#termination).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` or a value of `f` does not have the
    ///   workspace's number of components, if the workspace was made for a plan of
    ///   another dimension or is too small for this plan and `scope`, or if a QMC
    ///   source has the wrong dimension.
    /// - `InvalidInput` if a QMC source does not match the replicates and samples
    ///   (see [`UniformSource`]), or if the proposal returns an invalid point or
    ///   density (see [`MonteCarloProposal::sample`]).
    /// - `NonFinite` (origin `Callback`) if `f` returns NaN or infinity.
    /// - `ConvergenceFailure` if a budget stops a single run before its first
    ///   observation, or stops any replicate or stratum.
    /// - `ExecutionStopped` if a cancellation or deadline stops the call in the
    ///   same cases.
    /// - Errors returned by `f`, the proposal or the source.
    pub fn integrate_into<'scope, 'capability, F, V, Src, O>(
        &self,
        f: F,
        source: Src,
        out: &mut O,
        workspace: &mut MonteCarloWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        P: Sync,
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        Src: UniformSource<S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let f = &f;
        let scope = scope.into();
        workspace.reset();
        let components = workspace.requirements.components();
        let preflight = check_value_shape(out, components)
            .and_then(|()| self.check_capacity(workspace, scope.execution()))
            .and_then(|()| {
                source.check(
                    if self.strata.is_some() {
                        1
                    } else {
                        self.options.replicates()
                    },
                    self.options.samples().get(),
                    self.proposal.uniform_count(),
                )
            });
        if let Err(error) = preflight {
            return workspace.attempts.reject(error);
        }
        let report = {
            let MonteCarloWorkspace {
                attempts,
                lanes,
                children,
                solves,
                values,
                statistics,
                active,
                ..
            } = workspace;
            attempts.integrate(
                scope,
                self.algorithm_id(),
                components,
                SolveBudgets::unbounded(),
                |context| {
                    let dimension = self.bounds.len();
                    let degenerate = self.bounds.iter().any(|b| b.lower() == b.upper());
                    if dimension > 0 && (components == 0 || degenerate) {
                        values.fill(S::ZERO);
                        context.commit_accepted(PartialResultTrust::Admissible)?;
                        write_value(values, out)?;
                        return Ok(Outcome::Published {
                            termination: if degenerate {
                                IntegrationTermination::DegenerateDomain
                            } else {
                                IntegrationTermination::FixedRuleCompleted
                            },
                            subdivisions: 0,
                            errors: None,
                        });
                    }
                    if dimension == 0 {
                        if let Some(stop) = crate::workspace::admit(context, 1, false)? {
                            return Ok(Outcome::Unpublished(stop));
                        }
                        crate::evaluation::sample_rows(
                            1,
                            components,
                            values,
                            ExecutionPolicy::Sequential,
                            context,
                            &|_, row| crate::evaluation::write_row(&f(&[])?, row),
                        )?;
                        context.commit_accepted(PartialResultTrust::Admissible)?;
                        write_value(values, out)?;
                        return Ok(Outcome::Published {
                            termination: IntegrationTermination::FixedRuleCompleted,
                            subdivisions: 0,
                            errors: None,
                        });
                    }
                    let job = Job {
                        source: &source,
                        // Uniform plans map admitted coordinates directly.
                        proposal: if self.identity_map {
                            None
                        } else {
                            Some(&self.proposal)
                        },
                        function: f,
                        dimension,
                        uniforms: self.proposal.uniform_count(),
                    };
                    let count = self.children();
                    *active = count;
                    let independent = self.independent();
                    let termination = if independent {
                        let termination = self.single(&job, lanes, &mut children[0], context)?;
                        children[0].finalize_statistics(&self.options)?;
                        termination
                    } else {
                        let outcome = each_child(
                            lanes,
                            &mut children[..count],
                            &mut solves[..count],
                            self.options.execution_policy(),
                            context,
                            self.algorithm_id(),
                            &|index, lane, child, context| {
                                child.clear();
                                let volume =
                                    prepare(core::slice::from_mut(lane), self.child_bounds(index))?;
                                let progress = sample(
                                    &job,
                                    self.schedule(index)?,
                                    core::slice::from_mut(lane),
                                    ExecutionPolicy::Sequential,
                                    context,
                                    &accumulator::<S, V, Src>(volume),
                                    &mut |block| child.total.merge(block),
                                )?;
                                child.finish(progress, Src::INDEPENDENT)?;
                                if child.total.observations() > 0 {
                                    context.commit_accepted(PartialResultTrust::Admissible)?;
                                }
                                Ok(())
                            },
                        );
                        let finalized = children[..count]
                            .iter_mut()
                            .try_for_each(|child| child.finalize_statistics(&self.options));
                        outcome?;
                        finalized?;
                        self.aggregate(&children[..count], values, statistics)?
                    };
                    let Some(termination) = termination else {
                        return Ok(Outcome::Unpublished(
                            children[..count]
                                .iter()
                                .filter_map(super::workspace::Child::termination)
                                .find(|t| *t != IntegrationTermination::SampleBudgetCompleted)
                                .unwrap_or(IntegrationTermination::SampleBudgetCompleted),
                        ));
                    };
                    if independent {
                        values.copy_from_slice(children[0].values());
                        statistics.clone_from_slice(children[0].statistics());
                    }
                    context.commit_accepted(PartialResultTrust::Admissible)?;
                    write_value(values, out)?;
                    Ok(Outcome::Published {
                        termination,
                        subdivisions: if self.strata.is_some() { count - 1 } else { 0 },
                        errors: None,
                    })
                },
            )
        };
        if report.is_err() {
            workspace.statistics.fill(None);
        }
        report
    }

    fn check_capacity(
        &self,
        workspace: &MonteCarloWorkspace<S>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let required =
            self.workspace_requirements(workspace.requirements.components(), execution)?;
        if workspace.check(&required)? == Capacity::Sufficient {
            return Ok(());
        }
        let ([slots, workers], [have_slots, have_workers]) =
            (required.units(), workspace.requirements.units());
        if required.dimension() != workspace.requirements.dimension() {
            crate::kernel::shape(required.dimension(), workspace.requirements.dimension())
        } else if slots > have_slots {
            crate::kernel::shape(slots, have_slots)
        } else {
            crate::kernel::shape(workers, have_workers)
        }
    }

    /// Runs the single estimate; returns `None` when nothing was observed.
    fn single<Src, F, V>(
        &self,
        job: &Job<'_, Src, P, F>,
        lanes: &mut [kernel::Lane<S, Block<S>>],
        child: &mut super::workspace::Child<S>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Option<IntegrationTermination>, SciOxideError>
    where
        P: Sync,
        Src: UniformSource<S>,
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: AffineValue<Scale = S>,
    {
        let workers = kernel::lanes(
            self.options.execution_policy(),
            self.options.samples().get().div_ceil(super::SAMPLE_BLOCK),
            context.execution(),
        );
        let lanes = &mut lanes[..workers];
        child.clear();
        let volume = prepare(lanes, &self.bounds)?;
        let progress = sample(
            job,
            self.schedule(0)?,
            lanes,
            self.options.execution_policy(),
            context,
            &accumulator::<S, V, Src>(volume),
            &mut |block| child.total.merge(block),
        )?;
        child.finish(progress, Src::INDEPENDENT)?;
        Ok((child.total.observations() > 0).then(|| {
            progress
                .stop
                .unwrap_or(IntegrationTermination::SampleBudgetCompleted)
        }))
    }

    /// Combines complete children; returns `None` when any child stopped early.
    fn aggregate(
        &self,
        children: &[super::workspace::Child<S>],
        values: &mut [S],
        statistics: &mut [Option<MonteCarloStatistics<S>>],
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        for child in children {
            match child.termination() {
                Some(IntegrationTermination::SampleBudgetCompleted) => {}
                Some(_) => return Ok(None),
                None => {
                    return Err(super::invalid(
                        scioxide_core::argument!("policy"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "a child estimate is missing",
                    ));
                }
            }
        }
        let samples = children.iter().try_fold(0usize, |total, child| {
            total
                .checked_add(child.total.observations())
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("observations"))
                })
        })?;
        for (component, (value, statistic)) in
            values.iter_mut().zip(statistics.iter_mut()).enumerate()
        {
            let estimate = match &self.strata {
                Some(strata) => self.strata_estimate(children, strata, component, samples)?,
                None => {
                    let mut moments = OnlineMoments::new(MomentOrder::Second);
                    for child in children {
                        moments.observe(child.values()[component])?;
                    }
                    MonteCarloStatistics::from_moments(
                        VarianceBasis::Replicates,
                        moments.snapshot(),
                        samples,
                        None,
                        &self.options,
                    )?
                }
            };
            *value = estimate.estimate();
            *statistic = Some(estimate);
        }
        Ok(Some(IntegrationTermination::SampleBudgetCompleted))
    }
}

impl<S: RealScalar, P> MonteCarloIntegrator<S, P> {
    /// Sums the stratum estimates and variances, with Welch–Satterthwaite degrees of
    /// freedom.
    fn strata_estimate(
        &self,
        children: &[super::workspace::Child<S>],
        strata: &MiserStrata<S>,
        component: usize,
        samples: usize,
    ) -> Result<MonteCarloStatistics<S>, SciOxideError> {
        let mut total = ScaledAccumulator::ZERO;
        for child in children {
            total.add_value(child.values()[component])?;
        }
        let (variance, degrees) = welch_satterthwaite(
            children
                .iter()
                .zip(strata.samples())
                .map(|(child, &samples)| {
                    let variance = child.statistics()[component]
                        .as_ref()
                        .map(|statistics| statistics.estimator_variance())
                        .ok_or_else(|| {
                            super::invalid(
                                scioxide_core::argument!("strata"),
                                scioxide_core::InvalidInputReason::Inconsistent,
                                "every stratum needs an observation variance",
                            )
                        })?;
                    Ok((variance, samples))
                }),
            samples.saturating_sub(children.len()),
        )?;
        Ok(MonteCarloStatistics::new(
            Estimate {
                basis: VarianceBasis::Strata,
                units: children.len(),
                samples,
                moments: None,
                value: total.round()?,
                estimator_variance: variance,
                standard_error: variance.sqrt()?.round()?,
                degrees_of_freedom: Some(degrees),
                effective_sample_size: None,
            },
            &self.options,
        ))
    }
}

/// Returns the per-observation update of a block summary for a box of `volume`.
fn accumulator<S: RealScalar, V: AffineValue<Scale = S>, Src: super::Source<S>>(
    volume: Scaled<S>,
) -> impl Fn(&mut Block<S>, Draw<'_, S>, &V) -> Result<(), SciOxideError> + Sync {
    let native = volume
        .round()
        .ok()
        .filter(|v| v.is_finite() && v.abs() >= S::MIN_POSITIVE);
    #[inline(always)]
    move |block, draw, value| block.observe::<Src, V>(&draw, value, volume, native)
}

/// Estimates the scalar integral of f over a finite, positive-volume box.
///
/// Draws 4096 samples from stream 0 of rng without advancing the root. Reusing
/// the root reproduces the statistics. For independent runs use
/// [`MonteCarloIntegrator`] with [`RandomStreams`](crate::RandomStreams) and
/// distinct stream numbers. Work is O(4096d) plus callbacks; the standard error
/// decreases as N⁻¹ᐟ² and the default interval is 95% Student t.
///
/// # Errors
///
/// Rejects mismatched corners, non-finite or reversed bounds, dimension zero
/// and zero-width axes before calling f. Propagates generator, callback and
/// allocation errors and converts unsuccessful termination to an error.
///
/// ```
/// use scioxide_random::{Philox4x64, SeedableRng};
/// let rng = Philox4x64::from_seed(7);
/// let estimate = scioxide_integrate::integrate_monte_carlo(
///     |_: &[f64]| Ok(2.0), &[0.0], &[1.0], &rng,
/// )?;
/// assert!((estimate.estimate() - 2.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. B. Owen, Monte Carlo Theory, Methods and Examples (2013), chapter 2.
pub fn integrate_monte_carlo<S, F, R>(
    f: F,
    lower: &[S],
    upper: &[S],
    rng: &R,
) -> Result<MonteCarloStatistics<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<S, SciOxideError> + Sync,
    R: scioxide_random::Rng<S> + scioxide_random::SplittableRng + Sync,
{
    crate::kernel::shape(lower.len(), upper.len())?;
    let mut bounds =
        scioxide_core::backend::try_with_capacity(lower.len(), scioxide_core::quantity!("bounds"))?;
    for (&a, &b) in lower.iter().zip(upper) {
        bounds.push(scioxide_core::Interval::new(a, b)?);
    }
    if bounds.is_empty() {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("lower"),
            scioxide_core::InvalidInputReason::Empty,
            "Monte Carlo integration requires an axis",
        ));
    }
    let no_statistics = || {
        crate::kernel::invalid(
            scioxide_core::argument!("upper"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Monte Carlo integration requires positive-width axes",
        )
    };
    if bounds.iter().any(|bound| bound.lower() == bound.upper()) {
        return Err(no_statistics());
    }
    let plan = MonteCarloIntegrator::new(&bounds, MonteCarloOptions::default())?;
    let context = scioxide_core::ExecutionContext::sequential_native();
    let mut workspace = plan.workspace(1, context)?;
    let mut value = S::ZERO;
    let report = plan.integrate_into(
        &f,
        crate::RandomStreams::new(rng, 0),
        &mut value,
        &mut workspace,
        context,
    )?;
    crate::workspace::require_success(
        report.termination(),
        plan.algorithm_id(),
        workspace
            .last_attempt()
            .map_or_else(scioxide_core::SolveCounts::new, |attempt| attempt.counts()),
    )?;
    workspace.statistics()[0].clone().ok_or_else(no_statistics)
}
