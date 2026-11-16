//! Sources of uniforms for Monte Carlo plans: random streams and QMC replicates.

use super::ProposalUniforms;
use scioxide_core::{RealScalar, SciOxideError};
use scioxide_random::{QmcReplicatePlan, Rng, SplittableRng, fill_uniforms};

mod sealed {
    use super::ProposalUniforms;
    use scioxide_core::{RealScalar, SciOxideError};

    /// Sealed methods of [`super::UniformSource`].
    pub trait Source<S: RealScalar>: Sync {
        /// Generator state of one sample block.
        type Block;
        /// Whether observations are independent, so their variance estimates the error.
        const INDEPENDENT: bool;
        /// Checks the source against a plan's replicates, samples and uniforms per point.
        fn check(
            &self,
            replicates: usize,
            samples: usize,
            uniforms: usize,
        ) -> Result<(), SciOxideError>;
        /// Opens global block `block` of child `child`.
        fn open(&self, child: usize, block: u64) -> Result<Self::Block, SciOxideError>;
        /// Fills and checks the uniforms of observation `index` of `child`, adding the
        /// number of uniforms drawn to `draws`.
        fn fill<'a>(
            &self,
            block: &mut Self::Block,
            child: usize,
            index: usize,
            uniforms: &'a mut [S],
            draws: &mut usize,
        ) -> Result<ProposalUniforms<'a, S>, SciOxideError>;
    }
}

pub(super) use sealed::Source;

/// Source of the uniforms in `[0, 1)` that a Monte Carlo plan maps to sample points.
///
/// The trait is sealed. Two sources implement it:
///
/// - [`RandomStreams`] draws pseudo-random uniforms from substreams of a
///   splittable generator. The observations are independent.
/// - `&`[`QmcReplicatePlan`] supplies randomized quasi-Monte Carlo points. Each
///   randomization is one replicate of the plan. The points within a replicate
///   are not independent, so the standard error comes only from the spread of
///   the replicate estimates, and a single replicate gives no statistics.
///
/// A QMC source must match the plan. [`integrate_into`](crate::MonteCarloIntegrator::integrate_into)
/// returns `InvalidInput` if the number of randomizations differs from
/// [`replicates`](crate::MonteCarloOptions::replicates) or the points per
/// replicate differ from [`samples`](crate::MonteCarloOptions::samples), and
/// `DimensionMismatch` if the point dimension differs from the proposal's
/// `uniform_count()`.
///
/// Each uniform drawn counts as one evaluation of the sample provider; see
/// [`MonteCarloIntegrator::integrate_into`](crate::MonteCarloIntegrator::integrate_into).
///
/// # Examples
///
/// Integrate `x y` over the unit square with 8 randomized Sobol' replicates:
///
/// ```
/// use scioxide_core::{Budget, ExecutionContext, Interval};
/// use scioxide_integrate::{MonteCarloOptions, MonteCarloIntegrator, VarianceBasis};
/// use scioxide_random::{Philox4x64, QmcPlan, QmcReplicateOptions, QmcReplicatePlan};
/// let replicates = QmcReplicateOptions::default()
///     .with_replicates(8)
///     .with_points_per_replicate(256);
/// let sobol = QmcPlan::sobol(2)?;
/// let qmc = QmcReplicatePlan::new(&sobol, &Philox4x64::from_key([7, 0]), replicates)?;
/// let options = MonteCarloOptions::default()
///     .with_samples(Budget::try_from(256)?)
///     .with_replicates(8);
/// let plan = MonteCarloIntegrator::new(&[Interval::new(0.0, 1.0)?; 2], options)?;
/// let cx = ExecutionContext::sequential_native();
/// let mut workspace = plan.workspace(1, cx)?;
/// let mut value = 0.0;
/// let f = |x: &[f64]| Ok(x[0] * x[1]);
/// plan.integrate_into(&f, &qmc, &mut value, &mut workspace, cx)?;
/// let stats = workspace.statistics()[0].as_ref().expect("statistics");
/// assert_eq!(stats.basis(), VarianceBasis::Replicates);
/// // The exact integral is 1/4.
/// assert!((value - 0.25).abs() < 1e-3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait UniformSource<S: RealScalar>: sealed::Source<S> {}

/// Pseudo-random uniforms from one stream of a splittable generator.
///
/// A run draws its observations in blocks of [`SAMPLE_BLOCK`](crate::SAMPLE_BLOCK)
/// `= B`, and block `b` draws its uniforms in order from
/// `root.substream(stream, b)`. Blocks are numbered consecutively over the whole
/// run. Replicate `r` of `n` observations uses blocks `r⌈n/B⌉` to
/// `(r + 1)⌈n/B⌉ − 1`; VEGAS iterations and MISER pilot visits are numbered the
/// same way. Stratum `c` of a stratified plan uses the blocks after those of
/// strata `0..c`.
///
/// The root is read but never advanced. The same root, `stream`, plan and
/// integrand give the same estimate, statistics and counts for every execution
/// policy and number of workers, unless a cancellation or deadline stops the
/// run. Distinct `stream` values select distinct substreams; their independence
/// depends on the generator (see [`SplittableRng`]).
///
/// Substream selection is absolute within the original root: a generator split
/// from the root, or advanced by drawing from it, selects the same substreams as
/// the root. Use distinct `stream` values, not split roots, for distinct runs.
/// Training and evaluation both start at block 0, so use different `stream`
/// values for them unless the evaluation should reuse the training uniforms.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::RandomStreams;
/// use scioxide_random::{Pcg64, SeedableRng};
/// let root = Pcg64::from_seed(42);
/// let streams = RandomStreams::new(&root, 3);
/// assert_eq!(streams.stream(), 3);
/// ```
#[derive(Debug)]
pub struct RandomStreams<'a, R> {
    root: &'a R,
    stream: u64,
}

impl<R> Clone for RandomStreams<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> Copy for RandomStreams<'_, R> {}

impl<'a, R> RandomStreams<'a, R> {
    /// Selects stream `stream` of `root`.
    #[must_use]
    pub const fn new(root: &'a R, stream: u64) -> Self {
        Self { root, stream }
    }

    /// Returns the root generator.
    #[must_use]
    pub const fn root(&self) -> &'a R {
        self.root
    }

    /// Returns the selected stream.
    #[must_use]
    pub const fn stream(&self) -> u64 {
        self.stream
    }
}

impl<S, R> sealed::Source<S> for RandomStreams<'_, R>
where
    S: RealScalar,
    R: Rng<S> + SplittableRng + Sync,
{
    type Block = R;
    const INDEPENDENT: bool = true;

    fn check(&self, _: usize, _: usize, _: usize) -> Result<(), SciOxideError> {
        Ok(())
    }

    fn open(&self, _: usize, block: u64) -> Result<R, SciOxideError> {
        self.root.substream(self.stream, block)
    }

    #[inline]
    fn fill<'a>(
        &self,
        block: &mut R,
        _: usize,
        _: usize,
        uniforms: &'a mut [S],
        draws: &mut usize,
    ) -> Result<ProposalUniforms<'a, S>, SciOxideError> {
        fill_uniforms(block, uniforms, draws)?;
        Ok(ProposalUniforms::from_admitted(uniforms))
    }
}

impl<S, R> UniformSource<S> for RandomStreams<'_, R>
where
    S: RealScalar,
    R: Rng<S> + SplittableRng + Sync,
{
}

impl<S: RealScalar> sealed::Source<S> for &QmcReplicatePlan {
    type Block = ();
    const INDEPENDENT: bool = false;

    fn check(
        &self,
        replicates: usize,
        samples: usize,
        uniforms: usize,
    ) -> Result<(), SciOxideError> {
        if replicates != self.plans().len() {
            return Err(super::invalid(
                scioxide_core::argument!("source"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "QMC replicates must equal the plan's replicates",
            ));
        }
        if samples != self.points_per_replicate() {
            return Err(super::invalid(
                scioxide_core::argument!("source"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "QMC points per replicate must equal the plan's samples",
            ));
        }
        self.plans()
            .iter()
            .try_for_each(|plan| crate::kernel::shape(uniforms, plan.dim()))
    }

    fn open(&self, _: usize, _: u64) -> Result<(), SciOxideError> {
        Ok(())
    }

    #[inline]
    fn fill<'a>(
        &self,
        (): &mut (),
        child: usize,
        index: usize,
        uniforms: &'a mut [S],
        draws: &mut usize,
    ) -> Result<ProposalUniforms<'a, S>, SciOxideError> {
        *draws += uniforms.len();
        self.plans()[child].point_into(index as u64, uniforms)?;
        ProposalUniforms::new(uniforms)
    }
}

impl<S: RealScalar> UniformSource<S> for &QmcReplicatePlan {}
