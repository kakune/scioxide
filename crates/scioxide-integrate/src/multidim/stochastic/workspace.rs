//! Monte Carlo workspace: block summaries, per-child estimates and the final statistics.

use alloc::vec::Vec;

use scioxide_core::backend::{PartitionedSolve, try_filled, try_with_capacity};
use scioxide_core::{AffineValue, PromotingCompensatedSum, RealScalar, Scaled, SciOxideError};
use scioxide_stats::{MomentOrder, OnlineMoments};

use super::SAMPLE_BLOCK;
use super::kernel::{Draw, Lane, Progress, Summary};
use super::{MonteCarloOptions, MonteCarloStatistics, VarianceBasis};
use crate::report::{IntegrationReport, IntegrationTermination};
use crate::workspace::{Attempts, IntegrationRequirements, workspace_protocol};

/// Running sums of one real value component.
#[derive(Debug, Clone, Copy)]
struct Component<S: RealScalar> {
    moments: OnlineMoments<S>,
    sum: PromotingCompensatedSum<S>,
}

impl<S: RealScalar> Component<S> {
    fn new() -> Self {
        Self {
            moments: OnlineMoments::new(MomentOrder::Second),
            sum: PromotingCompensatedSum::ZERO,
        }
    }
}

/// Summary of weighted observations: per-component moments (independent sources)
/// or sums (QMC), and the weight sums of Kish's effective sample size.
#[derive(Debug, Clone)]
pub(super) struct Block<S: RealScalar> {
    observations: usize,
    units: usize,
    weights: PromotingCompensatedSum<S>,
    squares: PromotingCompensatedSum<S>,
    row: Vec<S>,
    samples: Vec<S>,
    independent: bool,
    components: Vec<Component<S>>,
}

impl<S: RealScalar> Summary for Block<S> {
    fn clear(&mut self) {
        self.independent = false;
        self.observations = 0;
        self.units = 0;
        self.weights = PromotingCompensatedSum::ZERO;
        self.squares = PromotingCompensatedSum::ZERO;
        self.components.fill(Component::new());
    }
    fn finish(&mut self) -> Result<(), SciOxideError> {
        if self.independent {
            for (component, samples) in self
                .components
                .iter_mut()
                .zip(self.samples.chunks_exact(SAMPLE_BLOCK))
            {
                component.moments =
                    OnlineMoments::from_slice(&samples[..self.observations], MomentOrder::Second)?;
            }
        }
        Ok(())
    }
}

impl<S: RealScalar> Block<S> {
    pub(super) fn allocate(components: usize, buffered: bool) -> Result<Self, SciOxideError> {
        Ok(Self {
            observations: 0,
            units: 0,
            weights: PromotingCompensatedSum::ZERO,
            squares: PromotingCompensatedSum::ZERO,
            independent: false,
            samples: try_filled(
                scioxide_core::backend::checked_mul(
                    components,
                    if buffered { SAMPLE_BLOCK } else { 0 },
                    scioxide_core::quantity!("mc_samples"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("mc_samples"),
            )?,
            row: try_filled(components, S::ZERO, scioxide_core::quantity!("mc_row"))?,
            components: try_filled(
                components,
                Component::new(),
                scioxide_core::quantity!("mc_components"),
            )?,
        })
    }

    /// Returns the number of observations so far.
    pub(super) const fn observations(&self) -> usize {
        self.observations
    }

    /// Add the observation `volume · f / q` of every component.
    ///
    /// Unit densities are counted exactly instead of summing rounded weights.
    /// `native_volume`, when present, is the box volume as a normal `S` value.
    #[inline(always)]
    pub(super) fn observe<Src: super::Source<S>, V: AffineValue<Scale = S>>(
        &mut self,
        draw: &Draw<'_, S>,
        value: &V,
        volume: Scaled<S>,
        native_volume: Option<S>,
    ) -> Result<(), SciOxideError> {
        let independent = Src::INDEPENDENT;
        let scalar = if V::FIXED_DIMENSION == Some(1) {
            crate::kernel::shape(1, self.row.len())?;
            Some(crate::evaluation::scalar_value(value)?)
        } else {
            crate::evaluation::write_row(value, &mut self.row)?;
            None
        };
        self.independent = independent;
        // The accumulator admitted native_volume once for this box. Only a
        // non-unit proposal density introduces new range-sensitive arithmetic.
        let native_scale = if draw.density == S::ONE {
            native_volume
        } else {
            native_volume.and_then(|volume| {
                let scale = volume / draw.density;
                (scale.is_finite() && scale.abs() >= S::MIN_POSITIVE).then_some(scale)
            })
        };
        let weight = if draw.density == S::ONE {
            self.units += 1;
            None
        } else {
            let native = S::ONE / draw.density;
            if native.is_finite() && native.abs() >= S::MIN_POSITIVE {
                self.weights.add_value(native)?;
                self.squares.add_product(native, native)?;
                None
            } else {
                let weight = Scaled::from_value(S::ONE)?.div(Scaled::from_value(draw.density)?)?;
                self.weights.add_scaled(weight)?;
                self.squares.add_scaled(weight.square()?)?;
                Some(weight)
            }
        };
        let native_observation = |value: S| {
            native_scale.and_then(|scale| {
                let product = scale * value;
                (value == S::ZERO || product.is_finite() && product.abs() >= S::MIN_POSITIVE)
                    .then_some(product)
            })
        };
        if let Some(value) = scalar {
            if let Some(observation) = native_observation(value) {
                // Fixed scalar callbacks need no row materialization or component
                // traversal. Their arithmetic and block positions are unchanged.
                if independent {
                    self.samples[self.observations] = observation;
                } else {
                    self.components[0].sum.add_value(observation)?;
                }
            } else {
                self.observe_scalar_extended(value, draw.density, volume, weight, independent)?;
            }
            self.observations += 1;
            return Ok(());
        }
        let mut extended_scale = None;
        for (index, (component, &value)) in self.components.iter_mut().zip(&self.row).enumerate() {
            let native = native_observation(value);
            if let Some(observation) = native {
                if independent {
                    self.samples[index * SAMPLE_BLOCK + self.observations] = observation;
                } else {
                    component.sum.add_value(observation)?;
                }
            } else {
                let scale = match extended_scale {
                    Some(scale) => scale,
                    None => {
                        let weight = match weight {
                            Some(weight) => weight,
                            None => Scaled::from_value(S::ONE)?
                                .div(Scaled::from_value(draw.density)?)?,
                        };
                        let scale = volume.mul(weight)?;
                        extended_scale = Some(scale);
                        scale
                    }
                };
                let observation = scale.mul(Scaled::from_value(value)?)?;
                if independent {
                    self.samples[index * SAMPLE_BLOCK + self.observations] = observation.round()?;
                } else {
                    component.sum.add_scaled(observation)?;
                }
            }
        }
        self.observations += 1;
        Ok(())
    }

    /// Adds one scalar observation in extended range, outside the hot loop.
    #[cold]
    #[inline(never)]
    fn observe_scalar_extended(
        &mut self,
        value: S,
        density: S,
        volume: Scaled<S>,
        weight: Option<Scaled<S>>,
        independent: bool,
    ) -> Result<(), SciOxideError> {
        let weight = match weight {
            Some(weight) => weight,
            None => Scaled::from_value(S::ONE)?.div(Scaled::from_value(density)?)?,
        };
        let observation = volume.mul(weight)?.mul(Scaled::from_value(value)?)?;
        if independent {
            self.samples[self.observations] = observation.round()?;
        } else {
            self.components[0].sum.add_scaled(observation)?;
        }
        Ok(())
    }

    /// Merges a later block into this summary.
    pub(super) fn merge(&mut self, other: &Self) -> Result<(), SciOxideError> {
        self.observations = self
            .observations
            .checked_add(other.observations)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("observations"))
            })?;
        self.units += other.units;
        self.weights.merge(other.weights)?;
        self.squares.merge(other.squares)?;
        for (component, other) in self.components.iter_mut().zip(&other.components) {
            component.moments.merge(&other.moments)?;
            component.sum.merge(other.sum)?;
        }
        Ok(())
    }

    /// Returns Kish's effective sample size `(Σw)² / Σw²`.
    fn effective_sample_size(&self) -> Result<S, SciOxideError> {
        if self.units == self.observations {
            return S::try_from_f64_lossy(self.units as f64);
        }
        let units = Scaled::from_value(S::try_from_f64_lossy(self.units as f64)?)?;
        let mut weights = self.weights;
        let mut squares = self.squares;
        weights.add_scaled(units)?;
        squares.add_scaled(units)?;
        weights.total()?.square()?.div(squares.total()?)?.round()
    }
}

/// Estimate, statistics and progress of one replicate, stratum or independent run.
#[derive(Debug, Clone)]
pub(super) struct Child<S: RealScalar> {
    pub(super) total: Block<S>,
    values: Vec<S>,
    statistics: Vec<Option<MonteCarloStatistics<S>>>,
    effective_sample_size: Option<S>,
    termination: Option<IntegrationTermination>,
    evaluations: usize,
}

impl<S: RealScalar> Child<S> {
    fn allocate(components: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            total: Block::allocate(components, false)?,
            values: try_filled(components, S::ZERO, scioxide_core::quantity!("mc_values"))?,
            statistics: try_filled(components, None, scioxide_core::quantity!("mc_statistics"))?,
            effective_sample_size: None,
            termination: None,
            evaluations: 0,
        })
    }

    pub(super) fn clear(&mut self) {
        self.total.clear();
        self.values.fill(S::ZERO);
        self.statistics.fill(None);
        self.effective_sample_size = None;
        self.termination = None;
        self.evaluations = 0;
    }

    /// Returns the termination of the last run, `None` before it ran.
    pub(super) const fn termination(&self) -> Option<IntegrationTermination> {
        self.termination
    }

    pub(super) fn values(&self) -> &[S] {
        &self.values
    }

    pub(super) fn statistics(&self) -> &[Option<MonteCarloStatistics<S>>] {
        &self.statistics
    }

    /// Computes the estimates from the accumulated observations.
    pub(super) fn finish(
        &mut self,
        progress: Progress,
        independent: bool,
    ) -> Result<(), SciOxideError> {
        self.evaluations = progress.evaluations;
        self.termination = Some(
            progress
                .stop
                .unwrap_or(IntegrationTermination::SampleBudgetCompleted),
        );
        let count = self.total.observations;
        if count == 0 {
            return Ok(());
        }
        let ess = self.total.effective_sample_size()?;
        self.effective_sample_size = Some(ess);
        let observations = Scaled::from_value(S::try_from_f64_lossy(count as f64)?)?;
        for (value, component) in self.values.iter_mut().zip(&self.total.components) {
            if independent {
                let moments = component.moments.snapshot();
                *value = moments.mean()?.ok_or_else(|| {
                    super::invalid(
                        scioxide_core::argument!("observations"),
                        scioxide_core::InvalidInputReason::Empty,
                        "missing sampled mean",
                    )
                })?;
            } else {
                *value = component.sum.total()?.div(observations)?.round()?;
            }
        }
        Ok(())
    }

    pub(super) fn finalize_statistics(
        &mut self,
        options: &MonteCarloOptions,
    ) -> Result<(), SciOxideError> {
        let count = self.total.observations;
        if count >= 2
            && self
                .total
                .components
                .first()
                .is_some_and(|c| !c.moments.is_empty())
        {
            for (statistics, component) in self.statistics.iter_mut().zip(&self.total.components) {
                *statistics = Some(MonteCarloStatistics::from_moments(
                    VarianceBasis::Observations,
                    component.moments.snapshot(),
                    count,
                    self.effective_sample_size,
                    options,
                )?);
            }
        }
        Ok(())
    }

    fn report(&self) -> Option<IntegrationReport<S>> {
        let termination = self.termination?;
        (self.total.observations > 0)
            .then(|| IntegrationReport::new(termination, self.evaluations, 0, None))
    }
}

/// Workspace of [`MonteCarloIntegrator::integrate_into`](crate::MonteCarloIntegrator::integrate_into)
/// and the statistics of its last result.
///
/// Create one with [`MonteCarloIntegrator::workspace`](crate::MonteCarloIntegrator::workspace)
/// for a fixed number of real value components. It serves any Monte Carlo plan
/// with the same number of components and coordinates whose requirements it
/// meets; `try_reserve` grows it for a plan that needs more replicates, strata
/// or workers. See [reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// After each call, [`statistics`](Self::statistics) holds the standard error
/// and confidence interval of every component, and [`children`](Self::children)
/// the result of every replicate or stratum. Each parallel worker stores
/// [`SAMPLE_BLOCK`] values per component.
#[derive(Debug)]
pub struct MonteCarloWorkspace<S: RealScalar> {
    pub(super) requirements: IntegrationRequirements,
    pub(super) attempts: Attempts<IntegrationReport<S>>,
    pub(super) lanes: Vec<Lane<S, Block<S>>>,
    pub(super) children: Vec<Child<S>>,
    pub(super) solves: Vec<PartitionedSolve>,
    pub(super) values: Vec<S>,
    pub(super) statistics: Vec<Option<MonteCarloStatistics<S>>>,
    pub(super) active: usize,
}

impl<S: RealScalar> MonteCarloWorkspace<S> {
    pub(super) fn allocate(requirements: &IntegrationRequirements) -> Result<Self, SciOxideError> {
        let components = requirements.components();
        let [slots, workers] = requirements.units();
        let coordinates = requirements.dimension();
        let mut children = try_with_capacity(slots, scioxide_core::quantity!("mc_children"))?;
        for _ in 0..slots {
            children.push(Child::allocate(components)?);
        }
        let mut solves = try_with_capacity(slots, scioxide_core::quantity!("mc_child_solves"))?;
        solves.resize_with(slots, PartitionedSolve::default);
        Ok(Self {
            requirements: *requirements,
            attempts: Attempts::default(),
            lanes: Lane::allocate_many(workers, coordinates, || Block::allocate(components, true))?,
            children,
            solves,
            values: try_filled(components, S::ZERO, scioxide_core::quantity!("mc_estimate"))?,
            statistics: try_filled(components, None, scioxide_core::quantity!("mc_statistics"))?,
            active: 0,
        })
    }

    /// Clears the previous call's estimates.
    pub(super) fn reset(&mut self) {
        self.statistics.fill(None);
        for child in &mut self.children[..self.active] {
            child.clear();
        }
        self.active = 0;
    }

    /// Returns the statistics of each real component of the last estimate.
    ///
    /// An entry is `None` when there is no variance estimate: the last call
    /// returned an error, a single run had fewer than two observations or drew
    /// from a QMC source, or the box has dimension zero or zero width. Clone an
    /// entry to call [`MonteCarloStatistics::confidence_interval`].
    #[must_use]
    pub fn statistics(&self) -> &[Option<MonteCarloStatistics<S>>] {
        &self.statistics
    }

    /// Returns the result of every replicate or stratum of the last call, or of
    /// its single run, in index order.
    ///
    /// The children are available even when a stop made the call return an
    /// error, so you can inspect how far each one got. The iterator is empty if
    /// the call failed before sampling, or if the box has dimension zero or zero
    /// width.
    pub fn children(&self) -> impl ExactSizeIterator<Item = MonteCarloChild<'_, S>> {
        self.children[..self.active]
            .iter()
            .map(|child| MonteCarloChild { child })
    }
}

workspace_protocol!(MonteCarloWorkspace, IntegrationReport<S>);

/// Result of one replicate, one stratum, or the single run of a Monte Carlo call.
///
/// Obtained from [`MonteCarloWorkspace::children`].
#[derive(Debug, Clone, Copy)]
pub struct MonteCarloChild<'a, S: RealScalar> {
    child: &'a Child<S>,
}

impl<'a, S: RealScalar> MonteCarloChild<'a, S> {
    /// Returns the termination and the number of integrand calls of this child,
    /// or `None` if it has no observations.
    #[must_use]
    pub fn report(&self) -> Option<IntegrationReport<S>> {
        self.child.report()
    }

    /// Returns this child's estimate of each component, or `None` if it has no
    /// observations.
    ///
    /// For a stratum, the estimate is the integral over that stratum.
    #[must_use]
    pub fn values(&self) -> Option<&'a [S]> {
        self.child.report().map(|_| self.child.values.as_slice())
    }

    /// Returns the statistics of this child's estimate from its observations.
    ///
    /// Entries are `None` for QMC replicates, whose points are not independent,
    /// and for fewer than two observations.
    #[must_use]
    pub fn statistics(&self) -> &'a [Option<MonteCarloStatistics<S>>] {
        &self.child.statistics
    }

    /// Returns Kish's effective sample size of this child's importance weights; see
    /// [`MonteCarloStatistics::effective_sample_size`].
    #[must_use]
    pub fn effective_sample_size(&self) -> Option<S> {
        self.child.effective_sample_size
    }
}
