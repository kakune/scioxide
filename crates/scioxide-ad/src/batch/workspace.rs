//! Caller-owned storage of batch evaluations.

use alloc::vec::Vec;

use scioxide_core::{Capacity, InvalidInputReason, RealScalar, SciOxideError};

use super::BatchPlan;
use crate::{Dual, Lanes, Tape, TapeOptions};

/// The largest number of workers a batch distributes blocks over.
pub(crate) const MAX_WORKERS: usize = 64;

/// The buffers a [`BatchPlan`] fixes for a number of workers: per-worker lane
/// buffers, per-block results and per-item staging. Tapes are not included; they
/// grow on first use, each within the byte limit of the plan that evaluates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchRequirements {
    workers: usize,
    blocks: usize,
    items: usize,
    parameters: usize,
    inputs: usize,
    data: usize,
    lanes: usize,
    /// Per-block gradients: `max(blocks, 1)·max(parameters, 1)`.
    gradients: usize,
    /// Per-item input gradients: `items·inputs`.
    matrix: usize,
    /// Dual seeds per worker: `inputs + parameters`.
    duals: usize,
}

impl BatchRequirements {
    pub(crate) fn for_plan<const W: usize>(
        plan: &BatchPlan<W>,
        workers: usize,
    ) -> Result<Self, SciOxideError> {
        if workers == 0 || workers > MAX_WORKERS {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("workers"),
                InvalidInputReason::OutOfRange,
                "a batch workspace serves 1 to 64 workers",
            ));
        }
        let overflow = || SciOxideError::size_overflow(scioxide_core::quantity!("batch_workspace"));
        let (blocks, parameters) = (plan.blocks(), plan.parameters());
        let gradients = blocks
            .max(1)
            .checked_mul(parameters.max(1))
            .filter(|g| g.checked_add(parameters).is_some())
            .ok_or_else(overflow)?;
        let matrix = plan
            .items()
            .checked_mul(plan.inputs())
            .ok_or_else(overflow)?;
        let duals = plan.inputs().checked_add(parameters).ok_or_else(overflow)?;
        let requirements = Self {
            workers,
            blocks,
            items: plan.items(),
            parameters,
            inputs: plan.inputs(),
            data: plan.data(),
            lanes: W,
            gradients,
            matrix,
            duals,
        };
        requirements.bytes::<f64>()?;
        Ok(requirements)
    }

    /// Returns the number of workers.
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }
    /// Returns the number of blocks.
    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.blocks
    }
    /// Returns the number of items.
    #[must_use]
    pub const fn items(&self) -> usize {
        self.items
    }

    /// Returns the bytes the buffers take for scalars of `S`, tapes excluded: per
    /// worker the lane buffers and the bookkeeping of its two tapes, per block the
    /// results and gradients, and per item the staged values and input gradients.
    ///
    /// # Errors
    ///
    /// `SizeOverflow` if the byte count overflows.
    pub fn bytes<S: RealScalar>(&self) -> Result<usize, SciOxideError> {
        let overflow = || SciOxideError::size_overflow(scioxide_core::quantity!("batch_bytes"));
        let lanes = |count: usize, size: usize| count.checked_mul(self.lanes)?.checked_mul(size);
        let scalar = size_of::<S>();
        let per_worker = lanes(self.parameters, scalar)
            .and_then(|b| b.checked_add(lanes(self.data, scalar)?))
            .and_then(|b| b.checked_add(lanes(self.inputs, 2 * scalar)?))
            .and_then(|b| b.checked_add(lanes(self.duals, 2 * scalar)?))
            .and_then(|b| b.checked_add(size_of::<Worker<S, 1>>()))
            .and_then(|b| b.checked_mul(self.workers));
        let per_block = self
            .blocks
            .checked_mul(size_of::<BlockStats<S>>())
            .and_then(|b| b.checked_add(self.gradients.checked_mul(2 * scalar)?))
            .and_then(|b| b.checked_add(self.parameters.checked_mul(scalar)?));
        let per_item = self
            .items
            .checked_mul(2 * scalar)
            .and_then(|b| b.checked_add(self.matrix.checked_mul(scalar)?));
        per_worker
            .and_then(|b| b.checked_add(per_block?))
            .and_then(|b| b.checked_add(per_item?))
            .ok_or_else(overflow)
    }
}

/// The per-block results reduced by the caller thread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BlockStats<S> {
    pub(crate) count: usize,
    pub(crate) sum: S,
    pub(crate) mean: S,
    pub(crate) m2: S,
    pub(crate) derivative: S,
    pub(crate) kinks: u64,
}

impl<S: RealScalar> BlockStats<S> {
    pub(crate) fn empty() -> Self {
        Self {
            count: 0,
            sum: S::ZERO,
            mean: S::ZERO,
            m2: S::ZERO,
            derivative: S::ZERO,
            kinks: 0,
        }
    }
}

/// The buffers and tapes of one worker.
pub(crate) struct Worker<S: RealScalar, const W: usize> {
    pub(crate) parameters: Vec<Lanes<S, W>>,
    pub(crate) data: Vec<Lanes<S, W>>,
    pub(crate) inputs: Vec<Lanes<S, W>>,
    pub(crate) directions: Vec<Lanes<S, W>>,
    pub(crate) duals: Vec<Dual<Lanes<S, W>>>,
    pub(crate) tape: Tape<Lanes<S, W>>,
    pub(crate) second: Tape<Dual<Lanes<S, W>>>,
}

impl<S: RealScalar, const W: usize> Worker<S, W> {
    fn new() -> Self {
        let options = TapeOptions::new();
        Self {
            parameters: Vec::new(),
            data: Vec::new(),
            inputs: Vec::new(),
            directions: Vec::new(),
            duals: Vec::new(),
            tape: Tape::new(options),
            second: Tape::new(options),
        }
    }
}

/// Storage of [`BatchPlan`] evaluations: per-worker buffers and tapes, per-block
/// results and per-item staging, so outputs are written only after every block
/// succeeded. Under sequential execution a warm evaluation allocates nothing.
pub struct BatchWorkspace<S: RealScalar, const W: usize> {
    pub(crate) workers: Vec<Worker<S, W>>,
    pub(crate) blocks: Vec<BlockStats<S>>,
    pub(crate) gradients: Vec<S>,
    pub(crate) scratch: Vec<S>,
    pub(crate) values: Vec<S>,
    pub(crate) input_gradients: Vec<S>,
    pub(crate) derivatives: Vec<S>,
}

impl<S: RealScalar, const W: usize> core::fmt::Debug for BatchWorkspace<S, W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BatchWorkspace")
            .field("workers", &self.workers.len())
            .field("blocks", &self.blocks.len())
            .finish_non_exhaustive()
    }
}

fn reserve<T: Clone>(vector: &mut Vec<T>, len: usize, fill: T) -> Result<(), SciOxideError> {
    if vector.len() < len {
        vector.try_reserve_exact(len - vector.len()).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("batch_workspace"), len)
        })?;
        vector.resize(len, fill);
    }
    Ok(())
}

impl<S: RealScalar, const W: usize> BatchWorkspace<S, W> {
    pub(crate) fn new() -> Self {
        Self {
            workers: Vec::new(),
            blocks: Vec::new(),
            gradients: Vec::new(),
            scratch: Vec::new(),
            values: Vec::new(),
            input_gradients: Vec::new(),
            derivatives: Vec::new(),
        }
    }

    /// Returns the number of workers the workspace serves.
    #[must_use]
    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    /// Grows the workspace to `requirements`.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` if a buffer cannot grow; the workspace stays usable.
    pub fn try_reserve(&mut self, requirements: &BatchRequirements) -> Result<(), SciOxideError> {
        let zero = S::ZERO;
        let lanes = Lanes::splat(zero);
        if self.workers.len() < requirements.workers {
            self.workers
                .try_reserve_exact(requirements.workers - self.workers.len())
                .map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("batch_workers"),
                        requirements.workers,
                    )
                })?;
            while self.workers.len() < requirements.workers {
                self.workers.push(Worker::new());
            }
        }
        for worker in &mut self.workers {
            reserve(&mut worker.parameters, requirements.parameters, lanes)?;
            reserve(&mut worker.data, requirements.data, lanes)?;
            reserve(&mut worker.inputs, requirements.inputs, lanes)?;
            reserve(&mut worker.directions, requirements.inputs, lanes)?;
            reserve(
                &mut worker.duals,
                requirements.duals,
                Dual::new(lanes, lanes),
            )?;
        }
        reserve(&mut self.blocks, requirements.blocks, BlockStats::empty())?;
        reserve(&mut self.gradients, requirements.gradients, zero)?;
        // The reduced gradients, then the gradient errors.
        let scratch = requirements.gradients + requirements.parameters;
        reserve(&mut self.scratch, scratch, zero)?;
        reserve(&mut self.values, requirements.items, zero)?;
        reserve(&mut self.derivatives, requirements.items, zero)?;
        reserve(&mut self.input_gradients, requirements.matrix, zero)?;
        Ok(())
    }

    /// Classifies the retained storage against `requirements`.
    ///
    /// # Errors
    ///
    /// None at present; the signature follows the workspace protocol.
    pub fn check(&self, requirements: &BatchRequirements) -> Result<Capacity, SciOxideError> {
        let enough = self.workers.len() >= requirements.workers
            && self.workers.iter().all(|worker| {
                worker.parameters.len() >= requirements.parameters
                    && worker.data.len() >= requirements.data
                    && worker.inputs.len() >= requirements.inputs
                    && worker.directions.len() >= requirements.inputs
                    && worker.duals.len() >= requirements.duals
            })
            && self.blocks.len() >= requirements.blocks
            && self.gradients.len() >= requirements.gradients
            && self.scratch.len() >= requirements.gradients + requirements.parameters
            && self.values.len() >= requirements.items
            && self.derivatives.len() >= requirements.items
            && self.input_gradients.len() >= requirements.matrix;
        Ok(if enough {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
}
