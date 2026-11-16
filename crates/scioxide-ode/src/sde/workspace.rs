//! Workspace and storage requirements of SDE plans.

use super::SdeReport;
use crate::shell::Shell;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem::size_of;
use scioxide_core::backend::{checked_add, checked_mul, try_reserve};
use scioxide_core::{
    AlgebraicScalar, AttemptRecord, Capacity, NumericScalar, Quantity, SciOxideError,
};

const BYTES: Quantity = scioxide_core::quantity!("sde_workspace_bytes");

/// State-scalar vectors: proposal, half step, refined candidate and five stage buffers.
const STATE_VECTORS: usize = 8;

/// Checked heap bytes of `state` values of `T`, `real` values of `T::Real` and `stack`
/// refinement intervals.
fn footprint<T: NumericScalar>(
    state: usize,
    real: usize,
    stack: usize,
) -> Result<usize, SciOxideError> {
    let state = checked_mul(state, size_of::<T>(), BYTES)?;
    let real = checked_mul(real, size_of::<T::Real>(), BYTES)?;
    let stack = checked_mul(stack, size_of::<(usize, usize)>(), BYTES)?;
    checked_add(checked_add(state, real, BYTES)?, stack, BYTES)
}

/// Describes the storage an [`SdeMethod`](crate::SdeMethod) needs for states of scalar type
/// `T`, as returned by
/// [`SdeMethod::workspace_requirements`](crate::SdeMethod::workspace_requirements).
///
/// The byte limit applies to all storage the workspace holds, including storage kept
/// from earlier, larger plans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdeRequirements<T> {
    dimension: usize,
    noise_dimension: usize,
    refinement_stack: usize,
    bytes: usize,
    byte_limit: Option<usize>,
    scalar: PhantomData<fn() -> T>,
}
impl<T: NumericScalar> SdeRequirements<T> {
    pub(super) fn new(
        dimension: usize,
        noise_dimension: usize,
        refinement_stack: usize,
        byte_limit: Option<usize>,
    ) -> Result<Self, SciOxideError> {
        let state = checked_mul(dimension, STATE_VECTORS, BYTES)?;
        let real = checked_add(checked_mul(dimension, 2, BYTES)?, noise_dimension, BYTES)?;
        Ok(Self {
            dimension,
            noise_dimension,
            refinement_stack,
            bytes: footprint::<T>(state, real, refinement_stack)?,
            byte_limit,
            scalar: PhantomData,
        })
    }
    /// Returns the state dimension.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the number of Wiener components.
    pub const fn noise_dimension(&self) -> usize {
        self.noise_dimension
    }
    /// Returns the bytes of solver scratch storage.
    ///
    /// The plan, the inputs, allocations of the callbacks and the sink, and allocator
    /// overhead are not included.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Returns the limit on the bytes a workspace may hold, if set.
    pub const fn byte_limit(&self) -> Option<usize> {
        self.byte_limit
    }
}

/// Stage buffers of one explicit step.
pub(super) struct StepScratch<T: NumericScalar> {
    pub(super) drift: Vec<T>,
    pub(super) diffusion: Vec<T>,
    pub(super) predictor: Vec<T>,
    pub(super) next_drift: Vec<T>,
    pub(super) next_diffusion: Vec<T>,
    pub(super) increment: Vec<T::Real>,
}

/// Numerical scratch of the stepping loop.
pub(super) struct Buffers<T: NumericScalar> {
    pub(super) full: Vec<T>,
    pub(super) half: Vec<T>,
    pub(super) candidate: Vec<T>,
    pub(super) errors: Vec<T::Real>,
    pub(super) scales: Vec<T::Real>,
    pub(super) pending: Vec<(usize, usize)>,
    pub(super) scratch: StepScratch<T>,
}
impl<T: NumericScalar> Buffers<T> {
    const fn new() -> Self {
        Self {
            full: Vec::new(),
            half: Vec::new(),
            candidate: Vec::new(),
            errors: Vec::new(),
            scales: Vec::new(),
            pending: Vec::new(),
            scratch: StepScratch {
                drift: Vec::new(),
                diffusion: Vec::new(),
                predictor: Vec::new(),
                next_drift: Vec::new(),
                next_diffusion: Vec::new(),
                increment: Vec::new(),
            },
        }
    }
    fn states(&self) -> [&Vec<T>; STATE_VECTORS] {
        let s = &self.scratch;
        [
            &self.full,
            &self.half,
            &self.candidate,
            &s.drift,
            &s.diffusion,
            &s.predictor,
            &s.next_drift,
            &s.next_diffusion,
        ]
    }
    fn states_mut(&mut self) -> [&mut Vec<T>; STATE_VECTORS] {
        let s = &mut self.scratch;
        [
            &mut self.full,
            &mut self.half,
            &mut self.candidate,
            &mut s.drift,
            &mut s.diffusion,
            &mut s.predictor,
            &mut s.next_drift,
            &mut s.next_diffusion,
        ]
    }
    /// Real vectors with their required lengths for `(dimension, noise_dimension)`.
    fn reals(&self, n: usize, m: usize) -> [(&Vec<T::Real>, usize); 3] {
        [
            (&self.errors, n),
            (&self.scales, n),
            (&self.scratch.increment, m),
        ]
    }
    fn reals_mut(&mut self, n: usize, m: usize) -> [(&mut Vec<T::Real>, usize); 3] {
        [
            (&mut self.errors, n),
            (&mut self.scales, n),
            (&mut self.scratch.increment, m),
        ]
    }
    /// Bytes held after growing every buffer to at least `(n, m, stack)`.
    fn retained(&self, n: usize, m: usize, stack: usize) -> Result<usize, SciOxideError> {
        let state = self
            .states()
            .into_iter()
            .try_fold(0, |sum, v| checked_add(sum, v.capacity().max(n), BYTES))?;
        let real = self.reals(n, m).into_iter().try_fold(0, |sum, (v, len)| {
            checked_add(sum, v.capacity().max(len), BYTES)
        })?;
        footprint::<T>(state, real, self.pending.capacity().max(stack))
    }
    fn fits(&self, requirements: &SdeRequirements<T>) -> bool {
        let (n, m) = (requirements.dimension, requirements.noise_dimension);
        self.states().into_iter().all(|v| v.capacity() >= n)
            && self
                .reals(n, m)
                .into_iter()
                .all(|(v, len)| v.capacity() >= len)
            && self.pending.capacity() >= requirements.refinement_stack
    }
    fn reserve(&mut self, requirements: &SdeRequirements<T>) -> Result<(), SciOxideError> {
        let (n, m) = (requirements.dimension, requirements.noise_dimension);
        for v in self.states_mut() {
            if v.capacity() < n {
                try_reserve(
                    v,
                    n - v.len(),
                    scioxide_core::quantity!("sde_state_scratch"),
                )?;
            }
        }
        for (v, len) in self.reals_mut(n, m) {
            if v.capacity() < len {
                try_reserve(
                    v,
                    len - v.len(),
                    scioxide_core::quantity!("sde_real_scratch"),
                )?;
            }
        }
        let stack = requirements.refinement_stack;
        if self.pending.capacity() < stack {
            let additional = stack - self.pending.len();
            try_reserve(
                &mut self.pending,
                additional,
                scioxide_core::quantity!("sde_refinement_stack"),
            )?;
        }
        Ok(())
    }
    /// Set the exact lengths of one epoch within reserved capacity; does not allocate
    /// after a successful reservation for `(n, m)`.
    pub(super) fn shape(&mut self, n: usize, m: usize) {
        for v in self.states_mut() {
            v.resize(n, T::ZERO);
        }
        for (v, len) in self.reals_mut(n, m) {
            v.resize(len, <T::Real as AlgebraicScalar>::ZERO);
        }
        self.pending.clear();
    }
}

/// Holds the step and refinement storage of [`SdeMethod`](crate::SdeMethod)s, and the record
/// of the last call.
///
/// [`SdeWorkspace::new`] creates an empty workspace;
/// [`SdeMethod::workspace`](crate::SdeMethod::workspace) and [`Self::try_reserve`] allocate
/// its storage. One workspace can serve several plans with the same state scalar type
/// and keeps the largest storage it has allocated. It stores no random state or path.
/// See [plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct SdeWorkspace<T: NumericScalar> {
    pub(super) buffers: Buffers<T>,
    pub(super) shell: Shell<SdeReport<T::Real>>,
}
impl<T: NumericScalar> Default for SdeWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: NumericScalar> core::fmt::Debug for SdeWorkspace<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SdeWorkspace")
            .field("retained_bytes", &self.buffers.retained(0, 0, 0).ok())
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}
impl<T: NumericScalar> SdeWorkspace<T> {
    /// Creates an empty workspace without allocating.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffers: Buffers::new(),
            shell: Shell::new(),
        }
    }

    /// Reports whether this workspace holds the storage of `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace already holds more bytes than the byte limit.
    /// - `SizeOverflow` if its size overflows `usize`.
    pub fn check(&self, requirements: &SdeRequirements<T>) -> Result<Capacity, SciOxideError> {
        if let Some(limit) = requirements.byte_limit {
            let retained = self.buffers.retained(0, 0, 0)?;
            if retained > limit {
                return Err(SciOxideError::resource_limit(BYTES, retained, limit));
            }
        }
        Ok(if self.buffers.fits(requirements) {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }

    /// Grows this workspace to the storage of `requirements`.
    ///
    /// The byte limit is checked against the size after growth before anything is
    /// allocated. If an allocation fails, the buffers grown so far keep their new size.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn try_reserve(&mut self, requirements: &SdeRequirements<T>) -> Result<(), SciOxideError> {
        if let Some(limit) = requirements.byte_limit {
            let retained = self.buffers.retained(
                requirements.dimension,
                requirements.noise_dimension,
                requirements.refinement_stack,
            )?;
            if retained > limit {
                return Err(SciOxideError::resource_limit(BYTES, retained, limit));
            }
        }
        self.buffers.reserve(requirements)
    }

    /// Returns the record of the last call: its outcome, work counts and report.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<SdeReport<T::Real>>> {
        self.shell.last_attempt()
    }
}
