//! Reusable staging storage for batch sampling, and the byte accounting behind
//! every configured storage limit.

use alloc::vec::Vec;

use scioxide_core::{Capacity, Quantity, SciOxideError};

const BYTES: Quantity = scioxide_core::quantity!("workspace_bytes");

/// Checked payload bytes of `elements` values of `T`.
pub(crate) fn storage_bytes<T>(elements: usize) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_mul(elements, core::mem::size_of::<T>(), BYTES)
}

/// Checks storage the crate allocates against a configured byte limit.
pub(crate) fn admit_bytes(bytes: usize, max_bytes: usize) -> Result<(), SciOxideError> {
    if bytes > max_bytes {
        Err(SciOxideError::resource_limit(BYTES, bytes, max_bytes))
    } else {
        Ok(())
    }
}

/// Shape of one batch: `samples` rows of `width` elements.
///
/// ```
/// use scioxide_random::{OutputPolicy, SamplingPolicy, SamplingRequest};
/// let policy = SamplingPolicy::rollback(OutputPolicy::Incremental);
/// let requirements = policy.workspace_requirements(SamplingRequest::rows(8, 3))?;
/// assert_eq!(requirements.elements(), 3); // one staged row
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingRequest {
    samples: usize,
    width: usize,
}

impl SamplingRequest {
    /// Creates a request for `samples` rows of `width` elements.
    ///
    /// A zero width is rejected when requirements are computed.
    pub const fn rows(samples: usize, width: usize) -> Self {
        Self { samples, width }
    }

    /// Creates a request for `samples` scalar draws.
    pub const fn scalars(samples: usize) -> Self {
        Self::rows(samples, 1)
    }

    // Atomic output stages the whole batch so nothing is published before
    // success. Incremental output only needs one row of staging, because a
    // vector-valued row must not be exposed half-written; scalar samples
    // are written directly.
    pub(crate) fn elements(self, output: crate::OutputPolicy) -> Result<usize, SciOxideError> {
        if self.width == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("width"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "sample width must be positive",
            ));
        }
        let len = scioxide_core::backend::checked_mul(
            self.samples,
            self.width,
            scioxide_core::quantity!("sampling_output"),
        )?;
        Ok(match output {
            crate::OutputPolicy::Atomic => len,
            crate::OutputPolicy::Incremental if self.width > 1 && self.samples > 0 => self.width,
            crate::OutputPolicy::Incremental => 0,
        })
    }
}

/// Workspace size needed by one batch: staging elements, `f64` scratch and a byte limit.
///
/// The limit applies to the workspace's total capacity, including capacity left
/// from earlier, larger batches. Requirements from a [`SamplingPolicy`](crate::SamplingPolicy)
/// have no limit; a law or sampler with a configured byte limit applies it when
/// the batch runs.
///
/// ```
/// use scioxide_random::{OutputPolicy, SamplingPolicy, SamplingRequest};
/// let policy = SamplingPolicy::advance_on_error(OutputPolicy::Atomic);
/// let requirements = policy.workspace_requirements(SamplingRequest::rows(8, 3))?;
/// assert_eq!(requirements.elements(), 24); // the whole batch is staged
/// assert_eq!(requirements.f64_scratch_elements(), 0);
/// assert_eq!(requirements.max_bytes(), usize::MAX); // policies impose no limit
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingRequirements {
    elements: usize,
    f64_scratch_elements: usize,
    max_bytes: usize,
}

impl SamplingRequirements {
    pub(crate) const fn new(elements: usize) -> Self {
        Self {
            elements,
            f64_scratch_elements: 0,
            max_bytes: usize::MAX,
        }
    }

    /// Apply a configured byte limit in addition to any existing one.
    pub(crate) const fn limited(self, max_bytes: usize) -> Self {
        Self {
            elements: self.elements,
            f64_scratch_elements: self.f64_scratch_elements,
            max_bytes: if max_bytes < self.max_bytes {
                max_bytes
            } else {
                self.max_bytes
            },
        }
    }

    /// Returns the number of staging elements of the output type.
    pub const fn elements(self) -> usize {
        self.elements
    }

    /// Returns the number of additional `f64` scratch elements.
    ///
    /// It does not depend on the output type: [`Dirichlet`](crate::Dirichlet)
    /// needs `f64` scratch even for `f32` output.
    pub const fn f64_scratch_elements(self) -> usize {
        self.f64_scratch_elements
    }

    pub(crate) const fn with_f64_scratch(mut self, elements: usize) -> Self {
        self.f64_scratch_elements = elements;
        self
    }

    /// Returns the byte limit on the total workspace capacity; `usize::MAX` means no limit.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
}

/// Reusable scratch storage for batch sampling and quasi-random batches.
///
/// Size it with [`SamplingPolicy::workspace_requirements`](crate::SamplingPolicy::workspace_requirements)
/// (or a law's own `workspace_requirements`) and [`Self::try_reserve`]. A batch
/// grows a workspace that is too small before it draws, so reserving in advance
/// only moves the allocation out of the batch. Contents may change on error, and
/// the workspace stays usable. See [reusing plans and
/// workspaces](crate#reusing-plans-and-workspaces).
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_random::{OutputPolicy, SamplingPolicy, SamplingRequest, SamplingWorkspace};
/// let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
/// let requirements = policy.workspace_requirements(SamplingRequest::scalars(16))?;
/// let mut workspace = SamplingWorkspace::<f64>::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SamplingWorkspace<T> {
    scratch: Vec<T>,
    f64_scratch: Vec<f64>,
}

impl<T> SamplingWorkspace<T> {
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            scratch: Vec::new(),
            f64_scratch: Vec::new(),
        }
    }

    fn bytes(&self, requirements: &SamplingRequirements) -> Result<usize, SciOxideError> {
        scioxide_core::backend::checked_add(
            storage_bytes::<T>(self.scratch.capacity().max(requirements.elements))?,
            storage_bytes::<f64>(
                self.f64_scratch
                    .capacity()
                    .max(requirements.f64_scratch_elements),
            )?,
            BYTES,
        )
    }

    /// Reports whether the current capacity covers `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the current capacity already exceeds the byte limit
    ///   of `requirements`.
    /// - `SizeOverflow` if a byte count overflows `usize`.
    pub fn check(&self, requirements: &SamplingRequirements) -> Result<Capacity, SciOxideError> {
        admit_bytes(
            self.bytes(&SamplingRequirements::new(0))?,
            requirements.max_bytes,
        )?;
        Ok(
            if self.scratch.capacity() >= requirements.elements
                && self.f64_scratch.capacity() >= requirements.f64_scratch_elements
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Grows the capacity to cover `requirements`, checking the byte limit first.
    ///
    /// Existing capacity is never released. A failed reservation may keep part
    /// of the growth.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit.
    /// - `SizeOverflow` if a byte count overflows `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn try_reserve(
        &mut self,
        requirements: &SamplingRequirements,
    ) -> Result<(), SciOxideError> {
        admit_bytes(self.bytes(requirements)?, requirements.max_bytes)?;
        // Reservation is fallible and happens before any generator draw, so an
        // allocation failure can never leave a batch half-sampled. Existing
        // capacity is reused across calls to keep repeated batches allocation-free.
        if self.scratch.capacity() < requirements.elements {
            let additional = requirements.elements - self.scratch.len();
            scioxide_core::backend::try_reserve(
                &mut self.scratch,
                additional,
                scioxide_core::quantity!("sampling_workspace"),
            )?;
        }
        if self.f64_scratch.capacity() < requirements.f64_scratch_elements {
            let additional = requirements.f64_scratch_elements - self.f64_scratch.len();
            scioxide_core::backend::try_reserve(
                &mut self.f64_scratch,
                additional,
                scioxide_core::quantity!("sampling_f64_scratch"),
            )?;
        }
        self.check(requirements).map(|_| ())
    }

    /// Reserve and fill exactly `requirements.elements()` staging values.
    pub(crate) fn stage(
        &mut self,
        requirements: &SamplingRequirements,
        value: T,
    ) -> Result<(), SciOxideError>
    where
        T: Clone,
    {
        self.try_reserve(requirements)?;
        self.scratch.clear();
        self.scratch.resize(requirements.elements, value);
        self.f64_scratch
            .resize(requirements.f64_scratch_elements, 0.0);
        Ok(())
    }

    /// Reserve and copy `values` into staging.
    pub(crate) fn stage_copy(
        &mut self,
        requirements: &SamplingRequirements,
        values: &[T],
    ) -> Result<(), SciOxideError>
    where
        T: Clone,
    {
        self.try_reserve(requirements)?;
        self.scratch.clear();
        self.scratch.extend_from_slice(values);
        Ok(())
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.scratch
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.scratch
    }
}

impl<T> Default for SamplingWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Writes sampled rows under a sampling policy; `max_bytes` is the sampler's staging limit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_rows<T: Copy, G: ?Sized, P: crate::StatePolicy<G>>(
    generator: &mut G,
    policy: crate::SamplingPolicy<P>,
    output: &mut [T],
    width: usize,
    zero: T,
    workspace: &mut SamplingWorkspace<T>,
    max_bytes: usize,
    mut sample: impl FnMut(&mut G, &mut [T]) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    sample_rows_scratch(
        generator,
        policy,
        output,
        width,
        zero,
        workspace,
        SamplingRequirements::new(0).limited(max_bytes),
        |generator, row, _, _| sample(generator, row),
    )
}

/// Writes sampled rows with reusable per-draw scratch: `scratch.elements` items of
/// the sample type staged after the rows, and its binary64 scratch. The row
/// staging is derived from the output; the requirements add scratch and limits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_rows_scratch<T: Copy, G: ?Sized, P: crate::StatePolicy<G>>(
    generator: &mut G,
    policy: crate::SamplingPolicy<P>,
    output: &mut [T],
    width: usize,
    zero: T,
    workspace: &mut SamplingWorkspace<T>,
    scratch: SamplingRequirements,
    mut sample: impl FnMut(&mut G, &mut [T], &mut [T], &mut [f64]) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    if width == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("width"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "sample width must be positive",
        ));
    }
    if !output.len().is_multiple_of(width) {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(width),
            scioxide_core::Dimensions::one(output.len() % width),
        ));
    }
    if output.is_empty() {
        return Ok(());
    }
    let staged = policy
        .workspace_requirements(SamplingRequest::rows(output.len() / width, width))?
        .elements;
    let requirements = SamplingRequirements::new(scioxide_core::backend::checked_add(
        staged,
        scratch.elements,
        BYTES,
    )?)
    .limited(scratch.max_bytes)
    .with_f64_scratch(scratch.f64_scratch_elements);
    workspace.stage(&requirements, zero)?;
    let atomic = policy.output_policy() == crate::OutputPolicy::Atomic;
    let SamplingWorkspace {
        scratch: rows,
        f64_scratch,
    } = workspace;
    let (rows, items) = rows.split_at_mut(staged);
    policy.run(generator, |generator| {
        if atomic {
            rows.chunks_exact_mut(width)
                .try_for_each(|row| sample(generator, row, items, f64_scratch))?;
            output.copy_from_slice(rows);
        } else if width == 1 {
            // A scalar sample is written in one store, so it can be published
            // directly through a stack row without workspace staging.
            for value in output {
                let mut row = [zero];
                sample(generator, &mut row, items, f64_scratch)?;
                *value = row[0];
            }
        } else {
            for row in output.chunks_exact_mut(width) {
                sample(generator, rows, items, f64_scratch)?;
                row.copy_from_slice(rows);
            }
        }
        Ok(())
    })
}

/// Byte accounting of one indexed batch: the configured limit, bytes the
/// sampler holds, and the scratch each worker owns.
#[derive(Clone, Copy)]
pub(crate) struct IndexedBytes {
    pub(crate) limit: usize,
    pub(crate) retained: usize,
    pub(crate) per_worker: usize,
}

/// The one indexed root-split shell. Row `i` draws from `root.split(i)`, so
/// results do not depend on the schedule. Each partition owns one scratch
/// value from `make`, reused by its rows; `draw` must not depend on what an
/// earlier row left in it. Ownership slots, per-worker scratch and the bytes
/// the sampler holds count against the limit, and staging receives the
/// remainder. Validation and staging happen before any draw, and the output is
/// written only if every row succeeds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_indexed<T: Copy + Send, R: crate::SplittableRng + Send + Sync, W: Send>(
    root: &R,
    output: &mut [T],
    width: usize,
    zero: T,
    workspace: &mut SamplingWorkspace<T>,
    execution: scioxide_core::ExecutionContext<'_>,
    policy: scioxide_core::ExecutionPolicy,
    bytes: IndexedBytes,
    mut make: impl FnMut() -> Result<W, SciOxideError>,
    draw: impl Fn(&mut R, &mut W, &mut [T]) -> Result<(), SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    if width == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("width"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "sample width must be positive",
        ));
    }
    if !output.len().is_multiple_of(width) {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(width),
            scioxide_core::Dimensions::one(output.len() % width),
        ));
    }
    if output.is_empty() {
        return Ok(());
    }
    execution.check()?;
    let rows = output.len() / width;
    let plan = execution.plan(
        policy,
        rows,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: core::num::NonZeroUsize::MIN,
            worker_limit: core::num::NonZeroUsize::MAX,
        },
    );
    let workers = plan.worker_count();
    let fixed = scioxide_core::backend::checked_add(
        bytes.retained,
        scioxide_core::backend::checked_mul(
            scioxide_core::backend::checked_add(
                core::mem::size_of::<scioxide_core::backend::PartitionSlot<(&mut [T], W)>>(),
                bytes.per_worker,
                BYTES,
            )?,
            workers,
            BYTES,
        )?,
        BYTES,
    )?;
    admit_bytes(fixed, bytes.limit)?;
    workspace.stage(
        &SamplingRequirements::new(output.len()).limited(bytes.limit - fixed),
        zero,
    )?;
    let mut slots = scioxide_core::backend::try_with_capacity(
        workers,
        scioxide_core::quantity!("execution_partitions"),
    )?;
    let mut remaining = workspace.as_mut_slice();
    for part in plan.partitions() {
        let (staged, tail) = remaining.split_at_mut(part.range().len() * width);
        remaining = tail;
        slots.push(scioxide_core::backend::PartitionSlot::new((
            staged,
            make()?,
        )));
    }
    execution.execute_partitioned(plan, &slots, &|partition, slot, _| {
        let (staged, mut scratch) = slot;
        for (i, row) in partition.range().zip(staged.chunks_exact_mut(width)) {
            execution.check()?;
            let id = u64::try_from(i).map_err(|_| {
                SciOxideError::invalid_input(
                    scioxide_core::argument!("output"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "sample stream index exceeds u64",
                )
            })?;
            draw(&mut root.split(id)?, &mut scratch, row)?;
        }
        Ok(())
    })?;
    drop(slots);
    output.copy_from_slice(workspace.as_slice());
    Ok(())
}

/// Writes scalar samples through [`sample_rows`] as one-element rows.
pub(crate) fn sample_scalars<T: Copy, G: ?Sized, P: crate::StatePolicy<G>>(
    generator: &mut G,
    policy: crate::SamplingPolicy<P>,
    output: &mut [T],
    zero: T,
    workspace: &mut SamplingWorkspace<T>,
    max_bytes: usize,
    mut draw: impl FnMut(&mut G) -> Result<T, SciOxideError>,
) -> Result<(), SciOxideError> {
    sample_rows(
        generator,
        policy,
        output,
        1,
        zero,
        workspace,
        max_bytes,
        |generator, row| {
            row[0] = draw(generator)?;
            Ok(())
        },
    )
}

/// Owned scalar batch: one allocation, then `n` draws. On error nothing is
/// returned and the generator keeps the words it consumed.
pub(crate) fn sample_owned<T: Clone, G: ?Sized>(
    generator: &mut G,
    n: usize,
    zero: T,
    mut draw: impl FnMut(&mut G) -> Result<T, SciOxideError>,
) -> Result<Vec<T>, SciOxideError> {
    let mut values = scioxide_core::backend::try_filled(
        n,
        zero,
        scioxide_core::quantity!("distribution_output"),
    )?;
    values.iter_mut().try_for_each(|value| {
        *value = draw(generator)?;
        Ok::<_, SciOxideError>(())
    })?;
    Ok(values)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::SciOxideErrorKind;

    #[test]
    fn prepared_capacity_is_reused() {
        let mut workspace = SamplingWorkspace::new();
        workspace
            .try_reserve(&SamplingRequirements::new(8))
            .unwrap();
        let capacity = workspace.scratch.capacity();
        workspace
            .stage(&SamplingRequirements::new(8), 1_u32)
            .unwrap();
        workspace
            .stage(&SamplingRequirements::new(4), 2_u32)
            .unwrap();
        assert_eq!(workspace.scratch.capacity(), capacity);
        assert_eq!(workspace.as_slice(), &[2; 4]);
    }

    #[test]
    fn retained_capacity_counts_against_limits() {
        let mut workspace = SamplingWorkspace::<f64>::new();
        workspace
            .try_reserve(&SamplingRequirements::new(4))
            .unwrap();
        let small = SamplingRequirements::new(1).limited(16);
        assert_eq!(small.max_bytes(), 16);
        assert_eq!(
            workspace.check(&small).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        let mut fresh = SamplingWorkspace::<f64>::new();
        assert_eq!(fresh.check(&small).unwrap(), Capacity::NeedsGrowth);
        fresh.try_reserve(&small).unwrap();
        assert_eq!(fresh.check(&small).unwrap(), Capacity::Sufficient);
        let over = SamplingRequirements::new(3).limited(16);
        assert_eq!(
            fresh.try_reserve(&over).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            SamplingWorkspace::<f64>::new()
                .try_reserve(&SamplingRequirements::new(usize::MAX))
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
        assert_eq!(
            SamplingWorkspace::<u8>::new()
                .try_reserve(&SamplingRequirements::new(usize::MAX))
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::AllocationFailure
        );
        assert_eq!(
            SamplingRequirements::new(1)
                .limited(8)
                .limited(16)
                .max_bytes(),
            8
        );
    }
}
