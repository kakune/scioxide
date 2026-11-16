//! Sampling methods of vector- and matrix-valued laws.
//!
//! A law implements [`RowLaw`] to draw one row. `vector_sampling!` derives
//! `sample_width`, `workspace_requirements`, `workspace`, `sample`, `sample_n`,
//! `sample_n_into` and `sample_indexed_into` from it. A matrix-valued sample is
//! one row, stored row by row in the order of `shape()`.
use crate::distribution::LawOptions;
use crate::workspace::{IndexedBytes, admit_bytes, storage_bytes};
use crate::{
    BitGenerator, SamplingPolicy, SamplingRequest, SamplingRequirements, SamplingWorkspace,
    SplittableRng, StatePolicy,
};
use alloc::vec::Vec;
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

/// Per-draw scratch reserved by the shared shells and sized by
/// [`RowLaw::scratch`]: `items` of the sample type and `f64` `reals`.
pub(super) struct Scratch<'a, T> {
    pub(super) items: &'a mut [T],
    pub(super) reals: &'a mut [f64],
}

/// Row kernel of one vector- or matrix-valued law.
pub(super) trait RowLaw<S: RealScalar, T: Copy> {
    /// Elements of one sample.
    fn width(&self) -> usize;
    /// Word allowance, byte limit and indexed execution policy.
    fn options(&self) -> LawOptions<S>;
    /// Bytes the prepared law stores, counted against its byte limit.
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    /// Per-draw scratch: elements of the sample type and `f64` elements.
    fn scratch(&self) -> (usize, usize) {
        (0, 0)
    }
    /// Draw one row; the law charges its own per-draw word allowance.
    fn draw<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        row: &mut [T],
        scratch: Scratch<'_, T>,
    ) -> Result<(), SciOxideError>;
}

fn limit<S: RealScalar, T: Copy>(law: &impl RowLaw<S, T>) -> Result<usize, SciOxideError> {
    let max_bytes = law.options().max_bytes();
    let retained = law.retained_bytes()?;
    admit_bytes(retained, max_bytes)?;
    Ok(max_bytes - retained)
}

fn scratch_requirements<S: RealScalar, T: Copy>(
    law: &impl RowLaw<S, T>,
) -> Result<SamplingRequirements, SciOxideError> {
    let (items, reals) = law.scratch();
    Ok(SamplingRequirements::new(items)
        .with_f64_scratch(reals)
        .limited(limit(law)?))
}

/// Staged rows plus per-draw scratch for `samples` rows, limited by the law's
/// byte limit minus its stored parameters. Empty batches need nothing.
pub(super) fn requirements<S: RealScalar, T: Copy, P>(
    law: &impl RowLaw<S, T>,
    policy: SamplingPolicy<P>,
    samples: usize,
) -> Result<SamplingRequirements, SciOxideError> {
    let rows = policy.workspace_requirements(SamplingRequest::rows(samples, law.width()))?;
    if samples == 0 {
        return Ok(rows);
    }
    let scratch = scratch_requirements(law)?;
    Ok(
        SamplingRequirements::new(scioxide_core::backend::checked_add(
            rows.elements(),
            scratch.elements(),
            scioxide_core::quantity!("workspace_bytes"),
        )?)
        .with_f64_scratch(scratch.f64_scratch_elements())
        .limited(scratch.max_bytes()),
    )
}

/// `n` owned rows, stored row by row, with one scratch reused by every draw.
/// The rows and the scratch are checked against the byte limit before the
/// allocation; the scratch is stored after the rows and truncated away.
pub(super) fn sample_n<S: RealScalar, T: Copy, R: BitGenerator + ?Sized>(
    law: &impl RowLaw<S, T>,
    rng: &mut R,
    n: usize,
    zero: T,
) -> Result<Vec<T>, SciOxideError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let (width, (items, reals)) = (law.width(), law.scratch());
    let quantity = scioxide_core::quantity!("vector_sample");
    let total = scioxide_core::backend::checked_mul(n, width, quantity)?;
    let elements = scioxide_core::backend::checked_add(total, items, quantity)?;
    admit_bytes(
        scioxide_core::backend::checked_add(
            storage_bytes::<T>(elements)?,
            storage_bytes::<f64>(reals)?,
            quantity,
        )?,
        limit(law)?,
    )?;
    let mut out = scioxide_core::backend::try_filled(elements, zero, quantity)?;
    let mut reals = scioxide_core::backend::try_filled(reals, 0.0, quantity)?;
    let (rows, items) = out.split_at_mut(total);
    for i in 0..n {
        law.draw(
            rng,
            &mut rows[i * width..(i + 1) * width],
            Scratch {
                items: &mut *items,
                reals: &mut reals,
            },
        )?;
    }
    out.truncate(total);
    Ok(out)
}

/// Complete rows into `output` under `policy`.
pub(super) fn sample_n_into<S: RealScalar, T: Copy, R: BitGenerator, P: StatePolicy<R>>(
    law: &impl RowLaw<S, T>,
    rng: &mut R,
    policy: SamplingPolicy<P>,
    output: &mut [T],
    zero: T,
    workspace: &mut SamplingWorkspace<T>,
) -> Result<(), SciOxideError> {
    crate::workspace::sample_rows_scratch(
        rng,
        policy,
        output,
        law.width(),
        zero,
        workspace,
        scratch_requirements(law)?,
        |rng, row, items, reals| law.draw(rng, row, Scratch { items, reals }),
    )
}

/// Rows drawn from `root.split(row)`, independent of the schedule; each worker
/// owns one scratch allocation.
pub(super) fn sample_indexed_into<
    S: RealScalar,
    T: Copy + Send,
    L: RowLaw<S, T> + Sync,
    R: BitGenerator + SplittableRng + Send + Sync,
>(
    law: &L,
    root: &R,
    output: &mut [T],
    zero: T,
    workspace: &mut SamplingWorkspace<T>,
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let (items, reals) = law.scratch();
    let quantity = scioxide_core::quantity!("vector_indexed_scratch");
    crate::workspace::sample_indexed(
        root,
        output,
        law.width(),
        zero,
        workspace,
        execution,
        law.options().execution_policy(),
        IndexedBytes {
            limit: law.options().max_bytes(),
            retained: law.retained_bytes()?,
            per_worker: scioxide_core::backend::checked_add(
                storage_bytes::<T>(items)?,
                storage_bytes::<f64>(reals)?,
                quantity,
            )?,
        },
        || {
            Ok((
                scioxide_core::backend::try_filled(items, zero, quantity)?,
                scioxide_core::backend::try_filled(reals, 0.0, quantity)?,
            ))
        },
        |rng, (items, reals), row| law.draw(rng, row, Scratch { items, reals }),
    )
}

/// The public sampling forms of a vector- or matrix-valued law.
macro_rules! vector_sampling {
    ($name:ident, $item:ty, $zero:expr $(, $bound:path)?) => {
        impl<S: RealScalar $(+ $bound)?> $name<S> {
            /// Returns the number of elements in one sample.
            ///
            /// A matrix-valued sample is stored row by row.
            pub fn sample_width(&self) -> usize {
                crate::distribution::rows::RowLaw::width(self)
            }
            /// Returns the workspace size that `sample_n_into` needs to draw
            /// `samples` samples under `policy`.
            ///
            /// The size covers the staged output rows and the scratch of one
            /// draw. Its byte limit is [`LawOptions::max_bytes`](crate::LawOptions::max_bytes)
            /// minus the bytes of the stored parameters. An empty batch needs no
            /// workspace.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if the size does not fit in `usize`.
            /// - `ResourceLimit` if the stored parameters already exceed the
            ///   byte limit.
            pub fn workspace_requirements<P>(
                &self,
                policy: crate::SamplingPolicy<P>,
                samples: usize,
            ) -> Result<crate::SamplingRequirements, SciOxideError> {
                crate::distribution::rows::requirements(self, policy, samples)
            }
            /// Returns a workspace reserved for `samples` samples under `policy`.
            ///
            /// # Errors
            ///
            /// - The errors of `workspace_requirements`.
            /// - `ResourceLimit` or `AllocationFailure` if the workspace cannot be
            ///   reserved.
            pub fn workspace<P>(
                &self,
                policy: crate::SamplingPolicy<P>,
                samples: usize,
            ) -> Result<crate::SamplingWorkspace<$item>, SciOxideError> {
                let mut workspace = crate::SamplingWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements(policy, samples)?)?;
                Ok(workspace)
            }
            /// Draws one sample into a new `Vec` of `sample_width()` elements.
            ///
            /// For many samples, `sample_n` allocates once, and
            /// `sample_n_into` with a reused workspace allocates nothing.
            ///
            /// # Errors
            ///
            /// - `ResourceLimit` or `AllocationFailure` if the sample and its
            ///   scratch exceed the byte limit or cannot be allocated.
            /// - `ConvergenceFailure` if the draw exceeds its
            ///   [word limit](crate::LawOptions::sample_word_budget) or its
            ///   candidate limit.
            /// - The numerical errors of the law's sampling method.
            pub fn sample<R: crate::BitGenerator>(
                &self,
                rng: &mut R,
            ) -> Result<alloc::vec::Vec<$item>, SciOxideError> {
                crate::distribution::rows::sample_n(self, rng, 1, $zero)
            }
            /// Draws `n` samples into a new `Vec` of `n · sample_width()`
            /// elements, one row per sample.
            ///
            /// The samples equal `n` calls of [`Self::sample`]. The rows and the
            /// scratch of one draw are allocated once, before the first draw;
            /// `n = 0` returns an empty vector and draws nothing. On error
            /// nothing is returned and `rng` stays advanced by the words it
            /// consumed; to restore it, use `sample_n_into` with
            /// [`SamplingPolicy::rollback`](crate::SamplingPolicy::rollback).
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if `n · sample_width()` does not fit in `usize`.
            /// - `ResourceLimit` or `AllocationFailure` if the rows and the
            ///   scratch exceed the byte limit or cannot be allocated. This is
            ///   checked before any draw.
            /// - The errors of [`Self::sample`] for each draw.
            pub fn sample_n<R: crate::BitGenerator>(
                &self,
                rng: &mut R,
                n: usize,
            ) -> Result<alloc::vec::Vec<$item>, SciOxideError> {
                crate::distribution::rows::sample_n(self, rng, n, $zero)
            }
            /// Draws `output.len() / sample_width()` samples into `output`, one
            /// row per sample.
            ///
            /// A workspace smaller than `workspace_requirements` grows before
            /// the first draw. What `output` and `rng` hold after an error follows
            /// `policy`, as described in [batch sampling](crate#batch-sampling). A
            /// row is written only when its draw is complete. On success the
            /// rows equal those of `sample_n`.
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `output.len()` is not a multiple of
            ///   `sample_width()`.
            /// - `ResourceLimit` or `AllocationFailure` if the workspace cannot
            ///   grow within the byte limit. This is checked before any draw.
            /// - The errors of [`Self::sample`] for each draw.
            pub fn sample_n_into<R: crate::BitGenerator>(
                &self,
                rng: &mut R,
                policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
                output: &mut [$item],
                workspace: &mut crate::SamplingWorkspace<$item>,
            ) -> Result<(), SciOxideError> {
                crate::distribution::rows::sample_n_into(self, rng, policy, output, $zero, workspace)
            }
            /// Draws one sample per row of `output`, row `i` from the stream
            /// `root.split(i)`.
            ///
            /// The rows run under the
            /// [execution policy](crate::LawOptions::execution_policy) of the law.
            /// The samples do not depend on the policy or on the number of
            /// workers. `output` is written only if every row succeeds, and
            /// `root` is never advanced. Each worker allocates the scratch of one
            /// draw.
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `output.len()` is not a multiple of
            ///   `sample_width()`.
            /// - `ResourceLimit` or `AllocationFailure` if the workspace and the
            ///   scratch of all workers exceed the byte limit or cannot be
            ///   allocated. This is checked before any draw.
            /// - The errors of `root.split`, of the executor, and of
            ///   [`Self::sample`] for each draw.
            pub fn sample_indexed_into<
                R: crate::BitGenerator + crate::SplittableRng + Send + Sync,
            >(
                &self,
                root: &R,
                output: &mut [$item],
                workspace: &mut crate::SamplingWorkspace<$item>,
                execution: scioxide_core::ExecutionContext<'_>,
            ) -> Result<(), SciOxideError>
            where
                $item: Send,
                Self: Sync,
            {
                crate::distribution::rows::sample_indexed_into(
                    self, root, output, $zero, workspace, execution,
                )
            }
        }
    };
}
pub(super) use vector_sampling;
