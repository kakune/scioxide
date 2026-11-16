//! Bounded random streams for parallel work.

use scioxide_core::{ExecutionPlan, RealScalar, SciOxideError, WorkPartition};

use crate::traits::{BitGenerator, Rng, SkipAhead, SplittableRng};

/// A generator restricted to a fixed range of `length` consecutive transitions of another.
///
/// A read or skip that would pass the end of the range fails with `OutOfBounds`
/// and does not advance the substream. A failure of the generator itself during
/// a read keeps the transitions it consumed, as it would without the bound. The
/// wrapped generator is not exposed. [`BitGenerator::fill_bytes`] writes
/// successive `next_u64` words as little-endian bytes, and on error the
/// complete words written so far stay in the output.
///
/// [`BitGenerator::position`] is `Some(n)` after `n` transitions of the range,
/// starting at zero. [`SkipAhead::skip_to`] uses the same origin and accepts the
/// end of the range. Backward skips and skips past the end fail without
/// changing the substream.
///
/// ```
/// use scioxide_random::{BitGenerator, BoundedSubstream, SeedableRng, SkipAhead, SplitMix64};
/// let base = SplitMix64::from_seed(42);
/// let mut chunk = BoundedSubstream::for_chunk(&base, 2, 4)?;
/// chunk.skip_to(4)?;
/// assert_eq!(chunk.position(), Some(4));
/// assert_eq!(chunk.remaining(), 0);
/// assert!(chunk.next_u64().is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct BoundedSubstream<R> {
    rng: R,
    start: u128,
    length: u128,
    consumed: u128,
}

impl<R: SkipAhead + Clone> BoundedSubstream<R> {
    /// Restricts `rng`, derived from the caller's `base`, to `length` transitions.
    fn new(
        rng: R,
        length: u128,
        length_argument: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        let start = tracked_position(&rng, scioxide_core::argument!("base"))?;
        start.checked_add(length).ok_or_else(|| {
            SciOxideError::invalid_input(
                length_argument,
                scioxide_core::InvalidInputReason::OutOfRange,
                alloc::format!("bounded substream end overflow: start={start}, length={length}"),
            )
        })?;
        // Validate the actual engine's finite endpoint before publishing a
        // range that promises this many native outputs.
        let mut end = rng.clone();
        end.skip_ahead(length)?;
        Ok(Self {
            rng,
            start,
            length,
            consumed: 0,
        })
    }

    /// Returns chunk `chunk_index` of `base`: transitions
    /// `chunk_index · chunk_len` to `(chunk_index + 1) · chunk_len` of a clone.
    ///
    /// Use it to cut one stream into contiguous chunks. When the work is
    /// described by an execution plan, use [`StreamPartitioner::for_partition`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `chunk_len` is zero, the chunk's start or end overflows,
    ///   or `base` does not track its position.
    /// - Any error of [`SkipAhead::skip_ahead`] on the clone.
    pub fn for_chunk(base: &R, chunk_index: u64, chunk_len: u64) -> Result<Self, SciOxideError> {
        if chunk_len == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("chunk_len"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "chunk length must be positive",
            ));
        }
        tracked_position(base, scioxide_core::argument!("base"))?;
        let offset = u128::from(chunk_index)
            .checked_mul(u128::from(chunk_len))
            .ok_or_else(|| {
                SciOxideError::invalid_input(
                    scioxide_core::argument!("chunk_index"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    alloc::format!(
                        "chunk offset overflow: index={chunk_index}, length={chunk_len}"
                    ),
                )
            })?;
        let mut rng = base.clone();
        rng.skip_ahead(offset)?;
        Self::new(
            rng,
            u128::from(chunk_len),
            scioxide_core::argument!("chunk_len"),
        )
    }

    /// Returns the number of transitions left in the range.
    pub const fn remaining(&self) -> u128 {
        self.length - self.consumed
    }

    #[inline]
    fn try_read<T>(
        &mut self,
        read: impl FnOnce(&mut R) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let remaining = self.length - self.consumed;
        if remaining == 0 {
            return Err(boundary_error());
        }
        let bound = self
            .rng
            .read_transition_bound(crate::traits::private::Token);
        if bound.is_some_and(|bound| bound <= remaining) {
            self.read_in_place(read)
        } else {
            self.read_transactionally(read)
        }
    }

    // A built-in engine's declared worst case fits the remaining range, so the
    // read cannot cross the end and needs no snapshot. Consumed transitions of
    // a failed read stay consumed, as for the unbounded engine.
    #[inline(always)]
    fn read_in_place<T>(
        &mut self,
        read: impl FnOnce(&mut R) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        match read(&mut self.rng) {
            Ok(value) => {
                let before = self.consumed;
                if self.sync_consumed()? == before {
                    return Err(overshoot());
                }
                Ok(value)
            }
            Err(error) => {
                self.sync_consumed()?;
                Err(error)
            }
        }
    }

    // Re-read the engine position after an in-place read; it must stay
    // within the range and never move backward.
    #[inline(always)]
    fn sync_consumed(&mut self) -> Result<u128, SciOxideError> {
        let consumed = self
            .rng
            .position()
            .and_then(|absolute| absolute.checked_sub(self.start))
            .filter(|&consumed| consumed >= self.consumed && consumed <= self.length)
            .ok_or_else(overshoot)?;
        self.consumed = consumed;
        Ok(consumed)
    }

    // One read may consume a variable number of native transitions (for
    // example `next_u64` on a 32-bit engine), so the budget is checked after
    // the fact on a clone: the read is committed only if the engine's new
    // position is within the range. Failed reads retain consumed transitions;
    // successful reads must also make progress.
    #[cold]
    fn read_transactionally<T>(
        &mut self,
        read: impl FnOnce(&mut R) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut candidate = self.rng.clone();
        let result = read(&mut candidate);
        let absolute = tracked_position(&candidate, scioxide_core::argument!("self"))?;
        let end = self.start + self.length;
        let before = self.start + self.consumed;
        if absolute < before || absolute > end || (absolute == before && result.is_ok()) {
            return Err(boundary_error());
        }
        self.rng = candidate;
        self.consumed = absolute - self.start;
        result
    }
}

impl<R: SkipAhead + Clone> SkipAhead for BoundedSubstream<R> {
    /// Advances by `transitions` within the range.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds`, without advancing, if the target lies past the end of the range.
    /// - Any error of the wrapped generator's [`SkipAhead::skip_ahead`].
    fn skip_ahead(&mut self, transitions: u128) -> Result<(), SciOxideError> {
        let target = self.consumed.checked_add(transitions).ok_or_else(|| {
            SciOxideError::out_of_bounds(
                scioxide_core::quantity!("bounded_substream"),
                usize::MAX,
                usize::MAX,
            )
        })?;
        if target > self.length {
            return Err(boundary_error());
        }

        // Work on a clone and cross-check the engine's reported position
        // against our own bookkeeping, so a mismatch leaves `self` untouched and
        // a faulty `SkipAhead` implementation cannot silently escape the range.
        let mut candidate = self.rng.clone();
        candidate.skip_ahead(transitions)?;
        let absolute = tracked_position(&candidate, scioxide_core::argument!("self"))?;
        let expected = self.start + target;
        if absolute != expected {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                alloc::format!(
                    "bounded skip reported absolute position {absolute}, expected {expected}"
                ),
            ));
        }
        self.rng = candidate;
        self.consumed = target;
        Ok(())
    }
    // Reads consume the inner engine's transitions one-for-one.
    #[inline]
    fn read_transition_bound(&self, token: crate::traits::private::Token) -> Option<u128> {
        self.rng.read_transition_bound(token)
    }
}

impl<R: SkipAhead + Clone> BitGenerator for BoundedSubstream<R> {
    #[inline]
    fn position(&self) -> Option<u128> {
        Some(self.consumed)
    }
    fn stream_identity(&self) -> Option<crate::StreamIdentity<'_>> {
        self.rng.stream_identity()
    }
    #[inline]
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        self.try_read(BitGenerator::next_u32)
    }
    #[inline]
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.try_read(BitGenerator::next_u64)
    }
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.try_read(BitGenerator::next_uniform_u32)
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.try_read(BitGenerator::next_uniform_u64)
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
        crate::util::fill_bytes_with(output, 8, || self.next_u64())
    }
}
impl<S: RealScalar, R: Rng<S> + SkipAhead + Clone> Rng<S> for BoundedSubstream<R> {
    #[inline]
    fn next_float(&mut self) -> Result<S, SciOxideError> {
        self.try_read(Rng::next_float)
    }
}
crate::transaction::clone_snapshot!([<R: SkipAhead + Clone>] BoundedSubstream<R>);

/// Absolute position of `rng`; an untracked engine is invalid input against `argument`.
fn tracked_position<R: SkipAhead>(
    rng: &R,
    argument: scioxide_core::Argument,
) -> Result<u128, SciOxideError> {
    rng.position().ok_or_else(|| {
        SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "an RNG with a tracked absolute position is required",
        )
    })
}

#[cold]
#[inline(never)]
fn boundary_error() -> SciOxideError {
    SciOxideError::out_of_bounds(
        scioxide_core::quantity!("bounded_substream"),
        usize::MAX,
        usize::MAX,
    )
}

/// A built-in engine moved outside its declared per-read transition bound.
#[cold]
#[inline(never)]
fn overshoot() -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("self"),
        scioxide_core::InvalidInputReason::Inconsistent,
        "bounded read moved the engine outside its declared transition bound",
    )
}

/// Assigns a reproducible, bounded random stream to each partition of an execution plan.
///
/// The stream of a partition depends only on the partition's identity
/// ([`WorkPartition`]), the base generator and a domain separator you choose.
/// It does not depend on thread scheduling or on the order in which workers
/// finish, so a parallel computation draws the same numbers for a given plan.
/// Use different domain separators for unrelated computations that share a
/// base generator and plan.
///
/// The partition identity and domain are mixed into a 128-bit key whose low
/// and high 64-bit words select the stream and substream of
/// [`SplittableRng::substream`]. The selectors are therefore arbitrary 64-bit
/// values, not small indices: a guarantee a generator states for small
/// indices, such as non-overlap of MRG32k3a substreams `j < 2⁵¹`, does not
/// apply. Whether two partitions overlap within `stream_len` outputs depends on
/// the generator's period and stream spacing; distinct selectors alone do not
/// establish statistical independence.
///
/// ```
/// use core::num::NonZeroUsize;
/// use scioxide_core::{ExecutionContext, ExecutionPolicy, PlanLimits};
/// use scioxide_random::{BitGenerator, Pcg64, SeedableRng, StreamPartitioner};
/// let one = NonZeroUsize::MIN;
/// let limits = PlanLimits { work_per_item: one, auto_threshold: one, worker_limit: one };
/// let plan = ExecutionContext::sequential().plan(ExecutionPolicy::Sequential, 8, limits);
/// let partitioner = StreamPartitioner::from_plan(plan, 17);
/// let base = Pcg64::from_seed(42);
/// for partition in plan.partitions() {
///     let mut stream = partitioner.for_partition(&base, &partition, 4)?;
///     stream.next_u64()?;
///     assert_eq!(stream.remaining(), 3);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct StreamPartitioner {
    plan: ExecutionPlan,
    domain: u64,
}

impl StreamPartitioner {
    /// Creates a partitioner for `plan` with the domain separator `domain`.
    pub const fn from_plan(plan: ExecutionPlan, domain: u64) -> Self {
        Self { plan, domain }
    }

    /// Returns the execution plan whose partitions this partitioner accepts.
    pub const fn plan(&self) -> ExecutionPlan {
        self.plan
    }

    /// Returns the domain separator.
    pub const fn domain(&self) -> u64 {
        self.domain
    }

    /// Returns the number of partitions in the plan.
    pub const fn num_partitions(&self) -> usize {
        self.plan.worker_count()
    }

    /// Returns the stream of `partition`, restricted to `length` transitions.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `partition` is not part of the plan, `length` is
    ///   zero, or the derived stream does not track its position.
    /// - Any error of [`SplittableRng::substream`], or of
    ///   [`SkipAhead::skip_ahead`] when checking the end of the range, for
    ///   example when the generator's stream is shorter than `length`.
    pub fn for_partition<R: SplittableRng + SkipAhead + Clone>(
        &self,
        base: &R,
        partition: &WorkPartition,
        stream_len: u128,
    ) -> Result<BoundedSubstream<R>, SciOxideError> {
        if stream_len == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("stream_len"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "partition stream length must be positive",
            ));
        }

        let expected = self
            .plan
            .partition(partition.worker_index())
            .ok_or_else(|| {
                SciOxideError::invalid_input(
                    scioxide_core::argument!("partition"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    alloc::format!(
                        "partition index {} is outside plan with {} partitions",
                        partition.worker_index(),
                        self.plan.worker_count()
                    ),
                )
            })?;
        if expected != *partition {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("partition"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "partition identity or range does not belong to the bound execution plan",
            ));
        }

        // The 128-bit partition key is split into the generator's two 64-bit
        // selectors (low word = stream, high word = subsequence). The engine's
        // stream spacing determines whether the resulting ranges overlap.
        let key = partition.id().stream_key(self.domain);
        #[allow(clippy::cast_possible_truncation)]
        let stream_id = key as u64;
        let subsequence_id = u64::try_from(key >> u64::BITS).map_err(|_| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("partition"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "partition stream key does not fit the generator stream contract",
            )
        })?;
        BoundedSubstream::new(
            base.substream(stream_id, subsequence_id)?,
            stream_len,
            scioxide_core::argument!("stream_len"),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::num::NonZeroUsize;

    use scioxide_core::{ExecutionContext, ExecutionPolicy, Executor, PartitionFn};

    use crate::prng::{Mrg32k3a, Pcg64, SplitMix64};
    use crate::traits::SeedableRng;

    struct PlanningExecutor(NonZeroUsize);

    impl Executor for PlanningExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            self.0
        }

        fn execute(
            &self,
            _context: ExecutionContext<'_>,
            _plan: ExecutionPlan,
            _operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("executor"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "planning-only executor cannot execute work",
            ))
        }
    }

    fn execution_plan(work_items: usize, workers: usize) -> ExecutionPlan {
        let workers = NonZeroUsize::new(workers).unwrap();
        let executor = PlanningExecutor(workers);
        ExecutionContext::new(&executor).plan(
            ExecutionPolicy::Parallel(workers),
            work_items,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: workers,
            },
        )
    }

    #[test]
    fn planned_streams_are_bounded_and_deterministic() {
        let base = Pcg64::from_seed(42);
        let plan = execution_plan(40, 4);
        let partition = plan.partitions().nth(2).unwrap();
        let partitioner = StreamPartitioner::from_plan(plan, 17);
        let mut first = partitioner.for_partition(&base, &partition, 2).unwrap();
        let mut replay = partitioner.for_partition(&base, &partition, 2).unwrap();
        assert_eq!(first.next_u64().unwrap(), replay.next_u64().unwrap());
        assert_eq!(first.next_u64().unwrap(), replay.next_u64().unwrap());
        assert_eq!(first.position(), Some(2));
        assert_eq!(first.remaining(), 0);
        let error = first.next_u64().unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
        assert_eq!(first.position(), Some(2));
    }

    #[test]
    fn partition_membership_domain_and_lengths_are_checked() {
        let base = Pcg64::from_seed(42);
        let plan = execution_plan(40, 4);
        let partition = plan.partitions().nth(1).unwrap();
        let partitioner = StreamPartitioner::from_plan(plan, 17);
        let foreign = execution_plan(40, 3).partitions().nth(1).unwrap();
        let error = partitioner.for_partition(&base, &foreign, 1).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        let error = partitioner.for_partition(&base, &partition, 0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);

        let mut first = partitioner.for_partition(&base, &partition, 1).unwrap();
        let mut other_domain = StreamPartitioner::from_plan(plan, 18)
            .for_partition(&base, &partition, 1)
            .unwrap();
        assert_ne!(first.next_u64().unwrap(), other_domain.next_u64().unwrap());
    }

    #[test]
    fn chunk_matches_sequential_generation_and_enforces_its_end() {
        let base = SplitMix64::from_seed(42);
        let mut chunk = BoundedSubstream::for_chunk(&base, 1, 2).unwrap();
        let mut sequential = base;
        sequential.skip_ahead(2).unwrap();
        assert_eq!(chunk.next_u64().unwrap(), sequential.next_u64().unwrap());
        assert_eq!(chunk.next_u64().unwrap(), sequential.next_u64().unwrap());
        assert!(chunk.next_u32().is_err());
    }

    #[test]
    fn relative_skip_cannot_cross_the_boundary() {
        let base = SplitMix64::from_seed(42);
        let mut chunk = BoundedSubstream::for_chunk(&base, 0, 10).unwrap();
        chunk.skip_ahead(9).unwrap();
        assert!(chunk.skip_ahead(2).is_err());
        assert_eq!(chunk.position(), Some(9));
        let _: f64 = chunk.next_float().unwrap();
        assert_eq!(chunk.remaining(), 0);
    }

    #[test]
    fn trait_skips_use_the_bounded_origin_and_reject_without_mutation() {
        fn seek<G: SkipAhead>(rng: &mut G, position: u128) -> Result<(), SciOxideError> {
            rng.skip_to(position)
        }
        let base = SplitMix64::from_seed(42);
        let mut chunk = BoundedSubstream::for_chunk(&base, 2, 4).unwrap();
        assert_eq!(chunk.position(), Some(0));
        seek(&mut chunk, 2).unwrap();
        let mut expected = chunk.clone();
        for target in [1, 5, u128::MAX] {
            assert_eq!(
                seek(&mut chunk, target).unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::OutOfBounds
            );
            assert_eq!(chunk.position(), Some(2));
        }
        assert!(chunk.skip_ahead(u128::MAX).is_err());
        assert_eq!(chunk.next_u64().unwrap(), expected.next_u64().unwrap());
        seek(&mut chunk, 4).unwrap();
        chunk.skip_ahead(0).unwrap();
        assert_eq!(chunk.position(), Some(4));
        assert_eq!(chunk.remaining(), 0);
        assert!(chunk.next_u64().is_err());
    }

    #[test]
    fn multi_transition_read_is_rejected_transactionally() {
        let base = Mrg32k3a::from_seed(42);
        let mut chunk = BoundedSubstream::for_chunk(&base, 0, 1).unwrap();
        assert!(chunk.next_u64().is_err());
        assert_eq!(chunk.position(), Some(0));
        assert!(chunk.next_u32().is_ok());
        assert_eq!(chunk.position(), Some(1));
    }

    #[test]
    fn failed_reads_preserve_consumption_inside_the_bound() {
        for length in [1, 4096] {
            let mut stream =
                BoundedSubstream::for_chunk(&Mrg32k3a::from_seed(7), 0, length).unwrap();
            let error = stream
                .try_read::<()>(|rng| {
                    rng.next_u32()?;
                    Err(SciOxideError::external(
                        "provider failed after consuming one word",
                    ))
                })
                .unwrap_err();
            assert!(
                matches!(error.view(), scioxide_core::SciOxideErrorView::External { detail }
                if detail == "provider failed after consuming one word")
            );
            assert_eq!(stream.position(), Some(1));
            assert_eq!(stream.remaining(), u128::from(length - 1));
        }
        // A failed read that crosses the boundary still publishes no state.
        let mut stream = BoundedSubstream::for_chunk(&Mrg32k3a::from_seed(7), 0, 1).unwrap();
        let error = stream
            .try_read::<()>(|rng| {
                rng.next_u64()?;
                Err(SciOxideError::external("provider failed beyond the bound"))
            })
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
        assert_eq!(stream.position(), Some(0));
    }

    // Mixed reads through a chunk equal the unbounded engine skipped to the
    // chunk start; the range end is exact and a refused read keeps state.
    fn assert_matches_unbounded<R>(base: &R, length: u64)
    where
        R: Rng<f64> + SkipAhead + Clone,
    {
        let mut chunk = BoundedSubstream::for_chunk(base, 3, length).unwrap();
        let mut reference = base.clone();
        reference.skip_ahead(3 * u128::from(length)).unwrap();
        for step in 0..length {
            match step % 3 {
                0 => assert_eq!(chunk.next_u64().unwrap(), reference.next_u64().unwrap()),
                1 => assert_eq!(chunk.next_u32().unwrap(), reference.next_u32().unwrap()),
                _ => assert_eq!(
                    chunk.next_float().unwrap().to_bits(),
                    reference.next_float().unwrap().to_bits()
                ),
            }
            assert_eq!(chunk.position(), Some(u128::from(step + 1)));
        }
        let error = chunk.next_uniform_u64().unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::OutOfBounds);
        assert_eq!(
            (chunk.position(), chunk.remaining()),
            (Some(u128::from(length)), 0)
        );
    }

    #[test]
    fn in_place_reads_match_the_unbounded_stream_to_the_end() {
        assert_matches_unbounded(&SplitMix64::from_seed(1346), 97);
        assert_matches_unbounded(&Pcg64::from_seed(1346), 97);
        assert_matches_unbounded(&crate::prng::Philox4x64::from_seed(1346), 97);
        assert_matches_unbounded(&crate::prng::Threefry4x64::from_seed(1346), 97);
    }

    #[test]
    fn variable_width_reads_fall_back_to_snapshots_near_the_end() {
        // Each normalized MRG32k3a word takes three digits; once fewer than
        // 3,072 transitions remain, reads run on snapshots and a read that
        // would cross the end fails without advancing.
        let base = Mrg32k3a::from_seed(1346);
        let length = 3 * 1_024 + 4;
        let mut chunk = BoundedSubstream::for_chunk(&base, 0, length).unwrap();
        let mut reference = base;
        while chunk.remaining() >= 3 {
            assert_eq!(
                chunk.next_uniform_u64().unwrap(),
                reference.next_uniform_u64().unwrap()
            );
            assert_eq!(chunk.position(), reference.position());
        }
        assert_eq!(chunk.remaining(), 1);
        assert_eq!(
            chunk.next_uniform_u64().unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::OutOfBounds
        );
        assert_eq!(chunk.remaining(), 1);
        assert_eq!(chunk.next_u32().unwrap(), reference.next_u32().unwrap());
        assert_eq!(chunk.remaining(), 0);
    }

    #[test]
    fn nested_substreams_enforce_the_inner_range() {
        let base = SplitMix64::from_seed(1346);
        let outer = BoundedSubstream::for_chunk(&base, 1, 10).unwrap();
        let mut inner = BoundedSubstream::for_chunk(&outer, 1, 4).unwrap();
        let mut reference = base;
        reference.skip_ahead(14).unwrap();
        for _ in 0..4 {
            assert_eq!(inner.next_u64().unwrap(), reference.next_u64().unwrap());
        }
        assert_eq!(
            inner.next_u64().unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::OutOfBounds
        );
        assert_eq!(inner.position(), Some(4));
    }

    #[test]
    fn untracked_generators_cannot_create_bounded_ranges() {
        let mut base = SplitMix64::from_seed(42);
        base.skip_ahead(u128::MAX).unwrap();
        base.skip_ahead(1).unwrap();
        let error = BoundedSubstream::for_chunk(&base, 0, 1).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn partition_order_and_parent_progress_do_not_change_selected_streams() {
        let mut base = Pcg64::from_seed(42);
        let plan = execution_plan(40, 4);
        let partitions: Vec<_> = plan.partitions().collect();
        let partitioner = StreamPartitioner::from_plan(plan, 17);
        let draw = |base: &Pcg64, partition: &WorkPartition| {
            let mut stream = partitioner.for_partition(base, partition, 3).unwrap();
            let root = stream.clone();
            let identity = root.stream_identity();
            let a: f32 = stream.next_float().unwrap();
            let b: f64 = stream.next_float().unwrap();
            let c = stream.next_u64().unwrap();
            assert!(stream.next_u32().is_err());
            assert_eq!(stream.stream_identity(), identity);
            (
                identity.map(|id| (id.algorithm(), id.selectors())),
                a.to_bits(),
                b.to_bits(),
                c,
            )
        };
        let expected: Vec<_> = partitions.iter().map(|p| draw(&base, p)).collect();
        base.skip_ahead(987).unwrap();
        for index in [3, 1, 0, 2] {
            assert_eq!(draw(&base, &partitions[index]), expected[index]);
        }
    }
}
