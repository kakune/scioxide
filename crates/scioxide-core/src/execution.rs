//! Execution contexts, executors, partitions, clocks, cancellation and processor features.
//!
//! See [execution contexts](crate#execution-contexts) for an overview.

#![deny(unsafe_code)]

use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::num::NonZeroUsize;
use core::ops::Range;
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;

use crate::{Capability, ExecutionStopReason, SciOxideError};

#[cfg(feature = "std")]
mod std_threads;

#[cfg(feature = "std")]
pub use std_threads::ThreadExecutor;

/// How many workers an operation may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[must_use]
pub enum ExecutionPolicy {
    /// Use one worker and process items in order.
    Sequential,
    /// Use several workers only when the operation's work reaches its threshold
    /// and each worker gets at least the executor's minimum work.
    #[default]
    Auto,
    /// Use at most the given number of workers.
    Parallel(NonZeroUsize),
}

/// Operation-specific limits used by [`ExecutionContext::plan`] to choose the number of partitions.
///
/// Every policy is bounded by `worker_limit`, the context's worker budget and
/// the number of work items. `Auto` also requires each partition to carry at
/// least [`Executor::auto_min_work_per_worker`] work units; if fewer than two
/// workers qualify, the whole operation runs on the calling thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct PlanLimits {
    /// Smallest number of work items for which `Auto` may use several workers.
    pub auto_threshold: NonZeroUsize,
    /// Estimated work units per item, used only by `Auto`.
    ///
    /// One unit is roughly one scalar operation. Use one when no estimate is
    /// available. The estimate affects only how work is divided.
    pub work_per_item: NonZeroUsize,
    /// Maximum number of partitions for this operation.
    pub worker_limit: NonZeroUsize,
}

/// Identity of a work partition, stable across runs and derived from its parents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct PartitionId(u128);

impl PartitionId {
    /// Identity of a top-level execution context.
    pub const ROOT: Self = Self(0xcbf2_9ce4_8422_2325_9e37_79b9_7f4a_7c15);

    /// Returns the identity of child partition `index`.
    pub const fn child(self, index: usize) -> Self {
        // A cheap multiplicative mix with a xor-rotate avalanche. Identities, and
        // the stream keys derived from them, depend only on the position in the
        // partition tree, never on which worker ran a partition or when. The
        // `+ 1` keeps child 0 distinct from its parent.
        let mixed = self
            .0
            .wrapping_add((index as u128).wrapping_add(1))
            .wrapping_mul(0x0000_0001_0000_01b3_9e37_79b9_7f4a_7c15);
        Self(mixed ^ mixed.rotate_left(47))
    }

    /// Returns a deterministic key for seeding the random stream `domain` of this partition.
    pub const fn stream_key(self, domain: u64) -> u128 {
        let mixed = self.0 ^ (domain as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        mixed ^ mixed.rotate_left(31)
    }
}

/// One contiguous, nonempty range of work items in an [`ExecutionPlan`].
///
/// Partitions of a plan do not overlap. Their identity and index do not
/// depend on the order in which they complete. See [`ExecutionPlan`] for an
/// example.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct WorkPartition {
    id: PartitionId,
    worker_index: usize,
    range: Range<usize>,
}

impl WorkPartition {
    /// Returns the partition identity.
    pub const fn id(&self) -> PartitionId {
        self.id
    }

    /// Returns the zero-based partition index.
    pub const fn worker_index(&self) -> usize {
        self.worker_index
    }

    /// Returns the half-open range of work items in this partition.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

/// Division of a number of work items into balanced contiguous partitions.
///
/// Created by [`ExecutionContext::plan`]. Any work count is valid, including
/// zero (no partitions) and `usize::MAX`. The partitions cover all items
/// without overlap. [`Self::reduce`] allocates storage for the partition
/// results.
///
/// ```
/// use core::num::NonZeroUsize;
/// use scioxide_core::{ExecutionContext, ExecutionPolicy, PlanLimits};
///
/// let limits = PlanLimits {
///     work_per_item: NonZeroUsize::MIN,
///     auto_threshold: NonZeroUsize::MIN,
///     worker_limit: NonZeroUsize::MIN,
/// };
/// let plan = ExecutionContext::sequential().plan(ExecutionPolicy::Sequential, 8, limits);
/// let partition = plan.partition(0).unwrap();
/// assert_eq!(partition.range(), 0..8);
/// assert!(plan.partition(1).is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct ExecutionPlan {
    parent: PartitionId,
    work_items: usize,
    worker_count: usize,
}

impl ExecutionPlan {
    fn resolve(
        parent: PartitionId,
        policy: ExecutionPolicy,
        work_items: usize,
        limits: PlanLimits,
        available: NonZeroUsize,
        minimum_work: NonZeroUsize,
    ) -> Self {
        let worker_count = if work_items == 0 {
            0
        } else {
            match policy {
                ExecutionPolicy::Sequential => 1,
                // Below the caller's threshold, dispatch overhead outweighs the work. Every
                // policy is also capped by the work-item count, so no partition is empty.
                ExecutionPolicy::Auto if work_items < limits.auto_threshold.get() => 1,
                ExecutionPolicy::Auto => {
                    // Division avoids overflowing a total-work product. Ceiling
                    // division ensures even the smallest balanced partition
                    // contains the executor's minimum estimated work.
                    let minimum_items = minimum_work.get().div_ceil(limits.work_per_item.get());
                    available
                        .get()
                        .min(limits.worker_limit.get())
                        .min((work_items / minimum_items).max(1))
                }
                ExecutionPolicy::Parallel(requested) => requested
                    .get()
                    .min(available.get())
                    .min(limits.worker_limit.get())
                    .min(work_items),
            }
        };
        Self {
            parent,
            work_items,
            worker_count,
        }
    }

    /// Returns the number of work items.
    pub const fn work_items(self) -> usize {
        self.work_items
    }

    /// Returns the number of partitions.
    pub const fn worker_count(self) -> usize {
        self.worker_count
    }

    /// Returns the partitions in index order.
    pub const fn partitions(self) -> WorkPartitions {
        WorkPartitions {
            plan: self,
            next_worker: 0,
        }
    }

    /// Returns partition `index`, or `None` if the plan has no such partition.
    pub fn partition(self, index: usize) -> Option<WorkPartition> {
        if index >= self.worker_count {
            return None;
        }
        let base = self.work_items / self.worker_count;
        let remainder = self.work_items % self.worker_count;
        // All intermediates are bounded by work_items, including usize::MAX.
        let start = base * index + index.min(remainder);
        let end = start + base + usize::from(index < remainder);
        Some(WorkPartition {
            id: self.parent.child(index),
            worker_index: index,
            range: start..end,
        })
    }

    /// Combines one result per partition with `reduce`, in partition index order.
    ///
    /// Executors may finish partitions in any order. Reducing in index order
    /// makes the result independent of scheduling.
    ///
    /// # Errors
    ///
    /// - `ExecutionFailure` if a result belongs to no partition of this plan,
    ///   or a partition has no result or more than one.
    /// - `AllocationFailure` if storage for the results cannot be reserved.
    /// - Any error returned by `reduce`, unchanged.
    pub fn reduce<T, A>(
        self,
        results: impl IntoIterator<Item = (WorkPartition, T)>,
        initial: A,
        reduce: impl FnMut(A, T) -> Result<A, SciOxideError>,
    ) -> Result<A, SciOxideError> {
        let mut slots = Vec::new();
        slots.try_reserve_exact(self.worker_count).map_err(|_| {
            SciOxideError::allocation_failure(
                crate::quantity!("deterministic_reduction_slots"),
                self.worker_count,
            )
        })?;
        slots.resize_with(self.worker_count, || None);

        // Results arrive in completion order. Placing them by partition index and
        // folding in that order makes floating-point reductions bitwise identical
        // under any scheduling. Unknown, duplicate, or missing results expose a
        // faulty executor instead of silently changing the answer.
        for (partition, value) in results {
            let index = partition.worker_index();
            if self.partition(index).as_ref() != Some(&partition) {
                return Err(SciOxideError::execution_failure(
                    None,
                    crate::ExecutionFailureReason::UnknownPartition,
                    "executor returned an unknown partition identity",
                ));
            }
            if slots[index].replace(value).is_some() {
                return Err(SciOxideError::execution_failure(
                    Some(index),
                    crate::ExecutionFailureReason::DuplicatePartition,
                    "executor returned a duplicate partition result",
                ));
            }
        }

        if let Some(index) = slots.iter().position(Option::is_none) {
            return Err(SciOxideError::execution_failure(
                Some(index),
                crate::ExecutionFailureReason::MissingPartition,
                "executor omitted a partition result",
            ));
        }

        slots.into_iter().flatten().try_fold(initial, reduce)
    }
}

/// Iterator over the partitions of an [`ExecutionPlan`], in index order.
///
/// Returned by [`ExecutionPlan::partitions`].
#[derive(Debug, Clone)]
#[must_use]
pub struct WorkPartitions {
    plan: ExecutionPlan,
    next_worker: usize,
}
impl Iterator for WorkPartitions {
    type Item = WorkPartition;
    fn next(&mut self) -> Option<Self::Item> {
        let partition = self.plan.partition(self.next_worker)?;
        self.next_worker += 1;
        Some(partition)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.plan.worker_count - self.next_worker;
        (remaining, Some(remaining))
    }
}
impl ExactSizeIterator for WorkPartitions {}

/// A value that one partition can take exactly once, even from another thread.
///
/// Used with [`ExecutionContext::execute_partitioned`] to hand each partition
/// its own input, such as a chunk of an output buffer. [`Self::take`] returns
/// the value to the first caller and `None` afterwards. The value must be
/// `Send` to share a slot between workers.
///
/// ```
/// use scioxide_core::backend::PartitionSlot;
///
/// let slot = PartitionSlot::new([1.0_f64, 2.0]);
/// assert_eq!(slot.take(), Some([1.0, 2.0]));
/// assert!(slot.was_claimed());
/// assert!(slot.take().is_none());
/// ```
#[must_use]
pub struct PartitionSlot<T> {
    claimed: AtomicBool,
    value: UnsafeCell<Option<T>>,
}

impl<T> PartitionSlot<T> {
    /// Creates a slot holding `value`.
    pub const fn new(value: T) -> Self {
        Self {
            claimed: AtomicBool::new(false),
            value: UnsafeCell::new(Some(value)),
        }
    }

    /// Returns an empty slot, marked as already taken, for unused entries of a fixed array.
    pub const fn empty() -> Self {
        Self {
            claimed: AtomicBool::new(true),
            value: UnsafeCell::new(None),
        }
    }

    /// Returns the value on the first call and `None` on every later call.
    #[expect(
        unsafe_code,
        reason = "atomic single-consumer claim grants exclusive cell access"
    )]
    pub fn take(&self) -> Option<T> {
        if self
            .claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return None;
        }
        // SAFETY: the successful atomic claim gives this caller exclusive
        // access. Every later caller returns before accessing the cell.
        unsafe { (*self.value.get()).take() }
    }

    /// Returns whether the value has been taken.
    pub fn was_claimed(&self) -> bool {
        self.claimed.load(Ordering::Acquire)
    }
}

#[expect(
    unsafe_code,
    reason = "only the winning atomic claimant accesses the Send value"
)]
// SAFETY: `take` grants cell access to exactly one caller through an atomic
// claim, and `T: Send` permits that value to move to an executor worker.
unsafe impl<T: Send> Sync for PartitionSlot<T> {}

/// Callback run for one partition, with the partition's input and a child execution context.
///
/// The child context cannot outlive the call.
pub type PartitionFn<'a, I = ()> =
    dyn for<'s> Fn(WorkPartition, I, ExecutionContext<'s>) -> Result<(), SciOxideError> + Sync + 'a;

/// Scheduler that runs the partitions of an [`ExecutionPlan`].
///
/// An implementation must finish every partition it started and release all
/// borrows before it returns or unwinds; it must not abandon a running callback
/// on a timeout. It may run partitions in any order. Callers combine results in
/// [`ExecutionPlan::partitions`] order.
pub trait Executor: Sync {
    /// Returns the maximum number of workers.
    fn worker_budget(&self) -> NonZeroUsize;

    /// Returns the minimum estimated work units per worker under `Auto`.
    ///
    /// The default is one, so only [`PlanLimits::auto_threshold`] applies.
    /// Executors with expensive dispatch may raise it. `Parallel` policies
    /// ignore it.
    fn auto_min_work_per_worker(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }

    /// Runs every partition of `plan` exactly once.
    ///
    /// # Errors
    ///
    /// - The error of the failed partition with the lowest index, returned
    ///   after all started partitions have finished.
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError>;
}

/// Executor that runs all partitions in order on the calling thread.
///
/// Panics in callbacks propagate.
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialExecutor;

impl Executor for SequentialExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }

    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let mut result = Ok(());
        for partition in plan.partitions() {
            result = result.and(invoke_partition(context, partition, operation));
        }
        result
    }
}

// Membership is scheduling bookkeeping, never numerical capability discovery.
// Context-carried budgets remain authoritative when scopes cross thread boundaries.
#[cfg(feature = "std")]
std::thread_local! { static EXECUTION_ACTIVE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) }; }
#[cfg(feature = "std")]
struct ExecutionMembership(Option<bool>);
#[cfg(feature = "std")]
impl ExecutionMembership {
    fn enter() -> Self {
        Self(
            EXECUTION_ACTIVE
                .try_with(|active| active.replace(true))
                .ok(),
        )
    }
    fn active() -> bool {
        EXECUTION_ACTIVE
            .try_with(core::cell::Cell::get)
            .unwrap_or(true)
    }
}
#[cfg(feature = "std")]
impl Drop for ExecutionMembership {
    fn drop(&mut self) {
        if let Some(previous) = self.0 {
            let _ = EXECUTION_ACTIVE.try_with(|active| active.set(previous));
        }
    }
}

fn invoke_partition(
    context: ExecutionContext<'_>,
    partition: WorkPartition,
    operation: &PartitionFn<'_>,
) -> Result<(), SciOxideError> {
    if context.callback_boundary {
        return operation(partition, (), context);
    }
    #[cfg(feature = "std")]
    let _membership = ExecutionMembership::enter();
    context.check()?;
    let child = context.child(partition.id());
    operation(partition, (), child)
}

static SEQUENTIAL_EXECUTOR: SequentialExecutor = SequentialExecutor;

/// Error returned by a clock, entropy source or other capability provider.
#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub struct CapabilityError {
    detail: alloc::borrow::Cow<'static, str>,
}

impl CapabilityError {
    /// Creates an error with the given description.
    pub fn new(detail: impl Into<alloc::borrow::Cow<'static, str>>) -> Self {
        Self {
            detail: detail.into(),
        }
    }

    /// Returns the description by value.
    pub fn into_detail(self) -> alloc::borrow::Cow<'static, str> {
        self.detail
    }

    /// Returns the description.
    pub fn detail(&self) -> &str {
        self.detail.as_ref()
    }
}

/// Source of cooperative cancellation requests.
pub trait Cancellation: Sync {
    /// Returns whether cancellation has been requested.
    fn is_cancelled(&self) -> bool;
}

/// Cancellation source that never requests cancellation.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NeverCancelled;

impl Cancellation for NeverCancelled {
    #[inline]
    fn is_cancelled(&self) -> bool {
        false
    }
}

static NEVER_CANCELLED: NeverCancelled = NeverCancelled;

/// Thread-safe cancellation flag for use with [`ExecutionContext::with_cancellation`].
#[derive(Debug, Default)]
pub struct CancellationToken {
    cancelled: AtomicBool,
}

impl CancellationToken {
    /// Creates a token that has not been cancelled.
    pub const fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
        }
    }

    /// Requests cancellation; repeated calls have no further effect.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Clears the request so the token can be reused for another solve.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }
}

impl Cancellation for CancellationToken {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Monotonic timestamp returned by a [`Clock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct MonotonicInstant(Duration);

impl MonotonicInstant {
    /// Creates a timestamp from the time elapsed since the clock's own starting point.
    pub const fn from_duration_since_epoch(duration: Duration) -> Self {
        Self(duration)
    }

    /// Returns the time elapsed since the clock's starting point.
    pub const fn duration_since_epoch(self) -> Duration {
        self.0
    }

    /// Returns the timestamp `duration` later, or `None` on overflow.
    pub fn checked_add(self, duration: Duration) -> Option<Self> {
        self.0.checked_add(duration).map(Self)
    }
}

/// Monotonic clock used for deadlines.
pub trait Clock: Sync {
    /// Returns the current time.
    fn now(&self) -> Result<MonotonicInstant, CapabilityError>;
}

/// A clock bound to one time origin, used by contexts and deadlines.
///
/// Keep the binding alive while contexts or deadlines borrow it. Two bindings
/// are different origins even when they wrap the same clock, and a context
/// rejects a deadline from a binding other than its own.
#[must_use]
pub struct ClockBinding<'a> {
    provider: &'a dyn Clock,
}
impl core::fmt::Debug for ClockBinding<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClockBinding").finish_non_exhaustive()
    }
}
impl<'a> ClockBinding<'a> {
    /// Binds `provider`.
    pub const fn new(provider: &'a dyn Clock) -> Self {
        Self { provider }
    }

    /// Returns the current time of the bound clock.
    ///
    /// # Errors
    ///
    /// - `CapabilityFailure` if the clock returns an error.
    pub fn now(&self) -> Result<MonotonicInstant, SciOxideError> {
        self.provider.now().map_err(|error| {
            SciOxideError::capability_failure(Capability::Clock, error.into_detail())
        })
    }
}

/// An absolute deadline on one [`ClockBinding`].
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Deadline<'a> {
    binding: &'a ClockBinding<'a>,
    at: MonotonicInstant,
}
impl<'a> Deadline<'a> {
    /// Creates a deadline at the instant `at` of `binding`.
    pub const fn at(binding: &'a ClockBinding<'a>, at: MonotonicInstant) -> Self {
        Self { binding, at }
    }
    /// Creates a deadline `duration` after the current time of `binding`.
    ///
    /// # Errors
    ///
    /// - `CapabilityFailure` if the clock fails or the deadline overflows.
    pub fn after(binding: &'a ClockBinding<'a>, duration: Duration) -> Result<Self, SciOxideError> {
        let at = binding.now()?.checked_add(duration).ok_or_else(|| {
            SciOxideError::capability_failure(Capability::Clock, "deadline overflow")
        })?;
        Ok(Self { binding, at })
    }
}

// Absence of Timing is the only clock-unavailable state. A deadline cannot
// exist separately from its immutable binding.
#[derive(Debug, Clone, Copy)]
struct Timing<'a> {
    binding: &'a ClockBinding<'a>,
    deadline: Option<MonotonicInstant>,
}
impl<'a> Timing<'a> {
    fn tighten(self, other: Self) -> Result<Self, SciOxideError> {
        if !core::ptr::eq(self.binding, other.binding) {
            return Err(SciOxideError::capability_failure(
                Capability::Clock,
                "cannot replace or combine different clock bindings; construct a new context",
            ));
        }
        let deadline = match (self.deadline, other.deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        Ok(Self { deadline, ..self })
    }
    #[inline(always)]
    fn reached(self) -> Result<bool, SciOxideError> {
        match self.deadline {
            Some(at) => self.binding.now().map(|now| now >= at),
            None => Ok(false),
        }
    }
}

/// Source of random bytes for seeding.
pub trait EntropySource: Sync {
    /// Fills `destination` with random bytes.
    fn fill_bytes(&self, destination: &mut [u8]) -> Result<(), CapabilityError>;
}

/// Processor instruction-set features that kernels may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[must_use]
pub enum ProcessorFeature {
    /// x86/x86_64 SSE2 instructions.
    X86Sse2,
    /// x86/x86_64 AVX2 instructions.
    X86Avx2,
    /// x86/x86_64 fused multiply-add instructions.
    X86Fma,
    /// x86/x86_64 AVX-512 foundation instructions.
    X86Avx512F,
    /// x86/x86_64 AVX-512 doubleword and quadword instructions.
    X86Avx512Dq,
    /// x86/x86_64 AVX-512 vector-length instructions.
    X86Avx512Vl,
    /// AArch64 Advanced SIMD instructions.
    Aarch64Neon,
}

/// Policy that can forbid kernels from using processor features.
///
/// Returning `false` excludes a feature even on hardware that has it.
/// Returning `true` only permits it: kernels use a feature only after checking
/// that the running processor supports it, by runtime detection with `std` or
/// from the compile-time target features without `std`. A `no_std` build that
/// needs SIMD must therefore enable the target features at compile time.
///
/// Answers should not change during an operation.
pub trait ProcessorCapabilities: Sync {
    /// Returns whether kernels may use `feature`, if the processor supports it.
    fn supports(&self, feature: ProcessorFeature) -> bool;
}

impl ProcessorFeature {
    /// Whether the running processor supports the feature, independent of any policy.
    pub(crate) fn verified(self) -> bool {
        #[cfg(feature = "std")]
        match self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Sse2 => std::is_x86_feature_detected!("sse2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Fma => std::is_x86_feature_detected!("fma"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Avx512F => std::is_x86_feature_detected!("avx512f"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Avx512Dq => std::is_x86_feature_detected!("avx512dq"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Avx512Vl => std::is_x86_feature_detected!("avx512vl"),
            #[cfg(target_arch = "aarch64")]
            Self::Aarch64Neon => std::arch::is_aarch64_feature_detected!("neon"),
            _ => false,
        }
        #[cfg(not(feature = "std"))]
        match self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Sse2 => cfg!(target_feature = "sse2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Avx2 => cfg!(target_feature = "avx2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Fma => cfg!(target_feature = "fma"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Avx512F => cfg!(target_feature = "avx512f"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Avx512Dq => cfg!(target_feature = "avx512dq"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::X86Avx512Vl => cfg!(target_feature = "avx512vl"),
            #[cfg(target_arch = "aarch64")]
            Self::Aarch64Neon => cfg!(target_feature = "neon"),
            _ => false,
        }
    }
}

/// Processor policy that forbids all features, so kernels use portable scalar code.
#[derive(Debug, Clone, Copy, Default)]
pub struct PortableProcessor;

impl ProcessorCapabilities for PortableProcessor {
    fn supports(&self, _feature: ProcessorFeature) -> bool {
        false
    }
}

static PORTABLE_PROCESSOR: PortableProcessor = PortableProcessor;

/// The executor, clock, cancellation flag, entropy source and processor policy a computation may use.
///
/// The context borrows all of them, and is cheap to copy. Work started from
/// inside a partition callback runs on that callback's thread. See
/// [execution contexts](crate#execution-contexts).
///
/// ```
/// use core::num::NonZeroUsize;
/// use scioxide_core::{ExecutionContext, ExecutionPolicy, PlanLimits};
///
/// let context = ExecutionContext::sequential();
/// let limits = PlanLimits {
///     work_per_item: NonZeroUsize::MIN,
///     auto_threshold: NonZeroUsize::MIN,
///     worker_limit: NonZeroUsize::MIN,
/// };
/// let plan = context.plan(ExecutionPolicy::Sequential, 2, limits);
/// context.execute(plan, &|partition, (), _| {
///     assert_eq!(partition.range(), 0..2);
///     Ok(())
/// })?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
#[must_use]
pub struct ExecutionContext<'a> {
    executor: &'a dyn Executor,
    cancellation: &'a dyn Cancellation,
    timing: Option<Timing<'a>>,
    entropy: Option<&'a dyn EntropySource>,
    processor: &'a dyn ProcessorCapabilities,

    budget: NonZeroUsize,
    parent: PartitionId,
    // The context wrapper owns callback setup for this dispatch. Direct
    // Executor calls still receive setup from invoke_partition.
    callback_boundary: bool,
}

impl core::fmt::Debug for ExecutionContext<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ExecutionContext")
            .field("timing", &self.timing)
            .field("has_entropy", &self.entropy.is_some())
            .field("budget", &self.budget)
            .field("parent", &self.parent)
            .finish_non_exhaustive()
    }
}

impl<'a> ExecutionContext<'a> {
    // Detached solve permits retain the parent's capabilities. An executor scope may narrow
    // parallelism/deadlines, but cannot remove the solve's deadline.
    pub(crate) fn inherit_solve_capabilities(
        mut self,
        parent: ExecutionContext<'a>,
    ) -> Result<Self, SciOxideError> {
        self.cancellation = parent.cancellation;
        self.entropy = parent.entropy;
        self.processor = parent.processor;

        self.budget = self.budget.min(parent.budget);
        self.timing = match (self.timing, parent.timing) {
            (Some(local), Some(inherited)) => Some(inherited.tighten(local)?),
            (local, inherited) => inherited.or(local),
        };
        Ok(self)
    }
    /// Creates a context that runs parallel work on `executor`, with portable kernels.
    ///
    /// Kernels use portable code only; [`Self::native`] creates the same
    /// context with kernels that may use the processor's native features. A
    /// context created inside a partition callback is limited to one worker,
    /// even if it is later moved to another thread.
    pub fn new(executor: &'a dyn Executor) -> Self {
        let context = Self {
            executor,
            cancellation: &NEVER_CANCELLED,
            timing: None,
            entropy: None,
            processor: &PORTABLE_PROCESSOR,

            budget: executor.worker_budget(),
            parent: PartitionId::ROOT,
            callback_boundary: false,
        };
        Self {
            budget: context.worker_budget(),
            ..context
        }
    }

    /// Returns a context that runs on the calling thread, with no clock, cancellation or entropy.
    ///
    /// Kernels use portable code only.
    #[inline]
    pub const fn sequential() -> ExecutionContext<'static> {
        ExecutionContext {
            executor: &SEQUENTIAL_EXECUTOR,
            cancellation: &NEVER_CANCELLED,
            timing: None,
            entropy: None,
            processor: &PORTABLE_PROCESSOR,

            budget: NonZeroUsize::MIN,
            parent: PartitionId::ROOT,
            callback_boundary: false,
        }
    }

    /// Returns a sequential context whose kernels may use the processor's native features.
    ///
    /// It uses `StdProcessor` with `std` and [`PortableProcessor`] without
    /// `std`; otherwise it equals [`Self::sequential`]. Native kernels may
    /// differ from portable kernels within their documented error bounds.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    ///
    /// let context = ExecutionContext::sequential_native();
    /// assert_eq!(context.worker_budget().get(), 1);
    /// assert!(context.clock().is_none());
    /// assert!(context.entropy().is_none());
    /// ```
    #[inline]
    pub const fn sequential_native() -> ExecutionContext<'static> {
        #[cfg(feature = "std")]
        {
            Self::sequential().with_processor(&StdProcessor)
        }
        #[cfg(not(feature = "std"))]
        {
            Self::sequential()
        }
    }

    /// Creates a context that runs parallel work on `executor`, with native kernels.
    ///
    /// It uses `StdProcessor` with `std` and [`PortableProcessor`] without
    /// `std`; otherwise it equals [`Self::new`]. `native(executor)` relates to
    /// `new(executor)` as [`Self::sequential_native`] relates to
    /// [`Self::sequential`], and `native(&SequentialExecutor)` equals
    /// `sequential_native()`. Native kernels may differ from portable kernels
    /// within their documented error bounds.
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, SequentialExecutor};
    ///
    /// let context = ExecutionContext::native(&SequentialExecutor);
    /// assert_eq!(context.worker_budget().get(), 1);
    /// assert!(context.clock().is_none());
    /// ```
    pub fn native(executor: &'a dyn Executor) -> Self {
        #[cfg(feature = "std")]
        {
            Self::new(executor).with_processor(&StdProcessor)
        }
        #[cfg(not(feature = "std"))]
        {
            Self::new(executor)
        }
    }

    /// Returns the context with a cancellation source.
    pub const fn with_cancellation(mut self, cancellation: &'a dyn Cancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    /// Returns the context with a clock; binding the same clock again keeps the deadline.
    ///
    /// # Errors
    ///
    /// - `CapabilityFailure` if the context already has a different binding.
    pub fn with_clock(self, binding: &'a ClockBinding<'a>) -> Result<Self, SciOxideError> {
        self.with_timing(Timing {
            binding,
            deadline: None,
        })
    }

    /// Returns the context with `deadline`, keeping an earlier deadline if there is one.
    ///
    /// # Errors
    ///
    /// - `CapabilityFailure` if `deadline` uses a different clock binding than the context.
    pub fn with_deadline(self, deadline: Deadline<'a>) -> Result<Self, SciOxideError> {
        self.with_timing(Timing {
            binding: deadline.binding,
            deadline: Some(deadline.at),
        })
    }

    fn with_timing(mut self, timing: Timing<'a>) -> Result<Self, SciOxideError> {
        self.timing = Some(match self.timing {
            Some(parent) => parent.tighten(timing)?,
            None => timing,
        });
        Ok(self)
    }

    /// Returns the context with a deadline `duration` from now, keeping an earlier deadline.
    ///
    /// A zero duration is valid and expires immediately.
    ///
    /// # Errors
    ///
    /// - `CapabilityUnavailable` if no clock is bound.
    /// - `CapabilityFailure` if the clock fails or the deadline overflows.
    pub fn with_time_limit(self, duration: Duration) -> Result<Self, SciOxideError> {
        let timing = self
            .timing
            .ok_or_else(|| SciOxideError::capability_unavailable(Capability::Clock))?;
        let deadline = Deadline::after(timing.binding, duration)?;
        self.with_timing(Timing {
            binding: deadline.binding,
            deadline: Some(deadline.at),
        })
    }

    /// Returns the context with an entropy source.
    pub const fn with_entropy(mut self, entropy: &'a dyn EntropySource) -> Self {
        self.entropy = Some(entropy);
        self
    }

    /// Returns the context with a processor-feature policy.
    pub const fn with_processor(mut self, processor: &'a dyn ProcessorCapabilities) -> Self {
        self.processor = processor;
        self
    }

    /// Returns the executor.
    pub const fn executor(self) -> &'a dyn Executor {
        self.executor
    }

    /// Returns the number of workers available at this nesting level.
    pub fn worker_budget(self) -> NonZeroUsize {
        #[cfg(feature = "std")]
        if ExecutionMembership::active() {
            return NonZeroUsize::MIN;
        }
        self.budget
    }

    /// Returns the entropy source, if any.
    pub const fn entropy(self) -> Option<&'a dyn EntropySource> {
        self.entropy
    }

    /// Returns the clock binding, if any.
    pub fn clock(self) -> Option<&'a ClockBinding<'a>> {
        self.timing.map(|timing| timing.binding)
    }

    /// Returns the processor-feature policy.
    pub const fn processor(self) -> &'a dyn ProcessorCapabilities {
        self.processor
    }

    /// Divides `work_items` items into partitions according to `policy` and `limits`.
    pub fn plan(
        self,
        policy: ExecutionPolicy,
        work_items: usize,
        limits: PlanLimits,
    ) -> ExecutionPlan {
        let available = self.worker_budget();
        let minimum_work = if policy == ExecutionPolicy::Auto
            && work_items >= limits.auto_threshold.get()
            && available != NonZeroUsize::MIN
        {
            self.executor.auto_min_work_per_worker()
        } else {
            NonZeroUsize::MIN
        };
        ExecutionPlan::resolve(
            self.parent,
            policy,
            work_items,
            limits,
            available,
            minimum_work,
        )
    }

    /// Runs `operation` once for each partition of `plan` on the executor.
    ///
    /// Cancellation and the deadline are checked before each partition. To give
    /// each partition its own input, use [`Self::execute_partitioned`].
    ///
    /// # Errors
    ///
    /// - `ExecutionStopped` on cancellation or when the deadline has passed.
    /// - `CapabilityFailure` if the clock fails.
    /// - `ExecutionFailure` if the executor runs a partition that is not in `plan`.
    /// - Any error returned by `operation` or the executor.
    pub fn execute(
        self,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let dispatch = Self {
            callback_boundary: true,
            ..self
        };
        self.executor.execute(dispatch, plan, &|partition, (), _| {
            if plan.partition(partition.worker_index()).as_ref() != Some(&partition) {
                return Err(SciOxideError::execution_failure(
                    None,
                    crate::ExecutionFailureReason::UnknownPartition,
                    "executor supplied an unknown partition",
                ));
            }
            #[cfg(feature = "std")]
            let _membership = ExecutionMembership::enter();
            self.check()?;
            let child = self.child(partition.id());
            operation(partition, (), child)
        })
    }

    /// Runs `f` once for each partition of `plan`, passing the input from the matching slot.
    ///
    /// `inputs` holds one [`PartitionSlot`] per partition, for example
    /// disjoint chunks of an output buffer. A partition takes its input even if
    /// execution later fails.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of slots differs from the number of partitions.
    /// - `ExecutionFailure` if a slot was already taken, or the executor runs a
    ///   partition that is not in `plan`, runs one twice or skips one.
    /// - The errors of [`Self::execute`].
    ///
    /// Run `cargo run -p scioxide-core --example workspace_operations` for a complete caller-owned storage example.
    pub fn execute_partitioned<I: Send>(
        self,
        plan: ExecutionPlan,
        inputs: &[PartitionSlot<I>],
        f: &PartitionFn<'_, I>,
    ) -> Result<(), SciOxideError> {
        if inputs.len() != plan.worker_count() {
            return Err(SciOxideError::dimension_mismatch(
                plan.worker_count(),
                inputs.len(),
            ));
        }
        if inputs.iter().any(PartitionSlot::was_claimed) {
            return Err(SciOxideError::execution_failure(
                None,
                crate::ExecutionFailureReason::DuplicatePartition,
                "partition inputs were already consumed",
            ));
        }
        self.execute(plan, &|partition, (), child| {
            let input = inputs[partition.worker_index()].take().ok_or_else(|| {
                SciOxideError::execution_failure(
                    None,
                    crate::ExecutionFailureReason::DuplicatePartition,
                    "executor repeated a partition",
                )
            })?;
            f(partition, input, child)
        })?;
        if inputs.iter().any(|slot| !slot.was_claimed()) {
            return Err(SciOxideError::execution_failure(
                None,
                crate::ExecutionFailureReason::MissingPartition,
                "executor omitted a partition",
            ));
        }
        Ok(())
    }

    /// Returns why work should stop, if cancellation was requested or the deadline passed.
    ///
    /// Cancellation is checked first. Returning the reason as a value lets an
    /// iterative solver return its last accepted state before it stops.
    ///
    /// # Errors
    ///
    /// - `CapabilityFailure` if the clock fails.
    #[inline(always)]
    pub fn stop_reason(self) -> Result<Option<ExecutionStopReason>, SciOxideError> {
        // Cancellation is checked first: it is a cheap atomic load that needs no
        // fallible clock call, and a fixed precedence keeps the reported reason
        // deterministic when both conditions hold.
        if self.cancellation.is_cancelled() {
            return Ok(Some(ExecutionStopReason::Cancelled));
        }
        if let Some(timing) = self.timing
            && timing.reached()?
        {
            return Ok(Some(ExecutionStopReason::DeadlineReached));
        }
        Ok(None)
    }

    /// Returns an error if cancellation was requested or the deadline passed.
    ///
    /// # Errors
    ///
    /// - `ExecutionStopped` on cancellation or when the deadline has passed.
    /// - `CapabilityFailure` if the clock fails.
    pub fn check(self) -> Result<(), SciOxideError> {
        if let Some(reason) = self.stop_reason()? {
            return Err(SciOxideError::execution_stopped(reason));
        }
        Ok(())
    }

    // Work inside a partition gets a single-worker budget, so a nested plan
    // runs sequentially on the current worker instead of oversubscribing the
    // executor or waiting on a pool that its own caller occupies.
    const fn child(self, parent: PartitionId) -> Self {
        Self {
            budget: NonZeroUsize::MIN,
            parent,
            callback_boundary: false,
            ..self
        }
    }
}

/// Monotonic clock based on [`std::time::Instant`].
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct StdClock {
    epoch: std::time::Instant,
}

#[cfg(feature = "std")]
impl StdClock {
    /// Creates a clock whose time origin is the current instant.
    pub fn new() -> Self {
        Self {
            epoch: std::time::Instant::now(),
        }
    }
}

#[cfg(feature = "std")]
impl Default for StdClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
impl Clock for StdClock {
    fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
        Ok(MonotonicInstant(self.epoch.elapsed()))
    }
}

/// Processor policy that permits every feature the running processor supports.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, Default)]
pub struct StdProcessor;

#[cfg(feature = "std")]
impl ProcessorCapabilities for StdProcessor {
    fn supports(&self, feature: ProcessorFeature) -> bool {
        feature.verified()
    }
}

// One protected destruction attempt bounds recovery work. A secondary payload
// may itself panic on drop, so retain it deliberately instead of recurring.
// This allocates no executor-owned recovery storage. Abort/nonreturning user
// destructors cannot be recovered by an unwind boundary.
#[cfg(feature = "std")]
fn discard_panic_payload(payload: std::boxed::Box<dyn core::any::Any + Send>) {
    if let Err(secondary) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(payload)))
    {
        core::mem::forget(secondary);
    }
}

#[cfg(test)]
mod tests {

    use alloc::{vec, vec::Vec};
    use core::sync::atomic::{AtomicU64, AtomicUsize};
    use std::sync::Mutex;

    use super::*;
    use crate::RangeAccumulator;

    fn non_zero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    #[test]
    fn native_sequential_changes_only_processor_policy() {
        const NATIVE: ExecutionContext<'static> = ExecutionContext::sequential_native();
        let portable = ExecutionContext::sequential();
        assert!(core::ptr::addr_eq(NATIVE.executor, portable.executor));
        assert!(core::ptr::addr_eq(
            NATIVE.cancellation,
            portable.cancellation
        ));
        assert_eq!(NATIVE.worker_budget(), NonZeroUsize::MIN);
        assert_eq!(NATIVE.parent, portable.parent);
        assert_eq!(NATIVE.callback_boundary, portable.callback_boundary);
        assert!(NATIVE.clock().is_none());
        assert!(NATIVE.entropy().is_none());
        assert!(NATIVE.check().is_ok());
        assert_eq!(
            NATIVE
                .with_time_limit(Duration::from_secs(1))
                .unwrap_err()
                .kind(),
            crate::SciOxideErrorKind::CapabilityUnavailable,
        );
        for feature in [
            ProcessorFeature::X86Sse2,
            ProcessorFeature::X86Avx2,
            ProcessorFeature::X86Fma,
            ProcessorFeature::X86Avx512F,
            ProcessorFeature::X86Avx512Dq,
            ProcessorFeature::X86Avx512Vl,
            ProcessorFeature::Aarch64Neon,
        ] {
            #[cfg(feature = "std")]
            let expected = StdProcessor.supports(feature);
            #[cfg(not(feature = "std"))]
            let expected = PortableProcessor.supports(feature);
            assert_eq!(NATIVE.processor().supports(feature), expected);
            assert!(!portable.processor().supports(feature));
        }
    }

    #[test]
    fn native_changes_only_processor_policy() {
        let native = ExecutionContext::native(&SEQUENTIAL_EXECUTOR);
        let portable = ExecutionContext::new(&SEQUENTIAL_EXECUTOR);
        let sequential_native = ExecutionContext::sequential_native();
        for context in [portable, sequential_native] {
            assert!(core::ptr::addr_eq(native.executor, context.executor));
            assert!(core::ptr::addr_eq(
                native.cancellation,
                context.cancellation
            ));
            assert_eq!(native.worker_budget(), context.worker_budget());
            assert_eq!(native.parent, context.parent);
            assert_eq!(native.callback_boundary, context.callback_boundary);
        }
        assert!(native.clock().is_none());
        assert!(native.entropy().is_none());
        assert!(native.check().is_ok());
        assert_eq!(
            native
                .with_time_limit(Duration::from_secs(1))
                .unwrap_err()
                .kind(),
            crate::SciOxideErrorKind::CapabilityUnavailable,
        );
        for feature in [
            ProcessorFeature::X86Sse2,
            ProcessorFeature::X86Avx2,
            ProcessorFeature::X86Fma,
            ProcessorFeature::X86Avx512F,
            ProcessorFeature::X86Avx512Dq,
            ProcessorFeature::X86Avx512Vl,
            ProcessorFeature::Aarch64Neon,
        ] {
            assert_eq!(
                native.processor().supports(feature),
                sequential_native.processor().supports(feature)
            );
            assert!(!portable.processor().supports(feature));
        }
    }

    #[test]
    fn partitions_are_stable_balanced_and_cover_the_domain() {
        let context = ExecutionContext::sequential();
        let plan = context.plan(
            ExecutionPolicy::Parallel(non_zero(4)),
            10,
            crate::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: non_zero(1),
                worker_limit: non_zero(8),
            },
        );
        let partitions = plan.partitions().collect::<Vec<_>>();
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].range(), 0..10);

        let executor = MockExecutor::new(4);
        let parallel = ExecutionContext::new(&executor).plan(
            ExecutionPolicy::Parallel(non_zero(4)),
            10,
            crate::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: non_zero(1),
                worker_limit: non_zero(8),
            },
        );
        assert_eq!(
            parallel
                .partitions()
                .map(|partition| partition.range())
                .collect::<Vec<_>>(),
            vec![0..3, 3..6, 6..8, 8..10]
        );
        assert_eq!(
            parallel
                .partitions()
                .map(|partition| partition.id())
                .collect::<Vec<_>>(),
            parallel
                .partitions()
                .map(|partition| partition.id())
                .collect::<Vec<_>>()
        );
    }

    #[derive(Debug)]
    struct MockExecutor {
        budget: NonZeroUsize,
        calls: AtomicUsize,
    }

    impl MockExecutor {
        fn new(budget: usize) -> Self {
            Self {
                budget: non_zero(budget),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Executor for MockExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            self.budget
        }

        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            SequentialExecutor.execute(context, plan, operation)
        }
    }

    #[test]
    fn injected_executor_is_reused_and_nested_budget_is_one() {
        let executor = MockExecutor::new(4);
        let context = ExecutionContext::new(&executor);
        let nested_budgets = Mutex::new(Vec::new());
        for _ in 0..3 {
            let plan = context.plan(
                ExecutionPolicy::Parallel(non_zero(4)),
                4,
                crate::PlanLimits {
                    work_per_item: core::num::NonZeroUsize::MIN,
                    auto_threshold: non_zero(1),
                    worker_limit: non_zero(4),
                },
            );
            context
                .execute(plan, &|_, (), scope| {
                    nested_budgets.lock().unwrap().push(scope.worker_budget());
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(executor.calls.load(Ordering::Relaxed), 3);
        assert!(
            nested_budgets
                .into_inner()
                .unwrap()
                .into_iter()
                .all(|budget| budget == NonZeroUsize::MIN)
        );
    }

    #[derive(Debug)]
    struct MockClock {
        ticks: AtomicU64,
    }

    impl Clock for MockClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_nanos(self.ticks.fetch_add(1, Ordering::Relaxed)),
            ))
        }
    }

    #[test]
    fn cancellation_and_deadline_are_typed() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = ExecutionContext::sequential()
            .with_cancellation(&cancellation)
            .check()
            .unwrap_err();
        assert!(matches!(
            (error).view(),
            crate::SciOxideErrorView::ExecutionStopped {
                reason: ExecutionStopReason::Cancelled,
                ..
            }
        ));

        let clock = MockClock {
            ticks: AtomicU64::new(0),
        };
        let clock_binding = crate::ClockBinding::new(&clock);
        let deadline = Deadline::at(
            &clock_binding,
            MonotonicInstant::from_duration_since_epoch(Duration::from_nanos(1)),
        );
        let context = ExecutionContext::sequential()
            .with_deadline(deadline)
            .unwrap();
        context.check().unwrap();
        assert!(matches!(
            (context.check()).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::ExecutionStopped {
                reason: ExecutionStopReason::DeadlineReached,
                ..
            }
        ));
    }

    #[test]
    fn deterministic_merge_order_is_independent_of_execution_order() {
        let executor = MockExecutor::new(4);
        let context = ExecutionContext::new(&executor);
        let plan = context.plan(
            ExecutionPolicy::Parallel(non_zero(4)),
            32,
            crate::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: non_zero(1),
                worker_limit: non_zero(4),
            },
        );
        let results = Mutex::new(Vec::new());
        let exact_results = Mutex::new(Vec::new());
        context
            .execute(plan, &|partition, (), _| {
                let mut range = partition.range();
                results.lock().unwrap().push((
                    partition.clone(),
                    range.clone().map(|value| value + 1).sum::<usize>(),
                ));
                let mut accumulator = RangeAccumulator::<f64>::new();
                range.try_for_each(|index| {
                    let (left, right) = match index {
                        0 => (f64::MAX, f64::MIN_POSITIVE),
                        1 => (f64::MIN_POSITIVE, f64::MAX),
                        _ => (0.0, 0.0),
                    };
                    accumulator.add_product(left, right)
                })?;
                exact_results.lock().unwrap().push((partition, accumulator));
                Ok(())
            })
            .unwrap();
        let mut results = results.into_inner().unwrap();
        results.reverse();
        let sum = plan
            .reduce(results, 0, |sum, value| Ok(sum + value))
            .unwrap();
        assert_eq!(sum, (1..=32).sum::<usize>());

        let mut exact_results = exact_results.into_inner().unwrap();
        exact_results.reverse();
        let exact = plan
            .reduce(
                exact_results,
                RangeAccumulator::<f64>::new(),
                |mut accumulator, partition| {
                    accumulator.merge(&partition)?;
                    Ok(accumulator)
                },
            )
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(exact.to_bits(), 0x401f_ffff_ffff_ffff);

        let first = plan.partitions().next().unwrap();
        assert!(matches!(
            (plan.reduce([(first.clone(), 1), (first, 2)], 0, |sum, value| Ok(
                sum + value
            )))
            .as_ref()
            .unwrap_err()
            .view(),
            crate::SciOxideErrorView::ExecutionFailure {
                partition: Some(0),
                ..
            }
        ));
    }

    #[test]
    fn scoped_partition_slot_is_single_consumer() {
        let mut value = 0;
        let slot = PartitionSlot::new(&mut value);
        let claims = AtomicUsize::new(0);
        std::thread::scope(|threads| {
            for _ in 0..4 {
                threads.spawn(|| {
                    if let Some(value) = slot.take() {
                        claims.fetch_add(1, Ordering::Relaxed);
                        *value = 17;
                    }
                });
            }
        });
        assert_eq!(claims.load(Ordering::Relaxed), 1);
        assert!(slot.take().is_none());
        drop(slot);
        assert_eq!(value, 17);
    }
}
