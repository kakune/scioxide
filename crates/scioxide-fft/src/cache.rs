//! A bounded least-recently-used cache of prepared FFT plans.
use crate::plan::PlanCore;
use crate::{FftLayout, FftPlan, FftPlanSpec};
use alloc::{sync::Arc, vec::Vec};
use scioxide_core::backend::{KernelBackend, try_with_capacity};
use scioxide_core::{ExecutionPolicy, SciOxideError};
/// Sets the size limits of an [`FftPlanCache`]: a number of entries and a
/// number of bytes.
///
/// The default allows 16 entries and 64 MiB.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_fft::FftCachePolicy;
///
/// let policy = FftCachePolicy::new(4, 1 << 20)?;
/// assert_eq!(
///     (policy.max_entries(), policy.max_plan_bytes()),
///     (4, 1 << 20)
/// );
/// let error = FftCachePolicy::new(0, 1 << 20).unwrap_err();
/// assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
/// let default = FftCachePolicy::default();
/// assert_eq!(
///     (default.max_entries(), default.max_plan_bytes()),
///     (16, 64 << 20)
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct FftCachePolicy {
    max_entries: usize,
    max_plan_bytes: usize,
}

impl FftCachePolicy {
    /// Creates a policy that keeps at most `max_entries` prepared plans and at
    /// most `max_plan_bytes` bytes of plan storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if either limit is zero.
    pub fn new(max_entries: usize, max_plan_bytes: usize) -> Result<Self, SciOxideError> {
        if max_entries == 0 || max_plan_bytes == 0 {
            return Err(SciOxideError::invalid_input(
                if max_entries == 0 {
                    scioxide_core::argument!("max_entries")
                } else {
                    scioxide_core::argument!("max_plan_bytes")
                },
                scioxide_core::InvalidInputReason::OutOfRange,
                "FFT cache bounds must both be non-zero",
            ));
        }
        Ok(Self {
            max_entries,
            max_plan_bytes,
        })
    }

    /// Returns the maximum number of entries.
    pub const fn max_entries(self) -> usize {
        self.max_entries
    }

    /// Returns the maximum number of bytes of plan storage.
    pub const fn max_plan_bytes(self) -> usize {
        self.max_plan_bytes
    }
}

impl Default for FftCachePolicy {
    fn default() -> Self {
        Self {
            max_entries: 16,
            max_plan_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Reports the activity and current contents of an [`FftPlanCache`].
///
/// The counters cover the life of the cache; [`FftPlanCache::clear`] does not
/// reset them. See the [`FftPlanCache`] example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct FftCacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    entries: usize,
    retained_plan_bytes: usize,
}

impl FftCacheStats {
    /// Returns the number of requests served from the cache.
    pub const fn hits(self) -> u64 {
        self.hits
    }

    /// Returns the number of requests that prepared a new plan.
    pub const fn misses(self) -> u64 {
        self.misses
    }

    /// Returns the number of entries removed to stay within the policy.
    pub const fn evictions(self) -> u64 {
        self.evictions
    }

    /// Returns the current number of entries.
    pub const fn entries(self) -> usize {
        self.entries
    }

    /// Returns the bytes of plan storage held by the current entries.
    pub const fn retained_plan_bytes(self) -> usize {
        self.retained_plan_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheKey {
    length: usize,
    /// Real and Hermitian layouts share one packed real core.
    packed: bool,
    execution_policy: ExecutionPolicy,
    backend: KernelBackend,
}

#[derive(Debug)]
struct CacheEntry<S: scioxide_core::RealScalar> {
    key: CacheKey,
    core: Arc<PlanCore<S>>,
}

/// Prepares [`FftPlan`]s and keeps their tables for reuse.
///
/// An entry is keyed by length, by complex or packed layout, by execution
/// policy and by the resolved CPU backend. Plans that differ only in direction
/// or normalization share an entry, and so do [`FftLayout::Real`] and
/// [`FftLayout::Hermitian`]. The returned plans share storage with the cache
/// and stay valid after their entry is removed.
///
/// When a new entry would exceed the [`FftCachePolicy`] limits, the least
/// recently used entries are removed first. A plan larger than the byte limit
/// is returned but not stored. Byte counts include the shared tables once and
/// exclude the cache's own entry list, which is allocated up front. Planning
/// contexts with a provider bypass the cache.
///
/// SciOxide has no global plan cache; create one where you need it.
///
/// # Examples
///
/// The inverse plan reuses the tables prepared for the forward plan: one miss,
/// one hit and one entry.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_fft::{
///     FftCachePolicy, FftDirection, FftLayout, FftPlanCache, FftPlanSpec, Normalization,
/// };
///
/// let mut cache = FftPlanCache::<f64>::new(FftCachePolicy::new(4, 1 << 20)?)?;
/// let spec = |direction| FftPlanSpec::new(64, direction, Normalization::Inverse, FftLayout::Complex);
/// let cx = ExecutionContext::sequential();
/// let forward = cache.plan(spec(FftDirection::Forward), cx)?;
/// let inverse = cache.plan(spec(FftDirection::Inverse), cx)?;
/// assert_eq!(inverse.spec().direction(), FftDirection::Inverse);
///
/// let stats = cache.stats();
/// assert_eq!((stats.misses(), stats.hits(), stats.entries()), (1, 1, 1));
/// assert!(stats.retained_plan_bytes() <= forward.plan_bytes());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
#[must_use]
pub struct FftPlanCache<S: scioxide_core::RealScalar> {
    policy: FftCachePolicy,
    entries: Vec<CacheEntry<S>>,
    retained_plan_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl<S: scioxide_core::RealScalar> FftPlanCache<S> {
    /// Creates an empty cache with the given limits.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the entry list cannot be allocated.
    pub fn new(policy: FftCachePolicy) -> Result<Self, SciOxideError> {
        Ok(Self {
            policy,
            entries: try_with_capacity(
                policy.max_entries,
                scioxide_core::quantity!("fft_plan_cache_entries"),
            )?,
            retained_plan_bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        })
    }

    /// Returns the size limits.
    pub const fn policy(&self) -> FftCachePolicy {
        self.policy
    }

    /// Returns the activity counters and current contents.
    pub const fn stats(&self) -> FftCacheStats {
        FftCacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            entries: self.entries.len(),
            retained_plan_bytes: self.retained_plan_bytes,
        }
    }

    /// Removes all entries. The counters and the allocated entry list are kept.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.retained_plan_bytes = 0;
    }

    // `entries` is ordered from least to most recently used. Linear scans and
    // `remove`/`push` are adequate because the entry bound is small, and they
    // keep eviction order fully deterministic. Plans are `Arc`-backed, so a
    // hit clones only a handle.
    fn find(&mut self, key: CacheKey) -> Option<Arc<PlanCore<S>>> {
        self.entries
            .iter()
            .position(|entry| entry.key == key)
            .map(|index| {
                self.hits = self.hits.saturating_add(1);
                let entry = self.entries.remove(index);
                let core = Arc::clone(&entry.core);
                self.entries.push(entry);
                core
            })
    }

    fn insert(&mut self, key: CacheKey, core: Arc<PlanCore<S>>) {
        self.misses = self.misses.saturating_add(1);
        let bytes = core.bytes();
        // A plan larger than the whole byte budget is returned to the caller
        // but never retained, so it cannot flush every other entry.
        if bytes > self.policy.max_plan_bytes {
            return;
        }
        // Evict oldest entries until both bounds admit the new plan. The byte
        // test is written as `retained > max - bytes` so it cannot overflow
        // (`bytes <= max` was checked above). Capacity was reserved for
        // `max_entries` up front, so the push below never reallocates.
        while self.entries.len() >= self.policy.max_entries
            || self.retained_plan_bytes > self.policy.max_plan_bytes - bytes
        {
            let entry = self.entries.remove(0);
            self.retained_plan_bytes = self.retained_plan_bytes.saturating_sub(entry.core.bytes());
            self.evictions = self.evictions.saturating_add(1);
        }
        self.retained_plan_bytes += bytes;
        self.entries.push(CacheEntry { key, core });
    }
}

impl<S: scioxide_core::RealScalar> FftPlanCache<S> {
    /// Returns a plan for `spec`, reusing a cached entry when one matches.
    ///
    /// The result is the same as [`FftPlan::build`] with the same arguments.
    /// `context` selects the CPU backend; with a provider, the plan is built
    /// without the cache. Neither the plan nor the cache keeps the context.
    ///
    /// # Errors
    ///
    /// The same as [`FftPlan::build`]. On error the cache entries are unchanged.
    pub fn plan<'context>(
        &mut self,
        spec: FftPlanSpec,
        context: impl Into<crate::FftPlanningContext<'context, S>>,
    ) -> Result<FftPlan<S>, SciOxideError> {
        let context = context.into();
        context.execution().check()?;
        let backend = context.resolve(spec.backend_strategy())?;
        // Provider plans depend on the borrowed provider and its limits,
        // neither of which is part of the cache key, so they bypass the cache
        // entirely. CPU cores are keyed by shape, policy and the *resolved*
        // backend, so an automatic request on different processors never
        // shares an entry.
        if context.has_provider() {
            return FftPlan::prepare(spec, backend, context);
        }
        let scale = FftPlan::<S>::scale(spec)?;
        let key = CacheKey {
            length: spec.length(),
            packed: spec.layout() != FftLayout::Complex,
            execution_policy: spec.execution_policy(),
            backend,
        };
        if let Some(core) = self.find(key) {
            return Ok(FftPlan::from_core(spec, scale, core));
        }
        let plan = FftPlan::prepare(spec, backend, context)?;
        self.insert(key, Arc::clone(&plan.core));
        Ok(plan)
    }
}
