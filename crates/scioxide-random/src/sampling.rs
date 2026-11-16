//! Shuffling, uniform choice and sampling without replacement.
//!
//! [`WeightedSampler`] builds an alias table once for repeated weighted draws in
//! expected O(1) time.

use alloc::vec::Vec;

use scioxide_core::SciOxideError;

use crate::traits::BitGenerator;
use crate::util::MAX_SAMPLING_ATTEMPTS;
use crate::workspace::SamplingWorkspace;

mod weighted;

pub use weighted::WeightedSampler;

/// Returns an integer uniformly distributed on `0..upper`.
///
/// The method is Lemire's nearly divisionless rejection on one uniform word
/// per attempt: [`BitGenerator::next_uniform_u32`] for `upper ≤ 2³²`,
/// [`BitGenerator::next_uniform_u64`] above. Rejected words are consumed, and
/// at most 1,024 words are tried.
///
/// ```
/// use scioxide_random::{SeedableRng, SplitMix64, uniform_index};
/// let mut rng = SplitMix64::from_seed(7);
/// let die = 1 + uniform_index(6, &mut rng)?;
/// assert!((1..=6).contains(&die));
/// assert!(uniform_index(0, &mut rng).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
/// - `InvalidInput` if `upper` is zero or exceeds `u64::MAX`.
/// - `ConvergenceFailure` if 1,024 words are rejected.
/// - Any error of the generator. Words drawn before the error stay consumed.
///
/// # References
///
/// D. Lemire, "Fast random integer generation in an interval", *ACM Trans.
/// Model. Comput. Simul.* 29(1), 2019.
#[allow(clippy::cast_possible_truncation)]
#[inline]
pub fn uniform_index<R: BitGenerator>(upper: usize, rng: &mut R) -> Result<usize, SciOxideError> {
    if upper == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("upper"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "uniform_index requires upper > 0",
        ));
    }

    let upper_u64 = u64::try_from(upper).map_err(|_| {
        SciOxideError::invalid_input(
            scioxide_core::argument!("upper"),
            scioxide_core::InvalidInputReason::OutOfRange,
            alloc::format!("upper bound {upper} does not fit into u64"),
        )
    })?;

    Ok(uniform_below(upper_u64, rng)? as usize)
}

/// Bounds served by 32-bit uniform words.
const NARROW_BOUND: u64 = 1 << 32;

/// Shared Lemire rejection; callers validate a nonzero bound.
#[inline]
pub(crate) fn uniform_below<R: BitGenerator + ?Sized>(
    upper: u64,
    rng: &mut R,
) -> Result<u64, SciOxideError> {
    // Lemire's (2019) nearly-divisionless method: the high half of x * upper
    // is uniform on [0, upper) except for a bias that lives only in products
    // whose low half falls below 2^w mod upper. That threshold is computed
    // only when `low < upper`, so the common path needs no division; the
    // rejection probability is below upper / 2^w. Bounds up to 2^32 use 32-bit
    // words, which halves the native cost of 32-bit engines.
    if upper <= NARROW_BOUND {
        for _ in 0..MAX_SAMPLING_ATTEMPTS {
            let product = u64::from(rng.next_uniform_u32()?) * upper;
            let low = product & (NARROW_BOUND - 1);
            if low < upper && low < (NARROW_BOUND - upper) % upper {
                continue;
            }
            return Ok(product >> u32::BITS);
        }
    } else {
        for _ in 0..MAX_SAMPLING_ATTEMPTS {
            let product = u128::from(rng.next_uniform_u64()?) * u128::from(upper);
            #[allow(clippy::cast_possible_truncation)]
            let low = product as u64;
            if low < upper && low < upper.wrapping_neg() % upper {
                continue;
            }
            #[allow(clippy::cast_possible_truncation)]
            return Ok((product >> u64::BITS) as u64);
        }
    }
    Err(lemire_exhausted())
}

#[cold]
#[inline(never)]
fn lemire_exhausted() -> SciOxideError {
    crate::law::iteration_limit(MAX_SAMPLING_ATTEMPTS)
}

/// Shuffles `slice` in place with the Fisher–Yates algorithm.
///
/// Every permutation is equally likely if `rng` is uniform. The shuffle walks
/// from the last position down and swaps position `i` with one drawn by
/// [`uniform_index`]`(i + 1)`: `n − 1` draws for a slice of length `n`, no
/// allocation and no clones. On error the generator stays advanced and the
/// slice holds a permutation of its input; for all-or-nothing output or a
/// restored generator, use [`shuffle_in_place`].
///
/// ```
/// use scioxide_random::{SeedableRng, SplitMix64, shuffle};
/// let mut deck = [1, 2, 3, 4, 5];
/// shuffle(&mut deck, &mut SplitMix64::from_seed(7))?;
/// let mut sorted = deck;
/// sorted.sort_unstable();
/// assert_eq!(sorted, [1, 2, 3, 4, 5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - The errors of [`uniform_index`].
///
/// # References
///
/// D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed., §3.4.2,
/// Algorithm P.
pub fn shuffle<T, R: BitGenerator>(slice: &mut [T], rng: &mut R) -> Result<(), SciOxideError> {
    for index in (1..slice.len()).rev() {
        let swap_index = uniform_index(index + 1, rng)?;
        slice.swap(index, swap_index);
    }
    Ok(())
}

/// Shuffles `slice` in place like [`shuffle`], under a [`SamplingPolicy`](crate::SamplingPolicy).
///
/// The draws and, on success, the permutation equal those of [`shuffle`].
/// `policy` decides whether the slice is permuted all-or-nothing or swap by
/// swap, and where the generator stands on error
/// ([batch sampling](crate#batch-sampling)). All-or-nothing output needs a
/// workspace of `slice.len()` elements (none below length two); incremental
/// output needs none. For all-or-nothing output the elements are cloned into
/// the workspace before the first draw and swapped back after the last,
/// without further clones.
///
/// ```
/// use scioxide_random::{OutputPolicy, SamplingPolicy, SamplingRequest, SeedableRng,
///     SplitMix64, shuffle_in_place};
/// let mut deck = [1, 2, 3, 4, 5];
/// let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
/// let mut workspace = policy.workspace(SamplingRequest::scalars(deck.len()))?;
/// shuffle_in_place(&mut deck, &mut SplitMix64::from_seed(7), policy, &mut workspace)?;
/// let mut sorted = deck;
/// sorted.sort_unstable();
/// assert_eq!(sorted, [1, 2, 3, 4, 5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `AllocationFailure` if the workspace cannot grow.
/// - The errors of [`uniform_index`].
pub fn shuffle_in_place<T: Clone, R: BitGenerator>(
    slice: &mut [T],
    rng: &mut R,
    policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
    workspace: &mut SamplingWorkspace<T>,
) -> Result<(), SciOxideError> {
    if slice.len() < 2 {
        return Ok(());
    }
    let atomic = policy.output_policy() == crate::OutputPolicy::Atomic;
    // Atomic output shuffles a staged copy and only swaps it into `slice` after
    // every draw succeeded, so a failing provider leaves the input untouched.
    if atomic {
        let requirements =
            policy.workspace_requirements(crate::SamplingRequest::scalars(slice.len()))?;
        workspace.stage_copy(&requirements, slice)?;
    }
    policy.run(rng, |rng| {
        if atomic {
            shuffle(workspace.as_mut_slice(), rng)?;
            slice.swap_with_slice(workspace.as_mut_slice());
            Ok(())
        } else {
            shuffle(slice, rng)
        }
    })
}

/// Returns a uniformly chosen element of `slice`, or `None` if it is empty.
///
/// It takes one draw of [`uniform_index`]; an empty slice draws nothing.
///
/// ```
/// use scioxide_random::{SeedableRng, SplitMix64, choose};
/// let mut rng = SplitMix64::from_seed(7);
/// let colors = ["red", "green", "blue"];
/// assert!(choose(&colors, &mut rng)?.is_some_and(|color| colors.contains(color)));
/// assert_eq!(choose::<u8, _>(&[], &mut rng)?, None);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - The errors of [`uniform_index`].
pub fn choose<'a, T, R: BitGenerator>(
    slice: &'a [T],
    rng: &mut R,
) -> Result<Option<&'a T>, SciOxideError> {
    if slice.is_empty() {
        return Ok(None);
    }
    let idx = uniform_index(slice.len(), rng)?;
    Ok(Some(&slice[idx]))
}

/// Returns `k` distinct indices drawn uniformly from `0..n` without replacement.
///
/// The algorithm is chosen by the sampled fraction. When `k >= n / 16`, a
/// partial Fisher–Yates shuffle of `0..n` draws exactly `k` indices with
/// `O(n)` work and storage, and the result is a uniformly random ordered
/// sample; in particular `k = n` returns a uniformly random permutation of
/// `0..n`. Smaller fractions use Floyd's algorithm with an open-addressed set
/// of at most `4k` slots: exactly `k` index draws, `O(k)` expected work and
/// storage. The order of the returned indices is then unspecified. Every index
/// draw uses [`uniform_index`]. Storage is allocated before the first draw. On
/// error nothing is returned and the generator stays advanced by the words it
/// consumed.
///
/// ```
/// use scioxide_random::{SeedableRng, SplitMix64, sample_indices};
/// let mut rng = SplitMix64::from_seed(7);
/// let mut sparse = sample_indices(1_000, 5, &mut rng)?; // Floyd
/// sparse.sort_unstable();
/// sparse.dedup();
/// assert!(sparse.len() == 5 && sparse.iter().all(|&i| i < 1_000));
/// let mut permutation = sample_indices(10, 10, &mut rng)?; // Fisher-Yates
/// permutation.sort_unstable();
/// assert!(permutation.into_iter().eq(0..10));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - [`scioxide_core::SciOxideErrorKind::InvalidInput`] if `k > n`.
/// - [`scioxide_core::SciOxideErrorKind::AllocationFailure`] if the working storage
///   cannot be allocated; nothing is drawn.
/// - The errors of [`uniform_index`].
///
/// # References
///
/// J. Bentley and B. Floyd, "Programming pearls: a sample of brilliance",
/// *Commun. ACM* 30(9), 1987, 754–757.
pub fn sample_indices<R: BitGenerator>(
    n: usize,
    k: usize,
    rng: &mut R,
) -> Result<Vec<usize>, SciOxideError> {
    if k > n {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("k"),
            scioxide_core::InvalidInputReason::OutOfRange,
            alloc::format!("sample_indices: k={k} exceeds n={n}"),
        ));
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    if k >= n / DENSE_FRACTION {
        let mut pool: Vec<usize> = scioxide_core::backend::try_with_capacity(
            n,
            scioxide_core::quantity!("sampled_indices"),
        )?;
        pool.extend(0..n);
        partial_shuffle(&mut pool, k, rng)?;
        pool.truncate(k);
        Ok(pool)
    } else {
        let mut selected = scioxide_core::backend::try_with_capacity(
            k,
            scioxide_core::quantity!("sampled_indices"),
        )?;
        let mut seen = IndexSet::new(k)?;
        floyd(n, k, &mut selected, &mut seen, rng)?;
        Ok(selected)
    }
}

/// `sample_indices` shuffles a full index buffer once `k >= n / DENSE_FRACTION`.
const DENSE_FRACTION: usize = 16;

// The first `k` positions of a Fisher-Yates pass over 0..n are a uniformly
// random ordered k-sample; the untouched tail is discarded.
fn partial_shuffle<R: BitGenerator>(
    pool: &mut [usize],
    k: usize,
    rng: &mut R,
) -> Result<(), SciOxideError> {
    let n = pool.len();
    for i in 0..k {
        let j = i + uniform_index(n - i, rng)?;
        pool.swap(i, j);
    }
    Ok(())
}

// Floyd's algorithm for sampling k distinct indices from 0..n. A compact
// open-addressed set avoids an allocator- and std-specific hash table.
// Invariant: after step j the set is a uniformly random subset of 0..=j of
// size j - (n - k) + 1. A collision with t is resolved by taking j, which is
// new at this step, so exactly k draws suffice.
fn floyd<R: BitGenerator>(
    n: usize,
    k: usize,
    selected: &mut Vec<usize>,
    seen: &mut IndexSet,
    rng: &mut R,
) -> Result<(), SciOxideError> {
    for j in (n - k)..n {
        let t = uniform_index(j + 1, rng)?;
        let chosen = if seen.insert(t) {
            t
        } else {
            seen.insert(j);
            j
        };
        selected.push(chosen);
    }
    Ok(())
}

/// Open-addressed set of indices below `usize::MAX`, which marks empty slots.
struct IndexSet {
    slots: Vec<usize>,
    shift: u32,
}

impl IndexSet {
    const EMPTY: usize = usize::MAX;

    fn new(expected: usize) -> Result<Self, SciOxideError> {
        // Power-of-two capacity at least twice the element count keeps the
        // linear-probing load factor <= 1/2 and lets a shift replace modulo.
        let doubled = scioxide_core::backend::checked_mul(
            expected,
            2,
            scioxide_core::quantity!("sampled_index_set"),
        )?;
        let capacity = scioxide_core::backend::checked_next_power_of_two(
            doubled.max(2),
            scioxide_core::quantity!("sampled_index_set"),
        )?;
        Ok(Self {
            slots: scioxide_core::backend::try_filled(
                capacity,
                Self::EMPTY,
                scioxide_core::quantity!("sampled_index_set"),
            )?,
            shift: u64::BITS - capacity.trailing_zeros(),
        })
    }

    /// Insert `value`, reporting whether it was absent.
    fn insert(&mut self, value: usize) -> bool {
        let mask = self.slots.len() - 1;
        let mut slot = self.home(value);
        loop {
            match self.slots[slot] {
                Self::EMPTY => {
                    self.slots[slot] = value;
                    return true;
                }
                stored if stored == value => return false,
                _ => slot = (slot + 1) & mask,
            }
        }
    }

    // Fibonacci hashing: the high bits of a golden-ratio product spread
    // consecutive indices across the table without long probe runs.
    #[allow(clippy::cast_possible_truncation)]
    fn home(&self, value: usize) -> usize {
        ((value as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> self.shift) as usize
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::SeedableRng;
    use scioxide_core::ConvergenceReason;

    struct ZeroWords {
        draws: usize,
    }

    impl BitGenerator for ZeroWords {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.draws += 1;
            Ok(0)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.draws += 1;
            Ok(0)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            self.draws += dest.len().div_ceil(8);
            dest.fill(0);
            Ok(())
        }
    }

    #[test]
    fn test_shuffle_preserves_elements() {
        let mut data = vec![1, 2, 3, 4, 5];
        let mut rng = SplitMix64::from_seed(42);
        let mut workspace = SamplingWorkspace::new();
        shuffle_in_place(
            &mut data,
            &mut rng,
            crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
            &mut workspace,
        )
        .unwrap();
        data.sort_unstable();
        assert_eq!(data, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn small_index_frequency_and_rejection_vectors() {
        // Fixed-seed frequency evidence complements the integer rejection proof.
        for upper in 2..=9 {
            let mut rng = SplitMix64::from_seed(610);
            let mut counts = vec![0usize; upper];
            for _ in 0..90_000 {
                counts[uniform_index(upper, &mut rng).unwrap()] += 1;
            }
            let expected = 90_000. / upper as f64;
            assert!(
                counts
                    .iter()
                    .all(|&count| (count as f64 - expected).abs() < 6. * expected.sqrt())
            );
        }
        let mut zero = ZeroWords { draws: 0 };
        assert_eq!(uniform_index(1, &mut zero).unwrap(), 0);
        assert_eq!(zero.draws, 1);
        assert!(uniform_index(0, &mut zero).is_err());
        assert_eq!(zero.draws, 1);
    }

    #[test]
    fn test_shuffle_empty() {
        let mut data: Vec<i32> = vec![];
        let mut rng = SplitMix64::from_seed(42);
        let mut workspace = SamplingWorkspace::new();
        assert!(
            shuffle_in_place(
                &mut data,
                &mut rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut workspace,
            )
            .is_ok()
        );
        assert!(shuffle(&mut data, &mut rng).is_ok());
    }

    #[test]
    #[allow(clippy::panic)]
    fn lemire_rejection_is_bounded_and_shuffle_is_transactional() {
        let mut rng = ZeroWords { draws: 0 };
        let error = uniform_index(3, &mut rng).unwrap_err();
        let scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::IterationLimit,
            diagnostics,
            ..
        } = error.view()
        else {
            panic!("unexpected error: {error:?}")
        };
        assert_eq!(
            diagnostics.counts().own().iterations(),
            MAX_SAMPLING_ATTEMPTS
        );
        assert_eq!(rng.draws, MAX_SAMPLING_ATTEMPTS);

        let mut values = [1, 2, 3, 4];
        let original = values;
        let mut rng = ZeroWords { draws: 0 };
        let mut workspace = SamplingWorkspace::new();
        let error = shuffle_in_place(
            &mut values,
            &mut rng,
            crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
            &mut workspace,
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!(values, original);
    }

    #[test]
    fn f32_strata_limit_precedes_allocation_and_rng_use() {
        let mut rng = SplitMix64::from_seed(42);
        let position = rng.position();
        let error = crate::latin_hypercube::<f32, _>((1 << 24) + 1, 2, &mut rng).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        assert_eq!(rng.position(), position);
    }

    #[test]
    fn checked_sampling_allocations_fail_before_rng_use() {
        let mut rng = SplitMix64::from_seed(42);
        let position = rng.position();
        let error = sample_indices(usize::MAX, usize::MAX, &mut rng).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::AllocationFailure
        );
        assert_eq!(rng.position(), position);
        let error = crate::SamplingPolicy::rollback(crate::OutputPolicy::Atomic)
            .workspace::<u8>(crate::SamplingRequest::scalars(usize::MAX))
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::AllocationFailure
        );
    }

    #[test]
    fn test_choose_some() {
        let data = [10, 20, 30];
        let mut rng = SplitMix64::from_seed(42);
        let choice = choose(&data, &mut rng).unwrap();
        assert!(choice.is_some());
        assert!(data.contains(choice.unwrap()));
    }

    #[test]
    fn test_choose_empty() {
        let data: [i32; 0] = [];
        let mut rng = SplitMix64::from_seed(42);
        assert!(choose(&data, &mut rng).unwrap().is_none());
    }

    #[test]
    fn test_sample_indices_basic() {
        let mut rng = SplitMix64::from_seed(42);
        let indices = sample_indices(100, 5, &mut rng).unwrap();
        assert_eq!(indices.len(), 5);
        // Check all distinct
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);
        // All in range
        for &i in &indices {
            assert!(i < 100);
        }
    }

    #[test]
    fn test_sample_indices_k_exceeds_n() {
        let mut rng = SplitMix64::from_seed(42);
        let error = sample_indices(3, 5, &mut rng).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_sample_indices_k_equals_n() {
        let mut rng = SplitMix64::from_seed(42);
        let indices = sample_indices(5, 5, &mut rng).unwrap();
        assert_eq!(indices.len(), 5);
        let mut sorted = indices;
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3, 4]);
    }

    /// Constant words that record the requested uniform width.
    struct WidthLog {
        word: u64,
        narrow: usize,
        wide: usize,
    }

    impl WidthLog {
        const fn new(word: u64) -> Self {
            Self {
                word,
                narrow: 0,
                wide: 0,
            }
        }
    }

    impl BitGenerator for WidthLog {
        #[allow(clippy::cast_possible_truncation)]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.narrow += 1;
            Ok(self.word as u32)
        }
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.wide += 1;
            Ok(self.word)
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_uniform_u32()
        }
        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_uniform_u64()
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    #[test]
    fn index_word_width_follows_the_bound() {
        for upper in [1_usize, 2, 1000, u32::MAX as usize] {
            let mut rng = WidthLog::new(u64::MAX);
            assert_eq!(uniform_index(upper, &mut rng).unwrap(), upper - 1);
            assert_eq!((rng.narrow, rng.wide), (1, 0), "{upper}");
        }
        // 2^32 is the widest bound decoded from a 32-bit word: the index is the word.
        let mut rng = WidthLog::new(0x1234_5678_9abc_def0);
        assert_eq!(uniform_below(1 << 32, &mut rng).unwrap(), 0x9abc_def0);
        assert_eq!((rng.narrow, rng.wide), (1, 0));
        for upper in [(1_u64 << 32) + 1, u64::MAX] {
            let mut rng = WidthLog::new(u64::MAX);
            assert_eq!(uniform_below(upper, &mut rng).unwrap(), upper - 1);
            assert_eq!((rng.narrow, rng.wide), (0, 1), "{upper}");
        }
    }

    #[test]
    fn narrow_rejection_exhausts_after_1024_words() {
        // 0 * 3 has low half 0 below 2^32 mod 3 = 1, so every word is rejected.
        let mut rng = WidthLog::new(0);
        let error = uniform_below(3, &mut rng).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        assert_eq!((rng.narrow, rng.wide), (MAX_SAMPLING_ATTEMPTS, 0));
        // A bound dividing 2^32 never rejects: 2^32 mod 4 = 0.
        let mut rng = WidthLog::new(0);
        assert_eq!(uniform_below(4, &mut rng).unwrap(), 0);
        assert_eq!(rng.narrow, 1);
    }

    #[test]
    fn sample_indices_regimes_select_uniform_subsets() {
        // n = 64: k = 3 uses Floyd's set, k = 4 the partial shuffle.
        for k in [3, 4, 32, 64] {
            let mut rng = SplitMix64::from_seed(1346 + k as u64);
            let trials = 20_000;
            let mut counts = [0_usize; 64];
            for _ in 0..trials {
                let indices = sample_indices(64, k, &mut rng).unwrap();
                assert_eq!(indices.len(), k);
                let mut seen = [false; 64];
                for &index in &indices {
                    assert!(!seen[index], "duplicate {index} for k={k}");
                    seen[index] = true;
                    counts[index] += 1;
                }
            }
            let p = k as f64 / 64.0;
            let expected = trials as f64 * p;
            let deviation = (expected * (1.0 - p)).sqrt().max(1.0);
            for (index, &count) in counts.iter().enumerate() {
                assert!(
                    (count as f64 - expected).abs() <= 6.0 * deviation,
                    "k={k} index={index}: {count} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn dense_sample_indices_are_uniformly_ordered() {
        // Every ordered pair from 0..4 has probability 1/12.
        let mut rng = SplitMix64::from_seed(1345);
        let trials = 24_000;
        let mut counts = [[0_usize; 4]; 4];
        for _ in 0..trials {
            let pair = sample_indices(4, 2, &mut rng).unwrap();
            counts[pair[0]][pair[1]] += 1;
        }
        let expected = trials as f64 / 12.0;
        for (first, row) in counts.iter().enumerate() {
            for (second, &count) in row.iter().enumerate() {
                if first == second {
                    assert_eq!(count, 0);
                } else {
                    assert!((count as f64 - expected).abs() <= 6.0 * expected.sqrt());
                }
            }
        }
    }

    #[test]
    fn empty_sample_draws_nothing() {
        let mut rng = WidthLog::new(0);
        assert!(sample_indices(10, 0, &mut rng).unwrap().is_empty());
        assert!(sample_indices(0, 0, &mut rng).unwrap().is_empty());
        assert_eq!((rng.narrow, rng.wide), (0, 0));
    }
}
