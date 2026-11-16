//! Exact combinatorial counts in `usize`, permutations, and enumeration and ranking
//! of permutations, combinations and multisets.
//!
//! Counts are computed exactly in integer arithmetic. A count that exceeds
//! `usize::MAX` is a `SizeOverflow` error, never a wrapped or rounded value.
//! Indices and ranks start at zero, and enumeration and ranking use
//! lexicographic order. Types that hold a `Vec` have constructors that
//! allocate; the forms that take a slice you provide do not.
use crate::backend::{checked_mul, try_copy_slice, try_filled};
use crate::size::validate_same_len;
use crate::{InvalidInputReason, SciOxideError};
use alloc::vec::Vec;

fn invalid(argument: crate::Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::Inconsistent, detail)
}

/// Computes `n!` exactly, with `0! = 1`.
///
/// # Errors
///
/// - `SizeOverflow` if `n!` exceeds `usize::MAX`.
pub fn checked_factorial(n: usize) -> Result<usize, SciOxideError> {
    (2..=n).try_fold(1, |value, factor| {
        checked_mul(value, factor, crate::quantity!("factorial"))
    })
}

/// Computes the multifactorial `n(n − step)(n − 2·step)⋯`, stopping before the
/// first factor below 2.
///
/// The empty product is 1, so `n = 0` and `n = 1` give 1. `step = 1` gives
/// `n!` and `step = 2` gives `n!!`.
///
/// # Errors
///
/// - `InvalidInput` (`Inconsistent`) if `step` is zero.
/// - `SizeOverflow` if the product exceeds `usize::MAX`.
pub fn checked_multifactorial(n: usize, step: usize) -> Result<usize, SciOxideError> {
    if step == 0 {
        return Err(invalid(
            crate::argument!("step"),
            "multifactorial step must be positive",
        ));
    }
    let mut factor = n;
    let mut value = 1;
    while factor > 1 {
        value = checked_mul(value, factor, crate::quantity!("multifactorial"))?;
        factor = factor.saturating_sub(step);
    }
    Ok(value)
}

/// Computes the double factorial `n!! = n(n − 2)(n − 4)⋯`, with `0!! = 1!! = 1`.
///
/// # Errors
///
/// - `SizeOverflow` if `n!!` exceeds `usize::MAX`.
pub fn checked_double_factorial(n: usize) -> Result<usize, SciOxideError> {
    checked_multifactorial(n, 2)
}

/// Computes the rising factorial (Pochhammer symbol) `a(a + 1)⋯(a + n − 1)`.
///
/// The empty product (`n = 0`) is 1, and `a = 0` with `n > 0` gives 0.
///
/// # Errors
///
/// - `SizeOverflow` if a factor or the product exceeds `usize::MAX`.
pub fn checked_rising_factorial(a: usize, n: usize) -> Result<usize, SciOxideError> {
    if n > 0 && a == 0 {
        return Ok(0);
    }
    (0..n).try_fold(1, |value, i| {
        let factor = a
            .checked_add(i)
            .ok_or_else(|| SciOxideError::size_overflow(crate::quantity!("rising_factorial")))?;
        checked_mul(value, factor, crate::quantity!("rising_factorial"))
    })
}

fn checked_stirling_magnitude(n: usize, k: usize, first: bool) -> Result<usize, SciOxideError> {
    if n == k {
        return Ok(1);
    }
    if k == 0 || k > n {
        return Ok(0);
    }
    if !first && k == 1 {
        return Ok(1);
    }
    // Store only ancestors of the requested cell. Overflow in an unrelated
    // column must not reject a representable answer near the diagonal.
    let length = k
        .checked_add(1)
        .ok_or_else(|| SciOxideError::size_overflow(crate::quantity!("stirling_scratch")))?;
    let mut cells = try_filled(length, 0_usize, crate::quantity!("stirling_scratch"))?;
    cells[0] = 1;
    for row in 1..=n {
        let lower = k.saturating_sub(n - row).max(1);
        for column in (lower..=k.min(row)).rev() {
            let factor = if first { row - 1 } else { column };
            cells[column] = cells[column]
                .checked_mul(factor)
                .and_then(|v| v.checked_add(cells[column - 1]))
                .ok_or_else(|| SciOxideError::size_overflow(crate::quantity!("stirling_number")))?;
        }
        cells[0] = 0;
    }
    Ok(cells[k])
}

/// Computes the signed Stirling number of the first kind `s(n, k)`.
///
/// `s(n, k) = (−1)^(n−k) c(n, k)`, where the unsigned number `c(n, k)` counts
/// permutations of `n` elements with exactly `k` cycles. `s(0, 0) = 1`, and
/// `s(n, k) = 0` for `k > n` or `k = 0 < n`. The value comes from the
/// recurrence `c(n, k) = (n − 1) c(n − 1, k) + c(n − 1, k − 1)`, evaluated over
/// the `k·(n − k + 1)` cells that lead to `c(n, k)`, one checked multiply-add
/// each, in one allocated row of `k + 1` words. A caller that must bound the
/// work compares `k·(n − k + 1)` with its limit before the call. A `k` whose
/// row cannot be reserved returns `AllocationFailure`; the function never
/// panics or aborts.
///
/// # Errors
///
/// - `SizeOverflow` if `s(n, k)` or an intermediate magnitude does not fit
///   (`isize` for the result, `usize` for intermediates).
/// - `AllocationFailure` if the row of `k + 1` words cannot be reserved.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_stirling_first_kind;
///
/// // Four elements form two cycles in 11 ways; the sign is (−1)^(4−2).
/// assert_eq!(checked_stirling_first_kind(4, 2)?, 11);
/// assert_eq!(checked_stirling_first_kind(4, 3)?, -6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_stirling_first_kind(n: usize, k: usize) -> Result<isize, SciOxideError> {
    let magnitude = checked_stirling_magnitude(n, k, true)?;
    if k <= n && !(n - k).is_multiple_of(2) {
        if magnitude == (isize::MAX as usize) + 1 {
            return Ok(isize::MIN);
        }
        isize::try_from(magnitude)
            .map(|v| -v)
            .map_err(|_| SciOxideError::size_overflow(crate::quantity!("signed_stirling_number")))
    } else {
        isize::try_from(magnitude)
            .map_err(|_| SciOxideError::size_overflow(crate::quantity!("signed_stirling_number")))
    }
}

/// Computes the Stirling number of the second kind `S(n, k)`.
///
/// `S(n, k)` counts partitions of `n` elements into `k` nonempty blocks.
/// `S(0, 0) = 1`, and `S(n, k) = 0` for `k > n` or `k = 0 < n`. The value comes
/// from the recurrence `S(n, k) = k S(n − 1, k) + S(n − 1, k − 1)`, evaluated
/// over the `k·(n − k + 1)` cells that lead to `S(n, k)`, one checked
/// multiply-add each, in one allocated row of `k + 1` words. A caller that must
/// bound the work compares `k·(n − k + 1)` with its limit before the call. A
/// `k` whose row cannot be reserved returns `AllocationFailure`; the function
/// never panics or aborts.
///
/// # Errors
///
/// - `SizeOverflow` if `S(n, k)` or an intermediate value exceeds `usize::MAX`.
/// - `AllocationFailure` if the row of `k + 1` words cannot be reserved.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_stirling_second_kind;
///
/// // Ten elements fall into three nonempty blocks in 9330 ways.
/// assert_eq!(checked_stirling_second_kind(10, 3)?, 9330);
/// assert_eq!(checked_stirling_second_kind(3, 4)?, 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_stirling_second_kind(n: usize, k: usize) -> Result<usize, SciOxideError> {
    checked_stirling_magnitude(n, k, false)
}

/// Computes the number of `k`-permutations of `n` elements, `n!/(n − k)!`.
///
/// This is the number of ordered selections of `k` distinct elements. It is
/// zero for `k > n` and one for `k = 0`.
///
/// # Errors
///
/// - `SizeOverflow` if the result exceeds `usize::MAX`.
pub fn checked_permutation_count(n: usize, k: usize) -> Result<usize, SciOxideError> {
    if k > n {
        return Ok(0);
    }
    (0..k).try_fold(1, |value, i| {
        checked_mul(value, n - i, crate::quantity!("permutation_count"))
    })
}

/// Computes the greatest common divisor by Euclid's algorithm, with `gcd(0, 0) = 0`.
///
/// ```
/// assert_eq!(scioxide_core::backend::gcd(18, 12), 6);
/// assert_eq!(scioxide_core::backend::gcd(0, 0), 0);
/// ```
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Computes the binomial coefficient `C(n, k) = n!/(k!(n − k)!)` exactly.
///
/// `C(n, k) = 0` for `k > n` and `C(n, 0) = 1`. Common factors are cancelled
/// before each multiplication, so no intermediate value exceeds the result.
/// The cost is O(min(k, n − k)) integer operations.
///
/// # Errors
///
/// - `SizeOverflow` if `C(n, k)` exceeds `usize::MAX`.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_binomial;
///
/// // C(5, 2) = 10 two-element subsets of a five-element set.
/// assert_eq!(checked_binomial(5, 2)?, 10);
/// assert_eq!(checked_binomial(2, 5)?, 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_binomial(n: usize, k: usize) -> Result<usize, SciOxideError> {
    if k > n {
        return Ok(0);
    }
    // After step i, the running value is exactly C(n - k + i, i). Intermediates
    // never exceed the result, so overflow is reported only when the answer
    // itself does not fit.
    let k = k.min(n - k);
    (1..=k).try_fold(1, |value, i| {
        let numerator = n - k + i;
        let divisor = gcd(numerator, i);
        // Coprimality implies the remaining denominator divides value exactly.
        checked_mul(
            value / (i / divisor),
            numerator / divisor,
            crate::quantity!("binomial_coefficient"),
        )
    })
}

/// Computes the number of multisets of size `k` from `n` elements, `C(n + k − 1, k)`.
///
/// This is the number of nondecreasing sequences of length `k` from `0..n`.
/// It is one for `k = 0` and zero for `n = 0 < k`.
///
/// # Errors
///
/// - `SizeOverflow` if the count, or `n + k − 1`, exceeds `usize::MAX`.
pub fn checked_multiset_count(n: usize, k: usize) -> Result<usize, SciOxideError> {
    if k == 0 {
        return Ok(1);
    }
    if n == 0 {
        return Ok(0);
    }
    if n == 1 {
        return Ok(1);
    }
    let upper = n
        .checked_add(k - 1)
        .ok_or_else(|| SciOxideError::size_overflow(crate::quantity!("multiset_count")))?;
    checked_binomial(upper, k)
}

/// A permutation of `0..n`, stored with its inverse.
///
/// `new_to_old[new] = old` and `old_to_new[old] = new`. Applying the
/// permutation to a vector gathers `output[i] = input[new_to_old[i]]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    forward: Vec<usize>,
    inverse: Vec<usize>,
}
impl Permutation {
    /// Creates a permutation from its `new_to_old` map and computes the inverse.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if the map is not a bijection of
    ///   `0..n`, that is, it has a repeated or out-of-range index.
    /// - `AllocationFailure` if the inverse cannot be stored.
    pub fn new(new_to_old: Vec<usize>) -> Result<Self, SciOxideError> {
        let mut inverse = try_filled(
            new_to_old.len(),
            usize::MAX,
            crate::quantity!("inverse_permutation"),
        )?;
        for (new, &old) in new_to_old.iter().enumerate() {
            if old >= inverse.len() || inverse[old] != usize::MAX {
                return Err(invalid(
                    crate::argument!("new_to_old"),
                    "indices are not a bijection",
                ));
            }
            inverse[old] = new;
        }
        Ok(Self {
            forward: new_to_old,
            inverse,
        })
    }

    /// Replaces this permutation with the identity of length `n`, reusing its storage.
    ///
    /// On error the permutation is unchanged, although its capacity may have
    /// grown.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot grow to `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Permutation;
    ///
    /// let mut p = Permutation::new(vec![2, 0, 1])?;
    /// p.try_reset_identity(2)?;
    /// assert_eq!(p.new_to_old(), &[0, 1]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reset_identity(&mut self, n: usize) -> Result<(), SciOxideError> {
        let q = crate::quantity!("identity_permutation");
        let forward = n.saturating_sub(self.forward.len());
        let inverse = n.saturating_sub(self.inverse.len());
        crate::backend::try_reserve(&mut self.forward, forward, q)?;
        crate::backend::try_reserve(&mut self.inverse, inverse, q)?;
        self.forward.resize(n, 0);
        self.inverse.resize(n, 0);
        write_identity(&mut self.forward);
        write_identity(&mut self.inverse);
        Ok(())
    }
    /// Returns the map from new position to original index.
    pub fn new_to_old(&self) -> &[usize] {
        &self.forward
    }
    /// Returns the map from original index to new position.
    pub fn old_to_new(&self) -> &[usize] {
        &self.inverse
    }
    /// Writes `output[i] = input[new_to_old[i]]`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` has a length other than `n`;
    ///   `output` is then unchanged.
    pub fn apply_into<T: Copy>(&self, input: &[T], output: &mut [T]) -> Result<(), SciOxideError> {
        self.as_view().apply_into(input, output)
    }
    /// Writes `output[new_to_old[i]] = input[i]`, the inverse of [`Self::apply_into`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` has a length other than `n`;
    ///   `output` is then unchanged.
    pub fn inverse_into<T: Copy>(
        &self,
        input: &[T],
        output: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.as_view().inverse_into(input, output)
    }
}

impl Permutation {
    /// Returns the identity permutation of `0..n`.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn identity(n: usize) -> Result<Self, SciOxideError> {
        let mut forward = try_filled(n, 0, crate::quantity!("identity_permutation"))?;
        write_identity(&mut forward);
        Self::new(forward)
    }

    /// Returns a borrowed view of the permutation.
    pub fn as_view(&self) -> PermutationView<'_> {
        PermutationView { map: &self.forward }
    }

    /// Returns the inverse permutation.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn inverse(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            forward: try_copy_slice(&self.inverse, crate::quantity!("inverse_permutation"))?,
            inverse: try_copy_slice(&self.forward, crate::quantity!("inverse_permutation"))?,
        })
    }

    /// Returns the composition with `new_to_old[i] = other.new_to_old[self.new_to_old[i]]`.
    ///
    /// Gathering with the result equals gathering with `other` first and then
    /// with `self`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ.
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn compose(&self, other: &Self) -> Result<Self, SciOxideError> {
        validate_same_len(self.forward.len(), other.forward.len())?;
        let mut forward = try_filled(
            self.forward.len(),
            0,
            crate::quantity!("permutation_composition"),
        )?;
        self.as_view().compose_into(other.as_view(), &mut forward)?;
        Self::new(forward)
    }
}

/// Returns the scratch length in bytes, `⌈n/8⌉`, needed to check a permutation of length `n`.
///
/// It is used by [`PermutationView::new`] and [`PermutationEnumerator::new`],
/// and does not overflow for any `n`.
pub const fn permutation_validation_scratch_bytes(n: usize) -> usize {
    n / 8 + (!n.is_multiple_of(8)) as usize
}

/// A borrowed permutation map `new_to_old`, checked to be a bijection of `0..n`.
///
/// Checking costs O(n) time and needs [`permutation_validation_scratch_bytes`]
/// bytes of scratch. Gathers, inverses and compositions cost O(n).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PermutationView<'a> {
    map: &'a [usize],
}
impl<'a> PermutationView<'a> {
    /// Creates a view after checking that every index in `0..map.len()` occurs exactly once.
    ///
    /// The contents of `scratch` are unspecified afterward.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `scratch` is shorter than
    ///   [`permutation_validation_scratch_bytes`]`(map.len())`.
    /// - `InvalidInput` (`Inconsistent`) if an index is repeated or out of range.
    pub fn new(map: &'a [usize], scratch: &mut [u8]) -> Result<Self, SciOxideError> {
        validate_permutation(map, scratch)?;
        Ok(Self { map })
    }
    /// Returns the map from new position to original index.
    pub fn new_to_old(self) -> &'a [usize] {
        self.map
    }

    /// Writes `output[i] = input[new_to_old[i]]`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` has a length other than `n`;
    ///   `output` is then unchanged.
    pub fn apply_into<T: Copy>(self, input: &[T], output: &mut [T]) -> Result<(), SciOxideError> {
        validate_same_len(self.map.len(), input.len())?;
        validate_same_len(self.map.len(), output.len())?;
        output
            .iter_mut()
            .zip(self.map)
            .for_each(|(out, &i)| *out = input[i]);
        Ok(())
    }
    /// Writes `output[new_to_old[i]] = input[i]`, the inverse of [`Self::apply_into`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` has a length other than `n`;
    ///   `output` is then unchanged.
    pub fn inverse_into<T: Copy>(self, input: &[T], output: &mut [T]) -> Result<(), SciOxideError> {
        validate_same_len(self.map.len(), input.len())?;
        validate_same_len(self.map.len(), output.len())?;
        for (&i, &value) in self.map.iter().zip(input) {
            output[i] = value;
        }
        Ok(())
    }
    /// Writes the inverse map, `output[new_to_old[i]] = i`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output.len() != n`; `output` is then unchanged.
    pub fn inverse_map_into(self, output: &mut [usize]) -> Result<(), SciOxideError> {
        validate_same_len(self.map.len(), output.len())?;
        for (i, &value) in self.map.iter().enumerate() {
            output[value] = i;
        }
        Ok(())
    }
    /// Writes the composition `output[i] = other[self[i]]`.
    ///
    /// Gathering with `output` equals gathering with `other` first and then
    /// with `self`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the lengths differ; `output` is then unchanged.
    pub fn compose_into(self, other: Self, output: &mut [usize]) -> Result<(), SciOxideError> {
        validate_same_len(self.map.len(), other.map.len())?;
        validate_same_len(self.map.len(), output.len())?;
        self.apply_into(other.map, output)
    }
}

// Validate the caller's `map` parameter.
fn validate_permutation(values: &[usize], scratch: &mut [u8]) -> Result<(), SciOxideError> {
    let required = permutation_validation_scratch_bytes(values.len());
    if scratch.len() < required {
        return Err(SciOxideError::dimension_mismatch(required, scratch.len()));
    }
    let scratch = &mut scratch[..required];
    scratch.fill(0);
    for &value in values {
        if value >= values.len() {
            return Err(invalid(
                crate::argument!("map"),
                "indices are not a bijection",
            ));
        }
        let byte = value / 8;
        let bit = 1_u8 << (value % 8);
        if scratch[byte] & bit != 0 {
            return Err(invalid(
                crate::argument!("map"),
                "indices are not a bijection",
            ));
        }
        scratch[byte] |= bit;
    }
    Ok(())
}

fn write_identity(values: &mut [usize]) {
    values
        .iter_mut()
        .enumerate()
        .for_each(|(i, value)| *value = i);
}

// Return the first changed position, or `None` at the final state.
fn advance_permutation(values: &mut [usize]) -> Option<usize> {
    // Lexicographic successor (Narayana): find the rightmost ascent, swap its
    // left element with the rightmost larger element, and reverse the
    // now-descending suffix.
    let i = (0..values.len().saturating_sub(1))
        .rev()
        .find(|&i| values[i] < values[i + 1])?;
    if let Some(j) = (i + 1..values.len()).rev().find(|&j| values[i] < values[j]) {
        values.swap(i, j);
        values[i + 1..].reverse();
    }
    Some(i)
}

/// Enumerates the permutations of `0..n` in lexicographic order.
///
/// The state is stored in `B`, which is a `Vec<usize>` for `PermutationEnumerator::identity`
/// or any mutable slice you provide. [`Self::advance`] computes the next
/// permutation in place (Narayana's algorithm).
///
/// ```
/// // The permutation after [0, 1, 2] in lexicographic order is [0, 2, 1].
/// let mut state = scioxide_core::PermutationEnumerator::identity(3)?;
/// assert!(state.advance());
/// assert_eq!(state.new_to_old(), &[0, 2, 1]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermutationEnumerator<B = Vec<usize>> {
    map: B,
}

impl<B: AsRef<[usize]> + AsMut<[usize]>> PermutationEnumerator<B> {
    /// Starts enumeration from the permutation already stored in `map`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `scratch` is shorter than
    ///   [`permutation_validation_scratch_bytes`]`(n)`.
    /// - `InvalidInput` (`Inconsistent`) if `map` is not a permutation of `0..n`.
    pub fn new(mut map: B, scratch: &mut [u8]) -> Result<Self, SciOxideError> {
        validate_permutation(map.as_mut(), scratch)?;
        Ok(Self { map })
    }

    /// Starts enumeration at the identity, overwriting `map`.
    pub fn identity_into(mut map: B) -> Self {
        write_identity(map.as_mut());
        Self { map }
    }

    /// Returns the current permutation as a `new_to_old` map.
    pub fn new_to_old(&self) -> &[usize] {
        self.map.as_ref()
    }

    /// Returns the current permutation as a [`PermutationView`].
    pub fn as_view(&self) -> PermutationView<'_> {
        PermutationView {
            map: self.map.as_ref(),
        }
    }

    /// Moves to the next permutation in lexicographic order.
    ///
    /// Returns `false`, and leaves the state unchanged, at the last
    /// permutation. Each step costs O(1) amortized over a full traversal and
    /// O(n) at worst.
    pub fn advance(&mut self) -> bool {
        advance_permutation(self.map.as_mut()).is_some()
    }

    /// Returns to the identity permutation.
    pub fn reset(&mut self) {
        write_identity(self.map.as_mut());
    }

    /// Returns the storage, which holds the current permutation.
    pub fn into_inner(self) -> B {
        self.map
    }
}
impl PermutationEnumerator<Vec<usize>> {
    /// Starts enumeration at the identity permutation of `0..n`.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the state cannot be stored.
    pub fn identity(n: usize) -> Result<Self, SciOxideError> {
        let mut map = try_filled(n, 0, crate::quantity!("permutation_state"))?;
        write_identity(&mut map);
        Ok(Self { map })
    }
}

// Validate a selection parameter `argument`.
fn validate_selection(
    n: usize,
    values: &[usize],
    repeated: bool,
    argument: crate::Argument,
) -> Result<(), SciOxideError> {
    if values.iter().any(|&v| v >= n)
        || values.windows(2).any(|pair| {
            if repeated {
                pair[0] > pair[1]
            } else {
                pair[0] >= pair[1]
            }
        })
    {
        return Err(invalid(
            argument,
            "indices must be in range and canonically ordered",
        ));
    }
    Ok(())
}

fn write_first_combination(n: usize, values: &mut [usize]) -> Result<(), SciOxideError> {
    if values.len() > n {
        return Err(invalid(
            crate::argument!("indices"),
            "selection exceeds universe",
        ));
    }
    write_identity(values);
    Ok(())
}

fn write_first_multiset(n: usize, values: &mut [usize]) -> Result<(), SciOxideError> {
    if n == 0 && !values.is_empty() {
        return Err(invalid(crate::argument!("indices"), "empty universe"));
    }
    values.fill(0);
    Ok(())
}

fn advance_combination(n: usize, values: &mut [usize]) -> bool {
    let k = values.len();
    if k == 0 || k == n {
        return false;
    }
    // Position i can hold at most n - k + i. Increment the rightmost position
    // below its maximum and reset everything after it to consecutive values.
    let Some(i) = (0..k).rev().find(|&i| values[i] < n - k + i) else {
        return false;
    };
    let first = values[i] + 1;
    values[i..]
        .iter_mut()
        .enumerate()
        .for_each(|(j, value)| *value = first + j);
    true
}

fn advance_multiset(n: usize, values: &mut [usize]) -> bool {
    if n <= 1 || values.is_empty() {
        return false;
    }
    let Some(i) = values.iter().rposition(|&v| v < n - 1) else {
        return false;
    };
    let next = values[i] + 1;
    values[i..].fill(next);
    true
}

/// Enumerates the `k`-element subsets of `0..n` in lexicographic order.
///
/// Each subset is a strictly increasing index list. The state is stored in
/// `B`, which is a `Vec<usize>` for `CombinationEnumerator::first` or any
/// mutable slice you provide.
///
/// ```
/// // The first 2-subset of {0, 1, 2} is {0, 1}.
/// let state = scioxide_core::CombinationEnumerator::first(3, 2)?;
/// assert_eq!(state.indices(), &[0, 1]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombinationEnumerator<B = Vec<usize>> {
    n: usize,
    indices: B,
}

impl<B: AsRef<[usize]> + AsMut<[usize]>> CombinationEnumerator<B> {
    /// Starts enumeration from the subset already stored in `indices`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if an index is not less than `n` or the
    ///   indices are not strictly increasing.
    pub fn new(n: usize, indices: B) -> Result<Self, SciOxideError> {
        validate_selection(n, indices.as_ref(), false, crate::argument!("indices"))?;
        Ok(Self { n, indices })
    }

    /// Starts enumeration at `[0, 1, …, k − 1]`, overwriting `indices`, where `k = indices.len()`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `k > n`; `indices` is then unchanged.
    pub fn first_into(n: usize, mut indices: B) -> Result<Self, SciOxideError> {
        write_first_combination(n, indices.as_mut())?;
        Ok(Self { n, indices })
    }

    /// Returns the current subset as increasing indices.
    pub fn indices(&self) -> &[usize] {
        self.indices.as_ref()
    }

    /// Moves to the next subset in lexicographic order, in O(k) time.
    ///
    /// Returns `false`, and leaves the state unchanged, at the last subset.
    pub fn advance(&mut self) -> bool {
        advance_combination(self.n, self.indices.as_mut())
    }

    /// Returns to the first subset `[0, 1, …, k − 1]`.
    pub fn reset(&mut self) {
        write_identity(self.indices.as_mut());
    }

    /// Returns the storage, which holds the current subset.
    pub fn into_inner(self) -> B {
        self.indices
    }
}
impl CombinationEnumerator<Vec<usize>> {
    /// Starts enumeration of the `k`-subsets of `0..n` at `[0, 1, …, k − 1]`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `k > n`.
    /// - `AllocationFailure` if the state cannot be stored.
    pub fn first(n: usize, k: usize) -> Result<Self, SciOxideError> {
        if k > n {
            return Err(invalid(crate::argument!("k"), "selection exceeds universe"));
        }
        let mut indices = try_filled(k, 0, crate::quantity!("combination_state"))?;
        write_identity(&mut indices);
        Ok(Self { n, indices })
    }
}

/// Enumerates the multisets of size `k` from `0..n` in lexicographic order.
///
/// Each multiset is a nondecreasing index list. The state is stored in `B`,
/// which is a `Vec<usize>` for `MultisetEnumerator::first` or any mutable
/// slice you provide.
///
/// ```
/// // The first multiset of size 2 from {0, 1, 2} is {0, 0}.
/// let state = scioxide_core::MultisetEnumerator::first(3, 2)?;
/// assert_eq!(state.indices(), &[0, 0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultisetEnumerator<B = Vec<usize>> {
    n: usize,
    indices: B,
}

impl<B: AsRef<[usize]> + AsMut<[usize]>> MultisetEnumerator<B> {
    /// Starts enumeration from the multiset already stored in `indices`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if an index is not less than `n` or the
    ///   indices decrease.
    pub fn new(n: usize, indices: B) -> Result<Self, SciOxideError> {
        validate_selection(n, indices.as_ref(), true, crate::argument!("indices"))?;
        Ok(Self { n, indices })
    }

    /// Starts enumeration at `[0, 0, …, 0]`, overwriting `indices`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `n = 0` and `indices` is not empty;
    ///   `indices` is then unchanged.
    pub fn first_into(n: usize, mut indices: B) -> Result<Self, SciOxideError> {
        write_first_multiset(n, indices.as_mut())?;
        Ok(Self { n, indices })
    }

    /// Returns the current multiset as nondecreasing indices.
    pub fn indices(&self) -> &[usize] {
        self.indices.as_ref()
    }

    /// Moves to the next multiset in lexicographic order, in O(k) time.
    ///
    /// Returns `false`, and leaves the state unchanged, at the last multiset.
    pub fn advance(&mut self) -> bool {
        advance_multiset(self.n, self.indices.as_mut())
    }

    /// Returns to the first multiset `[0, 0, …, 0]`.
    pub fn reset(&mut self) {
        self.indices.as_mut().fill(0);
    }

    /// Returns the storage, which holds the current multiset.
    pub fn into_inner(self) -> B {
        self.indices
    }
}
impl MultisetEnumerator<Vec<usize>> {
    /// Starts enumeration of the size-`k` multisets of `0..n` at `[0, 0, …, 0]`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if `n = 0` and `k > 0`.
    /// - `AllocationFailure` if the state cannot be stored.
    pub fn first(n: usize, k: usize) -> Result<Self, SciOxideError> {
        if n == 0 && k != 0 {
            return Err(invalid(crate::argument!("k"), "selection exceeds universe"));
        }
        let indices = try_filled(k, 0, crate::quantity!("multiset_state"))?;

        Ok(Self { n, indices })
    }
}

/// Returns the zero-based lexicographic rank of a permutation, in O(n²) time.
///
/// The rank is computed from the Lehmer code. The identity has rank 0 and the
/// reversed permutation has rank `n! − 1`.
///
/// # Errors
///
/// - `SizeOverflow` if `n!` exceeds `usize::MAX`.
pub fn permutation_rank(values: PermutationView<'_>) -> Result<usize, SciOxideError> {
    let values = values.new_to_old();
    // Lehmer code: digit i counts the later elements smaller than values[i],
    // weighted by (len - 1 - i)! in the factorial number system.
    let mut block = checked_factorial(values.len())?;
    let mut rank = 0;
    for (i, &value) in values.iter().enumerate() {
        block /= values.len() - i;
        let digit = values[i + 1..].iter().filter(|&&v| v < value).count();
        rank += digit * block;
    }
    Ok(rank)
}

fn validate_rank(rank: usize, count: usize) -> Result<(), SciOxideError> {
    if rank >= count {
        return Err(SciOxideError::out_of_bounds(
            crate::quantity!("combinatorial_rank"),
            rank,
            count,
        ));
    }
    Ok(())
}

/// Writes the permutation of `0..output.len()` with zero-based lexicographic rank `rank`.
///
/// This is the inverse of [`permutation_rank`] and costs O(n²) time.
///
/// # Errors
///
/// - `SizeOverflow` if `n!` exceeds `usize::MAX`.
/// - `OutOfBounds` if `rank >= n!`.
///
/// On error, `output` is unchanged.
pub fn permutation_unrank_into(mut rank: usize, output: &mut [usize]) -> Result<(), SciOxideError> {
    let mut block = checked_factorial(output.len())?;
    validate_rank(rank, block)?;
    write_identity(output);
    for i in 0..output.len() {
        block /= output.len() - i;
        let digit = rank / block;
        rank %= block;
        // The unused values in output[i..] stay sorted. Rotating moves the
        // digit-th smallest of them to position i and keeps the rest in order.
        output[i..=i + digit].rotate_right(1);
    }
    Ok(())
}

// Rank increasing indices; multisets supply the virtual transformation a[i]+i.
// Every binomial below counts a subset of the admitted total, so it fits usize.
fn selection_rank(n: usize, values: &[usize], repeated: bool) -> Result<usize, SciOxideError> {
    let mut start = 0;
    let mut rank = 0;
    for (i, &raw) in values.iter().enumerate() {
        let value = raw + if repeated { i } else { 0 };
        let remaining = values.len() - i;
        rank += checked_binomial(n - start, remaining)? - checked_binomial(n - value, remaining)?;
        start = value + 1;
    }
    Ok(rank)
}

/// Returns the zero-based lexicographic rank of a `k`-subset of `0..n`, in O(k²) time.
///
/// `values` lists the subset in strictly increasing order.
///
/// # Errors
///
/// - `InvalidInput` (`Inconsistent`) if an index is not less than `n` or the
///   indices are not strictly increasing.
/// - `SizeOverflow` if `C(n, k)` exceeds `usize::MAX`.
pub fn combination_rank(n: usize, values: &[usize]) -> Result<usize, SciOxideError> {
    validate_selection(n, values, false, crate::argument!("values"))?;
    checked_binomial(n, values.len())?;
    selection_rank(n, values, false)
}

/// Returns the zero-based lexicographic rank of a size-`k` multiset of `0..n`, in O(k²) time.
///
/// `values` lists the multiset in nondecreasing order.
///
/// # Errors
///
/// - `InvalidInput` (`Inconsistent`) if an index is not less than `n` or the
///   indices decrease.
/// - `SizeOverflow` if the number of multisets exceeds `usize::MAX`.
pub fn multiset_rank(n: usize, values: &[usize]) -> Result<usize, SciOxideError> {
    validate_selection(n, values, true, crate::argument!("values"))?;
    checked_multiset_count(n, values.len())?;
    if values.is_empty() || n == 1 {
        return Ok(0);
    }
    // A representable nontrivial count guarantees this transformed universe fits.
    selection_rank(n + (values.len() - 1), values, true)
}

// Binary search the first index whose cumulative skipped block exceeds rank.
// Total-count admission makes every subsequent binomial and index representable.
fn selection_unrank(
    n: usize,
    mut rank: usize,
    output: &mut [usize],
    repeated: bool,
) -> Result<(), SciOxideError> {
    let k = output.len();
    let mut start = 0;
    for (i, value) in output.iter_mut().enumerate() {
        let remaining = k - i;
        let total = checked_binomial(n - start, remaining)?;
        let mut low = start;
        let mut high = n - remaining + 1;
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            let skipped = total - checked_binomial(n - mid, remaining)?;
            if skipped <= rank {
                low = mid;
            } else {
                high = mid;
            }
        }
        rank -= total - checked_binomial(n - low, remaining)?;
        *value = low - if repeated { i } else { 0 };
        start = low + 1;
    }
    Ok(())
}

/// Writes the `k`-subset of `0..n` with zero-based lexicographic rank `rank`, where `k = output.len()`.
///
/// This is the inverse of [`combination_rank`] and costs O(k² log(n + 1)) time.
///
/// # Errors
///
/// - `SizeOverflow` if `C(n, k)` exceeds `usize::MAX`.
/// - `OutOfBounds` if `rank >= C(n, k)`.
///
/// On error, `output` is unchanged.
pub fn combination_unrank_into(
    n: usize,
    rank: usize,
    output: &mut [usize],
) -> Result<(), SciOxideError> {
    validate_rank(rank, checked_binomial(n, output.len())?)?;
    selection_unrank(n, rank, output, false)
}

/// Writes the size-`k` multiset of `0..n` with zero-based lexicographic rank `rank`, where `k = output.len()`.
///
/// This is the inverse of [`multiset_rank`] and costs O(k² log(n + k)) time.
///
/// # Errors
///
/// - `SizeOverflow` if the number of multisets exceeds `usize::MAX`.
/// - `OutOfBounds` if `rank` is not less than that number.
///
/// On error, `output` is unchanged.
pub fn multiset_unrank_into(
    n: usize,
    rank: usize,
    output: &mut [usize],
) -> Result<(), SciOxideError> {
    validate_rank(rank, checked_multiset_count(n, output.len())?)?;
    if output.is_empty() || n == 1 {
        output.fill(0);
        return Ok(());
    }
    selection_unrank(n + (output.len() - 1), rank, output, true)
}
