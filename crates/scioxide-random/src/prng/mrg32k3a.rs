//! L'Ecuyer's `MRG32k3a` combined multiple recursive generator with streams
//! and substreams.

use scioxide_core::SciOxideError;

use crate::util::TransitionCounter;

use crate::state::{self, GeneratorAlgorithm as Algorithm, Snapshot, StateEngine, StreamIdentity};
use crate::traits::{BitGenerator, SeedableRng, SkipAhead, SplittableRng};

const M1: u64 = 4_294_967_087;
const M2: u64 = 4_294_944_443;
const A12: u64 = 1_403_580;
const A13N: u64 = 810_728;
const A21: u64 = 527_612;
const A23N: u64 = 1_370_589;
#[cfg(test)]
const NORM: f64 = 1.0 / 4_294_967_088.0;

type Matrix = [[u64; 3]; 3];

// Companion matrices of the two order-3 recurrences acting on the column
// vector (x_{n-3}, x_{n-2}, x_{n-1}). The negative published coefficients
// `-a13` and `-a23` are stored as `m - a` so every entry is a canonical
// residue and the matrix powers below stay in unsigned arithmetic.
const A1: Matrix = [[0, 1, 0], [0, 0, 1], [M1 - A13N, A12, 0]];
const A2: Matrix = [[0, 1, 0], [0, 0, 1], [M2 - A23N, 0, A21]];

// Published RngStreams jump matrices A^(2^76).
const A1P76: Matrix = [
    [82_758_667, 1_871_391_091, 4_127_413_238],
    [3_672_831_523, 69_195_019, 1_871_391_091],
    [3_672_091_415, 3_528_743_235, 69_195_019],
];
const A2P76: Matrix = [
    [1_511_326_704, 3_759_209_742, 1_610_795_712],
    [4_292_754_251, 1_511_326_704, 3_889_917_532],
    [3_859_662_829, 4_292_754_251, 3_708_466_080],
];

// Published RngStreams jump matrices A^(2^127).
const A1P127: Matrix = [
    [2_427_906_178, 3_580_155_704, 949_770_784],
    [226_153_695, 1_230_515_664, 3_580_155_704],
    [1_988_835_001, 986_791_581, 1_230_515_664],
];
const A2P127: Matrix = [
    [1_464_411_153, 277_697_599, 1_610_723_613],
    [32_183_930, 1_464_411_153, 1_022_607_788],
    [2_824_425_944, 32_183_930, 2_093_834_863],
];

/// Multiply two 3-by-3 matrices modulo a constant MRG modulus.
///
/// Each dot product is below `3 * (2^32 - 1)^2 < 2^66`, so `u128`
/// accumulation is exact.
#[allow(clippy::cast_possible_truncation)]
fn mat_mul_mod(left: &Matrix, right: &Matrix, modulus: u64) -> Matrix {
    let modulus = u128::from(modulus);
    core::array::from_fn(|row| {
        core::array::from_fn(|column| {
            (0..3)
                .map(|index| u128::from(left[row][index]) * u128::from(right[index][column]))
                .sum::<u128>()
                .rem_euclid(modulus) as u64
        })
    })
}

/// Multiply a 3-by-3 matrix by a component state modulo its modulus.
#[allow(clippy::cast_possible_truncation)]
fn mat_vec_mod(matrix: &Matrix, vector: &[u64; 3], modulus: u64) -> [u64; 3] {
    let modulus = u128::from(modulus);
    core::array::from_fn(|row| {
        (0..3)
            .map(|column| u128::from(matrix[row][column]) * u128::from(vector[column]))
            .sum::<u128>()
            .rem_euclid(modulus) as u64
    })
}

/// Apply a matrix power directly to a component, avoiding an accumulated
/// matrix and its unnecessary identity products. A zero power returns the
/// component unchanged.
fn advance_component(
    mut component: [u64; 3],
    mut matrix: Matrix,
    mut exponent: u128,
    modulus: u64,
) -> [u64; 3] {
    while exponent != 0 {
        if exponent & 1 != 0 {
            component = mat_vec_mod(&matrix, &component, modulus);
        }
        exponent >>= 1;
        if exponent != 0 {
            matrix = mat_mul_mod(&matrix, &matrix, modulus);
        }
    }
    component
}

/// Apply independently powered component matrices to one six-word state.
fn jump_state(
    s1: &[u64; 3],
    s2: &[u64; 3],
    component_1: &Matrix,
    component_2: &Matrix,
    exponent: u128,
) -> ([u64; 3], [u64; 3]) {
    (
        advance_component(*s1, *component_1, exponent, M1),
        advance_component(*s2, *component_2, exponent, M2),
    )
}

/// Evaluate `(positive_coefficient * positive_state - negative_coefficient *
/// negative_state) mod modulus` without signed or wrapping arithmetic.
///
/// Each coefficient-state product is below `6.1e15`, well inside `u64`.
#[inline]
fn recurrence(
    positive_coefficient: u64,
    positive_state: u64,
    negative_coefficient: u64,
    negative_state: u64,
    modulus: u64,
) -> u64 {
    let positive = (positive_coefficient * positive_state) % modulus;
    let negative = (negative_coefficient * negative_state) % modulus;
    (positive + modulus - negative) % modulus
}

/// Combined multiple recursive generator `MRG32k3a` with period about 2¹⁹¹ and streams.
///
/// Two order-3 components run in exact integer arithmetic,
///
/// ```text
/// xₙ = (1403580 xₙ₋₂ − 810728 xₙ₋₃) mod m₁,     m₁ = 2³² − 209,
/// yₙ = (527612 yₙ₋₁ − 1370589 yₙ₋₃) mod m₂,     m₂ = 2³² − 22853,
/// ```
///
/// and the native output is `zₙ = (xₙ − yₙ) mod m₁`, taken in `[1, m₁]` (a zero difference gives
/// `m₁`), with the parameters of L'Ecuyer (1999). [`next_u32`](crate::BitGenerator::next_u32)
/// returns `zₙ`. An `f64` uniform is `zₙ / (m₁ + 1)`, in `(0, 1)`. An `f32` uniform uses the top
/// 24 bits of `zₙ`. [`next_u64`](crate::BitGenerator::next_u64) joins two outputs, the first in
/// the low half.
///
/// Each output costs O(1). [`SkipAhead`](crate::SkipAhead) jumps `n` steps in `O(log n)` with
/// powers of the 3 × 3 companion matrices. Stream `i` of
/// [`substream`](crate::SplittableRng::substream) starts `i · 2¹²⁷` steps after the seeded
/// state, and substream `j` a further `j · 2⁷⁶` steps, as in L'Ecuyer et al. (2002). Substreams
/// `j < 2⁵¹` of one stream do not overlap. [`from_seed`](crate::SeedableRng::from_seed) takes
/// the six state words from [`SplitMix64`](crate::SplitMix64) outputs reduced modulo `m₁` and
/// `m₂`. Native draws never fail; uniform-word normalization can exhaust its retry limit.
///
/// # References
///
/// - P. L'Ecuyer, "Good parameters and implementations for combined multiple recursive random
///   number generators", *Operations Research* 47(1), 1999, 159–164.
/// - P. L'Ecuyer, R. Simard, E. J. Chen and W. D. Kelton, "An object-oriented random-number
///   package with many long streams and substreams", *Operations Research* 50(6), 2002,
///   1073–1075.
///
/// # Examples
///
/// ```
/// use scioxide_random::{BitGenerator, Mrg32k3a, Rng, SerializableState, SkipAhead};
/// use scioxide_random::{SeedableRng, SplittableRng};
/// // Default package seed of L'Ecuyer et al. (2002) and its first native word and uniform.
/// let mut root = Mrg32k3a::from_state_words([12_345; 6])?;
/// assert_eq!(root.clone().next_u32()?, 545_508_589);
/// let first: f64 = root.next_float()?;
/// assert!((first - 0.127_011_122_046_577).abs() < 1e-12);
/// let mut stream = Mrg32k3a::from_seed(42).substream(2, 1)?; // stream 2, substream 1
/// stream.skip_ahead(10)?;
/// let restored = Mrg32k3a::import_state(&stream.export_state()?)?;
/// assert_eq!(restored.state_words(), stream.state_words());
/// assert_eq!(restored.position(), Some(10));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct Mrg32k3a {
    s1: [u64; 3],
    s2: [u64; 3],
    origin: ([u64; 3], [u64; 3]),
    selectors: (u64, u64),
    counter: TransitionCounter,
}

impl Mrg32k3a {
    /// Creates a generator from its six state words.
    ///
    /// Words `0..3` are `(xₙ₋₃, xₙ₋₂, xₙ₋₁)` of the first component and words `3..6` the same for
    /// the second. Each word must be reduced modulo its component's modulus. The generator starts
    /// at position zero, and its streams are measured from this state.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a word of the first component is at least `m₁`, a word of the second
    ///   is at least `m₂`, or either component is all zero.
    pub fn from_state_words(words: [u64; 6]) -> Result<Self, SciOxideError> {
        let [s10, s11, s12, s20, s21, s22] = words;
        let s1 = [s10, s11, s12];
        let s2 = [s20, s21, s22];
        Self::validate_components(&s1, &s2, scioxide_core::argument!("words"))?;
        Ok(Self {
            s1,
            s2,
            origin: (s1, s2),
            selectors: (0, 0),
            counter: TransitionCounter::from_position(Some(0)),
        })
    }

    fn validate_components(
        s1: &[u64; 3],
        s2: &[u64; 3],
        parameter: scioxide_core::Argument,
    ) -> Result<(), SciOxideError> {
        if s1.iter().any(|&word| word >= M1) {
            return Err(SciOxideError::invalid_input(
                parameter,
                scioxide_core::InvalidInputReason::OutOfRange,
                alloc::format!("MRG32k3a component 1 words must be less than {M1}"),
            ));
        }
        if s2.iter().any(|&word| word >= M2) {
            return Err(SciOxideError::invalid_input(
                parameter,
                scioxide_core::InvalidInputReason::OutOfRange,
                alloc::format!("MRG32k3a component 2 words must be less than {M2}"),
            ));
        }
        if *s1 == [0; 3] || *s2 == [0; 3] {
            return Err(SciOxideError::invalid_input(
                parameter,
                scioxide_core::InvalidInputReason::Inconsistent,
                "MRG32k3a component triples must not be all zero",
            ));
        }
        Ok(())
    }

    fn validate_imported_components(s1: &[u64; 3], s2: &[u64; 3]) -> Result<(), SciOxideError> {
        Self::validate_components(s1, s2, scioxide_core::argument!("bytes")).map_err(|error| {
            state::malformed(
                alloc::format!("invalid canonical MRG32k3a state: {error}"),
                scioxide_core::argument!("bytes"),
            )
        })
    }

    /// Performs one recurrence step. The caller updates the position counter.
    #[inline]
    fn next_word(&mut self) -> u32 {
        let x1 = recurrence(A12, self.s1[1], A13N, self.s1[0], M1);
        let x2 = recurrence(A21, self.s2[2], A23N, self.s2[0], M2);
        self.s1 = [self.s1[1], self.s1[2], x1];
        self.s2 = [self.s2[1], self.s2[2], x2];

        // Combination step `(x1 - x2) mod m1`, mapped into [1, m1] rather than
        // [0, m1): equality yields m1 instead of 0, as in RngStreams. The word
        // is therefore never zero and fits in u32, and the `f64` decoder
        // (divide by m1 + 1) produces values strictly inside (0, 1).
        let word = if x1 > x2 { x1 - x2 } else { x1 + M1 - x2 };
        #[allow(clippy::cast_possible_truncation)]
        {
            word as u32
        }
    }

    fn advance_state(&mut self, transitions: u128) {
        (self.s1, self.s2) = jump_state(&self.s1, &self.s2, &A1, &A2, transitions);
    }
}

impl BitGenerator for Mrg32k3a {
    #[inline]
    fn position(&self) -> Option<u128> {
        self.counter.position()
    }
    fn stream_identity(&self) -> Option<StreamIdentity<'_>> {
        Some(self.identity())
    }

    #[inline]
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        let word = self.next_word();
        self.counter.increment();
        Ok(word)
    }

    /// Returns two successive native words, the first in the low half.
    #[inline]
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        let low = u64::from(self.next_word());
        let high = u64::from(self.next_word());
        self.counter.increment_by(2);
        Ok(low | (high << 32))
    }

    /// Fills `destination` with little-endian native words, discarding the unused high bytes of
    /// the last word.
    fn fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), SciOxideError> {
        crate::util::fill_bytes_with(destination, 4, || Ok(u64::from(self.next_word())))?;
        self.counter = self.counter.advanced(destination.len().div_ceil(4) as u128);
        Ok(())
    }

    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        #[allow(clippy::cast_possible_truncation)]
        super::words::uniform_word(32, M1, || self.next_u32().map(|word| u64::from(word) - 1))
            .map(|word| word as u32)
    }

    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        super::words::uniform_word(64, M1, || self.next_u32().map(|word| u64::from(word) - 1))
    }
}

impl SeedableRng for Mrg32k3a {
    fn from_seed(seed: u64) -> Self {
        let mut splitmix = crate::prng::SplitMix64::from_seed(seed);
        // Reducing 64-bit SplitMix words modulo m1/m2 gives canonical residues;
        // the bias from `%` is below 2^-32 and irrelevant for seeding. Each
        // component must avoid the all-zero triple, which is a fixed point of
        // its recurrence.
        let mut s1 = core::array::from_fn(|_| splitmix.next_word() % M1);
        let mut s2 = core::array::from_fn(|_| splitmix.next_word() % M2);
        if s1 == [0; 3] {
            s1[0] = 1;
        }
        if s2 == [0; 3] {
            s2[0] = 1;
        }
        Self {
            s1,
            s2,
            origin: (s1, s2),
            selectors: (0, 0),
            counter: TransitionCounter::from_position(Some(0)),
        }
    }
}

impl SkipAhead for Mrg32k3a {
    fn skip_ahead(&mut self, transitions: u128) -> Result<(), SciOxideError> {
        self.advance_state(transitions);
        self.counter = self.counter.advanced(transitions);
        Ok(())
    }
    // A normalized 64-bit word takes three digits per candidate and at most 1,024 candidates.
    #[inline]
    fn read_transition_bound(&self, _: crate::traits::private::Token) -> Option<u128> {
        Some(3 * crate::util::MAX_SAMPLING_ATTEMPTS as u128)
    }
}

impl SplittableRng for Mrg32k3a {
    fn split(&self, stream_id: u64) -> Result<Self, SciOxideError> {
        self.substream(stream_id, 0)
    }
    fn substream(&self, stream_id: u64, subsequence_id: u64) -> Result<Self, SciOxideError> {
        // Streams are addressed from the recorded origin, not the current
        // state, so `substream(i, j)` is a pure function of (origin, i, j) no
        // matter how far this handle has advanced. The origin is shared by all
        // derived streams, keeping them on disjoint 2^127 / 2^76 blocks of one
        // sequence.
        let (s1, s2) = jump_state(
            &self.origin.0,
            &self.origin.1,
            &A1P127,
            &A2P127,
            u128::from(stream_id),
        );
        let (s1, s2) = jump_state(&s1, &s2, &A1P76, &A2P76, u128::from(subsequence_id));
        Ok(Self {
            s1,
            s2,
            origin: self.origin,
            selectors: (stream_id, subsequence_id),
            counter: TransitionCounter::from_position(Some(0)),
        })
    }
}

impl Mrg32k3a {
    /// Returns the six state words in the order of [`Self::from_state_words`].
    pub const fn state_words(&self) -> [u64; 6] {
        [
            self.s1[0], self.s1[1], self.s1[2], self.s2[0], self.s2[1], self.s2[2],
        ]
    }
    fn identity(&self) -> StreamIdentity<'static> {
        let (s1, s2) = self.origin;
        StreamIdentity::new(
            Algorithm::Mrg32k3a,
            [s1[0], s1[1], s1[2], s2[0], s2[1], s2[2]],
            Some(self.selectors),
        )
    }
}
impl StateEngine for Mrg32k3a {
    const ALGORITHM: Algorithm = Algorithm::Mrg32k3a;
    const WORDS: usize = 6;
    const HAS_STREAMS: bool = true;
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            identity: self.identity(),
            words: self.state_words(),
            parameter: 0,
            counter: self.counter.position(),
        }
    }
    fn from_snapshot(snapshot: Snapshot) -> Result<Self, SciOxideError> {
        let [a, b, c, d, e, f] = snapshot.words;
        let (s1, s2) = ([a, b, c], [d, e, f]);
        let [a, b, c, d, e, f] = snapshot.identity.origin;
        let origin = ([a, b, c], [d, e, f]);
        Self::validate_imported_components(&s1, &s2)?;
        Self::validate_imported_components(&origin.0, &origin.1)?;
        let selectors = snapshot.identity.selectors.ok_or_else(|| {
            state::malformed(
                "MRG requires stream selectors",
                scioxide_core::argument!("bytes"),
            )
        })?;
        Ok(Self {
            s1,
            s2,
            origin,
            selectors,
            counter: TransitionCounter::from_position(snapshot.counter),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{Rng, SerializableState};

    const REFERENCE_STATE: [u64; 6] = [12_345; 6];

    fn state_words(generator: &Mrg32k3a) -> [u64; 6] {
        [
            generator.s1[0],
            generator.s1[1],
            generator.s1[2],
            generator.s2[0],
            generator.s2[1],
            generator.s2[2],
        ]
    }

    #[test]
    fn checked_state_constructor_enforces_component_invariants() {
        assert_eq!(
            state_words(&Mrg32k3a::from_state_words(REFERENCE_STATE).unwrap()),
            REFERENCE_STATE
        );
        for invalid in [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [M1, 1, 1, 1, 1, 1],
            [1, 1, 1, M2, 1, 1],
        ] {
            let error = Mrg32k3a::from_state_words(invalid).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        }
    }

    #[test]
    fn next_u64_concatenates_two_successive_words_first_low() {
        let mut words = Mrg32k3a::from_seed(42);
        let low = words.next_u32().unwrap();
        let high = words.next_u32().unwrap();
        let mut combined = Mrg32k3a::from_seed(42);
        assert_eq!(
            combined.next_u64().unwrap(),
            u64::from(low) | (u64::from(high) << 32)
        );
        assert_ne!(low, 0);
        assert_ne!(high, 0);
        assert_eq!(combined.position(), Some(2));
    }

    #[test]
    fn zero_difference_uses_m1_as_the_native_word() {
        let mut generator = Mrg32k3a::from_state_words([0, 0, 1, 0, 1, 0]).unwrap();
        assert_eq!(generator.next_u32().unwrap(), u32::try_from(M1).unwrap());
    }

    #[test]
    fn native_float_paths_consume_one_word() {
        let mut words = Mrg32k3a::from_seed(123);
        let word = words.next_u32().unwrap();
        let mut f64_generator = Mrg32k3a::from_seed(123);
        let f64_value = <_ as Rng<f64>>::next_float(&mut f64_generator).unwrap();
        assert_eq!(f64_value, f64::from(word) * NORM);
        assert_eq!(f64_generator.position(), Some(1));

        let mut f32_generator = Mrg32k3a::from_seed(123);
        let f32_value = <_ as Rng<f32>>::next_float(&mut f32_generator).unwrap();
        assert_eq!(f32_value, crate::decode_u32::<f32>(word));
        assert_eq!(f32_generator.position(), Some(1));
    }

    #[test]
    fn maximum_native_word_remains_below_one_for_f32() {
        let mut generator = Mrg32k3a::from_state_words([0, 0, 1, 0, 1, 0]).unwrap();
        let value = <_ as Rng<f32>>::next_float(&mut generator).unwrap();
        assert_eq!(value, 1.0_f32.next_down());
    }

    #[test]
    fn fill_bytes_uses_little_endian_words_and_discards_partial_tail() {
        let mut words = Mrg32k3a::from_seed(7);
        let expected = [
            words.next_u32().unwrap().to_le_bytes(),
            words.next_u32().unwrap().to_le_bytes(),
        ]
        .concat();

        let mut generator = Mrg32k3a::from_seed(7);
        let mut bytes = [0_u8; 7];
        generator.fill_bytes(&mut bytes).unwrap();
        assert_eq!(bytes, expected[..7]);
        assert_eq!(generator.position(), Some(2));

        let mut whole = Mrg32k3a::from_seed(7);
        let mut together = [0_u8; 4];
        whole.fill_bytes(&mut together).unwrap();
        let mut split = Mrg32k3a::from_seed(7);
        let mut partial = [0_u8; 4];
        split.fill_bytes(&mut partial[..1]).unwrap();
        split.fill_bytes(&mut partial[1..]).unwrap();
        assert_ne!(partial, together);
        assert_eq!(split.position(), Some(2));
    }

    #[test]
    fn iteration_and_skip_reach_identical_states() {
        for transitions in [0_u128, 1, 2, 50, 1_000] {
            let mut iterated = Mrg32k3a::from_seed(99);
            (0..transitions).for_each(|_| {
                let _ = iterated.next_u32().unwrap();
            });
            let mut skipped = Mrg32k3a::from_seed(99);
            skipped.skip_ahead(transitions).unwrap();
            assert_eq!(state_words(&iterated), state_words(&skipped));
            assert_eq!(iterated.position(), skipped.position());
        }
    }

    #[test]
    fn position_overflow_becomes_untracked_without_stopping_the_stream() {
        let mut generator = Mrg32k3a::from_seed(7);
        generator.counter = TransitionCounter::from_position(Some(u128::MAX));
        let mut expected = generator.clone();
        expected.advance_state(2);
        generator.skip_ahead(1).unwrap();
        generator.skip_ahead(1).unwrap();
        assert_eq!(state_words(&generator), state_words(&expected));
        assert_eq!(generator.position(), None);
        assert!(generator.skip_to(0).is_err());
        assert_eq!(
            Mrg32k3a::import_state(&generator.export_state().unwrap())
                .unwrap()
                .position(),
            None
        );
    }

    #[test]
    fn split_and_substream_use_immutable_origin_and_reset_local_position() {
        let mut current = Mrg32k3a::from_seed(42);
        current.skip_ahead(17).unwrap();
        let split_zero = current.split(0).unwrap();
        assert_eq!(
            state_words(&split_zero),
            state_words(&Mrg32k3a::from_seed(42))
        );
        assert_eq!(split_zero.position(), Some(0));
        let substream_zero = current.substream(0, 0).unwrap();
        assert_eq!(
            state_words(&substream_zero),
            state_words(&Mrg32k3a::from_seed(42))
        );
        assert_eq!(substream_zero.position(), Some(0));

        let direct = current.split(2).unwrap();
        let composed = current.split(1).unwrap().split(2).unwrap();
        assert_eq!(state_words(&direct), state_words(&composed));

        let sub_direct = current.substream(1, 2).unwrap();
        let sub_composed = current.split(1).unwrap().substream(1, 2).unwrap();
        assert_eq!(state_words(&sub_direct), state_words(&sub_composed));
        assert_eq!(sub_direct.position(), Some(0));
    }

    #[test]
    fn large_absolute_stream_ids_ignore_prior_selection() {
        let base = Mrg32k3a::from_seed(42);
        let left = u64::MAX / 2;
        let direct = base.split(u64::MAX).unwrap();
        let composed = base.split(left).unwrap().split(u64::MAX).unwrap();
        assert_eq!(state_words(&direct), state_words(&composed));

        let direct = base.substream(u64::MAX, u64::MAX).unwrap();
        let composed = base
            .substream(left, left)
            .unwrap()
            .substream(u64::MAX, u64::MAX)
            .unwrap();
        assert_eq!(state_words(&direct), state_words(&composed));
    }

    #[test]
    fn integer_state_roundtrip_preserves_words_and_counter() {
        let mut generator = Mrg32k3a::from_seed(42);
        let _ = generator.next_u64().unwrap();
        let encoded = generator.export_state().unwrap();
        assert_eq!(&encoded[..4], b"GRN2");
        let restored = Mrg32k3a::import_state(&encoded).unwrap();
        assert_eq!(state_words(&restored), state_words(&generator));
        assert_eq!(restored.position(), Some(2));
    }

    #[test]
    fn modular_identity_is_stable() {
        let identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        assert_eq!(advance_component([1, 2, 3], identity, 100, M1), [1, 2, 3]);
    }
}
