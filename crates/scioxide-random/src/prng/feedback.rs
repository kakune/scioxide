//! Combined Tausworthe generators (L'Ecuyer 1996, 1999) and Ziff's four-tap
//! generalized feedback shift register (1998).

use super::classic::{ClassicState, classic_engine, dyadic_float};
use crate::SeedableRng;

classic_engine!(
    /// Three-component combined Tausworthe generator with 32-bit output and period about 2⁸⁸.
    ///
    /// Each component is a linear feedback shift register on a 32-bit word, and the output is the
    /// XOR of the three registers. The parameters are those of L'Ecuyer (1996) with the corrected
    /// seeding of L'Ecuyer (1999). Each output costs O(1), and the state is three words.
    ///
    /// [`from_seed`](crate::SeedableRng::from_seed) uses the low 32 bits of the seed (zero becomes
    /// 1), expands them with the congruential generator `x ← 69069 x mod 2³²`, raises any register
    /// below its minimum, and discards six outputs. Drawing never fails.
    ///
    /// # References
    ///
    /// - P. L'Ecuyer, "Maximally equidistributed combined Tausworthe generators", *Math. Comp.*
    ///   65(213), 1996, 203–213.
    /// - P. L'Ecuyer, "Tables of maximally equidistributed combined LFSR generators", *Math. Comp.*
    ///   68(225), 1999, 261–269.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, SeedableRng, SerializableState, Taus2};
    /// let mut rng = Taus2::from_seed(1);
    /// let checkpoint = rng.export_state()?;
    /// let word = rng.next_u32()?;
    /// let mut replay = Taus2::import_state(&checkpoint)?;
    /// assert_eq!(replay.next_u32()?, word);
    /// assert_eq!(replay.position(), Some(1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Taus2,
    3,
    32
);
classic_engine!(
    /// Four-component combined Tausworthe generator with 32-bit output and period about 2¹¹³.
    ///
    /// Each component is a linear feedback shift register on a 32-bit word, and the output is the
    /// XOR of the four registers, with the parameters of L'Ecuyer (1999). Each output costs O(1),
    /// and the state is four words. [`next_u64`](crate::BitGenerator::next_u64) joins two outputs,
    /// the first in the low half.
    ///
    /// [`from_seed`](crate::SeedableRng::from_seed) uses the low 32 bits of the seed (zero becomes
    /// 1), expands them with the congruential generator `x ← 69069 x mod 2³²`, raises any register
    /// below its minimum, and discards ten outputs. Drawing never fails.
    ///
    /// # References
    ///
    /// P. L'Ecuyer, "Tables of maximally equidistributed combined LFSR generators", *Math. Comp.*
    /// 68(225), 1999, 261–269.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, SeedableRng, SerializableState, Taus113};
    /// let mut rng = Taus113::from_seed(1);
    /// let word = rng.next_u64()?; // two 32-bit native words, first in the low half
    /// assert_eq!(rng.position(), Some(2));
    /// let mut buffer = [0_u8; Taus113::STATE_BYTES];
    /// rng.export_state_into(&mut buffer)?;
    /// let expected = rng.next_u32()?;
    /// assert_eq!(Taus113::import_state(&buffer)?.next_u32()?, expected);
    /// assert_eq!(Taus113::from_seed(1).next_u64()?, word);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Taus113,
    4,
    32
);
classic_engine!(
    /// Four-tap generalized feedback shift register with 32-bit output and period 2⁹⁶⁸⁹ − 1.
    ///
    /// The recurrence is
    ///
    /// ```text
    /// xₙ = xₙ₋₄₇₁ ⊕ xₙ₋₁₅₈₆ ⊕ xₙ₋₆₉₈₈ ⊕ xₙ₋₉₆₈₉,
    /// ```
    ///
    /// where `⊕` is bitwise XOR on 32-bit words (Ziff 1998). Each output costs O(1). The register
    /// holds 9,689 words, and the generator also stores its starting register for checkpoints.
    ///
    /// [`from_seed`](crate::SeedableRng::from_seed) uses the low 32 bits of the seed (zero becomes
    /// 4357) and fills the register bit by bit from the congruential generator
    /// `x ← 69069 x mod 2³²`, forcing a set of linearly independent words. Seeding costs about
    /// 5 × 10⁵ generator steps. Drawing never fails.
    ///
    /// # References
    ///
    /// R. M. Ziff, "Four-tap shift-register-sequence random-number generators", *Computers in
    /// Physics* 12(4), 1998, 385–392.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, Gfsr4, SeedableRng, SerializableState};
    /// let mut rng = Gfsr4::from_seed(4357);
    /// let checkpoint = rng.export_state()?;
    /// let words = [rng.next_u32()?, rng.next_u32()?];
    /// let mut replay = Gfsr4::import_state(&checkpoint)?;
    /// assert_eq!([replay.next_u32()?, replay.next_u32()?], words);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Gfsr4,
    9690,
    32,
    u32
);
dyadic_float!(Taus2, 32);
dyadic_float!(Taus113, 32);
dyadic_float!(Gfsr4, 32);

const MASK: u64 = 0xffff_ffff;
// One step of a 32-bit Tausworthe component in L'Ecuyer's shift form. `clear`
// masks the low bits that do not belong to the component's k-bit register;
// words are kept in u64 so the left shifts can be masked back to 32 bits.
fn lfsr(value: u64, a: u32, b: u32, c: u32, clear: u64) -> u64 {
    (((value & (MASK ^ clear)) << c) & MASK) ^ ((((value << a) & MASK) ^ value) >> b)
}
fn taus_seed<const N: usize>(seed: u64, minima: [u64; N]) -> [u64; N] {
    let mut value = seed & MASK;
    if value == 0 {
        value = 1;
    }
    // Each component register must have a nonzero value above the bits removed
    // by its `clear` mask (s1 > 1, s2 > 7, s3 > 15, s4 > 127), otherwise the
    // component degenerates to zero. The LCG-69069 expansion uses the "+ min"
    // repair to satisfy these lower bounds.
    minima.map(|min| {
        value = (69069 * value) & MASK;
        if value < min {
            value += min;
        }
        value
    })
}
fn valid_taus<const N: usize>(words: &[u64; N], minima: [u64; N]) -> bool {
    words
        .iter()
        .zip(minima)
        .all(|(&w, min)| w >= min && w <= MASK)
}
impl SeedableRng for Taus2 {
    fn from_seed(seed: u64) -> Self {
        let mut engine = Self(ClassicState::new(taus_seed(seed, [2, 8, 16])));
        // Warm-up steps from the reference seeder decorrelate the LCG-seeded
        // registers; the warmed state then becomes the root at position zero.
        for _ in 0..6 {
            engine.next_native();
        }
        Self(ClassicState::new(engine.0.words))
    }
}
impl Taus2 {
    fn valid(words: &[u64; 3]) -> bool {
        valid_taus(words, [2, 8, 16])
    }
    fn consistent(_: &[u64; 3], _: &[u64; 3]) -> bool {
        true
    }
    #[inline]
    pub(super) fn next_native(&mut self) -> u64 {
        let [a, b, c] = self.0.words;
        self.0.words = [
            lfsr(a, 13, 19, 12, 1),
            lfsr(b, 2, 25, 4, 7),
            lfsr(c, 3, 11, 17, 15),
        ];
        self.0.counter.increment();
        self.0.words.iter().fold(0, |a, &b| a ^ b)
    }
}
impl SeedableRng for Taus113 {
    fn from_seed(seed: u64) -> Self {
        let mut engine = Self(ClassicState::new(taus_seed(seed, [2, 8, 16, 128])));
        for _ in 0..10 {
            engine.next_native();
        }
        Self(ClassicState::new(engine.0.words))
    }
}
impl Taus113 {
    fn valid(words: &[u64; 4]) -> bool {
        valid_taus(words, [2, 8, 16, 128])
    }
    fn consistent(_: &[u64; 4], _: &[u64; 4]) -> bool {
        true
    }
    #[inline]
    pub(super) fn next_native(&mut self) -> u64 {
        let [a, b, c, d] = self.0.words;
        self.0.words = [
            lfsr(a, 6, 13, 18, 1),
            lfsr(b, 2, 27, 2, 7),
            lfsr(c, 13, 21, 7, 15),
            lfsr(d, 3, 12, 13, 127),
        ];
        self.0.counter.increment();
        self.0.words.iter().fold(0, |a, &b| a ^ b)
    }
}
impl SeedableRng for Gfsr4 {
    fn from_seed(seed: u64) -> Self {
        let mut value = seed & MASK;
        if value == 0 {
            value = 4357;
        }
        let mut words = [0_u32; 9690];
        // The compatibility seeder emits 16384 words and starts at cursor 32.
        // Retain only its active history, in chronological order (6728..16383,
        // then 0..32). Earlier cells can never affect a subsequent output.
        for original in 0_usize..16384 {
            let mut word = 0_u32;
            for _ in 0..32 {
                value = (69069 * value) & MASK;
                word = (word << 1) | u32::from(value >> 31 != 0);
            }
            if (7..=100).contains(&original) && (original - 7).is_multiple_of(3) {
                let bit = (original - 7) / 3;
                word = (word & (u32::MAX >> bit)) | (0x8000_0000 >> bit);
            }
            let index = (original + 16384 - 6728) % 16384;
            if index < 9689 {
                words[index] = word;
            }
        }
        // Word 9689 is the ring cursor (index of the most recent output), not
        // recurrence state; 9688 makes the next write land on slot 0.
        words[9689] = 9688;
        Self(ClassicState::new(words))
    }
}
impl Gfsr4 {
    fn valid(words: &[u32; 9690]) -> bool {
        words[9689] < 9689 && words[..9689].iter().any(|&w| w != 0)
    }
    fn consistent(origin: &[u32; 9690], _: &[u32; 9690]) -> bool {
        origin[9689] == 9688
    }
    #[inline]
    pub(super) fn next_native(&mut self) -> u64 {
        // x_n = x_{n-471} ^ x_{n-1586} ^ x_{n-6988} ^ x_{n-9689} over a ring of
        // exactly 9689 slots. The oldest lag equals the ring length, so the
        // slot being overwritten is read (as lag 9689) before it is replaced.
        const RING: usize = 9689;
        // Ring arithmetic by one conditional subtraction instead of a remainder.
        let wrap = |slot: usize| if slot >= RING { slot - RING } else { slot };
        let index = wrap(self.0.words[RING] as usize + 1);
        let words = &self.0.words;
        let value = words[index]
            ^ words[wrap(index + RING - 471)]
            ^ words[wrap(index + RING - 1586)]
            ^ words[wrap(index + RING - 6988)];
        self.0.words[index] = value;
        #[allow(clippy::cast_possible_truncation)]
        {
            self.0.words[9689] = index as u32;
        }
        self.0.counter.increment();
        u64::from(value)
    }
}
