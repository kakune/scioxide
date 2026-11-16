//! Lüscher's subtract-with-borrow generators with luxury levels, in exact
//! 24- and 48-bit integer arithmetic.

use super::classic::{ClassicState, classic_engine, dyadic_float};
use crate::SeedableRng;
use scioxide_core::{InvalidInputReason, SciOxideError};

macro_rules! luxury {
    ($(#[$attr:meta])* $name:ident, $($variant:ident = $value:literal),+) => {
        $(#[$attr])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(u8)]
        pub enum $name { $(#[doc = concat!("Luxury level ", stringify!($value), ".")] $variant = $value),+ }
        impl TryFrom<u8> for $name {
            type Error = SciOxideError;
            /// Converts a numeric luxury level.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `value` is not a level of this generator.
            fn try_from(value: u8) -> Result<Self, SciOxideError> {
                match value {
                    $($value => Ok(Self::$variant),)+
                    _ => Err(SciOxideError::invalid_input(

                        scioxide_core::argument!("value"),
                        InvalidInputReason::OutOfRange,
                        "unsupported luxury level",
                    )),
                }
            }
        }
    };
}
luxury!(
    /// Luxury level `0` to `4` of [`Ranlux`](crate::Ranlux).
    ///
    /// Level `k` delivers 24 outputs and discards the rest of every block of `p` recurrence steps,
    /// with `p` = 24, 48, 97, 223 or 389 for levels 0 to 4. Higher levels remove more of the
    /// correlations of the raw recurrence and cost proportionally more per output.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::RanluxLuxury;
    /// assert_eq!(RanluxLuxury::try_from(4)?, RanluxLuxury::Level4);
    /// assert!(RanluxLuxury::try_from(5).is_err());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    RanluxLuxury,
    Level0 = 0,
    Level1 = 1,
    Level2 = 2,
    Level3 = 3,
    Level4 = 4
);
luxury!(
    /// Luxury level `0` to `2` of [`Ranlxs`](crate::Ranlxs).
    ///
    /// Each refill of twelve 48-bit words takes 109, 202 or 397 recurrence steps for levels 0 to 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::RanlxsLuxury;
    /// assert_eq!(RanlxsLuxury::try_from(0)?, RanlxsLuxury::Level0);
    /// assert!(RanlxsLuxury::try_from(3).is_err());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    RanlxsLuxury,
    Level0 = 0,
    Level1 = 1,
    Level2 = 2
);
luxury!(
    /// Luxury level `1` or `2` of [`Ranlxd`](crate::Ranlxd).
    ///
    /// Each refill of twelve 48-bit words takes 202 or 397 recurrence steps for levels 1 and 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::RanlxdLuxury;
    /// assert_eq!(RanlxdLuxury::try_from(2)?, RanlxdLuxury::Level2);
    /// assert!(RanlxdLuxury::try_from(0).is_err());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    RanlxdLuxury,
    Level1 = 1,
    Level2 = 2
);

classic_engine!(
    /// RANLUX subtract-with-borrow generator with 24-bit output and period about 10¹⁷¹.
    ///
    /// The recurrence is
    ///
    /// ```text
    /// xₙ = (xₙ₋₁₀ − xₙ₋₂₄ − cₙ₋₁) mod 2²⁴,
    /// ```
    ///
    /// where the borrow `cₙ` is 1 if the subtraction went below zero and 0 otherwise (Lüscher
    /// 1994). The [`RanluxLuxury`](crate::RanluxLuxury) level sets how many recurrence values are
    /// discarded, so an output costs `O(p/24)` amortized for `p` steps per block.
    /// [`from_seed`](crate::SeedableRng::from_seed) uses level 3. The native output is the 24-bit
    /// value `xₙ`, returned by [`next_u32`](crate::BitGenerator::next_u32). Samplers combine
    /// several native words into each uniform word. Drawing never fails.
    ///
    /// # References
    ///
    /// - M. Lüscher, "A portable high-quality random number generator for lattice field theory
    ///   simulations", *Comput. Phys. Commun.* 79(1), 1994, 100–110.
    /// - F. James, "RANLUX: a Fortran implementation of the high-quality pseudorandom number
    ///   generator of Lüscher", *Comput. Phys. Commun.* 79(1), 1994, 111–114.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, Ranlux, RanluxLuxury, SeedableRng, SerializableState};
    /// // `from_seed` selects luxury level 3.
    /// let mut rng = Ranlux::with_luxury(7, RanluxLuxury::Level3);
    /// let checkpoint = rng.export_state()?;
    /// let native = rng.next_u32()?; // one 24-bit native word
    /// assert!(native < 1 << 24);
    /// assert_eq!(Ranlux::from_seed(7).next_u32()?, native);
    /// assert_eq!(Ranlux::import_state(&checkpoint)?.next_u32()?, native);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Ranlux,
    28,
    24
);
classic_engine!(
    /// Second-generation RANLUX generator RANLXS with 24-bit output.
    ///
    /// The underlying recurrence works on 48-bit words,
    ///
    /// ```text
    /// xₙ = (xₙ₋₅ − xₙ₋₁₂ − cₙ₋₁) mod 2⁴⁸,
    /// ```
    ///
    /// with borrow `cₙ` as in [`Ranlux`](crate::Ranlux). Each refill runs `p` recurrence steps
    /// set by the [`RanlxsLuxury`](crate::RanlxsLuxury) level and keeps the last twelve words.
    /// Each 48-bit word gives two 24-bit outputs, low half first, so an output costs `O(p/24)`
    /// amortized. [`from_seed`](crate::SeedableRng::from_seed) uses level 1. Drawing never fails.
    ///
    /// # References
    ///
    /// M. Lüscher, "A portable high-quality random number generator for lattice field theory
    /// simulations", *Comput. Phys. Commun.* 79(1), 1994, 100–110.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, Ranlxs, RanlxsLuxury, SeedableRng, SerializableState};
    /// // `from_seed` selects luxury level 1.
    /// let mut rng = Ranlxs::with_luxury(7, RanlxsLuxury::Level1);
    /// let checkpoint = rng.export_state()?;
    /// let native = rng.next_u32()?; // one 24-bit native word
    /// assert!(native < 1 << 24);
    /// assert_eq!(Ranlxs::from_seed(7).next_u32()?, native);
    /// assert_eq!(Ranlxs::import_state(&checkpoint)?.next_u32()?, native);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Ranlxs,
    17,
    24
);
classic_engine!(
    /// Second-generation RANLUX generator RANLXD with 48-bit output.
    ///
    /// The recurrence and refill are those of [`Ranlxs`](crate::Ranlxs), with the
    /// [`RanlxdLuxury`](crate::RanlxdLuxury) level setting the steps per refill. Each 48-bit word
    /// is one output, returned by [`next_u64`](crate::BitGenerator::next_u64), so an output costs
    /// `O(p/12)` amortized. [`from_seed`](crate::SeedableRng::from_seed) uses level 1. Drawing
    /// never fails.
    ///
    /// # References
    ///
    /// M. Lüscher, "A portable high-quality random number generator for lattice field theory
    /// simulations", *Comput. Phys. Commun.* 79(1), 1994, 100–110.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, Ranlxd, RanlxdLuxury, SeedableRng, SerializableState};
    /// // `from_seed` selects luxury level 1.
    /// let mut rng = Ranlxd::with_luxury(7, RanlxdLuxury::Level1);
    /// let checkpoint = rng.export_state()?;
    /// let native = rng.next_u64()?; // one 48-bit native word
    /// assert!(native < 1 << 48);
    /// assert_eq!(Ranlxd::from_seed(7).next_u64()?, native);
    /// assert_eq!(Ranlxd::import_state(&checkpoint)?.next_u64()?, native);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Ranlxd,
    17,
    48
);
dyadic_float!(Ranlux, 24);
dyadic_float!(Ranlxs, 24);
dyadic_float!(Ranlxd, 48);

const MASK24: u64 = (1 << 24) - 1;
const MASK48: u64 = (1 << 48) - 1;
const CLASSIC_STEPS: [usize; 5] = [24, 48, 97, 223, 389];
const SECOND_STEPS: [usize; 3] = [109, 202, 397];

// Pure fixed-point subtraction returns the new word and borrow bit.
// Subtract-with-borrow x_n = x_{n-s} - x_{n-r} - c_{n-1} mod 2^b is equivalent
// to Lüscher's floating recurrence scaled by 2^b, so integer state reproduces
// the reference outputs exactly.
fn subtract(left: u64, right: u64, carry: u64, mask: u64) -> (u64, u64) {
    (
        left.wrapping_sub(right).wrapping_sub(carry) & mask,
        u64::from(left < right + carry),
    )
}
impl SeedableRng for Ranlux {
    fn from_seed(seed: u64) -> Self {
        Self::with_luxury(seed, RanluxLuxury::Level3)
    }
}
impl Ranlux {
    /// Creates a generator at the given luxury level with James's seeding.
    ///
    /// The seed is reduced modulo 2147483563, and zero becomes 314159265. The register is filled
    /// from the congruential generator `x ← 40014 x mod 2147483563` (James 1994).
    pub fn with_luxury(seed: u64, luxury: RanluxLuxury) -> Self {
        let mut value = seed % 2_147_483_563;
        if value == 0 {
            value = 314_159_265;
        }
        let mut words = [0; 28];
        for word in &mut words[..24] {
            value = 40014 * value % 2_147_483_563;
            *word = value & MASK24;
        }
        // Layout: words 0..24 are the lag-24 register, 24 is the borrow, 25
        // the register cursor, 26 the count of outputs in the current block of
        // 24, and 27 the luxury level. The initial borrow follows James's
        // seeder (set when the last word is zero).
        words[24] = u64::from(words[23] == 0);
        words[25] = 23;
        words[27] = luxury as u64;
        Self(ClassicState::new(words))
    }
    // A subtract-with-borrow generator has two absorbing states: all zero with
    // no borrow, and all ones (2^b - 1) with borrow. Both are rejected.
    fn valid(w: &[u64; 28]) -> bool {
        w[..24].iter().all(|&v| v <= MASK24)
            && (w[24] != 0 || w[..24].iter().any(|&v| v != 0))
            && !(w[24] == 1 && w[..24].iter().all(|&v| v == MASK24))
            && w[24] <= 1
            && w[25] < 24
            && w[26] < 24
            && w[27] < 5
    }
    fn consistent(origin: &[u64; 28], current: &[u64; 28]) -> bool {
        origin[25] == 23
            && origin[26] == 0
            && origin[24] == u64::from(origin[23] == 0)
            && origin[27] == current[27]
    }
    fn step(&mut self) -> u64 {
        let w = &mut self.0.words;
        #[allow(clippy::cast_possible_truncation)]
        let i = w[25] as usize;
        let (value, carry) = subtract(w[(i + 10) % 24], w[i], w[24], MASK24);
        w[i] = value;
        w[24] = carry;
        w[25] = ((i + 23) % 24) as u64;
        value
    }
    #[inline]
    pub(super) fn next_native(&mut self) -> u64 {
        let value = self.step();
        self.0.words[26] += 1;
        // Lüscher's luxury decimation: after each 24 delivered values, discard
        // the next p - 24 recurrence values. Discarding breaks the long-range
        // correlations of the raw SWB recurrence; higher levels discard more.
        if self.0.words[26] == 24 {
            self.0.words[26] = 0;
            #[allow(clippy::cast_possible_truncation)]
            let steps = CLASSIC_STEPS[self.0.words[27] as usize];
            for _ in 24..steps {
                self.step();
            }
        }
        self.0.counter.increment();
        value
    }
}

fn second_seed(seed: u64, double: bool, level: u64) -> [u64; 17] {
    // Second-generation seeding fills twelve 48-bit words from a 31-bit
    // xor-shift register (tap 18), matching Lüscher's ranlxs/ranlxd reference.
    // The double-precision variant inverts every bit, as in the reference.
    let value = (seed & 0x7fff_ffff).max(1);
    let mut bits: [u64; 31] = core::array::from_fn(|i| (value >> i) & 1);
    let mut i = 0;
    let mut words = [0; 17];
    for word in &mut words[..12] {
        for _ in 0..48 {
            *word = (*word << 1) | (bits[i] ^ u64::from(double));
            bits[i] ^= bits[(i + 18) % 31];
            i = (i + 1) % 31;
        }
    }
    // Layout: 0..12 register, 12 borrow, 13 register cursor, 14 output cursor
    // (in 48-bit words for ranlxd, 24-bit halves for ranlxs), 15 the output
    // cursor value at which the next refill is due, 16 luxury level. Placing
    // the output cursor one before the refill point forces a refill on the
    // first draw.
    words[14] = if double { 11 } else { 23 };
    words[16] = level;
    words
}
fn valid_second(w: &[u64; 17], double: bool) -> bool {
    w[..12].iter().all(|&v| v <= MASK48)
        && (w[12] != 0 || w[..12].iter().any(|&v| v != 0))
        && !(w[12] == 1 && w[..12].iter().all(|&v| v == MASK48))
        && w[12] <= 1
        && w[13] < 12
        && w[14] < if double { 12 } else { 24 }
        && w[15] == w[13] * if double { 1 } else { 2 }
        && w[16] <= 2
        && (!double || w[16] >= 1)
}
fn consistent_second(origin: &[u64; 17], current: &[u64; 17], double: bool) -> bool {
    origin[12] == 0
        && origin[13] == 0
        && origin[14] == if double { 11 } else { 23 }
        && origin[15] == 0
        && origin[16] == current[16]
}
fn next_second(w: &mut [u64; 17], double: bool) -> u64 {
    let width = if double { 12 } else { 24 };
    w[14] = (w[14] + 1) % width;
    // The output cursor wrapping around to the refill point means all twelve
    // words produced by the previous refill have been delivered.
    if w[14] == w[15] {
        #[allow(clippy::cast_possible_truncation)]
        let steps = SECOND_STEPS[w[16] as usize];
        for _ in 0..steps {
            #[allow(clippy::cast_possible_truncation)]
            let i = w[13] as usize;
            let (value, carry) = subtract(w[(i + 7) % 12], w[i], w[12], MASK48);
            w[i] = value;
            w[12] = carry;
            w[13] = ((i + 1) % 12) as u64;
        }
        w[15] = w[13] * if double { 1 } else { 2 };
        w[14] = w[15];
    }
    #[allow(clippy::cast_possible_truncation)]
    let i = w[14] as usize;
    if double {
        w[i]
    // RANLXS splits every 48-bit word into two 24-bit outputs, low half first.
    } else if i.is_multiple_of(2) {
        w[i / 2] & MASK24
    } else {
        w[i / 2] >> 24
    }
}
impl SeedableRng for Ranlxs {
    fn from_seed(seed: u64) -> Self {
        Self::with_luxury(seed, RanlxsLuxury::Level1)
    }
}
impl Ranlxs {
    /// Creates a generator at the given luxury level from the low 31 bits of the seed.
    ///
    /// A zero seed becomes 1. The register is filled bit by bit from a 31-bit shift register.
    pub fn with_luxury(seed: u64, luxury: RanlxsLuxury) -> Self {
        Self(ClassicState::new(second_seed(seed, false, luxury as u64)))
    }
    fn valid(w: &[u64; 17]) -> bool {
        valid_second(w, false)
    }
    fn consistent(origin: &[u64; 17], current: &[u64; 17]) -> bool {
        consistent_second(origin, current, false)
    }
    #[inline]
    pub(super) fn next_native(&mut self) -> u64 {
        let value = next_second(&mut self.0.words, false);
        self.0.counter.increment();
        value
    }
}
impl SeedableRng for Ranlxd {
    fn from_seed(seed: u64) -> Self {
        Self::with_luxury(seed, RanlxdLuxury::Level1)
    }
}
impl Ranlxd {
    /// Creates a generator at the given luxury level from the low 31 bits of the seed.
    ///
    /// A zero seed becomes 1. The register is filled bit by bit from a 31-bit shift register.
    pub fn with_luxury(seed: u64, luxury: RanlxdLuxury) -> Self {
        Self(ClassicState::new(second_seed(seed, true, luxury as u64)))
    }
    fn valid(w: &[u64; 17]) -> bool {
        valid_second(w, true)
    }
    fn consistent(origin: &[u64; 17], current: &[u64; 17]) -> bool {
        consistent_second(origin, current, true)
    }
    #[inline]
    pub(super) fn next_native(&mut self) -> u64 {
        let value = next_second(&mut self.0.words, true);
        self.0.counter.increment();
        value
    }
}
