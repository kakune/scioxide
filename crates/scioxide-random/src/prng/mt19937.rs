//! MT19937 with Matsumoto and Nishimura's 2002 initialization.

use super::classic::{ClassicState, classic_engine, dyadic_float};
use crate::SeedableRng;
use scioxide_core::{InvalidInputReason, SciOxideError};

classic_engine!(
    /// Mersenne Twister MT19937 with 32-bit output and period 2¹⁹⁹³⁷ − 1.
    ///
    /// MT19937 is a twisted generalized feedback shift register on 624 words of 32 bits, with
    /// tempered output (Matsumoto and Nishimura 1998). Every 624 outputs the generator
    /// regenerates its whole state in O(624) work, so an output costs O(1) amortized.
    ///
    /// [`from_seed`](crate::SeedableRng::from_seed) uses the low 32 bits of the seed with the
    /// authors' 2002 `init_genrand` initialization, so seed 5489 gives the reference sequence.
    /// [`Mt19937::from_seed_words`] uses their `init_by_array` initialization. Drawing never
    /// fails.
    ///
    /// # References
    ///
    /// M. Matsumoto and T. Nishimura, "Mersenne twister: a 623-dimensionally equidistributed
    /// uniform pseudo-random number generator", *ACM Trans. Model. Comput. Simul.* 8(1), 1998,
    /// 3–30.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, Mt19937, SeedableRng, SerializableState};
    /// // Outputs of the authors' reference code for the scalar seed 5489 and an array key.
    /// let mut rng = Mt19937::from_seed(5489);
    /// assert_eq!(rng.next_u32()?, 3_499_211_612);
    /// let checkpoint = rng.export_state()?;
    /// assert_eq!(rng.next_u32()?, 581_869_302);
    /// assert_eq!(Mt19937::import_state(&checkpoint)?.next_u32()?, 581_869_302);
    /// let mut keyed = Mt19937::from_seed_words(&[0x123, 0x234, 0x345, 0x456])?;
    /// assert_eq!(keyed.next_u32()?, 1_067_595_299);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Mt19937,
    625,
    32
);
dyadic_float!(Mt19937, 32);
impl SeedableRng for Mt19937 {
    fn from_seed(seed: u64) -> Self {
        let mut words = [0; 625];
        words[0] = seed & 0xffff_ffff;
        for i in 1..624 {
            let prev = words[i - 1];
            words[i] = (1_812_433_253 * (prev ^ (prev >> 30)) + i as u64) & 0xffff_ffff;
        }
        // Word 624 is the output cursor. Setting it to 624 marks the buffer as
        // consumed so the first draw performs the twist, as in mt19937ar.
        words[624] = 624;
        Self(ClassicState::new(words))
    }
}
impl Mt19937 {
    /// Creates a generator from a key of 32-bit words with the authors' `init_by_array`
    /// initialization (2002).
    ///
    /// The key words are used in order. `[0]` is a valid one-word array key;
    /// array initialization differs from [`SeedableRng::from_seed`] with seed zero.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `key` is empty.
    pub fn from_seed_words(key: &[u32]) -> Result<Self, SciOxideError> {
        if key.is_empty() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("key"),
                InvalidInputReason::Empty,
                "MT19937 array seed must be nonempty",
            ));
        }
        let mut engine = Self::from_seed(19_650_218);
        let words = &mut engine.0.words;
        let (mut i, mut j) = (1, 0);
        for _ in 0..624.max(key.len()) {
            let prev = words[i - 1];
            words[i] = ((words[i] ^ ((prev ^ (prev >> 30)) * 1_664_525))
                .wrapping_add(u64::from(key[j]))
                .wrapping_add(j as u64))
                & 0xffff_ffff;
            i += 1;
            j += 1;
            if i == 624 {
                words[0] = words[623];
                i = 1;
            }
            if j == key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let prev = words[i - 1];
            words[i] = (words[i] ^ ((prev ^ (prev >> 30)) * 1_566_083_941)).wrapping_sub(i as u64)
                & 0xffff_ffff;
            i += 1;
            if i == 624 {
                words[0] = words[623];
                i = 1;
            }
        }
        // mt19937ar forces the MSB of word 0 so the state is never all zero in
        // the 19937 significant bits. The origin is replaced because the array
        // seeder, not `from_seed(19650218)`, defines the root of this stream.
        words[0] = 0x8000_0000;
        engine.0.origin = *words;
        Ok(engine)
    }
    fn valid(words: &[u64; 625]) -> bool {
        // Only the top bit of word 0 enters the recurrence, so a state whose
        // other 19937 significant bits are zero is the absorbing zero state.
        words[624] <= 624
            && words[..624].iter().all(|&w| w <= u64::from(u32::MAX))
            && (words[0] & 0x8000_0000 != 0 || words[1..624].iter().any(|&w| w != 0))
    }
    fn consistent(origin: &[u64; 625], _: &[u64; 625]) -> bool {
        origin[624] == 624
    }
    #[inline]
    pub(super) fn next_native(&mut self) -> u64 {
        if self.0.words[624] == 624 {
            self.twist();
        }
        let words = &mut self.0.words;
        #[allow(clippy::cast_possible_truncation)]
        let mut value = words[words[624] as usize];
        words[624] += 1;
        // Tempering improves equidistribution of the output bits; it is
        // invertible and does not feed back into the state.
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        self.0.counter.increment();
        value
    }

    /// Regenerate all 624 words in place and rewind the cursor.
    ///
    /// Word i combines the top bit of x_i with the low 31 bits of x_{i+1} and
    /// XORs x_{i+397}. Indices wrapping past 623 read words already updated in
    /// this pass, which is exactly the recurrence x_{k+624} = f(x_k, x_{k+1},
    /// x_{k+397}). The pass is split at the two wrap points, as in mt19937ar,
    /// so no index needs a remainder.
    #[inline(never)]
    fn twist(&mut self) {
        const N: usize = 624;
        const M: usize = 397;
        fn mix(upper: u64, lower: u64, far: u64) -> u64 {
            let pair = (upper & 0x8000_0000) | (lower & 0x7fff_ffff);
            far ^ (pair >> 1) ^ if pair & 1 == 0 { 0 } else { 0x9908_b0df }
        }
        let words = &mut self.0.words;
        for i in 0..N - M {
            words[i] = mix(words[i], words[i + 1], words[i + M]);
        }
        for i in N - M..N - 1 {
            words[i] = mix(words[i], words[i + 1], words[i + M - N]);
        }
        words[N - 1] = mix(words[N - 1], words[0], words[M - 1]);
        words[N] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitGenerator;
    #[test]
    fn published_scalar_and_array_vectors() {
        let mut scalar = Mt19937::from_seed(5489);
        assert_eq!(
            core::array::from_fn::<_, 10, _>(|_| scalar.next_u32().unwrap()),
            [
                3499211612, 581869302, 3890346734, 3586334585, 545404204, 4161255391, 3922919429,
                949333985, 2715962298, 1323567403
            ]
        );
        let mut array = Mt19937::from_seed_words(&[0x123, 0x234, 0x345, 0x456]).unwrap();
        assert_eq!(
            core::array::from_fn::<_, 10, _>(|_| array.next_u32().unwrap()),
            [
                1067595299, 955945823, 477289528, 4107218783, 4228976476, 3344332714, 3355579695,
                227628506, 810200273, 2591290167
            ]
        );
        assert!(Mt19937::from_seed_words(&[]).is_err());
    }
}
