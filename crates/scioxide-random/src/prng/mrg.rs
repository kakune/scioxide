//! L'Ecuyer's fifth-order multiple recursive generator (1993) and combined
//! multiple recursive generator (1996).

use super::classic::{ClassicState, classic_engine};
use crate::SeedableRng;

classic_engine!(
    /// Fifth-order multiple recursive generator modulo `2³¹ − 1` with period about 2¹⁵⁵.
    ///
    /// The recurrence is
    ///
    /// ```text
    /// xₙ = (107374182 xₙ₋₁ + 104480 xₙ₋₅) mod (2³¹ − 1),
    /// ```
    ///
    /// from L'Ecuyer, Blouin and Couture (1993). Each output costs O(1). The native output is the
    /// residue `xₙ` in `[0, 2³¹ − 1)`, returned by [`next_u32`](crate::BitGenerator::next_u32).
    /// [`next_uniform_u32`](crate::BitGenerator::next_uniform_u32) and the samplers build uniform
    /// words from several residues. An `f64` uniform is `xₙ / (2³¹ − 1)`. This is a different
    /// generator from [`Mrg32k3a`](crate::Mrg32k3a).
    ///
    /// [`from_seed`](crate::SeedableRng::from_seed) uses the low 32 bits of the seed (zero becomes
    /// 1), expands them with the congruential generator `x ← 69069 x mod 2³²`, and discards six
    /// outputs. Native draws never fail; uniform-word normalization can exhaust its retry limit.
    ///
    /// # References
    ///
    /// P. L'Ecuyer, F. Blouin and R. Couture, "A search for good multiple recursive random number
    /// generators", *ACM Trans. Model. Comput. Simul.* 3(2), 1993, 87–98.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, Mrg, SeedableRng, SerializableState};
    /// let mut rng = Mrg::from_seed(1);
    /// let checkpoint = rng.export_state()?;
    /// let native = rng.next_u32()?; // a residue below the modulus 2^31 - 1
    /// assert!(native < 2_147_483_647);
    /// let mut replay = Mrg::import_state(&checkpoint)?;
    /// assert_eq!(replay.next_u32()?, native);
    /// let _uniform: u32 = replay.next_uniform_u32()?; // full-width word built from residues
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Mrg,
    5,
    32,
    u64,
    M1
);
classic_engine!(
    /// Combined multiple recursive generator with two order-3 components and period about 2¹⁸⁵.
    ///
    /// The components are
    ///
    /// ```text
    /// xₙ = (63308 xₙ₋₂ − 183326 xₙ₋₃) mod m₁,    m₁ = 2³¹ − 1,
    /// yₙ = (86098 yₙ₋₁ − 539608 yₙ₋₃) mod m₂,    m₂ = 2145483479,
    /// ```
    ///
    /// and the output is `zₙ = (xₙ − yₙ) mod m₁` (L'Ecuyer 1996). Each output costs O(1). The
    /// native output is the residue `zₙ` in `[0, m₁)`, returned by
    /// [`next_u32`](crate::BitGenerator::next_u32). An `f64` uniform is `zₙ / m₁`. This is a
    /// different generator from [`Mrg32k3a`](crate::Mrg32k3a).
    ///
    /// [`from_seed`](crate::SeedableRng::from_seed) uses the low 32 bits of the seed (zero becomes
    /// 1), expands them with the congruential generator `x ← 69069 x mod 2³²`, and discards seven
    /// outputs. Native draws never fail; uniform-word normalization can exhaust its retry limit.
    ///
    /// # References
    ///
    /// P. L'Ecuyer, "Combined multiple recursive random number generators", *Operations Research*
    /// 44(5), 1996, 816–822.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, Cmrg, SeedableRng, SerializableState};
    /// let mut rng = Cmrg::from_seed(1);
    /// let checkpoint = rng.export_state()?;
    /// let native = rng.next_u32()?; // a residue below the modulus 2^31 - 1
    /// assert!(native < 2_147_483_647);
    /// let mut replay = Cmrg::import_state(&checkpoint)?;
    /// assert_eq!(replay.next_u32()?, native);
    /// assert_eq!(replay.position(), Some(1));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    Cmrg,
    6,
    32,
    u64,
    M1
);
const M1: u64 = 2_147_483_647;
const M2: u64 = 2_145_483_479;
fn seed_words<const N: usize>(seed: u64, moduli: [u64; N]) -> [u64; N] {
    // LCG-69069 expansion. `% m` can yield zero words; an
    // all-zero component is still practically impossible and `component_valid`
    // rejects it on import.
    let mut value = (seed & 0xffff_ffff).max(1);
    moduli.map(|m| {
        value = (69069 * value) & 0xffff_ffff;
        value % m
    })
}
fn component_valid(words: &[u64], modulus: u64) -> bool {
    words.iter().all(|&w| w < modulus) && words.iter().any(|&w| w != 0)
}
impl SeedableRng for Mrg {
    fn from_seed(seed: u64) -> Self {
        let mut engine = Self(ClassicState::new(seed_words(seed, [M1; 5])));
        for _ in 0..6 {
            engine.next_native();
        }
        Self(ClassicState::new(engine.0.words))
    }
}
impl Mrg {
    fn valid(w: &[u64; 5]) -> bool {
        component_valid(w, M1)
    }
    fn consistent(_: &[u64; 5], _: &[u64; 5]) -> bool {
        true
    }
    #[inline]
    pub(super) fn next_native(&mut self) -> u64 {
        let w = &mut self.0.words;
        // x_n = (a1 x_{n-1} + a5 x_{n-5}) mod (2^31 - 1), with the newest word
        // kept at index 0. Both products are below 2^58, so the sum is exact.
        let value = (107_374_182 * w[0] + 104_480 * w[4]) % M1;
        w.copy_within(..4, 1);
        w[0] = value;
        self.0.counter.increment();
        value
    }
}
impl SeedableRng for Cmrg {
    fn from_seed(seed: u64) -> Self {
        let mut engine = Self(ClassicState::new(seed_words(
            seed,
            [M1, M1, M1, M2, M2, M2],
        )));
        for _ in 0..7 {
            engine.next_native();
        }
        Self(ClassicState::new(engine.0.words))
    }
}
impl Cmrg {
    fn valid(w: &[u64; 6]) -> bool {
        component_valid(&w[..3], M1) && component_valid(&w[3..], M2)
    }
    fn consistent(_: &[u64; 6], _: &[u64; 6]) -> bool {
        true
    }
    #[inline]
    pub(super) fn next_native(&mut self) -> u64 {
        let w = &mut self.0.words;
        // Two order-3 components with negative coefficients. Each product is
        // reduced before the subtraction and `+ m` keeps the difference
        // non-negative, so unsigned arithmetic represents the signed recurrence.
        let x = (63308 * w[1] % M1 + M1 - 183326 * w[2] % M1) % M1;
        let y = (86098 * w[3] % M2 + M2 - 539608 * w[5] % M2) % M2;
        w.copy_within(0..2, 1);
        w.copy_within(3..5, 4);
        w[0] = x;
        w[3] = y;
        self.0.counter.increment();
        (x + M1 - y) % M1
    }
}
macro_rules! modular_float {
    ($name:ident) => {
        impl crate::float::FloatSource<f64> for $name {
            #[inline]
            fn draw_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> {
                #[allow(clippy::cast_precision_loss)]
                let value = self.next_native() as f64;
                // Native values lie in [0, m1), so dividing by m1 stays below 1.
                Ok(value / 2_147_483_647.0)
            }
        }
        impl crate::float::FloatSource<f32> for $name {
            #[inline]
            fn draw_float(&mut self) -> Result<f32, scioxide_core::SciOxideError> {
                // Truncate to 24 fraction bits; rounding must never produce 1.
                let bits = (self.next_native() << 24) / M1;
                Ok(crate::decode_u64(bits << 40))
            }
        }
    };
}
modular_float!(Mrg);
modular_float!(Cmrg);
