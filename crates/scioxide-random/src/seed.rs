//! Seed construction and deterministic seed expansion utilities.
//!
//! [`crate::SeedableRng::seed_from_entropy`] obtains a seed from an entropy source you pass,
//! and [`seed_sequence`] expands one seed into several.

use core::mem::size_of;

use alloc::vec::Vec;

use scioxide_core::{Capability, EntropySource, SciOxideError};

use crate::util::{SPLITMIX64_GAMMA, splitmix64_mix};
/// Reads a 64-bit seed from `entropy`.
///
/// # Errors
///
/// - `CapabilityFailure` if the source cannot supply the seed. The crate never
///   falls back to another source of entropy.
pub(crate) fn entropy_seed(source: &dyn EntropySource) -> Result<u64, SciOxideError> {
    let mut bytes = [0_u8; size_of::<u64>()];
    source.fill_bytes(&mut bytes).map_err(|error| {
        SciOxideError::capability_failure(Capability::Entropy, error.into_detail())
    })?;
    Ok(u64::from_le_bytes(bytes))
}

/// Returns `n` seeds derived from `seed`: the first `n` outputs of [`SplitMix64`](crate::SplitMix64) seeded with `seed`.
///
/// The seeds are reproducible, pairwise distinct, and carry no more entropy than `seed`. Use them
/// to seed several generators from one master seed.
///
/// # Errors
///
/// - `InvalidInput` if `n` is zero.
/// - `AllocationFailure` if the result cannot be allocated.
///
/// ```
/// use scioxide_random::{BitGenerator, Pcg64, SeedableRng, SplitMix64, seed_sequence};
/// let seeds = seed_sequence(42, 3)?;
/// let workers: Vec<Pcg64> = seeds.iter().map(|&seed| Pcg64::from_seed(seed)).collect();
/// assert_eq!(workers.len(), 3);
/// // The derived seeds are the SplitMix64 output stream for the same seed.
/// assert_eq!(seeds[0], SplitMix64::from_seed(42).next_u64()?);
/// assert!(seed_sequence(42, 0).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn seed_sequence(seed: u64, n: usize) -> Result<Vec<u64>, SciOxideError> {
    if n == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("n"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "seed_sequence requires n > 0",
        ));
    }

    // This is exactly the SplitMix64 output sequence for `seed`: a Weyl
    // increment followed by the bijective finalizer. Distinct indices therefore
    // yield distinct seeds for a fixed input (within the 2^64 period).
    let mut state = seed;
    let mut seeds =
        scioxide_core::backend::try_with_capacity(n, scioxide_core::quantity!("seed_sequence"))?;
    seeds.extend((0..n).map(|_| {
        state = state.wrapping_add(SPLITMIX64_GAMMA);
        splitmix64_mix(state)
    }));

    Ok(seeds)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{BitGenerator, SeedableRng};

    struct FixedEntropy(Result<[u8; 8], &'static str>);

    impl EntropySource for FixedEntropy {
        fn fill_bytes(&self, destination: &mut [u8]) -> Result<(), scioxide_core::CapabilityError> {
            let bytes = self.0.map_err(scioxide_core::CapabilityError::new)?;
            destination.copy_from_slice(&bytes);
            Ok(())
        }
    }

    #[test]
    fn explicit_entropy_is_reproducible_and_failure_is_typed() {
        assert_eq!(
            crate::SplitMix64::seed_from_entropy(&FixedEntropy(Ok(42_u64.to_le_bytes())))
                .unwrap()
                .next_u64()
                .unwrap(),
            crate::SplitMix64::from_seed(42).next_u64().unwrap()
        );
        let error =
            crate::SplitMix64::seed_from_entropy(&FixedEntropy(Err("offline"))).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::CapabilityFailure {
                capability: Capability::Entropy,
                ..
            }
        ));
    }

    #[test]
    fn test_seed_sequence_deterministic() {
        let seq1 = seed_sequence(42, 5).unwrap();
        let seq2 = seed_sequence(42, 5).unwrap();
        assert_eq!(seq1, seq2);
    }

    #[test]
    fn test_seed_sequence_known_answer() {
        assert_eq!(
            seed_sequence(42, 5).unwrap(),
            vec![
                0xbdd7_3226_2feb_6e95,
                0x28ef_e333_b266_f103,
                0x4752_6757_130f_9f52,
                0x581c_e1ff_0e4a_e394,
                0x09bc_585a_2448_23f2,
            ]
        );
    }

    #[test]
    fn test_seed_sequence_length() {
        let seq = seed_sequence(0, 10).unwrap();
        assert_eq!(seq.len(), 10);
    }

    #[test]
    fn test_seed_sequence_all_different() {
        let mut sorted = seed_sequence(123, 100).unwrap();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 100);
    }

    #[test]
    fn test_seed_sequence_zero_n_errors() {
        let error = seed_sequence(0, 0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_seed_sequence_reports_allocation_failure() {
        let error = seed_sequence(0, usize::MAX).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::AllocationFailure
        );
    }

    #[test]
    fn test_seed_sequence_different_seeds() {
        let seq1 = seed_sequence(1, 5).unwrap();
        let seq2 = seed_sequence(2, 5).unwrap();
        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_splitmix64_mix_avalanche() {
        // Two nearby inputs should produce very different outputs
        let a = super::splitmix64_mix(0);
        let b = super::splitmix64_mix(1);
        // At least half the bits should differ (avalanche property)
        let diff_bits = (a ^ b).count_ones();
        assert!(
            diff_bits >= 16,
            "poor avalanche: only {diff_bits} bits differ"
        );
    }
}
