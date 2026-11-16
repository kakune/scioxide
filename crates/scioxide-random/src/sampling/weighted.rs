//! Weighted sampling with the alias method.

use alloc::vec::Vec;

use scioxide_core::backend::{try_filled, try_with_capacity};
#[cfg(test)]
use scioxide_core::{InvalidInputReason, NonFiniteOrigin};
use scioxide_core::{RealScalar, SciOxideError};

use super::uniform_index;
use crate::traits::Rng;
use crate::util::validated_unit_float;
use crate::workspace::SamplingWorkspace;

/// Draws index `i` with probability `wᵢ / Σⱼ wⱼ` from fixed weights `w`, by the alias method.
///
/// Construction builds a probability table and an alias table in O(n) time and
/// O(n) storage. Each draw takes expected O(1) time. [`Self::sample_index`]
/// returns one index, [`Self::sample`] the element of a slice at it,
/// [`Self::sample_n`] a new `Vec` of indices, and [`Self::sample_n_into`]
/// fills a slice you provide. The sampler is immutable and stores only the
/// tables, not the values you sample from.
///
/// # References
///
/// M. D. Vose, "A linear algorithm for generating random numbers with a given
/// distribution", *IEEE Trans. Softw. Eng.* 17(9), 1991, 972–975.
///
/// ```
/// use scioxide_random::{SeedableRng, SplitMix64, WeightedSampler};
/// let sampler = WeightedSampler::new(&[0.0, 1.0, 3.0])?;
/// let mut rng = SplitMix64::from_seed(7);
/// let draws = sampler.sample_n(&mut rng, 1_000)?;
/// assert!(draws.iter().all(|&i| i != 0)); // zero weight is never drawn
/// let common = draws.iter().filter(|&&i| i == 2).count() as f64 / 1_000.0;
/// assert!((common - 0.75).abs() < 0.06); // 4.4 standard errors
/// assert_ne!(*sampler.sample(&mut rng, &["never", "rare", "common"])?, "never");
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct WeightedSampler<S: RealScalar> {
    probabilities: Vec<S>,
    aliases: Vec<usize>,
}

impl<S: RealScalar> WeightedSampler<S> {
    /// Creates a sampler from nonnegative finite `weights`, which need not sum to one.
    ///
    /// The weights are scaled by their largest positive value before a
    /// compensated sum is formed. This permits finite inputs such as repeated
    /// `f32::MAX` values whose direct sum would overflow.
    ///
    /// # Errors
    ///
    /// - [`scioxide_core::SciOxideErrorKind::InvalidInput`] if `weights` is empty, contains a
    ///   negative value, or has no positive entry.
    /// - [`scioxide_core::SciOxideErrorKind::NonFinite`] if an input or derived table value is not
    ///   finite.
    /// - [`scioxide_core::SciOxideErrorKind::ScalarConversion`] if the table length cannot be
    ///   represented exactly by `S`.
    /// - [`scioxide_core::SciOxideErrorKind::AllocationFailure`] if table or worklist storage
    ///   cannot be reserved.
    #[inline]
    pub fn new(weights: &[S]) -> Result<Self, SciOxideError> {
        Self::prepare(weights).map(|(sampler, _, _)| sampler)
    }

    /// Alias tables and log weights share validation and one normalization.
    pub(crate) fn with_log_weights(weights: &[S]) -> Result<(Self, Vec<S>), SciOxideError> {
        let (sampler, maximum, total) = Self::prepare(weights)?;
        let mut logs =
            try_with_capacity(weights.len(), scioxide_core::quantity!("mixture_weights"))?;
        let log_total = total.ln();
        logs.extend(weights.iter().map(|&weight| {
            let scaled = weight / maximum;
            // Preserve positive masses even when their normalized ratio underflows.
            let log = if scaled > S::ZERO {
                scaled.ln()
            } else {
                weight.ln() - maximum.ln()
            };
            log - log_total
        }));
        Ok((sampler, logs))
    }

    fn prepare(weights: &[S]) -> Result<(Self, S, S), SciOxideError> {
        let (maximum, positive_fallback) = validate_weights(weights)?;
        let count = scalar_count::<S>(weights.len())?;
        let total = compensated_scaled_total(weights, maximum)?;
        let (mut probabilities, mut aliases) = allocate_tables(weights.len())?;

        // Scaled masses n w_i / sum(w) average exactly 1, the invariant Vose's
        // alias construction balances around. Dividing every weight by the
        // maximum first keeps the sum finite for weights near the scalar's
        // maximum and avoids underflow when all weights are tiny.
        weights
            .iter()
            .zip(probabilities.iter_mut())
            .try_for_each(|(&weight, probability)| {
                let mass = (weight / maximum) * count / total;
                if !mass.is_finite() {
                    return Err(SciOxideError::non_finite(
                        scioxide_core::NonFiniteOrigin::Intermediate,
                        "a scaled alias-table mass is non-finite",
                    ));
                }
                *probability = mass;
                Ok(())
            })?;

        build_alias_tables(weights, positive_fallback, &mut probabilities, &mut aliases)?;

        Ok((
            Self {
                probabilities,
                aliases,
            },
            maximum,
            total,
        ))
    }

    /// Returns the number of weights, which is at least one.
    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        self.probabilities.len()
    }

    /// Draws one index.
    ///
    /// A draw chooses a column with [`uniform_index`](crate::uniform_index) and
    /// then reads one float from [`Rng::next_float`](crate::Rng::next_float) to
    /// decide between the column and its alias. A one-element sampler still
    /// makes both calls and returns zero.
    ///
    /// # Errors
    ///
    /// - The errors of [`uniform_index`](crate::uniform_index).
    /// - [`scioxide_core::SciOxideErrorKind::NonFinite`] if the RNG provider returns NaN or
    ///   infinity.
    /// - [`scioxide_core::SciOxideErrorKind::InvalidInput`] if the RNG provider returns a finite
    ///   draw outside `[0, 1)`.
    #[inline]
    pub fn sample_index<R: Rng<S>>(&self, rng: &mut R) -> Result<usize, SciOxideError> {
        // O(1) alias draw: pick a column uniformly, then keep it with its
        // retained probability or jump to its alias.
        let column = uniform_index(self.len(), rng)?;
        let uniform = validated_unit_float(rng, scioxide_core::argument!("rng"))?;
        Ok(if uniform < self.probabilities[column] {
            column
        } else {
            self.aliases[column]
        })
    }

    /// Draws one index and returns the element of `values` at it.
    ///
    /// The length of `values` is checked before the generator advances.
    ///
    /// # Errors
    ///
    /// - [`scioxide_core::SciOxideErrorKind::DimensionMismatch`] if `values.len() != self.len()`.
    /// - The errors of [`Self::sample_index`].
    pub fn sample<'a, T, R: Rng<S>>(
        &self,
        rng: &mut R,
        values: &'a [T],
    ) -> Result<&'a T, SciOxideError> {
        if values.len() != self.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.len()),
                scioxide_core::Dimensions::one(values.len()),
            ));
        }
        let index = self.sample_index(rng)?;
        Ok(&values[index])
    }

    /// Returns `n` independent indices in a new `Vec`.
    ///
    /// The indices equal `n` calls of [`Self::sample_index`]. The result is
    /// allocated once, before the first draw. On error nothing is returned and
    /// the generator stays advanced by the words it consumed; to restore it,
    /// use [`Self::sample_n_into`] with
    /// [`SamplingPolicy::rollback`](crate::SamplingPolicy::rollback).
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the output cannot be allocated.
    /// - The errors of [`Self::sample_index`].
    pub fn sample_n<R: Rng<S>>(&self, rng: &mut R, n: usize) -> Result<Vec<usize>, SciOxideError> {
        crate::workspace::sample_owned(rng, n, 0, |rng| self.sample_index(rng))
    }

    /// Fills `output` with independent indices.
    ///
    /// An empty output does not advance the generator. `policy` chooses what
    /// is written and where the generator stands on error
    /// ([batch sampling](crate#batch-sampling)). All-or-nothing output needs a
    /// workspace of `output.len()` indices; incremental output needs none. On
    /// success the indices equal those of [`Self::sample_n`].
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the workspace cannot grow.
    /// - The errors of [`Self::sample_index`].
    pub fn sample_n_into<R: Rng<S>>(
        &self,
        rng: &mut R,
        policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
        output: &mut [usize],
        workspace: &mut SamplingWorkspace<usize>,
    ) -> Result<(), SciOxideError> {
        crate::workspace::sample_scalars(rng, policy, output, 0, workspace, usize::MAX, |rng| {
            self.sample_index(rng)
        })
    }
}

fn validate_weights<S: RealScalar>(weights: &[S]) -> Result<(S, usize), SciOxideError> {
    if weights.is_empty() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::Empty,
            "WeightedSampler requires at least one weight",
        ));
    }

    let summary = weights.iter().copied().enumerate().try_fold(
        None,
        |maximum: Option<(S, usize)>, (index, weight)| {
            if !weight.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    "weights contain a non-finite value",
                ));
            }
            if weight < S::ZERO {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("weights"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "WeightedSampler requires non-negative weights",
                ));
            }
            Ok(
                if weight > S::ZERO && maximum.is_none_or(|(current, _)| weight > current) {
                    Some((weight, index))
                } else {
                    maximum
                },
            )
        },
    )?;

    summary.ok_or_else(|| {
        SciOxideError::invalid_input(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::Empty,
            "WeightedSampler requires at least one positive weight",
        )
    })
}

fn scalar_count<S: RealScalar>(count: usize) -> Result<S, SciOxideError> {
    let scalar = S::try_from_usize_exact(count)?;
    if !scalar.is_finite() || scalar <= S::ZERO {
        return Err(SciOxideError::scalar_conversion(
            scioxide_core::ConversionType::Usize,
            <S as scioxide_core::backend::Scalar>::kind().into(),
            scioxide_core::ScalarConversionReason::OutOfRange,
        ));
    }
    Ok(scalar)
}

fn compensated_scaled_total<S: RealScalar>(weights: &[S], maximum: S) -> Result<S, SciOxideError> {
    // Neumaier's variant of Kahan summation: the correction term captures the
    // rounding error of each addition regardless of which operand is larger,
    // so many small weights next to a large one still count.
    let (sum, correction) = weights.iter().fold((S::ZERO, S::ZERO), |state, &weight| {
        crate::util::neumaier(state, weight / maximum)
    });
    let total = sum + correction;
    if !total.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "the normalized weight sum is non-finite",
        ));
    }
    if total <= S::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "positive normalized weights produced a zero total",
        ));
    }
    Ok(total)
}

fn allocate_tables<S: RealScalar>(len: usize) -> Result<(Vec<S>, Vec<usize>), SciOxideError> {
    let probabilities = try_filled(
        len,
        S::ZERO,
        scioxide_core::quantity!("weighted_sampler_probabilities"),
    )?;
    let mut aliases = try_with_capacity(len, scioxide_core::quantity!("weighted_sampler_aliases"))?;
    aliases.extend(0..len);
    Ok((probabilities, aliases))
}

fn build_alias_tables<S: RealScalar>(
    weights: &[S],
    positive_fallback: usize,
    probabilities: &mut [S],
    aliases: &mut [usize],
) -> Result<(), SciOxideError> {
    let len = probabilities.len();
    let mut small = try_with_capacity(
        len,
        scioxide_core::quantity!("weighted_sampler_small_worklist"),
    )?;
    let mut large = try_with_capacity(
        len,
        scioxide_core::quantity!("weighted_sampler_large_worklist"),
    )?;
    let one = S::ONE;

    probabilities
        .iter()
        .copied()
        .enumerate()
        .for_each(|(index, mass)| {
            if mass < one {
                small.push(index);
            } else {
                large.push(index);
            }
        });

    // Vose's alias method: repeatedly pair an under-full column (mass < 1) with
    // an over-full one, fill the small column's remainder with the large
    // index, and move the large column's leftover mass back into the
    // appropriate worklist. Each step finalizes one column.
    while !small.is_empty() && !large.is_empty() {
        let small_index = small[small.len() - 1];
        small.truncate(small.len() - 1);
        let large_index = large[large.len() - 1];
        large.truncate(large.len() - 1);

        let small_mass = probabilities[small_index];
        probabilities[small_index] = small_mass.max(S::ZERO).min(one);
        aliases[small_index] = large_index;

        let remaining = probabilities[large_index] + small_mass - one;
        if !remaining.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Intermediate,
                "alias-table balancing produced a non-finite mass",
            ));
        }
        probabilities[large_index] = remaining.max(S::ZERO);
        if remaining < one {
            small.push(large_index);
        } else {
            large.push(large_index);
        }
    }

    // Columns left in either list differ from 1 only by accumulated rounding;
    // they are treated as exactly full so the table stays a valid partition.
    small.into_iter().chain(large).for_each(|index| {
        probabilities[index] = one;
        aliases[index] = index;
    });

    // Rounding may have left a sliver of probability on zero-weight columns
    // (or made them their own alias); force them to always redirect to a
    // positive-weight index so impossible outcomes can never be returned.

    weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, weight)| *weight == S::ZERO)
        .for_each(|(index, _)| {
            probabilities[index] = S::ZERO;
            if weights[aliases[index]] == S::ZERO {
                aliases[index] = positive_fallback;
            }
        });

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::{BitGenerator, SeedableRng};

    #[derive(Debug)]
    struct ScriptedRng {
        integers: Vec<u64>,
        floats: Vec<f64>,
        integer_index: usize,
        float_index: usize,
    }

    impl ScriptedRng {
        fn new(integers: Vec<u64>, floats: Vec<f64>) -> Self {
            Self {
                integers,
                floats,
                integer_index: 0,
                float_index: 0,
            }
        }
    }

    impl BitGenerator for ScriptedRng {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            Ok(self.next_u64().unwrap() as u32)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            let value = self.integers.get(self.integer_index).copied().unwrap_or(0);
            self.integer_index += 1;
            Ok(value)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.chunks_mut(8).for_each(|chunk| {
                let bytes = self.next_u64().unwrap().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            });
            Ok(())
        }
    }

    impl Rng<f64> for ScriptedRng {
        fn next_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> {
            let value = self.floats.get(self.float_index).copied().unwrap_or(0.0);
            self.float_index += 1;
            Ok(value)
        }
    }

    #[test]
    fn rejects_invalid_weight_inputs() {
        let error = WeightedSampler::<f64>::new(&[]).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        let error = WeightedSampler::new(&[0.0_f64, 0.0]).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        let error = WeightedSampler::new(&[1.0_f64, -1.0]).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
            .into_iter()
            .for_each(|invalid| {
                let error = WeightedSampler::new(&[1.0_f64, invalid]).unwrap_err();
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NonFinite {
                        origin: scioxide_core::NonFiniteOrigin::Input,
                        ..
                    }
                ));
            });
    }

    #[test]
    fn constructs_repeated_maximum_and_extreme_scale_tables() {
        assert_eq!(
            WeightedSampler::new(&[f32::MAX, f32::MAX]).unwrap().len(),
            2
        );
        assert_eq!(
            WeightedSampler::new(&[f64::MIN_POSITIVE, f64::MAX, 0.0])
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn singleton_has_fixed_rng_consumption() {
        let sampler = WeightedSampler::new(&[7.0_f64]).unwrap();
        let mut rng = ScriptedRng::new(vec![u64::MAX], vec![1.0 - f64::EPSILON]);

        assert_eq!(sampler.sample_index(&mut rng).unwrap(), 0);
        assert_eq!(rng.integer_index, 1);
        assert_eq!(rng.float_index, 1);
    }

    #[test]
    fn alias_decision_honors_uniform_endpoints() {
        let sampler = WeightedSampler::new(&[1.0_f64, 3.0]).unwrap();
        let mut lower = ScriptedRng::new(vec![0], vec![0.0]);
        let mut upper = ScriptedRng::new(vec![0], vec![1.0 - f64::EPSILON]);

        assert_eq!(sampler.sample_index(&mut lower).unwrap(), 0);
        assert_eq!(sampler.sample_index(&mut upper).unwrap(), 1);
    }

    #[test]
    fn malformed_provider_uniform_never_selects_an_alias() {
        let sampler = WeightedSampler::new(&[1.0_f64, 3.0]).unwrap();

        let mut non_finite = ScriptedRng::new(vec![0], vec![f64::NAN]);
        let error = sampler.sample_index(&mut non_finite).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        ));

        let mut out_of_range = ScriptedRng::new(vec![0], vec![1.0]);
        let error = sampler.sample_index(&mut out_of_range).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::OutOfRange,
                ..
            }
        ));
    }

    #[test]
    fn integer_column_rejection_retries_before_float_draw() {
        let sampler = WeightedSampler::new(&[1.0_f64, 1.0, 1.0]).unwrap();
        let mut rng = ScriptedRng::new(vec![0, 1], vec![0.0]);

        assert_eq!(sampler.sample_index(&mut rng).unwrap(), 0);
        assert_eq!(rng.integer_index, 2);
        assert_eq!(rng.float_index, 1);
    }

    #[test]
    fn zero_weights_are_never_returned() {
        let sampler = WeightedSampler::new(&[0.0_f64, 2.0, 0.0, 3.0]).unwrap();
        let mut rng = SplitMix64::from_seed(11);
        (0..10_000).for_each(|_| {
            let index = sampler.sample_index(&mut rng).unwrap();
            assert!(index == 1 || index == 3);
        });
    }

    #[test]
    fn value_mismatch_is_transactional() {
        let sampler = WeightedSampler::new(&[1.0_f64, 2.0]).unwrap();
        let mut rng = SplitMix64::from_seed(13);
        let position = rng.position();

        let error = sampler.sample(&mut rng, &["only one"]).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(rng.position(), position);
    }

    #[test]
    fn empty_bulk_output_does_not_advance_rng() {
        let sampler = WeightedSampler::new(&[1.0_f64, 2.0]).unwrap();
        let mut rng = SplitMix64::from_seed(17);
        let position = rng.position();

        let mut workspace = SamplingWorkspace::new();
        sampler
            .sample_n_into(
                &mut rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut [],
                &mut workspace,
            )
            .unwrap();
        assert_eq!(rng.position(), position);
    }

    #[test]
    fn scalar_value_and_bulk_paths_cover_multiple_values() {
        let sampler = WeightedSampler::new(&[1.0_f64, 2.0, 3.0]).unwrap();
        let values = ["a", "b", "c"];
        let mut scalar_rng = SplitMix64::from_seed(19);
        assert!(values.contains(sampler.sample(&mut scalar_rng, &values).unwrap()));

        let mut output = [usize::MAX; 64];
        let mut bulk_rng = SplitMix64::from_seed(23);
        let mut workspace = SamplingWorkspace::new();
        sampler
            .sample_n_into(
                &mut bulk_rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut output,
                &mut workspace,
            )
            .unwrap();
        assert!(output.into_iter().all(|index| index < sampler.len()));
    }

    #[test]
    fn bulk_provider_failure_preserves_output() {
        let sampler = WeightedSampler::new(&[1.0_f64, 3.0]).unwrap();
        let mut rng = ScriptedRng::new(vec![0, 0], vec![0.0, f64::NAN]);
        let mut output = [usize::MAX; 2];
        let mut workspace = SamplingWorkspace::new();

        let error = sampler
            .sample_n_into(
                &mut rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut output,
                &mut workspace,
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Callback,
                ..
            }
        ));
        assert_eq!(output, [usize::MAX; 2]);
    }

    #[test]
    fn same_seed_replays_the_same_sequence() {
        let sampler = WeightedSampler::new(&[1.0_f64, 4.0, 2.0, 8.0]).unwrap();
        let mut left = SplitMix64::from_seed(29);
        let mut right = SplitMix64::from_seed(29);

        let left_sequence = (0..128)
            .map(|_| sampler.sample_index(&mut left).unwrap())
            .collect::<Vec<_>>();
        let right_sequence = (0..128)
            .map(|_| sampler.sample_index(&mut right).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(left_sequence, right_sequence);
    }

    #[test]
    fn allocation_failure_is_structured() {
        let error = allocate_tables::<f64>(usize::MAX).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::AllocationFailure
        );
    }

    #[test]
    fn sampler_is_send_and_sync() {
        const fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<WeightedSampler<f32>>();
        assert_send_sync::<WeightedSampler<f64>>();
    }

    #[test]
    fn test_weighted_basic() {
        let items = ["a", "b", "c"];
        let weights = [0.0_f64, 1.0, 0.0];
        let mut rng = SplitMix64::from_seed(42);
        // With weight concentrated on "b", it should always be "b"
        for _ in 0..100 {
            let choice = WeightedSampler::new(&weights)
                .and_then(|sampler| sampler.sample(&mut rng, &items))
                .unwrap();
            assert_eq!(*choice, "b");
        }
    }

    #[test]
    fn test_weighted_dimension_mismatch() {
        let items = [1, 2, 3];
        let weights = [0.5_f64, 0.5];
        let mut rng = SplitMix64::from_seed(42);
        let error = WeightedSampler::new(&weights)
            .and_then(|sampler| sampler.sample(&mut rng, &items))
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
    }

    #[test]
    fn test_weighted_all_zero() {
        let items = [1, 2];
        let weights = [0.0_f64, 0.0];
        let mut rng = SplitMix64::from_seed(42);
        let error = WeightedSampler::new(&weights)
            .and_then(|sampler| sampler.sample(&mut rng, &items))
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_weighted_negative() {
        let items = [1, 2];
        let weights = [1.0_f64, -1.0];
        let mut rng = SplitMix64::from_seed(42);
        let error = WeightedSampler::new(&weights)
            .and_then(|sampler| sampler.sample(&mut rng, &items))
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_weighted_nan_errors() {
        let items = [1, 2];
        let weights = [1.0_f64, f64::NAN];
        let mut rng = SplitMix64::from_seed(42);
        let error = WeightedSampler::new(&weights)
            .and_then(|sampler| sampler.sample(&mut rng, &items))
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_weighted_infinity_errors() {
        let items = [1, 2];
        let weights = [1.0_f64, f64::INFINITY];
        let mut rng = SplitMix64::from_seed(42);
        let error = WeightedSampler::new(&weights)
            .and_then(|sampler| sampler.sample(&mut rng, &items))
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));

        let weights_neg = [1.0_f64, f64::NEG_INFINITY];
        let error = WeightedSampler::new(&weights_neg)
            .and_then(|sampler| sampler.sample(&mut rng, &items))
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn weighted_sampling_accepts_large_finite_weights() {
        let items = [1, 2];
        let sampler = WeightedSampler::new(&[f32::MAX, f32::MAX]).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..100 {
            assert!(items.contains(sampler.sample(&mut rng, &items).unwrap()));
        }
    }

    #[test]
    fn weighted_sampling_rejects_malformed_providers() {
        let items = [1, 2];
        let weights = [1.0_f64, 1.0];

        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut weighted_rng = ScriptedRng::new(vec![0], vec![value]);
            let error = WeightedSampler::new(&weights)
                .and_then(|sampler| sampler.sample(&mut weighted_rng, &items))
                .unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Callback,
                    ..
                }
            ));
        }

        for value in [-f64::MIN_POSITIVE, 1.0, 1.0 + f64::EPSILON] {
            let mut weighted_rng = ScriptedRng::new(vec![0], vec![value]);
            let error = WeightedSampler::new(&weights)
                .and_then(|sampler| sampler.sample(&mut weighted_rng, &items))
                .unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: scioxide_core::InvalidInputReason::OutOfRange,
                    ..
                }
            ));
        }
    }
}
