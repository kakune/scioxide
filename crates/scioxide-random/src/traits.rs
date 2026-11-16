//! Traits for generators, probability distributions and quasi-random sequences.
//!
//! - **RNG core**: [`BitGenerator`], [`Rng`], [`SeedableRng`]
//! - **Advanced RNG**: [`SkipAhead`], [`JumpableRng`], [`SplittableRng`], [`SerializableState`]
//! - **Sampling**: [`Distribution`], [`CountDistribution`]
//! - **Continuous capabilities**: [`Density`], [`LogDensity`], [`Cdf`], [`LogCdf`],
//!   [`Survival`], [`LogSurvival`], [`Quantile`], [`InverseSurvival`]
//! - **Count capabilities**: [`Mass`], [`LogMass`], [`CountSupport`],
//!   [`SignedCountSupport`], and the tail and inverse traits with integer arguments
//! - **Moments**: [`Mean`], [`Variance`], [`Entropy`]
//! - **Quasi-random**: [`QuasiRng`]
//!
//! Probability conventions shared by the capability traits: `cdf(x) = P(X <= x)`
//! and `sf(x) = P(X > x)`, with upper tails evaluated directly rather than as
//! `1 - cdf`; zero probability has logarithm `-∞`; continuous queries reject NaN
//! and infinite coordinates; quantiles are generalized inverses whose `p = 0` and
//! `p = 1` values are the support limits.

use alloc::vec::Vec;

use scioxide_core::{RealScalar, SciOxideError};

use crate::util::MAX_SAMPLING_ATTEMPTS;
use crate::workspace::{SamplingWorkspace, sample_owned, sample_rows, sample_scalars};
use crate::{SamplingPolicy, StatePolicy};

// ---------------------------------------------------------------------------
// RNG core traits
// ---------------------------------------------------------------------------

/// Source of random integer words, implemented by every generator.
///
/// A generator has two kinds of output. Native words ([`Self::next_u32`],
/// [`Self::next_u64`], [`Self::fill_bytes`]) are the outputs of the algorithm
/// as specified by its authors and may not span the full integer range; for example RANLUX
/// produces 24-bit values. Uniform words ([`Self::next_uniform_u32`],
/// [`Self::next_uniform_u64`]) are uniform over the full range, and every
/// integer and count sampler in this crate reads them; continuous samplers read
/// [`Rng::next_float`] instead. A generator whose native words are full
/// width returns them unchanged; the others combine several native words.
///
/// If you implement this trait, the uniform-word methods must return words
/// with every bit uniform. Samplers cannot check this from individual words.
///
/// ```
/// use scioxide_random::{BitGenerator, SeedableRng, SplitMix64};
/// let mut rng = SplitMix64::from_seed(42);
/// let word = rng.next_uniform_u64()?;
/// let mut bytes = [0_u8; 16];
/// rng.fill_bytes(&mut bytes)?;
/// assert_eq!(rng.position(), Some(3)); // one word, then two words of bytes
/// assert_eq!(SplitMix64::from_seed(42).next_uniform_u64()?, word);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait BitGenerator {
    /// Returns the number of native outputs since the start of the stream, or `None` if not tracked.
    ///
    /// Values discarded by luxury levels do not count. A recurrence generator
    /// stops tracking (returns `None`) when the count passes `u128::MAX` but keeps
    /// generating. A counter-based stream instead refuses reads past its end.
    /// The default implementation returns `None`.
    fn position(&self) -> Option<u128> {
        None
    }

    /// Returns the identity of the stream, or `None` if the generator has none.
    ///
    /// Drawing and skipping do not change it. See [`StreamIdentity`](crate::StreamIdentity).
    fn stream_identity(&self) -> Option<crate::StreamIdentity<'_>> {
        None
    }

    /// Returns the next native 32-bit output, which may not span the full `u32` range.
    ///
    /// # Errors
    /// - Any failure of the generator, such as the end of a finite stream.
    ///   Built-in generators do not advance on a failed read.
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError>;

    /// Returns the next native 64-bit output, which may not span the full `u64` range.
    ///
    /// # Errors
    /// - The errors of [`Self::next_u32`].
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError>;

    /// Returns a word uniform over the full `u32` range.
    ///
    /// A generator with full-width native output returns `next_u32`. A
    /// built-in generator with narrower output combines several native words
    /// by exact rejection: it accumulates digits in the native base and rejects
    /// an incomplete top interval. Unused bits are not cached for later calls.
    /// [`Self::position`] counts every native word consumed, rejected ones
    /// included.
    ///
    /// # Errors
    /// - Any failure of the generator.
    /// - `ConvergenceFailure` if a built-in generator rejects 1,024 candidates in
    ///   a row. Consumed native words stay consumed.
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError>;

    /// Returns a word uniform over the full `u64` range.
    ///
    /// A generator with full-width native output returns `next_u64`; otherwise
    /// it behaves as [`Self::next_uniform_u32`].
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, Ranlux, SeedableRng, decode_u64};
    /// let mut rng = Ranlux::from_seed(42);
    /// let uniform: f64 = decode_u64(rng.next_uniform_u64()?);
    /// assert!((0.0..1.0).contains(&uniform));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// - The errors of [`Self::next_uniform_u32`].
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError>;

    /// Fills `dest` with bytes of native output, which may not have all bits uniform.
    ///
    /// Each generator documents its byte order and the words it consumes. An
    /// empty `dest` consumes nothing.
    ///
    /// # Errors
    /// - Any failure of the generator, such as the end of a finite stream.
    ///   Counter-based generators check the whole fill before writing; a
    ///   [`BoundedSubstream`](crate::BoundedSubstream) keeps the complete words
    ///   written before the error.
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError>;
}

/// Source of uniform floating-point values in `[0, 1)`, read by every sampler.
///
/// Every built-in generator implements it for `f32` and `f64`. You can also
/// implement it for your own source; samplers then check each value, as
/// described under [uniform variates](crate#uniform-variates).
///
/// ```
/// use scioxide_random::{Rng, SeedableRng, SplitMix64};
/// let mut rng = SplitMix64::from_seed(42);
/// let x: f64 = rng.next_float()?;
/// let u: f32 = rng.next_open_float()?;
/// assert!((0.0..1.0).contains(&x));
/// assert!(u > 0.0 && u < 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Rng<S: RealScalar>: BitGenerator {
    /// Returns a uniform float in `[0, 1)`.
    ///
    /// This is the method to implement. Distributions and samplers check its
    /// result before use.
    ///
    /// # Errors
    /// - Any failure of the generator, such as the end of a finite stream.
    fn next_float(&mut self) -> Result<S, scioxide_core::SciOxideError>;

    /// Returns a uniform float in the open interval `(0, 1)`.
    ///
    /// Draws from [`Rng::next_float`], checks each value, and rejects zeros. Use
    /// it before a logarithm or an inverse CDF that is singular at 0.
    /// Samplers also validate the result of an overridden implementation;
    /// it must be finite and strictly between 0 and 1.
    ///
    /// # Errors
    ///
    /// - [`scioxide_core::SciOxideErrorKind::NonFinite`] if the provider returns NaN or infinity.
    /// - [`scioxide_core::SciOxideErrorKind::InvalidInput`] if the provider returns a finite value
    ///   outside `[0, 1)`.
    /// - [`scioxide_core::SciOxideErrorKind::ConvergenceFailure`] after 1,024 zeros in a row.
    /// - Any error of [`Rng::next_float`].
    #[inline]
    fn next_open_float(&mut self) -> Result<S, SciOxideError> {
        crate::util::open_probability(MAX_SAMPLING_ATTEMPTS, || {
            self.checked_float(scioxide_core::argument!("self"), private::Token)
        })
    }

    /// Checked half-open float for distribution and sampling consumers.
    ///
    /// Sealed: only SciOxide's own adapters, whose decoders prove the `[0, 1)`
    /// range, replace the validating default; custom providers always keep it.
    #[doc(hidden)]
    #[inline]
    fn checked_float(
        &mut self,
        argument: scioxide_core::Argument,
        _: private::Token,
    ) -> Result<S, SciOxideError> {
        crate::util::checked_unit_float(self.next_float()?, argument)
    }

    /// Validates an open-uniform override at the sampling boundary.
    #[doc(hidden)]
    #[inline]
    fn checked_open_float(&mut self, _: private::Token) -> Result<S, SciOxideError> {
        let value = crate::util::checked_unit_float(
            self.next_open_float()?,
            scioxide_core::argument!("rng"),
        )?;
        if value == S::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("rng"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "open-uniform provider returned zero",
            ));
        }
        Ok(value)
    }

    /// Batch hook used by [`crate::fill_uniforms`], which checks the counter
    /// capacity before calling it.
    #[doc(hidden)]
    fn fill_checked(
        &mut self,
        output: &mut [S],
        attempts: &mut usize,
        _: private::Token,
    ) -> Result<(), SciOxideError> {
        for value in output.iter_mut() {
            *attempts += 1;
            *value = self.next_float()?;
        }
        for &value in output.iter() {
            crate::util::checked_unit_float(value, scioxide_core::argument!("rng"))?;
        }
        Ok(())
    }
}

/// Sealed trait hooks: the token type is unnameable outside this crate, so
/// external implementations cannot replace methods that take it.
pub(crate) mod private {
    /// Witness that a hook call originates inside `scioxide-random`.
    #[derive(Debug, Clone, Copy)]
    pub struct Token;
}

// Borrowed trait objects remain usable with the sized generic distribution API.
// Preserve provider-specific endpoint behavior as well as scalar decoding.
impl<S: RealScalar, R: Rng<S> + ?Sized> Rng<S> for &mut R {
    #[inline]
    fn next_float(&mut self) -> Result<S, SciOxideError> {
        (**self).next_float()
    }
    #[inline]
    fn next_open_float(&mut self) -> Result<S, SciOxideError> {
        (**self).next_open_float()
    }
    #[inline]
    fn checked_open_float(&mut self, token: private::Token) -> Result<S, SciOxideError> {
        (**self).checked_open_float(token)
    }
    #[inline]
    fn checked_float(
        &mut self,
        argument: scioxide_core::Argument,
        token: private::Token,
    ) -> Result<S, SciOxideError> {
        (**self).checked_float(argument, token)
    }
    #[inline]
    fn fill_checked(
        &mut self,
        output: &mut [S],
        attempts: &mut usize,
        token: private::Token,
    ) -> Result<(), SciOxideError> {
        (**self).fill_checked(output, attempts, token)
    }
}

/// A generator created from a 64-bit seed.
///
/// Equal seeds give equal streams. [`Self::seed_from_entropy`] reads only the
/// entropy source you pass to it.
///
/// ```
/// use scioxide_core::{CapabilityError, EntropySource};
/// use scioxide_random::{BitGenerator, Pcg64, SeedableRng};
/// assert_eq!(Pcg64::from_seed(7).next_u64()?, Pcg64::from_seed(7).next_u64()?);
///
/// struct Fixed;
/// impl EntropySource for Fixed {
///     fn fill_bytes(&self, out: &mut [u8]) -> Result<(), CapabilityError> {
///         out.fill(0x5a);
///         Ok(())
///     }
/// }
/// let mut rng = Pcg64::seed_from_entropy(&Fixed)?;
/// rng.next_u64()?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait SeedableRng: Sized {
    /// Creates a generator from `seed`.
    #[must_use]
    fn from_seed(seed: u64) -> Self;

    /// Creates a generator from a 64-bit seed read from `source`.
    ///
    /// # Errors
    ///
    /// - [`scioxide_core::SciOxideErrorKind::CapabilityFailure`] if `source` cannot
    ///   supply the seed. No other source is tried.
    fn seed_from_entropy(source: &dyn scioxide_core::EntropySource) -> Result<Self, SciOxideError> {
        crate::seed::entropy_seed(source).map(Self::from_seed)
    }
}

// ---------------------------------------------------------------------------
// Advanced RNG traits
// ---------------------------------------------------------------------------

/// A generator that can advance by any number of outputs without generating them.
///
/// Recurrence generators use the closed form of the recurrence or polynomial
/// arithmetic, so a skip costs far less than generating the outputs; each
/// generator states its cost. Skipping lets you cut one stream into
/// reproducible chunks, as [`BoundedSubstream::for_chunk`](crate::BoundedSubstream::for_chunk) does.
///
/// ```
/// use scioxide_random::{BitGenerator, Pcg64, SeedableRng, SkipAhead};
/// let mut skipped = Pcg64::from_seed(42);
/// skipped.skip_ahead(1000)?;
/// let mut stepped = Pcg64::from_seed(42);
/// for _ in 0..1000 {
///     stepped.next_u64()?;
/// }
/// assert_eq!(skipped.position(), Some(1000));
/// assert_eq!(skipped.next_u64()?, stepped.next_u64()?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait SkipAhead: BitGenerator {
    /// Advances the generator by `n` native outputs.
    ///
    /// If the position passes `u128::MAX`, a recurrence generator still
    /// advances and [`BitGenerator::position`] becomes `None`.
    ///
    /// # Errors
    ///
    /// - A counter-based stream refuses to pass its end and stays unchanged.
    fn skip_ahead(&mut self, n: u128) -> Result<(), SciOxideError>;

    /// Advances the generator to the absolute position `index`.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds`, without changing the generator, if `index` is before the
    ///   current position or past the end of a finite stream.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the generator does not
    ///   track its position.
    ///
    /// ```
    /// use scioxide_random::{BitGenerator, SeedableRng, SkipAhead, SplitMix64};
    /// let mut rng = SplitMix64::from_seed(42);
    /// rng.skip_to(10)?;
    /// assert_eq!(rng.position(), Some(10));
    /// assert_eq!(rng.skip_to(9).unwrap_err().kind(),
    ///     scioxide_core::SciOxideErrorKind::OutOfBounds);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn skip_to(&mut self, index: u128) -> Result<(), SciOxideError> {
        let current = self.position().ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "absolute movement requires a tracked position",
            )
        })?;
        let distance = index.checked_sub(current).ok_or_else(|| {
            // Core bounds diagnostics use usize; saturate full-width positions.
            SciOxideError::out_of_bounds(
                scioxide_core::quantity!("generator_position"),
                usize::try_from(index).unwrap_or(usize::MAX),
                usize::try_from(current).unwrap_or(usize::MAX),
            )
        })?;
        self.skip_ahead(distance)
    }

    /// Worst-case native transitions consumed by one word or float read.
    ///
    /// Sealed: built-in engines declare their bound so bounded substreams can
    /// read in place while the bound fits the remaining range; `None` makes a
    /// bounded substream read from a copy and keep it only within the range,
    /// including transitions consumed before a provider failure.
    #[doc(hidden)]
    #[inline]
    fn read_transition_bound(&self, _: private::Token) -> Option<u128> {
        None
    }
}

/// A generator with fixed long jumps, used to give parallel workers non-overlapping blocks.
///
/// Each generator states its jump distances. Recurrence generators apply a
/// precomputed jump polynomial; counter-based generators add to the counter.
/// Successive jumps give disjoint blocks only if no worker draws more than one
/// jump distance.
///
/// ```
/// use scioxide_random::{BitGenerator, JumpableRng, Philox4x64, SeedableRng};
/// let mut rng = Philox4x64::from_seed(42);
/// rng.jump()?; // 2^32 outputs for Philox
/// assert_eq!(rng.position(), Some(1 << 32));
/// rng.long_jump()?; // 2^48 outputs
/// assert_eq!(rng.position(), Some((1 << 32) + (1 << 48)));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait JumpableRng {
    /// Advances the generator by its jump distance.
    ///
    /// # Errors
    ///
    /// - A counter-based stream refuses to jump past its end.
    fn jump(&mut self) -> Result<(), SciOxideError>;

    /// Advances the generator by its long-jump distance, larger than [`Self::jump`].
    ///
    /// # Errors
    ///
    /// - A counter-based stream refuses to jump past its end.
    fn long_jump(&mut self) -> Result<(), SciOxideError>;
}

/// A generator that derives numbered streams from its root, for parallel work.
///
/// Stream numbers are absolute: calling [`Self::split`] on a derived or advanced
/// generator selects from the same root. Selecting a stream does not change
/// the parent, and the new stream starts at position zero. Different numbers
/// give different sequences; whether they overlap within the outputs you draw
/// depends on the generator's period and stream spacing, which each generator
/// states.
///
/// ```
/// use scioxide_random::{BitGenerator, Pcg64, SeedableRng, SplittableRng};
/// let root = Pcg64::from_seed(42);
/// let mut worker = root.split(1)?;
/// worker.next_u64()?;
/// // Selection is absolute: an advanced worker still selects from the root.
/// assert_eq!(worker.split(0)?.next_u64()?, root.split(0)?.next_u64()?);
/// assert_eq!(root.substream(1, 0)?.next_u64()?, root.split(1)?.next_u64()?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait SplittableRng: Sized {
    /// Returns stream `stream_id` of the root, at position zero.
    ///
    /// # Errors
    ///
    /// - Returns an error if the generator does not support `stream_id`.
    fn split(&self, stream_id: u64) -> Result<Self, SciOxideError>;

    /// Returns substream `subseq_id` of stream `stream_id` of the root, at position zero.
    ///
    /// # Errors
    ///
    /// - Returns an error if the generator does not support the stream or substream index.
    fn substream(&self, stream_id: u64, subseq_id: u64) -> Result<Self, SciOxideError>;
}

/// A generator whose state can be saved to and restored from portable bytes.
///
/// A checkpoint starts with the magic bytes `GRN2` and format version 1 and
/// ends with a CRC-32C checksum. It holds the generator's stream identity, its
/// state and its position (or that the position is not tracked). All fields
/// are little-endian and of fixed width, so a checkpoint does not depend on the
/// platform, the pointer width or the float type you sample. The checksum
/// detects corruption; it does not authenticate the checkpoint.
///
/// ```
/// use scioxide_random::{BitGenerator, SeedableRng, SerializableState, SplitMix64};
/// let mut rng = SplitMix64::from_seed(42);
/// rng.next_u64()?;
/// let checkpoint = rng.export_state()?;
/// assert_eq!(checkpoint.len(), SplitMix64::STATE_BYTES);
/// let expected = rng.next_u64()?;
/// rng.restore_state(&checkpoint)?; // rewind this stream
/// assert_eq!(rng.next_u64()?, expected);
/// let mut imported = SplitMix64::import_state(&checkpoint)?;
/// assert_eq!(imported.next_u64()?, expected);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait SerializableState: BitGenerator + Sized {
    /// Length of a checkpoint in bytes, including the header and checksum.
    const STATE_BYTES: usize;

    /// Returns a checkpoint of the current state, without advancing the generator.
    ///
    /// # Errors
    /// - `AllocationFailure` if the byte vector cannot be allocated.
    /// - The errors of [`Self::export_state_into`].
    fn export_state(&self) -> Result<Vec<u8>, SciOxideError> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(Self::STATE_BYTES).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("random_generator_state"),
                Self::STATE_BYTES,
            )
        })?;
        bytes.resize(Self::STATE_BYTES, 0);
        self.export_state_into(&mut bytes)?;
        Ok(bytes)
    }

    /// Writes a checkpoint into `output`, which must have length [`Self::STATE_BYTES`].
    ///
    /// # Errors
    /// - `DimensionMismatch` if `output` has the wrong length. The output is
    ///   then unchanged.
    fn export_state_into(&self, output: &mut [u8]) -> Result<(), SciOxideError>;

    /// Creates a generator from a checkpoint.
    ///
    /// # Errors
    /// - `InvalidInput` if the bytes have the wrong length, a bad checksum, an
    ///   unknown version, a field outside its valid range, or the tag of another
    ///   algorithm.
    fn import_state(bytes: &[u8]) -> Result<Self, SciOxideError>;

    /// Restores a checkpoint taken from this generator's stream, to rewind or replay it.
    ///
    /// To switch to a different stream, use [`Self::import_state`].
    ///
    /// # Errors
    /// - The errors of [`Self::import_state`].
    /// - `InvalidInput` (`Inconsistent`) if the checkpoint belongs to another
    ///   stream.
    ///
    /// On error the generator is unchanged.
    fn restore_state(&mut self, bytes: &[u8]) -> Result<(), SciOxideError> {
        // Decode fully before touching `self`, and refuse checkpoints from
        // another stream: restoring is meant to rewind or replay this
        // sequence, not to swap it silently for a different one.
        let candidate = Self::import_state(bytes)?;
        if candidate.stream_identity() != self.stream_identity() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("bytes"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "checkpoint belongs to a different immutable stream",
            ));
        }
        *self = candidate;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Distribution traits
// ---------------------------------------------------------------------------

/// A probability distribution with real-valued samples.
///
/// Implementors provide [`Self::sample`]; the batch methods call it. A law
/// draws one sample with [`Self::sample`], `n` samples into a new `Vec` with
/// [`Self::sample_n`], and fills a slice you provide under a
/// [`SamplingPolicy`] with [`Self::sample_n_into`]. See
/// [batch sampling](crate#batch-sampling).
///
/// ```
/// use scioxide_random::{Distribution, Exponential, OutputPolicy, SamplingPolicy,
///     SamplingRequest, SeedableRng, SplitMix64};
/// let law = Exponential::new(2.0_f64)?;
/// let mut rng = SplitMix64::from_seed(42);
/// assert!(law.sample(&mut rng)? >= 0.0);
/// assert!(law.sample_n(&mut rng, 4)?.iter().all(|&x| x >= 0.0));
/// let policy = SamplingPolicy::advance_on_error(OutputPolicy::Incremental);
/// let mut workspace = policy.workspace(SamplingRequest::scalars(4))?;
/// let mut output = [0.0; 4];
/// law.sample_n_into(&mut rng, policy, &mut output, &mut workspace)?;
/// assert!(output.iter().all(|&x| x >= 0.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Distribution<S: RealScalar> {
    /// Byte limit of the buffers allocated by the default scalar batch methods.
    #[doc(hidden)]
    fn batch_byte_limit(&self, _: private::Token) -> usize {
        usize::MAX
    }
    /// Draws one sample.
    ///
    /// # Errors
    ///
    /// - Any error of the generator, or a non-finite or out-of-range value from
    ///   a custom [`Rng`].
    /// - `ConvergenceFailure` if a rejection loop or a [work limit](crate#work-limits)
    ///   is exhausted.
    /// - `NumericalInstability` if the sample overflows the output type.
    ///
    /// Each law lists further conditions.
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError>;
    /// Draws one sample, counting numerical work toward the work limit of `context`.
    ///
    /// Combinators such as [`Mixture`](crate::Mixture) call it so that a nested
    /// draw shares the enclosing law's work limit. The default counts one unit.
    ///
    /// # Errors
    /// - The errors of [`Self::sample`].
    /// - `ConvergenceFailure` if the shared work limit is exhausted.
    fn sample_scoped<R: Rng<S>>(
        &self,
        rng: &mut R,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::provider(context, || self.sample(rng))
    }

    /// Fills `output` with independent samples.
    ///
    /// `policy` chooses what is written and where the generator stands on
    /// error ([batch sampling](crate#batch-sampling)). Writing all-or-nothing
    /// needs a workspace of `output.len()` elements; writing incrementally
    /// needs none. On success the samples equal `output.len()` calls of
    /// [`Self::sample`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace exceeds a law's byte limit.
    /// - `AllocationFailure` if the workspace cannot grow.
    /// - Any error of [`Self::sample`].
    fn sample_n_into<R: Rng<S>>(
        &self,
        rng: &mut R,
        policy: SamplingPolicy<impl StatePolicy<R>>,
        output: &mut [S],
        workspace: &mut SamplingWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        sample_scalars(
            rng,
            policy,
            output,
            S::ZERO,
            workspace,
            self.batch_byte_limit(private::Token),
            |rng| self.sample(rng),
        )
    }

    /// Returns `n` independent samples in a new `Vec`.
    ///
    /// The result is allocated once, before the first draw, and the samples
    /// equal `n` calls of [`Self::sample`]. On error nothing is returned and
    /// the generator stays advanced by the words it consumed; to restore it,
    /// use [`Self::sample_n_into`] with [`SamplingPolicy::rollback`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the output exceeds a law's byte limit.
    /// - `SizeOverflow` if the output byte count overflows `usize`.
    /// - `AllocationFailure` if the output cannot be allocated.
    /// - Any error of [`Self::sample`].
    ///
    /// ```
    /// use scioxide_random::{Distribution, Normal, SeedableRng, SplitMix64};
    /// let mut rng = SplitMix64::from_seed(42);
    /// let samples = Normal::<f64>::standard().sample_n(&mut rng, 1_000)?;
    /// let mean = samples.iter().sum::<f64>() / 1_000.0;
    /// assert!(mean.abs() < 0.16); // five standard errors of 1/√1000
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn sample_n<R: Rng<S>>(&self, rng: &mut R, n: usize) -> Result<Vec<S>, SciOxideError> {
        crate::workspace::admit_bytes(
            crate::workspace::storage_bytes::<S>(n)?,
            self.batch_byte_limit(private::Token),
        )?;
        sample_owned(rng, n, S::ZERO, |rng| self.sample(rng))
    }
}

/// A discrete distribution with exact integer samples.
///
/// Every count law implements it, as do the prepared samplers
/// [`BinomialSampler`](crate::BinomialSampler),
/// [`PoissonSampler`](crate::PoissonSampler) and
/// [`SkellamSampler`](crate::SkellamSampler).
/// [`Binomial`](crate::Binomial), [`Poisson`](crate::Poisson) and
/// [`Skellam`](crate::Skellam) prepare their sampler once per call and draw
/// through it; for a loop of single draws, prepare the sampler once. Draws
/// read uniform words ([`BitGenerator::next_uniform_u64`], or
/// [`BitGenerator::next_uniform_u32`] for index draws below 2³²), so the
/// samples do not depend on the law's float type. A count that cannot be
/// represented is an error, never clipped. The
/// [work limits](crate#work-limits) of the law apply to each draw, and its
/// byte limit to the workspace and owned output of batches. The default batch
/// methods apply no byte limit; implement them to apply one.
///
/// ```
/// use scioxide_random::{CountDistribution, Geometric, OutputPolicy, SamplingPolicy,
///     SamplingRequest, SeedableRng, SplitMix64};
/// let law = Geometric::new(0.25_f64)?;
/// let mut rng = SplitMix64::from_seed(42);
/// assert!(law.sample(&mut rng)? >= 1);
/// assert!(law.sample_n(&mut rng, 8)?.iter().all(|&k| k >= 1));
/// let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
/// let mut workspace = policy.workspace(SamplingRequest::scalars(8))?;
/// let mut counts = [0; 8];
/// law.sample_n_into(&mut rng, policy, &mut counts, &mut workspace)?;
/// assert!(counts.iter().all(|&k| k >= 1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait CountDistribution {
    /// Sample type: `usize` for nonnegative counts, `i64` for signed counts.
    type Count: Copy + Default;

    /// Draws one count.
    ///
    /// Words consumed before an error stay consumed.
    ///
    /// # Errors
    ///
    /// - `ConvergenceFailure` if a rejection loop, the word limit or the work
    ///   limit is exhausted.
    /// - `NotImplemented` if the count would exceed the range the sampler
    ///   computes exactly; each law states its range.
    /// - Any error of the generator, such as the end of a finite stream.
    fn sample<R: BitGenerator + ?Sized>(&self, rng: &mut R) -> Result<Self::Count, SciOxideError>;

    /// Fills `output` with independent counts.
    ///
    /// `policy` chooses what is written and where the generator stands on
    /// error ([batch sampling](crate#batch-sampling)). Writing all-or-nothing
    /// needs a workspace of `output.len()` elements; writing incrementally
    /// needs none. On success the counts equal `output.len()` calls of
    /// [`Self::sample`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace exceeds the law's byte limit.
    /// - `AllocationFailure` if the workspace cannot grow.
    /// - Any error of [`Self::sample`].
    fn sample_n_into<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        policy: SamplingPolicy<impl StatePolicy<R>>,
        output: &mut [Self::Count],
        workspace: &mut SamplingWorkspace<Self::Count>,
    ) -> Result<(), SciOxideError> {
        sample_scalars(
            rng,
            policy,
            output,
            Self::Count::default(),
            workspace,
            usize::MAX,
            |rng| self.sample(rng),
        )
    }

    /// Returns `n` independent counts in a new `Vec`.
    ///
    /// The result is allocated once, before the first draw, and the counts
    /// equal `n` calls of [`Self::sample`]. On error nothing is returned and
    /// the generator stays advanced by the words it consumed; to restore it,
    /// use [`Self::sample_n_into`] with [`SamplingPolicy::rollback`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the output exceeds the law's byte limit.
    /// - `AllocationFailure` if the output cannot be allocated.
    /// - Any error of [`Self::sample`].
    fn sample_n<R: BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        n: usize,
    ) -> Result<Vec<Self::Count>, SciOxideError> {
        sample_owned(rng, n, Self::Count::default(), |rng| self.sample(rng))
    }
}

/// Probability density `f(x)` of a continuous law.
///
/// Points outside the support have density zero; this is a value, not an
/// error. Where the density is discontinuous, as at a support endpoint, the
/// law's documented version is returned: [`Uniform`](crate::Uniform) uses the
/// half-open `[low, high)`, and a density that diverges at an endpoint, such as
/// [`Gamma`](crate::Gamma) with shape below one at zero, is `+∞`. Vector- and
/// matrix-valued laws take their coordinates as `X`.
///
/// ```
/// use scioxide_random::{Density, Exponential};
/// let law = Exponential::new(2.0_f64)?;
/// assert!((law.pdf(0.5)? - 2.0 * (-1.0_f64).exp()).abs() < 1e-15);
/// assert_eq!(law.pdf(-1.0)?, 0.0); // outside the support
/// assert!(law.pdf(f64::NAN).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Density<S: RealScalar, X = S> {
    /// Returns the density at `x`; zero outside the support.
    ///
    /// # Errors
    /// - `NonFinite` if a coordinate is NaN or infinite.
    /// - `DimensionMismatch` or `InvalidInput` if a vector argument has the
    ///   wrong length or shape.
    /// - `NumericalInstability` if the density overflows.
    fn pdf(&self, x: X) -> Result<S, SciOxideError>;
}

/// Probability mass `P(X = k)` at an exact integer `k`.
///
/// `X` is `usize` for nonnegative counts (the default) and `i64` for signed
/// counts. Integers outside the support have mass zero; this is a value, not
/// an error.
///
/// ```
/// use scioxide_random::{Binomial, Mass};
/// let law = Binomial::new(4, 0.5_f64)?;
/// assert!((law.pmf(2)? - 0.375).abs() < 1e-15);
/// assert_eq!(law.pmf(5)?, 0.0); // outside 0..=4
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Mass<S: RealScalar, X = usize> {
    /// Returns `P(X = k)`; zero outside the support.
    ///
    /// # Errors
    /// - `NotImplemented` if `k` exceeds the range the law computes exactly,
    ///   such as magnitudes above `2^53`.
    fn pmf(&self, k: X) -> Result<S, SciOxideError>;
}

/// Cumulative distribution function `F(x) = P(X ≤ x)`, including the queried point.
///
/// Count laws take exact integer arguments: `cdf(k) = P(X <= k)` includes the
/// atom at `k`, with no continuity correction. The value is zero below the
/// support and one at or above its upper end.
///
/// ```
/// use scioxide_random::{Cdf, Normal, Poisson};
/// let normal = Normal::new(0.0_f64, 1.0)?;
/// assert!((normal.cdf(0.0)? - 0.5).abs() < 1e-15);
/// // P(X <= 1) = P(X = 0) + P(X = 1) includes the atom at one.
/// let poisson = Poisson::new(2.0_f64)?;
/// assert!((poisson.cdf(1)? - 3.0 * (-2.0_f64).exp()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Cdf<S: RealScalar, X = S> {
    /// Returns `P(X ≤ x)`.
    ///
    /// # Errors
    /// - `NonFinite` if `x` is NaN or infinite.
    /// - `NotImplemented` if an integer `x` exceeds the range the law computes
    ///   exactly.
    fn cdf(&self, x: X) -> Result<S, SciOxideError>;
}

/// Quantile function, the generalized inverse of the CDF: the smallest `x` with `F(x) ≥ p`.
///
/// Where the CDF is flat at level `p` (a gap in the support), continuous laws
/// return the left edge of the plateau; integer laws return the smallest `k`
/// with `cdf(k) >= p`. `p = 0` and `p = 1` return the lower and upper support
/// limits, which may be infinite; an unbounded limit of an integer law is not
/// representable and returns `NotImplemented`. Numerically inverted continuous
/// laws evaluate the smaller tail. Use [`InverseSurvival::isf`] for upper-tail
/// probabilities whose complement rounds to one in the scalar type.
///
/// ```
/// use scioxide_random::{Exponential, Histogram, Poisson, Quantile};
/// // Mass 1/2 on [0, 1] and on [2, 3]: the CDF is flat at 1/2 on [1, 2].
/// let gap = Histogram::from_masses(&[0.0_f64, 1.0, 2.0, 3.0], &[1.0, 0.0, 1.0])?;
/// assert_eq!(gap.quantile(0.5)?, 1.0); // left edge of the plateau
/// assert_eq!(Exponential::new(1.0_f64)?.quantile(1.0)?, f64::INFINITY);
/// let poisson = Poisson::new(2.0_f64)?;
/// assert_eq!(poisson.quantile(0.5)?, 2); // cdf(1) < 0.5 <= cdf(2)
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Quantile<S: RealScalar, X = S> {
    /// Returns the smallest `x` with `F(x) ≥ p`.
    ///
    /// # Errors
    /// - `NonFinite` if `p` is NaN or infinite.
    /// - `DomainError` if `p` is outside `[0, 1]`.
    /// - `NotImplemented` if the integer result cannot be represented.
    /// - `ConvergenceFailure` if a numerical inversion exhausts its
    ///   [work limit](crate#work-limits).
    fn quantile(&self, p: S) -> Result<X, SciOxideError>;

    /// Evaluates the same query, counting its work toward the work limit of `context`.
    ///
    /// Combinators call it so that nested queries share the enclosing law's
    /// [work limit](crate#work-limits). The default counts one unit. An
    /// implementation with iterative work must override it and count every
    /// evaluation.
    ///
    /// # Errors
    /// - The errors of the plain query.
    /// - `ConvergenceFailure` if the shared work limit is exhausted.
    fn quantile_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<X, SciOxideError> {
        crate::law::provider(context, || self.quantile(p))
    }
}

/// Expected value `E[X]`.
///
/// Implemented only where the law computes the mean itself: in closed form,
/// from special functions or series, or by numerical quadrature. There is no
/// generic fallback for other laws. An undefined or divergent mean for the
/// current parameters is a `DomainError` with
/// [`scioxide_core::DomainReason::Undefined`].
/// Vector- and matrix-valued laws return componentwise means as `Y`.
///
/// ```
/// use scioxide_core::{DomainReason, SciOxideErrorView};
/// use scioxide_random::{Exponential, Mean, StudentT};
/// assert!((Exponential::new(4.0_f64)?.mean()? - 0.25).abs() < 1e-15);
/// let undefined = StudentT::new(1.0_f64)?.mean().is_err_and(|error| matches!(
///     error.view(),
///     SciOxideErrorView::DomainError { reason: DomainReason::Undefined, .. }
/// ));
/// assert!(undefined);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Mean<S: RealScalar, Y = S> {
    /// Returns the mean.
    ///
    /// # Errors
    /// - `DomainError` (`Undefined`) if the mean does not exist.
    /// - `NumericalInstability` if it overflows.
    fn mean(&self) -> Result<Y, SciOxideError>;
    /// Evaluates the same query, counting its work toward the work limit of `context`.
    ///
    /// Combinators call it so that nested queries share the enclosing law's
    /// [work limit](crate#work-limits). The default counts one unit. An
    /// implementation with iterative work must override it and count every
    /// evaluation.
    ///
    /// # Errors
    /// - The errors of the plain query.
    /// - `ConvergenceFailure` if the shared work limit is exhausted.
    fn mean_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Y, SciOxideError> {
        crate::law::provider(context, || self.mean())
    }
}

/// Variance `E[(X - E[X])^2]`.
///
/// An undefined or divergent variance for the current parameters is a
/// `DomainError` with [`scioxide_core::DomainReason::Undefined`]. Vector- and
/// matrix-valued laws return componentwise variances as `Y`, not a covariance
/// matrix.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_random::{Exponential, StudentT, Variance};
/// assert!((Exponential::new(2.0_f64)?.variance()? - 0.25).abs() < 1e-15);
/// let infinite = StudentT::new(2.0_f64)?.variance();
/// assert_eq!(infinite.err().map(|error| error.kind()), Some(SciOxideErrorKind::DomainError));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Variance<S: RealScalar, Y = S> {
    /// Returns the variance.
    ///
    /// For a nondegenerate scalar law the result is positive: a variance that
    /// cancels or underflows to zero is an error, not zero.
    ///
    /// # Errors
    /// - `DomainError` (`Undefined`) if the variance does not exist.
    /// - Returns an error if it overflows, underflows to zero or cannot be
    ///   computed to the law's tolerance.
    fn variance(&self) -> Result<Y, SciOxideError>;
    /// Evaluates the same query, counting its work toward the work limit of `context`.
    ///
    /// Combinators call it so that nested queries share the enclosing law's
    /// [work limit](crate#work-limits). The default counts one unit. An
    /// implementation with iterative work must override it and count every
    /// evaluation.
    ///
    /// # Errors
    /// - The errors of the plain query.
    /// - `ConvergenceFailure` if the shared work limit is exhausted.
    fn variance_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Y, SciOxideError> {
        crate::law::provider(context, || self.variance())
    }
}

/// Shannon entropy in nats (natural logarithm).
///
/// Continuous laws return the differential entropy `-E[ln f(X)]`, which can be
/// negative; discrete laws return `-E[ln P(X)]`.
///
/// ```
/// use scioxide_random::{Entropy, Exponential, Normal};
/// let normal = Normal::new(0.0_f64, 1.0)?;
/// let expected = 0.5 * (core::f64::consts::TAU * core::f64::consts::E).ln();
/// assert!((normal.entropy()? - expected).abs() < 1e-15);
/// assert!(Exponential::new(4.0_f64)?.entropy()? < 0.0); // 1 - ln 4
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Entropy<S: RealScalar> {
    /// Returns the entropy in nats.
    ///
    /// # Errors
    /// - Returns an error if the entropy is undefined for the parameters or
    ///   overflows.
    /// - `ConvergenceFailure` if a series or quadrature exhausts its
    ///   [work limit](crate#work-limits).
    fn entropy(&self) -> Result<S, SciOxideError>;
    /// Evaluates the same query, counting its work toward the work limit of `context`.
    ///
    /// Combinators call it so that nested queries share the enclosing law's
    /// [work limit](crate#work-limits). The default counts one unit. An
    /// implementation with iterative work must override it and count every
    /// evaluation.
    ///
    /// # Errors
    /// - The errors of the plain query.
    /// - `ConvergenceFailure` if the shared work limit is exhausted.
    fn entropy_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::provider(context, || self.entropy())
    }
}

/// Natural logarithm of the probability density, `ln f(x)`.
///
/// Points outside the support give `-∞`; endpoint values follow the law's
/// documented density version, as for [`Density`]. Use it where
/// [`Density::pdf`] underflows.
///
/// ```
/// use scioxide_random::{Exponential, LogDensity, Normal};
/// let law = Exponential::new(2.0_f64)?;
/// assert!((law.log_pdf(0.5)? - (2.0_f64.ln() - 1.0)).abs() < 1e-15);
/// assert_eq!(law.log_pdf(-1.0)?, f64::NEG_INFINITY);
/// let normal = Normal::new(0.0_f64, 1.0)?;
/// let expected = -800.0 - 0.5 * core::f64::consts::TAU.ln(); // pdf(40) underflows
/// assert!((normal.log_pdf(40.0)? - expected).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait LogDensity<S: RealScalar, X = S> {
    /// Returns `ln f(x)`; `−∞` outside the support.
    ///
    /// # Errors
    /// Returns the errors of [`Density::pdf`].
    fn log_pdf(&self, x: X) -> Result<S, SciOxideError>;

    /// Evaluates the same query, counting its work toward the work limit of `context`.
    ///
    /// Combinators call it so that nested queries share the enclosing law's
    /// [work limit](crate#work-limits). The default counts one unit. An
    /// implementation with iterative work must override it and count every
    /// evaluation.
    ///
    /// # Errors
    /// - The errors of the plain query.
    /// - `ConvergenceFailure` if the shared work limit is exhausted.
    fn log_pdf_scoped(
        &self,
        x: X,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::provider(context, || self.log_pdf(x))
    }
}

/// Natural logarithm of the probability mass, `ln P(X = k)`.
///
/// Integers outside the support give `-∞`.
///
/// ```
/// use scioxide_random::{Binomial, LogMass, Poisson};
/// assert!((Poisson::new(3.0_f64)?.log_pmf(0)? + 3.0).abs() < 1e-14);
/// let binomial = Binomial::new(4, 0.5_f64)?;
/// assert_eq!(binomial.log_pmf(5)?, f64::NEG_INFINITY);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait LogMass<S: RealScalar, X = usize> {
    /// Returns `ln P(X = k)`; `−∞` outside the support.
    ///
    /// # Errors
    /// Returns the errors of [`Mass::pmf`].
    fn log_pmf(&self, k: X) -> Result<S, SciOxideError>;
}

/// Survival function `P(X > x)`, the probability strictly above the queried point.
///
/// Upper tails are evaluated directly, never as `1 - cdf`, so small upper
/// probabilities keep their relative accuracy. For count laws `sf(k)` excludes
/// the atom at `k`, so `cdf(k) + sf(k) = 1`.
///
/// ```
/// use scioxide_random::{Cdf, Exponential, Poisson, Survival};
/// let law = Exponential::new(2.0_f64)?;
/// assert_eq!(1.0 - law.cdf(30.0)?, 0.0); // the complement loses the tail
/// assert!((law.sf(30.0)? / (-60.0_f64).exp() - 1.0).abs() < 1e-14);
/// let poisson = Poisson::new(2.0_f64)?;
/// assert!((poisson.cdf(1)? + poisson.sf(1)? - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Survival<S: RealScalar, X = S> {
    /// Returns `P(X > x)`.
    ///
    /// # Errors
    /// Returns the errors of [`Cdf::cdf`].
    fn sf(&self, x: X) -> Result<S, SciOxideError>;
}

/// Logarithm of the cumulative distribution function, `ln P(X ≤ x)`.
///
/// Zero probability gives `-∞`. Deep lower tails stay finite where
/// [`Cdf::cdf`] underflows to zero.
///
/// ```
/// use scioxide_random::{Cdf, Exponential, LogCdf, Normal};
/// let normal = Normal::new(0.0_f64, 1.0)?;
/// assert_eq!(normal.cdf(-40.0)?, 0.0); // underflows
/// let log_p = normal.log_cdf(-40.0)?; // about -804.6
/// assert!(log_p > -805.0 && log_p < -804.0);
/// assert_eq!(Exponential::new(1.0_f64)?.log_cdf(-1.0)?, f64::NEG_INFINITY);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait LogCdf<S: RealScalar, X = S> {
    /// Returns `ln P(X ≤ x)`; `−∞` for zero probability.
    ///
    /// # Errors
    /// Returns the errors of [`Cdf::cdf`].
    fn log_cdf(&self, x: X) -> Result<S, SciOxideError>;

    /// Evaluates the same query, counting its work toward the work limit of `context`.
    ///
    /// Combinators call it so that nested queries share the enclosing law's
    /// [work limit](crate#work-limits). The default counts one unit. An
    /// implementation with iterative work must override it and count every
    /// evaluation.
    ///
    /// # Errors
    /// - The errors of the plain query.
    /// - `ConvergenceFailure` if the shared work limit is exhausted.
    fn log_cdf_scoped(
        &self,
        x: X,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::provider(context, || self.log_cdf(x))
    }
}

/// Natural logarithm of the upper probability, `ln P(X > x)`.
///
/// Zero probability gives `-∞`. Deep upper tails stay finite where
/// [`Survival::sf`] underflows to zero.
///
/// ```
/// use scioxide_random::{LogSurvival, Normal, Uniform};
/// let log_q = Normal::new(0.0_f64, 1.0)?.log_sf(40.0)?; // about -804.6
/// assert!(log_q > -805.0 && log_q < -804.0);
/// assert_eq!(Uniform::new(0.0_f64, 1.0)?.log_sf(1.0)?, f64::NEG_INFINITY);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait LogSurvival<S: RealScalar, X = S> {
    /// Returns `ln P(X > x)`; `−∞` for zero probability.
    ///
    /// # Errors
    /// Returns the errors of [`Cdf::cdf`].
    fn log_sf(&self, x: X) -> Result<S, SciOxideError>;

    /// Evaluates the same query, counting its work toward the work limit of `context`.
    ///
    /// Combinators call it so that nested queries share the enclosing law's
    /// [work limit](crate#work-limits). The default counts one unit. An
    /// implementation with iterative work must override it and count every
    /// evaluation.
    ///
    /// # Errors
    /// - The errors of the plain query.
    /// - `ConvergenceFailure` if the shared work limit is exhausted.
    fn log_sf_scoped(
        &self,
        x: X,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::provider(context, || self.log_sf(x))
    }
}

/// Inverse survival function: the point whose upper-tail probability
/// `P(X > x)` is `p`, computed without forming `1 - p`.
///
/// Mathematically `isf(p) = quantile(1 - p)`, but small `p` keeps its full
/// relative accuracy. Continuous laws return the right edge of a survival
/// plateau; integer laws return the smallest `k` with `sf(k) <= p`. `p = 0`
/// returns the upper and `p = 1` the lower support limit, with the
/// representability rule of [`Quantile`].
///
/// ```
/// use scioxide_random::{Histogram, InverseSurvival, Normal, Quantile};
/// let gap = Histogram::from_masses(&[0.0_f64, 1.0, 2.0, 3.0], &[1.0, 0.0, 1.0])?;
/// assert_eq!(gap.isf(0.5)?, 2.0); // right edge; quantile(0.5) is 1.0
/// let normal = Normal::new(0.0_f64, 1.0)?;
/// assert!((normal.isf(0.025)? + normal.quantile(0.025)?).abs() < 1e-12);
/// assert_eq!(normal.quantile(1.0 - 1e-300)?, f64::INFINITY); // 1 - p rounds to one
/// assert!((normal.isf(1e-300)? - 37.047).abs() < 1e-3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait InverseSurvival<S: RealScalar, X = S> {
    /// Returns the point `x` with `P(X > x) = p`.
    ///
    /// # Errors
    /// Returns the errors of [`Quantile::quantile`].
    fn isf(&self, p: S) -> Result<X, SciOxideError>;

    /// Evaluates the same query, counting its work toward the work limit of `context`.
    ///
    /// Combinators call it so that nested queries share the enclosing law's
    /// [work limit](crate#work-limits). The default counts one unit. An
    /// implementation with iterative work must override it and count every
    /// evaluation.
    ///
    /// # Errors
    /// - The errors of the plain query.
    /// - `ConvergenceFailure` if the shared work limit is exhausted.
    fn isf_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<X, SciOxideError> {
        crate::law::provider(context, || self.isf(p))
    }
}

/// Integer support of a law on nonnegative counts, independent of the float type.
///
/// Bounds are inclusive and describe the current parameters, so degenerate
/// parameters can narrow the support.
///
/// ```
/// use scioxide_random::{Binomial, CountSupport, Poisson};
/// let binomial = Binomial::new(10, 0.3_f64)?;
/// assert_eq!(binomial.count_support(), (0, Some(10)));
/// let certain = Binomial::new(10, 1.0_f64)?;
/// assert_eq!(certain.count_support(), (10, Some(10)));
/// assert_eq!(Poisson::new(2.0_f64)?.count_support(), (0, None));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait CountSupport {
    /// Returns the inclusive lower and upper bounds of the support; `None` is an unbounded upper end.
    fn count_support(&self) -> (usize, Option<usize>);
}

/// Integer support of a law on signed counts; either end may be unbounded.
///
/// ```
/// use scioxide_random::{SignedCountSupport, Skellam};
/// let law = Skellam::new(2.0_f64, 3.0)?;
/// assert_eq!(law.signed_count_support(), (None, None));
/// let nonnegative = Skellam::new(2.0_f64, 0.0)?;
/// assert_eq!(nonnegative.signed_count_support(), (Some(0), None));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait SignedCountSupport {
    /// Returns the inclusive bounds of the support; `None` is an unbounded end.
    fn signed_count_support(&self) -> (Option<i64>, Option<i64>);
}

// ---------------------------------------------------------------------------
// Quasi-random trait
// ---------------------------------------------------------------------------

/// A quasi-random (low-discrepancy) sequence of points in the unit cube `[0, 1)^d`.
///
/// Points have [`Self::dim`] coordinates in `[0, 1)`. Batches store points row
/// by row. [`QmcSequence`](crate::QmcSequence) implements it for every family
/// of [`QmcPlan`](crate::QmcPlan).
///
/// ```
/// use scioxide_random::{QmcPlan, QuasiRng};
/// let plan = QmcPlan::halton(2)?;
/// let mut cursor = plan.sequence::<f64>()?;
/// assert_eq!(cursor.dim(), 2);
/// let points = cursor.points(4)?;
/// assert_eq!(points.len(), 8);
/// assert!(points.iter().all(|x| (0.0..1.0).contains(x)));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait QuasiRng<S: RealScalar> {
    /// Returns the next point as a new vector.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the sequence is exhausted (after index `u64::MAX` for
    ///   the built-in families).
    /// - `AllocationFailure` if the vector cannot be allocated.
    fn next_point(&mut self) -> Result<Vec<S>, SciOxideError>;

    /// Writes the next point to `out`, which must have length [`Self::dim`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out.len() != dim()`.
    /// - `InvalidInput` if the sequence is exhausted.
    fn next_point_into(&mut self, out: &mut [S]) -> Result<(), SciOxideError>;

    /// Returns the number of coordinates of each point.
    fn dim(&self) -> usize;

    /// Advances the sequence by `n` points.
    ///
    /// Skipping to just past the last point leaves the sequence exhausted. A
    /// zero skip succeeds even when exhausted.
    ///
    /// # Errors
    ///
    /// - Returns an error, without changing the sequence, if the skip passes
    ///   the end of the sequence.
    fn skip(&mut self, n: u64) -> Result<(), SciOxideError>;

    /// Moves the sequence to point `index`, forward or backward.
    ///
    /// Built-in sequences accept every `u64` index, including `u64::MAX`, and
    /// seeking from the exhausted state makes points available again. Unlike
    /// [`SkipAhead::skip_to`], this can rewind. It does not change the
    /// randomization of the plan.
    ///
    /// # Errors
    ///
    /// - An implementation with a smaller index range rejects indices outside
    ///   it, without changing the sequence. Built-in sequences do not fail.
    ///
    /// ```
    /// use scioxide_random::{QmcPlan, QuasiRng};
    /// let plan = QmcPlan::sobol(1)?;
    /// let mut cursor = plan.sequence::<f64>()?;
    /// cursor.skip_to(u64::MAX)?;
    /// cursor.skip(1)?;
    /// assert_eq!(cursor.position().next_index(), None);
    /// cursor.skip_to(0)?;
    /// assert_eq!(cursor.next_point()?, [0.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn skip_to(&mut self, index: u64) -> Result<(), SciOxideError>;

    /// Writes consecutive points to `output`, one point per `dim()` values.
    ///
    /// An empty output does not advance the sequence. `policy` chooses what is
    /// written and whether the sequence is rewound on error
    /// ([batch sampling](crate#batch-sampling)). The workspace size is given by
    /// [`SamplingPolicy::workspace_requirements`]. On success the points equal
    /// those of [`Self::points`].
    ///
    /// # Errors
    ///
    /// - [`scioxide_core::SciOxideErrorKind::DimensionMismatch`] if `output.len()` is not an exact
    ///   multiple of `dim()`.
    /// - [`scioxide_core::SciOxideErrorKind::InvalidInput`] if an implementation reports zero
    ///   dimensions.
    /// - Any error returned by [`Self::next_point_into`].
    fn points_into(
        &mut self,
        policy: SamplingPolicy<impl StatePolicy<Self>>,
        output: &mut [S],
        workspace: &mut SamplingWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let dim = self.dim();
        if dim == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "quasi-random generators require dim() > 0",
            ));
        }

        let remainder = output.len() % dim;
        if remainder != 0 {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(dim),
                scioxide_core::Dimensions::one(remainder),
            ));
        }

        sample_rows(
            self,
            policy,
            output,
            dim,
            S::ZERO,
            workspace,
            usize::MAX,
            |generator, row| generator.next_point_into(row),
        )
    }

    /// Returns the next `n` points in a new `Vec`, point by point.
    ///
    /// The result is allocated once, before the first point, and returned only
    /// on success. After an error the sequence stays advanced past the points
    /// it produced; to rewind it, use [`Self::points_into`] with
    /// [`SamplingPolicy::rollback`].
    ///
    /// # Errors
    ///
    /// - [`scioxide_core::SciOxideErrorKind::InvalidInput`] if an implementation reports zero
    ///   dimensions.
    /// - [`scioxide_core::SciOxideErrorKind::SizeOverflow`] if `n * dim()` cannot be represented.
    /// - [`scioxide_core::SciOxideErrorKind::AllocationFailure`] if output storage cannot be
    ///   reserved.
    /// - Any error of [`Self::next_point_into`].
    fn points(&mut self, n: usize) -> Result<Vec<S>, SciOxideError> {
        let dim = self.dim();
        if dim == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "quasi-random generators require dim() > 0",
            ));
        }
        let len = scioxide_core::backend::checked_mul(
            n,
            dim,
            scioxide_core::quantity!("quasi_random_batch_output"),
        )?;
        let mut values = scioxide_core::backend::try_filled(
            len,
            S::ZERO,
            scioxide_core::quantity!("quasi_random_output"),
        )?;
        values
            .chunks_exact_mut(dim)
            .try_for_each(|row| self.next_point_into(row))?;
        Ok(values)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    struct EndpointRng {
        values: [f64; 2],
        position: usize,
    }

    struct ConstantRng(f64);

    impl BitGenerator for ConstantRng {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl Rng<f64> for ConstantRng {
        fn next_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> {
            Ok(self.0)
        }
    }

    impl EndpointRng {
        const fn zero_then_half() -> Self {
            Self {
                values: [0.0, 0.5],
                position: 0,
            }
        }
    }

    impl BitGenerator for EndpointRng {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl Rng<f64> for EndpointRng {
        fn next_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> {
            let value = self.values[self.position];
            self.position += 1;
            Ok(value)
        }
    }

    struct DirectDistribution;

    impl Distribution<f64> for DirectDistribution {
        fn sample<R: Rng<f64>>(&self, rng: &mut R) -> Result<f64, SciOxideError> {
            Ok(rng.next_float().unwrap())
        }
    }

    struct FailsOnHalf;

    impl Distribution<f64> for FailsOnHalf {
        fn sample<R: Rng<f64>>(&self, rng: &mut R) -> Result<f64, SciOxideError> {
            let value = rng.next_float().unwrap();
            if value == 0.5 {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("test_sample"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "the test distribution rejects one half",
                ));
            }
            Ok(value)
        }
    }

    #[test]
    fn next_float_preserves_the_valid_zero_endpoint() {
        let mut rng = EndpointRng::zero_then_half();
        assert_eq!(rng.next_float().unwrap(), 0.0);
        assert_eq!(rng.position, 1);
    }

    #[test]
    fn next_open_float_rejects_zero_instead_of_clamping_it() {
        let mut rng = EndpointRng::zero_then_half();
        assert_eq!(rng.next_open_float().unwrap(), 0.5);
        assert_eq!(rng.position, 2);
    }

    #[test]
    fn next_open_float_accepts_the_greatest_value_below_one() {
        let mut rng = ConstantRng(1.0_f64.next_down());
        assert_eq!(rng.next_open_float().unwrap(), 1.0_f64.next_down());
    }

    #[test]
    #[allow(clippy::panic)]
    fn next_open_float_bounds_repeated_zero_rejection() {
        let mut rng = ConstantRng(0.0);
        let error = rng.next_open_float().unwrap_err();
        let scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::IterationLimit,
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
    }

    #[test]
    fn default_sample_n_into_preserves_domain_values_in_order() {
        let mut rng = EndpointRng::zero_then_half();
        let mut output = [f64::NAN; 2];
        let mut workspace = SamplingWorkspace::new();

        assert!(
            DirectDistribution
                .sample_n_into(
                    &mut rng,
                    crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                    &mut output,
                    &mut workspace
                )
                .is_ok()
        );

        assert_eq!(output, [0.0, 0.5]);
    }

    #[test]
    fn default_sample_n_into_preserves_output_on_error() {
        let mut rng = EndpointRng::zero_then_half();
        let mut output = [f64::NAN; 2];
        let mut workspace = SamplingWorkspace::new();

        let error = FailsOnHalf
            .sample_n_into(
                &mut rng,
                crate::SamplingPolicy::advance_on_error(crate::OutputPolicy::Atomic),
                &mut output,
                &mut workspace,
            )
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);

        assert!(output.iter().all(|value| value.is_nan()));
    }
}
