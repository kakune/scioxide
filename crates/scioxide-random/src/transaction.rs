//! Output and generator-state policies for batch sampling.

use crate::{SamplingRequest, SamplingRequirements, SamplingWorkspace};
use scioxide_core::SciOxideError;

/// Chooses when a batch writes its samples to the output slice.
///
/// See [`SamplingPolicy`] for an example of both policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputPolicy {
    /// Writes the whole batch only if every draw succeeds; on error the output is unchanged.
    Atomic,
    /// Writes each completed sample (a row for vector-valued samples, a swap for a shuffle)
    /// as it goes. Completed samples stay written after a later error, even with
    /// [`Rollback`].
    Incremental,
}

/// A generator whose state can be saved and restored in memory, as [`Rollback`] requires.
///
/// A snapshot holds all state that affects later outputs, including the stream
/// identity. Taking a snapshot does not advance the generator. Restoring a
/// snapshot taken from the same generator cannot fail and does not allocate.
/// Side effects outside the generator are not restored. Built-in generators
/// snapshot by copying their state. For portable checkpoints that survive the
/// process, use [`SerializableState`](crate::SerializableState).
///
/// ```
/// use scioxide_random::{BitGenerator, SeedableRng, SnapshotGenerator, SplitMix64};
/// let mut rng = SplitMix64::from_seed(42);
/// let snapshot = rng.snapshot()?;
/// let first = rng.next_u64()?;
/// rng.restore_snapshot(snapshot);
/// assert_eq!(rng.next_u64()?, first);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait SnapshotGenerator {
    /// Saved generator state; it does not borrow the generator.
    type Snapshot;
    /// Returns the current state without changing it.
    ///
    /// # Errors
    /// Returns any failure of the implementation to save its state, such as an
    /// allocation failure. Built-in generators do not fail.
    fn snapshot(&self) -> Result<Self::Snapshot, SciOxideError>;
    /// Restores a snapshot taken from this generator. Internal scratch may differ.
    fn restore_snapshot(&mut self, snapshot: Self::Snapshot);
}

impl<G: SnapshotGenerator + ?Sized> SnapshotGenerator for &mut G {
    type Snapshot = G::Snapshot;
    fn snapshot(&self) -> Result<Self::Snapshot, SciOxideError> {
        (**self).snapshot()
    }
    fn restore_snapshot(&mut self, snapshot: Self::Snapshot) {
        (**self).restore_snapshot(snapshot);
    }
}

macro_rules! clone_snapshot {
    ([$($generics:tt)*] $engine:ty) => {
        impl $($generics)* crate::SnapshotGenerator for $engine {
            type Snapshot = Self;
            fn snapshot(&self) -> Result<Self, scioxide_core::SciOxideError> {
                Ok(self.clone())
            }
            fn restore_snapshot(&mut self, snapshot: Self) {
                *self = snapshot;
            }
        }
    };
    ($($engine:ty),+ $(,)?) => {$(
        crate::transaction::clone_snapshot!([] $engine);
    )+};
}
pub(crate) use clone_snapshot;
clone_snapshot!(
    crate::SplitMix64,
    crate::Lcg64,
    crate::Pcg64,
    crate::Mrg32k3a,
    crate::Xoshiro128StarStar,
    crate::Xoshiro256StarStar
);

/// Leaves the generator advanced by the draws made before an error.
///
/// Works with every generator, since it needs no snapshot. Select it with
/// [`SamplingPolicy::advance_on_error`].
///
/// ```
/// use scioxide_random::{BitGenerator, BoundedSubstream, Distribution, OutputPolicy,
///     SamplingPolicy, SamplingWorkspace, SeedableRng, SplitMix64, Uniform};
/// let base = SplitMix64::from_seed(42);
/// let mut rng = BoundedSubstream::for_chunk(&base, 0, 3)?; // three draws available
/// let law = Uniform::new(0.0_f64, 1.0)?;
/// let policy = SamplingPolicy::advance_on_error(OutputPolicy::Atomic);
/// let mut output = [0.0; 4];
/// let mut workspace = SamplingWorkspace::new();
/// assert!(law.sample_n_into(&mut rng, policy, &mut output, &mut workspace).is_err());
/// assert_eq!(rng.position(), Some(3)); // the three completed draws stay consumed
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AdvanceOnError;
/// Restores the generator to its state before the call if the call returns an error.
///
/// The [`OutputPolicy`] decides separately what stays written to the output.
/// Requires a [`SnapshotGenerator`]; select it with [`SamplingPolicy::rollback`].
///
/// ```
/// use scioxide_random::{BitGenerator, BoundedSubstream, Distribution, OutputPolicy,
///     SamplingPolicy, SamplingWorkspace, SeedableRng, SplitMix64, Uniform};
/// let base = SplitMix64::from_seed(42);
/// let mut rng = BoundedSubstream::for_chunk(&base, 0, 3)?; // three draws available
/// let law = Uniform::new(0.0_f64, 1.0)?;
/// let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
/// let mut workspace = SamplingWorkspace::new();
/// assert!(law.sample_n_into(&mut rng, policy, &mut [0.0; 4], &mut workspace).is_err());
/// assert_eq!(rng.position(), Some(0)); // restored
/// law.sample_n_into(&mut rng, policy, &mut [0.0; 3], &mut workspace)?;
/// assert_eq!(rng.position(), Some(3));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Rollback;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::AdvanceOnError {}
    impl Sealed for super::Rollback {}
}

/// Generator-state policy on error, implemented only by [`AdvanceOnError`] and [`Rollback`].
///
/// [`Rollback`] implements it only for a [`SnapshotGenerator`].
///
/// See [`AdvanceOnError`] and [`Rollback`] for examples.
pub trait StatePolicy<G: ?Sized>: sealed::Sealed {
    /// State saved before the batch.
    type Checkpoint;
    /// Saves the state needed to handle an error, before drawing.
    ///
    /// # Errors
    /// Returns snapshot capture errors.
    fn capture(generator: &G) -> Result<Self::Checkpoint, SciOxideError>;
    /// Applies the policy after the batch returned an error.
    fn on_error(generator: &mut G, checkpoint: Self::Checkpoint);
}
impl<G: ?Sized> StatePolicy<G> for AdvanceOnError {
    type Checkpoint = ();
    fn capture(_: &G) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn on_error(_: &mut G, (): ()) {}
}
impl<G: SnapshotGenerator + ?Sized> StatePolicy<G> for Rollback {
    type Checkpoint = G::Snapshot;
    fn capture(generator: &G) -> Result<Self::Checkpoint, SciOxideError> {
        generator.snapshot()
    }
    fn on_error(generator: &mut G, checkpoint: Self::Checkpoint) {
        generator.restore_snapshot(checkpoint);
    }
}

/// Output policy and generator-state policy of one batch that fills a slice you provide.
///
/// Every form that draws into a slice you provide (`sample_n_into`,
/// `points_into`, [`shuffle_in_place`](crate::shuffle_in_place), …) takes a
/// policy after the generator. The form validates its arguments and reserves
/// the workspace before it saves generator state or draws. An empty batch
/// neither saves state nor draws. The policies apply to returned errors, not
/// to panics. A successful batch draws the same words in the same order as the
/// same number of single draws, rejected proposals included. A policy has no
/// default: the caller chooses both what the output holds and where the
/// generator stands after a failure. Forms that return a new `Vec` take no
/// policy and leave the generator advanced on error. See
/// [batch sampling](crate#batch-sampling).
///
/// Reserve a workspace once, then reuse it for every
/// [`Distribution::sample_n_into`] batch. Below, a [`BoundedSubstream`] holds
/// six draws, so the second batch of four fails after two:
///
/// ```
/// use scioxide_random::{BitGenerator, BoundedSubstream, Distribution, OutputPolicy,
///     SamplingPolicy, SamplingRequest, SeedableRng, SplitMix64, Uniform};
/// let law = Uniform::new(1.0_f64, 2.0)?;
/// let mut rng = BoundedSubstream::for_chunk(&SplitMix64::from_seed(42), 0, 6)?;
/// let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
/// let mut workspace = policy.workspace(SamplingRequest::scalars(4))?;
/// let mut output = [0.0; 4];
/// law.sample_n_into(&mut rng, policy, &mut output, &mut workspace)?;
/// let first = output;
/// // Atomic + rollback: nothing is written and the generator is restored.
/// assert!(law.sample_n_into(&mut rng, policy, &mut output, &mut workspace).is_err());
/// assert_eq!((output, rng.position()), (first, Some(4)));
/// // Incremental + advance: the two completed draws are written and consumed.
/// let policy = SamplingPolicy::advance_on_error(OutputPolicy::Incremental);
/// assert!(law.sample_n_into(&mut rng, policy, &mut output, &mut workspace).is_err());
/// assert_eq!((&output[2..], rng.position()), (&first[2..], Some(6)));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// [`Distribution::sample_n_into`]: crate::Distribution::sample_n_into
/// [`BoundedSubstream`]: crate::BoundedSubstream
#[derive(Debug)]
pub struct SamplingPolicy<P> {
    output: OutputPolicy,
    state: core::marker::PhantomData<P>,
}
// The policy only names its state policy, so it is `Copy` for every `P`.
impl<P> Clone for SamplingPolicy<P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P> Copy for SamplingPolicy<P> {}
impl SamplingPolicy<AdvanceOnError> {
    /// Creates a policy that leaves the generator advanced on error ([`AdvanceOnError`]).
    pub const fn advance_on_error(output: OutputPolicy) -> Self {
        Self {
            output,
            state: core::marker::PhantomData,
        }
    }
}
impl SamplingPolicy<Rollback> {
    /// Creates a policy that restores the generator on error ([`Rollback`]).
    ///
    /// A generator without [`SnapshotGenerator`] does not compile with it:
    /// ```compile_fail
    /// use scioxide_random::{BitGenerator, Distribution, Normal, OutputPolicy, Rng,
    ///     SamplingPolicy, SamplingWorkspace};
    /// struct Provider;
    /// impl BitGenerator for Provider {
    ///     fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> { Ok({ 1 }) }
    ///     fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> { Ok({ 1 }) }
    ///     fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> { self.next_u32() }
    ///     fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> { self.next_u64() }
    ///     fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> { Ok({ out.fill(1); }) }
    /// }
    /// impl Rng<f64> for Provider { fn next_float(&mut self) -> Result<f64, scioxide_core::SciOxideError> { Ok({ 0.5 }) } }
    /// let policy = SamplingPolicy::rollback(OutputPolicy::Atomic);
    /// let mut workspace = SamplingWorkspace::new();
    /// Normal::<f64>::standard()
    ///     .sample_n_into(&mut Provider, policy, &mut [0.0], &mut workspace)
    ///     .unwrap();
    /// ```
    pub const fn rollback(output: OutputPolicy) -> Self {
        Self {
            output,
            state: core::marker::PhantomData,
        }
    }
}
impl<P> SamplingPolicy<P> {
    /// Returns the output policy.
    pub const fn output_policy(&self) -> OutputPolicy {
        self.output
    }

    /// Returns the workspace size a batch of `request` needs, without a byte limit.
    ///
    /// All-or-nothing output stages every element. Incremental output stages
    /// nothing for scalar samples and one row for vector-valued samples. A
    /// shuffle of `len` elements stages `len` elements, or none when
    /// incremental. Quasi-random sequences may also keep their own point scratch.
    ///
    /// # Errors
    /// - `InvalidInput` if the row width is zero.
    /// - `SizeOverflow` if the element count overflows `usize`.
    pub fn workspace_requirements(
        &self,
        request: SamplingRequest,
    ) -> Result<SamplingRequirements, SciOxideError> {
        Ok(SamplingRequirements::new(request.elements(self.output)?))
    }

    /// Creates a workspace reserved for `request`.
    ///
    /// # Errors
    /// The errors of [`Self::workspace_requirements`], or `AllocationFailure`.
    pub fn workspace<T>(
        &self,
        request: SamplingRequest,
    ) -> Result<SamplingWorkspace<T>, SciOxideError> {
        let mut workspace = SamplingWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    pub(crate) fn run<G: ?Sized, T>(
        self,
        generator: &mut G,
        operation: impl FnOnce(&mut G) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError>
    where
        P: StatePolicy<G>,
    {
        run::<G, T, P>(generator, operation)
    }
}

/// Runs a sampling operation under one generator-state policy.
pub(crate) fn run<G: ?Sized, T, P: StatePolicy<G>>(
    generator: &mut G,
    operation: impl FnOnce(&mut G) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let checkpoint = P::capture(generator)?;
    match operation(generator) {
        Ok(value) => Ok(value),
        Err(error) => {
            P::on_error(generator, checkpoint);
            Err(error)
        }
    }
}
