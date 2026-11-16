//! Sampled objectives, sample keys and replicated batch estimates.
use super::*;
use alloc::vec::Vec;

use scioxide_core::{
    Budget, BudgetKind, EvaluationRole, ExecutionPolicy,
    backend::{PartitionSlot, checked_mul, try_filled},
};
use scioxide_random::{BitGenerator, Philox4x64, SplittableRng};
use scioxide_stats::{MomentOrder, OnlineMoments};

/// States how repeated evaluations of a sampled objective at one point relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseModel {
    /// Every evaluation at a point returns the same value; the reported
    /// sampling error is zero.
    Deterministic,
    /// Evaluations with distinct samples are independent, with a common finite
    /// mean and variance. The solver assumes, without checking, that sampled
    /// objectives and gradients are unbiased.
    Independent,
}
/// Identifies one sample: the seed, what the sample is for, and its sequence
/// number.
///
/// The key does not depend on the number of workers, so results do not
/// either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizationSampleKey {
    /// Root seed from the solver options.
    pub seed: u64,
    /// Purpose of the sample: 0 for random variation, 1 for an objective, 2 for
    /// a gradient, 3 for the trust-region base point, 4 for a trust-region
    /// trial point.
    pub role: u8,
    /// Sequence number of the sample. It increases by one per sample and is
    /// saved in checkpoints.
    pub ordinal: u64,
}
/// Maps a [`OptimizationSampleKey`] to a `u64` sample token.
///
/// The solver passes the token to [`SampledObjective::sample_into`] or
/// [`SampledGradient::gradient_sample_into`], which may interpret it as a
/// dataset index, a mini-batch seed or a noise seed. Equal keys must give equal
/// tokens. Each call counts as one sample and one evaluation, even if it fails.
pub trait OptimizationSampler: Sync {
    /// Returns a value that identifies the sampler configuration. A checkpoint
    /// can be resumed only with a sampler that returns the same value.
    fn configuration_token(&self) -> u128;
    /// Returns the sample token for `key`.
    ///
    /// # Errors
    ///
    /// Implementations may return any error; the solver passes it through.
    fn sample(&self, key: OptimizationSampleKey) -> Result<u64, SciOxideError>;
}
/// Draws sample tokens from the Philox4x64-10 counter-based generator.
///
/// The token for a key is the first output of the substream numbered by the
/// key's sequence number, forked by its purpose, under the key's seed.
///
/// # References
///
/// J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random
/// numbers: as easy as 1, 2, 3", *Proc. SC11*, 2011.
/// <https://doi.org/10.1145/2063384.2063405>
#[derive(Debug, Clone, Copy, Default)]
pub struct PhiloxSampler;
impl OptimizationSampler for PhiloxSampler {
    fn configuration_token(&self) -> u128 {
        0x5048494c4f583031
    }
    fn sample(&self, key: OptimizationSampleKey) -> Result<u64, SciOxideError> {
        let root = Philox4x64::from_key([key.seed, 0]);
        root.substream(key.ordinal, 0)?.fork(key.role)?.next_u64()
    }
}
/// Defines a noisy or sampled objective `F(x, ξ)` with one or more real
/// components, all minimized.
///
/// The solver estimates `f(x) = E[F(x, ξ)]` by averaging evaluations with
/// distinct sample tokens `ξ` from an [`OptimizationSampler`]. With
/// [`NoiseModel::Independent`], distinct tokens must give independent
/// observations. Variables may be real or complex.
pub trait SampledObjective<S: NumericScalar>: Sync {
    /// Returns the number of variables.
    fn dimension(&self) -> usize;
    /// Returns the number of objective components; at least one.
    fn objectives(&self) -> usize;
    /// Returns how repeated evaluations relate; the default is
    /// [`NoiseModel::Independent`].
    fn noise_model(&self) -> NoiseModel {
        NoiseModel::Independent
    }
    /// Writes every component of `F(x, sample)` to `output`.
    ///
    /// # Errors
    ///
    /// Implementations may return any error; the solver passes it through.
    /// The solver rejects a non-finite output with `NonFinite`.
    fn sample_into(
        &self,
        x: &[S],
        sample: u64,
        output: &mut [S::Real],
    ) -> Result<(), SciOxideError>;
}
/// Adds a sampled gradient `∇ₓF(x, ξ)` to a single-component
/// [`SampledObjective`].
///
/// For complex variables the gradient follows the real inner product, so that
/// `F(x + h, ξ) ≈ F(x, ξ) + Re(gᴴ h)`. The stochastic solvers require exactly one
/// objective component.
pub trait SampledGradient<S: NumericScalar>: SampledObjective<S> {
    /// Writes every component of the gradient for `sample` to `output`.
    ///
    /// # Errors
    ///
    /// Implementations may return any error; the solver passes it through.
    /// The solver rejects a non-finite output with `NonFinite`.
    fn gradient_sample_into(
        &self,
        x: &[S],
        sample: u64,
        output: &mut [S],
    ) -> Result<(), SciOxideError>;
}
/// Wraps a [`SmoothGradient`](crate::SmoothGradient) as a deterministic
/// [`SampledGradient`] that ignores the sample token.
pub struct DeterministicGradient<'a, F: ?Sized>(pub &'a F);
impl<S: NumericScalar, F: crate::SmoothGradient<S> + Sync + ?Sized> SampledObjective<S>
    for DeterministicGradient<'_, F>
{
    fn dimension(&self) -> usize {
        self.0.dimension()
    }
    fn objectives(&self) -> usize {
        1
    }
    fn noise_model(&self) -> NoiseModel {
        NoiseModel::Deterministic
    }
    fn sample_into(&self, x: &[S], _: u64, output: &mut [S::Real]) -> Result<(), SciOxideError> {
        if output.len() != 1 {
            return Err(invalid(
                scioxide_core::argument!("output"),
                "scalar objective output shape",
            ));
        }
        output[0] = self.0.value(x)?;
        Ok(())
    }
}
impl<S: NumericScalar, F: crate::SmoothGradient<S> + Sync + ?Sized> SampledGradient<S>
    for DeterministicGradient<'_, F>
{
    fn gradient_sample_into(&self, x: &[S], _: u64, output: &mut [S]) -> Result<(), SciOxideError> {
        self.0.gradient_into(x, output)
    }
}
/// Configures how many replicated samples form one estimate, and the error
/// margin used in decisions.
///
/// An estimate averages `minimum` to `maximum` samples. Its error margin is the
/// multiplier times the sample standard error. When a decision cannot be made
/// within that margin, an adaptive count doubles the samples, up to `maximum`.
/// See also [Termination](crate#termination).
#[derive(Debug, Clone, Copy)]
pub struct ReplicationOptions<R: RealScalar> {
    pub(super) minimum: usize,
    pub(super) maximum: usize,
    pub(super) margin: R,
}
impl<R: RealScalar> ReplicationOptions<R> {
    /// Uses exactly `count` samples per estimate, with an error margin of two
    /// standard errors. The margin is not a confidence level.
    pub fn fixed(count: Budget) -> Self {
        Self {
            minimum: count.get(),
            maximum: count.get(),
            margin: R::TWO,
        }
    }
    /// Uses `minimum` samples per estimate, doubled while a decision is
    /// unresolved, up to `maximum`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `maximum < minimum`.
    pub fn adaptive(minimum: Budget, maximum: Budget) -> Result<Self, SciOxideError> {
        if maximum.get() < minimum.get() {
            return Err(invalid(
                scioxide_core::argument!("maximum"),
                "maximum replicates below minimum",
            ));
        }
        Ok(Self {
            maximum: maximum.get(),
            ..Self::fixed(minimum)
        })
    }
    /// Sets the multiplier of the sample standard error. The solver constructor
    /// requires it to be positive and finite.
    #[must_use]
    pub const fn with_error_multiplier(mut self, multiplier: R) -> Self {
        self.margin = multiplier;
        self
    }
    /// Returns the minimum number of samples per estimate.
    pub const fn minimum(self) -> usize {
        self.minimum
    }
    /// Returns the maximum number of samples per estimate.
    pub const fn maximum(self) -> usize {
        self.maximum
    }
    /// Returns the multiplier of the sample standard error.
    pub const fn error_multiplier(self) -> R {
        self.margin
    }
    pub(super) fn validate(self) -> Result<(), SciOxideError> {
        crate::proximal::positive(scioxide_core::argument!("multiplier"), self.margin)
    }
}

pub(super) struct Batch<S: NumericScalar> {
    width: usize,
    raw: Vec<S>,
    pub mean: Vec<S>,
    pub error: Vec<S::Real>,
    pub variance: Vec<S::Real>,
    pub count: usize,
    pub error_available: bool,
}
impl<S: NumericScalar> Batch<S> {
    pub(super) fn new(width: usize, maximum: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            width,
            raw: try_filled(
                checked_mul(width, maximum, scioxide_core::quantity!("batch.raw"))?,
                S::ZERO,
                scioxide_core::quantity!("batch.raw"),
            )?,
            mean: try_filled(width, S::ZERO, scioxide_core::quantity!("batch.mean"))?,
            error: try_filled(
                width,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("batch.error"),
            )?,
            variance: try_filled(
                width,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("batch.variance"),
            )?,
            count: 0,
            error_available: false,
        })
    }
    pub(super) fn reset(&mut self) {
        self.count = 0;
        self.error_available = false;
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn extend(
        &mut self,
        total: usize,
        model: NoiseModel,
        seed: u64,
        purpose: u8,
        cursor: &mut u64,
        sampler: &(impl OptimizationSampler + ?Sized),
        role: EvaluationRole,
        policy: ExecutionPolicy,
        context: &mut SolveContext<'_, '_>,
        evaluate: &(impl Fn(u64, &mut [S]) -> Result<(), SciOxideError> + Sync),
    ) -> Result<Option<OptimizeTermination>, SciOxideError> {
        if total < self.count || total > self.raw.len() / self.width {
            return Err(invalid(
                scioxide_core::argument!("work"),
                "replicate workspace size",
            ));
        }
        if let Some(reason) = stop(context)? {
            return Ok(Some(reason));
        }
        let added = total - self.count;
        if !context.can_consume(BudgetKind::Sample, added)?.is_granted() {
            return Ok(Some(OptimizeTermination::SampleLimit));
        }
        if !context
            .can_consume(
                BudgetKind::Evaluation,
                checked_mul(added, 2, scioxide_core::quantity!("provider_evaluations"))?,
            )?
            .is_granted()
        {
            return Ok(Some(OptimizeTermination::EvaluationLimit));
        }
        let first = *cursor;
        *cursor = cursor
            .checked_add(u64::try_from(added).map_err(|_| {
                SciOxideError::size_overflow(scioxide_core::quantity!("sample_cursor"))
            })?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("sample_cursor"))
            })?;
        // Admission precedes dispatch. Counts include reserved slots after executor failure.
        for _ in 0..added {
            if !context.try_consume(BudgetKind::Sample, 1)?.is_granted() {
                return Err(crate::context::budget_error(
                    context,
                    BudgetKind::Sample,
                    "sample batch admission failed",
                ));
            }
            if !context
                .record_provider(EvaluationRole::Sample)?
                .is_granted()
                || !context.record_provider(role)?.is_granted()
            {
                return Err(crate::context::budget_error(
                    context,
                    BudgetKind::Evaluation,
                    "sample provider admission failed",
                ));
            }
        }
        let values = &mut self.raw[self.count * self.width..total * self.width];
        values.fill(S::from_real(S::Real::NAN));
        let execution = context.execution();
        let limit = core::num::NonZeroUsize::MIN.saturating_add(63);
        let plan = execution.plan(
            policy,
            added,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: core::num::NonZeroUsize::MIN,
                worker_limit: limit,
            },
        );
        let mut slots: [_; 64] = core::array::from_fn(|_| PartitionSlot::empty());
        let mut tail = values;
        for partition in plan.partitions() {
            let (out, rest) = tail.split_at_mut(partition.range().len() * self.width);
            tail = rest;
            slots[partition.worker_index()] = PartitionSlot::new(out);
        }
        let width = self.width;
        let report = execution.execute_partitioned(
            plan,
            &slots[..plan.worker_count()],
            &|partition, out, _| {
                for (ordinal, row) in partition.range().zip(out.chunks_exact_mut(width)) {
                    // Cooperative checks occur before each callback; providers own in-call cancellation.
                    if let Some(reason) = execution.stop_reason()? {
                        return Err(SciOxideError::execution_stopped(reason));
                    }
                    let token = sampler.sample(OptimizationSampleKey {
                        seed,
                        role: purpose,
                        ordinal: first + ordinal as u64,
                    })?;
                    evaluate(token, row)?;
                    finite_vector(row)?;
                }
                Ok(())
            },
        );
        if let Err(error) = report {
            match error.view() {
                scioxide_core::SciOxideErrorView::ExecutionStopped { reason, .. } => {
                    return Ok(Some(match reason {
                        ExecutionStopReason::DeadlineReached => OptimizeTermination::TimeLimit,
                        _ => OptimizeTermination::Cancelled,
                    }));
                }
                _ => return Err(error),
            }
        }
        if let Some(reason) = stop(context)? {
            return Ok(Some(reason));
        }
        self.count = total;
        self.error_available = model == NoiseModel::Deterministic || total > 1;
        for j in 0..self.width {
            let mut re = OnlineMoments::new(MomentOrder::Second);
            let mut im = OnlineMoments::new(MomentOrder::Second);
            for row in self.raw[..total * self.width].chunks_exact(self.width) {
                re.observe(row[j].real_component())?;
                im.observe(row[j].imaginary_component())?;
            }
            let re = re.snapshot();
            let im = im.snapshot();
            self.mean[j] = S::from_components(
                re.mean()?
                    .ok_or_else(|| invalid(scioxide_core::argument!("work"), "empty mean"))?,
                im.mean()?
                    .ok_or_else(|| invalid(scioxide_core::argument!("work"), "empty mean"))?,
            );
            let (variance, error) = if model == NoiseModel::Deterministic {
                (
                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                    <_ as scioxide_core::AlgebraicScalar>::ZERO,
                )
            } else {
                let v = finite(
                    re.sample_variance()?
                        .unwrap_or(<_ as scioxide_core::AlgebraicScalar>::ZERO)
                        + im.sample_variance()?
                            .unwrap_or(<_ as scioxide_core::AlgebraicScalar>::ZERO),
                )?;
                let e = crate::proximal::norm(&[
                    S::from_real(
                        re.standard_error_of_sample_mean()?
                            .unwrap_or(<_ as scioxide_core::AlgebraicScalar>::ZERO),
                    ),
                    S::from_real(
                        im.standard_error_of_sample_mean()?
                            .unwrap_or(<_ as scioxide_core::AlgebraicScalar>::ZERO),
                    ),
                ])?;
                (v, e)
            };
            self.variance[j] = variance;
            self.error[j] = error;
        }
        Ok(None)
    }
}

/// Counter-based variation stream: word `k` is the sampler's variation token at ordinal `cursor + k`.
///
/// Every native or uniform word is one sampler invocation; 32-bit words keep
/// the high half of the token.
pub(super) struct SampleStream<'a, T: ?Sized> {
    pub(super) sampler: &'a T,
    pub(super) seed: u64,
    pub(super) cursor: &'a mut u64,
}
impl<T: OptimizationSampler + ?Sized> BitGenerator for SampleStream<'_, T> {
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        let ordinal = *self.cursor;
        *self.cursor = ordinal.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("sample_cursor"))
        })?;
        self.sampler.sample(OptimizationSampleKey {
            seed: self.seed,
            role: 0,
            ordinal,
        })
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok((self.next_u64()? >> 32) as u32)
    }
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.next_u64()
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.next_u32()
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
        for chunk in output.chunks_mut(8) {
            let bytes = self.next_u64()?.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        Ok(())
    }
}

/// Variation draws: one sample and one sample-provider evaluation each, counted
/// before the stream's cursor advances.
pub(super) struct VariationAdmission<'a, 's, 'c> {
    pub(super) context: &'a mut SolveContext<'s, 'c>,
}
impl crate::random::DrawAdmission for VariationAdmission<'_, '_, '_> {
    fn admit(&mut self) -> Result<Option<OptimizeTermination>, SciOxideError> {
        if let Some(s) = stop(self.context)? {
            return Ok(Some(s));
        }
        if !self
            .context
            .can_consume(BudgetKind::Sample, 1)?
            .is_granted()
        {
            return Ok(Some(OptimizeTermination::SampleLimit));
        }
        if !self
            .context
            .can_consume(BudgetKind::Evaluation, 1)?
            .is_granted()
        {
            return Ok(Some(OptimizeTermination::EvaluationLimit));
        }
        if !self
            .context
            .try_consume(BudgetKind::Sample, 1)?
            .is_granted()
            || !self
                .context
                .record_provider(EvaluationRole::Sample)?
                .is_granted()
        {
            return Err(invalid(
                scioxide_core::argument!("context"),
                "sample admission",
            ));
        }
        Ok(None)
    }
    fn refusal(&self, _: OptimizeTermination) -> SciOxideError {
        invalid(
            scioxide_core::argument!("context"),
            "variation sample not admitted",
        )
    }
}
