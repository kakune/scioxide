//! Exact-length complex FFT planning and execution.
//!
//! [`ComplexCore`] builds an immutable, deterministic execution plan. Lengths
//! composed from radices 2, 3, 4, 5, and 7 use mixed-radix Cooley-Tukey
//! stages. Suitable prime lengths use Rader's algorithm, while remaining
//! lengths use Bluestein's algorithm with a checked smooth convolution size.

use crate::numeric::reduced_root;
mod bluestein;
mod planner;
mod rader;
mod smooth;
use crate::scaling::scale_complex;
use crate::{FftAlgorithmKind, FftPlanDiagnostics};
use alloc::vec::Vec;
use bluestein::BluesteinPlan;
use core::mem::size_of;
use core::num::NonZeroUsize;
use planner::*;
use rader::RaderPlan;
use smooth::SmoothPlan;

use scioxide_core::Complex;
use scioxide_core::ExecutionPlan;
use scioxide_core::backend::PartitionSlot;
use scioxide_core::backend::{KernelBackend, KernelDispatch};
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{Argument, ExecutionContext, ExecutionPolicy, SciOxideError};
use scioxide_linalg::{VectorView, VectorViewMut};

use crate::lanes::{LaneMode, Lanes};
use crate::numeric::scalar_ratio_u128;
use crate::scaling::{BinaryScale, complex_max, range_exponent, restore_complex};
use crate::validation::check_complex_slice_finite;

/// Precomputed exact-length complex FFT plan.
///
/// The plan owns algorithm selection, permutation metadata, compact stage
/// twiddles, private convolution subplans, backend dispatch, and execution
/// policy. Plan construction performs all structural and allocation checks
/// before returning. Supplied-scratch execution performs no heap allocation
/// when forced sequential. Automatic and explicit parallel execution remain
/// sequential below 65,536 complex values; larger smooth transforms submit
/// deterministic partitions to the caller's reusable executor.
///
/// For bounded inputs, normalized round trips are tested against `64 * epsilon(S) * max(1,
/// log2(n))`; scale that absolute bound by the maximum input-component magnitude for larger
/// inputs.
#[derive(Debug, Clone)]
#[must_use]
pub(crate) struct ComplexCore<S: scioxide_core::RealScalar> {
    n: usize,
    /// Binary headroom `growth_exponent(n)` of native kernel outputs.
    growth: i32,
    execution_policy: ExecutionPolicy,
    backend: KernelBackend,
    scratch_len: usize,
    algorithm_scratch_len: usize,
    plan_bytes: usize,
    preparation_bytes: usize,
    diagnostics: FftPlanDiagnostics,
    algorithm: FftAlgorithm<S>,
}

#[derive(Debug, Clone)]
enum FftAlgorithm<S: scioxide_core::RealScalar> {
    Smooth(SmoothPlan<S>),
    Rader(RaderPlan<S>),
    Bluestein(BluesteinPlan<S>),
    Provider(crate::backend::ProviderKernel<S>),
}

impl<S: scioxide_core::RealScalar> ComplexCore<S> {
    /// Prepare with an already resolved CPU backend.
    pub(crate) fn with_backend_and_context<'a>(
        n: usize,
        execution_policy: ExecutionPolicy,
        backend: KernelBackend,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
        length: Argument,
        provider: Argument,
    ) -> Result<Self, SciOxideError> {
        let context = context.into();
        Self::build(
            n,
            execution_policy,
            backend,
            context,
            length,
            provider,
            usize::MAX,
        )
    }

    /// Bound native peak preparation and provider storage before allocation.
    pub(crate) fn with_preparation_limit(
        n: usize,
        execution_policy: ExecutionPolicy,
        strategy: crate::FftBackendStrategy,
        context: crate::FftPlanningContext<'_, S>,
        length: Argument,
        provider: Argument,
        preparation_limit: usize,
    ) -> Result<Self, SciOxideError> {
        crate::memory::admit_bytes(
            size_of::<Self>(),
            preparation_limit,
            scioxide_core::quantity!("fft_preparation_bytes"),
        )?;
        Self::build(
            n,
            execution_policy,
            context.resolve(strategy)?,
            context.with_provider_limits(crate::FftProviderLimits {
                preparation_bytes: preparation_limit.saturating_sub(size_of::<Self>()),
                ..crate::FftProviderLimits::default()
            }),
            length,
            provider,
            preparation_limit,
        )
    }

    fn build(
        n: usize,
        execution_policy: ExecutionPolicy,
        backend: KernelBackend,
        context: crate::FftPlanningContext<'_, S>,
        length: Argument,
        provider: Argument,
        preparation_limit: usize,
    ) -> Result<Self, SciOxideError> {
        crate::validation::nonzero(n, length)?;

        checked_mul(
            n,
            size_of::<Complex<S>>(),
            scioxide_core::quantity!("fft_data_bytes"),
        )?;
        let mut preparation_bytes = size_of::<Self>();
        let algorithm = if n > 1 {
            context
                .prepare(n, execution_policy, provider)?
                .map(FftAlgorithm::Provider)
        } else {
            None
        };
        let algorithm = match algorithm {
            Some(algorithm) => {
                if let FftAlgorithm::Provider(provider) = &algorithm {
                    preparation_bytes = checked_add(
                        size_of::<Self>(),
                        provider.plan.preparation_bytes(),
                        scioxide_core::quantity!("fft_preparation_bytes"),
                    )?;
                }
                algorithm
            }
            None => {
                context.execution().check()?;
                let dispatch = crate::backend::dispatch(backend, context.execution().processor())?;
                // Candidate selection evaluates at most two smooth cost plans.
                // Admit their temporary memo/radix buffers before selecting.
                if rader_candidate(n) {
                    let m = bluestein_len(n, length)?;
                    crate::memory::admit_bytes(
                        checked_add(
                            size_of::<Self>(),
                            planner_bytes(n - 1)?.max(planner_bytes(m)?),
                            scioxide_core::quantity!("fft_preparation_bytes"),
                        )?,
                        preparation_limit,
                        scioxide_core::quantity!("fft_preparation_bytes"),
                    )?;
                }
                let selection = NativeAlgorithm::select(n, length)?;
                preparation_bytes = selection.preparation_bound::<S>(n)?;
                crate::memory::admit_bytes(
                    preparation_bytes,
                    preparation_limit,
                    scioxide_core::quantity!("fft_preparation_bytes"),
                )?;
                selection.prepare::<S>(n, dispatch, length)?
            }
        };
        // The leading `n` scratch values are the transactional candidate that
        // receives the range-scaled input; algorithm scratch follows it. Length
        // one is rescaled directly and never stages.
        let algorithm_scratch_len = algorithm.scratch_len()?;
        let scratch_len = if n == 1 {
            0
        } else {
            checked_add(
                n,
                algorithm_scratch_len,
                scioxide_core::quantity!("transactional_fft_scratch"),
            )?
        };
        let heap_bytes = algorithm.heap_bytes()?;
        let stage_count = algorithm.stage_count();
        let diagnostics = FftPlanDiagnostics {
            algorithm: algorithm.kind(n),
            stage_count,
        };
        let diagnostics = match &algorithm {
            FftAlgorithm::Provider(plan) => {
                let reported = plan.diagnostics;
                if reported.algorithm() != FftAlgorithmKind::Provider {
                    return Err(SciOxideError::invalid_input(
                        provider,
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "provider diagnostics must describe a provider",
                    ));
                }
                reported
            }
            _ => diagnostics,
        };
        checked_mul(
            scratch_len,
            size_of::<Complex<S>>(),
            scioxide_core::quantity!("fft_scratch_bytes"),
        )?;
        let plan_bytes = checked_add(
            size_of::<Self>(),
            heap_bytes,
            scioxide_core::quantity!("fft_plan_bytes"),
        )?;
        Ok(Self {
            n,
            growth: crate::scaling::growth_exponent(n),
            execution_policy,
            backend,
            scratch_len,
            algorithm_scratch_len,
            plan_bytes,
            preparation_bytes,
            diagnostics,
            algorithm,
        })
    }

    /// Return the resolved kernel backend.
    pub(crate) const fn backend(&self) -> KernelBackend {
        self.backend
    }

    pub(crate) fn resolved_backend(&self) -> crate::FftBackend {
        match self.algorithm {
            FftAlgorithm::Provider(_) => crate::FftBackend::Provider,
            _ => crate::FftBackend::Cpu(self.backend),
        }
    }

    pub(crate) fn workspace_byte_limit(&self) -> usize {
        let bytes = |len: usize| len * size_of::<Complex<S>>();
        match &self.algorithm {
            FftAlgorithm::Provider(provider) => crate::workspace::nested_byte_limit(
                usize::MAX,
                bytes(self.scratch_len),
                [(provider.workspace_limit, bytes(provider.scratch_len))],
            ),
            _ => usize::MAX,
        }
    }
    /// Return resident plan bytes, including owned heap storage.
    #[must_use]
    pub(crate) const fn plan_bytes(&self) -> usize {
        self.plan_bytes
    }

    // NUFFT uses only powers of two. Radix-4 first (then radix-2) is
    // optimal for stage_cost on these lengths; its storage is known exactly.
    pub(crate) fn power_two_requirements(n: usize) -> Result<(usize, usize, usize), SciOxideError> {
        debug_assert!(n.is_power_of_two());
        let log = n.trailing_zeros() as usize;
        let stages = log.div_ceil(2);
        let q = scioxide_core::quantity!("fft_preparation_bytes");
        let resident = [
            size_of::<Self>(),
            checked_mul(n, size_of::<usize>(), q)?,
            checked_mul(n - 1, size_of::<Complex<S>>(), q)?,
            checked_mul(stages, size_of::<crate::kernel::Stage<S>>(), q)?,
        ]
        .into_iter()
        .try_fold(0, |sum, part| checked_add(sum, part, q))?;
        let peak = checked_add(resident, checked_mul(stages, size_of::<usize>(), q)?, q)?;
        Ok((resident, peak, smooth_scratch_len(n)))
    }

    /// Storage for a child whose parent owns scaling and publication.
    pub(crate) const fn bounded_scratch_len(&self) -> usize {
        self.algorithm_scratch_len
    }

    /// Candidate and native stage scratch for a small N-D block.
    pub(crate) fn lane_scratch_len(&self) -> usize {
        if self.n > crate::kernel::block::COMPLEX_ROWS
            && self.n <= crate::kernel::block::ROWS
            && self.smooth_stages().is_some()
        {
            2 * crate::kernel::block::WIDTH * self.n
        } else {
            self.algorithm_scratch_len
        }
    }

    pub(crate) fn bounded_workspace_byte_limit(&self) -> usize {
        match &self.algorithm {
            FftAlgorithm::Provider(p) => p.workspace_limit,
            _ => usize::MAX,
        }
    }

    pub(crate) const fn preparation_bytes(&self) -> usize {
        self.preparation_bytes
    }

    /// Return deterministic algorithm and stage-count diagnostics.
    pub(crate) const fn diagnostics(&self) -> FftPlanDiagnostics {
        self.diagnostics
    }

    /// Return the number of complex values required for supplied-scratch execution.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    pub(crate) fn transform(
        &self,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::validation::length(data.len(), self.n)?;
        crate::validation::sufficient(scratch.len(), self.scratch_len())?;
        if self.n == 1 {
            check_complex_slice_finite(data, scioxide_core::NonFiniteOrigin::Input)?;
            context.check()?;
            data[0] = singleton(data[0], scale)?;
            return Ok(());
        }
        let (candidate, algorithm_scratch) = scratch.split_at_mut(self.n);
        let restore = self.transform_candidate(
            || data.iter().copied(),
            Some(data),
            candidate,
            algorithm_scratch,
            inverse,
            scale,
            context,
        )?;
        publish(restore, candidate, data.iter_mut());
        Ok(())
    }

    pub(crate) fn transform_into_view(
        &self,
        input: VectorView<'_, Complex<S>>,
        mut output: VectorViewMut<'_, Complex<S>>,
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        Complex<S>: scioxide_core::NumericScalar,
    {
        crate::validation::length(output.len(), self.n)?;
        crate::validation::length(input.len(), self.n)?;
        crate::validation::sufficient(scratch.len(), self.scratch_len())?;
        if self.n == 1 {
            let value = input.get(0)?;
            check_complex_slice_finite(&[value], scioxide_core::NonFiniteOrigin::Input)?;
            context.check()?;
            output.set(0, singleton(value, scale)?)?;
            return Ok(());
        }
        let (candidate, algorithm_scratch) = scratch.split_at_mut(self.n);
        // Select contiguous borrows once. This keeps both the maximum scan
        // and scaled copy free of per-element stride/conjugation handling.
        let restore = if let Some(input) = input.as_contiguous_slice() {
            self.transform_candidate(
                || input.iter().copied(),
                Some(input),
                candidate,
                algorithm_scratch,
                inverse,
                scale,
                context,
            )?
        } else {
            self.transform_candidate(
                || input.iter(),
                None,
                candidate,
                algorithm_scratch,
                inverse,
                scale,
                context,
            )?
        };
        if let Some(output) = output.as_contiguous_slice_mut() {
            publish(restore, candidate, output.iter_mut());
        } else {
            publish(restore, candidate, output.iter_mut());
        }
        Ok(())
    }

    /// Scan and stage `input` into `candidate`, then execute it. Returns the
    /// pending binary restore, or `None` when `candidate` is already final.
    #[expect(
        clippy::too_many_arguments,
        reason = "the scanned input, its optional contiguous source and both buffers stay explicit"
    )]
    fn transform_candidate<I: Iterator<Item = Complex<S>>>(
        &self,
        input: impl Fn() -> I,
        source: Option<&[Complex<S>]>,
        candidate: &mut [Complex<S>],
        algorithm_scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<Option<BinaryScale<S>>, SciOxideError> {
        let maximum = match source {
            Some(source) => crate::scaling::complex_max_slice(source)?,
            None => complex_max(input())?,
        };
        context.check()?;
        if maximum == S::ZERO {
            candidate.fill(Complex::new(S::ZERO, S::ZERO));
            return Ok(None);
        }
        let exponent = range_exponent(maximum);
        let bounded = self.bounded(scale, exponent);
        // Input already in [0.5, 1) needs no range scaling: smooth plans
        // gather it straight from contiguous caller storage, with no staging.
        if let (0, true, Some(source), FftAlgorithm::Smooth(plan)) =
            (exponent, bounded, source, &self.algorithm)
        {
            plan.execute_from(
                source,
                candidate,
                algorithm_scratch,
                inverse,
                scale,
                self.execution_policy,
                context,
            )?;
            return Ok(Some(BinaryScale::Identity));
        }
        let scaling = BinaryScale::new(-exponent);
        scaling.apply_all(candidate.iter_mut().zip(input()));
        self.execute_candidate(
            candidate,
            algorithm_scratch,
            inverse,
            scale,
            exponent,
            bounded,
            context,
        )
    }

    /// Owned output is itself the unpublished candidate, requiring no second
    /// transaction buffer or preliminary clone of the input.
    pub(crate) fn transform_owned(
        &self,
        input: &[Complex<S>],
        inverse: bool,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<Vec<Complex<S>>, SciOxideError> {
        crate::validation::length(input.len(), self.n)?;
        let zero = Complex::new(S::ZERO, S::ZERO);
        let mut output = try_filled(self.n, zero, scioxide_core::quantity!("fft_output_buffer"))?;
        if self.n == 1 {
            check_complex_slice_finite(input, scioxide_core::NonFiniteOrigin::Input)?;
            context.check()?;
            output[0] = singleton(input[0], scale)?;
            return Ok(output);
        }
        let mut work = try_filled(
            self.algorithm_scratch_len,
            zero,
            scioxide_core::quantity!("fft_scratch"),
        )?;
        let restore = self.transform_candidate(
            || input.iter().copied(),
            Some(input),
            &mut output,
            &mut work,
            inverse,
            scale,
            context,
        )?;
        if let Some(restore) = restore {
            restore.restore_in_place(&mut output);
        }
        Ok(output)
    }

    /// Whether restoring a candidate scaled by `2^-exponent` stays finite.
    ///
    /// Native kernels bound outputs by `2^growth * scale` in the range-scaled
    /// frame. Scaling that finite bound by a power of two is exact until it
    /// overflows, which happens exactly when its binary exponent exceeds that
    /// of the largest finite scalar. Providers are never bounded, so their
    /// outputs are checked individually.
    fn bounded(&self, scale: S, exponent: i32) -> bool {
        !matches!(self.algorithm, FftAlgorithm::Provider(_))
            && self.growth + range_exponent(scale) + exponent <= range_exponent(S::MAX)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "candidate storage, range exponent and its bound decision stay explicit"
    )]
    fn execute_candidate(
        &self,
        candidate: &mut [Complex<S>],
        algorithm_scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        input_exponent: i32,
        bounded: bool,
        context: ExecutionContext<'_>,
    ) -> Result<Option<BinaryScale<S>>, SciOxideError> {
        // At the representability boundary, convolution rounding can turn a
        // finite normalized DC (e.g. the mean of MAX samples) into infinity.
        // Only this rare range needs a compensated reduction. It has ordinary
        // scalar precision.
        let dc = if !bounded && !matches!(self.algorithm, FftAlgorithm::Provider(_)) {
            let mut re = scioxide_core::ScaledAccumulator::ZERO;
            let mut im = scioxide_core::ScaledAccumulator::ZERO;
            for value in candidate.iter() {
                re.add_value(value.re)?;
                im.add_value(value.im)?;
            }
            Some(Complex::new(re.round()?, im.round()?) * scale)
        } else {
            None
        };
        self.execute_kernel::<false>(candidate, algorithm_scratch, inverse, scale, context)?;
        if let Some(dc) = dc {
            candidate[0] = dc;
        }
        if bounded {
            Ok(Some(BinaryScale::new(input_exponent)))
        } else {
            restore_complex(candidate, input_exponent, S::INFINITY)?;
            Ok(None)
        }
    }

    /// Execute finite components bounded by one, in parent-owned candidate
    /// storage. Parents own range scaling and the final output write.
    /// Cancellation, capabilities and provider validation remain mandatory.
    /// Native algorithms check bounds in their first input reads; providers
    /// use checked range measurement and restoration, including under bounded
    /// parent execution. Contract violations panic in every build.
    pub(crate) fn transform_bounded(
        &self,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::validation::length(data.len(), self.n)?;
        crate::validation::sufficient(scratch.len(), self.algorithm_scratch_len)?;
        context.check()?;
        self.execute_bounded(data, scratch, inverse, scale, context)?;
        context.check()
    }

    /// [`Self::transform_bounded`] for callers that already checked lengths,
    /// scratch and context.
    pub(crate) fn execute_bounded(
        &self,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if self.n == 1 {
            data[0] = singleton(crate::validation::bounded(data[0]), scale)?;
            Ok(())
        } else if matches!(self.algorithm, FftAlgorithm::Provider(_)) {
            self.transform_lane::<true>(data, scratch, inverse, scale, context)
        } else {
            self.execute_kernel::<true>(data, scratch, inverse, scale, context)
        }
    }

    /// The stage plan of a native smooth algorithm.
    pub(crate) fn smooth_stages(&self) -> Option<&crate::kernel::StagePlan<S>> {
        match &self.algorithm {
            FftAlgorithm::Smooth(plan) => Some(&plan.stages),
            _ => None,
        }
    }

    /// Transform every lane of one N-D axis on behalf of the parent, with one
    /// length, scratch and dispatch check for the whole axis.
    ///
    /// Bounded smooth lanes of up to [`crate::kernel::block::COMPLEX_ROWS`] points run
    /// across lanes, a block of four at a time, with the scalar codelets'
    /// arithmetic. Longer smooth lanes through 256 points batch the native
    /// stages with one-lane arithmetic; other bounded lanes run the unscaled kernel.
    /// Checked and provider lanes are independent batches, range-scaled,
    /// transformed and restored in place with the arithmetic of
    /// [`Self::transform`]; the lanes belong to the parent's candidate, so only
    /// the prepared lane scratch suffices. Provider lanes keep one provider call,
    /// context check and finiteness check each.
    pub(crate) fn transform_lanes(
        &self,
        lanes: Lanes<'_, Complex<S>>,
        scratch: &mut [Complex<S>],
        inverse: bool,
        mode: LaneMode,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::validation::length(lanes.len(), self.n)?;
        crate::validation::sufficient(scratch.len(), self.algorithm_scratch_len)?;
        match (mode, &self.algorithm) {
            (LaneMode::Bounded, FftAlgorithm::Smooth(plan))
                if self.n <= crate::kernel::block::COMPLEX_ROWS =>
            {
                let zero = crate::kernel::block::Block::new(
                    crate::kernel::block::Reals::splat(S::ZERO),
                    crate::kernel::block::Reals::splat(S::ZERO),
                );
                let rows = &mut [zero; crate::kernel::block::ROWS];
                lanes.try_for_each_run(context, |values, run| {
                    plan.stages
                        .transform_lanes(values, run, (inverse, scale), rows);
                    Ok(())
                })
            }
            (LaneMode::Bounded, FftAlgorithm::Smooth(plan))
                if self.n <= crate::kernel::block::ROWS =>
            {
                crate::validation::sufficient(scratch.len(), self.lane_scratch_len())?;
                lanes.try_for_each_run(context, |values, run| {
                    plan.stages
                        .transform_native_lanes(values, run, scratch, inverse, scale);
                    Ok(())
                })
            }
            (LaneMode::Bounded, _) => lanes.try_for_each(context, |lane| {
                self.execute_bounded(lane, scratch, inverse, scale, context)
            }),
            (LaneMode::Checked, _) => lanes.try_for_each(context, |lane| {
                self.transform_lane::<false>(lane, scratch, inverse, scale, context)
            }),
        }
    }

    /// Range-scale, transform, normalize and restore one lane of a parent's
    /// transaction candidate in place, with the arithmetic of [`Self::transform`].
    fn transform_lane<const BOUNDED: bool>(
        &self,
        data: &mut [Complex<S>],
        algorithm_scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if self.n == 1 {
            check_complex_slice_finite(data, scioxide_core::NonFiniteOrigin::Input)?;
            data[0] = singleton(data[0], scale)?;
            return Ok(());
        }
        let maximum = crate::scaling::complex_max_slice(data);
        if BOUNDED {
            assert!(
                maximum.as_ref().is_ok_and(|&maximum| maximum <= S::ONE),
                "bounded FFT input must have finite components with magnitude at most one"
            );
        }
        let maximum = maximum?;
        if maximum == S::ZERO {
            data.fill(Complex::new(S::ZERO, S::ZERO));
            return Ok(());
        }
        let exponent = range_exponent(maximum);
        let scaling = BinaryScale::new(-exponent);
        scaling.apply_all(crate::scaling::in_place(data));
        let bounded = self.bounded(scale, exponent);
        let restore = self.execute_candidate(
            data,
            algorithm_scratch,
            inverse,
            scale,
            exponent,
            bounded,
            context,
        )?;
        if let Some(restore) = restore {
            restore.restore_in_place(data);
        }
        Ok(())
    }

    fn execute_kernel<const BOUNDED: bool>(
        &self,
        candidate: &mut [Complex<S>],
        algorithm_scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match &self.algorithm {
            FftAlgorithm::Provider(plan) => {
                // Bounded provider lanes use transform_lane: their admission is
                // folded into the checked range measurement before callback entry.
                debug_assert!(!BOUNDED);
                plan.plan.execute(
                    if inverse {
                        crate::FftDirection::Inverse
                    } else {
                        crate::FftDirection::Forward
                    },
                    candidate,
                    &mut algorithm_scratch[..plan.scratch_len],
                    context,
                )?;
                // Providers are external code: re-check cancellation and
                // finiteness, and apply the direction scale here because the
                // provider contract is an unscaled DFT.
                context.check()?;
                check_complex_slice_finite(candidate, scioxide_core::NonFiniteOrigin::Callback)?;
                scale_complex(candidate, scale);
            }
            FftAlgorithm::Smooth(plan) => {
                plan.execute::<BOUNDED>(
                    candidate,
                    algorithm_scratch,
                    inverse,
                    scale,
                    self.execution_policy,
                    context,
                )?;
            }
            FftAlgorithm::Rader(plan) => {
                plan.execute::<BOUNDED>(candidate, algorithm_scratch, inverse, scale);
            }
            FftAlgorithm::Bluestein(plan) => {
                plan.execute::<BOUNDED>(candidate, algorithm_scratch, inverse, scale);
            }
        }

        Ok(())
    }
}

impl<S: scioxide_core::RealScalar> FftAlgorithm<S> {
    fn kind(&self, n: usize) -> FftAlgorithmKind {
        match self {
            Self::Smooth(_) if n == 1 => FftAlgorithmKind::Identity,
            Self::Smooth(_) => FftAlgorithmKind::MixedRadix,
            Self::Rader(_) => FftAlgorithmKind::Rader,
            Self::Bluestein(_) => FftAlgorithmKind::Bluestein,
            Self::Provider(_) => FftAlgorithmKind::Provider,
        }
    }

    fn stage_count(&self) -> usize {
        // Rader and Bluestein add two phase-bearing steps (pre- and
        // post-multiplication by chirp/kernel data) around the convolution.
        match self {
            Self::Smooth(plan) => plan.stage_count(),
            Self::Rader(plan) => plan.convolution.stage_count() + 2,
            Self::Bluestein(plan) => plan.convolution.stage_count() + 2,
            Self::Provider(plan) => plan.diagnostics.stage_count(),
        }
    }

    fn scratch_len(&self) -> Result<usize, SciOxideError> {
        // Smooth plans gather into an out-of-place buffer of their length;
        // Rader and Bluestein need a signal buffer plus a spectrum buffer of their convolution length. The
        // private convolution subplans run out of place between those two
        // buffers, so they need no scratch of their own.
        match self {
            Self::Smooth(plan) => Ok(smooth_scratch_len(plan.n)),
            Self::Rader(plan) => {
                checked_mul(plan.n - 1, 2, scioxide_core::quantity!("rader_fft_scratch"))
            }
            Self::Bluestein(plan) => checked_mul(
                plan.convolution.n,
                2,
                scioxide_core::quantity!("bluestein_fft_scratch"),
            ),
            Self::Provider(plan) => Ok(plan.scratch_len),
        }
    }

    fn heap_bytes(&self) -> Result<usize, SciOxideError> {
        match self {
            Self::Smooth(plan) => plan.heap_bytes(),
            Self::Rader(plan) => plan.heap_bytes(),
            Self::Bluestein(plan) => plan.heap_bytes(),
            Self::Provider(plan) => Ok(plan.bytes),
        }
    }
}

/// Apply the singleton gain before publication, preserving exact and rounded zeros.
fn singleton<S: scioxide_core::RealScalar>(
    value: Complex<S>,
    scale: S,
) -> Result<Complex<S>, SciOxideError> {
    let value = crate::scaling::scale_singleton(value, scale);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(crate::validation::arithmetic_overflow(
            "singleton FFT output exceeds scalar range",
        ))
    }
}

/// Smooth plans gather out of place; length one is the identity and needs none.
const fn smooth_scratch_len(n: usize) -> usize {
    if n == 1 { 0 } else { n }
}

/// Write a finished candidate, restoring its binary range when pending.
fn publish<'a, S: scioxide_core::RealScalar + 'a>(
    restore: Option<BinaryScale<S>>,
    candidate: &[Complex<S>],
    destination: impl Iterator<Item = &'a mut Complex<S>>,
) {
    let values = destination.zip(candidate.iter().copied());
    match restore {
        Some(restore) => restore.restore_all(values),
        None => values.for_each(|(destination, value)| *destination = value),
    }
}

#[cfg(test)]
#[expect(clippy::cast_lossless, clippy::cast_precision_loss)]
mod tests {
    #[cfg(feature = "std")]
    use super::smooth::FFT_PARALLEL_THRESHOLD;
    use super::*;
    use crate::Normalization;

    use crate::test_support::{assert_complex_close, assert_complex_l2_close, direct_dft};

    /// Prepare a sequential core, the public default policy, on `backend`.
    fn prepared<S: scioxide_core::RealScalar>(
        n: usize,
        backend: KernelBackend,
    ) -> Result<ComplexCore<S>, SciOxideError> {
        ComplexCore::with_backend_and_context(
            n,
            ExecutionPolicy::Sequential,
            backend,
            ExecutionContext::sequential_native(),
            scioxide_core::argument!("n"),
            scioxide_core::argument!("context"),
        )
    }

    /// The backend automatic selection resolves on this host.
    fn automatic() -> KernelBackend {
        crate::FftPlanningContext::<f64>::default()
            .resolve(crate::FftBackendStrategy::Automatic)
            .unwrap()
    }

    /// Zeroed scratch of the length a core requires.
    fn workspace<S: scioxide_core::RealScalar>(plan: &ComplexCore<S>) -> Vec<Complex<S>> {
        vec![Complex::new(S::ZERO, S::ZERO); plan.scratch_len()]
    }

    fn bounded_contract<S: scioxide_core::RealScalar>() {
        let context = ExecutionContext::sequential_native();
        for backend in crate::test_support::fft_backends(context.processor()) {
            // Identity, mixed radix, Rader and Bluestein all admit during
            // their existing first read pass, before publishing into data.
            for n in [1, 2, 3, 4, 5, 7, 16, 64, 17, 23, 1019] {
                let plan = prepared::<S>(n, backend).unwrap();
                let mut scratch = workspace(&plan);
                for index in [0, n / 2, n - 1] {
                    for bad in [
                        S::NAN,
                        S::INFINITY,
                        -S::INFINITY,
                        S::ONE + S::EPSILON,
                        -S::ONE - S::EPSILON,
                    ] {
                        for imaginary in [false, true] {
                            let mut input = vec![Complex::new(S::HALF, S::ZERO); n];
                            input[index] = if imaginary {
                                Complex::new(S::HALF, bad)
                            } else {
                                Complex::new(bad, S::ZERO)
                            };
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    let _ = plan.transform_bounded(
                                        &mut input,
                                        &mut scratch,
                                        false,
                                        S::ONE,
                                        context,
                                    );
                                }));
                            assert!(result.is_err(), "{backend:?} n={n}, index={index}");
                        }
                    }
                }
                let mut boundary = vec![Complex::new(S::ONE, -S::ONE); n];
                plan.transform_bounded(&mut boundary, &mut scratch, false, S::ONE, context)
                    .unwrap();
            }
        }
    }

    #[test]
    fn bounded_input_contract_is_enforced_before_execution() {
        bounded_contract::<f32>();
        bounded_contract::<f64>();
    }

    #[test]
    fn bounded_provider_inputs_are_admitted_before_callbacks_and_restored() {
        use crate::{
            FftDirection, FftProvider, FftProviderLimits, FftProviderPlan, FftProviderPolicy,
        };
        use alloc::sync::Arc;
        use core::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Debug)]
        struct Direct(Arc<AtomicUsize>);
        impl FftProvider<f64> for Direct {
            fn prepare(
                &self,
                _: usize,
                _: ExecutionPolicy,
                _: FftProviderLimits,
                _: ExecutionContext<'_>,
            ) -> Result<Option<Arc<dyn FftProviderPlan<f64>>>, SciOxideError> {
                Ok(Some(Arc::new(Direct(Arc::clone(&self.0)))))
            }
        }
        impl FftProviderPlan<f64> for Direct {
            fn workspace_len(&self) -> usize {
                0
            }
            fn plan_bytes(&self) -> usize {
                size_of::<Self>()
            }
            fn preparation_bytes(&self) -> usize {
                size_of::<Self>()
            }
            fn diagnostics(&self) -> FftPlanDiagnostics {
                FftPlanDiagnostics::for_provider(1)
            }
            fn execute(
                &self,
                direction: FftDirection,
                data: &mut [Complex<f64>],
                _: &mut [Complex<f64>],
                _: ExecutionContext<'_>,
            ) -> Result<(), SciOxideError> {
                self.0.fetch_add(1, Ordering::Relaxed);
                data.copy_from_slice(&direct_dft(data, direction == FftDirection::Inverse));
                Ok(())
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Direct(Arc::clone(&calls));
        let context = ExecutionContext::sequential();
        let plan = ComplexCore::with_backend_and_context(
            4,
            ExecutionPolicy::Sequential,
            KernelBackend::Scalar,
            crate::FftPlanningContext::with_provider(
                context,
                &provider,
                FftProviderPolicy::Required,
            ),
            scioxide_core::argument!("n"),
            scioxide_core::argument!("context"),
        )
        .unwrap();
        for bad in [
            f64::NAN,
            f64::INFINITY,
            -f64::INFINITY,
            1.0 + f64::EPSILON,
            -1.0 - f64::EPSILON,
        ] {
            let mut input = [Complex::new(0.5, 0.0); 4];
            input[3].im = bad;
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = plan.transform_bounded(&mut input, &mut [], false, 1.0, context);
                }))
                .is_err()
            );
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        for inverse in [false, true] {
            let input = [
                Complex::new(1.0, -1.0),
                Complex::new(-1.0, 0.5),
                Complex::new(0.5, 0.0),
                Complex::new(0.0, 0.5),
            ];
            let mut actual = input;
            plan.transform_bounded(&mut actual, &mut [], inverse, 0.25, context)
                .unwrap();
            let expected: Vec<_> = direct_dft(&input, inverse)
                .into_iter()
                .map(|z| z * 0.25)
                .collect();
            assert_complex_close(&actual, &expected, Complex::norm, |_| 8.0 * f64::EPSILON);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    fn singleton_paths<S: scioxide_core::RealScalar>()
    where
        Complex<S>: scioxide_core::NumericScalar<Real = S>,
    {
        use crate::lanes::{LaneGeometry, Placement};
        let plan = prepared::<S>(1, automatic()).unwrap();
        let cx = ExecutionContext::sequential();
        let half = S::HALF;
        for scale in [S::ZERO, half, S::ONE, S::TWO] {
            for value in [Complex::new(half, -half), Complex::new(-S::ZERO, S::ZERO)] {
                let expected = if value.re == S::ZERO || scale == S::ZERO {
                    Complex::new(S::ZERO, S::ZERO)
                } else {
                    Complex::new(value.re * scale, value.im * scale)
                };
                let check = |actual: Complex<S>| {
                    assert_eq!(actual, expected);
                    assert_eq!(actual.re.is_sign_negative(), expected.re.is_sign_negative());
                    assert_eq!(actual.im.is_sign_negative(), expected.im.is_sign_negative());
                };
                for inverse in [false, true] {
                    let mut data = [value];
                    plan.transform(&mut data, &mut [], inverse, scale, cx)
                        .unwrap();
                    check(data[0]);
                    check(plan.transform_owned(&[value], inverse, scale, cx).unwrap()[0]);
                    let mut data = [value];
                    plan.transform_bounded(&mut data, &mut [], inverse, scale, cx)
                        .unwrap();
                    check(data[0]);
                    let mut data = [value];
                    plan.execute_bounded(&mut data, &mut [], inverse, scale, cx)
                        .unwrap();
                    check(data[0]);
                    let mut output = [value; 2];
                    plan.transform_into_view(
                        VectorView::from_slice_with_stride(&[value; 2], 1, 2).unwrap(),
                        VectorViewMut::from_slice_with_stride(&mut output, 1, 2).unwrap(),
                        &mut [],
                        inverse,
                        scale,
                        cx,
                    )
                    .unwrap();
                    check(output[0]);
                    assert_eq!(output[1], value);
                    for mode in [LaneMode::Checked, LaneMode::Bounded] {
                        let mut data = [value; 3];
                        let geometry = LaneGeometry::new::<Complex<S>>(&[1, 3], &[3, 1], 0);
                        let mut staging = vec![value; geometry.staging_len()];
                        let lanes = Lanes::new(
                            &mut data,
                            Placement::new(geometry, &[1, 3], &[3, 1]),
                            &mut staging,
                        );
                        plan.transform_lanes(lanes, &mut [], inverse, mode, scale, cx)
                            .unwrap();
                        data.into_iter().for_each(check);
                    }
                }
            }
        }
        let input = [Complex::new(S::MAX, S::ZERO)];
        let mut data = input;
        assert!(matches!(
            plan.transform(&mut data, &mut [], false, S::TWO, cx)
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
        assert_eq!(data, input);
        let mut output = [Complex::new(half, -half)];
        let original = output;
        assert!(
            plan.transform_into_view(
                VectorView::from_slice(&input),
                VectorViewMut::from_slice(&mut output),
                &mut [],
                false,
                S::TWO,
                cx
            )
            .is_err()
        );
        assert_eq!(output, original);
    }

    #[test]
    fn singleton_scaling_matches_on_every_execution_path() {
        singleton_paths::<f32>();
        singleton_paths::<f64>();
    }

    #[test]
    fn power_two_estimator_matches_constructed_storage() {
        for log in 0..=16 {
            let n = 1 << log;
            let plan = prepared::<f64>(n, automatic()).unwrap();
            let (resident, peak, scratch) = ComplexCore::<f64>::power_two_requirements(n).unwrap();
            assert_eq!(resident, plan.plan_bytes());
            assert_eq!(peak, plan.preparation_bytes());
            assert_eq!(scratch, plan.bounded_scratch_len());
        }
    }

    #[test]
    fn bounded_children_use_exact_algorithm_storage() {
        let cx = ExecutionContext::sequential_native();
        for n in [1, 2, 3, 16, 17, 23, 32, 1000, 4096] {
            let plan = prepared::<f64>(n, automatic()).unwrap();
            let (forward, inverse) = Normalization::Orthonormal.fourier_scales(n).unwrap();
            let input = bounded_signal::<f64>(n);
            for (inverse, scale) in [(false, forward), (true, inverse)] {
                let mut expected = input.clone();
                plan.transform(&mut expected, &mut workspace(&plan), inverse, scale, cx)
                    .unwrap();
                let mut actual = input.clone();
                let required = plan.bounded_scratch_len();
                let mut work = vec![Complex::new(f64::NAN, f64::NAN); required];
                if required != 0 {
                    assert_eq!(
                        plan.transform_bounded(
                            &mut actual,
                            &mut work[..required - 1],
                            inverse,
                            scale,
                            cx
                        )
                        .unwrap_err()
                        .kind(),
                        scioxide_core::SciOxideErrorKind::DimensionMismatch
                    );
                    assert_eq!(actual, input);
                }
                plan.transform_bounded(&mut actual, &mut work, inverse, scale, cx)
                    .unwrap();
                assert_complex_close(&actual, &expected, Complex::norm, |_| {
                    roundtrip_bound::<f64>(n)
                });
                assert_eq!(plan.scratch_len() - required, if n == 1 { 0 } else { n });
            }
        }
    }

    use scioxide_core::RealScalar;

    fn c<S: scioxide_core::RealScalar>(re: f64, im: f64) -> Complex<S> {
        Complex::new(
            S::try_from_f64_lossy(re).unwrap(),
            S::try_from_f64_lossy(im).unwrap(),
        )
    }

    fn signal(n: usize) -> Vec<Complex<f64>> {
        (0..n)
            .map(|index| c(index as f64 * 0.25 - 1.0, 0.5 - index as f64 * 0.125))
            .collect()
    }

    fn bounded_signal<S: scioxide_core::RealScalar>(n: usize) -> Vec<Complex<S>> {
        (0..n)
            .map(|index| {
                let angle = 2.0 * std::f64::consts::PI * index as f64 / n as f64;
                c(angle.sin(), (3.0 * angle).cos() * 0.5)
            })
            .collect()
    }

    fn roundtrip_bound<S: scioxide_core::RealScalar>(n: usize) -> f64 {
        let epsilon = <S as RealScalar>::EPSILON.try_to_f64_exact().unwrap();
        // Two passes through the documented 2 epsilon log2(n) forward L2
        // envelope, measured relative to the correctly normalized reference.
        4.0 * epsilon * (n as f64).log2().max(1.0)
    }

    fn defining_sum_bound(input: &[Complex<f64>]) -> f64 {
        // Absolute complex error scaled by the defining sum's input mass.
        // Eight budgets kernel rounding and binary64 oracle phase evaluation.
        8.0 * f64::EPSILON
            * (input.len() as f64).log2().max(1.0)
            * input.iter().map(|value| value.norm()).sum::<f64>()
    }

    /// Transform forward then back with inverse normalization.
    fn roundtrip<S: scioxide_core::RealScalar>(
        plan: &ComplexCore<S>,
        input: &[Complex<S>],
        normalization: Normalization,
    ) -> Vec<Complex<S>> {
        let cx = ExecutionContext::sequential_native();
        let (forward, inverse) = normalization.fourier_scales(input.len()).unwrap();
        let spectrum = plan.transform_owned(input, false, forward, cx).unwrap();
        plan.transform_owned(&spectrum, true, inverse, cx).unwrap()
    }

    #[test]
    fn zero_is_rejected_and_one_is_identity() {
        let error = prepared::<f64>(0, automatic()).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        let plan = prepared::<f64>(1, automatic()).unwrap();
        let input = [c(3.0, -2.0)];
        let output = plan
            .transform_owned(&input, false, 1.0, ExecutionContext::sequential_native())
            .unwrap();
        assert_complex_close(&output, &input, Complex::norm, |_| 0.0);
    }

    #[test]
    fn plan_reports_algorithm_and_stage_count() {
        for n in [1, 16, 1000, 1009, 1022] {
            let plan = prepared::<f64>(n, KernelBackend::Scalar).unwrap();
            let diagnostics = plan.diagnostics();
            assert_eq!(diagnostics.stage_count(), plan.algorithm.stage_count());
            if n == 1 {
                assert_eq!(diagnostics.algorithm(), FftAlgorithmKind::Identity);
            }
        }
    }

    #[test]
    fn bluestein_chirps_remain_on_the_unit_circle() {
        let plan = prepared::<f64>(23, KernelBackend::Scalar).unwrap();
        assert!(matches!(plan.algorithm, FftAlgorithm::Bluestein(_)));
        let FftAlgorithm::Bluestein(bluestein) = &plan.algorithm else {
            return;
        };
        let tolerance = 4.0 * f64::EPSILON;
        bluestein.chirp.iter().for_each(|value| {
            assert!((value.norm_sqr() - 1.0).abs() <= tolerance);
        });
    }

    #[test]
    fn mixed_radix_lengths_match_direct_dft() {
        let cx = ExecutionContext::sequential_native();
        for n in [2, 3, 4, 5, 7, 8, 9, 16, 25, 49, 60, 210] {
            let input = signal(n);
            let plan = prepared(n, KernelBackend::Scalar).unwrap();
            for inverse in [false, true] {
                assert_complex_close(
                    &plan.transform_owned(&input, inverse, 1.0, cx).unwrap(),
                    &direct_dft(&input, inverse),
                    Complex::norm,
                    |_| defining_sum_bound(&input),
                );
            }
        }
    }

    #[test]
    fn rader_and_bluestein_match_direct_dft() {
        let cx = ExecutionContext::sequential_native();
        for n in [11, 13, 17, 23, 46] {
            let input = signal(n);
            let plan = prepared(n, automatic()).unwrap();
            assert_complex_close(
                &plan.transform_owned(&input, false, 1.0, cx).unwrap(),
                &direct_dft(&input, false),
                Complex::norm,
                |_| defining_sum_bound(&input),
            );
        }
    }

    #[test]
    fn planner_selects_both_prime_algorithms() {
        assert!(matches!(
            prepared::<f64>(17, automatic()).unwrap().algorithm,
            FftAlgorithm::Rader(_)
        ));
        assert!(matches!(
            prepared::<f64>(23, automatic()).unwrap().algorithm,
            FftAlgorithm::Bluestein(_)
        ));
    }

    #[test]
    fn forward_inverse_roundtrip_for_scalars_and_normalizations() {
        let input = signal(105);
        let plan = prepared(input.len(), automatic()).unwrap();
        for normalization in [
            Normalization::Forward,
            Normalization::Inverse,
            Normalization::Orthonormal,
            Normalization::Unscaled,
        ] {
            let expected: Vec<_> = input
                .iter()
                .copied()
                .map(|value| {
                    if normalization == Normalization::Unscaled {
                        value * input.len() as f64
                    } else {
                        value
                    }
                })
                .collect();
            assert_complex_l2_close(
                &roundtrip(&plan, &input, normalization),
                &expected,
                roundtrip_bound::<f64>(input.len()),
            );
        }

        let input: Vec<Complex<f32>> = (0..70)
            .map(|index| c(index as f64 * 0.125, 1.0 - index as f64 * 0.0625))
            .collect();
        let plan = prepared(input.len(), automatic()).unwrap();
        assert_complex_l2_close(
            &roundtrip(&plan, &input, Normalization::Inverse),
            &input,
            roundtrip_bound::<f32>(input.len()),
        );
    }

    #[test]
    fn structural_and_finite_failures_are_transactional() {
        let cx = ExecutionContext::sequential_native();
        let plan = prepared::<f64>(23, automatic()).unwrap();
        let original = signal(23);
        let mut data = original.clone();
        let mut short = vec![c(0.0, 0.0); plan.scratch_len() - 1];
        let error = plan
            .transform(&mut data, &mut short, false, 1.0, cx)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(data, original);

        let mut non_finite = original;
        non_finite[5].re = f64::NAN;
        let before = non_finite.clone();
        let error = plan
            .transform(&mut non_finite, &mut workspace(&plan), false, 1.0, cx)
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
        assert_eq!(non_finite[0], before[0]);
        assert!(non_finite[5].re.is_nan());
    }

    #[test]
    fn planning_overflow_is_structured() {
        let error = prepared::<f64>(usize::MAX, automatic()).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::SizeOverflow);
    }

    #[test]
    fn plan_reports_policy_backend_scratch_and_bytes() {
        let plan = prepared::<f64>(1000, KernelBackend::Scalar).unwrap();
        assert_eq!(plan.execution_policy, ExecutionPolicy::Sequential);
        assert_eq!(plan.backend(), KernelBackend::Scalar);
        assert_eq!(plan.scratch_len(), 2000);
        assert!(plan.plan_bytes() > size_of::<ComplexCore<f64>>());

        let short = prepared::<f64>(256, KernelBackend::Scalar).unwrap();
        assert_eq!(short.scratch_len(), 512);
        let input = bounded_signal::<f64>(256);
        assert_complex_close(
            &roundtrip(&short, &input, Normalization::Inverse),
            &input,
            Complex::norm,
            |_| roundtrip_bound::<f64>(256),
        );

        let mut non_finite = input;
        non_finite[113].im = f64::INFINITY;
        let before = non_finite.clone();
        let error = short
            .transform(
                &mut non_finite,
                &mut workspace(&short),
                false,
                1.0,
                ExecutionContext::sequential_native(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
        assert_eq!(non_finite, before);
    }

    #[test]
    fn direction_scaling_prevents_avoidable_intermediate_overflow() {
        let cx = ExecutionContext::sequential_native();
        for n in [4, 17, 23] {
            let plan = prepared::<f64>(n, automatic()).unwrap();
            let (forward, _) = Normalization::Forward.fourier_scales(n).unwrap();
            let mut data = vec![Complex::new(f64::MAX, 0.0); n];
            plan.transform(&mut data, &mut workspace(&plan), false, forward, cx)
                .unwrap();
            assert!(
                data.iter()
                    .all(|value| value.re.is_finite() && value.im.is_finite())
            );
            assert!(data[0].re > f64::MAX * 0.99);
        }

        let plan = prepared::<f64>(4, automatic()).unwrap();
        let (forward, _) = Normalization::Orthonormal.fourier_scales(4).unwrap();
        let mut data = vec![Complex::new(f64::MAX * 0.5, 0.0); 4];
        plan.transform(&mut data, &mut workspace(&plan), false, forward, cx)
            .unwrap();
        assert!(
            data.iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert!(data[0].re > f64::MAX * 0.99);
    }

    #[test]
    fn arithmetic_overflow_is_output_provenance_and_transactional() {
        let plan = prepared::<f64>(4, automatic()).unwrap();
        let mut data = vec![Complex::new(f64::MAX, 0.0); 4];
        let original = data.clone();
        let error = plan
            .transform(
                &mut data,
                &mut workspace(&plan),
                false,
                1.0,
                ExecutionContext::sequential_native(),
            )
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
        assert_eq!(data, original);
    }

    #[test]
    fn every_available_backend_and_policy_respects_roundtrip_bound() {
        fn check<S: scioxide_core::RealScalar>(
            n: usize,
            policy: ExecutionPolicy,
            backend: KernelBackend,
        ) {
            let cx = ExecutionContext::sequential_native();
            let plan = ComplexCore::with_backend_and_context(
                n,
                policy,
                backend,
                cx,
                scioxide_core::argument!("n"),
                scioxide_core::argument!("context"),
            )
            .unwrap();
            let input = bounded_signal::<S>(n);
            for normalization in [
                Normalization::Forward,
                Normalization::Inverse,
                Normalization::Orthonormal,
                Normalization::Unscaled,
            ] {
                let scale = if normalization == Normalization::Unscaled {
                    S::try_from_usize_exact(n).unwrap()
                } else {
                    S::ONE
                };
                let expected: Vec<_> = input.iter().map(|&value| value * scale).collect();
                assert_complex_l2_close(
                    &roundtrip(&plan, &input, normalization),
                    &expected,
                    roundtrip_bound::<S>(n),
                );
            }
        }
        let policies = [
            ExecutionPolicy::Sequential,
            ExecutionPolicy::Auto,
            ExecutionPolicy::Parallel(NonZeroUsize::new(2).unwrap()),
        ];
        for backend in
            crate::test_support::fft_backends(ExecutionContext::sequential_native().processor())
        {
            for policy in policies {
                for n in [1, 17, 23, 46, 70, 105] {
                    check::<f64>(n, policy, backend);
                    check::<f32>(n, policy, backend);
                }
            }
        }
    }

    // The partitioned prefix runs only with a parallel policy, a length at the
    // crossover and an executor offering several workers.
    #[cfg(feature = "std")]
    #[test]
    fn large_parallel_transform_matches_sequential() {
        let n = FFT_PARALLEL_THRESHOLD;
        let input = bounded_signal::<f64>(n);
        let sequential = prepared(n, automatic()).unwrap();
        let workers = NonZeroUsize::new(4).unwrap();
        let parallel = ComplexCore::with_backend_and_context(
            n,
            ExecutionPolicy::Parallel(workers),
            automatic(),
            ExecutionContext::sequential_native(),
            scioxide_core::argument!("n"),
            scioxide_core::argument!("context"),
        )
        .unwrap();
        let executor = scioxide_core::ThreadExecutor::new(workers);
        assert_complex_close(
            &parallel
                .transform_owned(&input, false, 1.0, ExecutionContext::new(&executor))
                .unwrap(),
            &sequential
                .transform_owned(&input, false, 1.0, ExecutionContext::sequential_native())
                .unwrap(),
            Complex::norm,
            |_| roundtrip_bound::<f64>(n),
        );
    }

    #[test]
    fn supplied_scratch_sequential_execution_allocates_nothing() {
        let plan = prepared::<f64>(1000, automatic()).unwrap();
        let mut data = bounded_signal(1000);
        let mut scratch = workspace(&plan);
        let result = plan.transform(
            &mut data,
            &mut scratch,
            false,
            1.0,
            ExecutionContext::sequential_native(),
        );
        result.unwrap();
    }

    #[test]
    fn available_backends_handle_unaligned_buffers_and_vector_tails() {
        let cx = ExecutionContext::sequential_native();
        for backend in crate::test_support::fft_backends(cx.processor()) {
            let plan = prepared::<f64>(125, backend).unwrap();
            let input = bounded_signal(125);
            let mut storage = vec![c(0.0, 0.0); 126];
            storage[1..].copy_from_slice(&input);
            let mut scratch_storage = vec![c(0.0, 0.0); plan.scratch_len() + 1];
            plan.transform(&mut storage[1..], &mut scratch_storage[1..], false, 1.0, cx)
                .unwrap();
            assert_complex_close(
                &storage[1..],
                &direct_dft(&input, false),
                Complex::norm,
                |_| 2e-8,
            );
        }
    }

    #[test]
    fn unavailable_backends_return_structured_errors() {
        // AVX-512 has no FFT leaves and is rejected on every host, even when
        // the token would verify; other backends only where unavailable.
        let context = ExecutionContext::sequential_native();
        core::iter::once(KernelBackend::Avx512)
            .chain(
                [KernelBackend::Avx2Fma, KernelBackend::Neon]
                    .into_iter()
                    .filter(|backend| !backend.is_available(context.processor())),
            )
            .for_each(|backend| {
                let error = prepared::<f64>(16, backend).unwrap_err();
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::InvalidInput {
                        reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                        ..
                    }
                ));
            });
    }

    #[test]
    fn convolution_subplans_are_smooth_and_non_recursive() {
        let rader = prepared::<f64>(17, automatic()).unwrap();
        let FftAlgorithm::Rader(rader) = &rader.algorithm else {
            unreachable!()
        };
        assert!(rader.convolution.n < rader.n);
        assert!(is_smooth(rader.convolution.n));

        let bluestein = prepared::<f64>(23, automatic()).unwrap();
        let FftAlgorithm::Bluestein(bluestein) = &bluestein.algorithm else {
            unreachable!()
        };
        assert!(is_smooth(bluestein.convolution.n));
        assert!(bluestein.convolution.n >= 2 * bluestein.n - 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn planning_and_parallel_crossover_are_deterministic() {
        let first = prepared::<f64>(1000, KernelBackend::Scalar).unwrap();
        let second = prepared::<f64>(1000, KernelBackend::Scalar).unwrap();
        let (FftAlgorithm::Smooth(first), FftAlgorithm::Smooth(second)) =
            (&first.algorithm, &second.algorithm)
        else {
            unreachable!()
        };
        assert_eq!(first.stages.permutation(), second.stages.permutation());
        assert!(first.stages.geometry().eq(second.stages.geometry()));

        let below = prepared::<f64>(4096, automatic()).unwrap();
        let FftAlgorithm::Smooth(below) = &below.algorithm else {
            unreachable!()
        };
        let prefix_end = below.stage_count().saturating_sub(2);
        assert_eq!(
            below
                .parallel_plan(
                    ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
                    prefix_end,
                    ExecutionContext::sequential_native(),
                )
                .worker_count(),
            1
        );

        let at_crossover = prepared::<f64>(FFT_PARALLEL_THRESHOLD, automatic()).unwrap();
        let FftAlgorithm::Smooth(at_crossover) = &at_crossover.algorithm else {
            unreachable!()
        };
        let prefix_end = at_crossover.stage_count().saturating_sub(2);
        let executor = scioxide_core::ThreadExecutor::new(NonZeroUsize::new(4).unwrap());
        let context = ExecutionContext::new(&executor).with_processor(&scioxide_core::StdProcessor);
        assert_eq!(
            at_crossover
                .parallel_plan(
                    ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
                    prefix_end,
                    context,
                )
                .worker_count(),
            4
        );
    }

    #[test]
    fn common_large_lengths_roundtrip_with_length_aware_bound() {
        for n in [1000, 4096] {
            let input = bounded_signal::<f64>(n);
            let plan = prepared(n, automatic()).unwrap();
            assert_complex_l2_close(
                &roundtrip(&plan, &input, Normalization::Inverse),
                &input,
                roundtrip_bound::<f64>(n),
            );
        }
    }

    #[test]
    fn fused_power_of_two_output_matches_scalar_plan() {
        let cx = ExecutionContext::sequential_native();
        if !KernelBackend::Avx2Fma.is_available(cx.processor()) {
            return;
        }
        let input = bounded_signal::<f64>(4096);
        let scalar = prepared(input.len(), KernelBackend::Scalar).unwrap();
        let avx2 = prepared(input.len(), KernelBackend::Avx2Fma).unwrap();
        assert_complex_close(
            &avx2.transform_owned(&input, false, 1.0, cx).unwrap(),
            &scalar.transform_owned(&input, false, 1.0, cx).unwrap(),
            Complex::norm,
            |_| 2e-8,
        );

        let mut non_finite = input;
        non_finite[2049].im = f64::NAN;
        let before = non_finite.clone();
        let error = avx2
            .transform(&mut non_finite, &mut workspace(&avx2), false, 1.0, cx)
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
        assert_eq!(non_finite[..2049], before[..2049]);
        assert!(non_finite[2049].im.is_nan());
        assert_eq!(non_finite[2050..], before[2050..]);
    }

    #[test]
    fn miri_forced_scalar_covers_all_codelets() {
        let cx = ExecutionContext::sequential_native();
        for n in [2, 3, 4, 5, 7, 60] {
            let input = bounded_signal::<f64>(n);
            let plan = prepared(n, KernelBackend::Scalar).unwrap();
            assert_complex_close(
                &plan.transform_owned(&input, false, 1.0, cx).unwrap(),
                &direct_dft(&input, false),
                Complex::norm,
                |_| defining_sum_bound(&input),
            );
        }
    }

    #[test]
    fn integer_bound_test_matches_scaled_bound_finiteness() {
        fn check<S: scioxide_core::RealScalar>() {
            let limit = range_exponent(<S as RealScalar>::MAX);
            for n in [2, 3, 64, 1000, 65_536] {
                let plan = prepared::<S>(n, KernelBackend::Scalar).unwrap();
                for normalization in [
                    Normalization::Unscaled,
                    Normalization::Inverse,
                    Normalization::Orthonormal,
                ] {
                    let (forward, inverse) = normalization.fourier_scales::<S>(n).unwrap();
                    for scale in [forward, inverse] {
                        let bound = S::ONE.scalbn(crate::scaling::growth_exponent(n)) * scale;
                        for exponent in -2 * limit..=limit + 4 {
                            assert_eq!(
                                plan.bounded(scale, exponent),
                                bound.scalbn(exponent).is_finite(),
                                "n={n} exponent={exponent}"
                            );
                        }
                    }
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn unit_range_contiguous_input_gathers_directly_with_staged_bits() {
        let context = ExecutionContext::sequential_native();
        for backend in crate::test_support::fft_backends(context.processor()) {
            for n in [2, 16, 64, 256, 512, 1000, 4096] {
                let plan = prepared::<f64>(n, backend).unwrap();
                let (forward, inverse) = Normalization::Orthonormal.fourier_scales(n).unwrap();
                // Components in [-0.75, 0.75] with maximum 0.75: range exponent zero.
                let input: Vec<_> = (0..n)
                    .map(|i| {
                        c(
                            ((i * 73 % 193) as f64 - 96.0) / 128.0,
                            (((i + 1) * 107 % 193) as f64 - 96.0) / 128.0,
                        )
                    })
                    .collect();
                let gapped: Vec<_> = input.iter().flat_map(|&v| [v, c(9.0, 9.0)]).collect();
                for (inverse, scale) in [(false, forward), (true, inverse)] {
                    let mut scratch = workspace(&plan);
                    let mut direct = vec![c(7.0, 7.0); n];
                    plan.transform_into_view(
                        VectorView::from_slice(&input),
                        VectorViewMut::from_slice(&mut direct),
                        &mut scratch,
                        inverse,
                        scale,
                        context,
                    )
                    .unwrap();
                    // A strided source is staged into the candidate first.
                    let mut staged = vec![c(7.0, 7.0); n];
                    plan.transform_into_view(
                        VectorView::from_slice_with_stride(&gapped, n, 2).unwrap(),
                        VectorViewMut::from_slice(&mut staged),
                        &mut scratch,
                        inverse,
                        scale,
                        context,
                    )
                    .unwrap();
                    assert_eq!(direct, staged, "{backend:?} n={n} inverse={inverse}");
                    let mut in_place = input.clone();
                    plan.transform(&mut in_place, &mut scratch, inverse, scale, context)
                        .unwrap();
                    assert_eq!(in_place, staged);
                }
                let mut non_finite = input.clone();
                non_finite[n / 2].im = f64::NAN;
                let mut output = vec![c(7.0, 7.0); n];
                let error = plan
                    .transform_into_view(
                        VectorView::from_slice(&non_finite),
                        VectorViewMut::from_slice(&mut output),
                        &mut workspace(&plan),
                        false,
                        forward,
                        context,
                    )
                    .unwrap_err();
                assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
                assert!(output.iter().all(|&v| v == c(7.0, 7.0)));
            }
        }
    }
}
