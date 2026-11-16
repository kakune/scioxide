//! Smooth mixed-radix stages, twiddles and scheduling.
use super::*;
use crate::kernel::{Permutation, Stage, StagePlan};
use core::ops::Range;
pub(super) const FFT_PARALLEL_THRESHOLD: usize = 65_536;
const FFT_AUTO_WORKERS: NonZeroUsize =
    NonZeroUsize::new(4).expect("resource limit must be positive");

/// Out-of-place mixed-radix plan. Every length gathers digit-reversed input
/// into scratch, fused with the leading stages the selected leaves support,
/// runs the middle stages there and writes the scaled final stage back.
#[derive(Debug, Clone)]
pub(super) struct SmoothPlan<S: scioxide_core::RealScalar> {
    pub(super) n: usize,
    pub(super) stages: StagePlan<S>,
}

// Planning helpers below report every failure
// that builds the plan; invariant violations are attributed to `length`, the
// caller parameter that determined the transform length.
impl<S: scioxide_core::RealScalar> SmoothPlan<S> {
    pub(super) fn new(
        n: usize,
        dispatch: KernelDispatch,
        length: Argument,
    ) -> Result<Self, SciOxideError> {
        let radices = best_radix_plan(n, length)?;
        let permutation = Permutation::new(compute_mixed_radix_permutation(n, &radices)?)?;
        let mut stages =
            try_with_capacity(radices.len(), scioxide_core::quantity!("fft_stage_plans"))?;
        // Decimation in time after the digit-reversal gather: stage k combines
        // `radix_k` interleaved sub-transforms of length `width` (the product
        // of all earlier radices) into blocks of `radix_k * width`.
        let mut width = 1_usize;
        for radix in radices {
            let (twiddles, roots) = stage_tables::<S>(radix, width, length)?;
            stages.push(Stage::new(dispatch, radix, width, twiddles, roots, length)?);
            width = checked_mul(width, radix, scioxide_core::quantity!("fft_stage_width"))?;
        }
        Ok(Self {
            n,
            stages: StagePlan::new(n, permutation, stages, dispatch, length)?,
        })
    }

    pub(super) fn stage_count(&self) -> usize {
        self.stages.stage_count()
    }

    // Unscaled out-of-place transform used only by the private Rader and
    // Bluestein convolutions.
    pub(super) fn transform_into(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        inverse: bool,
    ) {
        let completed = self.stages.gather(input, output, inverse, true);
        self.stages
            .run(output, completed..self.stage_count(), inverse);
    }

    /// Transform `data` in place through `n` values of scratch.
    pub(super) fn execute<const BOUNDED: bool>(
        &self,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        execution_policy: ExecutionPolicy,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let scratch = &mut scratch[..self.n];
        self.transform_into_partitioned::<BOUNDED>(
            data,
            scratch,
            inverse,
            execution_policy,
            context,
        )?;
        self.stages.finish(scratch, data, inverse, scale);
        Ok(())
    }

    /// Transform `input` into `output` through `n` values of scratch; the
    /// gather reads caller storage directly, with no staging copy.
    #[expect(
        clippy::too_many_arguments,
        reason = "borrowed source, destination and scratch stay explicit beside the call options"
    )]
    pub(super) fn execute_from(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        inverse: bool,
        scale: S,
        execution_policy: ExecutionPolicy,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let scratch = &mut scratch[..self.n];
        self.transform_into_partitioned::<false>(
            input,
            scratch,
            inverse,
            execution_policy,
            context,
        )?;
        self.stages.finish(scratch, output, inverse, scale);
        Ok(())
    }

    /// Gather and run every stage but the last into `output`.
    pub(super) fn transform_into_partitioned<const BOUNDED: bool>(
        &self,
        input: &[Complex<S>],
        output: &mut [Complex<S>],
        inverse: bool,
        execution_policy: ExecutionPolicy,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let stage_end = self.stage_count().saturating_sub(1);
        let first_completed = if BOUNDED {
            self.stages.gather_bounded(input, output, inverse, false)
        } else {
            self.stages.gather(input, output, inverse, false)
        };

        // A stage of radix r and width w only mixes values inside contiguous
        // blocks of r * w, and each earlier stage's block size divides the
        // later one's. Stages [first_completed, prefix_end) are therefore
        // independent across chunks of the last prefix stage's block size and
        // can run on disjoint partitions. The stage(s) after the prefix span
        // whole chunks and run sequentially; the final stage is left to the
        // caller so it can fuse scaling and write-back.
        let prefix_end = stage_end
            .saturating_sub(1)
            .max(first_completed)
            .min(stage_end);
        let parallel_plan = self.parallel_plan(execution_policy, prefix_end, context);
        if parallel_plan.worker_count() > 1 {
            self.execute_parallel_prefix(
                output,
                first_completed..prefix_end,
                self.stages.block(prefix_end - 1),
                inverse,
                parallel_plan,
                context,
            )?;
            self.stages.run(output, prefix_end..stage_end, inverse);
        } else {
            self.stages.run(output, first_completed..stage_end, inverse);
        }
        Ok(())
    }

    pub(super) fn parallel_plan(
        &self,
        policy: ExecutionPolicy,
        prefix_end: usize,
        context: ExecutionContext<'_>,
    ) -> ExecutionPlan {
        if self.n < FFT_PARALLEL_THRESHOLD || prefix_end == 0 {
            return context.plan(
                ExecutionPolicy::Sequential,
                1,
                scioxide_core::PlanLimits {
                    work_per_item: core::num::NonZeroUsize::MIN,
                    auto_threshold: NonZeroUsize::MIN,
                    worker_limit: FFT_AUTO_WORKERS,
                },
            );
        }
        let chunk_size = self.stages.block(prefix_end - 1);
        let work_items = self.n / chunk_size;
        context.plan(
            policy,
            work_items,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: FFT_AUTO_WORKERS,
            },
        )
    }

    pub(super) fn execute_parallel_prefix(
        &self,
        output: &mut [Complex<S>],
        stages: Range<usize>,
        chunk_size: usize,
        inverse: bool,
        plan: ExecutionPlan,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let mut remaining = output;
        // `parallel_plan` already caps both automatic and explicit requests at
        // FFT_AUTO_WORKERS. Keep its partition claims on the stack so prepared
        // streaming consumers do not allocate on every transform.
        let mut partitions = plan.partitions();
        let slots: [_; FFT_AUTO_WORKERS.get()] = core::array::from_fn(|_| {
            partitions
                .next()
                .map_or_else(PartitionSlot::empty, |partition| {
                    let partition_len = partition.range().len() * chunk_size;
                    let (worker_output, next) =
                        core::mem::take(&mut remaining).split_at_mut(partition_len);
                    remaining = next;
                    PartitionSlot::new(worker_output)
                })
        });
        context.execute_partitioned(
            plan,
            &slots[..plan.worker_count()],
            &|_, worker_output, _| {
                self.stages.run(worker_output, stages.clone(), inverse);
                Ok(())
            },
        )?;

        Ok(())
    }

    pub(super) fn heap_bytes(&self) -> Result<usize, SciOxideError> {
        self.stages.heap_bytes()
    }
}

/// Stage twiddles and odd-radix roots.
type StageTables<S> = (Vec<Complex<S>>, Vec<Complex<S>>);

/// Twiddles `w^(branch * index)` for branches `1..radix` and one period of
/// odd-radix roots, with `w = exp(-2*pi*i / (radix * width))`.
fn stage_tables<S: scioxide_core::RealScalar>(
    radix: usize,
    width: usize,
    length: Argument,
) -> Result<StageTables<S>, SciOxideError> {
    let twiddle_count = checked_mul(
        radix - 1,
        width,
        scioxide_core::quantity!("fft_stage_twiddle_count"),
    )?;
    let mut twiddles = try_with_capacity(
        twiddle_count,
        scioxide_core::quantity!("fft_stage_twiddles"),
    )?;
    let stage_size = checked_mul(radix, width, scioxide_core::quantity!("fft_stage_size"))?;
    // Layout: twiddles[(branch - 1) * width + index] = w^(branch * index)
    // with w = exp(-2*pi*i / (radix * width)); branch 0 is the identity and
    // is not stored. Only forward roots are kept; kernels conjugate them for
    // the inverse. Radix-4 tables reuse independently evaluated first-octant
    // roots through exact swaps/signs: no rounded product relates entries.
    // This cuts plan-time trigonometry by six without recurrence drift.
    if radix == 4 && width.is_multiple_of(2) {
        for index in 0..=width / 2 {
            twiddles.push(reduced_root::<S>(
                index as u128,
                stage_size as u128,
                false,
                length,
            )?);
        }
        for index in width / 2 + 1..width {
            let root = twiddles[width - index];
            twiddles.push(Complex::new(-root.im, -root.re));
        }
        for branch in 2..4 {
            for index in 0..width {
                let exponent = branch * index;
                let root = if exponent < width {
                    twiddles[exponent]
                } else if exponent < 2 * width {
                    let root = twiddles[exponent - width];
                    Complex::new(root.im, -root.re)
                } else {
                    let root = twiddles[exponent - 2 * width];
                    Complex::new(-root.re, -root.im)
                };
                twiddles.push(root);
            }
        }
    } else {
        for branch in 1..radix {
            for index in 0..width {
                let exponent = branch as u128 * index as u128;
                twiddles.push(reduced_root::<S>(
                    exponent,
                    stage_size as u128,
                    false,
                    length,
                )?);
            }
        }
    }

    // Store one period of odd-radix roots. Direct codelets index it with
    // output*input modulo radix; symmetric codelets use its first roots.
    // Radices 2 and 4 need only hard-coded signs and quarter turns.
    let root_count = crate::kernel::root_count(radix).unwrap_or(0);
    let mut roots = try_with_capacity(root_count, scioxide_core::quantity!("fft_radix_roots"))?;
    for exponent in 0..root_count {
        roots.push(reduced_root::<S>(
            exponent as u128,
            radix as u128,
            false,
            length,
        )?);
    }
    Ok((twiddles, roots))
}

fn permutation_index(mut source: usize, radices: &[usize]) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("fft_permutation_index");
    radices
        .iter()
        .rev()
        .copied()
        .try_fold(0, |destination, radix| {
            let digit = source % radix;
            source /= radix;
            checked_add(checked_mul(destination, radix, q)?, digit, q)
        })
}

pub(super) fn compute_mixed_radix_permutation(
    n: usize,
    radices: &[usize],
) -> Result<Vec<usize>, SciOxideError> {
    let mut permutation = try_filled(n, 0, scioxide_core::quantity!("fft_permutation"))?;
    for source in 0..n {
        permutation[permutation_index(source, radices)?] = source;
    }
    Ok(permutation)
}
