//! Typed complex FFT codelet dispatch.
//!
//! [`StagePlan`] owns a mixed-radix stage sequence: its twiddle tables, its
//! digit-gather permutation and the leaves selected with verified CPU
//! capabilities. Construction proves every leaf precondition once: supported
//! radices, table lengths, the decimation-in-time width chain and whole stage
//! blocks of the transform length. Specialised leaves may encode width in
//! their machine code and index without bounds checks, so the immutable plan
//! carries that proof and each execution entry checks only its borrowed
//! lengths, once per call rather than once per stage.

mod arch;
mod batch;
pub(crate) mod block;
pub(crate) mod scalar;
#[cfg(test)]
mod tests;
mod tile;

pub(crate) use arch::{nufft_interpolate3, nufft_spread, nufft_tile, run_pass};

use alloc::vec::Vec;
use core::ops::Range;
use scioxide_core::backend::{
    ComplexSlice, ComplexSliceMut, KernelDispatch, ScalarValue, checked_add, checked_mul,
};
use scioxide_core::{Argument, Complex, RealScalar, SciOxideError};

/// Immutable plan-owned gather indices, range-checked once at preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Permutation(Vec<usize>);
impl Permutation {
    pub(crate) fn new(indices: Vec<usize>) -> Result<Self, SciOxideError> {
        if let Some(&index) = indices.iter().find(|&&index| index >= indices.len()) {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("fft_permutation"),
                index,
                indices.len(),
            ));
        }
        Ok(Self(indices))
    }
}
impl core::ops::Deref for Permutation {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        &self.0
    }
}

/// # Safety
/// The selected ISA must be available. `data` contains whole nonzero
/// `radix * width` blocks; twiddles have `(radix - 1) * width` entries and
/// roots have `root_count(radix)` entries. Any specialized radix/width must
/// match the selected leaf.
type StageLeaf<S> = unsafe fn(
    data: &mut [Complex<S>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
);

/// # Safety
/// The selected ISA must be available. All slices have the plan length,
/// divisible by 16, except `second_twiddles`, which has 12 entries. Every
/// permutation source must index `input`.
type FusedGatherLeaf<S> = unsafe fn(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    permutation: &[usize],
    second_twiddles: &[Complex<S>],
);

/// # Safety
/// The selected ISA must be available. Data contains whole `16 * width`
/// blocks, and the two tables have `3 * width` and `12 * width` entries.
/// The first width must match the selected specialization.
type FusedStageLeaf<S> = unsafe fn(&mut [Complex<S>], &[Complex<S>], &[Complex<S>]);

/// # Safety
/// The selected ISA must be available. Both buffers have `radix * width`
/// entries, with the stage's complete twiddle/root tables, matching any
/// selected specialization. `scale` must be finite.
type FinalStageLeaf<S> = unsafe fn(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    radix: usize,
    width: usize,
    twiddles: &[Complex<S>],
    roots: &[Complex<S>],
    scale: S,
);

/// # Safety
/// The selected ISA must be available. All slices have the same length and
/// every permutation source must index the input buffer.
type PermutationLeaf<S> = unsafe fn(&[Complex<S>], &mut [Complex<S>], &[usize]);

/// # Safety
/// The selected ISA must be available. All buffers/permutation have the same
/// length, divisible by the supported radix, and every source is in range.
/// `roots` contains `root_count(radix)` entries for this width-one stage.
type GatherLeaf<S> = unsafe fn(
    input: &[Complex<S>],
    output: &mut [Complex<S>],
    permutation: &[usize],
    radix: usize,
    roots: &[Complex<S>],
);

/// In-place mixed-radix stage over whole `radix * width` blocks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StageKernel<S>(StageLeaf<S>);
impl<S> StageKernel<S> {
    pub(crate) const fn new(leaf: StageLeaf<S>) -> Self {
        Self(leaf)
    }
}

/// Digit gather fused with radix-4 stages of widths 1 and 4 (16-point blocks).
/// The second leaf checks bounded inputs in the same gathered loads.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedGatherKernel<S>(FusedGatherLeaf<S>, FusedGatherLeaf<S>);
impl<S> FusedGatherKernel<S> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn new(leaf: FusedGatherLeaf<S>, bounded: FusedGatherLeaf<S>) -> Self {
        Self(leaf, bounded)
    }
}

/// Radix-4 stage at `width` fused with radix 4 at `4 * width`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedStageKernel<S>(FusedStageLeaf<S>);
impl<S> FusedStageKernel<S> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn new(leaf: FusedStageLeaf<S>) -> Self {
        Self(leaf)
    }
}

/// Out-of-place scaled final stage.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FinalStageKernel<S>(FinalStageLeaf<S>);
impl<S> FinalStageKernel<S> {
    pub(crate) const fn new(leaf: FinalStageLeaf<S>) -> Self {
        Self(leaf)
    }
}

/// Digit-permutation copy, with ordinary and bounded-input leaves.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PermutationKernel<S>(PermutationLeaf<S>, PermutationLeaf<S>);
impl<S> PermutationKernel<S> {
    pub(crate) const fn new(leaf: PermutationLeaf<S>, bounded: PermutationLeaf<S>) -> Self {
        Self(leaf, bounded)
    }
}

/// Digit gather fused with a width-one first stage, ordinary and bounded.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GatherKernel<S>(GatherLeaf<S>, GatherLeaf<S>);
impl<S> GatherKernel<S> {
    pub(crate) const fn new(leaf: GatherLeaf<S>, bounded: GatherLeaf<S>) -> Self {
        Self(leaf, bounded)
    }
}

pub(crate) const fn root_count(radix: usize) -> Option<usize> {
    match radix {
        2 | 4 => Some(0),
        3 | 5 | 7 => Some(radix),
        _ => None,
    }
}

/// Direction-specific leaves stored by one stage plan.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StageKernels<S> {
    pub(crate) forward: StageKernel<S>,
    pub(crate) inverse: StageKernel<S>,
    pub(crate) fused_gather_forward: Option<FusedGatherKernel<S>>,
    pub(crate) fused_gather_inverse: Option<FusedGatherKernel<S>>,
    pub(crate) fused_stage_forward: Option<FusedStageKernel<S>>,
    pub(crate) fused_stage_inverse: Option<FusedStageKernel<S>>,
    pub(crate) final_forward: FinalStageKernel<S>,
    pub(crate) final_inverse: FinalStageKernel<S>,
    pub(crate) permute: PermutationKernel<S>,
    pub(crate) gather_forward: GatherKernel<S>,
    pub(crate) gather_inverse: GatherKernel<S>,
}

/// Immutable leaves selected with verified CPU capabilities at preparation.
/// Projection uses core's sealed scalar representation without casts.
#[derive(Debug, Clone, Copy)]
enum SelectedKernels {
    F32(StageKernels<f32>),
    F64(StageKernels<f64>),
}

/// Exact component representation of one selected leaf set. A stage selects
/// its leaves from the plan scalar's representation, so projections of that
/// scalar's borrows always agree with the selected variant.
trait Lane: Copy {
    fn kernels(leaves: &SelectedKernels) -> &StageKernels<Self>;
    fn slice<S: RealScalar>(values: &[Complex<S>]) -> &[Complex<Self>];
    fn slice_mut<S: RealScalar>(values: &mut [Complex<S>]) -> &mut [Complex<Self>];
    fn value<S: RealScalar>(value: S) -> Self;
}

macro_rules! lane {
    ($scalar:ty, $variant:ident) => {
        impl Lane for $scalar {
            #[inline(always)]
            fn kernels(leaves: &SelectedKernels) -> &StageKernels<Self> {
                match leaves {
                    SelectedKernels::$variant(kernels) => kernels,
                    _ => unreachable!("FFT leaves are selected for the plan scalar"),
                }
            }
            #[inline(always)]
            fn slice<S: RealScalar>(values: &[Complex<S>]) -> &[Complex<Self>] {
                match S::complex_slice(values) {
                    ComplexSlice::$variant(values) => values,
                    _ => unreachable!("FFT leaves are selected for the plan scalar"),
                }
            }
            #[inline(always)]
            fn slice_mut<S: RealScalar>(values: &mut [Complex<S>]) -> &mut [Complex<Self>] {
                match S::complex_slice_mut(values) {
                    ComplexSliceMut::$variant(values) => values,
                    _ => unreachable!("FFT leaves are selected for the plan scalar"),
                }
            }
            #[inline(always)]
            fn value<S: RealScalar>(value: S) -> Self {
                match S::value(value) {
                    ScalarValue::$variant(value) => value,
                    _ => unreachable!("FFT leaves are selected for the plan scalar"),
                }
            }
        }
    };
}
lane!(f32, F32);
lane!(f64, F64);

/// One validated decimation-in-time stage of a [`StagePlan`].
///
/// Layout: `twiddles[(branch - 1) * width + index] = w^(branch * index)` with
/// `w = exp(-2*pi*i / (radix * width))`; `roots` holds one period of radix
/// roots for odd radices only. Fields stay private so the geometry proven at
/// preparation cannot change before execution.
#[derive(Debug, Clone)]
pub(crate) struct Stage<S> {
    radix: usize,
    width: usize,
    twiddles: Vec<Complex<S>>,
    roots: Vec<Complex<S>>,
    leaves: SelectedKernels,
    pair_with_next: bool,
}

impl<S: RealScalar> Stage<S> {
    /// Validate one stage's geometry and tables, then select its leaves.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` attributed to `length` for an unsupported radix,
    /// zero or overflowing width, or tables that do not match the geometry.
    pub(crate) fn new(
        dispatch: KernelDispatch,
        radix: usize,
        width: usize,
        twiddles: Vec<Complex<S>>,
        roots: Vec<Complex<S>>,
        length: Argument,
    ) -> Result<Self, SciOxideError> {
        let expected_twiddles = radix
            .checked_sub(1)
            .and_then(|branches| branches.checked_mul(width));
        let valid = width != 0
            && radix.checked_mul(width).is_some()
            && root_count(radix) == Some(roots.len())
            && expected_twiddles == Some(twiddles.len());
        if !valid {
            return Err(SciOxideError::invalid_input(
                length,
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "FFT stage tables do not match a supported stage geometry",
            ));
        }
        let leaves = match S::complex_slice(&[]) {
            ComplexSlice::F32(_) => SelectedKernels::F32(arch::kernels_f32(dispatch, radix, width)),
            ComplexSlice::F64(_) => SelectedKernels::F64(arch::kernels_f64(dispatch, radix, width)),
        };
        Ok(Self {
            radix,
            width,
            twiddles,
            roots,
            leaves,
            pair_with_next: false,
        })
    }

    const fn block(&self) -> usize {
        self.radix * self.width
    }

    fn has_fused_gather(&self) -> bool {
        match &self.leaves {
            SelectedKernels::F32(k) => k.fused_gather_forward.is_some(),
            SelectedKernels::F64(k) => k.fused_gather_forward.is_some(),
        }
    }
}

/// Leading work fused with the digit gather.
#[derive(Debug, Clone, Copy)]
enum GatherPrefix {
    Permutation,
    Single,
    Pair,
    Triple,
}

impl GatherPrefix {
    fn select<S: RealScalar>(n: usize, stages: &[Stage<S>], end: usize) -> Self {
        if end >= 2 && stages[0].radix == 4 && stages[1].radix == 4 && stages[0].has_fused_gather()
        {
            if end >= 3 && stages[2].radix == 4 {
                Self::Triple
            } else {
                Self::Pair
            }
        } else if n <= 4096 && end >= 1 {
            Self::Single
        } else {
            Self::Permutation
        }
    }
}

/// A validated mixed-radix stage sequence for one transform length.
#[derive(Debug, Clone)]
pub(crate) struct StagePlan<S> {
    n: usize,
    permutation: Permutation,
    stages: Vec<Stage<S>>,
    gather_all: GatherPrefix,
    gather_prefix: GatherPrefix,
    /// Verified dispatch selecting the build of across-lane passes.
    dispatch: KernelDispatch,
}

impl<S: RealScalar> StagePlan<S> {
    /// Prove the stage chain for length `n`: stage `k` combines `radix_k`
    /// interleaved sub-transforms whose width is the product of all earlier
    /// radices, the final block is `n`, and the permutation has length `n`.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` for a permutation of another length and
    /// `InvalidInput` attributed to `length` for a broken width chain.
    pub(crate) fn new(
        n: usize,
        permutation: Permutation,
        mut stages: Vec<Stage<S>>,
        dispatch: KernelDispatch,
        length: Argument,
    ) -> Result<Self, SciOxideError> {
        if permutation.len() != n {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(n),
                scioxide_core::Dimensions::one(permutation.len()),
            ));
        }
        let covered = stages.iter().try_fold(1, |width, stage| {
            (stage.width == width).then(|| stage.block())
        });
        if covered != Some(n) {
            return Err(SciOxideError::invalid_input(
                length,
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "mixed-radix stages did not cover the transform length",
            ));
        }
        for index in 1..stages.len() {
            stages[index - 1].pair_with_next =
                stages[index - 1].radix == 4 && stages[index].radix == 4;
        }
        let gather_all = GatherPrefix::select(n, &stages, stages.len());
        let gather_prefix = GatherPrefix::select(n, &stages, stages.len().saturating_sub(1));
        Ok(Self {
            n,
            permutation,
            stages,
            gather_all,
            gather_prefix,
            dispatch,
        })
    }

    pub(crate) fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Block size `radix * width` of stage `index`; later blocks are multiples.
    pub(crate) fn block(&self, index: usize) -> usize {
        self.stages[index].block()
    }

    #[cfg(all(test, feature = "std"))]
    pub(crate) fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    #[cfg(all(test, feature = "std"))]
    pub(crate) fn geometry(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.stages.iter().map(|stage| (stage.radix, stage.width))
    }

    pub(crate) fn heap_bytes(&self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("fft_plan_bytes");
        let initial = checked_add(
            checked_mul(self.permutation.0.capacity(), size_of::<usize>(), q)?,
            checked_mul(self.stages.capacity(), size_of::<Stage<S>>(), q)?,
            q,
        )?;
        self.stages.iter().try_fold(initial, |bytes, stage| {
            let tables = checked_add(stage.twiddles.capacity(), stage.roots.capacity(), q)?;
            checked_add(bytes, checked_mul(tables, size_of::<Complex<S>>(), q)?, q)
        })
    }
}

impl<S: RealScalar> StagePlan<S> {
    /// Transform the `run` lanes of `values` in place across lanes, a block
    /// of [`block::WIDTH`] lanes at a time through `rows`, and multiply them
    /// by `scale`, for lengths up to [`block::ROWS`]; see [`tile`]. The pass
    /// runs as built for the plan's verified dispatch.
    pub(crate) fn transform_lanes(
        &self,
        values: &mut [Complex<S>],
        run: crate::kernel::block::Run,
        (inverse, scale): (bool, S),
        rows: &mut [block::Block<S, { block::WIDTH }>; block::ROWS],
    ) {
        admit(run.len == self.n && self.n <= block::ROWS && scale.is_finite());
        arch::run_pass(
            self.dispatch,
            LanePass {
                plan: self,
                values,
                run,
                inverse,
                scale,
                rows,
            },
        );
    }
}

/// One run of complex lanes transformed across lanes.
struct LanePass<'p, 'v, S: RealScalar> {
    plan: &'p StagePlan<S>,
    values: &'v mut [Complex<S>],
    run: crate::kernel::block::Run,
    inverse: bool,
    scale: S,
    rows: &'v mut [block::Block<S, { block::WIDTH }>; block::ROWS],
}

impl<S: RealScalar> block::BlockPass for LanePass<'_, '_, S> {
    type Output = ();
    #[inline(always)]
    fn run(self) {
        tile::transform_run(
            self.plan,
            self.values,
            self.run,
            (self.inverse, self.scale),
            self.rows,
        );
    }
}

/// The single release guard of an execution entry. Plans only build valid
/// geometry, so failure is an asserted internal inconsistency; in release
/// builds it still prevents unsafe leaf entry with mismatched borrows.
fn admit(valid: bool) {
    assert!(valid, "invalid FFT kernel geometry");
}
