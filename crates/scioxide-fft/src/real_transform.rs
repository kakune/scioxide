//! Discrete cosine and sine transform plans of types I–IV.
mod lanes;
mod planning;
mod publication;
mod short;
mod staging;
mod tables;
#[cfg(test)]
mod tests;

use crate::kernel::block::Run;
use crate::kernel::block::{
    Addressing, Adjacent, Block, Contiguous, Get, Interleaved, Layout, Load, Map, Put, ROWS, Reals,
    Scale, Spread, Store, WIDTH, deinterleave, deinterleave_partial, interleave,
    interleave_partial,
};
use crate::lanes::{LaneMode, Lanes};
use crate::numeric::reduced_root;
use crate::numeric::scalar_from_usize_rounded;
use crate::real_fft::RealFftPlan;
use crate::scaling::BinaryScale;
use crate::types::Normalization;
use crate::validation::real_transform_overflow as overflow;
use crate::{DctType, DstType, FftDirection};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{Argument, Complex, ExecutionContext, ExecutionPolicy, SciOxideError};
use tables::*;

/// Selects a DCT or DST and its type.
///
/// The type is that of the forward transform; an inverse plan computes its
/// dual type. The definitions are listed on [`RealTransformPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealTransformKind {
    /// Discrete cosine transform.
    Dct(DctType),
    /// Discrete sine transform.
    Dst(DstType),
}
impl From<DctType> for RealTransformKind {
    fn from(value: DctType) -> Self {
        Self::Dct(value)
    }
}
impl From<DstType> for RealTransformKind {
    fn from(value: DstType) -> Self {
        Self::Dst(value)
    }
}
impl RealTransformKind {
    const fn kernel(self) -> KernelType {
        match self {
            Self::Dct(DctType::I) => KernelType::CosineOne,
            Self::Dst(DstType::I) => KernelType::SineOne,
            Self::Dct(DctType::II) | Self::Dst(DstType::II) => KernelType::Two,
            Self::Dct(DctType::III) | Self::Dst(DstType::III) => KernelType::Three,
            Self::Dct(DctType::IV) | Self::Dst(DstType::IV) => KernelType::Four,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelType {
    CosineOne,
    SineOne,
    Two,
    Three,
    Four,
}

/// Computes a discrete cosine or sine transform (DCT or DST) of type I–IV and
/// fixed length.
///
/// # Definitions
///
/// For real input `x` of length `n`, the output `y` has length `n` and indices
/// `0 ≤ k < n`. Sum bounds are inclusive and an empty sum is zero. These are
/// the unscaled transforms, used with [`Normalization::Unscaled`]:
///
/// | Type | DCT `y[k]` | Minimum `n` | Round-trip factor `R` |
/// |---|---|---|---|
/// | I | `x[0] + (−1)ᵏ x[n−1] + 2 Σ_{j=1}^{n−2} x[j] cos(πjk/(n−1))` | 2 | `2(n−1)` |
/// | II | `Σ_{j=0}^{n−1} x[j] cos(π(j+½)k/n)` | 1 | `n` |
/// | III | `x[0] + 2 Σ_{j=1}^{n−1} x[j] cos(πj(k+½)/n)` | 1 | `n` |
/// | IV | `Σ_{j=0}^{n−1} x[j] cos(π(j+½)(k+½)/n)` | 1 | `n/2` |
///
/// | Type | DST `y[k]` | Minimum `n` | Round-trip factor `R` |
/// |---|---|---|---|
/// | I | `Σ_{j=0}^{n−1} x[j] sin(π(j+1)(k+1)/(n+1))` | 1 | `(n+1)/2` |
/// | II | `Σ_{j=0}^{n−1} x[j] sin(π(j+½)(k+1)/n)` | 1 | `n` |
/// | III | `(−1)ᵏ x[n−1] + 2 Σ_{j=0}^{n−2} x[j] sin(π(j+1)(k+½)/n)` | 1 | `n` |
/// | IV | `Σ_{j=0}^{n−1} x[j] sin(π(j+½)(k+½)/n)` | 1 | `n/2` |
///
/// The forward transform computes the selected type. The inverse transform
/// computes the dual type: I for I, III for II, II for III and IV for IV.
/// Applying both unscaled directions multiplies the input by `R`.
/// [`Normalization::Forward`] divides the forward output by `R`, and
/// [`Normalization::Inverse`] divides the inverse output by `R`; both give an
/// identity round trip.
///
/// [`Normalization::Orthonormal`] makes the transform matrix orthogonal. It
/// modifies the type actually computed (the dual type for an inverse plan) as
/// follows, in order: weight the input, apply the scale, then divide the
/// listed outputs by `√2`.
///
/// | Computed DCT | Input weight | Scale | Outputs divided by `√2` |
/// |---|---|---|---|
/// | I | `x[0]`, `x[n−1]` times `√2` | `1/√(2(n−1))` | `y[0]`, `y[n−1]` |
/// | II | none | `√(2/n)` | `y[0]` |
/// | III | `x[0]` times `√2` | `1/√(2n)` | none |
/// | IV | none | `√(2/n)` | none |
///
/// | Computed DST | Input weight | Scale | Outputs divided by `√2` |
/// |---|---|---|---|
/// | I | none | `√(2/(n+1))` | none |
/// | II | none | `√(2/n)` | `y[n−1]` |
/// | III | `x[n−1]` times `√2` | `1/√(2n)` | none |
/// | IV | none | `√(2/n)` | none |
///
/// At `n = 1` an endpoint that appears twice in this table is adjusted once.
///
/// # Method
///
/// Each transform is computed with one FFT in `O(n log n)` operations. DCT-I
/// and DST-I use a real FFT of length `2(n−1)` and `2(n+1)` of the symmetric
/// extension. Types II and III of even length use a complex FFT of length
/// `n/2` (Makhoul, 1980), and of odd length a real FFT of length `n`. Type IV
/// uses a complex FFT of length `n/2` for even `n` and a real FFT of length `n`
/// for odd `n`. DST types II–IV are computed from the DCT of the same type by
/// reversing the order and alternating the signs of the samples.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_fft::{DctType, FftDirection, Normalization, RealTransformKind};
/// use scioxide_fft::{RealTransformPlan, RealTransformPlanSpec};
/// let kind = RealTransformKind::Dct(DctType::II);
/// let spec = RealTransformPlanSpec::new(4, kind, FftDirection::Forward, Normalization::Orthonormal);
/// let plan = RealTransformPlan::<f64>::new(spec)?;
/// let mut workspace = plan.workspace(())?;
/// let mut output = [0.0; 4];
/// for amplitude in [1.0, 2.0] {
///     let cx = ExecutionContext::sequential();
///     plan.execute_into(&[amplitude; 4], &mut output, &mut workspace, cx)?;
///     // Orthonormal DCT-II of a constant: y[0] = √(2/4)·4a/√2 = 2a, other bins 0.
///     assert!((output[0] - 2.0 * amplitude).abs() < 1e-12);
///     assert!(output[1..].iter().all(|x| x.abs() < 1e-12));
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. Makhoul, "A fast cosine transform in one and two dimensions",
///   *IEEE Trans. Acoust. Speech Signal Process.* 28(1), 1980.
/// - V. Britanak, P. C. Yip and K. R. Rao, *Discrete Cosine and Sine
///   Transforms: General Properties, Fast Algorithms and Integer
///   Approximations*, Academic Press, 2007.
#[derive(Debug, Clone)]
#[must_use]
pub struct RealTransformPlan<S: scioxide_core::RealScalar> {
    plan_bytes: usize,
    spec: RealTransformPlanSpec,
    kernel: Kernel<S>,
    phases: Vec<Complex<S>>,
    scaling: Scaling<S>,
    short: Option<ShortTransform<S>>,
    scratch_len: usize,
    /// Lane-block layout for N-D passes of even II/III/IV kernels.
    blocks: Option<LaneBlocks>,
    /// Natural slot to native digit-order row for smooth lane blocks.
    digit_rows: Vec<usize>,
    /// Bound on normalized outputs before binary restoration; infinite for
    /// providers, whose outputs are checked individually.
    output_bound: S,
}

#[derive(Debug, Clone)]
enum Kernel<S: scioxide_core::RealScalar> {
    Real(RealFftPlan<S>),
    EvenHalf(RealFftPlan<S>),
    EvenIv(crate::complex::ComplexCore<S>),
}

/// N-D lane passes of an even II/III/IV kernel transform blocks of [`WIDTH`]
/// lanes. Scalar/AVX2 smooth children use stack rows; other native smooth
/// children reserve two lane buffers. Scratch also covers single-lane fallback
/// calls. `dispatch` selects block passes.
#[derive(Debug, Clone, Copy)]
struct LaneBlocks {
    scratch_len: usize,
    dispatch: scioxide_core::backend::KernelDispatch,
}

#[derive(Debug, Clone, Copy)]
struct Scaling<S> {
    input_first: S,
    input_last: S,
    output_scale: S,
    output_first: S,
    output_last: S,
    output_single: S,
}

/// The same prepared, factored butterfly serves public vectors and bounded
/// tensor lanes. Keeping endpoint factors outside the sum preserves the raw
/// transform's cancellation and avoids a separate dense two-by-two kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShortTransform<S> {
    kernel: ShortKernel<S>,
    scaling: Scaling<S>,
    sine: bool,
}

#[derive(Debug, Clone, Copy)]
enum ShortKernel<S> {
    One,
    Two(S),
    Three(S),
    Four(Complex<S>),
}

/// Describes a DCT or DST plan: length, type, direction and normalization.
#[derive(Debug, Clone, Copy)]
pub struct RealTransformPlanSpec {
    length: usize,
    kind: RealTransformKind,
    direction: FftDirection,
    normalization: Normalization,
    execution_policy: ExecutionPolicy,
    backend_strategy: crate::FftBackendStrategy,
}
impl RealTransformPlanSpec {
    /// Returns the transform length.
    pub const fn length(&self) -> usize {
        self.length
    }
    /// Returns the transform kind.
    pub const fn kind(&self) -> RealTransformKind {
        self.kind
    }
    /// Returns the direction.
    pub const fn direction(&self) -> FftDirection {
        self.direction
    }
    /// Returns the normalization.
    pub const fn normalization(&self) -> Normalization {
        self.normalization
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution_policy
    }
    /// Returns the backend strategy.
    pub const fn backend_strategy(&self) -> crate::FftBackendStrategy {
        self.backend_strategy
    }

    /// Creates a spec with sequential execution and automatic backend selection.
    pub const fn new(
        length: usize,
        kind: RealTransformKind,
        direction: FftDirection,
        normalization: Normalization,
    ) -> Self {
        Self {
            length,
            kind,
            direction,
            normalization,
            execution_policy: ExecutionPolicy::Sequential,
            backend_strategy: crate::FftBackendStrategy::Automatic,
        }
    }
    /// Sets the execution policy.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }
    /// Sets the backend strategy.
    pub const fn with_backend_strategy(mut self, backend: crate::FftBackendStrategy) -> Self {
        self.backend_strategy = backend;
        self
    }
}
impl<S: scioxide_core::RealScalar> RealTransformPlan<S> {
    /// Bound a bounded lane's normalized output, including its fixed binary
    /// restoration. Providers keep an infinite bound and checked output.
    pub(crate) fn bounded_output_bound(&self) -> S {
        self.output_bound.scalbn(2)
    }
    /// Returns the bytes this plan occupies, including its FFT tables.
    pub const fn plan_bytes(&self) -> usize {
        self.plan_bytes
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        let q = scioxide_core::quantity!("fft_plan_bytes");
        let child = match &self.kernel {
            Kernel::Real(p) | Kernel::EvenHalf(p) => p.plan_bytes() - size_of::<RealFftPlan<S>>(),
            Kernel::EvenIv(p) => p.plan_bytes() - size_of::<crate::complex::ComplexCore<S>>(),
        };
        checked_add(
            checked_add(
                size_of::<Self>(),
                crate::memory::vector_bytes(&self.phases),
                q,
            )?,
            checked_add(child, crate::memory::vector_bytes(&self.digit_rows), q)?,
            q,
        )
    }

    /// Returns the spec the plan was prepared from.
    pub const fn spec(&self) -> &RealTransformPlanSpec {
        &self.spec
    }

    /// Prepares a plan for the processor of
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the length is zero, or one for DCT-I, or if a
    ///   required backend is unavailable.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the tables or
    ///   workspace size cannot be represented or reserved.
    pub fn new(spec: RealTransformPlanSpec) -> Result<Self, SciOxideError> {
        Self::build(spec, scioxide_core::ExecutionContext::sequential_native())
    }
    /// Prepares a plan with the processor features and provider of `context`.
    ///
    /// See [Backends and providers](crate#backends-and-providers).
    ///
    /// # Errors
    ///
    /// - The errors of [`RealTransformPlan::new`].
    /// - The errors of the provider, if it fails to prepare a kernel.
    pub fn build<'a>(
        spec: RealTransformPlanSpec,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
    ) -> Result<Self, SciOxideError> {
        Self::with_backend_context(
            spec.length,
            spec.kind,
            spec.direction,
            spec.normalization,
            spec.execution_policy,
            spec.backend_strategy,
            context,
            scioxide_core::argument!("spec.length"),
        )
    }

    /// Validate one tensor-axis length, attributing
    /// a rejected extent to its `length` argument.
    pub(crate) fn validate_nd_length(
        n: usize,
        kind: RealTransformKind,
        length: Argument,
    ) -> Result<(), SciOxideError> {
        validate_length(n, kind.kernel(), length)
    }

    /// Returns the backend of each complex FFT the plan runs.
    ///
    /// Items can repeat.
    pub fn resolved_backends(&self) -> impl Iterator<Item = crate::FftBackend> + '_ {
        let (complex, real) = match &self.kernel {
            Kernel::Real(p) | Kernel::EvenHalf(p) => (None, Some(p)),
            Kernel::EvenIv(p) => (Some(p.resolved_backend()), None),
        };
        complex
            .into_iter()
            .chain(real.into_iter().flat_map(|p| p.resolved_backends()))
    }

    /// Returns the scratch length, in complex elements, that execution needs.
    ///
    /// Let `R(m)` be the scratch of a packed real FFT plan of length `m` (see
    /// [`FftPlan::workspace_requirements`](crate::FftPlan::workspace_requirements))
    /// and `C(m)` the algorithm scratch of a complex FFT of length `m` (`m` for
    /// 2/3/5/7-smooth `m > 1`, none for `m = 1`). The scratch length is
    ///
    /// | Transform | Scratch |
    /// |---|---|
    /// | DCT-I | `n + R(2(n−1))` |
    /// | DST-I | `n + R(2(n+1))` |
    /// | II, III, `n` odd | `n + R(n)` |
    /// | II, III, `n ≥ 4` even | `3n/2 + 1 + C(n/2)` |
    /// | IV, `n` odd | `n + R(n)` |
    /// | IV, `n ≥ 4` even | `3n/2 + C(n/2)` |
    /// | II, III, IV, `n ≤ 2` | `n` |
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    /// - `ResourceLimit` if it exceeds a provider's workspace limit.
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<crate::FftWorkspaceRequirements, SciOxideError> {
        self.requirements(self.scratch_len)
    }

    /// Child scratch of an N-D lane pass; see [`Self::lane_scratch_len`].
    pub(crate) fn lane_workspace_requirements(
        &self,
    ) -> Result<crate::FftWorkspaceRequirements, SciOxideError> {
        self.requirements(self.lane_scratch_len())
    }

    /// `len` complex values embedding the child's own scratch and limit.
    fn requirements(&self, len: usize) -> Result<crate::FftWorkspaceRequirements, SciOxideError> {
        let (child_scratch, limit) = match &self.kernel {
            Kernel::Real(p) | Kernel::EvenHalf(p) => (p.scratch_len(), p.workspace_byte_limit()),
            Kernel::EvenIv(p) => (p.bounded_scratch_len(), p.bounded_workspace_byte_limit()),
        };
        let requirements = crate::FftWorkspaceRequirements::new::<S>(len)?;
        requirements.with_limit(crate::workspace::nested_byte_limit(
            usize::MAX,
            requirements.bytes(),
            [(limit, child_scratch * core::mem::size_of::<Complex<S>>())],
        ))
    }

    /// Allocates a workspace of the length reported by
    /// [`RealTransformPlan::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// - The errors of [`RealTransformPlan::workspace_requirements`].
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, request: ()) -> Result<crate::FftWorkspace<S>, SciOxideError> {
        crate::workspace::allocate(&self.workspace_requirements(request)?)
    }

    /// Transforms `input` and writes the result to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` does not have the plan
    ///   length, or `scratch` is shorter than
    ///   [`RealTransformPlan::workspace_requirements`] reports.
    /// - The shared execution errors listed under [Errors](crate#errors).
    pub fn execute_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.check_buffers(input, output, scratch)?;
        context.check()?;
        // One binary scale bounds the input and endpoint weights; restoration
        // follows the unscaled child.
        let Some(exponent) = lane_exponent(input)? else {
            output.fill(S::ZERO);
            return Ok(());
        };
        if let Some(short) = self.short {
            let result = short.scaled(input, exponent)?;
            context.check()?;
            output.copy_from_slice(&result[..self.spec.length]);
            return Ok(());
        }
        let (candidate, work) = scratch.split_at_mut(self.spec.length);
        let raw = self.raw_spectrum(input, work, BinaryScale::new(-exponent), context)?;
        context.check()?;
        let restore = BinaryScale::new(exponent);
        if self.output_bound.scalbn(exponent).is_finite() {
            // Every restored output is provably finite: publish directly.
            self.publish_restored(raw, restore, |i, value| output[i] = value);
            return Ok(());
        }
        self.publish_restored(raw, restore, |i, value| candidate[i].re = value);
        if candidate.iter().any(|value| !value.re.is_finite()) {
            return Err(crate::validation::real_transform_overflow());
        }
        output
            .iter_mut()
            .zip(candidate)
            .for_each(|(out, value)| *out = value.re);
        Ok(())
    }

    /// Raw kernel actually executed: inverse plans run the dual type.
    fn raw_type(&self) -> KernelType {
        let ty = self.spec.kind.kernel();
        if self.spec.direction == FftDirection::Inverse {
            dual_type(ty)
        } else {
            ty
        }
    }

    fn check_buffers(
        &self,
        input: &[S],
        output: &[S],
        scratch: &[Complex<S>],
    ) -> Result<(), SciOxideError> {
        crate::validation::length(input.len(), self.spec.length)?;
        crate::validation::length(output.len(), self.spec.length)?;
        crate::validation::sufficient(scratch.len(), self.scratch_len)
    }
}

/// Range exponent, plus the two-bit endpoint and growth margin, of finite
/// `values`; `None` for an all-zero lane.
fn lane_exponent<S: scioxide_core::RealScalar>(values: &[S]) -> Result<Option<i32>, SciOxideError> {
    let maximum = crate::scaling::real_max_slice(values)?;
    Ok((maximum != S::ZERO).then(|| crate::scaling::range_exponent(maximum) + 2))
}
