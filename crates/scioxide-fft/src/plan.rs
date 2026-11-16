//! Fourier plan specifications and prepared one-dimensional Fourier plans.

use alloc::sync::Arc;
use core::mem::size_of;

use scioxide_core::backend::{KernelBackend, checked_add};
use scioxide_core::{Complex, ExecutionContext, ExecutionPolicy, SciOxideError};

use crate::complex::ComplexCore;
use crate::io::layout_error;
use crate::real_fft::RealFftPlan;
use crate::types::{FftPlanDiagnostics, Normalization};
use crate::{FftInput, FftOutput, FftWorkspaceRequirements};

/// Selects the forward or inverse transform of a plan.
///
/// The forward transform uses the exponent sign −1 and the inverse +1; see
/// [Conventions](crate#conventions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum FftDirection {
    /// Transform samples into frequency bins.
    Forward,
    /// Transform frequency bins into samples.
    Inverse,
}

/// Selects the input and output storage of a Fourier plan.
///
/// Packed layouts store the `⌊n/2⌋ + 1` bins with nonnegative frequency of a
/// Hermitian sequence of logical length `n`. N-D plans pack the last selected
/// axis; the other selected axes keep their full extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum FftLayout {
    /// Full complex input and output, each with the logical transform length.
    Complex,
    /// Real samples and `⌊n/2⌋ + 1` packed Hermitian frequency bins, as in
    /// [`rfft`](crate::rfft) and [`irfft`](crate::irfft).
    Real,
    /// `⌊n/2⌋ + 1` packed Hermitian samples and a real spectrum, as in
    /// [`hfft`](crate::hfft) and [`ihfft`](crate::ihfft).
    Hermitian,
}

/// Selects the CPU kernels a plan is prepared with.
///
/// See [Backends and providers](crate#backends-and-providers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum FftBackendStrategy {
    /// Uses AVX2/FMA or NEON when the planning context reports it, and the
    /// portable scalar kernels otherwise.
    Automatic,
    /// Uses the given backend, or fails preparation if it is unavailable.
    ///
    /// FFT kernels exist for [`KernelBackend::Scalar`], [`KernelBackend::Avx2Fma`]
    /// and [`KernelBackend::Neon`]. Requiring [`KernelBackend::Avx512`] fails
    /// preparation with `InvalidInput`
    /// ([`UnsupportedConfiguration`](scioxide_core::InvalidInputReason::UnsupportedConfiguration))
    /// on every host, even with a provider. Requiring a backend that the planning
    /// context does not report fails in the same way.
    Required(KernelBackend),
}

/// Describes a one-dimensional Fourier plan: length, direction, normalization
/// and layout.
///
/// The length `n` is the logical transform length; for packed layouts it is
/// the length of the real sequence, not the number of bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct FftPlanSpec {
    length: usize,
    direction: FftDirection,
    normalization: Normalization,
    layout: FftLayout,
    execution_policy: ExecutionPolicy,
    backend_strategy: FftBackendStrategy,
}

impl FftPlanSpec {
    /// Creates a spec with sequential execution and automatic backend selection.
    pub const fn new(
        length: usize,
        direction: FftDirection,
        normalization: Normalization,
        layout: FftLayout,
    ) -> Self {
        Self {
            length,
            direction,
            normalization,
            layout,
            execution_policy: ExecutionPolicy::Sequential,
            backend_strategy: FftBackendStrategy::Automatic,
        }
    }

    /// Sets the execution policy.
    pub const fn with_execution_policy(mut self, execution_policy: ExecutionPolicy) -> Self {
        self.execution_policy = execution_policy;
        self
    }

    /// Sets the backend strategy.
    pub const fn with_backend_strategy(mut self, backend_strategy: FftBackendStrategy) -> Self {
        self.backend_strategy = backend_strategy;
        self
    }

    /// Returns the logical transform length.
    pub const fn length(self) -> usize {
        self.length
    }

    /// Returns the transform direction.
    pub const fn direction(self) -> FftDirection {
        self.direction
    }

    /// Returns the normalization.
    pub const fn normalization(self) -> Normalization {
        self.normalization
    }

    /// Returns the storage layout.
    pub const fn layout(self) -> FftLayout {
        self.layout
    }

    /// Returns the execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution_policy
    }

    /// Returns the backend strategy.
    pub const fn backend_strategy(self) -> FftBackendStrategy {
        self.backend_strategy
    }
}

#[derive(Debug)]
pub(crate) struct PlanCore<S: scioxide_core::RealScalar> {
    kind: CoreKind<S>,
    bytes: usize,
}

#[derive(Debug)]
enum CoreKind<S: scioxide_core::RealScalar> {
    Complex(ComplexCore<S>),
    Real(RealFftPlan<S>),
}

impl<S: scioxide_core::RealScalar> PlanCore<S> {
    fn new(kind: CoreKind<S>) -> Result<Self, SciOxideError> {
        let (bytes, inline) = match &kind {
            CoreKind::Complex(p) => (p.plan_bytes(), size_of::<ComplexCore<S>>()),
            CoreKind::Real(p) => (p.plan_bytes(), size_of::<RealFftPlan<S>>()),
        };
        let bytes = checked_add(
            bytes - inline,
            size_of::<Self>() + 2 * size_of::<usize>(),
            scioxide_core::quantity!("fft_shared_plan_storage"),
        )?;
        checked_add(
            size_of::<FftPlan<S>>(),
            bytes,
            scioxide_core::quantity!("fft_shared_plan_storage"),
        )?;
        Ok(Self { kind, bytes })
    }
    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Computes a one-dimensional complex, real or Hermitian DFT of fixed length.
///
/// The plan fixes the length, direction, normalization and layout of its
/// [`FftPlanSpec`]. Cloning a plan is cheap: clones share the prepared tables.
///
/// Lengths with no prime factor above 7 use mixed-radix Cooley–Tukey stages
/// of radix 2, 3, 4, 5 and 7. A prime length whose `n − 1` has no prime factor
/// above 7 uses Rader's algorithm when its cost model favors it. Every other
/// length uses Bluestein's algorithm, a convolution of 2/3/5/7-smooth length at
/// least `2n − 1`. Every length costs `O(n log n)` operations, and
/// [`FftPlan::diagnostics`] reports the selected algorithm. Packed layouts use
/// a complex FFT of length `n/2` for even `n`.
///
/// # Accuracy
///
/// Forward complex and packed-real transforms are tested against
/// high-precision reference DFTs with a relative error in the 2-norm of at
/// most `2 ε max(1, log₂ n)`, where `ε` is the machine epsilon of `S`. The
/// tested lengths are 2/3/5/7-smooth lengths up to 6720, Rader primes up to
/// 65537 and Bluestein primes up to 10007, for `f32` and `f64` and every CPU
/// backend. This is a normwise bound: bins much smaller than the largest bin
/// can have a larger relative error. It does not cover external providers.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext};
/// use scioxide_fft::{FftDirection, FftLayout, FftPlan, FftPlanSpec, Normalization};
/// let direction = FftDirection::Forward;
/// let spec = FftPlanSpec::new(4, direction, Normalization::Inverse, FftLayout::Complex);
/// let plan = FftPlan::<f64>::new(spec)?;
/// let mut workspace = plan.workspace(())?;
/// let mut output = [Complex::new(0.0, 0.0); 4];
/// for amplitude in [1.0, 2.0] {
///     let input = [Complex::new(amplitude, 0.0); 4];
///     let cx = ExecutionContext::sequential();
///     plan.execute_into(&input, &mut output, &mut workspace, cx)?;
///     // The DFT of a constant c is n·c at frequency 0 and zero elsewhere.
///     assert!((output[0] - Complex::new(4.0 * amplitude, 0.0)).norm() < 1e-12);
///     assert!(output[1..].iter().all(|z| z.norm() < 1e-12));
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of
///   complex Fourier series", *Math. Comp.* 19(90), 1965.
/// - C. M. Rader, "Discrete Fourier transforms when the number of data samples
///   is prime", *Proc. IEEE* 56(6), 1968.
/// - L. I. Bluestein, "A linear filtering approach to the computation of
///   discrete Fourier transform", *IEEE Trans. Audio Electroacoust.* 18(4), 1970.
#[derive(Debug, Clone)]
#[must_use]
pub struct FftPlan<S: scioxide_core::RealScalar> {
    spec: FftPlanSpec,
    scale: S,
    pub(crate) core: Arc<PlanCore<S>>,
}

impl<S: scioxide_core::RealScalar> FftPlan<S> {
    pub(crate) fn shares_core(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.core, &other.core)
    }

    /// Prepares a plan for the processor of
    /// [`ExecutionContext::sequential_native`].
    ///
    /// Use [`FftPlan::build`] to pass another planning context or an
    /// [`FftProvider`](crate::FftProvider).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the length is zero, or if a required backend is
    ///   unavailable (see [`FftBackendStrategy::Required`]).
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the tables or
    ///   workspace size cannot be represented or reserved.
    pub fn new(spec: FftPlanSpec) -> Result<Self, SciOxideError> {
        Self::build(spec, scioxide_core::ExecutionContext::sequential_native())
    }

    /// Prepares a plan with the processor features and provider of `context`.
    ///
    /// The plan does not borrow `context` after it returns. A plain
    /// [`ExecutionContext`] selects CPU kernels from its processor features;
    /// [`ExecutionContext::sequential`] reports none, so it selects the scalar
    /// kernels. See [Backends and providers](crate#backends-and-providers).
    ///
    /// # Errors
    ///
    /// - The errors of [`FftPlan::new`].
    /// - `ExecutionStopped` if the context is already cancelled or past its
    ///   deadline.
    /// - The errors of the provider, if it fails to prepare a kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, SciOxideErrorKind, backend::KernelBackend};
    /// use scioxide_fft::{FftBackend, FftBackendStrategy, FftDirection, FftLayout};
    /// use scioxide_fft::{FftPlan, FftPlanSpec, Normalization};
    /// let (direction, norm) = (FftDirection::Inverse, Normalization::Inverse);
    /// let spec = |backend| {
    ///     FftPlanSpec::new(32, direction, norm, FftLayout::Complex)
    ///         .with_backend_strategy(FftBackendStrategy::Required(backend))
    /// };
    /// let cx = ExecutionContext::sequential_native();
    /// let plan = FftPlan::<f64>::build(spec(KernelBackend::Scalar), cx)?;
    /// let scalar = FftBackend::Cpu(KernelBackend::Scalar);
    /// assert_eq!(plan.resolved_backends().next(), Some(scalar));
    /// // FFT has no AVX-512 kernels, so requiring that backend always fails.
    /// let error = FftPlan::<f64>::build(spec(KernelBackend::Avx512), cx).unwrap_err();
    /// assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn build<'a>(
        spec: FftPlanSpec,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
    ) -> Result<Self, SciOxideError> {
        let context = context.into();
        context.execution().check()?;
        let backend = context.resolve(spec.backend_strategy())?;
        Self::prepare(spec, backend, context)
    }

    pub(crate) fn prepare(
        spec: FftPlanSpec,
        backend: KernelBackend,
        context: crate::FftPlanningContext<'_, S>,
    ) -> Result<Self, SciOxideError> {
        let length = scioxide_core::argument!("spec.length");
        let provider = scioxide_core::argument!("context");
        let core = match spec.layout {
            FftLayout::Complex => CoreKind::Complex(ComplexCore::with_backend_and_context(
                spec.length,
                spec.execution_policy,
                backend,
                context,
                length,
                provider,
            )?),
            FftLayout::Real | FftLayout::Hermitian => {
                CoreKind::Real(RealFftPlan::with_backend_and_context(
                    spec.length,
                    spec.execution_policy,
                    backend,
                    context,
                    length,
                    provider,
                )?)
            }
        };
        let core = PlanCore::new(core)?;
        let plan = Self::from_core(spec, Self::scale(spec)?, Arc::new(core));
        let _ = plan.workspace_requirements(())?;
        Ok(plan)
    }

    pub(crate) fn scale(spec: FftPlanSpec) -> Result<S, SciOxideError> {
        crate::validation::nonzero(spec.length, scioxide_core::argument!("spec.length"))?;
        let (forward, inverse) = spec.normalization.fourier_scales(spec.length)?;
        Ok(match spec.direction {
            FftDirection::Forward => forward,
            FftDirection::Inverse => inverse,
        })
    }

    pub(crate) fn from_core(spec: FftPlanSpec, scale: S, core: Arc<PlanCore<S>>) -> Self {
        Self { spec, scale, core }
    }

    /// Returns the spec the plan was prepared from.
    pub const fn spec(&self) -> &FftPlanSpec {
        &self.spec
    }

    /// Returns the backend of each complex FFT the plan runs.
    ///
    /// A packed plan of odd composite length can run several complex FFTs, so
    /// the iterator can yield more than one item, and items can repeat.
    pub fn resolved_backends(&self) -> impl Iterator<Item = crate::FftBackend> + '_ {
        let (complex, real) = match &self.core.kind {
            CoreKind::Complex(plan) => (Some(plan.resolved_backend()), None),
            CoreKind::Real(plan) => (None, Some(plan)),
        };
        complex
            .into_iter()
            .chain(real.into_iter().flat_map(|p| p.resolved_backends()))
    }

    /// Returns the scratch length, in complex elements, that execution needs.
    ///
    /// Let `A(m)` be the scratch of a complex FFT of length `m`, including any
    /// scratch a provider declares: `m` for 2/3/5/7-smooth `m > 1`, `2(m − 1)`
    /// for Rader and twice the convolution length for Bluestein. A complex plan
    /// needs `n + A(n)` elements for `n > 1` and none for `n = 1`. For packed
    /// layouts let `R(n)` be the requirement:
    ///
    /// ```text
    /// R(n) = n/2 + 1 + A(n/2)                    n even
    /// R(n) = (n + 1)/2 + (p − 1)m/2 + R(m) + A(m)  n = p m odd, p ∈ {3, 5, 7}, m > 1
    /// R(n) = n + A(n)                            other odd n > 1
    /// R(1) = 1
    /// ```
    ///
    /// where `p` is the smallest of 3, 5, 7 that divides `n`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    /// - `ResourceLimit` if it exceeds a provider's workspace limit.
    pub fn workspace_requirements(
        &self,
        (): (),
    ) -> Result<FftWorkspaceRequirements, SciOxideError> {
        let (len, limit) = match &self.core.kind {
            CoreKind::Complex(p) => (p.scratch_len(), p.workspace_byte_limit()),
            CoreKind::Real(p) => (p.scratch_len(), p.workspace_byte_limit()),
        };
        FftWorkspaceRequirements::new::<S>(len)?.with_limit(limit)
    }

    /// Returns the bytes this plan occupies, including its shared tables.
    ///
    /// Clones report the same tables in full; an [`FftPlanCache`](crate::FftPlanCache)
    /// counts shared tables once.
    pub fn plan_bytes(&self) -> usize {
        size_of::<Self>() + self.core.bytes()
    }

    /// Returns the selected algorithm and its number of stages.
    pub fn diagnostics(&self) -> FftPlanDiagnostics {
        match &self.core.kind {
            CoreKind::Complex(p) => p.diagnostics(),
            CoreKind::Real(p) => p.diagnostics(),
        }
    }

    /// Allocates a workspace of the length reported by
    /// [`FftPlan::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// - The errors of [`FftPlan::workspace_requirements`].
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, request: ()) -> Result<crate::FftWorkspace<S>, SciOxideError> {
        crate::workspace::allocate(&self.workspace_requirements(request)?)
    }

    /// Transforms `input` and writes the result to `output`.
    ///
    /// Both arguments take anything that converts into [`FftInput`] and
    /// [`FftOutput`]: real or complex slices, arrays, vectors and views, and
    /// [`HermitianSpectrum`](crate::HermitianSpectrum) references for packed
    /// bins. The variants must match the layout and direction:
    ///
    /// | Layout | Forward | Inverse |
    /// |---|---|---|
    /// | [`FftLayout::Complex`] | complex → complex | complex → complex |
    /// | [`FftLayout::Real`] | real → Hermitian | Hermitian → real |
    /// | [`FftLayout::Hermitian`] | Hermitian → real | real → Hermitian |
    ///
    /// Packed Hermitian input is checked as described in
    /// [Hermitian input](crate#hermitian-input). `scratch` must hold at least
    /// the length from [`FftPlan::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the input or output
    ///   variant does not match the layout and direction.
    /// - `InvalidInput` if packed input is not Hermitian within the tolerance.
    /// - `DimensionMismatch` if the input, output or logical length differs from
    ///   the plan length, or `scratch` is too short.
    /// - The shared execution errors listed under [Errors](crate#errors).
    pub fn execute_into<'i, 'o>(
        &self,
        input: impl Into<FftInput<'i, S>>,
        output: impl Into<FftOutput<'o, S>>,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        S: scioxide_core::NumericScalar<Real = S>,
        Complex<S>: scioxide_core::NumericScalar<Real = S>,
    {
        let (input, output) = (input.into(), output.into());
        // Only the (layout, direction, input, output) combinations with a
        // defined meaning dispatch; any other variant pairing is a caller
        // layout error rather than an implicit conversion. Packed spectra also
        // carry their logical length, which must match the plan because
        // n = 2m and n = 2m + 1 share the same bin count.
        //
        // The Hermitian layout is the conjugate dual of the real layout:
        //   hfft(x)  = irfft_unscaled(conj(x)) * scale
        //   ihfft(y) = conj(rfft_unscaled(y) * scale)
        // so it runs the opposite real kernel with a conjugated packed side.
        let hermitian = self.spec.layout == FftLayout::Hermitian;
        let samples_to_bins = hermitian == (self.spec.direction == FftDirection::Inverse);
        match (&self.core.kind, self.spec.direction, input, output) {
            (
                CoreKind::Complex(plan),
                FftDirection::Forward,
                FftInput::Complex(input),
                FftOutput::Complex(output),
            ) => plan.transform_into_view(input, output, scratch, false, self.scale, context),
            (
                CoreKind::Complex(plan),
                FftDirection::Inverse,
                FftInput::Complex(input),
                FftOutput::Complex(output),
            ) => plan.transform_into_view(input, output, scratch, true, self.scale, context),
            (
                CoreKind::Real(plan),
                _,
                FftInput::Real(input),
                FftOutput::Hermitian { logical_len, bins },
            ) if samples_to_bins => {
                crate::validation::length(logical_len, self.spec.length)?;
                plan.forward_into(self.scale, input, bins, hermitian, scratch, context)
            }
            (
                CoreKind::Real(plan),
                _,
                FftInput::Hermitian { logical_len, bins },
                FftOutput::Real(output),
            ) if !samples_to_bins => {
                crate::validation::length(logical_len, self.spec.length)?;
                plan.inverse_into(self.scale, bins, output, hermitian, scratch, context)
            }
            _ => Err(layout_error(scioxide_core::argument!("input"))),
        }
    }

    /// Algorithm scratch of an N-D lane child; the parent holds the output candidates.
    pub(crate) fn lane_workspace_requirements(
        &self,
    ) -> Result<crate::FftWorkspaceRequirements, SciOxideError> {
        let CoreKind::Complex(plan) = &self.core.kind else {
            return Err(layout_error(scioxide_core::argument!("self")));
        };
        crate::FftWorkspaceRequirements::new::<S>(plan.lane_scratch_len())?
            .with_limit(plan.bounded_workspace_byte_limit())
    }

    pub(crate) fn execute_owned(
        &self,
        input: &[Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<alloc::vec::Vec<Complex<S>>, SciOxideError> {
        let CoreKind::Complex(plan) = &self.core.kind else {
            return Err(layout_error(scioxide_core::argument!("self")));
        };
        plan.transform_owned(
            input,
            self.spec.direction == FftDirection::Inverse,
            self.scale,
            context,
        )
    }

    /// Transform the lanes of one N-D axis for the parent plan;
    /// see [`ComplexCore::transform_lanes`].
    pub(crate) fn transform_lanes(
        &self,
        lanes: crate::lanes::Lanes<'_, Complex<S>>,
        scratch: &mut [Complex<S>],
        mode: crate::lanes::LaneMode,
        scale: S,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let CoreKind::Complex(plan) = &self.core.kind else {
            return Err(layout_error(scioxide_core::argument!("self")));
        };
        plan.transform_lanes(
            lanes,
            scratch,
            self.spec.direction == FftDirection::Inverse,
            mode,
            scale,
            context,
        )
    }

    /// Transforms `data` in place with a complex plan.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the plan layout is not
    ///   [`FftLayout::Complex`].
    /// - `DimensionMismatch` if `data` does not have the plan length or `scratch`
    ///   is too short.
    /// - The shared execution errors listed under [Errors](crate#errors).
    pub fn execute_in_place(
        &self,
        data: &mut [Complex<S>],
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let CoreKind::Complex(plan) = &self.core.kind else {
            return Err(layout_error(scioxide_core::argument!("self")));
        };
        plan.transform(
            data,
            scratch,
            self.spec.direction == FftDirection::Inverse,
            self.scale,
            context,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FftCachePolicy, FftPlanCache, HermitianSpectrum};

    #[test]
    fn cached_variants_share_the_same_allocation() {
        for (layout, variant) in [
            (FftLayout::Complex, FftLayout::Complex),
            (FftLayout::Real, FftLayout::Real),
            (FftLayout::Real, FftLayout::Hermitian),
        ] {
            let mut cache = FftPlanCache::<f32>::new(FftCachePolicy::default()).unwrap();
            let cx = ExecutionContext::sequential();
            let first = cache
                .plan(
                    FftPlanSpec::new(23, FftDirection::Forward, Normalization::Inverse, layout),
                    cx,
                )
                .unwrap();
            {
                let other = cache
                    .plan(
                        FftPlanSpec::new(
                            23,
                            FftDirection::Inverse,
                            Normalization::Orthonormal,
                            variant,
                        ),
                        cx,
                    )
                    .unwrap();
                assert!(Arc::ptr_eq(&first.core, &other.core));
            };
        }
    }

    #[test]
    fn cache_reuses_and_evicts_least_recently_used_plans() {
        let policy = FftCachePolicy::new(2, usize::MAX).unwrap();
        let mut cache = FftPlanCache::<f64>::new(policy).unwrap();
        {
            let spec = |length| {
                FftPlanSpec::new(
                    length,
                    FftDirection::Forward,
                    Normalization::Inverse,
                    FftLayout::Complex,
                )
            };

            let first = cache
                .plan(spec(8), crate::FftPlanningContext::default())
                .unwrap();
            let reused = cache
                .plan(spec(8), crate::FftPlanningContext::default())
                .unwrap();
            assert!(Arc::ptr_eq(&first.core, &reused.core));
            let _ = cache
                .plan(spec(9), crate::FftPlanningContext::default())
                .unwrap();
            let _ = cache
                .plan(spec(10), crate::FftPlanningContext::default())
                .unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.hits(), 1);
        assert_eq!(stats.misses(), 3);
        assert_eq!(stats.evictions(), 1);
        assert_eq!(stats.entries(), 2);
        assert!(stats.retained_plan_bytes() <= policy.max_plan_bytes());
    }

    #[test]
    fn complex_directional_plans_round_trip() {
        let input = [
            Complex::new(1.0, -0.5),
            Complex::new(2.0, 0.25),
            Complex::new(-1.0, 0.75),
            Complex::new(0.5, -1.25),
        ];
        let spec = |direction| {
            FftPlanSpec::new(
                input.len(),
                direction,
                Normalization::Inverse,
                FftLayout::Complex,
            )
        };
        let forward = FftPlan::<f64>::build(
            spec(FftDirection::Forward),
            crate::FftPlanningContext::default(),
        )
        .unwrap();
        let inverse = FftPlan::<f64>::build(
            spec(FftDirection::Inverse),
            crate::FftPlanningContext::default(),
        )
        .unwrap();
        let mut spectrum = [Complex::new(0.0, 0.0); 4];
        let mut output = [Complex::new(0.0, 0.0); 4];
        let mut forward_scratch = forward.workspace(()).unwrap();
        let mut inverse_scratch = inverse.workspace(()).unwrap();
        forward
            .execute_into(
                &input,
                &mut spectrum,
                &mut forward_scratch,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        inverse
            .execute_into(
                &spectrum,
                &mut output,
                &mut inverse_scratch,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        input.iter().zip(output).for_each(|(expected, actual)| {
            assert!((expected.re - actual.re).abs() < 1e-12);
            assert!((expected.im - actual.im).abs() < 1e-12);
        });
    }

    #[test]
    fn packed_real_plan_round_trip_preserves_logical_shape() {
        let input = [1.0_f64, -2.0, 0.5, 4.0, -1.5];
        let spec = |direction| {
            FftPlanSpec::new(
                input.len(),
                direction,
                Normalization::Inverse,
                FftLayout::Real,
            )
        };
        let forward = FftPlan::build(
            spec(FftDirection::Forward),
            crate::FftPlanningContext::default(),
        )
        .unwrap();
        let inverse = FftPlan::build(
            spec(FftDirection::Inverse),
            crate::FftPlanningContext::default(),
        )
        .unwrap();
        let mut spectrum = HermitianSpectrum::zeros(input.len()).unwrap();
        let mut output = [0.0; 5];
        forward
            .execute_into(
                &input,
                &mut spectrum,
                &mut forward.workspace(()).unwrap(),
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        inverse
            .execute_into(
                &spectrum,
                &mut output,
                &mut inverse.workspace(()).unwrap(),
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        input
            .iter()
            .zip(output)
            .for_each(|(expected, actual)| assert!((expected - actual).abs() < 1e-12));
    }

    #[test]
    fn layout_failure_leaves_output_unchanged() {
        let plan = FftPlan::<f64>::build(
            FftPlanSpec::new(
                2,
                FftDirection::Forward,
                Normalization::Inverse,
                FftLayout::Complex,
            ),
            crate::FftPlanningContext::default(),
        )
        .unwrap();
        let mut output = [Complex::new(7.0, 8.0); 2];
        let original = output;
        let error = plan
            .execute_into(
                &[1.0_f64, 2.0],
                &mut output,
                &mut plan.workspace(()).unwrap(),
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        assert_eq!(output, original);
    }
}
