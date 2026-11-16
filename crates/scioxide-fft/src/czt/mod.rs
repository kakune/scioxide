//! Chirp z-transform plans.
//!
//! [`CztContour`] and [`CztPlanSpec`] describe the transform. [`CztPlan`]
//! evaluates it by Bluestein convolution (`convolution`) or direct summation
//! (`direct`).
mod contour;
mod convolution;
mod direct;
mod spec;

pub use contour::CztContour;
pub use spec::{CztLimits, CztPlanSpec};

use crate::FftWorkspaceRequirements;
use crate::validation::{invalid, length};
use alloc::vec::Vec;
use convolution::Convolution;
use direct::Direct;
use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{Complex, ExecutionContext, RealScalar, SciOxideError};

/// Algorithm that a [`CztPlan`] selected during preparation.
///
/// The selection rule is stated on [`CztPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CztAlgorithm {
    /// Bluestein's algorithm: a circular convolution evaluated with FFTs of
    /// length `L = next_fast_len(n + m − 1)`, costing two length-`L` FFTs plus
    /// `O(n + m)` operations.
    Bluestein,
    /// Direct summation with an extended exponent range, costing `O(n·m)`.
    ScaledDirect,
}

/// Evaluates the chirp z-transform of a sequence on a geometric contour.
///
/// For an input `x` of length `n` and the `m` contour points `z_k` of a
/// [`CztContour`], the plan computes
///
/// ```text
/// y[k] = s · Σⱼ x[j] z_k^(−j),   j = 0, …, n − 1,   k = 0, …, m − 1,
/// z_k  = exp(ρ₀ + k ρ + 2πi (t₀ + k t)),
/// ```
///
/// where `ρ₀` and `ρ` are the initial log-radius and its step, `t₀` and `t`
/// the initial phase and its step in turns, and `s` the scale. `s` is `1/n`
/// for [`Normalization::Forward`](crate::Normalization::Forward), `1/√n` for
/// [`Normalization::Orthonormal`](crate::Normalization::Orthonormal) and one
/// otherwise. On the contour [`CztContour::dft`]`(n)` with `m = n` this is the
/// forward DFT; [`CztContour::zoom`] samples a frequency band.
///
/// # Method
///
/// Preparation selects the algorithm, reported by [`Self::algorithm`]:
///
/// - Contours on the unit circle (`ρ₀ = ρ = 0`) use Bluestein's algorithm.
/// - Other contours with `n·m ≤ 4096` use direct summation.
/// - Larger contours use Bluestein's algorithm when
///   `χ = |ρ| · max(n, m)² ≤ 0.5`, and direct summation otherwise.
///
/// Bluestein's algorithm writes `jk = (j² + k² − (k − j)²)/2` and turns the sum
/// into a convolution with a chirp, evaluated with two FFTs of length
/// `L = next_fast_len(n + m − 1)` plus `O(n + m)` operations. The radial factor
/// `exp(−j ρ₀)` is applied to the input with a separate binary exponent, so the
/// contour radii need not be representable in `S`. Direct summation costs
/// `O(n·m)` per execution; plans with `n·m ≤ 4096` store the coefficients.
/// Each output row uses compensated sums in working precision when its error
/// bound allows and extended-exponent arithmetic otherwise, in a fixed order
/// that does not depend on the number of threads. Direct summation checks for
/// cancellation and deadlines at least every 256 terms.
///
/// # Accuracy
///
/// For Bluestein's algorithm with the native FFT kernels, each output
/// component satisfies
///
/// ```text
/// |y[k] − ŷ[k]| ≤ |s| e^(2χ) (128 ε h B + 32 ε₆₄ (D + χ B)) + U,
/// h = max(1, log₂ L),
/// B = Σⱼ (|Re x[j]| + |Im x[j]|) e^(−j ρ₀),
/// D = Σⱼ (|Re x[j]| + |Im x[j]|) e^(−j ρ₀) (1 + |j ρ₀|),
/// ```
///
/// where `ŷ` is the computed result, `ε` the machine epsilon of `S` and `ε₆₄`
/// that of `f64`. The term `U = 128 η 2^E L² h + 2η` covers gradual underflow;
/// `η` is the smallest positive subnormal of `S` and `2^E` the power of two by
/// which the algorithm rescales the output. The bound is absolute, so
/// outputs much smaller than the weighted input mass `B` can have a large
/// relative error. An injected FFT provider has its own accuracy.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext};
/// use scioxide_fft::{CztAlgorithm, CztContour, CztPlan, CztPlanSpec, Normalization};
///
/// // On the contour z_k = exp(2πik/4) the CZT is the DFT, so a constant
/// // input has all of its energy in bin 0.
/// let spec = CztPlanSpec::new(4, 4, CztContour::dft(4)?, Normalization::Unscaled);
/// let plan = CztPlan::<f64>::new(spec)?;
/// assert_eq!(plan.algorithm(), CztAlgorithm::Bluestein);
/// let (input, mut output) = ([Complex::new(1.0, 0.0); 4], [Complex::new(0.0, 0.0); 4]);
/// let (mut workspace, cx) = (plan.workspace(())?, ExecutionContext::sequential_native());
/// plan.execute_into(&input[..], &mut output[..], &mut workspace, cx)?;
/// assert!((output[0] - Complex::new(4.0, 0.0)).norm() < 1e-12);
/// assert!(output[1..].iter().all(|z| z.norm() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - L. R. Rabiner, R. W. Schafer and C. M. Rader, "The chirp z-transform
///   algorithm", *IEEE Trans. Audio Electroacoust.* 17(2), 86–92, 1969.
/// - L. I. Bluestein, "A linear filtering approach to the computation of
///   discrete Fourier transform", *IEEE Trans. Audio Electroacoust.* 18(4),
///   451–455, 1970.
#[derive(Debug, Clone)]
pub struct CztPlan<S: RealScalar> {
    plan_bytes: usize,
    spec: CztPlanSpec<S>,
    preparation: Preparation<S>,
    scratch: FftWorkspaceRequirements,
    scale: S,
}
#[derive(Debug, Clone)]
#[cfg_attr(
    target_pointer_width = "64",
    expect(
        clippy::large_enum_variant,
        reason = "inline storage avoids another allocation and is included in preparation limits"
    )
)]
enum Preparation<S: RealScalar> {
    Convolution(Convolution<S>),
    Direct(Direct<S>),
}
impl<S: RealScalar> CztPlan<S> {
    /// Returns the backend of the FFT used by Bluestein's algorithm.
    ///
    /// Direct summation uses no FFT, so the iterator is then empty.
    pub fn resolved_backends(&self) -> impl Iterator<Item = crate::FftBackend> + '_ {
        let backend = match &self.preparation {
            Preparation::Convolution(c) => Some(c.fft.resolved_backend()),
            Preparation::Direct(_) => None,
        };
        backend.into_iter()
    }
    /// Prepares a plan with the default processor capabilities.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `spec.n` or `spec.m` is zero.
    /// - `SizeOverflow` if `n + m − 1` or a storage size overflows `usize`.
    /// - `ResourceLimit` if preparation or the workspace exceeds a byte limit in
    ///   [`CztLimits`].
    /// - `NumericalInstability` if a radial factor lies outside the extended
    ///   exponent range.
    /// - `AllocationFailure` if storage cannot be reserved.
    /// - Errors from preparing the FFT, as for [`FftPlan::build`](crate::FftPlan::build).
    pub fn new(spec: CztPlanSpec<S>) -> Result<Self, SciOxideError> {
        Self::build(spec, scioxide_core::ExecutionContext::sequential_native())
    }
    /// Prepares a plan with the processor capabilities and FFT provider of
    /// `context`.
    ///
    /// The plan does not borrow `context`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], plus:
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the spec requires a
    ///   backend that `context` does not verify.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn build<'a>(
        spec: CztPlanSpec<S>,
        context: impl Into<crate::FftPlanningContext<'a, S>>,
    ) -> Result<Self, SciOxideError> {
        let (n, m) = (spec.n, spec.m);
        if n == 0 || m == 0 {
            return Err(invalid(
                scioxide_core::InvalidInputReason::Empty,
                if n == 0 {
                    scioxide_core::argument!("spec.n")
                } else {
                    scioxide_core::argument!("spec.m")
                },
                "input and output lengths must be positive",
            ));
        }
        checked_add(n, m - 1, scioxide_core::quantity!("czt_convolution_length"))?;

        let context = context.into();
        context.execution().check()?;
        crate::memory::admit_bytes(
            size_of::<Self>(),
            spec.limits.preparation_bytes,
            scioxide_core::quantity!("czt_preparation_bytes"),
        )?;
        let backend = context.resolve(spec.backend)?;
        let ar = spec.contour.log_radius().try_to_f64_exact()?;
        let dr = spec.contour.log_radius_step().try_to_f64_exact()?;
        let extent = n.max(m) as f64;
        // Radius is an input weighting, not a convolution condition number.
        // Only the quadratic spiral chirps amplify convolution cancellation.
        // Small off-circle sums retain the componentwise direct contract.
        let small = n.checked_mul(m).is_some_and(|work| work <= 4096);
        let off_circle = ar != 0.0 || dr != 0.0;
        let scale = spec.normalization.fourier_scales(n)?.0;
        let preparation = if (!off_circle || !small) && dr.abs() * extent * extent <= 0.5 {
            Preparation::Convolution(Convolution::prepare(&spec, backend, context, scale)?)
        } else {
            Preparation::Direct(Direct::prepare(&spec, small, context.execution())?)
        };
        let scratch = preparation.requirements(&spec)?;
        let mut plan = Self {
            plan_bytes: 0,
            spec,
            preparation,
            scratch,
            scale,
        };
        plan.plan_bytes = plan.retained_bytes()?;
        Ok(plan)
    }
    /// Returns the specification the plan was prepared from.
    pub const fn spec(&self) -> &CztPlanSpec<S> {
        &self.spec
    }
    /// Returns the algorithm selected during preparation.
    pub fn algorithm(&self) -> CztAlgorithm {
        if matches!(self.preparation, Preparation::Convolution(_)) {
            CztAlgorithm::Bluestein
        } else {
            CztAlgorithm::ScaledDirect
        }
    }
    /// Returns the workspace size for [`Self::execute_into`] (request `()`) or
    /// for [`Self::execute_axis_into`] (an [`AxisWorkspaceRequest`](crate::AxisWorkspaceRequest)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` or `InvalidInput` if an axis request does not fit
    ///   the plan.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if the size exceeds `limits.workspace_bytes`.
    pub fn workspace_requirements<'r>(
        &self,
        request: impl Into<crate::AxisWorkspaceRequest<'r>>,
    ) -> Result<FftWorkspaceRequirements, SciOxideError> {
        crate::axis::requirements(self, request.into())
    }

    /// Exact number of complex scratch elements.
    const fn scratch_len(&self) -> usize {
        self.scratch.complex_len()
    }
    /// Allocates a workspace for the given request.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<'r>(
        &self,
        request: impl Into<crate::AxisWorkspaceRequest<'r>>,
    ) -> Result<crate::FftWorkspace<S>, SciOxideError> {
        crate::workspace::allocate(&self.workspace_requirements(request.into())?)
    }

    // Owned output for the allocating conveniences; prepared plans publish
    // only caller-storage forms.
    pub(crate) fn execute_owned(
        &self,
        input: crate::FftInput<'_, S>,
    ) -> Result<Vec<Complex<S>>, SciOxideError>
    where
        Complex<S>: scioxide_core::NumericScalar,
    {
        let mut output = try_filled(
            self.spec.m,
            <scioxide_core::Complex<_> as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("czt_output"),
        )?;
        self.execute_into(
            input,
            &mut output,
            &mut self.workspace(())?,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Evaluate the prepared transform into the scratch candidate prefix,
    /// reporting every failure.
    pub(crate) fn evaluate(
        &self,
        input: impl Fn(usize) -> Result<Complex<S>, SciOxideError>,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::validation::sufficient(scratch.len(), self.scratch_len())?;
        context.check()?;
        // Validate and measure the whole input before any arithmetic; the
        // result is built in the `candidate` prefix of scratch, which callers
        // copy out only after success.
        let mut magnitude = S::ZERO;
        for j in 0..self.spec.n {
            let x = input(j)?;
            crate::validation::check_complex_slice_finite(
                &[x],
                scioxide_core::NonFiniteOrigin::Input,
            )?;
            magnitude = magnitude.max(x.re.abs()).max(x.im.abs());
        }
        context.check()?;
        let (candidate, work) = scratch.split_at_mut(self.spec.m);
        if magnitude.is_zero() {
            candidate.fill(<scioxide_core::Complex<_> as scioxide_core::AlgebraicScalar>::ZERO);
            return Ok(());
        }
        match &self.preparation {
            Preparation::Convolution(c) => {
                c.evaluate(&input, magnitude, candidate, work, context)?
            }
            Preparation::Direct(d) => {
                self.evaluate_direct(d, &input, magnitude, candidate, work, context)?
            }
        }
        context.check()
    }
    /// Evaluates the transform of one real or complex vector into `output`.
    ///
    /// The input may be strided or conjugated. `scratch` needs the length given
    /// by [`Self::workspace_requirements`]`(())`; a plan workspace dereferences
    /// to such a slice.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the input length is not `n`, the output length
    ///   is not `m`, or `scratch` is too short.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if `output` is not complex.
    /// - `NonFinite` if the input contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if a result is not representable.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn execute_into<'i, 'o>(
        &self,
        input: impl Into<crate::FftInput<'i, S>>,
        output: impl Into<crate::FftOutput<'o, S>>,
        scratch: &mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>
    where
        Complex<S>: scioxide_core::NumericScalar,
    {
        let input = crate::io::FullInput::try_from(input.into())?;
        let input_len = input.len();
        let crate::FftOutput::Complex(mut output) = output.into() else {
            return Err(invalid(
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                scioxide_core::argument!("output"),
                "CZT output must be complex",
            ));
        };
        length(input_len, self.spec.n)?;
        length(output.len(), self.spec.m)?;
        match input {
            crate::io::FullInput::Complex(view) => {
                self.evaluate(|j| view.get(j), scratch, context)?
            }
            crate::io::FullInput::Real(view) => self.evaluate(
                |j| Ok(Complex::new(view.get(j)?, S::ZERO)),
                scratch,
                context,
            )?,
        }
        for (k, value) in scratch[..self.spec.m].iter().enumerate() {
            output.set(k, *value)?;
        }
        Ok(())
    }
    /// Transforms every lane of `input` along `axis` into `output`.
    ///
    /// The other axes are batch axes. The input may broadcast along them; the
    /// output must not alias itself. Either every lane is written or, on error,
    /// none is. The CZT counts no iterations or evaluations; `scope` supplies
    /// the execution context and may nest the call in a parent solve.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `axis` is not an axis of `input`.
    /// - `DimensionMismatch` if the extent along `axis` is not `n` for the
    ///   input or `m` for the output, or the batch shapes differ.
    /// - `DimensionMismatch` if `workspace` is too small for this request.
    /// - `NonFinite` if the input contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if a result is not representable.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn execute_axis_into<'scope, 'cap, T: scioxide_core::NumericScalar<Real = S>>(
        &self,
        input: scioxide_core::NdView<'_, '_, T>,
        axis: usize,
        output: scioxide_core::NdViewMut<'_, '_, Complex<S>>,
        workspace: &mut crate::FftWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'cap>>,
    ) -> Result<(), SciOxideError> {
        crate::axis::execute(self, input, axis, output, workspace, scope.into())
    }
    /// Returns the bytes the plan occupies, including its FFT tables.
    pub const fn plan_bytes(&self) -> usize {
        self.plan_bytes
    }
    fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        let bytes = size_of::<Self>();
        match &self.preparation {
            Preparation::Convolution(c) => c.retained_bytes(bytes),
            Preparation::Direct(d) => d.retained_bytes(bytes),
        }
    }
}

impl<S: RealScalar> Preparation<S> {
    // Plain execution storage: the candidate output plus algorithm work.
    // Child FFT scratch keeps its own workspace limit; CZT limits cap the total.
    fn requirements(
        &self,
        spec: &CztPlanSpec<S>,
    ) -> Result<FftWorkspaceRequirements, SciOxideError> {
        let work = match self {
            Self::Convolution(c) => c.work_len()?,
            Self::Direct(d) => d.work_len(spec.n)?,
        };
        let scratch = FftWorkspaceRequirements::new::<S>(checked_add(
            spec.m,
            work,
            scioxide_core::quantity!("czt_scratch"),
        )?)?;
        let child = match self {
            Self::Convolution(c) => Some((
                c.fft.bounded_workspace_byte_limit(),
                c.fft.bounded_scratch_len() * size_of::<Complex<S>>(),
            )),
            Self::Direct(_) => None,
        };
        scratch.with_limit(crate::workspace::nested_byte_limit(
            spec.limits.workspace_bytes,
            scratch.bytes(),
            child,
        ))
    }
}

impl<S: RealScalar> crate::axis::AxisPlan<S> for CztPlan<S> {
    fn lane_lens(&self) -> (usize, usize) {
        (self.spec.n, self.spec.m)
    }
    fn scratch(&self) -> Result<FftWorkspaceRequirements, SciOxideError> {
        Ok(self.scratch)
    }
    fn workspace_bytes(&self) -> usize {
        self.spec.limits.workspace_bytes
    }
    fn lane<'w>(
        &self,
        source: &[Complex<S>],
        scratch: &'w mut [Complex<S>],
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<&'w [Complex<S>], SciOxideError> {
        self.evaluate(|j| Ok(source[j]), scratch, context.execution())?;
        Ok(&scratch[..self.spec.m])
    }
}
