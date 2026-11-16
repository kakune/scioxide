//! Minimum-phase conversion of FIR filters from the log-magnitude spectrum.
use crate::util::{
    empty, exact_len, finite, inconsistent, invalid, scalar_from_usize_rounded as scalar,
    unsupported,
};
use crate::{SignalRequirements, SignalWorkspace};
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar, ResidualTolerance,
    SciOxideError,
};
use scioxide_fft::{FftDirection, FftLayout, FftPlan, FftPlanSpec, Normalization};

/// Algorithm of a minimum-phase conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimumPhaseMethod {
    /// Homomorphic (cepstral) method: folds the real cepstrum of the target
    /// magnitude onto nonnegative quefrencies. It accepts real and complex
    /// filters of any length.
    Homomorphic,
    /// Hilbert method for real Type I linear-phase filters, such as equiripple
    /// designs.
    ///
    /// With the zero-phase amplitude `A`, its extremes `A_min` and `A_max`, the
    /// lift `δ = max(0, −A_min)` and the normalization
    /// `ν = (2 / (√(A_max + δ) + √(2 − A_max + δ)))²`, the target magnitude is
    /// `√(ν (A + δ))`. The minimum phase is recovered with a discrete Hilbert
    /// transform of the log magnitude. It requires the half-length convention
    /// and an odd number of at least three taps.
    Hilbert,
}
/// Length and magnitude of the minimum-phase result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimumPhaseConvention {
    /// Returns `N` taps whose magnitude response approximates `|H|`.
    SameLength,
    /// Returns `⌈N/2⌉` taps whose magnitude response approximates `√|H|`. For
    /// a linear-phase filter this is the minimum-phase spectral factor.
    HalfLength,
}
/// Treatment of zero and near-zero magnitudes on the FFT grid.
///
/// The logarithm of the magnitude is taken without any added offset, so a
/// sampled zero has to be either rejected or raised to a floor. The threshold
/// or floor must lie in `(0, 1)`; otherwise preparing the plan returns
/// `InvalidInput`.
#[derive(Clone, Copy, Debug)]
pub enum SpectralZeroPolicy<S: RealScalar> {
    /// Returns `InvalidInput` if a magnitude is at or below
    /// `relative_threshold` times the peak magnitude.
    Reject {
        /// Threshold relative to the peak magnitude, in `(0, 1)`.
        relative_threshold: S,
    },
    /// Raises magnitudes below `relative_floor` times the peak magnitude to
    /// that floor.
    Regularize {
        /// Floor relative to the peak magnitude, in `(0, 1)`.
        relative_floor: S,
    },
}
/// Options of a minimum-phase conversion.
///
/// The defaults are the homomorphic method, the half-length convention, the
/// default FFT length, a relative floor of `32ε` (with `ε` the machine epsilon
/// of `S`), the default symmetry tolerance and sequential execution.
/// [`MinimumPhasePlan::new`] checks the options.
///
/// ```
/// use scioxide_signal::{MinimumPhaseConvention, MinimumPhaseOptions};
/// let options = MinimumPhaseOptions::<f64>::new()
///     .with_convention(MinimumPhaseConvention::SameLength)
///     .with_fft_length(Some(256));
/// assert_eq!(options.fft_length(), Some(256));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MinimumPhaseOptions<S: RealScalar> {
    method: MinimumPhaseMethod,
    convention: MinimumPhaseConvention,
    fft_length: Option<usize>,
    spectral_zeros: SpectralZeroPolicy<S>,
    symmetry_tolerance: ResidualTolerance<S>,
    execution_policy: ExecutionPolicy,
}
impl<S: RealScalar> Default for MinimumPhaseOptions<S> {
    fn default() -> Self {
        Self {
            method: MinimumPhaseMethod::Homomorphic,
            convention: MinimumPhaseConvention::HalfLength,
            fft_length: None,
            spectral_zeros: SpectralZeroPolicy::Regularize {
                relative_floor: S::EPSILON * (S::TWO).powi(5),
            },
            symmetry_tolerance: ResidualTolerance::default(),
            execution_policy: ExecutionPolicy::Sequential,
        }
    }
}
impl<S: RealScalar> MinimumPhaseOptions<S> {
    /// Creates the default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the algorithm.
    pub const fn with_method(mut self, method: MinimumPhaseMethod) -> Self {
        self.method = method;
        self
    }
    /// Sets the length and magnitude convention.
    pub const fn with_convention(mut self, convention: MinimumPhaseConvention) -> Self {
        self.convention = convention;
        self
    }
    /// Sets the FFT length, or `None` for the default
    /// `max(200 (N − 1), N)` rounded up to a power of two, for `N` input taps.
    ///
    /// The grid is never refined automatically. A longer FFT reduces the
    /// aliasing of the cepstrum.
    pub const fn with_fft_length(mut self, fft_length: Option<usize>) -> Self {
        self.fft_length = fft_length;
        self
    }
    /// Sets the treatment of zero and near-zero magnitudes.
    pub const fn with_spectral_zeros(mut self, spectral_zeros: SpectralZeroPolicy<S>) -> Self {
        self.spectral_zeros = spectral_zeros;
        self
    }
    /// Sets the tolerance with which the Hilbert method checks that the input
    /// taps are symmetric.
    pub const fn with_symmetry_tolerance(mut self, tolerance: ResidualTolerance<S>) -> Self {
        self.symmetry_tolerance = tolerance;
        self
    }
    /// Sets the execution policy of the plan's FFTs.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }
    /// Returns the algorithm.
    pub const fn method(self) -> MinimumPhaseMethod {
        self.method
    }
    /// Returns the length and magnitude convention.
    pub const fn convention(self) -> MinimumPhaseConvention {
        self.convention
    }
    /// Returns the FFT length, or `None` for the default.
    pub const fn fft_length(self) -> Option<usize> {
        self.fft_length
    }
    /// Returns the treatment of zero and near-zero magnitudes.
    pub const fn spectral_zeros(self) -> SpectralZeroPolicy<S> {
        self.spectral_zeros
    }
    /// Returns the symmetry tolerance of the Hilbert method.
    pub const fn symmetry_tolerance(self) -> ResidualTolerance<S> {
        self.symmetry_tolerance
    }
    /// Returns the execution policy of the plan's FFTs.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.execution_policy
    }
}
/// Diagnostics of a minimum-phase conversion, measured on the FFT grid.
///
/// The values describe the result at the grid frequencies only. They estimate
/// the error between grid points but do not bound it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinimumPhaseDiagnostics<S: RealScalar> {
    fft_length: usize,
    spectral_ratio: S,
    regularized_bins: usize,
    relative_floor: S,
    magnitude_error: S,
    discarded_energy_fraction: S,
    hilbert_lift: S,
    hilbert_normalization: S,
}
impl<S: RealScalar> MinimumPhaseDiagnostics<S> {
    /// Returns the FFT length.
    pub const fn fft_length(self) -> usize {
        self.fft_length
    }
    /// Returns the ratio of the smallest to the largest target magnitude
    /// before flooring. Zero means that the grid samples a spectral zero.
    pub const fn spectral_ratio(self) -> S {
        self.spectral_ratio
    }
    /// Returns the number of grid points raised to the floor.
    pub const fn regularized_bins(self) -> usize {
        self.regularized_bins
    }
    /// Returns the relative floor that was applied, or zero for
    /// [`SpectralZeroPolicy::Reject`].
    pub const fn relative_floor(self) -> S {
        self.relative_floor
    }
    /// Returns the largest difference between the magnitude response of the
    /// result and the target magnitude, divided by the largest target
    /// magnitude before flooring.
    pub const fn magnitude_error(self) -> S {
        self.magnitude_error
    }
    /// Returns the energy of the minimum-phase sequence beyond the returned
    /// taps, divided by its total energy over the FFT length.
    pub const fn discarded_energy_fraction(self) -> S {
        self.discarded_energy_fraction
    }
    /// Returns the lift `δ` that the Hilbert method adds to the zero-phase
    /// amplitude, in units of the input response, or zero for the homomorphic
    /// method.
    pub const fn hilbert_lift(self) -> S {
        self.hilbert_lift
    }
    /// Returns the factor by which the Hilbert method scales the lifted
    /// zero-phase amplitude, or one for the homomorphic method.
    pub const fn hilbert_normalization(self) -> S {
        self.hilbert_normalization
    }
}
/// Converts FIR filters of a fixed length to minimum phase.
///
/// A filter with transfer function `H(z) = Σₖ h[k] z^(−k)` has minimum phase
/// when all zeros of `H` lie inside or on the unit circle. The plan computes
/// such a filter `h_min` whose magnitude response approximates a target `M`.
/// For the homomorphic method `M` is `|H|` or `√|H|` according to
/// [`MinimumPhaseConvention`], and for the Hilbert method it is described on
/// [`MinimumPhaseMethod::Hilbert`]. The result is rotated so that `h_min[0]`
/// is real and positive.
///
/// With the homomorphic method and an FFT of length `L`, the plan computes the
/// real cepstrum `c = IDFT(log M)` on the grid, folds it to
/// `ĉ[0] = c[0]`, `ĉ[k] = 2c[k]` for `0 < k < L/2`, `ĉ[L/2] = c[L/2]` and zero
/// beyond, and returns the first taps of `IDFT(exp(DFT(ĉ)))`. Each conversion
/// costs a few FFTs of length `L`, `O(L log L)` in total. The workspace holds
/// `3L` complex values plus the FFT scratch, and its request type is `()`; see
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{MinimumPhaseConvention, MinimumPhaseOptions, MinimumPhasePlan};
/// let options = MinimumPhaseOptions::new()
///     .with_convention(MinimumPhaseConvention::SameLength)
///     .with_fft_length(Some(256));
/// let plan = MinimumPhasePlan::<f64>::new(2, options)?;
/// let mut workspace = plan.workspace(())?;
/// let mut output = [0.0; 2];
/// // 0.5 − z⁻¹ has its zero at z = 2. The minimum-phase filter 1 − 0.5 z⁻¹
/// // has its zero at z = 0.5 and the same magnitude response.
/// let context = ExecutionContext::sequential_native();
/// plan.reconstruct_into(&[0.5, -1.0], &mut output, &mut workspace, context)?;
/// assert!((output[0] - 1.0).abs() < 1e-10 && (output[1] + 0.5).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd
/// ed., Pearson, 2010, ch. 13.
#[derive(Clone, Debug)]
pub struct MinimumPhasePlan<T: NumericScalar> {
    forward: FftPlan<T::Real>,
    inverse: FftPlan<T::Real>,
    input_len: usize,
    output_len: usize,
    complex_len: usize,
    options: MinimumPhaseOptions<T::Real>,
    marker: PhantomData<T>,
}
impl<T: NumericScalar> MinimumPhasePlan<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Prepares the conversion of filters with `input_len` taps.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `input_len` is zero.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the Hilbert method is
    ///   combined with complex taps, the same-length convention, or an
    ///   `input_len` that is even or less than three.
    /// - `InvalidInput` if the threshold or floor of the
    ///   [`SpectralZeroPolicy`] lies outside `(0, 1)`, and `NonFinite` if it is
    ///   NaN or infinite.
    /// - `InvalidInput` (`Inconsistent`) if the FFT length is less than
    ///   `input_len`.
    /// - `SizeOverflow` if the default FFT length is not representable.
    pub fn new(
        input_len: usize,
        options: MinimumPhaseOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            input_len,
            options,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
    /// Prepares the conversion like [`new`](Self::new), planning the FFTs with
    /// the execution context `cx`.
    ///
    /// The FFTs of the plan run with the execution policy of the options.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`new`](Self::new), and `ExecutionStopped` on
    /// cancellation or deadline expiry.
    pub fn build(
        input_len: usize,
        options: MinimumPhaseOptions<T::Real>,
        cx: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        cx.check()?;
        if input_len == 0 {
            return Err(empty(
                scioxide_core::argument!("input_len"),
                "input length must be positive",
            ));
        }
        let complex =
            T::from_components(T::Real::ZERO, T::Real::ONE).imaginary_component() != T::Real::ZERO;
        if options.method == MinimumPhaseMethod::Hilbert
            && (complex
                || input_len < 3
                || input_len.is_multiple_of(2)
                || options.convention != MinimumPhaseConvention::HalfLength)
        {
            return Err(unsupported(
                scioxide_core::argument!("options.method"),
                "Hilbert method needs real odd input_len >= 3 and half-length convention",
            ));
        }
        let threshold = match options.spectral_zeros {
            SpectralZeroPolicy::Reject { relative_threshold } => relative_threshold,
            SpectralZeroPolicy::Regularize { relative_floor } => relative_floor,
        };
        crate::util::check_real_slice_finite(&[threshold])?;
        if threshold <= T::Real::ZERO || threshold >= T::Real::ONE {
            return Err(invalid(
                scioxide_core::argument!("options.spectral_zeros"),
                "spectral threshold must lie in (0, 1)",
            ));
        }
        let n = match options.fft_length {
            Some(n) => n,
            None => checked_mul(
                input_len - 1,
                200,
                scioxide_core::quantity!("minimum_phase_fft_length"),
            )?
            .max(input_len)
            .checked_next_power_of_two()
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("minimum_phase_fft_length"))
            })?,
        };
        if n < input_len {
            return Err(inconsistent(
                scioxide_core::argument!("options.fft_length"),
                "FFT length must be at least the input length",
            ));
        }
        let mut cache =
            scioxide_fft::FftPlanCache::new(scioxide_fft::FftCachePolicy::new(1, usize::MAX)?)?;
        let spec = |direction| {
            FftPlanSpec::new(n, direction, Normalization::Inverse, FftLayout::Complex)
                .with_execution_policy(options.execution_policy)
        };
        let forward = cache.plan(spec(FftDirection::Forward), cx)?;
        let inverse = cache.plan(spec(FftDirection::Inverse), cx)?;
        let fft_scratch = forward
            .workspace_requirements(())?
            .complex_len()
            .max(inverse.workspace_requirements(())?.complex_len());
        let complex_len = checked_add(
            checked_mul(n, 3, scioxide_core::quantity!("minimum_phase_workspace"))?,
            fft_scratch,
            scioxide_core::quantity!("minimum_phase_workspace"),
        )?;
        let output_len = match options.convention {
            MinimumPhaseConvention::SameLength => input_len,
            MinimumPhaseConvention::HalfLength => input_len / 2 + input_len % 2,
        };
        Ok(Self {
            forward,
            inverse,
            input_len,
            output_len,
            complex_len,
            options,
            marker: PhantomData,
        })
    }
    /// Returns the number of input taps.
    pub fn input_len(&self) -> usize {
        self.input_len
    }
    /// Returns the number of output taps.
    pub fn output_len(&self) -> usize {
        self.output_len
    }
    /// Returns the FFT length.
    pub fn fft_length(&self) -> usize {
        self.forward.spec().length()
    }
    /// Returns the workspace requirements, which preparation fixes.
    ///
    /// # Errors
    ///
    /// This query does not fail.
    pub fn workspace_requirements(
        &self,
        _request: (),
    ) -> Result<SignalRequirements, SciOxideError> {
        Ok(SignalRequirements::complex(self.complex_len))
    }
    /// Allocates a workspace for [`Self::reconstruct_into`].
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the workspace cannot be
    /// allocated.
    pub fn workspace(&self, request: ()) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(request)?)
    }
    /// Computes the minimum-phase filter of `input` and writes its
    /// [`output_len`](Self::output_len) taps to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` differs from the prepared
    ///   lengths, or if the workspace is too small.
    /// - `NonFinite` if a tap of `input` is NaN or infinite.
    /// - `DomainError` if all taps of `input` are zero.
    /// - `InvalidInput` (`Inconsistent`) with the Hilbert method if `input` is
    ///   not symmetric within the symmetry tolerance, or if `A_max + δ ≤ 0` or
    ///   `A_max ≥ 2 + δ` (see [`MinimumPhaseMethod::Hilbert`]).
    /// - `InvalidInput` if [`SpectralZeroPolicy::Reject`] finds a magnitude at
    ///   or below its threshold.
    /// - `NumericalInstability` if the magnitude spectrum overflows or
    ///   underflows, or if the leading tap of the result vanishes or overflows.
    /// - `ExecutionStopped` on cancellation or deadline expiry.
    pub fn reconstruct_into(
        &self,
        input: &[T],
        output: &mut [T],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<MinimumPhaseDiagnostics<T::Real>, SciOxideError> {
        let input_argument = scioxide_core::argument!("input");
        let finite = |value| finite(value);
        exact_len(self.input_len, input.len())?;
        exact_len(self.output_len, output.len())?;
        let scratch = ws.buffers(&self.workspace_requirements(())?)?.complex;
        cx.check()?;
        let mut scale = T::Real::ZERO;
        for x in input {
            if !x.real_component().is_finite() || !x.imaginary_component().is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    "coefficients must be finite",
                ));
            }
            scale = scale
                .max(x.real_component().abs())
                .max(x.imaginary_component().abs());
        }
        if scale == T::Real::ZERO {
            return Err(SciOxideError::domain_error(
                input_argument,
                scioxide_core::DomainReason::Undefined,
                "the minimum phase of an all-zero input is undefined",
            ));
        }
        let n = self.fft_length();
        let (data, rest) = scratch.split_at_mut(n);
        let (target, rest) = rest.split_at_mut(n);
        let (verify, fft_scratch) = rest.split_at_mut(n);
        data.fill(Complex::new(T::Real::ZERO, T::Real::ZERO));
        data.iter_mut().zip(input.iter()).for_each(|(z, x)| {
            *z = Complex::new(x.real_component() / scale, x.imaginary_component() / scale)
        });
        if self.options.method == MinimumPhaseMethod::Hilbert {
            let tol = self.options.symmetry_tolerance;
            if data[..self.input_len]
                .iter()
                .zip(data[..self.input_len].iter().rev())
                .any(|(a, b)| {
                    (a.re - b.re).abs()
                        > tol.absolute() / scale + tol.relative() * a.re.abs().max(b.re.abs())
                })
            {
                return Err(inconsistent(
                    input_argument,
                    "Hilbert method requires symmetric input within tolerance",
                ));
            }
        }
        self.forward.execute_in_place(data, fft_scratch, cx)?;
        let mut lift = T::Real::ZERO;
        let mut normalization = T::Real::ONE;
        let gain = if self.options.method == MinimumPhaseMethod::Homomorphic {
            target
                .iter_mut()
                .zip(data.iter())
                .for_each(|(m, z)| m.re = z.norm());
            if self.options.convention == MinimumPhaseConvention::HalfLength {
                target.iter_mut().for_each(|z| z.re = z.re.sqrt());
                scale.sqrt()
            } else {
                scale
            }
        } else {
            let delay = scalar::<T::Real>(self.input_len / 2)?;
            let step = T::Real::TAU * delay / scalar::<T::Real>(n)?;
            for (k, (m, z)) in target.iter_mut().zip(data.iter()).enumerate() {
                let (sin, cos) = (step * scalar::<T::Real>(k)?).sin_cos();
                m.re = finite((z.re * cos - z.im * sin) * scale)?;
            }
            let low = target
                .iter()
                .map(|z| z.re)
                .fold(T::Real::INFINITY, T::Real::min);
            let high = target
                .iter()
                .map(|z| z.re)
                .fold(T::Real::NEG_INFINITY, T::Real::max);
            lift = (-low).max(T::Real::ZERO);
            let a = finite(high + lift)?;
            let b = finite(T::Real::TWO - high + lift)?;
            if a <= T::Real::ZERO || b <= T::Real::ZERO {
                return Err(inconsistent(
                    input_argument,
                    "Hilbert method requires a spectrum below unity and zero gain",
                ));
            }
            let sum = a.sqrt() + b.sqrt();
            normalization = finite((T::Real::TWO) / sum)?;
            normalization = finite(normalization * normalization)?;
            target.iter_mut().try_for_each(|z| {
                z.re = finite(((z.re + lift).max(T::Real::ZERO) * normalization).sqrt())?;
                Ok::<_, SciOxideError>(())
            })?;
            T::Real::ONE
        };
        let peak = target
            .iter()
            .map(|z| z.re)
            .fold(T::Real::ZERO, T::Real::max);
        let minimum = target
            .iter()
            .map(|z| z.re)
            .fold(T::Real::INFINITY, T::Real::min);
        if !peak.is_finite() || peak == T::Real::ZERO {
            return Err(SciOxideError::numerical_instability(
                if peak.is_finite() {
                    scioxide_core::NumericalIssue::Underflow
                } else {
                    scioxide_core::NumericalIssue::Overflow
                },
                "magnitude spectrum of a nonzero input must be finite and nonzero",
            ));
        }
        let (threshold, regularize) = match self.options.spectral_zeros {
            SpectralZeroPolicy::Reject { relative_threshold } => (relative_threshold, false),
            SpectralZeroPolicy::Regularize { relative_floor } => (relative_floor, true),
        };
        let floor = threshold * peak;
        if floor == T::Real::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "relative spectral threshold is not representable at this scale",
            ));
        }
        let mut regularized_bins = 0;
        for (z, m) in data.iter_mut().zip(target.iter_mut()) {
            // Preserve the unfloored target for independent truncation/floor diagnostics.
            m.im = m.re;
            if m.re <= floor && !regularize {
                return Err(invalid(
                    input_argument,
                    "spectrum has near-zero bins and the policy rejects them",
                ));
            }
            if m.re < floor {
                m.re = floor;
                regularized_bins += 1;
            }
            *z = Complex::new(m.re.ln(), T::Real::ZERO);
        }
        self.inverse.execute_in_place(data, fft_scratch, cx)?;
        if self.options.method == MinimumPhaseMethod::Homomorphic {
            for (k, z) in data.iter_mut().enumerate() {
                if k > n / 2 {
                    *z = Complex::new(T::Real::ZERO, T::Real::ZERO);
                } else if k != 0 && !(n.is_multiple_of(2) && k == n / 2) {
                    *z *= Complex::new(T::Real::TWO, T::Real::ZERO);
                }
            }
            self.forward.execute_in_place(data, fft_scratch, cx)?;
            data.iter_mut().for_each(|z| *z = z.exp());
        } else {
            // Modified DHT: +/- time-index sign recovers the minimum-phase log phase.
            for (k, z) in data.iter_mut().enumerate() {
                if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
                    *z = Complex::new(T::Real::ZERO, T::Real::ZERO);
                } else if k > n / 2 {
                    *z = -*z;
                }
            }
            self.forward.execute_in_place(data, fft_scratch, cx)?;
            data.iter_mut()
                .zip(target.iter())
                .for_each(|(z, m)| *z = z.exp() * Complex::new(m.re, T::Real::ZERO));
        }
        self.inverse.execute_in_place(data, fft_scratch, cx)?;
        let first = data[0].norm();
        if first == T::Real::ZERO || !first.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "minimum-phase leading coefficient is zero or non-finite",
            ));
        }
        let rotation = Complex::new(data[0].re / first, -data[0].im / first);
        let real_output =
            T::from_components(T::Real::ZERO, T::Real::ONE).imaginary_component() == T::Real::ZERO;
        data.iter_mut().try_for_each(|z| {
            *z *= rotation;
            finite(z.re)?;
            finite(z.im)?;
            if real_output {
                z.im = T::Real::ZERO;
            }
            Ok::<_, SciOxideError>(())
        })?;
        let max_tap = data
            .iter()
            .map(|z| z.norm())
            .fold(T::Real::ZERO, T::Real::max);
        let mut total = T::Real::ZERO;
        let mut tail = T::Real::ZERO;
        for (k, z) in data.iter().enumerate() {
            let energy = Complex::new(z.re / max_tap, z.im / max_tap).norm_sqr();
            total += energy;
            if k >= self.output_len {
                tail += energy;
            }
        }
        verify.fill(Complex::new(T::Real::ZERO, T::Real::ZERO));
        verify[..self.output_len].copy_from_slice(&data[..self.output_len]);
        self.forward.execute_in_place(verify, fft_scratch, cx)?;
        let error = verify
            .iter()
            .zip(target.iter())
            .map(|(z, m)| (z.norm() / peak - m.im / peak).abs())
            .fold(T::Real::ZERO, T::Real::max);
        // Validate scaled output before committing any caller coefficient.
        data[..self.output_len].iter_mut().try_for_each(|z| {
            z.re = finite(z.re * gain)?;
            z.im = finite(z.im * gain)?;
            Ok::<_, SciOxideError>(())
        })?;
        if data[0].re <= T::Real::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "gain restoration destroyed the causal leading coefficient",
            ));
        }
        let report = MinimumPhaseDiagnostics {
            fft_length: n,
            spectral_ratio: minimum / peak,
            regularized_bins,
            relative_floor: if regularize { threshold } else { T::Real::ZERO },
            magnitude_error: finite(error)?,
            discarded_energy_fraction: finite(tail / total)?,
            hilbert_lift: lift,
            hilbert_normalization: normalization,
        };
        cx.check()?;
        output
            .iter_mut()
            .zip(data.iter().take(self.output_len))
            .for_each(|(y, z)| *y = T::from_components(z.re, z.im));
        Ok(report)
    }
}
/// Computes the minimum-phase spectral factor of the FIR filter `input`.
///
/// For `N` taps with response `H`, the result is the `⌈N/2⌉` taps of a causal
/// minimum-phase filter whose magnitude response approximates `√|H|`; for a
/// linear-phase filter this is its minimum-phase spectral factor. The default
/// [`MinimumPhaseOptions`] apply: the homomorphic method of
/// [`MinimumPhasePlan`] on an FFT grid of `max(200 (N − 1), N)` points rounded
/// up to a power of two, with magnitudes below `32ε` of the peak raised to
/// that floor, where `ε` is the machine epsilon. Each conversion costs a few
/// FFTs of the grid length `L`, `O(L log L)` in total.
/// [`minimum_phase_with_options`] takes other options and also returns the
/// diagnostics of the conversion.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `input` is empty.
/// - `SizeOverflow` if the FFT length is not representable.
/// - `NonFinite` if a tap is NaN or infinite.
/// - `DomainError` if all taps are zero.
/// - `NumericalInstability` if the magnitude spectrum overflows or
///   underflows, or if the leading tap of the result vanishes or overflows.
///
/// # Examples
///
/// ```
/// use scioxide_signal::minimum_phase;
/// // (1 − 0.5 z⁻¹)(0.5 − z⁻¹) is linear phase with |H| = |1 − 0.5 z⁻¹|²:
/// // its spectral factor is 1 − 0.5 z⁻¹.
/// let taps = minimum_phase(&[0.5_f64, -1.25, 0.5])?;
/// assert_eq!(taps.len(), 2);
/// assert!((taps[0] - 1.0).abs() < 1e-12 && (taps[1] + 0.5).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd
/// ed., Pearson, 2010, ch. 13.
pub fn minimum_phase<T: NumericScalar>(input: &[T]) -> Result<Vec<T>, SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    Ok(minimum_phase_with_options(input, MinimumPhaseOptions::default())?.0)
}
/// Computes the minimum-phase filter of `input` with `options`, and the
/// diagnostics of the conversion.
///
/// This prepares a [`MinimumPhasePlan`] for `input.len()` taps and calls
/// [`MinimumPhasePlan::reconstruct_into`] once.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `input` is empty.
/// - The other errors of [`MinimumPhasePlan::new`] and
///   [`MinimumPhasePlan::reconstruct_into`].
///
/// # Examples
///
/// ```
/// use scioxide_signal::{MinimumPhaseConvention, MinimumPhaseOptions, minimum_phase_with_options};
/// // 0.5 − z⁻¹ has the minimum-phase counterpart 1 − 0.5 z⁻¹.
/// let options = MinimumPhaseOptions::new().with_convention(MinimumPhaseConvention::SameLength);
/// let (taps, diagnostics) = minimum_phase_with_options(&[0.5_f64, -1.0], options)?;
/// assert!((taps[0] - 1.0).abs() < 1e-10 && diagnostics.magnitude_error() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd
/// ed., Pearson, 2010, ch. 13.
pub fn minimum_phase_with_options<T: NumericScalar>(
    input: &[T],
    options: MinimumPhaseOptions<T::Real>,
) -> Result<(Vec<T>, MinimumPhaseDiagnostics<T::Real>), SciOxideError>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    if input.is_empty() {
        return Err(empty(
            scioxide_core::argument!("input"),
            "input length must be positive",
        ));
    }
    let plan = MinimumPhasePlan::new(input.len(), options)?;
    let mut workspace = plan.workspace(())?;
    let mut output = try_filled(
        plan.output_len(),
        T::ZERO,
        scioxide_core::quantity!("minimum_phase_output"),
    )?;
    let report = plan.reconstruct_into(
        input,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok((output, report))
}
