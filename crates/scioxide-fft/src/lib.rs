//! Fast Fourier transforms and related spectral transforms.
//!
//! The crate computes discrete Fourier transforms of complex, real and
//! Hermitian data in one or more dimensions, discrete cosine and sine
//! transforms of types I–IV, the chirp z-transform, nonuniform FFTs of
//! types 1–3, and Hankel transforms on logarithmic (FFTLog) and Bessel-zero
//! grids. Every transform supports `f32` and `f64`, takes an explicit
//! [`Normalization`], and has the length of its input: no transform pads or
//! truncates implicitly.
//!
//! # Quick start
//!
//! ```
//! use scioxide_core::Complex;
//! use scioxide_fft::{Normalization, fft, ifft};
//! // x[j] = cos(2πj/8) has the unscaled DFT X[1] = X[7] = 4, and X[k] = 0 otherwise.
//! let x: Vec<_> = (0..8)
//!     .map(|j| Complex::new((2.0 * core::f64::consts::PI * j as f64 / 8.0).cos(), 0.0))
//!     .collect();
//! let spectrum = fft(&x, Normalization::Inverse)?;
//! for (k, bin) in spectrum.iter().enumerate() {
//!     let expected = if k == 1 || k == 7 { 4.0 } else { 0.0 };
//!     assert!((*bin - Complex::new(expected, 0.0)).norm() < 1e-12);
//! }
//! let recovered = ifft(&spectrum, Normalization::Inverse)?;
//! for (a, b) in recovered.iter().zip(&x) {
//!     assert!((*a - *b).norm() < 1e-12);
//! }
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Problem | Entry points | Method |
//! |---|---|---|
//! | DFT of complex data | [`fft`], [`ifft`], [`FftPlan`] | mixed-radix Cooley–Tukey (radices 2, 3, 4, 5, 7), Rader for suitable primes, Bluestein otherwise; `O(n log n)` for every `n` |
//! | DFT of real data | [`rfft`], [`irfft`], [`FftPlan`] with [`FftLayout::Real`] | half-length complex FFT for even `n`; real-data decimation or a full complex FFT for odd `n` |
//! | Hermitian data to a real spectrum | [`hfft`], [`ihfft`], [`FftPlan`] with [`FftLayout::Hermitian`] | as for real data |
//! | Selected axes of an N-D array | [`fftn`], [`ifftn`], [`rfftn`], [`irfftn`], [`hfftn`], [`ihfftn`], [`FftNdPlan`] | one 1-D transform per selected axis |
//! | DCT and DST, types I–IV | [`dct`], [`idct`], [`dst`], [`idst`], [`dctn`], [`dstn`], [`RealTransformPlan`], [`RealTransformNdPlan`] | one real or complex FFT per transform; `O(n log n)` |
//! | z-transform on a spiral contour, zoomed band | [`czt`], [`zoom_fft`], [`CztPlan`] | Bluestein convolution, or direct summation for small off-circle or strongly spiraling contours |
//! | Fourier sums at nonuniform points or frequencies | [`nufft_type1`], [`nufft_type2`], [`nufft_type3`], [`NufftPlan`] | Kaiser–Bessel spreading and an oversampled FFT, to a requested accuracy |
//! | Hankel transform on a logarithmic grid | [`fht`], [`ifht`], [`FftLogPlan`] | FFTLog: two FFTs and a Mellin-space multiplication |
//! | Hankel transform on a Bessel-zero grid | [`dht`], [`idht`], [`DiscreteHankelPlan`] | dense quadrature matrix, `O(n²)` per transform |
//! | Frequency grids, shifts, fast lengths | [`fftfreq`], [`rfftfreq`], [`fftshift`], [`ifftshift`], [`fftshift_nd`], [`ifftshift_nd`], [`next_fast_len`], [`prev_fast_len`] | |
//!
//! Every length is supported, but lengths whose only prime factors are 2, 3,
//! 5 and 7 are fastest. [`next_fast_len`] returns the next such length if you
//! want to zero-pad.
//!
//! # Conventions
//!
//! For a sequence `x` of length `n`, the forward and inverse transforms are,
//! before normalization,
//!
//! ```text
//! forward:  X[k] = Σⱼ x[j] exp(−2πi jk/n),   k = 0, …, n − 1,
//! inverse:  x[j] = Σₖ X[k] exp(+2πi jk/n),   j = 0, …, n − 1,
//! ```
//!
//! with both sums over `0, …, n − 1`. Bin `k` holds the frequency `k/n` cycles
//! per sample for `k < n/2` and `(k − n)/n` for `k ≥ n/2`. [`fftfreq`] returns
//! these frequencies for a given sample spacing, and [`fftshift`] moves bin 0 to
//! the center.
//!
//! A real sequence of length `n` has a Hermitian spectrum,
//! `X[n − k] = conj(X[k])`, so the real transforms store only bins
//! `0, …, ⌊n/2⌋`. A [`HermitianSpectrum`] keeps these `⌊n/2⌋ + 1` bins together
//! with `n`, so odd and even lengths are distinguished. N-D transforms pack the
//! last selected axis in the same way ([`HermitianSpectrumNd`]). Unselected
//! axes of an N-D array are independent batches.
//!
//! The DCT and DST definitions are on [`RealTransformPlan`]. The chirp
//! z-transform, nonuniform and Hankel conventions are on [`CztPlan`],
//! [`NufftPlan`], [`FftLogPlan`] and [`DiscreteHankelPlan`].
//!
//! # Normalization
//!
//! Applying the unscaled forward and inverse DFT of length `n` multiplies a
//! sequence by `n`. [`Normalization`] says where the factor `1/n` goes:
//!
//! | [`Normalization`] | Forward scale | Inverse scale |
//! |---|---|---|
//! | `Inverse` | 1 | `1/n` |
//! | `Forward` | `1/n` | 1 |
//! | `Orthonormal` | `1/√n` | `1/√n` |
//! | `Unscaled` | 1 | 1 |
//!
//! Use the same value in both directions for an identity round trip (except
//! `Unscaled`). N-D transforms use the product of the selected axis lengths. The
//! packed real layout does not change the scale. The DCT and DST use their own
//! round-trip factors, listed on [`RealTransformPlan`].
//!
//! Kernels scale the input by a power of two before the transform and apply the
//! inverse scaling together with the normalization at the end, with one
//! rounding. Inputs close to the overflow threshold therefore transform without
//! intermediate overflow. Each batch of an N-D transform is scaled separately,
//! so a batch of large values does not flush the components of a small one.
//!
//! # Hermitian input
//!
//! [`irfft`], [`irfftn`], [`hfft`], [`hfftn`] and the plans with packed input
//! check that the input is Hermitian to rounding error. For each independent
//! spectrum, let `M` be its largest absolute real or imaginary component, `ε`
//! the machine epsilon and `η` the smallest positive subnormal number. With
//! `τ = max(8εM, η)`:
//!
//! - a self-conjugate bin (DC, and the Nyquist bin for even `n`) must have
//!   `|Im| ≤ τ`;
//! - in N-D transforms, the DC and Nyquist planes of the packed axis store both
//!   bins of a conjugate pair; their real parts must differ by at most `2τ` and
//!   their imaginary parts must sum to at most `2τ` in magnitude.
//!
//! Inputs within these limits are made exactly Hermitian in scratch storage;
//! your input is not modified. This projection changes the transform by at most
//! `√2 n (τ + εM + η)` before normalization. Inputs outside the limits return
//! `InvalidInput`; they are not projected silently.
//!
//! ```
//! use scioxide_core::{Complex, SciOxideErrorKind};
//! use scioxide_fft::{HermitianSpectrum, Normalization, irfft};
//!
//! // The DC bin has imaginary part 1e-20 = M, far above τ = 8εM.
//! let bins = HermitianSpectrum::from_vec(
//!     4,
//!     vec![
//!         Complex::new(0.0_f64, 1e-20),
//!         Complex::new(1e-20, 0.0),
//!         Complex::new(0.0, 0.0),
//!     ],
//! )?;
//! let error = irfft(&bins, Normalization::Inverse).unwrap_err();
//! assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: an empty 1-D input, a DCT-I shorter than two samples, a
//!   spectrum that fails the [Hermitian check](#hermitian-input), a repeated or
//!   out-of-range axis, an invalid specification parameter, an [`FftInput`] or [`FftOutput`]
//!   variant that does not match the plan's layout, or a required backend that
//!   the planning context does not provide.
//! - `NonFinite`: an input or parameter is NaN or infinite (origin `Input`), or
//!   an injected [`FftProvider`] returned one (origin `Callback`).
//! - `NumericalInstability`: an output overflows although the input is finite
//!   (`Overflow`), an internal rounding check of a packed real transform fails
//!   (`Breakdown`), or a nonzero coefficient computed during preparation, such
//!   as an FFTLog multiplier, rounds to zero (`Underflow`). Outputs that
//!   underflow to zero are not an error.
//! - `DimensionMismatch`: an output, view or workspace has the wrong shape or
//!   is too short.
//! - `Singular`: an FFTLog plan needs the reciprocal of a zero Mellin
//!   coefficient or meets a pole.
//! - `ResourceLimit`, `AllocationFailure`, `SizeOverflow`: a configured byte
//!   limit is exceeded, or storage cannot be reserved or its size overflows.
//! - `ConvergenceFailure`: Hankel or FFTLog preparation exceeds its work limit.
//! - `ExecutionStopped`: cancellation or a deadline stopped execution.
//!
//! On error, outputs are unchanged as described in
//! [Output on error](scioxide_core#output-on-error). This includes the gaps of a
//! strided output and all lanes of an axis transform.
//!
//! # Accuracy
//!
//! `ε` is the machine epsilon of the scalar type.
//!
//! | Transform | Error |
//! |---|---|
//! | Forward complex and packed real FFT, native kernels | relative `L²` error at most `2ε max(1, log₂ n)` |
//! | [`CztPlan`], [`zoom_fft`] | componentwise bound stated on [`CztPlan`] |
//! | [`NufftPlan`] | error estimate relative to the input `ℓ¹` norm meets the requested accuracy; the estimate is not a bound |
//! | [`DiscreteHankelPlan`] | componentwise bound stated on [`DiscreteHankelPlan`] |
//!
//! The FFT bound is tested against DFTs computed in 100- and 160-digit
//! arithmetic for 2-, 3-, 5- and 7-smooth lengths up to 6720, the odd length
//! 3375, Rader primes up to 65537 and Bluestein primes up to 10007, in `f32` and
//! `f64` and on every native backend. It is a normwise bound: bins much smaller
//! than the largest bin can have a large relative error. It does not cover
//! injected providers.
//!
//! The DC bin is computed with the same floating-point rounding as the other
//! bins; it is not an exactly rounded sum.
//!
//! # Reusing plans and workspaces
//!
//! The one-call functions prepare a plan, run it once and return a `Vec`. They
//! take contiguous data: the N-D functions read a dense row-major slice with
//! its shape. For repeated transforms of the same size, for strided N-D views
//! and for settings beyond the defaults, describe the transform with a spec
//! such as [`FftPlanSpec`], prepare a plan once, and call `execute_into` with a
//! workspace from `workspace`, as described in
//! [Plans and workspaces](scioxide_core#plans-and-workspaces). `execute_into`
//! takes slices, views and [`HermitianSpectrum`] references directly. A plan
//! fixes its direction and normalization; prepare a second plan for the
//! opposite direction. Prepared execution with a large enough workspace does not
//! allocate. [`FftPlanCache`] keeps prepared FFT plans in a bounded cache that
//! you own; SciOxide has no global plan cache.
//!
//! ```
//! use scioxide_core::{Complex, ExecutionContext};
//! use scioxide_fft::{FftDirection, FftLayout, FftPlan, FftPlanSpec, Normalization};
//!
//! let spec = FftPlanSpec::new(4, FftDirection::Forward, Normalization::Inverse, FftLayout::Complex);
//! let plan = FftPlan::<f64>::new(spec)?;
//! let (mut workspace, cx) = (plan.workspace(())?, ExecutionContext::sequential());
//! let mut output = [Complex::new(0.0, 0.0); 4];
//! for amplitude in [1.0, 2.0] {
//!     let input = [Complex::new(amplitude, 0.0); 4];
//!     plan.execute_into(&input, &mut output, &mut workspace, cx)?;
//!     // The DC bin of a constant sequence is the sum of its samples.
//!     assert!((output[0] - Complex::new(4.0 * amplitude, 0.0)).norm() < 1e-12);
//! }
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! The CZT, NUFFT, Hankel and FFTLog plans also transform one axis of an N-D
//! array with `execute_axis_into`. The NUFFT, Hankel and FFTLog plans implement
//! [`LinearOperator`](scioxide_core::LinearOperator) and
//! [`AdjointOperator`](scioxide_core::AdjointOperator). Each application, and each
//! axis call with at least one lane, counts as one operator evaluation under
//! [Solve accounting](scioxide_core#solve-accounting); the FFTs inside it do not
//! count.
//!
//! # Backends and providers
//!
//! Native kernels exist for portable scalar code, AVX2 with FMA on x86-64 and
//! NEON on AArch64. `new` prepares a plan for the CPU features of the host
//! under `std`, and for the portable scalar kernels without `std`. `build` takes an
//! [execution context](scioxide_core#execution-contexts) or an
//! [`FftPlanningContext`] and uses only the CPU features it provides; a plain
//! [`ExecutionContext::sequential`](scioxide_core::ExecutionContext::sequential)
//! selects the scalar kernels. [`FftBackendStrategy`] selects a backend
//! automatically or requires one, and each plan reports its choice through
//! `resolved_backends`. A prepared plan runs with any execution context; it uses
//! only the context's executor, cancellation flag and deadline.
//!
//! An [`FftProvider`] passed through [`FftPlanningContext`] replaces the complex
//! FFT kernels of every transform in this crate. SciOxide still applies the
//! normalization, checks the provider's output for NaN and infinity, and writes
//! the output only on success. Plans prepared with a provider are not cached by
//! [`FftPlanCache`].
//!
//! # Feature flags
//!
//! - `std` (default): `new` detects the CPU features of the host at run time.
//!
//! Without default features the crate is `no_std` and needs only `alloc`. The
//! API is the same, and `new` prepares plans for the portable scalar kernels.
//!
//! # References
//!
//! - J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of
//!   complex Fourier series", *Math. Comp.* 19(90), 297–301, 1965.
//! - C. M. Rader, "Discrete Fourier transforms when the number of data samples
//!   is prime", *Proc. IEEE* 56(6), 1107–1108, 1968.
//! - L. I. Bluestein, "A linear filtering approach to the computation of
//!   discrete Fourier transform", *IEEE Trans. Audio Electroacoust.* 18(4),
//!   451–455, 1970.
//! - L. R. Rabiner, R. W. Schafer and C. M. Rader, "The chirp z-transform
//!   algorithm", *IEEE Trans. Audio Electroacoust.* 17(2), 86–92, 1969.
//! - A. H. Barnett, J. Magland and L. af Klinteberg, "A parallel nonuniform
//!   fast Fourier transform library based on an 'exponential of semicircle'
//!   kernel", *SIAM J. Sci. Comput.* 41(5), C479–C504, 2019.
//! - P. J. Beatty, D. G. Nishimura and J. M. Pauly, "Rapid gridding
//!   reconstruction with a minimal oversampling ratio", *IEEE Trans. Med.
//!   Imaging* 24(6), 799–808, 2005.
//! - A. J. S. Hamilton, "Uncorrelated modes of the non-linear power spectrum",
//!   *Mon. Not. R. Astron. Soc.* 312(2), 257–284, 2000.
//! - C. Van Loan, *Computational Frameworks for the Fast Fourier Transform*,
//!   SIAM, 1992.

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![deny(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg_attr(all(test, not(feature = "std")), macro_use)]
extern crate alloc;
#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod chirp;
mod czt;
pub use czt::{CztAlgorithm, CztContour, CztLimits, CztPlan, CztPlanSpec};
mod nufft;
pub use nufft::{NufftDiagnostics, NufftLimits, NufftPlan, NufftPlanSpec, NufftTransform};
mod cache;
mod complex;
mod frequency;
mod hermitian;
mod io;
mod memory;
mod numeric;
mod plan;
mod real_fft;
mod scaling;
mod smooth_len;
mod tensor;
#[cfg(test)]
#[path = "../tests/support/numerics.rs"]
mod test_support;
mod trigonometric;
mod types;
mod validation;

mod kernel;

// -- Re-exports: complex FFT ------------------------------------------------
pub use simple::{fft, ifft};

// -- Re-exports: real FFT ---------------------------------------------------
pub use simple::{irfft, rfft};

// -- Re-exports: plan --------------------------------------------------------
pub use cache::{FftCachePolicy, FftCacheStats, FftPlanCache};
pub use io::{FftInput, FftOutput, HermitianSpectrum};
pub use plan::{FftBackendStrategy, FftDirection, FftLayout, FftPlan, FftPlanSpec};
pub use types::{FftAlgorithmKind, FftPlanDiagnostics, Normalization};

// -- Re-exports: DCT/DST ----------------------------------------------------
pub use trigonometric::{dct, dst, idct, idst};
pub use types::{DctType, DstType};

// -- Re-exports: utilities --------------------------------------------------
pub use frequency::{fftfreq, fftshift, ifftshift, next_fast_len, prev_fast_len, rfftfreq};

mod lanes;
mod nd;
pub use io::HermitianSpectrumNd;
pub use nd::{FftNdInput, FftNdOutput, FftNdPlan, FftNdPlanSpec};
pub use simple::{fftn, fftshift_nd, hfftn, ifftn, ifftshift_nd, ihfftn, irfftn, rfftn};

mod real_transform;
pub use real_transform::{RealTransformKind, RealTransformPlan, RealTransformPlanSpec};
mod real_nd;
pub use real_nd::{RealTransformNdPlan, RealTransformNdPlanSpec, dctn, dstn, idctn, idstn};

// Test-only source dependency on the workspace reference loader and its fixtures.
// It is included directly because its owning tool also tests scioxide-fft; there
// is no Cargo dependency from this published crate to that consumer tool.

mod backend;
pub use backend::{
    FftBackend, FftPlanningContext, FftProvider, FftProviderLimits, FftProviderPlan,
    FftProviderPolicy,
};

mod hankel;
mod resource;
pub use hankel::{
    DiscreteHankelDiagnostics, DiscreteHankelPlan, DiscreteHankelPlanSpec, HankelLimits,
};
mod fftlog;
pub use fftlog::{
    FftLogDiagnostics, FftLogLimits, FftLogPlan, FftLogPlanSpec, fftlog_low_ringing_offset,
};

mod axis;

mod simple;
pub use simple::{
    czt, dht, fht, hfft, idht, ifht, ihfft, nufft_type1, nufft_type2, nufft_type3, zoom_fft,
};

mod workspace;
pub use workspace::{
    AxisWorkspaceRequest, FftWorkspace, FftWorkspaceRequirements, RealTransformNdWorkspace,
    RealTransformNdWorkspaceRequirements,
};
