//! One-call transforms that prepare a plan, run it once and return a new vector.

use crate::validation::invalid;
use crate::{
    FftDirection, FftInput, FftLayout, FftNdPlan, FftNdPlanSpec, FftPlan, FftPlanSpec,
    HermitianSpectrumNd,
};
use alloc::vec::Vec;
use scioxide_core::ExecutionContext;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{Complex, NdView, NdViewMut, RealScalar, SciOxideError, Shape};

use crate::{
    CztContour, CztPlan, CztPlanSpec, DiscreteHankelPlan, DiscreteHankelPlanSpec, FftLogPlan,
    FftLogPlanSpec, HermitianSpectrum, Normalization, NufftPlan, NufftPlanSpec,
};

/// Returns the length of a real or complex one-shot input, which must be
/// nonempty; packed input is rejected.
fn full_len<S: RealScalar>(input: FftInput<'_, S>) -> Result<usize, SciOxideError>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    let n = crate::io::FullInput::try_from(input)?.len();
    crate::validation::nonzero(n, scioxide_core::argument!("input"))?;
    Ok(n)
}

/// Computes the chirp z-transform of `input` at `m` points of a geometric contour.
///
/// ```text
/// X[k] = Σⱼ x[j] z_k^(−j),   z_k = a · w^(−k),   0 ≤ j < n,  0 ≤ k < m,
/// ```
///
/// where `x` is the real or complex sequence `input` of length `n`, `a` is the
/// first contour point and `w` is a nonzero complex ratio. Real samples give
/// the same result as complex samples with zero imaginary parts. With `a = 1`
/// and `w = exp(−2πi/n)`, `X` is the forward DFT. [`Normalization::Forward`]
/// divides by `n`, [`Normalization::Orthonormal`] by `√n`, and the other two
/// leave the sum unscaled.
///
/// Contours on the unit circle use Bluestein's algorithm: two FFTs of length at
/// least `n + m − 1`. Some off-circle spirals use an `O(n m)` direct sum; see
/// [`CztPlan`], which also avoids repeating the preparation for many inputs.
///
/// # Errors
///
/// - `InvalidInput` (`UnsupportedConfiguration`) if `input` is packed
///   ([`FftInput::Hermitian`]).
/// - `InvalidInput` (`Empty`) if `input` is empty or `m` is zero.
/// - `InvalidInput` (`OutOfRange`) if `w` or `a` is zero.
/// - `NonFinite` if `w`, `a` or an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{Normalization, czt};
///
/// // With w = −i, the contour 1, i, −1 gives the first three bins of the
/// // 4-point DFT of [1, 2, 0, 0]: X[k] = 1 + 2 (−i)^k.
/// let x = [Complex::new(1.0_f64, 0.0), Complex::new(2.0, 0.0)];
/// let (w, a) = (Complex::new(0.0, -1.0), Complex::new(1.0, 0.0));
/// let bins = czt(&x, 3, w, a, Normalization::Unscaled)?;
/// assert!((bins[0] - Complex::new(3.0, 0.0)).norm() < 1e-12);
/// assert!((bins[1] - Complex::new(1.0, -2.0)).norm() < 1e-12);
/// assert!((bins[2] - Complex::new(-1.0, 0.0)).norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - L. R. Rabiner, R. W. Schafer and C. M. Rader, "The chirp z-transform
///   algorithm", *IEEE Trans. Audio Electroacoust.* 17(2), 1969, 86–92.
/// - L. I. Bluestein, "A linear filtering approach to the computation of
///   discrete Fourier transform", *IEEE Trans. Audio Electroacoust.* 18(4),
///   1970, 451–455.
pub fn czt<'i, S: RealScalar>(
    input: impl Into<FftInput<'i, S>>,
    m: usize,
    w: Complex<S>,
    a: Complex<S>,
    normalization: Normalization,
) -> Result<Vec<Complex<S>>, SciOxideError>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    let input = input.into();
    let n = full_len(input)?;
    crate::validation::nonzero(m, scioxide_core::argument!("m"))?;
    let contour = CztContour::from_wa(w, a)?;
    CztPlan::new(CztPlanSpec::new(n, m, contour, normalization))?.execute_owned(input)
}

/// Computes the discrete-time Fourier transform of `input` at `m` equally spaced
/// frequencies in a band.
///
/// ```text
/// X[k] = Σⱼ x[j] exp(−2πi j f_k / f_s),   f_k = start + k Δ,   0 ≤ k < m,
/// ```
///
/// where `x` is the real or complex sequence `input` and `f_s` is
/// `sample_rate`. The step is `Δ = (end − start)/(m − 1)` when `endpoint` is
/// true and `Δ = (end − start)/m` otherwise, so `end` is included only when
/// `endpoint` is true. A single point samples `start`. Negative and descending
/// bands are valid. Normalization scales by the input length as in [`czt`],
/// which this function calls with a unit-circle contour. Use [`CztPlan`] with
/// [`CztContour::zoom`] to transform many inputs.
///
/// # Errors
///
/// - `InvalidInput` (`UnsupportedConfiguration`) if `input` is packed
///   ([`FftInput::Hermitian`]).
/// - `InvalidInput` (`Empty`) if `input` is empty or `m` is zero.
/// - `InvalidInput` (`OutOfRange`) if `sample_rate` is not positive.
/// - `NonFinite` if `start`, `end`, `sample_rate` or an input value is NaN or
///   infinite.
/// - `NumericalInstability` (`Overflow`) if `start / sample_rate` or the step
///   overflows, or a result is not representable.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{Normalization, zoom_fft};
///
/// // X(f) = 1 + 2 exp(−2πi f) at f = 0, 0.125, 0.25.
/// let x = [1.0_f64, 2.0];
/// let bins = zoom_fft(&x, 0.0, 0.25, 3, 1.0, true, Normalization::Unscaled)?;
/// assert!((bins[0] - Complex::new(3.0, 0.0)).norm() < 1e-12);
/// assert!((bins[2] - Complex::new(1.0, -2.0)).norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn zoom_fft<'i, S: RealScalar>(
    input: impl Into<FftInput<'i, S>>,
    start: S,
    end: S,
    m: usize,
    sample_rate: S,
    endpoint: bool,
    normalization: Normalization,
) -> Result<Vec<Complex<S>>, SciOxideError>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    let input = input.into();
    let n = full_len(input)?;
    let contour = CztContour::zoom(start, end, m, sample_rate, endpoint)?;
    CztPlan::new(CztPlanSpec::new(n, m, contour, normalization))?.execute_owned(input)
}

/// Copies real samples into new complex storage with zero imaginary parts.
fn complexified<S: RealScalar>(
    input: &[S],
    quantity: scioxide_core::Quantity,
) -> Result<Vec<Complex<S>>, SciOxideError> {
    let mut values = try_with_capacity(input.len(), quantity)?;
    values.extend(input.iter().map(|&x| Complex::new(x, S::ZERO)));
    Ok(values)
}

/// Copies the real parts of `values` into new storage.
fn real_parts<S: RealScalar>(
    values: &[Complex<S>],
    quantity: scioxide_core::Quantity,
) -> Result<Vec<S>, SciOxideError> {
    let mut real = try_with_capacity(values.len(), quantity)?;
    real.extend(values.iter().map(|z| z.re));
    Ok(real)
}

const fn direction(inverse: bool) -> FftDirection {
    if inverse {
        FftDirection::Inverse
    } else {
        FftDirection::Forward
    }
}

// Shared real-input adapter for `fht` and `ifht`.
fn fftlog_real<S: RealScalar>(
    input: &[S],
    dln: S,
    mu: S,
    offset: S,
    bias: S,
    inverse: bool,
) -> Result<Vec<S>, SciOxideError> {
    let plan = FftLogPlan::new(FftLogPlanSpec::new(
        input.len(),
        dln,
        mu,
        offset,
        Complex::new(bias, S::ZERO),
        direction(inverse),
    ))?;
    let values = complexified(input, scioxide_core::quantity!("fftlog_simple_input"))?;
    let mut output = try_filled(
        input.len(),
        Complex::new(S::ZERO, S::ZERO),
        scioxide_core::quantity!("fftlog_simple_output"),
    )?;
    plan.execute_into(
        &values,
        &mut output,
        &mut plan.workspace(())?,
        ExecutionContext::sequential_native(),
    )?;
    real_parts(&output, scioxide_core::quantity!("fftlog_real_output"))
}

/// Computes the fast Hankel transform of real samples on a logarithmic grid
/// (FFTLog).
///
/// The discrete transform approximates
///
/// ```text
/// A(k) = ∫₀^∞ a(r) J_μ(k r) k dr
/// ```
///
/// with input samples at `ln r_j = (j − j_c) dln` and output samples at
/// `ln k_j = offset + (j − j_c) dln`, where `j_c = (n − 1)/2` and `n` is the
/// input length. `dln` is the logarithmic spacing, `mu` the Bessel order `μ`,
/// `offset` the logarithm of `r_c k_c` for the central radius `r_c = 1`, and
/// `bias` the exponent of the power-law bias. The input is treated as periodic,
/// with no padding or tapering. For real input and real bias the result is
/// real, and the rounding error in the imaginary part is discarded.
///
/// The method costs two FFTs of length `n` after preparing `n` Mellin
/// coefficients. [`FftLogPlanSpec::new`] takes the same arguments in the same
/// order for repeated or complex transforms, and [`fftlog_low_ringing_offset`]
/// chooses an `offset` that reduces ringing.
///
/// [`fftlog_low_ringing_offset`]: crate::fftlog_low_ringing_offset
///
/// # Errors
///
/// - `InvalidInput` if `input` is empty or `dln` is not positive.
/// - `NonFinite` if a parameter or sample is NaN or infinite.
/// - `Singular` if a forward Mellin coefficient has a pole.
/// - `NumericalInstability` if a nonzero Mellin coefficient underflows to zero,
///   a coefficient or bias weight overflows, or a result is not representable.
///
/// # Examples
///
/// ```
/// use scioxide_fft::fht;
///
/// let a = [1.0_f64, -0.5, 2.0, 0.25, 0.75];
/// let transformed = fht(&a, 0.3, 0.5, 0.1, 0.2)?;
/// assert_eq!(transformed.len(), a.len());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. J. S. Hamilton, "Uncorrelated modes of the non-linear power
///   spectrum", *Mon. Not. R. Astron. Soc.* 312, 2000, 257–284.
/// - J. D. Talman, "Numerical Fourier and Bessel transforms in logarithmic
///   variables", *J. Comput. Phys.* 29, 1978, 35–48.
pub fn fht<S: RealScalar>(
    input: &[S],
    dln: S,
    mu: S,
    offset: S,
    bias: S,
) -> Result<Vec<S>, SciOxideError> {
    fftlog_real(input, dln, mu, offset, bias, false)
}

/// Computes the inverse of the discrete fast Hankel transform [`fht`].
///
/// The arguments have the same meaning and order as for [`fht`], with `input`
/// sampled on the `k` grid and the result on the `r` grid. The function applies
/// the exact inverse of the discrete map of [`fht`], so `ifht(fht(a))` returns
/// `a` up to rounding when no Mellin coefficient is zero. It returns the real
/// part of the result.
///
/// # Errors
///
/// - `InvalidInput` if `input` is empty or `dln` is not positive.
/// - `NonFinite` if a parameter or sample is NaN or infinite.
/// - `Singular` if a forward Mellin coefficient is zero, so that its inverse
///   is undefined.
/// - `NumericalInstability` if a nonzero Mellin coefficient underflows to zero,
///   a coefficient or bias weight overflows, or a result is not representable.
///
/// # Examples
///
/// ```
/// use scioxide_fft::{fht, ifht};
///
/// let a = [1.0_f64, -0.5, 2.0, 0.25, 0.75];
/// let transformed = fht(&a, 0.3, 0.5, 0.1, 0.2)?;
/// let recovered = ifht(&transformed, 0.3, 0.5, 0.1, 0.2)?;
/// for (&x, y) in recovered.iter().zip(a) {
///     assert!((x - y).abs() < 1e-11);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ifht<S: RealScalar>(
    input: &[S],
    dln: S,
    mu: S,
    offset: S,
    bias: S,
) -> Result<Vec<S>, SciOxideError> {
    fftlog_real(input, dln, mu, offset, bias, true)
}

// Shared real-input adapter for `dht` and `idht`. `order` and `extent` are
// checked here under their own names before the plan examines the zeros.
fn hankel_real<S: RealScalar>(
    input: &[S],
    zeros: &[S],
    order: S,
    extent: S,
    direction: FftDirection,
) -> Result<Vec<S>, SciOxideError> {
    let zeros_argument = scioxide_core::argument!("zeros");
    crate::validation::nonzero(zeros.len(), zeros_argument)?;
    let n = zeros.len() - 1;
    crate::validation::nonzero(n, zeros_argument)?;
    crate::validation::length(input.len(), n)?;
    if !order.is_finite() || !extent.is_finite() {
        return Err(crate::validation::nonfinite());
    }
    if order <= -S::HALF {
        return Err(invalid(
            scioxide_core::InvalidInputReason::OutOfRange,
            scioxide_core::argument!("order"),
            "finite order > -1/2 required",
        ));
    }
    if extent <= S::ZERO {
        return Err(invalid(
            scioxide_core::InvalidInputReason::OutOfRange,
            scioxide_core::argument!("extent"),
            "positive extent required",
        ));
    }
    let plan =
        DiscreteHankelPlan::new(zeros, DiscreteHankelPlanSpec::new(order, extent, direction))?;
    let values = complexified(input, scioxide_core::quantity!("dht_simple_input"))?;
    let mut output = try_filled(
        n,
        Complex::new(S::ZERO, S::ZERO),
        scioxide_core::quantity!("dht_simple_output"),
    )?;
    plan.execute_into(
        &values,
        &mut output,
        &mut plan.workspace(())?,
        ExecutionContext::sequential_native(),
    )?;
    // The transform matrix is real, so the imaginary parts are exactly zero.
    real_parts(&output, scioxide_core::quantity!("dht_real_output"))
}

/// Computes the discrete Hankel transform of order `ν` of real samples by
/// quadrature at the zeros of the Bessel function `J_ν`.
///
/// The transform approximates
///
/// ```text
/// F(k) = ∫₀^R f(r) J_ν(k r) r dr
/// ```
///
/// from samples `f(r_j)` at `r_j = R z_j / z_(n+1)` and returns `F(k_m)` at
/// `k_m = z_m / R`, for `1 ≤ j, m ≤ n`. Here `ν = order > −1/2`, `R = extent`,
/// and `z_1 < z_2 < …` are the positive zeros of `J_ν`. You supply the first
/// `n + 1` zeros in `zeros`, so `input` has `n = zeros.len() − 1` samples; they
/// are checked with the default tolerance of [`DiscreteHankelPlanSpec`]. The
/// quadrature matrix is real, so the result is real. [`idht`] applies the
/// inverse quadrature.
///
/// The transform costs `O(n²)` Bessel evaluations to prepare and `O(n²)`
/// operations to apply. Use [`DiscreteHankelPlan`] for complex samples,
/// another zero tolerance or limits, or to transform many inputs.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `zeros` has fewer than two entries.
/// - `DimensionMismatch` if `input.len()` is not `zeros.len() − 1`.
/// - `NonFinite` if `order`, `extent`, a zero or a sample is NaN or infinite.
/// - `InvalidInput` (`OutOfRange`) if `order ≤ −1/2` or `extent ≤ 0`, or if
///   `zeros` are not the first consecutive positive zeros of `J_ν` to within
///   the default tolerance.
/// - `NumericalInstability` if a node, frequency or result is not
///   representable, or a quadrature weight underflows.
///
/// # Examples
///
/// ```
/// use core::f64::consts::PI;
/// use scioxide_fft::dht;
///
/// // The positive zeros of J_{1/2} are kπ. Two zeros give one node, and the
/// // quadrature is F(k_1) = R² f(r_1) / (2π).
/// let zeros = [PI, 2.0 * PI];
/// let transform = dht(&[1.0_f64], &zeros, 0.5, 4.0)?;
/// assert!((transform[0] - 8.0 / PI).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// H. F. Johnson, "An improved method for computing a discrete Hankel
/// transform", *Comput. Phys. Commun.* 43(2), 1987, 181–202.
pub fn dht<S: RealScalar>(
    input: &[S],
    zeros: &[S],
    order: S,
    extent: S,
) -> Result<Vec<S>, SciOxideError> {
    hankel_real(input, zeros, order, extent, FftDirection::Forward)
}

/// Computes the inverse discrete Hankel transform of order `ν` of real samples.
///
/// With the zeros, order and extent of [`dht`], `input` holds samples at
/// `k_m = z_m / R` and the result values at `r_m = R z_m / z_(n+1)`, for
/// `1 ≤ m ≤ n`. The sum is that of [`dht`] with the factor `1 / R²` in place
/// of `(R / z_(n+1))²` (see [`DiscreteHankelPlan`]). Both directions are
/// quadratures, not exact inverses of each other, so `idht(dht(f))`
/// approximates `f` with the quadrature error.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `zeros` has fewer than two entries.
/// - `DimensionMismatch` if `input.len()` is not `zeros.len() − 1`.
/// - `NonFinite` if `order`, `extent`, a zero or a sample is NaN or infinite.
/// - `InvalidInput` (`OutOfRange`) if `order ≤ −1/2` or `extent ≤ 0`, or if
///   `zeros` are not the first consecutive positive zeros of `J_ν` to within
///   the default tolerance of [`DiscreteHankelPlanSpec`].
/// - `NumericalInstability` if a node, frequency or result is not
///   representable, or a quadrature weight underflows.
///
/// # Examples
///
/// ```
/// use core::f64::consts::PI;
/// use scioxide_fft::{dht, idht};
///
/// // For ν = 1/2 the zeros are kπ; seventeen zeros give sixteen nodes.
/// let zeros: Vec<f64> = (1..=17).map(|k| k as f64 * PI).collect();
/// let f: Vec<f64> = (0..16).map(|j| j as f64).collect();
/// let transform = dht(&f, &zeros, 0.5, 4.0)?;
/// let back = idht(&transform, &zeros, 0.5, 4.0)?;
/// assert!(back.iter().zip(&f).all(|(a, b)| (a - b).abs() < 1e-9));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn idht<S: RealScalar>(
    input: &[S],
    zeros: &[S],
    order: S,
    extent: S,
) -> Result<Vec<S>, SciOxideError> {
    hankel_real(input, zeros, order, extent, FftDirection::Inverse)
}

// Shared adapter for the NUFFT one-shots: `accuracy` is a parameter here, so it
// is checked and named before the spec is built, and packed input is rejected
// before preparation.
fn nufft<S: RealScalar>(
    transform: crate::NufftTransform<'_, '_, S>,
    input: FftInput<'_, S>,
    direction: FftDirection,
    normalization: Normalization,
    accuracy: S,
) -> Result<Vec<Complex<S>>, SciOxideError>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    crate::io::FullInput::try_from(input)?;
    if !accuracy.is_finite() {
        return Err(crate::validation::nonfinite());
    }
    if accuracy <= S::ZERO || accuracy >= S::ONE {
        return Err(invalid(
            scioxide_core::InvalidInputReason::OutOfRange,
            scioxide_core::argument!("accuracy"),
            "accuracy must be finite and strictly between zero and one",
        ));
    }
    let spec = NufftPlanSpec::new(direction, normalization, accuracy);
    NufftPlan::new(transform, spec)?.execute_owned(input)
}

/// Computes a type-1 nonuniform FFT: sums at nonuniform points evaluated on a
/// grid of integer modes.
///
/// ```text
/// f[k] = Σⱼ c_j exp(∓i k · x_j),   k ∈ K,
/// ```
///
/// where `x_j` are the rows of `points` (shape `[count, d]`), `c_j` the real or
/// complex entries of `input`, and `K` the integer modes with
/// `−⌊N_a/2⌋ ≤ k_a < ⌈N_a/2⌉` on each axis `a`, for `modes = [N_1, …, N_d]`.
/// The output lists `K` in row-major order with each axis in increasing `k`.
///
/// [`FftDirection::Forward`] selects the sign `−` and [`FftDirection::Inverse`]
/// the sign `+`; `normalization` scales by the source count; and each output
/// is within about `accuracy · Σⱼ |c_j|` of the exact sum. The grid is
/// oversampled by 2 and no resource limits apply; see [`NufftPlan`] for the
/// method and its cost, and for other oversampling ratios, limits, execution
/// policies and backends.
///
/// # Errors
///
/// - `InvalidInput` (`UnsupportedConfiguration`) if `input` is packed
///   ([`FftInput::Hermitian`]).
/// - `InvalidInput` if `accuracy` is not strictly between zero and one or a
///   mode count is zero.
/// - `DimensionMismatch` if `points` is not of rank two, `modes.len()` differs
///   from the point dimension, or `input.len()` from the number of points.
/// - `NonFinite` if `accuracy`, a coordinate or an input value is NaN or
///   infinite.
/// - `NumericalInstability` if `accuracy` is below the rounding error the
///   method can reach.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, NdView, StridedLayout};
/// use scioxide_fft::{FftDirection, Normalization, nufft_type1};
///
/// let x = [-1.5_f64, -0.25, 0.7, 1.3];
/// let points = NdView::new(&x, StridedLayout::new(&[4, 1], &[1, 1])?)?;
/// let c = [1.0_f64; 4];
/// let values = nufft_type1(points, &c, &[5], FftDirection::Forward, Normalization::Unscaled, 1e-8)?;
/// // Modes are −2, …, 2; the k = 0 entry is Σ c_j = 4.
/// assert!((values[2] - Complex::new(4.0, 0.0)).norm() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn nufft_type1<'i, S: RealScalar>(
    points: NdView<'_, '_, S>,
    input: impl Into<FftInput<'i, S>>,
    modes: &[usize],
    direction: FftDirection,
    normalization: Normalization,
    accuracy: S,
) -> Result<Vec<Complex<S>>, SciOxideError>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    let transform = crate::NufftTransform::Type1 { points, modes };
    nufft(transform, input.into(), direction, normalization, accuracy)
}

/// Computes a type-2 nonuniform FFT: a Fourier series on a grid of integer
/// modes evaluated at nonuniform points.
///
/// ```text
/// c_j = Σ_{k ∈ K} f[k] exp(∓i k · x_j),
/// ```
///
/// where `x_j` are the rows of `points` (shape `[count, d]`), `f` holds the
/// real or complex coefficients in `input`, and `K` and its row-major order
/// are as in [`nufft_type1`]. This evaluates the sum; it does not solve the
/// inverse problem of type 1.
///
/// [`FftDirection::Forward`] selects the sign `−` and [`FftDirection::Inverse`]
/// the sign `+`; `normalization` scales by the mode count; and each output is
/// within about `accuracy · Σ_k |f[k]|` of the exact sum. The grid is
/// oversampled by 2 and no resource limits apply; see [`NufftPlan`] for the
/// method and its cost, and for other settings.
///
/// # Errors
///
/// - `InvalidInput` (`UnsupportedConfiguration`) if `input` is packed
///   ([`FftInput::Hermitian`]).
/// - `InvalidInput` if `accuracy` is not strictly between zero and one or a
///   mode count is zero.
/// - `DimensionMismatch` if `points` is not of rank two, `modes.len()` differs
///   from the point dimension, or `input.len()` from the product of `modes`.
/// - `NonFinite` if `accuracy`, a coordinate or an input value is NaN or
///   infinite.
/// - `NumericalInstability` if `accuracy` is below the rounding error the
///   method can reach.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, NdView, StridedLayout};
/// use scioxide_fft::{FftDirection, Normalization, nufft_type2};
///
/// let x = [-1.5_f64, -0.25, 0.7, 1.3];
/// let points = NdView::new(&x, StridedLayout::new(&[4, 1], &[1, 1])?)?;
/// let modes = [Complex::new(1.0, 0.0); 5];
/// let (direction, norm) = (FftDirection::Forward, Normalization::Unscaled);
/// let values = nufft_type2(points, &modes, &[5], direction, norm, 1e-8)?;
/// // Unit coefficients on k = −2, …, 2 sum to 1 + 2 cos x + 2 cos 2x.
/// let exact = 1.0 + 2.0 * x[2].cos() + 2.0 * (2.0 * x[2]).cos();
/// assert!((values[2] - Complex::new(exact, 0.0)).norm() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn nufft_type2<'i, S: RealScalar>(
    points: NdView<'_, '_, S>,
    input: impl Into<FftInput<'i, S>>,
    modes: &[usize],
    direction: FftDirection,
    normalization: Normalization,
    accuracy: S,
) -> Result<Vec<Complex<S>>, SciOxideError>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    let transform = crate::NufftTransform::Type2 { points, modes };
    nufft(transform, input.into(), direction, normalization, accuracy)
}

/// Computes a type-3 nonuniform FFT: sums at nonuniform points evaluated at
/// nonuniform angular frequencies.
///
/// ```text
/// f_k = Σⱼ c_j exp(∓i s_k · x_j),
/// ```
///
/// where `x_j` are the rows of `points` (shape `[count, d]`), `s_k` the rows of
/// `frequencies` (shape `[targets, d]`, in radians per unit of `x`) and `c_j`
/// the real or complex entries of `input`.
///
/// [`FftDirection::Forward`] selects the sign `−` and [`FftDirection::Inverse`]
/// the sign `+`; `normalization` scales by the source count; and each output
/// is within about `accuracy · Σⱼ |c_j|` of the exact sum. The grids are
/// oversampled by 2 and no resource limits apply; see [`NufftPlan`] for the
/// method and its cost, and for other settings.
///
/// # Errors
///
/// - `InvalidInput` (`UnsupportedConfiguration`) if `input` is packed
///   ([`FftInput::Hermitian`]).
/// - `InvalidInput` if `accuracy` is not strictly between zero and one.
/// - `DimensionMismatch` if `points` or `frequencies` is not of rank two, their
///   dimensions differ, or `input.len()` differs from the number of points.
/// - `NonFinite` if `accuracy`, a coordinate, a frequency or an input value is
///   NaN or infinite.
/// - `NumericalInstability` if `accuracy` is below the rounding error the
///   method can reach.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, NdView, StridedLayout};
/// use scioxide_fft::{FftDirection, Normalization, nufft_type3};
///
/// let x = [-1.5_f64, -0.25, 0.7, 1.3];
/// let omega = [0.0_f64, -0.8, 1.1];
/// let points = NdView::new(&x, StridedLayout::new(&[4, 1], &[1, 1])?)?;
/// let frequencies = NdView::new(&omega, StridedLayout::new(&[3, 1], &[1, 1])?)?;
/// let c = [Complex::new(1.0, 0.0); 4];
/// let (direction, norm) = (FftDirection::Forward, Normalization::Unscaled);
/// let values = nufft_type3(points, &c, frequencies, direction, norm, 1e-8)?;
/// // At frequency zero every term is c_j = 1.
/// assert!((values[0] - Complex::new(4.0, 0.0)).norm() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn nufft_type3<'i, S: RealScalar>(
    points: NdView<'_, '_, S>,
    input: impl Into<FftInput<'i, S>>,
    frequencies: NdView<'_, '_, S>,
    direction: FftDirection,
    normalization: Normalization,
    accuracy: S,
) -> Result<Vec<Complex<S>>, SciOxideError>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    let transform = crate::NufftTransform::Type3 {
        points,
        frequencies,
    };
    nufft(transform, input.into(), direction, normalization, accuracy)
}

/// Computes the forward DFT of a Hermitian-symmetric sequence, which is real.
///
/// `input` holds the first `⌊n/2⌋ + 1` samples `h[0], …, h[⌊n/2⌋]` of a
/// sequence of logical length `n` with `h[n − j] = conj(h[j])`; the length `n`,
/// odd or even, is stored in the [`HermitianSpectrum`]. The result is
///
/// ```text
/// y[k] = Σⱼ h[j] exp(−2πi jk/n),   0 ≤ k < n,
/// ```
///
/// scaled as described in [Normalization](crate#normalization).
/// [`ihfft`] is the inverse. Use an [`FftPlan`](crate::FftPlan) with
/// [`FftLayout::Hermitian`] and [`FftDirection::Forward`] to transform many
/// inputs.
///
/// # Errors
///
/// - `InvalidInput` if `h[0]` or, for even `n`, `h[n/2]` has an imaginary part
///   larger than the [Hermitian input](crate#hermitian-input) tolerance.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{HermitianSpectrum, Normalization, hfft};
///
/// // h = [1, 2 + i, −0.5, −0.5, 2 − i]; y[0] is the sum of h, 4.
/// let h = vec![
///     Complex::new(1.0_f64, 0.0),
///     Complex::new(2.0, 1.0),
///     Complex::new(-0.5, 0.0),
/// ];
/// let samples = HermitianSpectrum::from_vec(5, h)?;
/// let spectrum = hfft(&samples, Normalization::Inverse)?;
/// assert_eq!(spectrum.len(), 5);
/// assert!((spectrum[0] - 4.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn hfft<S>(
    input: &HermitianSpectrum<S>,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError>
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    bins_to_samples(
        input,
        FftDirection::Forward,
        normalization,
        FftLayout::Hermitian,
    )
}

/// Computes the inverse DFT of a real sequence and returns its Hermitian half.
///
/// For real `y` of length `n`, the inverse DFT
///
/// ```text
/// h[j] = Σₖ y[k] exp(+2πi jk/n),   0 ≤ j < n,
/// ```
///
/// scaled as described in [Normalization](crate#normalization), satisfies
/// `h[n − j] = conj(h[j])`. The result keeps `h[0], …, h[⌊n/2⌋]` and the
/// logical length `n`, so [`hfft`] inverts it for odd and even `n`. Use an
/// [`FftPlan`](crate::FftPlan) with [`FftLayout::Hermitian`] and
/// [`FftDirection::Inverse`] to transform many inputs.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `input` is empty.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// ```
/// use scioxide_fft::{Normalization, ihfft};
///
/// let spectrum = [1.0_f64, 2.0, -0.5, 0.75, 1.25];
/// let samples = ihfft(&spectrum, Normalization::Inverse)?;
/// assert_eq!((samples.logical_len(), samples.len()), (5, 3));
/// // With inverse normalization, h[0] is the mean of the spectrum, 0.9.
/// assert!((samples.bins()[0].re - 0.9).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ihfft<S>(
    input: &[S],
    normalization: Normalization,
) -> Result<HermitianSpectrum<S>, SciOxideError>
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    samples_to_bins(
        input,
        FftDirection::Inverse,
        normalization,
        FftLayout::Hermitian,
    )
}

/// Computes the forward discrete Fourier transform of a complex sequence.
///
/// ```text
/// X[k] = Σⱼ x[j] exp(−2πi jk/n),   0 ≤ k < n,
/// ```
///
/// where `n = input.len()`, scaled as described in
/// [Normalization](crate#normalization). The result has length `n`, in the
/// frequency order of [`fftfreq`](crate::fftfreq). Real samples use
/// [`rfft`].
///
/// Every length costs `O(n log n)`. Lengths whose prime factors are 2, 3, 5
/// and 7 use mixed-radix Cooley–Tukey stages. Other lengths use Rader's
/// algorithm for suitable primes and Bluestein's algorithm otherwise.
/// [`next_fast_len`] gives the next 2/3/5/7-smooth length. Use an [`FftPlan`]
/// to transform many sequences of the same length.
///
/// [`next_fast_len`]: crate::next_fast_len
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `input` is empty.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// The DFT of a constant sequence is concentrated in `X[0]`, the sum of the
/// samples, which inverse normalization leaves unscaled.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{Normalization, fft};
///
/// let input = [Complex::new(1.0_f64, 0.0); 4];
/// let spectrum = fft(&input, Normalization::Inverse)?;
/// assert_eq!(spectrum.len(), input.len());
/// assert!((spectrum[0] - Complex::new(4.0, 0.0)).norm() < 1e-12);
/// assert!(spectrum[1..].iter().all(|z| z.norm() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation
///   of complex Fourier series", *Math. Comp.* 19, 1965, 297–301.
/// - C. M. Rader, "Discrete Fourier transforms when the number of data
///   samples is prime", *Proc. IEEE* 56(6), 1968, 1107–1108.
/// - L. I. Bluestein, "A linear filtering approach to the computation of
///   discrete Fourier transform", *IEEE Trans. Audio Electroacoust.* 18(4),
///   1970, 451–455.
pub fn fft<S: scioxide_core::RealScalar>(
    input: &[Complex<S>],
    normalization: Normalization,
) -> Result<Vec<Complex<S>>, SciOxideError> {
    complex_transform(input, false, normalization)
}

/// Computes the inverse discrete Fourier transform of a complex sequence.
///
/// ```text
/// x[j] = Σₖ X[k] exp(+2πi jk/n),   0 ≤ j < n,
/// ```
///
/// where `n = input.len()`, scaled as described in
/// [Normalization](crate#normalization). With the same normalization other
/// than `Unscaled`, `ifft(fft(x))` returns `x` up to rounding. The method and
/// cost are those of [`fft`].
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `input` is empty.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// Use the same normalization for both directions to recover the input.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{Normalization, fft, ifft};
///
/// let input = [Complex::new(1.0_f64, 2.0), Complex::new(3.0, -1.0)];
/// let spectrum = fft(&input, Normalization::Inverse)?;
/// let recovered = ifft(&spectrum, Normalization::Inverse)?;
/// for (a, b) in recovered.iter().zip(input) {
///     assert!((*a - b).norm() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ifft<S: scioxide_core::RealScalar>(
    input: &[Complex<S>],
    normalization: Normalization,
) -> Result<Vec<Complex<S>>, SciOxideError> {
    complex_transform(input, true, normalization)
}

fn complex_transform<S: scioxide_core::RealScalar>(
    input: &[Complex<S>],
    inverse: bool,
    normalization: Normalization,
) -> Result<Vec<Complex<S>>, SciOxideError> {
    let plan = FftPlan::<S>::build(
        FftPlanSpec::new(
            input.len(),
            direction(inverse),
            normalization,
            FftLayout::Complex,
        ),
        crate::FftPlanningContext::default(),
    )?;
    plan.execute_owned(input, scioxide_core::ExecutionContext::sequential_native())
}

// Allocating helpers below report their own buffers.
fn complex_nd<S: RealScalar>(
    input: &[Complex<S>],
    shape: &[usize],
    axes: &[usize],
    normalization: Normalization,
    direction: FftDirection,
) -> Result<Vec<Complex<S>>, SciOxideError> {
    crate::validation::length(input.len(), Shape::new(shape)?.len())?;
    let spec = FftNdPlanSpec::new(shape, axes, direction, normalization, FftLayout::Complex)?;
    let plan = FftNdPlan::new(spec)?;
    let mut output = try_filled(
        input.len(),
        Complex::new(S::ZERO, S::ZERO),
        scioxide_core::quantity!("n_d_fft_output"),
    )?;
    let layout = plan.dense_layout()?;
    plan.execute_into(
        NdView::new(input, layout)?,
        NdViewMut::new(&mut output, layout)?,
        &mut plan.workspace(())?,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
/// Computes the forward DFT of a complex array along the selected axes.
///
/// The transform is the one-dimensional [`fft`] applied along each axis in
/// `axes`. Unselected axes index independent transforms. `input` holds an array
/// of shape `shape` in row-major order (last axis fastest), and the result is a
/// new array of the same shape and order. Normalization uses the product of the
/// selected extents; see [Normalization](crate#normalization). Use an
/// [`FftNdPlan`] for strided arrays or to transform many arrays of the same
/// shape.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` (`Inconsistent`) if an axis appears twice.
/// - `InvalidInput` (`Empty`) if a selected axis has extent zero.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// Transform along the rows of a 2 × 3 array; each row is a separate
/// transform.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{Normalization, fftn};
///
/// let input = [Complex::new(1.0_f64, 0.0); 6];
/// let spectrum = fftn(&input, &[2, 3], &[1], Normalization::Inverse)?;
/// for row in spectrum.chunks_exact(3) {
///     assert!((row[0] - Complex::new(3.0, 0.0)).norm() < 1e-12);
///     assert!(row[1..].iter().all(|z| z.norm() < 1e-12));
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn fftn<S: RealScalar>(
    input: &[Complex<S>],
    shape: &[usize],
    axes: &[usize],
    normalization: Normalization,
) -> Result<Vec<Complex<S>>, SciOxideError> {
    complex_nd(input, shape, axes, normalization, FftDirection::Forward)
}
/// Computes the inverse DFT of a complex array along the selected axes.
///
/// The transform is the one-dimensional [`ifft`] applied along each axis in
/// `axes`, with the conventions of [`fftn`]: `input` and the result are arrays
/// of shape `shape` in row-major order.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` (`Inconsistent`) if an axis appears twice.
/// - `InvalidInput` (`Empty`) if a selected axis has extent zero.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// Inverse normalization divides by the product of the selected extents, so
/// the round trip returns the input.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{Normalization, fftn, ifftn};
///
/// let input = [Complex::new(1.0_f64, 0.5); 6];
/// let norm = Normalization::Inverse;
/// let spectrum = fftn(&input, &[2, 3], &[0, 1], norm)?;
/// let recovered = ifftn(&spectrum, &[2, 3], &[0, 1], norm)?;
/// for (a, b) in recovered.iter().zip(input) {
///     assert!((*a - b).norm() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ifftn<S: RealScalar>(
    input: &[Complex<S>],
    shape: &[usize],
    axes: &[usize],
    normalization: Normalization,
) -> Result<Vec<Complex<S>>, SciOxideError> {
    complex_nd(input, shape, axes, normalization, FftDirection::Inverse)
}

fn real_nd<S: RealScalar>(
    input: &[S],
    shape: &[usize],
    axes: &[usize],
    normalization: Normalization,
    layout: FftLayout,
    direction: FftDirection,
) -> Result<HermitianSpectrumNd<S>, SciOxideError> {
    crate::validation::length(input.len(), Shape::new(shape)?.len())?;
    let plan = FftNdPlan::new(FftNdPlanSpec::new(
        shape,
        axes,
        direction,
        normalization,
        layout,
    )?)?;
    let packed_shape = plan.packed_shape();
    let mut bins = try_filled(
        scioxide_core::Shape::new(packed_shape)?.len(),
        Complex::new(S::ZERO, S::ZERO),
        scioxide_core::quantity!("n_d_packed_output"),
    )?;
    plan.execute_into(
        NdView::new(input, plan.dense_layout()?)?,
        NdViewMut::new(&mut bins, plan.packed_layout()?)?,
        &mut plan.workspace(())?,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    HermitianSpectrumNd::from_vec(shape, axes, bins)
}
fn packed_nd<S: RealScalar>(
    input: &HermitianSpectrumNd<S>,
    normalization: Normalization,
    layout: FftLayout,
    direction: FftDirection,
) -> Result<Vec<S>, SciOxideError> {
    let plan = FftNdPlan::new(FftNdPlanSpec::new(
        input.shape(),
        input.axes(),
        direction,
        normalization,
        layout,
    )?)?;
    let mut output = try_filled(
        scioxide_core::Shape::new(plan.spec().shape())?.len(),
        S::ZERO,
        scioxide_core::quantity!("n_d_real_output"),
    )?;
    plan.execute_into(
        NdView::new(input.bins(), plan.packed_layout()?)?,
        NdViewMut::new(&mut output, plan.dense_layout()?)?,
        &mut plan.workspace(())?,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
/// Computes the forward DFT of a real array along the selected axes and keeps
/// the nonredundant half.
///
/// `input` holds a real array of shape `shape` in row-major order (last axis
/// fastest). The result is [`fftn`] of the input with the last axis in `axes`
/// cut to its first `⌊n/2⌋ + 1` bins, where `n` is that axis's extent; the
/// other bins follow from Hermitian symmetry. The returned
/// [`HermitianSpectrumNd`] stores the bins in row-major order together with the
/// logical shape and the axes, which [`irfftn`] uses to invert the transform.
/// Use an [`FftNdPlan`] for strided arrays.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`.
/// - `InvalidInput` (`Empty`) if `axes` is empty or a selected axis has extent
///   zero.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` (`Inconsistent`) if an axis appears twice.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// The last selected axis is packed: a logical shape of [2, 3] becomes [2, 2].
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{Normalization, rfftn};
///
/// let spectrum = rfftn(&[1.0_f64; 6], &[2, 3], &[0, 1], Normalization::Inverse)?;
/// assert_eq!(spectrum.shape(), &[2, 3]); // Logical shape is preserved.
/// assert_eq!(spectrum.bins().len(), 4); // Packed shape is [2, 2].
/// assert!((spectrum.bins()[0] - Complex::new(6.0, 0.0)).norm() < 1e-12);
/// assert!(spectrum.bins()[1..].iter().all(|z| z.norm() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn rfftn<S: RealScalar>(
    input: &[S],
    shape: &[usize],
    axes: &[usize],
    normalization: Normalization,
) -> Result<HermitianSpectrumNd<S>, SciOxideError> {
    real_nd(
        input,
        shape,
        axes,
        normalization,
        FftLayout::Real,
        FftDirection::Forward,
    )
}
/// Computes the real inverse DFT of a Hermitian-symmetric array given by its
/// nonredundant half.
///
/// This inverts [`rfftn`]: `input` holds the packed bins, the logical shape
/// and the selected axes, and the result is the real array of the logical
/// shape in row-major order. The bins must be [Hermitian](crate#hermitian-input)
/// to within rounding. Use [`FftNdPlan::execute_into`] for strided bins.
///
/// # Errors
///
/// - `InvalidInput` if the zero-frequency or Nyquist planes of the packed
///   axis are not Hermitian to within the
///   [Hermitian input](crate#hermitian-input) tolerance.
/// - `NonFinite` if a bin is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// ```
/// use scioxide_fft::{Normalization, irfftn, rfftn};
///
/// let input = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let norm = Normalization::Inverse;
/// let spectrum = rfftn(&input, &[2, 3], &[0, 1], norm)?;
/// let recovered = irfftn(&spectrum, norm)?;
/// assert_eq!(recovered.len(), input.len());
/// for (a, b) in recovered.iter().zip(input) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn irfftn<S: RealScalar>(
    input: &HermitianSpectrumNd<S>,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    packed_nd(input, normalization, FftLayout::Real, FftDirection::Inverse)
}
/// Computes the forward DFT of a Hermitian-symmetric array given by its
/// nonredundant half; the result is real.
///
/// This is the N-D form of [`hfft`]: `input` holds the packed samples, the
/// logical shape and the selected axes, with the last selected axis cut to
/// `⌊n/2⌋ + 1` entries. The result is the real array of the logical shape in
/// row-major order. The samples must be [Hermitian](crate#hermitian-input) to
/// within rounding. [`ihfftn`] is the inverse. Use [`FftNdPlan::execute_into`]
/// for strided samples.
///
/// # Errors
///
/// - `InvalidInput` if the zero-frequency or Nyquist planes of the packed
///   axis are not Hermitian to within the
///   [Hermitian input](crate#hermitian-input) tolerance.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// A Hermitian impulse at the origin transforms to a constant real spectrum.
/// Inverse normalization leaves this forward transform unscaled.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{HermitianSpectrumNd, Normalization, hfftn};
///
/// let zero = Complex::new(0.0, 0.0);
/// let bins = vec![Complex::new(1.0_f64, 0.0), zero, zero, zero];
/// let samples = HermitianSpectrumNd::from_vec(&[2, 3], &[0, 1], bins)?;
/// let spectrum = hfftn(&samples, Normalization::Inverse)?;
/// assert_eq!(spectrum.len(), 6);
/// assert!(spectrum.iter().all(|x| (x - 1.0).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn hfftn<S: RealScalar>(
    input: &HermitianSpectrumNd<S>,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError> {
    packed_nd(
        input,
        normalization,
        FftLayout::Hermitian,
        FftDirection::Forward,
    )
}
/// Computes the inverse DFT of a real array along the selected axes and keeps
/// the nonredundant half.
///
/// This is the N-D form of [`ihfft`]: `input` holds a real array of shape
/// `shape` in row-major order, and the result is [`ifftn`] of the input with
/// the last axis in `axes` cut to its first `⌊n/2⌋ + 1` entries. The returned
/// [`HermitianSpectrumNd`] stores them in row-major order together with the
/// logical shape and the axes, which [`hfftn`] uses to invert the transform.
/// Use an [`FftNdPlan`] for strided arrays.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`.
/// - `InvalidInput` (`Empty`) if `axes` is empty or a selected axis has extent
///   zero.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` (`Inconsistent`) if an axis appears twice.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// A constant real spectrum produces a packed Hermitian impulse. Only the last
/// selected axis is packed, and inverse normalization divides by 6.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{Normalization, ihfftn};
///
/// let samples = ihfftn(&[1.0_f64; 6], &[2, 3], &[0, 1], Normalization::Inverse)?;
/// assert_eq!(samples.shape(), &[2, 3]);
/// assert_eq!(samples.bins().len(), 4); // Packed shape is [2, 2].
/// assert!((samples.bins()[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
/// assert!(samples.bins()[1..].iter().all(|z| z.norm() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ihfftn<S: RealScalar>(
    input: &[S],
    shape: &[usize],
    axes: &[usize],
    normalization: Normalization,
) -> Result<HermitianSpectrumNd<S>, SciOxideError> {
    real_nd(
        input,
        shape,
        axes,
        normalization,
        FftLayout::Hermitian,
        FftDirection::Inverse,
    )
}

fn shift_nd<T: Copy>(
    input: &[T],
    shape: &[usize],
    axes: &[usize],
    inverse: bool,
) -> Result<Vec<T>, SciOxideError> {
    let checked = Shape::new(shape)?;
    crate::validation::length(input.len(), checked.len())?;
    // Shifts are permutations, so selected empty axes are valid.
    for (i, &axis) in axes.iter().enumerate() {
        checked.extent(axis)?;
        if axes[..i].contains(&axis) {
            return Err(invalid(
                scioxide_core::InvalidInputReason::Inconsistent,
                scioxide_core::argument!("axes"),
                "shift axes must be distinct",
            ));
        }
    }
    let mut output = try_with_capacity(input.len(), scioxide_core::quantity!("n_d_shift_output"))?;
    // Output coordinate c reads source (c + shift) mod n, i.e. a left rotation
    // by ceil(n/2) (fftshift) or floor(n/2) (ifftshift), matching the 1-D
    // helpers; the two differ only for odd n, which is what makes them mutual
    // inverses. The source offset accumulates the row-major strides from the
    // last axis; every partial product divides the element count.
    let source = |index: usize| {
        let (offset, _, _) = shape.iter().enumerate().rev().fold(
            (0, 1, index),
            |(offset, stride, remaining), (axis, &n)| {
                let coordinate = remaining % n;
                let shifted = if axes.contains(&axis) {
                    let shift = if inverse { n / 2 } else { n / 2 + n % 2 };
                    if coordinate >= n - shift {
                        coordinate - (n - shift)
                    } else {
                        coordinate + shift
                    }
                } else {
                    coordinate
                };
                (offset + shifted * stride, stride * n, remaining / n)
            },
        );
        input[offset]
    };
    output.extend((0..input.len()).map(source));
    Ok(output)
}
/// Moves the zero-frequency entry to the center of each selected axis.
///
/// Along each axis in `axes` with extent `n`, the entries are rotated left by
/// `⌈n/2⌉`, so the zero-frequency entry moves to index `⌊n/2⌋`. This is
/// [`fftshift`](crate::fftshift) applied along several axes of an array.
/// `input` holds an array of shape `shape` in row-major order (last axis
/// fastest), and the result is a new array of the same shape and order. Copy a
/// strided array to a dense one first. Axes of extent zero are allowed.
/// [`ifftshift_nd`] is the inverse.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` (`Inconsistent`) if an axis appears twice.
///
/// # Examples
///
/// Shift only the second axis, leaving the rows independent.
///
/// ```
/// use scioxide_fft::fftshift_nd;
///
/// let output = fftshift_nd(&[0, 1, 2, 3, 4, 5], &[2, 3], &[1])?;
/// assert_eq!(output, [2, 0, 1, 5, 3, 4]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn fftshift_nd<T: Copy>(
    input: &[T],
    shape: &[usize],
    axes: &[usize],
) -> Result<Vec<T>, SciOxideError> {
    shift_nd(input, shape, axes, false)
}
/// Moves the center entry of each selected axis back to index zero.
///
/// Along each axis in `axes` with extent `n`, the entries are rotated left by
/// `⌊n/2⌋`. This inverts [`fftshift_nd`]; the two differ for odd `n`. `input`
/// and the result are arrays of shape `shape` in row-major order. Axes of
/// extent zero are allowed.
///
/// # Errors
///
/// - `SizeOverflow` if the product of `shape` overflows `usize`.
/// - `DimensionMismatch` if `input.len()` is not the product of `shape`.
/// - `OutOfBounds` if an axis is not less than `shape.len()`.
/// - `InvalidInput` (`Inconsistent`) if an axis appears twice.
///
/// # Examples
///
/// Undo the shift along the second axis of the [`fftshift_nd`] example.
///
/// ```
/// use scioxide_fft::ifftshift_nd;
///
/// let output = ifftshift_nd(&[2, 0, 1, 5, 3, 4], &[2, 3], &[1])?;
/// assert_eq!(output, [0, 1, 2, 3, 4, 5]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ifftshift_nd<T: Copy>(
    input: &[T],
    shape: &[usize],
    axes: &[usize],
) -> Result<Vec<T>, SciOxideError> {
    shift_nd(input, shape, axes, true)
}

/// Computes the forward DFT of a real sequence and keeps the bins of
/// nonnegative frequency.
///
/// For real `x` of length `n` the DFT `X[k] = Σⱼ x[j] exp(−2πi jk/n)` satisfies
/// `X[n − k] = conj(X[k])`, so the result keeps `X[0], …, X[⌊n/2⌋]` together
/// with the length `n`, odd or even. Normalization is as described in
/// [Normalization](crate#normalization); packing does not change the scale.
/// [`irfft`] is the inverse. Use an [`FftPlan`] with [`FftLayout::Real`] to
/// transform many sequences of the same length.
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `input` is empty.
/// - `NonFinite` if an input value is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// The spectrum of a constant sequence of odd length 5 has three bins, and
/// only the zero-frequency bin is nonzero.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{Normalization, rfft};
///
/// let spectrum = rfft(&[1.0_f64; 5], Normalization::Inverse)?;
/// assert_eq!(spectrum.logical_len(), 5);
/// assert_eq!(spectrum.bins().len(), 3);
/// assert!((spectrum.bins()[0] - Complex::new(5.0, 0.0)).norm() < 1e-12);
/// assert!(spectrum.bins()[1..].iter().all(|z| z.norm() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn rfft<S>(
    input: &[S],
    normalization: Normalization,
) -> Result<HermitianSpectrum<S>, SciOxideError>
where
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    samples_to_bins(input, FftDirection::Forward, normalization, FftLayout::Real)
}

/// Computes the real inverse DFT of a sequence given by its bins of
/// nonnegative frequency.
///
/// `input` holds `X[0], …, X[⌊n/2⌋]` and the length `n` of the result; the
/// other bins are `X[n − k] = conj(X[k])`. The result is
/// `x[j] = Σₖ X[k] exp(+2πi jk/n)` for `0 ≤ j < n`, scaled as described in
/// [Normalization](crate#normalization). `X[0]` and, for even `n`, `X[n/2]`
/// must be real to within the [Hermitian input](crate#hermitian-input)
/// tolerance; their imaginary parts are then set to zero.
///
/// # Errors
///
/// - `InvalidInput` if `X[0]` or, for even `n`, `X[n/2]` has an imaginary part
///   above the [Hermitian input](crate#hermitian-input) tolerance.
/// - `NonFinite` if a bin is NaN or infinite.
/// - `NumericalInstability` (`Overflow`) if a result is not representable.
///
/// # Examples
///
/// The spectrum stores the length 5, so `irfft` returns five samples.
///
/// ```
/// use scioxide_fft::{Normalization, irfft, rfft};
///
/// let input = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let spectrum = rfft(&input, Normalization::Inverse)?;
/// let recovered = irfft(&spectrum, Normalization::Inverse)?;
/// assert_eq!(recovered.len(), input.len());
/// for (a, b) in recovered.iter().zip(input) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn irfft<S>(
    input: &HermitianSpectrum<S>,
    normalization: Normalization,
) -> Result<Vec<S>, SciOxideError>
where
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    bins_to_samples(input, FftDirection::Inverse, normalization, FftLayout::Real)
}

/// Runs a one-off 1-D plan from real samples to packed bins
/// (`rfft` or `ihfft`, according to `direction` and `layout`).
fn samples_to_bins<S>(
    input: &[S],
    direction: FftDirection,
    normalization: Normalization,
    layout: FftLayout,
) -> Result<HermitianSpectrum<S>, SciOxideError>
where
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let plan = FftPlan::<S>::build(
        FftPlanSpec::new(input.len(), direction, normalization, layout),
        crate::FftPlanningContext::default(),
    )?;
    let mut output = HermitianSpectrum::zeros(input.len())?;
    plan.execute_into(
        input,
        &mut output,
        &mut plan.workspace(())?,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}

/// Runs a one-off 1-D plan from packed bins to real samples
/// (`irfft` or `hfft`, according to `direction` and `layout`).
fn bins_to_samples<S>(
    input: &HermitianSpectrum<S>,
    direction: FftDirection,
    normalization: Normalization,
    layout: FftLayout,
) -> Result<Vec<S>, SciOxideError>
where
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let n = input.logical_len();
    let plan = FftPlan::<S>::build(
        FftPlanSpec::new(n, direction, normalization, layout),
        crate::FftPlanningContext::default(),
    )?;
    let mut output = try_filled(
        n,
        <S as scioxide_core::AlgebraicScalar>::ZERO,
        scioxide_core::quantity!("real_fft_samples"),
    )?;
    plan.execute_into(
        input,
        &mut output,
        &mut plan.workspace(())?,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}

#[cfg(test)]
mod complex_tests {
    use super::*;
    use crate::test_support::{assert_complex_close, component_max};

    fn c<S: scioxide_core::RealScalar>(re: f64, im: f64) -> Complex<S> {
        Complex::new(
            S::try_from_f64_lossy(re).unwrap(),
            S::try_from_f64_lossy(im).unwrap(),
        )
    }

    #[test]
    fn test_fft_empty() {
        let error = fft::<f64>(&[], Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_ifft_empty() {
        let error = ifft::<f64>(&[], Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn test_fft_non_finite_input() {
        let input = [Complex::new(1.0_f64, f64::NAN)];
        let error = fft(&input, Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn test_ifft_non_finite_input() {
        let input = [Complex::new(f64::INFINITY, 0.0)];
        let error = ifft(&input, Normalization::Inverse).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn test_fft_power_of_two_input() {
        let input: Vec<Complex<f64>> = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let result = fft(&input, Normalization::Inverse).unwrap();
        assert_eq!(result.len(), 4);
        let expected: Vec<Complex<f64>> =
            vec![c(10.0, 0.0), c(-2.0, 2.0), c(-2.0, 0.0), c(-2.0, -2.0)];
        assert_complex_close(&result, &expected, component_max, |_| 1e-12);
    }

    #[test]
    fn test_fft_non_power_of_two_input() {
        let input: Vec<Complex<f64>> = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        let result = fft(&input, Normalization::Inverse).unwrap();
        assert_eq!(result.len(), 3);
        let root_three_over_two = 3.0_f64.sqrt() / 2.0;
        let expected: Vec<Complex<f64>> = vec![
            c(6.0, 0.0),
            c(-1.5, root_three_over_two),
            c(-1.5, -root_three_over_two),
        ];
        assert_complex_close(&result, &expected, component_max, |_| 1e-12);
    }

    #[test]
    fn test_fft_single_element() {
        let input: Vec<Complex<f64>> = vec![c(42.0, -7.0)];
        let result = fft(&input, Normalization::Inverse).unwrap();
        assert_eq!(result.len(), 1);
        assert_complex_close(&result, &input, component_max, |_| 1e-14);
    }
}
