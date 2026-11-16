//! Input and output types and packed Hermitian spectra.
use crate::tensor::{copy_metadata, validate_axes};
use crate::validation::invalid;
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::{Argument, Complex, SciOxideError};
use scioxide_core::{RealScalar, Shape};
/// Stores the first half of a Hermitian sequence together with its full length.
///
/// A sequence `X` of length `n` is Hermitian if `X[n − k] = conj(X[k])`; the DFT
/// of a real signal is an example. The bins `X[0], …, X[⌊n/2⌋]` determine it.
/// Lengths `2m` and `2m + 1` both pack into `m + 1` bins, so the spectrum keeps
/// the logical length `n` to select the inverse transform. [`rfft`](crate::rfft) and
/// [`ihfft`](crate::ihfft) return this type, and [`irfft`](crate::irfft) and
/// [`hfft`](crate::hfft) take it. See [Conventions](crate#conventions).
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_fft::{HermitianSpectrum, Normalization, irfft};
///
/// // Signals of length 4 and 5 both pack into 3 bins.
/// let bins = vec![
///     Complex::new(6.0_f64, 0.0),
///     Complex::new(-1.0, 1.0),
///     Complex::new(-2.0, 0.0),
/// ];
/// let even = HermitianSpectrum::from_vec(4, bins.clone())?;
/// let odd = HermitianSpectrum::from_vec(5, bins)?;
/// assert_eq!((even.len(), odd.len()), (3, 3));
/// assert_eq!(irfft(&even, Normalization::Inverse)?.len(), 4);
/// assert_eq!(irfft(&odd, Normalization::Inverse)?.len(), 5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct HermitianSpectrum<S: scioxide_core::RealScalar> {
    logical_len: usize,
    bins: Vec<Complex<S>>,
}

impl<S: scioxide_core::RealScalar> HermitianSpectrum<S> {
    /// Creates a spectrum from the first `⌊logical_len/2⌋ + 1` values of a
    /// Hermitian sequence of length `logical_len`.
    ///
    /// Hermitian symmetry of the DC and Nyquist bins is not checked here; it
    /// is checked by the inverse transform (see
    /// [Hermitian input](crate#hermitian-input)).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `logical_len` is zero.
    /// - `DimensionMismatch` if `bins.len()` is not `⌊logical_len/2⌋ + 1`.
    pub fn from_vec(logical_len: usize, bins: Vec<Complex<S>>) -> Result<Self, SciOxideError> {
        validate_packed_len(logical_len, bins.len())?;
        Ok(Self { logical_len, bins })
    }

    /// Creates a zero spectrum of logical length `logical_len`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `logical_len` is zero.
    /// - `AllocationFailure` if the bins cannot be allocated.
    pub fn zeros(logical_len: usize) -> Result<Self, SciOxideError> {
        let bin_len = packed_len(logical_len)?;
        let bins = try_filled(
            bin_len,
            Complex::new(S::ZERO, S::ZERO),
            scioxide_core::quantity!("packed_hermitian_spectrum"),
        )?;
        Ok(Self { logical_len, bins })
    }

    /// Returns the logical length `n`.
    pub const fn logical_len(&self) -> usize {
        self.logical_len
    }

    /// Returns the number of bins, `⌊n/2⌋ + 1`.
    #[allow(
        clippy::len_without_is_empty,
        reason = "packed spectra always contain at least one bin"
    )]
    pub const fn len(&self) -> usize {
        self.bins.len()
    }

    /// Returns the bins `X[0], …, X[⌊n/2⌋]`.
    pub fn bins(&self) -> &[Complex<S>] {
        &self.bins
    }

    /// Returns the bins mutably.
    pub fn bins_mut(&mut self) -> &mut [Complex<S>] {
        &mut self.bins
    }

    /// Returns the bins as a vector, dropping the logical length.
    pub fn into_vec(self) -> Vec<Complex<S>> {
        self.bins
    }
}

/// Selects real, complex or packed input for a one-dimensional transform.
///
/// [`FftPlan::execute_into`](crate::FftPlan::execute_into) and the chirp-z and
/// nonuniform plans take any value that converts into this type. Slices, arrays
/// and vectors of real or complex values convert into the matching variant, a
/// [`VectorView`](scioxide_linalg::VectorView) selects strided storage, and a
/// [`&HermitianSpectrum`](HermitianSpectrum) or [`Self::hermitian`] gives packed
/// bins with their logical length. The variant must match the plan's layout and
/// direction; the chirp-z and nonuniform plans accept only real or complex
/// input.
///
/// # Examples
///
/// A real forward plan reads the samples 1, …, 5 from every second element of
/// interleaved storage and writes packed bins. The DC bin is their sum, 15.
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext};
/// use scioxide_fft::{FftDirection, FftInput, FftLayout, FftOutput, FftPlan, FftPlanSpec};
/// use scioxide_linalg::VectorView;
/// let (dir, norm) = (FftDirection::Forward, scioxide_fft::Normalization::Inverse);
/// let plan = FftPlan::<f64>::new(FftPlanSpec::new(5, dir, norm, FftLayout::Real))?;
/// let interleaved = [1.0, -9.0, 2.0, -9.0, 3.0, -9.0, 4.0, -9.0, 5.0];
/// let input = FftInput::Real(VectorView::from_slice_with_stride(&interleaved, 5, 2)?);
/// let mut bins = [Complex::new(0.0, 0.0); 3];
/// let output = FftOutput::hermitian(5, &mut bins);
/// let cx = ExecutionContext::sequential();
/// plan.execute_into(input, output, &mut plan.workspace(())?, cx)?;
/// assert!((bins[0] - Complex::new(15.0, 0.0)).norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub enum FftInput<'a, S: scioxide_core::RealScalar>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    /// Complex samples or bins of the full length.
    Complex(scioxide_linalg::VectorView<'a, Complex<S>>),
    /// Real samples, or a real spectrum for a Hermitian inverse.
    Real(scioxide_linalg::VectorView<'a, S>),
    /// Packed bins `X[0], …, X[⌊n/2⌋]` of a Hermitian sequence.
    Hermitian {
        /// Length `n` of the full sequence.
        logical_len: usize,
        /// The `⌊n/2⌋ + 1` bins.
        bins: scioxide_linalg::VectorView<'a, Complex<S>>,
    },
}

/// Selects real, complex or packed output for a one-dimensional transform.
///
/// The counterpart of [`FftInput`], with the same conversions from mutable
/// slices, arrays, vectors and views; a
/// [`&mut HermitianSpectrum`](HermitianSpectrum) or [`Self::hermitian`] receives
/// packed bins. Only the elements that the view addresses are written; elements
/// between strides keep their values.
///
/// # Examples
///
/// A real inverse plan with the same normalization recovers the samples from
/// their packed bins.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_fft::{FftDirection, FftLayout, FftPlan, FftPlanSpec, rfft};
/// let samples = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let (dir, norm) = (FftDirection::Inverse, scioxide_fft::Normalization::Inverse);
/// let spectrum = rfft(&samples, norm)?;
/// let plan = FftPlan::<f64>::new(FftPlanSpec::new(5, dir, norm, FftLayout::Real))?;
/// let (mut out, cx) = ([0.0; 5], ExecutionContext::sequential());
/// plan.execute_into(&spectrum, &mut out, &mut plan.workspace(())?, cx)?;
/// assert!(out.iter().zip(samples).all(|(a, b)| (a - b).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub enum FftOutput<'a, S: scioxide_core::RealScalar>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    /// Complex samples or bins of the full length.
    Complex(scioxide_linalg::VectorViewMut<'a, Complex<S>>),
    /// Real samples, or a real spectrum for a Hermitian forward transform.
    Real(scioxide_linalg::VectorViewMut<'a, S>),
    /// Packed bins `X[0], …, X[⌊n/2⌋]` of a Hermitian sequence.
    Hermitian {
        /// Length `n` of the full sequence.
        logical_len: usize,
        /// The `⌊n/2⌋ + 1` bins.
        bins: scioxide_linalg::VectorViewMut<'a, Complex<S>>,
    },
}

impl<'a, S: scioxide_core::RealScalar> FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    /// Creates packed input from the contiguous bins `X[0], …, X[⌊n/2⌋]` of a
    /// Hermitian sequence of length `n = logical_len`.
    ///
    /// Construction checks nothing; execution returns `DimensionMismatch` if
    /// `logical_len` differs from the plan length or `bins.len()` from
    /// `⌊n/2⌋ + 1`. Use the [`Self::Hermitian`] variant for strided bins.
    pub const fn hermitian(logical_len: usize, bins: &'a [Complex<S>]) -> Self {
        Self::Hermitian {
            logical_len,
            bins: scioxide_linalg::VectorView::from_slice(bins),
        }
    }
}

impl<'a, S: scioxide_core::RealScalar> FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    /// Creates packed output from contiguous storage for the bins
    /// `X[0], …, X[⌊n/2⌋]` of a Hermitian sequence of length `n = logical_len`.
    ///
    /// Construction checks nothing; execution returns `DimensionMismatch` if
    /// `logical_len` differs from the plan length or `bins.len()` from
    /// `⌊n/2⌋ + 1`. Use the [`Self::Hermitian`] variant for strided bins.
    pub fn hermitian(logical_len: usize, bins: &'a mut [Complex<S>]) -> Self {
        Self::Hermitian {
            logical_len,
            bins: scioxide_linalg::VectorViewMut::from_slice(bins),
        }
    }
}

impl<'a, S: scioxide_core::RealScalar> From<&'a HermitianSpectrum<S>> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(spectrum: &'a HermitianSpectrum<S>) -> Self {
        Self::hermitian(spectrum.logical_len, &spectrum.bins)
    }
}
impl<'a, S: scioxide_core::RealScalar> From<&'a mut HermitianSpectrum<S>> for FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(spectrum: &'a mut HermitianSpectrum<S>) -> Self {
        Self::hermitian(spectrum.logical_len, &mut spectrum.bins)
    }
}

impl<'a, S: scioxide_core::RealScalar> From<&'a [S]> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a [S]) -> Self {
        Self::Real(scioxide_linalg::VectorView::from_slice(values))
    }
}
impl<'a, S: scioxide_core::RealScalar> From<&'a Vec<S>> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a Vec<S>) -> Self {
        Self::from(&values[..])
    }
}
impl<'a, S: scioxide_core::RealScalar, const N: usize> From<&'a [S; N]> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a [S; N]) -> Self {
        Self::from(&values[..])
    }
}

impl<'a, S: scioxide_core::RealScalar> From<&'a [Complex<S>]> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a [Complex<S>]) -> Self {
        Self::Complex(scioxide_linalg::VectorView::from_slice(values))
    }
}
impl<'a, S: scioxide_core::RealScalar> From<&'a Vec<Complex<S>>> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a Vec<Complex<S>>) -> Self {
        Self::from(&values[..])
    }
}
impl<'a, S: scioxide_core::RealScalar, const N: usize> From<&'a [Complex<S>; N]> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a [Complex<S>; N]) -> Self {
        Self::from(&values[..])
    }
}

impl<'a, S: scioxide_core::RealScalar> From<&'a mut [S]> for FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a mut [S]) -> Self {
        Self::Real(scioxide_linalg::VectorViewMut::from_slice(values))
    }
}
impl<'a, S: scioxide_core::RealScalar> From<&'a mut Vec<S>> for FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a mut Vec<S>) -> Self {
        Self::from(&mut values[..])
    }
}
impl<'a, S: scioxide_core::RealScalar, const N: usize> From<&'a mut [S; N]> for FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a mut [S; N]) -> Self {
        Self::from(&mut values[..])
    }
}

impl<'a, S: scioxide_core::RealScalar> From<&'a mut [Complex<S>]> for FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a mut [Complex<S>]) -> Self {
        Self::Complex(scioxide_linalg::VectorViewMut::from_slice(values))
    }
}
impl<'a, S: scioxide_core::RealScalar> From<&'a mut Vec<Complex<S>>> for FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a mut Vec<Complex<S>>) -> Self {
        Self::from(&mut values[..])
    }
}
impl<'a, S: scioxide_core::RealScalar, const N: usize> From<&'a mut [Complex<S>; N]>
    for FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a mut [Complex<S>; N]) -> Self {
        Self::from(&mut values[..])
    }
}

impl<'a, S: scioxide_core::RealScalar> From<scioxide_linalg::VectorViewMut<'a, Complex<S>>>
    for FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(view: scioxide_linalg::VectorViewMut<'a, Complex<S>>) -> Self {
        Self::Complex(view)
    }
}

impl<'a, S: scioxide_core::RealScalar> From<scioxide_linalg::VectorViewMut<'a, S>>
    for FftOutput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(view: scioxide_linalg::VectorViewMut<'a, S>) -> Self {
        Self::Real(view)
    }
}

impl<'a, S: scioxide_core::RealScalar> From<scioxide_linalg::VectorView<'a, Complex<S>>>
    for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(view: scioxide_linalg::VectorView<'a, Complex<S>>) -> Self {
        Self::Complex(view)
    }
}

impl<'a, S: scioxide_core::RealScalar> From<scioxide_linalg::VectorView<'a, S>> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(view: scioxide_linalg::VectorView<'a, S>) -> Self {
        Self::Real(view)
    }
}

impl<'a, S: scioxide_core::RealScalar> From<&'a mut [Complex<S>]> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a mut [Complex<S>]) -> Self {
        Self::from(&*values)
    }
}

impl<'a, S: scioxide_core::RealScalar> From<&'a mut [S]> for FftInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    fn from(values: &'a mut [S]) -> Self {
        Self::from(&*values)
    }
}

// Packed bin count for the `logical_len` parameter.
pub(crate) fn packed_len(logical_len: usize) -> Result<usize, SciOxideError> {
    crate::validation::nonzero(logical_len, scioxide_core::argument!("logical_len"))?;
    Ok(logical_len / 2 + 1)
}

fn validate_packed_len(logical_len: usize, actual_len: usize) -> Result<(), SciOxideError> {
    let expected_len = packed_len(logical_len)?;
    crate::validation::length(actual_len, expected_len)
}

// The `argument` does not match the plan's layout.
pub(crate) fn layout_error(argument: Argument) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::UnsupportedConfiguration,
        "input and output variants must match the plan direction and layout",
    )
}

/// Stores the packed N-D spectrum of a real array together with its shape and
/// transformed axes.
///
/// The bins are in row-major order with the shape of the real array, except
/// that the last transformed axis of length `n` holds `⌊n/2⌋ + 1` bins.
/// [`rfftn`](crate::rfftn) and [`ihfftn`](crate::ihfftn) return this type, and
/// [`irfftn`](crate::irfftn) and [`hfftn`](crate::hfftn) take it.
#[derive(Debug, Clone)]
pub struct HermitianSpectrumNd<S: RealScalar> {
    shape: Vec<usize>,
    axes: Vec<usize>,
    bins: Vec<Complex<S>>,
}
impl<S: RealScalar> HermitianSpectrumNd<S> {
    /// Creates a spectrum from row-major `bins` of a real array with shape
    /// `shape`, transformed along `axes`.
    ///
    /// Hermitian symmetry and finiteness are checked by the inverse transform,
    /// not here.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if an axis is not less than the rank of `shape`.
    /// - `InvalidInput` (`Empty`) if `axes` is empty or a transformed extent is
    ///   zero, and (`Inconsistent`) if an axis is repeated.
    /// - `DimensionMismatch` if `bins` does not have the packed length.
    /// - `SizeOverflow` if the number of elements overflows `usize`.
    /// - `AllocationFailure` if the shape or axes cannot be copied.
    pub fn from_vec(
        shape: &[usize],
        axes: &[usize],
        bins: Vec<Complex<S>>,
    ) -> Result<Self, SciOxideError> {
        let axes_argument = scioxide_core::argument!("axes");
        validate_axes(
            shape,
            axes,
            scioxide_core::argument!("shape"),
            axes_argument,
        )?;
        let axis = *axes.last().ok_or_else(|| {
            invalid(
                scioxide_core::InvalidInputReason::Empty,
                axes_argument,
                "packed spectra require an axis",
            )
        })?;
        let mut packed_shape = copy_metadata(shape)?;
        packed_shape[axis] = packed_len(shape[axis])?;
        crate::validation::length(bins.len(), Shape::new(&packed_shape)?.len())?;
        Ok(Self {
            shape: copy_metadata(shape)?,
            axes: copy_metadata(axes)?,
            bins,
        })
    }
    /// Returns the shape of the real array.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Returns the transformed axes in order; the last is packed.
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
    /// Returns the bins in row-major order.
    pub fn bins(&self) -> &[Complex<S>] {
        &self.bins
    }
    /// Returns the bins mutably.
    pub fn bins_mut(&mut self) -> &mut [Complex<S>] {
        &mut self.bins
    }
    /// Returns the bins as a vector, dropping the shape and axes.
    pub fn into_vec(self) -> Vec<Complex<S>> {
        self.bins
    }
}

/// Validated full samples for specialized transforms that reject packed input.
pub(crate) enum FullInput<'a, S: RealScalar>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    Real(scioxide_linalg::VectorView<'a, S>),
    Complex(scioxide_linalg::VectorView<'a, Complex<S>>),
}
impl<'a, S: RealScalar> TryFrom<FftInput<'a, S>> for FullInput<'a, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    type Error = SciOxideError;
    fn try_from(input: FftInput<'a, S>) -> Result<Self, Self::Error> {
        match input {
            FftInput::Real(view) => Ok(Self::Real(view)),
            FftInput::Complex(view) => Ok(Self::Complex(view)),
            FftInput::Hermitian { .. } => Err(invalid(
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                scioxide_core::argument!("input"),
                "input must be real or full complex",
            )),
        }
    }
}
impl<S: RealScalar> FullInput<'_, S>
where
    Complex<S>: scioxide_core::NumericScalar,
{
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Real(view) => view.len(),
            Self::Complex(view) => view.len(),
        }
    }
}
