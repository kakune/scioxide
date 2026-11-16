//! Owned synthesis and spectrograms, and plans with a supplied dual window.
use super::*;
impl<T: NumericScalar> StftPlan<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Reconstructs a record of `signal_len` samples from its STFT
    /// coefficients and returns it.
    ///
    /// # Errors
    ///
    /// - The errors of [`StftPlan::synthesize_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn synthesize(
        &self,
        coefficients: &[Complex<T::Real>],
        signal_len: usize,
    ) -> Result<Vec<T>, SciOxideError> {
        let mut output = try_filled(
            signal_len,
            T::ZERO,
            scioxide_core::quantity!("istft_output"),
        )?;
        let mut ws = self.workspace(StftRequest::Synthesis(signal_len))?;
        self.synthesize_into(
            coefficients,
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Computes the power spectrogram `|X[p, k]|²` of `input` and returns it,
    /// frame by frame.
    ///
    /// This is the owned form of [`StftPlan::spectrogram_into`], which defines
    /// the values. The result has [`StftPlan::output_len`] values,
    /// [`StftPlan::bins`] per frame, and costs one FFT per frame.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a power overflows.
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{StftOptions, StftPlan};
    /// let plan = StftPlan::<f64>::new(&[0.5, 1.0, 0.5, 0.0], 2, StftOptions::new())?;
    /// let input = [1.0, -2.0, 0.5, 3.0, -1.0];
    /// let power = plan.spectrogram(&input)?;
    /// // Each value is the squared magnitude of the STFT coefficient.
    /// let coefficients = plan.analyze(&input)?;
    /// assert_eq!(power.len(), coefficients.len());
    /// for (p, x) in power.iter().zip(&coefficients) {
    ///     assert!((p - x.norm_sqr()).abs() < 1e-12);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn spectrogram(&self, input: &[T]) -> Result<Vec<T::Real>, SciOxideError> {
        let mut output = try_filled(
            self.output_len(input.len())?,
            T::Real::ZERO,
            scioxide_core::quantity!("spectrogram_output"),
        )?;
        let mut ws = self.workspace(StftRequest::Spectrogram(input.len()))?;
        self.spectrogram_into(
            input,
            &mut output,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Prepares an STFT with hop `hop` from a synthesis window `dual`,
    /// analyzing with its minimum-norm dual.
    ///
    /// The analysis window is [`canonical_stft_dual`] of `dual` and `hop`, and
    /// synthesis uses `dual` itself.
    ///
    /// # Errors
    ///
    /// - The errors of [`canonical_stft_dual`] for `dual`, and of
    ///   [`StftPlan::new`] for the analysis window.
    /// - `InvalidInput` (`Inconsistent`) if `dual` fails the reconstruction
    ///   check against the analysis window.
    pub fn from_dual(
        dual: &[T],
        hop: usize,
        options: StftOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let window = canonical_stft_dual(dual, hop, options.tolerance)?;
        Self::new(&window, hop, options)?.replace_dual(dual)
    }
    /// Prepares an STFT with analysis window `window`, synthesis window `dual`
    /// and hop `hop`.
    ///
    /// The pair must satisfy the dual condition of [`canonical_stft_dual`]:
    /// with `h = hop`, for each residue `r = 0, …, h − 1`,
    /// `|Σ_q conj(w[r + q·h]) d[r + q·h] − 1|` is at most `atol + rtol` of the
    /// options' tolerance.
    ///
    /// # Errors
    ///
    /// - The errors of [`StftPlan::new`].
    /// - `InvalidInput` (`Inconsistent`) if `window` and the hop violate NOLA,
    ///   or `dual` fails the reconstruction check.
    /// - `DimensionMismatch` if `dual` and `window` differ in length.
    /// - `NonFinite` if a coefficient of `dual` is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{StftOptions, StftPlan, canonical_stft_dual};
    ///
    /// let window = [0.5_f64, 1.0, 0.5, 0.25];
    /// let options = StftOptions::new();
    /// let dual = canonical_stft_dual(&window, 2, options.tolerance())?;
    /// let plan = StftPlan::from_windows(&window, &dual, 2, options)?;
    /// assert_eq!(plan.dual_window(), Some(&dual[..]));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn from_windows(
        window: &[T],
        dual: &[T],
        hop: usize,
        options: StftOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::new(window, hop, options)?.replace_dual(dual)
    }
    /// Returns the half-open range of sample indices that frame `frame` covers.
    ///
    /// The range is `p·h − ⌊m/2⌋ .. p·h − ⌊m/2⌋ + m` for frame `p`, hop `h`
    /// and window length `m`. It can extend beyond the record on either side.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if an end of the range does not fit in `i64`.
    pub fn frame_sample_range(&self, frame: i64) -> Result<core::ops::Range<i64>, SciOxideError> {
        let start = self.start(frame);
        let end = start + self.frame.window.len() as i128;
        Ok(i64::try_from(start).map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("stft_frame_support"))
        })?..i64::try_from(end).map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("stft_frame_support"))
        })?)
    }
}
