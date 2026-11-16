//! Scratch-length requirements of borrowed scratch slices.

/// Reports the minimum length of a scratch slice that you pass to a polynomial
/// function.
///
/// A function that borrows a scratch slice reports the length it needs through
/// `workspace_requirements`. The slice is never resized. A shorter slice is a
/// `DimensionMismatch` error before any numerical work, and a longer slice is
/// accepted. The scratch contents may change on success or on error (see
/// [Output on error](scioxide_core#output-on-error)).
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_poly::ApproxPolynomial;
///
/// // p(x) = 1·(1 − x) + 3·x in the Bernstein basis on [0, 1], so p(0.5) = 2.
/// let p = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0)?;
/// let mut scratch = vec![0.0_f64; p.workspace_requirements(1)?.scratch_len()];
/// let mut value = [0.0];
/// p.evaluate_batch_into(
///     &[0.5],
///     &mut value,
///     &mut scratch,
///     ExecutionContext::sequential(),
/// )?;
/// assert!((value[0] - 2.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolynomialScratchRequirements {
    scratch_len: usize,
}

impl PolynomialScratchRequirements {
    pub(crate) const fn new(scratch_len: usize) -> Self {
        Self { scratch_len }
    }
    /// Returns the minimum number of scratch elements.
    ///
    /// ```
    /// use scioxide_poly::ApproxPolynomial;
    ///
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// // Values and derivatives at three points, and no basis scratch.
    /// let requirements = p.workspace_requirements(3)?;
    /// assert_eq!(requirements.scratch_len(), 6);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn scratch_len(self) -> usize {
        self.scratch_len
    }
}
