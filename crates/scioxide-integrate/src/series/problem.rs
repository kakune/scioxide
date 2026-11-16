//! Term providers and the tail information needed by Euler–Maclaurin summation.

use scioxide_core::{NumericScalar, SciOxideError};

/// Terms `aₙ` of an infinite series, evaluated by index.
///
/// A term is a slice of values: a slice of length one is a scalar series, and a
/// longer slice is a vector series whose components are summed independently.
/// Closures `Fn(usize, &mut [T]) -> Result<(), SciOxideError>` implement this
/// trait. See [Integrands](crate#integrands) for how callback errors and
/// non-finite values are handled.
pub trait SeriesTerms<T: NumericScalar>: Sync {
    /// Writes the term with index `index` to `output`.
    ///
    /// `output` has the length of one term. Its contents are unspecified if the
    /// call fails.
    ///
    /// # Errors
    ///
    /// Returns any error of the provider; the summation returns it unchanged.
    fn term(&self, index: usize, output: &mut [T]) -> Result<(), SciOxideError>;
}

impl<T: NumericScalar, F: Fn(usize, &mut [T]) -> Result<(), SciOxideError> + Sync> SeriesTerms<T>
    for F
{
    fn term(&self, index: usize, output: &mut [T]) -> Result<(), SciOxideError> {
        self(index, output)
    }
}

/// Terms `aₙ = f(n)` of a smooth function `f`, with the integral and
/// derivatives of `f` that Euler–Maclaurin summation needs.
///
/// For a tail starting at index `n` and `p` corrections, the summation uses
///
/// ```text
/// Σₖ₌ₙ^∞ f(k) = ∫ₙ^∞ f(x) dx + f(n)/2 − Σⱼ₌₁^p (B₂ⱼ/(2j)!) f^(2j−1)(n) + Rₚ,
/// |Rₚ| ≤ (|B₂ₚ|/(2p)!) ∫ₙ^∞ |f^(2p)(x)| dx,
/// ```
///
/// where `B₂ⱼ` are the Bernoulli numbers and `f^(m)` is the `m`-th derivative
/// of `f`. You are responsible for the assumptions behind this formula: `f`
/// extends to a smooth function on `[n, ∞)`, its odd derivatives tend to zero
/// at infinity, and `f^(2p)` is absolutely integrable there. Every error bound
/// you return must hold for each component of the term. Under these
/// assumptions, the error of the estimate is at most the sum of your declared
/// errors, the remainder bound and rounding errors. SciOxide does not check the
/// assumptions.
///
/// # References
///
/// NIST DLMF, §2.10(i), Euler–Maclaurin formula. <https://dlmf.nist.gov/2.10#i>
pub trait EulerMaclaurinProvider<T: NumericScalar>: SeriesTerms<T> {
    /// Writes an approximation of `∫ f(x) dx` over `[at, ∞)` to `value` and a
    /// bound on its absolute error to `absolute_error`.
    ///
    /// # Errors
    ///
    /// Returns any error of the provider; the summation returns it unchanged.
    fn tail_integral(
        &self,
        at: usize,
        value: &mut [T],
        absolute_error: &mut [T::Real],
    ) -> Result<(), SciOxideError>;

    /// Writes an approximation of the odd derivative `f^(order)(at)` to `value`
    /// and a bound on its absolute error to `absolute_error`.
    ///
    /// # Errors
    ///
    /// Returns any error of the provider; the summation returns it unchanged.
    fn odd_derivative(
        &self,
        at: usize,
        order: usize,
        value: &mut [T],
        absolute_error: &mut [T::Real],
    ) -> Result<(), SciOxideError>;

    /// Writes an upper bound on `∫ |f^(order)(x)| dx` over `[at, ∞)` to
    /// `bound`, for an even `order`.
    ///
    /// # Errors
    ///
    /// Returns any error of the provider; the summation returns it unchanged.
    fn even_derivative_l1_bound(
        &self,
        at: usize,
        order: usize,
        bound: &mut [T::Real],
    ) -> Result<(), SciOxideError>;
}

/// Series to sum, in the form the chosen [`SeriesMethod`](crate::SeriesMethod)
/// needs.
#[derive(Clone, Copy)]
pub enum SeriesProblem<'a, T: NumericScalar> {
    /// Terms only, for [`LevinU`](crate::SeriesMethod::LevinU) and
    /// [`WynnEpsilon`](crate::SeriesMethod::WynnEpsilon).
    Terms(&'a dyn SeriesTerms<T>),
    /// Terms with tail integrals and derivatives, for
    /// [`EulerMaclaurin`](crate::SeriesMethod::EulerMaclaurin).
    EulerMaclaurin(&'a dyn EulerMaclaurinProvider<T>),
}
