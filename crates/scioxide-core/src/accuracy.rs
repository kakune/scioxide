/// Accuracy level of the products, sums and norms in SciOxide's linear-algebra kernels.
///
/// `Standard` uses ordinary floating-point arithmetic with rescaling to avoid
/// overflow. `Exact` accumulates every product exactly and rounds the result
/// once, so it is correctly rounded.
///
/// # Error bounds for `Standard`
///
/// Let `u` be the unit roundoff, `γ_k = k·ε/(1 − k·ε)` with `ε` the machine
/// epsilon, and `u_sub` the smallest subnormal of the output type times the
/// rescaling factor. The bounds below are informative when `k·ε < 1`.
///
/// ```text
/// inner product of length k:   |error| ≤ γ_k · Σ |a_j b_j| + k·u_sub
/// GEMM with inner dimension k: |error_ij| ≤ γ_k · (|A||B|)_ij + k·u_sub
/// ```
///
/// GEMV, dot products, SYRK and TRMM use the inner-product bound, and so do
/// sparse inner products. A complex component counts as at most `2k` real
/// products. Affine products `αAB + βC` add `|α||A||B| + |β||C|` and two
/// operations, and rank-2k updates include both products. AXPY, AXPBY and plane
/// rotations use the two-product bound (four real products for complex values).
/// ASUM uses the sum of component magnitudes and the component count.
///
/// Triangular solves (TRSM and vector substitution) satisfy the componentwise
/// backward-error bound `|residual| ≤ γ_(n+1) · (|T||X| + |αB|)`, plus the
/// rounding of rescaled underflow. The forward error also depends on the
/// condition number.
///
/// Euclidean norms use blocked squares or Blue's scaled sum of squares, with
/// relative error at most `γ_(2n+5)` plus the final subnormal rounding, where
/// `n` counts real components. `Exact` norms use the scaled sum-of-squares recurrence.
/// Neither level promises a correctly rounded square root.
///
/// # Reproducibility
///
/// For fixed options and backend, `Standard` results are bitwise independent of
/// the number of workers. Other instruction-set backends satisfy the same error
/// bound. `Exact` results are also independent of the instruction set.
///
/// # Errors
///
/// Kernels that use this setting return `NonFinite` for NaN or infinite
/// components and `NumericalInstability` (`Overflow`) when a rescaled result is
/// not representable. See [output on error](crate#output-on-error).
///
/// ```
/// use scioxide_core::Accuracy;
///
/// assert_eq!(Accuracy::default(), Accuracy::Standard);
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Accuracy {
    /// Ordinary arithmetic with rescaling to avoid overflow; reproducible for a fixed backend.
    #[default]
    Standard,
    /// Exact product accumulation with one final rounding.
    Exact,
}
