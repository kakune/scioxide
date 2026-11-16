//! Level 1 reductions, swaps and plane rotations on vector views.

use scioxide_core::AlgebraicScalar;
use scioxide_core::{Dimensions, NumericScalar, ProductSum, RealScalar, SciOxideError};

use crate::{
    Accuracy, DenseWorkspaceRequest, LinalgExecutionOptions, LinalgWorkspace, VectorView,
    VectorViewMut,
};

fn equal_lengths(left: usize, right: usize) -> Result<(), SciOxideError> {
    if left != right {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(left),
            Dimensions::one(right),
        ));
    }
    Ok(())
}

impl<S: NumericScalar> VectorView<'_, S> {
    /// Computes `Σ (|Re xᵢ| + |Im xᵢ|)`, the BLAS absolute sum.
    ///
    /// For real vectors this is the 1-norm `‖x‖₁`. For complex vectors it is not the 1-norm,
    /// because each entry contributes `|Re xᵢ| + |Im xᵢ|` instead of `|xᵢ|`. The sum uses
    /// `Standard` [`Accuracy`](scioxide_core::Accuracy); with `Exact` in its options,
    /// [`asum_scoped`](Self::asum_scoped) rounds the sum once.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `NumericalInstability` if the sum overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::VectorView;
    ///
    /// let x = VectorView::from_slice(&[1.0_f64, -2.0, 3.0]);
    /// let sum = x.asum()?;
    /// // |1| + |-2| + |3| = 6.
    /// assert!((sum - 6.0).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn asum(&self) -> Result<S::Real, SciOxideError> {
        self.asum_scoped(
            &crate::LinalgExecutionOptions::default(),
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    /// Computes [`asum`](Self::asum) with `options` in the caller's execution context, which can
    /// select [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), attach a BLAS provider or stop the
    /// computation.
    ///
    /// # Errors
    ///
    /// The errors of [`asum`](Self::asum), and:
    ///
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn asum_scoped<'scope, 'a>(
        &self,
        options: &crate::LinalgExecutionOptions,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<S::Real, SciOxideError> {
        let context = context.into().with_options(options)?.kernel();

        let values = self.try_finite()?;
        let context: crate::KernelContext<'_> = context;
        context.execution().check()?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas()
            && let Some(result) = crate::blas::dispatch::asum(
                *self,
                dispatch,
                context.accuracy(),
                crate::kernel::provider_policy_workers(context),
            )?
        {
            return Ok(result);
        }
        if context.accuracy() == Accuracy::Standard {
            let e = crate::kernel::standard::exponent(values.maximum)?;
            let sum = values.iter().fold(S::Real::ZERO, |sum, value| {
                sum + value.real_component().abs().scalbn(-e)
                    + value.imaginary_component().abs().scalbn(-e)
            });
            return crate::kernel::standard::restore(sum, e);
        }
        let mut sum = ProductSum::<S::Real>::new();
        values.iter().try_for_each(|value| {
            sum.add_product(value.real_component().abs(), S::Real::ONE)?;
            sum.add_product(value.imaginary_component().abs(), S::Real::ONE)
        })?;
        sum.finish()
    }

    /// Returns the zero-based index of the first entry that maximizes `|Re xᵢ| + |Im xᵢ|`.
    ///
    /// This is the BLAS IAMAX convention. Returns `None` for an empty vector. The sums are
    /// compared exactly, without rounding, in `O(n)` operations.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - Execution errors as described in [Execution](crate#execution).
    pub fn iamax(&self) -> Result<Option<usize>, SciOxideError> {
        self.iamax_scoped(
            &crate::LinalgExecutionOptions::default(),
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    /// Computes [`iamax`](Self::iamax) with `options` in the caller's execution context, which can
    /// select [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), attach a BLAS provider or stop the
    /// computation.
    ///
    /// # Errors
    ///
    /// The errors of [`iamax`](Self::iamax), and:
    ///
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn iamax_scoped<'scope, 'a>(
        &self,
        options: &crate::LinalgExecutionOptions,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<Option<usize>, SciOxideError> {
        let context = context.into().with_options(options)?.kernel();

        let values = self.try_finite()?;
        let context: crate::KernelContext<'_> = context;
        context.execution().check()?;
        if self.is_empty() {
            return Ok(None);
        }
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas()
            && let Some(result) = crate::blas::dispatch::iamax(
                *self,
                dispatch,
                context.accuracy(),
                crate::kernel::provider_policy_workers(context),
            )?
        {
            return Ok(Some(result));
        }
        let mut best: Option<(usize, AbsoluteSumKey<S::Real>)> = None;
        for (index, value) in values.iter().enumerate() {
            let key = AbsoluteSumKey::new(
                value.real_component().abs(),
                value.imaginary_component().abs(),
            );
            if best.is_none_or(|(_, best_key)| key > best_key) {
                best = Some((index, key));
            }
        }
        Ok(best.map(|(index, _)| index))
    }
}

/// Exact sort key of `a + b` for finite `a, b ≥ 0`.
///
/// Ordinary sums are the TwoSum pair `(fl(a + b), a + b − fl(a + b))`, which
/// orders lexicographically like the exact sum because rounding is monotone.
/// A sum that rounds past the largest finite value forms a higher tier keyed by
/// the halves: with `2^emax` the first power of two above the largest finite
/// value and `p` the precision, both summands are then at least
/// `2^(emax − p − 1)`, so halving them is exact.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct AbsoluteSumKey<R> {
    overflowed: bool,
    sum: R,
    residual: R,
}

impl<R: RealScalar> AbsoluteSumKey<R> {
    fn new(a: R, b: R) -> Self {
        let (sum, residual) = scioxide_core::backend::two_sum(a, b);
        if sum.is_finite() {
            return Self {
                overflowed: false,
                sum,
                residual,
            };
        }
        let (sum, residual) = scioxide_core::backend::two_sum(a * R::HALF, b * R::HALF);
        Self {
            overflowed: true,
            sum,
            residual,
        }
    }
}

impl VectorView<'_, f32> {
    /// Computes the `f32` dot product `Σ xᵢ yᵢ`, accumulated and returned in `f64`.
    ///
    /// Each product of two `f32` values is exact in `f64`. Under `Accuracy::Standard` the
    /// products are summed in `f64` arithmetic. Under `Accuracy::Exact` the sum is rounded
    /// once to `f64`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` and `other` differ in length.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `NumericalInstability` if the sum overflows.
    /// - Execution errors as described in [Execution](crate#execution).
    pub fn dot_wide<'v>(
        &self,
        other: impl Into<VectorView<'v, f32>>,
    ) -> Result<f64, SciOxideError> {
        self.dot_wide_scoped(
            other,
            &crate::LinalgExecutionOptions::default(),
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    /// Computes [`dot_wide`](Self::dot_wide) with `options` in the caller's execution context,
    /// which can select [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), attach a BLAS provider
    /// or stop the computation.
    ///
    /// # Errors
    ///
    /// The errors of [`dot_wide`](Self::dot_wide), and:
    ///
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn dot_wide_scoped<'v, 'scope, 'a>(
        &self,
        other: impl Into<VectorView<'v, f32>>,
        options: &crate::LinalgExecutionOptions,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<f64, SciOxideError> {
        let other = other.into();
        let context = context.into().with_options(options)?.kernel();

        equal_lengths(self.len(), other.len())?;
        let left = self.try_finite()?;
        let right = other.try_finite()?;
        let context: crate::KernelContext<'_> = context;
        context.execution().check()?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas()
            && let Some(result) = crate::blas::dispatch::wide_dot(
                *self,
                other,
                None,
                dispatch,
                context.accuracy(),
                crate::kernel::provider_policy_workers(context),
            )?
        {
            return Ok(result);
        }
        if context.accuracy() == Accuracy::Standard {
            // Binary32 products and any addressable vector sum fit binary64.
            let sum = left
                .iter()
                .zip(right.iter())
                .fold(0.0_f64, |sum, (a, b)| f64::from(a) * f64::from(b) + sum);
            return crate::checked::finite_candidate(sum);
        }
        let mut sum = ProductSum::<f64>::new();
        left.iter()
            .zip(right.iter())
            .try_for_each(|(a, b)| sum.add_product(f64::from(a), f64::from(b)))?;
        sum.finish()
    }

    /// Computes `alpha + Σ xᵢ yᵢ` for `f32` vectors.
    ///
    /// Under `Accuracy::Exact` the result is rounded once to `f32`. Under
    /// `Accuracy::Standard` it is computed in `f32` arithmetic with power-of-two rescaling to
    /// avoid overflow.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `self` and `other` differ in length.
    /// - `NonFinite` if `alpha` or an entry is NaN or infinite.
    /// - `NumericalInstability` if the result overflows.
    /// - Execution errors as described in [Execution](crate#execution).
    pub fn dot_plus<'v>(
        &self,
        other: impl Into<VectorView<'v, f32>>,
        alpha: f32,
    ) -> Result<f32, SciOxideError> {
        self.dot_plus_scoped(
            other,
            alpha,
            &crate::LinalgExecutionOptions::default(),
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    /// Computes [`dot_plus`](Self::dot_plus) with `options` in the caller's execution context,
    /// which can select [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), attach a BLAS provider
    /// or stop the computation.
    ///
    /// # Errors
    ///
    /// The errors of [`dot_plus`](Self::dot_plus), and:
    ///
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn dot_plus_scoped<'v, 'scope, 'a>(
        &self,
        other: impl Into<VectorView<'v, f32>>,
        alpha: f32,
        options: &crate::LinalgExecutionOptions,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<f32, SciOxideError> {
        let other = other.into();
        let context = context.into().with_options(options)?.kernel();

        equal_lengths(self.len(), other.len())?;
        crate::checked::validate_input(alpha, "alpha")?;
        let left = self.try_finite()?;
        let right = other.try_finite()?;
        let context: crate::KernelContext<'_> = context;
        context.execution().check()?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas()
            && let Some(result) = crate::blas::dispatch::wide_dot(
                *self,
                other,
                Some(alpha),
                dispatch,
                context.accuracy(),
                crate::kernel::provider_policy_workers(context),
            )?
        {
            return Ok(result as f32);
        }
        if context.accuracy() == Accuracy::Standard {
            use crate::kernel::standard;
            let (a, b) = standard::product_scales(left.maximum, right.maximum, left.len())?;
            let sum = |a: i32, b: i32| {
                left.iter().zip(right.iter()).fold(0.0_f32, |sum, (x, y)| {
                    standard::scale(x, -a) * standard::scale(y, -b) + sum
                })
            };
            // As in Standard dots, normalize only after an unscaled overflow.
            if a + b > 0 {
                let unscaled = sum(0, 0) + alpha;
                if unscaled.is_finite() {
                    return Ok(unscaled);
                }
            }
            let exponent = (a + b).max(standard::exponent(alpha.abs())?);
            return standard::restore(
                standard::scale(sum(a, b), a + b - exponent) + standard::scale(alpha, -exponent),
                exponent,
            );
        }
        let mut sum = ProductSum::<f32>::new();
        sum.add_product(alpha, 1.0)?;
        left.iter()
            .zip(right.iter())
            .try_for_each(|(a, b)| sum.add_product(a, b))?;
        sum.finish()
    }
}

impl<S: NumericScalar> VectorViewMut<'_, S> {
    /// Exchanges the entries of `first` and `other`.
    ///
    /// Without a BLAS provider no workspace is used. With a provider, reserve
    /// [`DenseWorkspaceRequest::vector_pair`] for the copies passed to it.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the vectors differ in length.
    /// - `NonFinite` if an entry of either vector is NaN or infinite.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn swap_in_place<'scope, 'a>(
        options: &LinalgExecutionOptions,
        mut first: Self,
        mut other: Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        equal_lengths(first.len(), other.len())?;
        if options.byte_limit().is_some() {
            let _ = workspace.check(&crate::LinalgRequirements::new(options.byte_limit()))?;
        }
        let _ = first.as_view().try_finite()?;
        let _ = other.as_view().try_finite()?;
        let context: crate::KernelContext<'_> = context;
        context.execution().check()?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas() {
            let request =
                DenseWorkspaceRequest::vector_pair(first.len(), options).for_kernels(context);
            workspace.try_reserve(&LinalgWorkspace::<S>::workspace_requirements(&request)?)?;
            let (left, right) = crate::workspace::split_mut(
                workspace.scalar_mut(crate::kernel::CANDIDATE),
                first.len(),
            )?;
            if left.len() < first.len() || right.len() < first.len() {
                return Err(SciOxideError::dimension_mismatch(
                    Dimensions::two(first.len(), first.len()),
                    Dimensions::two(left.len(), right.len()),
                ));
            }

            for index in 0..first.len() {
                left[index] = first.as_view().get_element(index);
                right[index] = other.as_view().get_element(index);
            }
            if crate::blas::dispatch::swap(
                &mut left[..first.len()],
                &mut right[..first.len()],
                dispatch,
                context.accuracy(),
                crate::kernel::provider_worker_bound::<S>(
                    first.len(),
                    1,
                    1,
                    options.execution_policy(),
                    context,
                )?,
            )? == crate::blas::dispatch::BlasRoute::Provider
            {
                for index in 0..first.len() {
                    first.set_element(index, left[index]);
                    other.set_element(index, right[index]);
                }
                return Ok(());
            }
        }
        for index in 0..first.len() {
            let value = first.as_view().get_element(index);
            first.set_element(index, other.as_view().get_element(index));
            other.set_element(index, value);
        }
        Ok(())
    }
}

fn rotated<R: RealScalar>(x: R, y: R, cosine: R, sine: R) -> Result<(R, R), SciOxideError> {
    let mut first = ProductSum::new();
    first.add_product(cosine, x)?;
    first.add_product(sine, y)?;
    let mut second = ProductSum::new();
    second.add_product(cosine, y)?;
    second.subtract_product(sine, x)?;
    Ok((first.finish()?, second.finish()?))
}

impl<R: RealScalar> VectorViewMut<'_, R> {
    /// Applies a real plane rotation to every pair `(xᵢ, yᵢ)` of `first` and `other`.
    ///
    /// Each pair becomes `xᵢ := c xᵢ + s yᵢ`, `yᵢ := c yᵢ − s xᵢ`, with `c` and `s` from
    /// `rotation` (BLAS ROT). The coefficients are used as given; `c² + s² = 1` is not
    /// checked. Reserve [`DenseWorkspaceRequest::vector_pair`] in `workspace`. Under
    /// `Accuracy::Exact` each new entry is rounded once.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the vectors differ in length.
    /// - `NonFinite` if an entry of either vector is NaN or infinite.
    /// - `NumericalInstability` if a rotated entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn rotate_in_place<'scope, 'a>(
        rotation: GivensRotation<R>,
        options: &LinalgExecutionOptions,
        mut first: Self,
        mut other: Self,
        workspace: &mut LinalgWorkspace<'_, R>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let GivensRotation { cosine, sine } = rotation;
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let request = DenseWorkspaceRequest::vector_pair(first.len(), options).for_kernels(context);
        workspace.try_reserve(&LinalgWorkspace::<R>::workspace_requirements(&request)?)?;
        let (left, right) = crate::workspace::split_mut(
            workspace.scalar_mut(crate::kernel::CANDIDATE),
            first.len(),
        )?;
        equal_lengths(first.len(), other.len())?;
        if left.len() < first.len() || right.len() < first.len() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(first.len(), first.len()),
                Dimensions::two(left.len(), right.len()),
            ));
        }
        crate::checked::validate_input(cosine, "cosine")?;
        crate::checked::validate_input(sine, "sine")?;
        let x = first.as_view().try_finite()?;
        let y = other.as_view().try_finite()?;
        let first_affine =
            crate::kernel::standard::Affine::new(cosine, x.maximum, sine, y.maximum)?;
        let second = crate::kernel::standard::Affine::new(-sine, x.maximum, cosine, y.maximum)?;
        let context: crate::KernelContext<'_> = context;
        context.execution().check()?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas() {
            for index in 0..first.len() {
                left[index] = first.as_view().get_element(index);
                right[index] = other.as_view().get_element(index);
            }
            if crate::blas::dispatch::rotate(crate::blas::dispatch::RotationRequest {
                blas_operation: crate::BlasOperation::Rot,
                x: &mut left[..first.len()],
                y: &mut right[..first.len()],
                cosine,
                sine,
                parameters: None,
                dispatch,
                accuracy: context.accuracy(),
                workers: crate::kernel::provider_worker_bound::<R>(
                    first.len(),
                    1,
                    1,
                    options.execution_policy(),
                    context,
                )?,
            })? == crate::blas::dispatch::BlasRoute::Provider
            {
                for index in 0..first.len() {
                    first.set_element(index, left[index]);
                    other.set_element(index, right[index]);
                }
                return Ok(());
            }
        }
        for index in 0..first.len() {
            let (x, y) = if context.accuracy() == Accuracy::Standard {
                let x = first.as_view().get_element(index);
                let y = other.as_view().get_element(index);
                (
                    crate::checked::finite_candidate(first_affine.apply(x, y))?,
                    crate::checked::finite_candidate(second.apply(x, y))?,
                )
            } else {
                rotated(
                    first.as_view().get_element(index),
                    other.as_view().get_element(index),
                    cosine,
                    sine,
                )?
            };
            left[index] = x;
            right[index] = y;
        }
        for index in 0..first.len() {
            first.set_element(index, left[index]);
            other.set_element(index, right[index]);
        }
        Ok(())
    }
}

/// Real plane rotation `G = [c s; −s c]` with finite cosine `c` and sine `s`.
///
/// `G` maps `[x; y]` to `[c x + s y; c y − s x]`. The coefficients are stored as given and
/// are not normalized. [`givens_rotation`] returns one with `c² + s² = 1`.
///
/// # Examples
///
/// ```
/// let rotation = scioxide_linalg::GivensRotation::new(0.6_f64, 0.8)?;
/// assert_eq!(rotation.cosine(), 0.6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GivensRotation<R: RealScalar> {
    cosine: R,
    sine: R,
}
impl<R: RealScalar> GivensRotation<R> {
    /// Creates a rotation from its cosine and sine.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `cosine` or `sine` is NaN or infinite.
    pub fn new(cosine: R, sine: R) -> Result<Self, SciOxideError> {
        crate::checked::validate_input(cosine, "cosine")?;
        crate::checked::validate_input(sine, "sine")?;
        Ok(Self { cosine, sine })
    }
    /// Returns the cosine `c`.
    pub const fn cosine(self) -> R {
        self.cosine
    }
    /// Returns the sine `s`.
    pub const fn sine(self) -> R {
        self.sine
    }
}

/// Result of [`givens_rotation`]: the rotation, `r`, and the BLAS reconstruction value `z`.
///
/// The rotation `[c s; −s c]` maps `[a; b]` to `[r; 0]`, with `r = ±√(a² + b²)`. The sign
/// of `r` is that of `a` if `|a| > |b|` and that of `b` otherwise (BLAS ROTG convention).
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct GivensRotationResult<R: RealScalar> {
    /// Rotated first coordinate.
    pub(crate) r: R,
    /// BLAS reconstruction value `z`.
    pub(crate) z: R,
    /// Generated rotation.
    pub(crate) rotation: GivensRotation<R>,
}

impl<R: RealScalar> GivensRotationResult<R> {
    /// Returns `r`, the first coordinate after rotation.
    pub const fn r(&self) -> R {
        self.r
    }

    /// Returns the BLAS ROTG value `z`, from which the rotation can be rebuilt.
    ///
    /// `z = s` if `|a| > |b|`. Otherwise `z = 1/c` if `c ≠ 0` and `z = 1` if `c = 0`. For
    /// `a = b = 0`, `z = 0`.
    pub const fn z(&self) -> R {
        self.z
    }

    /// Returns the rotation.
    pub const fn rotation(&self) -> GivensRotation<R> {
        self.rotation
    }
}

/// Computes the real plane rotation that maps `[a; b]` to `[r; 0]`.
///
/// Returns `c`, `s` and `r` with `[c s; −s c] [a; b] = [r; 0]` and `c² + s² = 1`, with the
/// sign conventions and the reconstruction value `z` of BLAS ROTG described on
/// [`GivensRotationResult`]. For `a = b = 0` the rotation is the identity and `r = 0`.
/// Inputs of extreme magnitude are rescaled by powers of two, so `r` overflows only if
/// `√(a² + b²)` does.
///
/// # Errors
///
/// - `NonFinite` if `a` or `b` is NaN or infinite.
/// - `NumericalInstability` if `r` overflows.
/// - Execution errors as described in [Execution](crate#execution).
///
/// # Examples
///
/// ```
/// use scioxide_linalg::givens_rotation;
/// let rotation = givens_rotation(
///     3.0_f64,
///     4.0)?;
/// // r = √(3² + 4²) = 5, and the second coordinate −3 s + 4 c vanishes.
/// assert!((rotation.r() - 5.0).abs() < 1e-14);
/// assert!((-rotation.rotation().sine() * 3.0 + rotation.rotation().cosine() * 4.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn givens_rotation<R: RealScalar>(
    a: R,
    b: R,
) -> Result<GivensRotationResult<R>, SciOxideError> {
    givens_rotation_scoped(
        a,
        b,
        &crate::LinalgExecutionOptions::default(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
}

/// Computes [`givens_rotation`](givens_rotation) with `options` in the caller's execution context,
/// which can select [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), attach a BLAS provider or
/// stop the computation.
///
/// # Errors
///
/// The errors of [`givens_rotation`](givens_rotation), and:
///
/// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
pub fn givens_rotation_scoped<'scope, 'a, R: RealScalar>(
    a: R,
    b: R,
    options: &crate::LinalgExecutionOptions,
    context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
) -> Result<GivensRotationResult<R>, SciOxideError> {
    let context = context.into().with_options(options)?.kernel();

    crate::checked::validate_input(a, "a")?;
    crate::checked::validate_input(b, "b")?;
    let context: crate::KernelContext<'_> = context;
    context.execution().check()?;
    #[cfg(feature = "blas")]
    if let Some(dispatch) = context.blas()
        && let Some(result) = crate::blas::dispatch::rotation_generator(
            a,
            b,
            dispatch,
            context.accuracy(),
            crate::kernel::provider_policy_workers(context),
        )?
    {
        return Ok(result);
    }
    if a == R::ZERO && b == R::ZERO {
        return Ok(GivensRotationResult {
            r: R::ZERO,
            z: R::ZERO,
            rotation: GivensRotation::new(R::ONE, R::ZERO)?,
        });
    }
    let (rotation, mut r) = crate::rotation::Rotation::generate(a, b)?;
    let (mut c, mut s) = (rotation.c, rotation.s);
    let roe = if a.abs() > b.abs() { a } else { b };
    if r.copysign(roe) != r {
        r = -r;
        c = -c;
        s = -s;
    }
    let z = if a.abs() > b.abs() {
        s
    } else if c != R::ZERO {
        crate::checked::div(R::ONE, c)?
    } else {
        R::ONE
    };
    Ok(GivensRotationResult {
        r,
        z,
        rotation: GivensRotation::new(c, s)?,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    #[test]
    fn reductions_follow_blas_component_rules() {
        let values = [
            Complex::new(9.0_f64, 0.0),
            Complex::new(8.0, -8.0),
            Complex::new(-8.0, 8.0),
        ];
        let view = VectorView::from_slice(&values);
        assert_eq!(
            view.asum_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap(),
            41.0
        );
        assert_eq!(
            view.iamax_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap(),
            Some(1)
        );
        assert_eq!(
            VectorView::from_slice(&[] as &[f32])
                .iamax_scoped(
                    &crate::LinalgExecutionOptions::new(),
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap(),
            None
        );
    }

    #[test]
    fn iamax_compares_absolute_sums_exactly() {
        let index = |values: &[Complex<f64>]| VectorView::from_slice(values).iamax().unwrap();
        // Equal sums 139: the first entry wins.
        assert_eq!(
            index(&[Complex::new(16.0, 123.0), Complex::new(65.0, 74.0)]),
            Some(0)
        );
        // Exact sums 4361241660720539 < 4361241660720540, both above 2^52.
        assert_eq!(
            index(&[
                Complex::new(3_210_835_642_594_549.0, 1_150_406_018_125_990.0),
                Complex::new(302_957_924_514_085.0, 4_058_283_736_206_455.0),
            ]),
            Some(1)
        );
        // Sums beyond the largest finite value differ only in the smaller component.
        let max = f64::MAX;
        assert_eq!(
            index(&[
                Complex::new(max, max * 0.5),
                Complex::new(max * 0.5 + 2.0_f64.powi(970), max),
                Complex::new(max, 1.0),
            ]),
            Some(1)
        );
        // An overflowing sum outranks every representable one.
        assert_eq!(
            index(&[Complex::new(max, 0.0), Complex::new(max * 0.75, max * 0.75)]),
            Some(1)
        );
        // Subnormal components still separate sums near the largest value.
        let tiny = f64::from_bits(1);
        assert_eq!(
            index(&[Complex::new(max * 0.5, 0.0), Complex::new(max * 0.5, tiny)]),
            Some(1)
        );
    }

    #[test]
    fn rotation_is_transactional_for_strided_vectors() {
        let mut left = [3.0_f64, 123.0, 4.0, 456.0];
        let mut right = [4.0_f64, 789.0, -3.0, 321.0];
        let x = VectorViewMut::from_slice_with_stride(&mut left, 2, 2).unwrap();
        let y = VectorViewMut::from_slice_with_stride(&mut right, 2, 2).unwrap();
        let GivensRotationResult { rotation, .. } = givens_rotation_scoped(
            3.0,
            4.0,
            &crate::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        crate::VectorViewMut::rotate_in_place(
            rotation,
            &crate::LinalgExecutionOptions::new(),
            x,
            y,
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert!((left[0] - 5.0).abs() < 1e-14);
        assert!(right[0].abs() < 1e-14);
        assert_eq!(left[1], 123.0);
        assert_eq!(right[1], 789.0);

        let before_left = left;
        let before_right = right;
        let x = VectorViewMut::from_slice_with_stride(&mut left, 2, 2).unwrap();
        let y = VectorViewMut::from_slice_with_stride(&mut right, 2, 2).unwrap();
        assert!(
            crate::VectorViewMut::rotate_in_place(
                crate::GivensRotation::new(f64::MAX, f64::MAX).unwrap(),
                &crate::LinalgExecutionOptions::new(),
                x,
                y,
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(left, before_left);
        assert_eq!(right, before_right);
    }

    #[test]
    fn test_widened_and_offset_dot_keep_cancelled_units() {
        let x = [16_777_216.0_f32, 1.0, -16_777_216.0];
        let y = [1.0_f32; 3];
        let left = VectorView::from_slice(&x);
        let right = VectorView::from_slice(&y);
        assert_eq!(
            left.dot_wide_scoped(
                right,
                &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                crate::LinalgExecutionContext::sequential()
            )
            .unwrap(),
            1.0
        );
        assert_eq!(
            left.dot_plus_scoped(
                right,
                2.0,
                &crate::LinalgExecutionOptions::new().with_accuracy(Accuracy::Exact),
                crate::LinalgExecutionContext::sequential()
            )
            .unwrap(),
            3.0
        );
    }
}
