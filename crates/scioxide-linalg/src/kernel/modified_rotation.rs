//! Real modified Givens transformations in the BLAS flag forms.

use scioxide_core::{Dimensions, NumericalIssue, ProductSum, RealScalar, Scaled, SciOxideError};

use crate::VectorViewMut;

/// Form of a modified Givens transformation `H`, as encoded by the BLAS flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModifiedRotationFlag {
    /// `H = I` (BLAS flag `−2`).
    Identity,
    /// `H = [h11 h12; h21 h22]` with all four entries stored (flag `−1`).
    Full,
    /// `H = [1 h12; h21 1]` (flag `0`).
    UnitDiagonal,
    /// `H = [h11 1; −1 h22]` (flag `1`).
    UnitOffDiagonal,
}

/// Real modified Givens transformation `H`, a 2 × 2 matrix in one of the BLAS flag forms.
///
/// `H` maps `[x; y]` to `[h11 x + h12 y; h21 x + h22 y]`. It is computed by
/// [`modified_givens_rotation`] and applied to vectors by
/// [`VectorViewMut::modified_rotate_in_place`].
#[derive(Debug, Clone, Copy)]
pub struct ModifiedRotation<R: RealScalar> {
    flag: ModifiedRotationFlag,
    h11: R,
    h21: R,
    h12: R,
    h22: R,
}

impl<R: RealScalar> ModifiedRotation<R> {
    /// Returns the flag form.
    pub const fn flag(&self) -> ModifiedRotationFlag {
        self.flag
    }
    /// Returns `[h11, h12, h21, h22]` in row-major order, with the entries implied by the
    /// flag form filled in.
    pub fn coefficients(&self) -> [R; 4] {
        [self.h11, self.h12, self.h21, self.h22]
    }
    /// Returns the BLAS parameter array `[flag, h11, h21, h12, h22]`.
    ///
    /// `flag` is `−2`, `−1`, `0` or `1` as listed on [`ModifiedRotationFlag`]. The entries
    /// implied by the flag form are filled in.
    pub fn blas_parameters(&self) -> [R; 5] {
        let flag = match self.flag {
            ModifiedRotationFlag::Identity => -(R::TWO),
            ModifiedRotationFlag::Full => -R::ONE,
            ModifiedRotationFlag::UnitDiagonal => R::ZERO,
            ModifiedRotationFlag::UnitOffDiagonal => R::ONE,
        };
        [flag, self.h11, self.h21, self.h12, self.h22]
    }
    #[cfg(feature = "blas")]
    fn from_validated_blas_parameters(parameters: [R; 5]) -> Self {
        let flag = parameters[0];
        let two = R::TWO;
        let (flag, h11, h21, h12, h22) = if flag == -two {
            (
                ModifiedRotationFlag::Identity,
                R::ONE,
                R::ZERO,
                R::ZERO,
                R::ONE,
            )
        } else if flag == -R::ONE {
            (
                ModifiedRotationFlag::Full,
                parameters[1],
                parameters[2],
                parameters[3],
                parameters[4],
            )
        } else if flag == R::ZERO {
            (
                ModifiedRotationFlag::UnitDiagonal,
                R::ONE,
                parameters[2],
                parameters[3],
                R::ONE,
            )
        } else {
            debug_assert_eq!(flag, R::ONE);
            (
                ModifiedRotationFlag::UnitOffDiagonal,
                parameters[1],
                -R::ONE,
                R::ONE,
                parameters[4],
            )
        };
        Self {
            flag,
            h11,
            h21,
            h12,
            h22,
        }
    }
    fn apply(self, x: R, y: R) -> Result<(R, R), SciOxideError> {
        let mut first = ProductSum::new();
        first.add_product(self.h11, x)?;
        first.add_product(self.h12, y)?;
        let mut second = ProductSum::new();
        second.add_product(self.h21, x)?;
        second.add_product(self.h22, y)?;
        Ok((first.finish()?, second.finish()?))
    }
}

/// Result of [`modified_givens_rotation`]: updated weights, first coordinate and `H`.
#[derive(Debug, Clone, Copy)]
pub struct ModifiedRotationResult<R: RealScalar> {
    /// Updated weight `d1`.
    pub(crate) d1: R,
    /// Updated weight `d2`.
    pub(crate) d2: R,
    /// Updated coordinate `x1`.
    pub(crate) x1: R,
    /// Transformation `H`.
    pub(crate) rotation: ModifiedRotation<R>,
}

impl<R: RealScalar> ModifiedRotationResult<R> {
    /// Returns the updated weight `d1`.
    pub const fn d1(&self) -> R {
        self.d1
    }

    /// Returns the updated weight `d2`.
    pub const fn d2(&self) -> R {
        self.d2
    }

    /// Returns the updated first coordinate `x1`.
    pub const fn x1(&self) -> R {
        self.x1
    }

    /// Returns the transformation `H`, to apply to further vector pairs.
    pub const fn rotation(&self) -> ModifiedRotation<R> {
        self.rotation
    }
}

fn zero_result<R: RealScalar>() -> ModifiedRotationResult<R> {
    ModifiedRotationResult {
        d1: R::ZERO,
        d2: R::ZERO,
        x1: R::ZERO,
        rotation: ModifiedRotation {
            flag: ModifiedRotationFlag::Full,
            h11: R::ZERO,
            h21: R::ZERO,
            h12: R::ZERO,
            h22: R::ZERO,
        },
    }
}

/// Computes the modified Givens transformation that zeroes the second entry of
/// `(√d1 x1, √d2 y1)`.
///
/// The weights `d1`, `d2` and coordinates `x1`, `y1` describe the vector
/// `(√d1 x1, √d2 y1)`. The result holds `H` with `H [x1; y1] = [x1'; 0]` and the new weights
/// `d1'`, `d2'`, so that `√d1' x1'` is the norm of that vector up to sign. Working with `H`
/// and the weights instead of the rotation needs no square roots (Gentleman, 1973). The
/// convention is that of BLAS ROTMG:
///
/// - if `d2 y1 = 0`, `H` is the identity and the inputs are returned unchanged;
/// - if `d1 < 0`, or the update would make a weight negative, the result is zero: `H = 0` in
///   the `Full` form and `d1' = d2' = x1' = 0`;
/// - `d1'` and `|d2'|` are rescaled by powers of `γ² = 2²⁴` into `(γ⁻², γ²)`, which puts `H`
///   in the `Full` form.
///
/// Extreme inputs use independent binary exponents until the final conversion,
/// so intermediate underflow cannot select the identity transformation.
///
/// # Errors
///
/// - `NonFinite` if `d1`, `d2`, `x1` or `y1` is NaN or infinite.
/// - `NumericalInstability` if a final weight, coordinate or coefficient overflows
///   or a nonzero extended-range result rounds to zero, or ordinary rescaling
///   does not finish within 128 steps.
/// - Execution errors as described in [Execution](crate#execution).
///
/// # Examples
///
/// ```
/// use scioxide_linalg::modified_givens_rotation;
/// let result = modified_givens_rotation(
///     1.0_f64,
///     1.0,
///     3.0,
///     4.0)?;
/// assert!(result.d1().is_finite() && result.d2().is_finite());
/// // √d1' x1' = ‖(3, 4)‖₂ = 5.
/// assert!((result.d1().sqrt() * result.x1() - 5.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - W. M. Gentleman, "Least squares computations by Givens transformations without square
///   roots", *J. Inst. Maths Applics* 12, 1973.
/// - C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T. Krogh, "Basic Linear Algebra
///   Subprograms for Fortran usage", *ACM Transactions on Mathematical Software* 5, 1979.
pub fn modified_givens_rotation<R: RealScalar>(
    d1: R,
    d2: R,
    x1: R,
    y1: R,
) -> Result<ModifiedRotationResult<R>, SciOxideError> {
    modified_givens_rotation_scoped(
        d1,
        d2,
        x1,
        y1,
        &crate::LinalgExecutionOptions::default(),
        scioxide_core::ExecutionContext::sequential_native(),
    )
}

/// Computes [`modified_givens_rotation`](modified_givens_rotation) with `options` in the caller's
/// execution context, which can select [`Accuracy::Exact`](scioxide_core::Accuracy::Exact), attach a
/// BLAS provider or stop the computation.
///
/// # Errors
///
/// The errors of [`modified_givens_rotation`](modified_givens_rotation), and:
///
/// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
pub fn modified_givens_rotation_scoped<'scope, 'a, R: RealScalar>(
    mut d1: R,
    mut d2: R,
    mut x1: R,
    y1: R,
    options: &crate::LinalgExecutionOptions,
    context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
) -> Result<ModifiedRotationResult<R>, SciOxideError> {
    let context = context.into().with_options(options)?.kernel();

    for (name, value) in [("d1", d1), ("d2", d2), ("x1", x1), ("y1", y1)] {
        crate::checked::validate_input(value, name)?;
    }
    let context: crate::KernelContext<'_> = context;
    context.execution().check()?;
    // Eighth-root bounds leave headroom for the products, ratios and updates
    // before ROTMG's binary rescaling. Outside this interval retain exponents
    // throughout, including when an external provider is attached.
    let lower = R::MIN_POSITIVE.sqrt().sqrt().sqrt();
    let upper = R::MAX.sqrt().sqrt().sqrt();
    if [d1, d2, x1, y1]
        .iter()
        .any(|x| *x != R::ZERO && (x.abs() < lower || x.abs() > upper))
    {
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas() {
            let _ = crate::blas::dispatch::unsupported(dispatch)?;
        }
        return scaled_generator(d1, d2, x1, y1);
    }
    #[cfg(feature = "blas")]
    if let Some(dispatch) = context.blas()
        && let Some((next_d1, next_d2, next_x1, parameters)) =
            crate::blas::dispatch::modified_rotation_generator(
                d1,
                d2,
                x1,
                y1,
                dispatch,
                context.accuracy(),
                crate::kernel::provider_policy_workers(context),
            )?
    {
        return Ok(ModifiedRotationResult {
            d1: next_d1,
            d2: next_d2,
            x1: next_x1,
            rotation: ModifiedRotation::from_validated_blas_parameters(parameters),
        });
    }
    if d1 < R::ZERO {
        return Ok(zero_result());
    }
    let p2 = crate::checked::mul(d2, y1)?;
    if p2 == R::ZERO {
        return Ok(ModifiedRotationResult {
            d1,
            d2,
            x1,
            rotation: ModifiedRotation {
                flag: ModifiedRotationFlag::Identity,
                h11: R::ONE,
                h21: R::ZERO,
                h12: R::ZERO,
                h22: R::ONE,
            },
        });
    }
    let p1 = crate::checked::mul(d1, x1)?;
    let q2 = crate::checked::mul(p2, y1)?;
    let q1 = crate::checked::mul(p1, x1)?;
    let (mut flag, mut h11, mut h21, mut h12, mut h22);
    if q1.abs() > q2.abs() {
        h11 = R::ONE;
        h22 = R::ONE;
        h21 = crate::checked::div(-y1, x1)?;
        h12 = crate::checked::div(p2, p1)?;
        let mut sum = ProductSum::new();
        sum.add_product(R::ONE, R::ONE)?;
        sum.subtract_product(h12, h21)?;
        let u = sum.finish()?;
        if u <= R::ZERO {
            return Ok(zero_result());
        }
        flag = ModifiedRotationFlag::UnitDiagonal;
        d1 = crate::checked::div(d1, u)?;
        d2 = crate::checked::div(d2, u)?;
        x1 = crate::checked::mul(x1, u)?;
    } else {
        if q2 < R::ZERO {
            return Ok(zero_result());
        }
        flag = ModifiedRotationFlag::UnitOffDiagonal;
        h11 = crate::checked::div(p1, p2)?;
        h21 = -R::ONE;
        h12 = R::ONE;
        h22 = crate::checked::div(x1, y1)?;
        let mut sum = ProductSum::new();
        sum.add_product(R::ONE, R::ONE)?;
        sum.add_product(h11, h22)?;
        let u = sum.finish()?;
        let next_d1 = crate::checked::div(d2, u)?;
        d2 = crate::checked::div(d1, u)?;
        d1 = next_d1;
        x1 = crate::checked::mul(y1, u)?;
    }

    let gamma = R::try_from_f64_exact(4096.0)?;
    let gamma_squared = crate::checked::mul(gamma, gamma)?;
    let reciprocal_squared = crate::checked::div(R::ONE, gamma_squared)?;
    for _ in 0..128 {
        if d1 == R::ZERO || d1 > reciprocal_squared && d1 < gamma_squared {
            break;
        }
        if flag == ModifiedRotationFlag::UnitDiagonal {
            h11 = R::ONE;
            h22 = R::ONE;
        } else if flag == ModifiedRotationFlag::UnitOffDiagonal {
            h21 = -R::ONE;
            h12 = R::ONE;
        }
        flag = ModifiedRotationFlag::Full;
        if d1 <= reciprocal_squared {
            d1 = crate::checked::mul(d1, gamma_squared)?;
            x1 = crate::checked::div(x1, gamma)?;
            h11 = crate::checked::div(h11, gamma)?;
            h12 = crate::checked::div(h12, gamma)?;
        } else {
            d1 = crate::checked::div(d1, gamma_squared)?;
            x1 = crate::checked::mul(x1, gamma)?;
            h11 = crate::checked::mul(h11, gamma)?;
            h12 = crate::checked::mul(h12, gamma)?;
        }
    }
    if d1 != R::ZERO && (d1 <= reciprocal_squared || d1 >= gamma_squared) {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "modified rotation rescaling did not finish",
        ));
    }
    for _ in 0..128 {
        if d2 == R::ZERO || d2.abs() > reciprocal_squared && d2.abs() < gamma_squared {
            break;
        }
        if flag == ModifiedRotationFlag::UnitDiagonal {
            h11 = R::ONE;
            h22 = R::ONE;
        } else if flag == ModifiedRotationFlag::UnitOffDiagonal {
            h21 = -R::ONE;
            h12 = R::ONE;
        }
        flag = ModifiedRotationFlag::Full;
        if d2.abs() <= reciprocal_squared {
            d2 = crate::checked::mul(d2, gamma_squared)?;
            h21 = crate::checked::div(h21, gamma)?;
            h22 = crate::checked::div(h22, gamma)?;
        } else {
            d2 = crate::checked::div(d2, gamma_squared)?;
            h21 = crate::checked::mul(h21, gamma)?;
            h22 = crate::checked::mul(h22, gamma)?;
        }
    }
    if d2 != R::ZERO && (d2.abs() <= reciprocal_squared || d2.abs() >= gamma_squared) {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "modified rotation rescaling did not finish",
        ));
    }
    Ok(ModifiedRotationResult {
        d1,
        d2,
        x1,
        rotation: ModifiedRotation {
            flag,
            h11,
            h21,
            h12,
            h22,
        },
    })
}

// ROTMG in independent binary exponents. Only the published coefficients and
// coordinates are projected; a nonzero product cannot turn into a zero branch.
fn scaled_generator<R: RealScalar>(
    d1: R,
    d2: R,
    x1: R,
    y1: R,
) -> Result<ModifiedRotationResult<R>, SciOxideError> {
    if d1 < R::ZERO {
        return Ok(zero_result());
    }
    if d2 == R::ZERO || y1 == R::ZERO {
        return Ok(ModifiedRotationResult {
            d1,
            d2,
            x1,
            rotation: ModifiedRotation {
                flag: ModifiedRotationFlag::Identity,
                h11: R::ONE,
                h12: R::ZERO,
                h21: R::ZERO,
                h22: R::ONE,
            },
        });
    }
    let (mut d1, mut d2, mut x1, y1) = (
        Scaled::from_value(d1)?,
        Scaled::from_value(d2)?,
        Scaled::from_value(x1)?,
        Scaled::from_value(y1)?,
    );
    let one = Scaled::from_value(R::ONE)?;
    let zero = Scaled::ZERO;
    let p1 = d1.mul(x1)?;
    let p2 = d2.mul(y1)?;
    let q1 = p1.mul(x1)?;
    let q2 = p2.mul(y1)?;
    let (mut flag, mut h11, mut h12, mut h21, mut h22);
    if q1.abs_cmp(q2).is_gt() {
        h11 = one;
        h22 = one;
        h21 = (-y1).div(x1)?;
        h12 = p2.div(p1)?;
        let u = one.difference_of_products(one, h12, h21)?;
        if !u.cmp_value(zero).is_gt() {
            return Ok(zero_result());
        }
        flag = ModifiedRotationFlag::UnitDiagonal;
        d1 = d1.div(u)?;
        d2 = d2.div(u)?;
        x1 = x1.mul(u)?;
    } else {
        if q2.cmp_value(zero).is_lt() {
            return Ok(zero_result());
        }
        flag = ModifiedRotationFlag::UnitOffDiagonal;
        h11 = p1.div(p2)?;
        h12 = one;
        h21 = -one;
        h22 = x1.div(y1)?;
        let u = one.difference_of_products(one, -h11, h22)?;
        let next_d1 = d2.div(u)?;
        d2 = d1.div(u)?;
        d1 = next_d1;
        x1 = y1.mul(u)?;
    }
    let lower = one.shifted(-24)?;
    let upper = one.shifted(24)?;
    // Each step changes a weight exponent by 24; finite IEEE inputs and their
    // fixed number of products/quotients bound the number of steps.
    while !d1.is_zero() && (d1.abs_le(lower) || !d1.abs_cmp(upper).is_lt()) {
        flag = ModifiedRotationFlag::Full;
        let shift = if d1.abs_le(lower) { 12 } else { -12 };
        d1 = d1.shifted(2 * shift)?;
        x1 = x1.shifted(-shift)?;
        h11 = h11.shifted(-shift)?;
        h12 = h12.shifted(-shift)?;
    }
    while !d2.is_zero() && (d2.abs_le(lower) || !d2.abs_cmp(upper).is_lt()) {
        flag = ModifiedRotationFlag::Full;
        let shift = if d2.abs_le(lower) { 12 } else { -12 };
        d2 = d2.shifted(2 * shift)?;
        h21 = h21.shifted(-shift)?;
        h22 = h22.shifted(-shift)?;
    }
    Ok(ModifiedRotationResult {
        d1: d1.try_value()?,
        d2: d2.try_value()?,
        x1: x1.try_value()?,
        rotation: ModifiedRotation {
            flag,
            h11: h11.try_value()?,
            h12: h12.try_value()?,
            h21: h21.try_value()?,
            h22: h22.try_value()?,
        },
    })
}

impl<R: RealScalar> VectorViewMut<'_, R> {
    /// Applies a modified Givens transformation `H` to every pair `(xᵢ, yᵢ)` of `first` and
    /// `other`.
    ///
    /// Each pair becomes `[xᵢ; yᵢ] := H [xᵢ; yᵢ]` (BLAS ROTM), with each new entry rounded
    /// once. Reserve
    /// [`DenseWorkspaceRequest::vector_pair`](crate::DenseWorkspaceRequest::vector_pair) in
    /// `workspace`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the vectors differ in length.
    /// - `NonFinite` if an entry of either vector is NaN or infinite.
    /// - `NumericalInstability` if a transformed entry overflows.
    /// - Workspace and execution errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces) and [Execution](crate#execution).
    /// - `CapabilityUnavailable` or `CapabilityFailure` from an attached BLAS provider.
    pub fn modified_rotate_in_place<'scope, 'a>(
        rotation: ModifiedRotation<R>,
        options: &crate::LinalgExecutionOptions,
        mut first: Self,
        mut other: Self,
        workspace: &mut crate::LinalgWorkspace<'_, R>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into().with_options(options)?.kernel();
        context.execution().check()?;
        let request =
            crate::DenseWorkspaceRequest::vector_pair(first.len(), options).for_kernels(context);
        workspace.try_reserve(&crate::LinalgWorkspace::<R>::workspace_requirements(
            &request,
        )?)?;
        let (left, right) = crate::workspace::split_mut(
            workspace.scalar_mut(crate::kernel::CANDIDATE),
            first.len(),
        )?;
        if first.len() != other.len() || left.len() < first.len() || right.len() < first.len() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::two(first.len(), first.len()),
                Dimensions::two(other.len(), left.len().min(right.len())),
            ));
        }
        let _ = first.as_view().try_finite()?;
        let _ = other.as_view().try_finite()?;
        let context: crate::KernelContext<'_> = context;
        context.execution().check()?;
        #[cfg(feature = "blas")]
        if let Some(dispatch) = context.blas() {
            for index in 0..first.len() {
                left[index] = first.as_view().get_element(index);
                right[index] = other.as_view().get_element(index);
            }
            if crate::blas::dispatch::rotate(crate::blas::dispatch::RotationRequest {
                blas_operation: crate::BlasOperation::Rotm,
                x: &mut left[..first.len()],
                y: &mut right[..first.len()],
                cosine: R::ZERO,
                sine: R::ZERO,
                parameters: Some(rotation.blas_parameters()),
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
            let (x, y) = rotation.apply(
                first.as_view().get_element(index),
                other.as_view().get_element(index),
            )?;
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn modified_generator_matches_reference_flag_forms() {
        let result = modified_givens_rotation_scoped(
            1.0_f64,
            1.0,
            3.0,
            4.0,
            &crate::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(
            result.rotation.flag(),
            ModifiedRotationFlag::UnitOffDiagonal
        );
        assert_eq!(result.rotation.coefficients(), [0.75, 1.0, -1.0, 0.75]);
        assert_eq!(result.rotation.apply(3.0, 4.0).unwrap(), (6.25, 0.0));
        assert_eq!(result.x1, 6.25);

        let identity = modified_givens_rotation_scoped(
            1.0_f32,
            0.0,
            3.0,
            4.0,
            &crate::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(identity.rotation.flag(), ModifiedRotationFlag::Identity);
        assert_eq!(identity.rotation.apply(3.0, 4.0).unwrap(), (3.0, 4.0));
    }

    #[test]
    fn extreme_weights_preserve_the_weighted_norm_and_annihilate_the_second_entry() {
        fn check<R: RealScalar>(d: R, x: R) {
            for ratio in [R::ONE, R::TWO] {
                let first_input = ratio * x;
                let result = modified_givens_rotation(d, d, first_input, x).unwrap();
                assert_ne!(result.rotation.flag(), ModifiedRotationFlag::Identity);
                let (first, second) = result.rotation.apply(first_input, x).unwrap();
                let expected = d.sqrt() * x * ratio.hypot(R::ONE);
                let actual = result.d1.sqrt() * result.x1;
                let tolerance = R::try_from_f64_exact(32.0).unwrap() * R::EPSILON;
                assert!((actual / expected - R::ONE).abs() <= tolerance);
                assert!((first / result.x1 - R::ONE).abs() <= tolerance);
                assert!(second.abs() <= tolerance * first.abs());
            }
            let identity = modified_givens_rotation(d, R::ZERO, x, x).unwrap();
            assert_eq!(identity.rotation.flag(), ModifiedRotationFlag::Identity);
            let zero = modified_givens_rotation(-d, d, x, x).unwrap();
            assert_eq!(zero.d1, R::ZERO);
            assert_eq!(zero.x1, R::ZERO);
        }
        check(1e-300_f64, 1e-100);
        check(1e300_f64, 1e100);
        check(1e-30_f32, 1e-20);
        check(1e30_f32, 1e10);
    }

    #[test]
    fn extended_rotation_preserves_flags_and_reports_unrepresentable_outputs() {
        for (d1, d2, x, y) in [(2.0_f64, 1.0, 3.0, 4.0), (1.0, -1.0, 4.0, 3.0)] {
            let expected = modified_givens_rotation(d1, d2, x, y).unwrap();
            let actual = scaled_generator(d1, d2, x, y).unwrap();
            assert_eq!(actual.rotation.flag(), expected.rotation.flag());
            for (a, b) in actual
                .rotation
                .coefficients()
                .into_iter()
                .zip(expected.rotation.coefficients())
            {
                assert!((a - b).abs() <= 32.0 * f64::EPSILON * b.abs().max(1.0));
            }
        }
        let error = modified_givens_rotation(1e300_f64, 1e300, 1e300, 1e300).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn modified_application_is_transactional_for_strides() {
        let rotation = modified_givens_rotation_scoped(
            1.0_f64,
            1.0,
            3.0,
            4.0,
            &crate::LinalgExecutionOptions::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap()
        .rotation;
        let mut left = [3.0, 99.0, 6.0];
        let mut right = [4.0, 88.0, 8.0];
        let x = VectorViewMut::from_slice_with_stride(&mut left, 2, 2).unwrap();
        let y = VectorViewMut::from_slice_with_stride(&mut right, 2, 2).unwrap();
        crate::VectorViewMut::modified_rotate_in_place(
            rotation,
            &crate::LinalgExecutionOptions::new(),
            x,
            y,
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(left, [6.25, 99.0, 12.5]);
        assert_eq!(right, [0.0, 88.0, 0.0]);
    }
}
