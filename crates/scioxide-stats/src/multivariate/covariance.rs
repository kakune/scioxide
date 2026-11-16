//! Sample, frequency-weighted and shrinkage covariance estimators.
use super::center::{CenteredData, centered, column_centers};
use super::storage::reserve_multivariate;
use super::{
    Extended, MultivariateOptions, PreciseCenter, conjugate, extended, norm_squared, project,
};
use crate::error_helpers::{count as number, invalid};
use alloc::vec::Vec;
use scioxide_core::{
    AlgebraicScalar, ExecutionContext, NumericScalar, NumericalIssue, ProductSum, RealScalar,
    Scaled, ScaledAccumulator, SciOxideError,
};
use scioxide_linalg::{Matrix, MatrixView};

/// Computes one real covariance entry in `f64` with compensated products.
///
/// Returns `None` when cancellation or the exponent range makes this fast path
/// unreliable; the caller then uses the extended-range reduction.
fn native_covariance<T: NumericScalar, const WEIGHTED: bool>(
    data: MatrixView<'_, T>,
    j: usize,
    k: usize,
    weights: &[T::Real],
    centers: Option<&[PreciseCenter]>,
    divisor: f64,
) -> Result<Option<T>, SciOxideError> {
    if !T::IS_REAL {
        return Ok(None);
    }
    let center =
        |index: usize| centers.map_or(Some([0.0; 2]), |centers| centers[index].native_real());
    let (Some(left_center), Some(right_center)) = (center(j), center(k)) else {
        return Ok(None);
    };
    let (left, right) = (data.col(j)?, data.col(k)?);
    let mut admitted = true;
    let sums = scioxide_core::compensated_reduce(left.iter().zip(right.iter()).enumerate().map(
        |(i, (a, b))| {
            let w = if WEIGHTED { weights[i].into_f64() } else { 1.0 };
            if w == 0.0 {
                return [0.0; 3];
            }
            let (a, b) = (a.real_component().into_f64(), b.real_component().into_f64());
            admitted &= crate::arithmetic::native(a, 128)
                & crate::arithmetic::native(b, 128)
                & crate::arithmetic::native(w, 64);
            let a = crate::arithmetic::normalized_deviation(a, left_center);
            let b = crate::arithmetic::normalized_deviation(b, right_center);
            admitted &= crate::arithmetic::native(a, 256) & crate::arithmetic::native(b, 256);
            let product = a * b;
            let residual = a.mul_add(b, -product);
            let high = w * product;
            let low = w.mul_add(product, -high) + w * residual;
            [high, low, high.abs()]
        },
    ));
    let Ok([mut high, low, mass]) = sums else {
        return Ok(None);
    };
    if !admitted {
        return Ok(None);
    }
    high.merge(low)?;
    let value = high.total()?;
    if j != k && value.abs() <= crate::arithmetic::PRODUCT_CANCELLATION * mass.total()? {
        return Ok(None);
    }
    let Ok(value) = T::Real::try_from_f64_lossy(value / divisor) else {
        return Ok(None);
    };
    Ok(value.is_finite().then_some(T::from_real(value)))
}

/// Computes the result of [`sample_covariance_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`sample_covariance_with_options`] for the fixed default settings.
pub fn sample_covariance<T: NumericScalar>(
    data: MatrixView<'_, T>,
) -> Result<(Vec<T>, Matrix<T>), SciOxideError> {
    sample_covariance_with_options(data, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("data"),
        )
    })
}
/// Computes the column means and the sample covariance matrix of `data`.
///
/// Rows are observations and columns are variables. With `n` rows, the entry for
/// columns `j` and `k` is
///
/// ```text
/// Σ_jk = (1/(n − 1)) Σᵢ conj(x_ij − x̄_j) (x_ik − x̄_k),
/// ```
///
/// where `x̄_j` is the mean of column `j`. The result is Hermitian. The returned
/// means are rounded to the scalar type, but centering subtracts the
/// compensated means, so a large common offset does not cancel the deviations.
/// Real products are summed with compensation, and entries that cancel or leave
/// the `f64` exponent range are recomputed in extended range.
///
/// The cost is O(np²) time and O(np + p²) memory for `p` columns.
/// `options.max_bytes()` limits that memory. The rank policy is not used.
///
/// # Errors
///
/// - `InvalidInput` if `data` has fewer than two rows, no columns, or more rows
///   than the scalar type counts exactly (2⁵³ for `f64`, 2²⁴ for `f32`).
/// - `NonFinite` if `data` contains NaN or infinity.
/// - `NumericalInstability` if a mean or covariance entry overflows the scalar type.
/// - `ResourceLimit` if the memory needed exceeds `options.max_bytes()`.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::MatrixView;
/// use scioxide_stats::{MultivariateOptions, sample_covariance_with_options};
///
/// // Columns (1, 3) and (2, 6): variances 2 and 8, covariance 4.
/// let data = [1.0_f64, 2.0, 3.0, 6.0];
/// let (mean, covariance) = sample_covariance_with_options(
///     MatrixView::from_slice(2, 2, &data)?,
///     MultivariateOptions::new().with_max_bytes(4096),
/// )?;
/// assert_eq!(mean, [2.0, 4.0]);
/// assert_eq!(covariance.as_slice(), &[2.0, 4.0, 4.0, 8.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn sample_covariance_with_options<T: NumericScalar>(
    data: MatrixView<'_, T>,
    options: MultivariateOptions<T::Real>,
) -> Result<(Vec<T>, Matrix<T>), SciOxideError> {
    let max_bytes = options.max_bytes();
    let elements = data
        .rows()
        .checked_mul(data.cols())
        .and_then(|v| v.checked_mul(3))
        .and_then(|v| {
            data.cols()
                .checked_mul(data.cols())
                .and_then(|q| v.checked_add(q))
        })
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_elements"))
        })?;
    reserve_multivariate::<T>(elements, data.cols(), 0, max_bytes)?;
    let CenteredData { means, values, .. } =
        centered(data, max_bytes, ExecutionContext::sequential())?;
    let p = values.cols();
    reserve_multivariate::<T>(
        scioxide_core::backend::checked_mul(p, p, scioxide_core::quantity!("multivariate_bytes"))?,
        0,
        0,
        max_bytes,
    )?;
    let mut covariance = Matrix::zeros(p, p)?;
    let divisor = T::from_real(number::<T::Real>(values.rows() - 1)?);
    let inv_divisor = T::from_real(T::Real::ONE / number::<T::Real>(values.rows() - 1)?);
    for j in 0..p {
        for k in j..p {
            if let Some(value) = native_covariance::<T, false>(
                values.view(),
                j,
                k,
                &[],
                None,
                (values.rows() - 1) as f64,
            )? {
                covariance.view_mut().set(j, k, value)?;
                covariance.view_mut().set(k, j, value.conj())?;
                continue;
            }
            let mut sum = ProductSum::<T>::new();
            for i in 0..values.rows() {
                sum.add_product(values.view().get(i, j)?.conj(), values.view().get(i, k)?)?;
            }
            let value = match sum.finish() {
                Ok(raw) => raw / divisor,
                Err(_) => {
                    let mut normalized = ProductSum::<T>::new();
                    for i in 0..values.rows() {
                        normalized.add_triple_product(
                            values.view().get(i, j)?.conj(),
                            values.view().get(i, k)?,
                            inv_divisor,
                        )?;
                    }
                    normalized.finish()?
                }
            };
            covariance.view_mut().set(j, k, value)?;
            covariance.view_mut().set(k, j, value.conj())?;
        }
    }
    Ok((means, covariance))
}

/// Computes the column means and the unbiased frequency-weighted covariance.
///
/// Weights are nonnegative integer replication counts whose total `N` lies in
/// `[2, 2⁵³]` for `f64` or `[2, 2²⁴]` for `f32`. The divisor is `N − 1`. Rows with
/// zero weight do not contribute, but must still be finite. The cost is O(np²).
///
/// # Errors
///
/// Returns `InvalidInput` for an empty matrix, a weight count that differs from
/// the row count, or weights that are not integer counts within the limits.
pub(crate) fn frequency_covariance_with_options<T: NumericScalar>(
    data: MatrixView<'_, T>,
    weights: &[T::Real],
    max_bytes: usize,
) -> Result<(Vec<T>, Matrix<T>), SciOxideError> {
    let (n, p) = (data.rows(), data.cols());
    if n == 0 || p == 0 || weights.len() != n {
        return Err(invalid(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "frequency covariance shape or weight model",
        ));
    }
    let elements = p
        .checked_mul(p)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(p))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_elements"))
        })?;
    reserve_multivariate::<T>(elements, p, 0, max_bytes)?;
    let mut count = 0_u64;
    let max_exact = 1_u64 << T::Real::MANTISSA_DIGITS;
    for &w in weights {
        let value = w.try_to_f64_exact()?;
        if !value.is_finite() || value < 0.0 || value != value.floor() || value > max_exact as f64 {
            return Err(invalid(
                scioxide_core::argument!("weights"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "frequency weights must be nonnegative integers within the count limit",
            ));
        }
        // Each validated integer fits u64 on both 32- and 64-bit targets.
        count = count
            .checked_add(value as u64)
            .filter(|&v| v <= max_exact)
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("weights"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "frequency total outside supported range",
                )
            })?;
    }
    if count < 2 {
        return Err(invalid(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "frequency total outside supported range",
        ));
    }
    let (means, centers) =
        column_centers(data, Some(weights), count, ExecutionContext::sequential())?;
    let divisor = Scaled::from_value((count - 1) as f64)?;
    let mut covariance = Matrix::zeros(p, p)?;
    for j in 0..p {
        for k in j..p {
            if let Some(value) = native_covariance::<T, true>(
                data,
                j,
                k,
                weights,
                Some(&centers),
                (count - 1) as f64,
            )? {
                covariance.view_mut().set(j, k, value)?;
                covariance.view_mut().set(k, j, value.conj())?;
                continue;
            }
            let mut sum = PreciseCenter::ZERO;
            for (i, &w) in weights.iter().enumerate() {
                if w == T::Real::ZERO {
                    continue;
                }
                let left = centers[j].difference(extended(data.get(i, j)?)?)?;
                let right = centers[k].difference(extended(data.get(i, k)?)?)?;
                sum.add(
                    conjugate(left)
                        .mul(right)?
                        .mul_real(Scaled::from_value(w.try_to_f64_exact()?)?)?,
                )?;
            }
            let raw = sum.value()?;
            let value: T = project(Extended::from_components(
                raw.real().div(divisor)?,
                raw.imaginary().div(divisor)?,
            ))?;
            covariance.view_mut().set(j, k, value)?;
            covariance.view_mut().set(k, j, value.conj())?;
        }
    }
    Ok((means, covariance))
}

/// Shrinks a covariance matrix toward a multiple of the identity.
///
/// For a `p × p` Hermitian matrix `S` and intensity `α ∈ [0, 1]`, returns
///
/// ```text
/// Ŝ = (1 − α) S + α μ I,   μ = tr(S)/p.
/// ```
///
/// `α = 0` returns `S` unchanged and `α = 1` returns `μ I`. The trace is summed in
/// extended range. The cost is O(p²). To choose `α` from data, use
/// [`ledoit_wolf_covariance_with_options`].
///
/// # Errors
///
/// - `InvalidInput` if `covariance` is empty or not square, or if `intensity` is
///   not in `[0, 1]`.
/// - `InvalidInput` if `covariance` is not exactly Hermitian, has a negative or
///   non-real diagonal entry, or contains NaN or infinity.
pub fn shrink_covariance<T: NumericScalar>(
    covariance: MatrixView<'_, T>,
    intensity: T::Real,
) -> Result<Matrix<T>, SciOxideError> {
    let p = covariance.rows();
    if p == 0
        || covariance.cols() != p
        || !intensity.is_finite()
        || intensity < T::Real::ZERO
        || intensity > T::Real::ONE
    {
        return Err(invalid(
            scioxide_core::argument!("intensity"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "shrinkage shape or intensity",
        ));
    }
    let mut trace = ScaledAccumulator::<f64>::ZERO;
    for i in 0..p {
        let d = covariance.get(i, i)?;
        if !d.has_finite_components()
            || d.imaginary_component() != T::Real::ZERO
            || d.real_component() < T::Real::ZERO
        {
            return Err(invalid(
                scioxide_core::argument!("covariance"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid covariance diagonal",
            ));
        }
        trace.add_value(d.real_component().try_to_f64_exact()?)?;
        for j in 0..i {
            let a = covariance.get(i, j)?;
            let b = covariance.get(j, i)?;
            if !a.has_finite_components() || !b.has_finite_components() || a != b.conj() {
                return Err(invalid(
                    scioxide_core::argument!("covariance"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "covariance must be Hermitian",
                ));
            }
        }
    }
    let target = trace.total()?.div(Scaled::from_value(p as f64)?)?;
    let fraction = Scaled::from_value(intensity.try_to_f64_exact()?)?;
    let complement = Scaled::ONE.sub(fraction)?;
    let mut out = Matrix::zeros(p, p)?;
    for i in 0..p {
        for j in 0..p {
            let entry = extended(covariance.get(i, j)?)?;
            let value: T = if i == j {
                T::from_real(
                    crate::arithmetic::affine_scaled(entry.real(), target, complement, fraction)?
                        .round_to()?,
                )
            } else {
                project(entry.mul_real(complement)?)?
            };
            out.view_mut().set(i, j, value)?;
        }
    }
    Ok(out)
}

/// Computes the result of [`ledoit_wolf_covariance_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`ledoit_wolf_covariance_with_options`] for the fixed default settings.
pub fn ledoit_wolf_covariance<T: NumericScalar>(
    data: MatrixView<'_, T>,
) -> Result<(T::Real, Matrix<T>), SciOxideError> {
    ledoit_wolf_covariance_with_options(data, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("data"),
        )
    })
}
/// Computes the Ledoit–Wolf shrinkage covariance and its shrinkage intensity.
///
/// Rows of `data` are independent, unweighted observations. With `xᵢ` the
/// centered row `i`, `n` rows and `p` columns, the intensity `α` is
///
/// ```text
/// S = (1/n) Σᵢ xᵢ xᵢᴴ,        μ = tr(S)/p,
/// δ = ‖S − μ I‖²_F,           β = (1/n²) Σᵢ ‖xᵢ xᵢᴴ − S‖²_F,
/// α = min(β/δ, 1),            α = 1 if δ = 0,
/// ```
///
/// where `‖·‖_F` is the Frobenius norm. The function returns `α` and the shrunk
/// covariance `(n/(n − 1)) ((1 − α) S + α μ I)`, which uses the sample divisor
/// `n − 1`. Constant or isotropic data give `α = 1`. The fourth-order sums and
/// the ratio are computed in extended range, so `α` does not change when the
/// data are scaled by a common factor.
///
/// The cost is O(np²) time and O(np + p²) memory. `options.max_bytes()` limits
/// that memory. The rank policy is not used. To shrink with a fixed intensity,
/// use [`shrink_covariance`].
///
/// # Errors
///
/// - `InvalidInput` if `data` has fewer than two rows, no columns, or more rows
///   than the scalar type counts exactly (2⁵³ for `f64`, 2²⁴ for `f32`).
/// - `NonFinite` if `data` contains NaN or infinity.
/// - `NumericalInstability` if a covariance entry overflows the scalar type.
/// - `ResourceLimit` if the memory needed exceeds `options.max_bytes()`.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_stats::{MultivariateOptions, ledoit_wolf_covariance_with_options};
///
/// // S = 2.5 everywhere, δ = 12.5, β = 36/16, so α = 36/(16 · 12.5) = 0.18.
/// let data = Matrix::<f64>::from_vec(4, 2, vec![-2., -2., -1., -1., 1., 1., 2., 2.])?;
/// let (intensity, _) = ledoit_wolf_covariance_with_options(data.view(), MultivariateOptions::new())?;
/// assert!((intensity - 0.18_f64).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional
/// covariance matrices", *Journal of Multivariate Analysis* 88 (2004), 365–411.
pub fn ledoit_wolf_covariance_with_options<T: NumericScalar>(
    data: MatrixView<'_, T>,
    options: MultivariateOptions<T::Real>,
) -> Result<(T::Real, Matrix<T>), SciOxideError> {
    let max_bytes = options.max_bytes();
    let elements = data
        .rows()
        .checked_mul(data.cols())
        .and_then(|v| v.checked_mul(3))
        .and_then(|v| {
            data.cols()
                .checked_mul(data.cols())
                .and_then(|q| q.checked_mul(2))
                .and_then(|q| v.checked_add(q))
        })
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_elements"))
        })?;
    let p = data.cols();
    let entries = p.checked_mul(p).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_elements"))
    })?;
    reserve_multivariate::<T>(elements, p, entries, max_bytes)?;
    let CenteredData { values: x, .. } = centered(data, max_bytes, ExecutionContext::sequential())?;
    let n = x.rows();
    let nr = Scaled::from_value(n as f64)?;
    let mut mle = scioxide_core::backend::try_filled(
        entries,
        Extended::ZERO,
        scioxide_core::quantity!("multivariate_covariance"),
    )?;
    let mut trace = ScaledAccumulator::ZERO;
    for i in 0..p {
        for j in i..p {
            let mut sum = PreciseCenter::ZERO;
            for row in 0..n {
                sum.add(
                    conjugate(extended(x.view().get(row, i)?)?)
                        .mul(extended(x.view().get(row, j)?)?)?,
                )?;
            }
            let raw = sum.value()?;
            let value = Extended::from_components(raw.real().div(nr)?, raw.imaginary().div(nr)?);
            mle[i * p + j] = value;
            mle[j * p + i] = conjugate(value);
            if i == j {
                trace.add_scaled(value.real())?;
            }
        }
    }
    let target = trace.total()?.div(Scaled::from_value(p as f64)?)?;
    let mut delta = ScaledAccumulator::ZERO;
    let mut beta = ScaledAccumulator::ZERO;
    for i in 0..p {
        for j in 0..p {
            let s = mle[i * p + j];
            let target_entry =
                Extended::from_components(if i == j { target } else { Scaled::ZERO }, Scaled::ZERO);
            delta.add_scaled(norm_squared(s.sub(target_entry)?)?)?;
            for row in 0..n {
                let outer = conjugate(extended(x.view().get(row, i)?)?)
                    .mul(extended(x.view().get(row, j)?)?)?;
                beta.add_scaled(norm_squared(outer.sub(s)?)?)?;
            }
        }
    }
    let delta = delta.total()?;
    let intensity = if delta.is_zero() {
        Scaled::ONE
    } else {
        let ratio = beta.total()?.div(nr.square()?.mul(delta)?)?;
        if ratio.is_sign_negative() {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "negative Ledoit-Wolf intensity",
            ));
        }
        // Compare before projection: even an intensity above the IEEE range
        // has a well-defined clipped value, but NaN is never a clamp operand.
        if ratio.abs_le(Scaled::ONE) {
            ratio
        } else {
            Scaled::ONE
        }
    };
    let complement = Scaled::ONE.sub(intensity)?;
    let sample_factor = nr.div(Scaled::from_value((n - 1) as f64)?)?;
    let mut covariance = Matrix::zeros(p, p)?;
    for i in 0..p {
        for j in i..p {
            let s = mle[i * p + j];
            let shrunk = if i == j {
                Extended::from_components(
                    crate::arithmetic::affine_scaled(s.real(), target, complement, intensity)?,
                    Scaled::ZERO,
                )
            } else {
                s.mul_real(complement)?
            };
            let value: T = project(shrunk.mul_real(sample_factor)?)?;
            covariance.view_mut().set(i, j, value)?;
            covariance.view_mut().set(j, i, value.conj())?;
        }
    }
    Ok((intensity.round_to()?, covariance))
}
