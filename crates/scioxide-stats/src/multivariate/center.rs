//! Compensated column means and centered copies of data matrices.
use super::storage::reserve_multivariate;
use super::{Extended, PreciseCenter, extended, project};
use crate::arithmetic::{native, promote, shift, shifted_center};
use crate::error_helpers::invalid;
use alloc::vec::Vec;
use scioxide_core::{
    AlgebraicScalar, ExecutionContext, NumericScalar, RealScalar, Scaled, SciOxideError,
};
use scioxide_linalg::{Matrix, MatrixView};

pub(super) fn column_centers<T: NumericScalar>(
    data: MatrixView<'_, T>,
    weights: Option<&[T::Real]>,
    count: u64,
    context: ExecutionContext<'_>,
) -> Result<(Vec<T>, Vec<PreciseCenter>), SciOxideError> {
    let mut means = scioxide_core::backend::try_filled(
        data.cols(),
        T::ZERO,
        scioxide_core::quantity!("multivariate_means"),
    )?;
    let mut centers = scioxide_core::backend::try_filled(
        data.cols(),
        PreciseCenter::ZERO,
        scioxide_core::quantity!("multivariate_centers"),
    )?;
    let total = Scaled::from_value(count as f64)?;
    for (j, center) in centers.iter_mut().enumerate() {
        context.check()?;
        if let Some(native) = native_column_center(data, j, weights, count, context)? {
            means[j] = project(native.value()?)?;
            *center = native;
            continue;
        }
        let mut origin = None;
        let mut shifted = PreciseCenter::ZERO;
        for i in 0..data.rows() {
            if i % 1024 == 0 {
                context.check()?;
            }
            // Validate even zero-weight rows, as for scalar weighted moments.
            let value = extended(data.get(i, j)?)?;
            let weight = weights.map_or(T::Real::ONE, |w| w[i]);
            if weight == T::Real::ZERO {
                continue;
            }
            let base = *origin.get_or_insert(PreciseCenter::from_value(value));
            shifted.add(
                base.difference(value)?
                    .mul_real(Scaled::from_value(weight.try_to_f64_exact()?)?)?,
            )?;
        }
        let mut result = origin.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("weights"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty frequency support",
            )
        })?;
        let shift = shifted.value()?;
        result.add(Extended::from_components(
            shift.real().div(total)?,
            shift.imaginary().div(total)?,
        ))?;
        means[j] = project(result.value()?)?;
        *center = result;
    }
    Ok((means, centers))
}

// Complete ordinary columns share scalar batch compensation. Both Cartesian
// components keep their center corrections; extreme columns replay unchanged.
fn native_column_center<T: NumericScalar>(
    data: MatrixView<'_, T>,
    column: usize,
    weights: Option<&[T::Real]>,
    count: u64,
    context: ExecutionContext<'_>,
) -> Result<Option<PreciseCenter>, SciOxideError> {
    if data.rows() == 0 {
        return Ok(None);
    }
    let first_row = weights
        .and_then(|w| w.iter().position(|&w| w != T::Real::ZERO))
        .unwrap_or(0);
    let first = data.get(first_row, column)?;
    let origins = [
        first.real_component().into_f64(),
        first.imaginary_component().into_f64(),
    ];
    let mut admitted = true;
    let mut failure = None;
    let column = data.col(column)?;
    let terms = column.iter().enumerate().scan((), |(), (i, value)| {
        if i % 1024 == 0
            && let Err(error) = context.check()
        {
            failure = Some(error);
            return None;
        }
        let components = [
            value.real_component().into_f64(),
            value.imaginary_component().into_f64(),
        ];
        admitted &= components[0].is_finite() & components[1].is_finite();
        let weight = weights.map_or(1.0, |w| w[i].into_f64());
        if weight == 0.0 {
            return Some([0.0; 4]);
        }
        admitted &= native(components[0], 64) & native(components[1], 64) & native(weight, 64);
        let [rh, rl] = shift(components[0], origins[0]);
        let [ih, il] = shift(components[1], origins[1]);
        Some([weight * rh, weight * rl, weight * ih, weight * il])
    });
    let shifted = if T::IS_REAL {
        scioxide_core::compensated_reduce(terms.map(|[rh, rl, ..]| [rh, rl])).map(|[rh, rl]| {
            [
                rh,
                rl,
                scioxide_core::CompensatedSum::ZERO,
                scioxide_core::CompensatedSum::ZERO,
            ]
        })
    } else {
        scioxide_core::compensated_reduce(terms)
    };
    if let Some(error) = failure {
        return Err(error);
    }
    let Ok([rh, rl, ih, il]) = shifted else {
        return Ok(None);
    };
    if !admitted {
        return Ok(None);
    }
    let (real, imaginary) = ([rh, rl], [ih, il]);
    Ok(Some(PreciseCenter {
        real: promote(shifted_center(origins[0], real, count as f64)?)?,
        imaginary: promote(shifted_center(origins[1], imaginary, count as f64)?)?,
    }))
}

fn native_centered<T: NumericScalar>(value: T, [real, imaginary]: [[f64; 2]; 2]) -> Option<T> {
    let (r, i) = (
        value.real_component().into_f64(),
        value.imaginary_component().into_f64(),
    );
    if !native(r, 64) || !native(i, 64) {
        return None;
    }
    Some(T::from_components(
        T::Real::try_from_f64_lossy(crate::arithmetic::normalized_deviation(r, real)).ok()?,
        T::Real::try_from_f64_lossy(crate::arithmetic::normalized_deviation(i, imaginary)).ok()?,
    ))
}

pub(crate) struct CenteredData<T: NumericScalar> {
    pub(crate) means: Vec<T>,
    pub(crate) centers: Vec<PreciseCenter>,
    pub(crate) values: Matrix<T>,
}

pub(crate) fn centered<T: NumericScalar>(
    data: MatrixView<'_, T>,
    limit: usize,
    context: ExecutionContext<'_>,
) -> Result<CenteredData<T>, SciOxideError> {
    let (n, p) = (data.rows(), data.cols());
    if n < 2 || p == 0 || (n as u128) > (1_u128 << T::Real::MANTISSA_DIGITS) {
        return Err(invalid(
            scioxide_core::argument!("data"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "at least two exactly countable rows and one column required",
        ));
    }
    context.check()?;
    reserve_multivariate::<T>(
        n.checked_mul(p)
            .and_then(|v| v.checked_add(p))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_elements"))
            })?,
        p,
        0,
        limit,
    )?;
    let (means, centers) = column_centers(data, None, n as u64, context)?;
    let mut centered = Matrix::zeros(n, p)?;
    for (j, &center) in centers.iter().enumerate() {
        let native = center.native_parts();
        for i in 0..n {
            if i % 256 == 0 {
                context.check()?;
            }
            let source = data.get(i, j)?;
            let value = if let Some(value) = native.and_then(|parts| native_centered(source, parts))
            {
                value
            } else {
                project(center.difference(extended(source)?)?)?
            };
            centered.view_mut().set(i, j, value)?;
        }
    }
    Ok(CenteredData {
        means,
        centers,
        values: centered,
    })
}
