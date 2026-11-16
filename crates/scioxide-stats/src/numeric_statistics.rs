//! Sum, mean, variance and standard deviation of real or complex samples.
use scioxide_core::{NumericScalar, RealScalar, Scaled, SciOxideError};

/// Arithmetic summaries that are also defined for complex samples.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ArithmeticStatistic {
    /// Sum of the observations.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Variance `Σ |xᵢ − x̄|² / (n − ddof)`.
    Variance(usize),
    /// Square root of the variance.
    StandardDeviation(usize),
}
/// Result of an arithmetic summary: a location in the scalar type, or a real dispersion.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ArithmeticValue<T: NumericScalar> {
    /// Sum or mean, in the scalar type of the data.
    Location(T),
    /// Nonnegative real variance or standard deviation.
    Dispersion(T::Real),
}
/// Returns the compensated sum or mean of a real or complex slice in native `f64` arithmetic.
///
/// Returns `None` if a component lies outside the native range; the caller
/// then uses the extended-range path.
pub(crate) fn native_location<T: NumericScalar>(
    values: &[T],
    sum: bool,
) -> Result<Option<T>, SciOxideError> {
    if values.is_empty() {
        return Ok(sum.then_some(T::ZERO));
    }
    let native = values.iter().fold(true, |native, &value| {
        native
            & crate::arithmetic::native(value.real_component().into_f64(), 64)
            & (T::IS_REAL || crate::arithmetic::native(value.imaginary_component().into_f64(), 64))
    });
    if !native {
        return Ok(None);
    }
    let totals = if T::IS_REAL {
        scioxide_core::compensated_reduce(
            values
                .iter()
                .map(|&value| [value.real_component().into_f64()]),
        )
        .map(|[real]| [real, scioxide_core::CompensatedSum::ZERO])
    } else {
        scioxide_core::compensated_reduce(values.iter().map(|&value| {
            [
                value.real_component().into_f64(),
                value.imaginary_component().into_f64(),
            ]
        }))
    };
    let Ok([real, imaginary]) = totals else {
        return Ok(None);
    };
    let project = |value: scioxide_core::CompensatedSum<f64>| {
        T::Real::try_from_f64_lossy(if sum {
            value.total()?
        } else {
            crate::arithmetic::sum_mean(value.parts(), values.len() as f64)
        })
    };
    Ok(Some(T::from_components(
        project(real)?,
        if T::IS_REAL {
            <T::Real as scioxide_core::AlgebraicScalar>::ZERO
        } else {
            project(imaginary)?
        },
    )))
}

/// Returns the variance or standard deviation of a short real slice in native `f64` arithmetic.
///
/// Four interleaved sums estimate the mean offset from the first value. Their
/// error enters the variance quadratically, as O(n³ ε²), which is below
/// working precision for `n ≤ 1024`. The squared differences keep the exact
/// shift from the origin and use compensated summation. Values within `±64` in
/// binary exponent keep every intermediate normal in `f64`. Returns `None` in
/// all other cases, and the caller uses the extended-range path.
#[inline]
fn native_dispersion<T: NumericScalar>(
    values: &[T],
    ddof: usize,
    deviation: bool,
) -> Result<Option<T::Real>, SciOxideError> {
    if !T::IS_REAL || values.len() <= ddof || values.len() > 1024 {
        return Ok(None);
    }
    let origin = values[0].real_component().into_f64();
    if !crate::arithmetic::native(origin, 64) {
        return Ok(None);
    }
    let mut admitted = true;
    let mut shifted = [0.0; 4];
    let mut blocks = values[1..].chunks_exact(4);
    for block in blocks.by_ref() {
        for (sum, &value) in shifted.iter_mut().zip(block) {
            let x = value.real_component().into_f64();
            admitted &= crate::arithmetic::native(x, 64);
            *sum += x - origin;
        }
    }
    for &value in blocks.remainder() {
        let x = value.real_component().into_f64();
        admitted &= crate::arithmetic::native(x, 64);
        shifted[0] += x - origin;
    }
    if !admitted {
        return Ok(None);
    }
    let offset = ((shifted[0] + shifted[1]) + (shifted[2] + shifted[3])) / values.len() as f64;
    let square = |value: T| {
        let [hi, lo] = crate::arithmetic::shift(value.real_component().into_f64(), origin);
        let delta = (hi - offset) + lo;
        [delta * delta]
    };
    let squares = if values.len() <= 64 {
        // Short, admitted pilots need no staging block. The indexed reducer
        // retains exactly the ordinary reducer's four-lane arithmetic order.
        scioxide_core::backend::compensated_indexed(values.len(), |i| square(values[i]))
    } else {
        scioxide_core::compensated_reduce(values.iter().copied().map(square)).ok()
    };
    let Some([squares]) = squares else {
        return Ok(None);
    };
    let variance = squares.total()? / (values.len() - ddof) as f64;
    if !variance.is_finite() || variance < 0.0 {
        return Ok(None);
    }
    let value = if deviation { variance.sqrt() } else { variance };
    match T::Real::try_from_f64_lossy(value) {
        Ok(projected)
            if value == 0.0 || projected != <T::Real as scioxide_core::AlgebraicScalar>::ZERO =>
        {
            Ok(Some(projected))
        }
        _ => Scaled::<f64>::from_value(value)?.round_to().map(Some),
    }
}

/// Computes an arithmetic summary of real or complex observations.
///
/// # Errors
///
/// Returns an error for an empty or non-finite sample, for `n ≤ ddof`, or if
/// the result cannot be represented.
#[inline]
pub(crate) fn numeric_statistic<T: NumericScalar>(
    values: &[T],
    statistic: ArithmeticStatistic,
) -> Result<ArithmeticValue<T>, SciOxideError> {
    if values.is_empty() && matches!(statistic, ArithmeticStatistic::Sum) {
        return Ok(ArithmeticValue::Location(T::ZERO));
    }
    if matches!(
        statistic,
        ArithmeticStatistic::Sum | ArithmeticStatistic::Mean
    ) && let Some(value) =
        native_location(values, matches!(statistic, ArithmeticStatistic::Sum))?
    {
        return Ok(ArithmeticValue::Location(value));
    }
    if let ArithmeticStatistic::Variance(ddof) | ArithmeticStatistic::StandardDeviation(ddof) =
        statistic
        && let Some(value) = native_dispersion(
            values,
            ddof,
            matches!(statistic, ArithmeticStatistic::StandardDeviation(_)),
        )?
    {
        return Ok(ArithmeticValue::Dispersion(value));
    }
    let locations = crate::standardize::Locations::new(values, scioxide_core::argument!("values"))?;
    Ok(match statistic {
        ArithmeticStatistic::Sum | ArithmeticStatistic::Mean => {
            let [re, im] = locations.extended()?;
            let factor = if matches!(statistic, ArithmeticStatistic::Sum) {
                Scaled::<f64>::from_value((values.len()) as f64)?
            } else {
                Scaled::<f64>::ONE
            };
            ArithmeticValue::Location(T::from_components(
                re.value()?.mul(factor)?.round_to()?,
                if T::IS_REAL {
                    <T::Real as scioxide_core::AlgebraicScalar>::ZERO
                } else {
                    im.value()?.mul(factor)?.round_to()?
                },
            ))
        }
        ArithmeticStatistic::Variance(ddof) => {
            ArithmeticValue::Dispersion(locations.variance(values, ddof)?.round_to()?)
        }
        ArithmeticStatistic::StandardDeviation(ddof) => {
            ArithmeticValue::Dispersion(locations.variance(values, ddof)?.sqrt()?.round_to()?)
        }
    })
}
