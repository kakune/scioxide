//! Least-squares detrending of sampled signals.
use crate::util::{capacity, invalid, scalar_from_usize_rounded as scalar};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::try_filled;
use scioxide_core::{ExecutionContext, NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{Scaled, ScaledAccumulator};
/// Trend that [`detrend`] and [`detrend_into`] remove from a record or from each
/// segment.
#[derive(Clone, Copy, Debug)]
pub enum DetrendKind {
    /// The arithmetic mean of the segment.
    Constant,
    /// The least-squares straight line `α + β k` in the sample index `k`. A
    /// segment of one sample uses its mean.
    Linear,
}
/// Removes the least-squares constant or linear trend from a record.
///
/// The result is the residual `x[k] − trend(k)` of the fit of `kind` to all of
/// `input`: its mean for [`DetrendKind::Constant`] and the straight line
/// `α + β k` in the sample index `k` for [`DetrendKind::Linear`]. A record of
/// one sample gives zero and an empty record gives an empty result. Complex
/// samples are fitted componentwise. The cost is `O(n)` for `n` samples, and
/// the accuracy is that of [`detrend_into`], which also detrends segments
/// between breakpoints. This allocates the result and runs with
/// [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` if a residual overflows.
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{DetrendKind, detrend};
/// // The deviations from the line 1 + 2k have no mean and no slope, so they remain.
/// let residual = detrend(&[1.0_f64, 3.5, 4.0, 7.5, 9.0], DetrendKind::Linear)?;
/// for (r, d) in residual.iter().zip([0.0, 0.5, -1.0, 0.5, 0.0]) {
///     assert!((r - d).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn detrend<T: NumericScalar>(input: &[T], kind: DetrendKind) -> Result<Vec<T>, SciOxideError> {
    let mut output = try_filled(
        input.len(),
        T::ZERO,
        scioxide_core::quantity!("detrend_output"),
    )?;
    detrend_into(
        input,
        kind,
        &[],
        &mut output,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
/// Removes the least-squares constant or linear trend from each segment of `input`.
///
/// `breakpoints` are strictly increasing indices in `1..input.len()` that start
/// new segments; `0` and `input.len()` are implicit. Each segment is fitted
/// independently, and `output[k]` receives the residual `x[k] − trend(k)`. A
/// segment of one sample gives zero, and an empty input with no breakpoints is
/// valid. Complex samples are fitted componentwise. `output` must hold at least
/// `input.len()` values, and only that prefix is written.
///
/// The fit accumulates moments in the sample index centered on the segment
/// midpoint. The cost is O(n) for `n` samples. Each segment is fitted twice,
/// once to check it before any output is written. Execution is sequential.
///
/// # Accuracy
///
/// For a segment of length `n`, each component of each residual has absolute
/// error at most `γ(8n + 32) · M + (8n + 32) η`, where `M` is the largest
/// component magnitude in the segment, `γ(k) = k u / (1 − k u)` with
/// `u = ε / 2`, `η` is the smallest positive subnormal, and `(8n + 32) u < 1`.
/// Segments whose moments would overflow are fitted in extended-exponent
/// arithmetic, so only a residual outside the range of the scalar type fails.
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `breakpoints` is not strictly increasing
///   or a breakpoint is outside `1..input.len()`.
/// - `DimensionMismatch` if `output` is shorter than `input`.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` if a residual overflows.
/// - `ExecutionStopped` on cancellation or a passed deadline.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{DetrendKind, detrend_into};
/// let mut output = [0.0; 3];
/// detrend_into(
///     &[1.0_f64, 2.0, 3.0],
///     DetrendKind::Linear,
///     &[],
///     &mut output,
///     ExecutionContext::sequential_native(),
/// )?;
/// // A straight line has zero residual.
/// assert!(output.iter().all(|y| y.abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn detrend_into<T: NumericScalar>(
    input: &[T],
    kind: DetrendKind,
    breakpoints: &[usize],
    output: &mut [T],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    capacity(input.len(), output.len())?;
    check_breakpoints(input.len(), breakpoints)?;
    context.check()?;
    for_each_segment(input, breakpoints, |_, segment| {
        detrend_segment(segment, kind, |_, _| {})
    })?;
    context.check()?;
    for_each_segment(input, breakpoints, |start, segment| {
        detrend_segment(segment, kind, |i, value| output[start + i] = value)
    })
}

/// Crate-internal detrending staged through `input.len()` scratch, used by
/// spectral and STFT preprocessing that already owns staging storage.
pub(crate) fn detrend_staged<T: NumericScalar>(
    input: &[T],
    kind: DetrendKind,
    breakpoints: &[usize],
    output: &mut [T],
    scratch: &mut [T],
) -> Result<(), SciOxideError> {
    capacity(input.len(), output.len())?;
    capacity(input.len(), scratch.len())?;
    check_breakpoints(input.len(), breakpoints)?;
    for_each_segment(input, breakpoints, |start, segment| {
        detrend_segment(segment, kind, |i, value| scratch[start + i] = value)
    })?;
    output[..input.len()].copy_from_slice(&scratch[..input.len()]);
    Ok(())
}
fn check_breakpoints(len: usize, breakpoints: &[usize]) -> Result<(), SciOxideError> {
    let mut previous = 0;
    for &b in breakpoints {
        if b <= previous || b >= len {
            return Err(invalid(
                scioxide_core::argument!("breakpoints"),
                "breakpoints must be strictly increasing interior indices",
            ));
        }
        previous = b;
    }
    Ok(())
}
/// Visit each nonempty segment with its start index.
fn for_each_segment<T>(
    input: &[T],
    breakpoints: &[usize],
    mut visit: impl FnMut(usize, &[T]) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let mut start = 0;
    for end in breakpoints
        .iter()
        .copied()
        .chain(core::iter::once(input.len()))
    {
        if end > start {
            visit(start, &input[start..end])?;
        }
        start = end;
    }
    Ok(())
}
/// Accumulate both components and moments together, then emit residuals in
/// index order through `emit`. A rejected working-precision frame re-emits
/// every residual from the extended-range fit, so the last value emitted for
/// each index is the result.
fn detrend_segment<T: NumericScalar>(
    input: &[T],
    kind: DetrendKind,
    mut emit: impl FnMut(usize, T),
) -> Result<(), SciOxideError> {
    let n = scalar::<T::Real>(input.len())?;
    let inverse = T::Real::ONE / n;
    let midpoint = scalar::<T::Real>(input.len() - 1)? * T::Real::HALF;
    let linear = matches!(kind, DetrendKind::Linear) && input.len() > 1;
    let mut sums = [T::ZERO; 4];
    let mut cross = [T::ZERO; 4];
    let headroom = T::Real::MAX / n;
    let mut ordinary = [true; 4];
    let mut accumulate = |lane: usize, i: usize, x: T| -> Result<(), SciOxideError> {
        // This single comparison also rejects NaN and infinity. Only a declined
        // ordinary frame needs a separate input-error classification pass.
        ordinary[lane] &= x.real_component().abs() <= headroom;
        if !T::IS_REAL {
            ordinary[lane] &= x.imaginary_component().abs() <= headroom;
        }
        sums[lane] = sums[lane] + x;
        if linear {
            let u = (T::Real::try_from_f64_lossy(i as f64)? - midpoint) * inverse;
            cross[lane] = cross[lane] + x * T::from_real(u);
        }
        Ok(())
    };
    let mut chunks = input.chunks_exact(4);
    for (block, chunk) in chunks.by_ref().enumerate() {
        for (lane, &x) in chunk.iter().enumerate() {
            accumulate(lane, block * 4 + lane, x)?;
        }
    }
    for (lane, &x) in chunks.remainder().iter().enumerate() {
        accumulate(lane, input.len() / 4 * 4 + lane, x)?;
    }
    let ordinary = ordinary.into_iter().all(core::convert::identity);
    if !ordinary {
        crate::util::inputs(input)?;
    }
    let mean = ((sums[0] + sums[1]) + (sums[2] + sums[3])) * T::from_real(inverse);
    let slope = if linear {
        let norm = (n / T::Real::try_from_f64_exact(12.)?) * (T::Real::ONE - inverse * inverse);
        ((cross[0] + cross[1]) + (cross[2] + cross[3])) / T::from_real(norm)
    } else {
        T::ZERO
    };
    if ordinary && mean.has_finite_components() && slope.has_finite_components() {
        let mut finite = true;
        for (i, &x) in input.iter().enumerate() {
            let trend = if linear {
                slope * T::from_real((T::Real::try_from_f64_lossy(i as f64)? - midpoint) * inverse)
            } else {
                T::ZERO
            };
            let y = (x - mean) - trend;
            finite &= y.has_finite_components();
            emit(i, y);
        }
        if finite {
            return Ok(());
        }
    }
    extended_segment(input, kind, emit)
}

/// Fit each component in extended range and emit the rounded residuals.
fn extended_segment<T: NumericScalar>(
    input: &[T],
    kind: DetrendKind,
    mut emit: impl FnMut(usize, T),
) -> Result<(), SciOxideError> {
    let n = scalar::<T::Real>(input.len())?;
    let real = component_fit(input, kind, false)?;
    let imaginary = if T::IS_REAL {
        None
    } else {
        component_fit(input, kind, true)?
    };
    let midpoint = scalar::<T::Real>(input.len() - 1)? * T::Real::HALF;
    let coordinate = |i: usize| -> Result<Scaled<T::Real>, SciOxideError> {
        Scaled::from_value((scalar::<T::Real>(i)? - midpoint) / n)
    };
    let residual = |fit: Fit<T::Real>, i: usize, x: T::Real| -> Result<T::Real, SciOxideError> {
        match fit {
            None => Ok(T::Real::ZERO),
            Some((mean, slope)) => Scaled::from_value(x)?
                .sub(mean)?
                .sub(slope.mul(coordinate(i)?)?)?
                .round(),
        }
    };
    for (i, &x) in input.iter().enumerate() {
        emit(
            i,
            T::from_components(
                residual(real, i, x.real_component())?,
                residual(imaginary, i, x.imaginary_component())?,
            ),
        );
    }
    Ok(())
}

/// Extended-range `(mean, slope)` of one component; `None` when the component
/// is identically zero, whose residual is zero.
type Fit<S> = Option<(Scaled<S>, Scaled<S>)>;

/// Fit one component of a segment in extended range.
fn component_fit<T: NumericScalar>(
    input: &[T],
    kind: DetrendKind,
    imaginary: bool,
) -> Result<Fit<T::Real>, SciOxideError> {
    let component = |x: T| {
        if imaginary {
            x.imaginary_component()
        } else {
            x.real_component()
        }
    };
    if input.iter().all(|&x| component(x) == T::Real::ZERO) {
        return Ok(None);
    }
    let n = scalar::<T::Real>(input.len())?;
    let count = Scaled::from_value(n)?;
    let midpoint = scalar::<T::Real>(input.len() - 1)? * T::Real::HALF;
    let coordinate = |i: usize| -> Result<Scaled<T::Real>, SciOxideError> {
        Scaled::from_value((scalar::<T::Real>(i)? - midpoint) / n)
    };
    let mut sum = ScaledAccumulator::ZERO;
    for &x in input {
        sum.add_value(component(x))?;
    }
    let mean = sum.total()?.div(count)?;
    let slope = if matches!(kind, DetrendKind::Linear) && input.len() > 1 {
        let mut cross = ScaledAccumulator::ZERO;
        let mut norm = ScaledAccumulator::ZERO;
        for (i, &x) in input.iter().enumerate() {
            let u = coordinate(i)?;
            cross.add_product(Scaled::from_value(component(x))?.sub(mean)?, u)?;
            norm.add_product(u, u)?;
        }
        cross.total()?.div(norm.total()?)?
    } else {
        Scaled::ZERO
    };
    Ok(Some((mean, slope)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::util::error_kind;
    use alloc::vec;
    use scioxide_core::{Complex, SciOxideErrorKind};

    fn native() -> ExecutionContext<'static> {
        ExecutionContext::sequential_native()
    }

    fn compare<T: NumericScalar>() {
        let value = |x| T::Real::try_from_f64_lossy(x).unwrap();
        for n in [1, 2, 3, 31, 4096] {
            let input: alloc::vec::Vec<_> = (0..n)
                .map(|i| {
                    T::from_components(
                        value(1024. + (i % 17) as f64 / 8.),
                        value((i % 13) as f64 - 6.),
                    )
                })
                .collect();
            for kind in [DetrendKind::Constant, DetrendKind::Linear] {
                let mut actual = vec![T::ZERO; n];
                let mut exact = actual.clone();
                detrend_into(&input, kind, &[], &mut actual, native()).unwrap();
                extended_segment(&input, kind, |i, y| exact[i] = y).unwrap();
                let ku = value((8 * n + 32) as f64) * T::Real::EPSILON * T::Real::HALF;
                let bound = ku / (T::Real::ONE - ku) * value(1026.);
                for (a, b) in actual.iter().zip(exact) {
                    assert!((a.real_component() - b.real_component()).abs() <= bound);
                    assert!((a.imaginary_component() - b.imaginary_component()).abs() <= bound);
                }
            }
        }
        let input = [
            T::from_real(T::Real::MAX),
            T::from_real(value(3e-30)),
            T::from_real(-T::Real::MAX),
        ];
        let mut output = [T::ZERO; 3];
        detrend_into(&input, DetrendKind::Constant, &[], &mut output, native()).unwrap();
        assert!(
            (output[1].real_component() / value(2e-30) - T::Real::ONE).abs()
                <= value(8.) * T::Real::EPSILON
        );
    }

    #[test]
    fn late_segment_failures_preserve_output() {
        for (input, kind) in [
            ([1., 2., f64::NAN], SciOxideErrorKind::NonFinite),
            // The mean is finite, but the residual `MAX + MAX / 3` overflows.
            (
                [-f64::MAX, -f64::MAX, f64::MAX],
                SciOxideErrorKind::NumericalInstability,
            ),
        ] {
            let mut output = [17.; 3];
            assert_eq!(
                error_kind(detrend_into(
                    &input,
                    DetrendKind::Constant,
                    &[],
                    &mut output,
                    native()
                )),
                kind
            );
            assert_eq!(output, [17.; 3]);
        }
        let mut output = [17.; 3];
        assert_eq!(
            error_kind(detrend_into(
                &[1., 2., f64::NAN],
                DetrendKind::Linear,
                &[1],
                &mut output,
                native()
            )),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(output, [17.; 3]);
    }

    #[test]
    fn staged_and_direct_detrending_agree() {
        let input = [3.0_f64, -1.0, 4.0, 1.0, -5.0, 9.0];
        let mut direct = [0.0; 6];
        let mut staged = [0.0; 6];
        let mut scratch = [0.0; 6];
        detrend_into(&input, DetrendKind::Linear, &[2], &mut direct, native()).unwrap();
        detrend_staged(&input, DetrendKind::Linear, &[2], &mut staged, &mut scratch).unwrap();
        assert_eq!(direct, staged);
        let cancelled = scioxide_core::CancellationToken::new();
        cancelled.cancel();
        let mut untouched = [7.0; 6];
        assert_eq!(
            error_kind(detrend_into(
                &input,
                DetrendKind::Constant,
                &[],
                &mut untouched,
                ExecutionContext::sequential().with_cancellation(&cancelled)
            )),
            SciOxideErrorKind::ExecutionStopped
        );
        assert_eq!(untouched, [7.0; 6]);
    }

    #[test]
    fn fused_moments_match_extended_reference() {
        compare::<f32>();
        compare::<f64>();
        compare::<Complex<f32>>();
        compare::<Complex<f64>>();
    }
}
