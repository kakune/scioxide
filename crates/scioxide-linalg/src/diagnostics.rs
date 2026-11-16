//! Residual and backward-error checks of linear solutions, and the norms and
//! substitutions used by the condition estimators.
//!
//! A dense matrix is treated as a full band, so dense, band, tridiagonal and
//! triangular factors use the same code: column-sum norms, substitution with a
//! separate exponent for condition estimation, and the normwise backward error.
//! Coefficients are read only inside the band.

use core::ops::Range;

use crate::{Accuracy, KernelContext, MatrixView, VectorView};
use scioxide_core::{
    AlgebraicScalar, DiagonalKind, NumericScalar, RealScalar, ResidualReport, Scaled, ScaledValue,
    SciOxideError, Triangle,
};

impl<S: RealScalar + NumericScalar<Real = S>> MatrixView<'_, S> {
    /// Checks that `x` solves `A x = b` to a componentwise backward-error
    /// tolerance and returns `max_i |(b − A x)_i|`.
    ///
    /// Row `i` passes when
    ///
    /// ```text
    /// |(b − A x)_i| ≤ atol + rtol (|b_i| + Σ_j |a_ij| |x_j|),
    /// ```
    ///
    /// with `atol` and `rtol` from `tolerance`. The residuals and the right-hand
    /// side are accumulated exactly with a separate exponent and rounded once,
    /// so they do not overflow or lose accuracy to cancellation. A zero row of
    /// `A` passes only if `|b_i| ≤ atol + rtol |b_i|`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have `A.rows()` entries or
    ///   `solution` does not have `A.cols()` entries.
    /// - `NonFinite` if `A`, `rhs` or `solution` contains NaN or infinity.
    /// - `NumericalInstability` (`Breakdown`) if a row fails the test.
    /// - `NumericalInstability` (`Overflow`) if every row passes but
    ///   `max_i |(b − A x)_i|` exceeds the largest finite value.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Tolerance;
    /// let a = scioxide_linalg::MatrixView::from_slice(1, 1, &[2.0_f64])?;
    /// assert_eq!(a.verify_solution(&[4.0], &[2.0], Tolerance::new(0.0, 1e-12)?)?, 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn verify_solution(
        &self,
        rhs: &[S],
        solution: &[S],
        tolerance: scioxide_core::Tolerance<S>,
    ) -> Result<S, SciOxideError> {
        validate_solution(*self, rhs, solution)?;
        let mut maximum = Scaled::ZERO;
        for (i, &b) in rhs.iter().enumerate() {
            let row = residual_row(*self, i, b, solution)?;
            let residual = row.residual.abs();
            let bound = Scaled::from_value(tolerance.absolute())?
                .add(Scaled::from_value(tolerance.relative())?.mul(row.envelope)?)?;
            if !residual.abs_le(bound) {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    alloc::format!(
                        "linear solution row {i} failed componentwise backward-error check"
                    ),
                ));
            }
            if maximum.abs_le(residual) {
                maximum = residual;
            }
        }
        maximum.round()
    }
    /// Checks that `x` solves `A x = b` to a normwise backward-error tolerance
    /// and returns `‖b − A x‖∞`.
    ///
    /// The test is
    ///
    /// ```text
    /// ‖b − A x‖∞ ≤ atol + rtol (‖A‖∞ ‖x‖∞ + ‖b‖∞),
    /// ```
    ///
    /// with `atol` and `rtol` from `tolerance`. Unlike
    /// [`Self::verify_solution`], it accepts rounding errors in an equation whose
    /// exact right-hand side is zero. Sums are accumulated exactly with a
    /// separate exponent, so large rows do not overflow.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` does not have `A.rows()` entries or
    ///   `solution` does not have `A.cols()` entries.
    /// - `NonFinite` if `A`, `rhs` or `solution` contains NaN or infinity.
    /// - `NumericalInstability` (`Breakdown`) if the test fails.
    /// - `NumericalInstability` (`Overflow`) if the test passes but
    ///   `‖b − A x‖∞` exceeds the largest finite value.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Tolerance;
    /// let a = scioxide_linalg::MatrixView::from_slice(1, 1, &[2.0_f64])?;
    /// assert_eq!(
    ///     a.verify_solution_normwise(&[4.0], &[2.0], Tolerance::new(0.0, 1e-12)?)?,
    ///     0.0
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn verify_solution_normwise(
        &self,
        rhs: &[S],
        solution: &[S],
        tolerance: scioxide_core::Tolerance<S>,
    ) -> Result<S, SciOxideError> {
        validate_solution(*self, rhs, solution)?;
        let mut residual = Scaled::ZERO;
        let mut matrix_norm = Scaled::ZERO;
        for (i, &b) in rhs.iter().enumerate() {
            let row = residual_row(*self, i, b, solution)?;
            if residual.abs_le(row.residual) {
                residual = row.residual.abs();
            }
            if matrix_norm.abs_le(row.matrix_norm) {
                matrix_norm = row.matrix_norm;
            }
        }
        let maximum = |values: &[S]| {
            values.iter().fold(<S as AlgebraicScalar>::ZERO, |a, &v| {
                a.max(NumericScalar::abs(v))
            })
        };
        let scale = matrix_norm
            .mul(Scaled::from_value(maximum(solution))?)?
            .add(Scaled::from_value(maximum(rhs))?)?;
        let bound = Scaled::from_value(tolerance.absolute())?
            .add(Scaled::from_value(tolerance.relative())?.mul(scale)?)?;
        if !residual.abs_le(bound) {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "linear solution failed normwise backward-error check",
            ));
        }
        residual.round()
    }

    /// Writes the residual `b − A x` to `output`, each entry rounded once from
    /// the exact sum.
    ///
    /// `scratch` must have at least `A.rows()` entries.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` or `output` does not have `A.rows()`
    ///   entries, `solution` does not have `A.cols()` entries, or `scratch` is
    ///   too short.
    /// - `NonFinite` if `A`, `rhs` or `solution` contains NaN or infinity.
    /// - `NumericalInstability` (`Overflow`) if an entry of the residual exceeds
    ///   the largest finite value.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = scioxide_linalg::MatrixView::from_slice(1, 1, &[2.0_f64])?;
    /// let mut residual = [99.0];
    /// a.residual_into(&[5.0], &[2.0], &mut residual, &mut [0.0])?;
    /// // 5 − 2 · 2 = 1
    /// assert_eq!(residual, [1.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn residual_into(
        &self,
        rhs: &[S],
        solution: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        validate_solution(*self, rhs, solution)?;
        if output.len() != self.rows() || scratch.len() < self.rows() {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows(), self.rows()],
                [output.len(), scratch.len()],
            ));
        }
        for (i, &b) in rhs.iter().enumerate() {
            scratch[i] = -residual_row(*self, i, b, solution)?.residual.round()?;
        }
        output.copy_from_slice(&scratch[..self.rows()]);
        Ok(())
    }
}

fn validate_solution<S: RealScalar + NumericScalar<Real = S>>(
    matrix: MatrixView<'_, S>,
    rhs: &[S],
    solution: &[S],
) -> Result<(), SciOxideError> {
    if rhs.len() != matrix.rows() || solution.len() != matrix.cols() {
        return Err(SciOxideError::dimension_mismatch(
            [matrix.rows(), matrix.cols()],
            [rhs.len(), solution.len()],
        ));
    }
    for &value in rhs.iter().chain(solution) {
        crate::checked::validate_input(value, "linear solution")?;
    }
    Ok(())
}
struct ResidualRow<S: RealScalar> {
    residual: Scaled<S>,
    envelope: Scaled<S>,
    matrix_norm: Scaled<S>,
}
fn residual_row<S: RealScalar + NumericScalar<Real = S>>(
    matrix: MatrixView<'_, S>,
    row: usize,
    b: S,
    solution: &[S],
) -> Result<ResidualRow<S>, SciOxideError> {
    let mut residual = scioxide_core::RangeAccumulator::<S>::new();
    let mut scale = scioxide_core::RangeAccumulator::<S>::new();
    let mut norm = scioxide_core::RangeAccumulator::<S>::new();
    residual.add(-b)?;
    scale.add(NumericScalar::abs(b))?;
    for (j, &x) in solution.iter().enumerate() {
        let a = matrix.get(row, j)?;
        residual.add_product(a, x)?;
        scale.add_product(NumericScalar::abs(a), NumericScalar::abs(x))?;
        norm.add(NumericScalar::abs(a))?;
    }
    Ok(ResidualRow {
        residual: residual.finish_scaled::<S>()?,
        envelope: scale.finish_scaled::<S>()?,
        matrix_norm: norm.finish_scaled::<S>()?,
    })
}

/// Operator shape whose entry `(i, j)` is stored when `i <= j + lower` and
/// `j <= i + upper`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Band {
    rows: usize,
    cols: usize,
    lower: usize,
    upper: usize,
}

impl Band {
    /// Square band of order `order`.
    pub(crate) const fn new(order: usize, lower: usize, upper: usize) -> Self {
        Self {
            rows: order,
            cols: order,
            lower,
            upper,
        }
    }

    /// Every entry of a `rows × cols` matrix.
    pub(crate) const fn dense(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            lower: rows.saturating_sub(1),
            upper: cols.saturating_sub(1),
        }
    }

    /// Band of the transposed operator.
    pub(crate) const fn transpose(self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
            lower: self.upper,
            upper: self.lower,
        }
    }

    fn rows_of(self, col: usize) -> Range<usize> {
        col.saturating_sub(self.upper)
            ..col
                .saturating_add(self.lower)
                .saturating_add(1)
                .min(self.rows)
    }

    fn columns_of(self, row: usize) -> Range<usize> {
        row.saturating_sub(self.lower)
            ..row
                .saturating_add(self.upper)
                .saturating_add(1)
                .min(self.cols)
    }
}

/// Column-sum norm `max_j sum_i |a_ij|` without overflowing the scalar range.
///
/// Each column adds its rows in increasing order; a column whose ordinary sum
/// is not finite is summed again in extended range. The infinity norm is the
/// one-norm of the transpose.
pub(crate) fn one_norm<S: NumericScalar>(
    band: Band,
    element: impl Fn(usize, usize) -> S,
) -> Result<Scaled<S::Real>, SciOxideError> {
    const BLOCK: usize = 32;
    // Empty shapes may have an unbounded other extent, such as `usize::MAX × 0`.
    if band.rows == 0 || band.cols == 0 {
        return Ok(Scaled::ZERO);
    }
    let mut maximum = Scaled::ZERO;
    let mut ordinary_maximum = S::Real::ZERO;
    // Accumulate a small block of columns together so row-major inputs are
    // streamed rather than reread with a cache-line stride for every column.
    for start in (0..band.cols).step_by(BLOCK) {
        let end = (start + BLOCK).min(band.cols);
        let mut sums = [S::Real::ZERO; BLOCK];
        for row in band.rows_of(start).start..band.rows_of(end - 1).end {
            let columns = band.columns_of(row);
            for col in columns.start.max(start)..columns.end.min(end) {
                sums[col - start] += element(row, col).abs();
            }
        }
        for (col, &ordinary) in (start..end).zip(&sums) {
            if ordinary.is_finite() {
                ordinary_maximum = ordinary_maximum.max(ordinary);
                continue;
            }
            let sum = band.rows_of(col).try_fold(Scaled::ZERO, |sum, row| {
                sum.add(ScaledValue::from_value(element(row, col))?.magnitude()?)
            })?;
            if !sum.abs_le(maximum) {
                maximum = sum;
            }
        }
    }
    let ordinary = Scaled::from_value(ordinary_maximum)?;
    Ok(if ordinary.abs_le(maximum) {
        maximum
    } else {
        ordinary
    })
}

/// [`one_norm`] of `cols` columns whose band entries `column(j)` yields in
/// ascending row order.
///
/// Compact column storage is streamed directly. Short columns reduce locally;
/// longer independent columns interleave. Each sums in ascending row order, so the
/// result equals [`one_norm`] of the same band exactly.
/// Ordinary column maxima are reduced before one extended-range conversion;
/// only overflowing columns require independent extended-range sums.
pub(crate) fn column_one_norm<S: NumericScalar, I: Iterator<Item = S> + Clone>(
    cols: usize,
    column: impl Fn(usize) -> I,
) -> Result<Scaled<S::Real>, SciOxideError> {
    const LANES: usize = 8;
    let mut maximum = Scaled::ZERO;
    let mut ordinary_maximum = S::Real::ZERO;
    let mut observe = |col, ordinary: S::Real| -> Result<(), SciOxideError> {
        if ordinary.is_finite() {
            ordinary_maximum = ordinary_maximum.max(ordinary);
        } else {
            let sum = column(col).try_fold(Scaled::ZERO, |sum, value| {
                sum.add(ScaledValue::from_value(value)?.magnitude()?)
            })?;
            if !sum.abs_le(maximum) {
                maximum = sum;
            }
        }
        Ok(())
    };
    if cols > 0
        && column(cols / 2)
            .size_hint()
            .1
            .is_some_and(|length| length <= LANES)
    {
        // Short bands do less work than maintaining eight optional iterator states.
        for col in 0..cols {
            let ordinary = column(col).fold(S::Real::ZERO, |sum, value| sum + value.abs());
            observe(col, ordinary)?;
        }
    } else {
        for start in (0..cols).step_by(LANES) {
            let width = LANES.min(cols - start);
            let mut streams: [Option<I>; LANES] =
                core::array::from_fn(|lane| (lane < width).then(|| column(start + lane)));
            let mut sums = [S::Real::ZERO; LANES];
            while streams.iter().any(Option::is_some) {
                for (sum, stream) in sums.iter_mut().zip(&mut streams) {
                    match stream.as_mut().map(Iterator::next) {
                        Some(Some(value)) => *sum += value.abs(),
                        Some(None) => *stream = None,
                        None => {}
                    }
                }
            }
            for (col, &ordinary) in (start..start + width).zip(&sums) {
                observe(col, ordinary)?;
            }
        }
    }
    let ordinary = Scaled::from_value(ordinary_maximum)?;
    Ok(if ordinary.abs_le(maximum) {
        maximum
    } else {
        ordinary
    })
}

/// Dense [`one_norm`].
///
/// Row-major and column-major views accumulate the same per-column order
/// through contiguous slices, so the result is identical.
pub(crate) fn matrix_one_norm<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
) -> Result<Scaled<S::Real>, SciOxideError> {
    const LANES: usize = 8;
    let (rows, cols) = (matrix.rows(), matrix.cols());
    let element = |i, j| matrix.get_element(i, j);
    if rows == 0 || cols == 0 || (matrix.col_stride() != 1 && matrix.row_stride() != 1) {
        return one_norm(Band::dense(rows, cols), element);
    }
    let data = matrix.data;
    let mut maximum = Scaled::ZERO;
    let mut ordinary_maximum = S::Real::ZERO;
    let mut retain = |col: usize, ordinary: S::Real| -> Result<(), SciOxideError> {
        if ordinary.is_finite() {
            ordinary_maximum = ordinary_maximum.max(ordinary);
        } else {
            let sum = (0..rows).try_fold(Scaled::ZERO, |sum, row| {
                sum.add(ScaledValue::from_value(element(row, col))?.magnitude()?)
            })?;
            if !sum.abs_le(maximum) {
                maximum = sum;
            }
        }
        Ok(())
    };
    let mut sums = [S::Real::ZERO; LANES];
    for start in (0..cols).step_by(LANES) {
        let width = LANES.min(cols - start);
        sums.fill(S::Real::ZERO);
        if matrix.col_stride() == 1 {
            // Rows are contiguous: add each row's segment across the lanes.
            for row in 0..rows {
                let offset = row * matrix.row_stride() + start;
                for (sum, value) in sums.iter_mut().zip(&data[offset..offset + width]) {
                    *sum += value.abs();
                }
            }
        } else {
            // Columns are contiguous: interleave independent column streams.
            let columns: [&[S]; LANES] = core::array::from_fn(|lane| {
                let offset = (start + lane.min(width - 1)) * matrix.col_stride();
                &data[offset..offset + rows]
            });
            for row in 0..rows {
                for (sum, column) in sums.iter_mut().zip(&columns) {
                    *sum += column[row].abs();
                }
            }
        }
        for (lane, &sum) in sums[..width].iter().enumerate() {
            retain(start + lane, sum)?;
        }
    }
    // Taking the maximum commutes with the exact scalar-to-Scaled conversion.
    // Tall transposed views need only one conversion instead of one per row.
    let ordinary = Scaled::from_value(ordinary_maximum)?;
    Ok(if ordinary.abs_le(maximum) {
        maximum
    } else {
        ordinary
    })
}

/// Triangular substitution with a separate exponent, for condition-estimator
/// solves, with `triangle` of a band of half-width `bandwidth`. Only that
/// triangle and an explicit diagonal are read; `adjoint` solves with the
/// conjugate transpose.
pub(crate) fn substitute<S: NumericScalar>(
    values: &mut [ScaledValue<S>],
    triangle: Triangle,
    diagonal: DiagonalKind,
    bandwidth: usize,
    adjoint: bool,
    element: impl Fn(usize, usize) -> S,
) -> Result<(), SciOxideError> {
    let n = values.len();
    let forward = (triangle == Triangle::Lower) != adjoint;
    let coefficient = |i, j| {
        if adjoint {
            element(j, i).conj()
        } else {
            element(i, j)
        }
    };
    for step in 0..n {
        let i = if forward { step } else { n - 1 - step };
        let mut known = if forward {
            i.saturating_sub(bandwidth)..i
        } else {
            i + 1..i.saturating_add(bandwidth).saturating_add(1).min(n)
        };
        let value = known.try_fold(values[i], |value, j| {
            value.sub(ScaledValue::from_value(coefficient(i, j))?.mul(values[j])?)
        })?;
        values[i] = match diagonal {
            DiagonalKind::Unit => value,
            DiagonalKind::Explicit => value.div(ScaledValue::from_value(coefficient(i, i))?)?,
        };
    }
    Ok(())
}

/// Dense [`substitute`] with the leading `values.len()` block of `matrix`.
pub(crate) fn matrix_substitute<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    triangle: Triangle,
    diagonal: DiagonalKind,
    adjoint: bool,
    values: &mut [ScaledValue<S>],
) -> Result<(), SciOxideError> {
    let bandwidth = values.len();
    substitute(values, triangle, diagonal, bandwidth, adjoint, |i, j| {
        matrix.get_element(i, j)
    })
}

/// Normwise backward error `||b - Ax||∞ / (||A||∞ ||x||∞ + ||b||∞)`, with
/// `norm = ||A||∞`.
///
/// Each row takes one ordinary signed product, dispatched when
/// `contiguous_row(i)` borrows the coefficients of the band columns of row `i`.
/// Rows whose ordinary residual leaves the scalar range are evaluated again with
/// exact accumulation over the band. Every row is accumulated exactly under
/// [`Accuracy::Exact`], and when the denominator is below `μ / ε` (`μ` the
/// smallest positive normal number): there the absolute rounding of subnormal
/// products is not small relative to the denominator, and an ordinary residual
/// could hide the error of the solution. The denominator bounds every row sum
/// `|b_i| + sum_j |a_ij| |x_j|`, so rounding it decides whether the report is
/// representable, up to rounding.
pub(crate) fn backward_error<'a, S: NumericScalar>(
    band: Band,
    element: impl Fn(usize, usize) -> S,
    contiguous_row: impl Fn(usize) -> Option<&'a [S]>,
    norm: Scaled<S::Real>,
    solution: VectorView<'_, S>,
    rhs: VectorView<'_, S>,
    context: KernelContext<'_>,
) -> Result<ResidualReport<S::Real>, SciOxideError> {
    context.execution().check()?;
    let xnorm = infinity_norm(solution)?;
    let bnorm = infinity_norm(rhs)?;
    let denominator = norm.mul(xnorm)?.add(bnorm)?;
    let exact = context.accuracy() == Accuracy::Exact
        || !Scaled::from_value(S::Real::MIN_POSITIVE / S::Real::EPSILON)?.abs_le(denominator);
    let mut raw = Scaled::ZERO;
    let mut ordinary_raw = S::Real::ZERO;
    for row in 0..band.rows {
        let columns = band.columns_of(row);
        if !exact {
            let product = if let Some(coefficients) = contiguous_row(row)
                && let Some(x) = solution.as_contiguous_slice()
                && let Some(dot) =
                    crate::kernel::arch::dot(coefficients, &x[columns.clone()], context.dispatch())?
            {
                dot
            } else {
                crate::kernel::standard::dot(
                    columns
                        .clone()
                        .map(|col| (element(row, col), solution.get_element(col))),
                )
            };
            let ordinary = rhs.get_element(row) - product;
            if ordinary.has_finite_components() {
                let magnitude = ordinary.abs();
                if magnitude.is_finite() {
                    ordinary_raw = ordinary_raw.max(magnitude);
                } else {
                    let residual = ScaledValue::from_value(ordinary)?.magnitude()?;
                    if !residual.abs_le(raw) {
                        raw = residual;
                    }
                }
                continue;
            }
        }
        let mut sum = scioxide_core::ProductSum::new();
        sum.add_product(rhs.get_element(row), S::ONE)?;
        for col in columns {
            sum.subtract_product(element(row, col), solution.get_element(col))?;
        }
        let residual = scioxide_core::backend::product_sum_scaled(&sum)?.magnitude()?;
        if !residual.abs_le(raw) {
            raw = residual;
        }
    }
    let ordinary_raw = Scaled::from_value(ordinary_raw)?;
    if !ordinary_raw.abs_le(raw) {
        raw = ordinary_raw;
    }
    let scaling = if denominator.is_zero() {
        scioxide_core::ResidualScaling::unscaled()
    } else {
        scioxide_core::ResidualScaling::new(
            scioxide_core::ResidualScalingRule::BackwardError,
            denominator.round()?,
        )?
    };
    context.execution().check()?;
    ResidualReport::new(
        raw.round()?,
        scioxide_core::NormSelection::Infinity,
        scaling,
    )
}

/// Dense [`backward_error`], dispatching contiguous rows.
pub(crate) fn matrix_backward_error<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    norm: Scaled<S::Real>,
    solution: VectorView<'_, S>,
    rhs: VectorView<'_, S>,
    context: KernelContext<'_>,
) -> Result<ResidualReport<S::Real>, SciOxideError> {
    backward_error(
        Band::dense(matrix.rows(), matrix.cols()),
        |i, j| matrix.get_element(i, j),
        |i| matrix.row(i).ok()?.as_contiguous_slice(),
        norm,
        solution,
        rhs,
        context,
    )
}

fn infinity_norm<S: NumericScalar>(
    values: VectorView<'_, S>,
) -> Result<Scaled<S::Real>, SciOxideError> {
    let ordinary = values
        .iter()
        .fold(S::Real::ZERO, |maximum, value| maximum.max(value.abs()));
    if ordinary.is_finite() {
        return Scaled::from_value(ordinary);
    }
    values.iter().try_fold(Scaled::ZERO, |maximum, value| {
        let value = ScaledValue::from_value(value)?.magnitude()?;
        Ok(if value.abs_le(maximum) {
            maximum
        } else {
            value
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn subnormal_backward_error_accumulates_exactly() -> Result<(), SciOxideError> {
        // Each product 3u · 0.5 = 1.5u rounds to 2u, so the ordinary residual
        // of b = 8u is 0 while the exact one is 8u − 6u = 2u.
        let unit = f64::from_bits(1);
        let matrix = [3.0 * unit; 4];
        let matrix = MatrixView::from_slice(1, 4, &matrix)?;
        let report = matrix_backward_error(
            matrix,
            Scaled::from_value(12.0 * unit)?,
            VectorView::from_slice(&[0.5; 4]),
            VectorView::from_slice(&[8.0 * unit]),
            KernelContext::sequential_native(),
        )?;
        assert_eq!(report.raw_norm(), 2.0 * unit);
        // ‖A‖∞ ‖x‖∞ + ‖b‖∞ = 12u · 0.5 + 8u = 14u.
        assert!((report.scaled_norm() - 1.0 / 7.0).abs() < 1e-15);
        Ok(())
    }

    fn compact_column_norms<S: NumericScalar>() -> Result<(), SciOxideError> {
        use crate::{BandedMatrixView, HermitianBandMatrixView};
        for n in [0_usize, 1, 7, 8, 9, 19] {
            for scale in [S::Real::ONE, S::Real::MAX] {
                let value = |i: usize, j: usize| {
                    let real = S::Real::try_from_f64_exact((i % 5 + 1) as f64 / 8.0)? * scale;
                    let imaginary = if i == j {
                        S::Real::ZERO
                    } else {
                        -S::Real::try_from_f64_exact((j % 3 + 1) as f64 / 4.0)? * scale
                    };
                    Ok::<S, SciOxideError>(S::from_components(real, imaginary))
                };
                for (lower, upper) in [(0, 0), (n / 2, n / 3), (n.saturating_sub(1), 0)] {
                    let band = Band::new(n, lower, upper);
                    let stride = if n == 0 { 0 } else { lower + upper + 3 };
                    let mut data = alloc::vec![S::from_real(S::Real::NAN); n * stride];
                    for j in 0..n {
                        for i in band.rows_of(j) {
                            data[j * stride + upper + i - j] = value(i, j)?;
                        }
                    }
                    let a = BandedMatrixView::new(n, n, lower, upper, stride, &data)?;
                    assert_eq!(
                        column_one_norm(n, |j| a.column(j).iter().copied())?,
                        one_norm(band, |i, j| a.element(i, j))?
                    );
                }
                for k in [0, n / 2] {
                    for triangle in [Triangle::Lower, Triangle::Upper] {
                        let mut data = alloc::vec![S::from_real(S::Real::NAN); n * (k + 1)];
                        for j in 0..n {
                            let rows = match triangle {
                                Triangle::Lower => j..(j + k + 1).min(n),
                                Triangle::Upper => j.saturating_sub(k)..j + 1,
                            };
                            for i in rows {
                                let slot = match triangle {
                                    Triangle::Lower => i - j,
                                    Triangle::Upper => k + i - j,
                                };
                                data[j * (k + 1) + slot] = value(i, j)?;
                            }
                        }
                        let a = HermitianBandMatrixView::new(n, k, triangle, &data)?;
                        assert_eq!(
                            column_one_norm(n, |j| a.column(j))?,
                            one_norm(Band::new(n, k, k), |i, j| a.element(i, j))?
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn compact_column_one_norms_preserve_entrywise_sums() -> Result<(), SciOxideError> {
        compact_column_norms::<f32>()?;
        compact_column_norms::<f64>()?;
        compact_column_norms::<scioxide_core::Complex<f32>>()?;
        compact_column_norms::<scioxide_core::Complex<f64>>()
    }

    #[test]
    fn contiguous_one_norms_match_the_entrywise_band_sums() -> Result<(), SciOxideError> {
        use scioxide_core::Complex;
        let real: Vec<f64> = (0..23 * 19)
            .map(|k| ((k * 37 % 101) as f64 - 50.0) * if k % 3 == 0 { 1e306 } else { 1.0 })
            .collect();
        let complex: Vec<Complex<f64>> = real
            .iter()
            .map(|&value| Complex::new(value * 0.5, value.abs().sqrt()))
            .collect();
        let base = crate::MatrixView::from_slice(23, 19, &real)?;
        for view in [
            base,
            base.transpose(),
            base.block(2, 3, 17, 9)?,
            base.block(2, 3, 17, 9)?.transpose(),
        ] {
            let generic = one_norm(Band::dense(view.rows(), view.cols()), |i, j| {
                view.get_element(i, j)
            })?;
            assert_eq!(matrix_one_norm(view)?, generic);
        }
        let complex = crate::MatrixView::from_slice(23, 19, &complex)?;
        for view in [complex, complex.transpose(), complex.block(1, 4, 9, 15)?] {
            let generic = one_norm(Band::dense(view.rows(), view.cols()), |i, j| {
                view.get_element(i, j)
            })?;
            assert_eq!(matrix_one_norm(view)?, generic);
        }
        Ok(())
    }

    #[test]
    fn band_norms_match_the_expanded_dense_matrix() -> Result<(), SciOxideError> {
        let band = Band::new(5, 1, 2);
        let inside = |i: usize, j: usize| i <= j + 1 && j <= i + 2;
        let value = |i: usize, j: usize| (i * 5 + j) as f64 - 7.5;
        // Entries outside the band are never read.
        let element = |i, j| {
            assert!(inside(i, j));
            value(i, j)
        };
        let dense = crate::Matrix::from_vec(
            5,
            5,
            (0..25)
                .map(|k| {
                    let (i, j) = (k / 5, k % 5);
                    if inside(i, j) { value(i, j) } else { 0. }
                })
                .collect(),
        )?;
        assert_eq!(one_norm(band, element)?, matrix_one_norm(dense.view())?);
        assert_eq!(
            one_norm(band.transpose(), |i, j| element(j, i))?,
            matrix_one_norm(dense.view().transpose())?
        );
        let wide = crate::Matrix::<f64>::from_vec(2, 3, alloc::vec![1., -2., 3., -4., 5., -6.])?;
        assert_eq!(matrix_one_norm(wide.view())?.round()?, 9.);
        assert_eq!(matrix_one_norm(wide.view().transpose())?.round()?, 15.);
        for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
            assert!(matrix_one_norm(crate::Matrix::<f64>::zeros(rows, cols)?.view())?.is_zero());
        }
        for band in [Band::dense(usize::MAX, 0), Band::dense(0, usize::MAX)] {
            assert!(one_norm(band, |_, _| f64::NAN)?.is_zero());
        }
        // A column sum beyond the scalar range is accumulated in extended range.
        let tall = crate::Matrix::from_vec(2, 1, alloc::vec![f64::MAX, -f64::MAX])?;
        let norm = matrix_one_norm(tall.view())?;
        assert_eq!(norm.div(Scaled::from_value(f64::MAX)?)?.round()?, 2.);
        Ok(())
    }

    #[test]
    fn backward_error_overflows_only_at_the_normwise_denominator() -> Result<(), SciOxideError> {
        let x = [1., -1.];
        let b = [0., -1.];
        let report = |entries: Vec<f64>| {
            let a = crate::Matrix::from_vec(2, 2, entries)?;
            matrix_backward_error(
                a.view(),
                matrix_one_norm(a.view().transpose())?,
                VectorView::from_slice(&x),
                VectorView::from_slice(&b),
                KernelContext::sequential(),
            )
        };
        // Both residuals are zero, but the first row envelope exceeds the range.
        let error = report(alloc::vec![f64::MAX, f64::MAX, 0., 1.]).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
        let quarter = f64::MAX / 4.;
        let residual = report(alloc::vec![quarter, quarter, 0., 1.])?;
        assert_eq!(residual.raw_norm(), 0.);
        assert_eq!(residual.scaling().denominator(), 2. * quarter + 1.);
        Ok(())
    }
}
