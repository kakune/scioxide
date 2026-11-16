//! Contingency tables: chi-square and power-divergence tests, association measures,
//! exact tests and p-value adjustment.

use crate::error_helpers::{finite, invalid};
use scioxide_core::{RealScalar, SciOxideError};
use scioxide_random::{ChiSquared, Survival};

mod exact;
pub use exact::{
    ContingencyDesign, ExactContingencyResult, UnconditionalOptions, barnard_exact_test,
    barnard_exact_test_with_options, boschloo_exact_test, boschloo_exact_test_with_options,
    fisher_exact_test, fisher_exact_test_with_options,
};
mod multiple;
pub use multiple::{
    CombinedPValue, FalseDiscoveryMethod, FamilywiseMethod, PValueAdjustment,
    PValueAdjustmentWorkspace, PValueCombination, adjust_pvalues, adjust_pvalues_into,
    adjust_pvalues_with_options, combine_pvalues,
};

/// Contingency table of counts, borrowed in row-major order.
///
/// Counts may be zero. The table needs at least one row and one column.
#[derive(Clone, Copy, Debug)]
pub struct ContingencyView<'a> {
    counts: &'a [usize],
    rows: usize,
    columns: usize,
}

impl<'a> ContingencyView<'a> {
    /// Creates a `rows × columns` view of `counts`, which is in row-major order.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `rows` or `columns` is zero, or
    ///   `counts.len() != rows * columns`.
    pub fn new(counts: &'a [usize], rows: usize, columns: usize) -> Result<Self, SciOxideError> {
        if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(counts.len()) {
            return Err(invalid(
                scioxide_core::argument!("counts"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid contingency shape",
            ));
        }
        Ok(Self {
            counts,
            rows,
            columns,
        })
    }

    /// Returns the number of rows.
    pub const fn rows(self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub const fn columns(self) -> usize {
        self.columns
    }
    /// Returns the counts in row-major order.
    pub const fn counts(self) -> &'a [usize] {
        self.counts
    }

    /// Writes the row and column totals to `row` and `column` and returns the
    /// grand total.
    ///
    /// The totals are exact integers. The cost is O(R C) time.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `row.len()` is not the number of rows or
    ///   `column.len()` not the number of columns.
    /// - `SizeOverflow` if the grand total overflows `u128`.
    pub fn margins_into(
        self,
        row: &mut [u128],
        column: &mut [u128],
    ) -> Result<u128, SciOxideError> {
        if row.len() != self.rows || column.len() != self.columns {
            return Err(invalid(
                scioxide_core::argument!("row"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "margin output shape mismatch",
            ));
        }
        let total = self.counts.iter().try_fold(0_u128, |sum, &count| {
            sum.checked_add(count as u128).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })
        })?;
        // Every partial margin is bounded by the checked grand total.
        row.fill(0);
        column.fill(0);
        for (index, &count) in self.counts.iter().enumerate() {
            row[index / self.columns] += count as u128;
            column[index % self.columns] += count as u128;
        }
        Ok(total)
    }

    /// Writes the expected counts under independence, `Eᵢⱼ = rᵢ cⱼ / n`, to
    /// `output`.
    ///
    /// `rᵢ` and `cⱼ` are the row and column totals and `n` the grand total.
    /// `rows` and `columns` receive the totals, as in [`Self::margins_into`].
    /// The cost is O(R C) time.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `output.len()` differs from the number of cells.
    /// - `InvalidInput` if the grand total is zero.
    /// - The errors of [`Self::margins_into`].
    pub fn expected_into(
        self,
        output: &mut [f64],
        rows: &mut [u128],
        columns: &mut [u128],
    ) -> Result<(), SciOxideError> {
        if output.len() != self.counts.len() {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "expected output shape mismatch",
            ));
        }
        let total = self.margins_into(rows, columns)?;
        if total == 0 {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty contingency table",
            ));
        }
        // A count table held in memory has practical totals far below f64 overflow.
        // Division first avoids overflow in the product of two large margins.
        let n = total as f64;
        for (index, slot) in output.iter_mut().enumerate() {
            *slot = (rows[index / self.columns] as f64 / n) * columns[index % self.columns] as f64;
        }
        Ok(())
    }
}

/// Exponent `λ` of the Cressie–Read power-divergence statistic.
///
/// ```text
/// 2nI^λ = 2/(λ(λ + 1)) Σᵢ Oᵢ ((Oᵢ/Eᵢ)^λ − 1),
/// ```
///
/// where `Oᵢ` and `Eᵢ` are the observed and expected counts. `λ = 1` gives
/// Pearson's chi-square and `λ = 0` (as a limit) the likelihood-ratio G-test.
/// Other common values are `λ = −1/2` (Freeman–Tukey), `λ = −1` (modified
/// log-likelihood) and `λ = 2/3` (recommended by Cressie and Read). Any finite
/// `λ` is accepted.
///
/// # Examples
///
/// ```
/// use scioxide_stats::CressieReadParameter;
/// assert_eq!(CressieReadParameter::PEARSON.lambda(), 1.0);
/// assert!(CressieReadParameter::new(f64::NAN).is_err());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CressieReadParameter(f64);
impl CressieReadParameter {
    /// `λ = 1`: Pearson's chi-square.
    pub const PEARSON: Self = Self(1.0);
    /// `λ = 0`: the likelihood-ratio G-test.
    pub const G_TEST: Self = Self(0.0);
    /// Creates the exponent `λ = lambda`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lambda` is NaN or infinite.
    pub fn new(lambda: f64) -> Result<Self, SciOxideError> {
        if !lambda.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "Cressie–Read exponent must be finite",
            ));
        }
        Ok(Self(lambda))
    }
    /// Returns `λ`.
    pub const fn lambda(self) -> f64 {
        self.0
    }
}
impl Default for CressieReadParameter {
    fn default() -> Self {
        Self::PEARSON
    }
}

/// Result of a chi-square or power-divergence test.
///
/// The p-value is the large-sample chi-square approximation, not an exact
/// probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquareResult {
    /// Power-divergence statistic.
    statistic: f64,
    /// Upper tail probability of the chi-square distribution at the statistic.
    p_value: f64,
    /// Degrees of freedom of the chi-square distribution.
    degrees_of_freedom: usize,
}
impl ChiSquareResult {
    /// Returns the statistic.
    pub const fn statistic(&self) -> f64 {
        self.statistic
    }
    /// Returns the p value.
    pub const fn p_value(&self) -> f64 {
        self.p_value
    }
    /// Returns the degrees of freedom.
    pub const fn degrees_of_freedom(&self) -> usize {
        self.degrees_of_freedom
    }
}

fn divergence_term(observed: f64, expected: f64, lambda: f64) -> Result<f64, SciOxideError> {
    if expected <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("expected"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "nonpositive expected frequency",
        ));
    }
    if observed == 0.0 {
        return if lambda <= -1.0 {
            Err(invalid(
                scioxide_core::argument!("observed"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "zero observed frequency is singular for this divergence",
            ))
        } else if lambda == 1.0 {
            Ok(expected)
        } else {
            Ok(0.0)
        };
    }
    let ratio = observed / expected;
    if lambda == 1.0 {
        return finite((observed - expected).powi(2) / expected);
    }
    if lambda == 0.0 {
        return finite(2.0 * observed * ratio.ln());
    }
    if lambda == -1.0 {
        return finite(2.0 * expected * (expected / observed).ln());
    }
    finite(2.0 * observed * (lambda * ratio.ln()).exp_m1() / (lambda * (lambda + 1.0)))
}

fn chi_result(statistic: f64, df: usize) -> Result<ChiSquareResult, SciOxideError> {
    if df == 0 {
        return Err(invalid(
            scioxide_core::argument!("ddof"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "nonpositive chi-square degrees of freedom",
        ));
    }
    let statistic = finite(statistic)?.max(0.0);
    let p_value = ChiSquared::<f64>::new(df as f64)?.sf(statistic)?;
    Ok(ChiSquareResult {
        statistic,
        p_value,
        degrees_of_freedom: df,
    })
}

/// Performs the Cressie–Read power-divergence goodness-of-fit test of `k`
/// observed frequencies.
///
/// `expected` defaults to equal frequencies. The statistic is that of
/// [`CressieReadParameter`]. The p-value is the upper tail of the chi-square
/// distribution with `k − 1 − ddof` degrees of freedom, where `ddof` is the
/// number of parameters estimated to obtain `expected`. The cost is O(k) time
/// and O(1) memory.
///
/// # Errors
///
/// - `InvalidInput` if `k < 2` or `expected` has a different length.
/// - `InvalidInput` if a frequency is negative, the observed total is zero, or
///   the observed and expected totals differ by more than `64 ε` relative.
/// - `InvalidInput` if `ddof ≥ k − 1`.
/// - `InvalidInput` if an expected frequency is zero, or an observed frequency
///   is zero and `λ ≤ −1`.
/// - An error if a frequency is NaN or infinite.
///
/// # References
///
/// N. Cressie and T. R. C. Read, "Multinomial goodness-of-fit tests", *J. Roy.
/// Statist. Soc. Ser. B* 46 (1984), 440–464.
pub fn power_divergence_test<S: RealScalar>(
    observed: &[S],
    expected: Option<&[S]>,
    divergence: CressieReadParameter,
    ddof: usize,
) -> Result<ChiSquareResult, SciOxideError> {
    let lambda = divergence.lambda();
    if observed.len() < 2 || expected.is_some_and(|values| values.len() != observed.len()) {
        return Err(invalid(
            scioxide_core::argument!("expected"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "goodness-of-fit shape mismatch",
        ));
    }
    let mut total = 0.0;
    let mut expected_total = 0.0;
    for (index, &value) in observed.iter().enumerate() {
        let o = finite(value.into_f64())?;
        if o < 0.0 {
            return Err(invalid(
                scioxide_core::argument!("observed"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "negative observed frequency",
            ));
        }
        total += o;
        if let Some(values) = expected {
            let e = finite(values[index].into_f64())?;
            if e < 0.0 {
                return Err(invalid(
                    scioxide_core::argument!("expected"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "negative expected frequency",
                ));
            }
            expected_total += e;
        }
    }
    finite(total)?;
    if total <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("observed"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "zero observed total",
        ));
    }
    if expected.is_some()
        && (total - expected_total).abs() > 64.0 * S::EPSILON.into_f64() * total.max(expected_total)
    {
        return Err(invalid(
            scioxide_core::argument!("expected"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "observed and expected totals differ",
        ));
    }
    let equal = total / observed.len() as f64;
    let mut statistic = 0.0;
    for (index, &value) in observed.iter().enumerate() {
        let e = expected.map_or(equal, |values| values[index].into_f64());
        statistic += divergence_term(value.into_f64(), e, lambda)?;
    }
    let df = observed
        .len()
        .checked_sub(1)
        .and_then(|v| v.checked_sub(ddof))
        .ok_or_else(|| {
            invalid(
                scioxide_core::argument!("ddof"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "nonpositive chi-square degrees of freedom",
            )
        })?;
    chi_result(statistic, df)
}

/// Performs the chi-square test of independence in an `R × C` contingency
/// table.
///
/// The expected counts are `Eᵢⱼ = rᵢ cⱼ / n`, and the statistic is the
/// power divergence selected by `divergence`. The p-value is the upper tail of
/// the chi-square distribution with `(R − 1)(C − 1)` degrees of freedom. For a
/// `2 × 2` table with the Pearson statistic, `correction` applies Yates's
/// continuity correction `Σ (|O − E| − 1/2)² / E`. It has no effect on larger tables.
/// The cost is O(R C) time and O(R + C) storage for the margins.
///
/// # Errors
///
/// - `InvalidInput` if the table has fewer than two rows or columns, or a row or
///   column total is zero.
/// - `InvalidInput` if `correction` is `HalfUnit` and `divergence` is not
///   Pearson's.
/// - `InvalidInput` if a zero count meets `λ ≤ −1`.
/// - The errors of [`ContingencyView::margins_into`].
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ContingencyView, ContinuityCorrection};
/// use scioxide_stats::{CressieReadParameter, power_divergence_contingency_test};
///
/// // All expected counts are 15, so X² = 4 · 5²/15 = 20/3.
/// let table = ContingencyView::new(&[10, 20, 20, 10], 2, 2)?;
/// let pearson = CressieReadParameter::PEARSON;
/// let none = ContinuityCorrection::None;
/// let test = power_divergence_contingency_test(table, pearson, none)?;
/// assert!((test.statistic() - 20.0 / 3.0).abs() < 1e-12);
/// assert_eq!(test.degrees_of_freedom(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn power_divergence_contingency_test(
    table: ContingencyView<'_>,
    divergence: CressieReadParameter,
    correction: crate::ContinuityCorrection,
) -> Result<ChiSquareResult, SciOxideError> {
    if table.rows < 2 || table.columns < 2 {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "chi-square requires two rows and columns",
        ));
    }
    let lambda = divergence.lambda();
    if correction == crate::ContinuityCorrection::HalfUnit && lambda != 1.0 {
        return Err(invalid(
            scioxide_core::argument!("correction"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Yates correction requires Pearson divergence",
        ));
    }
    let mut rows = scioxide_core::backend::try_filled(
        table.rows(),
        0_u128,
        scioxide_core::quantity!("contingency_rows"),
    )?;
    let mut columns = scioxide_core::backend::try_filled(
        table.columns(),
        0_u128,
        scioxide_core::quantity!("contingency_columns"),
    )?;
    let total = table.margins_into(&mut rows, &mut columns)?;
    if total == 0 || rows.contains(&0) || columns.contains(&0) {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "zero contingency margin",
        ));
    }
    let n = total as f64;
    let mut statistic = 0.0;
    for (index, &value) in table.counts.iter().enumerate() {
        let e = (rows[index / table.columns] as f64 / n) * columns[index % table.columns] as f64;
        let o = value as f64;
        if correction == crate::ContinuityCorrection::HalfUnit
            && table.rows == 2
            && table.columns == 2
        {
            statistic += ((o - e).abs() - 0.5).max(0.0).powi(2) / e;
        } else {
            statistic += divergence_term(o, e, lambda)?;
        }
    }
    chi_result(
        statistic,
        (table.rows - 1)
            .checked_mul(table.columns - 1)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?,
    )
}

/// Selects a measure of association for nominal variables derived from
/// Pearson's chi-square statistic `X²`.
///
/// `n` is the grand total and `R`, `C` the numbers of rows and columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssociationMeasure {
    /// Cramér's `V = √(X² / (n min(R − 1, C − 1)))`, in `[0, 1]`.
    CramerV,
    /// Tschuprow's `T = √(X² / (n √((R − 1)(C − 1))))`, in `[0, 1]`.
    TschuprowT,
    /// Pearson's contingency coefficient `C = √(X² / (X² + n))`.
    Pearson,
}

/// Computes a measure of association between the row and column variables of a
/// contingency table.
///
/// `X²` is Pearson's statistic without continuity correction.
///
/// # Errors
///
/// - The errors of [`power_divergence_contingency_test`], for example a table with fewer than
///   two rows or columns, or a zero row or column total.
pub fn contingency_association(
    table: ContingencyView<'_>,
    method: AssociationMeasure,
) -> Result<f64, SciOxideError> {
    let chi = power_divergence_contingency_test(
        table,
        CressieReadParameter::default(),
        crate::ContinuityCorrection::None,
    )?
    .statistic;
    let total = table.counts.iter().try_fold(0_u128, |a, &v| {
        a.checked_add(v as u128).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })
    })? as f64;
    let value = match method {
        AssociationMeasure::CramerV => {
            (chi / (total * (table.rows - 1).min(table.columns - 1) as f64)).sqrt()
        }
        AssociationMeasure::TschuprowT => {
            (chi / (total * ((table.rows - 1) as f64 * (table.columns - 1) as f64).sqrt())).sqrt()
        }
        AssociationMeasure::Pearson => (chi / (chi + total)).sqrt(),
    };
    finite(value)
}

/// Computes the sample odds ratio `a d / (b c)` of the `2 × 2` table
/// `[[a, b], [c, d]]`.
///
/// The result is 0 when `a d = 0` and `+∞` when `b c = 0`.
///
/// # Errors
///
/// - `InvalidInput` if the table is not `2 × 2`, or both `a d` and `b c` are
///   zero.
pub fn odds_ratio(table: ContingencyView<'_>) -> Result<f64, SciOxideError> {
    let [a, b, c, d] = four(table)?;
    let numerator = a as f64 * d as f64;
    let denominator = b as f64 * c as f64;
    if numerator == 0.0 && denominator == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "undefined odds ratio",
        ));
    }
    Ok(numerator / denominator)
}

/// Computes the relative risk `(a/(a + b)) / (c/(c + d))` of the `2 × 2` table
/// `[[a, b], [c, d]]`.
///
/// Rows are the exposure groups and the first column is the event. The result
/// is `+∞` when `c = 0` and `a > 0`.
///
/// # Errors
///
/// - `InvalidInput` if the table is not `2 × 2`, a row total is zero, or
///   `a = c = 0`.
pub fn relative_risk(table: ContingencyView<'_>) -> Result<f64, SciOxideError> {
    let [a, b, c, d] = four(table)?;
    let first = (a as u128 + b as u128) as f64;
    let second = (c as u128 + d as u128) as f64;
    if first == 0.0 || second == 0.0 || (a == 0 && c == 0) {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "undefined relative risk",
        ));
    }
    Ok((a as f64 / first) / (c as f64 / second))
}

fn four(table: ContingencyView<'_>) -> Result<[usize; 4], SciOxideError> {
    if table.rows != 2 || table.columns != 2 {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "requires a 2x2 table",
        ));
    }
    Ok([
        table.counts[0],
        table.counts[1],
        table.counts[2],
        table.counts[3],
    ])
}

/// Tests independence using Pearson's chi-square statistic with Yates's correction for a 2 × 2 table.
///
/// Uses [`power_divergence_contingency_test`] with the [one-shot defaults](crate#one-shot-forms).
///
/// # Errors
///
/// Returns the errors of the configured contingency test.
pub fn chi2_contingency_test(table: ContingencyView<'_>) -> Result<ChiSquareResult, SciOxideError> {
    power_divergence_contingency_test(
        table,
        CressieReadParameter::default(),
        crate::ContinuityCorrection::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_margins_expectations_and_divergence() {
        let counts = [10, 20, 20, 10];
        let table = ContingencyView::new(&counts, 2, 2).unwrap();
        let mut rows = [0; 2];
        let mut columns = [0; 2];
        assert_eq!(table.margins_into(&mut rows, &mut columns).unwrap(), 60);
        assert_eq!(rows, [30, 30]);
        assert_eq!(columns, [30, 30]);
        let mut expected = [0.0; 4];
        table
            .expected_into(&mut expected, &mut rows, &mut columns)
            .unwrap();
        assert_eq!(expected, [15.0; 4]);
        let chi = power_divergence_contingency_test(
            table,
            CressieReadParameter::default(),
            crate::ContinuityCorrection::None,
        )
        .unwrap();
        assert!((chi.statistic - 20.0 / 3.0).abs() < 1e-12);
        let yates = power_divergence_contingency_test(
            table,
            CressieReadParameter::default(),
            crate::ContinuityCorrection::HalfUnit,
        )
        .unwrap();
        assert!((yates.statistic - 81.0 / 15.0).abs() < 1e-12);
        let goodness = power_divergence_test(
            &[10.0_f32, 20.0],
            Some(&[15.0, 15.0]),
            crate::CressieReadParameter::PEARSON,
            0,
        )
        .unwrap();
        assert!((goodness.statistic - 10.0 / 3.0).abs() < 1e-12);
        assert!(
            (contingency_association(table, AssociationMeasure::CramerV).unwrap()
                - (1.0_f64 / 9.0).sqrt())
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn zero_margins_and_wide_counts() {
        let all_large = [usize::MAX; 4];
        let table = ContingencyView::new(&all_large, 2, 2).unwrap();
        let mut rows = [0; 2];
        let mut columns = [0; 2];
        assert_eq!(
            table.margins_into(&mut rows, &mut columns).unwrap(),
            4 * usize::MAX as u128
        );
        assert!(
            power_divergence_contingency_test(
                ContingencyView::new(&[0, 0, 1, 2], 2, 2).unwrap(),
                CressieReadParameter::default(),
                crate::ContinuityCorrection::None
            )
            .is_err()
        );
        assert!(ContingencyView::new(&[1, 2, 3], usize::MAX, 2).is_err());
        assert!(
            power_divergence_test(
                &[0.0_f64, 1.0],
                Some(&[0.5, 0.5]),
                CressieReadParameter::new(-1.0).unwrap(),
                0
            )
            .is_err()
        );
    }
}
