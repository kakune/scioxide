//! Exact tests for contingency tables: Fisher, Barnard and Boschloo.
use super::{ContingencyView, four, invalid};
use crate::{ClassicalOptions, TestAlternative, TestCalibration};
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
#[cfg_attr(test, allow(unused_imports))]
use scioxide_core::RealScalar;
use scioxide_core::{
    SciOxideError,
    backend::{try_filled, try_with_capacity},
};

/// Sampling design an exact contingency test conditions on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContingencyDesign {
    /// Both row and column totals are fixed (Fisher's exact test).
    FixedMargins,
    /// Only the column totals are fixed: two independent binomial samples. The
    /// p-value is maximized over the unknown common success probability
    /// (Barnard's and Boschloo's tests).
    FixedColumns,
}

/// Result of an exact contingency test.
///
/// For the unconditional tests the p-value is a maximum over the nuisance
/// probability, computed only to within [`UnconditionalOptions::tolerance`].
/// The result brackets it as `p_value_lower() ≤ p ≤ p_value()`, and
/// [`ExactContingencyResult::p_value`] is the conservative upper end. For
/// Fisher's test the two are equal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactContingencyResult {
    statistic: f64,
    p_value: f64,
    p_value_lower: f64,
    design: ContingencyDesign,
    tables: usize,
}
impl ExactContingencyResult {
    /// Returns the test statistic: the sample odds ratio for Fisher's `2 × 2`
    /// test, the probability of the observed table for Fisher's `R × C` test,
    /// the Wald statistic for Barnard's test and Fisher's p-value for
    /// Boschloo's test.
    pub const fn statistic(&self) -> f64 {
        self.statistic
    }
    /// Returns the p-value. For the unconditional tests it is the upper end of
    /// the bracket.
    pub const fn p_value(&self) -> f64 {
        self.p_value
    }
    /// Returns the lower end of the p-value bracket: the largest tail
    /// probability found at an evaluated nuisance probability.
    pub const fn p_value_lower(&self) -> f64 {
        self.p_value_lower
    }
    /// Returns the sampling design the test conditions on.
    pub const fn design(&self) -> ContingencyDesign {
        self.design
    }
    /// Returns the number of tables evaluated.
    pub const fn tables(&self) -> usize {
        self.tables
    }
}

fn charge(work: &mut usize, amount: usize, limit: usize) -> Result<(), SciOxideError> {
    *work = work
        .checked_add(amount)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::work(
        *work,
        limit,
        scioxide_core::BudgetKind::Node,
        "exact tables exceed options.max_work",
    )
}

fn bytes(count: usize, element: usize, limit: usize) -> Result<(), SciOxideError> {
    let required = scioxide_core::backend::checked_mul(
        count,
        element,
        scioxide_core::quantity!("exact_workspace_bytes"),
    )?;
    crate::error_helpers::limit(
        required,
        limit,
        scioxide_core::quantity!("exact_workspace_bytes"),
    )
}

fn fisher_2x2(
    cells: [u128; 4],
    alternative: TestAlternative,
    work: &mut usize,
    max_work: usize,
    max_bytes: usize,
) -> Result<(f64, usize), SciOxideError> {
    let [a, b, c, d] = cells;
    let row = a
        .checked_add(b)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let column = a
        .checked_add(c)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let total = row
        .checked_add(c)
        .and_then(|v| v.checked_add(d))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let low = row.saturating_add(column).saturating_sub(total);
    let high = row.min(column);
    let width = usize::try_from(high - low + 1).map_err(|_| {
        invalid(
            scioxide_core::argument!("options"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "exact support too large",
        )
    })?;
    charge(work, width, max_work)?;
    bytes(width, core::mem::size_of::<f64>(), max_bytes)?;
    let mut logs = try_filled(
        width,
        0.0_f64,
        scioxide_core::quantity!("fisher_log_masses"),
    )?;
    for index in 1..width {
        let x = low + index as u128 - 1;
        let numerator = (row - x) as f64 * (column - x) as f64;
        let denominator = (x + 1) as f64 * (total - row - (column - x) + 1) as f64;
        logs[index] = logs[index - 1] + numerator.ln() - denominator.ln();
    }
    let observed = usize::try_from(a - low).map_err(|_| {
        invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "observed cell outside support",
        )
    })?;
    let pivot = logs[observed];
    let normalization = logs.iter().try_fold(f64::NEG_INFINITY, |sum, &v| {
        scioxide_core::logaddexp(sum, v)
    })?;
    let mut tail = f64::NEG_INFINITY;
    for (index, &value) in logs.iter().enumerate() {
        let selected = match alternative {
            TestAlternative::Less => index <= observed,
            TestAlternative::Greater => index >= observed,
            TestAlternative::TwoSided => {
                value <= pivot + 64.0 * f64::EPSILON * pivot.abs().max(1.0)
            }
        };
        if selected {
            tail = scioxide_core::logaddexp(tail, value)?;
        }
    }
    Ok(((tail - normalization).exp().min(1.0), width))
}

fn log_factorial(n: usize) -> Result<f64, SciOxideError> {
    scioxide_special::ln_factorial::<f64>(n)
}

struct RxC<'a> {
    rows: usize,
    columns: usize,
    row_remaining: Vec<u128>,
    column_remaining: Vec<u128>,
    cells: Vec<usize>,
    observed_score: f64,
    tail_log: f64,
    total_log: f64,
    tables: usize,
    options: &'a ClassicalOptions,
}

impl RxC<'_> {
    fn visit(&mut self, index: usize) -> Result<(), SciOxideError> {
        if index == self.cells.len() {
            charge(&mut self.tables, 1, self.options.max_work())?;
            // The margin-dependent normalizer cancels in both ordering and
            // tail normalization; omitting it avoids large-log cancellation.
            let mut log = 0.0;
            for &cell in &self.cells {
                log -= log_factorial(cell)?;
            }
            self.total_log = scioxide_core::logaddexp(self.total_log, log)?;
            if log <= self.observed_score + 64.0 * f64::EPSILON * self.observed_score.abs().max(1.0)
            {
                self.tail_log = scioxide_core::logaddexp(self.tail_log, log)?;
            }
            return Ok(());
        }
        let row = index / self.columns;
        let column = index % self.columns;
        let low = if row + 1 == self.rows || column + 1 == self.columns {
            if row + 1 == self.rows {
                self.column_remaining[column]
            } else {
                self.row_remaining[row]
            }
        } else {
            let remaining = self.column_remaining[column + 1..]
                .iter()
                .copied()
                .sum::<u128>();
            self.row_remaining[row].saturating_sub(remaining)
        };
        let high = if row + 1 == self.rows || column + 1 == self.columns {
            low
        } else {
            self.row_remaining[row].min(self.column_remaining[column])
        };
        if high > self.row_remaining[row] || high > self.column_remaining[column] {
            return Ok(());
        }
        let mut value = low;
        loop {
            self.cells[index] = usize::try_from(value).map_err(|_| {
                invalid(
                    scioxide_core::argument!("table"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "table cell exceeds usize",
                )
            })?;
            self.row_remaining[row] -= value;
            self.column_remaining[column] -= value;
            self.visit(index + 1)?;
            self.row_remaining[row] += value;
            self.column_remaining[column] += value;
            if value == high {
                break;
            }
            value += 1;
        }
        Ok(())
    }
}

/// Computes the result of [`fisher_exact_test_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`fisher_exact_test_with_options`] for the fixed default settings.
pub fn fisher_exact_test(
    table: ContingencyView<'_>,
) -> Result<ExactContingencyResult, SciOxideError> {
    fisher_exact_test_with_options(table, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("table"),
        )
    })
}
/// Performs Fisher's exact test of independence in an `R × C` contingency
/// table.
///
/// With all row and column totals fixed, the table has the multivariate
/// hypergeometric distribution under independence. The two-sided p-value sums
/// the probabilities of all tables with the same totals that are no more
/// probable than the observed one. For `2 × 2` tables, `Less` and `Greater`
/// use the tails of the upper-left count. Larger tables support only the
/// two-sided test and are enumerated in full (the Freeman–Halton extension).
/// [`ClassicalOptions::max_work`] bounds the number of tables and
/// [`ClassicalOptions::max_workspace_bytes`] the scratch.
///
/// # Errors
///
/// - `InvalidInput` if the table has fewer than two rows or columns, a row or
///   column total is zero, or the grand total exceeds `2⁵³`.
/// - `InvalidInput` if the calibration is [`TestCalibration::Asymptotic`], or
///   the table is larger than `2 × 2` and the alternative is not two-sided.
/// - `ConvergenceFailure` if the enumeration exceeds `max_work`.
/// - `ResourceLimit` if the scratch exceeds `max_workspace_bytes`.
///
/// # References
///
/// A. Agresti, *Categorical Data Analysis*, 3rd ed., Wiley, 2013.
///
/// G. H. Freeman and J. H. Halton, "Note on an exact treatment of contingency,
/// goodness of fit and other problems of significance", *Biometrika* 38 (1951),
/// 141–149.
pub fn fisher_exact_test_with_options(
    table: ContingencyView<'_>,
    options: ClassicalOptions,
) -> Result<ExactContingencyResult, SciOxideError> {
    if table.rows() < 2
        || table.columns() < 2
        || options.calibration() == TestCalibration::Asymptotic
    {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Fisher requires a genuine table and exact calibration",
        ));
    }
    let margin_slots = table
        .rows()
        .checked_add(table.columns())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let margin_bytes = margin_slots
        .checked_mul(core::mem::size_of::<u128>())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    if table.rows() == 2 && table.columns() == 2 {
        crate::error_helpers::limit(
            margin_bytes,
            options.max_workspace_bytes(),
            scioxide_core::quantity!("exact_bytes"),
        )?;
    } else {
        let all_slots = margin_slots
            .checked_add(table.counts().len())
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        bytes(
            all_slots,
            core::mem::size_of::<u128>(),
            options.max_workspace_bytes(),
        )?;
    }
    let mut row = try_filled(
        table.rows(),
        0_u128,
        scioxide_core::quantity!("fisher_row_margins"),
    )?;
    let mut column = try_filled(
        table.columns(),
        0_u128,
        scioxide_core::quantity!("fisher_column_margins"),
    )?;
    let total = table.margins_into(&mut row, &mut column)?;
    if total == 0 || row.contains(&0) || column.contains(&0) {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "zero Fisher margin",
        ));
    }
    if total > (1_u128 << 53) {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "exact tail ordering needs counts within f64 integer precision",
        ));
    }
    if table.rows() == 2 && table.columns() == 2 {
        let [a, b, c, d] = four(table)?;
        let mut work = 0;
        let (p, tables) = fisher_2x2(
            [a as u128, b as u128, c as u128, d as u128],
            options.alternative(),
            &mut work,
            options.max_work(),
            options.max_workspace_bytes() - margin_bytes,
        )?;
        let statistic = if b == 0 || c == 0 {
            f64::INFINITY
        } else {
            (a as f64 * d as f64) / (b as f64 * c as f64)
        };
        return Ok(ExactContingencyResult {
            statistic,
            p_value: p,
            p_value_lower: p,
            design: ContingencyDesign::FixedMargins,
            tables,
        });
    }
    if options.alternative() != TestAlternative::TwoSided {
        return Err(invalid(
            scioxide_core::argument!("options.alternative"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "R×C Fisher supports two-sided ordering only",
        ));
    }
    let n = usize::try_from(total).map_err(|_| {
        invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "R×C total exceeds factorial domain",
        )
    })?;
    let mut constant = -log_factorial(n)?;
    for &sum in row.iter().chain(column.iter()) {
        constant += log_factorial(usize::try_from(sum).map_err(|_| {
            invalid(
                scioxide_core::argument!("table"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "margin exceeds factorial domain",
            )
        })?)?;
    }
    let mut observed_score = 0.0;
    for &count in table.counts() {
        observed_score -= log_factorial(count)?;
    }
    let mut runner = RxC {
        rows: table.rows(),
        columns: table.columns(),
        row_remaining: row,
        column_remaining: column,
        cells: try_filled(
            table.counts().len(),
            0,
            scioxide_core::quantity!("fisher_table"),
        )?,
        observed_score,
        tail_log: f64::NEG_INFINITY,
        total_log: f64::NEG_INFINITY,
        tables: 0,
        options: &options,
    };
    runner.visit(0)?;
    let p = (runner.tail_log - runner.total_log).exp().min(1.0);
    Ok(ExactContingencyResult {
        statistic: (constant + observed_score).exp(),
        p_value: p,
        p_value_lower: p,
        design: ContingencyDesign::FixedMargins,
        tables: runner.tables,
    })
}

/// Options of [`barnard_exact_test_with_options`] and [`boschloo_exact_test_with_options`].
///
/// The default uses the default [`ClassicalOptions`], the pooled variance and
/// a tolerance of `10⁻⁹`.
#[derive(Clone, Copy, Debug)]
pub struct UnconditionalOptions {
    /// Alternative and work and byte limits.
    classical: ClassicalOptions,
    /// Variance estimate of Barnard's Wald statistic.
    pooled: crate::VariancePooling,
    /// Absolute width of the p-value bracket.
    tolerance: f64,
}

impl UnconditionalOptions {
    /// Returns the classical options, which supply the alternative and the
    /// work and byte limits. Asymptotic calibration is rejected.
    pub const fn classical(&self) -> ClassicalOptions {
        self.classical
    }
    /// Sets the classical options.
    pub const fn with_classical(mut self, value: ClassicalOptions) -> Self {
        self.classical = value;
        self
    }
    /// Returns the variance estimate of Barnard's Wald statistic.
    pub const fn pooled(&self) -> crate::VariancePooling {
        self.pooled
    }
    /// Sets the variance estimate of Barnard's Wald statistic.
    pub const fn with_pooled(mut self, value: crate::VariancePooling) -> Self {
        self.pooled = value;
        self
    }
    /// Returns the absolute width to which the maximum over the nuisance
    /// probability is bracketed. The bracket accounts for the branch and bound
    /// but not for rounding errors.
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }
    /// Sets the width of the p-value bracket, in `(0, 1)`.
    pub const fn with_tolerance(mut self, value: f64) -> Self {
        self.tolerance = value;
        self
    }
}

impl Default for UnconditionalOptions {
    fn default() -> Self {
        Self {
            classical: ClassicalOptions::default(),
            pooled: crate::VariancePooling::Pooled,
            tolerance: 1e-9,
        }
    }
}

fn wald(i: usize, j: usize, n1: usize, n2: usize, pooled: bool) -> f64 {
    let a = i as f64 / n1 as f64;
    let b = j as f64 / n2 as f64;
    let variance = if pooled {
        let p = (i as f64 + j as f64) / (n1 as f64 + n2 as f64);
        p * (1.0 - p) * (1.0 / n1 as f64 + 1.0 / n2 as f64)
    } else {
        a * (1.0 - a) / n1 as f64 + b * (1.0 - b) / n2 as f64
    };
    if variance == 0.0 {
        if a == b {
            0.0
        } else {
            (a - b).signum() * f64::INFINITY
        }
    } else {
        (a - b) / variance.sqrt()
    }
}

fn optimize(
    coeff: Vec<f64>,
    options: UnconditionalOptions,
    work: &mut usize,
    used_bytes: usize,
) -> Result<(f64, f64), SciOxideError> {
    let size = coeff.len();
    let mut best = coeff[0].max(coeff[size - 1]);
    let mut stack = try_with_capacity(1, scioxide_core::quantity!("nuisance_interval_stack"))?;
    stack.push(coeff);
    while let Some((position, bound)) = stack
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.iter().copied().fold(0.0_f64, f64::max)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
    {
        if bound <= best + options.tolerance {
            break;
        }
        let current = stack.swap_remove(position);
        let n = current.len();
        charge(
            work,
            n.checked_mul(n).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?,
            options.classical.max_work(),
        )?;
        let mut left = try_filled(
            n,
            0.0,
            scioxide_core::quantity!("nuisance_left_coefficients"),
        )?;
        let mut right = try_filled(
            n,
            0.0,
            scioxide_core::quantity!("nuisance_right_coefficients"),
        )?;
        let mut row = current;
        for level in 0..n {
            left[level] = row[0];
            right[n - 1 - level] = row[n - 1 - level];
            for index in 0..n - 1 - level {
                row[index] = 0.5 * (row[index] + row[index + 1]);
            }
        }
        best = best.max(left[n - 1]);
        for child in [left, right] {
            let upper = child.iter().copied().fold(0.0_f64, f64::max);
            if upper > best + options.tolerance {
                let slots = stack
                    .len()
                    .checked_add(3)
                    .and_then(|v| v.checked_mul(n))
                    .ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
                    })?;
                let storage = slots
                    .checked_mul(core::mem::size_of::<f64>())
                    .and_then(|v| v.checked_add(used_bytes))
                    .ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
                    })?;
                crate::error_helpers::limit(
                    storage,
                    options.classical.max_workspace_bytes(),
                    scioxide_core::quantity!("exact_bytes"),
                )?;
                stack.push(child);
            }
        }
    }
    Ok((best.min(1.0), (best + options.tolerance).min(1.0)))
}

fn unconditional(
    table: ContingencyView<'_>,
    options: UnconditionalOptions,
    boschloo: bool,
    work: &mut usize,
) -> Result<ExactContingencyResult, SciOxideError> {
    let [a, b, c, d] = four(table)?;
    if options.classical.calibration() == TestCalibration::Asymptotic
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || options.tolerance >= 1.0
    {
        return Err(invalid(
            scioxide_core::argument!("options"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid unconditional exact options",
        ));
    }
    let n1 = a
        .checked_add(c)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let n2 = b
        .checked_add(d)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    if n1 == 0 || n2 == 0 {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "unconditional test needs two positive column sizes",
        ));
    }
    let n = n1
        .checked_add(n2)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    if n as u128 > (1_u128 << 53) {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "exact ordering needs counts within f64 integer precision",
        ));
    }
    let states = n1
        .checked_add(1)
        .and_then(|v| n2.checked_add(1).and_then(|w| v.checked_mul(w)))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let required = scioxide_core::backend::checked_add(
        states,
        *work,
        scioxide_core::quantity!("unconditional_work"),
    )?;
    crate::error_helpers::work(
        required,
        options.classical.max_work(),
        scioxide_core::BudgetKind::Node,
        "unconditional nuisance tables exceed options.max_work",
    )?;
    let count = n
        .checked_add(1)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let arrays = count
        .checked_mul(5)
        .and_then(|v| v.checked_mul(core::mem::size_of::<f64>()))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        arrays,
        options.classical.max_workspace_bytes(),
        scioxide_core::quantity!("exact_bytes"),
    )?;
    let mut logfac = try_filled(
        count,
        0.0_f64,
        scioxide_core::quantity!("unconditional_factorial_logs"),
    )?;
    for index in 2..=n {
        logfac[index] = logfac[index - 1] + (index as f64).ln();
    }
    let choose =
        |size: usize, selected: usize| logfac[size] - logfac[selected] - logfac[size - selected];
    let observed = if boschloo {
        let (p, _) = fisher_2x2(
            [a as u128, b as u128, c as u128, d as u128],
            options.classical.alternative(),
            work,
            options.classical.max_work(),
            options.classical.max_workspace_bytes(),
        )?;
        p
    } else {
        wald(
            a,
            b,
            n1,
            n2,
            options.pooled == crate::VariancePooling::Pooled,
        )
    };
    let mut coeff = try_filled(
        count,
        0.0_f64,
        scioxide_core::quantity!("unconditional_bernstein_coefficients"),
    )?;
    for i in 0..=n1 {
        for j in 0..=n2 {
            charge(work, 1, options.classical.max_work())?;
            let statistic = if boschloo {
                fisher_2x2(
                    [i as u128, j as u128, (n1 - i) as u128, (n2 - j) as u128],
                    options.classical.alternative(),
                    work,
                    options.classical.max_work(),
                    options.classical.max_workspace_bytes(),
                )?
                .0
            } else {
                wald(
                    i,
                    j,
                    n1,
                    n2,
                    options.pooled == crate::VariancePooling::Pooled,
                )
            };
            let selected = if boschloo {
                statistic <= observed + 64.0 * f64::EPSILON
            } else {
                match options.classical.alternative() {
                    TestAlternative::Less => statistic <= observed,
                    TestAlternative::Greater => statistic >= observed,
                    TestAlternative::TwoSided => statistic.abs() >= observed.abs(),
                }
            };
            if selected {
                let k = i + j;
                coeff[k] += (choose(n1, i) + choose(n2, j) - choose(n, k)).exp();
            }
        }
    }
    for value in &mut coeff {
        *value = value.min(1.0);
    }
    let (lower, upper) = optimize(coeff, options, work, arrays)?;
    Ok(ExactContingencyResult {
        statistic: observed,
        p_value: upper,
        p_value_lower: lower,
        design: ContingencyDesign::FixedColumns,
        tables: states,
    })
}

/// Computes the result of [`barnard_exact_test_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`barnard_exact_test_with_options`] for the fixed default settings.
pub fn barnard_exact_test(
    table: ContingencyView<'_>,
) -> Result<ExactContingencyResult, SciOxideError> {
    barnard_exact_test_with_options(table, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("table"),
        )
    })
}
/// Performs Barnard's unconditional exact test of equal proportions in a
/// `2 × 2` table.
///
/// The columns are two independent binomial samples of fixed sizes `n₁` and
/// `n₂`, and the first row counts successes. Tables are ordered by the Wald
/// statistic `(p̂₁ − p̂₂)/σ̂`, with the pooled or unpooled variance selected by
/// [`UnconditionalOptions::pooled`]. The p-value is the maximum over the common
/// success probability `π` of the tail probability. The maximum is found by
/// branch and bound on the Bernstein form of the tail polynomial in `π`, to the
/// width [`UnconditionalOptions::tolerance`]. When only the column totals are
/// fixed by design, the test is usually more powerful than Fisher's.
///
/// # Errors
///
/// - `InvalidInput` if the table is not `2 × 2`, a column total is zero, or
///   the grand total exceeds `2⁵³`.
/// - `InvalidInput` if the calibration is [`TestCalibration::Asymptotic`], or
///   `tolerance` is not in `(0, 1)`.
/// - `ConvergenceFailure` if the `(n₁ + 1)(n₂ + 1)` tables or the branch and
///   bound exceed `max_work`.
/// - `ResourceLimit` if the scratch exceeds `max_workspace_bytes`.
///
/// # References
///
/// G. A. Barnard, "A new test for 2 × 2 tables", *Nature* 156 (1945), 177.
pub fn barnard_exact_test_with_options(
    table: ContingencyView<'_>,
    options: UnconditionalOptions,
) -> Result<ExactContingencyResult, SciOxideError> {
    let mut work = 0;
    unconditional(table, options, false, &mut work)
}

/// Computes the result of [`boschloo_exact_test_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`boschloo_exact_test_with_options`] for the fixed default settings.
pub fn boschloo_exact_test(
    table: ContingencyView<'_>,
) -> Result<ExactContingencyResult, SciOxideError> {
    boschloo_exact_test_with_options(table, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("table"),
        )
    })
}
/// Performs Boschloo's unconditional exact test of equal proportions in a
/// `2 × 2` table.
///
/// The design and the maximization over the nuisance probability are those of
/// [`barnard_exact_test_with_options`], but tables are ordered by their one-sided Fisher p-value.
/// It is uniformly more powerful than Fisher's one-sided exact test. The two-sided
/// p-value is twice the smaller one-sided p-value, capped at 1.
///
/// # Errors
///
/// - `InvalidInput` if the table is not `2 × 2`, a column total is zero, or
///   the grand total exceeds `2⁵³`.
/// - `InvalidInput` if the calibration is [`TestCalibration::Asymptotic`], or
///   `tolerance` is not in `(0, 1)`.
/// - `ConvergenceFailure` if the `(n₁ + 1)(n₂ + 1)` tables or the branch and
///   bound exceed `max_work`.
/// - `ResourceLimit` if the scratch exceeds `max_workspace_bytes`.
///
/// # References
///
/// R. D. Boschloo, "Raised conditional level of significance for the 2 × 2-table
/// when testing the equality of two probabilities", *Statist. Neerlandica* 24
/// (1970), 1–35.
pub fn boschloo_exact_test_with_options(
    table: ContingencyView<'_>,
    options: UnconditionalOptions,
) -> Result<ExactContingencyResult, SciOxideError> {
    let mut work = 0;
    if options.classical.alternative() != TestAlternative::TwoSided {
        return unconditional(table, options, true, &mut work);
    }
    let less = unconditional(
        table,
        options.with_classical(options.classical.with_alternative(TestAlternative::Less)),
        true,
        &mut work,
    )?;
    let greater = unconditional(
        table,
        options.with_classical(options.classical.with_alternative(TestAlternative::Greater)),
        true,
        &mut work,
    )?;
    Ok(ExactContingencyResult {
        statistic: (2.0 * less.statistic.min(greater.statistic)).min(1.0),
        p_value: (2.0 * less.p_value.min(greater.p_value)).min(1.0),
        p_value_lower: (2.0 * less.p_value_lower.min(greater.p_value_lower)).min(1.0),
        design: ContingencyDesign::FixedColumns,
        tables: less.tables.checked_add(greater.tables).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choose(n: usize, k: usize) -> u64 {
        if k > n {
            return 0;
        }
        (0..k.min(n - k)).fold(1_u64, |v, i| v * (n - i) as u64 / (i + 1) as u64)
    }

    fn reference_fisher(cells: [usize; 4]) -> f64 {
        let [a, b, c, d] = cells;
        let row = a + b;
        let column = a + c;
        let total = row + c + d;
        let low = row.saturating_add(column).saturating_sub(total);
        let high = row.min(column);
        let denominator = choose(total, row) as f64;
        let observed = choose(column, a) as f64 * choose(total - column, row - a) as f64;
        (low..=high)
            .filter(|&x| {
                choose(column, x) as f64 * choose(total - column, row - x) as f64 <= observed
            })
            .map(|x| {
                choose(column, x) as f64 * choose(total - column, row - x) as f64 / denominator
            })
            .sum()
    }

    #[test]
    fn fisher_enumerated_two_by_two_and_rectangular() {
        let cells = [1, 3, 3, 1];
        let table = ContingencyView::new(&cells, 2, 2).unwrap();
        let result = fisher_exact_test_with_options(table, ClassicalOptions::default()).unwrap();
        assert!((result.p_value - 34.0 / 70.0).abs() < 1e-13);
        assert_eq!(result.tables, 5);
        assert!((result.statistic - 1.0 / 9.0).abs() < 1e-15);
        let rectangular = [2, 0, 0, 0, 1, 1];
        let table = ContingencyView::new(&rectangular, 2, 3).unwrap();
        let result = fisher_exact_test_with_options(table, ClassicalOptions::default()).unwrap();
        assert!((result.p_value - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(result.tables, 4);
    }

    #[test]
    fn unconditional_small_tables_and_budgets() {
        let cells = [1, 2, 2, 1];
        let table = ContingencyView::new(&cells, 2, 2).unwrap();
        let options = UnconditionalOptions::default()
            .with_tolerance(1e-6)
            .with_classical(ClassicalOptions::default().with_max_work(1_000_000));
        for result in [
            barnard_exact_test_with_options(table, options).unwrap(),
            boschloo_exact_test_with_options(table, options).unwrap(),
        ] {
            assert!((0.0..=1.0).contains(&result.p_value));
            assert!(result.p_value >= result.p_value_lower);
            assert!(result.p_value - result.p_value_lower <= 2e-6);
        }
        let tiny = options.with_classical(ClassicalOptions::default().with_max_work(1));
        assert!(barnard_exact_test_with_options(table, tiny).is_err());
        let tiny = ContingencyView::new(&[1, 0, 0, 1], 2, 2).unwrap();
        let barnard = barnard_exact_test_with_options(tiny, options).unwrap();
        let boschloo = boschloo_exact_test_with_options(tiny, options).unwrap();
        assert!((barnard.p_value_lower - 0.5).abs() < 1e-12);
        assert!((boschloo.p_value_lower - 0.5).abs() < 1e-12);
    }

    #[test]
    fn exhaustive_small_fisher_probability_order_and_permutations() {
        for a in 0..=3 {
            for b in 0..=3 {
                for c in 0..=3 {
                    for d in 0..=3 {
                        if a + b == 0 || c + d == 0 || a + c == 0 || b + d == 0 {
                            continue;
                        }
                        let cells = [a, b, c, d];
                        let table = ContingencyView::new(&cells, 2, 2).unwrap();
                        let p = fisher_exact_test_with_options(table, ClassicalOptions::default())
                            .unwrap()
                            .p_value;
                        assert!((p - reference_fisher(cells)).abs() < 1e-12, "{cells:?}");
                        let flipped = [c, d, a, b];
                        let q = fisher_exact_test_with_options(
                            ContingencyView::new(&flipped, 2, 2).unwrap(),
                            ClassicalOptions::default(),
                        )
                        .unwrap()
                        .p_value;
                        assert!((p - q).abs() < 1e-12, "{cells:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn barnard_published_vaccine_example_and_fisher_contrast() {
        let cells = [7, 12, 8, 3];
        let table = ContingencyView::new(&cells, 2, 2).unwrap();
        let classical = ClassicalOptions::default()
            .with_alternative(TestAlternative::Less)
            .with_max_work(10_000_000);
        let fisher = fisher_exact_test_with_options(table, classical).unwrap();
        assert!((fisher.p_value - 0.0640).abs() < 1e-4);
        let barnard = barnard_exact_test_with_options(
            table,
            UnconditionalOptions::default()
                .with_classical(classical)
                .with_tolerance(1e-5),
        )
        .unwrap();
        assert!((barnard.p_value_lower - 0.03407).abs() < 5e-4);
        assert!(
            barnard.p_value - barnard.p_value_lower <= 1.000_001e-5,
            "lower={} upper={}",
            barnard.p_value_lower,
            barnard.p_value
        );
    }
}
