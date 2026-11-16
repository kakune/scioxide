//! Order statistics: quantiles, ranks, modes, expectiles and percentiles of a score.

use crate::arithmetic::Compensated;
use crate::arithmetic::count_ratio;
use crate::error_helpers::invalid;
use crate::ranking::{finite_cmp, tie_runs};
use crate::{QuantileMethod, QuantileProbability, QuantileResult};
use alloc::vec::Vec;
use scioxide_core::Scaled;
use scioxide_core::backend::try_filled;
use scioxide_core::{Argument, NonFiniteOrigin, RealScalar, SciOxideError};

mod expectile;

/// Returns the percentile of a score from the counts `lower = #{x < score}`
/// and `upper = #{x ≤ score}` in a sample of `len` observations.
pub(crate) fn percentile_from_counts<S: RealScalar>(
    len: usize,
    lower: usize,
    upper: usize,
    score: S,
    kind: PercentileKind,
) -> Result<OrderResult<S>, SciOxideError> {
    if len == 0 {
        return Err(invalid(
            scioxide_core::argument!("samples"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty retained sample",
        ));
    }
    if !score.is_finite() {
        return Err(invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "score must be finite",
        ));
    }
    let rank = match kind {
        PercentileKind::Strict => Scaled::<f64>::from_value(lower as f64)?,
        PercentileKind::Weak => Scaled::<f64>::from_value(upper as f64)?,
        PercentileKind::Mean | PercentileKind::Rank => {
            let extra = usize::from(kind == PercentileKind::Rank && upper > lower);
            Scaled::<f64>::from_value(lower as f64)?
                .add(Scaled::<f64>::from_value(upper as f64)?)?
                .add(Scaled::<f64>::from_value(extra as f64)?)?
                .div(Scaled::<f64>::from_value(2.0)?)?
        }
    };
    let percentage = rank
        .div(Scaled::<f64>::from_value(len as f64)?)?
        .mul(Scaled::<f64>::from_value(100.0)?)?
        .round_to()?;
    Ok(OrderResult::PercentileOfScore { percentage, kind })
}

/// Rule for ranking tied observations. Ranks start at one.
///
/// A group of `t` tied observations occupies the ranks `r + 1, …, r + t`,
/// where `r` observations are smaller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankMethod {
    /// The mean of the occupied ranks, `r + (t + 1)/2`, which can be a
    /// half-integer.
    Average,
    /// The lowest occupied rank, `r + 1`.
    Minimum,
    /// The highest occupied rank, `r + t`.
    Maximum,
    /// The rank of the value among the distinct values, so ranks have no gaps.
    Dense,
    /// Distinct ranks, assigned to tied observations in input order.
    Ordinal,
}

/// A rank: an integer or a half-integer, stored exactly as the mean of two
/// integer bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank {
    lower: usize,
    upper: usize,
}
impl Rank {
    /// Creates a rank from one-based bounds `lower ≤ upper`.
    pub(crate) const fn from_bounds(lower: usize, upper: usize) -> Self {
        Self { lower, upper }
    }
    /// Returns the integer bounds `(lower, upper)`. The rank is their mean.
    pub const fn bounds(self) -> (usize, usize) {
        (self.lower, self.upper)
    }
    /// Returns the rank as an exact `f64`.
    ///
    /// # Errors
    ///
    /// - [`ScalarConversion`](scioxide_core::SciOxideErrorKind::ScalarConversion)
    ///   if a bound exceeds 2⁵³ and is not exactly representable.
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if the
    ///   rank is a half-integer of at least 2⁵², which `f64` cannot represent.
    pub fn to_f64(self) -> Result<f64, SciOxideError> {
        let low = f64::try_from_usize_exact(self.lower)?;
        let high = f64::try_from_usize_exact(self.upper)?;
        let value = low + (high - low) / 2.0;
        if self.lower % 2 != self.upper % 2 && value >= 4_503_599_627_370_496.0 {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "half-integer rank is not representable in f64",
            ));
        }
        Ok(value)
    }
}

impl<S: RealScalar> OrderResult<S> {
    /// Returns the result as one scalar.
    ///
    /// # Errors
    /// Fails if a percentage or tie correction is not representable in `S`.
    pub(crate) fn scalar(self) -> Result<S, SciOxideError> {
        match self {
            Self::Quantile(q) => Ok(q.value()),
            Self::Mode(m) => Ok(m.value),
            Self::Expectile(e) => Ok(e.value),
            Self::Iqr { value, .. } => Ok(value),
            Self::PercentileOfScore { percentage, .. } => crate::error_helpers::number(percentage),
            Self::TieCorrection(v) => crate::error_helpers::number(v),
        }
    }
}

/// Convention for the percentile of a score `s` in a sample of `n`
/// observations.
///
/// With `L` observations less than `s` and `U` observations at most `s`, the
/// percentile is `100 · c / n`, where `c` depends on the variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PercentileKind {
    /// `c = L`: the percentage of observations below `s`.
    Strict,
    /// `c = U`: the percentage of observations at most `s`.
    Weak,
    /// `c = (L + U)/2`: the mean of the strict and weak percentages.
    Mean,
    /// `c = (L + U + 1)/2`, the mean rank of the observations equal to `s`,
    /// if `s` occurs in the sample, and `c = L` otherwise.
    Rank,
}

/// The most frequent value of a sample, with its count.
///
/// If several values share the highest count, the smallest one is returned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeResult<S> {
    /// The modal value. `−0.0` and `+0.0` count as one value, returned as
    /// `+0.0`.
    pub value: S,
    /// The number of observations equal to `value`.
    pub count: usize,
}

/// The `p`-expectile of a sample, with its relative balance residual.
///
/// For observations `xᵢ` with weights `wᵢ` and `0 < p < 1`, the `p`-expectile
/// is the unique root `t` of the balance equation
///
/// ```text
/// B(t) = (1 − p) Σ_{xᵢ ≤ t} wᵢ (xᵢ − t) + p Σ_{xᵢ > t} wᵢ (xᵢ − t) = 0,
/// ```
///
/// which minimizes the asymmetric squared loss `Σ wᵢ |p − 1{xᵢ ≤ t}| (xᵢ − t)²`.
/// The `1/2`-expectile is the weighted mean. At `p = 0` and `p = 1` the
/// result is the smallest and the largest observation.
///
/// `B` is strictly decreasing and linear between adjacent observations. The
/// root is bracketed by bisection over the order statistics with median
/// selection, in expected O(n) time. The sign of `B` at each pivot is exact.
/// The result is the closed-form root on the final bracket, so no iteration
/// tolerance is involved.
///
/// # References
///
/// W. K. Newey and J. L. Powell, "Asymmetric least squares estimation and
/// testing", *Econometrica* 55(4), 819–847, 1987.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpectileResult<S> {
    /// The expectile `t`.
    pub value: S,
    /// The asymmetry level `p`.
    pub probability: QuantileProbability,
    /// The relative residual `|B(t)| / Σ wᵢ cᵢ |xᵢ − t|` at the returned `t`,
    /// with `cᵢ = 1 − p` for `xᵢ ≤ t` and `cᵢ = p` otherwise.
    ///
    /// It lies in `[0, 1]` and is 0 for an exact balance or a constant
    /// sample. It measures the balance, not the error in `t`.
    pub relative_balance_residual: f64,
}

/// Order statistic evaluated on each lane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum OrderedStatistic<S: RealScalar> {
    /// Sample quantile with a Hyndman–Fan definition.
    Quantile(QuantileProbability, QuantileMethod),
    /// Smallest mode and its count.
    Mode,
    /// Expectile.
    Expectile(QuantileProbability),
    /// Interquartile range `Q(3/4) − Q(1/4)`.
    Iqr(QuantileMethod),
    /// Percentile of a score.
    PercentileOfScore(S, PercentileKind),
    /// Tie correction `1 − Σ (t³ − t)/(n³ − n)` over tie groups of size `t`.
    TieCorrection,
}

/// Value of an order statistic, with the definition used to compute it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderResult<S: RealScalar> {
    /// A sample quantile.
    Quantile(QuantileResult<S>),
    /// A mode.
    Mode(ModeResult<S>),
    /// An expectile.
    Expectile(ExpectileResult<S>),
    /// The interquartile range `Q(3/4) − Q(1/4)`.
    Iqr {
        /// The interquartile range.
        value: S,
        /// The Hyndman–Fan definition used for both quartiles.
        method: QuantileMethod,
    },
    /// The percentile of a score.
    PercentileOfScore {
        /// The percentile, in `[0, 100]`.
        percentage: f64,
        /// The convention for observations equal to the score.
        kind: PercentileKind,
    },
    /// The tie correction `1 − Σ (t³ − t)/(n³ − n)` over tie groups of size
    /// `t`, as used by rank tests. It is 1 for fewer than two observations.
    TieCorrection(f64),
}

/// Returns the mode or the tie correction of a nonempty sorted sequence.
fn tie_run_statistic<T, S: RealScalar>(
    sorted: &[T],
    value: impl Fn(&T) -> S,
    statistic: OrderedStatistic<S>,
) -> Result<OrderResult<S>, SciOxideError> {
    let len = sorted.len();
    let runs = || tie_runs(sorted, |a, b| value(a) == value(b));
    match statistic {
        OrderedStatistic::Mode => {
            let mut result = ModeResult {
                value: value(&sorted[0]),
                count: 0,
            };
            for (run, members) in runs() {
                if run.len() > result.count {
                    result = ModeResult {
                        value: value(&members[0]),
                        count: run.len(),
                    };
                }
            }
            Ok(OrderResult::Mode(result))
        }
        OrderedStatistic::TieCorrection => {
            if len < 2 {
                return Ok(OrderResult::TieCorrection(1.0));
            }
            let n = Scaled::<f64>::from_value(len as f64)?;
            let denominator = n.mul(n)?.sub(Scaled::<f64>::ONE)?;
            let mut correction = Compensated::<Scaled<f64>>::zero();
            for (run, _) in runs() {
                let t = Scaled::<f64>::from_value(run.len() as f64)?;
                // Sum positive complements instead of subtracting nearly equal cubes.
                let contribution = count_ratio(run.len(), len)?
                    .mul(n.sub(t)?.mul(n.add(t)?)?.div(denominator)?)?;
                correction = correction.added(contribution)?;
            }
            Ok(OrderResult::TieCorrection(correction.total()?.round_to()?))
        }
        _ => Err(invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "statistic does not use tie runs",
        )),
    }
}

#[derive(Clone, Copy)]
struct Entry<S> {
    value: S,
    weight: S,
    index: usize,
    count: usize,
}

/// Reusable O(n) storage for order statistics of one lane.
///
/// Percentiles of a score count in O(n). Unweighted quantiles, the
/// interquartile range and expectiles use selection in expected O(n). Other
/// statistics sort in O(n log n). `−0.0` is stored as `+0.0`.
pub(crate) struct RawOrderWorkspace<S: RealScalar> {
    entries: Vec<Entry<S>>,
    values: Vec<S>,
    pub(crate) ranks: Vec<Option<Rank>>,
    len: usize,
    masses: Vec<(S, S)>,
    frequencies: Vec<(S, u64)>,
}
impl<S: RealScalar> Default for RawOrderWorkspace<S> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            values: Vec::new(),
            ranks: Vec::new(),
            len: 0,
            masses: Vec::new(),
            frequencies: Vec::new(),
        }
    }
}
impl<S: RealScalar> RawOrderWorkspace<S> {
    pub(crate) fn required_bytes(n: usize) -> Result<usize, SciOxideError> {
        crate::estimation::storage_bytes(&[
            (n, core::mem::size_of::<Entry<S>>()),
            (n, core::mem::size_of::<S>()),
            (n, core::mem::size_of::<Option<Rank>>()),
            (n, core::mem::size_of::<(S, S)>()),
            (n, core::mem::size_of::<(S, u64)>()),
        ])
    }
    /// Grows the storage to hold `observations` values.
    ///
    /// # Errors
    /// Fails on allocation failure, leaving the previous storage in place.
    pub(crate) fn reserve(&mut self, observations: usize) -> Result<(), SciOxideError> {
        if self.entries.len() >= observations {
            return Ok(());
        }
        let entry = Entry {
            value: S::ZERO,
            weight: S::ONE,
            index: 0,
            count: 1,
        };
        let entries = try_filled(
            observations,
            entry,
            scioxide_core::quantity!("order_entries"),
        )?;
        let values = try_filled(
            observations,
            S::ZERO,
            scioxide_core::quantity!("order_values"),
        )?;
        let ranks = try_filled(observations, None, scioxide_core::quantity!("order_ranks"))?;
        let masses = try_filled(
            observations,
            (S::ZERO, S::ZERO),
            scioxide_core::quantity!("order_mass_scratch"),
        )?;
        let frequencies = try_filled(
            observations,
            (S::ZERO, 0),
            scioxide_core::quantity!("order_frequency_scratch"),
        )?;
        *self = Self {
            masses,
            frequencies,
            entries,
            values,
            ranks,
            len: 0,
        };
        Ok(())
    }
    /// Returns the largest lane length the storage holds.
    pub(crate) fn capacity(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn begin(&mut self, len: usize, argument: Argument) -> Result<(), SciOxideError> {
        if len > self.capacity() {
            return Err(invalid(
                argument,
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient order workspace",
            ));
        }
        self.len = 0;
        Ok(())
    }
    pub(crate) fn push_scalar(&mut self, value: S) {
        self.values[self.len] = if value == S::ZERO { S::ZERO } else { value };
        self.len += 1;
    }
    pub(crate) fn evaluate_scalar(
        &mut self,
        statistic: OrderedStatistic<S>,
    ) -> Result<OrderResult<S>, SciOxideError> {
        let values = &mut self.values[..self.len];
        match statistic {
            OrderedStatistic::Quantile(p, method) => {
                crate::empirical::quantile_in_place(values, p, method).map(OrderResult::Quantile)
            }
            OrderedStatistic::Iqr(method) => {
                let lo = crate::empirical::quantile_in_place(
                    values,
                    QuantileProbability::new(0.25)?,
                    method,
                )?
                .value();
                let hi = crate::empirical::quantile_in_place(
                    values,
                    QuantileProbability::new(0.75)?,
                    method,
                )?
                .value();
                Ok(OrderResult::Iqr {
                    value: Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(hi))?
                        .sub(Scaled::<f64>::from_value(
                            scioxide_core::RealScalar::into_f64(lo),
                        )?)?
                        .round_to()?,
                    method,
                })
            }
            _ => Err(invalid(
                scioxide_core::argument!("statistic"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "statistic does not use scalar selection",
            )),
        }
    }
    pub(crate) fn push(
        &mut self,
        value: S,
        weight: S,
        index: usize,
        argument: Argument,
    ) -> Result<(), SciOxideError> {
        if !weight.is_finite() || weight < S::ZERO {
            return Err(invalid(
                argument,
                scioxide_core::InvalidInputReason::OutOfRange,
                "weights must be finite and nonnegative",
            ));
        }
        if weight == S::ZERO {
            return Ok(());
        }
        if !value.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "nonfinite observation",
            ));
        }
        let slot = self.entries.get_mut(self.len).ok_or_else(|| {
            invalid(
                argument,
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient order workspace",
            )
        })?;
        *slot = Entry {
            value: if value == S::ZERO { S::ZERO } else { value },
            weight,
            index,
            count: 1,
        };
        self.len += 1;
        Ok(())
    }
    pub(crate) fn push_frequency(
        &mut self,
        value: S,
        count: usize,
        index: usize,
        argument: Argument,
    ) -> Result<(), SciOxideError> {
        if count == 0 {
            return Ok(());
        }
        self.push(value, S::ONE, index, argument)?;
        self.entries[self.len - 1].count = count;
        Ok(())
    }
    pub(crate) fn evaluate_weighted(
        &mut self,
        statistic: OrderedStatistic<S>,
        frequency: bool,
    ) -> Result<OrderResult<S>, SciOxideError> {
        if frequency {
            let OrderedStatistic::Quantile(p, method) = statistic else {
                return Err(invalid(
                    scioxide_core::argument!("statistic"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "frequency weights support quantiles only",
                ));
            };
            for (entry, pair) in self.entries[..self.len].iter().zip(&mut self.frequencies) {
                *pair = (entry.value, entry.count as u64);
            }
            crate::frequency_quantile_in_place(&mut self.frequencies[..self.len], p, method)
                .map(OrderResult::Quantile)
        } else {
            match statistic {
                OrderedStatistic::Expectile(p) => self.expectile(p).map(OrderResult::Expectile),
                OrderedStatistic::Quantile(p, QuantileMethod::NearestRank) => {
                    for (entry, pair) in self.entries[..self.len].iter().zip(&mut self.masses) {
                        *pair = (entry.value, entry.weight);
                    }
                    crate::weighted_inverse_ecdf_quantile_in_place(&mut self.masses[..self.len], p)
                        .map(OrderResult::Quantile)
                }
                _ => Err(invalid(
                    scioxide_core::argument!("statistic"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "relative masses support inverse ECDF and expectiles only",
                )),
            }
        }
    }
    pub(crate) fn sort(&mut self, argument: Argument) -> Result<(), SciOxideError> {
        if self.len == 0 {
            return Err(invalid(
                argument,
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty retained sample",
            ));
        }
        self.entries[..self.len]
            .sort_unstable_by(|a, b| finite_cmp(&a.value, &b.value).then(a.index.cmp(&b.index)));
        Ok(())
    }
    pub(crate) fn load(&mut self, samples: &[S]) -> Result<(), SciOxideError> {
        let argument = scioxide_core::argument!("samples");
        self.begin(samples.len(), argument)?;
        for (index, &value) in samples.iter().enumerate() {
            self.push(value, S::ONE, index, argument)?;
        }
        self.sort(argument)
    }
    /// Evaluates `statistic` on unweighted observations.
    ///
    /// # Errors
    /// Fails for insufficient capacity, an empty or non-finite sample, a
    /// non-finite score, or a result that is not representable.
    pub(crate) fn evaluate(
        &mut self,
        samples: &[S],
        statistic: OrderedStatistic<S>,
    ) -> Result<OrderResult<S>, SciOxideError> {
        self.begin(samples.len(), scioxide_core::argument!("samples"))?;
        match statistic {
            OrderedStatistic::PercentileOfScore(score, kind) => {
                let (lower, upper, finite) =
                    samples
                        .iter()
                        .fold((0, 0, true), |(lower, upper, finite), &value| {
                            (
                                lower + usize::from(value < score),
                                upper + usize::from(value <= score),
                                finite & value.is_finite(),
                            )
                        });
                if !finite {
                    return Err(SciOxideError::non_finite(
                        NonFiniteOrigin::Input,
                        "nonfinite observation",
                    ));
                }
                percentile_from_counts(samples.len(), lower, upper, score, kind)
            }
            OrderedStatistic::Quantile(..) | OrderedStatistic::Iqr(_) => {
                for &value in samples {
                    self.push_scalar(value);
                }
                self.evaluate_scalar(statistic)
            }
            OrderedStatistic::Expectile(p) => {
                for (index, &value) in samples.iter().enumerate() {
                    self.push(value, S::ONE, index, scioxide_core::argument!("samples"))?;
                }
                self.expectile(p).map(OrderResult::Expectile)
            }
            OrderedStatistic::Mode | OrderedStatistic::TieCorrection => {
                // Tie runs depend only on values: sort plain finite scalars, whose
                // duplicates the pattern-defeating sort groups cheaply.
                for &value in samples {
                    if !value.is_finite() {
                        return Err(SciOxideError::non_finite(
                            NonFiniteOrigin::Input,
                            "nonfinite observation",
                        ));
                    }
                    self.push_scalar(value);
                }
                if self.len == 0 {
                    return Err(invalid(
                        scioxide_core::argument!("samples"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "empty retained sample",
                    ));
                }
                let values = &mut self.values[..self.len];
                crate::ranking::sort_finite(values);
                tie_run_statistic(values, |&value| value, statistic)
            }
        }
    }

    pub(crate) fn rank_sorted(&mut self, method: RankMethod) {
        let runs = tie_runs(&self.entries[..self.len], |a, b| a.value == b.value);
        for (dense, (run, members)) in runs.enumerate() {
            for (position, entry) in members.iter().enumerate() {
                let rank = match method {
                    RankMethod::Average => run.rank(),
                    RankMethod::Minimum => Rank::from_bounds(run.start + 1, run.start + 1),
                    RankMethod::Maximum => Rank::from_bounds(run.end, run.end),
                    RankMethod::Dense => Rank::from_bounds(dense + 1, dense + 1),
                    RankMethod::Ordinal => {
                        Rank::from_bounds(run.start + position + 1, run.start + position + 1)
                    }
                };
                self.ranks[entry.index] = Some(rank);
            }
        }
    }
    fn expectile(&mut self, p: QuantileProbability) -> Result<ExpectileResult<S>, SciOxideError> {
        let entries = &self.entries[..self.len];
        let first = entries.first().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("samples"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty retained sample",
            )
        })?;
        let (low, high, native) = entries.iter().fold(
            (first.value, first.value, true),
            |(low, high, native), entry| {
                (
                    if entry.value < low { entry.value } else { low },
                    if entry.value > high {
                        entry.value
                    } else {
                        high
                    },
                    native
                        & crate::arithmetic::native(entry.value.into_f64(), 64)
                        & crate::arithmetic::native(entry.weight.into_f64(), 64),
                )
            },
        );
        if p.get() == 0.0 || p.get() == 1.0 || low == high {
            return Ok(ExpectileResult {
                value: if p.get() == 1.0 { high } else { low },
                probability: p,
                relative_balance_residual: 0.0,
            });
        }
        let native = native
            && crate::arithmetic::native(p.get(), 64)
            && crate::arithmetic::native(1.0 - p.get(), 64);
        expectile::expectile(&mut self.entries[..self.len], p, low, high, native)
    }
}

/// Returns the ranks of `samples` in input order.
///
/// # Errors
/// Fails for an empty or non-finite sample, or allocation failure.
pub(crate) fn rankdata<S: RealScalar>(
    samples: &[S],
    method: RankMethod,
) -> Result<Vec<Rank>, SciOxideError> {
    let mut workspace = RawOrderWorkspace::default();
    workspace.reserve(samples.len())?;
    workspace.load(samples)?;
    workspace.rank_sorted(method);
    let mut result = scioxide_core::backend::try_with_capacity(
        samples.len(),
        scioxide_core::quantity!("rank_output"),
    )?;
    for rank in &workspace.ranks {
        result.push(rank.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("samples"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "missing finite rank",
            )
        })?);
    }
    Ok(result)
}
