//! Pool-adjacent-violators pools in compensated `f64` arithmetic, for inputs
//! whose values and weights have binary exponents within `±64`.
//!
//! In that range, products have exponents within `±130`, and sums over `usize`
//! counts stay far from overflow. Cancellation, division by the largest mass and
//! weighted squared residuals stay normal `f64` numbers. Pool updates therefore
//! use error-free transformations without range checks, and the results are
//! checked when the fit is written.
use super::{Entry, IsotonicBlock, IsotonicDiagnostics};
use crate::error_helpers::invalid;
use scioxide_core::backend::two_sum;
use scioxide_core::{Argument, CompensatedSum, RealScalar, SciOxideError, compensated_reduce};

/// Mass and weighted sum of a pool, each as a two-part compensated `f64` sum.
#[derive(Clone, Copy)]
pub(super) struct Sums {
    mass: [f64; 2],
    sum: [f64; 2],
}

impl Sums {
    pub(super) const ZERO: Self = Self {
        mass: [0.0; 2],
        sum: [0.0; 2],
    };

    #[inline]
    pub(super) fn single(entry: Entry) -> Self {
        Self {
            mass: [entry.weight, 0.0],
            sum: [entry.weight * entry.value, 0.0],
        }
    }

    /// Merges two pools. Keeps both corrections and the exact rounding error of
    /// the leading sum; only the additions of corrections round, as in Sum2 of
    /// Ogita, Rump and Oishi.
    #[inline]
    pub(super) fn combined(self, right: Self) -> Self {
        let (mass, mass_error) = two_sum(self.mass[0], right.mass[0]);
        let (sum, sum_error) = two_sum(self.sum[0], right.sum[0]);
        Self {
            mass: [mass, (self.mass[1] + right.mass[1]) + mass_error],
            sum: [sum, (self.sum[1] + right.sum[1]) + sum_error],
        }
    }

    /// Returns the pooled weighted mean, clamped to `[low, high]`, the range of the members.
    #[inline]
    pub(super) fn mean(self, low: f64, high: f64) -> f64 {
        ((self.sum[0] + self.sum[1]) / (self.mass[0] + self.mass[1])).clamp(low, high)
    }

    /// Returns both parts of the mass; their sum is the pooled mass.
    pub(super) const fn mass(self) -> [f64; 2] {
        self.mass
    }

    /// Returns both parts of the weighted sum; their sum is the pooled weighted sum.
    pub(super) const fn sum(self) -> [f64; 2] {
        self.sum
    }

    fn check(self, mean: f64) -> Result<(), SciOxideError> {
        if self
            .mass
            .into_iter()
            .chain(self.sum)
            .chain([mean])
            .all(f64::is_finite)
            && self.mass[0] + self.mass[1] > 0.0
        {
            Ok(())
        } else {
            Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "native isotonic pool left its admitted arithmetic frame",
            ))
        }
    }
}

/// Pool sums together with the smallest and largest member values, for clamping the mean.
#[derive(Clone, Copy)]
pub(super) struct Pool {
    start: usize,
    sums: Sums,
    mean: f64,
    low: f64,
    high: f64,
}

impl Pool {
    pub(super) const EMPTY: Self = Self {
        start: 0,
        sums: Sums::ZERO,
        mean: 0.0,
        low: 0.0,
        high: 0.0,
    };

    #[inline]
    fn single(start: usize, entry: Entry) -> Self {
        Self {
            start,
            sums: Sums::single(entry),
            mean: entry.value,
            low: entry.value,
            high: entry.value,
        }
    }

    /// Merges two pools without recomputing the mean. The mean is recomputed at
    /// run boundaries, which avoids a division for each member of a run.
    #[inline]
    fn combined(self, right: Self) -> Self {
        Self {
            start: self.start,
            sums: self.sums.combined(right.sums),
            mean: self.mean,
            low: self.low.min(right.low),
            high: self.high.max(right.high),
        }
    }

    #[inline]
    fn centered(mut self) -> Self {
        self.mean = self.sums.mean(self.low, self.high);
        self
    }
}

#[inline]
fn ordered<const INCREASING: bool>(left: f64, right: f64) -> bool {
    if INCREASING {
        left < right
    } else {
        left > right
    }
}

fn project<S: RealScalar>(value: f64) -> Result<Option<S>, SciOxideError> {
    if value > S::MAX.into_f64() {
        Ok(None)
    } else {
        Ok(Some(S::try_from_f64_lossy(value)?))
    }
}

pub(super) fn fit<S: RealScalar, const INCREASING: bool>(
    entries: &[Entry],
    pools: &mut [Pool],
    fitted: &mut [S],
    blocks: &mut [IsotonicBlock<S>],
    argument: Argument,
) -> Result<IsotonicDiagnostics<S>, SciOxideError> {
    let mut input = entries
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, e)| e.weight > 0.0)
        .peekable();
    let mut count = 0;
    let mut retained = 0;
    while let Some((index, entry)) = input.next() {
        retained += 1;
        let mut pool = Pool::single(if count == 0 { 0 } else { index }, entry);
        let mut previous = entry.value;
        let mut combined = false;
        // Every member of an opposite-ordered run must pool with its predecessor,
        // for any positive weights. Delay mean divisions until the run ends.
        while let Some(&(index, entry)) = input.peek() {
            if ordered::<INCREASING>(previous, entry.value) {
                break;
            }
            input.next();
            pool = pool.combined(Pool::single(index, entry));
            previous = entry.value;
            retained += 1;
            combined = true;
        }
        if combined {
            pool = pool.centered();
        }
        while count > 0 {
            let left = pools[count - 1];
            if ordered::<INCREASING>(left.mean, pool.mean) {
                break;
            }
            pool = left.combined(pool).centered();
            count -= 1;
        }
        pools[count] = pool;
        count += 1;
    }
    if !entries.is_empty() && count == 0 {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "nonempty lane has no positive retained mass",
        ));
    }

    let pools = &pools[..count];
    let mut mass = CompensatedSum::ZERO;
    let mut block_mass = CompensatedSum::ZERO;
    let mut block_count = 0;
    for (index, &pool) in pools.iter().enumerate() {
        pool.sums.check(pool.mean)?;
        let end = pools
            .get(index + 1)
            .map_or(entries.len(), |next| next.start);
        let value = S::try_from_f64_lossy(pool.mean)?;
        let value = if value == S::ZERO { S::ZERO } else { value };
        fitted[pool.start..end].fill(value);
        if block_count > 0 && blocks[block_count - 1].value == value {
            blocks[block_count - 1].end = end;
        } else {
            if block_count > 0 {
                blocks[block_count - 1].weight = project(block_mass.total()?)?;
            }
            blocks[block_count] = IsotonicBlock {
                start: pool.start,
                end,
                value,
                weight: Some(S::ZERO),
            };
            block_count += 1;
            block_mass = CompensatedSum::ZERO;
        }
        for part in pool.sums.mass() {
            mass.add(part)?;
            block_mass.add(part)?;
        }
    }
    if block_count > 0 {
        blocks[block_count - 1].weight = project(block_mass.total()?)?;
    }
    // Evaluate loss at the published scalar, including f32 rounding. Omitted
    // entries have sanitized zero mass/value, so they contribute exactly zero.
    let [loss] = compensated_reduce(fitted.iter().zip(entries).map(|(&fit, entry)| {
        let residual = entry.value - fit.into_f64();
        [entry.weight * residual * residual]
    }))?;
    Ok(IsotonicDiagnostics {
        block_count,
        retained,
        excluded: entries.len() - retained,
        weight: project(mass.total()?)?,
        squared_error: project(loss.total()?)?,
    })
}
