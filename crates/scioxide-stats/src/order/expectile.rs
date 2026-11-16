//! Expectiles by bisection over order statistics with exact balance signs.
//!
//! The balance `B(t) = (1 − p) Σ_{x ≤ t} w (x − t) + p Σ_{x > t} w (x − t)` is
//! continuous and strictly decreasing, so the expectile is its unique root.
//! Median pivots shrink the unresolved support. Compensated sums of the mass
//! and of the weighted values, shifted by the first pivot, summarize each
//! discarded side. The sign of `B` at a pivot comes from these sums when it
//! exceeds a rounding-error bound, and otherwise from an exact accumulation
//! over every entry, so the final bracket contains the root. On that bracket
//! `B` is linear, and its root is computed in closed form and clamped into
//! the bracket.
//!
//! Lanes whose values and weights have binary exponents within `±64` use
//! `f64` sums. Wider lanes use the same partitions with extended-range sums. The constant-size
//! equations at partition boundaries always use extended-range arithmetic.

use super::{Entry, ExpectileResult, QuantileProbability, finite_cmp};
use crate::arithmetic::{advanced_center_scaled, divided_center_scaled, promote, shift};
use core::cmp::Ordering;
use scioxide_core::backend::two_product;
use scioxide_core::{
    CompensatedSum, NumericalIssue, RangeAccumulator, RealScalar, Scaled, ScaledAccumulator,
    SciOxideError, compensated_reduce,
};

/// Total mass and origin-shifted weighted sum of a set of entries.
trait Summary: Copy {
    const ZERO: Self;
    fn of<S: RealScalar>(entries: &[Entry<S>], origin: f64) -> Result<Self, SciOxideError>;
    fn merged(self, other: Self) -> Result<Self, SciOxideError>;
    /// Both two-part sums, promoted exactly: `[mass, sum]`.
    fn extended(self) -> Result<[ScaledAccumulator<f64>; 2], SciOxideError>;
}

/// `f64` summaries for values and weights with binary exponents within `±64`,
/// where differences, weight products and their rounding errors stay normal.
#[derive(Clone, Copy)]
struct Native {
    mass: CompensatedSum<f64>,
    sum: CompensatedSum<f64>,
}

impl Summary for Native {
    const ZERO: Self = Self {
        mass: CompensatedSum::ZERO,
        sum: CompensatedSum::ZERO,
    };

    fn of<S: RealScalar>(entries: &[Entry<S>], origin: f64) -> Result<Self, SciOxideError> {
        let [mass, mut sum, low] = compensated_reduce(entries.iter().map(|entry| {
            let w = entry.weight.into_f64();
            let [high, residual] = shift(entry.value.into_f64(), origin);
            let (product, error) = two_product(w, high);
            [w, product, w * residual + error]
        }))?;
        sum.merge(low)?;
        Ok(Self { mass, sum })
    }

    fn merged(mut self, other: Self) -> Result<Self, SciOxideError> {
        self.mass.merge(other.mass)?;
        self.sum.merge(other.sum)?;
        Ok(self)
    }

    fn extended(self) -> Result<[ScaledAccumulator<f64>; 2], SciOxideError> {
        Ok([promote(self.mass.parts())?, promote(self.sum.parts())?])
    }
}

/// Extended-range summaries for lanes outside that range.
#[derive(Clone, Copy)]
struct Extended {
    mass: ScaledAccumulator<f64>,
    sum: ScaledAccumulator<f64>,
}

impl Summary for Extended {
    const ZERO: Self = Self {
        mass: ScaledAccumulator::ZERO,
        sum: ScaledAccumulator::ZERO,
    };

    fn of<S: RealScalar>(entries: &[Entry<S>], origin: f64) -> Result<Self, SciOxideError> {
        let origin = Scaled::from_value(origin)?;
        let mut summary = Self::ZERO;
        for entry in entries {
            let w = Scaled::from_value(entry.weight.into_f64())?;
            // A two-term compensated sum is an exact TwoSum contrast.
            let mut contrast = ScaledAccumulator::from_value(entry.value.into_f64())?;
            contrast.sub_scaled(origin)?;
            summary.mass.add_scaled(w)?;
            summary.sum = advanced_center_scaled(summary.sum, contrast, [w, Scaled::ZERO])?;
        }
        Ok(summary)
    }

    fn merged(mut self, other: Self) -> Result<Self, SciOxideError> {
        self.mass.merge(other.mass)?;
        self.sum.merge(other.sum)?;
        Ok(self)
    }

    fn extended(self) -> Result<[ScaledAccumulator<f64>; 2], SciOxideError> {
        Ok([self.mass, self.sum])
    }
}

fn breakdown() -> SciOxideError {
    SciOxideError::numerical_instability(
        NumericalIssue::Breakdown,
        "expectile left its support bracket",
    )
}

/// Returns `a + p (b − a)` for two-part `a` and `b`, keeping the product
/// rounding errors.
fn interpolated(
    a: ScaledAccumulator<f64>,
    b: ScaledAccumulator<f64>,
    p: Scaled<f64>,
) -> Result<ScaledAccumulator<f64>, SciOxideError> {
    let mut difference = b;
    for part in a.parts() {
        difference.sub_scaled(part)?;
    }
    advanced_center_scaled(a, difference, [p, Scaled::ZERO])
}

/// Returns the balance at a pivot at distance `d` (`contrast`) from the
/// origin: with `A = S_L − d M_L` and `C = S_R − d M_R`, `B = A + p (C − A)`.
/// The factor `1 − p` is never rounded.
fn balance<A: Summary>(
    left: A,
    right: A,
    contrast: ScaledAccumulator<f64>,
    p: Scaled<f64>,
) -> Result<Scaled<f64>, SciOxideError> {
    let side = |summary: A| -> Result<ScaledAccumulator<f64>, SciOxideError> {
        let [mass, sum] = summary.extended()?;
        let [high, low] = contrast.parts();
        advanced_center_scaled(sum, mass, [-high, -low])
    };
    interpolated(side(left)?, side(right)?, p)?.total()
}

/// Returns the exact sign of the balance at `pivot`, summed over every entry.
#[cold]
fn exact_balance<S: RealScalar>(
    entries: &[Entry<S>],
    pivot: S,
    p: f64,
) -> Result<Ordering, SciOxideError> {
    let mut balance = RangeAccumulator::<f64>::new();
    let v = pivot.into_f64();
    for entry in entries {
        let (x, w) = (entry.value.into_f64(), entry.weight.into_f64());
        if entry.value <= pivot {
            balance.add_product(w, x)?;
            balance.subtract_product(w, v)?;
            balance.subtract_triple_product(p, w, x)?;
            balance.add_triple_product(p, w, v)?;
        } else {
            balance.add_triple_product(p, w, x)?;
            balance.subtract_triple_product(p, w, v)?;
        }
    }
    Ok(balance.sign())
}

/// Computes the expectile of checked entries with finite values and positive
/// weights, for `0 < p < 1` and `low < high`. `native` selects the `f64`
/// summaries.
pub(super) fn expectile<S: RealScalar>(
    entries: &mut [Entry<S>],
    p: QuantileProbability,
    low: S,
    high: S,
    native: bool,
) -> Result<ExpectileResult<S>, SciOxideError> {
    let value = if native {
        root::<S, Native>(entries, p.get(), low, high)?
    } else {
        root::<S, Extended>(entries, p.get(), low, high)?
    };
    let residual = if native {
        native_residual(entries, value, p.get())?
    } else {
        extended_residual(entries, value, p.get())?
    };
    Ok(ExpectileResult {
        value,
        probability: p,
        relative_balance_residual: residual,
    })
}

fn root<S: RealScalar, A: Summary>(
    entries: &mut [Entry<S>],
    probability: f64,
    mut low: S,
    mut high: S,
) -> Result<S, SciOxideError> {
    let p = Scaled::from_value(probability)?;
    let n = entries.len();
    // Every contrast from an origin inside the support is at most the range.
    // Compensated reductions, merges and the constant-size equations each
    // lose at most a small multiple of (n + 64) u^2 of mass * range; 64
    // covers the multiple with margin.
    let range = Scaled::from_value(high.into_f64())?.sub(Scaled::from_value(low.into_f64())?)?;
    let gamma = Scaled::from_value((n as f64 + 64.0) * 64.0 * f64::EPSILON * f64::EPSILON)?;
    let mut certified = None;
    let mut origin = None;
    let mut left = A::ZERO;
    let mut right = A::ZERO;
    let (mut start, mut end) = (0, n);
    while start < end {
        let middle = start + (end - start) / 2;
        let (below, pivot, above) = entries[start..end]
            .select_nth_unstable_by(middle - start, |a, b| finite_cmp(&a.value, &b.value));
        let v = pivot.value;
        let o = *origin.get_or_insert(v.into_f64());
        let pivot_summary = A::of(core::slice::from_ref(pivot), o)?;
        let lower = left.merged(A::of(below, o)?)?;
        let upper = right.merged(A::of(above, o)?)?;
        let with_pivot = lower.merged(pivot_summary)?;
        // The first partition summarizes every entry.
        let bound = match certified {
            Some(bound) => bound,
            None => {
                let mass = with_pivot.merged(upper)?.extended()?[0].total()?;
                let bound = gamma.mul(mass)?.mul(range)?;
                certified = Some(bound);
                bound
            }
        };
        let mut contrast = ScaledAccumulator::from_value(v.into_f64())?;
        contrast.sub_scaled(Scaled::from_value(o)?)?;
        let estimate = balance(with_pivot, upper, contrast, p)?;
        let sign = if estimate.abs_cmp(bound) == Ordering::Greater {
            estimate.cmp_value(Scaled::ZERO)
        } else {
            exact_balance(entries, v, probability)?
        };
        match sign {
            Ordering::Equal => return Ok(v),
            Ordering::Less => {
                right = upper.merged(pivot_summary)?;
                high = v;
                end = middle;
            }
            Ordering::Greater => {
                left = with_pivot;
                low = v;
                start = middle + 1;
            }
        }
    }
    // Every entry now lies in `left` (at most `low`) or `right` (at least
    // `high`), and the root lies strictly between them.
    let origin = origin.ok_or_else(breakdown)?;
    let [left_mass, left_sum] = left.extended()?;
    let [right_mass, right_sum] = right.extended()?;
    let candidate: S = divided_center_scaled(
        ScaledAccumulator::from_value(origin)?,
        interpolated(left_sum, right_sum, p)?,
        interpolated(left_mass, right_mass, p)?,
    )?
    .total()?
    .round_to()?;
    let candidate = if candidate < low {
        low
    } else if candidate > high {
        high
    } else {
        candidate
    };
    Ok(if candidate == S::ZERO {
        S::ZERO
    } else {
        candidate
    })
}

/// Returns the relative balance residual at `value` with `f64` sums. Both
/// parts of `x − value` and the leading product errors are included.
fn native_residual<S: RealScalar>(
    entries: &[Entry<S>],
    value: S,
    p: f64,
) -> Result<f64, SciOxideError> {
    let [balance, balance_error, scale, scale_error] =
        compensated_reduce(entries.iter().map(|e| {
            let w = e.weight.into_f64();
            let [delta, delta_error] = shift(value.into_f64(), e.value.into_f64());
            let factor = if e.value > value { p } else { 1.0 - p };
            let (product, error) = two_product(delta, w);
            let error = error + delta_error * w;
            let (term, rounding) = two_product(product, factor);
            let error = rounding + error * factor;
            [
                term,
                error,
                term.abs(),
                if delta < 0.0 { -error } else { error },
            ]
        }))?;
    let balance = merged(balance, balance_error)?;
    let scale = merged(scale, scale_error)?;
    relative(balance.abs(), scale)
}

fn merged(mut sum: CompensatedSum<f64>, other: CompensatedSum<f64>) -> Result<f64, SciOxideError> {
    sum.merge(other)?;
    sum.total()
}

/// Returns the relative balance residual at `value` with extended-range sums.
fn extended_residual<S: RealScalar>(
    entries: &[Entry<S>],
    value: S,
    p: f64,
) -> Result<f64, SciOxideError> {
    let right = Scaled::from_value(p)?;
    let left = Scaled::ONE.sub(right)?;
    let center = Scaled::from_value(value.into_f64())?;
    let mut balance = ScaledAccumulator::ZERO;
    let mut scale = ScaledAccumulator::ZERO;
    for e in entries {
        let term = Scaled::from_value(e.value.into_f64())?
            .sub(center)?
            .mul(Scaled::from_value(e.weight.into_f64())?)?
            .mul(if e.value > value { right } else { left })?;
        balance.add_scaled(term)?;
        scale.add_scaled(term.abs())?;
    }
    let scale = scale.total()?;
    if scale.is_zero() {
        return Ok(0.0);
    }
    // A ratio below the binary64 range is an exact balance for reporting.
    Ok(balance
        .total()?
        .abs()
        .div(scale)?
        .try_value_to()
        .unwrap_or(0.0_f64)
        .min(1.0))
}

fn relative(balance: f64, scale: f64) -> Result<f64, SciOxideError> {
    let residual = if scale == 0.0 { 0.0 } else { balance / scale };
    if residual.is_finite() {
        // One-signed terms give a ratio of one up to their final rounding.
        Ok(residual.min(1.0))
    } else {
        Err(breakdown())
    }
}
