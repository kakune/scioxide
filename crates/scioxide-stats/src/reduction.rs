//! Policies for missing and non-finite values, merge order, weight models and trim limits.
use scioxide_core::{RealScalar, SciOxideError};
fn non_finite_error(subject: &'static str) -> SciOxideError {
    SciOxideError::non_finite(
        scioxide_core::NonFiniteOrigin::Input,
        alloc::format!("{subject} must be finite"),
    )
}
/// How to treat entries that a validity mask marks as missing.
///
/// See [Data, weights and missing values](crate#data-weights-and-missing-values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[must_use]
pub enum MissingPolicy {
    /// Returns an error at the first missing entry.
    #[default]
    Reject,
    /// Leaves missing entries out of the estimate.
    Skip,
}

/// How to treat NaN or infinite values in entries that are not marked missing.
///
/// See [Data, weights and missing values](crate#data-weights-and-missing-values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[must_use]
pub enum NonFinitePolicy {
    /// Returns an error at the first NaN or infinite value.
    #[default]
    Reject,
    /// Leaves NaN and infinite values out of the estimate.
    Skip,
}

/// Order in which batch moment reductions combine observations.
///
/// The two orders give the same estimates up to rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[must_use]
pub enum MergePolicy {
    /// Visits the observations of each lane once, in index order, on one thread.
    #[default]
    InputOrder,
    /// Splits each lane into the partitions of the execution plan, which may run
    /// in parallel, and merges the partition results in partition order. The
    /// result does not depend on thread scheduling.
    StablePartitions,
}

/// Name of a sampling weight model, as reported by weighted moments and mean intervals.
///
/// The weight model types [`crate::Frequency`], [`crate::Reliability`] and
/// [`crate::Analytic`] select the model. This enum only reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[must_use]
pub enum WeightSemantics {
    /// Relative importance, including normalized probabilities. The effective
    /// sample size is Kish's `W²/Q`, with `W = Σ wᵢ` and `Q = Σ wᵢ²`.
    Reliability,
    /// Integer replication counts.
    Frequency,
    /// Inverse variances up to a common factor, the estimated noise scale.
    Analytic,
}

/// Lower and upper value limits for trimmed statistics.
///
/// An observation `x` is kept if `lower ≤ x ≤ upper`. A missing bound does not
/// restrict. Each bound is inclusive by default and can be made exclusive with
/// [`Self::with_inclusive`]. Equal bounds with an exclusive end keep nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct TrimLimits<S: RealScalar> {
    lower: Option<S>,
    upper: Option<S>,
    include_lower: bool,
    include_upper: bool,
}
impl<S: RealScalar> TrimLimits<S> {
    /// Creates inclusive limits from optional bounds.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a bound is NaN or infinite.
    /// - `InvalidInput` if `lower > upper`.
    pub fn new(lower: Option<S>, upper: Option<S>) -> Result<Self, SciOxideError> {
        if lower.is_some_and(|v| !v.is_finite()) || upper.is_some_and(|v| !v.is_finite()) {
            return Err(non_finite_error("trim bound"));
        }
        if lower.zip(upper).is_some_and(|(l, u)| l > u) {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("lower"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "lower trim bound exceeds upper bound",
            ));
        }
        Ok(Self {
            lower,
            upper,
            include_lower: true,
            include_upper: true,
        })
    }
    /// Sets whether the lower and upper bounds are inclusive or exclusive.
    pub const fn with_inclusive(
        mut self,
        lower: crate::TrimInclusion,
        upper: crate::TrimInclusion,
    ) -> Self {
        self.include_lower = matches!(lower, crate::TrimInclusion::Inclusive);
        self.include_upper = matches!(upper, crate::TrimInclusion::Inclusive);
        self
    }
    /// Returns the lower bound, or `None` if there is none.
    pub const fn lower(self) -> Option<S> {
        self.lower
    }
    /// Returns the upper bound, or `None` if there is none.
    pub const fn upper(self) -> Option<S> {
        self.upper
    }
    /// Returns whether the lower and upper bounds are inclusive, in that order.
    pub const fn inclusive(self) -> (bool, bool) {
        (self.include_lower, self.include_upper)
    }
    pub(crate) fn contains(self, value: S) -> bool {
        self.lower
            .is_none_or(|l| value > l || (self.include_lower && value == l))
            && self
                .upper
                .is_none_or(|u| value < u || (self.include_upper && value == u))
    }
}
