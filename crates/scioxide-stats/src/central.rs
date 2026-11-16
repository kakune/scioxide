//! Central-moment states, their pairwise update formulas, and moment conventions.
//!
//! The pairwise update formulas follow Pébay et al. (2016),
//! doi:10.1007/s00180-015-0637-z. Moments are normalized by mass, so raw power
//! sums that could overflow are never formed.

use crate::arithmetic::{
    Arithmetic, Shares, advanced_center, advanced_center_scaled, center_difference,
    center_difference_sum, counted_center, native, native_center, native_delta, native_scaled,
    offset_parts, promote,
};
use crate::error_helpers::invalid;
use scioxide_core::{InvalidInputReason, RealScalar, SciOxideError};

use core::cmp::Ordering;
use scioxide_core::{Scaled, ScaledAccumulator};

/// Highest central moment that an accumulator tracks.
///
/// Higher orders cost more arithmetic per observation. Asking an accumulator
/// for a moment above its order returns a `NotImplemented` error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MomentOrder {
    /// Tracks the mean, the extrema and the second central moment.
    Second,
    /// Also tracks the third central moment.
    Third,
    /// Also tracks the fourth central moment.
    Fourth,
}

impl MomentOrder {
    /// Returns the highest tracked central-moment order: 2, 3 or 4.
    pub const fn maximum(self) -> usize {
        match self {
            Self::Second => 2,
            Self::Third => 3,
            Self::Fourth => 4,
        }
    }
}

/// Small-sample correction for skewness and kurtosis.
///
/// With central moments `mₖ = Σ wᵢ (xᵢ − x̄)ᵏ / Σ wᵢ`, the uncorrected
/// coefficients are `g₁ = m₃ / m₂^(3/2)` and `g₂ = m₄ / m₂² − 3`. The corrected
/// coefficients for `n` independent observations are
///
/// ```text
/// G₁ = g₁ √(n(n − 1)) / (n − 2),                       n ≥ 3,
/// G₂ = (n − 1) ((n + 1) g₂ + 6) / ((n − 2)(n − 3)),    n ≥ 4.
/// ```
///
/// # References
///
/// D. N. Joanes and C. A. Gill, "Comparing measures of sample skewness and
/// kurtosis", *The Statistician* 47(1), 183–189, 1998.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiasCorrection {
    /// Uses the population coefficients `g₁` and `g₂`.
    None,
    /// Uses the adjusted Fisher–Pearson coefficients `G₁` and `G₂`.
    Corrected,
}

/// Convention for reporting kurtosis, the fourth standardized moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KurtosisConvention {
    /// Excess kurtosis `m₄ / m₂² − 3`, which is 0 for a normal distribution.
    Fisher,
    /// Kurtosis `m₄ / m₂²`, which is 3 for a normal distribution.
    Pearson,
}

/// Central-moment state in native `f64` or extended-range arithmetic.
///
/// Data in the native range never converts powers to extended range during an
/// update. Promotion keeps both parts of the center.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Central<S: RealScalar> {
    Native(NativeCentral<S>),
    Extended(ExtendedCentral<S>),
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NativeCentral<S: RealScalar> {
    mean: scioxide_core::CompensatedSum<f64>,
    moments: [f64; 3],
    minimum: S,
    maximum: S,
}
impl<S: RealScalar> Central<S> {
    pub(crate) fn from_native_parts(
        mean: [f64; 2],
        moments: [f64; 3],
        minimum: S,
        maximum: S,
    ) -> Result<Self, SciOxideError> {
        let (high, low) = scioxide_core::backend::two_sum(mean[0], mean[1]);
        let mean = [high, low];
        if native(mean[0], 64)
            && native(mean[1], 256)
            && moments.into_iter().all(|v| native(v, 256))
        {
            return Ok(Self::Native(NativeCentral {
                mean: scioxide_core::CompensatedSum::from_parts(mean[0], mean[1])?,
                moments,
                minimum,
                maximum,
            }));
        }
        Ok(Self::Extended(ExtendedCentral {
            mean: promote(mean)?,
            m2: Scaled::from_value(moments[0])?,
            m3: Scaled::from_value(moments[1])?,
            m4: Scaled::from_value(moments[2])?,
            minimum,
            maximum,
        }))
    }
    pub(crate) fn native_second(self) -> Option<f64> {
        match self {
            Self::Native(state) => Some(state.moments[0]),
            Self::Extended(_) => None,
        }
    }
    pub(crate) fn native_center_parts(self) -> Option<[f64; 2]> {
        match self {
            Self::Native(state) => Some(state.mean.parts()),
            Self::Extended(_) => None,
        }
    }
    pub(crate) fn empty() -> Self {
        Self::Native(NativeCentral {
            mean: scioxide_core::CompensatedSum::ZERO,
            moments: [0.0; 3],
            minimum: S::ZERO,
            maximum: S::ZERO,
        })
    }
    pub(crate) fn singleton(value: S) -> Result<Self, SciOxideError> {
        if native_delta(S::ZERO, value).is_some() {
            Ok(Self::Native(NativeCentral {
                mean: scioxide_core::CompensatedSum::from_parts(value.into_f64(), 0.0)?,
                moments: [0.0; 3],
                minimum: value,
                maximum: value,
            }))
        } else {
            Ok(Self::Extended(ExtendedCentral::singleton(value)?))
        }
    }
    pub(crate) fn from_extended(value: ExtendedCentral<S>) -> Result<Self, SciOxideError> {
        let native = || -> Option<NativeCentral<S>> {
            let [hi, lo] = native_center(value.mean)?;
            Some(NativeCentral {
                mean: scioxide_core::CompensatedSum::from_parts(hi, lo).ok()?,
                moments: [
                    native_scaled(value.m2, 256)?,
                    native_scaled(value.m3, 256)?,
                    native_scaled(value.m4, 256)?,
                ],
                minimum: value.minimum,
                maximum: value.maximum,
            })
        };
        Ok(native().map_or(Self::Extended(value), Self::Native))
    }
    pub(crate) fn extended(self) -> Result<ExtendedCentral<S>, SciOxideError> {
        match self {
            Self::Extended(value) => Ok(value),
            Self::Native(value) => Ok(ExtendedCentral {
                mean: promote(value.mean.parts())?,
                m2: Scaled::from_value(value.moments[0])?,
                m3: Scaled::from_value(value.moments[1])?,
                m4: Scaled::from_value(value.moments[2])?,
                minimum: value.minimum,
                maximum: value.maximum,
            }),
        }
    }
    pub(crate) fn mean(self) -> Result<ScaledAccumulator<f64>, SciOxideError> {
        match self {
            Self::Extended(value) => Ok(value.mean),
            Self::Native(value) => promote(value.mean.parts()),
        }
    }
    pub(crate) fn m2(self) -> Result<Scaled<f64>, SciOxideError> {
        self.moment(2, MomentOrder::Second)
    }
    pub(crate) fn minimum(self) -> S {
        match self {
            Self::Native(v) => v.minimum,
            Self::Extended(v) => v.minimum,
        }
    }
    pub(crate) fn maximum(self) -> S {
        match self {
            Self::Native(v) => v.maximum,
            Self::Extended(v) => v.maximum,
        }
    }
    pub(crate) fn is_constant(self) -> bool {
        match self {
            Self::Native(v) => v.moments[0] == 0.0,
            Self::Extended(v) => v.m2.is_zero(),
        }
    }
    pub(crate) fn moment(
        self,
        order: usize,
        tracked: MomentOrder,
    ) -> Result<Scaled<f64>, SciOxideError> {
        require_order(order, tracked)?;
        match self {
            Self::Extended(v) => v.moment(order, tracked),
            Self::Native(v) => Scaled::from_value(match order {
                0 => 1.0,
                1 => 0.0,
                _ => v.moments[order - 2],
            }),
        }
    }
    pub(crate) fn skewness(
        self,
        count: usize,
        tracked: MomentOrder,
        correction: BiasCorrection,
    ) -> Result<Option<S>, SciOxideError> {
        self.extended()?.skewness(count, tracked, correction)
    }
    pub(crate) fn kurtosis(
        self,
        count: usize,
        tracked: MomentOrder,
        correction: BiasCorrection,
        convention: KurtosisConvention,
    ) -> Result<Option<S>, SciOxideError> {
        self.extended()?
            .kurtosis(count, tracked, correction, convention)
    }
    pub(crate) fn k_statistic(
        self,
        order: usize,
        count: usize,
        tracked: MomentOrder,
    ) -> Result<Option<Scaled<f64>>, SciOxideError> {
        self.extended()?.k_statistic(order, count, tracked)
    }
    pub(crate) fn k_statistic_variance(
        self,
        order: usize,
        count: usize,
        tracked: MomentOrder,
    ) -> Result<Option<Scaled<f64>>, SciOxideError> {
        self.extended()?.k_statistic_variance(order, count, tracked)
    }
    /// Adds one finite observation to a state of `count − 1` observations.
    ///
    /// It uses the native update when the value is in range, and otherwise
    /// merges a one-observation state with two-part count shares.
    ///
    /// # Errors
    ///
    /// Returns count, range or numerical errors of the extended merge.
    #[inline]
    pub(crate) fn observed(
        self,
        value: S,
        count: usize,
        order: MomentOrder,
    ) -> Result<Self, SciOxideError> {
        if let Some(state) = self.observed_native(value, count, order) {
            return Ok(state);
        }
        let shares = Shares::counts(count - 1, 1, count)?;
        self.merged(Self::singleton(value)?, shares, order)
    }
    #[inline(always)]
    pub(crate) fn observed_native(
        self,
        value: S,
        count: usize,
        order: MomentOrder,
    ) -> Option<Self> {
        if count as u128 > (1_u128 << 53) {
            return None;
        }
        let Self::Native(left) = self else {
            return None;
        };
        let value64 = native_delta(S::ZERO, value)?;
        let right = NativeCentral {
            mean: scioxide_core::CompensatedSum::from_parts(value64, 0.0).ok()?,
            moments: [0.0; 3],
            minimum: value,
            maximum: value,
        };
        left.merged(
            right,
            Advance::Observed {
                count: count as f64,
            },
            order,
        )
        .map(Self::Native)
    }
    /// Merges with shares of rounded native masses.
    #[inline]
    pub(crate) fn merged_native_fractions(
        self,
        right: Self,
        p: f64,
        q: f64,
        order: MomentOrder,
    ) -> Result<Self, SciOxideError> {
        if let (Self::Native(left), Self::Native(right)) = (self, right)
            && let Some(candidate) = left.merged(right, Advance::Rounded { p, q }, order)
        {
            return Ok(Self::Native(candidate));
        }
        Ok(Self::Extended(self.extended()?.merged(
            right.extended()?,
            Shares::rounded(Scaled::from_value(p)?, Scaled::from_value(q)?),
            order,
        )?))
    }
    #[inline]
    pub(crate) fn merged(
        self,
        right: Self,
        shares: Shares,
        order: MomentOrder,
    ) -> Result<Self, SciOxideError> {
        let native_share = |[high, low]: [Scaled<f64>; 2]| {
            Some([native_scaled(high, 256)?, native_scaled(low, 256)?])
        };
        if let (Self::Native(left), Self::Native(right)) = (self, right)
            && let (Some(p), Some(q)) = (native_share(shares.p), native_share(shares.q))
            && let Some(candidate) = left.merged(right, Advance::Shares { p, q }, order)
        {
            return Ok(Self::Native(candidate));
        }
        Ok(Self::Extended(self.extended()?.merged(
            right.extended()?,
            shares,
            order,
        )?))
    }
}
/// How a native merge moves the center by the difference `right − left`.
#[derive(Clone, Copy)]
enum Advance {
    /// Rounded shares of native masses. Rounding the shares already costs an
    /// ulp of the scaled difference, so one rounded product suffices.
    Rounded { p: f64, q: f64 },
    /// Two-part shares of exact counts. Uses both difference parts, both share
    /// parts and the rounding error of the leading product.
    Shares { p: [f64; 2], q: [f64; 2] },
    /// One observation joining `count − 1`. Divides the two-part difference by
    /// `count` and keeps the division residual.
    Observed { count: f64 },
}

impl<S: RealScalar> NativeCentral<S> {
    /// Merges two states pairwise.
    ///
    /// The update formulas use the rounded difference and shares. The center
    /// moves as `advance` selects.
    #[inline(always)]
    fn merged(self, right: Self, advance: Advance, order: MomentOrder) -> Option<Self> {
        let difference = offset_parts(self.mean.parts(), right.mean.parts())
            .ok()?
            .parts();
        let delta = native_delta(0.0, difference[0] + difference[1])?;
        let (p, q) = match advance {
            Advance::Rounded { p, q } => (p, q),
            Advance::Shares { p, q } => (p[0], q[0]),
            Advance::Observed { count } => {
                let q = 1.0 / count;
                (1.0 - q, q)
            }
        };
        let moments = merged_powers(self.moments, right.moments, delta, p, q, order).ok()?;
        // Guard the state consumed by the next polynomial, including cancellation.
        if moments.into_iter().any(|v| !native(v, 256)) || moments[0] < 0.0 || moments[2] < 0.0 {
            return None;
        }
        // Partition merges start at the smaller endpoint so tiny partition
        // masses retain their contribution even when the opposite share
        // rounds to one; an observation divides exactly instead.
        let left_base = || -> Option<bool> {
            Some(self.mean.total().ok()?.abs() <= right.mean.total().ok()?.abs())
        };
        let mean = match advance {
            Advance::Observed { count } => counted_center(self.mean, difference, count, q),
            Advance::Shares { p, q } => {
                if left_base()? {
                    advanced_center(self.mean, difference, q)
                } else {
                    advanced_center(right.mean, difference, [-p[0], -p[1]])
                }
            }
            Advance::Rounded { .. } => {
                let (mut mean, step) = if left_base()? {
                    (self.mean, delta * q)
                } else {
                    (right.mean, -delta * p)
                };
                mean.add(step).map(|()| mean)
            }
        }
        .ok()?;
        let [hi, lo] = mean.parts();
        if !native(hi, 64) || !native(lo, 256) {
            return None;
        }
        Some(Self {
            mean,
            moments,
            minimum: self.minimum.min(right.minimum),
            maximum: self.maximum.max(right.maximum),
        })
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ExtendedCentral<S: RealScalar> {
    pub mean: ScaledAccumulator<f64>,
    pub m2: Scaled<f64>,
    pub m3: Scaled<f64>,
    pub m4: Scaled<f64>,
    pub minimum: S,
    pub maximum: S,
}

impl<S: RealScalar> ExtendedCentral<S> {
    pub(crate) fn singleton(value: S) -> Result<Self, SciOxideError> {
        Ok(Self {
            mean: ScaledAccumulator::from_value(scioxide_core::RealScalar::into_f64(value))?,
            m2: Scaled::<f64>::ZERO,
            m3: Scaled::<f64>::ZERO,
            m4: Scaled::<f64>::ZERO,
            minimum: value,
            maximum: value,
        })
    }

    /// Merges two states pairwise in extended range.
    ///
    /// The center uses both difference parts, both share parts and the
    /// rounding errors of the leading products.
    #[inline(always)]
    pub(crate) fn merged(
        self,
        right: Self,
        shares: Shares,
        order: MomentOrder,
    ) -> Result<Self, SciOxideError> {
        let left_moments = [self.m2, self.m3, self.m4];
        let right_moments = [right.m2, right.m3, right.m4];
        let difference = center_difference_sum(self.mean, right.mean)?;
        let delta = difference.total()?;
        // Start at the smaller endpoint so tiny partition masses retain their
        // contribution even when the opposite share rounds to one.
        let mean = if self.mean.total()?.abs_cmp(right.mean.total()?) != Ordering::Greater {
            advanced_center_scaled(self.mean, difference, shares.q)?
        } else {
            advanced_center_scaled(right.mean, difference, shares.p.map(|part| -part))?
        };
        let [p, q] = [shares.p[0], shares.q[0]];
        let moments = merged_powers(left_moments, right_moments, delta, p, q, order)?;
        if [moments[0], moments[2]]
            .into_iter()
            .any(|m| m.is_sign_negative() && !m.is_zero())
        {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Cancellation,
                "even central moment became negative",
            ));
        }
        Ok(Self {
            mean,
            m2: moments[0],
            m3: moments[1],
            m4: moments[2],
            minimum: self.minimum.min(right.minimum),
            maximum: self.maximum.max(right.maximum),
        })
    }

    pub(crate) fn moment(
        self,
        order: usize,
        tracked: MomentOrder,
    ) -> Result<Scaled<f64>, SciOxideError> {
        require_order(order, tracked)?;
        Ok(match order {
            0 => Scaled::<f64>::ONE,
            1 => Scaled::<f64>::ZERO,
            2 => self.m2,
            3 => self.m3,
            _ => self.m4,
        })
    }

    pub(crate) fn skewness(
        self,
        count: usize,
        tracked: MomentOrder,
        correction: BiasCorrection,
    ) -> Result<Option<S>, SciOxideError> {
        require_order(3, tracked)?;
        if count == 0 || self.m2.is_zero() || (correction == BiasCorrection::Corrected && count < 3)
        {
            return Ok(None);
        }
        let value = self.m3.div(self.m2)?.div(self.m2.sqrt()?)?;
        let value = if correction == BiasCorrection::Corrected {
            value
                .mul(
                    Scaled::<f64>::from_value((count) as f64)?
                        .mul(Scaled::<f64>::from_value((count - 1) as f64)?)?
                        .sqrt()?,
                )?
                .div(Scaled::<f64>::from_value((count - 2) as f64)?)?
        } else {
            value
        };
        value.round_to().map(Some)
    }

    pub(crate) fn kurtosis(
        self,
        count: usize,
        tracked: MomentOrder,
        correction: BiasCorrection,
        convention: KurtosisConvention,
    ) -> Result<Option<S>, SciOxideError> {
        require_order(4, tracked)?;
        if count == 0 || self.m2.is_zero() || (correction == BiasCorrection::Corrected && count < 4)
        {
            return Ok(None);
        }
        let three = Scaled::<f64>::from_value(3.0)?;
        let excess = self.m4.div(self.m2.square()?)?.sub(three)?;
        let excess = if correction == BiasCorrection::Corrected {
            let n = Scaled::<f64>::from_value((count) as f64)?;
            excess
                .mul(n.add(Scaled::<f64>::ONE)?)?
                .add(Scaled::<f64>::from_value(6.0)?)?
                .mul(Scaled::<f64>::from_value((count - 1) as f64)?)?
                .div(
                    Scaled::<f64>::from_value((count - 2) as f64)?
                        .mul(Scaled::<f64>::from_value((count - 3) as f64)?)?,
                )?
        } else {
            excess
        };
        (if convention == KurtosisConvention::Pearson {
            excess.add(three)?
        } else {
            excess
        })
        .round_to()
        .map(Some)
    }

    pub(crate) fn k_statistic_variance(
        self,
        order: usize,
        count: usize,
        tracked: MomentOrder,
    ) -> Result<Option<Scaled<f64>>, SciOxideError> {
        if !(1..=2).contains(&order) {
            return Err(invalid(
                scioxide_core::argument!("order"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "k-statistic variance order must be 1 or 2",
            ));
        }
        let Some(k2) = self.k_statistic(2, count, tracked)? else {
            return Ok(None);
        };
        let n = Scaled::<f64>::from_value((count) as f64)?;
        if order == 1 {
            return k2.div(n).map(Some);
        }
        let Some(k4) = self.k_statistic(4, count, tracked)? else {
            return Ok(None);
        };
        k2.square()?
            .mul(n)?
            .mul(Scaled::<f64>::from_value(2.0)?)?
            .add(k4.mul(Scaled::<f64>::from_value((count - 1) as f64)?)?)?
            .div(n.mul(n.add(Scaled::<f64>::ONE)?)?)
            .map(Some)
    }

    pub(crate) fn k_statistic(
        self,
        order: usize,
        count: usize,
        tracked: MomentOrder,
    ) -> Result<Option<Scaled<f64>>, SciOxideError> {
        if !(1..=4).contains(&order) {
            return Err(invalid(
                scioxide_core::argument!("order"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "k-statistic order must be in 1..=4",
            ));
        }
        require_order(order, tracked)?;
        if count < order {
            return Ok(None);
        }
        let n = Scaled::<f64>::from_value((count) as f64)?;
        let result = match order {
            1 => self.mean.total()?,
            2 => self
                .m2
                .mul(n)?
                .div(Scaled::<f64>::from_value((count - 1) as f64)?)?,
            3 => self.m3.mul(n.square()?)?.div(
                Scaled::<f64>::from_value((count - 1) as f64)?
                    .mul(Scaled::<f64>::from_value((count - 2) as f64)?)?,
            )?,
            _ => self
                .m4
                .mul(n.add(Scaled::<f64>::ONE)?)?
                .sub(
                    self.m2
                        .square()?
                        .mul(Scaled::<f64>::from_value((count - 1) as f64)?)?
                        .mul(Scaled::<f64>::from_value(3.0)?)?,
                )?
                .mul(n.square()?)?
                .div(
                    Scaled::<f64>::from_value((count - 1) as f64)?
                        .mul(Scaled::<f64>::from_value((count - 2) as f64)?)?
                        .mul(Scaled::<f64>::from_value((count - 3) as f64)?)?,
                )?,
        };
        Ok(Some(result))
    }
}

pub(crate) fn require_order(requested: usize, tracked: MomentOrder) -> Result<(), SciOxideError> {
    if requested > tracked.maximum() {
        Err(SciOxideError::not_implemented(
            "requested moment was not accumulated",
        ))
    } else {
        Ok(())
    }
}

/// Rejects merging an incompatible `other` accumulator into this one.
pub(crate) fn compatible(compatible: bool) -> Result<(), SciOxideError> {
    if compatible {
        Ok(())
    } else {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("other"),
            InvalidInputReason::Inconsistent,
            "moment orders and weight semantics must agree",
        ))
    }
}

/// Merges normalized cross moments pairwise (Pébay et al., 2016).
///
/// It uses native arithmetic while every operand stays in the native range.
#[inline]
pub(crate) fn merge_cross(
    left: Scaled<f64>,
    right: Scaled<f64>,
    left_mean: (ScaledAccumulator<f64>, ScaledAccumulator<f64>),
    right_mean: (ScaledAccumulator<f64>, ScaledAccumulator<f64>),
    p: Scaled<f64>,
    q: Scaled<f64>,
) -> Result<Scaled<f64>, SciOxideError> {
    let dx = center_difference(left_mean.0, right_mean.0)?;
    let dy = center_difference(left_mean.1, right_mean.1)?;
    let native = || -> Option<Scaled<f64>> {
        let value = cross_polynomial(
            native_scaled(left, 256)?,
            native_scaled(right, 256)?,
            native_delta(0.0, native_scaled(dx, 64)?)?,
            native_delta(0.0, native_scaled(dy, 64)?)?,
            native_scaled(p, 256)?,
            native_scaled(q, 256)?,
        )
        .ok()?;
        Scaled::<f64>::from_value(value).ok()
    };
    if let Some(value) = native() {
        return Ok(value);
    }
    cross_polynomial(left, right, dx, dy, p, q)
}

#[inline(always)]
fn cross_polynomial<N: Arithmetic>(
    left: N,
    right: N,
    dx: N,
    dy: N,
    p: N,
    q: N,
) -> Result<N, SciOxideError> {
    left.mul(p)?
        .add(right.mul(q)?)?
        .add(dx.mul(dy)?.mul(p)?.mul(q)?)
}

#[inline(always)]
fn merged_powers<N: Arithmetic>(
    left: [N; 3],
    right: [N; 3],
    delta: N,
    p: N,
    q: N,
    order: MomentOrder,
) -> Result<[N; 3], SciOxideError> {
    let d2 = delta.square()?;
    let m2 = p.mul(left[0].add(q.mul(d2)?)?)?.add(q.mul(right[0])?)?;
    if order == MomentOrder::Second {
        return Ok([m2, N::number(0.0)?, N::number(0.0)?]);
    }
    let pq = p.mul(q)?;
    let m3 = p
        .mul(
            left[1].add(
                q.mul(delta)?.mul(
                    N::number(3.0)?
                        .mul(right[0].sub(left[0])?)?
                        .add(p.sub(q)?.mul(d2)?)?,
                )?,
            )?,
        )?
        .add(q.mul(right[1])?)?;
    if order == MomentOrder::Third {
        return Ok([m2, m3, N::number(0.0)?]);
    }
    let m4 = p
        .mul(
            left[2].add(
                q.mul(delta)?.mul(
                    N::number(4.0)?.mul(right[1].sub(left[1])?)?.add(
                        delta.mul(
                            N::number(6.0)?
                                .mul(q.mul(left[0])?.add(p.mul(right[0])?)?)?
                                .add(p.square()?.sub(pq)?.add(q.square()?)?.mul(d2)?)?,
                        )?,
                    )?,
                )?,
            )?,
        )?
        .add(q.mul(right[2])?)?;
    Ok([m2, m3, m4])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod native_tests {
    use super::*;
    #[test]
    fn native_updates_and_promotion_preserve_compensated_state() {
        for order in [MomentOrder::Second, MomentOrder::Third, MomentOrder::Fourth] {
            let mut native = Central::singleton(1.0_f64).unwrap();
            let mut extended = native.extended().unwrap();
            for (i, x) in [1.0 + f64::EPSILON, -3.0, 0.25, 8.0]
                .into_iter()
                .enumerate()
            {
                let n = i + 2;
                let shares = Shares::counts(n - 1, 1, n).unwrap();
                native = native.observed_native(x, n, order).unwrap();
                extended = extended
                    .merged(ExtendedCentral::singleton(x).unwrap(), shares, order)
                    .unwrap();
                for k in 2..=order.maximum() {
                    let a = native.moment(k, order).unwrap().round().unwrap();
                    let b = extended.moment(k, order).unwrap().round().unwrap();
                    assert!((a - b).abs() <= 64.0 * f64::EPSILON * b.abs().max(1.0));
                }
            }
            let shares = Shares::counts(5, 1, 6).unwrap();
            let promoted = native
                .merged(Central::singleton(1e200).unwrap(), shares, order)
                .unwrap();
            assert!(matches!(promoted, Central::Extended(_)));
            let expected = extended
                .merged(ExtendedCentral::singleton(1e200).unwrap(), shares, order)
                .unwrap();
            assert_eq!(promoted.mean().unwrap(), expected.mean);
            assert_eq!(promoted.m2().unwrap(), expected.m2);
        }
    }
}
