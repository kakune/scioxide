//! Group sizes, means and variances for tests on independent groups.
use crate::arithmetic::{
    advanced_center_scaled, center_difference, center_difference_sum, divided_center_scaled,
};
use crate::error_helpers::invalid;
use crate::{MomentOrder, OnlineMoments};
use alloc::vec::Vec;
use scioxide_core::{RealScalar, Scaled, ScaledAccumulator, SciOxideError};

#[derive(Clone, Copy)]
pub(crate) struct GroupSummary {
    pub n: f64,
    pub mean: ScaledAccumulator<f64>,
    pub variance: Scaled<f64>,
}

impl GroupSummary {
    pub(crate) fn from_sample<S: RealScalar>(sample: &[S]) -> Result<Self, SciOxideError> {
        if sample.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("samples"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty group",
            ));
        }
        let snapshot = OnlineMoments::from_slice(sample, MomentOrder::Second)?.snapshot();
        Ok(Self {
            n: sample.len() as f64,
            mean: snapshot.state.mean()?,
            variance: snapshot.variance_scaled(1)?.unwrap_or(Scaled::ZERO),
        })
    }

    pub(crate) fn from_scaled(sample: &[Scaled<f64>]) -> Result<Self, SciOxideError> {
        let second = crate::arithmetic::scaled_second(
            sample.len(),
            |i| crate::arithmetic::native_scaled(sample[i], 64),
            |i| Ok(sample[i]),
        )?;
        Self::from_second(sample.len(), second)
    }

    pub(crate) fn from_native(sample: &[f64]) -> Result<Self, SciOxideError> {
        let second = crate::arithmetic::scaled_second(
            sample.len(),
            |i| crate::arithmetic::native(sample[i], 64).then_some(sample[i]),
            |i| Scaled::from_value(sample[i]),
        )?;
        Self::from_second(sample.len(), second)
    }

    fn from_second(
        n: usize,
        (mean, population): (ScaledAccumulator<f64>, Scaled<f64>),
    ) -> Result<Self, SciOxideError> {
        let n = n as f64;
        Ok(Self {
            n,
            mean,
            variance: if n > 1.0 {
                population
                    .mul(Scaled::from_value(n)?)?
                    .div(Scaled::from_value(n - 1.0)?)?
            } else {
                Scaled::ZERO
            },
        })
    }

    pub(crate) fn precision(self) -> Result<Scaled<f64>, SciOxideError> {
        Scaled::from_value(self.n)?.div(self.variance)
    }
}

pub(crate) struct GroupSummaries {
    pub groups: Vec<GroupSummary>,
    pub total: usize,
    pub within: Scaled<f64>,
}

impl GroupSummaries {
    pub(crate) fn prepare<S: RealScalar>(
        samples: &[&[S]],
        max_bytes: usize,
    ) -> Result<Self, SciOxideError> {
        if samples.len() < 2 {
            return Err(invalid(
                scioxide_core::argument!("samples"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "at least two groups required",
            ));
        }
        let total = samples.iter().try_fold(0usize, |n, sample| {
            n.checked_add(sample.len()).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("group_observations"))
            })
        })?;
        Self::from_summaries(
            samples.len(),
            total,
            max_bytes,
            samples
                .iter()
                .map(|sample| GroupSummary::from_sample(sample)),
        )
    }

    pub(crate) fn from_summaries(
        count: usize,
        total: usize,
        max_bytes: usize,
        summaries: impl Iterator<Item = Result<GroupSummary, SciOxideError>>,
    ) -> Result<Self, SciOxideError> {
        if total as u128 > (1u128 << 53) {
            return Err(invalid(
                scioxide_core::argument!("samples"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "group count is not exactly representable",
            ));
        }
        crate::error_helpers::limit(
            scioxide_core::backend::checked_mul(
                count,
                core::mem::size_of::<GroupSummary>(),
                scioxide_core::quantity!("group_summary_bytes"),
            )?,
            max_bytes,
            scioxide_core::quantity!("group_summary_bytes"),
        )?;
        let mut groups = scioxide_core::backend::try_with_capacity(
            count,
            scioxide_core::quantity!("group_summaries"),
        )?;
        let mut within = ScaledAccumulator::ZERO;
        for summary in summaries {
            let group = summary?;
            within.add_scaled(group.variance.mul(Scaled::from_value(group.n - 1.0)?)?)?;
            groups.push(group);
        }
        Ok(Self {
            groups,
            total,
            within: within.total()?,
        })
    }

    pub(crate) fn pooled(&self) -> Result<(Scaled<f64>, f64), SciOxideError> {
        let df = self.total as f64 - self.groups.len() as f64;
        if df <= 0.0 || self.within.is_zero() {
            return Err(invalid(
                scioxide_core::argument!("groups"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "positive residual degrees and pooled variance required",
            ));
        }
        Ok((self.within.div(Scaled::from_value(df)?)?, df))
    }

    pub(crate) fn positive_variances(&self) -> Result<(), SciOxideError> {
        if self
            .groups
            .iter()
            .any(|g| g.n <= 1.0 || g.variance.is_zero())
        {
            return Err(invalid(
                scioxide_core::argument!("groups"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "procedure requires at least two nonconstant observations per group",
            ));
        }
        Ok(())
    }

    /// Weighted mean of the group means, centered on the mean of smallest
    /// magnitude, and the total weight.
    pub(crate) fn weighted_center(
        &self,
        weight: impl Fn(GroupSummary) -> Result<Scaled<f64>, SciOxideError>,
    ) -> Result<(ScaledAccumulator<f64>, Scaled<f64>), SciOxideError> {
        let origin = self
            .groups
            .iter()
            .try_fold(self.groups[0].mean, |origin, group| {
                Ok::<_, SciOxideError>(if group.mean.total()?.abs_le(origin.total()?) {
                    group.mean
                } else {
                    origin
                })
            })?;
        let mut mass = ScaledAccumulator::ZERO;
        let mut shifted = ScaledAccumulator::ZERO;
        for &group in &self.groups {
            let w = weight(group)?;
            mass.add_scaled(w)?;
            // Form the compensated contrast before weighting: products of
            // large translated centers can lose the entire small contrast.
            shifted = advanced_center_scaled(
                shifted,
                center_difference_sum(origin, group.mean)?,
                [w, Scaled::ZERO],
            )?;
        }
        Ok((divided_center_scaled(origin, shifted, mass)?, mass.total()?))
    }

    pub(crate) fn anova(&self) -> Result<(Scaled<f64>, f64), SciOxideError> {
        let (variance, df) = self.pooled()?;
        let (mean, _) = self.weighted_center(|g| Scaled::from_value(g.n))?;
        let mut between = ScaledAccumulator::ZERO;
        for group in &self.groups {
            between.add_scaled(
                center_difference(mean, group.mean)?
                    .square()?
                    .mul(Scaled::from_value(group.n)?)?,
            )?;
        }
        Ok((
            between
                .total()?
                .div(Scaled::from_value(self.groups.len() as f64 - 1.0)?)?
                .div(variance)?,
            df,
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn summary_budget_is_the_scaled_storage_width() {
        let samples = [&[1., 2., 3.][..], &[2., 4., 6.]];
        let bytes = 2 * core::mem::size_of::<GroupSummary>();
        assert!(GroupSummaries::prepare(&samples, bytes).is_ok());
        assert!(GroupSummaries::prepare(&samples, bytes - 1).is_err());
        assert!(
            crate::PreparedGroups::new(
                &samples,
                crate::StatisticsOptions::new().with_max_bytes(bytes)
            )
            .is_ok()
        );
        assert!(
            crate::PreparedGroups::new(
                &samples,
                crate::StatisticsOptions::new().with_max_bytes(bytes - 1)
            )
            .is_err()
        );
    }

    #[test]
    fn weighted_centers_keep_the_division_residual() {
        // Origin -1 (the last smallest magnitude), shifted sum 2 over mass 3:
        // dropping the residual of 2/3 lands one ULP away from -1/3.
        let summaries = GroupSummaries::prepare(&[&[1.0][..], &[-1.0, -1.0]], usize::MAX).unwrap();
        let (mean, mass) = summaries
            .weighted_center(|g| Scaled::from_value(g.n))
            .unwrap();
        assert_eq!(mean.round().unwrap(), -1.0 / 3.0);
        assert_eq!(mass.round().unwrap(), 3.0);
    }

    #[test]
    fn scaled_centers_keep_low_parts_through_cancellation() {
        let values = [
            Scaled::from_value(-f64::MAX).unwrap(),
            Scaled::ONE,
            Scaled::from_value(f64::MAX).unwrap(),
        ];
        let group = GroupSummary::from_scaled(&values).unwrap();
        assert!((group.mean.round().unwrap() - 1.0 / 3.0).abs() < 1e-15);
        let summaries =
            GroupSummaries::prepare(&[&[-f64::MAX][..], &[1.], &[f64::MAX]], usize::MAX).unwrap();
        let (mean, _) = summaries
            .weighted_center(|g| Scaled::from_value(g.n))
            .unwrap();
        assert!((mean.round().unwrap() - 1.0 / 3.0).abs() < 1e-15);
    }
}
