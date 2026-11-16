use super::*;
impl<S: RealScalar> LocalFilterPlan<S> {
    // Pre-optimization window definition, retained only for bitwise regression tests.
    pub(super) fn reference(&self, center: S, window: &mut [S]) -> Result<S, SciOxideError> {
        let scale = window.iter().fold(center.abs(), |m, x| m.max(x.abs()));
        if scale == S::ZERO {
            return Ok(center);
        }
        match self.kind {
            LocalFilter::Median | LocalFilter::Rank { .. } => {
                let rank = if let LocalFilter::Rank { rank } = self.kind {
                    rank
                } else {
                    window.len() / 2
                };
                let (_, value, _) = window.select_nth_unstable_by(rank, |a, b| {
                    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                });
                Ok(*value)
            }
            LocalFilter::Hampel { threshold } => {
                window.sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
                });
                let median = window[window.len() / 2];
                let mad = median_deviation(window)?;
                let factor = Scaled::from_value(S::try_from_f64_lossy(1.482602218505602)?)?;
                let distance = Scaled::from_value(center)?
                    .sub(Scaled::from_value(median)?)?
                    .abs();
                let limit = mad.mul(factor)?.mul(Scaled::from_value(threshold)?)?;
                let replace = distance.abs_cmp(limit) == core::cmp::Ordering::Greater;
                Ok(if replace { median } else { center })
            }
            LocalFilter::Wiener { noise_variance } => {
                if noise_variance == S::ZERO {
                    return Ok(center);
                }
                if window.iter().all(|x| *x == center) {
                    return Ok(center);
                }
                let (mean, variance) = local_moments(window.iter().copied(), window.len())?;
                let noise = Scaled::from_value(noise_variance)?;
                if variance.abs_le(noise) {
                    return mean.round();
                }
                let delta = mean.sub(Scaled::from_value(center)?)?;
                Scaled::from_value(center)?
                    .add(delta.mul(noise)?.div(variance)?)?
                    .round()
            }
        }
    }
}
