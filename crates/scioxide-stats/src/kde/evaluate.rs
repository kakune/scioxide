//! Density and log-density evaluation at single points and tiles of query points.
use super::RawGaussianKde;
use super::kernel::QuerySum;
use super::workspace::QUERY_TILE;
use crate::error_helpers::invalid;
use scioxide_core::Scaled;
use scioxide_core::logsumexp;
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

impl<S: RealScalar> RawGaussianKde<S> {
    /// Evaluates the density at `point`, using `scratch` for intermediate values.
    ///
    /// # Errors
    /// Fails on a dimension mismatch, a non-finite point, scratch that is too
    /// short, or a density that overflows `S`.
    pub(crate) fn pdf(&self, point: &[S], scratch: &mut [f64]) -> Result<S, SciOxideError> {
        self.evaluate(point, scratch, false)
    }
    /// Evaluates the log density at `point`. Returns `−∞` outside the support.
    ///
    /// # Errors
    /// Fails on an invalid point, scratch that is too short, or a log density
    /// that cannot be represented in `S`.
    pub(crate) fn log_pdf(&self, point: &[S], scratch: &mut [f64]) -> Result<S, SciOxideError> {
        self.evaluate(point, scratch, true)
    }
    fn evaluate(&self, point: &[S], scratch: &mut [f64], log: bool) -> Result<S, SciOxideError> {
        if point.len() != self.dimensions
            || scratch.len() < self.dimensions
            || point.iter().any(|x| !x.is_finite())
        {
            return Err(invalid(
                scioxide_core::argument!("point"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "KDE query/scratch mismatch or non-finite query",
            ));
        }
        if !self.contains(point) {
            return Ok(if log { S::NEG_INFINITY } else { S::ZERO });
        }
        let ordinary = if log {
            self.ordinary::<true>(point, scratch)
        } else {
            self.ordinary::<false>(point, scratch)
        };
        match ordinary {
            Some(value) => S::try_from_f64_lossy(value),
            None => self.evaluate_scaled(point, scratch, log, ExecutionContext::sequential()),
        }
    }
    fn contains(&self, point: &[S]) -> bool {
        !self
            .support
            .as_ref()
            .is_some_and(|bounds| bounds.iter().zip(point).any(|(b, x)| !b.contains(*x)))
    }
    fn ordinary<const LOG: bool>(&self, point: &[S], scratch: &mut [f64]) -> Option<f64> {
        let mut sum = QuerySum::zero();
        if let Some(interval) = self
            .ordinary_interval
            .as_ref()
            .filter(|r| r.contains(point[0].into_f64()))
        {
            let point = point[0].into_f64();
            for (center, weight) in self.centers.iter().zip(&self.weights) {
                let z = (point - center.into_f64()) * interval.inverse;
                sum.add_bounded::<LOG>((0.5 * z) * z, weight)?;
            }
            return sum.finish::<LOG>();
        }
        for (center, weight) in self
            .centers
            .chunks_exact(self.dimensions)
            .zip(&self.weights)
        {
            sum.add::<LOG>(self.kernel.half_distance(point, center, scratch)?, weight)?;
        }
        sum.finish::<LOG>()
    }
    fn evaluate_scaled(
        &self,
        point: &[S],
        scratch: &mut [f64],
        log: bool,
        context: ExecutionContext<'_>,
    ) -> Result<S, SciOxideError> {
        let mut value = f64::NEG_INFINITY;
        for (i, weight) in self.weights.iter().enumerate() {
            if i % 256 == 0 {
                context.check()?;
            }
            let k = self.kernel.log_value(
                |j| {
                    Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(point[j]))?.sub(
                        Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(
                            self.centers[i * self.dimensions + j],
                        ))?,
                    )
                },
                scratch,
            )?;
            value = logsumexp(&[value, weight.log_mass + k - weight.log_truncation])?;
        }
        if log {
            if !value.is_finite() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "log-density is outside the scalar's finite range",
                ));
            }
            S::try_from_f64_lossy(value)
        } else {
            S::try_from_f64_lossy(value.exp())
        }
    }
    pub(super) fn evaluate_tile<const LOG: bool>(
        &self,
        points: &[S],
        scratch: &mut [f64],
        context: ExecutionContext<'_>,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        let d = self.dimensions;
        let mut sums = [QuerySum::zero(); QUERY_TILE];
        let mut active = [false; QUERY_TILE];
        let mut fallback = [false; QUERY_TILE];
        for (point, active) in points.chunks_exact(d).zip(&mut active) {
            *active = self.contains(point);
        }
        if active.iter().any(|&value| value) {
            if let Some(interval) = self
                .ordinary_interval
                .as_ref()
                .filter(|r| points.iter().all(|point| r.contains(point.into_f64())))
            {
                'centers: for (i, (center, weight)) in
                    self.centers.iter().zip(&self.weights).enumerate()
                {
                    if i % 32 == 0 {
                        context.check()?;
                    }
                    let center = center.into_f64();
                    for (sum, point) in sums.iter_mut().zip(points) {
                        let z = (point.into_f64() - center) * interval.inverse;
                        if sum.add_bounded::<LOG>((0.5 * z) * z, weight).is_none() {
                            // A sum overflow can still occur after pair admission.
                            // Restart the tile without per-pair fallback bookkeeping.
                            fallback.fill(true);
                            break 'centers;
                        }
                    }
                }
            } else {
                for (i, (center, weight)) in
                    self.centers.chunks_exact(d).zip(&self.weights).enumerate()
                {
                    if i % 32 == 0 {
                        // Eight queries per tile: poll at most every 256 pairs.
                        context.check()?;
                    }
                    for (lane, (point, scratch)) in points
                        .chunks_exact(d)
                        .zip(scratch.chunks_exact_mut(d))
                        .enumerate()
                    {
                        if active[lane] && !fallback[lane] {
                            fallback[lane] = self
                                .kernel
                                .half_distance(point, center, scratch)
                                .and_then(|distance| sums[lane].add::<LOG>(distance, weight))
                                .is_none();
                        }
                    }
                }
            }
        }
        for (lane, ((point, scratch), out)) in points
            .chunks_exact(d)
            .zip(scratch.chunks_exact_mut(d))
            .zip(output)
            .enumerate()
        {
            *out = if !active[lane] {
                if LOG { S::NEG_INFINITY } else { S::ZERO }
            } else if let Some(value) = (!fallback[lane])
                .then(|| sums[lane].finish::<LOG>())
                .flatten()
            {
                S::try_from_f64_lossy(value)?
            } else {
                // Restart the whole query; a partial ordinary sum cannot retain
                // contributions that lost range before normalization.
                self.evaluate_scaled(point, scratch, LOG, context)?
            };
        }
        Ok(())
    }
}
