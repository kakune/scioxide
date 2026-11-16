//! Nested Boole-rule estimates and the nested grid of the Simpson rule.
use crate::geometry::IntervalGeometry;
use scioxide_core::{Interval, RealScalar, Scaled, ScaledAccumulator, SciOxideError};

pub(crate) struct SimpsonEstimate<S: RealScalar> {
    pub(crate) value: S,
    pub(crate) error: S,
    pub(crate) roundoff_floor: S,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct SimpsonKernel<S: RealScalar> {
    coarse_weights: [S; 9],
    fine_weights: [S; 9],
    ninety: S,
}

impl<S: RealScalar> SimpsonKernel<S> {
    pub(crate) fn prepare() -> Result<Self, SciOxideError> {
        let mut coarse_weights = [S::ZERO; 9];
        let mut fine_weights = [S::ZERO; 9];
        for (index, (coarse, fine)) in [7, 0, 32, 0, 12, 0, 32, 0, 7]
            .into_iter()
            .zip([7, 32, 12, 32, 14, 32, 12, 32, 7])
            .enumerate()
        {
            coarse_weights[index] = S::try_from_usize_exact(coarse)?;
            fine_weights[index] = S::try_from_usize_exact(fine)?;
        }
        Ok(Self {
            coarse_weights,
            fine_weights,
            ninety: S::try_from_usize_exact(90)?,
        })
    }

    /// Compares the nine-point composite Boole rule with the five-point Boole
    /// rule. Four times the full difference estimates the error of the finer
    /// rule without assuming smoothness at steps, kinks or endpoints. The
    /// estimate is heuristic: finite samples can miss narrow features.
    pub(crate) fn estimate(
        &self,
        geometry: &IntervalGeometry<S>,
        samples: [S; 9],
    ) -> Result<SimpsonEstimate<S>, SciOxideError> {
        let range = scioxide_core::backend::ShortReductionRange::new(samples.len());
        let safe = |x| range.admits(x);
        if let Some(span) = geometry.native_span().filter(|s| safe(*s))
            && samples.iter().all(|&x| safe(x))
        {
            use scioxide_core::CompensatedSum;
            let mut coarse = CompensatedSum::ZERO;
            let mut fine = CompensatedSum::ZERO;
            let mut absolute = CompensatedSum::ZERO;
            for (i, &sample) in samples.iter().enumerate() {
                coarse.add((self.coarse_weights[i] * S::TWO) * sample)?;
                fine.add(self.fine_weights[i] * sample)?;
                absolute.add(self.fine_weights[i] * sample.abs())?;
            }
            let factor = (span * S::HALF) / self.ninety;
            let error = scioxide_core::PromotingCompensatedSum::from_compensated(fine)
                .difference(scioxide_core::PromotingCompensatedSum::from_compensated(
                    coarse,
                ))?
                .abs();
            return Ok(SimpsonEstimate {
                value: factor * fine.total()?,
                error: (error * factor)
                    * S::try_from_usize_exact(crate::kernel::EXTRAPOLATION_ERROR_GUARD)?,
                roundoff_floor: crate::kernel::roundoff_floor(factor * absolute.total()?)?,
            });
        }
        let mut coarse_sum = ScaledAccumulator::ZERO;
        let mut fine_sum = ScaledAccumulator::ZERO;
        let mut absolute_sum = ScaledAccumulator::ZERO;
        for (index, &sample) in samples.iter().enumerate() {
            coarse_sum.add_product(
                Scaled::from_value(self.coarse_weights[index])?,
                Scaled::from_value(sample)?,
            )?;
            fine_sum.add_product(
                Scaled::from_value(self.fine_weights[index])?,
                Scaled::from_value(sample)?,
            )?;
            absolute_sum.add_product(
                Scaled::from_value(self.fine_weights[index])?,
                Scaled::from_value(sample.abs())?,
            )?;
        }
        let coarse = geometry
            .span()?
            .mul(coarse_sum.total()?)?
            .div(Scaled::from_value(self.ninety)?)?;
        let factor = geometry
            .half_span()?
            .div(Scaled::from_value(self.ninety)?)?;
        let fine = factor.mul(fine_sum.total()?)?;
        let resabs = factor.mul(absolute_sum.total()?)?;
        Ok(SimpsonEstimate {
            value: fine.round()?,
            // A non-smooth feature can move within the nested grid rather than follow
            // the smooth O(h^6) model. Keep four times the full extrapolant change.
            error: crate::kernel::diagnostic(
                fine.sub(coarse)?
                    .abs()
                    .mul(Scaled::from_value(S::try_from_usize_exact(
                        crate::kernel::EXTRAPOLATION_ERROR_GUARD,
                    )?)?)?
                    .round(),
            )?,
            roundoff_floor: crate::kernel::scaled_roundoff_floor(resabs)?,
        })
    }
}

/// Grid points built from repeated midpoints, so that a child grid repeats the
/// stored samples of its parent exactly.
pub(crate) fn simpson_grid<S: RealScalar>(
    interval: Interval<S>,
    divisions: usize,
) -> Result<[S; 17], SciOxideError> {
    // Points are produced by repeated exact midpoints (coarsest stride first) instead of
    // `lower + i * h`, so the grid of a child interval is bit-identical to the matching
    // half of its parent's grid. That identity is what makes reusing retained samples
    // correct. `divisions` is 8 or 16; unused trailing entries stay at `lower`.
    let mut points = [interval.lower(); 17];
    points[divisions] = interval.upper();
    let mut stride = divisions / 2;
    while stride > 0 {
        for index in (stride..divisions).step_by(2 * stride) {
            points[index] =
                Interval::new(points[index - stride], points[index + stride])?.midpoint()?;
        }
        stride /= 2;
    }
    Ok(points)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::{AdaptiveIntegrator, AdaptiveRule, SubdivisionOptions};
    use scioxide_core::{Budget, BudgetKind, ErrorTolerance, NormSelection};

    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    use super::*;
    use crate::IntegrationTermination;

    fn exact() -> SubdivisionOptions<f64> {
        SubdivisionOptions::default()
            .with_tolerance(ErrorTolerance::scalar(0.0, 0.0, NormSelection::Infinity).unwrap())
    }

    #[test]
    fn smooth_and_polynomial_integrals_converge() {
        let integrator =
            AdaptiveIntegrator::new(AdaptiveRule::Simpson, SubdivisionOptions::default()).unwrap();
        let (sine, _) = integrator
            .integrate(
                |x: f64| Ok(x.sin()),
                Interval::new(0.0, core::f64::consts::PI).unwrap(),
            )
            .unwrap();
        assert!((sine - 2.0).abs() < 1.0e-10);
        let (quadratic, _) = integrator
            .integrate(|x: f64| Ok(x * x), Interval::new(0.0, 1.0).unwrap())
            .unwrap();
        assert!((quadratic - 1.0 / 3.0).abs() < 1.0e-10);
    }

    #[test]
    fn f32_uses_the_same_scheduler() {
        let integrator =
            AdaptiveIntegrator::<f32>::new(AdaptiveRule::Simpson, SubdivisionOptions::default())
                .unwrap();
        let (value, _) = integrator
            .integrate(
                |x: f32| Ok(x.sin()),
                Interval::new(0.0_f32, core::f32::consts::PI).unwrap(),
            )
            .unwrap();
        assert!((value - 2.0).abs() < 1.0e-3);
    }

    #[test]
    fn evaluation_cap_returns_a_measured_full_domain_estimate() {
        let options = exact().with_max_evaluations(Some(Budget::try_from(18).unwrap()));
        let (_, report) = AdaptiveIntegrator::new(AdaptiveRule::Simpson, options)
            .unwrap()
            .integrate(|x: f64| Ok(x.exp()), Interval::new(0.0, 1.0).unwrap())
            .unwrap();
        // Ten root callbacks, nine nodes and the witness, then one split.
        assert_eq!(report.evaluations(), 18);
        assert_eq!(report.subdivisions(), 1);
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert!(report.error_estimate().unwrap().is_finite());
    }

    #[test]
    fn split_budget_is_reserved_atomically() {
        let calls = AtomicUsize::new(0);
        let function = |x: f64| {
            calls.fetch_add(1, Relaxed);
            Ok(x.powi(8))
        };
        let options = exact().with_max_evaluations(Some(Budget::try_from(15).unwrap()));
        let (_, report) = AdaptiveIntegrator::new(AdaptiveRule::Simpson, options)
            .unwrap()
            .integrate(&function, Interval::new(0.0, 1.0).unwrap())
            .unwrap();
        assert_eq!(report.evaluations(), 10);
        assert_eq!(report.evaluations(), calls.load(Relaxed));
        assert_eq!(
            report.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert!(report.error_estimate().unwrap() > 0.0);
    }
}
