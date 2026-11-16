//! Regional rules of oscillatory plans and counted phase evaluation.
use super::{
    OscillatoryMethod, OscillatoryPhaseInverse, OscillatoryQuadraturePlan, levin::LevinWorkspace,
    number,
};
use crate::IntegrationTermination;
use crate::adaptive::regional::{OptionAllowance, Preparation, Prepared, Region, RegionalKernel};
use crate::kernel::{AffineMap1D, checked_callback_value, checked_intermediate, roundoff_floor};
use crate::weighted::solve::Sampler;
use crate::{FourierWeight, RegionalProgress};
use scioxide_core::RangeAccumulator;
use scioxide_core::backend::checked_mul;
use scioxide_core::{
    Complex, EvaluationRole, Interval, NumericScalar, RealScalar, SciOxideError, SolveContext,
};

/// Counts one provider call toward the evaluation limit.
pub(super) fn charge(
    context: &mut SolveContext<'_, '_>,
    role: EvaluationRole,
) -> Result<(), SciOxideError> {
    if !context.record_provider(role)?.is_granted() {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("context"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "admitted regional provider budget exhausted",
        ));
    }
    Ok(())
}
/// Borrowed phase function: one combined evaluation and an optional inverse.
#[derive(Clone, Copy)]
pub(super) struct Phase<'a, S: RealScalar> {
    pub(super) evaluate: &'a dyn Fn(S) -> Result<(S, S), SciOxideError>,
    pub(super) inverse: Option<&'a dyn OscillatoryPhaseInverse<S>>,
}
pub(super) struct Kernel<'a, 'b, S: RealScalar>
where
    Complex<S>: NumericScalar<Real = S>,
{
    pub(crate) plan: &'a OscillatoryQuadraturePlan<S>,
    pub(crate) phase: Phase<'a, S>,
    pub(crate) sampler: &'a mut Sampler<'b, S>,
    pub(crate) levin: Option<&'a mut LevinWorkspace<S>>,
    pub(crate) totals: &'a mut [[RangeAccumulator<S>; 4]],
}
impl<S: RealScalar> Kernel<'_, '_, S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    /// Evaluates `(g(x), g'(x))` as one counted provider call; the derivative
    /// must lie in the declared bounds of the root region.
    fn phase(
        &self,
        x: S,
        root: usize,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(S, S), SciOxideError> {
        charge(context, EvaluationRole::Function)?;
        let (value, derivative) = (self.phase.evaluate)(x)?;
        let value = checked_callback_value(value)?;
        let derivative = checked_callback_value(derivative)?;
        let declaration = self.plan.phase.regions[root];
        if derivative < declaration.derivative.lower()
            || derivative > declaration.derivative.upper()
            || (!declaration.stationary && derivative == S::ZERO)
        {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("phase"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "phase derivative violates the declared region",
            ));
        }
        Ok((value, derivative))
    }
    fn ordinary(
        &mut self,
        region: Region<S>,
        panels: usize,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        context: &mut SolveContext<'_, '_>,
        report: &mut RegionalProgress,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        self.totals
            .iter_mut()
            .for_each(|row| *row = core::array::from_fn(|_| RangeAccumulator::new()));
        // Fallback for weakly oscillatory or stationary regions: the oscillating factor is
        // multiplied into the amplitude and integrated with G7K15 on `panels` equal panels.
        // Panel sums accumulate in `totals` so the region reports one combined estimate.
        let rule = self.plan.kronrod.rule();
        let outer = AffineMap1D::from_interval(region.geometry)?.ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "empty regional interval",
            )
        })?;
        let d = estimates.len();
        for panel in 0..panels {
            let a = outer
                .point(-S::ONE + number::<S>(2)? * number::<S>(panel)? / number::<S>(panels)?)?;
            let b = outer.point(
                -S::ONE + number::<S>(2)? * number::<S>(panel + 1)? / number::<S>(panels)?,
            )?;
            let map = AffineMap1D::from_interval(Interval::new(a, b)?)?.ok_or_else(|| {
                crate::kernel::invalid(
                    scioxide_core::argument!("self"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "phase-resolution panel collapsed",
                )
            })?;
            let mut points = [S::ZERO; 15];
            let mut factors = [S::ONE; 15];
            rule.points(map, &mut points)?;
            if !distinct(&points, a, b, true) {
                return Ok(Some(IntegrationTermination::GeometryFailure));
            }
            let mut phase_error = S::ZERO;
            if self.plan.phase.omega != S::ZERO {
                for (i, &x) in points.iter().enumerate() {
                    // The derivative is not needed by the rule; it enforces the caller's
                    // declared derivative bounds that chose this regime.
                    let (value, _) = self.phase(x, region.root, context)?;
                    let phase = checked_intermediate(self.plan.phase.omega * value)?;
                    // A phase argument of size |phi| carries ~eps |phi| absolute error, which
                    // perturbs sin/cos by the same amount; for large omega this dominates the
                    // rule's own roundoff and is added to the arithmetic error below.
                    phase_error = phase_error.max(number::<S>(4)? * S::EPSILON * phase.abs());
                    factors[i] = project(self.plan.weight, phase.sin(), phase.cos());
                }
            }
            report.num_evals += (self.sampler)(&points, &mut samples[..15 * d], context)?;
            for (c, sums) in self.totals.iter_mut().enumerate() {
                let estimate = rule.apply_samples::<0>(map.half_weight()?, |i| {
                    checked_intermediate(samples[i * d + c] * factors[i])
                })?;
                let amplitude = (0..15).fold(S::ZERO, |m, i| m.max(samples[i * d + c].abs()));
                let arithmetic = checked_intermediate(
                    estimate.roundoff_floor() + (b - a) * amplitude * phase_error,
                )?;
                for (sum, value) in
                    sums.iter_mut()
                        .zip([estimate.value(), estimate.error(), arithmetic, S::ZERO])
                {
                    sum.add(value)?;
                }
            }
        }
        for (out, sums) in estimates.iter_mut().zip(self.totals.iter()) {
            for (value, sum) in out.iter_mut().zip(sums) {
                *value = sum.finish()?;
            }
        }
        Ok(None)
    }
    fn levin(
        &mut self,
        region: Region<S>,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        context: &mut SolveContext<'_, '_>,
        report: &mut RegionalProgress,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        // Collocation points are the 25 Chebyshev extrema cos(j pi / 24), in decreasing
        // order, so index 0 is the right endpoint and index 24 the left one.
        let map = AffineMap1D::from_interval(region.geometry)?.ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "empty Levin region",
            )
        })?;
        let mut points = [S::ZERO; 25];
        let mut derivatives = points;
        let mut phases = points;
        for (j, x) in points.iter_mut().enumerate() {
            *x = map.point(self.plan.cosines[1][j])?;
        }
        if !distinct(
            &points,
            region.geometry.lower(),
            region.geometry.upper(),
            false,
        ) {
            return Ok(Some(IntegrationTermination::GeometryFailure));
        }
        for (j, &x) in points.iter().enumerate() {
            let (value, derivative) = self.phase(x, region.root, context)?;
            phases[j] = checked_intermediate(self.plan.phase.omega * value)?;
            derivatives[j] = derivative;
        }
        let half = region.geometry.width()? / number::<S>(2)?;
        let workspace = self.levin.as_mut().ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "missing Levin workspace",
            )
        })?;
        let weights = workspace.weights(
            &self.plan.cosines,
            &derivatives,
            half,
            self.plan.phase.omega,
            [phases[24], phases[0]],
            context,
        )?;
        let d = estimates.len();
        report.num_evals += (self.sampler)(&points, &mut samples[..25 * d], context)?;
        let Some((high, low, solve_error)) = weights else {
            // A rank-deficient or ill-conditioned collocation has no estimate. The
            // sampled amplitude mass bounds the oscillatory integral heuristically,
            // so the scheduler bisects this region; children sweeping at most 64
            // radians use the ordinary rule.
            let width = region.geometry.width()?;
            for (c, out) in estimates.iter_mut().enumerate() {
                let amplitude = samples
                    .chunks_exact(d)
                    .take(25)
                    .fold(S::ZERO, |m, row| m.max(row[c].abs()));
                let mass = checked_intermediate(width * amplitude)?;
                *out = [S::ZERO, mass, roundoff_floor(mass)?, S::ZERO];
            }
            return Ok(None);
        };
        let phase_error = number::<S>(4)? * S::EPSILON * phases[0].abs().max(phases[24].abs());
        let width = region.geometry.width()?;
        for (c, out) in estimates.iter_mut().enumerate() {
            let mut fine = RangeAccumulator::new();
            let mut coarse = RangeAccumulator::new();
            let mut mass = RangeAccumulator::new();
            let mut top = [RangeAccumulator::new(), RangeAccumulator::new()];
            let mut amplitude = S::ZERO;
            for j in 0..25 {
                let sample = samples[j * d + c];
                amplitude = amplitude.max(sample.abs());
                fine.add_product(sample, project(self.plan.weight, high[j].im, high[j].re))?;
                coarse.add_product(sample, project(self.plan.weight, low[j].im, low[j].re))?;
                mass.add_product(sample.abs(), high[j].norm())?;
                for (sum, &row) in top.iter_mut().zip(&self.plan.top[j]) {
                    sum.add_product(sample, row)?;
                }
            }
            let value = fine.finish()?;
            let mass = mass.finish()?;
            // Error: a safety factor of 4 on the difference to the nested 13-point Levin
            // estimate, or the amplitude-resolution bound when larger. The nested
            // collocations share endpoint asymptotics once omega h is large, so only the
            // resolution bound sees interior jumps, kinks and cusps; |sin| and |cos| are
            // at most one, so the width bounds the weight's mass. Arithmetic: roundoff of
            // the weighted sum, plus the forward error of the collocation weights and of
            // the endpoint phases, both scaled by the L1 mass.
            let nested = checked_intermediate(number::<S>(4)? * (value - coarse.finish()?).abs())?;
            let resolution = crate::weighted::rule::resolution_bound(
                [top[0].finish()?, top[1].finish()?],
                amplitude,
                width,
            );
            *out = [
                value,
                nested.max(resolution),
                checked_intermediate(roundoff_floor(mass)? + mass * (solve_error + phase_error))?,
                S::ZERO,
            ];
        }
        Ok(None)
    }
    fn filon(
        &mut self,
        region: Region<S>,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        context: &mut SolveContext<'_, '_>,
        report: &mut RegionalProgress,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        // Filon in the phase variable: substituting phi = g(x) turns the integral into
        // int f(x(phi)) / |g'(x(phi))| e^{i omega phi} dphi over [g(a), g(b)]. The smooth
        // factor is interpolated at Chebyshev points in phi and integrated exactly against
        // analytic Fourier-Chebyshev moments, which is accurate for any omega. This
        // needs a monotone phase (checked from the endpoint values) and its inverse.
        let (lower, upper) = (region.geometry.lower(), region.geometry.upper());
        let start = self.phase(lower, region.root, context)?;
        let end = self.phase(upper, region.root, context)?;
        let increasing = self.plan.phase.regions[region.root].derivative.lower() > S::ZERO;
        if (increasing && end.0 <= start.0) || (!increasing && start.0 <= end.0) {
            return Err(crate::kernel::invalid(
                scioxide_core::argument!("phase"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "phase endpoints violate declared monotonicity",
            ));
        }
        let domain = Interval::new(start.0.min(end.0), start.0.max(end.0))?;
        let map = AffineMap1D::from_interval(domain)?.ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("phase"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "phase interval collapsed",
            )
        })?;
        let inverse = self.phase.inverse.ok_or_else(|| {
            crate::kernel::invalid(
                scioxide_core::argument!("phase"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Filon requires an inverse phase",
            )
        })?;
        // Node 0 is the largest phase and node 24 the smallest; mapping +-1 is exact, so
        // both reuse the endpoint evaluations instead of inverting the phase again.
        let (largest, smallest) = if increasing {
            ((upper, end), (lower, start))
        } else {
            ((lower, start), (upper, end))
        };
        let mut points = [S::ZERO; 25];
        let mut jacobians = points;
        let mut inverse_error = S::ZERO;
        for j in 0..25 {
            let target = map.point(self.plan.cosines[1][j])?;
            let (x, (actual, derivative)) = match j {
                0 => largest,
                24 => smallest,
                _ => {
                    charge(context, EvaluationRole::Function)?;
                    let x = checked_callback_value(inverse.invert(region.geometry, target)?)?;
                    if x < lower || x > upper {
                        return Err(crate::kernel::invalid(
                            scioxide_core::argument!("phase"),
                            scioxide_core::InvalidInputReason::OutOfRange,
                            "phase inverse is outside its branch interval",
                        ));
                    }
                    (x, self.phase(x, region.root, context)?)
                }
            };
            // The user's inverse is verified by re-evaluating the phase: a residual above
            // working precision means a wrong branch or an inaccurate inverse, which Filon
            // cannot tolerate. The accepted residual, times omega, is a phase error term.
            let discrepancy = (actual - target).abs();
            if discrepancy
                > number::<S>(64)? * S::EPSILON * S::ONE.max(actual.abs()).max(target.abs())
            {
                return Err(crate::kernel::invalid(
                    scioxide_core::argument!("phase"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "phase inverse residual exceeds working precision",
                ));
            }
            inverse_error = inverse_error.max(checked_intermediate(
                self.plan.phase.omega.abs() * discrepancy,
            )?);
            jacobians[j] = checked_intermediate(S::ONE / derivative.abs())?;
            points[j] = x;
        }
        if !distinct(&points, lower, upper, false) {
            return Ok(Some(IntegrationTermination::GeometryFailure));
        }
        let moments = crate::weighted::moments::fourier_on_interval(
            self.plan.phase.omega,
            self.plan.weight,
            domain,
        )?;
        // The phase-domain width bounds the mass of |cos| or |sin| for the
        // amplitude-resolution term, which sees features the nested 12/24 difference
        // misses once the phase parameter is large.
        let contraction = crate::weighted::rule::MomentRule::new(
            moments,
            &self.plan.cosines,
            None,
            domain.width()?,
            None,
        )?;
        let d = estimates.len();
        report.num_evals += (self.sampler)(&points, &mut samples[..25 * d], context)?;
        let phase_error = inverse_error;
        for (c, out) in estimates.iter_mut().enumerate() {
            let mut values = [S::ZERO; 25];
            for (j, value) in values.iter_mut().enumerate() {
                *value = checked_intermediate(samples[j * d + c] * jacobians[j])?;
            }
            *out = contraction.estimate(&values, &self.plan.top, &self.plan.cosines)?;
            // Filon's prepared moment uncertainty is arithmetic forward error
            // of its interpolation weights, reported with the other roundoff terms.
            out[2] = checked_intermediate(out[2] + out[3])?;
            out[3] = S::ZERO;
            let amplitude = samples
                .chunks_exact(d)
                .take(25)
                .fold(S::ZERO, |m, row| m.max(row[c].abs()));
            out[2] =
                checked_intermediate(out[2] + region.geometry.width()? * amplitude * phase_error)?;
        }
        Ok(None)
    }
}
impl<S: RealScalar> RegionalKernel<S> for Kernel<'_, '_, S>
where
    Complex<S>: NumericScalar<Real = S>,
{
    // Refinement switches between ordinary, Levin and Filon rules, and a failed
    // collocation reports its amplitude mass until its children are resolved, so
    // QUADPACK's fixed-rule slow-progress counters do not apply. Arithmetic-floor
    // and geometry stagnation still stop the solve.
    const SLOW_PROGRESS: bool = false;
    fn refine(
        &mut self,
        region: Region<S>,
        _slot: usize,
        _estimates: &[[S; 4]],
    ) -> Result<crate::adaptive::regional::Refinement<S, Self::Prepared>, SciOxideError> {
        crate::adaptive::regional::bisect(self, region)
    }
    fn midpoint(&self, region: Region<S>) -> Result<Option<S>, SciOxideError> {
        region.geometry.midpoint().map(Some)
    }
    type Prepared = Option<usize>;
    type Allowance = OptionAllowance;
    fn prepare(&mut self, region: Region<S>) -> Result<Preparation<Self::Prepared>, SciOxideError> {
        if let Some(panels) = self.plan.phase.ordinary_panels(
            region.geometry,
            region.root,
            self.plan.method == OscillatoryMethod::Levin,
            self.plan.options.ordinary_panels().get(),
        )? {
            if panels > self.plan.options.ordinary_panels().get() {
                return Ok(Preparation::Stop(
                    crate::IntegrationTermination::RefinementLimit,
                ));
            }
            // Provider work per sample: the amplitude, plus one combined phase call when
            // the oscillation is present. Levin: 25 amplitude + 25 phase + 2 factorizations
            // + 2 solves = 54. Filon: 25 amplitude + 25 phase (two at the endpoints) + 23
            // interior inverses = 73.
            let samples = checked_mul(panels, 15, scioxide_core::quantity!("panel_samples"))?;
            Ok(Preparation::Ready(Prepared {
                rule: Some(panels),
                evaluations: samples,
                providers: checked_mul(
                    samples,
                    if self.plan.phase.omega == S::ZERO {
                        1
                    } else {
                        2
                    },
                    scioxide_core::quantity!("region_providers"),
                )?,
            }))
        } else {
            Ok(Preparation::Ready(Prepared {
                rule: None,
                evaluations: 25,
                providers: match self.plan.method {
                    OscillatoryMethod::Levin => 54,
                    OscillatoryMethod::Filon => 73,
                },
            }))
        }
    }
    fn measure(
        &mut self,
        region: Region<S>,
        panels: &Self::Prepared,
        _slot: usize,
        samples: &mut [S],
        estimates: &mut [[S; 4]],
        context: &mut SolveContext<'_, '_>,
        allowance: &mut OptionAllowance,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        let report = &mut allowance.report;
        if let Some(panels) = panels {
            self.ordinary(region, *panels, samples, estimates, context, report)
        } else {
            match self.plan.method {
                OscillatoryMethod::Levin => self.levin(region, samples, estimates, context, report),
                OscillatoryMethod::Filon => self.filon(region, samples, estimates, context, report),
            }
        }
    }
}
fn project<S: RealScalar>(weight: FourierWeight, sine: S, cosine: S) -> S {
    match weight {
        FourierWeight::Sine => sine,
        FourierWeight::Cosine => cosine,
    }
}
fn distinct<S: RealScalar>(points: &[S], a: S, b: S, open: bool) -> bool {
    let Ok(interval) = Interval::new(a, b) else {
        return false;
    };
    if open {
        crate::kernel::mapped_geometry(points.iter().copied(), interval, false)
    } else {
        crate::kernel::mapped_geometry(points.iter().copied(), interval, true)
    }
}
