//! Spherical spline fits that place their own knots to meet a residual target.

use scioxide_core::backend::{try_copy_slice, try_with_capacity};
use scioxide_core::{
    AffineValue, AlgorithmId, ExecutionContext, RealScalar, SciOxideError, SolveScope, Tolerance,
};

use super::basis::resolve;
use super::model::SphericalSpline;
use super::plan::{SphericalSamples, SphericalSplineFit, SphericalSplinePlan};
use crate::adaptive::{self, AdaptiveSplineOptions, Candidate, sample_dimension};
use crate::protocol::ExtrapolationPolicy;

/// Identity of the adaptive spherical fit in solve accounting and errors.
const ALGORITHM: AlgorithmId =
    scioxide_core::algorithm_id!("interp", "spherical-adaptive-spline", 1);

/// Fits the smoothest [`SphericalSpline`] whose residual at scattered samples on
/// the sphere meets a bound, and returns it.
///
/// `points` holds the sites `[θ, φ]` in radians, with colatitude `θ` in
/// `[0, π]` and longitude `φ` modulo `2π`, and the samples get unit weights.
/// With `s = residual`, `(atol, rtol)` from `tolerance` and
/// `τ = atol + rtol·s`, the fit inserts colatitude and longitude breakpoints
/// until the sum of squared residuals `R = Σᵢ |yᵢ − f(θᵢ, φᵢ)|²` satisfies
/// `R ≤ s + τ`, and then searches for the largest smoothing parameter `λ`
/// that keeps that bound, until `|R − s| ≤ τ`. `s = 0` asks for an
/// interpolating spline, `R ≤ atol`. The search is that of
/// [`fit_spherical_spline_scoped`] with
/// [`SplineFitTarget::Residual`](crate::SplineFitTarget::Residual) and the
/// default [`SplineFitLimits`](crate::SplineFitLimits): 128 knots per axis,
/// 64 breakpoint insertions and 32 penalty trials.
///
/// A search that stops before meeting its target is an error here;
/// [`fit_spherical_spline_with_options`] takes gridded samples, weights and
/// options, and returns the fit with its
/// [`AdaptiveSplineReport`](crate::AdaptiveSplineReport) whatever the status.
/// A tolerance near `ε` leaves the penalty search a window it may not find
/// within 32 trials; `rtol = 10⁻⁵` suits a smoothing bound `s > 0`.
///
/// # Errors
///
/// - The errors of [`fit_spherical_spline_scoped`] for an invalid `residual`
///   or invalid samples.
/// - `ConvergenceFailure` if the search stops before meeting its target:
///   `TruncationLimit` at the knot or insertion limit, `Stagnation` when no
///   span can be split, `IterationLimit` at the penalty-trial limit.
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_interp::{ExtrapolationPolicy, Interpolant, fit_spherical_spline};
/// use std::f64::consts::PI;
/// let points: Vec<[f64; 2]> = (0..=16)
///     .flat_map(|i| (0..16).map(move |j| [PI * f64::from(i) / 16.0, PI * f64::from(j) / 8.0]))
///     .collect();
/// let heights: Vec<f64> = points.iter().map(|p| p[0].cos()).collect();
/// let surface = fit_spherical_spline(&points, &heights, 1e-6, Tolerance::new(0.0, 1e-5)?)?;
/// let value = surface.evaluate([0.6, 1.0], ExtrapolationPolicy::Reject)?;
/// assert!((value - 0.6_f64.cos()).abs() < 3e-4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn fit_spherical_spline<S: RealScalar, V: AffineValue<Scale = S>>(
    points: &[[S; 2]],
    values: &[V],
    residual: S,
    tolerance: Tolerance<S>,
) -> Result<SphericalSpline<S, V::Owned>, SciOxideError> {
    let fit = fit_spherical_spline_with_options(
        SphericalSamples::Scattered(points),
        values,
        None,
        AdaptiveSplineOptions::residual_bound(residual, tolerance),
    )?;
    fit.report().require_success(ALGORITHM)?;
    Ok(fit.into_spline())
}

/// Fits a [`SphericalSpline`] to samples on the sphere as configured by `options`,
/// places the knots automatically, and returns the fit with its report.
///
/// This is [`fit_spherical_spline_scoped`] with
/// [`ExecutionContext::sequential_native`], on the calling thread; that
/// function describes the method. An unsuccessful status is reported in
/// [`SphericalSplineFit::report`], not as an error.
///
/// # Errors
///
/// - The errors of [`fit_spherical_spline_scoped`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_interp::{
///     AdaptiveSplineOptions, ExtrapolationPolicy, Interpolant, SphericalSamples, SplineFitTarget,
///     fit_spherical_spline_with_options,
/// };
/// use std::f64::consts::PI;
/// let samples: Vec<[f64; 2]> = (0..=16)
///     .flat_map(|i| (0..16).map(move |j| [PI * i as f64 / 16.0, PI * j as f64 / 8.0]))
///     .collect();
/// let values: Vec<f64> = samples.iter().map(|p| p[0].cos()).collect();
/// let options = AdaptiveSplineOptions::new()
///     .with_target(SplineFitTarget::Residual(1e-6))
///     .with_residual_tolerance(Tolerance::new(1e-16, 1e-5)?);
/// let fit = fit_spherical_spline_with_options(
///     SphericalSamples::Scattered(&samples),
///     &values,
///     None,
///     options,
/// )?;
/// assert!(fit.report().refinements() > 0 && fit.report().residual() <= 1.00001e-6);
/// let value = fit
///     .spline()
///     .evaluate([0.6, 1.0], ExtrapolationPolicy::Reject)?;
/// assert!((value - 0.6_f64.cos()).abs() < 3e-4);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn fit_spherical_spline_with_options<S: RealScalar, V: AffineValue<Scale = S>>(
    samples: SphericalSamples<'_, S>,
    values: &[V],
    weights: Option<&[S]>,
    options: AdaptiveSplineOptions<S>,
) -> Result<SphericalSplineFit<S, V::Owned>, SciOxideError> {
    fit_spherical_spline_scoped(
        samples,
        values,
        weights,
        options,
        ExecutionContext::sequential_native(),
    )
}

/// Fits a [`SphericalSpline`] to samples on the sphere within `scope`,
/// placing the knots automatically.
///
/// Each trial is a [`SphericalSplinePlan`] fit with the current breakpoints,
/// which minimizes the weighted squared residual plus `λ` times the
/// second-difference penalty described there. The first trial uses the polar
/// modes only, with colatitude breakpoints `[0, π]` and four longitude
/// intervals. The [`SplineFitTarget`](crate::SplineFitTarget) in `options`
/// selects the rest:
///
/// - `Penalty(λ)` fits once with the initial breakpoints; use
///   [`SphericalSplinePlan`] to choose your own.
/// - `Residual(s)` inserts breakpoints until the weighted residual is at most
///   `s`, then searches for the largest `λ` that keeps it there. Insertions
///   alternate between colatitude and longitude, starting with colatitude, and
///   split the span with the largest residual sum.
///
/// [`AdaptiveSplineReport::termination`](crate::AdaptiveSplineReport::termination)
/// tells whether the target was met. Each trial counts as one iteration toward
/// the budgets in `options`, and its factorization runs in a child scope.
///
/// # Errors
///
/// - `InvalidInput` if the target is negative or not finite, the knot limit is
///   below eight, a grid axis is not strictly increasing, a weight is negative
///   or no weight is positive.
/// - `DomainError` if a sample colatitude is outside `[0, π]`.
/// - `DimensionMismatch` if the values differ in shape, or the number of values
///   or weights differs from the number of samples.
/// - `NonFinite` if a sample, value or weight is NaN or infinite.
/// - `ResourceLimit` if a trial needs more memory than the limit in `options`.
/// - `NumericalInstability` (`IllConditioned`) if a design is numerically rank
///   deficient.
/// - `ConvergenceFailure` if the budget refuses the first trial, and
///   `ExecutionStopped` on cancellation or a deadline.
/// - `AllocationFailure` if storage cannot be reserved.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_interp::{
///     AdaptiveSplineOptions, AdaptiveSplineTermination, SphericalSamples, SplineFitTarget,
///     fit_spherical_spline_scoped, fit_spherical_spline_with_options,
/// };
/// use std::f64::consts::PI;
/// let samples: Vec<[f64; 2]> = (0..=8)
///     .flat_map(|i| (0..8).map(move |j| [PI * i as f64 / 8.0, PI * j as f64 / 4.0]))
///     .collect();
/// let values: Vec<[f64; 2]> = samples.iter().map(|p| [p[0].cos(), 1.0]).collect();
/// let options = AdaptiveSplineOptions::new().with_target(SplineFitTarget::Penalty(0.0));
/// let grid = SphericalSamples::Scattered(&samples);
/// let scoped = fit_spherical_spline_scoped(
///     grid,
///     &values,
///     None,
///     options,
///     ExecutionContext::sequential_native(),
/// )?;
/// let owned = fit_spherical_spline_with_options(grid, &values, None, options)?;
/// assert_eq!(
///     scoped.report().termination(),
///     AdaptiveSplineTermination::FixedPenalty
/// );
/// assert_eq!(
///     scoped.spline().coefficients(),
///     owned.spline().coefficients()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn fit_spherical_spline_scoped<'scope, 'capability, S, V>(
    samples: SphericalSamples<'_, S>,
    values: &[V],
    weights: Option<&[S]>,
    options: AdaptiveSplineOptions<S>,
    scope: impl Into<SolveScope<'scope, 'capability>>,
) -> Result<SphericalSplineFit<S, V::Owned>, SciOxideError>
where
    S: RealScalar,
    V: AffineValue<Scale = S>,
{
    options.validate()?;
    let count = samples.count()?;
    let dimension = sample_dimension(values, count)?;
    let limits = options.limits();
    limits.check_spherical::<S>(count, 6, dimension, 0)?;
    let points = samples.points()?;
    let weights = adaptive::weights(weights, count)?;
    let mut coordinates = [
        try_with_capacity(count, scioxide_core::quantity!("latitudes"))?,
        try_with_capacity(count, scioxide_core::quantity!("longitudes"))?,
    ];
    for &point in points.iter() {
        let point = resolve(
            point,
            ExtrapolationPolicy::Reject,
            scioxide_core::argument!("samples"),
        )?
        .effective();
        coordinates[0].push(point[0]);
        coordinates[1].push(point[1]);
    }
    let (pi, two) = (S::PI, S::TWO);
    let mut axes = [
        try_copy_slice(&[S::ZERO, pi], scioxide_core::quantity!("latitude_breaks"))?,
        try_copy_slice(
            &[S::ZERO, pi / two, pi, pi + pi / two, pi + pi],
            scioxide_core::quantity!("longitude_breaks"),
        )?,
    ];
    let result = adaptive::search(
        scope,
        ALGORITHM,
        ((count as u128) << 64) | dimension as u128,
        &mut axes,
        &options,
        |axes, lambda, context| {
            let plan = SphericalSplinePlan::build(
                &axes[0],
                &axes[1],
                SphericalSamples::Scattered(&points),
                Some(&weights),
                lambda,
                limits,
                &mut *context,
            )?;
            let (fit, workspace) = plan.fit_scoped(values, &mut *context)?;
            Ok(Candidate {
                model: fit.spline,
                residuals: try_copy_slice(
                    workspace.residuals(),
                    scioxide_core::quantity!("residuals"),
                )?,
                report: fit.report,
            })
        },
        |axes, residuals| {
            // Alternate axes; longitude refinement has no effect until an
            // interior colatitude lane exists, so colatitude is preferred at
            // equal refinement depth.
            let preferred = if axes[0].len() < 3 || axes[0].len() - 2 <= axes[1].len() - 5 {
                0
            } else {
                1
            };
            for axis in [preferred, 1 - preferred] {
                if axes[axis].len() + 6 >= limits.max_knots() {
                    continue;
                }
                if let Some(knot) = adaptive::insertion(&axes[axis], &coordinates[axis], residuals)?
                {
                    adaptive::insert(&mut axes[axis], knot)?;
                    return Ok(true);
                }
            }
            Ok(false)
        },
    )?;
    Ok(SphericalSplineFit {
        spline: result.model,
        report: result.report,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{AdaptiveSplineTermination, SplineFitTarget};
    use alloc::vec::Vec;
    use core::f64::consts::PI;
    use scioxide_core::{Budget, BudgetKind, SolveBudgets, Tolerance};

    fn samples() -> Vec<[f64; 2]> {
        (0..=16)
            .flat_map(|i| (0..16).map(move |j| [PI * i as f64 / 16., PI * j as f64 / 8.]))
            .collect()
    }

    #[test]
    fn residual_target_refines_and_budget_stops_after_a_fit() {
        let samples = samples();
        let values: Vec<f64> = samples.iter().map(|p| p[0].cos()).collect();
        let options = AdaptiveSplineOptions::new()
            .with_target(SplineFitTarget::Residual(1e-6))
            .with_residual_tolerance(Tolerance::new(1e-16, 1e-5).unwrap());
        let fit = fit_spherical_spline_with_options(
            SphericalSamples::Scattered(&samples),
            &values,
            None,
            options.with_limits(crate::SplineFitLimits::new().with_max_refinements(2)),
        )
        .unwrap();
        assert_eq!(
            fit.report().termination(),
            AdaptiveSplineTermination::RefinementLimit
        );
        assert_eq!(
            (fit.report().refinements(), fit.report().fit_trials()),
            (2, 3)
        );
        let limited = fit_spherical_spline_with_options(
            SphericalSamples::Scattered(&samples),
            &values,
            None,
            options.with_budgets(
                SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE),
            ),
        )
        .unwrap();
        assert_eq!(
            limited.report().termination(),
            AdaptiveSplineTermination::BudgetExhausted
        );
        assert_eq!(limited.report().fit_trials(), 1);
    }

    #[test]
    fn options_and_values_are_validated_first() {
        let samples = samples();
        let options = AdaptiveSplineOptions::<f64>::new();
        assert!(
            fit_spherical_spline_with_options(
                SphericalSamples::Scattered(&samples),
                &[1.0; 3],
                None,
                options
            )
            .is_err()
        );
        assert!(
            fit_spherical_spline_with_options(
                SphericalSamples::Scattered(&samples),
                &vec![1.0; samples.len()],
                None,
                options.with_target(SplineFitTarget::Residual(f64::NAN)),
            )
            .is_err()
        );
    }
}
