//! Smoothing and interpolating cubic spline curves through ordered points.

use crate::NodeSeparation;
use crate::validate::{validate_count, validate_length, validate_nodes};
use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, try_copy_slice, try_filled};
use scioxide_core::{
    AffineValue, AlgorithmId, ExecutionContext, RealScalar, ScaledNorm, SciOxideError, SolveScope,
    Tolerance,
};

use crate::adaptive::{
    self, AdaptiveSplineOptions, AdaptiveSplineReport, Candidate, sample_dimension,
};
use crate::checked::intermediate;
use crate::error::invalid;
use crate::spline::preparation::parametric_bytes;
use crate::{BSpline, KnotSide, PenalizedSplinePlan, SplinePenalty};

/// Identity of the adaptive parametric fit in solve accounting and errors.
const ALGORITHM: AlgorithmId = scioxide_core::algorithm_id!("interp", "parametric-spline", 1);

/// Selects the parameter values `t_i` of the samples of a parametric curve.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     AdaptiveSplineOptions, CurveParameterization, SplineFitTarget,
///     fit_parametric_spline_with_options,
/// };
/// let points: Vec<[f64; 2]> = (0..5).map(|i| [3.0 * i as f64, 4.0 * i as f64]).collect();
/// let options = AdaptiveSplineOptions::new().with_target(SplineFitTarget::Penalty(0.0));
/// let fit =
///     fit_parametric_spline_with_options(&points, CurveParameterization::ChordLength, None, options)?;
/// assert_eq!(fit.parameters(), [0.0, 0.25, 0.5, 0.75, 1.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub enum CurveParameterization<'a, S: RealScalar> {
    /// Uses these finite, strictly increasing parameters, in their own units.
    Supplied(&'a [S]),
    /// Uses the cumulative chord length scaled to `[0, 1]`:
    /// `t_0 = 0`, `t_i = t_{i−1} + ‖y_i − y_{i−1}‖`, divided by the total length.
    /// The norm is the Euclidean norm over all real value components, which is
    /// the Hermitian norm for complex values.
    ChordLength,
}

/// Holds the result of [`fit_parametric_spline_with_options`]: the curve, the sample
/// parameters it used and the fit report.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     AdaptiveSplineOptions, CurveParameterization, ExtrapolationPolicy, Interpolant,
///     SplineFitTarget, fit_parametric_spline_with_options,
/// };
/// let parameters = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
/// let points: Vec<[f64; 2]> = parameters.iter().map(|&t| [t, t * t]).collect();
/// let options = AdaptiveSplineOptions::new().with_target(SplineFitTarget::Penalty(0.0));
/// let fit = fit_parametric_spline_with_options(
///     &points,
///     CurveParameterization::Supplied(&parameters),
///     None,
///     options,
/// )?;
/// assert_eq!(fit.parameters(), parameters);
/// let [x, y] = fit.spline().evaluate([0.5], ExtrapolationPolicy::Reject)?;
/// assert!((x - 0.5).abs() < 1e-12 && (y - 0.25).abs() < 1e-12);
/// let curve = fit.into_spline();
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct ParametricSplineFit<S: RealScalar, V: AffineValue<Scale = S> = S> {
    spline: BSpline<S, V, 1>,
    parameters: Vec<S>,
    report: AdaptiveSplineReport<S>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> ParametricSplineFit<S, V> {
    /// Returns the fitted cubic curve; derivatives and integrals are with respect
    /// to the parameter.
    pub const fn spline(&self) -> &BSpline<S, V, 1> {
        &self.spline
    }

    /// Returns the sample parameters in input order.
    pub fn parameters(&self) -> &[S] {
        &self.parameters
    }

    /// Returns the fit report.
    pub const fn report(&self) -> &AdaptiveSplineReport<S> {
        &self.report
    }

    /// Returns the fitted curve, consuming the result.
    pub fn into_spline(self) -> BSpline<S, V, 1> {
        self.spline
    }
}

/// Fits the smoothest cubic spline curve `f(t)` whose residual at the ordered
/// samples `y_i` meets a bound, and returns it.
///
/// The samples get chord-length parameters on `[0, 1]`
/// ([`CurveParameterization::ChordLength`]) and unit weights. With
/// `s = residual`, `(atol, rtol)` from `tolerance` and `τ = atol + rtol·s`,
/// the fit inserts knots until the sum of squared residuals
/// `R = Σᵢ |yᵢ − f(tᵢ)|²` satisfies `R ≤ s + τ`, and then searches for the
/// largest smoothing parameter `λ` that keeps that bound, until
/// `|R − s| ≤ τ`. `s = 0` asks for an interpolating curve, `R ≤ atol`. The
/// search is that of [`fit_parametric_spline_scoped`] with
/// [`SplineFitTarget::Residual`](crate::SplineFitTarget::Residual) and the
/// default [`SplineFitLimits`](crate::SplineFitLimits): 128 knots, 64 knot
/// insertions and 32 penalty trials.
///
/// A search that stops before meeting its target is an error here;
/// [`fit_parametric_spline_with_options`] takes the parameters, weights and
/// options, and returns the fit with its
/// [`AdaptiveSplineReport`](crate::AdaptiveSplineReport) whatever the status.
/// A tolerance near `ε` leaves the penalty search a window it may not find
/// within 32 trials; `rtol = 10⁻⁵` suits a smoothing bound `s > 0`.
///
/// # Errors
///
/// - The errors of [`fit_parametric_spline_scoped`] for an invalid `residual`
///   or invalid samples.
/// - `ConvergenceFailure` if the search stops before meeting its target:
///   `TruncationLimit` at the knot or insertion limit, `Stagnation` when no
///   span can be split, `IterationLimit` at the penalty-trial limit.
///
/// # Examples
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_interp::{Univariate, fit_parametric_spline};
/// let points: Vec<[f64; 2]> = (0..41)
///     .map(|i| f64::from(i) / 40.0)
///     .map(|t| [t, (8.0 * t).sin()])
///     .collect();
/// let curve = fit_parametric_spline(&points, 1e-6, Tolerance::new(0.0, 1e-5)?)?;
/// let [x, y] = curve.value_at(0.0)?;
/// assert!(x.abs() < 1e-3 && y.abs() < 1e-3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. H. C. Eilers and B. D. Marx, "Flexible smoothing with B-splines and
/// penalties", *Statistical Science* 11(2), 1996, 89–121.
pub fn fit_parametric_spline<S: RealScalar, V: AffineValue<Scale = S>>(
    values: &[V],
    residual: S,
    tolerance: Tolerance<S>,
) -> Result<BSpline<S, V::Owned, 1>, SciOxideError> {
    let fit = fit_parametric_spline_with_options(
        values,
        CurveParameterization::ChordLength,
        None,
        AdaptiveSplineOptions::residual_bound(residual, tolerance),
    )?;
    fit.report().require_success(ALGORITHM)?;
    Ok(fit.into_spline())
}

/// Fits a cubic spline curve `f(t)` to ordered samples `y_i` as configured by
/// `options`, and returns the fit with its report.
///
/// This is [`fit_parametric_spline_scoped`] with
/// [`ExecutionContext::sequential_native`], on the calling thread; that
/// function describes the method. An unsuccessful status is reported in
/// [`ParametricSplineFit::report`], not as an error.
///
/// # Errors
///
/// - The errors of [`fit_parametric_spline_scoped`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::{
///     AdaptiveSplineOptions, AdaptiveSplineTermination, CurveParameterization, ExtrapolationPolicy,
///     Interpolant, SplineFitTarget, fit_parametric_spline_with_options,
/// };
/// let parameters = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
/// let values: Vec<_> = parameters.iter().map(|&t| Complex::new(t, t * t)).collect();
/// let options = AdaptiveSplineOptions::new().with_target(SplineFitTarget::Penalty(0.0));
/// let fit = fit_parametric_spline_with_options(
///     &values,
///     CurveParameterization::Supplied(&parameters),
///     None,
///     options,
/// )?;
/// assert_eq!(
///     fit.report().termination(),
///     AdaptiveSplineTermination::FixedPenalty
/// );
/// let point = fit.spline().evaluate([0.5], ExtrapolationPolicy::Reject)?;
/// assert!((point.im - 0.25).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn fit_parametric_spline_with_options<S: RealScalar, V: AffineValue<Scale = S>>(
    values: &[V],
    parameterization: CurveParameterization<'_, S>,
    weights: Option<&[S]>,
    options: AdaptiveSplineOptions<S>,
) -> Result<ParametricSplineFit<S, V::Owned>, SciOxideError> {
    fit_parametric_spline_scoped(
        values,
        parameterization,
        weights,
        options,
        ExecutionContext::sequential_native(),
    )
}

/// Fits a cubic spline curve `f(t)` to ordered samples `y_i` within `scope`.
///
/// The samples are values of any affine type, such as points `[f64; 2]` or
/// complex numbers, and get parameters `t_i` from `parameterization`. The
/// curve is a cubic B-spline on a clamped knot vector over `[t_0, t_{m−1}]`
/// that minimizes
///
/// ```text
/// Σ_i w_i |y_i − f(t_i)|² + λ ‖Δ²c‖²
/// ```
///
/// over its coefficients `c`, with weights `w_i` (one by default) and the
/// second-difference penalty of a [`PenalizedSplinePlan`]. The fit starts with
/// a single cubic piece. The [`SplineFitTarget`](crate::SplineFitTarget) in
/// `options` selects the rest:
///
/// - `Penalty(λ)` solves once on the single piece; use
///   [`PenalizedSplinePlan`] to fit with knots of your choice.
/// - `Residual(s)` inserts knots until the weighted residual is at most `s`,
///   then searches for the largest `λ` that keeps it there.
///
/// [`AdaptiveSplineReport::termination`] tells whether the target was met.
/// Each trial fit counts as one iteration toward the budgets in `options`,
/// and its banded Cholesky factorization runs in a child scope. For `m`
/// samples, `n` controls and `d` real components, a trial costs
/// `O(m (d + log n) + n d)` and stores `O(m + n d)` scalars. The memory limit
/// applies to this storage plus the best fit kept from earlier trials.
///
/// # Errors
///
/// - `InvalidInput` if the target is negative or not finite, the knot limit is
///   below eight, there are fewer than four samples, values have zero
///   components, supplied parameters are not strictly increasing, a weight is
///   negative or no weight is positive, or two adjacent points coincide under
///   [`CurveParameterization::ChordLength`].
/// - `DimensionMismatch` if the values differ in shape, or the number of
///   parameters or weights differs from the number of values.
/// - `NonFinite` if a value, weight or parameter is NaN or infinite.
/// - `ResourceLimit` if a trial needs more memory than the limit in `options`.
/// - `DomainError` if a normal matrix is not positive definite.
/// - `ConvergenceFailure` if the budget refuses the first trial, and
///   `ExecutionStopped` on cancellation or a deadline.
/// - `AllocationFailure` if storage cannot be reserved.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_interp::{
///     AdaptiveSplineOptions, CurveParameterization, SplineFitTarget, fit_parametric_spline_scoped,
///     fit_parametric_spline_with_options,
/// };
/// let points: Vec<[f64; 2]> = (0..6).map(|i| [i as f64, (i * i) as f64]).collect();
/// let options = AdaptiveSplineOptions::new().with_target(SplineFitTarget::Penalty(0.0));
/// let chord = CurveParameterization::ChordLength;
/// let scoped = fit_parametric_spline_scoped(
///     &points,
///     chord,
///     None,
///     options,
///     ExecutionContext::sequential_native(),
/// )?;
/// let owned = fit_parametric_spline_with_options(&points, chord, None, options)?;
/// assert_eq!(scoped.parameters(), owned.parameters());
/// assert_eq!(
///     scoped.spline().coefficients(),
///     owned.spline().coefficients()
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. H. C. Eilers and B. D. Marx, "Flexible smoothing with B-splines and
/// penalties", *Statistical Science* 11(2), 1996, 89–121.
pub fn fit_parametric_spline_scoped<'scope, 'capability, S, V>(
    values: &[V],
    parameterization: CurveParameterization<'_, S>,
    weights: Option<&[S]>,
    options: AdaptiveSplineOptions<S>,
    scope: impl Into<SolveScope<'scope, 'capability>>,
) -> Result<ParametricSplineFit<S, V::Owned>, SciOxideError>
where
    S: RealScalar,
    V: AffineValue<Scale = S>,
{
    options.validate()?;
    let samples = values.len();
    validate_count(samples, 4, scioxide_core::argument!("values"))?;
    let dimension = sample_dimension(values, samples)?;
    let limits = options.limits();
    limits.admit(parametric_bytes::<S>(samples, 4, dimension)?)?;
    let parameters = parameters(values, dimension, parameterization)?;
    let weights = adaptive::weights(weights, samples)?;
    let mut axes = [try_copy_slice(
        &[parameters[0], parameters[samples - 1]],
        scioxide_core::quantity!("breakpoints"),
    )?];
    let mut fitted = adaptive::search(
        scope,
        ALGORITHM,
        ((samples as u128) << 64) | dimension as u128,
        &mut axes,
        &options,
        |axes, lambda, context| {
            let controls = checked_add(axes[0].len(), 2, scioxide_core::quantity!("controls"))?;
            limits.admit(parametric_bytes::<S>(samples, controls, dimension)?)?;
            let basis =
                adaptive::open_cubic(&axes[0], scioxide_core::argument!("parameterization"))?;
            let plan = PenalizedSplinePlan::build(
                basis,
                try_copy_slice(&parameters, scioxide_core::quantity!("parameters"))?,
                Some(try_copy_slice(
                    &weights,
                    scioxide_core::quantity!("weights"),
                )?),
                SplinePenalty::new(lambda, 2)?,
                &mut *context,
            )?;
            let spline = plan.fit_scoped(values, &mut *context)?.into_spline();
            let residuals = residuals(&spline, values, &parameters, &weights, dimension)?;
            let residual = residuals
                .iter()
                .try_fold(S::ZERO, |sum, &r| intermediate(sum + r))?;
            Ok(Candidate {
                model: spline,
                residuals,
                report: AdaptiveSplineReport::fixed(residual, lambda, plan.diagonal_ratio()),
            })
        },
        |axes, residuals| {
            if axes[0].len() + 6 >= limits.max_knots() || axes[0].len() + 2 >= samples {
                return Ok(false);
            }
            if let Some(knot) = adaptive::insertion(&axes[0], &parameters, residuals)? {
                adaptive::insert(&mut axes[0], knot)?;
                Ok(true)
            } else {
                Ok(false)
            }
        },
    )?;
    fitted.report = fitted.report.with_geometry(
        [fitted.model.bases()[0].knots().len(), 0],
        [2, 0],
        None,
        [parameters[0], S::ZERO],
        [
            intermediate(parameters[samples - 1] - parameters[0])?,
            S::ONE,
        ],
    );
    Ok(ParametricSplineFit {
        spline: fitted.model,
        parameters,
        report: fitted.report,
    })
}

/// Supplied parameters, validated, or normalized cumulative chord lengths.
fn parameters<S: RealScalar, V: AffineValue<Scale = S>>(
    values: &[V],
    dimension: usize,
    parameterization: CurveParameterization<'_, S>,
) -> Result<Vec<S>, SciOxideError> {
    let samples = values.len();
    if let CurveParameterization::Supplied(p) = parameterization {
        validate_length(samples, p.len())?;
        validate_nodes(
            p,
            1,
            NodeSeparation::Exact,
            scioxide_core::argument!("parameterization"),
        )?;
        return try_copy_slice(p, scioxide_core::quantity!("parameters"));
    }
    let mut p = try_filled(samples, S::ZERO, scioxide_core::quantity!("parameters"))?;
    for (i, pair) in values.windows(2).enumerate() {
        let mut norm = ScaledNorm::<S>::new();
        for component in 0..dimension {
            norm.add_magnitude(
                (pair[1].component(component)? - pair[0].component(component)?).abs(),
            )?;
        }
        let distance = norm.finish()?;
        if distance == S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("values"),
                "chord-length parameters reject coincident adjacent points",
            ));
        }
        p[i + 1] = intermediate(p[i] + distance)?;
        if p[i + 1] <= p[i] {
            return Err(invalid(
                scioxide_core::argument!("values"),
                "chord-length increments are not representable",
            ));
        }
    }
    let length = p[samples - 1];
    for x in &mut p {
        *x /= length;
    }
    if p.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid(
            scioxide_core::argument!("values"),
            "normalized chord parameters lost ordering",
        ));
    }
    Ok(p)
}

/// Weighted squared residual of each sample, summed over value components.
///
/// Predictions use the local basis row exactly as curve evaluation does.
fn residuals<S: RealScalar, V: AffineValue<Scale = S>, W: AffineValue<Scale = S>>(
    spline: &BSpline<S, W, 1>,
    values: &[V],
    parameters: &[S],
    weights: &[S],
    dimension: usize,
) -> Result<Vec<S>, SciOxideError> {
    let basis = &spline.bases()[0];
    let coefficients = spline.coefficients();
    let mut residuals = try_filled(values.len(), S::ZERO, scioxide_core::quantity!("residuals"))?;
    let mut row = [S::ZERO; 4];
    for (((slot, value), &t), &weight) in residuals
        .iter_mut()
        .zip(values)
        .zip(parameters)
        .zip(weights)
    {
        let first = basis.local(t, 0, KnotSide::Right, &mut row)?;
        let scale = weight.sqrt();
        let mut residual = S::ZERO;
        for component in 0..dimension {
            let mut prediction = S::ZERO;
            for (offset, &b) in row.iter().enumerate() {
                if b != S::ZERO {
                    prediction += b * coefficients[(first + offset) * dimension + component];
                }
            }
            let d = (crate::checked::output(prediction)? - value.component(component)?).abs();
            residual = intermediate(residual + (scale * d) * (scale * d))?;
        }
        *slot = residual;
    }
    Ok(residuals)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{AdaptiveSplineTermination, SplineFitLimits, SplineFitTarget};
    use scioxide_core::{SciOxideErrorKind, Tolerance};

    #[test]
    fn chord_lengths_and_supplied_parameters() {
        let points: Vec<[f64; 2]> = (0..5).map(|i| [3.0 * f64::from(i), 0.0]).collect();
        let p = parameters(&points, 2, CurveParameterization::ChordLength).unwrap();
        assert_eq!(p, [0.0, 0.25, 0.5, 0.75, 1.0]);
        assert!(parameters(&[[0.0, 0.0]; 2], 2, CurveParameterization::ChordLength).is_err());
        assert!(parameters(&points, 2, CurveParameterization::Supplied(&[0.0, 1.0])).is_err());
    }

    #[test]
    fn penalty_search_limit_is_a_termination() {
        let p: Vec<f64> = (0..41).map(|i| f64::from(i) / 40.).collect();
        let values: Vec<[f64; 2]> = p.iter().map(|&x| [x, (8. * x).sin()]).collect();
        let options = AdaptiveSplineOptions::new()
            .with_target(SplineFitTarget::Residual(1e-6))
            .with_residual_tolerance(Tolerance::new(1e-16, 1e-5).unwrap())
            .with_limits(SplineFitLimits::new().with_max_penalty_iterations(0));
        let fit = fit_parametric_spline_with_options(
            &values,
            CurveParameterization::Supplied(&p),
            None,
            options,
        )
        .unwrap();
        assert_eq!(
            fit.report().termination(),
            AdaptiveSplineTermination::PenaltySearchLimit
        );
        assert_eq!(fit.report().penalty(), 0.0);
        assert!(fit.report().residual() <= 1.00001e-6);
    }

    #[test]
    fn byte_limit_is_a_resource_limit() {
        let options = AdaptiveSplineOptions::<f64>::new()
            .with_limits(SplineFitLimits::new().with_max_workspace_bytes(1));
        let error = fit_parametric_spline_with_options(
            &[0.0, 1.0, 2.0, 3.0],
            CurveParameterization::Supplied(&[0.0, 0.25, 0.5, 1.0]),
            None,
            options,
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    }

    #[test]
    fn banded_trial_limit_admits_large_sample_sets_and_checks_the_boundary() {
        let p: Vec<_> = (0..1024).map(|i| f64::from(i) / 1023.).collect();
        let values: Vec<_> = p.iter().map(|&x| [x, x * x]).collect();
        let required = parametric_bytes::<f64>(p.len(), 4, 2).unwrap();
        assert!(required < 128 * 1024);
        let options = AdaptiveSplineOptions::new()
            .with_target(SplineFitTarget::Penalty(0.))
            .with_limits(SplineFitLimits::new().with_max_workspace_bytes(required));
        let fit = fit_parametric_spline_with_options(
            &values,
            CurveParameterization::Supplied(&p),
            None,
            options,
        )
        .unwrap();
        assert!(fit.report().residual() < 1e-23);
        let error = fit_parametric_spline_with_options(
            &values,
            CurveParameterization::Supplied(&p),
            None,
            options.with_limits(SplineFitLimits::new().with_max_workspace_bytes(required - 1)),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
        assert_eq!(
            parametric_bytes::<f64>(usize::MAX, 4, 2)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
        assert_eq!(
            parametric_bytes::<f64>(4, 4, usize::MAX)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
    }
}
