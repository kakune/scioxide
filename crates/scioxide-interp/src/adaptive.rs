//! Targets, limits, reports and the knot refinement and penalty search of the
//! spherical and parametric spline fits.

use alloc::vec::Vec;

use scioxide_core::backend::{checked_add, checked_mul, try_copy_slice, try_filled};
use scioxide_core::{
    AffineValue, AlgorithmId, Argument, ConvergenceReason, ProblemToken, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, SolveIdentity, SolveScope, SolveState, Tolerance,
};

use crate::BSplineBasis;
use crate::checked::intermediate;
use crate::error::{admit_iteration, invalid, iteration_denied, unconverged};
use crate::validate::{Weights, validate_count, validate_weights};

/// The budget refused the first trial, so no model exists.
const FIRST_TRIAL: &str = "iteration budget denied the first spline fit trial";

/// Bytes counted for the dense QR/LQ kernels of a spherical factorization:
/// fixed 32×32 tiles, exponent-split panels and accumulator bins, which do not
/// scale with the problem. 32 MiB bounds them for every scalar width and
/// packing layout.
pub(crate) const SPHERE_KERNEL_BYTES: usize = 32 * 1024 * 1024;

/// Selects what an adaptive spline fit aims for: a given smoothing parameter
/// or a bound on the residual.
///
/// The fits minimize `R + λ P`, where `R` is the weighted sum of squared
/// residuals and `P` is the roughness penalty of the fit. With a residual
/// bound, the fit places knots and chooses `λ`; with a penalty, it uses the
/// initial knots and the given `λ`.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{AdaptiveSplineOptions, SplineFitTarget};
/// let options = AdaptiveSplineOptions::<f64>::new().with_target(SplineFitTarget::Residual(1e-6));
/// assert_eq!(options.target(), SplineFitTarget::Residual(1e-6));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplineFitTarget<S: RealScalar> {
    /// Minimizes `R + λ P` on the initial knots with this `λ ≥ 0`; no knots are
    /// inserted.
    Penalty(S),
    /// Inserts knots until `R` is at most this bound `s ≥ 0` plus the residual
    /// tolerance, then searches for the largest `λ` that still meets it.
    ///
    /// Each insertion splits the span with the largest residual sum at the
    /// midpoint of its interior samples. The penalty search starts at `λ = 1`,
    /// multiplies `λ` by 16 until the bound fails, and then bisects.
    Residual(S),
}

/// Sets limits on one spline fit, counting every trial of an adaptive search.
///
/// The defaults are 128 knots per axis, 64 knot insertions, 32 penalty trials
/// and 64 MiB of working memory. Exceeding the memory limit, or the knot limit
/// with the fixed breakpoints of a [`crate::SphericalSplinePlan`], is a
/// `ResourceLimit` error. In an adaptive search, the knot, insertion and
/// penalty limits instead end the search with the matching
/// [`AdaptiveSplineTermination`].
///
/// # Examples
///
/// ```
/// use scioxide_interp::SplineFitLimits;
/// let limits = SplineFitLimits::new()
///     .with_max_refinements(8)
///     .with_max_workspace_bytes(1 << 26);
/// assert_eq!((limits.max_refinements(), limits.max_knots()), (8, 128));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct SplineFitLimits {
    max_knots: usize,
    max_refinements: usize,
    max_penalty_iterations: usize,
    max_workspace_bytes: usize,
}

impl Default for SplineFitLimits {
    fn default() -> Self {
        Self::new()
    }
}

impl SplineFitLimits {
    /// Returns the default limits.
    pub const fn new() -> Self {
        Self {
            max_knots: 128,
            max_refinements: 64,
            max_penalty_iterations: 32,
            max_workspace_bytes: 64 * 1024 * 1024,
        }
    }

    /// Sets the largest knot-vector length per axis, counting the repeated end
    /// knots; adaptive fits require at least eight.
    pub const fn with_max_knots(mut self, knots: usize) -> Self {
        self.max_knots = knots;
        self
    }

    /// Sets the largest number of knot insertions; zero disables refinement.
    pub const fn with_max_refinements(mut self, refinements: usize) -> Self {
        self.max_refinements = refinements;
        self
    }

    /// Sets the largest number of penalty trials; zero keeps the fit with `λ = 0`.
    pub const fn with_max_penalty_iterations(mut self, iterations: usize) -> Self {
        self.max_penalty_iterations = iterations;
        self
    }

    /// Sets the memory limit in bytes for the working storage of a fit.
    ///
    /// The fit compares an upper bound on its working storage, including the
    /// factors it keeps between trials, with this limit before it allocates.
    /// The bound excludes your input and the returned spline.
    pub const fn with_max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = bytes;
        self
    }

    /// Returns the largest knot-vector length per axis.
    pub const fn max_knots(&self) -> usize {
        self.max_knots
    }

    /// Returns the largest number of knot insertions.
    pub const fn max_refinements(&self) -> usize {
        self.max_refinements
    }

    /// Returns the largest number of penalty trials.
    pub const fn max_penalty_iterations(&self) -> usize {
        self.max_penalty_iterations
    }

    /// Returns the memory limit in bytes.
    pub const fn max_workspace_bytes(&self) -> usize {
        self.max_workspace_bytes
    }

    /// Upper bound on the dense storage of a fit with `rows` design rows,
    /// `columns` coefficients and `components` right-hand sides.
    pub(crate) fn dense_bytes<S: RealScalar>(
        rows: usize,
        columns: usize,
        components: usize,
    ) -> Result<usize, SciOxideError> {
        // Eight dense payloads cover the design, augmented QR, packed factor,
        // factor staging, RHS/solutions, retained feasible trial, and scratch.
        // Scalar/index/basis vectors are included at 256 bytes per entry.
        let quantity = scioxide_core::quantity!("workspace_entries");
        let entries = checked_add(
            checked_mul(
                checked_add(rows, columns, quantity)?,
                checked_add(columns, components, quantity)?,
                quantity,
            )?,
            checked_add(rows, columns, quantity)?,
            quantity,
        )?;
        checked_mul(
            entries,
            core::mem::size_of::<S>().max(32) * 8,
            scioxide_core::quantity!("workspace_bytes"),
        )
    }

    /// Check `required` bytes against the memory limit.
    pub(crate) const fn admit(&self, required: usize) -> Result<(), SciOxideError> {
        if required > self.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("spline_workspace_bytes"),
                required,
                self.max_workspace_bytes,
            ));
        }
        Ok(())
    }

    /// Check a spherical fit against the memory limit: its dense storage,
    /// `extra` bytes and the factorization kernels.
    pub(crate) fn check_spherical<S: RealScalar>(
        &self,
        rows: usize,
        columns: usize,
        components: usize,
        extra: usize,
    ) -> Result<(), SciOxideError> {
        let quantity = scioxide_core::quantity!("spline_workspace_bytes");
        self.admit(checked_add(
            checked_add(
                Self::dense_bytes::<S>(rows, columns, components)?,
                extra,
                quantity,
            )?,
            SPHERE_KERNEL_BYTES,
            quantity,
        )?)
    }
}

/// Configures the spherical and parametric spline fits.
///
/// The defaults are the interpolation target `Residual(0)`, a residual
/// tolerance of machine epsilon, the default [`SplineFitLimits`] and no
/// budget limits. Each trial fit, whether after a knot insertion or in the
/// penalty search, counts as one iteration toward the budgets. Factorizations
/// inside a trial run in child scopes and do not count toward them; see
/// [solve accounting](scioxide_core#solve-accounting).
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, BudgetKind, SolveBudgets, Tolerance};
/// use scioxide_interp::{AdaptiveSplineOptions, SplineFitLimits, SplineFitTarget};
/// let options = AdaptiveSplineOptions::<f64>::new()
///     .with_target(SplineFitTarget::Residual(1e-8))
///     .with_residual_tolerance(Tolerance::new(1e-16, 1e-6)?)
///     .with_limits(SplineFitLimits::new().with_max_refinements(16))
///     .with_budgets(
///         SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(40)?),
///     );
/// assert_eq!(options.limits().max_refinements(), 16);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct AdaptiveSplineOptions<S: RealScalar> {
    target: SplineFitTarget<S>,
    residual_tolerance: Tolerance<S>,
    limits: SplineFitLimits,
    budgets: SolveBudgets,
}

impl<S: RealScalar> Default for AdaptiveSplineOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar> AdaptiveSplineOptions<S> {
    /// Returns the default options.
    pub fn new() -> Self {
        Self {
            target: SplineFitTarget::Residual(S::ZERO),
            residual_tolerance: Tolerance::from_machine_epsilon(),
            limits: SplineFitLimits::new(),
            budgets: SolveBudgets::unbounded(),
        }
    }

    /// Sets the penalty or residual target.
    pub const fn with_target(mut self, target: SplineFitTarget<S>) -> Self {
        self.target = target;
        self
    }

    /// Sets the tolerance on the weighted sum of squared residuals.
    ///
    /// A target `s` is met when the residual is at most `s` plus the tolerance
    /// threshold at `s`.
    pub const fn with_residual_tolerance(mut self, tolerance: Tolerance<S>) -> Self {
        self.residual_tolerance = tolerance;
        self
    }

    /// Sets the knot, trial and memory limits.
    pub const fn with_limits(mut self, limits: SplineFitLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Sets the budgets of the search, with one iteration per trial fit.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }

    /// Returns the penalty or residual target.
    pub const fn target(&self) -> SplineFitTarget<S> {
        self.target
    }

    /// Returns the residual tolerance.
    pub const fn residual_tolerance(&self) -> Tolerance<S> {
        self.residual_tolerance
    }

    /// Returns the knot, trial and memory limits.
    pub const fn limits(&self) -> SplineFitLimits {
        self.limits
    }

    /// Returns the budgets of the search.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }

    /// The options of the one-shot fits: the residual bound `residual` within `tolerance`.
    pub(crate) fn residual_bound(residual: S, tolerance: Tolerance<S>) -> Self {
        Self::new()
            .with_target(SplineFitTarget::Residual(residual))
            .with_residual_tolerance(tolerance)
    }

    /// Validate the target and the knot limit.
    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        let value = match self.target {
            SplineFitTarget::Penalty(x) | SplineFitTarget::Residual(x) => x,
        };
        if !value.is_finite() || value < S::ZERO || self.limits.max_knots < 8 {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "fit target must be finite/nonnegative and knot limit at least eight",
            ));
        }
        Ok(())
    }
}

/// Reports why a spherical or parametric spline fit stopped.
///
/// Every variant comes with the last fit that the search accepted. Only
/// [`Self::FixedPenalty`] and [`Self::ResidualSatisfied`] are successes.
///
/// # Examples
///
/// ```
/// use scioxide_interp::AdaptiveSplineTermination;
/// assert!(AdaptiveSplineTermination::ResidualSatisfied.is_success());
/// assert!(!AdaptiveSplineTermination::PenaltySearchLimit.is_success());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptiveSplineTermination {
    /// The fit with the given penalty was solved.
    FixedPenalty,
    /// The residual bound holds, and the penalty search reached the target
    /// within its tolerance or could not change `λ` further in floating point.
    ResidualSatisfied,
    /// The residual bound holds, but the penalty search used all its trials
    /// before reaching the target within its tolerance.
    PenaltySearchLimit,
    /// The knot or insertion limit was reached before the residual bound held.
    RefinementLimit,
    /// No span had samples to split before the residual bound held.
    NoAdmissibleKnot,
    /// The iteration budget ran out after at least one fit.
    BudgetExhausted,
}

impl AdaptiveSplineTermination {
    /// Returns whether the fit met its target.
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::FixedPenalty | Self::ResidualSatisfied)
    }
}

/// Reports the outcome of a spherical or parametric spline fit.
///
/// The report gives the termination status, the residual and penalty of the
/// returned fit, the trial counts and the knots used. The values describe the
/// computed fit and do not bound its error.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     AdaptiveSplineOptions, AdaptiveSplineTermination, CurveParameterization, SplineFitTarget,
///     fit_parametric_spline_with_options,
/// };
/// let parameters = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
/// let values: Vec<f64> = parameters.iter().map(|t| t * t).collect();
/// let options = AdaptiveSplineOptions::new().with_target(SplineFitTarget::Penalty(0.0));
/// let fit = fit_parametric_spline_with_options(
///     &values,
///     CurveParameterization::Supplied(&parameters),
///     None,
///     options,
/// )?;
/// let report = fit.report();
/// assert_eq!(
///     report.termination(),
///     AdaptiveSplineTermination::FixedPenalty
/// );
/// assert!(report.residual() < 1e-24 && report.fit_trials() == 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct AdaptiveSplineReport<S: RealScalar> {
    termination: AdaptiveSplineTermination,
    residual: S,
    penalty: S,
    refinements: usize,
    penalty_iterations: usize,
    fit_trials: usize,
    reciprocal_condition: S,
    knot_counts: [usize; 2],
    continuity: [usize; 2],
    pole_continuity: Option<usize>,
    coordinate_origin: [S; 2],
    coordinate_scale: [S; 2],
}

impl<S: RealScalar> AdaptiveSplineReport<S> {
    /// Fails with `ConvergenceFailure` unless the fit met its target; for the one-shot fits.
    pub(crate) fn require_success(&self, algorithm: AlgorithmId) -> Result<(), SciOxideError> {
        let reason = match self.termination {
            AdaptiveSplineTermination::FixedPenalty
            | AdaptiveSplineTermination::ResidualSatisfied => return Ok(()),
            AdaptiveSplineTermination::RefinementLimit => ConvergenceReason::TruncationLimit,
            AdaptiveSplineTermination::NoAdmissibleKnot => ConvergenceReason::Stagnation,
            AdaptiveSplineTermination::PenaltySearchLimit
            | AdaptiveSplineTermination::BudgetExhausted => ConvergenceReason::IterationLimit,
        };
        Err(unconverged(
            reason,
            algorithm,
            self.fit_trials,
            "spline fit stopped before meeting its residual target",
        ))
    }

    /// Report of one fixed-penalty objective.
    pub(crate) fn fixed(residual: S, penalty: S, reciprocal_condition: S) -> Self {
        Self {
            termination: AdaptiveSplineTermination::FixedPenalty,
            residual,
            penalty,
            refinements: 0,
            penalty_iterations: 0,
            fit_trials: 1,
            reciprocal_condition,
            knot_counts: [0, 0],
            continuity: [2, 0],
            pole_continuity: None,
            coordinate_origin: [S::ZERO; 2],
            coordinate_scale: [S::ONE; 2],
        }
    }

    /// Set knot counts, continuity and coordinate normalization.
    pub(crate) const fn with_geometry(
        mut self,
        knot_counts: [usize; 2],
        continuity: [usize; 2],
        pole_continuity: Option<usize>,
        coordinate_origin: [S; 2],
        coordinate_scale: [S; 2],
    ) -> Self {
        self.knot_counts = knot_counts;
        self.continuity = continuity;
        self.pole_continuity = pole_continuity;
        self.coordinate_origin = coordinate_origin;
        self.coordinate_scale = coordinate_scale;
        self
    }

    /// Returns the termination status.
    pub const fn termination(&self) -> AdaptiveSplineTermination {
        self.termination
    }

    /// Returns the weighted sum of squared residuals `Σ_i w_i |f(x_i) − y_i|²`,
    /// summed over all real value components.
    pub const fn residual(&self) -> S {
        self.residual
    }

    /// Returns the smoothing parameter `λ` of the returned fit.
    pub const fn penalty(&self) -> S {
        self.penalty
    }

    /// Returns the number of inserted interior knots.
    pub const fn refinements(&self) -> usize {
        self.refinements
    }

    /// Returns the number of penalty trials.
    pub const fn penalty_iterations(&self) -> usize {
        self.penalty_iterations
    }

    /// Returns the number of trial fits, including those after knot insertions.
    pub const fn fit_trials(&self) -> usize {
        self.fit_trials
    }

    /// Returns a cheap reciprocal conditioning indicator of the final solve.
    ///
    /// Small values warn of an ill-conditioned fit. It is not a bound on the
    /// condition number.
    pub const fn reciprocal_condition(&self) -> S {
        self.reciprocal_condition
    }

    /// Returns the knot-vector length of each axis; the second entry is zero for curves.
    pub const fn knot_counts(&self) -> [usize; 2] {
        self.knot_counts
    }

    /// Returns the continuity order at interior knots of each axis; the second
    /// entry is zero for curves.
    pub const fn continuity(&self) -> [usize; 2] {
        self.continuity
    }

    /// Returns the continuity order at the poles of a spherical fit, or `None` for curves.
    pub const fn pole_continuity(&self) -> Option<usize> {
        self.pole_continuity
    }

    /// Returns the origin of each normalized coordinate, in parameter units.
    pub const fn coordinate_origin(&self) -> [S; 2] {
        self.coordinate_origin
    }

    /// Returns the scale of each normalized coordinate; an unused curve axis has scale one.
    pub const fn coordinate_scale(&self) -> [S; 2] {
        self.coordinate_scale
    }
}

/// One fitted objective: its model, per-sample weighted squared residuals and report.
pub(crate) struct Candidate<M, S: RealScalar> {
    pub(crate) model: M,
    pub(crate) residuals: Vec<S>,
    pub(crate) report: AdaptiveSplineReport<S>,
}

/// Validate one equally shaped, finite value per sample and return its dimension.
pub(crate) fn sample_dimension<V: AffineValue>(
    values: &[V],
    expected: usize,
) -> Result<usize, SciOxideError> {
    if values.len() != expected {
        return Err(crate::error::shape(expected, values.len()));
    }
    let first = values
        .first()
        .ok_or_else(|| crate::error::shape(expected.max(1), 0))?;
    let dimension = first.dimension()?;
    if dimension == 0 {
        return Err(invalid(
            scioxide_core::argument!("values"),
            "fit values need at least one real component",
        ));
    }
    for value in values {
        scioxide_core::check_value_shape(value, dimension)?;
        scioxide_core::validate_value(value)?;
    }
    Ok(dimension)
}

/// Copy or default (all ones) the caller's `weights` for `samples` observations.
pub(crate) fn weights<S: RealScalar>(
    values: Option<&[S]>,
    samples: usize,
) -> Result<Vec<S>, SciOxideError> {
    validate_count(samples, 1, scioxide_core::argument!("samples"))?;
    match values {
        Some(w) => {
            validate_weights(
                w,
                samples,
                Weights::Observed,
                scioxide_core::argument!("weights"),
            )?;
            try_copy_slice(w, scioxide_core::quantity!("weights"))
        }
        None => try_filled(samples, S::ONE, scioxide_core::quantity!("weights")),
    }
}

/// Open cubic basis on strictly increasing `breaks`, blaming `argument`.
pub(crate) fn open_cubic<S: RealScalar>(
    breaks: &[S],
    argument: Argument,
) -> Result<BSplineBasis<S>, SciOxideError> {
    BSplineBasis::with_repeated_ends(breaks, 2, 1, 3, argument)
}

/// Midpoint of the interior samples of the residual-heaviest splittable span,
/// with stable coordinate ties.
pub(crate) fn insertion<S: RealScalar>(
    breaks: &[S],
    coordinates: &[S],
    residuals: &[S],
) -> Result<Option<S>, SciOxideError> {
    let mut best = None;
    let mut highest = -S::ONE;
    for span in breaks.windows(2) {
        let mut score = S::ZERO;
        let mut lower = None;
        let mut upper = None;
        for (&x, &r) in coordinates.iter().zip(residuals) {
            if x >= span[0] && x <= span[1] {
                score = intermediate(score + r)?;
                if x > span[0] && x < span[1] {
                    lower = Some(lower.map_or(x, |v: S| v.min(x)));
                    upper = Some(upper.map_or(x, |v: S| v.max(x)));
                }
            }
        }
        if let (Some(a), Some(b)) = (lower, upper) {
            let knot = scioxide_core::checked_midpoint(a, b)?;
            if score > highest && knot > span[0] && knot < span[1] {
                best = Some(knot);
                highest = score;
            }
        }
    }
    Ok(best)
}

/// Insert a breakpoint in order.
pub(crate) fn insert<S: RealScalar>(breaks: &mut Vec<S>, knot: S) -> Result<(), SciOxideError> {
    breaks.try_reserve(1).map_err(|_| {
        SciOxideError::allocation_failure(
            scioxide_core::quantity!("breakpoints"),
            core::mem::size_of::<S>(),
        )
    })?;
    let index = breaks.partition_point(|x| *x < knot);
    breaks.insert(index, knot);
    Ok(())
}

/// Run the bounded refinement and monotone penalty search in a new solve of
/// `scope` with the budgets [`AdaptiveSplineOptions::budgets`].
///
/// `trial` fits one objective on the current axes with a penalty, running any
/// nested factorization in a child scope of its context; `refine` inserts one
/// knot from per-sample residuals and reports whether it did. Each trial
/// counts as one iteration. The result is the last accepted fit with its
/// termination and counts; running out of budget before the first fit is a
/// `ConvergenceFailure`.
pub(crate) fn search<'scope, 'capability, M, S: RealScalar>(
    scope: impl Into<SolveScope<'scope, 'capability>>,
    algorithm: AlgorithmId,
    shape: u128,
    axes: &mut [Vec<S>],
    options: &AdaptiveSplineOptions<S>,
    trial: impl FnMut(&[Vec<S>], S, &mut SolveContext<'_, '_>) -> Result<Candidate<M, S>, SciOxideError>,
    refine: impl FnMut(&mut [Vec<S>], &[S]) -> Result<bool, SciOxideError>,
) -> Result<Candidate<M, S>, SciOxideError> {
    let identity = SolveIdentity::new(ProblemToken::new(0), algorithm, shape, 0);
    scope.into().run(
        &mut SolveState::new(),
        identity,
        options.budgets,
        |context| search_in(axes, options, context, trial, refine),
    )
}

/// Record the stop of a search on its last accepted candidate.
fn finish<M, S: RealScalar>(
    mut best: Candidate<M, S>,
    termination: AdaptiveSplineTermination,
    counts: [usize; 3],
) -> Candidate<M, S> {
    let [refinements, iterations, trials] = counts;
    best.report.termination = termination;
    best.report.refinements = refinements;
    best.report.penalty_iterations = iterations;
    best.report.fit_trials = trials;
    best
}

/// The search of [`search`] inside its solve context.
fn search_in<M, S: RealScalar>(
    axes: &mut [Vec<S>],
    options: &AdaptiveSplineOptions<S>,
    context: &mut SolveContext<'_, '_>,
    mut trial: impl FnMut(
        &[Vec<S>],
        S,
        &mut SolveContext<'_, '_>,
    ) -> Result<Candidate<M, S>, SciOxideError>,
    mut refine: impl FnMut(&mut [Vec<S>], &[S]) -> Result<bool, SciOxideError>,
) -> Result<Candidate<M, S>, SciOxideError> {
    let target = match options.target {
        SplineFitTarget::Penalty(lambda) => {
            if !admit_iteration(context)? {
                return Err(iteration_denied(context, FIRST_TRIAL));
            }
            return trial(axes, lambda, context);
        }
        SplineFitTarget::Residual(target) => target,
    };
    let tolerance = options.residual_tolerance.threshold(target)?;
    let bound = intermediate(target + tolerance)?;
    let limits = options.limits;
    if !admit_iteration(context)? {
        return Err(iteration_denied(context, FIRST_TRIAL));
    }
    let mut best = trial(axes, S::ZERO, context)?;
    let mut refinements = 0;
    let mut trials = 1;
    while best.report.residual > bound {
        let stop = if refinements == limits.max_refinements
            || axes.iter().all(|a| a.len() + 6 >= limits.max_knots)
        {
            Some(AdaptiveSplineTermination::RefinementLimit)
        } else if !refine(axes, &best.residuals)? {
            Some(AdaptiveSplineTermination::NoAdmissibleKnot)
        } else if !admit_iteration(context)? {
            Some(AdaptiveSplineTermination::BudgetExhausted)
        } else {
            None
        };
        if let Some(termination) = stop {
            return Ok(finish(best, termination, [refinements, 0, trials]));
        }
        best = trial(axes, S::ZERO, context)?;
        refinements += 1;
        trials += 1;
    }
    // Retain the feasible endpoint throughout a bounded monotone penalty search.
    let mut termination = AdaptiveSplineTermination::ResidualSatisfied;
    let mut iterations = 0;
    if target > S::ZERO {
        let mut low = S::ZERO;
        let mut high = S::ONE;
        let mut bracketed = false;
        termination = loop {
            if iterations == limits.max_penalty_iterations {
                break AdaptiveSplineTermination::PenaltySearchLimit;
            }
            let lambda = if bracketed {
                scioxide_core::checked_midpoint(low, high)?
            } else {
                high
            };
            if lambda == low || (bracketed && lambda == high) {
                break AdaptiveSplineTermination::ResidualSatisfied;
            }
            if !admit_iteration(context)? {
                break AdaptiveSplineTermination::BudgetExhausted;
            }
            let candidate = trial(axes, lambda, context)?;
            iterations += 1;
            trials += 1;
            if candidate.report.residual <= bound {
                low = lambda;
                best = candidate;
                if (best.report.residual - target).abs() <= tolerance {
                    break AdaptiveSplineTermination::ResidualSatisfied;
                }
                if !bracketed {
                    let next = high * S::try_from_usize_exact(16)?;
                    if !next.is_finite() {
                        break AdaptiveSplineTermination::ResidualSatisfied;
                    }
                    high = next;
                }
            } else {
                high = lambda;
                bracketed = true;
            }
        };
    }
    Ok(finish(best, termination, [refinements, iterations, trials]))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::BudgetKind;
    use scioxide_core::{Budget, ExecutionContext, SciOxideErrorKind};

    /// A synthetic objective whose residual is `scale / knots + penalty`.
    fn run(
        options: AdaptiveSplineOptions<f64>,
        scale: f64,
    ) -> Result<Candidate<(), f64>, SciOxideError> {
        let mut axes = [vec![0.0, 1.0]];
        search(
            ExecutionContext::sequential_native(),
            scioxide_core::algorithm_id!("interp", "adaptive-test", 1),
            0,
            &mut axes,
            &options,
            |axes, lambda, _| {
                let residual = scale / axes[0].len() as f64 + lambda;
                Ok(Candidate {
                    model: (),
                    residuals: vec![residual],
                    report: AdaptiveSplineReport::fixed(residual, lambda, 1.0),
                })
            },
            |axes, _| {
                let last = axes[0][axes[0].len() - 1];
                axes[0].push(last + 1.0);
                Ok(true)
            },
        )
    }

    #[test]
    fn unsuccessful_terminations_map_to_convergence_reasons() {
        use scioxide_core::SciOxideErrorView;
        let algorithm = scioxide_core::algorithm_id!("interp", "adaptive-test", 1);
        for (termination, expected) in [
            (AdaptiveSplineTermination::FixedPenalty, None),
            (AdaptiveSplineTermination::ResidualSatisfied, None),
            (
                AdaptiveSplineTermination::RefinementLimit,
                Some(ConvergenceReason::TruncationLimit),
            ),
            (
                AdaptiveSplineTermination::NoAdmissibleKnot,
                Some(ConvergenceReason::Stagnation),
            ),
            (
                AdaptiveSplineTermination::PenaltySearchLimit,
                Some(ConvergenceReason::IterationLimit),
            ),
            (
                AdaptiveSplineTermination::BudgetExhausted,
                Some(ConvergenceReason::IterationLimit),
            ),
        ] {
            let mut report = AdaptiveSplineReport::fixed(0.5, 0.0, 1.0);
            report.termination = termination;
            report.fit_trials = 7;
            let result = report.require_success(algorithm);
            match expected {
                None => assert!(result.is_ok()),
                Some(expected) => assert!(matches!(
                    result.unwrap_err().view(),
                    SciOxideErrorView::ConvergenceFailure { reason, diagnostics, .. }
                        if reason == expected
                            && diagnostics.algorithm() == Some(algorithm)
                            && diagnostics.counts().own().iterations() == 7
                )),
            }
        }
    }

    #[test]
    fn limits_and_options_round_trip() {
        let limits = SplineFitLimits::new()
            .with_max_knots(9)
            .with_max_refinements(2)
            .with_max_penalty_iterations(3)
            .with_max_workspace_bytes(4);
        assert_eq!(
            [
                limits.max_knots(),
                limits.max_refinements(),
                limits.max_penalty_iterations(),
                limits.max_workspace_bytes()
            ],
            [9, 2, 3, 4]
        );
        assert_eq!(SplineFitLimits::default(), SplineFitLimits::new());
        let error = limits
            .admit(SplineFitLimits::dense_bytes::<f64>(1, 1, 1).unwrap())
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
        let error = SplineFitLimits::new()
            .check_spherical::<f64>(1, 1, 1, usize::MAX)
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::SizeOverflow);
        let options = AdaptiveSplineOptions::<f64>::new().with_limits(limits);
        assert!(options.validate().is_ok());
        assert!(
            options
                .with_limits(limits.with_max_knots(7))
                .validate()
                .is_err()
        );
        assert!(
            options
                .with_target(SplineFitTarget::Penalty(-1.0))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn penalty_search_limit_only_when_unsatisfied() {
        // Residual 1/len + λ against target 0.75: refinement stops at two knots
        // (0.5) and the penalty search needs several bisections.
        let options = AdaptiveSplineOptions::new()
            .with_target(SplineFitTarget::Residual(0.75))
            .with_residual_tolerance(Tolerance::new(1e-3, 0.0).unwrap());
        let unlimited = run(options, 1.0).unwrap();
        assert_eq!(
            unlimited.report.termination(),
            AdaptiveSplineTermination::ResidualSatisfied
        );
        let needed = unlimited.report.penalty_iterations();
        assert!(needed > 1);
        let exact = run(
            options.with_limits(SplineFitLimits::new().with_max_penalty_iterations(needed)),
            1.0,
        )
        .unwrap();
        assert_eq!(
            exact.report.termination(),
            AdaptiveSplineTermination::ResidualSatisfied
        );
        let short = run(
            options.with_limits(SplineFitLimits::new().with_max_penalty_iterations(needed - 1)),
            1.0,
        )
        .unwrap();
        assert_eq!(
            short.report.termination(),
            AdaptiveSplineTermination::PenaltySearchLimit
        );
        assert_eq!(short.report.penalty_iterations(), needed - 1);
    }

    #[test]
    fn budgets_end_the_search_after_an_accepted_trial() {
        let options = AdaptiveSplineOptions::new()
            .with_target(SplineFitTarget::Residual(0.1))
            .with_budgets(
                SolveBudgets::unbounded()
                    .with_limit(BudgetKind::Iteration, Budget::try_from(2).unwrap()),
            );
        let result = run(options, 1.0).unwrap();
        assert_eq!(
            result.report.termination(),
            AdaptiveSplineTermination::BudgetExhausted
        );
        assert_eq!(result.report.fit_trials(), 2);
        let limited = run(
            options
                .with_budgets(SolveBudgets::unbounded())
                .with_limits(SplineFitLimits::new().with_max_refinements(1)),
            1.0,
        )
        .unwrap();
        assert_eq!(
            limited.report.termination(),
            AdaptiveSplineTermination::RefinementLimit
        );
    }

    #[test]
    fn insertion_prefers_heaviest_span() {
        let knot = insertion(
            &[0.0_f64, 1.0, 2.0],
            &[0.2, 0.4, 1.2, 1.8],
            &[1.0, 1.0, 5.0, 5.0],
        )
        .unwrap()
        .unwrap();
        assert!((knot - 1.5).abs() < 1e-15);
        let mut breaks = vec![0.0, 2.0];
        insert(&mut breaks, 1.0).unwrap();
        assert_eq!(breaks, [0.0, 1.0, 2.0]);
        assert!(
            insertion(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 1.0])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn values_and_weights_are_validated() {
        assert_eq!(sample_dimension(&[[1.0, 2.0], [3.0, 4.0]], 2).unwrap(), 2);
        assert_eq!(
            sample_dimension(&[1.0, 2.0], 3).unwrap_err().kind(),
            SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(
            sample_dimension(&[f64::NAN], 1).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
        assert!(sample_dimension::<Vec<f64>>(&[vec![]], 1).is_err());
        assert_eq!(weights::<f64>(None, 2).unwrap(), [1.0, 1.0]);
        assert!(weights(Some(&[0.0, 0.0]), 2).is_err());
        assert!(weights(Some(&[1.0]), 2).is_err());
        assert!(open_cubic(&[0.0, 0.0], scioxide_core::argument!("breaks")).is_err());
        assert_eq!(
            open_cubic(&[0.0, 0.5, 1.0], scioxide_core::argument!("breaks"))
                .unwrap()
                .control_count(),
            5
        );
    }
}
