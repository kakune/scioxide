//! Projected limited-memory BFGS for bound constraints.
//!
//! The reduced direction is `-P_F H P_F g`, with every gradient-blocked active
//! coordinate and any remaining outward component removed. Numerical failure or
//! loss of descent after shortening resets the history and retries projected
//! steepest descent; success still requires the projected-gradient test.

use alloc::vec::Vec;
use core::time::Duration;

use crate::attempt::{Accepted, Entry, Shape};
use crate::quasi_newton::scaled_dot;
use crate::unconstrained::FirstOrderStorage;
use crate::utils::{finish_first_order, line_search_termination, validate_x0};
use crate::{
    BacktrackingLineSearch, BoundedObjective, CurvaturePolicy, HessianUpdateReport, LineSearch,
    ObjectiveFunction, OptimizeResult, OptimizeTermination,
};
use scioxide_core::{
    AttemptKind, Bounds, Budget, GradientTolerance, NumericScalar, RealScalar, SciOxideError,
    SolveScope,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{
    DirectSolveOptions, DirectSolveTermination, LinalgExecutionOptions, LinalgWorkspace,
    LuDecomposition, LuWorkspaceRequest, MatrixView, RankPolicy, VectorView, VectorViewMut,
};

fn dot<S: RealScalar>(a: &[S], b: &[S]) -> Result<S, SciOxideError> {
    scaled_dot(a, b)?.round()
}
use crate::validation::invalid;
fn breakdown(message: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::Breakdown, message)
}
fn finite<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(breakdown("non-finite projected model"))
    }
}
/// Sets the stopping tolerance, limits, memory, derivative source and curvature policy of [`LbfgsB`].
///
/// The defaults are the default [`GradientTolerance`], 1000 iterations, a
/// memory of 10 pairs, the default [`DerivativeStrategy`], the default
/// [`CurvaturePolicy`] and no time limit. The line search is Armijo
/// backtracking that allows a rounding slack of 8 `ε` in the objective. The
/// stopping test is described under [Termination](crate#termination).
#[derive(Debug, Clone, Copy)]
pub struct LbfgsBOptions<S: RealScalar> {
    gradient: GradientTolerance<S>,
    iterations: Budget,
    memory: Budget,
    derivative: DerivativeStrategy,
    curvature: CurvaturePolicy<S>,
    line_search: BacktrackingLineSearch<S>,
    time_limit: Option<Duration>,
}
impl<S: RealScalar> Default for LbfgsBOptions<S> {
    fn default() -> Self {
        Self {
            gradient: GradientTolerance::default(),
            iterations: Budget::ONE.saturating_add(999),
            memory: Budget::ONE.saturating_add(9),
            derivative: DerivativeStrategy::default(),
            curvature: CurvaturePolicy::default(),
            line_search: BacktrackingLineSearch::default().with_roundoff_allowance(),
            time_limit: None,
        }
    }
}
impl<S: RealScalar> LbfgsBOptions<S> {
    /// Sets the tolerance on the projected gradient.
    pub const fn with_gradient_tolerance(mut self, value: GradientTolerance<S>) -> Self {
        self.gradient = value;
        self
    }
    /// Sets the iteration limit.
    pub const fn with_max_iterations(mut self, value: Budget) -> Self {
        self.iterations = value;
        self
    }
    /// Sets the number `m` of stored correction pairs.
    pub const fn with_memory(mut self, value: Budget) -> Self {
        self.memory = value;
        self
    }
    /// Sets the order in which derivative sources are tried.
    pub const fn with_derivative_strategy(mut self, value: DerivativeStrategy) -> Self {
        self.derivative = value;
        self
    }
    /// Sets the curvature policy of the updates.
    pub const fn with_curvature_policy(mut self, value: CurvaturePolicy<S>) -> Self {
        self.curvature = value;
        self
    }
    /// Sets a wall-clock time limit. The execution context must provide a clock.
    pub const fn with_time_limit(mut self, value: Duration) -> Self {
        self.time_limit = Some(value);
        self
    }
    /// Returns the tolerance on the projected gradient.
    pub const fn gradient_tolerance(self) -> GradientTolerance<S> {
        self.gradient
    }
    /// Returns the iteration limit.
    pub const fn max_iterations(self) -> Budget {
        self.iterations
    }
    /// Returns the number `m` of stored correction pairs.
    pub const fn memory(self) -> Budget {
        self.memory
    }
    /// Returns the order in which derivative sources are tried.
    pub const fn derivative_strategy(self) -> DerivativeStrategy {
        self.derivative
    }
    /// Returns the curvature policy of the updates.
    pub const fn curvature_policy(self) -> CurvaturePolicy<S> {
        self.curvature
    }
    /// Returns the time limit, if set.
    pub const fn time_limit(self) -> Option<Duration> {
        self.time_limit
    }
}

/// Projected L-BFGS history and iterate storage, O(nm + m²).
#[derive(Debug)]
pub(crate) struct LbfgsBStorage<S: RealScalar + NumericScalar<Real = S>> {
    first: FirstOrderStorage<S>,
    projected: S,
    coefficients: Vec<S>,
    coefficient_product: Vec<S>,
    projection: Vec<S>,
    norms: Vec<S>,
    physical_basis: Vec<S>,
    factor: LuDecomposition<S>,
    linear_workspace: LinalgWorkspace<'static, S>,
    #[cfg(test)]
    direction_trace: Option<Vec<DirectionObservation>>,
}
#[cfg(test)]
#[derive(Debug)]
struct DirectionObservation {
    active: Vec<bool>,
    history_before: usize,
    history_after: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for LbfgsBStorage<S> {
    fn default() -> Self {
        Self {
            first: FirstOrderStorage::new(),
            projected: S::ZERO,
            coefficients: Vec::new(),
            coefficient_product: Vec::new(),
            projection: Vec::new(),
            norms: Vec::new(),
            physical_basis: Vec::new(),
            factor: LuDecomposition::empty(),
            linear_workspace: LinalgWorkspace::new(),
            #[cfg(test)]
            direction_trace: None,
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> LbfgsBStorage<S> {
    fn fits(&self, dimension: usize, memory: usize) -> bool {
        self.first.fits(dimension, memory, false)
    }
    fn try_reserve(&mut self, dimension: usize, memory: usize) -> Result<(), SciOxideError> {
        if !self.fits(dimension, memory) {
            let mut next = Self::default();
            next.first.try_reserve(dimension, memory)?;
            let k = scioxide_core::backend::checked_mul(
                2,
                memory,
                scioxide_core::quantity!("history_columns"),
            )?;
            let square = scioxide_core::backend::checked_mul(
                k,
                k,
                scioxide_core::quantity!("history_coefficients"),
            )?;
            crate::storage::resize(&mut next.first.history_ss, square / 4, S::ZERO)?;
            crate::storage::resize(&mut next.coefficients, square, S::ZERO)?;
            crate::storage::resize(&mut next.coefficient_product, k, S::ZERO)?;
            crate::storage::resize(&mut next.projection, k, S::ZERO)?;
            crate::storage::resize(&mut next.norms, k, S::ZERO)?;
            let basis = scioxide_core::backend::checked_mul(
                k,
                dimension,
                scioxide_core::quantity!("history_fallback_basis"),
            )?;
            crate::storage::resize(&mut next.physical_basis, basis, S::ZERO)?;
            next.factor.try_reserve(k)?;
            let options = LinalgExecutionOptions::new()
                .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);
            for request in [
                LuWorkspaceRequest::factor(k, &options),
                LuWorkspaceRequest::solve(k, &options),
            ] {
                next.linear_workspace
                    .try_reserve(&LuDecomposition::<S>::workspace_requirements(&request)?)?;
            }
            *self = next;
        }
        Ok(())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Accepted<S> for LbfgsBStorage<S> {
    fn accepted(&self) -> Option<crate::attempt::AcceptedState<'_, S>> {
        self.first.accepted()
    }
}

/// Describes the workspace that an [`LbfgsB`] solve needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LbfgsBWorkspaceRequirements {
    dimension: usize,
    history: usize,
}

impl LbfgsBWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }

    /// Returns the number of stored correction pairs.
    pub const fn history(self) -> usize {
        self.history
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Shape<LbfgsBStorage<S>>
    for LbfgsBWorkspaceRequirements
{
    fn fits(&self, storage: &LbfgsBStorage<S>) -> bool {
        storage.fits(self.dimension, self.history)
    }
    fn prepare(&self, storage: &mut LbfgsBStorage<S>) -> Result<(), SciOxideError> {
        storage.try_reserve(self.dimension, self.history)
    }
}

crate::attempt::local_workspace!(
    /// Holds the storage and the record of the last call for [`LbfgsB`]:
    /// O(nm + m²) values for `n` variables and memory `m`.
    LbfgsBWorkspace,
    LbfgsBStorage<S>,
    LbfgsBWorkspaceRequirements
);

/// Reject a start of the wrong shape, with nonfinite entries or outside the box.
fn validate_start<S, F>(objective: &F, bounds: &Bounds<S>, x0: &[S]) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    validate_x0(objective, x0)?;
    if x0
        .iter()
        .zip(bounds.as_slice())
        .any(|(&v, &(lo, hi))| lo.is_some_and(|b| v < b) || hi.is_some_and(|b| v > b))
    {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("x0"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "initial point is outside bounds",
        ));
    }
    Ok(())
}

fn projected_gradient<S: RealScalar>(
    x: &[S],
    gradient: &[S],
    bounds: &Bounds<S>,
) -> Result<(S, S), SciOxideError> {
    let mut norm: S = <_ as scioxide_core::AlgebraicScalar>::ZERO;
    let mut l2: S = <_ as scioxide_core::AlgebraicScalar>::ZERO;
    for ((&value, &g), &(lo, hi)) in x.iter().zip(gradient).zip(bounds.as_slice()) {
        let component = if matches!((lo, hi), (Some(lower), Some(upper)) if lower == upper)
            || (lo.is_some_and(|lower| value == lower)
                && g >= <_ as scioxide_core::AlgebraicScalar>::ZERO)
            || (hi.is_some_and(|upper| value == upper)
                && g <= <_ as scioxide_core::AlgebraicScalar>::ZERO)
        {
            <_ as scioxide_core::AlgebraicScalar>::ZERO
        } else {
            g.abs()
        };
        norm = norm.max(component);
        l2 = l2.hypot(component);
    }
    Ok((finite(norm)?, finite(l2)?))
}

fn blocked<S: RealScalar>(x: S, g: S, bounds: (Option<S>, Option<S>)) -> bool {
    matches!(bounds, (Some(lo), Some(hi)) if lo == hi)
        || (bounds.0.is_some_and(|lo| x == lo) && g >= S::ZERO)
        || (bounds.1.is_some_and(|hi| x == hi) && g <= S::ZERO)
}

// Project the unit endpoint before searching its feasible segment. Comparing
// against a finite bound gap also handles an overflowing unprojected endpoint.
fn feasible_component<S: RealScalar>(
    x: S,
    mut d: S,
    bounds: (Option<S>, Option<S>),
) -> Result<S, SciOxideError> {
    if let Some(hi) = bounds.1
        && d > S::ZERO
    {
        let gap = hi - x;
        if d >= gap {
            d = gap;
        }
    }
    if let Some(lo) = bounds.0
        && d < S::ZERO
    {
        let gap = lo - x;
        if d <= gap {
            d = gap;
        }
    }
    let trial = crate::utils::trial_component(x, S::ONE, d)?;
    let end = crate::geometry::project_component(trial, bounds);
    if end == trial {
        Ok(d)
    } else {
        // Subtracting an opposite-sign origin can round the bound gap outward.
        // One inward ULP makes its exact endpoint feasible before the line
        // search rounds any point on this segment.
        let gap = scioxide_core::checked_difference(end, x)?;
        let endpoint = crate::utils::trial_component(x, S::ONE, gap)?;
        Ok(
            if crate::geometry::project_component(endpoint, bounds) == endpoint {
                gap
            } else {
                gap.next_after(S::ZERO)
            },
        )
    }
}

// BLNZ compact B = theta I - theta W K^-1 W', with unit-norm
// W=[S,Y]. Diagonal equilibration makes K dimensionless; only its O(m³)
// factorization changes when history changes. All vector products cost O(nm).
fn history_gram<S: RealScalar>(first: &FirstOrderStorage<S>, row: usize, col: usize) -> S {
    let len = first.history_len;
    let m = first.history_capacity;
    let i = (first.history_start + row % len) % m;
    let j = (first.history_start + col % len) % m;
    match (row < len, col < len) {
        (true, true) => first.history_ss[i * m + j],
        (true, false) => first.history_sy[i * m + j],
        (false, true) => first.history_sy[j * m + i],
        (false, false) => first.history_yy[i * m + j],
    }
}

fn projected_model_curvature<S: RealScalar>(
    storage: &mut LbfgsBStorage<S>,
    execution: scioxide_core::ExecutionContext<'_>,
) -> Result<(scioxide_core::Scaled<S>, S), SciOxideError> {
    let first = &mut storage.first;
    let m = first.history_len;
    let k = 2 * m;
    let theta = if m == 0 {
        scioxide_core::Scaled::from_value(S::ONE)?
    } else {
        let slot = (first.history_start + m - 1) % first.history_capacity;
        let range = slot * first.dimension..(slot + 1) * first.dimension;
        scaled_dot(&first.history_y[range.clone()], &first.history_y[range])?
            .div(first.history_curvature[slot])?
    };
    let ss = scaled_dot(&first.s, &first.s)?;
    let base = theta.mul(ss)?;
    if m == 0 {
        return Ok((base, S::ZERO));
    }
    if !first.history_cache_valid {
        return Err(breakdown("history Gram range is not representable"));
    }
    let theta = theta.round()?;
    for (row, norm) in storage.norms[..k].iter_mut().enumerate() {
        *norm = finite(history_gram(first, row, row).sqrt())?;
        if *norm <= S::ZERO {
            return Err(breakdown("zero compact history column"));
        }
    }
    let c = &mut storage.coefficients[..k * k];
    c.fill(S::ZERO);
    for row in 0..m {
        for col in 0..m {
            let denom = scioxide_core::Scaled::from_value(storage.norms[row])?
                .mul(scioxide_core::Scaled::from_value(storage.norms[col])?)?;
            c[row * k + col] = scioxide_core::Scaled::from_value(history_gram(first, row, col))?
                .div(denom)?
                .round()?;
            if row > col {
                let denom = scioxide_core::Scaled::from_value(storage.norms[row])?
                    .mul(scioxide_core::Scaled::from_value(storage.norms[m + col])?)?;
                let value = scioxide_core::Scaled::from_value(history_gram(first, row, m + col))?
                    .div(denom)?
                    .round()?;
                c[row * k + m + col] = value;
                c[(m + col) * k + row] = value;
            }
        }
        c[(m + row) * k + m + row] = -scioxide_core::Scaled::from_value(theta)?
            .mul(scioxide_core::Scaled::from_value(history_gram(
                first,
                row,
                m + row,
            ))?)?
            .div(scioxide_core::Scaled::from_value(history_gram(
                first,
                m + row,
                m + row,
            ))?)?
            .round()?;
    }
    for (row, projection) in storage.projection[..k].iter_mut().enumerate() {
        let slot = (first.history_start + row % m) % first.history_capacity;
        let range = slot * first.dimension..(slot + 1) * first.dimension;
        let column = if row < m {
            &first.history_s[range]
        } else {
            &first.history_y[range]
        };
        *projection = scaled_dot(column, &first.s)?
            .div(scioxide_core::Scaled::from_value(storage.norms[row])?)?
            .round()?;
    }
    let options = LinalgExecutionOptions::new()
        .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential);
    LuDecomposition::factor_into(
        MatrixView::from_slice(k, k, c)?,
        &options,
        &mut storage.factor,
        &mut storage.linear_workspace,
        execution,
    )
    .map_err(|error| {
        if matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ) {
            breakdown("singular compact history model")
        } else {
            error
        }
    })?;
    let report = storage.factor.solve_into(
        VectorView::from_slice(&storage.projection[..k]),
        &DirectSolveOptions::new(RankPolicy::DimensionScaledEpsilon)
            .with_execution_options(options),
        VectorViewMut::from_slice(&mut storage.coefficient_product[..k]),
        &mut storage.linear_workspace,
        execution,
    )?;
    if report.termination() != DirectSolveTermination::Accepted {
        return Err(breakdown("unresolved compact history solve"));
    }
    let correction = scioxide_core::Scaled::from_value(theta)?.mul(scaled_dot(
        &storage.projection[..k],
        &storage.coefficient_product[..k],
    )?)?;
    let magnitude = finite(
        base.round()?.abs()
            + storage.projection[..k]
                .iter()
                .zip(&storage.coefficient_product[..k])
                .fold(S::ZERO, |sum, (&a, &b)| sum + (a * b * theta).abs()),
    )?;
    let error_units = S::try_from_f64_lossy(64.0 * (k as f64) * (k as f64))?;
    let error = finite(error_units * S::EPSILON * magnitude)?;
    Ok((base.sub(correction)?, error))
}

fn normalize_column<S: RealScalar>(
    values: &mut [S],
    denominator: scioxide_core::Scaled<S>,
) -> Result<(), SciOxideError> {
    let native = denominator
        .round()
        .ok()
        .filter(|value| value.abs() >= S::MIN_POSITIVE);
    for value in values {
        if let Some(denominator) = native {
            let candidate = *value / denominator;
            if candidate.is_finite() {
                *value = candidate;
                continue;
            }
        }
        *value = scioxide_core::Scaled::from_value(*value)?
            .div(denominator)?
            .round()?;
    }
    Ok(())
}

// Unresolved compact systems retry the normalized physical columns.
// This exceptional O(nm²) path retains the previous range-safe BFGS model;
// routine model products never rebuild dimension-wide columns.
fn physical_model_product<S: RealScalar>(
    storage: &mut LbfgsBStorage<S>,
    execution: scioxide_core::ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let first = &mut storage.first;
    let n = first.dimension;
    let m = first.history_len;
    if n == 0 {
        return Ok(());
    }
    let one = scioxide_core::Scaled::from_value(S::ONE)?;
    let theta = if m == 0 {
        one
    } else {
        let slot = (first.history_start + m - 1) % first.history_capacity;
        let range = slot * n..(slot + 1) * n;
        scaled_dot(&first.history_y[range.clone()], &first.history_y[range])?
            .div(first.history_curvature[slot])?
    };
    for j in 0..m {
        execution.check()?;
        let slot = (first.history_start + j) % first.history_capacity;
        let range = slot * n..(slot + 1) * n;
        let s = &first.history_s[range.clone()];
        let y = &first.history_y[range];
        let (previous, next) = storage.physical_basis.split_at_mut(2 * j * n);
        let (bs, next) = next.split_at_mut(n);
        bs.fill(S::ZERO);
        super::lbfgs::axpy(bs, theta, s)?;
        for (column, basis) in previous.chunks_exact(n).enumerate() {
            let coefficient = scaled_dot(basis, s)?;
            super::lbfgs::axpy(
                bs,
                if column % 2 == 0 {
                    -coefficient
                } else {
                    coefficient
                },
                basis,
            )?;
        }
        let curvature = scaled_dot(s, bs)?;
        if !crate::quasi_newton::positive(curvature) {
            return Err(breakdown("physical Hessian lost positive curvature"));
        }
        normalize_column(bs, curvature.sqrt()?)?;
        next[..n].copy_from_slice(y);
        normalize_column(&mut next[..n], first.history_curvature[slot].sqrt()?)?;
    }
    first.r.fill(S::ZERO);
    super::lbfgs::axpy(&mut first.r, theta, &first.s)?;
    for (column, basis) in storage.physical_basis[..2 * m * n]
        .chunks_exact(n)
        .enumerate()
    {
        let coefficient = scaled_dot(basis, &first.s)?;
        super::lbfgs::axpy(
            &mut first.r,
            if column % 2 == 0 {
                -coefficient
            } else {
                coefficient
            },
            basis,
        )?;
    }
    Ok(())
}

fn prepare_projected_correction<S: RealScalar>(
    storage: &mut LbfgsBStorage<S>,
    policy: CurvaturePolicy<S>,
    execution: scioxide_core::ExecutionContext<'_>,
) -> Result<HessianUpdateReport, SciOxideError> {
    let scalar = projected_model_curvature(storage, execution);
    if let Ok((curvature, uncertainty)) = scalar {
        let sy = scaled_dot(&storage.first.s, &storage.first.y)?;
        let error = scioxide_core::Scaled::from_value(uncertainty)?;
        let bound = curvature.mul(scioxide_core::Scaled::from_value(policy.threshold())?)?;
        if crate::quasi_newton::positive(curvature)
            && sy.cmp_value(bound.add(error)?) == core::cmp::Ordering::Greater
        {
            let cutoff = scaled_dot(&storage.first.s, &storage.first.s)?
                .mul(scaled_dot(&storage.first.y, &storage.first.y)?)?
                .sqrt()?
                .mul(scioxide_core::Scaled::from_value(S::EPSILON)?)?;
            return Ok(if crate::quasi_newton::positive(sy) && !sy.abs_le(cutoff) {
                HessianUpdateReport::Updated
            } else {
                HessianUpdateReport::Skipped(crate::UpdateSkipReason::Curvature)
            });
        }
        if policy.action() == crate::CurvatureAction::Skip
            && crate::quasi_newton::positive(curvature)
            && sy.cmp_value(bound.sub(error)?) == core::cmp::Ordering::Less
        {
            return Ok(HessianUpdateReport::Skipped(
                crate::UpdateSkipReason::Curvature,
            ));
        }
    } else if let Err(error) = scalar
        && !matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        )
    {
        return Err(error);
    }
    let result = physical_model_product(storage, execution).and_then(|()| {
        crate::quasi_newton::damp_pair(
            &storage.first.s,
            &mut storage.first.y,
            &storage.first.r,
            policy,
        )
    });
    match result {
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ) =>
        {
            Ok(HessianUpdateReport::Skipped(
                crate::UpdateSkipReason::Unrepresentable,
            ))
        }
        result => result,
    }
}

fn projected_direction<S: RealScalar>(
    first: &mut FirstOrderStorage<S>,
    bounds: &Bounds<S>,
) -> Result<S, SciOxideError> {
    for ((q, (&x, &g)), &bound) in first
        .q
        .iter_mut()
        .zip(first.x.iter().zip(&first.gradient))
        .zip(bounds.as_slice())
    {
        *q = if blocked(x, g, bound) { S::ZERO } else { g };
    }
    match super::lbfgs::two_loop_recursion(first) {
        Ok(()) => first
            .direction
            .iter_mut()
            .zip(&first.r)
            .for_each(|(d, &r)| *d = -r),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ) =>
        {
            first.history_len = 0;
            first.history_start = 0;
            first
                .direction
                .iter_mut()
                .zip(&first.gradient)
                .for_each(|(d, &g)| *d = -g);
        }
        Err(error) => return Err(error),
    }
    for retry in [false, true] {
        if retry {
            first.history_len = 0;
            first.history_start = 0;
            first
                .direction
                .iter_mut()
                .zip(&first.gradient)
                .for_each(|(d, &g)| *d = -g);
        }
        for (((d, &x), &g), &bound) in first
            .direction
            .iter_mut()
            .zip(&first.x)
            .zip(&first.gradient)
            .zip(bounds.as_slice())
        {
            // Apply the free-coordinate projector on both sides of the inverse
            // model: coupling must not move gradient-blocked coordinates inward.
            if blocked(x, g, bound) || blocked(x, -*d, bound) {
                *d = S::ZERO;
            }
        }
        if !scaled_dot(&first.gradient, &first.direction)?.is_sign_negative() {
            continue;
        }
        crate::quasi_newton::descent_slope(&first.gradient, &mut first.direction, false)?;
        for ((d, &x), &bound) in first
            .direction
            .iter_mut()
            .zip(&first.x)
            .zip(bounds.as_slice())
        {
            *d = feasible_component(x, *d, bound)?;
        }
        let slope = dot(&first.gradient, &first.direction)?;
        if slope < S::ZERO {
            return Ok(slope);
        }
    }
    Ok(S::ZERO)
}

#[allow(clippy::too_many_arguments)]
fn bounded_gradient<S, F>(
    objective: &F,
    bounds: &Bounds<S>,
    x: &[S],
    value: S,
    strategy: DerivativeStrategy,
    output: &mut [S],
    finite_difference: &mut crate::DerivativeWorkspace<S>,
    execution: scioxide_core::ExecutionContext<'_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    let _ = finite_difference.acquire_into(
        strategy,
        x,
        Some(bounds),
        Some(&[value]),
        |analytic, out| {
            if analytic {
                objective.gradient_into(x, out)
            } else {
                objective.automatic_gradient_into(x, out)
            }
        },
        |point, out| {
            out[0] = objective.eval(point)?;
            Ok(())
        },
        output,
        execution,
    )?;
    Ok(())
}

fn bound_violation<S: RealScalar>(x: &[S], bounds: &Bounds<S>) -> S {
    crate::geometry::bound_violation(x, bounds.as_slice())
}

#[allow(clippy::too_many_arguments)]
fn finish<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut LbfgsBStorage<S>,
    output: &mut OptimizeResult<S>,
    bounds: &Bounds<S>,
    value: S,
    iterations: usize,
    termination: OptimizeTermination,
    message: &'static str,
) -> Result<(), SciOxideError> {
    finish_first_order(
        &mut workspace.first,
        output,
        value,
        true,
        iterations,
        termination,
        message,
    )?;
    let (inf, l2) = projected_gradient(&workspace.first.x, &workspace.first.gradient, bounds)?;
    workspace.projected = inf;
    output.diagnostics.stationarity = Some(l2);
    output.diagnostics.bound_violation = Some(bound_violation(&workspace.first.x, bounds));
    output.diagnostics.primal_feasibility = output.diagnostics.bound_violation;
    Ok(())
}

/// Minimizes a smooth function subject to bounds `l ≤ x ≤ u` by a projected limited-memory BFGS method.
///
/// The bounds come from the [`BoundedObjective`](crate::BoundedObjective);
/// either side of a bound may be absent. Coordinates at a bound where the
/// negative gradient points out of the box are held fixed. On the remaining free coordinates `F`
/// the search direction is `−P_F H P_F g`, where `H` is the L-BFGS
/// inverse-Hessian approximation and `P_F` the projection onto `F`. The method
/// then searches the projected path back into the box with an Armijo
/// backtracking line search. If the direction fails, the method discards its
/// history and takes a projected steepest-descent step. It converges when the
/// infinity norm of the projected gradient, whose components at active bounds
/// are zero, passes the gradient threshold of [Termination](crate#termination);
/// the reported stationarity is the Euclidean norm of the same vector.
///
/// An iteration costs O(nm) arithmetic plus O(m³) for the curvature test,
/// where `m` is the memory, and the workspace holds O(nm + m²) values. The
/// method does not compute the generalized Cauchy point of Byrd, Lu, Nocedal
/// and Zhu.
///
/// # Examples
///
/// ```
/// use scioxide_core::Bounds;
/// use scioxide_optimize::minimize_bounded;
/// let bounds = Bounds::new(vec![(Some(0.0), Some(1.0)); 2])?;
/// let result = minimize_bounded(
///     |x: &[f64]| Ok((x[0] - 2.0).powi(2) + x[1] * x[1]),
///     &[0.25, 0.5], &bounds,
/// )?;
/// assert!((result.x()[0] - 1.0).abs() < 1e-6);
/// assert!(result.x()[1].abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - D. P. Bertsekas, "Projected Newton methods for optimization problems
///   with simple constraints", *SIAM J. Control Optim.* 20(2), 221–246, 1982.
/// - D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large
///   scale optimization", *Math. Program.* 45, 503–528, 1989.
#[derive(Debug, Clone, Copy)]
pub struct LbfgsB<S: RealScalar + NumericScalar<Real = S>> {
    options: LbfgsBOptions<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for LbfgsB<S> {
    fn default() -> Self {
        Self::new(LbfgsBOptions::default())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> LbfgsB<S> {
    /// Creates the optimizer with the given options.
    pub const fn new(options: LbfgsBOptions<S>) -> Self {
        Self { options }
    }
    /// Returns the options.
    pub const fn options(&self) -> &LbfgsBOptions<S> {
        &self.options
    }

    /// Returns the workspace requirements for `dimension` variables.
    ///
    /// # Errors
    ///
    /// This method does not fail.
    pub const fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<LbfgsBWorkspaceRequirements, SciOxideError> {
        Ok(LbfgsBWorkspaceRequirements {
            dimension,
            history: self.options.memory.get(),
        })
    }

    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// reserved.
    pub fn workspace(&self, dimension: usize) -> Result<LbfgsBWorkspace<S>, SciOxideError> {
        let mut workspace = LbfgsBWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Minimizes `problem` from `x0` and writes the result to `output`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent [`SolveContext`](scioxide_core::SolveContext). When a limit,
    /// cancellation or deadline stops the solve after it has accepted a point,
    /// the call returns `Ok` with an unsuccessful termination. On error,
    /// `output` is unchanged and [`LbfgsBWorkspace::last_attempt`] describes the
    /// failed call.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x0.len()` differs from `problem.dim()`.
    /// - `InvalidInput` if `x0` lies outside the bounds.
    /// - `NonFinite` if `x0`, or the objective or gradient at `x0`, is NaN
    ///   or infinite.
    /// - `CapabilityUnavailable` if a time limit is set and `scope` has no clock.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation
    ///   or deadline stops the solve before it accepts a point.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the
    ///   workspace cannot grow.
    ///
    /// Errors returned by the objective are returned unchanged.
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut LbfgsBWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        validate_start(problem, problem.bounds, x0)?;
        workspace.try_reserve(&self.workspace_requirements(x0.len())?)?;
        crate::attempt::preflight(&scope, self.options.time_limit)?;
        let entry = Entry::Call {
            identity: self.identity(scioxide_core::ProblemToken::new(0), problem.bounds),
            budgets: super::settings::iteration_budgets(self.options.iterations),
        };
        crate::attempt::minimize(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| {
                self.minimize_in_context(problem, x0, staged, storage, context)
            },
        )
    }

    /// Identify method, objective, box, and immutable options for a solve epoch.
    pub(crate) fn identity(
        &self,
        problem: scioxide_core::ProblemToken,
        bounds: &Bounds<S>,
    ) -> scioxide_core::SolveIdentity {
        scioxide_core::SolveIdentity::new(
            problem,
            scioxide_core::algorithm_id!("optimize", "L-BFGS-B", 1),
            bounds.len() as u128,
            crate::context::options_token(&(self.options, bounds.as_slice())),
        )
    }

    /// Run one epoch in `context` and write projected-gradient diagnostics.
    pub(crate) fn minimize_in_context(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut LbfgsBStorage<S>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if let Some(limit) = self.options.time_limit {
            context.limit_time(limit)?;
        }
        let session = crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
        let observed = session.observe(problem);
        let outcome =
            self.minimize_kernel(&observed, problem.bounds, x0, output, workspace, &session);
        let completed =
            crate::utils::complete_first_order(outcome, &mut workspace.first, output, &session);
        if completed.is_ok() && workspace.first.accepted_gradient {
            let (inf, l2) = projected_gradient(
                &workspace.first.x,
                &workspace.first.gradient,
                problem.bounds,
            )?;
            workspace.projected = inf;
            output.diagnostics.stationarity = Some(l2);
            output.diagnostics.bound_violation =
                Some(bound_violation(&workspace.first.x, problem.bounds));
            output.diagnostics.primal_feasibility = output.diagnostics.bound_violation;
        }
        completed
    }

    #[allow(clippy::too_many_arguments)]
    fn minimize_kernel<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        objective: &F,
        bounds: &Bounds<S>,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut LbfgsBStorage<S>,
        session: &crate::context::SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        validate_start(objective, bounds, x0)?;
        if !objective.is_in_domain(x0)? {
            return Err(invalid(
                scioxide_core::argument!("x0"),
                "initial point is outside objective domain",
            ));
        }
        workspace.try_reserve(x0.len(), self.options.memory.get())?;
        output.try_reserve(x0.len())?;
        workspace.first.reset();
        workspace.first.x.copy_from_slice(x0);

        let mut value = objective.eval(&workspace.first.x)?;
        if !value.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "objective returned a non-finite value",
            ));
        }
        workspace.first.accepted_value = Some(value);
        bounded_gradient(
            objective,
            bounds,
            &workspace.first.x,
            value,
            self.options.derivative,
            &mut workspace.first.gradient,
            crate::derivatives::prepared(&mut workspace.first.finite_difference)?,
            session.execution()?,
        )?;
        workspace.first.accepted_gradient = true;
        for iteration in 0..self.options.iterations.get() {
            if let Some(stop) = session.iteration()? {
                return finish(
                    workspace,
                    output,
                    bounds,
                    value,
                    iteration,
                    stop,
                    "execution limit reached",
                );
            }
            let (projected, _) =
                projected_gradient(&workspace.first.x, &workspace.first.gradient, bounds)?;
            workspace.projected = projected;
            if self.options.gradient.is_satisfied(
                projected,
                crate::utils::gradient_scale(value, &workspace.first.x),
            )? {
                return finish(
                    workspace,
                    output,
                    bounds,
                    value,
                    iteration,
                    OptimizeTermination::GradientSmall,
                    "projected gradient below tolerance",
                );
            }
            #[cfg(test)]
            let history_before = workspace.first.history_len;
            let slope = projected_direction(&mut workspace.first, bounds)?;
            #[cfg(test)]
            if let Some(trace) = &mut workspace.direction_trace {
                trace.push(DirectionObservation {
                    active: workspace
                        .first
                        .x
                        .iter()
                        .zip(&workspace.first.gradient)
                        .zip(bounds.as_slice())
                        .map(|((&x, &g), &b)| blocked(x, g, b))
                        .collect(),
                    history_before,
                    history_after: workspace.first.history_len,
                });
            }
            if slope >= S::ZERO {
                return finish(
                    workspace,
                    output,
                    bounds,
                    value,
                    iteration,
                    OptimizeTermination::Stagnation,
                    "no representable feasible descent direction",
                );
            }
            let trial = self.options.line_search.search_into(
                objective,
                &workspace.first.x,
                value,
                &workspace.first.direction,
                slope,
                self.options.derivative,
                &mut workspace.first.trial_x,
                &mut workspace.first.trial_gradient,
                &mut workspace.first.s,
                &mut workspace.first.y,
                crate::derivatives::prepared(&mut workspace.first.finite_difference)?,
                crate::LineSearchScope::observed(session),
                None,
            )?;
            if let Some(stop) = line_search_termination(trial) {
                return finish(
                    workspace,
                    output,
                    bounds,
                    value,
                    iteration,
                    stop,
                    "line search found no accepted trial",
                );
            }
            if let Some(stop) = session.termination()? {
                return finish(
                    workspace,
                    output,
                    bounds,
                    value,
                    iteration,
                    stop,
                    "execution limit reached during line search",
                );
            }
            bounded_gradient(
                objective,
                bounds,
                &workspace.first.trial_x,
                trial.value(),
                self.options.derivative,
                &mut workspace.first.trial_gradient,
                crate::derivatives::prepared(&mut workspace.first.finite_difference)?,
                session.execution()?,
            )?;
            for i in 0..x0.len() {
                workspace.first.s[i] = workspace.first.trial_x[i] - workspace.first.x[i];
                workspace.first.y[i] =
                    workspace.first.trial_gradient[i] - workspace.first.gradient[i];
            }
            let report = prepare_projected_correction(
                workspace,
                self.options.curvature,
                session.execution()?,
            )?;
            session.accept(scioxide_core::PartialResultTrust::Admissible)?;
            if matches!(
                report,
                HessianUpdateReport::Skipped(crate::UpdateSkipReason::Unrepresentable)
            ) {
                workspace.first.history_len = 0;
                workspace.first.history_start = 0;
            }
            if !matches!(report, HessianUpdateReport::Skipped(_)) {
                let curvature = scaled_dot(&workspace.first.s, &workspace.first.y)?;
                crate::unconstrained::lbfgs::push_history(&mut workspace.first, curvature);
            }
            workspace.first.globalization.record_update(report);
            core::mem::swap(&mut workspace.first.x, &mut workspace.first.trial_x);
            core::mem::swap(
                &mut workspace.first.gradient,
                &mut workspace.first.trial_gradient,
            );
            value = trial.value();
            workspace.first.accepted_value = Some(value);
            workspace.first.accepted_gradient = true;
            workspace.first.accepted_iterations = iteration + 1;
        }
        finish(
            workspace,
            output,
            bounds,
            value,
            self.options.iterations.get(),
            OptimizeTermination::IterationLimit,
            "maximum iterations reached",
        )
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> LbfgsB<S> {
    /// Minimizes the objective with this method's settings and owned workspace.
    ///
    /// Returns the complete result, including unsuccessful termination. Call
    /// `require_success()` to require the stopping test to pass.
    ///
    /// # Errors
    ///
    /// Returns the input, callback, allocation and numerical errors of `minimize_into`.
    pub fn minimize(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem.dim())?;
        let mut output = OptimizeResult::empty();
        self.minimize_into(
            problem,
            x0,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<'a, S: RealScalar + NumericScalar<Real = S>>
    crate::traits::Optimizer<S, BoundedObjective<'a, S>> for LbfgsB<S>
{
    fn minimize(
        &self,
        problem: &BoundedObjective<'a, S>,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        self.minimize(problem, x0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn coupled_active_direction<S: RealScalar>() {
        let cv = |x| S::try_from_f64_lossy(x).unwrap();
        let mut first = FirstOrderStorage::new();
        first.try_reserve(3, 2).unwrap();
        first.x.fill(S::ZERO);
        first.gradient.copy_from_slice(&[cv(10.), S::ONE, S::ONE]);
        // This feasible secant belongs to the SPD Hessian
        // [[2,1,0],[1,1,0],[0,0,1]]. Its inverse history couples coordinate 0
        // inward, making the unprojected slope positive despite positive s'y.
        first.s.copy_from_slice(&[-S::ONE, S::ONE, S::ZERO]);
        first.y.copy_from_slice(&[-S::ONE, S::ZERO, S::ZERO]);
        let curvature = scaled_dot(&first.s, &first.y).unwrap();
        super::super::lbfgs::push_history(&mut first, curvature);
        let bounds = Bounds::new(alloc::vec![
            (Some(S::ZERO), None),
            (None, None),
            (None, None)
        ])
        .unwrap();
        let slope = projected_direction(&mut first, &bounds).unwrap();
        assert!(slope < S::ZERO);
        assert_eq!(first.direction[0], S::ZERO);
        assert_eq!(first.history_len, 1);
        assert!(first.direction[1] < S::ZERO && first.direction[2] < S::ZERO);
    }

    #[test]
    fn coupled_active_direction_preserves_history() {
        coupled_active_direction::<f64>();
        coupled_active_direction::<f32>();
    }

    struct BoxQuadratic<S> {
        gram: Vec<S>,
        linear: Vec<S>,
    }
    impl<S: RealScalar> ObjectiveFunction<S> for BoxQuadratic<S> {
        fn dim(&self) -> usize {
            self.linear.len()
        }
        fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
            Ok(self.gram.chunks(x.len()).zip(x).zip(&self.linear).fold(
                S::ZERO,
                |sum, ((row, &x_i), &h)| {
                    let product = row.iter().zip(x).fold(S::ZERO, |sum, (&a, &x)| sum + a * x);
                    sum + x_i * (product / S::TWO - h)
                },
            ))
        }
        fn gradient_into(&self, x: &[S], out: &mut [S]) -> Result<bool, SciOxideError> {
            for ((g, row), &h) in out
                .iter_mut()
                .zip(self.gram.chunks(x.len()))
                .zip(&self.linear)
            {
                *g = row.iter().zip(x).fold(-h, |sum, (&a, &x)| sum + a * x);
            }
            Ok(true)
        }
    }

    fn seeded_box_quadratics<S: RealScalar>() {
        use scioxide_random::{BitGenerator, SeedableRng, SplitMix64, decode_u64};
        let mut rng = SplitMix64::from_seed(0x2394_2393);
        let mut uniform = || decode_u64::<f64>(rng.next_u64().unwrap());
        let cv = |x| S::try_from_f64_lossy(x).unwrap();
        for n in 4..=12 {
            for case in 0..4 {
                let a: Vec<_> = (0..(n + 3) * n).map(|_| 2. * uniform() - 1.).collect();
                let gram: Vec<_> = (0..n)
                    .flat_map(|i| {
                        let a = &a;
                        (0..n).map(move |j| {
                            cv(a.chunks(n).map(|row| row[i] * row[j]).sum::<f64>()
                                + if i == j { 0.1 } else { 0. })
                        })
                    })
                    .collect();
                let linear = (0..n).map(|_| cv(4. * uniform() - 2.)).collect();
                let objective = BoxQuadratic { gram, linear };
                let endpoints: Vec<_> = (0..n)
                    .map(|_| (Some(cv(-0.2 - uniform())), Some(cv(0.2 + uniform()))))
                    .collect();
                let initial: Vec<_> = endpoints
                    .iter()
                    .map(|&(lo, hi)| (lo.unwrap() + hi.unwrap()) / S::TWO)
                    .collect();
                let bounds = Bounds::new(endpoints).unwrap();
                let problem = BoundedObjective::new(&objective, &bounds).unwrap();
                for memory in [1, 10] {
                    let solver = LbfgsB::new(
                        LbfgsBOptions::default().with_memory(Budget::try_from(memory).unwrap()),
                    );
                    let mut workspace = solver.workspace(n).unwrap();
                    workspace.storage.direction_trace = Some(Vec::new());
                    let mut result = OptimizeResult::empty();
                    solver
                        .minimize_into(
                            &problem,
                            &initial,
                            &mut result,
                            &mut workspace,
                            scioxide_core::ExecutionContext::sequential(),
                        )
                        .unwrap();
                    assert_eq!(
                        result.termination(),
                        OptimizeTermination::GradientSmall,
                        "n {n}, case {case}, memory {memory}: {result:?}"
                    );
                    let trace = workspace.storage.direction_trace.as_ref().unwrap();
                    let last = trace.last().unwrap();
                    for observation in trace.iter().rev().take_while(|o| o.active == last.active) {
                        assert_eq!(
                            observation.history_before, observation.history_after,
                            "settled active set lost history: n {n}, case {case}, memory {memory}: {trace:?}"
                        );
                    }
                    // Recompute stationarity in f64 from the returned point,
                    // independently of the solver's projected-gradient routine.
                    let x: Vec<_> = result
                        .x()
                        .iter()
                        .map(|v| v.try_to_f64_exact().unwrap())
                        .collect();
                    let projected = objective
                        .gram
                        .chunks(n)
                        .zip(&objective.linear)
                        .zip(bounds.as_slice())
                        .zip(&x)
                        .map(|(((row, h), &(lo, hi)), &value)| {
                            let lo = lo.unwrap().try_to_f64_exact().unwrap();
                            let hi = hi.unwrap().try_to_f64_exact().unwrap();
                            assert!((lo..=hi).contains(&value));
                            let gradient = row
                                .iter()
                                .zip(&x)
                                .map(|(a, x)| a.try_to_f64_exact().unwrap() * x)
                                .sum::<f64>()
                                - h.try_to_f64_exact().unwrap();
                            ((value - gradient).clamp(lo, hi) - value).abs()
                        })
                        .fold(0., f64::max);
                    let coordinate = x.iter().map(|v| v.abs()).fold(1., f64::max);
                    let value = objective
                        .eval(result.x())
                        .unwrap()
                        .try_to_f64_exact()
                        .unwrap();
                    let tolerance = 16.
                        * S::EPSILON.try_to_f64_exact().unwrap().sqrt()
                        * (1. + (value.abs() / coordinate).max(1.));
                    assert!(
                        projected <= tolerance,
                        "projected gradient {projected}: {result:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn seeded_box_quadratics_retain_settled_history_f64() {
        seeded_box_quadratics::<f64>();
    }
    #[test]
    fn seeded_box_quadratics_retain_settled_history_f32() {
        seeded_box_quadratics::<f32>();
    }

    fn model_curvature<S: RealScalar>() {
        let cv = |x| S::try_from_f64_lossy(x).unwrap();
        let mut storage = LbfgsBStorage::default();
        storage.try_reserve(5, 2).unwrap();
        let pairs = [
            ([1., 0.3, -0.2, 0.1, 0.2], [2., 0.1, -0.3, 0.2, 0.4]),
            ([0.2, 1., 0.4, -0.1, 0.3], [0.7, 2.4, 0.9, -0.2, 0.6]),
        ];
        for (s, y) in pairs {
            storage.first.s.copy_from_slice(&s.map(cv));
            storage.first.y.copy_from_slice(&y.map(cv));
            let sy = scaled_dot(&storage.first.s, &storage.first.y).unwrap();
            super::super::lbfgs::push_history(&mut storage.first, sy);
        }
        let theta = storage.first.history_yy[3] / storage.first.history_sy[3];
        let mut dense = [[S::ZERO; 5]; 5];
        for (i, row) in dense.iter_mut().enumerate() {
            row[i] = theta;
        }
        for (s, y) in pairs {
            let s = s.map(cv);
            let y = y.map(cv);
            let mut bs = [S::ZERO; 5];
            for (out, row) in bs.iter_mut().zip(&dense) {
                *out = row.iter().zip(s).fold(S::ZERO, |sum, (&a, b)| sum + a * b);
            }
            let sb = scaled_dot(&s, &bs).unwrap().round().unwrap();
            let sy = scaled_dot(&s, &y).unwrap().round().unwrap();
            for (i, row) in dense.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    *value = *value - bs[i] * bs[j] / sb + y[i] * y[j] / sy;
                }
            }
        }
        let step = [cv(0.5), -S::ONE, cv(2.), cv(0.3), cv(-0.7)];
        storage.first.s.copy_from_slice(&step);
        let mut expected = [S::ZERO; 5];
        for (out, row) in expected.iter_mut().zip(&dense) {
            *out = row
                .iter()
                .zip(step)
                .fold(S::ZERO, |sum, (&a, b)| sum + a * b);
        }
        let exact = scaled_dot(&step, &expected).unwrap().round().unwrap();
        let (curvature, _) =
            projected_model_curvature(&mut storage, scioxide_core::ExecutionContext::sequential())
                .unwrap();
        assert!((curvature.round().unwrap() - exact).abs() <= cv(1024.) * S::EPSILON * exact.abs());
        physical_model_product(&mut storage, scioxide_core::ExecutionContext::sequential())
            .unwrap();
        for (&a, &b) in storage.first.r.iter().zip(&expected) {
            assert!((a - b).abs() <= cv(1024.) * S::EPSILON * (S::ONE + b.abs()));
        }
        storage.first.y.copy_from_slice(&expected);
        assert_eq!(
            prepare_projected_correction(
                &mut storage,
                CurvaturePolicy::default(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap(),
            HessianUpdateReport::Updated
        );
        storage.first.y.iter_mut().for_each(|y| *y = -*y);
        storage.first.history_cache_valid = false;
        assert_eq!(
            prepare_projected_correction(
                &mut storage,
                CurvaturePolicy::default(),
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap(),
            HessianUpdateReport::Damped
        );
        assert_eq!(storage.first.s, step);
        assert!(crate::quasi_newton::positive(
            scaled_dot(&storage.first.s, &storage.first.y).unwrap()
        ));
    }

    #[test]
    fn coefficient_curvature_and_physical_retry_match_dense_bfgs() {
        model_curvature::<f32>();
        model_curvature::<f64>();
    }

    fn directions<S: RealScalar>() {
        let cv = |x| S::try_from_f64_lossy(x).unwrap();
        let mut first = FirstOrderStorage::new();
        first.try_reserve(3, 2).unwrap();
        first.x.copy_from_slice(&[S::ZERO, S::ONE, cv(2.0)]);
        first.gradient.copy_from_slice(&[S::ONE, -S::ONE, cv(3.0)]);
        let bounds = Bounds::new(alloc::vec![
            (Some(S::ZERO), None),
            (None, Some(S::ONE)),
            (Some(cv(2.0)), Some(cv(2.0)))
        ])
        .unwrap();
        assert_eq!(projected_direction(&mut first, &bounds).unwrap(), S::ZERO);
        first.gradient.copy_from_slice(&[-S::ONE, S::ONE, cv(3.0)]);
        assert!(projected_direction(&mut first, &bounds).unwrap() < S::ZERO);
        assert!(first.direction[0] > S::ZERO && first.direction[1] < S::ZERO);
        assert_eq!(first.direction[2], S::ZERO);
        assert_eq!(
            feasible_component(cv(0.9), cv(2.0), (None, Some(S::ONE))).unwrap(),
            S::ONE - cv(0.9)
        );
        assert_eq!(
            feasible_component(S::MAX, S::MAX, (None, Some(S::MAX))).unwrap(),
            S::ZERO
        );
        let x = -S::ONE.scalbn(S::MANTISSA_DIGITS as i32);
        let step = feasible_component(x, -S::TWO * x, (None, Some(cv(3.0)))).unwrap();
        let endpoint = crate::utils::trial_component(x, S::ONE, step).unwrap();
        assert!(endpoint <= cv(3.0) && endpoint >= x);
    }

    #[test]
    fn projected_directions_and_overflowing_endpoints() {
        directions::<f32>();
        directions::<f64>();
    }
}
