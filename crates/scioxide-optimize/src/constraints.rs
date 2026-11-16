//! Nonlinear equality, inequality and bound constraints, and their evaluation storage.

use alloc::boxed::Box;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

use core::fmt;

use crate::DerivativeWorkspace;
use scioxide_core::backend::{try_filled, try_reserve};
use scioxide_core::{Bounds, NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Matrix;

use crate::traits::ConstraintFunction;

/// Counts the rows of a [`Constraints`] collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstraintProfile {
    equality_count: usize,
    inequality_count: usize,
    has_bounds: bool,
}

impl ConstraintProfile {
    pub(crate) const EMPTY: Self = Self {
        equality_count: 0,
        inequality_count: 0,
        has_bounds: false,
    };

    /// Returns the number of equality rows.
    #[must_use]
    pub const fn equality_count(self) -> usize {
        self.equality_count
    }

    /// Returns the number of inequality rows. Each finite side of a bound counts
    /// as one row.
    #[must_use]
    pub const fn inequality_count(self) -> usize {
        self.inequality_count
    }

    /// Returns `true` if the collection contains at least one set of bounds.
    #[must_use]
    pub const fn has_bounds(self) -> bool {
        self.has_bounds
    }

    pub(crate) const fn total_count(self) -> usize {
        self.equality_count + self.inequality_count
    }

    pub(crate) const fn from_counts(equality_count: usize, inequality_count: usize) -> Self {
        Self {
            equality_count,
            inequality_count,
            has_bounds: false,
        }
    }
}

/// Holds the values of the equality rows `h(x)` and inequality rows `g(x)` of a
/// [`Constraints`] collection at one point.
///
/// A point is feasible when `h(x) = 0` and `g(x) ≤ 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintValues<S: RealScalar> {
    equalities: Vec<S>,
    inequalities: Vec<S>,
}

impl<S: RealScalar> ConstraintValues<S> {
    /// Returns the equality values `h(x)` in the order the rows were added.
    #[must_use]
    pub fn equalities(&self) -> &[S] {
        &self.equalities
    }

    /// Returns the inequality values `g(x)` in the order the rows were added.
    ///
    /// A set of bounds contributes `lᵢ − xᵢ` and then `xᵢ − uᵢ` for each
    /// variable `i`, skipping infinite sides.
    #[must_use]
    pub fn inequalities(&self) -> &[S] {
        &self.inequalities
    }
}

/// Counts the callback work of one [`Constraints::jacobian_into`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[must_use]
pub struct ConstraintJacobianStats {
    constraint_evaluations: usize,
    jacobian_evaluations: usize,
}

impl ConstraintJacobianStats {
    /// Returns the number of constraint value calls made for finite differences.
    pub const fn constraint_evaluations(self) -> usize {
        self.constraint_evaluations
    }

    /// Returns the number of nonlinear Jacobian rows computed.
    pub const fn jacobian_evaluations(self) -> usize {
        self.jacobian_evaluations
    }
}

enum ConstraintEntry<S: RealScalar + NumericScalar<Real = S>> {
    Equality(Box<dyn ConstraintFunction<S> + Send + Sync>),
    Inequality(Box<dyn ConstraintFunction<S> + Send + Sync>),
    Bounds(Bounds<S>),
}

/// Stores the constraints of a nonlinear program: equalities, inequalities and bounds.
///
/// A point `x` is feasible when every equality row satisfies `h(x) = 0`, every
/// inequality row satisfies `g(x) ≤ 0`, and `x` lies in every set of bounds.
/// Build a collection with [`Constraints::new`] and the consuming builders
/// [`equality`](Constraints::equality), [`inequality`](Constraints::inequality)
/// and [`bounds`](Constraints::bounds). A finished collection is immutable and
/// can be shared between threads.
///
/// All sets of bounds are intersected into one box. [`Sqp`](crate::Sqp),
/// [`InteriorPoint`](crate::InteriorPoint), [`Cobyla`](crate::Cobyla) and
/// [`Cobyqa`](crate::Cobyqa) evaluate the problem only inside that box.
/// [`PenaltyMethod`](crate::PenaltyMethod) and
/// [`AugmentedLagrangian`](crate::AugmentedLagrangian) treat each finite bound
/// side as an inequality row, so their iterates may leave the box.
///
/// # Examples
///
/// ```
/// use scioxide_core::Bounds;
/// use scioxide_optimize::{ConstraintFn, Constraints};
///
/// let constraints = Constraints::new(2)?
///     .equality(ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0)))?
///     .bounds(Bounds::new(vec![(Some(0.0), None), (None, Some(1.0))])?)?;
/// assert_eq!(constraints.profile().inequality_count(), 2);
/// // A second box whose intersection with the first is empty is rejected.
/// assert!(constraints.bounds(Bounds::new(vec![(None, Some(-1.0)), (None, None)])?).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct Constraints<S: RealScalar + NumericScalar<Real = S>> {
    dimension: usize,
    entries: Vec<ConstraintEntry<S>>,
    profile: ConstraintProfile,
    hard_box: Vec<(Option<S>, Option<S>)>,
}

impl<S> fmt::Debug for Constraints<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Constraints")
            .field("dimension", &self.dimension)
            .field("entry_count", &self.entries.len())
            .field("profile", &self.profile)
            .finish()
    }
}

impl<S> Constraints<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Creates an empty collection for points of dimension `dimension`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `dimension` is zero.
    pub fn new(dimension: usize) -> Result<Self, SciOxideError> {
        if dimension == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("dimension"),
                scioxide_core::InvalidInputReason::Empty,
                "constraint dimension must be positive",
            ));
        }
        Ok(Self {
            dimension,
            entries: Vec::new(),
            profile: ConstraintProfile::EMPTY,
            hard_box: try_filled(
                dimension,
                (None, None),
                scioxide_core::quantity!("constraint_box"),
            )?,
        })
    }

    /// Returns the dimension of the points.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the row counts.
    #[must_use]
    pub const fn profile(&self) -> ConstraintProfile {
        self.profile
    }

    /// Returns `true` if no constraints have been added.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Adds the equality row `h(x) = 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `function` has a different dimension.
    pub fn equality<F>(mut self, function: F) -> Result<Self, SciOxideError>
    where
        F: ConstraintFunction<S> + Send + Sync + 'static,
    {
        self.validate_function_dimension(&function)?;
        let equality_count = self.profile.equality_count.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("equality_constraint_count"))
        })?;
        try_reserve(
            &mut self.entries,
            1,
            scioxide_core::quantity!("constraint_collection"),
        )?;
        self.entries
            .push(ConstraintEntry::Equality(Box::new(function)));
        self.profile.equality_count = equality_count;
        Ok(self)
    }

    /// Adds the inequality row `g(x) ≤ 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `function` has a different dimension.
    pub fn inequality<F>(mut self, function: F) -> Result<Self, SciOxideError>
    where
        F: ConstraintFunction<S> + Send + Sync + 'static,
    {
        self.validate_function_dimension(&function)?;
        let inequality_count = self
            .profile
            .inequality_count
            .checked_add(1)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "inequality_constraint_count"
                ))
            })?;
        try_reserve(
            &mut self.entries,
            1,
            scioxide_core::quantity!("constraint_collection"),
        )?;
        self.entries
            .push(ConstraintEntry::Inequality(Box::new(function)));
        self.profile.inequality_count = inequality_count;
        Ok(self)
    }

    /// Adds the bounds `lᵢ ≤ xᵢ ≤ uᵢ` and intersects them with earlier bounds.
    ///
    /// Each finite side adds one inequality row.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `bounds` has a different dimension.
    /// - `InvalidInput` if the intersection with earlier bounds is empty.
    pub fn bounds(mut self, bounds: Bounds<S>) -> Result<Self, SciOxideError> {
        if bounds.len() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dimension),
                scioxide_core::Dimensions::one(bounds.len()),
            ));
        }
        let disjoint =
            self.hard_box
                .iter()
                .zip(bounds.as_slice())
                .any(|(&(lo, hi), &(lower, upper))| {
                    let lo = tighten(lo, lower, true);
                    let hi = tighten(hi, upper, false);
                    matches!((lo, hi), (Some(lo), Some(hi)) if lo > hi)
                });
        if disjoint {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("bounds"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "bound intersection is empty",
            ));
        }
        let side_count = bounds
            .as_slice()
            .iter()
            .try_fold(0_usize, |count, (lower, upper)| {
                count
                    .checked_add(usize::from(lower.is_some()) + usize::from(upper.is_some()))
                    .ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!(
                            "bound_constraint_count"
                        ))
                    })
            })?;
        let inequality_count = self
            .profile
            .inequality_count
            .checked_add(side_count)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "inequality_constraint_count"
                ))
            })?;
        try_reserve(
            &mut self.entries,
            1,
            scioxide_core::quantity!("constraint_collection"),
        )?;
        for (slot, &(lower, upper)) in self.hard_box.iter_mut().zip(bounds.as_slice()) {
            *slot = (tighten(slot.0, lower, true), tighten(slot.1, upper, false));
        }
        self.entries.push(ConstraintEntry::Bounds(bounds));
        self.profile.inequality_count = inequality_count;
        self.profile.has_bounds = true;
        Ok(self)
    }

    /// Allocates a workspace for [`evaluate_into`](Self::evaluate_into) and
    /// [`jacobian_into`](Self::jacobian_into).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self) -> Result<ConstraintsWorkspace<S>, SciOxideError> {
        ConstraintsWorkspace::new(self)
    }

    /// Evaluates every row at `x` and returns the values.
    ///
    /// # Errors
    ///
    /// The same as [`evaluate_into`](Self::evaluate_into).
    pub fn evaluate(&self, x: &[S]) -> Result<ConstraintValues<S>, SciOxideError> {
        let mut workspace = self.workspace()?;
        let mut equalities = try_filled(
            self.profile.equality_count,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("constraint_equality_values"),
        )?;
        let mut inequalities = try_filled(
            self.profile.inequality_count,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("constraint_inequality_values"),
        )?;
        self.evaluate_into(x, &mut equalities, &mut inequalities, &mut workspace)?;
        Ok(ConstraintValues {
            equalities,
            inequalities,
        })
    }

    /// Evaluates every row at `x` into `equalities` and `inequalities`.
    ///
    /// The slices must have the lengths given by [`profile`](Self::profile).
    /// Rows are ordered as in [`ConstraintValues`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x`, an output slice or `workspace` does not
    ///   match the collection.
    /// - `NonFinite` if `x` or a constraint value is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a bound row `lᵢ − xᵢ` or
    ///   `xᵢ − uᵢ` overflows.
    /// - Errors returned by a constraint callback, unchanged.
    pub fn evaluate_into(
        &self,
        x: &[S],
        equalities: &mut [S],
        inequalities: &mut [S],
        workspace: &mut ConstraintsWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.evaluate_observed(x, equalities, inequalities, workspace, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_observed(
        &self,
        x: &[S],
        equalities: &mut [S],
        inequalities: &mut [S],
        workspace: &mut ConstraintsWorkspace<S>,
        observer: Option<&dyn crate::context::Observation>,
    ) -> Result<(), SciOxideError> {
        self.validate_evaluation_preflight(x, equalities, inequalities, workspace)?;
        self.evaluate_working_core(
            x,
            &mut workspace.equalities,
            &mut workspace.inequalities,
            observer,
            true,
        )?;
        equalities.copy_from_slice(&workspace.equalities);
        inequalities.copy_from_slice(&workspace.inequalities);
        Ok(())
    }

    /// Evaluate a solver trial; `false` reports a non-finite value, which
    /// rejects the trial, and leaves the caller outputs unchanged.
    pub(crate) fn evaluate_trial_observed(
        &self,
        x: &[S],
        equalities: &mut [S],
        inequalities: &mut [S],
        workspace: &mut ConstraintsWorkspace<S>,
        observer: Option<&dyn crate::context::Observation>,
    ) -> Result<bool, SciOxideError> {
        self.validate_evaluation_preflight(x, equalities, inequalities, workspace)?;
        self.evaluate_working_core(
            x,
            &mut workspace.equalities,
            &mut workspace.inequalities,
            observer,
            false,
        )?;
        if workspace
            .equalities
            .iter()
            .chain(&workspace.inequalities)
            .any(|value| !value.is_finite())
        {
            return Ok(false);
        }
        equalities.copy_from_slice(&workspace.equalities);
        inequalities.copy_from_slice(&workspace.inequalities);
        Ok(true)
    }

    fn evaluate_working_core(
        &self,
        x: &[S],
        equalities: &mut [S],
        inequalities: &mut [S],
        observer: Option<&dyn crate::context::Observation>,
        validate: bool,
    ) -> Result<(), SciOxideError> {
        let mut equality_index = 0;
        let mut inequality_index = 0;
        let evaluate = |function: &dyn ConstraintFunction<S>| {
            let value = crate::context::invoke(
                observer,
                scioxide_core::EvaluationRole::Constraint,
                || function.eval(x),
            )?;
            if validate {
                validate_callback_value(value)
            } else {
                Ok(value)
            }
        };
        for entry in &self.entries {
            match entry {
                ConstraintEntry::Equality(function) => {
                    equalities[equality_index] = evaluate(function.as_ref())?;
                    equality_index += 1;
                }
                ConstraintEntry::Inequality(function) => {
                    inequalities[inequality_index] = evaluate(function.as_ref())?;
                    inequality_index += 1;
                }
                ConstraintEntry::Bounds(bounds) => {
                    evaluate_bounds_into(bounds, x, inequalities, &mut inequality_index)?;
                }
            }
        }
        Ok(())
    }

    /// Computes the Jacobian of all rows at `x` into `output`.
    ///
    /// `output` has one row per constraint, equality rows first, and one column
    /// per variable. `equality_values` and `inequality_values` must be the
    /// values at `x`, as returned by [`evaluate_into`](Self::evaluate_into);
    /// finite differences reuse them as the base point. `strategy` selects how
    /// rows without an analytic Jacobian are differentiated. Bound rows are
    /// exact unit rows.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x`, the values, `output` or `workspace` does
    ///   not match the collection.
    /// - `NonFinite` if `x` or a supplied value is NaN or infinite.
    /// - Errors from a constraint callback or from differentiating it.
    pub fn jacobian_into(
        &self,
        x: &[S],
        equality_values: &[S],
        inequality_values: &[S],
        strategy: DerivativeStrategy,
        output: &mut Matrix<S>,
        workspace: &mut ConstraintsWorkspace<S>,
    ) -> Result<ConstraintJacobianStats, SciOxideError> {
        self.jacobian_observed(
            x,
            equality_values,
            inequality_values,
            strategy,
            output,
            workspace,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn jacobian_observed(
        &self,
        x: &[S],
        equality_values: &[S],
        inequality_values: &[S],
        strategy: DerivativeStrategy,
        output: &mut Matrix<S>,
        workspace: &mut ConstraintsWorkspace<S>,
        observer: Option<&dyn crate::context::Observation>,
    ) -> Result<ConstraintJacobianStats, SciOxideError> {
        self.validate_jacobian_preflight(x, equality_values, inequality_values, output, workspace)?;
        let dimension = workspace.dimension;
        let finite_difference = workspace.finite_difference.as_mut().ok_or_else(|| {
            SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(1, self.dimension),
                scioxide_core::Dimensions::two(0, dimension),
            )
        })?;
        let stats = self.jacobian_working_core(
            x,
            equality_values,
            inequality_values,
            strategy,
            &mut workspace.jacobian,
            &mut workspace.row,
            finite_difference,
            observer,
        )?;
        output
            .as_mut_slice()
            .copy_from_slice(workspace.jacobian.as_slice());
        Ok(stats)
    }

    #[allow(clippy::too_many_arguments)]
    fn jacobian_working_core(
        &self,
        x: &[S],
        equality_values: &[S],
        inequality_values: &[S],
        strategy: DerivativeStrategy,
        output: &mut Matrix<S>,
        row: &mut [S],
        finite_difference: &mut DerivativeWorkspace<S>,
        observer: Option<&dyn crate::context::Observation>,
    ) -> Result<ConstraintJacobianStats, SciOxideError> {
        output
            .as_mut_slice()
            .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
        let mut equality_index = 0;
        let mut inequality_index = 0;
        let mut stats = ConstraintJacobianStats::default();
        for entry in &self.entries {
            match entry {
                ConstraintEntry::Equality(function) => {
                    let row_index = equality_index;
                    let base = equality_values[equality_index];
                    equality_index += 1;
                    evaluate_jacobian_row(
                        function.as_ref(),
                        x,
                        base,
                        strategy,
                        row,
                        finite_difference,
                        &mut stats,
                        observer,
                    )?;
                    let offset = row_index * self.dimension;
                    output.as_mut_slice()[offset..offset + self.dimension].copy_from_slice(row);
                }
                ConstraintEntry::Inequality(function) => {
                    let row_index = self.profile.equality_count + inequality_index;
                    let base = inequality_values[inequality_index];
                    inequality_index += 1;
                    evaluate_jacobian_row(
                        function.as_ref(),
                        x,
                        base,
                        strategy,
                        row,
                        finite_difference,
                        &mut stats,
                        observer,
                    )?;
                    let offset = row_index * self.dimension;
                    output.as_mut_slice()[offset..offset + self.dimension].copy_from_slice(row);
                }
                ConstraintEntry::Bounds(bounds) => {
                    fill_bound_jacobian(
                        bounds,
                        self.dimension,
                        self.profile.equality_count,
                        &mut inequality_index,
                        output,
                    );
                }
            }
        }
        Ok(stats)
    }

    fn validate_function_dimension<F>(&self, function: &F) -> Result<(), SciOxideError>
    where
        F: ConstraintFunction<S>,
    {
        if function.dim() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dimension),
                scioxide_core::Dimensions::one(function.dim()),
            ));
        }
        Ok(())
    }

    fn validate_evaluation_preflight(
        &self,
        x: &[S],
        equalities: &[S],
        inequalities: &[S],
        workspace: &ConstraintsWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        validate_input(self.dimension, x)?;
        validate_value_shapes(self.profile, equalities, inequalities)?;
        workspace.validate(self.dimension, self.profile)?;
        validate_finite_input(x)
    }

    fn validate_jacobian_preflight(
        &self,
        x: &[S],
        equality_values: &[S],
        inequality_values: &[S],
        output: &Matrix<S>,
        workspace: &ConstraintsWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        validate_input(self.dimension, x)?;
        validate_value_shapes(self.profile, equality_values, inequality_values)?;
        let rows = self.profile.total_count();
        validate_matrix_shape(rows, self.dimension, output)?;
        workspace.validate(self.dimension, self.profile)?;
        validate_finite_input(x)?;
        validate_finite_values(equality_values, inequality_values)
    }

    /// Intersection of every bound set, one `(lower, upper)` pair per variable;
    /// `None` sides are unbounded. Nonempty by construction.
    pub(crate) fn hard_box(&self) -> &[(Option<S>, Option<S>)] {
        &self.hard_box
    }

    pub(crate) fn bound_sets(&self) -> impl Iterator<Item = &Bounds<S>> {
        self.entries.iter().filter_map(|entry| match entry {
            ConstraintEntry::Bounds(bounds) => Some(bounds),
            ConstraintEntry::Equality(_) | ConstraintEntry::Inequality(_) => None,
        })
    }

    /// Nonlinear rows only, in equality-then-inequality insertion order.
    pub(crate) fn nonlinear_functions(
        &self,
    ) -> impl Iterator<Item = &(dyn ConstraintFunction<S> + Send + Sync)> {
        self.entries
            .iter()
            .filter_map(|entry| match entry {
                ConstraintEntry::Equality(function) => Some(function.as_ref()),
                _ => None,
            })
            .chain(self.entries.iter().filter_map(|entry| match entry {
                ConstraintEntry::Inequality(function) => Some(function.as_ref()),
                _ => None,
            }))
    }
}

/// Tighten one optional side: the larger lower or the smaller upper endpoint.
fn tighten<S: RealScalar>(current: Option<S>, next: Option<S>, lower: bool) -> Option<S> {
    match (current, next) {
        (Some(a), Some(b)) => Some(if lower { a.max(b) } else { a.min(b) }),
        (a, b) => a.or(b),
    }
}

/// Stores scratch space for evaluating a [`Constraints`] collection.
#[derive(Debug)]
#[must_use]
pub struct ConstraintsWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    dimension: usize,
    equality_count: usize,
    inequality_count: usize,
    equalities: Vec<S>,
    inequalities: Vec<S>,
    row: Vec<S>,
    jacobian: Matrix<S>,
    finite_difference: Option<DerivativeWorkspace<S>>,
}

impl<S> ConstraintsWorkspace<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Allocates a workspace for `constraints`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn new(constraints: &Constraints<S>) -> Result<Self, SciOxideError> {
        let mut workspace = Self::empty();
        workspace.try_reserve(constraints.dimension, constraints.profile)?;
        Ok(workspace)
    }

    pub(crate) const fn empty() -> Self {
        Self {
            dimension: 0,
            equality_count: 0,
            inequality_count: 0,
            equalities: Vec::new(),
            inequalities: Vec::new(),
            row: Vec::new(),
            jacobian: Matrix::empty(),
            finite_difference: None,
        }
    }

    pub(crate) fn try_reserve(
        &mut self,
        dimension: usize,
        profile: ConstraintProfile,
    ) -> Result<(), SciOxideError> {
        resize(
            &mut self.equalities,
            profile.equality_count,
            scioxide_core::quantity!("constraint_equality_staging"),
        )?;
        resize(
            &mut self.inequalities,
            profile.inequality_count,
            scioxide_core::quantity!("constraint_inequality_staging"),
        )?;
        resize(
            &mut self.row,
            dimension,
            scioxide_core::quantity!("constraint_jacobian_row_staging"),
        )?;
        self.jacobian
            .try_resize_flat(profile.total_count(), dimension)?;
        if self
            .finite_difference
            .as_ref()
            .is_none_or(|workspace| workspace.input_dimension() != dimension)
        {
            self.finite_difference = Some(crate::DerivativeWorkspace::new(
                scioxide_differentiate::DerivativeOperation::Gradient,
                dimension,
                1,
            )?);
        }
        self.dimension = dimension;
        self.equality_count = profile.equality_count;
        self.inequality_count = profile.inequality_count;
        Ok(())
    }

    fn validate(&self, dimension: usize, profile: ConstraintProfile) -> Result<(), SciOxideError> {
        let compatible = self.dimension == dimension
            && self.equality_count == profile.equality_count
            && self.inequality_count == profile.inequality_count
            && self.equalities.len() == profile.equality_count
            && self.inequalities.len() == profile.inequality_count
            && self.row.len() == dimension
            && self.jacobian.rows() == profile.total_count()
            && self.jacobian.cols() == dimension
            && self
                .finite_difference
                .as_ref()
                .is_some_and(|workspace| workspace.input_dimension() == dimension);
        if !compatible {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(profile.total_count(), dimension),
                scioxide_core::Dimensions::two(
                    self.equality_count + self.inequality_count,
                    self.dimension,
                ),
            ));
        }
        Ok(())
    }
}

fn resize<S: RealScalar + NumericScalar<Real = S>>(
    values: &mut Vec<S>,
    len: usize,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    crate::storage::resize_with_quantity(
        values,
        len,
        <S as scioxide_core::AlgebraicScalar>::ZERO,
        quantity,
    )
}

use crate::validation::slice_shape as validate_input;

use crate::validation::finite_input as validate_finite_input;

fn validate_value_shapes<S>(
    profile: ConstraintProfile,
    equalities: &[S],
    inequalities: &[S],
) -> Result<(), SciOxideError> {
    if equalities.len() != profile.equality_count || inequalities.len() != profile.inequality_count
    {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(profile.equality_count, profile.inequality_count),
            scioxide_core::Dimensions::two(equalities.len(), inequalities.len()),
        ));
    }
    Ok(())
}

use crate::validation::matrix_shape as validate_matrix_shape;

fn validate_finite_values<S: RealScalar>(
    equalities: &[S],
    inequalities: &[S],
) -> Result<(), SciOxideError> {
    crate::validation::finite_input(equalities)?;
    crate::validation::finite_input(inequalities)
}

use crate::validation::callback_scalar as validate_callback_value;

fn evaluate_bounds_into<S>(
    bounds: &Bounds<S>,
    x: &[S],
    inequalities: &mut [S],
    inequality_index: &mut usize,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    for (index, &(lower, upper)) in bounds.as_slice().iter().enumerate() {
        if let Some(lower) = lower {
            inequalities[*inequality_index] = validate_bound_value(lower - x[index])?;
            *inequality_index += 1;
        }
        if let Some(upper) = upper {
            inequalities[*inequality_index] = validate_bound_value(x[index] - upper)?;
            *inequality_index += 1;
        }
    }
    Ok(())
}

fn validate_bound_value<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if !value.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "finite bound arithmetic produced a non-finite value",
        ));
    }
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_jacobian_row<S>(
    function: &(dyn ConstraintFunction<S> + Send + Sync),
    x: &[S],
    value: S,
    strategy: DerivativeStrategy,
    output: &mut [S],
    finite_difference: &mut DerivativeWorkspace<S>,
    stats: &mut ConstraintJacobianStats,
    observer: Option<&dyn crate::context::Observation>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let execution = observer.map_or_else(
        || Ok(scioxide_core::ExecutionContext::sequential()),
        |o| o.execution(),
    )?;
    let (_, function_evaluations) = finite_difference.acquire_into(
        strategy,
        x,
        None,
        Some(&[value]),
        |analytic, out| {
            crate::context::invoke(
                observer,
                scioxide_core::EvaluationRole::ConstraintJacobian,
                || {
                    if analytic {
                        function.jacobian_into(x, out)
                    } else {
                        function.automatic_jacobian_into(x, out)
                    }
                },
            )
        },
        |point, out| {
            out[0] =
                crate::context::invoke(observer, scioxide_core::EvaluationRole::Constraint, || {
                    function.eval(point)
                })
                .and_then(validate_callback_value)?;
            Ok(())
        },
        output,
        execution,
    )?;
    stats.constraint_evaluations = stats
        .constraint_evaluations
        .checked_add(function_evaluations)
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("constraint_evaluation_count"))
        })?;
    stats.jacobian_evaluations = stats.jacobian_evaluations.checked_add(1).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("constraint_jacobian_count"))
    })?;
    Ok(())
}

fn fill_bound_jacobian<S>(
    bounds: &Bounds<S>,
    dimension: usize,
    equality_count: usize,
    inequality_index: &mut usize,
    jacobian: &mut Matrix<S>,
) where
    S: RealScalar + NumericScalar<Real = S>,
{
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    for (variable, &(lower, upper)) in bounds.as_slice().iter().enumerate() {
        if lower.is_some() {
            let row = equality_count + *inequality_index;
            jacobian.as_mut_slice()[row * dimension..(row + 1) * dimension].fill(zero);
            jacobian.as_mut_slice()[row * dimension + variable] = -one;
            *inequality_index += 1;
        }
        if upper.is_some() {
            let row = equality_count + *inequality_index;
            jacobian.as_mut_slice()[row * dimension..(row + 1) * dimension].fill(zero);
            jacobian.as_mut_slice()[row * dimension + variable] = one;
            *inequality_index += 1;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::ConstraintFn;

    struct NativeEquality;

    impl ConstraintFunction<f64> for NativeEquality {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] + x[1] - 1.0)
        }

        fn jacobian_into(&self, _x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output.copy_from_slice(&[1.0, 1.0]);
            Ok(true)
        }
    }

    struct NativeInequality;

    impl ConstraintFunction<f64> for NativeInequality {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] + x[1] - 2.0)
        }
    }

    struct FailingJacobian {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    impl ConstraintFunction<f64> for FailingJacobian {
        fn dim(&self) -> usize {
            1
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0])
        }

        fn jacobian_into(&self, _x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            output[0] = 1.0;
            if self.fail {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("x"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "injected failure",
                ));
            }
            Ok(true)
        }
    }

    #[test]
    fn builders_validate_dimensions_and_preserve_stable_order() {
        let wrong = ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1]));
        assert!(matches!(
            (Constraints::<f64>::new(1).unwrap().equality(wrong))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));

        let constraints = Constraints::new(1)
            .unwrap()
            .inequality(ConstraintFn::new(1, |x: &[f64]| Ok(x[0] + 1.0)))
            .unwrap()
            .equality(ConstraintFn::new(1, |x: &[f64]| Ok(x[0] - 1.0)))
            .unwrap()
            .bounds(Bounds::new(vec![(Some(0.0), Some(2.0))]).unwrap())
            .unwrap();
        let values = constraints.evaluate(&[0.5]).unwrap();
        assert_eq!(values.equalities(), &[-0.5]);
        assert_eq!(values.inequalities(), &[1.5, -0.5, -1.5]);
    }

    #[test]
    fn evaluation_preflight_and_callback_failures_are_transactional() {
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = Arc::clone(&calls);
        let constraints = Constraints::new(1)
            .unwrap()
            .equality(ConstraintFn::new(1, move |_: &[f64]| {
                callback_calls.fetch_add(1, Ordering::Relaxed);
                Ok(f64::NAN)
            }))
            .unwrap();
        let mut workspace = constraints.workspace().unwrap();
        let unrelated = Constraints::<f64>::new(1).unwrap();
        let mut incompatible_workspace = unrelated.workspace().unwrap();
        let mut equality = [7.0];

        assert!(matches!(
            (constraints.evaluate_into(&[], &mut equality, &mut [], &mut workspace))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(equality, [7.0]);
        assert!(matches!(
            (constraints.evaluate_into(&[0.0], &mut [], &mut [], &mut workspace))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
        assert!(matches!(
            (constraints.evaluate_into(
                &[0.0],
                &mut equality,
                &mut [],
                &mut incompatible_workspace,
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
        assert!(matches!(
            (constraints.evaluate_into(&[f64::NAN], &mut equality, &mut [], &mut workspace))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(equality, [7.0]);
        assert!(matches!(
            (constraints.evaluate_into(&[0.0], &mut equality, &mut [], &mut workspace))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(equality, [7.0]);
    }

    #[test]
    fn jacobian_preflight_and_callback_failures_are_transactional() {
        let calls = Arc::new(AtomicUsize::new(0));
        let constraints = Constraints::new(1)
            .unwrap()
            .equality(FailingJacobian {
                calls: Arc::clone(&calls),
                fail: false,
            })
            .unwrap()
            .inequality(FailingJacobian {
                calls: Arc::clone(&calls),
                fail: true,
            })
            .unwrap();
        let values = constraints.evaluate(&[0.5]).unwrap();
        let mut workspace = constraints.workspace().unwrap();
        let mut wrong_output = Matrix::zeros(1, 1).unwrap();

        assert!(matches!(
            (constraints.jacobian_into(
                &[0.5],
                values.equalities(),
                values.inequalities(),
                DerivativeStrategy::Analytic,
                &mut wrong_output,
                &mut workspace,
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        let mut output = Matrix::from_vec(2, 1, vec![7.0, 8.0]).unwrap();
        assert!(matches!(
            (constraints.jacobian_into(
                &[0.5],
                values.equalities(),
                values.inequalities(),
                DerivativeStrategy::Analytic,
                &mut output,
                &mut workspace,
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(output.as_slice(), &[7.0, 8.0]);
    }

    #[test]
    fn jacobian_groups_rows_and_reports_finite_difference_work() {
        let constraints = Constraints::new(2)
            .unwrap()
            .inequality(ConstraintFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1])))
            .unwrap()
            .equality(
                ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1])).with_jacobian(
                    |_: &[f64], output| {
                        output.copy_from_slice(&[1.0, 1.0]);
                        Ok(())
                    },
                ),
            )
            .unwrap();
        let values = constraints.evaluate(&[2.0, 3.0]).unwrap();
        let mut workspace = constraints.workspace().unwrap();
        let mut jacobian = Matrix::zeros(2, 2).unwrap();
        let stats = constraints
            .jacobian_into(
                &[2.0, 3.0],
                values.equalities(),
                values.inequalities(),
                DerivativeStrategy::default(),
                &mut jacobian,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(jacobian.as_slice()[..2], [1.0, 1.0]);
        assert!((jacobian.as_slice()[2] - 4.0).abs() < 1e-5);
        assert!((jacobian.as_slice()[3] - 1.0).abs() < 1e-5);
        assert_eq!(stats.jacobian_evaluations(), 2);
        assert_eq!(stats.constraint_evaluations(), 2);
    }

    #[test]
    fn pure_and_native_callbacks_have_equivalent_values_derivatives_and_counters() {
        let pure = Constraints::new(2)
            .unwrap()
            .equality(
                ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0)).with_jacobian(
                    |_: &[f64], output| {
                        output.copy_from_slice(&[1.0, 1.0]);
                        Ok(())
                    },
                ),
            )
            .unwrap()
            .inequality(ConstraintFn::new(2, |x: &[f64]| {
                Ok(x[0] * x[0] + x[1] - 2.0)
            }))
            .unwrap();
        let native = Constraints::new(2)
            .unwrap()
            .equality(NativeEquality)
            .unwrap()
            .inequality(NativeInequality)
            .unwrap();
        let point = [0.25, 0.5];
        let pure_values = pure.evaluate(&point).unwrap();
        let native_values = native.evaluate(&point).unwrap();
        assert_eq!(pure_values, native_values);

        let mut pure_workspace = pure.workspace().unwrap();
        let mut native_workspace = native.workspace().unwrap();
        let mut pure_jacobian = Matrix::zeros(2, 2).unwrap();
        let mut native_jacobian = Matrix::zeros(2, 2).unwrap();
        let pure_stats = pure
            .jacobian_into(
                &point,
                pure_values.equalities(),
                pure_values.inequalities(),
                DerivativeStrategy::default(),
                &mut pure_jacobian,
                &mut pure_workspace,
            )
            .unwrap();
        let native_stats = native
            .jacobian_into(
                &point,
                native_values.equalities(),
                native_values.inequalities(),
                DerivativeStrategy::default(),
                &mut native_jacobian,
                &mut native_workspace,
            )
            .unwrap();

        pure_jacobian
            .as_slice()
            .iter()
            .zip(native_jacobian.as_slice())
            .for_each(|(&pure, &native)| assert!((pure - native).abs() < 1e-12));
        assert_eq!(pure_stats, native_stats);
        assert_eq!(pure_stats.constraint_evaluations(), 2);
        assert_eq!(pure_stats.jacobian_evaluations(), 2);
    }

    #[test]
    fn collection_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Constraints<f64>>();
    }
}
