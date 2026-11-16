//! Problem definitions: linear, quadratic and mixed-integer programs, bounded
//! objectives and constrained problems.

#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_core::{Bounds, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_sparse::CsrMatrix;

/// Holds the coefficient matrix of a linear constraint system or a quadratic
/// objective, dense or sparse.
///
/// Sparse matrices use compressed sparse row (CSR) storage. A solver keeps the
/// sparse form until it needs a dense workspace.
#[derive(Debug, Clone)]
#[must_use]
pub enum CoefficientMatrix<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Dense coefficients, stored contiguously in row-major order.
    Dense(Matrix<S>),
    /// Sparse coefficients in compressed sparse row storage.
    Sparse(CsrMatrix<S>),
}

impl<S> CoefficientMatrix<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Returns the number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.rows(),
            Self::Sparse(matrix) => matrix.rows(),
        }
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.cols(),
            Self::Sparse(matrix) => matrix.cols(),
        }
    }

    /// Returns whether the coefficients are stored sparse.
    #[must_use]
    pub const fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    /// Returns the coefficient at (`row`, `col`).
    ///
    /// # Errors
    ///
    /// Returns [`OutOfBounds`](scioxide_core::SciOxideErrorKind::OutOfBounds) if
    /// either index lies outside the matrix.
    pub fn get(&self, row: usize, col: usize) -> Result<S, SciOxideError> {
        match self {
            Self::Dense(matrix) => matrix.view().get(row, col),
            Self::Sparse(matrix) => matrix.get(row, col),
        }
    }

    pub(crate) fn validate_finite(&self) -> Result<(), SciOxideError> {
        let values: &[S] = match self {
            Self::Dense(matrix) => matrix.as_slice(),
            Self::Sparse(matrix) => matrix.values(),
        };
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                alloc::format!("stored coefficient {index} must be finite"),
            ));
        }
        Ok(())
    }

    pub(crate) fn write_row(&self, row: usize, output: &mut [S]) -> Result<(), SciOxideError> {
        if row >= self.rows() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("coefficient_row"),
                row,
                self.rows(),
            ));
        }
        if output.len() != self.cols() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.cols()),
                scioxide_core::Dimensions::one(output.len()),
            ));
        }
        output.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
        match self {
            Self::Dense(matrix) => {
                let start = row * matrix.cols();
                output.copy_from_slice(&matrix.as_slice()[start..start + matrix.cols()]);
            }
            Self::Sparse(matrix) => {
                let start = matrix.row_ptr()[row];
                let end = matrix.row_ptr()[row + 1];
                for (&col, &value) in matrix.col_indices()[start..end]
                    .iter()
                    .zip(&matrix.values()[start..end])
                {
                    output[col] = value;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn row_entries(
        &self,
        row: usize,
    ) -> Result<impl Iterator<Item = (usize, S)> + '_, SciOxideError> {
        if row >= self.rows() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("coefficient_row"),
                row,
                self.rows(),
            ));
        }
        let range = match self {
            Self::Dense(m) => row * m.cols()..(row + 1) * m.cols(),
            Self::Sparse(m) => m.row_ptr()[row]..m.row_ptr()[row + 1],
        };
        Ok(range.map(move |k| match self {
            Self::Dense(m) => (k - row * m.cols(), m.as_slice()[k]),
            Self::Sparse(m) => (m.col_indices()[k], m.values()[k]),
        }))
    }
}

impl<S> From<Matrix<S>> for CoefficientMatrix<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    fn from(value: Matrix<S>) -> Self {
        Self::Dense(value)
    }
}

impl<S> From<CsrMatrix<S>> for CoefficientMatrix<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    fn from(value: CsrMatrix<S>) -> Self {
        Self::Sparse(value)
    }
}

/// Gives the relation of one row `aᵀx` of a linear constraint system to its
/// right-hand side `b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum LinearConstraintSense {
    /// `aᵀx ≤ b`.
    LessEqual,
    /// `aᵀx = b`.
    Equal,
    /// `aᵀx ≥ b`.
    GreaterEqual,
}

/// Holds a system of linear constraints `aᵢᵀx (≤ | = | ≥) bᵢ`, one relation
/// per row.
#[derive(Debug, Clone)]
#[must_use]
pub struct LinearConstraints<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    coefficients: CoefficientMatrix<S>,
    rhs: Vec<S>,
    senses: Vec<LinearConstraintSense>,
}

impl<S> LinearConstraints<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Creates the system from the coefficient matrix, the right-hand side and
    /// the relation of each row.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if the matrix has no rows or no columns.
    /// - `DimensionMismatch` if `rhs` or `senses` does not have one entry per
    ///   row.
    /// - `NonFinite` if a coefficient or right-hand-side entry is NaN or
    ///   infinite.
    pub fn new<M>(
        coefficients: M,
        rhs: Vec<S>,
        senses: Vec<LinearConstraintSense>,
    ) -> Result<Self, SciOxideError>
    where
        M: Into<CoefficientMatrix<S>>,
    {
        let coefficients = coefficients.into();
        let rows = coefficients.rows();
        let cols = coefficients.cols();
        if rows == 0 || cols == 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("coefficients"),
                scioxide_core::InvalidInputReason::Empty,
                "linear constraints must have at least one row and one column",
            ));
        }
        if rhs.len() != rows {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(rows),
                scioxide_core::Dimensions::one(rhs.len()),
            ));
        }
        if senses.len() != rows {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(rows),
                scioxide_core::Dimensions::one(senses.len()),
            ));
        }
        coefficients.validate_finite()?;
        if let Some(index) = rhs.iter().position(|value| !value.is_finite()) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                alloc::format!("right-hand-side entry {index} must be finite"),
            ));
        }
        Ok(Self {
            coefficients,
            rhs,
            senses,
        })
    }

    /// Creates linear constraints `Ax ≤ b`, with one relation for every row.
    ///
    /// Allocates `O(m)` senses for `m` rows and uses [`Self::new`] to validate data.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::new`] or `AllocationFailure` for the senses.
    pub fn less_equal<M: Into<CoefficientMatrix<S>>>(
        coefficients: M,
        rhs: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        let senses = scioxide_core::backend::try_filled(
            rhs.len(),
            LinearConstraintSense::LessEqual,
            scioxide_core::quantity!("senses"),
        )?;
        Self::new(coefficients, rhs, senses)
    }

    /// Creates linear constraints `Ax = b`, with one relation for every row.
    ///
    /// Allocates `O(m)` senses for `m` rows and uses [`Self::new`] to validate data.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::new`] or `AllocationFailure` for the senses.
    pub fn equal<M: Into<CoefficientMatrix<S>>>(
        coefficients: M,
        rhs: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        let senses = scioxide_core::backend::try_filled(
            rhs.len(),
            LinearConstraintSense::Equal,
            scioxide_core::quantity!("senses"),
        )?;
        Self::new(coefficients, rhs, senses)
    }

    /// Creates linear constraints `Ax ≥ b`, with one relation for every row.
    ///
    /// Allocates `O(m)` senses for `m` rows and uses [`Self::new`] to validate data.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::new`] or `AllocationFailure` for the senses.
    pub fn greater_equal<M: Into<CoefficientMatrix<S>>>(
        coefficients: M,
        rhs: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        let senses = scioxide_core::backend::try_filled(
            rhs.len(),
            LinearConstraintSense::GreaterEqual,
            scioxide_core::quantity!("senses"),
        )?;
        Self::new(coefficients, rhs, senses)
    }

    /// Returns the coefficient matrix.
    pub const fn coefficients(&self) -> &CoefficientMatrix<S> {
        &self.coefficients
    }

    /// Returns the right-hand side, one entry per row.
    #[must_use]
    pub fn rhs(&self) -> &[S] {
        &self.rhs
    }

    /// Returns the relation of each row.
    pub fn senses(&self) -> &[LinearConstraintSense] {
        &self.senses
    }

    /// Returns the number of rows.
    #[must_use]
    #[allow(
        clippy::len_without_is_empty,
        reason = "LinearConstraints is validated as non-empty"
    )]
    pub fn len(&self) -> usize {
        self.coefficients.rows()
    }
}

/// Describes a linear program
///
/// ```text
/// min cᵀx   subject to   aᵢᵀx (≤ | = | ≥) bᵢ,   l ≤ x ≤ u,
/// ```
///
/// with objective coefficients `c ∈ ℝⁿ`, optional [`LinearConstraints`] and
/// optional bounds. Without explicit bounds every variable satisfies `x ≥ 0`.
/// Solve it with [`SimplexSolver`](crate::SimplexSolver) or
/// [`InteriorPointLp`](crate::InteriorPointLp); the documentation of
/// [`SimplexSolver`](crate::SimplexSolver) has an example.
#[derive(Debug, Clone)]
#[must_use]
pub struct LpProblem<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    objective: Vec<S>,
    constraints: Option<LinearConstraints<S>>,
    bounds: Option<Bounds<S>>,
}

impl<S> LpProblem<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Creates a linear program with objective coefficients `objective`, no
    /// constraints and the default bounds `x ≥ 0`.
    ///
    /// [`Self::with_bounds`] replaces the default bounds. In a [`Bounds`] value,
    /// `None` is an infinite endpoint.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `objective` is empty.
    /// - `NonFinite` if an objective coefficient is NaN or infinite.
    pub fn new(objective: Vec<S>) -> Result<Self, SciOxideError> {
        validate_objective(&objective)?;
        Ok(Self {
            objective,
            constraints: None,
            bounds: None,
        })
    }

    /// Adds the linear constraints.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of columns differs from the number
    ///   of variables.
    /// - `InvalidInput` if constraints are already set.
    pub fn with_constraints(
        mut self,
        constraints: LinearConstraints<S>,
    ) -> Result<Self, SciOxideError> {
        if self.constraints.is_some() {
            return Err(duplicate_attachment(scioxide_core::argument!(
                "constraints"
            )));
        }
        validate_constraint_dimension(&constraints, self.dimension())?;
        self.constraints = Some(constraints);
        Ok(self)
    }

    /// Sets the bounds `l ≤ x ≤ u`, replacing the default `x ≥ 0`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `bounds.len()` differs from the number of
    ///   variables.
    /// - `InvalidInput` if bounds are already set.
    pub fn with_bounds(mut self, bounds: Bounds<S>) -> Result<Self, SciOxideError> {
        if self.bounds.is_some() {
            return Err(duplicate_attachment(scioxide_core::argument!("bounds")));
        }
        validate_bounds_dimension(&bounds, self.dimension())?;
        self.bounds = Some(bounds);
        Ok(self)
    }

    /// Returns the number of variables `n`.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.objective.len()
    }

    /// Returns the objective coefficients `c`.
    #[must_use]
    pub fn objective(&self) -> &[S] {
        &self.objective
    }

    /// Returns the linear constraints, if set.
    #[must_use]
    pub const fn constraints(&self) -> Option<&LinearConstraints<S>> {
        self.constraints.as_ref()
    }

    /// Returns the explicit bounds; `None` means the default `x ≥ 0`.
    #[must_use]
    pub const fn bounds(&self) -> Option<&Bounds<S>> {
        self.bounds.as_ref()
    }

    pub(crate) fn update_bounds(
        &mut self,
        entries: &[(Option<S>, Option<S>)],
    ) -> Result<(), SciOxideError> {
        self.bounds
            .as_mut()
            .ok_or_else(|| {
                crate::validation::invalid(
                    scioxide_core::argument!("bounds"),
                    "retained LP requires explicit bounds",
                )
            })?
            .copy_from_slice(entries)
    }
}

/// Selects how [`QpProblem`] treats a Hessian `Q` that is not symmetric.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HessianSymmetry {
    /// Requires `Q` to be exactly symmetric.
    #[default]
    RequireSymmetric,
    /// Uses the symmetric part `(Q + Qᵀ)/2`, which defines the same objective.
    SymmetricPart,
}

/// Describes a quadratic program
///
/// ```text
/// min ½ xᵀQx + cᵀx   subject to   aᵢᵀx (≤ | = | ≥) bᵢ,   l ≤ x ≤ u,
/// ```
///
/// with a symmetric `n × n` Hessian `Q`, linear coefficients `c ∈ ℝⁿ`,
/// optional [`LinearConstraints`] and optional bounds. Without explicit
/// bounds every variable is free. Solve it with
/// [`QpSolver`](crate::QpSolver).
#[derive(Debug, Clone)]
#[must_use]
pub struct QpProblem<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    hessian: CoefficientMatrix<S>,
    symmetry: HessianSymmetry,
    linear_objective: Vec<S>,
    constraints: Option<LinearConstraints<S>>,
    bounds: Option<Bounds<S>>,
}

impl<S> QpProblem<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Creates a quadratic program with Hessian `hessian`, dense or sparse, and
    /// linear coefficients `linear_objective`, without constraints or bounds.
    ///
    /// The Hessian must be exactly symmetric. Use [`Self::new_with_symmetry`]
    /// to accept a nonsymmetric `Q` through its symmetric part.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `linear_objective` is empty.
    /// - `DimensionMismatch` if the Hessian is not `n × n`, where `n` is the
    ///   length of `linear_objective`.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `InvalidInput` (`Inconsistent`) if the Hessian is not exactly
    ///   symmetric.
    pub fn new<M>(hessian: M, linear_objective: Vec<S>) -> Result<Self, SciOxideError>
    where
        M: Into<CoefficientMatrix<S>>,
    {
        Self::new_with_symmetry(hessian, linear_objective, HessianSymmetry::RequireSymmetric)
    }

    /// Creates a quadratic program and sets how a nonsymmetric Hessian is
    /// treated.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`]; the symmetry check applies only with
    /// [`HessianSymmetry::RequireSymmetric`].
    pub fn new_with_symmetry<M: Into<CoefficientMatrix<S>>>(
        hessian: M,
        linear_objective: Vec<S>,
        symmetry: HessianSymmetry,
    ) -> Result<Self, SciOxideError> {
        validate_objective(&linear_objective)?;
        let hessian = hessian.into();
        let dimension = linear_objective.len();
        if hessian.rows() != dimension || hessian.cols() != dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(dimension, dimension),
                scioxide_core::Dimensions::two(hessian.rows(), hessian.cols()),
            ));
        }
        hessian.validate_finite()?;
        if symmetry == HessianSymmetry::RequireSymmetric {
            validate_hessian_symmetry(&hessian)?;
        }
        Ok(Self {
            hessian,
            symmetry,
            linear_objective,
            constraints: None,
            bounds: None,
        })
    }

    /// Returns how a nonsymmetric Hessian is treated.
    pub const fn symmetry(&self) -> HessianSymmetry {
        self.symmetry
    }

    /// Adds the linear constraints.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of columns differs from the number
    ///   of variables.
    /// - `InvalidInput` if constraints are already set.
    pub fn with_constraints(
        mut self,
        constraints: LinearConstraints<S>,
    ) -> Result<Self, SciOxideError> {
        if self.constraints.is_some() {
            return Err(duplicate_attachment(scioxide_core::argument!(
                "constraints"
            )));
        }
        validate_constraint_dimension(&constraints, self.dimension())?;
        self.constraints = Some(constraints);
        Ok(self)
    }

    /// Sets the bounds `l ≤ x ≤ u`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `bounds.len()` differs from the number of
    ///   variables.
    /// - `InvalidInput` if bounds are already set.
    pub fn with_bounds(mut self, bounds: Bounds<S>) -> Result<Self, SciOxideError> {
        if self.bounds.is_some() {
            return Err(duplicate_attachment(scioxide_core::argument!("bounds")));
        }
        validate_bounds_dimension(&bounds, self.dimension())?;
        self.bounds = Some(bounds);
        Ok(self)
    }

    /// Returns the number of variables `n`.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.linear_objective.len()
    }

    /// Returns the Hessian `Q`.
    pub const fn hessian(&self) -> &CoefficientMatrix<S> {
        &self.hessian
    }

    /// Returns the linear coefficients `c`.
    #[must_use]
    pub fn linear_objective(&self) -> &[S] {
        &self.linear_objective
    }

    /// Returns the linear constraints, if set.
    #[must_use]
    pub const fn constraints(&self) -> Option<&LinearConstraints<S>> {
        self.constraints.as_ref()
    }

    /// Returns the bounds; `None` means every variable is free.
    #[must_use]
    pub const fn bounds(&self) -> Option<&Bounds<S>> {
        self.bounds.as_ref()
    }
}

/// Describes a mixed-integer linear program: an [`LpProblem`], its linear
/// relaxation, in which the listed variables must take integer values.
///
/// Solve it with [`BranchAndBound`](crate::BranchAndBound).
#[derive(Debug, Clone)]
#[must_use]
pub struct MipProblem<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    relaxation: LpProblem<S>,
    integer_variables: Vec<usize>,
}

impl<S> MipProblem<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Creates the problem from its relaxation and the indices of the integer
    /// variables.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `integer_variables` is empty.
    /// - `OutOfBounds` if an index is not less than the number of variables.
    /// - `InvalidInput` (`Inconsistent`) if an index occurs twice.
    pub fn new(
        relaxation: LpProblem<S>,
        integer_variables: Vec<usize>,
    ) -> Result<Self, SciOxideError> {
        if integer_variables.is_empty() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("integer_variables"),
                scioxide_core::InvalidInputReason::Empty,
                "a MIP must contain at least one integer-constrained variable",
            ));
        }
        let dimension = relaxation.dimension();
        let mut seen = try_filled(
            dimension,
            false,
            scioxide_core::quantity!("mip_integer_variable_validation_flags"),
        )?;
        for &index in &integer_variables {
            if index >= dimension {
                return Err(SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("mip_integer_variable"),
                    index,
                    dimension,
                ));
            }
            if seen[index] {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("integer_variables"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    alloc::format!("integer variable {index} was specified more than once"),
                ));
            }
            seen[index] = true;
        }
        Ok(Self {
            relaxation,
            integer_variables,
        })
    }

    /// Returns the linear relaxation.
    pub const fn relaxation(&self) -> &LpProblem<S> {
        &self.relaxation
    }

    /// Returns the indices of the integer variables, in the order given.
    #[must_use]
    pub fn integer_variables(&self) -> &[usize] {
        &self.integer_variables
    }
}

fn validate_objective<S>(objective: &[S]) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    if objective.is_empty() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("objective"),
            scioxide_core::InvalidInputReason::Empty,
            "objective must contain at least one variable",
        ));
    }
    if let Some(index) = objective.iter().position(|value| !value.is_finite()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            alloc::format!("objective coefficient {index} must be finite"),
        ));
    }
    Ok(())
}

fn validate_constraint_dimension<S>(
    constraints: &LinearConstraints<S>,
    dimension: usize,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    if constraints.coefficients().cols() != dimension {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(constraints.len(), dimension),
            scioxide_core::Dimensions::two(
                constraints.coefficients().rows(),
                constraints.coefficients().cols(),
            ),
        ));
    }
    Ok(())
}

fn validate_bounds_dimension<S>(bounds: &Bounds<S>, dimension: usize) -> Result<(), SciOxideError>
where
    S: RealScalar,
{
    if bounds.len() != dimension {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(dimension),
            scioxide_core::Dimensions::one(bounds.len()),
        ));
    }
    Ok(())
}

fn validate_hessian_symmetry<S>(hessian: &CoefficientMatrix<S>) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let dimension = hessian.rows();
    for row in 0..dimension {
        for col in (row + 1)..dimension {
            let forward = hessian.get(row, col)?;
            let reverse = hessian.get(col, row)?;
            if forward != reverse {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("hessian"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    alloc::format!(
                        "Hessian entries ({row}, {col}) and ({col}, {row}) are asymmetric"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn duplicate_attachment(argument: scioxide_core::Argument) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::Inconsistent,
        alloc::format!("{argument} were already attached"),
    )
}

impl<S: RealScalar + NumericScalar<Real = S>> scioxide_core::LinearOperator<S>
    for CoefficientMatrix<S>
{
    fn rows(&self) -> usize {
        self.rows()
    }
    fn cols(&self) -> usize {
        self.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        match self {
            Self::Dense(m) => Ok(m.rows()),
            Self::Sparse(m) => scioxide_core::LinearOperator::apply_workspace_len(m),
        }
    }
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Dense(m) => scioxide_core::LinearOperator::apply_into(
                &m.view().try_finite()?,
                input,
                output,
                workspace,
            ),
            Self::Sparse(m) => {
                scioxide_core::LinearOperator::apply_into(m, input, output, workspace)
            }
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> scioxide_core::AdjointOperator<S>
    for CoefficientMatrix<S>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        match self {
            Self::Dense(m) => Ok(m.cols()),
            Self::Sparse(m) => scioxide_core::AdjointOperator::apply_adjoint_workspace_len(m),
        }
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Dense(m) => scioxide_core::AdjointOperator::apply_adjoint_into(
                &m.view().try_finite()?,
                input,
                output,
                workspace,
            ),
            Self::Sparse(m) => {
                scioxide_core::AdjointOperator::apply_adjoint_into(m, input, output, workspace)
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_sparse::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};

    fn dense_constraints() -> LinearConstraints<f64> {
        LinearConstraints::new(
            Matrix::from_vec(1, 2, vec![1.0, 1.0]).unwrap(),
            vec![2.0],
            vec![LinearConstraintSense::LessEqual],
        )
        .unwrap()
    }

    #[test]
    fn linear_constraints_validate_shapes_and_finiteness() {
        assert!(matches!(
            (LinearConstraints::new(Matrix::zeros(0, 2).unwrap(), Vec::<f64>::new(), Vec::new()))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::Empty,
                ..
            })
        ));
        let matrix = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
        assert!(matches!(
            (LinearConstraints::new(
                matrix.clone(),
                vec![1.0],
                vec![LinearConstraintSense::Equal; 2]
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
        assert!(matches!(
            (LinearConstraints::new(matrix, vec![1.0, 2.0], vec![LinearConstraintSense::Equal]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
        let non_finite = Matrix::from_vec(1, 1, vec![f64::NAN]).unwrap();
        assert!(matches!(
            (LinearConstraints::new(non_finite, vec![0.0], vec![LinearConstraintSense::Equal]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
        assert!(matches!(
            (LinearConstraints::new(
                Matrix::from_vec(1, 1, vec![1.0_f64]).unwrap(),
                vec![f64::INFINITY],
                vec![LinearConstraintSense::Equal]
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }

    #[test]
    fn lp_rejects_empty_non_finite_and_mismatched_inputs() {
        assert!(matches!(
            (LpProblem::<f64>::new(Vec::new()))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::Empty,
                ..
            })
        ));
        assert!(matches!(
            (LpProblem::new(vec![f64::INFINITY]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
        assert!(matches!(
            (LpProblem::new(vec![1.0_f64])
                .unwrap()
                .with_constraints(dense_constraints()))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn bounds_are_dimension_checked_and_single_assignment() {
        let problem = LpProblem::new(vec![1.0_f64, 2.0]).unwrap();
        assert!(matches!(
            (problem
                .clone()
                .with_bounds(Bounds::new(vec![(None, None)]).unwrap()))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
        let bounded = problem
            .with_bounds(Bounds::from_indexed(2, Vec::new()).unwrap())
            .unwrap();
        assert!(matches!(
            (bounded.with_bounds(Bounds::new(vec![(None, None); 2]).unwrap()))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::Inconsistent,
                ..
            })
        ));
    }

    #[test]
    fn qp_validates_shape_finiteness_and_symmetry() {
        assert!(matches!(
            (QpProblem::new(Matrix::zeros(0, 0).unwrap(), Vec::<f64>::new()))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::Empty,
                ..
            })
        ));
        assert!(matches!(
            (QpProblem::new(Matrix::zeros(2, 1).unwrap(), vec![0.0_f64, 0.0]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
        assert!(matches!(
            (QpProblem::new(
                Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 0.0, 1.0]).unwrap(),
                vec![0.0, 0.0]
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::Inconsistent,
                ..
            })
        ));
        assert!(matches!(
            (QpProblem::new(
                Matrix::from_vec(1, 1, vec![f64::INFINITY]).unwrap(),
                vec![0.0]
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }

    #[test]
    fn sparse_constraints_and_hessian_are_retained() {
        let mut constraint_coo = SparseBuilder::new(1, 2);
        constraint_coo.push(0, 0, 1.0_f64).unwrap();
        constraint_coo.push(0, 1, 1.0).unwrap();
        let constraints = LinearConstraints::new(
            constraint_coo
                .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
                .unwrap(),
            vec![3.0],
            vec![LinearConstraintSense::LessEqual],
        )
        .unwrap();
        assert!(constraints.coefficients().is_sparse());

        let mut hessian_coo = SparseBuilder::new(2, 2);
        hessian_coo.push(0, 0, 2.0_f64).unwrap();
        hessian_coo.push(1, 1, 2.0).unwrap();
        let problem = QpProblem::new(
            hessian_coo
                .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
                .unwrap(),
            vec![-1.0, -1.0],
        )
        .unwrap()
        .with_constraints(constraints)
        .unwrap();
        assert!(problem.hessian().is_sparse());
        assert!(
            problem
                .constraints()
                .is_some_and(|value| value.coefficients().is_sparse())
        );
    }

    #[test]
    fn asymmetric_hessian_requires_explicit_interpretation() {
        let delta = f64::SQRT_EPSILON;
        let hessian = Matrix::from_vec(2, 2, vec![2.0, 1.0, 1.0 + delta, 2.0]).unwrap();
        assert!(QpProblem::new(hessian.clone(), vec![0.0, 0.0]).is_err());
        assert!(
            QpProblem::new_with_symmetry(hessian, vec![0.0, 0.0], HessianSymmetry::SymmetricPart)
                .is_ok()
        );
    }

    #[test]
    fn mip_validates_integrality_metadata() {
        let relaxation = LpProblem::new(vec![1.0_f64, 2.0]).unwrap();
        assert!(matches!(
            (MipProblem::new(relaxation.clone(), Vec::new()))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::Empty,
                ..
            })
        ));
        assert!(matches!(
            (MipProblem::new(relaxation.clone(), vec![2]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::OutOfBounds { .. })
        ));
        assert!(matches!(
            (MipProblem::new(relaxation, vec![1, 1]))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::Inconsistent,
                ..
            })
        ));
    }
}

/// Borrowed operator; a dense matrix keeps its finiteness check for repeated products.
#[derive(Clone, Copy)]
pub(crate) enum BorrowedAdjointOperator<'a, S: NumericScalar> {
    Dense(scioxide_linalg::FiniteMatrixView<'a, S>),
    Other(&'a dyn scioxide_core::AdjointOperator<S>),
}
impl<S: NumericScalar> BorrowedAdjointOperator<'_, S> {
    pub(crate) fn as_operator(&self) -> &dyn scioxide_core::AdjointOperator<S> {
        match self {
            Self::Dense(matrix) => matrix,
            Self::Other(operator) => *operator,
        }
    }
    pub(crate) fn dense(&self) -> Option<scioxide_linalg::MatrixView<'_, S>> {
        match self {
            Self::Dense(matrix) => Some(matrix.as_view()),
            Self::Other(_) => None,
        }
    }
}

/// Pairs an objective with box bounds `l ≤ x ≤ u`: the problem of
/// [`LbfgsB`](crate::LbfgsB), [`Powell`](crate::Powell) and the global
/// searches.
///
/// A `None` endpoint is infinite. The global searches require finite
/// endpoints. A point is in the domain if it is finite, lies in
/// the box and is in the domain of the objective.
pub struct BoundedObjective<'a, S: RealScalar + NumericScalar<Real = S>> {
    objective: &'a dyn crate::ObjectiveFunction<S>,
    shared: Option<&'a (dyn crate::ObjectiveFunction<S> + Sync)>,
    pub(crate) bounds: &'a Bounds<S>,
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> BoundedObjective<'a, S> {
    /// Creates the problem from an objective and its bounds.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `bounds.len()` differs from the dimension
    /// of `objective`.
    pub fn new(
        objective: &'a dyn crate::ObjectiveFunction<S>,
        bounds: &'a Bounds<S>,
    ) -> Result<Self, SciOxideError> {
        if objective.dim() != bounds.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(objective.dim()),
                scioxide_core::Dimensions::one(bounds.len()),
            ));
        }
        Ok(Self {
            objective,
            shared: None,
            bounds,
        })
    }
    /// Creates the problem from a `Sync` objective and its bounds.
    ///
    /// Global searches with a parallel execution policy evaluate the objective
    /// on several threads and require this form. See
    /// [Randomness and parallelism](crate#randomness-and-parallelism).
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `bounds.len()` differs from the dimension
    /// of `objective`.
    pub fn new_sync(
        objective: &'a (dyn crate::ObjectiveFunction<S> + Sync),
        bounds: &'a Bounds<S>,
    ) -> Result<Self, SciOxideError> {
        let mut problem = Self::new(objective, bounds)?;
        problem.shared = Some(objective);
        Ok(problem)
    }
    /// `Sync` objective, if the problem was built with [`Self::new_sync`].
    pub(crate) const fn sync_objective(
        &self,
    ) -> Option<&'a (dyn crate::ObjectiveFunction<S> + Sync)> {
        self.shared
    }
    /// Returns the objective.
    pub const fn objective(&self) -> &'a dyn crate::ObjectiveFunction<S> {
        self.objective
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> crate::ObjectiveFunction<S>
    for BoundedObjective<'_, S>
{
    fn dim(&self) -> usize {
        self.objective.dim()
    }
    fn bounds(&self) -> Option<&Bounds<S>> {
        Some(self.bounds)
    }
    fn is_in_domain(&self, x: &[S]) -> Result<bool, SciOxideError> {
        if x.len() != self.dim() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dim()),
                scioxide_core::Dimensions::one(x.len()),
            ));
        }
        Ok(x.iter().zip(self.bounds.as_slice()).all(|(&v, &(lo, hi))| {
            v.is_finite() && lo.is_none_or(|b| v >= b) && hi.is_none_or(|b| v <= b)
        }) && self.objective.is_in_domain(x)?)
    }
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        self.objective.eval(x)
    }
    fn gradient_into(&self, x: &[S], out: &mut [S]) -> Result<bool, SciOxideError> {
        self.objective.gradient_into(x, out)
    }
    fn automatic_gradient_into(&self, x: &[S], out: &mut [S]) -> Result<bool, SciOxideError> {
        self.objective.automatic_gradient_into(x, out)
    }
}

/// Pairs an objective `f` with constraints: the problem
///
/// ```text
/// min f(x)   subject to   h(x) = 0,   g(x) ≤ 0,   l ≤ x ≤ u
/// ```
///
/// of [`PenaltyMethod`](crate::PenaltyMethod),
/// [`AugmentedLagrangian`](crate::AugmentedLagrangian), [`Sqp`](crate::Sqp),
/// [`InteriorPoint`](crate::InteriorPoint), [`Cobyla`](crate::Cobyla) and
/// [`Cobyqa`](crate::Cobyqa).
///
/// The bounds belong to the [`Constraints`](crate::Constraints), set with
/// [`Constraints::bounds`](crate::Constraints::bounds). An optional exact
/// [`LagrangianHessian`](crate::LagrangianHessian) replaces the quasi-Newton
/// approximation in [`InteriorPoint`](crate::InteriorPoint); the other
/// methods ignore it. The problem does not implement
/// [`ObjectiveFunction`](crate::ObjectiveFunction), so it cannot be passed to
/// an unconstrained solver by mistake.
///
/// # Examples
///
/// ```
/// use scioxide_optimize::{ConstrainedProblem, ConstraintFn, Constraints, ObjectiveFn, Sqp};
///
/// // min x₀² + x₁² subject to x₀ + x₁ = 1 has the solution (½, ½).
/// let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
/// let constraints = Constraints::new(2)?
///     .equality(ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0)))?;
/// let problem = ConstrainedProblem::new(&objective, &constraints)?;
/// let result = Sqp::default().minimize(&problem, &[2.0, 0.0])?;
/// assert!((result.x()[0] - 0.5).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct ConstrainedProblem<'a, S: RealScalar + NumericScalar<Real = S>> {
    objective: &'a dyn crate::ObjectiveFunction<S>,
    constraints: &'a crate::Constraints<S>,
    hessian: Option<&'a dyn crate::LagrangianHessian<S>>,
}
impl<S: RealScalar + NumericScalar<Real = S>> core::fmt::Debug for ConstrainedProblem<'_, S> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ConstrainedProblem")
            .field("dimension", &self.objective.dim())
            .field("constraints", self.constraints)
            .field("lagrangian_hessian", &self.hessian.is_some())
            .finish()
    }
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> ConstrainedProblem<'a, S> {
    /// Creates the problem from an objective and its constraints.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the dimensions of `objective` and `constraints`
    ///   differ.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if `objective` has its own
    ///   bounds; set them on the constraints instead.
    pub fn new(
        objective: &'a dyn crate::ObjectiveFunction<S>,
        constraints: &'a crate::Constraints<S>,
    ) -> Result<Self, SciOxideError> {
        if objective.dim() != constraints.dimension() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(constraints.dimension()),
                scioxide_core::Dimensions::one(objective.dim()),
            ));
        }
        if objective.bounds().is_some() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("objective"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "attach bounds through Constraints::bounds",
            ));
        }
        Ok(Self {
            objective,
            constraints,
            hessian: None,
        })
    }
    /// Sets the exact Hessian of the Lagrangian. An error from it is returned,
    /// never replaced by an approximation.
    #[must_use]
    pub const fn with_lagrangian_hessian(
        mut self,
        hessian: &'a dyn crate::LagrangianHessian<S>,
    ) -> Self {
        self.hessian = Some(hessian);
        self
    }
    /// Returns the objective.
    pub const fn objective(&self) -> &'a dyn crate::ObjectiveFunction<S> {
        self.objective
    }
    /// Returns the constraints, including the problem's bounds.
    pub const fn constraints(&self) -> &'a crate::Constraints<S> {
        self.constraints
    }
    /// Returns the exact Hessian of the Lagrangian, if set.
    pub const fn lagrangian_hessian(&self) -> Option<&'a dyn crate::LagrangianHessian<S>> {
        self.hessian
    }
    /// Returns the number of variables.
    pub const fn dim(&self) -> usize {
        self.constraints.dimension()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> LpProblem<S> {
    /// Solves this problem by revised simplex with default settings.
    ///
    /// Uses [`crate::SimplexSolver`] with its default limit of
    /// `max(1000, 10(n + m))` iterations, where `n` is the number of variables
    /// and `m` the number of constraint rows, and at most 32 basis updates
    /// between factorizations. Original-problem feasibility and optimality
    /// tests use absolute and relative tolerances `1e-8` in `f64`, `1e-6` in `f32`.
    /// Dense basis solves cost `O(m²)` per pivot for `m` rows, with occasional
    /// `O(m³)` refactorizations; the number of pivots can be exponential.
    ///
    /// # Errors
    ///
    /// Returns input and numerical errors of [`crate::SimplexSolver::solve`],
    /// `DomainError` for verified infeasibility or unboundedness, and the
    /// errors of [`crate::ProgrammingResult::require_success`] for other unsuccessful stops.
    /// Reaching the iteration limit returns `ConvergenceFailure` with
    /// [`scioxide_core::ConvergenceReason::IterationLimit`].
    ///
    /// # Examples
    ///
    /// ```
    /// let result = scioxide_optimize::LpProblem::new(vec![1.0_f64])?.solve()?;
    /// assert!(result.x()[0].abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// Chvátal, *Linear Programming*, chapters 7–8, 1983.
    pub fn solve(&self) -> Result<crate::ProgrammingResult<S>, SciOxideError> {
        let result = crate::SimplexSolver::<S>::default().solve(self)?;
        result.require_success()?;
        Ok(result)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> QpProblem<S> {
    /// Solves this quadratic program with automatic method selection.
    ///
    /// Uses [`crate::QpSolver`] with 1000 iterations: factorization for an
    /// unconstrained problem, projected gradients for bounds, and an active set
    /// for general constraints. Dense factorizations cost `O(n³)` for `n` variables.
    /// Original-problem KKT tests use absolute and relative tolerances `1e-8`
    /// in `f64`, `1e-6` in `f32`; conditioning limits forward accuracy.
    ///
    /// # Errors
    ///
    /// Returns input and numerical errors of [`crate::QpSolver::solve`],
    /// `DomainError` for verified infeasibility or unboundedness, and the
    /// errors of [`crate::ProgrammingResult::require_success`] for other unsuccessful stops.
    ///
    /// # Examples
    ///
    /// ```
    /// let h = scioxide_linalg::Matrix::from_rows(&[[2.0_f64]])?;
    /// let result = scioxide_optimize::QpProblem::new(h, vec![-4.0])?.solve()?;
    /// assert!((result.x()[0] - 2.0).abs() < 1e-8);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// Nocedal and Wright, *Numerical Optimization*, 2nd ed., chapter 16.
    pub fn solve(&self) -> Result<crate::ProgrammingResult<S>, SciOxideError> {
        let result = crate::QpSolver::<S>::default().solve(self)?;
        result.require_success()?;
        Ok(result)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> MipProblem<S> {
    /// Solves this problem by branch and bound with default settings.
    ///
    /// Uses [`crate::BranchAndBound`] with 1000 nodes and 1000 LP iterations per
    /// node, reliability branching and presolve. The integrality tolerance is
    /// `√ε`; LP feasibility and optimality tolerances are `1e-8` in `f64`, `1e-6`
    /// in `f32`. Each node solves an LP; tree size can be exponential.
    ///
    /// # Errors
    ///
    /// Returns input and numerical errors of [`crate::BranchAndBound::solve`],
    /// `DomainError` for verified infeasibility or unboundedness, and the
    /// errors of [`crate::MipResult::require_success`] for other unsuccessful stops.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_optimize::{LpProblem, MipProblem};
    /// let lp = LpProblem::new(vec![1.0_f64])?;
    /// let result = MipProblem::new(lp, vec![0])?.solve()?;
    /// assert!(result.x()[0].abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// Land and Doig, *Econometrica* 28(3), 497–520, 1960.
    pub fn solve(&self) -> Result<crate::MipResult<S>, SciOxideError> {
        let result = crate::BranchAndBound::<S>::default().solve(self)?;
        result.require_success()?;
        Ok(result)
    }
}
