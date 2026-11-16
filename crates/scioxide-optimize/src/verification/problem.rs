//! Borrowed original-coordinate verification inputs.
use super::{finite, invalid, shape};
use crate::{CoefficientMatrix, LinearConstraintSense, LpProblem, QpProblem};

use scioxide_core::{AdjointOperator, Bounds, NumericScalar, RealScalar, SciOxideError};

/// Borrows the data of an LP, `minimize cᵀx` subject to linear rows and bounds,
/// for [`ProgrammingVerifier`](crate::ProgrammingVerifier).
///
/// The constraint matrix may be any [`AdjointOperator`]; it must apply its true
/// transpose and return finite products. The data are used as given, without
/// scaling. Variables are free unless bounds are attached, except that
/// [`Self::from_problem`] keeps the `x ≥ 0` default of an [`LpProblem`] without
/// bounds.
#[derive(Clone, Copy)]
pub struct LpVerificationProblem<'a, S: RealScalar + NumericScalar<Real = S>> {
    pub(super) objective: &'a [S],
    pub(super) operator: Option<&'a dyn AdjointOperator<S>>,
    pub(super) rhs: &'a [S],
    pub(super) senses: &'a [LinearConstraintSense],
    pub(super) bounds: Option<&'a Bounds<S>>,
    pub(super) nonnegative: bool,
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> LpVerificationProblem<'a, S> {
    /// Borrows the objective `c` of an LP with free variables and no constraints.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `objective` is empty, or `NonFinite` if it
    /// contains NaN or infinity.
    pub fn new(objective: &'a [S]) -> Result<Self, SciOxideError> {
        if objective.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("objective"),
                "empty verification objective",
            ));
        }
        finite(objective)?;
        Ok(Self {
            objective,
            operator: None,
            rhs: &[],
            senses: &[],
            bounds: None,
            nonnegative: false,
        })
    }
    /// Attaches constraint rows `operator`, with one right-hand side and one sense per row.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if constraints are already attached.
    /// - `DimensionMismatch` if the operator, `rhs` and `senses` disagree in size.
    /// - `NonFinite` if `rhs` contains NaN or infinity.
    pub fn with_constraints(
        mut self,
        operator: &'a dyn AdjointOperator<S>,
        rhs: &'a [S],
        senses: &'a [LinearConstraintSense],
    ) -> Result<Self, SciOxideError> {
        if self.operator.is_some() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "duplicate verification constraints",
            ));
        }
        shape(self.dimension(), operator.cols())?;
        shape(operator.rows(), rhs.len())?;
        shape(rhs.len(), senses.len())?;
        finite(rhs)?;
        self.operator = Some(operator);
        self.rhs = rhs;
        self.senses = senses;
        Ok(self)
    }
    /// Attaches variable bounds.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if bounds are already attached, or
    /// `DimensionMismatch` if `bounds` has the wrong length.
    pub fn with_bounds(mut self, bounds: &'a Bounds<S>) -> Result<Self, SciOxideError> {
        if self.bounds.is_some() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "duplicate verification bounds",
            ));
        }
        shape(self.dimension(), bounds.as_slice().len())?;
        self.bounds = Some(bounds);
        self.nonnegative = false;
        Ok(self)
    }
    /// Borrows the data of `problem` without copying.
    pub fn from_problem(problem: &'a LpProblem<S>) -> Self {
        Self {
            objective: problem.objective(),
            operator: problem
                .constraints()
                .map(|c| c.coefficients() as &dyn AdjointOperator<S>),
            rhs: problem.constraints().map_or(&[], |c| c.rhs()),
            senses: problem.constraints().map_or(&[], |c| c.senses()),
            bounds: problem.bounds(),
            nonnegative: problem.bounds().is_none(),
        }
    }
    /// Returns the number of variables.
    pub fn dimension(&self) -> usize {
        self.objective.len()
    }
    /// Returns the number of constraint rows, not counting variable bounds.
    pub fn constraints_len(&self) -> usize {
        self.rhs.len()
    }
    pub(super) fn bound(&self, i: usize) -> (Option<S>, Option<S>) {
        self.bounds.map_or(
            (
                self.nonnegative
                    .then_some(<S as scioxide_core::AlgebraicScalar>::ZERO),
                None,
            ),
            |b| b.as_slice()[i],
        )
    }
}

#[derive(Clone, Copy)]
enum HessianCoefficients<'a, S: RealScalar + NumericScalar<Real = S>> {
    Owned(&'a CoefficientMatrix<S>),
    Dense(&'a scioxide_linalg::MatrixView<'a, S>),
    Sparse(&'a scioxide_sparse::CsrMatrix<S>),
}

/// Borrows the data of a QP, `minimize ½ xᵀQx + cᵀx` subject to linear rows and
/// bounds, for [`ProgrammingVerifier`](crate::ProgrammingVerifier).
///
/// The gradient is computed as `(Q + Qᵀ)x/2 + c`, so a nonsymmetric `Q` is read
/// as its symmetric part. Convexity can be checked only when the coefficients
/// of `Q` are available ([`Self::from_dense`], [`Self::from_sparse`],
/// [`Self::from_problem`]); a Hessian given as an operator ([`Self::new`]) gets
/// a `KktOnly` report at best.
#[derive(Clone, Copy)]
pub struct QpVerificationProblem<'a, S: RealScalar + NumericScalar<Real = S>> {
    pub(super) linear: LpVerificationProblem<'a, S>,
    hessian: crate::problem::BorrowedAdjointOperator<'a, S>,
    coefficients: Option<HessianCoefficients<'a, S>>,
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> QpVerificationProblem<'a, S> {
    /// Borrows a Hessian operator and the linear data of the QP.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `hessian` is not square of the dimension of `linear`.
    pub fn new(
        hessian: &'a dyn AdjointOperator<S>,
        linear: LpVerificationProblem<'a, S>,
    ) -> Result<Self, SciOxideError> {
        shape(linear.dimension(), hessian.rows())?;
        shape(linear.dimension(), hessian.cols())?;
        Ok(Self {
            linear,
            hessian: crate::problem::BorrowedAdjointOperator::Other(hessian),
            coefficients: None,
        })
    }
    /// Borrows a dense Hessian and the linear data of the QP.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `hessian` is not square of the dimension of
    /// `linear`, or `NonFinite` if it contains NaN or infinity.
    pub fn from_dense(
        hessian: &'a scioxide_linalg::MatrixView<'a, S>,
        linear: LpVerificationProblem<'a, S>,
    ) -> Result<Self, SciOxideError> {
        shape(linear.dimension(), hessian.rows())?;
        shape(linear.dimension(), hessian.cols())?;
        Ok(Self {
            linear,
            hessian: crate::problem::BorrowedAdjointOperator::Dense(hessian.try_finite()?),
            coefficients: Some(HessianCoefficients::Dense(hessian)),
        })
    }
    /// Borrows a CSR Hessian and the linear data of the QP, without copying.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `hessian` is not square of the dimension of `linear`.
    pub fn from_sparse(
        hessian: &'a scioxide_sparse::CsrMatrix<S>,
        linear: LpVerificationProblem<'a, S>,
    ) -> Result<Self, SciOxideError> {
        let mut problem = Self::new(hessian, linear)?;
        problem.coefficients = Some(HessianCoefficients::Sparse(hessian));
        Ok(problem)
    }
    pub(super) fn hessian(&self) -> &dyn AdjointOperator<S> {
        self.hessian.as_operator()
    }
    pub(super) fn inspectable(&self) -> bool {
        self.coefficients.is_some()
    }
    pub(super) fn coefficient(&self, i: usize, j: usize) -> Result<S, SciOxideError> {
        match self.coefficients {
            Some(HessianCoefficients::Owned(m)) => m.get(i, j),
            Some(HessianCoefficients::Dense(m)) => m.get(i, j),
            Some(HessianCoefficients::Sparse(m)) => m.get(i, j),
            None => Err(invalid(
                scioxide_core::argument!("problem"),
                "operator Hessian has no inspectable coefficients",
            )),
        }
    }
    /// Borrows the data of `problem` without copying.
    pub fn from_problem(problem: &'a QpProblem<S>) -> Self {
        Self {
            linear: LpVerificationProblem {
                objective: problem.linear_objective(),
                operator: problem
                    .constraints()
                    .map(|c| c.coefficients() as &dyn AdjointOperator<S>),
                rhs: problem.constraints().map_or(&[], |c| c.rhs()),
                senses: problem.constraints().map_or(&[], |c| c.senses()),
                bounds: problem.bounds(),
                nonnegative: false,
            },
            hessian: crate::problem::BorrowedAdjointOperator::Other(problem.hessian()),
            coefficients: Some(HessianCoefficients::Owned(problem.hessian())),
        }
    }
    /// Returns the linear part of the problem.
    pub const fn linear(&self) -> &LpVerificationProblem<'a, S> {
        &self.linear
    }
}

/// Explicitly sample a stable linear operator's columns into bounded sparse storage.
fn sample_columns<S: RealScalar + NumericScalar<Real = S>>(
    operator: &dyn AdjointOperator<S>,
    maximum_entries: usize,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<scioxide_sparse::CsrMatrix<S>, SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let mut input = scioxide_core::backend::try_filled(
        operator.cols(),
        zero,
        scioxide_core::quantity!("operator_column_input"),
    )?;
    let mut output = scioxide_core::backend::try_filled(
        operator.rows(),
        zero,
        scioxide_core::quantity!("operator_column_output"),
    )?;
    let mut scratch = scioxide_core::backend::try_filled(
        operator.apply_workspace_len()?,
        zero,
        scioxide_core::quantity!("operator_column_scratch"),
    )?;
    let mut builder = scioxide_sparse::SparseBuilder::new(operator.rows(), operator.cols());
    let mut entries = 0;
    for column in 0..operator.cols() {
        input[column] = one;
        session.invoke(scioxide_core::EvaluationRole::LinearOperator, || {
            operator.apply_into(&input, &mut output, &mut scratch)
        })?;
        finite(&output)?;
        input[column] = zero;
        for (row, &value) in output.iter().enumerate() {
            if value != zero {
                if entries == maximum_entries {
                    return Err(SciOxideError::resource_limit(
                        scioxide_core::quantity!("operator_entries"),
                        entries.saturating_add(1),
                        maximum_entries,
                    ));
                }
                builder.push(row, column, value)?;
                entries += 1;
            }
        }
    }
    builder.finish_csr(
        scioxide_sparse::DuplicatePolicy::Reject,
        scioxide_sparse::ExplicitZeroPolicy::Preserve,
    )
}
impl<S: RealScalar + NumericScalar<Real = S>> LpProblem<S> {
    /// Creates an LP with sparse coefficients from borrowed LP data.
    ///
    /// When the constraints are an operator, it is applied once per variable to
    /// sample its columns, so it must be linear and give the same result on
    /// every call. Bounds and row senses are copied. `maximum_entries` limits the
    /// number of stored nonzeros; it is not a limit on work.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the matrix has more than `maximum_entries` nonzeros.
    /// - `NonFinite` if the operator returns NaN or infinity.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    /// - Errors returned by the operator.
    pub fn from_view(
        view: &LpVerificationProblem<'_, S>,
        maximum_entries: usize,
    ) -> Result<Self, SciOxideError> {
        let identity = crate::context::identity(
            scioxide_core::algorithm_id!("optimize", "LP-column-preparation", 1),
            scioxide_core::ProblemToken::new(0),
            view.dimension(),
            maximum_entries as u128,
        )?;
        crate::context::sequential(identity, scioxide_core::Budget::ONE, |context| {
            let session =
                crate::context::SolveSession::new(context, scioxide_core::argument!("context"));
            prepare_lp(view, maximum_entries, &session)
        })
    }
}
fn prepare_lp<S: RealScalar + NumericScalar<Real = S>>(
    view: &LpVerificationProblem<'_, S>,
    maximum_entries: usize,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<LpProblem<S>, SciOxideError> {
    use scioxide_core::backend::{try_copy_slice, try_with_capacity};
    let mut problem = LpProblem::new(try_copy_slice(
        view.objective,
        scioxide_core::quantity!("borrowed_lp_objective"),
    )?)?;
    if let Some(operator) = view.operator {
        problem = problem.with_constraints(crate::LinearConstraints::new(
            sample_columns(operator, maximum_entries, session)?,
            try_copy_slice(view.rhs, scioxide_core::quantity!("borrowed_lp_rhs"))?,
            try_copy_slice(view.senses, scioxide_core::quantity!("borrowed_lp_senses"))?,
        )?)?;
    }
    let mut bounds = try_with_capacity(
        view.dimension(),
        scioxide_core::quantity!("borrowed_lp_bounds"),
    )?;
    bounds.extend((0..view.dimension()).map(|i| view.bound(i)));
    problem.with_bounds(Bounds::new(bounds)?)
}
impl<S: RealScalar + NumericScalar<Real = S>> QpProblem<S> {
    /// Creates a QP with sparse coefficients from borrowed QP data.
    ///
    /// Operators for the constraints and the Hessian are applied once per
    /// variable to sample their columns, so they must be linear and give the
    /// same result on every call. `symmetry` says how to read an asymmetric
    /// Hessian. Each matrix may have at most `maximum_entries` stored nonzeros;
    /// this is not a limit on work. Convexity is checked later by
    /// [`QpSolver`](crate::QpSolver).
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if a matrix has more than `maximum_entries` nonzeros.
    /// - `InvalidInput` if the Hessian is not symmetric as `symmetry` requires.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    /// - Errors returned by an operator.
    pub fn from_view(
        view: &QpVerificationProblem<'_, S>,
        symmetry: crate::HessianSymmetry,
        maximum_entries: usize,
    ) -> Result<Self, SciOxideError> {
        let identity = crate::context::identity(
            scioxide_core::algorithm_id!("optimize", "QP-column-preparation", 1),
            scioxide_core::ProblemToken::new(0),
            view.linear.dimension(),
            maximum_entries as u128,
        )?;
        crate::context::sequential(identity, scioxide_core::Budget::ONE, |context| {
            let session =
                crate::context::SolveSession::new(context, scioxide_core::argument!("context"));
            let linear = prepare_lp(&view.linear, maximum_entries, &session)?;
            let mut problem = QpProblem::new_with_symmetry(
                sample_columns(view.hessian(), maximum_entries, &session)?,
                scioxide_core::backend::try_copy_slice(
                    linear.objective(),
                    scioxide_core::quantity!("borrowed_qp_linear_objective"),
                )?,
                symmetry,
            )?;
            if let Some(c) = linear.constraints() {
                problem = problem.with_constraints(c.clone())?;
            }
            if let Some(b) = linear.bounds() {
                problem = problem.with_bounds(b.clone())?;
            }
            Ok(problem)
        })
    }
}
