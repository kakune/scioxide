//! Basis views, injectable basis factorizations and bounded eta updates.
use super::standard_form::StandardForm;
use crate::context::SolveSession;
use alloc::vec::Vec;

use scioxide_core::backend::{checked_product, try_filled, try_with_capacity};
use scioxide_core::checked_float_quotient;
use scioxide_core::{
    AdjointOperator, ExecutionContext, ExecutionPolicy, LinearOperator, NumericScalar, RealScalar,
    SciOxideError,
};
use scioxide_linalg::{LinalgWorkspace, LuDecomposition, Matrix};
use scioxide_sparse::{
    DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder, SparseLu, SparseLuLimits, SparseLuSymbolic,
    SparseLuWorkspace, SparseOperation,
};

/// Borrows the square basis matrix of a simplex iteration.
///
/// Columns are standard-form columns of the basic variables. Artificial columns
/// are unit vectors and are not stored. Rows deleted as redundant are omitted,
/// and the remaining rows are renumbered from zero.
#[derive(Clone, Copy)]
pub struct BasisMatrix<'a, S: RealScalar + NumericScalar<Real = S>> {
    form: &'a StandardForm<S>,
    indices: &'a [usize],
    rows: &'a [usize],
    artificial_rows: &'a [usize],
    row_positions: &'a [usize],
}
impl<S: RealScalar + NumericScalar<Real = S>> BasisMatrix<'_, S> {
    /// Returns the order of the basis matrix.
    pub fn order(&self) -> usize {
        self.rows.len()
    }
    /// Returns the nonzero entries `(row, value)` of basis column `column`.
    ///
    /// # Errors
    ///
    /// Returns `OutOfBounds` if `column >= self.order()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::SciOxideError;
    /// use scioxide_optimize::BasisMatrix;
    ///
    /// fn stored_entries(basis: BasisMatrix<'_, f64>) -> Result<usize, SciOxideError> {
    ///     (0..basis.order()).try_fold(0, |count, column| {
    ///         Ok(count + basis.column_entries(column)?.count())
    ///     })
    /// }
    /// ```
    pub fn column_entries(
        &self,
        column: usize,
    ) -> Result<impl Iterator<Item = (usize, S)> + '_, SciOxideError> {
        if column >= self.order() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("basis_column"),
                column,
                self.order(),
            ));
        }
        let column = self.indices[column];
        let structural = (column < self.form.columns).then_some(column);
        let artificial =
            (column >= self.form.columns).then(|| self.artificial_rows[column - self.form.columns]);
        Ok(structural
            .into_iter()
            .flat_map(|c| self.form.column_entries(c))
            .chain(artificial.map(|row| (row, <S as scioxide_core::AlgebraicScalar>::ONE)))
            .filter_map(|(row, value)| {
                let position = self.row_positions[row];
                (position != usize::MAX).then_some((position, value))
            }))
    }
    /// Returns the coefficient at `(row, col)` without forming the basis matrix.
    ///
    /// # Errors
    ///
    /// Returns `OutOfBounds` if `row` or `col` is not below [`Self::order`].
    pub fn get(&self, row: usize, col: usize) -> Result<S, SciOxideError> {
        if row >= self.order() || col >= self.order() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("basis_coefficient"),
                row.max(col),
                self.order(),
            ));
        }
        let column = self.indices[col];
        Ok(if column < self.form.columns {
            self.form.coefficient(self.rows[row], column)
        } else if self.artificial_rows[column - self.form.columns] == self.rows[row] {
            <S as scioxide_core::AlgebraicScalar>::ONE
        } else {
            <S as scioxide_core::AlgebraicScalar>::ZERO
        })
    }
}
/// Factors simplex basis matrices and solves with them.
///
/// Install an implementation with [`SimplexWorkspace::with_basis_provider`](crate::SimplexWorkspace::with_basis_provider)
/// to replace the built-in LU factorization. An implementation may use a
/// direct factorization or an iterative solve, and manages its own storage.
/// [`DenseBasisProvider`] and [`SparseBasisProvider`] are the built-in choices.
pub trait BasisProvider<S: RealScalar + NumericScalar<Real = S>> {
    /// Factors the current basis matrix `basis`.
    ///
    /// No explicit inverse is required.
    ///
    /// # Errors
    ///
    /// Returns an error if the basis has an unsupported shape, storage cannot be
    /// allocated, the factorization fails, or `execution` stops the work.
    fn factor(
        &mut self,
        basis: BasisMatrix<'_, S>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>;
    /// Solves `B y = rhs`, or `Bᵀ y = rhs` when `transpose` is true, into `output`.
    ///
    /// `B` is the basis matrix of the last [`Self::factor`] call.
    ///
    /// # Errors
    ///
    /// Returns an error if no factorization is available, the lengths do not
    /// match its order, the solve fails numerically, or `execution` stops the
    /// work.
    fn solve_into(
        &mut self,
        rhs: &[S],
        output: &mut [S],
        transpose: bool,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>;
}
/// Factors simplex bases by dense LU with partial pivoting, up to a fixed order.
///
/// Solves with `B` and `Bᵀ` share one factorization.
pub struct DenseBasisProvider<S: RealScalar + NumericScalar<Real = S>> {
    maximum: usize,
    prepared_order: Option<usize>,
    identity: bool,
    matrix: Matrix<S>,
    factor: LuDecomposition<S>,
    work: LinalgWorkspace<'static, S>,
    scratch: Vec<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> DenseBasisProvider<S> {
    fn lazy(maximum: usize) -> Self {
        Self {
            maximum,
            prepared_order: None,
            identity: false,
            matrix: Matrix::empty(),
            factor: LuDecomposition::empty(),
            work: LinalgWorkspace::new(),
            scratch: Vec::new(),
        }
    }
    /// Creates a provider with storage for bases of order up to `maximum`.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// allocated.
    pub fn new(maximum: usize) -> Result<Self, SciOxideError> {
        let mut provider = Self::lazy(maximum);
        provider.matrix = Matrix::zeros(maximum, maximum)?;
        provider.factor.try_reserve(maximum)?;
        provider
            .work
            .try_reserve(&LuDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LuWorkspaceRequest::factor(
                    maximum,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                ),
            )?)?;
        provider.scratch = try_filled(
            checked_product(
                &[2, maximum],
                scioxide_core::quantity!("dense_basis_scratch"),
            )?,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("dense_basis_scratch"),
        )?;
        Ok(provider)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> BasisProvider<S> for DenseBasisProvider<S> {
    fn factor(
        &mut self,
        basis: BasisMatrix<'_, S>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = basis.order();
        if n > self.maximum {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("basis"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "basis exceeds prepared order",
            ));
        }
        execution.check()?;
        self.prepared_order = None;
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        self.matrix.try_resize_flat(n, n)?;
        self.matrix.as_mut_slice().fill(zero);
        crate::storage::resize(
            &mut self.scratch,
            checked_product(&[2, n], scioxide_core::quantity!("dense_basis_scratch"))?,
            zero,
        )?;
        let mut identity = true;
        for j in 0..n {
            let mut diagonal = false;
            for (i, value) in basis.column_entries(j)? {
                self.matrix.as_mut_slice()[i * n + j] = value;
                identity &= i == j && value == one;
                diagonal |= i == j && value == one;
            }
            identity &= diagonal;
        }
        if !identity {
            LuDecomposition::factor_into(
                &self.matrix,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut self.factor,
                &mut self.work,
                execution,
            )?;
            if self
                .factor
                .analyze_rank(scioxide_linalg::RankPolicy::Exact)?
                .rank()
                != n
            {
                return Err(SciOxideError::singular(None));
            }
        }
        self.identity = identity;
        self.prepared_order = Some(n);
        Ok(())
    }
    fn solve_into(
        &mut self,
        rhs: &[S],
        output: &mut [S],
        transpose: bool,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let order = self.prepared_order.ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "no successfully prepared basis",
            )
        })?;
        if rhs.len() != order || output.len() != order {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(order),
                scioxide_core::Dimensions::one(rhs.len().min(output.len())),
            ));
        }
        execution.check()?;
        let (candidate, scratch) = self.scratch.split_at_mut(order);
        if self.identity {
            if rhs.iter().any(|v| !v.is_finite()) {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    "nonfinite RHS",
                ));
            }
            candidate.copy_from_slice(rhs);
        } else if transpose {
            self.factor
                .inverse_operator()
                .apply_adjoint_into(rhs, candidate, scratch)?;
        } else {
            self.factor
                .inverse_operator()
                .apply_into(rhs, candidate, scratch)?;
        }
        execution.check()?;
        output.copy_from_slice(candidate);
        Ok(())
    }
}
/// Factors simplex bases by sparse LU, within explicit fill and storage limits.
///
/// When the sparsity pattern of the basis is unchanged, the symbolic analysis is
/// reused and only the numerical factorization is repeated.
pub struct SparseBasisProvider<S: RealScalar + NumericScalar<Real = S>> {
    limits: SparseLuLimits,
    factor: Option<SparseLu<S>>,
    work: Option<SparseLuWorkspace<S>>,
    scratch: Vec<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> SparseBasisProvider<S> {
    /// Creates a provider with the given limits on factor entries and workspace bytes.
    pub const fn new(limits: SparseLuLimits) -> Self {
        Self {
            limits,
            factor: None,
            work: None,
            scratch: Vec::new(),
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> BasisProvider<S> for SparseBasisProvider<S> {
    fn factor(
        &mut self,
        basis: BasisMatrix<'_, S>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = basis.order();
        let mut builder = SparseBuilder::new(n, n);
        for j in 0..n {
            for (i, value) in basis.column_entries(j)? {
                builder.push(i, j, value)?;
            }
        }
        let matrix = builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)?;
        if let (Some(factor), Some(work)) = (&mut self.factor, &mut self.work)
            && factor.symbolic().pattern() == matrix.structural_pattern()
        {
            return factor.refactor(
                matrix.view(),
                scioxide_sparse::SparseSolveOptions::new(),
                work,
                execution,
            );
        }
        let plan = SparseLuSymbolic::analyze(
            matrix.structural_pattern(),
            scioxide_sparse::SparseOrdering::default(),
            self.limits,
        )?;
        let work = (plan).workspace(())?;
        let factor = SparseLu::factor_scoped(
            plan,
            matrix.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            execution,
        )?;
        let len = checked_product(&[2, n], scioxide_core::quantity!("basis_scratch"))?;
        let extra = len.saturating_sub(self.scratch.len());
        scioxide_core::backend::try_reserve(
            &mut self.scratch,
            extra,
            scioxide_core::quantity!("basis_scratch"),
        )?;
        self.scratch
            .resize(len, <S as scioxide_core::AlgebraicScalar>::ZERO);
        self.factor = Some(factor);
        self.work = Some(work);
        Ok(())
    }
    fn solve_into(
        &mut self,
        rhs: &[S],
        output: &mut [S],
        transpose: bool,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let factor = self.factor.as_mut().ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "no prepared factor",
            )
        })?;
        factor.solve_into(
            rhs,
            if transpose {
                SparseOperation::Transpose
            } else {
                SparseOperation::Normal
            },
            scioxide_sparse::SparseSolveOptions::new(),
            output,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut self.scratch),
            execution,
        )
    }
}
/// Reborrow a borrowed provider for one nested call, shortening its object lifetime.
pub(super) fn lend<'s, S: RealScalar + NumericScalar<Real = S>>(
    provider: &'s mut Option<&mut (dyn BasisProvider<S> + Send)>,
) -> Option<&'s mut (dyn BasisProvider<S> + Send)> {
    provider
        .as_mut()
        .map(|provider| &mut **provider as &mut (dyn BasisProvider<S> + Send))
}
#[allow(
    clippy::large_enum_variant,
    reason = "Own provider state directly without an extra heap allocation"
)]
enum Backend<'a, S: RealScalar + NumericScalar<Real = S>> {
    Dense(DenseBasisProvider<S>),
    Sparse(SparseBasisProvider<S>),
    Injected(&'a mut (dyn BasisProvider<S> + Send)),
}
impl<S: RealScalar + NumericScalar<Real = S>> Backend<'_, S> {
    fn provider(&mut self) -> &mut dyn BasisProvider<S> {
        match self {
            Self::Dense(p) => p,
            Self::Sparse(p) => p,
            Self::Injected(p) => *p,
        }
    }
}
pub(super) struct Basis<'a, S: RealScalar + NumericScalar<Real = S>> {
    pub indices: Vec<usize>,
    pub rows: Vec<usize>,
    pub artificial_rows: Vec<usize>,
    membership: Vec<bool>,
    row_positions: Vec<usize>,
    pub generation: usize,
    backend: Backend<'a, S>,
    updates: Vec<S>,
    pivots: Vec<usize>,
    scratch: Vec<S>,
    max_updates: usize,
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> Basis<'a, S> {
    pub(super) fn new(
        form: &StandardForm<S>,
        max_updates: usize,
        limits: Option<SparseLuLimits>,
        provider: Option<&'a mut (dyn BasisProvider<S> + Send)>,
    ) -> Result<Self, SciOxideError> {
        let z = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let o = <S as scioxide_core::AlgebraicScalar>::ONE;
        let m = form.rows;
        let mut indices = try_with_capacity(m, scioxide_core::quantity!("simplex_basis_indices"))?;
        let mut rows = try_with_capacity(m, scioxide_core::quantity!("simplex_rows"))?;
        let mut artificial_rows =
            try_with_capacity(m, scioxide_core::quantity!("simplex_artificials"))?;
        let mut occurrences = try_filled(
            form.columns,
            0usize,
            scioxide_core::quantity!("basis_column_counts"),
        )?;
        for &column in &form.column_indices {
            occurrences[column] += 1;
        }
        for row in 0..m {
            rows.push(row);
            let unit = (form.transformed_variables..form.columns)
                .chain(0..form.transformed_variables)
                .find(|&col| {
                    occurrences[col] == 1
                        && form.coefficient(row, col) == o
                        && form.upper_bounds[col].is_none_or(|upper| form.rhs[row] <= upper)
                });
            indices.push(match unit {
                Some(j) => j,
                None => {
                    artificial_rows.push(row);
                    form.columns + artificial_rows.len() - 1
                }
            });
        }
        let backend = if let Some(provider) = provider {
            Backend::Injected(provider)
        } else if let Some(limits) = limits {
            Backend::Sparse(SparseBasisProvider::new(limits))
        } else {
            Backend::Dense(DenseBasisProvider::lazy(m))
        };
        let mut membership = try_filled(
            form.columns + artificial_rows.len(),
            false,
            scioxide_core::quantity!("basis_membership"),
        )?;
        for &column in &indices {
            membership[column] = true;
        }
        let row_positions = scioxide_core::backend::try_copy_slice(
            &rows,
            scioxide_core::quantity!("basis_row_positions"),
        )?;
        Ok(Self {
            indices,
            rows,
            artificial_rows,
            membership,
            row_positions,
            generation: 0,
            backend,
            updates: try_filled(
                checked_product(
                    &[m, max_updates],
                    scioxide_core::quantity!("basis_update_capacity"),
                )?,
                z,
                scioxide_core::quantity!("basis_updates"),
            )?,
            pivots: try_with_capacity(
                max_updates,
                scioxide_core::quantity!("basis_update_pivots"),
            )?,
            scratch: try_filled(m, z, scioxide_core::quantity!("basis_solve_scratch"))?,
            max_updates,
        })
    }
    pub(super) fn contains(&self, column: usize) -> bool {
        self.membership[column]
    }
    pub(super) fn column(&self, form: &StandardForm<S>, column: usize, output: &mut [S]) {
        output.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
        if column < form.columns {
            for (row, value) in form.column_entries(column) {
                let position = self.row_positions[row];
                if position != usize::MAX {
                    output[position] = value;
                }
            }
        } else {
            let position = self.row_positions[self.artificial_rows[column - form.columns]];
            if position != usize::MAX {
                output[position] = <S as scioxide_core::AlgebraicScalar>::ONE;
            }
        }
    }
    pub(super) fn column_dot(
        &self,
        form: &StandardForm<S>,
        column: usize,
        x: &[S],
    ) -> Result<S, SciOxideError> {
        if column >= form.columns {
            let position = self.row_positions[self.artificial_rows[column - form.columns]];
            return Ok(if position == usize::MAX {
                <S as scioxide_core::AlgebraicScalar>::ZERO
            } else {
                x[position]
            });
        }
        let entries = || {
            form.column_entries(column).filter_map(|(row, value)| {
                let position = self.row_positions[row];
                (position != usize::MAX).then(|| (value, x[position]))
            })
        };
        crate::utils::compensated_products(entries)
    }

    pub(super) fn refactor(
        &mut self,
        form: &StandardForm<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        self.membership.fill(false);
        for &column in &self.indices {
            self.membership[column] = true;
        }
        self.row_positions.fill(usize::MAX);
        for (position, &row) in self.rows.iter().enumerate() {
            self.row_positions[row] = position;
        }
        let basis = BasisMatrix {
            form,
            indices: &self.indices,
            rows: &self.rows,
            artificial_rows: &self.artificial_rows,
            row_positions: &self.row_positions,
        };
        session.invoke(scioxide_core::EvaluationRole::Factorization, || {
            self.backend.provider().factor(basis, session.execution()?)
        })?;
        self.pivots.clear();
        self.generation += 1;
        Ok(())
    }
    pub(super) fn solve_checked(
        &mut self,
        form: &StandardForm<S>,
        rhs: &[S],
        output: &mut [S],
        transpose: bool,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        if self.pivots.is_empty()
            && matches!(&self.backend,Backend::Dense(provider) if provider.identity)
        {
            return self.solve(rhs, output, transpose, session);
        }
        for attempt in 0..2 {
            self.solve(rhs, output, transpose, session)?;
            let basis = BasisMatrix {
                form,
                indices: &self.indices,
                rows: &self.rows,
                artificial_rows: &self.artificial_rows,
                row_positions: &self.row_positions,
            };
            let mut valid = true;
            for (i, &b) in rhs.iter().enumerate() {
                let mut residual = scioxide_core::ProductSum::new();
                let mut scale = scioxide_core::ProductSum::new();
                let one = <S as scioxide_core::AlgebraicScalar>::ONE;
                residual.add_product(-b, one)?;
                scale.add_product(NumericScalar::abs(b), one)?;
                for (j, &x) in output.iter().enumerate() {
                    let a = if transpose {
                        basis.get(j, i)?
                    } else {
                        basis.get(i, j)?
                    };
                    residual.add_product(a, x)?;
                    scale.add_product(NumericScalar::abs(a), NumericScalar::abs(x))?;
                }
                valid &= NumericScalar::abs(residual.finish()?)
                    <= S::SQRT_EPSILON
                        * scale.finish()?.max(
                            rhs.iter()
                                .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |a, b| {
                                    a.max(NumericScalar::abs(*b))
                                }),
                        );
            }
            if valid {
                return Ok(());
            }
            if attempt == 0 && !self.pivots.is_empty() {
                self.refactor(form, session)?;
            } else {
                break;
            }
        }
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "basis action failed its scaled backward-error check",
        ))
    }
    pub(super) fn solve(
        &mut self,
        rhs: &[S],
        output: &mut [S],
        transpose: bool,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.indices.len();
        self.scratch[..n].copy_from_slice(rhs);
        if transpose {
            for (k, &pivot) in self.pivots.iter().enumerate().rev() {
                let d = &self.updates[k * n..(k + 1) * n];
                let mut sum = scioxide_core::CompensatedSum::ZERO;
                let mut ordinary = sum.add(self.scratch[pivot]).is_ok();
                for (i, &v) in d.iter().enumerate() {
                    if i != pivot {
                        let product = -v * self.scratch[i];
                        if scioxide_core::backend::standard::product_needs_retry(
                            -v,
                            self.scratch[i],
                            product,
                        ) || sum.add(product).is_err()
                        {
                            ordinary = false;
                            break;
                        }
                    }
                }
                let numerator = if ordinary && let Ok(value) = sum.total() {
                    value
                } else {
                    let mut sum = scioxide_core::ProductSum::new();
                    sum.add_product(
                        self.scratch[pivot],
                        <S as scioxide_core::AlgebraicScalar>::ONE,
                    )?;
                    for (i, &v) in d.iter().enumerate() {
                        if i != pivot {
                            sum.add_product(-v, self.scratch[i])?;
                        }
                    }
                    sum.finish()?
                };
                self.scratch[pivot] = checked_float_quotient(numerator, d[pivot])?;
            }
            session.invoke(scioxide_core::EvaluationRole::LinearSolve, || {
                self.backend.provider().solve_into(
                    &self.scratch[..n],
                    output,
                    true,
                    session.execution()?,
                )
            })?;
        } else {
            session.invoke(scioxide_core::EvaluationRole::LinearSolve, || {
                self.backend
                    .provider()
                    .solve_into(rhs, output, false, session.execution()?)
            })?;
            for (k, &pivot) in self.pivots.iter().enumerate() {
                let d = &self.updates[k * n..(k + 1) * n];
                output[pivot] = checked_float_quotient(output[pivot], d[pivot])?;
                for (i, &v) in d.iter().enumerate() {
                    if i != pivot {
                        output[i] =
                            crate::utils::ordinary_trial_component(output[i], -output[pivot], v)?;
                    }
                }
            }
        }
        Ok(())
    }
    pub(super) fn replace(
        &mut self,
        row: usize,
        column: usize,
        direction: &[S],
        form: &StandardForm<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        self.membership[self.indices[row]] = false;
        self.membership[column] = true;
        self.indices[row] = column;
        // Exchanging identical columns changes the basic cost, not B itself.
        if direction.iter().enumerate().all(|(i, &v)| {
            v == if i == row {
                <S as scioxide_core::AlgebraicScalar>::ONE
            } else {
                <S as scioxide_core::AlgebraicScalar>::ZERO
            }
        }) {
            return Ok(());
        }
        if self.pivots.len() == self.max_updates {
            return self.refactor(form, session);
        }
        let n = self.indices.len();
        let offset = self.pivots.len() * n;
        self.updates[offset..offset + n].copy_from_slice(direction);
        self.pivots.push(row);
        Ok(())
    }
}
