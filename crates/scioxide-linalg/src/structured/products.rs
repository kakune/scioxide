//! Kronecker and block diagonal operators built from borrowed operators, and Khatri–Rao products.
use super::direct::{admit, check_apply, identity, narrow, size, wide};
use crate::kernel::CANDIDATE;
use crate::{
    DirectSolveTermination, LinalgExecutionContext, LinalgWorkspace, Matrix, MatrixView,
    MatrixViewMut, VectorView, VectorViewMut,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AdjointOperator, BlockOperator, LinearOperator, NumericScalar, RealScalar, SciOxideError,
    Tolerance,
};

/// Options for [`KroneckerOperator::solve_into`] and [`BlockDiagonalOperator::solve_into`].
///
/// With the tolerance `(atol, rtol)`, a solution `x` of `A x = b` is accepted when
///
/// ```text
/// ‖b − A x‖∞ ≤ atol · s + rtol · (‖b‖∞ + ‖A x‖∞),   s = max(‖b‖∞, ‖A x‖∞)
/// ```
///
/// The default tolerance is [`Tolerance::from_sqrt_epsilon`]. The execution
/// options set the byte limit of the workspace, the iteration budget, and
/// the execution policy, accuracy and time limit of the operator
/// applications.
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_linalg::CompositeSolveOptions;
/// let options = CompositeSolveOptions::<f64>::new()
///     .with_tolerance(Tolerance::new(0.0, 1e-12)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct CompositeSolveOptions<R: RealScalar> {
    tolerance: Tolerance<R>,
    execution: crate::LinalgExecutionOptions,
}
impl<R: RealScalar> CompositeSolveOptions<R> {
    /// Returns the default options.
    pub fn new() -> Self {
        Self {
            tolerance: Tolerance::from_sqrt_epsilon(),
            execution: crate::LinalgExecutionOptions::new(),
        }
    }
    /// Returns the execution options.
    pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
        self.execution
    }
    /// Sets the execution options.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }
    /// Sets the residual tolerance.
    pub const fn with_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Returns the residual tolerance.
    pub const fn tolerance(&self) -> Tolerance<R> {
        self.tolerance
    }
}
impl<R: RealScalar> Default for CompositeSolveOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Acceptance and backward error of a Kronecker or block diagonal solve.
///
/// The backward error is `‖b − A x‖∞ / (‖b‖∞ + ‖A x‖∞)` for the original
/// operator `A`, and zero when both norms vanish. The acceptance test is
/// described on [`CompositeSolveOptions`].
///
/// ```
/// use scioxide_linalg::{CompositeSolveReport, DirectSolveTermination};
/// fn accepted(report: CompositeSolveReport<f64>) -> bool {
///     report.termination() == DirectSolveTermination::Accepted
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct CompositeSolveReport<R: RealScalar> {
    termination: DirectSolveTermination,
    backward_error: R,
}

impl<R: RealScalar> CompositeSolveReport<R> {
    /// Returns [`DirectSolveTermination::Accepted`] or
    /// [`DirectSolveTermination::ResidualRejected`].
    pub const fn termination(self) -> DirectSolveTermination {
        self.termination
    }

    /// Returns the backward error, also for a rejected solve.
    pub const fn backward_error(self) -> R {
        self.backward_error
    }
}

fn sum(parts: &[usize]) -> Result<usize, SciOxideError> {
    parts.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("operator_product"))
        })
    })
}
struct Adjoint<'a, T: NumericScalar>(&'a (dyn AdjointOperator<T> + Sync));
impl<T: NumericScalar> LinearOperator<T> for Adjoint<'_, T> {
    fn rows(&self) -> usize {
        self.0.cols()
    }
    fn cols(&self) -> usize {
        self.0.rows()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.0.apply_adjoint_workspace_len()
    }
    fn apply_into(&self, x: &[T], y: &mut [T], w: &mut [T]) -> Result<(), SciOxideError> {
        self.0
            .apply_adjoint_into(x, y, &mut w[..self.0.apply_adjoint_workspace_len()?])
    }
    fn apply_into_scoped(
        &self,
        x: &[T],
        y: &mut [T],
        w: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.0.apply_adjoint_into_scoped(
            x,
            y,
            &mut w[..self.0.apply_adjoint_workspace_len()?],
            context,
        )
    }
}
/// Kronecker product `A ⊗ B` of two borrowed operators, applied without forming it.
///
/// For `A` of size `m × n` and `B` of size `p × q`, `A ⊗ B` has size
/// `mp × nq`, and its entry in row `i p + r` and column `k q + l` is
/// `A[i][k] B[r][l]`: the index of `B` varies fastest. A product is computed
/// as
///
/// ```text
/// (A ⊗ B) vec(X) = vec(B X Aᵀ)
/// ```
///
/// where `vec` stacks columns and `X` is `q × n`. This takes `n` products
/// with `B` and `p` products with `A`.
///
/// `A` and `B` can be any [`AdjointOperator`]: dense, sparse, structured or
/// matrix-free. Their dimensions and workspace sizes must not change while
/// they are borrowed. The operator implements [`LinearOperator`] and
/// [`AdjointOperator`].
pub struct KroneckerOperator<'a, T: NumericScalar> {
    a: &'a (dyn AdjointOperator<T> + Sync),
    b: &'a (dyn AdjointOperator<T> + Sync),
    rows: usize,
    cols: usize,
}
impl<'a, T: NumericScalar> KroneckerOperator<'a, T> {
    /// Creates the Kronecker product `a ⊗ b`.
    ///
    /// # Errors
    /// - `SizeOverflow` if `m p` or `n q` overflows `usize`.
    pub fn new(
        a: &'a (dyn AdjointOperator<T> + Sync),
        b: &'a (dyn AdjointOperator<T> + Sync),
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            a,
            b,
            rows: size(
                a.rows(),
                b.rows(),
                scioxide_core::quantity!("kronecker_rows"),
            )?,
            cols: size(
                a.cols(),
                b.cols(),
                scioxide_core::quantity!("kronecker_columns"),
            )?,
        })
    }
    /// Candidate scratch elements of [`Self::solve_into`] with these inverse
    /// factors; requested through [`crate::OperatorWorkspaceRequest::kronecker_solve`].
    pub(crate) fn solve_scratch(
        &self,
        inverses: [&dyn LinearOperator<T>; 2],
    ) -> Result<usize, SciOxideError> {
        let [a_inverse, b_inverse] = inverses;
        self.check_inverses(a_inverse, b_inverse)?;
        sum(&[
            self.rows,
            self.rows,
            self.rows,
            kron_len(a_inverse, b_inverse)?.max(self.apply_workspace_len()?),
        ])
    }
    fn check_inverses(
        &self,
        a: &dyn LinearOperator<T>,
        b: &dyn LinearOperator<T>,
    ) -> Result<(), SciOxideError> {
        [(self.a, a), (self.b, b)]
            .into_iter()
            .try_for_each(|(factor, inverse)| square_inverse(factor, inverse))
    }
    /// Solves `(A ⊗ B) x = b` with inverses of `A` and `B` that you provide.
    ///
    /// The residual of the result is checked before it is accepted.
    ///
    /// Since `(A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹`, the solve applies `inverses = [A⁻¹, B⁻¹]`
    /// as a Kronecker product. The inverses are typically the inverse
    /// operators of prepared factorizations, such as
    /// [`LuDecomposition::inverse_operator`](crate::LuDecomposition::inverse_operator).
    /// No dense matrix is formed. The result is multiplied by `A ⊗ B` and
    /// accepted if the residual passes the tolerance in `options`. A rejected
    /// solve returns `Ok` with [`DirectSolveTermination::ResidualRejected`] and
    /// does not write `out`. Applying the inverses and checking the residual
    /// count one iteration each toward the iteration budget. The workspace is
    /// sized by
    /// [`OperatorWorkspaceRequest::kronecker_solve`](crate::OperatorWorkspaceRequest::kronecker_solve).
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `A` or `B` is not square, an inverse does not
    ///   have the shape of its factor, or `rhs` or `out` does not have length
    ///   `m p`.
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    /// - `NonFinite` if `rhs` contains NaN or an infinity, or an operator
    ///   returns a non-finite value.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Any error returned by the operators.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn solve_into<'rhs, 'scope, 'capability>(
        &self,
        inverses: [&dyn LinearOperator<T>; 2],
        rhs: impl Into<VectorView<'rhs, T>>,
        options: &CompositeSolveOptions<T::Real>,
        mut out: VectorViewMut<'_, T>,
        workspace: &mut LinalgWorkspace<'_, T>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<CompositeSolveReport<T::Real>, SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(&options.execution)?;
        context.execution().check()?;
        let rhs = rhs.into();
        let requirements = LinalgWorkspace::<T>::workspace_requirements(
            &crate::OperatorWorkspaceRequest::kronecker_solve(self, inverses, &options.execution)?,
        )?;
        let [a_inverse, b_inverse] = inverses;
        if rhs.len() != self.rows || out.len() != self.cols {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows, self.cols],
                [rhs.len(), out.len()],
            ));
        }
        for value in rhs.iter() {
            crate::checked::validate_input(value, "Kronecker right-hand side")?;
        }
        workspace.try_reserve(&requirements)?;
        let len = requirements.scalar(CANDIDATE);
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "kronecker_solve", 1),
                self.rows,
            ),
            options.execution.solve_budgets(),
            |workspace, kernels, scope| {
                let scratch = &mut workspace.scalar_mut(CANDIDATE)[..len];
                // `(A (x) B)^{-1} = A^{-1} (x) B^{-1}`, so the solve is a
                // Kronecker product of the caller-supplied inverses. Those
                // inverses are untrusted approximations; the candidate is
                // multiplied back by `A (x) B` and accepted only if the
                // residual passes the tolerance.
                let (input, rest) = crate::workspace::split_mut(scratch, self.rows)?;
                for (value, entry) in input.iter_mut().zip(rhs.iter()) {
                    *value = entry;
                }
                let (candidate, rest) = crate::workspace::split_mut(rest, self.rows)?;
                let (check, work) = crate::workspace::split_mut(rest, self.rows)?;
                if self.rows != 0 {
                    admit(kernels.execution(), scope)?;
                }
                kron_apply(
                    a_inverse,
                    b_inverse,
                    input,
                    candidate,
                    work,
                    Some(&mut scioxide_core::OperatorContext::with_kernels(
                        scope, &kernels,
                    )),
                )
                .and_then(scioxide_core::Invocation::require)?;
                if self.rows != 0 {
                    admit(kernels.execution(), scope)?;
                }
                self.apply_into_scoped(
                    candidate,
                    check,
                    work,
                    &mut scioxide_core::OperatorContext::with_kernels(scope, &kernels),
                )
                .and_then(scioxide_core::Invocation::require)?;
                let report = assess_residual(input, check, options.tolerance)?;
                kernels.execution().check()?;
                if report.termination.is_success() {
                    for (i, &value) in candidate.iter().enumerate() {
                        out.set_element(i, value);
                    }
                }
                Ok(report)
            },
        )
    }
}
fn kron_len<T: NumericScalar>(
    a: &dyn LinearOperator<T>,
    b: &dyn LinearOperator<T>,
) -> Result<usize, SciOxideError> {
    if a.rows() == 0 || a.cols() == 0 || b.rows() == 0 || b.cols() == 0 {
        return Ok(0);
    }
    sum(&[
        size(
            a.cols(),
            b.rows(),
            scioxide_core::quantity!("kronecker_workspace"),
        )?,
        size(
            a.rows(),
            b.rows(),
            scioxide_core::quantity!("kronecker_workspace"),
        )?,
        a.cols(),
        a.rows(),
        a.apply_workspace_len()?.max(b.apply_workspace_len()?),
    ])
}
fn kron_apply<T: NumericScalar>(
    a: &dyn LinearOperator<T>,
    b: &dyn LinearOperator<T>,
    x: &[T],
    y: &mut [T],
    w: &mut [T],
    mut context: Option<&mut scioxide_core::OperatorContext<'_, '_, '_, T>>,
) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
    let rows = size(
        a.rows(),
        b.rows(),
        scioxide_core::quantity!("kronecker_rows"),
    )?;
    let cols = size(
        a.cols(),
        b.cols(),
        scioxide_core::quantity!("kronecker_columns"),
    )?;
    check_apply([cols, rows, kron_len(a, b)?], x.len(), y.len(), w.len())?;
    crate::checked::validate_inputs(x)?;
    if rows == 0 || cols == 0 {
        y.fill(T::ZERO);
        return Ok(scioxide_core::Invocation::Evaluated(()));
    }
    let (temp, w) = crate::workspace::split_mut(
        w,
        size(
            a.cols(),
            b.rows(),
            scioxide_core::quantity!("kronecker_workspace"),
        )?,
    )?;
    let (stage, w) = crate::workspace::split_mut(w, rows)?;
    let (input, w) = crate::workspace::split_mut(w, a.cols())?;
    let (output, w) = crate::workspace::split_mut(w, a.rows())?;
    // `(A (x) B) vec(X) = vec(B X A^T)` without forming the Kronecker
    // product: x is viewed as `a.cols()` blocks of length `b.cols()`. First B
    // is applied to every block (`temp = B X`), then A is applied to every
    // strided "row" gathered across blocks. The result is staged and copied
    // out only after the final finiteness check.
    for k in 0..a.cols() {
        let outcome = match context.as_deref_mut() {
            Some(context) => b.apply_into_scoped(
                &x[k * b.cols()..(k + 1) * b.cols()],
                &mut temp[k * b.rows()..(k + 1) * b.rows()],
                &mut w[..b.apply_workspace_len()?],
                context,
            ),
            None => b
                .apply_into(
                    &x[k * b.cols()..(k + 1) * b.cols()],
                    &mut temp[k * b.rows()..(k + 1) * b.rows()],
                    &mut w[..b.apply_workspace_len()?],
                )
                .map(scioxide_core::Invocation::Evaluated),
        }?;
        if !outcome.is_evaluated() {
            return Ok(outcome);
        }
    }
    crate::checked::provider_output(temp)?;
    for i in 0..b.rows() {
        for (k, v) in input.iter_mut().enumerate() {
            *v = temp[k * b.rows() + i];
        }
        let outcome = match context.as_deref_mut() {
            Some(context) => {
                a.apply_into_scoped(input, output, &mut w[..a.apply_workspace_len()?], context)
            }
            None => a
                .apply_into(input, output, &mut w[..a.apply_workspace_len()?])
                .map(scioxide_core::Invocation::Evaluated),
        }?;
        if !outcome.is_evaluated() {
            return Ok(outcome);
        }
        for (k, &v) in output.iter().enumerate() {
            stage[k * b.rows() + i] = v;
        }
    }
    crate::checked::provider_output(stage)?;
    y.copy_from_slice(stage);
    Ok(scioxide_core::Invocation::Evaluated(()))
}
impl<T: NumericScalar> LinearOperator<T> for KroneckerOperator<'_, T> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        kron_len(self.a, self.b)
    }
    fn apply_into(&self, x: &[T], y: &mut [T], w: &mut [T]) -> Result<(), SciOxideError> {
        kron_apply(self.a, self.b, x, y, w, None).and_then(scioxide_core::Invocation::require)
    }
    fn apply_into_scoped(
        &self,
        x: &[T],
        y: &mut [T],
        w: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        kron_apply(self.a, self.b, x, y, w, Some(context))
    }
}
impl<T: NumericScalar> AdjointOperator<T> for KroneckerOperator<'_, T> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        kron_len(&Adjoint(self.a), &Adjoint(self.b))
    }
    fn apply_adjoint_into(&self, x: &[T], y: &mut [T], w: &mut [T]) -> Result<(), SciOxideError> {
        kron_apply(&Adjoint(self.a), &Adjoint(self.b), x, y, w, None)
            .and_then(scioxide_core::Invocation::require)
    }
    fn apply_adjoint_into_scoped(
        &self,
        x: &[T],
        y: &mut [T],
        w: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        kron_apply(&Adjoint(self.a), &Adjoint(self.b), x, y, w, Some(context))
    }
}
impl<T: NumericScalar> BlockOperator<T> for KroneckerOperator<'_, T> {}
/// Block diagonal operator `diag(A₁, …, A_k)` of borrowed blocks, applied without forming it.
///
/// Blocks may be rectangular or empty. With block `Aᵢ` of size `mᵢ × nᵢ`,
/// the operator has `Σ mᵢ` rows and `Σ nᵢ` columns, and block `i` maps the
/// `nᵢ` entries of `x` after those of the earlier blocks to the
/// corresponding `mᵢ` entries of `y`. The operator implements
/// [`LinearOperator`] and [`AdjointOperator`].
pub struct BlockDiagonalOperator<'a, T: NumericScalar> {
    blocks: &'a [&'a (dyn AdjointOperator<T> + Sync)],
    rows: usize,
    cols: usize,
}
impl<'a, T: NumericScalar> BlockDiagonalOperator<'a, T> {
    /// Creates the block diagonal operator of `blocks`, without copying the list.
    ///
    /// # Errors
    /// - `SizeOverflow` if the row or column sum overflows `usize`.
    pub fn new(blocks: &'a [&'a (dyn AdjointOperator<T> + Sync)]) -> Result<Self, SciOxideError> {
        let (rows, cols) = blocks.iter().try_fold((0, 0), |(r, c), b| {
            Ok::<_, SciOxideError>((sum(&[r, b.rows()])?, sum(&[c, b.cols()])?))
        })?;
        Ok(Self { blocks, rows, cols })
    }
    /// Candidate scratch elements of [`Self::solve_into`] with these inverse
    /// blocks; requested through
    /// [`crate::OperatorWorkspaceRequest::block_diagonal_solve`].
    pub(crate) fn solve_scratch(
        &self,
        inverses: &[&dyn LinearOperator<T>],
    ) -> Result<usize, SciOxideError> {
        if inverses.len() != self.blocks.len() {
            return Err(SciOxideError::dimension_mismatch(
                [self.blocks.len()],
                [inverses.len()],
            ));
        }
        let mut work = self.apply_workspace_len()?;
        for (&block, &inverse) in self.blocks.iter().zip(inverses) {
            square_inverse(block, inverse)?;
            if inverse.rows() != 0 {
                work = work.max(inverse.apply_workspace_len()?);
            }
        }
        sum(&[self.rows, self.rows, self.rows, work])
    }
    /// Solves `diag(A₁, …, A_k) x = b` block by block with inverses you provide.
    ///
    /// The residual of the result is checked before it is accepted.
    ///
    /// `inverses[i]` applies `Aᵢ⁻¹`, so every block must be square. The
    /// inverses are typically the inverse operators of prepared
    /// factorizations. The result is multiplied by the block diagonal operator
    /// and accepted if the residual passes the tolerance in `options`. A
    /// rejected solve returns `Ok` with
    /// [`DirectSolveTermination::ResidualRejected`] and does not write `out`.
    /// Each nonempty inverse block and the residual check count one iteration
    /// each toward the iteration budget. The workspace is sized by
    /// [`OperatorWorkspaceRequest::block_diagonal_solve`](crate::OperatorWorkspaceRequest::block_diagonal_solve).
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # Errors
    /// - `DimensionMismatch` if the number of inverses differs from the number
    ///   of blocks, a block is not square, an inverse does not have the shape of
    ///   its block, or `rhs` or `out` has the wrong length.
    /// - `SizeOverflow` if the workspace size overflows `usize`.
    /// - `NonFinite` if `rhs` contains NaN or an infinity, or an operator
    ///   returns a non-finite value.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Any error returned by the operators.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn solve_into<'rhs, 'scope, 'capability>(
        &self,
        inverses: &[&dyn LinearOperator<T>],
        rhs: impl Into<VectorView<'rhs, T>>,
        options: &CompositeSolveOptions<T::Real>,
        mut out: VectorViewMut<'_, T>,
        workspace: &mut LinalgWorkspace<'_, T>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<CompositeSolveReport<T::Real>, SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(&options.execution)?;
        context.execution().check()?;
        let rhs = rhs.into();
        let requirements = LinalgWorkspace::<T>::workspace_requirements(
            &crate::OperatorWorkspaceRequest::block_diagonal_solve(
                self,
                inverses,
                &options.execution,
            )?,
        )?;
        if rhs.len() != self.rows || out.len() != self.cols {
            return Err(SciOxideError::dimension_mismatch(
                [self.rows, self.cols],
                [rhs.len(), out.len()],
            ));
        }
        for value in rhs.iter() {
            crate::checked::validate_input(value, "block-diagonal right-hand side")?;
        }
        workspace.try_reserve(&requirements)?;
        let len = requirements.scalar(CANDIDATE);
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "block_diagonal_solve", 1),
                self.rows,
            ),
            options.execution.solve_budgets(),
            |workspace, kernels, scope| {
                let scratch = &mut workspace.scalar_mut(CANDIDATE)[..len];
                let (input, rest) = crate::workspace::split_mut(scratch, self.rows)?;
                for (value, entry) in input.iter_mut().zip(rhs.iter()) {
                    *value = entry;
                }
                let (candidate, rest) = crate::workspace::split_mut(rest, self.rows)?;
                let (check, work) = crate::workspace::split_mut(rest, self.rows)?;
                let mut offset = 0;
                for inverse in inverses.iter().filter(|inverse| inverse.rows() != 0) {
                    let end = offset + inverse.rows();
                    admit(kernels.execution(), scope)?;
                    inverse
                        .apply_into_scoped(
                            &input[offset..end],
                            &mut candidate[offset..end],
                            &mut work[..inverse.apply_workspace_len()?],
                            &mut scioxide_core::OperatorContext::with_kernels(scope, &kernels),
                        )
                        .and_then(scioxide_core::Invocation::require)?;
                    offset = end;
                }
                crate::checked::provider_output(candidate)?;
                if self.rows != 0 {
                    admit(kernels.execution(), scope)?;
                }
                self.apply_into_scoped(
                    candidate,
                    check,
                    work,
                    &mut scioxide_core::OperatorContext::with_kernels(scope, &kernels),
                )
                .and_then(scioxide_core::Invocation::require)?;
                let report = assess_residual(input, check, options.tolerance)?;
                kernels.execution().check()?;
                if report.termination.is_success() {
                    for (i, &value) in candidate.iter().enumerate() {
                        out.set_element(i, value);
                    }
                }
                Ok(report)
            },
        )
    }
    fn len(&self, adj: bool) -> Result<usize, SciOxideError> {
        if self.rows == 0 || self.cols == 0 {
            return Ok(0);
        }
        let work = self.blocks.iter().try_fold(0usize, |n, b| {
            if b.rows() == 0 || b.cols() == 0 {
                return Ok(n);
            }
            Ok::<_, SciOxideError>(n.max(if adj {
                b.apply_adjoint_workspace_len()?
            } else {
                b.apply_workspace_len()?
            }))
        })?;
        sum(&[if adj { self.cols } else { self.rows }, work])
    }
    fn apply(
        &self,
        adj: bool,
        x: &[T],
        y: &mut [T],
        w: &mut [T],
        mut context: Option<&mut scioxide_core::OperatorContext<'_, '_, '_, T>>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        let (rows, cols) = if adj {
            (self.cols, self.rows)
        } else {
            (self.rows, self.cols)
        };
        check_apply([cols, rows, self.len(adj)?], x.len(), y.len(), w.len())?;
        crate::checked::validate_inputs(x)?;
        if rows == 0 || cols == 0 {
            y.fill(T::ZERO);
            return Ok(scioxide_core::Invocation::Evaluated(()));
        }
        let (stage, work) = crate::workspace::split_mut(w, rows)?;
        let (mut r, mut c) = (0, 0);
        for b in self.blocks {
            let (m, n) = if adj {
                (b.cols(), b.rows())
            } else {
                (b.rows(), b.cols())
            };
            if m == 0 || n == 0 {
                stage[r..r + m].fill(T::ZERO);
            } else if adj {
                let outcome = match context.as_deref_mut() {
                    Some(context) => b.apply_adjoint_into_scoped(
                        &x[c..c + n],
                        &mut stage[r..r + m],
                        &mut work[..b.apply_adjoint_workspace_len()?],
                        context,
                    ),
                    None => b
                        .apply_adjoint_into(
                            &x[c..c + n],
                            &mut stage[r..r + m],
                            &mut work[..b.apply_adjoint_workspace_len()?],
                        )
                        .map(scioxide_core::Invocation::Evaluated),
                }?;
                if !outcome.is_evaluated() {
                    return Ok(outcome);
                }
            } else {
                let outcome = match context.as_deref_mut() {
                    Some(context) => b.apply_into_scoped(
                        &x[c..c + n],
                        &mut stage[r..r + m],
                        &mut work[..b.apply_workspace_len()?],
                        context,
                    ),
                    None => b
                        .apply_into(
                            &x[c..c + n],
                            &mut stage[r..r + m],
                            &mut work[..b.apply_workspace_len()?],
                        )
                        .map(scioxide_core::Invocation::Evaluated),
                }?;
                if !outcome.is_evaluated() {
                    return Ok(outcome);
                }
            }
            r += m;
            c += n;
        }
        crate::checked::provider_output(stage)?;
        y.copy_from_slice(stage);
        Ok(scioxide_core::Invocation::Evaluated(()))
    }
}
impl<T: NumericScalar> LinearOperator<T> for BlockDiagonalOperator<'_, T> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.len(false)
    }
    fn apply_into(&self, x: &[T], y: &mut [T], w: &mut [T]) -> Result<(), SciOxideError> {
        self.apply(false, x, y, w, None)
            .and_then(scioxide_core::Invocation::require)
    }
    fn apply_into_scoped(
        &self,
        x: &[T],
        y: &mut [T],
        w: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply(false, x, y, w, Some(context))
    }
}
impl<T: NumericScalar> AdjointOperator<T> for BlockDiagonalOperator<'_, T> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.len(true)
    }
    fn apply_adjoint_into(&self, x: &[T], y: &mut [T], w: &mut [T]) -> Result<(), SciOxideError> {
        self.apply(true, x, y, w, None)
            .and_then(scioxide_core::Invocation::require)
    }
    fn apply_adjoint_into_scoped(
        &self,
        x: &[T],
        y: &mut [T],
        w: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply(true, x, y, w, Some(context))
    }
}
impl<T: NumericScalar> BlockOperator<T> for BlockDiagonalOperator<'_, T> {}
fn assess_residual<T: NumericScalar>(
    rhs: &[T],
    ax: &[T],
    tolerance: Tolerance<T::Real>,
) -> Result<CompositeSolveReport<T::Real>, SciOxideError> {
    // Composite applications have checked their provider output.
    let bnorm = rhs.iter().fold(T::Real::ZERO, |v, x| v.max(x.abs()));
    let anorm = ax.iter().fold(T::Real::ZERO, |v, x| v.max(x.abs()));
    let scale = bnorm.max(anorm);
    if scale == T::Real::ZERO {
        return Ok(CompositeSolveReport {
            termination: DirectSolveTermination::Accepted,
            backward_error: T::Real::ZERO,
        });
    }
    let error = rhs.iter().zip(ax).fold(T::Real::ZERO, |v, (&b, &a)| {
        v.max((b / T::from_real(scale) - a / T::from_real(scale)).abs())
    });
    let reference = bnorm / scale + anorm / scale;
    let termination = if tolerance.is_residual_ok(error, reference)? {
        DirectSolveTermination::Accepted
    } else {
        DirectSolveTermination::ResidualRejected
    };
    Ok(CompositeSolveReport {
        termination,
        backward_error: error / reference,
    })
}
/// Returns the Khatri–Rao product of `a` and `b`, the column-wise Kronecker product.
///
/// For `a` of size `m × n` and `b` of size `p × n`, column `j` of the
/// `mp × n` result is `a[:, j] ⊗ b[:, j]`, so the entry in row `i p + r`
/// and column `j` is `a[i][j] b[r][j]`.
///
/// ```
/// use scioxide_linalg::{Matrix, khatri_rao};
/// // [2, 3]ᵀ ⊗ [2, 3]ᵀ = [4, 6, 6, 9]ᵀ.
/// let a = Matrix::from_rows(&[[2.0_f64], [3.0]])?;
/// let product = khatri_rao(&a, &a)?;
/// assert_eq!(product.as_slice(), &[4.0, 6.0, 6.0, 9.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
/// - The errors of [`khatri_rao_into`].
/// - `AllocationFailure` if the result cannot be allocated.
pub fn khatri_rao<'a, 'b, T: NumericScalar>(
    a: impl Into<MatrixView<'a, T>>,
    b: impl Into<MatrixView<'b, T>>,
) -> Result<Matrix<T>, SciOxideError> {
    khatri_rao_with_options(a, b, &crate::LinalgExecutionOptions::default())
}

/// Runs [`khatri_rao`](khatri_rao) with the byte limit of `options` applied to the result.
///
/// # Errors
///
/// The errors of [`khatri_rao`](khatri_rao), and:
///
/// - `ResourceLimit` if the result exceeds the byte limit in `options`.
pub fn khatri_rao_with_options<'a, 'b, T: NumericScalar>(
    a: impl Into<MatrixView<'a, T>>,
    b: impl Into<MatrixView<'b, T>>,
    options: &crate::LinalgExecutionOptions,
) -> Result<Matrix<T>, SciOxideError> {
    let (a, b) = (a.into(), b.into());
    let rows = khatri_rao_rows(a, b)?;
    let quantity = scioxide_core::quantity!("khatri_rao_bytes");
    super::constructors::admit_owned::<T>(size(rows, a.cols(), quantity)?, options, quantity)?;
    let mut out = Matrix::zeros(rows, a.cols())?;
    khatri_rao_into(
        a,
        b,
        out.view_mut(),
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Ok(out)
}

fn khatri_rao_rows<T: NumericScalar>(
    a: MatrixView<'_, T>,
    b: MatrixView<'_, T>,
) -> Result<usize, SciOxideError> {
    if a.cols() != b.cols() {
        return Err(SciOxideError::dimension_mismatch([a.cols()], [b.cols()]));
    }
    let rows = size(
        a.rows(),
        b.rows(),
        scioxide_core::quantity!("khatri_rao_rows"),
    )?;
    size(
        rows,
        a.cols(),
        scioxide_core::quantity!("khatri_rao_elements"),
    )?;
    Ok(rows)
}
/// Writes the Khatri–Rao product of `a` and `b` into `out`.
///
/// The product is defined on [`khatri_rao`]. `out` may be a strided view. The
/// function counts no iterations.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::{Matrix, khatri_rao_into};
/// let a = Matrix::from_vec(2, 1, vec![2.0_f64, 3.0])?;
/// let mut output = Matrix::zeros(4, 1)?;
/// khatri_rao_into(a.view(), a.view(), output.view_mut(), ExecutionContext::sequential_native())?;
/// assert_eq!(output.as_slice(), &[4.0, 6.0, 6.0, 9.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
/// - `DimensionMismatch` if `a` and `b` have different numbers of columns,
///   or `out` is not `mp × n`.
/// - `SizeOverflow` if `m p` or `m p n` overflows `usize`.
/// - `NonFinite` if `a` or `b` contains NaN or an infinity.
/// - `NumericalInstability` if a product overflows.
/// - `ExecutionStopped` if the context is cancelled or its deadline passes.
pub fn khatri_rao_into<'a, 'b, 'scope, 'capability, T: NumericScalar>(
    a: impl Into<MatrixView<'a, T>>,
    b: impl Into<MatrixView<'b, T>>,
    mut out: MatrixViewMut<'_, T>,
    context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
) -> Result<(), SciOxideError> {
    let (a, b) = (a.into(), b.into());
    let context = context.into();
    let execution = context.execution();
    execution.check()?;
    let rows = khatri_rao_rows(a, b)?;
    if out.rows() != rows || out.cols() != a.cols() {
        return Err(SciOxideError::dimension_mismatch(
            [rows, a.cols()],
            [out.rows(), out.cols()],
        ));
    }
    if rows == 0 || a.cols() == 0 {
        return Ok(());
    }
    for publish in [false, true] {
        execution.check()?;
        for i in 0..rows {
            if !publish {
                execution.check()?;
            }
            for j in 0..a.cols() {
                let av = a.get_element(i / b.rows(), j);
                let bv = b.get_element(i % b.rows(), j);
                crate::checked::validate_input(av, "left coefficient")?;
                crate::checked::validate_input(bv, "right coefficient")?;
                let value = narrow(wide(av)?.mul(wide(bv)?)?)?;
                if publish {
                    out.set_element(i, j, value);
                }
            }
        }
    }
    Ok(())
}
// A solve through an inverse needs a square factor and an inverse of its shape.
fn square_inverse<T: NumericScalar>(
    factor: &dyn LinearOperator<T>,
    inverse: &dyn LinearOperator<T>,
) -> Result<(), SciOxideError> {
    let (rows, cols) = (factor.rows(), factor.cols());
    if rows != cols {
        return Err(SciOxideError::dimension_mismatch(
            [rows, rows],
            [rows, cols],
        ));
    }
    if inverse.rows() != rows || inverse.cols() != cols {
        return Err(SciOxideError::dimension_mismatch(
            [rows, cols],
            [inverse.rows(), inverse.cols()],
        ));
    }
    Ok(())
}
