//! Eigenvalues and first eigenvector components of a symmetric tridiagonal
//! matrix (Golub–Welsch) in O(n) storage, by implicit QL iteration.
use super::*;
use crate::HermitianEigenFirstComponents;

pub(in crate::eigen) fn requirements<R: RealScalar + NumericScalar<Real = R>>(
    request: &HermitianEigenWorkspaceRequest<R>,
) -> Result<LinalgRequirements, SciOxideError> {
    if request.storage != HermitianStorage::Tridiagonal || request.options.selects() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("request"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "first components require a complete absolute-accuracy tridiagonal job",
        ));
    }
    let mut requirements =
        LinalgRequirements::new(request.options.execution_options().byte_limit());
    *requirements.scalar_mut(ScalarRegion::Direct) = request.order;
    *requirements.real_mut(RealRegion::Direct) = checked_product(
        &[4, request.order],
        scioxide_core::quantity!("first_components_scratch"),
    )?;
    requirements.required_bytes::<R>()?;
    Ok(requirements)
}

impl<R: RealScalar + NumericScalar<Real = R>> HermitianEigenFirstComponents<R> {
    /// Computes the eigenvalues and first eigenvector components of a symmetric tridiagonal matrix.
    ///
    /// `diagonal` holds the `n` diagonal entries and `off_diagonal` the `n − 1`
    /// entries of the sub- and superdiagonal. The conventions are stated on
    /// [`HermitianEigenFirstComponents`]; the order and deflation tolerance
    /// come from [`HermitianEigenOptions`].
    ///
    /// Implicit QL iteration with Wilkinson shifts applies its rotations only
    /// to the first row of the eigenvector matrix, so the cost is O(n²) time
    /// and O(n) memory instead of the O(n³) time and O(n²) memory of a full
    /// eigendecomposition.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `off_diagonal.len()` is not
    ///   `diagonal.len().saturating_sub(1)`.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `ConvergenceFailure` if the iteration budget is exhausted.
    /// - `NumericalInstability` if an eigenvalue overflows when scaled back to
    ///   the input units.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::HermitianEigenFirstComponents;
    /// // [[2, 1], [1, 2]] has eigenvalue 1 with eigenvector (1, −1) / √2.
    /// let result = HermitianEigenFirstComponents::tridiagonal(
    ///     &[2.0_f64, 2.0],
    ///     &[1.0])?;
    /// assert!((result.eigenvalues()[0] - 1.0).abs() < 1e-14);
    /// assert!((result.first_components()[0].powi(2) - 0.5).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// G. H. Golub and J. H. Welsch, "Calculation of Gauss quadrature rules",
    /// *Math. Comp.* 23, 1969.
    pub fn tridiagonal(diagonal: &[R], off_diagonal: &[R]) -> Result<Self, SciOxideError> {
        Self::tridiagonal_with_options(
            diagonal,
            off_diagonal,
            &crate::HermitianEigenOptions::default(),
        )
    }

    /// Runs [`tridiagonal`](Self::tridiagonal) with `options`, which set the eigenvalue selection,
    /// ordering and accuracy, the deflation tolerance and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`tridiagonal`](Self::tridiagonal), and:
    ///
    /// - `InvalidInput` if `options` selects an index range or
    ///   [`EigenvalueAccuracy::Relative`](crate::EigenvalueAccuracy::Relative).
    pub fn tridiagonal_with_options(
        diagonal: &[R],
        off_diagonal: &[R],
        options: &HermitianEigenOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::tridiagonal_into(
            diagonal,
            off_diagonal,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the eigenvalues and first eigenvector components into `output`.
    ///
    /// This is the reusable form of [`Self::tridiagonal`], which describes the
    /// method. Size the workspace with
    /// [`HermitianEigenWorkspaceRequest::tridiagonal`]. Without an explicit
    /// iteration budget, QL iteration may take `30 max(n, 1)` sweeps.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::tridiagonal`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn tridiagonal_into<'scope, 'a>(
        diagonal: &[R],
        off_diagonal: &[R],
        options: &HermitianEigenOptions<R>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, R>,
        context: impl Into<LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        scope.kernel().execution().check()?;
        let scale = validate_tridiagonal(diagonal, off_diagonal, options)?;
        let n = diagonal.len();
        let request = HermitianEigenWorkspaceRequest::tridiagonal(n, options);
        workspace.try_reserve(&requirements(&request)?)?;
        output.reserve(n)?;
        let report = workspace.run_scope(
            scope,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "tridiagonal_first_components", 1),
                n as u128,
                0,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                workspace.with_direct(|scalars, reals, _, _| {
                    let (components, _) = crate::workspace::split_mut(scalars, n)?;
                    let (values, reals) = crate::workspace::split_mut(reals, n)?;
                    let (off, rotations) = crate::workspace::split_mut(reals, n)?;
                    values.copy_from_slice(diagonal);
                    normalize_slice(values, scale);
                    off.fill(R::ZERO);
                    off[..off_diagonal.len()].copy_from_slice(off_diagonal);
                    normalize_slice(&mut off[..off_diagonal.len()], scale);
                    // Accumulate only column zero of the transposed eigenvectors.
                    components.fill(R::ZERO);
                    if let Some(first) = components.first_mut() {
                        *first = R::ONE;
                    }
                    let mut vectors = ScratchMatrix::new(components, n, 1)?;
                    tridiagonal_ql(
                        values,
                        off,
                        Some(QlVectors::new(&mut vectors, rotations, n)?),
                        options,
                        kernels.execution(),
                        solve,
                    )?;
                    rescale_real_values(values, scale)?;
                    order_real_values(values, Some(&mut vectors), options.ordering())?;
                    validate_real_output(values)?;
                    validate_scalar_output(vectors.as_slice())?;
                    kernels.execution().check()?;
                    let _ = solve.counts().total()?;
                    output.eigenvalues.clear();
                    output.eigenvalues.extend_from_slice(values);
                    output.first_components.clear();
                    output
                        .first_components
                        .extend_from_slice(vectors.as_slice());
                    Ok(())
                })?;
                Ok(super::super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )?;
        output.report = Some(report);
        Ok(report)
    }
}
