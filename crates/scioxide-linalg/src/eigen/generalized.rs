//! Dense Hermitian-definite eigenproblems `A v = λ B v`, by Cholesky reduction to standard form.
use super::generalized_types::{
    GeneralizedHermitianEigenDecomposition, GeneralizedHermitianEigenvaluesBuffer,
    GeneralizedHermitianEigenvaluesResult,
};
use super::hermitian::{Candidate, Job, dense_candidate, dense_requirements};
use super::{
    GeneralizedHermitianEigenWorkspaceRequest, HermitianEigenOptions, hermitian_reduction_scale,
};
use crate::kernel::GemmConfig;
use crate::kernel::level3::triangular_with_context;
use crate::kernel::{GemmOp, Triangle};
use crate::kernel::{WorkspaceGemmProblem, gemm_workspace_with_context};
use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::workspace::{RealRegion, ScalarRegion};
use crate::{
    CholeskyDecomposition, KernelContext, LinalgExecutionContext, LinalgRequirements,
    LinalgWorkspace, MatrixView,
};
use crate::{
    FiniteMatrixView, HermitianInput, HermitianProjectionReport, MatrixSide, MatrixViewMut,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::DiagonalKind;
use scioxide_core::Scaled;
use scioxide_core::backend::checked_add;
use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar, SciOxideError, SolveContext,
};

/// Prefix A/B, congruence and Cholesky storage; vector jobs also reserve
/// normalized V, the stacked residual product, and real residual indicators.
fn prefix(order: usize, job: Job) -> Result<[usize; 3], SciOxideError> {
    let square = scioxide_core::backend::checked_product(
        &[order, order, if job == Job::Vectors { 7 } else { 4 }],
        scioxide_core::quantity!("generalized_hermitian_scratch"),
    )?;
    Ok([square, if job == Job::Vectors { order } else { 0 }, 0])
}

/// Shared scratch for a Hermitian-definite pair: the reduction prefix plus the
/// sequential maximum of Cholesky, triangular solves and the reduced job.
pub(super) fn generalized_requirements<S: NumericScalar>(
    request: &GeneralizedHermitianEigenWorkspaceRequest<S::Real>,
    job: Job,
) -> Result<LinalgRequirements, SciOxideError> {
    let request = &request.dense;
    reject_selection(&request.options)?;
    let n = request.order;
    let mut cholesky =
        crate::CholeskyWorkspaceRequest::factor(n, &request.options.execution_options())
            .with_worker_bound(request.workers);
    if matches!(request.packing, crate::workspace::Packing::Provider) {
        cholesky = cholesky.with_provider_packing();
    }
    let mut requirements = CholeskyDecomposition::<S>::workspace_requirements(&cholesky)?
        .max(&request.product::<S>(n, n, n, false)?)
        .max(&dense_requirements::<S>(request, job)?);
    if job == Job::Vectors {
        requirements =
            requirements.max(&request.product::<S>(super::support::times(2, n)?, n, n, false)?);
    }
    let [scalars, reals, _] = prefix(n, job)?;
    let quantity = scioxide_core::quantity!("generalized_hermitian_scratch");
    *requirements.scalar_mut(ScalarRegion::Direct) =
        checked_add(requirements.scalar(ScalarRegion::Direct), scalars, quantity)?;
    *requirements.real_mut(RealRegion::Direct) =
        checked_add(requirements.real(RealRegion::Direct), reals, quantity)?;
    requirements.byte_limit = request.options.execution_options().byte_limit();
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

fn reject_selection<R: RealScalar>(
    options: &HermitianEigenOptions<R>,
) -> Result<(), SciOxideError> {
    if options.index_range().is_some() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("options"),
            scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            "generalized Hermitian jobs require the complete spectrum",
        ));
    }
    Ok(())
}

fn restore<R: RealScalar>(values: &mut [R], a: R, b: R) -> Result<(), SciOxideError> {
    values.iter_mut().try_for_each(|v| {
        *v = Scaled::from_value(*v)?
            .mul(Scaled::from_value(a)?)?
            .div(Scaled::from_value(b)?)?
            .round()?;
        Ok(())
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn residuals<S: NumericScalar>(
    pair: MatrixView<'_, S>,
    vectors: &impl MatrixStorage<S>,
    values: &[S::Real],
    residuals: &mut [S::Real],
    normalized: &mut [S],
    product: &mut [S],
    policy: ExecutionPolicy,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let n = values.len();
    let dimension = crate::checked::rounded_count::<S::Real>(n.max(1))?;
    for col in 0..n {
        let scale = (0..n)
            .map(|row| vectors.get_element(row, col).abs())
            .fold(S::Real::ZERO, S::Real::max);
        crate::checked::finite_candidate(scale)?;
        for row in 0..n {
            normalized[row * n + col] =
                crate::rotation::divide_real(vectors.get_element(row, col), scale);
        }
    }
    gemm_workspace_with_context(
        WorkspaceGemmProblem {
            left: FiniteMatrixView::from_workspace(pair),
            right: FiniteMatrixView::from_workspace(MatrixView::from_slice(n, n, normalized)?),
            output: MatrixViewMut::from_slice(2 * n, n, product)?,
            config: GemmConfig::ordinary(),
        },
        policy,
        workspace,
        context.with_accuracy(scioxide_core::Accuracy::Standard),
    )?;
    for (col, (&lambda, result)) in values.iter().zip(residuals).enumerate() {
        let lscale = lambda.abs().max(S::Real::ONE);
        let mut error = S::Real::ZERO;
        for row in 0..n {
            let av = crate::rotation::divide_real(product[row * n + col], lscale);
            let bv = product[(n + row) * n + col] * S::from_real(lambda / lscale);
            error = error.max(crate::checked::finite_candidate(av - bv)?.abs());
        }
        *result = error / dimension / (S::Real::ONE / lscale + lambda.abs() / lscale);
    }
    Ok(())
}

/// A shape-checked pair with explicit interpretations.
#[derive(Clone, Copy)]
struct Pair<'m, S: NumericScalar> {
    a: MatrixView<'m, S>,
    b: MatrixView<'m, S>,
    projections: [Option<HermitianProjectionReport<S::Real>>; 2],
}

/// Generalized candidate in normalized pencil units before eigenvalue restoration.
struct GeneralizedCandidate<'c, 'a, 'g, S: NumericScalar> {
    inner: Candidate<'c, 'a, 'g, S>,
    residuals: &'c [S::Real],
    projections: [Option<HermitianProjectionReport<S::Real>>; 2],
}

/// Check shapes and reserve scratch for the explicit input interpretations.
fn prepare<'m, S: NumericScalar>(
    a: MatrixView<'m, S>,
    b: MatrixView<'m, S>,
    interpretations: [HermitianInput<S::Real>; 2],
    job: Job,
    options: &HermitianEigenOptions<S::Real>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<Pair<'m, S>, SciOxideError> {
    context.execution().check()?;
    reject_selection(options)?;
    let n = a.rows();
    if a.cols() != n {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(n, n),
            scioxide_core::Dimensions::two(n, a.cols()),
        ));
    }
    if b.rows() != n || b.cols() != n {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(n, n),
            scioxide_core::Dimensions::two(b.rows(), b.cols()),
        ));
    }
    let requirements = generalized_requirements::<S>(
        &GeneralizedHermitianEigenWorkspaceRequest::new(n, options).for_kernels(context),
        job,
    )?;
    workspace.try_reserve(&requirements)?;
    let projections = workspace.with_prefixes(prefix(n, job)?, |scalars, _, _, _| {
        let mut projections = [None; 2];
        for (index, source) in [a, b].into_iter().enumerate() {
            let target = &mut scalars[index * n * n..(index + 1) * n * n];
            projections[index] = crate::decompose::hermitian::prepare(
                source,
                interpretations[index],
                &mut MatrixViewMut::from_slice(n, n, target)?,
                if index == 0 {
                    scioxide_core::argument!("a")
                } else {
                    scioxide_core::argument!("b")
                },
            )?;
        }
        Ok(projections)
    })?;
    Ok(Pair { a, b, projections })
}

fn generalized_identity(order: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("linalg", "generalized_hermitian_eigen", 1),
        order as u128,
        0,
    )
}

/// Reduce `A v = lambda B v` to `C w = lambda w` and solve it in scratch.
fn generalized_candidate<S: NumericScalar, T>(
    pair: Pair<'_, S>,
    job: Job,
    options: &HermitianEigenOptions<S::Real>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
    solve: &mut SolveContext<'_, '_>,
    consume: impl FnOnce(GeneralizedCandidate<'_, '_, '_, S>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let n = pair.a.rows();
    let policy = options.execution_options().execution_policy();
    workspace.with_prefixes(prefix(n, job)?, |scalars, reals, _, child| {
        let (stacked, scalars) = crate::workspace::split_mut(scalars, 2 * n * n)?;
        let (reduced, scalars) = crate::workspace::split_mut(scalars, n * n)?;
        let (lower, residual_scratch) = crate::workspace::split_mut(scalars, n * n)?;
        let mut stacked = ScratchMatrix::new(stacked, 2 * n, n)?;
        let mut reduced = ScratchMatrix::new(reduced, n, n)?;
        let mut lower = ScratchMatrix::new(lower, n, n)?;
        let projections = pair.projections;
        let residual_values = &mut reals[..if job == Job::Vectors { n } else { 0 }];
        // Standard reduction: with `B = L L^H`,
        // `A v = lambda B v` becomes `C w = lambda w` for the Hermitian
        // `C = L^{-1} A L^{-H}` and `v = L^{-H} w`. A and B are normalized
        // independently to O(1) first, so lambda is recovered as
        // `lambda_C * a_scale / b_scale` (in extended range) and v as
        // `v_C / sqrt(b_scale)`.
        let ascale = hermitian_reduction_scale(crate::kernel::standard::slice_maximum(
            &stacked.as_slice()[..n * n],
        ));
        let bscale = hermitian_reduction_scale(crate::kernel::standard::slice_maximum(
            &stacked.as_slice()[n * n..],
        ));
        super::normalize_slice(&mut stacked.as_mut_slice()[..n * n], ascale);
        super::normalize_slice(&mut stacked.as_mut_slice()[n * n..], bscale);
        for i in 0..n {
            if pair.b.get_element(i, i).real_component() != S::Real::ZERO
                && stacked.get_element(n + i, i) == S::ZERO
            {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "normalization destroyed a definiteness pivot",
                ));
            }
        }
        CholeskyDecomposition::factor_exact_lower(
            stacked.try_view()?.block(n, 0, n, n)?,
            scioxide_core::argument!("b"),
            lower.as_mut_slice(),
            policy,
            child,
            context,
        )?;
        reduced
            .as_mut_slice()
            .copy_from_slice(&stacked.as_slice()[..n * n]);
        for (side, op) in [
            (MatrixSide::Left, GemmOp::None),
            (MatrixSide::Right, GemmOp::Adjoint),
        ] {
            triangular_with_context(
                crate::kernel::level3::TriangularAction::Solve,
                crate::kernel::TriangularOperand {
                    view: lower.try_view()?,
                    triangle: Triangle::Lower,
                    op,
                    diagonal: DiagonalKind::Explicit,
                },
                reduced.try_view_mut()?,
                crate::kernel::TriangularUpdate {
                    side,
                    alpha: S::ONE,
                    policy,
                },
                child,
                context,
            )?;
        }
        // The congruence is Hermitian. Retain its computed lower triangle rather
        // than interpreting roundoff between the two triangular solves as input asymmetry.
        for row in 0..n {
            reduced.as_mut_slice()[row * n + row] =
                S::from_real(reduced.get_element(row, row).real_component());
            for col in 0..row {
                reduced.as_mut_slice()[col * n + row] = reduced.get_element(row, col).conj();
            }
        }
        let reduced = reduced
            .try_view()?
            .try_finite_dispatched(context.dispatch())?
            .as_view();
        dense_candidate(
            reduced,
            HermitianInput::Exact,
            job,
            options,
            child,
            context,
            solve,
            |inner| {
                if job == Job::Vectors {
                    triangular_with_context(
                        crate::kernel::level3::TriangularAction::Solve,
                        crate::kernel::TriangularOperand {
                            view: lower.try_view()?,
                            triangle: Triangle::Lower,
                            op: GemmOp::Adjoint,
                            diagonal: DiagonalKind::Explicit,
                        },
                        inner.vectors.try_view_mut()?,
                        crate::kernel::TriangularUpdate {
                            side: MatrixSide::Left,
                            alpha: S::ONE,
                            policy,
                        },
                        inner.gemm,
                        context,
                    )?;
                    let (normalized, product) =
                        crate::workspace::split_mut(residual_scratch, n * n)?;
                    residuals(
                        stacked.try_view()?,
                        &*inner.vectors,
                        inner.values,
                        residual_values,
                        normalized,
                        product,
                        policy,
                        inner.gemm,
                        context,
                    )?;
                    let root = bscale.sqrt();
                    for value in inner.vectors.as_mut_slice() {
                        *value = crate::rotation::divide_real(*value, root);
                        crate::checked::finite_candidate(*value)?;
                    }
                }
                restore(inner.values, ascale, bscale)?;
                context.execution().check()?;
                consume(GeneralizedCandidate {
                    inner,
                    residuals: residual_values,
                    projections,
                })
            },
        )
    })
}

impl<R: RealScalar> GeneralizedHermitianEigenvaluesResult<R> {
    /// Computes the eigenvalues of `A v = λ B v` for Hermitian `A` and positive definite `B`.
    ///
    /// `a_input` and `b_input` define `A` and `B` as described on
    /// [`HermitianInput`]. The eigenvalues are real and sorted by
    /// [`HermitianEigenOptions::ordering`]. The complete spectrum is always
    /// computed.
    ///
    /// The pair is reduced to a standard Hermitian problem. With the Cholesky
    /// factorization `B = L Lᴴ`, the eigenvalues of `A v = λ B v` are those of
    /// `C = L⁻¹ A L⁻ᴴ`, and `v = L⁻ᴴ w` for each eigenvector `w` of `C`. `C`
    /// is solved as in
    /// [`HermitianEigenDecomposition::decompose`](crate::HermitianEigenDecomposition::decompose).
    /// `A` and `B` are each scaled by their largest entry magnitude first. The
    /// cost is O(n³) time and O(n²) memory.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` is not square or `b` does not have the shape of `a`.
    /// - `NonFinite` if an entry read under `a_input` or `b_input` is NaN or infinite.
    /// - `InvalidInput` if `a` or `b` violates its [`HermitianInput`]: not
    ///   exactly Hermitian under `Exact`, a diagonal entry with a nonzero
    ///   imaginary part, or a projection change above its tolerance.
    /// - `DomainError` if `B` is not numerically positive definite.
    /// - `ConvergenceFailure` if the iteration budget is exhausted.
    /// - `NumericalInstability` if scaling `B` turns a nonzero diagonal entry
    ///   into zero, or if a result overflows when scaled back.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{GeneralizedHermitianEigenvaluesResult, Matrix};
    /// // Diagonal pair: λ = 2 / 1 and 6 / 2.
    /// let a = Matrix::from_vec(2, 2, vec![2.0_f64, 0.0, 0.0, 6.0])?;
    /// let b = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// let values = GeneralizedHermitianEigenvaluesResult::compute(
    ///     &a,
    ///     &b,
    ///     scioxide_linalg::HermitianInput::Exact,
    ///     scioxide_linalg::HermitianInput::Exact)?;
    /// assert!((values.eigenvalues()[0] - 2.0).abs() < 1e-14);
    /// assert!((values.eigenvalues()[1] - 3.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    /// Hopkins, 2013, §8.7.
    pub fn compute<'m, S: NumericScalar<Real = R>>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        a_input: HermitianInput<S::Real>,
        b_input: HermitianInput<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::compute_with_options(
            a,
            b,
            a_input,
            b_input,
            &crate::HermitianEigenOptions::default(),
        )
    }

    /// Runs [`compute`](Self::compute) with `options`, which set the eigenvalue selection, ordering
    /// and accuracy, the deflation tolerance and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`compute`](Self::compute), and:
    ///
    /// - `InvalidInput` if `options` selects an index range.
    pub fn compute_with_options<'m, S: NumericScalar<Real = R>>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        a_input: HermitianInput<S::Real>,
        b_input: HermitianInput<S::Real>,
        options: &HermitianEigenOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let mut output = GeneralizedHermitianEigenvaluesBuffer::empty();
        let report = GeneralizedHermitianEigenvaluesBuffer::compute_into(
            a,
            b,
            a_input,
            b_input,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(Self { output, report })
    }
}

impl<R: RealScalar> GeneralizedHermitianEigenvaluesBuffer<R> {
    /// Computes the eigenvalues of a Hermitian-definite pair into `output`.
    ///
    /// This is the reusable form of [`GeneralizedHermitianEigenvaluesResult::compute`],
    /// which describes the method. Size the workspace with
    /// [`GeneralizedHermitianEigenWorkspaceRequest::new`]. The iterations of
    /// the tridiagonal solver are recorded by [`LinalgWorkspace::last_attempt`].
    ///
    /// # Errors
    ///
    /// - The errors of [`GeneralizedHermitianEigenvaluesResult::compute`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn compute_into<'scope, 'm, 'a, S: NumericScalar<Real = R>>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        a_input: HermitianInput<S::Real>,
        b_input: HermitianInput<S::Real>,
        options: &HermitianEigenOptions<R>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let pair = prepare(
            a.into(),
            b.into(),
            [a_input, b_input],
            Job::Values,
            options,
            workspace,
            scope.kernel(),
        )?;
        let order = pair.a.rows();
        super::types::reserve(
            &mut output.eigenvalues,
            order,
            scioxide_core::quantity!("output.eigenvalues"),
        )?;
        workspace.run_scope(
            scope,
            generalized_identity(order),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                generalized_candidate(
                    pair,
                    Job::Values,
                    options,
                    workspace,
                    kernels,
                    solve,
                    |candidate| {
                        output.eigenvalues.clear();
                        output.eigenvalues.extend_from_slice(candidate.inner.values);
                        output.projections = candidate.projections;
                        Ok(())
                    },
                )?;
                Ok(super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )
    }
}
impl<S: NumericScalar> GeneralizedHermitianEigenDecomposition<S> {
    /// Computes the eigenvalues and `B`-orthonormal eigenvectors of `A v = λ B v`.
    ///
    /// `A` is Hermitian and `B` Hermitian positive definite; `a_input` and
    /// `b_input` define them as described on [`HermitianInput`]. The
    /// eigenvalues are real and sorted by [`HermitianEigenOptions::ordering`],
    /// and column `j` of the eigenvector matrix `V` belongs to eigenvalue `j`.
    /// The columns satisfy `Vᴴ B V = I` to working accuracy. The complete
    /// spectrum is always computed.
    ///
    /// The pair is reduced to a standard Hermitian problem. With the Cholesky
    /// factorization `B = L Lᴴ`, the eigenvalues of `A v = λ B v` are those of
    /// `C = L⁻¹ A L⁻ᴴ`, and `v = L⁻ᴴ w` for each eigenvector `w` of `C`. `C`
    /// is solved as in
    /// [`HermitianEigenDecomposition::decompose`](crate::HermitianEigenDecomposition::decompose).
    /// `A` and `B` are each scaled by their largest entry magnitude first. The
    /// cost is O(n³) time and O(n²) memory. The result also contains a scaled
    /// residual for each eigenpair (see [`Self::residuals`]).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` is not square or `b` does not have the shape of `a`.
    /// - `NonFinite` if an entry read under `a_input` or `b_input` is NaN or infinite.
    /// - `InvalidInput` if `a` or `b` violates its [`HermitianInput`]: not
    ///   exactly Hermitian under `Exact`, a diagonal entry with a nonzero
    ///   imaginary part, or a projection change above its tolerance.
    /// - `DomainError` if `B` is not numerically positive definite.
    /// - `ConvergenceFailure` if the iteration budget is exhausted.
    /// - `NumericalInstability` if scaling `B` turns a nonzero diagonal entry
    ///   into zero, or if a result overflows when scaled back.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{GeneralizedHermitianEigenDecomposition, Matrix};
    /// let a = Matrix::from_vec(2, 2, vec![2.0_f64, 0.0, 0.0, 6.0])?;
    /// let b = Matrix::from_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// let eigen = GeneralizedHermitianEigenDecomposition::decompose(
    ///     &a,
    ///     &b,
    ///     scioxide_linalg::HermitianInput::Exact,
    ///     scioxide_linalg::HermitianInput::Exact)?;
    /// // λ = 3 has eigenvector e₂ / √2, since e₂ᴴ B e₂ = 2.
    /// assert!((eigen.eigenvalues()[1] - 3.0).abs() < 1e-14);
    /// assert!((eigen.eigenvectors().view().get(1, 1)?.abs() - 0.5_f64.sqrt()).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    /// Hopkins, 2013, §8.7.
    pub fn decompose<'m>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        a_input: HermitianInput<S::Real>,
        b_input: HermitianInput<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::decompose_with_options(
            a,
            b,
            a_input,
            b_input,
            &crate::HermitianEigenOptions::default(),
        )
    }

    /// Runs [`decompose`](Self::decompose) with `options`, which set the eigenvalue selection,
    /// ordering and accuracy, the deflation tolerance and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`decompose`](Self::decompose), and:
    ///
    /// - `InvalidInput` if `options` selects an index range.
    pub fn decompose_with_options<'m>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        a_input: HermitianInput<S::Real>,
        b_input: HermitianInput<S::Real>,
        options: &HermitianEigenOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::decompose_into(
            a,
            b,
            a_input,
            b_input,
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
    /// Computes the eigendecomposition of a Hermitian-definite pair into `output`.
    ///
    /// This is the reusable form of [`Self::decompose`], which describes the
    /// method and conventions. Size the workspace with
    /// [`GeneralizedHermitianEigenWorkspaceRequest::new`]. The iterations of
    /// the tridiagonal solver are recorded by [`LinalgWorkspace::last_attempt`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::decompose`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn decompose_into<'scope, 'm, 'a>(
        a: impl Into<MatrixView<'m, S>>,
        b: impl Into<MatrixView<'m, S>>,
        a_input: HermitianInput<S::Real>,
        b_input: HermitianInput<S::Real>,
        options: &HermitianEigenOptions<S::Real>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let pair = prepare(
            a.into(),
            b.into(),
            [a_input, b_input],
            Job::Vectors,
            options,
            workspace,
            scope.kernel(),
        )?;
        let order = pair.a.rows();
        let quantity = scioxide_core::quantity!("output.eigenvalues");
        super::types::reserve(&mut output.eigenvalues, order, quantity)?;
        super::types::reserve(&mut output.residuals, order, quantity)?;
        output.eigenvectors.try_reserve_shape(order, order)?;
        let report = workspace.run_scope(
            scope,
            generalized_identity(order),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                generalized_candidate(
                    pair,
                    Job::Vectors,
                    options,
                    workspace,
                    kernels,
                    solve,
                    |candidate| {
                        let vectors = &candidate.inner.vectors;
                        output
                            .eigenvectors
                            .try_resize_flat(vectors.rows(), vectors.cols())?;
                        output
                            .eigenvectors
                            .as_mut_slice()
                            .copy_from_slice(vectors.as_slice());
                        output.eigenvalues.clear();
                        output.eigenvalues.extend_from_slice(candidate.inner.values);
                        output.projections = candidate.projections;
                        output.residuals.clear();
                        output.residuals.extend_from_slice(candidate.residuals);
                        Ok(())
                    },
                )?;
                Ok(super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )?;
        output.report = Some(report);
        Ok(report)
    }
}
