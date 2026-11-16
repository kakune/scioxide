//! Factorization and solve of symmetric KKT systems.
pub(crate) mod assembly;
mod factor;
pub(crate) use factor::Factor;

use scioxide_core::{ExecutionContext, NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::{LdlDecomposition, LinalgWorkspace, MatrixView, VectorView, VectorViewMut};
use scioxide_sparse::LdlInertia;
use scioxide_sparse::{
    DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder, SparseLdl, SparseSymmetricLimits,
    SparseSymmetricSymbolic, SparseSymmetricWorkspace,
};

/// Factors and solves the real symmetric linear systems of the LP, QP, SQP and
/// nonlinear interior-point solvers.
///
/// A solver passes the full symmetric matrix `K` of each system to
/// [`factor`](Self::factor), then calls [`solve_into`](Self::solve_into) for
/// one or more right-hand sides. Depending on the method, `K` is a saddle-point
/// (KKT) matrix with Hessian and constraint Jacobian blocks, or the normal
/// matrix of an interior-point LP step. Each solver documents the order of the
/// systems it passes. Implement this trait to use your own direct or iterative
/// solver; [`DenseKktProvider`] and [`SparseKktProvider`] are the built-in
/// implementations.
///
/// `factor` must return the inertia of `K`: the numbers of positive, negative
/// and zero eigenvalues. Solutions must be finite. Solvers check the residual
/// of every solution against the original system, so an inaccurate solve is
/// detected, not trusted. After a failed `factor`, `solve_into` must fail
/// until the next successful `factor`. On error, `solve_into` must leave
/// `output` unchanged.
pub trait KktProvider<S: RealScalar + NumericScalar<Real = S>> {
    /// Factors the symmetric matrix `matrix` and returns its inertia.
    ///
    /// # Errors
    ///
    /// Implementations return `DimensionMismatch` or `InvalidInput` for a
    /// matrix they cannot accept, and resource, execution or numerical errors
    /// from the factorization.
    fn factor(
        &mut self,
        matrix: MatrixView<'_, S>,
        execution: ExecutionContext<'_>,
    ) -> Result<LdlInertia, SciOxideError>;
    /// Solves `K x = rhs` with the last factorization and writes `x` to `output`.
    ///
    /// # Errors
    ///
    /// Implementations return `InvalidInput` if no successful factorization
    /// exists, `DimensionMismatch` for slices of the wrong length, and resource,
    /// execution or numerical errors from the solve.
    fn solve_into(
        &mut self,
        rhs: &[S],
        output: &mut [S],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError>;
}
/// Reborrow a lent provider for one nested call, shortening its object lifetime.
pub(crate) fn lend<'s, S: RealScalar + NumericScalar<Real = S>>(
    provider: &'s mut Option<&mut dyn KktProvider<S>>,
) -> Option<&'s mut dyn KktProvider<S>> {
    provider
        .as_mut()
        .map(|provider| &mut **provider as &mut dyn KktProvider<S>)
}
#[derive(Debug, Clone, Copy)]
enum RankFallback<S: RealScalar> {
    Direct,
    Spectral(scioxide_linalg::RankPolicy<S>),
    LeastSquares(scioxide_linalg::RankPolicy<S>),
}

/// Solves KKT systems with a dense `L D Lᵀ` factorization.
///
/// Each matrix is first scaled symmetrically by powers of two, with four
/// sweeps of Ruiz equilibration, and then factored by
/// [`LdlDecomposition`](scioxide_linalg::LdlDecomposition) with bounded rook
/// pivoting. A factorization costs about `n³/3` flops and `O(n²)` storage for
/// a system of order `n`. Storage is allocated once, up to the maximum order
/// given to [`new`](Self::new).
///
/// [`KktProvider::factor`] returns `InvalidInput` if the matrix is larger than
/// that order or not symmetric, `DimensionMismatch` if it is not square,
/// `NonFinite` if it contains NaN or infinity, and `NumericalInstability`
/// (`Overflow`) if scaling overflows.
///
/// # References
///
/// - D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms
///   in matrices", Technical Report RAL-TR-2001-034, Rutherford Appleton
///   Laboratory, 2001.
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, §11.1.
#[derive(Debug)]
pub struct DenseKktProvider<S: RealScalar + NumericScalar<Real = S>> {
    order: usize,
    factor: LdlDecomposition<S>,
    work: LinalgWorkspace<'static, S>,
    equilibrated: scioxide_linalg::Matrix<S>,
    exponents: alloc::vec::Vec<i32>,
    rhs: alloc::vec::Vec<S>,
    solution: alloc::vec::Vec<S>,
    rank_policy: RankFallback<S>,
    spectral: scioxide_linalg::HermitianEigenDecomposition<S>,
    spectral_work: LinalgWorkspace<'static, S>,
    svd: scioxide_linalg::ThinSvdDecomposition<S>,
    svd_work: LinalgWorkspace<'static, S>,
    singular: bool,
    ready: bool,
}
impl<S: RealScalar + NumericScalar<Real = S>> DenseKktProvider<S> {
    /// Allocates storage for systems of order up to `maximum_order`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn new(maximum_order: usize) -> Result<Self, SciOxideError> {
        let mut factor = LdlDecomposition::empty();
        factor.try_reserve(maximum_order)?;
        let mut work = LinalgWorkspace::new();
        work.try_reserve(
            &scioxide_linalg::LdlDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LdlWorkspaceRequest::factor(
                    maximum_order,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?,
        )?;
        work.try_reserve(
            &scioxide_linalg::LdlDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LdlWorkspaceRequest::solve_matrix(
                    maximum_order,
                    1,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?,
        )?;
        Ok(Self {
            order: maximum_order,
            rank_policy: RankFallback::Direct,
            spectral: scioxide_linalg::HermitianEigenDecomposition::empty(),
            spectral_work: LinalgWorkspace::new(),
            svd: scioxide_linalg::ThinSvdDecomposition::empty(),
            svd_work: LinalgWorkspace::new(),
            singular: false,
            ready: false,
            factor,
            work,
            equilibrated: scioxide_linalg::Matrix::zeros(maximum_order, maximum_order)?,
            exponents: scioxide_core::backend::try_filled(
                maximum_order,
                0,
                scioxide_core::quantity!("kkt_equilibration"),
            )?,
            rhs: scioxide_core::backend::try_filled(
                maximum_order,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("kkt_rhs"),
            )?,
            solution: scioxide_core::backend::try_filled(
                maximum_order,
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("kkt_solution"),
            )?,
        })
    }
    /// Reserve the minimum-norm SVD fallback of least-squares KKT solves, so a
    /// prepared provider never grows it on a singular factor.
    pub(crate) fn reserve_least_squares_fallback(&mut self) -> Result<(), SciOxideError> {
        let options = scioxide_linalg::LinalgExecutionOptions::new();
        self.svd.try_reserve(self.order, self.order)?;
        for request in [
            scioxide_linalg::SvdWorkspaceRequest::new(self.order, self.order, &options),
            scioxide_linalg::SvdWorkspaceRequest::solve(self.order, self.order, &options),
        ] {
            self.svd_work.try_reserve(
                &scioxide_linalg::ThinSvdDecomposition::<S>::workspace_requirements(&request)?,
            )?;
        }
        Ok(())
    }
    pub(crate) fn use_rank_policy(&mut self, policy: Option<scioxide_linalg::RankPolicy<S>>) {
        self.rank_policy = policy.map_or(RankFallback::Direct, RankFallback::Spectral);
    }
    pub(crate) fn use_least_squares_rank_policy(
        &mut self,
        policy: Option<scioxide_linalg::RankPolicy<S>>,
    ) {
        self.rank_policy = policy.map_or(RankFallback::Direct, RankFallback::LeastSquares);
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> KktProvider<S> for DenseKktProvider<S> {
    fn factor(
        &mut self,
        matrix: MatrixView<'_, S>,
        execution: ExecutionContext<'_>,
    ) -> Result<LdlInertia, SciOxideError> {
        self.ready = false;
        if matrix.rows() > self.order {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("matrix"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "system exceeds prepared maximum order",
            ));
        }
        let n = matrix.rows();
        if matrix.cols() != n {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(n, n),
                scioxide_core::Dimensions::two(n, matrix.cols()),
            ));
        }
        self.equilibrated.try_resize_flat(n, n)?;
        for i in 0..n {
            for j in 0..n {
                self.equilibrated.as_mut_slice()[i * n + j] = matrix.get(i, j)?;
            }
        }
        crate::utils::validate_finite(self.equilibrated.as_slice())?;
        for i in 0..n {
            for j in 0..i {
                if self.equilibrated.as_slice()[i * n + j]
                    != self.equilibrated.as_slice()[j * n + i]
                {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("matrix"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "KKT matrix must be symmetric",
                    ));
                }
            }
        }
        self.exponents[..n].fill(0);
        // Four symmetric max-norm sweeps (Ruiz equilibration). Powers of two
        // preserve symmetry exactly and do not introduce rounding in scaling.
        for _ in 0..4 {
            for i in 0..n {
                let maximum = self.equilibrated.as_slice()[i * n..(i + 1) * n]
                    .iter()
                    .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |a, &b| {
                        a.max(NumericScalar::abs(b))
                    });
                let shift = if maximum == <S as scioxide_core::AlgebraicScalar>::ZERO {
                    0
                } else {
                    -(scioxide_core::Scaled::from_value(maximum)?
                        .exponent()
                        .div_euclid(2) as i32)
                };
                self.exponents[i] += shift;
                // A congruence changes row and column i together. The diagonal
                // receives the scale twice, and inertia is unchanged.
                for j in 0..n {
                    let value = self.equilibrated.as_slice()[i * n + j].scalbn(if i == j {
                        2 * shift
                    } else {
                        shift
                    });
                    if !value.is_finite() {
                        return Err(SciOxideError::numerical_instability(
                            scioxide_core::NumericalIssue::Overflow,
                            "KKT equilibration overflow",
                        ));
                    }
                    self.equilibrated.as_mut_slice()[i * n + j] = value;
                    self.equilibrated.as_mut_slice()[j * n + i] = value;
                }
            }
        }
        LdlDecomposition::factor_into(
            self.equilibrated.view(),
            scioxide_linalg::HermitianInput::Exact,
            &scioxide_linalg::LinalgExecutionOptions::new(),
            &mut self.factor,
            &mut self.work,
            execution,
        )?;
        let inertia = self.factor.inertia(scioxide_linalg::RankPolicy::Exact)?;
        self.singular = match self.rank_policy {
            RankFallback::Spectral(policy) => !self.factor.analyze_rank(policy)?.is_full_rank(),
            RankFallback::LeastSquares(_) => inertia.zero() != 0,
            RankFallback::Direct => false,
        };
        if self.singular {
            // Use original units so the minimum-norm solution is independent
            // of the congruence equilibration used by the direct LDL solve.
            if matches!(self.rank_policy, RankFallback::LeastSquares(_)) {
                scioxide_linalg::ThinSvdDecomposition::compute_into(
                    matrix,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    &mut self.svd,
                    &mut self.svd_work,
                    execution,
                )?;
            } else {
                scioxide_linalg::HermitianEigenDecomposition::decompose_into(
                    matrix,
                    scioxide_linalg::HermitianInput::Exact,
                    &scioxide_linalg::HermitianEigenOptions::default(),
                    &mut self.spectral,
                    &mut self.spectral_work,
                    execution,
                )?;
            }
        }
        self.ready = true;
        Ok(inertia)
    }
    fn solve_into(
        &mut self,
        rhs: &[S],
        output: &mut [S],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if !self.ready {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "no prepared factor",
            ));
        }
        let n = self.equilibrated.rows();
        if rhs.len() != n || output.len() != n {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(n, n),
                scioxide_core::Dimensions::two(rhs.len(), output.len()),
            ));
        }
        if self.singular {
            if let RankFallback::LeastSquares(policy) = self.rank_policy {
                self.svd.solve_into(
                    VectorView::from_slice(rhs),
                    policy,
                    <S as scioxide_core::AlgebraicScalar>::ZERO,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    VectorViewMut::from_slice(&mut self.solution[..n]),
                    &mut self.svd_work,
                    execution,
                )?;
                crate::utils::validate_finite(&self.solution[..n])?;
                output.copy_from_slice(&self.solution[..n]);
                return Ok(());
            }
            let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
            self.solution[..n].fill(zero);
            let scale = self
                .spectral
                .eigenvalues()
                .iter()
                .fold(zero, |a, &b| a.max(NumericScalar::abs(b)));
            let threshold = match self.rank_policy {
                RankFallback::Spectral(scioxide_linalg::RankPolicy::Threshold(tolerance)) => {
                    tolerance.threshold(scale)?
                }
                RankFallback::Spectral(scioxide_linalg::RankPolicy::Exact) => zero,
                _ => scioxide_core::checked_float_product(
                    scale,
                    S::EPSILON * S::try_from_usize_exact(n)?,
                )?,
            };
            for (j, &eigenvalue) in self.spectral.eigenvalues().iter().enumerate() {
                if NumericScalar::abs(eigenvalue) <= threshold {
                    continue;
                }
                let mut projection = scioxide_core::ProductSum::new();
                for (i, &b) in rhs.iter().enumerate() {
                    projection
                        .add_product(self.spectral.eigenvectors().as_slice()[i * n + j], b)?;
                }
                let weight =
                    scioxide_core::checked_float_quotient(projection.finish()?, eigenvalue)?;
                for (i, value) in self.solution[..n].iter_mut().enumerate() {
                    *value = crate::utils::trial_component(
                        *value,
                        weight,
                        self.spectral.eigenvectors().as_slice()[i * n + j],
                    )?;
                }
            }
            crate::utils::validate_finite(&self.solution[..n])?;
            output.copy_from_slice(&self.solution[..n]);
            return Ok(());
        }
        for (i, &value) in rhs.iter().enumerate() {
            self.rhs[i] = value.scalbn(self.exponents[i]);
        }
        crate::utils::validate_finite(&self.rhs[..n])?;
        self.factor
            .solve_into(
                VectorView::from_slice(&self.rhs[..n]),
                &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
                VectorViewMut::from_slice(&mut self.solution[..n]),
                &mut self.work,
                execution,
            )
            .and_then(|report| report.termination().require_accepted())?;
        for (value, &exponent) in self.solution[..n].iter_mut().zip(&self.exponents[..n]) {
            *value = value.scalbn(exponent);
        }
        crate::utils::validate_finite(&self.solution[..n])?;
        output.copy_from_slice(&self.solution[..n]);
        Ok(())
    }
}
/// Solves KKT systems with a sparse `L D Lᵀ` factorization.
///
/// The provider converts each matrix to sparse form, keeping the diagonal and
/// every nonzero entry, and factors it with
/// [`SparseLdl`](scioxide_sparse::SparseLdl). The fill-reducing ordering and
/// symbolic analysis are computed again only when the sparsity pattern
/// changes; otherwise the numeric factorization is updated in place. The
/// limits passed to [`new`](Self::new) bound the symbolic analysis, the fill
/// and the workspace, and exceeding them is a `ResourceLimit` error.
///
/// [`KktProvider::factor`] returns `InvalidInput` if the matrix is not square.
#[derive(Debug)]
pub struct SparseKktProvider<S: RealScalar + NumericScalar<Real = S>> {
    limits: SparseSymmetricLimits,
    ready: bool,
    factor: Option<SparseLdl<S>>,
    workspace: Option<SparseSymmetricWorkspace<S>>,
    scratch: alloc::vec::Vec<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> SparseKktProvider<S> {
    /// Creates a provider with the limits for symbolic analysis, fill and
    /// workspace size.
    pub const fn new(limits: SparseSymmetricLimits) -> Self {
        Self {
            limits,
            ready: false,
            factor: None,
            workspace: None,
            scratch: alloc::vec::Vec::new(),
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> KktProvider<S> for SparseKktProvider<S> {
    fn factor(
        &mut self,
        matrix: MatrixView<'_, S>,
        execution: ExecutionContext<'_>,
    ) -> Result<LdlInertia, SciOxideError> {
        self.ready = false;
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("matrix"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "KKT must be square",
            ));
        }
        let mut builder = SparseBuilder::new(n, n);
        for i in 0..n {
            for j in 0..n {
                let value = matrix.get(i, j)?;
                if value != <S as scioxide_core::AlgebraicScalar>::ZERO || i == j {
                    builder.push(i, j, value)?;
                }
            }
        }
        let sparse = builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)?;
        if let (Some(factor), Some(work)) = (&mut self.factor, &mut self.workspace)
            && factor.symbolic().pattern() == sparse.structural_pattern()
        {
            factor.refactor(
                sparse.view(),
                scioxide_sparse::SparseSolveOptions::new(),
                work,
                execution,
            )?;
            let inertia = factor.inertia()?;
            self.ready = true;
            return Ok(inertia);
        }
        let symbolic = SparseSymmetricSymbolic::analyze(
            sparse.structural_pattern(),
            scioxide_sparse::SparseOrdering::default(),
            None,
            self.limits,
        )?;
        let work = (symbolic).workspace(())?;
        let factor = SparseLdl::factor_scoped(
            symbolic,
            sparse.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            execution,
        )?;
        let inertia = factor.inertia()?;
        let len = scioxide_core::backend::checked_product(
            &[2, n],
            scioxide_core::quantity!("sparse_kkt_solve_scratch"),
        )?;
        let extra = len.saturating_sub(self.scratch.len());
        scioxide_core::backend::try_reserve(
            &mut self.scratch,
            extra,
            scioxide_core::quantity!("sparse_kkt_solve_scratch"),
        )?;
        self.scratch
            .resize(len, <S as scioxide_core::AlgebraicScalar>::ZERO);
        self.factor = Some(factor);
        self.workspace = Some(work);
        self.ready = true;
        Ok(inertia)
    }
    fn solve_into(
        &mut self,
        rhs: &[S],
        output: &mut [S],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if !self.ready {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "no prepared factor",
            ));
        }
        let factor = self.factor.as_mut().ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "no prepared factor",
            )
        })?;
        factor.solve_into(
            rhs,
            scioxide_sparse::SparseOperation::Normal,
            scioxide_sparse::SparseSolveOptions::new(),
            output,
            &mut scioxide_sparse::SparseSolveWorkspace::borrowed(&mut self.scratch),
            execution,
        )
    }
}

/// Original-system componentwise backward residual, using exact product sums.
pub(crate) fn verify_residual<S: RealScalar + NumericScalar<Real = S>>(
    matrix: MatrixView<'_, S>,
    rhs: &[S],
    solution: &[S],
) -> Result<(), SciOxideError> {
    verify_residual_with_floor(
        matrix,
        rhs,
        solution,
        <S as scioxide_core::AlgebraicScalar>::ZERO,
    )
}

/// Checks the residual with an absolute allowance `floor` for near-zero
/// equations. Original-model residuals are checked separately.
pub(crate) fn verify_residual_with_floor<S: RealScalar + NumericScalar<Real = S>>(
    matrix: MatrixView<'_, S>,
    rhs: &[S],
    solution: &[S],
    floor: S,
) -> Result<(), SciOxideError> {
    matrix.verify_solution(
        rhs,
        solution,
        scioxide_core::Tolerance::new(floor, S::SQRT_EPSILON)?,
    )?;
    Ok(())
}

pub(crate) fn validate_inertia(inertia: LdlInertia, order: usize) -> Result<(), SciOxideError> {
    if inertia.total() != order {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("provider"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "KKT provider returned an invalid inertia dimension",
        ));
    }
    Ok(())
}

/// Checks for the expected saddle-point inertia: `positive` positive and
/// `negative` negative eigenvalues and no zero eigenvalue. A dimension error
/// means the provider returned an inertia of the wrong order.
pub(crate) fn matches_inertia(
    inertia: LdlInertia,
    positive: usize,
    negative: usize,
) -> Result<bool, SciOxideError> {
    let order = scioxide_core::backend::checked_add(
        positive,
        negative,
        scioxide_core::quantity!("kkt_inertia"),
    )?;
    validate_inertia(inertia, order)?;
    Ok(inertia.positive() == positive && inertia.negative() == negative && inertia.zero() == 0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn failed_preparation_invalidates_dense_and_sparse_solve_state() {
        let mut dense = DenseKktProvider::<f64>::new(2).unwrap();
        let mut sparse = SparseKktProvider::new(SparseSymmetricLimits::new(16, 32, 100_000));
        let good = MatrixView::from_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]).unwrap();
        let bad = MatrixView::from_slice(2, 2, &[2.0, f64::NAN, f64::NAN, 3.0]).unwrap();
        for provider in [&mut dense as &mut dyn KktProvider<f64>, &mut sparse] {
            let inertia = provider
                .factor(good, ExecutionContext::sequential())
                .unwrap();
            assert!(matches_inertia(inertia, 2, 0).unwrap());
            let mut output = [99.0; 2];
            provider
                .solve_into(&[4.0, 6.0], &mut output, ExecutionContext::sequential())
                .unwrap();
            assert!((output[0] - 2.0).abs() < 1e-12 && (output[1] - 2.0).abs() < 1e-12);
            assert!(
                provider
                    .factor(bad, ExecutionContext::sequential())
                    .is_err()
            );
            output = [99.0; 2];
            assert!(
                provider
                    .solve_into(&[4.0, 6.0], &mut output, ExecutionContext::sequential())
                    .is_err()
            );
            assert_eq!(output, [99.0; 2]);
            provider
                .factor(good, ExecutionContext::sequential())
                .unwrap();
            provider
                .solve_into(&[4.0, 6.0], &mut output, ExecutionContext::sequential())
                .unwrap();
            assert!((output[0] - 2.0).abs() < 1e-12 && (output[1] - 2.0).abs() < 1e-12);
        }
        assert!(
            matches_inertia(LdlInertia::new(1, 1, 0).unwrap(), 2, 0).is_ok_and(|matches| !matches)
        );
        assert!(matches_inertia(LdlInertia::new(1, 0, 0).unwrap(), 2, 0).is_err());
    }
}
