//! Direct factorizations with an SVD fallback for ill-conditioned interpolation systems.

use scioxide_core::{
    ExecutionContext, NumericScalar, RankThreshold, RealScalar, SciOxideError, SciOxideErrorKind,
};
use scioxide_linalg::{
    CholeskyDecomposition, CholeskyPivoting, CholeskyWorkspaceRequest, DirectSolveOptions,
    HermitianInput, HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting,
    HouseholderWorkspaceRequest, LinalgExecutionOptions, LinalgRequirements, LinalgWorkspace,
    LuDecomposition, LuWorkspaceRequest, Matrix, RankPolicy, SvdWorkspaceRequest,
    ThinSvdDecomposition, Vector, VectorViewMut,
};

/// Names the matrix factorization used to solve a fitting or interpolation system.
///
/// Construction first factors the system directly: Cholesky for a
/// positive-definite kernel or covariance matrix, LU for a symmetric
/// indefinite system, column-pivoted QR for a least-squares design. If that
/// factorization fails, finds the matrix numerically rank deficient, or
/// estimates a reciprocal condition number below the accepted minimum, the
/// system is factored again with the singular value decomposition. A
/// covariance matrix of [`crate::GaussianProcessModel`] whose Cholesky
/// factorization fails is rejected with `DomainError` instead; it is factored
/// with the SVD only when the Cholesky factor exists but is rank deficient or
/// too ill-conditioned.
/// [`crate::RbfConditioning`] and [`crate::LocalFitReport`] report the result.
///
/// # References
///
/// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
///   Hopkins University Press, 2013, §3.4, §4.2, §5.4 and §8.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitFactorization {
    /// Cholesky factorization without pivoting, for a positive-definite kernel
    /// or covariance matrix.
    Cholesky,
    /// LU factorization with partial pivoting, for a symmetric indefinite system.
    Lu,
    /// QR factorization with column pivoting, for a rectangular least-squares
    /// design such as a local polynomial fit.
    PivotedQr,
    /// Singular value decomposition, used when the direct factorization fails
    /// or is too ill-conditioned.
    Svd,
}

/// Names the norm of a reported reciprocal condition number `1 / κ(A)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitConditionNorm {
    /// An estimate of `1 / (‖A‖₁ ‖A⁻¹‖₁)` from a few solves with the factors,
    /// reported for the direct factorizations.
    One,
    /// The exact 2-norm value `σ_min / σ_max`, the smallest singular value
    /// divided by the largest, reported for the SVD.
    Two,
}

#[derive(Clone, Copy)]
pub(crate) enum SystemKind {
    PositiveDefinite,
    Covariance,
    Symmetric,
    LeastSquares,
}

#[derive(Debug, Clone)]
enum Decomposition<S: RealScalar + NumericScalar<Real = S>> {
    Cholesky(CholeskyDecomposition<S>),
    Lu(LuDecomposition<S>),
    Qr(HouseholderDecomposition<S>),
    Svd(ThinSvdDecomposition<S>),
}

/// One stored factorization with its numerical rank and reciprocal condition.
#[derive(Debug, Clone)]
pub(crate) struct Factor<S: RealScalar + NumericScalar<Real = S>> {
    decomposition: Decomposition<S>,
    pub(crate) rank: usize,
    pub(crate) reciprocal_condition: S,
}

impl<S: RealScalar + NumericScalar<Real = S>> Factor<S> {
    pub(crate) const fn method(&self) -> FitFactorization {
        match self.decomposition {
            Decomposition::Cholesky(_) => FitFactorization::Cholesky,
            Decomposition::Lu(_) => FitFactorization::Lu,
            Decomposition::Qr(_) => FitFactorization::PivotedQr,
            Decomposition::Svd(_) => FitFactorization::Svd,
        }
    }

    pub(crate) const fn condition_norm(&self) -> FitConditionNorm {
        match self.decomposition {
            Decomposition::Svd(_) => FitConditionNorm::Two,
            _ => FitConditionNorm::One,
        }
    }

    /// Solve using the rank already accepted by preparation, without refactoring.
    pub(crate) fn solve_into(
        &self,
        rhs: &Vector<S>,
        output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let acceptance = DirectSolveOptions::new(RankPolicy::Exact);
        let report = match &self.decomposition {
            Decomposition::Cholesky(factor) => {
                factor.solve_into(rhs, &acceptance, output, workspace, context)?
            }
            Decomposition::Lu(factor) => {
                factor.solve_into(rhs, &acceptance, output, workspace, context)?
            }
            Decomposition::Qr(factor) => {
                factor.solve_into(rhs, &acceptance, output, workspace, context)?
            }
            Decomposition::Svd(factor) => {
                return factor.solve_into(
                    rhs,
                    RankPolicy::Exact,
                    S::ZERO,
                    &LinalgExecutionOptions::new(),
                    output,
                    workspace,
                    context,
                );
            }
        };
        if report.termination().is_success() {
            Ok(())
        } else {
            Err(SciOxideError::singular(None))
        }
    }
}

/// Workspace requirements for solving with any of the factorizations, including
/// the SVD fallback.
pub(crate) fn solve_requirements<S: RealScalar + NumericScalar<Real = S>>(
    rows: usize,
    cols: usize,
) -> Result<LinalgRequirements, SciOxideError> {
    let options = LinalgExecutionOptions::new();
    let n = rows.max(cols);
    let mut requirements = ThinSvdDecomposition::<S>::workspace_requirements(
        &SvdWorkspaceRequest::solve(rows, cols, &options),
    )?
    .max(&CholeskyDecomposition::<S>::workspace_requirements(
        &CholeskyWorkspaceRequest::solve(n, &options),
    )?)
    .max(&LuDecomposition::<S>::workspace_requirements(
        &LuWorkspaceRequest::solve(n, &options),
    )?);
    if rows >= cols {
        requirements = requirements.max(&HouseholderDecomposition::<S>::workspace_requirements(
            &HouseholderWorkspaceRequest::solve(rows, cols, HouseholderOrientation::Qr, &options),
        )?);
    }
    Ok(requirements)
}

/// Factor directly, falling back to the SVD only for an ill-conditioned system.
/// Rank-deficient fallback factors are returned for callers to diagnose or reject.
pub(crate) fn factor<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    kind: SystemKind,
    threshold: RankThreshold<S>,
    minimum_rcond: S,
    options: LinalgExecutionOptions,
    context: ExecutionContext<'_>,
) -> Result<Factor<S>, SciOxideError> {
    context.check()?;
    let direct_options =
        LinalgExecutionOptions::new().with_execution_policy(options.execution_policy());
    let primary = direct_factor(matrix, kind, threshold, &direct_options, context);
    match primary {
        Ok(primary) => {
            // Condition-scaled matrix magnitude supplements pivot rank screening.
            // Inconclusive systems receive the original singular-value policy.
            let scale = matrix
                .as_slice()
                .iter()
                .fold(S::ZERO, |a, x| a.max(x.abs()));
            let rank_screen = threshold.threshold(scale)?;
            if primary.rank == matrix.cols()
                && primary.reciprocal_condition >= minimum_rcond
                && primary.reciprocal_condition * scale > rank_screen
            {
                return Ok(primary);
            }
        }
        // A covariance requires a successful positive-definite factor. There
        // is no useful SVD solve to retain after that admission fails.
        Err(error) if matches!(kind, SystemKind::Covariance) => return Err(error),
        Err(error)
            if matches!(
                error.kind(),
                SciOxideErrorKind::Singular | SciOxideErrorKind::NumericalInstability
            ) || (matches!(kind, SystemKind::PositiveDefinite)
                && error.kind() == SciOxideErrorKind::DomainError) => {}
        Err(error) => return Err(error),
    };
    svd_factor(matrix, threshold, &options, context)
}

fn direct_factor<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    kind: SystemKind,
    threshold: RankThreshold<S>,
    options: &LinalgExecutionOptions,
    context: ExecutionContext<'_>,
) -> Result<Factor<S>, SciOxideError> {
    let mut workspace = LinalgWorkspace::new();
    let policy = RankPolicy::Threshold(threshold);
    let (decomposition, rank, condition) = match kind {
        SystemKind::PositiveDefinite | SystemKind::Covariance => {
            let mut f = CholeskyDecomposition::empty();
            CholeskyDecomposition::factor_into(
                matrix,
                HermitianInput::Exact,
                CholeskyPivoting::None,
                options,
                &mut f,
                &mut workspace,
                context,
            )?;
            let rank = f.definiteness_report(policy)?.effective_rank().rank();
            let condition = f.condition_estimate_scoped(options, &mut workspace, context)?;
            (Decomposition::Cholesky(f), rank, condition)
        }
        SystemKind::Symmetric => {
            let mut f = LuDecomposition::empty();
            LuDecomposition::factor_into(matrix, options, &mut f, &mut workspace, context)?;
            let rank = f.analyze_rank(policy)?.rank();
            let condition = f.condition_estimate_scoped(options, &mut workspace, context)?;
            (Decomposition::Lu(f), rank, condition)
        }
        SystemKind::LeastSquares => {
            let mut f = HouseholderDecomposition::empty();
            HouseholderDecomposition::factor_into(
                matrix,
                HouseholderOrientation::Qr,
                HouseholderPivoting::Column,
                options,
                &mut f,
                &mut workspace,
                context,
            )?;
            let rank = f.analyze_rank(policy)?.rank();
            let condition = f.condition_estimate_scoped(options, &mut workspace, context)?;
            (Decomposition::Qr(f), rank, condition)
        }
    };
    Ok(Factor {
        decomposition,
        rank,
        reciprocal_condition: condition.reciprocal().value(),
    })
}

fn svd_factor<S: RealScalar + NumericScalar<Real = S>>(
    matrix: &Matrix<S>,
    threshold: RankThreshold<S>,
    options: &LinalgExecutionOptions,
    context: ExecutionContext<'_>,
) -> Result<Factor<S>, SciOxideError> {
    let mut svd = ThinSvdDecomposition::empty();
    ThinSvdDecomposition::compute_into(
        matrix,
        options,
        &mut svd,
        &mut LinalgWorkspace::new(),
        context,
    )?;
    let rank = svd.analyze_rank(RankPolicy::Threshold(threshold))?.rank();
    let largest = svd.singular_values().first().copied().unwrap_or(S::ZERO);
    let smallest = svd.singular_values().last().copied().unwrap_or(S::ZERO);
    let reciprocal_condition = if largest == S::ZERO {
        S::ZERO
    } else {
        smallest / largest
    };
    if !reciprocal_condition.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite reciprocal condition estimate",
        ));
    }
    Ok(Factor {
        decomposition: Decomposition::Svd(svd),
        rank,
        reciprocal_condition,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn prepare(matrix: &Matrix<f64>, kind: SystemKind, minimum: f64) -> Factor<f64> {
        factor(
            matrix,
            kind,
            RankThreshold::new(0.0, 1e-12).unwrap(),
            minimum,
            LinalgExecutionOptions::new(),
            ExecutionContext::sequential_native(),
        )
        .unwrap()
    }

    #[test]
    fn direct_paths_retain_the_solve_factor() {
        let spd = Matrix::from_vec(2, 2, vec![4.0, 0.0, 0.0, 1.0]).unwrap();
        let indefinite = Matrix::from_vec(2, 2, vec![0.0, 2.0, 2.0, 0.0]).unwrap();
        let design = Matrix::from_vec(3, 2, vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        for (a, kind, method, rhs, expected) in [
            (
                &spd,
                SystemKind::PositiveDefinite,
                FitFactorization::Cholesky,
                vec![8.0, 3.0],
                [2.0, 3.0],
            ),
            (
                &indefinite,
                SystemKind::Symmetric,
                FitFactorization::Lu,
                vec![6.0, 4.0],
                [2.0, 3.0],
            ),
            (
                &design,
                SystemKind::LeastSquares,
                FitFactorization::PivotedQr,
                vec![-1.0, 2.0, 5.0],
                [2.0, 3.0],
            ),
        ] {
            let f = prepare(a, kind, 0.0);
            assert_eq!(f.method(), method);
            assert_eq!(f.condition_norm(), FitConditionNorm::One);
            assert_eq!(f.rank, 2);
            let mut w = LinalgWorkspace::new();
            w.try_reserve(&solve_requirements::<f64>(a.rows(), a.cols()).unwrap())
                .unwrap();
            let mut solution = Vector::zeros(2).unwrap();
            f.solve_into(
                &Vector::from_vec(rhs),
                solution.view_mut(),
                &mut w,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
            for (&actual, expected) in solution.as_slice().iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn condition_gate_falls_back_to_original_singular_value_diagnostics() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 2.0]).unwrap();
        let direct = prepare(&a, SystemKind::PositiveDefinite, 0.0);
        assert!(direct.reciprocal_condition < 0.32);
        let fallback = prepare(&a, SystemKind::PositiveDefinite, 0.32);
        let oracle = svd_factor(
            &a,
            RankThreshold::new(0.0, 1e-12).unwrap(),
            &LinalgExecutionOptions::new(),
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(fallback.method(), FitFactorization::Svd);
        assert_eq!(fallback.condition_norm(), FitConditionNorm::Two);
        assert_eq!(fallback.rank, oracle.rank);
        assert!((fallback.reciprocal_condition - oracle.reciprocal_condition).abs() < 1e-14);
        assert!(fallback.reciprocal_condition > 0.32);
    }

    #[test]
    fn breakdown_and_rank_screens_use_svd_without_accepting_indefinite_covariance() {
        let singular = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let fallback = prepare(&singular, SystemKind::PositiveDefinite, 0.0);
        assert_eq!(fallback.method(), FitFactorization::Svd);
        assert_eq!(fallback.rank, 1);
        let zero = Matrix::<f64>::zeros(1, 1).unwrap();
        let zero_factor = prepare(&zero, SystemKind::PositiveDefinite, 0.0);
        assert_eq!(zero_factor.rank, 0);
        assert_eq!(zero_factor.reciprocal_condition, 0.0);
        let indefinite = Matrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let result = factor(
            &indefinite,
            SystemKind::Covariance,
            RankThreshold::default(),
            0.0,
            LinalgExecutionOptions::new(),
            ExecutionContext::sequential_native(),
        );
        assert!(matches!(result, Err(error) if error.kind() == SciOxideErrorKind::DomainError));
    }
    #[test]
    fn svd_iteration_allowance_does_not_limit_the_direct_path() {
        let mut entries = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                let r = (i as f64 - j as f64) * 0.4;
                entries.push((-r * r).exp());
            }
        }
        let a = Matrix::from_vec(8, 8, entries).unwrap();
        let options =
            LinalgExecutionOptions::new().with_iteration_budget(scioxide_core::Budget::ONE);
        let threshold = RankThreshold::new(0.0, 1e-14).unwrap();
        let direct = factor(
            &a,
            SystemKind::PositiveDefinite,
            threshold,
            0.0,
            options,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
        assert_eq!(direct.method(), FitFactorization::Cholesky);
        let fallback = factor(
            &a,
            SystemKind::PositiveDefinite,
            threshold,
            1.0,
            options,
            ExecutionContext::sequential_native(),
        );
        assert!(
            matches!(fallback, Err(error) if error.kind() == SciOxideErrorKind::ConvergenceFailure)
        );
    }
}
