//! Dense singular value decomposition with values-only, thin and full jobs.
//!
//! The input is scaled by a power of two and reduced to real upper bidiagonal
//! form by Householder (Golub–Kahan) bidiagonalization. Values-only jobs then
//! run dqds on the bidiagonal. Vector jobs run implicit-shift bidiagonal QR up
//! to order 64 and bidiagonal divide and conquer with QR leaves of order 32
//! above it. The Householder reflectors are applied back only to the requested
//! vector columns. Panel widths and crossovers do not depend on CPU features.

use super::util::reserve_vec;
use crate::{LinalgExecutionOptions, LinalgWorkspace};
use alloc::vec::Vec;
use scioxide_core::{ExecutionContext, ExecutionPolicy};
use typed::{SvdJob, compute_scoped, prepare, publish_full};

use scioxide_core::{EstimateReliability, NumericScalar, RealScalar, SciOxideError};

use crate::analysis::condition_ratio;
use crate::dense::{Matrix, Vector};
use crate::{ConditionEstimate, RankPolicy, RankReport};

mod bidiagonal_qr;
mod divide_conquer;
mod dqds;
pub(crate) mod solve;
mod typed;

pub use typed::{SvdValues, SvdWorkspaceRequest, ThinSvdDecomposition};
pub(crate) use typed::{full_candidate, thin_candidate, values_candidate};

/// Full singular value decomposition `A = U Σ Vᴴ` of a dense `m × n` matrix.
///
/// With `k = min(m, n)`:
///
/// ```text
/// A = U Σ Vᴴ,   U: m × m unitary,   Σ: m × n diagonal,   Vᴴ: n × n unitary
/// ```
///
/// The diagonal of `Σ` holds the singular values `σ₁ ≥ σ₂ ≥ … ≥ σₖ ≥ 0`, returned
/// by [`singular_values`](Self::singular_values) as a slice of length `k`. Exact
/// zeros are stored as `+0.0`, and ties are ordered deterministically.
/// [`u_matrix`](Self::u_matrix) returns `U` and [`vt_matrix`](Self::vt_matrix)
/// returns `Vᴴ`, the conjugate transpose of `V`, so row `i` of `Vᴴ` is `vᵢᴴ`. The
/// last `m − k` columns of `U` are orthogonal to the range of `A`, and the last
/// `n − k` columns of `V` are orthogonal to the range of `Aᴴ`.
///
/// Singular vectors are determined only up to a unit phase. For each `i < k`, the
/// entry of largest magnitude in column `i` of `U` (the first one on ties) is made
/// real and positive, and row `i` of `Vᴴ` is scaled to match. For a repeated
/// singular value, only the spanned subspace is unique.
///
/// The singular values are the raw output of the decomposition. Numerical rank,
/// truncation and solve acceptance need an explicit [`RankPolicy`]: see
/// [`analyze_rank`](Self::analyze_rank), [`solve`](Self::solve) and
/// [`pseudoinverse`](Self::pseudoinverse).
///
/// Use [`ThinSvdDecomposition`] when only the first `k` singular vectors are needed,
/// and [`SvdValues`] when only the singular values are needed.
///
/// The matrix is scaled by a power of two and reduced to real upper bidiagonal form
/// by Householder bidiagonalization (Golub & Van Loan, §5.4). Wide inputs are
/// reduced in adjoint orientation. Bidiagonals up to order 64 are diagonalized by
/// implicit-shift bidiagonal QR (Golub & Van Loan, §8.6), which chases each block
/// from its larger end diagonal entry towards the smaller one
/// (Demmel & Kahan 1990), so graded blocks converge in either orientation.
/// Larger ones use bidiagonal divide and conquer (Gu & Eisenstat 1995) with such
/// QR leaves, which forms both vector factors without a Gram matrix. The cost is
/// `O(m n k)` time. The Householder reduction is normwise backward stable, so the
/// singular values have absolute errors of order `ε σ₁`.
///
/// The iteration limit of the execution options bounds the total number of QR
/// sweeps and secular-equation iterations. Without a limit it is `30 k L`, where
/// `L = 1` for `k ≤ 64` and `L = 1 + ⌈log₂⌈k/32⌉⌉` otherwise.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
///
/// let a = Matrix::<f64>::from_vec(3, 3, vec![
///     1.0, 0.0, 0.0,
///     0.0, 2.0, 0.0,
///     0.0, 0.0, 3.0,
/// ]).unwrap();
/// let svd = scioxide_linalg::SvdDecomposition::compute(
///     &a)
/// .unwrap();
/// assert_eq!(svd.singular_values().len(), 3);
/// ```
///
/// # References
///
/// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins,
///   2013, §5.4 and §8.6.
/// - J. Demmel and W. Kahan, "Accurate singular values of bidiagonal matrices",
///   *SIAM J. Sci. Stat. Comput.* 11(5), 1990.
/// - M. Gu and S. C. Eisenstat, "A divide-and-conquer algorithm for the bidiagonal
///   SVD", *SIAM J. Matrix Anal. Appl.* 16(1), 1995.
#[derive(Debug, Clone)]
#[must_use]
pub struct SvdDecomposition<S: NumericScalar> {
    /// Left singular vectors U (m x m).
    u: Matrix<S>,
    /// Singular values in descending order.
    singular_values: Vec<S::Real>,
    /// Right singular vectors V^H (n x n).
    vt: Matrix<S>,
    /// Number of rows of the original matrix.
    rows: usize,
    /// Number of columns of the original matrix.
    cols: usize,
}

impl<S: NumericScalar> SvdDecomposition<S> {
    /// Solves the truncated, damped least-squares problem `min ‖A x − b‖₂² + λ ‖x‖₂²`.
    ///
    /// With `λ = damping` and the singular triplets `(σᵢ, uᵢ, vᵢ)` of `A`:
    ///
    /// ```text
    /// x = Σ_{σᵢ > τ} σᵢ / (σᵢ² + λ) · (uᵢᴴ b) · vᵢ
    /// ```
    ///
    /// The threshold `τ` comes from `rank_policy` (see [`analyze_rank`](Self::analyze_rank)).
    /// With `λ = 0` this is the minimum-norm least-squares solution restricted to the
    /// singular vectors with `σᵢ > τ`. The inner products `uᵢᴴ b` and the sum over `i` use
    /// compensated dot products. Use [`solve_into`](Self::solve_into) to reuse output and workspace.
    ///
    /// # Errors
    ///
    /// The errors of [`solve_into`](Self::solve_into).
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{SvdDecomposition, Matrix, RankPolicy};
    /// let a = Matrix::from_vec(1, 2, vec![1.0_f64, 1.0])?;
    /// let factor = SvdDecomposition::compute(&a)?;
    /// let x = factor.solve(&[2.0][..], RankPolicy::Exact, 0.0)?;
    /// // The minimum-norm solution of x₀ + x₁ = 2 is (1, 1).
    /// assert!(x.as_slice().iter().all(|value| (value - 1.0).abs() < 1e-14));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve<'b>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        rank_policy: RankPolicy<S::Real>,
        damping: S::Real,
    ) -> Result<Vector<S>, SciOxideError> {
        let mut output = Vector::zeros(self.cols)?;
        self.solve_into(
            rhs,
            rank_policy,
            damping,
            &crate::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Solves the truncated, damped least-squares problem of [`solve`](Self::solve) into `output`.
    ///
    /// `rhs` has `rows` elements and `output` has `cols` elements. Both may be strided.
    /// Size the workspace with [`SvdWorkspaceRequest::solve`]. The solve counts as one
    /// iteration toward the iteration limit of `options`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs.len() != rows` or `output.len() != cols`.
    /// - `NonFinite` if `damping` or an entry of `rhs` is not finite.
    /// - `InvalidInput` if `damping < 0`, or the errors of `rank_policy`.
    /// - `NumericalInstability` if a filtered coefficient overflows, for example
    ///   `1/σᵢ` for a subnormal `σᵢ` kept by [`RankPolicy::Exact`].
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        rhs: impl Into<crate::VectorView<'b, S>>,
        rank_policy: RankPolicy<S::Real>,
        damping: S::Real,
        options: &crate::LinalgExecutionOptions,
        output: crate::VectorViewMut<'_, S>,
        workspace: &mut crate::LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        self.spectral_view()?.solve_into(
            rhs,
            rank_policy,
            damping,
            options,
            output,
            workspace,
            context,
        )
    }

    /// Returns the truncated Moore–Penrose pseudoinverse `A⁺ = Σ_{σᵢ > τ} σᵢ⁻¹ vᵢ uᵢᴴ` (`n × m`).
    ///
    /// The threshold `τ` comes from `rank_policy`.
    ///
    /// # Errors
    ///
    /// The errors of [`pseudoinverse_into`](Self::pseudoinverse_into).
    pub fn pseudoinverse(
        &self,
        rank_policy: RankPolicy<S::Real>,
    ) -> Result<Matrix<S>, SciOxideError> {
        let mut output = Matrix::zeros(self.cols, self.rows)?;
        self.pseudoinverse_into(
            rank_policy,
            &crate::LinalgExecutionOptions::new(),
            output.view_mut(),
            &mut crate::LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Writes the truncated pseudoinverse of [`pseudoinverse`](Self::pseudoinverse) to `output`.
    ///
    /// `output` is `cols × rows`. Size the workspace with
    /// [`SvdWorkspaceRequest::pseudoinverse`]. Each of the `rows` output columns counts
    /// as one iteration toward the iteration limit of `options`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is not `cols × rows`.
    /// - The errors of `rank_policy`.
    /// - `NumericalInstability` if a reciprocal `1/σᵢ` overflows.
    /// - `ConvergenceFailure` if the iteration limit is smaller than `rows`.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn pseudoinverse_into<'scope, 'capability>(
        &self,
        rank_policy: RankPolicy<S::Real>,
        options: &crate::LinalgExecutionOptions,
        output: crate::MatrixViewMut<'_, S>,
        workspace: &mut crate::LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        self.spectral_view()?
            .pseudoinverse_into(rank_policy, options, output, workspace, context)
    }

    fn spectral_view(
        &self,
    ) -> Result<crate::decompose::svd::solve::SpectralView<'_, S>, SciOxideError> {
        Ok(crate::decompose::svd::solve::SpectralView {
            u: self.u.view(),
            values: &self.singular_values,
            vt: self
                .vt
                .view()
                .block(0, 0, self.singular_values.len(), self.cols)?,
        })
    }
    /// Returns the `min(m, n)` singular values in descending order.
    #[must_use]
    pub fn singular_values(&self) -> &[S::Real] {
        &self.singular_values
    }

    /// Returns the numerical rank: the number of singular values above the policy threshold.
    ///
    /// # Errors
    ///
    /// The errors of `policy` when its threshold is evaluated.
    pub fn analyze_rank(
        &self,
        policy: RankPolicy<S::Real>,
    ) -> Result<RankReport<S::Real>, SciOxideError> {
        spectrum_rank(&self.singular_values, self.rows, self.cols, policy)
    }

    /// Returns the reciprocal two-norm condition number `σₖ / σ₁`.
    ///
    /// No rank threshold is applied: a singular matrix gives `0`.
    ///
    /// # Errors
    ///
    /// - [`DomainError`](scioxide_core::SciOxideErrorKind::DomainError) if `min(m, n) = 0`,
    ///   because the condition number is undefined.
    pub fn condition_estimate(&self) -> Result<ConditionEstimate<S::Real>, SciOxideError> {
        spectrum_condition_estimate(&self.singular_values)
    }

    /// Returns the left singular vectors `U` (`m × m`, column `i` is `uᵢ`).
    pub const fn u_matrix(&self) -> &Matrix<S> {
        &self.u
    }

    /// Returns the conjugate-transposed right singular vectors `Vᴴ` (`n × n`, row `i` is `vᵢᴴ`).
    pub const fn vt_matrix(&self) -> &Matrix<S> {
        &self.vt
    }

    /// Returns the number of rows `m` of the decomposed matrix.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns `n` of the decomposed matrix.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }
}

impl<S: NumericScalar> SvdDecomposition<S> {
    /// Returns the workspace size for [`compute_into`](Self::compute_into), or for a solve or
    /// pseudoinverse request.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size or the default iteration limit overflows `usize`.
    /// - `ResourceLimit` if the size exceeds the byte limit of the request options.
    pub fn workspace_requirements(
        request: &SvdWorkspaceRequest,
    ) -> Result<crate::LinalgRequirements, SciOxideError> {
        request.requirements::<S>(SvdJob::Full)
    }
    /// Allocates a workspace sized by [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn workspace(
        request: &SvdWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}
impl<S: NumericScalar> SvdDecomposition<S> {
    /// Computes the full SVD of a dense matrix.
    ///
    /// The input may be any view, including strided and conjugated ones.
    ///
    /// # Errors
    ///
    /// The errors of [`compute_into`](Self::compute_into).
    pub fn compute<'m>(matrix: impl Into<crate::MatrixView<'m, S>>) -> Result<Self, SciOxideError> {
        Self::compute_with_options(matrix, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`compute`](Self::compute) with `options`, which set the product accuracy and the
    /// limits described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`compute`](Self::compute), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn compute_with_options<'m>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        let mut workspace = LinalgWorkspace::new();
        Self::compute_into(
            matrix,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the full SVD of a dense matrix into `output`.
    ///
    /// `output` grows to the shape of `matrix` as needed. Size the workspace with
    /// [`SvdWorkspaceRequest::new`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an entry of `matrix` is NaN or infinite.
    /// - `ConvergenceFailure` if bidiagonal QR or divide and conquer exceeds the
    ///   iteration limit.
    /// - `NumericalInstability` if undoing the input scaling overflows.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn compute_into<'scope, 'm, 'a>(
        matrix: impl Into<crate::MatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<crate::LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(options)?;
        let context = scope.kernel();
        let matrix = matrix.into();
        let options = *options;

        let rows = matrix.rows();
        let cols = matrix.cols();
        let order = rows.min(cols);
        let finite = prepare(matrix, workspace, SvdJob::Full, options, context)?;
        output.u.try_reserve_shape(rows, rows)?;
        output.vt.try_reserve_shape(cols, cols)?;
        reserve_vec(
            &mut output.singular_values,
            order,
            scioxide_core::quantity!("output.singular_values"),
        )?;
        compute_scoped::<S, true, true, _>(finite, workspace, options, scope, |scratch| {
            publish_full(output, scratch, rows, cols)
        })?;
        Ok(())
    }
}
impl<S: NumericScalar> SvdDecomposition<S> {
    /// Returns an empty decomposition to use as the output of [`compute_into`](Self::compute_into).
    pub const fn empty() -> Self {
        Self {
            u: Matrix::empty(),
            singular_values: Vec::new(),
            vt: Matrix::empty(),
            rows: 0,
            cols: 0,
        }
    }
}
pub(crate) fn spectrum_rank<R: RealScalar>(
    values: &[R],
    rows: usize,
    cols: usize,
    policy: RankPolicy<R>,
) -> Result<RankReport<R>, SciOxideError> {
    crate::analysis::analyze_magnitudes(values.iter().copied(), rows.min(cols), rows, cols, policy)
}
pub(crate) fn spectrum_condition_estimate<R: RealScalar>(
    values: &[R],
) -> Result<ConditionEstimate<R>, SciOxideError> {
    let Some(&largest) = values.first() else {
        return Err(crate::analysis::empty_condition());
    };
    let smallest = values.last().copied().unwrap_or(R::ZERO);
    let reciprocal = condition_ratio(smallest, largest)?;
    ConditionEstimate::new(reciprocal, EstimateReliability::ModelBased)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::RankThreshold;

    const TOL_F64: f64 = 1e-10;
    const TOL_F32: f32 = 1e-4;

    fn effective_rank<S: NumericScalar>(svd: &SvdDecomposition<S>) -> usize {
        svd.analyze_rank(RankPolicy::DimensionScaledEpsilon)
            .unwrap()
            .rank()
    }

    /// Reconstruct A = U * S * Vt from an SVD decomposition.
    fn reconstruct<S: NumericScalar>(
        svd: &SvdDecomposition<S>,
    ) -> Result<Matrix<S>, SciOxideError> {
        let m = svd.rows();
        let n = svd.cols();
        let k = svd.singular_values().len();

        let mut sigma = Matrix::<S>::zeros(m, n).unwrap();
        for i in 0..k.min(m).min(n) {
            let _ = sigma
                .view_mut()
                .set(i, i, S::from_real(svd.singular_values()[i]));
        }

        let us = svd.u_matrix().view().mul_mat(&sigma)?;
        us.view().mul_mat(svd.vt_matrix())
    }

    /// Check that two f64 matrices are approximately equal.
    fn assert_mat_approx(lhs: &Matrix<f64>, rhs: &Matrix<f64>, tol: f64) {
        assert_eq!(lhs.rows(), rhs.rows(), "row count mismatch");
        assert_eq!(lhs.cols(), rhs.cols(), "col count mismatch");
        for i in 0..lhs.rows() {
            for j in 0..lhs.cols() {
                let diff = (lhs.view().get(i, j).unwrap() - rhs.view().get(i, j).unwrap()).abs();
                assert!(
                    diff < tol,
                    "mismatch at ({i}, {j}): {} vs {}, diff = {diff}",
                    lhs.view().get(i, j).unwrap(),
                    rhs.view().get(i, j).unwrap()
                );
            }
        }
    }

    /// Check that two f32 matrices are approximately equal.
    fn assert_mat_approx_f32(lhs: &Matrix<f32>, rhs: &Matrix<f32>, tol: f32) {
        assert_eq!(lhs.rows(), rhs.rows(), "row count mismatch");
        assert_eq!(lhs.cols(), rhs.cols(), "col count mismatch");
        for i in 0..lhs.rows() {
            for j in 0..lhs.cols() {
                let diff = (lhs.view().get(i, j).unwrap() - rhs.view().get(i, j).unwrap()).abs();
                assert!(
                    diff < tol,
                    "mismatch at ({i}, {j}): {} vs {}, diff = {diff}",
                    lhs.view().get(i, j).unwrap(),
                    rhs.view().get(i, j).unwrap()
                );
            }
        }
    }

    // ---- Identity decomposition ----

    #[test]
    fn test_decompose_identity() {
        let eye = Matrix::<f64>::identity(3).unwrap();
        let svd = crate::SvdDecomposition::compute(&eye).unwrap();

        for &s in svd.singular_values() {
            assert!((s - 1.0).abs() < TOL_F64, "sigma = {s}, expected 1.0");
        }

        let recon = reconstruct(&svd).unwrap();
        assert_mat_approx(&eye, &recon, TOL_F64);
    }

    // ---- U * S * Vt = A roundtrip ----

    #[test]
    fn test_usvt_equals_a() {
        let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let recon = reconstruct(&svd).unwrap();
        assert_mat_approx(&a, &recon, TOL_F64);
    }

    // ---- U orthogonality ----

    #[test]
    fn test_u_orthogonal() {
        let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let u = svd.u_matrix();
        let uh = u.view().adjoint().try_to_matrix().unwrap();
        let uhu = uh.view().mul_mat(u).unwrap();
        let eye = Matrix::<f64>::identity(u.rows()).unwrap();
        assert_mat_approx(&uhu, &eye, TOL_F64);
    }

    // ---- V orthogonality ----

    #[test]
    fn test_v_orthogonal() {
        let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let vt = svd.vt_matrix();
        let vth = vt.view().adjoint().try_to_matrix().unwrap();
        let vtvth = vt.view().mul_mat(&vth).unwrap();
        let eye = Matrix::<f64>::identity(vt.rows()).unwrap();
        assert_mat_approx(&vtvth, &eye, TOL_F64);
    }

    // ---- Singular values are sorted descending ----

    #[test]
    fn test_singular_values_descending() {
        let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let sigma = svd.singular_values();
        for pair in sigma.windows(2) {
            assert!(
                pair[0] >= pair[1],
                "not descending: {} < {}",
                pair[0],
                pair[1]
            );
        }
    }

    // ---- Known singular values ----

    #[test]
    fn test_singular_values_known() {
        let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
            .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let sigma = svd.singular_values();
        assert!((sigma[0] - 3.0).abs() < TOL_F64, "sigma[0] = {}", sigma[0]);
        assert!((sigma[1] - 2.0).abs() < TOL_F64, "sigma[1] = {}", sigma[1]);
        assert!((sigma[2] - 1.0).abs() < TOL_F64, "sigma[2] = {}", sigma[2]);
    }

    // ---- Solve 2x2 ----

    #[test]
    fn test_solve_2x2() {
        let a = Matrix::<f64>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let b = Vector::from_vec(vec![5.0, 11.0]);
        let x = svd
            .solve(
                &b,
                RankPolicy::DimensionScaledEpsilon,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
            )
            .unwrap();
        assert!(
            (x.view().get(0).unwrap() - 1.0).abs() < TOL_F64,
            "x[0] = {}",
            x.view().get(0).unwrap()
        );
        assert!(
            (x.view().get(1).unwrap() - 2.0).abs() < TOL_F64,
            "x[1] = {}",
            x.view().get(1).unwrap()
        );
    }

    // ---- Pseudoinverse ----

    #[test]
    fn test_pseudoinverse() {
        let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let pinv = svd
            .pseudoinverse(RankPolicy::DimensionScaledEpsilon)
            .unwrap();
        let a_pinv = a.view().mul_mat(&pinv).unwrap();
        let a_pinv_a = a_pinv.view().mul_mat(&a).unwrap();
        assert_mat_approx(&a, &a_pinv_a, 1e-8);
    }

    // ---- Rank (full) ----

    #[test]
    fn test_rank_full() {
        let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
            .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        assert_eq!(effective_rank(&svd), 3);
    }

    // ---- Rank deficient ----

    #[test]
    fn test_rank_deficient() {
        let a = Matrix::<f64>::from_vec(3, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0])
            .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        assert_eq!(effective_rank(&svd), 1);
    }

    // ---- Rectangular m > n ----

    #[test]
    fn test_decompose_rectangular() {
        let a = Matrix::<f64>::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        assert_eq!(svd.rows(), 3);
        assert_eq!(svd.cols(), 2);
        assert_eq!(svd.singular_values().len(), 2);
        assert_eq!(svd.u_matrix().rows(), 3);
        assert_eq!(svd.u_matrix().cols(), 3);
        assert_eq!(svd.vt_matrix().rows(), 2);
        assert_eq!(svd.vt_matrix().cols(), 2);

        let recon = reconstruct(&svd).unwrap();
        assert_mat_approx(&a, &recon, TOL_F64);
    }

    // ---- Wide matrix m < n ----

    #[test]
    fn test_decompose_wide() {
        let a = Matrix::<f64>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        assert_eq!(svd.rows(), 2);
        assert_eq!(svd.cols(), 3);
        assert_eq!(svd.singular_values().len(), 2);

        let recon = reconstruct(&svd).unwrap();
        assert_mat_approx(&a, &recon, TOL_F64);
    }

    // ---- Dimensions of U, S, Vt ----

    #[test]
    fn test_non_square_rectangular() {
        let a =
            Matrix::<f64>::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        assert_eq!(svd.u_matrix().rows(), 4);
        assert_eq!(svd.u_matrix().cols(), 4);
        assert_eq!(svd.vt_matrix().rows(), 2);
        assert_eq!(svd.vt_matrix().cols(), 2);
        assert_eq!(svd.singular_values().len(), 2);

        let recon = reconstruct(&svd).unwrap();
        assert_mat_approx(&a, &recon, TOL_F64);
    }

    // ---- f32 test ----

    #[test]
    fn test_decompose_f32() {
        let a = Matrix::<f32>::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
            .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let sigma = svd.singular_values();
        assert!((sigma[0] - 3.0).abs() < TOL_F32, "sigma[0] = {}", sigma[0]);
        assert!((sigma[1] - 2.0).abs() < TOL_F32, "sigma[1] = {}", sigma[1]);
        assert!((sigma[2] - 1.0).abs() < TOL_F32, "sigma[2] = {}", sigma[2]);

        let recon = reconstruct(&svd).unwrap();
        assert_mat_approx_f32(&a, &recon, TOL_F32);
    }

    // ---- 2x2 known SVD ----

    #[test]
    fn test_decompose_2x2() {
        let a = Matrix::<f64>::from_vec(2, 2, vec![3.0, 0.0, 0.0, 4.0]).unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let sigma = svd.singular_values();
        assert!((sigma[0] - 4.0).abs() < TOL_F64, "sigma[0] = {}", sigma[0]);
        assert!((sigma[1] - 3.0).abs() < TOL_F64, "sigma[1] = {}", sigma[1]);

        let recon = reconstruct(&svd).unwrap();
        assert_mat_approx(&a, &recon, TOL_F64);
    }

    #[test]
    fn dimension_scaled_rank_policy_formula_f64() {
        let a = Matrix::from_vec(
            3,
            3,
            vec![
                1.0,
                0.0,
                0.0,
                0.0,
                4.0 * f64::EPSILON,
                0.0,
                0.0,
                0.0,
                2.0 * f64::EPSILON,
            ],
        )
        .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let rank = svd
            .analyze_rank(RankPolicy::DimensionScaledEpsilon)
            .unwrap();
        assert!((rank.threshold() - 3.0 * f64::EPSILON).abs() < f64::EPSILON * f64::EPSILON);
        assert_eq!(rank.rank(), 2);
    }

    #[test]
    fn dimension_scaled_rank_policy_formula_f32() {
        let a = Matrix::from_vec(
            3,
            3,
            vec![
                1.0_f32,
                0.0,
                0.0,
                0.0,
                4.0 * f32::EPSILON,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
        .unwrap();
        let svd = crate::SvdDecomposition::compute(&a).unwrap();
        let rank = svd
            .analyze_rank(RankPolicy::DimensionScaledEpsilon)
            .unwrap();
        assert!((rank.threshold() - 3.0 * f32::EPSILON).abs() < f32::EPSILON * f32::EPSILON);
        assert_eq!(rank.rank(), 2);
    }

    #[test]
    fn dimension_scaled_rank_is_scale_invariant() {
        for scale in [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)] {
            let a = Matrix::from_vec(2, 2, vec![scale, 0.0, 0.0, scale * f64::EPSILON]).unwrap();
            assert_eq!(
                effective_rank(&crate::SvdDecomposition::compute(&a).unwrap()),
                1
            );
        }

        for scale in [2.0_f32.powi(-60), 1.0, 2.0_f32.powi(60)] {
            let a = Matrix::from_vec(2, 2, vec![scale, 0.0, 0.0, scale * f32::EPSILON]).unwrap();
            assert_eq!(
                effective_rank(&crate::SvdDecomposition::compute(&a).unwrap()),
                1
            );
        }
    }

    #[test]
    fn test_scaled_svd_reconstruction_and_condition_f64() {
        for scale in [2.0_f64.powi(-500), 2.0_f64.powi(500)] {
            let a = Matrix::<f64>::from_vec(
                3,
                3,
                vec![
                    3.0 * scale,
                    0.0,
                    0.0,
                    0.0,
                    2.0 * scale,
                    0.0,
                    0.0,
                    0.0,
                    scale,
                ],
            )
            .unwrap();
            let svd = crate::SvdDecomposition::compute(&a).unwrap();
            assert_eq!(effective_rank(&svd), 3);
            assert!(
                (svd.condition_estimate().unwrap().reciprocal().value() - 1.0 / 3.0).abs() < 1e-12
            );
            assert_mat_approx(&a, &reconstruct(&svd).unwrap(), scale * 1e-12);
        }
    }

    #[test]
    fn test_scaled_svd_reconstruction_and_condition_f32() {
        for scale in [2.0_f32.powi(-60), 2.0_f32.powi(60)] {
            let a = Matrix::<f32>::from_vec(
                3,
                3,
                vec![
                    3.0 * scale,
                    0.0,
                    0.0,
                    0.0,
                    2.0 * scale,
                    0.0,
                    0.0,
                    0.0,
                    scale,
                ],
            )
            .unwrap();
            let svd = crate::SvdDecomposition::compute(&a).unwrap();
            assert_eq!(effective_rank(&svd), 3);
            assert!(
                (svd.condition_estimate().unwrap().reciprocal().value() - 1.0 / 3.0).abs() < 1e-5
            );
            assert_mat_approx_f32(&a, &reconstruct(&svd).unwrap(), scale * 1e-5);
        }
    }

    #[test]
    fn dimension_scaled_pseudoinverse_is_scale_aware() {
        for scale in [1e-100, 1e100] {
            let a = Matrix::from_vec(2, 2, vec![scale, 0.0, 0.0, 0.0]).unwrap();
            let pseudoinverse = crate::SvdDecomposition::compute(&a)
                .unwrap()
                .pseudoinverse(RankPolicy::DimensionScaledEpsilon)
                .unwrap();
            let projected = a
                .view()
                .mul_mat(&pseudoinverse)
                .unwrap()
                .view()
                .mul_mat(&a)
                .unwrap();
            assert_mat_approx(&a, &projected, scale * 1e-12);
        }

        for scale in [1e-10_f32, 1e10] {
            let a = Matrix::from_vec(2, 2, vec![scale, 0.0, 0.0, 0.0]).unwrap();
            let pseudoinverse = crate::SvdDecomposition::compute(&a)
                .unwrap()
                .pseudoinverse(RankPolicy::DimensionScaledEpsilon)
                .unwrap();
            let projected = a
                .view()
                .mul_mat(&pseudoinverse)
                .unwrap()
                .view()
                .mul_mat(&a)
                .unwrap();
            assert_mat_approx_f32(&a, &projected, scale * 1e-5);
        }
    }

    #[test]
    fn rank_policy_tolerances_are_checked() {
        assert!(matches!(
            (RankThreshold::new(-1.0, 0.0))
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        for tolerance in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                (RankThreshold::new(tolerance, 0.0))
                    .as_ref()
                    .err()
                    .unwrap()
                    .view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
    }

    #[test]
    fn zero_tolerance_never_publishes_overflowing_reciprocals() {
        let smallest = f64::from_bits(1);
        let matrix = Matrix::from_vec(1, 1, vec![smallest]).unwrap();
        let full = crate::SvdDecomposition::compute(&matrix).unwrap();
        assert_eq!(full.analyze_rank(RankPolicy::Exact).unwrap().rank(), 1);
        assert!(matches!(
            (full.pseudoinverse(RankPolicy::Exact))
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));

        let thin = crate::ThinSvdDecomposition::compute(&matrix).unwrap();
        assert_eq!(thin.analyze_rank(RankPolicy::Exact).unwrap().rank(), 1);
        assert!(matches!(
            (thin.solve(
                &Vector::from_vec(vec![1.0]),
                RankPolicy::Exact,
                <_ as scioxide_core::AlgebraicScalar>::ZERO
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
        assert!(matches!(
            (thin.pseudoinverse(RankPolicy::Exact))
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }

    #[test]
    fn raw_condition_is_independent_of_effective_rank_policy() {
        let zero = Matrix::<f64>::zeros(2, 2).unwrap();
        let zero_svd = crate::SvdDecomposition::compute(&zero).unwrap();
        assert_eq!(
            zero_svd.condition_estimate().unwrap().reciprocal().value(),
            0.0
        );
        assert_eq!(effective_rank(&zero_svd), 0);

        // A nonzero diagonal below the rank threshold tests raw conditioning
        // without depending on roundoff residue from an exactly singular input.
        let deficient = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, f64::EPSILON / 4.0]).unwrap();
        let deficient_svd = crate::SvdDecomposition::compute(&deficient).unwrap();
        assert_eq!(effective_rank(&deficient_svd), 1);
        assert!(
            deficient_svd
                .condition_estimate()
                .unwrap()
                .reciprocal()
                .value()
                > 0.0
        );

        let empty = Matrix::<f64>::zeros(0, 0).unwrap();
        let empty_svd = crate::SvdDecomposition::compute(&empty).unwrap();
        assert_eq!(effective_rank(&empty_svd), 0);
        assert!(matches!(
            (empty_svd.condition_estimate())
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
}
