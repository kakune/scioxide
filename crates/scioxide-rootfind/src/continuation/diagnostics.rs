use super::*;
use scioxide_core::{ExecutionPolicy, RankThreshold};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, Matrix, RankPolicy, SvdValues,
};
/// Reports numerical indicators of a branch point at an accepted point.
///
/// The fields are floating-point estimates, not proofs. A branch point is
/// suspected when the `n × (n + 1)` Jacobian `[F_x F_λ]` loses rank, when the
/// bordered matrix `B = [F_x F_λ; (D⁻² t)ᵀ]` is singular, or when the sign of
/// `det B` changes between consecutive points. Here `t` is the unit tangent and
/// `D` the diagonal of coordinate scales from [`ContinuationOptions::with_scales`].
///
/// # References
///
/// E. L. Allgower and K. Georg, *Introduction to Numerical Continuation
/// Methods*, SIAM, 2003.
#[derive(Debug, Clone, Copy)]
pub struct ContinuationDiagnostic<R: RealScalar> {
    /// Estimated rank of `[F_x F_λ]`, when available.
    pub(super) augmented_rank: Option<usize>,
    /// Ratio of the smallest to the largest singular value, in `[0, 1]`.
    pub(super) reciprocal_condition: Option<R>,
    /// Sign of the bordered determinant, when available.
    pub(super) bordered_sign: Option<i8>,
    /// Whether rank loss, a zero determinant or a sign change suggests a branch point.
    pub(super) branch_point_candidate: bool,
}
impl<R: RealScalar> ContinuationDiagnostic<R> {
    /// Creates a diagnostic, for use by a [`ContinuationDiagnosticProvider`].
    ///
    /// `bordered_sign` is −1, 0 or 1. Continuation also marks the point as a
    /// branch-point candidate when `bordered_sign` has the opposite nonzero sign
    /// at the previous point.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `reciprocal_condition` is not finite or not in
    ///   `[0, 1]`, or `bordered_sign` is not −1, 0 or 1.
    pub fn new(
        augmented_rank: Option<usize>,
        reciprocal_condition: Option<R>,
        bordered_sign: Option<i8>,
        branch_point_candidate: bool,
    ) -> Result<Self, SciOxideError> {
        let value = Self {
            augmented_rank,
            reciprocal_condition,
            bordered_sign,
            branch_point_candidate,
        };
        value.validate()?;
        Ok(value)
    }
    /// Returns the estimated rank of `[F_x F_λ]`, when available.
    pub const fn augmented_rank(&self) -> Option<usize> {
        self.augmented_rank
    }
    /// Returns the ratio of the smallest to the largest singular value of `[F_x F_λ]`.
    pub const fn reciprocal_condition(&self) -> Option<R> {
        self.reciprocal_condition
    }
    /// Returns the sign of the bordered determinant `det B` (−1, 0 or 1), when available.
    pub const fn bordered_sign(&self) -> Option<i8> {
        self.bordered_sign
    }
    /// Returns whether this point is a suspected branch point.
    pub const fn branch_point_candidate(&self) -> bool {
        self.branch_point_candidate
    }

    /// Rejects an invalid diagnostic from the provider `p`.
    pub(super) fn validate(&self) -> Result<(), SciOxideError> {
        if self
            .reciprocal_condition
            .is_some_and(|v| !v.is_finite() || v < R::ZERO || v > R::ONE)
            || self.bordered_sign.is_some_and(|v| !(-1..=1).contains(&v))
        {
            return Err(invalid(
                scioxide_core::argument!("p"),
                InvalidInputReason::Inconsistent,
                "invalid injected branch diagnostic",
            ));
        }
        Ok(())
    }
}
/// Computes a branch-point diagnostic at each accepted point.
///
/// Implement this for large sparse or matrix-free problems, where the dense
/// diagnostic of [`ContinuationOptions::with_dense_diagnostics`] costs too much.
/// Attach it with [`ContinuationProblem::with_diagnostic`]. Each call counts as
/// one `Event` evaluation; work done inside should be charged to `context`.
/// The implementation provides its own scratch storage.
pub trait ContinuationDiagnosticProvider<R: RealScalar> {
    /// Computes the diagnostic of `problem` at `point = (x, λ)` with unit tangent `tangent`.
    ///
    /// # Errors
    ///
    /// Errors are returned by [`Continuation::advance`] unchanged, and the
    /// point is not accepted. A returned diagnostic whose rank exceeds `n` is an
    /// `InvalidInput` error.
    fn evaluate(
        &self,
        problem: &dyn JacobianOperator<R>,
        point: &[R],
        tangent: &[R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<ContinuationDiagnostic<R>, SciOxideError>;
}
pub(super) struct DenseDiagnostic<R: RealScalar> {
    matrix: Matrix<R>,
    bordered: Matrix<R>,
    direction: Vec<R>,
    product: Vec<R>,
    svd: LinalgWorkspace<'static, R>,
    values: SvdValues<R>,
    lu: LuDecomposition<R>,
    lu_workspace: LinalgWorkspace<'static, R>,
}
impl<R: RealScalar> DenseDiagnostic<R> {
    pub(super) fn required_bytes(m: usize) -> Result<usize, SciOxideError> {
        use scioxide_core::backend::{checked_add, checked_mul};
        let q = scioxide_core::quantity!("continuation_diagnostic_bytes");
        let n = m - 1;
        let scalars = checked_add(
            checked_add(checked_mul(n, m, q)?, checked_mul(m, m, q)?, q)?,
            checked_add(checked_add(m, n, q)?, n, q)?,
            q,
        )?;
        let execution = LinalgExecutionOptions::new();
        crate::workspace::sum([
            Ok(checked_mul(scalars, core::mem::size_of::<R>(), q)?),
            LuDecomposition::<R>::factor_storage_bytes(m),
            SvdValues::<R>::workspace_requirements::<R>(
                &scioxide_linalg::SvdWorkspaceRequest::new(n, m, &execution),
            )?
            .required_bytes::<R>(),
            LuDecomposition::<R>::workspace_requirements(
                &scioxide_linalg::LuWorkspaceRequest::factor(m, &execution),
            )?
            .required_bytes::<R>(),
        ])
    }
    pub(super) fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        crate::workspace::sum([
            self.matrix.retained_bytes(),
            self.bordered.retained_bytes(),
            crate::workspace::buffer(&self.direction),
            crate::workspace::buffer(&self.product),
            self.svd.retained_bytes(),
            self.values.retained_bytes(),
            self.lu.retained_bytes(),
            self.lu_workspace.retained_bytes(),
        ])
    }

    /// Allocates the dense diagnostic storage for `m = n + 1` coordinates.
    pub(super) fn new(m: usize) -> Result<Self, SciOxideError> {
        let mut matrix = Matrix::zeros(m - 1, m)?;
        let mut bordered = Matrix::zeros(m, m)?;
        for i in 0..m {
            bordered.as_mut_slice()[i * m + i] = R::ONE;
            if i < m - 1 {
                matrix.as_mut_slice()[i * m + i] = R::ONE;
            }
        }
        let mut svd = LinalgWorkspace::new();
        let mut values = SvdValues::empty();
        scioxide_linalg::SvdValues::compute_into(
            &matrix,
            &LinalgExecutionOptions::new(),
            &mut values,
            &mut svd,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        let mut lu = LuDecomposition::empty();
        let mut lu_workspace = LinalgWorkspace::new();
        LuDecomposition::factor_into(
            &bordered,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut lu,
            &mut lu_workspace,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok(Self {
            matrix,
            bordered,
            direction: scioxide_core::backend::try_filled(
                m,
                R::ZERO,
                scioxide_core::quantity!("workspace.dense.direction"),
            )?,
            product: scioxide_core::backend::try_filled(
                m - 1,
                R::ZERO,
                scioxide_core::quantity!("workspace.dense.product"),
            )?,
            svd,
            values,
            lu,
            lu_workspace,
        })
    }
    /// Computes the rank of `[F_x F_λ]` by SVD and the sign of the bordered determinant by LU.
    pub(super) fn evaluate(
        &mut self,
        p: &dyn JacobianOperator<R>,
        point: &[R],
        row: &[R],
        threshold: R,
        scratch: &mut [R],
        c: &mut SolveContext<'_, '_>,
    ) -> KernelResult<ContinuationDiagnostic<R>> {
        let m = point.len();
        let n = m - 1;
        let len = p.jvp_workspace_len()?;
        for col in 0..m {
            self.direction.fill(R::ZERO);
            self.direction[col] = R::ONE;
            self.product.fill(R::NAN);
            admit(c, EvaluationRole::LinearOperator)?;
            p.jvp_into_scoped(
                point,
                &self.direction,
                &mut self.product,
                &mut scratch[..len],
                c,
            )?;
            finite(&self.product, NonFiniteOrigin::Callback)?;
            for (r, &v) in self.product.iter().enumerate() {
                self.matrix.as_mut_slice()[r * m + col] = v;
                self.bordered.as_mut_slice()[r * m + col] = v;
            }
        }
        self.bordered.as_mut_slice()[n * m..].copy_from_slice(row);
        admit(c, EvaluationRole::Factorization)?;
        scioxide_linalg::SvdValues::compute_into(
            &self.matrix,
            &LinalgExecutionOptions::new(),
            &mut self.values,
            &mut self.svd,
            c.execution(),
        )?;
        let rank = self
            .values
            .analyze_rank(RankPolicy::Threshold(RankThreshold::new(
                R::ZERO,
                threshold,
            )?))?
            .rank();
        let s = self.values.singular_values();
        let reciprocal = if s[0] == R::ZERO {
            R::ZERO
        } else {
            s[n - 1] / s[0]
        };
        admit(c, EvaluationRole::Factorization)?;
        let sign = match LuDecomposition::factor_into(
            &self.bordered,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut self.lu,
            &mut self.lu_workspace,
            c.execution(),
        ) {
            Ok(()) => {
                let phase = self.lu.slogdet()?.phase();
                if phase > R::ZERO {
                    1
                } else if phase < R::ZERO {
                    -1
                } else {
                    0
                }
            }
            Err(error) => match error.view() {
                scioxide_core::SciOxideErrorView::Singular { .. } => 0,
                _ => return Err(error.into()),
            },
        };
        Ok(ContinuationDiagnostic {
            augmented_rank: Some(rank),
            reciprocal_condition: Some(reciprocal),
            bordered_sign: Some(sign),
            branch_point_candidate: rank < n || sign == 0,
        })
    }
}
