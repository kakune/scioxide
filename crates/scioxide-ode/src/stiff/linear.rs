//! Linear storage and the shared linear-solve contracts of stiff systems.
//!
//! [`LinearStorage`] owns one prepared backend: an assembled dense matrix with a pivoted LU,
//! an assembled CSR matrix with a sparse factor, or GMRES storage for a borrowed operator.
//! Dense factors reject exact zero pivots and solve without nested accounting or
//! diagnostics. Sparse reports map unaccepted terminations to [`SciOxideError::singular`]
//! through [`accepted`]. Iterative solves verify the true defect with [`verify_defect`];
//! verified direct solves check the normwise backward error with [`verify_backward_error`].
use crate::safety::invalid;
use crate::stiff::accuracy::StageAccuracy;
use crate::stiff::sparse::SparseStageFactor;
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_mul, try_filled};
use scioxide_core::{
    Budget, ExecutionContext, LinearOperator, NumericScalar, Preconditioner, RealScalar,
    ResidualTolerance, SciOxideError, SolveContext, StructuralPattern,
};
use scioxide_linalg::backend::lu::LuFactor;
use scioxide_linalg::{
    GmresSolver, IterativeWorkspace, IterativeWorkspaceRequest, Matrix, Preconditioning,
    VectorView, VectorViewMut,
};
use scioxide_sparse::{CsrMatrix, SparseLuLimits, SparseLuSymbolic, SparseValueWorkspace};

/// Selects the linear solver for the stage systems of implicit methods.
///
/// The selected backend is used as given; a failure does not fall back to
/// another one. The Jacobian comes from the method's
/// [`DerivativeStrategy`](scioxide_differentiate::DerivativeStrategy), with sources
/// that depend on the backend:
///
/// | Backend | Jacobian sources |
/// |---|---|
/// | `Dense` | analytic, automatic differentiation, or finite differences by column |
/// | `Sparse` | analytic values on the pattern, or finite differences with a column coloring of the pattern |
/// | `Krylov` | analytic or finite-difference Jacobian–vector products |
///
/// Sparse and Krylov backends need the identity mass matrix.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum OdeStageBackend {
    /// Assembles a dense Jacobian and solves with LU factorization with partial
    /// pivoting. This is the default.
    #[default]
    Dense,
    /// Assembles the Jacobian on a sparsity pattern and solves with sparse LU.
    ///
    /// A pattern with nonzeros only on the main and first off-diagonals is
    /// factored as a tridiagonal matrix. [`GaussRk`](crate::GaussRk) factors its
    /// coupled `2n×2n` system, whose blocks repeat this pattern, and `limits` apply
    /// to that system. [`RadauIia`](crate::RadauIia) factors each of its `n×n`
    /// shifted matrices on this pattern, with `limits` applied to each factor.
    ///
    /// The pattern is checked when the method prepares its workspace: a pattern
    /// that is not `n×n` or lacks a diagonal entry is an `InvalidInput` error.
    Sparse {
        /// Square pattern of the Jacobian; it must contain every diagonal entry.
        pattern: StructuralPattern,
        /// Fill and memory limits of the sparse factorization.
        limits: SparseLuLimits,
    },
    /// Solves with restarted GMRES on Jacobian–vector products, without forming the
    /// Jacobian.
    ///
    /// Systems of one stage block are preconditioned with
    /// [`OdeSystem::stage_preconditioner`](crate::OdeSystem::stage_preconditioner),
    /// and the coupled system of [`GaussRk`](crate::GaussRk) with
    /// [`OdeSystem::coupled_preconditioner`](crate::OdeSystem::coupled_preconditioner).
    /// Finite-difference products are accurate to about `√ε` relative to the state
    /// scale, where `ε` is the machine epsilon. Stage tolerances below that level,
    /// such as the startup of fixed-order BDF at working precision, need analytic
    /// products.
    Krylov {
        /// Maximum number of GMRES iterations per linear solve.
        iterations: Budget,
        /// Restart length: the maximum number of Krylov basis vectors kept.
        restart: Budget,
        /// Scratch length, in scalars, that one analytic Jacobian–vector product needs.
        operator_scratch: usize,
        /// Scratch length, in scalars, that one preconditioner application needs.
        preconditioner_scratch: usize,
    },
}

impl OdeStageBackend {
    /// Validate that a sparse pattern is square of order `n` with every diagonal.
    pub(crate) fn validate(&self, n: usize) -> Result<(), SciOxideError> {
        let Self::Sparse { pattern, .. } = self else {
            return Ok(());
        };
        let p = pattern.view();
        if p.rows() != n || p.cols() != n {
            return Err(invalid(
                scioxide_core::argument!("backend.pattern"),
                "sparse pattern must match stage dimension",
            ));
        }
        if (0..n).any(|row| {
            p.col_indices()[p.row_ptr()[row]..p.row_ptr()[row + 1]]
                .binary_search(&row)
                .is_err()
        }) {
            return Err(invalid(
                scioxide_core::argument!("backend.pattern"),
                "sparse stage pattern must include every diagonal",
            ));
        }
        Ok(())
    }
}

/// Relative linear tolerance for an iterative correction of `residual` at `state`:
/// one percent of the smallest requested component scale relative to the residual's
/// RMS magnitude, never looser than one percent.
pub(crate) fn iterative_tolerance<S, T>(
    accuracy: StageAccuracy<'_, S, T>,
    state: &[T],
    residual: &[T],
) -> Result<S, SciOxideError>
where
    S: RealScalar,
    T: NumericScalar<Real = S>,
{
    let percent = S::try_from_f64_lossy(0.01)?;
    let norm = crate::safety::scaled_rms_norm(residual)?;
    if norm <= S::ZERO {
        return Ok(percent);
    }
    let n = accuracy.dimension();
    state
        .iter()
        .enumerate()
        .try_fold(percent, |relative, (i, &value)| {
            Ok(relative.min(percent * accuracy.component_scale(i % n, value)? / norm))
        })
}

/// Require `‖applied − rhs‖₂ ≤ relative·‖rhs‖₂` and return the achieved ratio.
pub(crate) fn verify_defect<T: NumericScalar>(
    applied: &[T],
    rhs: &[T],
    relative: T::Real,
) -> Result<T::Real, SciOxideError> {
    let mut defect = scioxide_core::ScaledNorm::new();
    let mut norm = scioxide_core::ScaledNorm::new();
    for (&actual, &expected) in applied.iter().zip(rhs) {
        defect.add_magnitude(crate::safety::state_difference(actual, expected)?.abs())?;
        norm.add_magnitude(expected.abs())?;
    }
    let (defect, norm) = (defect.finish()?, norm.finish()?);
    if defect > relative * norm {
        return Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::NestedSolver,
            scioxide_core::ConvergenceDiagnostics::new(),
            "linear defect exceeds requested accuracy",
        ));
    }
    Ok(if norm == T::Real::ZERO {
        T::Real::ZERO
    } else {
        defect / norm
    })
}

/// Require the normwise backward error of a direct solve `A x = c`,
///
/// ```text
/// η = ‖c − A x‖∞ / (‖A‖∞ ‖x‖∞ + ‖c‖∞),
/// ```
///
/// to be at most `relative`, and return `η`. `applied` holds `A x` and
/// `matrix_norm` holds `‖A‖∞`. A backward-stable factorization attains `η` of a
/// small multiple of the unit roundoff whatever the condition of `A`, unlike the
/// relative residual `‖c − A x‖ / ‖c‖`.
///
/// # References
///
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, Theorem 7.1 (Rigal–Gaches).
pub(crate) fn verify_backward_error<T: NumericScalar>(
    applied: &[T],
    rhs: &[T],
    solution: &[T],
    matrix_norm: scioxide_core::Scaled<T::Real>,
    relative: T::Real,
) -> Result<T::Real, SciOxideError> {
    let maximum = |values: &[T]| values.iter().fold(T::Real::ZERO, |m, v| m.max(v.abs()));
    let mut defect = T::Real::ZERO;
    for (&actual, &expected) in applied.iter().zip(rhs) {
        defect = defect.max(crate::safety::state_difference(actual, expected)?.abs());
    }
    let scale = matrix_norm
        .mul(scioxide_core::Scaled::from_value(maximum(solution))?)?
        .add(scioxide_core::Scaled::from_value(maximum(rhs))?)?;
    let defect = scioxide_core::Scaled::from_value(defect)?;
    if !defect.abs_le(scale.mul(scioxide_core::Scaled::from_value(relative)?)?) {
        return Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::NestedSolver,
            scioxide_core::ConvergenceDiagnostics::new(),
            "linear backward error exceeds requested accuracy",
        ));
    }
    if scale.is_zero() {
        Ok(T::Real::ZERO)
    } else {
        defect.div(scale)?.round()
    }
}

// A shape-only provider prepares Krylov storage without executing callbacks.
struct Shape {
    n: usize,
    scratch: usize,
}
impl<T: NumericScalar> LinearOperator<T> for Shape {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn apply_into(&self, _: &[T], _: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        Err(invalid(
            scioxide_core::argument!("backend"),
            "shape query cannot execute",
        ))
    }
}
impl<T: NumericScalar> Preconditioner<T> for Shape {
    fn dimension(&self) -> usize {
        self.n
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.scratch)
    }
    fn precondition_into(&self, _: &[T], _: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        Err(invalid(
            scioxide_core::argument!("backend"),
            "shape query cannot execute",
        ))
    }
}

/// Prepared storage of one linear backend; preparation is the only allocation boundary.
// Caller-prepared inline storage avoids allocating an indirection per backend.
#[allow(clippy::large_enum_variant)]
pub(crate) enum LinearStorage<T: NumericScalar> {
    Dense {
        matrix: Matrix<T>,
        factor: LuFactor<T>,
    },
    Sparse {
        matrix: CsrMatrix<T>,
        values: SparseValueWorkspace<T>,
        factor: SparseStageFactor<T>,
        scratch: Vec<T>,
    },
    Krylov {
        work: IterativeWorkspace<T>,
        iterations: Budget,
        restart: Budget,
        operator_scratch: usize,
        preconditioner_scratch: usize,
    },
}

impl<T: NumericScalar> LinearStorage<T> {
    /// Reserve storage for an `n`-dimensional system. A Krylov `operator_scratch` is the
    /// scratch of one application of the operator that will be solved.
    pub(crate) fn new(n: usize, backend: &OdeStageBackend) -> Result<Self, SciOxideError> {
        Ok(match backend {
            OdeStageBackend::Dense => {
                let mut matrix = Matrix::empty();
                matrix.try_resize_flat(n, n)?;
                Self::Dense {
                    matrix,
                    factor: LuFactor::new(n)?,
                }
            }
            OdeStageBackend::Sparse { pattern, limits } => {
                backend.validate(n)?;
                Self::sparse(SparseLuSymbolic::analyze(
                    pattern.view(),
                    scioxide_sparse::SparseOrdering::default(),
                    *limits,
                )?)?
            }
            OdeStageBackend::Krylov {
                iterations,
                restart,
                operator_scratch,
                preconditioner_scratch,
            } => {
                if n == 0 {
                    return Err(invalid(
                        scioxide_core::argument!("dimension"),
                        "Krylov stage dimension must be positive",
                    ));
                }
                let solver = GmresSolver::<T>::new(
                    ResidualTolerance::new(T::Real::EPSILON, T::Real::EPSILON)?,
                    *iterations,
                )
                .with_restart(*restart);
                let work = solver.workspace(
                    &IterativeWorkspaceRequest::operator::<T>(&Shape {
                        n,
                        scratch: *operator_scratch,
                    })?
                    .with_preconditioner::<T>(Preconditioning::Inverse(
                        &Shape {
                            n,
                            scratch: *preconditioner_scratch,
                        },
                    ))?,
                )?;
                Self::Krylov {
                    work,
                    iterations: *iterations,
                    restart: *restart,
                    operator_scratch: *operator_scratch,
                    preconditioner_scratch: *preconditioner_scratch,
                }
            }
        })
    }

    /// Prepare an assembled sparse system with the caller's structural ordering.
    pub(crate) fn sparse(symbolic: SparseLuSymbolic) -> Result<Self, SciOxideError> {
        let p = symbolic.pattern();
        let n = p.rows();
        let matrix = CsrMatrix::from_pattern(
            p.to_owned()?,
            try_filled(
                p.col_indices().len(),
                T::ZERO,
                scioxide_core::quantity!("sparse_stage.values"),
            )?,
        )?;
        Ok(Self::Sparse {
            values: matrix.workspace(())?,
            matrix,
            factor: SparseStageFactor::new(symbolic)?,
            scratch: try_filled(
                checked_mul(n, 2, scioxide_core::quantity!("sparse_stage.scratch"))?,
                T::ZERO,
                scioxide_core::quantity!("sparse_stage.scratch"),
            )?,
        })
    }

    /// Prepare the bordered block-bidiagonal collocation system.
    pub(crate) fn bordered_sparse(
        pattern: scioxide_core::StructuralPatternView<'_>,
        n: usize,
        k: usize,
        nodes: usize,
        limits: SparseLuLimits,
    ) -> Result<Self, SciOxideError> {
        let matrix = CsrMatrix::from_pattern(
            pattern.to_owned()?,
            try_filled(
                pattern.col_indices().len(),
                T::ZERO,
                scioxide_core::quantity!("sparse_stage.values"),
            )?,
        )?;
        Ok(Self::Sparse {
            values: matrix.workspace(())?,
            matrix,
            factor: SparseStageFactor::bordered(pattern, n, k, nodes, limits)?,
            scratch: try_filled(
                pattern.rows() * 2,
                T::ZERO,
                scioxide_core::quantity!("sparse_stage.scratch"),
            )?,
        })
    }
    /// Change a direct system's active shape while keeping its prepared capacity.
    pub(crate) fn resize_direct(
        &mut self,
        n: usize,
        pattern: scioxide_core::StructuralPatternView<'_>,
        limits: SparseLuLimits,
        pattern_scratch: &mut [usize],
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Dense { matrix, factor } => {
                matrix.try_resize_flat(n, n)?;
                matrix.as_mut_slice().fill(T::ZERO);
                factor.try_reserve(n)?;
            }
            Self::Sparse {
                matrix,
                values,
                factor,
                scratch,
            } => {
                matrix.try_assign_pattern(pattern, pattern_scratch)?;
                values.try_reserve(&matrix.workspace_requirements(())?)?;
                factor.reprepare_natural(pattern, limits)?;
                scioxide_core::backend::try_reserve(
                    scratch,
                    (n * 2).saturating_sub(scratch.len()),
                    scioxide_core::quantity!("sparse_stage.scratch"),
                )?;
                scratch.resize(n * 2, T::ZERO);
            }
            Self::Krylov { .. } => {
                return Err(invalid(
                    scioxide_core::argument!("workspace"),
                    "direct resize requires assembled storage",
                ));
            }
        }
        Ok(())
    }

    /// Whether solves run GMRES on a borrowed operator instead of an assembled factor.
    pub(crate) const fn is_iterative(&self) -> bool {
        matches!(self, Self::Krylov { .. })
    }

    /// Scratch reserved for one operator and one preconditioner application.
    pub(crate) const fn iterative_scratch(&self) -> (usize, usize) {
        match self {
            Self::Krylov {
                operator_scratch,
                preconditioner_scratch,
                ..
            } => (*operator_scratch, *preconditioner_scratch),
            _ => (0, 0),
        }
    }

    /// Assembled dense matrix storage.
    pub(crate) fn dense_mut(&mut self) -> Result<&mut Matrix<T>, SciOxideError> {
        match self {
            Self::Dense { matrix, .. } => Ok(matrix),
            _ => Err(invalid(
                scioxide_core::argument!("workspace"),
                "dense stage storage mismatch",
            )),
        }
    }

    /// Fill the assembled sparse values in the order of their sorted pattern.
    pub(crate) fn assemble_sparse(
        &mut self,
        fill: impl FnOnce(
            scioxide_core::StructuralPatternView<'_>,
            &mut [T],
        ) -> Result<(), SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let Self::Sparse {
            matrix,
            values,
            factor,
            ..
        } = self
        else {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "sparse stage storage mismatch",
            ));
        };
        let pattern = factor.pattern()?;
        matrix.update_values(|_, candidate| fill(pattern, candidate), values)
    }

    /// Factor the assembled matrix.
    pub(crate) fn factor(&mut self, execution: ExecutionContext<'_>) -> Result<(), SciOxideError> {
        match self {
            Self::Dense { matrix, factor } => factor.factor(matrix.view(), execution),
            Self::Sparse { matrix, factor, .. } => factor.refactor(matrix.view(), execution),
            Self::Krylov { .. } => Err(invalid(
                scioxide_core::argument!("workspace"),
                "Krylov storage has no assembled factor",
            )),
        }
    }

    /// Solve with the prepared factor.
    pub(crate) fn solve_direct(
        &mut self,
        rhs: &[T],
        output: &mut [T],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Dense { factor, .. } => factor.solve_into(rhs, output, execution),
            Self::Sparse {
                factor, scratch, ..
            } => factor.solve_into(rhs, output, scratch, execution),
            Self::Krylov { .. } => Err(invalid(
                scioxide_core::argument!("workspace"),
                "Krylov storage has no assembled factor",
            )),
        }
    }

    /// Apply the assembled matrix.
    pub(crate) fn apply_assembled(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Dense { matrix, .. } => matrix
                .view()
                .try_finite()?
                .apply_into(input, output, scratch),
            Self::Sparse { matrix, .. } => matrix.apply_into(input, output, scratch),
            Self::Krylov { .. } => Err(invalid(
                scioxide_core::argument!("workspace"),
                "Krylov storage has no assembled matrix",
            )),
        }
    }

    /// The infinity norm `‖A‖∞ = maxᵢ Σⱼ |aᵢⱼ|` of the assembled matrix `A`, the
    /// operator actually solved: adjoint stages assemble the transposed matrix itself.
    pub(crate) fn assembled_norm_inf(
        &self,
    ) -> Result<scioxide_core::Scaled<T::Real>, SciOxideError> {
        let row_norm = |row: &[T]| {
            row.iter()
                .try_fold(scioxide_core::Scaled::ZERO, |sum, value| {
                    sum.add(scioxide_core::Scaled::from_value(value.abs())?)
                })
        };
        let mut norm = scioxide_core::Scaled::ZERO;
        let mut observe = |row: &[T]| -> Result<(), SciOxideError> {
            let candidate = row_norm(row)?;
            if norm.abs_le(candidate) {
                norm = candidate;
            }
            Ok(())
        };
        match self {
            Self::Dense { matrix, .. } => {
                for row in matrix.as_slice().chunks(matrix.cols().max(1)) {
                    observe(row)?;
                }
            }
            Self::Sparse { matrix, .. } => {
                for bounds in matrix.row_ptr().windows(2) {
                    observe(&matrix.values()[bounds[0]..bounds[1]])?;
                }
            }
            Self::Krylov { .. } => {
                return Err(invalid(
                    scioxide_core::argument!("workspace"),
                    "Krylov storage has no assembled matrix",
                ));
            }
        }
        Ok(norm)
    }

    /// Run restarted GMRES on `operator` from a zero start, within `limit` iterations.
    /// Unsuccessful terminations become typed errors; stops remain stops.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_iterative(
        &mut self,
        operator: &dyn LinearOperator<T>,
        preconditioner: Option<&dyn Preconditioner<T>>,
        rhs: &[T],
        output: &mut [T],
        relative: T::Real,
        limit: usize,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<usize, SciOxideError> {
        self.gmres(
            operator,
            preconditioner,
            rhs,
            output,
            relative,
            limit,
            context,
        )?
        .map_err(|reason| {
            SciOxideError::convergence_failure(
                reason,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
                "Krylov stage did not converge",
            )
        })
    }

    /// Restarted GMRES into `output`. `Ok(Err(reason))` reports a solve that stopped
    /// short of `relative` for a numerical reason (iteration limit, stagnation or
    /// breakdown), leaving the caller to treat it as a failed stage; budget and
    /// execution stops and provider errors are errors.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gmres(
        &mut self,
        operator: &dyn LinearOperator<T>,
        preconditioner: Option<&dyn Preconditioner<T>>,
        rhs: &[T],
        output: &mut [T],
        relative: T::Real,
        limit: usize,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Result<usize, scioxide_core::ConvergenceReason>, SciOxideError> {
        let Self::Krylov {
            work,
            iterations,
            restart,
            operator_scratch,
            preconditioner_scratch,
        } = self
        else {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "direct storage cannot run Krylov iterations",
            ));
        };
        if operator.apply_workspace_len()? > *operator_scratch
            || preconditioner.is_some_and(|p| p.dimension() != operator.rows())
            || preconditioner.map_or(Ok(0), Preconditioner::precondition_workspace_len)?
                > *preconditioner_scratch
        {
            return Err(invalid(
                scioxide_core::argument!("system"),
                "operator or preconditioner exceeds prepared Krylov capacity",
            ));
        }
        let solver = GmresSolver::<T>::new(
            ResidualTolerance::new(T::Real::ZERO, relative)?,
            Budget::try_from(iterations.get().min(limit))?,
        )
        .with_restart(*restart);
        output.fill(T::ZERO);
        let report = solver
            .solve_into(
                operator,
                VectorView::from_slice(rhs),
                preconditioner.map_or(Preconditioning::None, Preconditioning::Inverse),
                VectorViewMut::from_slice(output),
                work,
                &mut *context,
            )
            .map_err(|error| shared_evaluation_limit(error, context))?;
        if report.is_success() {
            return Ok(Ok(report.iterations()));
        }
        use scioxide_linalg::LinearSolveTermination;
        match report.termination() {
            LinearSolveTermination::ExecutionStopped(reason) => {
                Err(SciOxideError::execution_stopped(reason))
            }
            LinearSolveTermination::EvaluationLimit => {
                let budget = context.effective_budget_report(scioxide_core::BudgetKind::Evaluation);
                if budget.exhausted() {
                    return Err(crate::termination::Fault::Denied(budget).into());
                }
                Err(SciOxideError::convergence_failure(
                    scioxide_core::ConvergenceReason::EvaluationLimit,
                    scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
                    "Krylov stage exhausted its evaluation limit",
                ))
            }
            LinearSolveTermination::IterationLimit => {
                Ok(Err(scioxide_core::ConvergenceReason::IterationLimit))
            }
            _ => Ok(Err(scioxide_core::ConvergenceReason::NestedSolver)),
        }
    }
}

/// A Krylov solver stopped by the exhausted shared evaluation budget, before or after a
/// residual was available, is a budget stop, not a numerical failure: give its
/// report-less evaluation-limit error the denial's budget report so integrators end
/// with the accepted state.
fn shared_evaluation_limit(error: SciOxideError, context: &SolveContext<'_, '_>) -> SciOxideError {
    let budget = context.effective_budget_report(scioxide_core::BudgetKind::Evaluation);
    let reportless = matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::EvaluationLimit,
            diagnostics,
            ..
        } if diagnostics.budget().is_none()
    );
    if reportless && budget.exhausted() {
        crate::termination::Fault::Denied(budget).into()
    } else {
        error
    }
}

impl<T: NumericScalar> core::fmt::Debug for LinearStorage<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Dense { .. } => "LinearStorage::Dense",
            Self::Sparse { .. } => "LinearStorage::Sparse",
            Self::Krylov { .. } => "LinearStorage::Krylov",
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Hilbert matrix of order 10 (condition number about 1.6·10¹³) with `c = e₁`.
    fn hilbert_solve(backend: &OdeStageBackend) -> (LinearStorage<f64>, Vec<f64>, Vec<f64>) {
        let n = 10;
        let mut storage = LinearStorage::new(n, backend).unwrap();
        let hilbert = |i: usize, j: usize| 1.0 / (i + j + 1) as f64;
        match &mut storage {
            LinearStorage::Dense { matrix, .. } => {
                for (k, entry) in matrix.as_mut_slice().iter_mut().enumerate() {
                    *entry = hilbert(k / n, k % n);
                }
            }
            _ => storage
                .assemble_sparse(|pattern, values| {
                    for (row, bounds) in pattern.row_ptr().windows(2).enumerate() {
                        let columns = &pattern.col_indices()[bounds[0]..bounds[1]];
                        for (value, &column) in values[bounds[0]..bounds[1]].iter_mut().zip(columns)
                        {
                            *value = hilbert(row, column);
                        }
                    }
                    Ok(())
                })
                .unwrap(),
        }
        storage.factor(ExecutionContext::sequential()).unwrap();
        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        let mut solution = vec![0.0; n];
        storage
            .solve_direct(&rhs, &mut solution, ExecutionContext::sequential())
            .unwrap();
        let mut applied = vec![0.0; n];
        storage
            .apply_assembled(&solution, &mut applied, &mut [0.0; 10])
            .unwrap();
        (storage, solution, applied)
    }

    #[test]
    fn direct_solves_meet_the_backward_error_but_not_the_relative_residual() {
        let n = 10;
        let full = StructuralPattern::new(
            n,
            n,
            (0..=n).map(|row| row * n).collect(),
            (0..n * n).map(|k| k % n).collect(),
        )
        .unwrap();
        for backend in [
            OdeStageBackend::Dense,
            OdeStageBackend::Sparse {
                pattern: full,
                limits: SparseLuLimits::new(1 << 12, 1 << 20),
            },
        ] {
            let (storage, solution, applied) = hilbert_solve(&backend);
            let mut rhs = vec![0.0; n];
            rhs[0] = 1.0;
            let relative = 64.0 * f64::EPSILON;
            assert!(verify_defect(&applied, &rhs, relative).is_err());
            // ‖H‖∞ is the first row sum, the harmonic number H₁₀.
            let norm = storage.assembled_norm_inf().unwrap().round().unwrap();
            let harmonic: f64 = (1..=n).map(|k| 1.0 / k as f64).sum();
            assert!((norm - harmonic).abs() <= 4.0 * f64::EPSILON * harmonic);
            let eta = verify_backward_error(
                &applied,
                &rhs,
                &solution,
                storage.assembled_norm_inf().unwrap(),
                relative,
            )
            .unwrap();
            assert!(eta > 0.0 && eta <= relative);
        }
    }

    #[test]
    fn backward_error_rejects_a_perturbed_solution_and_accepts_zero_data() {
        let (storage, mut solution, mut applied) = hilbert_solve(&OdeStageBackend::Dense);
        let mut rhs = vec![0.0; 10];
        rhs[0] = 1.0;
        solution[3] *= 1.0 + 1e-6;
        storage
            .apply_assembled(&solution, &mut applied, &mut [0.0; 10])
            .unwrap();
        let norm = storage.assembled_norm_inf().unwrap();
        let error = verify_backward_error(&applied, &rhs, &solution, norm, 1e-10).unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        let zero = [0.0; 10];
        assert_eq!(
            verify_backward_error(&zero, &zero, &zero, norm, 0.0).unwrap(),
            0.0
        );
    }
}
