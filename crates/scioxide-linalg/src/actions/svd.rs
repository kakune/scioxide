//! Partial SVD by thick-restarted Golub–Kahan bidiagonalization with two-sided residual checks.
use super::workspace::{PartialSvdRequirements, SvdLayout};
use super::{ActionSlot, SpectralTermination, Work, run_action};
use crate::kernel::finite;
use crate::kernel::subspace as krylov;
use crate::workspace::SolveSlot;
use crate::workspace::accounting::{PlannedStorage, resize};
use crate::workspace::accounting::{StorageBytes, StoragePlan, bytes, product, sum};
use crate::{LinalgWorkspace, Matrix, ThinSvdDecomposition, VectorView};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::try_filled;
use scioxide_core::{
    AdjointOperator, AttemptRecord, Budget, NumericScalar, RealScalar, SciOxideError, Tolerance,
};
/// Solver for the largest singular triplets of an operator by restarted Lanczos bidiagonalization.
///
/// `A` is a `rows × cols` operator, used only through products `A x` and
/// `Aᴴ y`; `Aᴴ A` is never formed. From a right starting vector, Golub–Kahan
/// bidiagonalization builds orthonormal bases `V` and `U` of dimension up to
/// `m` with `A V = U B`, where `B` is upper bidiagonal. Each new vector is
/// reorthogonalized against the whole basis by classical Gram–Schmidt, with a
/// second pass when the norm drops by more than a factor `1/√2`. The SVD of
/// `B` gives the Ritz triplets. A restart keeps `max(count, d/2)` Ritz vectors,
/// but at most `d − 1` for basis dimension `d`, together with their residual
/// couplings, as in the augmented restart of Baglama and Reichel (2005).
/// Triplets that already meet the tolerance are kept first and checked again
/// in later cycles.
///
/// Triplets are accepted only when both residuals, computed from fresh
/// products with `A` and `Aᴴ`, meet the tolerance; either residual alone can
/// be small for a poor triplet. These checks run in the first and last cycles,
/// when the bases become invariant, and in any cycle where the estimate
/// `β |eₘᵀ ũ|` of some wanted triplet meets the tolerance. A cycle costs about
/// `2 (m − kept)` products, plus two per checked triplet.
///
/// The value holds only the settings; `rows` and `cols` are read from the
/// operator of each call. Defaults: basis dimension 30, at most 100 000
/// operator applications and residual tolerance `atol = rtol = √ε`. The
/// iteration limit of the execution options bounds the restart cycles (default
/// 100). The dense SVDs of the projected matrices use the other execution
/// options but no iteration limit.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, PartialSvdSolver, SpectralTermination};
/// let a = Matrix::<f64>::identity(2)?;
/// let (output, report) = PartialSvdSolver::new(1).solve(&a.view().try_finite()?, &[1.0, 0.0])?;
/// assert_eq!(report.termination(), SpectralTermination::Converged);
/// // Every singular value of the identity is 1.
/// assert!((output.singular_values()[0] - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - G. H. Golub and W. Kahan, "Calculating the singular values and
///   pseudo-inverse of a matrix", *SIAM J. Numer. Anal.* Ser. B 2(2), 1965.
/// - J. Baglama and L. Reichel, "Augmented implicitly restarted Lanczos
///   bidiagonalization methods", *SIAM J. Sci. Comput.* 27(1), 2005.
#[derive(Debug, Clone, Copy)]
pub struct PartialSvdSolver<R: RealScalar> {
    execution: crate::LinalgExecutionOptions,
    /// Number of leading triplets requested.
    count: usize,
    /// Maximum left and right basis dimension per cycle.
    basis: Budget,
    /// Total forward/adjoint provider calls, including both residual equations.
    applications: Budget,
    /// Independent residual tolerance relative to each singular value.
    tolerance: scioxide_core::ResidualTolerance<R>,
}
impl<R: RealScalar> PartialSvdSolver<R> {
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options; see [Execution](crate#execution).
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Sets the maximum dimension `m` of the left and right Krylov bases (default 30).
    ///
    /// `m` must be at least `count + 2`, unless `m ≥ min(rows, cols)`.
    pub const fn with_basis(mut self, basis: Budget) -> Self {
        self.basis = basis;
        self
    }

    /// Sets the maximum number of forward plus adjoint operator applications,
    /// including residual checks (default 100 000).
    pub const fn with_applications(mut self, applications: Budget) -> Self {
        self.applications = applications;
        self
    }

    /// Sets the residual tolerance (default `atol = rtol = √ε`).
    ///
    /// A triplet `(σ, u, v)` is accepted when
    /// `max(‖A v − σ u‖₂, ‖Aᴴ u − σ v‖₂) ≤ atol + rtol σ`.
    pub const fn with_tolerance(mut self, tolerance: scioxide_core::ResidualTolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Returns the number of singular triplets requested.
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Returns the maximum Krylov basis dimension.
    pub const fn basis(&self) -> Budget {
        self.basis
    }

    fn iteration_budget(&self) -> Budget {
        self.execution
            .iteration_budget()
            .unwrap_or(Budget::ONE.saturating_add(99))
    }

    /// Returns the maximum number of operator applications.
    pub const fn applications(&self) -> Budget {
        self.applications
    }

    /// Returns the residual tolerance.
    pub const fn tolerance(&self) -> scioxide_core::ResidualTolerance<R> {
        self.tolerance
    }
}
impl<R: RealScalar> PartialSvdSolver<R> {
    /// Returns default options for the `count` largest singular triplets.
    pub fn new(count: usize) -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            count,
            basis: Budget::ONE.saturating_add(29),
            applications: Budget::ONE.saturating_add(99_999),
            tolerance: scioxide_core::ResidualTolerance::from_tolerance(
                Tolerance::from_sqrt_epsilon(),
            ),
        }
    }
}
/// Largest singular triplets `(σ, u, v)` computed by [`PartialSvdSolver`].
///
/// Singular values are in descending order. The left vectors are the columns
/// of a `rows × count` matrix and the right vectors the columns of a
/// `cols × count` matrix, so that `A v ≈ σ u`; `V` is stored, not `Vᴴ`. Each
/// triplet has two residuals, `‖A v − σ u‖₂` and `‖Aᴴ u − σ v‖₂`, computed
/// from fresh products.
///
/// The buffer can hold fewer triplets than requested, and only `Converged`
/// guarantees that every triplet meets the tolerance. After a limit or
/// cancellation, it holds the most recent triplets whose residuals were both
/// computed, possibly none.
#[derive(Debug)]
pub struct PartialSvdBuffer<S: NumericScalar> {
    values: Vec<S::Real>,
    u: Matrix<S>,
    v: Matrix<S>,
    residuals: Vec<[S::Real; 2]>,
    count: usize,
}
impl<S: NumericScalar> ActionSlot for PartialSvdWorkspace<S> {
    fn slot(&mut self) -> &mut SolveSlot<()> {
        &mut self.slot
    }
}
impl<S: NumericScalar> PartialSvdWorkspace<S> {
    /// Returns the attempt record of the most recent solve, or `None` before the first.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.slot.last_attempt()
    }
}
impl<S: NumericScalar> PartialSvdBuffer<S> {
    /// Allocates a buffer for `count` triplets of `rows × cols` operators.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a buffer size overflows `usize`.
    /// - `AllocationFailure` if an allocation fails.
    pub fn new(rows: usize, cols: usize, count: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            values: try_filled(
                count,
                S::Real::ZERO,
                scioxide_core::quantity!("result.values"),
            )?,
            u: Matrix::zeros(rows, count)?,
            v: Matrix::zeros(cols, count)?,
            residuals: try_filled(
                count,
                [S::Real::ZERO; 2],
                scioxide_core::quantity!("result.residuals"),
            )?,
            count: 0,
        })
    }
    /// Returns the singular values in descending order; there may be fewer than requested.
    pub fn singular_values(&self) -> &[S::Real] {
        &self.values[..self.count]
    }
    /// Returns the left singular vectors `u` as the columns of a `rows × r` view.
    pub fn left_vectors(&self) -> crate::MatrixView<'_, S> {
        let mut view = self.u.view();
        view.cols = self.count;
        view
    }
    /// Returns the right singular vectors `v` as the columns of a `cols × r` view.
    pub fn right_vectors(&self) -> crate::MatrixView<'_, S> {
        let mut view = self.v.view();
        view.cols = self.count;
        view
    }
    /// Returns `[‖A v − σ u‖₂, ‖Aᴴ u − σ v‖₂]` for each triplet.
    pub fn residuals(&self) -> &[[S::Real; 2]] {
        &self.residuals[..self.count]
    }
}
#[derive(Debug, Clone, Copy)]
struct PartialSvdCall<R: RealScalar> {
    rows: usize,
    cols: usize,
    options: PartialSvdSolver<R>,
}
/// Workspace of [`PartialSvdSolver`].
///
/// It holds O((rows + cols) m + m²) scalars for basis dimension `m`.
#[derive(Debug)]
pub struct PartialSvdWorkspace<S: NumericScalar> {
    candidate_cycles: usize,
    proposal_cycles: usize,
    slot: SolveSlot<()>,
    rows: usize,
    cols: usize,
    dimension: usize,
    u: Vec<S>,
    v: Vec<S>,
    left: Vec<S>,
    coefficients: Vec<S>,
    right: Vec<S>,
    forward: Vec<S>,
    adjoint: Vec<S>,
    retained_u: Vec<S>,
    retained_v: Vec<S>,
    beta: S::Real,
    locked: Vec<bool>,
    order: Vec<usize>,
    small: Matrix<S>,
    factor: ThinSvdDecomposition<S>,
    factor_work: LinalgWorkspace<'static, S>,
    candidate: PartialSvdBuffer<S>,
    proposal: PartialSvdBuffer<S>,
}
impl<R: RealScalar> PartialSvdCall<R> {
    fn workspace_requirements<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
    ) -> Result<PartialSvdRequirements<S>, SciOxideError> {
        crate::checked::expect_shape([self.rows, self.cols], [a.rows(), a.cols()])?;
        let rows = self.rows;
        let cols = self.cols;
        let d = rows.min(cols).min(self.options.basis.get());
        let k = self.options.count;
        let options = super::projected_execution_options(self.options.execution);
        let dense = ThinSvdDecomposition::<S>::workspace_requirements(
            &crate::SvdWorkspaceRequest::new(d, d, &options),
        )?
        .max(&ThinSvdDecomposition::<S>::workspace_requirements(
            &crate::SvdWorkspaceRequest::new(d.saturating_sub(1), d, &options),
        )?)
        .max(&krylov::block_requirements::<S>(rows, d, d, &options)?)
        .max(&krylov::block_requirements::<S>(cols, d, d, &options)?);
        let payload = sum([
            bytes::<S>(sum([
                product(&[d, d])?,
                product(&[2, rows, d])?,
                product(&[cols, sum([product(&[2, d])?, 1])?])?,
                rows,
                cols,
                a.apply_workspace_len()?,
                a.apply_adjoint_workspace_len()?,
                product(&[3, d, d])?,
                product(&[2, k, sum([rows, cols])?])?,
            ])?)?,
            bytes::<R>(sum([d, product(&[6, k])?])?)?,
            bytes::<bool>(d)?,
            bytes::<usize>(d)?,
        ])?;
        PartialSvdRequirements::new(
            SvdLayout {
                rows: self.rows,
                cols: self.cols,
                basis: self.rows.min(self.cols).min(self.options.basis.get()),
                count: self.options.count,
                forward: a.apply_workspace_len()?,
                adjoint: a.apply_adjoint_workspace_len()?,
            },
            self.options.execution,
            payload,
            [dense, Default::default()],
            Default::default(),
        )
    }
    fn workspace<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
    ) -> Result<PartialSvdWorkspace<S>, SciOxideError> {
        let requirements = self.workspace_requirements(a)?;
        let mut workspace = PartialSvdWorkspace::new();
        workspace.try_reserve(&requirements)?;
        Ok(workspace)
    }
    fn solve<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        initial: &[S],
    ) -> Result<(PartialSvdBuffer<S>, PartialSvdReport), SciOxideError> {
        let mut workspace = self.workspace(a)?;
        let mut output = PartialSvdBuffer::new(self.rows, self.cols, self.options.count)?;
        let report = self.solve_into(
            a,
            initial,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok((output, report))
    }
    fn solve_into<'s, 'c, S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        initial: &[S],
        result: &mut PartialSvdBuffer<S>,
        w: &mut PartialSvdWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<PartialSvdReport, SciOxideError> {
        w.slot.reject();
        let requirements = self.workspace_requirements(a)?;
        let wanted = self.options.count;
        crate::checked::expect_shape([self.cols], [initial.len()])?;
        crate::checked::expect_shape([self.rows], [result.u.rows()])?;
        crate::checked::expect_shape([self.cols], [result.v.rows()])?;
        crate::checked::expect_shape([wanted], [result.values.len()])?;
        crate::checked::validate_values(
            initial,
            scioxide_core::NonFiniteOrigin::Input,
            "partial SVD start must be finite",
        )?;
        if wanted > 0 && initial.iter().all(|&x| x == S::ZERO) {
            return Err(crate::checked::out_of_range(
                scioxide_core::argument!("initial"),
                "starting vector must be nonzero",
            ));
        }
        w.prepare(&requirements)?;
        run_action(
            w,
            context.into(),
            scioxide_core::algorithm_id!("linear", "PartialSVD", 1),
            self.options.applications,
            &self.options.execution,
            |w, work| self.run(a, initial, result, w, work),
            |report: &PartialSvdReport| report.termination() == SpectralTermination::Converged,
        )
    }

    fn run<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        initial: &[S],
        result: &mut PartialSvdBuffer<S>,
        w: &mut PartialSvdWorkspace<S>,
        mut work: Work<'_, '_, '_>,
    ) -> Result<PartialSvdReport, SciOxideError> {
        let execution = work.execution;

        let rows = self.rows;
        let cols = self.cols;
        let wanted = self.options.count;
        if w.forward.len() != a.apply_workspace_len()?
            || w.adjoint.len() != a.apply_adjoint_workspace_len()?
        {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("a"),
                "operator changed its scratch requirement after preparation",
            ));
        }

        w.candidate.count = 0;
        w.proposal.count = 0;
        w.candidate_cycles = 0;
        let mut termination = if wanted == 0 {
            SpectralTermination::Converged
        } else {
            SpectralTermination::IterationLimit
        };
        let outcome = (|| {
            execution.check()?;
            if wanted > 0 {
                // Preflight rejected a zero start, and the norm bounds its largest entry.
                let norm = crate::reduction::l2_norm_with_context(
                    VectorView::from_slice(initial),
                    work.kernels,
                )?;
                for (v, &x) in w.v[..cols].iter_mut().zip(initial) {
                    *v = x / S::from_real(norm);
                }
                w.small.try_resize_flat(w.dimension, w.dimension)?;
                w.small.as_mut_slice().fill(S::ZERO);
                let mut start = 0;
                'cycles: for cycle in 0..self.options.iteration_budget().get() {
                    if !work.iteration()? {
                        termination = SpectralTermination::EvaluationLimit;
                        break 'cycles;
                    }
                    let core::ops::ControlFlow::Continue((dim, happy, left_breakdown)) =
                        self.expand_bidiagonal(a, w, &mut work, start)?
                    else {
                        termination = SpectralTermination::EvaluationLimit;
                        break 'cycles;
                    };
                    if dim == 0 {
                        termination = SpectralTermination::InvariantSubspace;
                        break;
                    }
                    w.proposal_cycles = cycle + 1;
                    let core::ops::ControlFlow::Continue((count, converged)) = self
                        .verify_singular_triplets(
                            a,
                            w,
                            &mut work,
                            (dim, left_breakdown),
                            cycle == 0
                                || cycle + 1 == self.options.iteration_budget().get()
                                || happy
                                || dim == rows.min(cols),
                        )?
                    else {
                        termination = SpectralTermination::EvaluationLimit;
                        break 'cycles;
                    };
                    if count > 0 {
                        core::mem::swap(&mut w.candidate, &mut w.proposal);
                        core::mem::swap(&mut w.candidate_cycles, &mut w.proposal_cycles);
                        w.proposal.count = 0;
                    }
                    if converged {
                        termination = SpectralTermination::Converged;
                        break;
                    }
                    if happy || dim == rows.min(cols) {
                        termination = SpectralTermination::InvariantSubspace;
                        break;
                    }
                    if dim <= wanted {
                        break;
                    }
                    start = self.restart(w, dim, work.kernels)?;
                }
            }
            execution.check()
        })();
        termination = super::action_termination(outcome, termination)?;
        if w.proposal.count > 0 && w.proposal.count >= w.candidate.count {
            core::mem::swap(&mut w.candidate, &mut w.proposal);
            core::mem::swap(&mut w.candidate_cycles, &mut w.proposal_cycles);
        }
        result.count = w.candidate.count;
        super::publish_snapshot(
            w.candidate.count,
            &w.candidate.values,
            &w.candidate.residuals,
            &mut result.values,
            &mut result.residuals,
            [
                (&w.candidate.u, &mut result.u),
                (&w.candidate.v, &mut result.v),
            ],
            (R::ZERO, [R::ZERO; 2]),
        );
        Ok(PartialSvdReport {
            termination,
            applications: work.count,
            cycles: w.candidate_cycles,
        })
    }
    fn expand_bidiagonal<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        w: &mut PartialSvdWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        start: usize,
    ) -> Result<core::ops::ControlFlow<(), (usize, bool, bool)>, SciOxideError> {
        let rows = self.rows;
        let cols = self.cols;
        let capacity = w.dimension;

        // The retained diagonal block and its coupling to v_start form
        // an arrowhead; ordinary bidiagonalization resumes beyond that column.
        let mut previous = R::ZERO;
        let mut dim = start;
        let mut happy = false;
        let mut left_breakdown = false;
        for column in start..capacity {
            work.execution.check()?;
            if !work.apply(
                a,
                &w.v[column * cols..(column + 1) * cols],
                &mut w.left,
                &mut w.forward,
            )? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            let reference = finite::norm(&w.left)?;
            if column > start {
                w.small.as_mut_slice()[(column - 1) * capacity + column] = S::from_real(previous);
            }
            for j in 0..column {
                let c = w.small.get_element(j, column);
                finite::axpy(
                    &mut w.left,
                    -c,
                    &w.u[j * rows..(j + 1) * rows],
                    work.kernels.dispatch(),
                )?;
            }
            let alpha = krylov::project(
                &mut w.left,
                &w.u[..column * rows],
                &mut w.coefficients,
                |j, c| {
                    w.small.as_mut_slice()[j * capacity + column] =
                        w.small.get_element(j, column) + c
                },
                work.kernels,
            )?;
            if alpha <= <R as RealScalar>::EPSILON * reference {
                happy = true;
                left_breakdown = true;
                break;
            }
            for i in 0..rows {
                w.u[column * rows + i] = w.left[i] / S::from_real(alpha);
            }
            w.small.as_mut_slice()[column * capacity + column] = S::from_real(alpha);
            dim = column + 1;
            if !work.adjoint(
                a,
                &w.u[column * rows..(column + 1) * rows],
                &mut w.right,
                &mut w.adjoint,
            )? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            for i in 0..cols {
                w.right[i] = w.right[i] - S::from_real(alpha) * w.v[column * cols + i];
            }
            previous = krylov::project(
                &mut w.right,
                &w.v[..(column + 1) * cols],
                &mut w.coefficients,
                |_, _| {},
                work.kernels,
            )?;
            w.beta = previous;
            if previous <= <R as RealScalar>::EPSILON * alpha {
                happy = true;
                break;
            }
            for i in 0..cols {
                w.v[(column + 1) * cols + i] = w.right[i] / S::from_real(previous);
            }
        }
        Ok(core::ops::ControlFlow::Continue((
            dim,
            happy,
            left_breakdown,
        )))
    }
    fn restart<S: NumericScalar<Real = R>>(
        &self,
        w: &mut PartialSvdWorkspace<S>,
        dim: usize,
        context: crate::KernelContext<'_>,
    ) -> Result<usize, SciOxideError> {
        // Baglama--Reichel thick restart: A V = U Sigma and
        // A^H U = V Sigma + beta v_next e_last^T U_small.
        // Retained target Ritz directions are soft locked: they are never
        // discarded at restart, and both residuals are rechecked for acceptance.
        let keep = self.options.count.max(dim / 2).min(dim - 1);
        for (i, index) in w.order[..keep].iter_mut().enumerate() {
            *index = i;
        }
        w.order[..keep].sort_unstable_by_key(|&index| (!w.locked[index], index));
        krylov::combine_block(
            crate::MatrixViewMut::from_slice(
                keep,
                self.rows,
                &mut w.retained_u[..keep * self.rows],
            )?
            .transpose(),
            &w.u[..dim * self.rows],
            |j, target| w.factor.u_matrix().get_element(j, w.order[target]),
            &mut w.coefficients,
            &mut w.factor_work,
            self.options.execution.execution_policy(),
            context,
        )?;
        krylov::combine_block(
            crate::MatrixViewMut::from_slice(
                keep,
                self.cols,
                &mut w.retained_v[..keep * self.cols],
            )?
            .transpose(),
            &w.v[..dim * self.cols],
            |j, target| w.factor.vt_matrix().get_element(w.order[target], j).conj(),
            &mut w.coefficients,
            &mut w.factor_work,
            self.options.execution.execution_policy(),
            context,
        )?;
        w.u[..keep * self.rows].copy_from_slice(&w.retained_u[..keep * self.rows]);
        w.v.copy_within(dim * self.cols..(dim + 1) * self.cols, keep * self.cols);
        w.v[..keep * self.cols].copy_from_slice(&w.retained_v[..keep * self.cols]);
        w.small.try_resize_flat(w.dimension, w.dimension)?;
        w.small.as_mut_slice().fill(S::ZERO);
        for j in 0..keep {
            let index = w.order[j];
            w.small.as_mut_slice()[j * w.dimension + j] =
                S::from_real(w.factor.singular_values()[index]);
            w.small.as_mut_slice()[j * w.dimension + keep] =
                S::from_real(w.beta) * w.factor.u_matrix().get_element(dim - 1, index).conj();
        }
        Ok(keep)
    }
    fn verify_singular_triplets<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        w: &mut PartialSvdWorkspace<S>,
        work: &mut Work<'_, '_, '_>,
        expansion: (usize, bool),
        force_verification: bool,
    ) -> Result<core::ops::ControlFlow<(), (usize, bool)>, SciOxideError> {
        let rows = self.rows;
        let cols = self.cols;
        let capacity = w.dimension;
        let wanted = self.options.count;

        let (dim, left_breakdown) = expansion;
        let right_dimension = dim + usize::from(left_breakdown);
        if dim < capacity {
            for i in 0..dim {
                for j in 0..right_dimension {
                    let value = w.small.get_element(i, j);
                    w.small.as_mut_slice()[i * right_dimension + j] = value;
                }
            }
            w.small.try_resize_flat(dim, right_dimension)?;
        }
        crate::ThinSvdDecomposition::compute_into(
            &w.small,
            &super::projected_execution_options(self.options.execution),
            &mut w.factor,
            &mut w.factor_work,
            work.kernels.scoped(&mut *work.solve),
        )?;
        w.locked.fill(false);
        let count = wanted.min(dim);
        let mut converged = count == wanted;
        if !force_verification {
            let mut ready = false;
            for target in 0..count {
                let estimate = w.beta * w.factor.u_matrix().get_element(dim - 1, target).abs();
                ready |= estimate
                    <= self
                        .options
                        .tolerance
                        .threshold(w.factor.singular_values()[target])?;
            }
            if !ready {
                return Ok(core::ops::ControlFlow::Continue((0, false)));
            }
        }
        krylov::combine_block(
            crate::MatrixViewMut::from_slice(count, rows, &mut w.retained_u[..count * rows])?
                .transpose(),
            &w.u[..dim * rows],
            |j, target| w.factor.u_matrix().get_element(j, target),
            &mut w.coefficients,
            &mut w.factor_work,
            self.options.execution.execution_policy(),
            work.kernels,
        )?;
        krylov::combine_block(
            crate::MatrixViewMut::from_slice(count, cols, &mut w.retained_v[..count * cols])?
                .transpose(),
            &w.v[..right_dimension * cols],
            |j, target| w.factor.vt_matrix().get_element(target, j).conj(),
            &mut w.coefficients,
            &mut w.factor_work,
            self.options.execution.execution_policy(),
            work.kernels,
        )?;
        for target in 0..count {
            let sigma = w.factor.singular_values()[target];
            for (i, &value) in w.retained_u[target * rows..(target + 1) * rows]
                .iter()
                .enumerate()
            {
                w.proposal.u.as_mut_slice()[i * wanted + target] = value;
            }
            for (i, &value) in w.retained_v[target * cols..(target + 1) * cols]
                .iter()
                .enumerate()
            {
                w.proposal.v.as_mut_slice()[i * wanted + target] = value;
            }
            for (i, value) in w.right.iter_mut().enumerate() {
                *value = w.proposal.v.get_element(i, target);
            }
            if !work.apply(a, &w.right, &mut w.left, &mut w.forward)? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            for i in 0..rows {
                w.left[i] = w.left[i] - S::from_real(sigma) * w.proposal.u.get_element(i, target);
            }
            // Both residuals `||A v - sigma u||` and `||A^H u - sigma v||`
            // are checked with the true operator; either alone can be
            // small for a poor singular triplet.
            let left_residual = crate::reduction::l2_norm_with_context(
                VectorView::from_slice(&w.left),
                work.kernels,
            )?;
            for i in 0..rows {
                w.left[i] = w.proposal.u.get_element(i, target);
            }
            if !work.adjoint(a, &w.left, &mut w.right, &mut w.adjoint)? {
                return Ok(core::ops::ControlFlow::Break(()));
            }
            for i in 0..cols {
                w.right[i] = w.right[i] - S::from_real(sigma) * w.proposal.v.get_element(i, target);
            }
            let right_residual = crate::reduction::l2_norm_with_context(
                VectorView::from_slice(&w.right),
                work.kernels,
            )?;
            w.proposal.values[target] = sigma;
            w.proposal.residuals[target] = [left_residual, right_residual];
            w.proposal.count = target + 1;
            w.locked[target] =
                left_residual.max(right_residual) <= self.options.tolerance.threshold(sigma)?;
            converged &= w.locked[target];
        }
        Ok(core::ops::ControlFlow::Continue((count, converged)))
    }
}
impl<R: RealScalar> PartialSvdSolver<R> {
    /// Checks the settings against `rows × cols` operators and returns the
    /// per-call value.
    fn call(&self, rows: usize, cols: usize) -> Result<PartialSvdCall<R>, SciOxideError> {
        super::validate_restart_capacity(rows.min(cols), self.count, self.basis)?;
        Ok(PartialSvdCall {
            rows,
            cols,
            options: *self,
        })
    }
    /// Returns the workspace storage needed to compute singular triplets of `a`.
    ///
    /// The operator is not applied; only its shape and scratch lengths are read.
    /// The storage includes the dense SVD of the projected matrix. The settings
    /// are checked first, so the execution methods report the same errors
    /// before any work.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`, argument `self.count`) if the requested
    ///   count exceeds `min(rows, cols)`.
    /// - `InvalidInput` (`Inconsistent`, argument `self.basis`) if the basis
    ///   dimension is less than `count + 2` and less than `min(rows, cols)`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the options.
    /// - Errors from the scratch-length queries of `a`.
    pub fn workspace_requirements<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
    ) -> Result<PartialSvdRequirements<S>, SciOxideError> {
        self.call(a.rows(), a.cols())?.workspace_requirements(a)
    }

    /// Allocates a workspace for `a`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
    ) -> Result<PartialSvdWorkspace<S>, SciOxideError> {
        self.call(a.rows(), a.cols())?.workspace(a)
    }

    /// Computes the singular triplets with a new workspace and result buffer.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`], [`PartialSvdBuffer::new`] and
    /// [`Self::solve_into`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, PartialSvdSolver};
    /// let a = Matrix::<f64>::identity(2)?;
    /// let method = PartialSvdSolver::new(1);
    /// let (output, report) = method.solve(&a.view().try_finite()?, &[1., 0.])?;
    /// // Every singular value of the identity is 1.
    /// assert!((output.singular_values()[0] - 1.).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn solve<S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        initial: &[S],
    ) -> Result<(PartialSvdBuffer<S>, PartialSvdReport), SciOxideError> {
        self.call(a.rows(), a.cols())?.solve(a, initial)
    }

    /// Computes the largest singular triplets of `a` from the right starting vector `initial`.
    ///
    /// The triplets are written to `result`. `initial` has length `cols` and
    /// must be nonzero.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `DimensionMismatch` if `initial` does not have length `cols`, or
    ///   `result` was allocated for another shape or count.
    /// - `NonFinite` if `initial` has a non-finite entry, or if `a` returns a
    ///   non-finite value.
    /// - `InvalidInput` (`OutOfRange`) if `initial` is zero.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `InvalidInput` (`Inconsistent`) if the scratch lengths of `a` changed
    ///   after the workspace was reserved.
    /// - Errors of the dense SVD, and errors returned by `a`.
    pub fn solve_into<'s, 'c, S: NumericScalar<Real = R>>(
        &self,
        a: &dyn AdjointOperator<S>,
        initial: &[S],
        result: &mut PartialSvdBuffer<S>,
        w: &mut PartialSvdWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<PartialSvdReport, SciOxideError> {
        self.call(a.rows(), a.cols())?
            .solve_into(a, initial, result, w, context)
    }
}

impl<S: NumericScalar> PartialSvdBuffer<S> {
    const fn empty() -> Self {
        Self {
            values: Vec::new(),
            u: Matrix::empty(),
            v: Matrix::empty(),
            residuals: Vec::new(),
            count: 0,
        }
    }
}

impl<S: NumericScalar> StorageBytes for PartialSvdBuffer<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.values.storage_bytes()?,
            self.u.storage_bytes()?,
            self.v.storage_bytes()?,
            self.residuals.storage_bytes()?,
        ])
    }
}

impl<S: NumericScalar> Default for PartialSvdWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> PartialSvdWorkspace<S> {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            candidate_cycles: 0,
            proposal_cycles: 0,
            slot: SolveSlot::new(),
            rows: 0,
            cols: 0,
            dimension: 0,
            u: Vec::new(),
            v: Vec::new(),
            left: Vec::new(),
            coefficients: Vec::new(),
            right: Vec::new(),
            forward: Vec::new(),
            adjoint: Vec::new(),
            retained_u: Vec::new(),
            retained_v: Vec::new(),
            beta: S::Real::ZERO,
            locked: Vec::new(),
            order: Vec::new(),
            small: Matrix::empty(),
            factor: ThinSvdDecomposition::empty(),
            factor_work: LinalgWorkspace::new(),
            candidate: PartialSvdBuffer::empty(),
            proposal: PartialSvdBuffer::empty(),
        }
    }
    /// Returns whether the workspace already holds the storage that `requirements` describes.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit of
    ///   `requirements`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    pub fn check(
        &self,
        requirements: &PartialSvdRequirements<S>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        PlannedStorage::check(self, requirements)
    }
    /// Grows the workspace to the storage that `requirements` describes.
    ///
    /// Buffers that are already large enough keep their capacity, and the
    /// attempt record is kept. If growth fails, some buffers may already have
    /// grown; the workspace stays usable.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit of
    ///   `requirements`.
    /// - `AllocationFailure` if an allocation fails.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    pub fn try_reserve(
        &mut self,
        requirements: &PartialSvdRequirements<S>,
    ) -> Result<(), SciOxideError> {
        PlannedStorage::reserve(self, requirements)
    }
}
impl<S: NumericScalar> PlannedStorage<S> for PartialSvdWorkspace<S> {
    type Requirements = PartialSvdRequirements<S>;
    fn storage_plan(
        &self,
        requirements: &PartialSvdRequirements<S>,
    ) -> Result<StoragePlan, SciOxideError> {
        let SvdLayout {
            rows,
            cols,
            basis: d,
            count,
            forward,
            adjoint,
        } = requirements.layout;
        let d = rows.min(cols).min(d);
        let mut plan = StoragePlan::new(self.storage_bytes()?);
        plan.buffer(&self.u, product(&[rows, d])?)?;
        plan.buffer(&self.v, product(&[cols, sum([d, 1])?])?)?;
        plan.buffer(&self.coefficients, product(&[d, d])?)?;
        plan.buffer(&self.left, rows)?;
        plan.buffer(&self.right, cols)?;
        plan.buffer(&self.forward, forward)?;
        plan.buffer(&self.adjoint, adjoint)?;
        plan.buffer(&self.retained_u, product(&[rows, d])?)?;
        plan.buffer(&self.retained_v, product(&[cols, d])?)?;
        plan.buffer(&self.locked, d)?;
        plan.buffer(&self.order, d)?;
        plan.matrix(&self.small, d, d)?;
        self.factor.action_storage(&mut plan, d, d)?;
        plan.dense(&self.factor_work, &requirements.storage.dense[0])?;
        plan.matrix(&self.candidate.u, rows, count)?;
        plan.matrix(&self.candidate.v, cols, count)?;
        plan.buffer(&self.candidate.values, count)?;
        plan.buffer(&self.candidate.residuals, count)?;
        plan.matrix(&self.proposal.u, rows, count)?;
        plan.matrix(&self.proposal.v, cols, count)?;
        plan.buffer(&self.proposal.values, count)?;
        plan.buffer(&self.proposal.residuals, count)?;
        Ok(plan)
    }
    fn configure(&mut self, requirements: &PartialSvdRequirements<S>) -> Result<(), SciOxideError> {
        let SvdLayout {
            rows,
            cols,
            basis: d,
            count,
            forward,
            adjoint,
        } = requirements.layout;
        let d = rows.min(cols).min(d);
        resize(&mut self.u, product(&[rows, d])?, S::ZERO)?;
        resize(&mut self.v, product(&[cols, sum([d, 1])?])?, S::ZERO)?;
        resize(&mut self.coefficients, product(&[d, d])?, S::ZERO)?;
        resize(&mut self.left, rows, S::ZERO)?;
        resize(&mut self.right, cols, S::ZERO)?;
        resize(&mut self.forward, forward, S::ZERO)?;
        resize(&mut self.adjoint, adjoint, S::ZERO)?;
        resize(&mut self.retained_u, product(&[rows, d])?, S::ZERO)?;
        resize(&mut self.retained_v, product(&[cols, d])?, S::ZERO)?;
        resize(&mut self.locked, d, false)?;
        resize(&mut self.order, d, 0)?;
        self.small.try_resize_flat(d, d)?;
        self.factor.reserve_result(d, d)?;
        self.factor_work
            .try_reserve(&requirements.storage.dense[0])?;
        self.candidate.u.try_resize_flat(rows, count)?;
        self.candidate.v.try_resize_flat(cols, count)?;
        resize(&mut self.candidate.values, count, S::Real::ZERO)?;
        resize(&mut self.candidate.residuals, count, [S::Real::ZERO; 2])?;
        self.proposal.u.try_resize_flat(rows, count)?;
        self.proposal.v.try_resize_flat(cols, count)?;
        resize(&mut self.proposal.values, count, S::Real::ZERO)?;
        resize(&mut self.proposal.residuals, count, [S::Real::ZERO; 2])?;
        self.rows = rows;
        self.cols = cols;
        self.dimension = d;
        Ok(())
    }
}
impl<S: NumericScalar> StorageBytes for PartialSvdWorkspace<S> {
    fn storage_bytes(&self) -> Result<usize, SciOxideError> {
        sum([
            self.u.storage_bytes()?,
            self.v.storage_bytes()?,
            self.left.storage_bytes()?,
            self.coefficients.storage_bytes()?,
            self.right.storage_bytes()?,
            self.forward.storage_bytes()?,
            self.adjoint.storage_bytes()?,
            self.retained_u.storage_bytes()?,
            self.retained_v.storage_bytes()?,
            self.locked.storage_bytes()?,
            self.order.storage_bytes()?,
            self.small.storage_bytes()?,
            self.factor.storage_bytes()?,
            self.factor_work.storage_bytes()?,
            self.candidate.storage_bytes()?,
            self.proposal.storage_bytes()?,
        ])
    }
}

/// Termination status and work counts of a partial SVD.
#[derive(Debug, Clone, Copy)]
pub struct PartialSvdReport {
    termination: SpectralTermination,
    applications: usize,
    cycles: usize,
}
impl PartialSvdReport {
    /// Returns the termination status.
    pub const fn termination(&self) -> SpectralTermination {
        self.termination
    }
    /// Returns the number of forward plus adjoint operator applications.
    pub const fn applications(&self) -> usize {
        self.applications
    }
    /// Returns the cycle, counted from 1, in which the returned triplets were
    /// computed, or zero if there are none.
    pub const fn cycles(&self) -> usize {
        self.cycles
    }
}
